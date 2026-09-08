//! Stdio MCP server. Tool name `probe_model` matches Jwrede/llmprobe;
//! the payload is canact host-policy JSON, not TTFT.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::{
    CatalogPriors, HostPolicyMeta, KeyRoute, OpenAiCompatClient, ProbeCache, ProbeError,
    ProbeRunner, SuiteTier, claude_code_access_token, looks_cheap, provider_from_base_url,
    refuse_cloud_without_key, resolve_api_key_from, resolve_host_catalog,
};

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Serve MCP over stdin/stdout until EOF. Returns a process exit code.
pub fn run_mcp_stdio() -> u8 {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut reader = stdin.lock();
    loop {
        let msg = match read_message(&mut reader) {
            Ok(Some(v)) => v,
            Ok(None) => return 0,
            Err(err) => {
                eprintln!("error: mcp read: {err}");
                return 1;
            }
        };
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        if method.starts_with("notifications/") {
            continue;
        }
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
        let result = match method {
            "initialize" => initialize_result(),
            "ping" => json!({}),
            "tools/list" => tools_list(),
            "tools/call" => match handle_tools_call(&params) {
                Ok(v) => v,
                Err(err) => {
                    if let Err(write_err) = write_message(&mut stdout, &error_response(id, err)) {
                        eprintln!("error: mcp write: {write_err}");
                        return 1;
                    }
                    continue;
                }
            },
            _ => {
                if let Err(write_err) = write_message(
                    &mut stdout,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("method not found: {method}") }
                    }),
                ) {
                    eprintln!("error: mcp write: {write_err}");
                    return 1;
                }
                continue;
            }
        };
        if let Err(err) = write_message(
            &mut stdout,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result
            }),
        ) {
            eprintln!("error: mcp write: {err}");
            return 1;
        }
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": "canact",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

fn tools_list() -> Value {
    json!({
        "tools": [{
            "name": "probe_model",
            "description": "Probe a model and return canact host-policy JSON (max_tools, edit ladder, XML/JSON repair, measured context). Not TTFT or latency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Model id" },
                    "provider": { "type": "string", "description": "Provider name" },
                    "base_url": { "type": "string", "description": "OpenAI-compatible base URL" },
                    "api_key_env": { "type": "string", "description": "Env var holding the API key (never pass the key itself)" },
                    "cache": { "type": "string", "description": "Probe cache path" },
                    "suite": {
                        "type": "string",
                        "description": "policy (default), full, or all. cheap/full remain aliases."
                    },
                    "cheap": { "type": "boolean", "description": "Alias of suite=policy" },
                    "full": { "type": "boolean", "description": "Alias of suite=full" },
                    "vision": {
                        "type": "boolean",
                        "description": "true runs vision; false skips it. Omit to use the host catalog."
                    },
                    "force": { "type": "boolean" },
                    "advertised_context": { "type": "integer", "minimum": 1 }
                },
                "required": ["model"]
            }
        }]
    })
}

fn handle_tools_call(params: &Value) -> Result<Value, String> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing tool name".to_owned())?;
    if name != "probe_model" {
        return Err(format!("unknown tool: {name}"));
    }
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let envelope = rt.block_on(probe_model_args(&args))?;
    let text = serde_json::to_string_pretty(&envelope).map_err(|e| e.to_string())?;
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false
    }))
}

async fn probe_model_args(args: &Value) -> Result<Value, String> {
    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let api_key_env = args.get("api_key_env").and_then(Value::as_str);
    let named_key = match api_key_env {
        Some(var) if !var.is_empty() => std::env::var(var).ok().filter(|s| !s.is_empty()),
        _ => None,
    };
    let route = mcp_resolve_key_route(
        api_key_env,
        named_key,
        std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        std::env::var("XAI_API_KEY").ok().filter(|s| !s.is_empty()),
        anthropic_env_key(),
        provider_given,
    );
    probe_model_with_route(args, route, api_key_env).await
}

async fn probe_model_with_route(
    args: &Value,
    route: KeyRoute,
    api_key_env: Option<&str>,
) -> Result<Value, String> {
    let model = args
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "model is required".to_owned())?
        .to_owned();
    let advertised = json_u32(args.get("advertised_context"));
    let cheap = json_bool(args.get("cheap")).unwrap_or(false);
    let full = json_bool(args.get("full")).unwrap_or(false);
    let suite = match args.get("suite").and_then(Value::as_str) {
        Some(raw) => SuiteTier::parse(raw)
            .ok_or_else(|| format!("unknown suite={raw} (expected policy, full, or all)"))?,
        None if full => SuiteTier::Full,
        None if cheap => SuiteTier::Policy,
        None => SuiteTier::Policy,
    };
    let vision_flag = json_bool(args.get("vision"));
    let vision = vision_flag.unwrap_or(false);
    let force = json_bool(args.get("force")).unwrap_or(false);
    let cache_path = args
        .get("cache")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .map(expand_tilde)
        .unwrap_or_else(default_cache_path);
    let mut cache = ProbeCache::load(&cache_path)
        .map_err(|e| format!("failed to load cache {}: {e}", cache_path.display()))?;

    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let api_key = route.key.clone();
    let base_url = args
        .get("base_url")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| route.default_base_url(provider_given.as_deref().unwrap_or("")));
    let provider = provider_given.unwrap_or_else(|| provider_from_base_url(&base_url));
    if !force {
        if let Some(profile) = cache.get_with_suite(&model, &provider, suite, vision, advertised) {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true, true, suite, advertised,
            )));
        }
        if advertised.is_none() && vision_flag.is_none() {
            if let Some((profile, _cheap_row, stored_advertised)) =
                cache.find_profile_unspecified_catalog_suite(&model, &provider, suite)
            {
                return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                    true,
                    true,
                    suite,
                    stored_advertised,
                )));
            }
        }
        if matches!(suite, SuiteTier::Policy) && !vision {
            if let Some((profile, cheap_row)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
            {
                let hit = if cheap_row {
                    SuiteTier::Policy
                } else {
                    SuiteTier::Full
                };
                return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                    true, true, hit, advertised,
                )));
            }
        }
    }
    if refuse_cloud_without_key(api_key.as_deref(), &base_url) {
        return Err(mcp_missing_key_error(api_key_env));
    }
    let hints = resolve_host_catalog(
        advertised,
        vision_flag,
        &base_url,
        api_key.as_deref(),
        &model,
    )
    .await;
    let advertised = hints.advertised_context_tokens;
    let vision = hints.supports_vision == Some(true);
    if !force {
        if let Some(profile) = cache.get_with_suite(&model, &provider, suite, vision, advertised) {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true, true, suite, advertised,
            )));
        }
        if matches!(suite, SuiteTier::Policy) && !vision {
            if let Some((profile, cheap_row)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
            {
                let hit = if cheap_row {
                    SuiteTier::Policy
                } else {
                    SuiteTier::Full
                };
                return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                    true, true, hit, advertised,
                )));
            }
        }
    }
    let catalog = CatalogPriors {
        advertised_context_tokens: advertised,
        supports_vision: hints.supports_vision,
        supports_tools: None,
    };
    let throttle = looks_cheap(&provider, &model, &base_url);
    let client = OpenAiCompatClient::new(base_url, api_key, model, provider, catalog)
        .map_err(|e| e.to_string())?;
    let mut runner = ProbeRunner::new(client).suite(suite);
    if throttle {
        runner = runner.throttled();
    }
    let run = runner.run_detailed().await.map_err(|e| match e {
        ProbeError::Auth(msg) => format!("authentication error: {msg}"),
        other => other.to_string(),
    })?;
    if let Err(err) = run.persist(&mut cache, &cache_path) {
        eprintln!("warning: failed to save probe cache: {err}");
    }
    Ok(run.host_policy_envelope())
}

/// Injected-key MCP route. Tests pass values so they do not race on env.
fn mcp_resolve_key_route(
    api_key_env: Option<&str>,
    named_key: Option<String>,
    openai: Option<String>,
    openrouter: Option<String>,
    xai: Option<String>,
    anthropic: Option<String>,
    provider: &str,
) -> KeyRoute {
    match api_key_env {
        Some(var) if !var.is_empty() => KeyRoute {
            key: named_key,
            from_openrouter: var == "OPENROUTER_API_KEY",
            from_xai: var == "XAI_API_KEY",
            from_anthropic: var == "ANTHROPIC_AUTH_TOKEN" || var == "ANTHROPIC_API_KEY",
        },
        _ => resolve_api_key_from(None, openai, openrouter, xai, anthropic, provider),
    }
}

/// Named `api_key_env` does not fall back to OPENAI_API_KEY / XAI_API_KEY.
fn mcp_missing_key_error(api_key_env: Option<&str>) -> String {
    match api_key_env {
        Some(var) if !var.is_empty() => format!("{var} is unset or empty"),
        _ => "set api_key_env (or OPENAI_API_KEY / OPENROUTER_API_KEY / XAI_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY), or pass base_url for a local host"
            .to_owned(),
    }
}

fn anthropic_env_key() -> Option<String> {
    std::env::var("ANTHROPIC_AUTH_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .or_else(claude_code_access_token)
}

fn default_cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("canact")
        .join("probes.json")
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(home) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(home));
        }
    }
    dirs::home_dir()
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return home_dir().unwrap_or(path);
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    path
}

fn json_bool(v: Option<&Value>) -> Option<bool> {
    let v = v?;
    v.as_bool()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn json_u32(v: Option<&Value>) -> Option<u32> {
    let v = v?;
    v.as_u64()
        .map(|n| n as u32)
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn error_response(id: Value, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{ "type": "text", "text": message }],
            "isError": true
        }
    })
}

const MAX_MCP_BYTES: u64 = 8 * 1024 * 1024;

fn read_limited_line(reader: &mut impl BufRead) -> Result<Option<String>, String> {
    let mut line = String::new();
    let n = {
        let mut limited = reader.take(MAX_MCP_BYTES + 1);
        limited.read_line(&mut line).map_err(|e| e.to_string())?
    };
    if n == 0 {
        return Ok(None);
    }
    if line.len() as u64 > MAX_MCP_BYTES {
        return Err("line too large".to_owned());
    }
    Ok(Some(line))
}

fn read_content_length_body(
    reader: &mut impl BufRead,
    first_line: &str,
) -> Result<Option<Value>, String> {
    let mut headers = first_line.to_owned();
    loop {
        let Some(next) = read_limited_line(reader)? else {
            return Err("eof during headers".to_owned());
        };
        if next == "\r\n" || next == "\n" {
            break;
        }
        headers.push_str(&next);
    }
    let mut len = None;
    for header in headers.lines() {
        let header = header.trim_end_matches('\r');
        if let Some(rest) = header.strip_prefix("Content-Length:") {
            len = rest.trim().parse::<usize>().ok();
        }
    }
    let len = len.ok_or_else(|| "missing Content-Length".to_owned())?;
    if len as u64 > MAX_MCP_BYTES {
        return Err("Content-Length too large".to_owned());
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    serde_json::from_slice(&buf)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn read_message(reader: &mut impl BufRead) -> Result<Option<Value>, String> {
    loop {
        let Some(line) = read_limited_line(reader)? else {
            return Ok(None);
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return serde_json::from_str(trimmed)
                .map(Some)
                .map_err(|e| e.to_string());
        }
        return read_content_length_body(reader, &line);
    }
}

fn write_message(writer: &mut impl Write, value: &Value) -> Result<(), String> {
    let mut body = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    body.push(b'\n');
    writer.write_all(&body).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ANTHROPIC_BASE_URL, XAI_BASE_URL};
    use std::io::Cursor;

    #[test]
    fn mcp_openrouter_provider_uses_openrouter_key_when_xai_also_set() {
        let route = mcp_resolve_key_route(
            None,
            None,
            None,
            Some("sk-or-env".to_owned()),
            Some("xai-env".to_owned()),
            None,
            "openrouter",
        );
        assert_eq!(
            route.key.as_deref(),
            Some("sk-or-env"),
            "MCP provider=openrouter must not send XAI_API_KEY to OpenRouter"
        );
        assert!(route.from_openrouter);
        assert!(!route.from_xai);
        assert_eq!(
            route.default_base_url("openrouter"),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn mcp_anthropic_provider_uses_anthropic_key_when_xai_also_set() {
        let route = mcp_resolve_key_route(
            None,
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "anthropic",
        );
        assert_eq!(
            route.key.as_deref(),
            Some("sk-ant-env"),
            "MCP provider=anthropic must not send XAI_API_KEY to Anthropic"
        );
        assert!(route.from_anthropic);
        assert!(!route.from_xai);
        assert_eq!(route.default_base_url("anthropic"), ANTHROPIC_BASE_URL);
    }

    #[tokio::test]
    async fn mcp_force_without_key_skips_catalog_lookup() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let route = KeyRoute {
            key: None,
            from_openrouter: false,
            from_xai: false,
            from_anthropic: false,
        };
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "force": true,
            "cache": cache_path.to_str().expect("utf8"),
        });
        let err = probe_model_with_route(&args, route, None)
            .await
            .unwrap_err();
        assert_eq!(err, mcp_missing_key_error(None));
        assert!(
            crate::adapters::openai::take_catalog_lookups().is_empty(),
            "must not call catalog"
        );
    }

    #[tokio::test]
    async fn mcp_named_api_key_env_unset_skips_catalog_lookup() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let route = mcp_resolve_key_route(
            Some("FOO_KEY"),
            None,
            Some("sk-openai".to_owned()),
            Some("sk-or".to_owned()),
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "openai",
        );
        assert_eq!(route.key, None);
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "force": true,
            "cache": cache_path.to_str().expect("utf8"),
            "api_key_env": "FOO_KEY",
        });
        let err = probe_model_with_route(&args, route, Some("FOO_KEY"))
            .await
            .unwrap_err();
        assert_eq!(err, "FOO_KEY is unset or empty");
        let lookups = crate::adapters::openai::take_catalog_lookups();
        assert!(
            lookups.is_empty(),
            "named unset api_key_env must not call catalog, got {lookups:?}"
        );
    }

    #[test]
    fn mcp_named_api_key_env_unset_names_the_var() {
        let route = mcp_resolve_key_route(
            Some("FOO_KEY"),
            None,
            Some("sk-openai".to_owned()),
            Some("sk-or".to_owned()),
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "openai",
        );
        assert_eq!(
            route.key, None,
            "named api_key_env must not fall back to OPENAI_API_KEY / XAI_API_KEY"
        );
        let err = mcp_missing_key_error(Some("FOO_KEY"));
        assert_eq!(err, "FOO_KEY is unset or empty");
        assert!(
            !err.contains("OPENAI_API_KEY") && !err.contains("XAI_API_KEY"),
            "named api_key_env error must not list fallback env vars: {err}"
        );
    }

    fn route_url(
        provider: &str,
        from_openrouter: bool,
        from_xai: bool,
        from_anthropic: bool,
    ) -> String {
        KeyRoute {
            key: None,
            from_openrouter,
            from_xai,
            from_anthropic,
        }
        .default_base_url(provider)
    }

    #[test]
    fn openai_provider_stays_on_openai_when_only_openrouter_env() {
        assert_eq!(
            route_url("openai", true, false, false),
            "https://api.openai.com/v1",
            "MCP provider openai must not use OpenRouter when only OPENROUTER_API_KEY is set"
        );
        assert_eq!(
            route_url("api.openai.com", true, false, false),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            route_url("", true, false, false),
            "https://openrouter.ai/api/v1",
            "empty provider plus OpenRouter env must keep #116 OpenRouter default"
        );
        assert_eq!(
            route_url("127.0.0.1:1234", false, false, false),
            "http://127.0.0.1:1234/v1",
            "MCP provider 127.0.0.1:1234 without base_url must stay on loopback"
        );
        assert_eq!(
            route_url("localhost:11434", true, false, false),
            "http://localhost:11434/v1"
        );
        assert_eq!(route_url("xai", false, false, false), XAI_BASE_URL);
        assert_eq!(
            route_url("", false, true, false),
            XAI_BASE_URL,
            "empty provider plus XAI_API_KEY must default to api.x.ai"
        );
        assert_eq!(route_url("claude", false, false, false), ANTHROPIC_BASE_URL);
        assert_eq!(
            route_url("", false, false, true),
            ANTHROPIC_BASE_URL,
            "empty provider plus ANTHROPIC_* must default to api.anthropic.com"
        );
        assert_eq!(
            route_url("", false, true, true),
            XAI_BASE_URL,
            "empty provider plus both XAI and Anthropic keys must keep the xAI default"
        );
    }

    #[test]
    fn tools_list_names_probe_model_not_ttft() {
        let list = tools_list();
        let desc = list["tools"][0]["description"].as_str().expect("desc");
        assert_eq!(list["tools"][0]["name"], "probe_model");
        assert!(desc.contains("host-policy"), "{desc}");
        assert!(desc.contains("Not TTFT"), "{desc}");
    }

    #[test]
    fn content_length_round_trip() {
        let original = json!({"jsonrpc":"2.0","id":1,"method":"ping"});
        let mut buf = Vec::new();
        write_message(&mut buf, &original).expect("write");
        let mut cursor = Cursor::new(buf);
        let got = read_message(&mut cursor).expect("read").expect("eof");
        assert_eq!(got, original);
    }

    #[test]
    fn content_length_input_still_accepted() {
        let original = json!({"jsonrpc":"2.0","id":1,"method":"ping"});
        let body = serde_json::to_vec(&original).expect("json");
        let mut framed = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        framed.extend_from_slice(&body);
        let mut cursor = Cursor::new(framed);
        let got = read_message(&mut cursor).expect("read").expect("eof");
        assert_eq!(got, original);
    }

    #[test]
    fn content_length_too_large_is_error() {
        let mut cursor = Cursor::new(b"Content-Length: 999999999\r\n\r\n");
        let err = read_message(&mut cursor).expect_err("cap");
        assert!(err.contains("too large"), "{err}");
    }

    #[test]
    fn ndjson_initialize_line_is_accepted() {
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let mut cursor = Cursor::new(format!("{line}\n"));
        let got = read_message(&mut cursor).expect("read").expect("eof");
        assert_eq!(got["method"], "initialize");
        assert_eq!(got["id"], 1);
    }

    #[test]
    fn ndjson_skips_empty_lines() {
        let mut cursor = Cursor::new("\n\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n");
        let got = read_message(&mut cursor).expect("read").expect("eof");
        assert_eq!(got["method"], "ping");
    }

    #[test]
    fn write_message_is_ndjson() {
        let original = json!({"jsonrpc":"2.0","id":1,"result":{}});
        let mut buf = Vec::new();
        write_message(&mut buf, &original).expect("write");
        let s = String::from_utf8(buf).expect("utf8");
        assert!(s.ends_with('\n'), "{s:?}");
        assert!(
            !s.contains("Content-Length"),
            "MCP 2024-11-05 writes json+newline, not LSP headers: {s:?}"
        );
        let parsed: Value = serde_json::from_str(s.trim_end()).expect("json");
        assert_eq!(parsed, original);
    }

    #[test]
    fn ndjson_line_too_large_is_error() {
        let mut huge = "{".to_string();
        huge.push_str(&"x".repeat(8 * 1024 * 1024));
        huge.push('\n');
        let mut cursor = Cursor::new(huge);
        let err = read_message(&mut cursor).expect_err("cap");
        assert!(err.contains("too large"), "{err}");
    }

    #[test]
    fn expand_tilde_joins_home_for_export_dir() {
        let home = dirs::home_dir().expect("home");
        assert_eq!(
            expand_tilde(PathBuf::from("~/overlays")),
            home.join("overlays")
        );
        assert_eq!(expand_tilde(PathBuf::from("~")), home);
        assert_eq!(
            expand_tilde(PathBuf::from("/tmp/overlays")),
            PathBuf::from("/tmp/overlays")
        );
    }

    #[test]
    fn default_cache_path_joins_dirs_cache() {
        let expected = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("canact")
            .join("probes.json");
        assert_eq!(default_cache_path(), expected);
    }
}
