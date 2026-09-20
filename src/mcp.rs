//! Stdio MCP server. Tool name `probe_model` matches Jwrede/llmprobe;
//! the payload is canact host-policy JSON, not TTFT.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::{
    CatalogPriors, HostPolicyMeta, KeyRoute, OpenAiCompatClient, ProbeCache, ProbeError,
    ProbeRunner, SuiteTier, claude_code_access_token, finalize_key_route,
    is_anthropic_provider_label, is_bedrock_provider_label, is_groq_provider_label, looks_cheap,
    refuse_cloud_without_key, resolve_api_key_from, resolve_host_catalog,
    should_load_claude_code_login, should_load_xai_oauth, uses_xai_credentials,
    xai_oauth_access_token,
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
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let api_key_env = trim_api_key_env(args.get("api_key_env").and_then(Value::as_str));
    let named_key = mcp_named_or_route_key(api_key_env, provider_given);
    let openai = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let openrouter = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let has_explicit_base = args
        .get("base_url")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    let skip_oauth = api_key_env.is_some_and(|v| !v.is_empty());
    let other_before_xai = openai.is_some() || openrouter.is_some() || named_key.is_some();
    let xai = if skip_oauth {
        std::env::var("XAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("GROK_API_KEY").ok().filter(|s| !s.is_empty()))
    } else {
        xai_key_for_route(provider_given, other_before_xai, has_explicit_base)
    };
    let other_cloud_keys =
        openai.is_some() || openrouter.is_some() || xai.is_some() || named_key.is_some();
    let anthropic = if skip_oauth {
        None
    } else {
        anthropic_key_for_route(provider_given, other_cloud_keys, has_explicit_base)
    };
    let route = mcp_resolve_key_route(
        api_key_env,
        named_key,
        openai,
        openrouter,
        xai,
        anthropic,
        provider_given,
    );
    probe_model_with_route(args, route, api_key_env).await
}

async fn probe_model_with_route(
    args: &Value,
    first: KeyRoute,
    api_key_env: Option<&str>,
) -> Result<Value, String> {
    let api_key_env = trim_api_key_env(api_key_env);
    let model = args
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "model is required".to_owned())?
        .to_owned();
    let advertised = match args.get("advertised_context") {
        None => None,
        Some(v) => match json_u32(Some(v)) {
            Some(n) if n >= 1 => Some(n),
            _ => return Err("advertised_context must be >= 1".to_owned()),
        },
    };
    let cheap = present_json_bool(args, "cheap")?.unwrap_or(false);
    let full = present_json_bool(args, "full")?.unwrap_or(false);
    let suite = match args.get("suite") {
        None if full => SuiteTier::Full,
        None => SuiteTier::Policy,
        Some(v) => {
            let parsed = v
                .as_str()
                .map(str::trim)
                .and_then(SuiteTier::parse)
                .ok_or_else(|| {
                    format!(
                        "unknown suite={} (expected policy, full, or all)",
                        json_label(v)
                    )
                })?;
            if cheap && parsed != SuiteTier::Policy {
                return Err(format!("cheap conflicts with suite={}", json_label(v)));
            }
            if full && parsed != SuiteTier::Full {
                return Err(format!("full conflicts with suite={}", json_label(v)));
            }
            parsed
        }
    };
    let vision_flag = present_json_bool(args, "vision")?;
    let vision = vision_flag.unwrap_or(false);
    let force = present_json_bool(args, "force")?.unwrap_or(false);
    let cache_path = match args.get("cache").and_then(Value::as_str) {
        None => default_cache_path(),
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                default_cache_path()
            } else {
                expand_tilde(PathBuf::from(trimmed))
            }
        }
    };
    let mut cache = ProbeCache::load(&cache_path)
        .map_err(|e| format!("failed to load cache {}: {e}", cache_path.display()))?;

    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let explicit_base_url = args
        .get("base_url")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let (route, base_url, provider) =
        finalize_key_route(provider_given, explicit_base_url, first, |provider| {
            let named_key = mcp_named_or_route_key(api_key_env, provider);
            let openai = std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|s| !s.is_empty());
            let openrouter = std::env::var("OPENROUTER_API_KEY")
                .ok()
                .filter(|s| !s.is_empty());
            let skip_oauth = api_key_env.is_some_and(|v| !v.is_empty());
            let other_before_xai = openai.is_some() || openrouter.is_some() || named_key.is_some();
            let xai = if skip_oauth {
                std::env::var("XAI_API_KEY")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .or_else(|| std::env::var("GROK_API_KEY").ok().filter(|s| !s.is_empty()))
            } else {
                xai_key_for_route(provider, other_before_xai, false)
            };
            let other_cloud_keys =
                openai.is_some() || openrouter.is_some() || xai.is_some() || named_key.is_some();
            let anthropic = if skip_oauth {
                None
            } else {
                anthropic_key_for_route(provider, other_cloud_keys, false)
            };
            mcp_resolve_key_route(
                api_key_env,
                named_key,
                openai,
                openrouter,
                xai,
                anthropic,
                provider,
            )
        });
    let api_key = route.key.clone();
    if !force {
        if let Some(profile) = cache.get_with_suite(&model, &provider, suite, vision, advertised) {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true, true, suite, advertised,
            )));
        }
        if advertised.is_none()
            && vision_flag.is_none()
            && let Some((profile, _cheap_row, stored_advertised)) =
                cache.find_profile_unspecified_catalog_suite(&model, &provider, suite)
        {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true,
                true,
                suite,
                stored_advertised,
            )));
        }
        if matches!(suite, SuiteTier::Policy)
            && !vision
            && let Some((profile, hit_suite)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
        {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true, true, hit_suite, advertised,
            )));
        }
    }
    if refuse_cloud_without_key(api_key.as_deref(), &base_url) {
        return Err(mcp_missing_key_error(api_key_env, &provider));
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
        if matches!(suite, SuiteTier::Policy)
            && !vision
            && let Some((profile, hit_suite)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
        {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
                true, true, hit_suite, advertised,
            )));
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

fn trim_api_key_env(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|s| !s.is_empty())
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
    match trim_api_key_env(api_key_env) {
        Some(var) => KeyRoute {
            key: named_key,
            from_openrouter: var == "OPENROUTER_API_KEY",
            from_xai: var == "XAI_API_KEY" || var == "GROK_API_KEY",
            from_anthropic: var == "ANTHROPIC_AUTH_TOKEN" || var == "ANTHROPIC_API_KEY",
        },
        None if is_groq_provider_label(provider) || is_bedrock_provider_label(provider) => {
            resolve_api_key_from(named_key, None, None, None, None, provider)
        }
        None => resolve_api_key_from(None, openai, openrouter, xai, anthropic, provider),
    }
}

fn mcp_named_or_route_key(api_key_env: Option<&str>, provider: &str) -> Option<String> {
    match trim_api_key_env(api_key_env) {
        Some(var) => std::env::var(var).ok().filter(|s| !s.is_empty()),
        None if is_groq_provider_label(provider) => {
            std::env::var("GROQ_API_KEY").ok().filter(|s| !s.is_empty())
        }
        None if is_bedrock_provider_label(provider) => std::env::var("AWS_BEARER_TOKEN_BEDROCK")
            .ok()
            .filter(|s| !s.is_empty()),
        None => None,
    }
}

/// Named `api_key_env` does not fall back to OPENAI_API_KEY / XAI_API_KEY.
fn mcp_missing_key_error(api_key_env: Option<&str>, provider: &str) -> String {
    match trim_api_key_env(api_key_env) {
        Some(var) => format!("{var} is unset or empty"),
        _ if uses_xai_credentials(provider) => {
            "set api_key_env or XAI_API_KEY for xAI (OPENAI_API_KEY is not sent)".to_owned()
        }
        _ if is_anthropic_provider_label(provider) => {
            "set api_key_env, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY for Anthropic (OPENAI_API_KEY is not sent)".to_owned()
        }
        _ if is_groq_provider_label(provider) => {
            "set api_key_env or GROQ_API_KEY for Groq (OPENAI_API_KEY is not sent)".to_owned()
        }
        _ if is_bedrock_provider_label(provider) => {
            "set api_key_env or AWS_BEARER_TOKEN_BEDROCK for Amazon Bedrock (OPENAI_API_KEY is not sent)".to_owned()
        }
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
}

fn xai_key_for_route(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> Option<String> {
    std::env::var("XAI_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("GROK_API_KEY").ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            if should_load_xai_oauth(provider, other_cloud_keys, explicit_base_url) {
                xai_oauth_access_token()
            } else {
                None
            }
        })
}

fn anthropic_key_for_route(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> Option<String> {
    anthropic_env_key().or_else(|| {
        if should_load_claude_code_login(provider, other_cloud_keys, explicit_base_url) {
            claude_code_access_token()
        } else {
            None
        }
    })
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
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    path
}

fn json_bool(v: Option<&Value>) -> Option<bool> {
    let v = v?;
    v.as_bool()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

fn present_json_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None => Ok(None),
        Some(v) => json_bool(Some(v))
            .map(Some)
            .ok_or_else(|| format!("{key} must be a boolean")),
    }
}

fn json_label(v: &Value) -> String {
    match v.as_str() {
        Some(s) => s.to_owned(),
        None => v.to_string(),
    }
}

fn json_u32(v: Option<&Value>) -> Option<u32> {
    let v = v?;
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
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
    use crate::{
        ANTHROPIC_BASE_URL, BEDROCK_BASE_URL, CapabilityProfile, GROQ_BASE_URL, XAI_BASE_URL,
    };
    use std::ffi::OsString;
    use std::io::Cursor;
    use std::sync::{Mutex, MutexGuard};

    fn mcp_empty_route() -> KeyRoute {
        KeyRoute {
            key: None,
            from_openrouter: false,
            from_xai: false,
            from_anthropic: false,
        }
    }

    struct IsolatedApiKeyEnv {
        _lock: MutexGuard<'static, ()>,
        openai: Option<OsString>,
        grok: Option<OsString>,
    }

    impl IsolatedApiKeyEnv {
        fn set_openai(value: &str) -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let openai = std::env::var_os("OPENAI_API_KEY");
            let grok = std::env::var_os("GROK_API_KEY");
            unsafe {
                std::env::set_var("OPENAI_API_KEY", value);
                std::env::remove_var("GROK_API_KEY");
            }
            Self {
                _lock: lock,
                openai,
                grok,
            }
        }
    }

    impl Drop for IsolatedApiKeyEnv {
        fn drop(&mut self) {
            unsafe {
                match &self.openai {
                    Some(v) => std::env::set_var("OPENAI_API_KEY", v),
                    None => std::env::remove_var("OPENAI_API_KEY"),
                }
                match &self.grok {
                    Some(v) => std::env::set_var("GROK_API_KEY", v),
                    None => std::env::remove_var("GROK_API_KEY"),
                }
            }
        }
    }

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
        assert_eq!(err, mcp_missing_key_error(None, "openai"));
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
        let err = mcp_missing_key_error(Some("FOO_KEY"), "openai");
        assert_eq!(err, "FOO_KEY is unset or empty");
        assert!(
            !err.contains("OPENAI_API_KEY") && !err.contains("XAI_API_KEY"),
            "named api_key_env error must not list fallback env vars: {err}"
        );
    }

    #[test]
    fn mcp_padded_api_key_env_looks_up_trimmed_name() {
        let _env = IsolatedApiKeyEnv::set_openai("sk-mcp-padded-lookup");
        let named = mcp_named_or_route_key(Some(" OPENAI_API_KEY "), "openai");
        assert_eq!(named.as_deref(), Some("sk-mcp-padded-lookup"));
    }

    #[test]
    fn mcp_whitespace_only_api_key_env_is_omitted() {
        let route = mcp_resolve_key_route(
            Some("   "),
            None,
            Some("sk-openai".to_owned()),
            Some("sk-or".to_owned()),
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "openai",
        );
        assert_eq!(
            route.key.as_deref(),
            Some("sk-openai"),
            "whitespace-only api_key_env must omit, not name a padded env var"
        );
        let err = mcp_missing_key_error(Some("   "), "openai");
        assert_eq!(err, mcp_missing_key_error(None, "openai"));
        assert!(
            !err.contains("is unset or empty"),
            "whitespace-only api_key_env must not say the padded name is unset: {err}"
        );
    }

    #[test]
    fn mcp_padded_api_key_env_sets_from_flags_like_unpadded() {
        let padded_or = mcp_resolve_key_route(
            Some(" OPENROUTER_API_KEY "),
            Some("sk-or".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        let unpadded_or = mcp_resolve_key_route(
            Some("OPENROUTER_API_KEY"),
            Some("sk-or".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        assert_eq!(padded_or.from_openrouter, unpadded_or.from_openrouter);
        assert!(padded_or.from_openrouter);
        assert!(!padded_or.from_xai);
        assert!(!padded_or.from_anthropic);

        let padded_xai = mcp_resolve_key_route(
            Some(" XAI_API_KEY "),
            Some("xai-named".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        let unpadded_xai = mcp_resolve_key_route(
            Some("XAI_API_KEY"),
            Some("xai-named".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        assert_eq!(padded_xai.from_xai, unpadded_xai.from_xai);
        assert!(padded_xai.from_xai);

        let padded_ant = mcp_resolve_key_route(
            Some(" ANTHROPIC_API_KEY "),
            Some("sk-ant".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        let unpadded_ant = mcp_resolve_key_route(
            Some("ANTHROPIC_API_KEY"),
            Some("sk-ant".to_owned()),
            None,
            None,
            None,
            None,
            "",
        );
        assert_eq!(padded_ant.from_anthropic, unpadded_ant.from_anthropic);
        assert!(padded_ant.from_anthropic);
    }

    #[tokio::test]
    async fn mcp_padded_api_key_env_missing_key_names_trimmed_var() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let route = mcp_resolve_key_route(
            Some(" FOO_KEY "),
            None,
            Some("sk-openai".to_owned()),
            Some("sk-or".to_owned()),
            Some("xai-env".to_owned()),
            Some("sk-ant".to_owned()),
            "openai",
        );
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "force": true,
            "cache": cache_path.to_str().expect("utf8"),
            "api_key_env": " FOO_KEY ",
        });
        let err = probe_model_with_route(&args, route, Some(" FOO_KEY "))
            .await
            .unwrap_err();
        assert_eq!(err, "FOO_KEY is unset or empty");
    }

    #[tokio::test]
    async fn mcp_advertised_context_zero_is_refused() {
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "advertised_context": 0,
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .unwrap_err();
        assert_eq!(err, "advertised_context must be >= 1");
    }

    #[tokio::test]
    async fn mcp_advertised_context_present_invalid_is_refused() {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        cache.put_with_suite(
            CapabilityProfile::unprobed("gpt-4o", "openai"),
            SuiteTier::Policy,
            false,
            None,
        );
        cache.save(&cache_path).expect("save");
        let cache_str = cache_path.to_str().expect("utf8");
        for advertised in [json!(-1), json!(0.5), json!("nope"), json!(null)] {
            let args = json!({
                "model": "gpt-4o",
                "provider": "openai",
                "cache": cache_str,
                "advertised_context": advertised,
            });
            let err = probe_model_with_route(&args, mcp_empty_route(), None)
                .await
                .unwrap_err();
            assert_eq!(
                err, "advertised_context must be >= 1",
                "present {advertised} must not omit"
            );
        }
    }

    #[tokio::test]
    async fn mcp_advertised_context_padded_string_parses() {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        let mut omitted = CapabilityProfile::unprobed("gpt-4o", "openai");
        omitted.effective_context_tokens = Some(111);
        cache.put_with_suite(omitted, SuiteTier::Policy, false, None);
        let mut padded = CapabilityProfile::unprobed("gpt-4o", "openai");
        padded.effective_context_tokens = Some(4096);
        cache.put_with_suite(padded, SuiteTier::Policy, false, Some(4096));
        cache.save(&cache_path).expect("save");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_path.to_str().expect("utf8"),
            "advertised_context": " 4096 ",
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("padded advertised_context must hit ctx4096");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["advertisedContextTokens"], 4096, "{envelope}");
    }

    #[tokio::test]
    async fn mcp_advertised_context_omitted_stays_none() {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        cache.put_with_suite(
            CapabilityProfile::unprobed("gpt-4o", "openai"),
            SuiteTier::Policy,
            false,
            None,
        );
        cache.save(&cache_path).expect("save");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_path.to_str().expect("utf8"),
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("omitted advertised_context is a cache hit");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(
            envelope["advertisedContextTokens"],
            Value::Null,
            "{envelope}"
        );
    }

    fn seed_openai_cache(
        suite: SuiteTier,
        vision: bool,
        advertised: Option<u32>,
        tokens: Option<u32>,
    ) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        let mut profile = CapabilityProfile::unprobed("gpt-4o", "openai");
        profile.effective_context_tokens = tokens;
        cache.put_with_suite(profile, suite, vision, advertised);
        cache.save(&cache_path).expect("save");
        (dir, cache_path.to_str().expect("utf8").to_owned())
    }

    struct IsolatedHome {
        _lock: MutexGuard<'static, ()>,
        prev_home: Option<OsString>,
        prev_userprofile: Option<OsString>,
        prev_xdg: Option<OsString>,
        _dir: tempfile::TempDir,
    }

    impl IsolatedHome {
        fn new() -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let dir = tempfile::tempdir().expect("temp home");
            let prev_home = std::env::var_os("HOME");
            let prev_userprofile = std::env::var_os("USERPROFILE");
            let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
            let xdg = dir.path().join("cache");
            unsafe {
                std::env::set_var("HOME", dir.path());
                std::env::set_var("USERPROFILE", dir.path());
                std::env::set_var("XDG_CACHE_HOME", &xdg);
            }
            Self {
                _lock: lock,
                prev_home,
                prev_userprofile,
                prev_xdg,
                _dir: dir,
            }
        }
    }

    impl Drop for IsolatedHome {
        fn drop(&mut self) {
            unsafe {
                match &self.prev_home {
                    Some(v) => std::env::set_var("HOME", v),
                    None => std::env::remove_var("HOME"),
                }
                match &self.prev_userprofile {
                    Some(v) => std::env::set_var("USERPROFILE", v),
                    None => std::env::remove_var("USERPROFILE"),
                }
                match &self.prev_xdg {
                    Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                    None => std::env::remove_var("XDG_CACHE_HOME"),
                }
            }
        }
    }

    #[tokio::test]
    async fn mcp_cheap_true_conflicts_with_suite_full() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Full, false, None, Some(999));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
            "cheap": true,
            "suite": "full",
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .unwrap_err();
        assert!(
            err.contains("cheap") && err.contains("suite"),
            "cheap+suite=full must name cheap vs suite: {err}"
        );
    }

    #[tokio::test]
    async fn mcp_full_true_conflicts_with_suite_policy() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Policy, false, None, Some(111));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
            "full": true,
            "suite": "policy",
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .unwrap_err();
        assert!(
            err.contains("full") && err.contains("suite"),
            "full+suite=policy must name full vs suite: {err}"
        );
    }

    #[tokio::test]
    async fn mcp_padded_suite_parses_full() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Full, false, None, Some(999));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
            "suite": " full ",
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("padded suite must parse Full");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["suite"], "full", "{envelope}");
    }

    #[tokio::test]
    async fn mcp_present_suite_non_string_is_refused() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Policy, false, None, Some(111));
        for suite in [json!(null), json!(true), json!(1)] {
            let args = json!({
                "model": "gpt-4o",
                "provider": "openai",
                "cache": cache_str,
                "suite": suite,
            });
            let err = probe_model_with_route(&args, mcp_empty_route(), None)
                .await
                .unwrap_err();
            assert!(
                err.contains("unknown suite"),
                "present {suite} must not omit to policy: {err}"
            );
        }
    }

    #[tokio::test]
    async fn mcp_present_invalid_bools_are_refused() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Policy, false, None, Some(111));
        for (key, value) in [
            ("cheap", json!("yes")),
            ("cheap", json!(2)),
            ("cheap", json!(null)),
            ("full", json!("yes")),
            ("vision", json!("yes")),
            ("vision", json!(null)),
            ("force", json!("yes")),
        ] {
            let mut args = json!({
                "model": "gpt-4o",
                "provider": "openai",
                "cache": cache_str,
            });
            args[key] = value.clone();
            let err = probe_model_with_route(&args, mcp_empty_route(), None)
                .await
                .unwrap_err();
            assert!(
                err.contains(key),
                "present {key}={value} must name {key}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn mcp_padded_cheap_true_conflicts_with_suite_full() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Full, false, None, Some(999));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
            "cheap": " true ",
            "suite": "full",
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .unwrap_err();
        assert!(
            err.contains("cheap") && err.contains("suite"),
            "padded cheap true must parse and conflict with suite=full: {err}"
        );
    }

    #[tokio::test]
    async fn mcp_omitted_cheap_stays_policy() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Policy, false, None, Some(111));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("omitted cheap is policy");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["suite"], "policy", "{envelope}");
    }

    #[tokio::test]
    async fn mcp_cheap_and_full_without_suite_is_full() {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        let mut policy = CapabilityProfile::unprobed("gpt-4o", "openai");
        policy.effective_context_tokens = Some(111);
        cache.put_with_suite(policy, SuiteTier::Policy, false, None);
        let mut full = CapabilityProfile::unprobed("gpt-4o", "openai");
        full.effective_context_tokens = Some(999);
        cache.put_with_suite(full, SuiteTier::Full, false, None);
        cache.save(&cache_path).expect("save");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_path.to_str().expect("utf8"),
            "cheap": true,
            "full": true,
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("cheap+full with no suite is Full");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["suite"], "full", "{envelope}");
    }

    #[tokio::test]
    async fn mcp_whitespace_cache_uses_default_path() {
        let _home = IsolatedHome::new();
        let cache_path = default_cache_path();
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent).expect("cache dir");
        }
        let mut cache = ProbeCache::default();
        let mut profile = CapabilityProfile::unprobed("gpt-4o", "openai");
        profile.effective_context_tokens = Some(4242);
        cache.put_with_suite(profile, SuiteTier::Policy, false, None);
        cache.save(&cache_path).expect("save");
        for cache_arg in ["", "   "] {
            let args = json!({
                "model": "gpt-4o",
                "provider": "openai",
                "cache": cache_arg,
            });
            let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
                .await
                .unwrap_or_else(|err| {
                    panic!("whitespace cache {cache_arg:?} must use default: {err}")
                });
            assert_eq!(envelope["fromCache"], true, "{cache_arg:?} {envelope}");
            assert_eq!(
                envelope["effectiveContextTokens"], 4242,
                "{cache_arg:?} {envelope}"
            );
        }
    }

    #[tokio::test]
    async fn mcp_vision_omitted_uses_catalog_cache() {
        let (_dir, cache_str) = seed_openai_cache(SuiteTier::Policy, true, None, Some(777));
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": cache_str,
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None)
            .await
            .expect("omitted vision uses catalog cache");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["effectiveContextTokens"], 777, "{envelope}");
    }

    #[test]
    fn mcp_named_grok_api_key_env_sets_from_xai() {
        let route = mcp_resolve_key_route(
            Some("GROK_API_KEY"),
            Some("xai-from-grok-env".to_owned()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "",
        );
        assert_eq!(route.key.as_deref(), Some("xai-from-grok-env"));
        assert!(route.from_xai);
        assert_eq!(route.default_base_url(""), XAI_BASE_URL);

        let xai = mcp_resolve_key_route(
            Some("XAI_API_KEY"),
            Some("xai-named".into()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "",
        );
        assert_eq!(xai.key.as_deref(), Some("xai-named"));
        assert!(xai.from_xai);
        assert_eq!(xai.default_base_url(""), XAI_BASE_URL);
    }

    #[test]
    fn mcp_named_groq_and_bedrock_ignore_openai_and_name_route_env() {
        let groq = mcp_resolve_key_route(
            None,
            Some("gsk-test".to_owned()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "groq",
        );
        assert_eq!(groq.key.as_deref(), Some("gsk-test"));
        assert!(!groq.from_xai);
        assert_eq!(groq.default_base_url("groq"), GROQ_BASE_URL);
        let groq_openai = mcp_resolve_key_route(
            None,
            None,
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "groq",
        );
        assert!(
            groq_openai.key.is_none(),
            "provider=groq must not send OPENAI_API_KEY"
        );
        let groq_err = mcp_missing_key_error(None, "groq");
        assert!(groq_err.contains("GROQ_API_KEY"), "{groq_err}");
        assert!(
            !groq_err.contains("set api_key_env (or OPENAI_API_KEY"),
            "Groq missing-key must not list OPENAI_API_KEY as the fix: {groq_err}"
        );

        let bedrock = mcp_resolve_key_route(
            None,
            Some("bedrock-token".to_owned()),
            Some("sk-openai".to_owned()),
            None,
            None,
            None,
            "amazon-bedrock",
        );
        assert_eq!(bedrock.key.as_deref(), Some("bedrock-token"));
        assert_eq!(bedrock.default_base_url("amazon-bedrock"), BEDROCK_BASE_URL);
        let bedrock_err = mcp_missing_key_error(None, "bedrock");
        assert!(
            bedrock_err.contains("AWS_BEARER_TOKEN_BEDROCK"),
            "{bedrock_err}"
        );
        assert!(
            !bedrock_err.contains("set api_key_env (or OPENAI_API_KEY"),
            "Bedrock missing-key must not list OPENAI_API_KEY as the fix: {bedrock_err}"
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
