//! Stdio MCP server. Tool name `probe_model` matches Jwrede/llmprobe;
//! the payload is canact host-policy JSON, not TTFT.

use std::io::{BufRead, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};

use crate::endpoint::mcp_tool_base_url_is_loopback;
use crate::{
    CatalogPriors, HostPolicyMeta, KeyRoute, OpenAiCompatClient, ProbeCache, ProbeError,
    ProbeRunner, ProbeTool, SuiteTier, claude_code_access_token, finalize_key_route,
    invalid_explicit_base_url, is_anthropic_provider_label, is_bedrock_provider_label,
    is_groq_provider_label, is_openai_codex_provider_label, is_openai_provider_label, looks_cheap,
    openrouter_default_ok, present_base_url, present_secret, refuse_cloud_without_key,
    resolve_api_key_from, resolve_host_catalog, shipped_profile_base_conflict,
    should_load_claude_code_login, should_load_xai_oauth, uses_xai_credentials,
    with_route_error_label, xai_oauth_access_token,
};

const PROTOCOL_VERSION: &str = "2024-11-05";

const MCP_API_KEY_ENV_ALLOWLIST: &[&str] = &[
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
];

/// Trust flags for the stdio MCP server. The default is strict.
#[derive(Clone, Debug, Default)]
pub struct McpServerOptions {
    /// Let the tool pass a cache path outside the default cache directory.
    pub allow_cache: bool,
    /// Let the tool pass a base URL that is not a loopback host.
    pub allow_base_url: bool,
    /// Env var the tool may read. The tool cannot name a different variable.
    pub api_key_env: Option<String>,
    /// Base URL for probes. The tool cannot replace it.
    pub base_url: Option<String>,
}

/// Serve MCP over stdin/stdout until EOF. Returns a process exit code.
pub fn run_mcp_stdio() -> u8 {
    run_mcp_stdio_with(McpServerOptions::default())
}

/// Serve MCP over stdin/stdout with the human's trust flags.
pub fn run_mcp_stdio_with(options: McpServerOptions) -> u8 {
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
            "tools/call" => match handle_tools_call(&params, &options) {
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
                    "advertised_context": { "type": "integer", "minimum": 1 },
                    "tools": {
                        "type": "array",
                        "description": "Caller tools. Each item has name, description, and parameters. Omit for builtin probe tools.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "description": { "type": "string" },
                                "parameters": { "description": "JSON Schema for the tool arguments" }
                            },
                            "required": ["name", "description", "parameters"]
                        }
                    }
                },
                "required": ["model"]
            }
        }]
    })
}

fn handle_tools_call(params: &Value, policy: &McpServerOptions) -> Result<Value, String> {
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
    let envelope = rt.block_on(probe_model_args(&args, policy))?;
    let text = serde_json::to_string_pretty(&envelope).map_err(|e| e.to_string())?;
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false
    }))
}

async fn probe_model_args(args: &Value, policy: &McpServerOptions) -> Result<Value, String> {
    parse_mcp_tools(args.get("tools"))?;
    reject_invalid_explicit_base_urls(args, policy)?;
    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let tool_env = trim_api_key_env(args.get("api_key_env").and_then(Value::as_str));
    let (effective_env, effective_url, tool_supplied) =
        enforce_mcp_tool_policy(tool_env, mcp_present_base_url(args), policy)?;
    let route = load_mcp_route(
        provider_given,
        effective_env.as_deref(),
        tool_supplied,
        !tool_supplied && effective_url.is_some(),
        policy,
    )
    .map_err(|msg| format!("authentication error: {msg}"))?;
    probe_model_with_route(args, route, tool_env, policy).await
}

async fn probe_model_with_route(
    args: &Value,
    first: KeyRoute,
    api_key_env: Option<&str>,
    policy: &McpServerOptions,
) -> Result<Value, String> {
    let caller_tools = parse_mcp_tools(args.get("tools"))?;
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
    reject_invalid_explicit_base_urls(args, policy)?;
    let (effective_env, effective_url, tool_supplied) =
        enforce_mcp_tool_policy(api_key_env, mcp_present_base_url(args), policy)?;
    let cache_path = opened_mcp_cache_path(&mcp_cache_path(args), policy)?;
    let mut cache = ProbeCache::load(&cache_path)
        .map_err(|e| format!("failed to load cache {}: {e}", cache_path.display()))?;

    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    let (route, base_url, provider) =
        finalize_key_route(provider_given, effective_url.clone(), first, |provider| {
            load_mcp_route(
                provider,
                effective_env.as_deref(),
                tool_supplied,
                !tool_supplied && effective_url.is_some(),
                policy,
            )
        })
        .map_err(with_route_error_label)?;
    if let Some(msg) = shipped_profile_base_conflict(&provider, &base_url) {
        return Err(msg.to_owned());
    }
    let api_key = route.key.clone();
    if !force {
        if let Some(profile) = cache.get_with_suite_tools(
            &model,
            &provider,
            suite,
            vision,
            advertised,
            caller_tools.as_deref(),
        ) {
            return Ok(profile.host_policy_envelope_with(mcp_host_meta(
                suite,
                advertised,
                caller_tools.as_deref(),
            )));
        }
        if caller_tools.is_none()
            && advertised.is_none()
            && vision_flag.is_none()
            && let Some((profile, _cheap_row, stored_advertised)) =
                cache.find_profile_unspecified_catalog_suite(&model, &provider, suite)
        {
            return Ok(profile.host_policy_envelope_with(mcp_host_meta(
                suite,
                stored_advertised,
                None,
            )));
        }
        if caller_tools.is_none()
            && matches!(suite, SuiteTier::Policy)
            && !vision
            && let Some((profile, hit_suite)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
        {
            return Ok(
                profile.host_policy_envelope_with(mcp_host_meta(hit_suite, advertised, None))
            );
        }
    }
    if refuse_cloud_without_key(api_key.as_deref(), &base_url) {
        return Err(mcp_refusal_key_error(
            effective_env.as_deref(),
            &provider,
            policy,
        ));
    }
    // Catalog and probe futures embed wiremux client state. Inlined, this
    // future is ~217KB and overflows the Windows 1MB test stack before a
    // refused base_url can return.
    let hints = Box::pin(resolve_host_catalog(
        advertised,
        vision_flag,
        &base_url,
        api_key.as_deref(),
        &model,
    ))
    .await;
    let advertised = hints.advertised_context_tokens;
    let vision = hints.supports_vision == Some(true);
    if !force {
        if let Some(profile) = cache.get_with_suite_tools(
            &model,
            &provider,
            suite,
            vision,
            advertised,
            caller_tools.as_deref(),
        ) {
            return Ok(profile.host_policy_envelope_with(mcp_host_meta(
                suite,
                advertised,
                caller_tools.as_deref(),
            )));
        }
        if caller_tools.is_none()
            && matches!(suite, SuiteTier::Policy)
            && !vision
            && let Some((profile, hit_suite)) =
                cache.find_profile_with_cost_and_advertised(&model, &provider, advertised)
        {
            return Ok(
                profile.host_policy_envelope_with(mcp_host_meta(hit_suite, advertised, None))
            );
        }
    }
    let catalog = CatalogPriors {
        advertised_context_tokens: advertised,
        supports_vision: hints.supports_vision,
        supports_tools: None,
    };
    let throttle = looks_cheap(&provider, &model, &base_url);
    let run = Box::pin(async move {
        let client = OpenAiCompatClient::new(base_url, api_key, model, provider, catalog)
            .map_err(|e| e.to_string())?;
        let mut runner = ProbeRunner::new(client).suite(suite);
        if throttle {
            runner = runner.throttled();
        }
        if let Some(list) = caller_tools {
            runner = runner.with_tools(list);
        }
        runner.run_detailed().await.map_err(|e| match e {
            ProbeError::Auth(msg) => format!("authentication error: {msg}"),
            other => other.to_string(),
        })
    })
    .await?;
    if let Err(err) = run.persist(&mut cache, &cache_path) {
        eprintln!("warning: failed to save probe cache: {err}");
    }
    Ok(run.host_policy_envelope())
}

#[derive(serde::Deserialize)]
struct McpCallerTool {
    name: String,
    description: String,
    parameters: Value,
}

fn parse_mcp_tools(raw: Option<&Value>) -> Result<Option<Vec<ProbeTool>>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let Some(items) = raw.as_array() else {
        return Err("failed to parse tools: expected a JSON array of tools".to_owned());
    };
    for (index, item) in items.iter().enumerate() {
        if !item.is_object() {
            return Err(format!(
                "failed to parse tools: item {index} must be an object with name, description, and parameters"
            ));
        }
    }
    let rows: Vec<McpCallerTool> = serde_json::from_value(raw.clone())
        .map_err(|err| format!("failed to parse tools: {err}"))?;
    Ok(Some(
        rows.into_iter()
            .map(|row| ProbeTool {
                name: row.name,
                description: row.description,
                parameters: row.parameters,
            })
            .collect(),
    ))
}

fn mcp_host_meta(
    suite: SuiteTier,
    advertised: Option<u32>,
    tools: Option<&[ProbeTool]>,
) -> HostPolicyMeta {
    let meta = HostPolicyMeta::for_suite(true, true, suite, advertised);
    match tools {
        Some(list) => meta.with_tool_digest(crate::tool_digest::probe_tools_digest(list)),
        None => meta,
    }
}

fn trim_api_key_env(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|s| !s.is_empty())
}

fn mcp_present_base_url(args: &Value) -> Option<&str> {
    present_base_url(args.get("base_url").and_then(Value::as_str))
}

fn reject_invalid_explicit_base_urls(
    args: &Value,
    policy: &McpServerOptions,
) -> Result<(), String> {
    let tool_raw = args.get("base_url").and_then(Value::as_str);
    if let Some(msg) = invalid_explicit_base_url(tool_raw) {
        return Err(msg);
    }
    if present_base_url(tool_raw).is_none()
        && let Some(msg) = invalid_explicit_base_url(policy.base_url.as_deref())
    {
        return Err(msg);
    }
    Ok(())
}

fn api_key_env_allowed(name: &str) -> bool {
    MCP_API_KEY_ENV_ALLOWLIST.contains(&name)
}

fn pinned_api_key_env(policy: &McpServerOptions) -> Option<&str> {
    policy
        .api_key_env
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn pinned_base_url(policy: &McpServerOptions) -> Option<&str> {
    policy
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn enforce_mcp_tool_policy(
    tool_env: Option<&str>,
    tool_url: Option<&str>,
    policy: &McpServerOptions,
) -> Result<(Option<String>, Option<String>, bool), String> {
    let effective_env = permitted_api_key_env(tool_env, policy)?;
    let (effective_url, tool_supplied) = resolve_effective_base_url(tool_url, policy)?;
    Ok((effective_env, effective_url, tool_supplied))
}

fn permitted_api_key_env(
    tool_name: Option<&str>,
    policy: &McpServerOptions,
) -> Result<Option<String>, String> {
    let tool = trim_api_key_env(tool_name);
    let pin = pinned_api_key_env(policy);
    match (tool, pin) {
        (Some(tool), Some(pin)) if tool != pin => {
            Err("api_key_env is pinned by the server".to_owned())
        }
        (Some(name), Some(_)) | (None, Some(name)) => Ok(Some(name.to_owned())),
        (Some(name), None) if api_key_env_allowed(name) => Ok(Some(name.to_owned())),
        (Some(_), None) => Err("api_key_env name not allowed".to_owned()),
        (None, None) => Ok(None),
    }
}

fn resolve_effective_base_url(
    tool_url: Option<&str>,
    policy: &McpServerOptions,
) -> Result<(Option<String>, bool), String> {
    let tool = tool_url.map(str::trim).filter(|s| !s.is_empty());
    let pin = pinned_base_url(policy);
    match (tool, pin) {
        (Some(tool), Some(pin)) if tool != pin => {
            Err("base_url is pinned by the server".to_owned())
        }
        (_, Some(pin)) => Ok((Some(pin.to_owned()), false)),
        (Some(tool), None) if policy.allow_base_url || mcp_tool_base_url_is_loopback(tool) => {
            Ok((Some(tool.to_owned()), true))
        }
        (Some(_), None) => Err("base_url is not a loopback host".to_owned()),
        (None, None) => Ok((None, false)),
    }
}

fn load_mcp_route(
    provider: &str,
    api_key_env: Option<&str>,
    tool_supplied_url: bool,
    human_base_url: bool,
    policy: &McpServerOptions,
) -> Result<KeyRoute, String> {
    let named_key = read_named_or_route_key(api_key_env, provider, policy);
    // `should_load_*` still loads logins for a named anthropic/xai provider.
    // Skip those helpers for a tool-supplied URL, including finalize.
    // The env vars stay readable. Only the login helpers are skipped.
    let skip_oauth = tool_supplied_url || api_key_env.is_some();
    let openai = mcp_env_nonempty("OPENAI_API_KEY");
    let openrouter = mcp_env_nonempty("OPENROUTER_API_KEY");
    let other_before_xai = openai.is_some() || openrouter.is_some() || named_key.is_some();
    let xai = if skip_oauth {
        mcp_env_nonempty("XAI_API_KEY").or_else(|| mcp_env_nonempty("GROK_API_KEY"))
    } else {
        xai_key_for_route(provider, other_before_xai, human_base_url)?
    };
    let other_cloud_keys =
        openai.is_some() || openrouter.is_some() || xai.is_some() || named_key.is_some();
    let anthropic = if skip_oauth {
        anthropic_env_key()
    } else {
        anthropic_key_for_route(provider, other_cloud_keys, human_base_url)?
    };
    Ok(mcp_resolve_key_route(
        api_key_env,
        named_key,
        openai,
        openrouter,
        xai,
        anthropic,
        provider,
    ))
}

fn mcp_env_nonempty(name: &str) -> Option<String> {
    present_secret(std::env::var(name).ok())
}

fn mcp_refusal_key_error(
    api_key_env: Option<&str>,
    provider: &str,
    policy: &McpServerOptions,
) -> String {
    if let Some(var) = trim_api_key_env(api_key_env)
        && !api_key_env_allowed(var)
        && pinned_api_key_env(policy) == Some(var)
    {
        return format!("{var} is unset or empty");
    }
    mcp_missing_key_error(api_key_env, provider)
}

fn mcp_cache_path(args: &Value) -> PathBuf {
    match args.get("cache").and_then(Value::as_str) {
        None => default_cache_path(),
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                default_cache_path()
            } else {
                expand_tilde(PathBuf::from(trimmed))
            }
        }
    }
}

fn opened_mcp_cache_path(path: &Path, policy: &McpServerOptions) -> Result<PathBuf, String> {
    if policy.allow_cache {
        return Ok(path.to_path_buf());
    }
    let outside = "cache path is outside the default cache directory";
    let Some(candidate) = resolve_opened_cache_path(path) else {
        return Err(outside.to_owned());
    };
    let default_file = default_cache_path();
    let Some(root) = default_file.parent() else {
        return Err(outside.to_owned());
    };
    let Some(located_root) = resolve_opened_cache_path(root) else {
        return Err(outside.to_owned());
    };
    if candidate.starts_with(&located_root) {
        Ok(candidate)
    } else {
        Err(outside.to_owned())
    }
}

#[cfg(test)]
fn mcp_cache_path_within_default(path: &Path) -> bool {
    opened_mcp_cache_path(path, &McpServerOptions::default()).is_ok()
}

/// `..` is applied after a directory symlink is followed. Collapsing
/// `..` first would hide that link, and the open would still enter it.
fn resolve_opened_cache_path(path: &Path) -> Option<PathBuf> {
    let mut links = 0u8;
    let opened = walk_opened_cache_path(path, &mut links)?;
    let mut full = opened.existing;
    for name in opened.pending {
        full.push(name);
    }
    if full.as_os_str().is_empty() {
        None
    } else {
        Some(full)
    }
}

struct OpenedCachePath {
    existing: PathBuf,
    pending: Vec<std::ffi::OsString>,
}

fn walk_opened_cache_path(path: &Path, links: &mut u8) -> Option<OpenedCachePath> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut state = OpenedCachePath {
        existing: PathBuf::new(),
        pending: Vec::new(),
    };
    for comp in absolute.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => {
                state.existing.push(comp);
                state.pending.clear();
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if state.pending.pop().is_none() {
                    state.existing.pop();
                }
            }
            Component::Normal(name) => {
                push_opened_cache_name(&mut state, name.to_os_string(), links)?;
            }
        }
    }
    Some(state)
}

fn push_opened_cache_name(
    state: &mut OpenedCachePath,
    name: std::ffi::OsString,
    links: &mut u8,
) -> Option<()> {
    if !state.pending.is_empty() {
        state.pending.push(name);
        return Some(());
    }
    let next = state.existing.join(&name);
    match std::fs::symlink_metadata(&next) {
        Ok(meta) if meta.file_type().is_symlink() => {
            *links = links.saturating_add(1);
            if *links > 40 {
                return None;
            }
            let target = std::fs::read_link(&next).ok()?;
            let combined = if target.is_absolute() {
                target
            } else {
                state.existing.join(target)
            };
            let followed = walk_opened_cache_path(&combined, links)?;
            state.existing = followed.existing;
            state.pending = followed.pending;
            Some(())
        }
        Ok(_) => {
            state.existing = next;
            Some(())
        }
        Err(_) => {
            state.pending.push(name);
            Some(())
        }
    }
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
    let named_key = present_secret(named_key);
    let openai = present_secret(openai);
    let openrouter = present_secret(openrouter);
    let xai = present_secret(xai);
    let anthropic = present_secret(anthropic);
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
        Some(var) if api_key_env_allowed(var) => mcp_env_nonempty(var),
        Some(_) => None,
        None if is_groq_provider_label(provider) => mcp_env_nonempty("GROQ_API_KEY"),
        None if is_bedrock_provider_label(provider) => mcp_env_nonempty("AWS_BEARER_TOKEN_BEDROCK"),
        None => None,
    }
}

fn read_named_or_route_key(
    api_key_env: Option<&str>,
    provider: &str,
    policy: &McpServerOptions,
) -> Option<String> {
    match trim_api_key_env(api_key_env) {
        Some(var) if api_key_env_allowed(var) => mcp_env_nonempty(var),
        Some(var) if pinned_api_key_env(policy) == Some(var) => mcp_env_nonempty(var),
        Some(_) => None,
        None => mcp_named_or_route_key(None, provider),
    }
}

/// Named `api_key_env` does not fall back to OPENAI_API_KEY / XAI_API_KEY.
fn mcp_missing_key_error(api_key_env: Option<&str>, provider: &str) -> String {
    match trim_api_key_env(api_key_env) {
        Some(var) if !api_key_env_allowed(var) => "api_key_env name not allowed".to_owned(),
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
        _ if is_openai_codex_provider_label(provider) => {
            "set api_key_env or OPENAI_API_KEY".to_owned()
        }
        _ if openrouter_default_ok(provider) && !provider.is_empty() => {
            "set api_key_env, OPENROUTER_API_KEY, or OPENAI_API_KEY for OpenRouter".to_owned()
        }
        _ if is_openai_provider_label(provider) || provider.trim().is_empty() => {
            "set api_key_env or OPENAI_API_KEY".to_owned()
        }
        _ => "set api_key_env (or OPENAI_API_KEY / OPENROUTER_API_KEY / XAI_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY), or pass base_url for a local host"
            .to_owned(),
    }
}

fn first_present_secret(first: Option<String>, second: Option<String>) -> Option<String> {
    present_secret(first).or_else(|| present_secret(second))
}

fn anthropic_env_key() -> Option<String> {
    first_present_secret(
        std::env::var("ANTHROPIC_AUTH_TOKEN").ok(),
        std::env::var("ANTHROPIC_API_KEY").ok(),
    )
}

fn xai_key_for_route(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> Result<Option<String>, String> {
    if let Some(key) = first_present_secret(
        std::env::var("XAI_API_KEY").ok(),
        std::env::var("GROK_API_KEY").ok(),
    ) {
        return Ok(Some(key));
    }
    if should_load_xai_oauth(provider, other_cloud_keys, explicit_base_url) {
        xai_oauth_access_token()
    } else {
        Ok(None)
    }
}

fn anthropic_key_for_route(
    provider: &str,
    other_cloud_keys: bool,
    explicit_base_url: bool,
) -> Result<Option<String>, String> {
    if let Some(key) = anthropic_env_key() {
        return Ok(Some(key));
    }
    if should_load_claude_code_login(provider, other_cloud_keys, explicit_base_url) {
        claude_code_access_token()
    } else {
        Ok(None)
    }
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

    fn cache_ok() -> McpServerOptions {
        McpServerOptions {
            allow_cache: true,
            ..McpServerOptions::default()
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
        let err = probe_model_with_route(&args, route, None, &cache_ok())
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
        let err = probe_model_with_route(&args, route, Some("FOO_KEY"), &cache_ok())
            .await
            .unwrap_err();
        assert_eq!(err, "api_key_env name not allowed");
        let lookups = crate::adapters::openai::take_catalog_lookups();
        assert!(
            lookups.is_empty(),
            "named unset api_key_env must not call catalog, got {lookups:?}"
        );
    }

    #[test]
    fn mcp_unknown_api_key_env_is_rejected() {
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
        assert_eq!(err, "api_key_env name not allowed");
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
    fn mcp_whitespace_only_base_url_is_not_explicit() {
        let args = json!({ "base_url": "  " });
        let has_explicit_base = mcp_present_base_url(&args).is_some();
        assert!(
            !has_explicit_base,
            "whitespace-only MCP base_url must not skip oauth"
        );
        assert!(should_load_xai_oauth("", false, has_explicit_base));
        assert_eq!(mcp_present_base_url(&args).map(str::to_owned), None);

        let empty = json!({ "base_url": "" });
        assert!(mcp_present_base_url(&empty).is_none());

        let padded = json!({ "base_url": " http://127.0.0.1:11434 " });
        assert_eq!(
            mcp_present_base_url(&padded),
            Some("http://127.0.0.1:11434")
        );
        assert!(!should_load_xai_oauth(
            "",
            false,
            mcp_present_base_url(&padded).is_some()
        ));
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
    async fn mcp_padded_unknown_api_key_env_is_rejected() {
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
        let err = probe_model_with_route(&args, route, Some(" FOO_KEY "), &cache_ok())
            .await
            .unwrap_err();
        assert_eq!(err, "api_key_env name not allowed");
    }

    #[tokio::test]
    async fn mcp_advertised_context_zero_is_refused() {
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "advertised_context": 0,
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
            let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
            let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
            let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let err = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
            let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
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
    fn mcp_named_env_key_trims_newline() {
        let route = mcp_resolve_key_route(
            Some("OPENAI_API_KEY"),
            Some("sk-real\n".into()),
            None,
            None,
            None,
            None,
            "openai",
        );
        assert_eq!(route.key.as_deref(), Some("sk-real"));
        assert!(!route.from_openrouter);
    }

    #[test]
    fn mcp_named_env_whitespace_key_is_absent() {
        let route = mcp_resolve_key_route(
            Some("OPENAI_API_KEY"),
            Some("  ".into()),
            None,
            None,
            None,
            None,
            "openai",
        );
        assert_eq!(route.key, None);
    }

    #[test]
    fn whitespace_xai_env_does_not_hide_grok_env() {
        let key = first_present_secret(Some("  \n".into()), Some("grok-real\n".into()));
        assert_eq!(key.as_deref(), Some("grok-real"));
    }

    #[test]
    fn whitespace_only_env_pair_is_absent() {
        assert_eq!(
            first_present_secret(Some(" ".into()), Some("\n".into())),
            None
        );
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

        let codex = mcp_resolve_key_route(
            None,
            None,
            Some("sk-openai".to_owned()),
            Some("sk-or".to_owned()),
            Some("xai-key".to_owned()),
            None,
            "openai-codex",
        );
        assert_eq!(codex.key.as_deref(), Some("sk-openai"));
        assert!(!codex.from_openrouter);
        assert!(!codex.from_xai);
        let codex_or = mcp_resolve_key_route(
            None,
            None,
            None,
            Some("sk-or".to_owned()),
            None,
            None,
            "openai-codex",
        );
        assert!(
            codex_or.key.is_none(),
            "provider=openai-codex must not send OPENROUTER_API_KEY"
        );
        let codex_err = mcp_missing_key_error(None, "openai-codex");
        assert!(codex_err.contains("OPENAI_API_KEY"), "{codex_err}");
        assert!(
            !codex_err.contains("OPENROUTER_API_KEY"),
            "Codex missing-key must not list OpenRouter: {codex_err}"
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
        let openai_err = mcp_missing_key_error(None, "openai");
        assert!(openai_err.contains("OPENAI_API_KEY"), "{openai_err}");
        assert!(
            !openai_err.contains("OPENROUTER_API_KEY"),
            "provider=openai must not tell the host to set OPENROUTER_API_KEY: {openai_err}"
        );
        assert!(
            !openai_err.contains("XAI_API_KEY") && !openai_err.contains("ANTHROPIC_API_KEY"),
            "provider=openai must not list other clouds: {openai_err}"
        );
        let openrouter_err = mcp_missing_key_error(None, "openrouter");
        assert!(
            openrouter_err.contains("OPENROUTER_API_KEY")
                && openrouter_err.contains("OPENAI_API_KEY"),
            "{openrouter_err}"
        );
        assert!(
            !openrouter_err.contains("XAI_API_KEY")
                && !openrouter_err.contains("ANTHROPIC_API_KEY"),
            "provider=openrouter must not list xAI or Anthropic: {openrouter_err}"
        );
        let empty_err = mcp_missing_key_error(None, "");
        assert_eq!(empty_err, "set api_key_env or OPENAI_API_KEY");
        assert!(
            !empty_err.contains("XAI_API_KEY"),
            "empty provider defaults to OpenAI and must not list other clouds: {empty_err}"
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
    fn mcp_tools_argument_is_optional_and_parses_the_array() {
        let schema = &tools_list()["tools"][0]["inputSchema"];
        assert_eq!(schema["required"], json!(["model"]));
        let tools = &schema["properties"]["tools"];
        assert_eq!(tools["type"], "array");
        assert_eq!(
            tools["items"]["required"],
            json!(["name", "description", "parameters"])
        );

        assert!(parse_mcp_tools(None).expect("omit").is_none());
        let empty = parse_mcp_tools(Some(&json!([])))
            .expect("empty")
            .expect("some");
        assert!(empty.is_empty());
        let missing =
            parse_mcp_tools(Some(&json!([{"name": "lookup_issue"}]))).expect_err("fields");
        assert!(missing.contains("parse"), "{missing}");
        let object = parse_mcp_tools(Some(&json!({}))).expect_err("object");
        assert!(
            object.contains("expected a JSON array of tools"),
            "{object}"
        );
        assert!(!object.contains("expected a sequence"), "{object}");
        assert!(!object.contains("McpCallerTool"), "{object}");
        let string_item = parse_mcp_tools(Some(&json!(["lookup_issue"]))).expect_err("string");
        assert!(
            string_item.contains("item 0 must be an object with name, description, and parameters"),
            "{string_item}"
        );
        assert!(!string_item.contains("McpCallerTool"), "{string_item}");

        let raw = json!([{
            "name": "lookup_issue",
            "description": "Look up one issue.",
            "parameters": {"type": "object"}
        }]);
        let parsed = parse_mcp_tools(Some(&raw)).expect("array").expect("some");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "lookup_issue");
        assert_eq!(parsed[0].description, "Look up one issue.");
    }

    #[tokio::test]
    async fn mcp_bad_tools_object_errors_before_connect() {
        let args = json!({
            "model": "x",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:1/v1",
            "tools": {}
        });
        let err = probe_model_args(&args, &McpServerOptions::default())
            .await
            .expect_err("object is not a tool list");
        assert!(err.contains("expected a JSON array of tools"), "{err}");
        assert!(!err.contains("expected a sequence"), "{err}");
        assert!(!err.contains("authentication error"), "{err}");
        assert!(!err.contains("connection"), "{err}");
        assert!(!err.contains("os error"), "{err}");
        assert!(!err.contains("set --api-key"), "{err}");
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

    struct FooKeyEnv {
        prev: Option<OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    impl FooKeyEnv {
        fn set(value: &str) -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var_os("FOO_KEY");
            unsafe {
                std::env::set_var("FOO_KEY", value);
            }
            Self { prev, _lock: lock }
        }
    }

    impl Drop for FooKeyEnv {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var("FOO_KEY", v),
                    None => std::env::remove_var("FOO_KEY"),
                }
            }
        }
    }

    struct LoginEnvGuard {
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl LoginEnvGuard {
        fn clear() -> Self {
            let keys = [
                "XAI_API_KEY",
                "GROK_API_KEY",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
            ];
            let saved = keys
                .into_iter()
                .map(|key| {
                    let prev = std::env::var_os(key);
                    unsafe {
                        std::env::remove_var(key);
                    }
                    (key, prev)
                })
                .collect();
            Self { saved }
        }
    }

    impl Drop for LoginEnvGuard {
        fn drop(&mut self) {
            unsafe {
                for (key, prev) in &self.saved {
                    match prev {
                        Some(v) => std::env::set_var(key, v),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    struct OauthShortCircuit;

    impl OauthShortCircuit {
        fn enable() -> Self {
            crate::claude_code::oauth_test_hook::reset();
            crate::claude_code::oauth_test_hook::SHORT_CIRCUIT
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Self
        }
    }

    impl Drop for OauthShortCircuit {
        fn drop(&mut self) {
            crate::claude_code::oauth_test_hook::SHORT_CIRCUIT
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn oauth_counts() -> (usize, usize) {
        (
            crate::claude_code::oauth_test_hook::CLAUDE_ENTRIES
                .load(std::sync::atomic::Ordering::SeqCst),
            crate::claude_code::oauth_test_hook::XAI_ENTRIES
                .load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    fn seed_model_cache(model: &str, provider: &str, tokens: u32) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let mut cache = ProbeCache::default();
        let mut profile = CapabilityProfile::unprobed(model, provider);
        profile.effective_context_tokens = Some(tokens);
        cache.put_with_suite(profile, SuiteTier::Policy, false, None);
        cache.save(&cache_path).expect("save");
        (dir, cache_path.to_str().expect("utf8").to_owned())
    }

    #[test]
    fn mcp_rejects_unknown_api_key_env() {
        let _foo = FooKeyEnv::set("canary-secret");
        let named = mcp_named_or_route_key(Some("FOO_KEY"), "openai");
        assert_ne!(
            named.as_deref(),
            Some("canary-secret"),
            "unknown api_key_env must not be read"
        );
        let err = mcp_missing_key_error(Some("FOO_KEY"), "openai");
        assert!(
            err.contains("name not allowed"),
            "unknown api_key_env must be rejected, got {err}"
        );
    }

    #[test]
    fn mcp_known_api_key_env_still_resolves() {
        let _env = IsolatedApiKeyEnv::set_openai("sk-known-openai");
        let named = mcp_named_or_route_key(Some("OPENAI_API_KEY"), "openai");
        assert_eq!(named.as_deref(), Some("sk-known-openai"));
        for name in [
            "GROQ_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
            "OPENROUTER_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ] {
            let err = mcp_missing_key_error(Some(name), "openai");
            assert!(
                !err.contains("name not allowed"),
                "{name} must stay allowed, got {err}"
            );
        }
    }

    // One probe future each. A single async test that awaited both overflowed
    // the Windows libtest stack.
    #[tokio::test]
    async fn mcp_named_anthropic_tool_url_skips_oauth() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let _circuit = OauthShortCircuit::enable();

        let (_dir, cache_str) = seed_model_cache("claude-test", "anthropic", 111);
        let args = json!({
            "model": "claude-test",
            "provider": "anthropic",
            "base_url": "http://127.0.0.1:11434/v1",
            "cache": cache_str,
        });
        let envelope = probe_model_args(&args, &cache_ok())
            .await
            .expect("named anthropic plus a loopback tool url");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(
            oauth_counts(),
            (0, 0),
            "named anthropic plus a tool base_url must not read stored logins"
        );
        assert!(
            crate::adapters::openai::take_catalog_lookups().is_empty(),
            "oauth skip test must not call catalog"
        );
    }

    #[tokio::test]
    async fn mcp_finalize_tool_url_skips_oauth() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let _circuit = OauthShortCircuit::enable();

        let (_dir, cache_str) = seed_model_cache("grok-test", "api.x.ai", 222);
        let args = json!({
            "model": "grok-test",
            "base_url": "https://api.x.ai/v1",
            "cache": cache_str,
        });
        let public_ok = McpServerOptions {
            allow_cache: true,
            allow_base_url: true,
            ..McpServerOptions::default()
        };
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &public_ok)
            .await
            .expect("finalize tool url reaches the cache");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(
            oauth_counts(),
            (0, 0),
            "finalize must not read stored logins for a tool base_url"
        );
        assert!(
            crate::adapters::openai::take_catalog_lookups().is_empty(),
            "oauth skip test must not call catalog"
        );
    }

    async fn refused_tool_base_url(url: &str) -> String {
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "base_url": url,
        });
        probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
            .await
            .expect_err(url)
    }

    async fn cached_tool_base_url(url: &str, cache_str: &str, what: &str) -> Value {
        let args = json!({
            "model": "llama",
            "provider": "ollama",
            "base_url": url,
            "cache": cache_str,
        });
        probe_model_with_route(&args, mcp_empty_route(), None, &cache_ok())
            .await
            .expect(what)
    }

    #[test]
    fn probe_model_future_fits_a_one_megabyte_stack() {
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
        });
        let policy = McpServerOptions::default();
        let fut = probe_model_with_route(&args, mcp_empty_route(), None, &policy);
        let size = std::mem::size_of_val(&fut);
        assert!(
            size < 128 * 1024,
            "probe future is {size} bytes; Windows tests overflow near 1MB"
        );
    }

    #[test]
    fn mcp_public_base_url_requires_server_flag() {
        // Sequential block_on keeps each probe future on its own. One async
        // fn with these awaits overflowed the Windows libtest stack.
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let _env = LoginEnvGuard::clear();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        for url in [
            "https://api.openai.com/v1",
            "http://127.0.0.1@evil.example/v1",
            "http://localhost./v1",
            "http://2130706433/v1",
            "http://127.0.0.1.nip.io/v1",
        ] {
            let err = rt.block_on(refused_tool_base_url(url));
            assert!(
                err.contains("base_url"),
                "{url} must be refused as a tool base_url, got {err}"
            );
            assert!(
                !err.contains("authentication error"),
                "{url} must be refused before login, got {err}"
            );
        }
        assert!(
            crate::adapters::openai::take_catalog_lookups().is_empty(),
            "refused base_url must not call catalog"
        );

        let (_dir, cache_str) = seed_model_cache("llama", "ollama", 333);
        let envelope = rt.block_on(cached_tool_base_url(
            "http://127.0.0.1:11434/v1",
            &cache_str,
            "loopback tool base_url stays allowed",
        ));
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(envelope["effectiveContextTokens"], 333, "{envelope}");

        let envelope = rt.block_on(cached_tool_base_url(
            "http://evil.example@127.0.0.1:11434/v1",
            &cache_str,
            "userinfo before a loopback host stays allowed",
        ));
        assert_eq!(envelope["fromCache"], true, "{envelope}");
    }

    fn write_openai_cache(path: &std::path::Path, tokens: u32) -> Vec<u8> {
        let mut cache = ProbeCache::default();
        let mut profile = CapabilityProfile::unprobed("gpt-4o", "openai");
        profile.effective_context_tokens = Some(tokens);
        cache.put_with_suite(profile, SuiteTier::Policy, false, None);
        cache.save(path).expect("save");
        std::fs::read(path).expect("read")
    }

    fn isolated_default_cache_dir() -> std::path::PathBuf {
        let dir = default_cache_path()
            .parent()
            .expect("cache dir")
            .to_path_buf();
        std::fs::create_dir_all(&dir).expect("default dir");
        dir
    }

    #[tokio::test]
    async fn mcp_cache_outside_default_dir_is_refused() {
        let _home = IsolatedHome::new();
        let outside = tempfile::tempdir().expect("temp");
        let outside_path = outside.path().join("probes.json");
        let before = write_openai_cache(&outside_path, 444);
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": outside_path.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("outside cache");
        assert!(
            err.contains("cache"),
            "outside cache must be refused, got {err}"
        );
        assert_eq!(std::fs::read(&outside_path).expect("still"), before);
    }

    #[tokio::test]
    async fn mcp_cache_dotdot_escape_is_refused() {
        let _home = IsolatedHome::new();
        let default_dir = isolated_default_cache_dir();
        let escape = default_dir.join("..").join("canact-escape-probes.json");
        let before = write_openai_cache(&escape, 555);
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": escape.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("dotdot cache");
        assert!(err.contains("cache"), "{err}");
        assert_eq!(std::fs::read(&escape).expect("escape still"), before);
    }

    #[tokio::test]
    async fn mcp_cache_inside_default_dir_is_allowed() {
        let _home = IsolatedHome::new();
        let inside = isolated_default_cache_dir().join("other.json");
        write_openai_cache(&inside, 666);
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": inside.to_str().expect("utf8"),
        });
        let envelope =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect("cache inside the default directory stays allowed");
        assert_eq!(envelope["effectiveContextTokens"], 666, "{envelope}");
    }

    #[tokio::test]
    async fn mcp_allow_cache_reads_outside_file() {
        let _home = IsolatedHome::new();
        let outside = tempfile::tempdir().expect("temp");
        let outside_path = outside.path().join("probes.json");
        let before = write_openai_cache(&outside_path, 444);
        let allowed = McpServerOptions {
            allow_cache: true,
            ..McpServerOptions::default()
        };
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": outside_path.to_str().expect("utf8"),
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &allowed)
            .await
            .expect("allow_cache reads an outside file");
        assert_eq!(envelope["effectiveContextTokens"], 444, "{envelope}");
        assert_eq!(std::fs::read(&outside_path).expect("unchanged"), before);
    }

    #[tokio::test]
    async fn mcp_missing_dotdot_cache_is_not_created() {
        let _home = IsolatedHome::new();
        let missing = isolated_default_cache_dir()
            .join("..")
            .join("canact-not-created.json");
        assert!(!missing.exists(), "precondition");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": missing.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("missing dotdot cache");
        assert!(err.contains("cache"), "{err}");
        assert!(!missing.exists(), "refused cache path must not be created");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_cache_symlink_outside_default_dir_is_refused() {
        let _home = IsolatedHome::new();
        let outside = tempfile::tempdir().expect("temp");
        let outside_path = outside.path().join("probes.json");
        let before = write_openai_cache(&outside_path, 444);
        let link = isolated_default_cache_dir().join("escape-link.json");
        std::os::unix::fs::symlink(&outside_path, &link).expect("symlink");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": link.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("symlink cache");
        assert!(err.contains("cache"), "{err}");
        assert_eq!(
            std::fs::read(&outside_path).expect("symlink target"),
            before
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_cache_symlink_ancestor_missing_leaf_is_refused() {
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let outside = tempfile::tempdir().expect("temp");
        let link = isolated_default_cache_dir().join("escape-dir");
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");
        let missing = link.join("not-created.json");
        assert!(!missing.exists(), "precondition");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": missing.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("symlink ancestor");
        assert!(
            err.contains("cache path is outside the default cache directory"),
            "{err}"
        );
        assert!(
            !outside.path().join("not-created.json").exists(),
            "refused cache path must not be created"
        );
    }

    #[cfg(unix)]
    #[test]
    fn mcp_cache_broken_symlink_ancestor_is_refused() {
        let _home = IsolatedHome::new();
        let dir = isolated_default_cache_dir();
        let missing_target = dir.join("..").join("canact-broken-target");
        assert!(!missing_target.exists(), "precondition");
        let link = dir.join("broken-link");
        std::os::unix::fs::symlink(&missing_target, &link).expect("symlink");
        let missing = link.join("not-created.json");
        assert!(
            !mcp_cache_path_within_default(&missing),
            "a broken symlink ancestor is outside the default cache directory"
        );
    }

    #[test]
    fn mcp_missing_file_inside_default_dir_is_allowed() {
        let _home = IsolatedHome::new();
        let missing = isolated_default_cache_dir().join("brand-new.json");
        assert!(!missing.exists(), "precondition");
        assert!(
            mcp_cache_path_within_default(&missing),
            "a new file inside the default cache directory stays allowed"
        );
    }

    #[test]
    fn mcp_dotdot_through_real_directory_stays_inside() {
        let _home = IsolatedHome::new();
        let sub = isolated_default_cache_dir().join("subdir");
        std::fs::create_dir_all(&sub).expect("subdir");
        let path = sub.join("..").join("brand-new.json");
        assert!(
            mcp_cache_path_within_default(&path),
            "a .. through a real directory stays inside the default cache directory"
        );
    }

    #[cfg(unix)]
    #[test]
    fn mcp_cache_dotdot_through_symlink_dir_is_refused() {
        let _home = IsolatedHome::new();
        let outside = tempfile::tempdir().expect("temp");
        let link = isolated_default_cache_dir().join("escape-dir");
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");
        let tricky = link.join("..").join("escaped.json");
        assert!(
            !mcp_cache_path_within_default(&tricky),
            "a .. after a directory symlink leaves the default cache directory"
        );
        let rel_outside = isolated_default_cache_dir()
            .join("..")
            .join("relative-outside");
        std::fs::create_dir_all(&rel_outside).expect("relative outside");
        let rel_link = isolated_default_cache_dir().join("rel-link");
        std::os::unix::fs::symlink("../relative-outside", &rel_link).expect("relative symlink");
        let rel_tricky = rel_link.join("..").join("escaped.json");
        assert!(
            !mcp_cache_path_within_default(&rel_tricky),
            "a .. after a relative directory symlink leaves the default cache directory"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_cache_dotdot_through_symlink_dir_is_not_created() {
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let outside = tempfile::tempdir().expect("temp");
        let dir = isolated_default_cache_dir();
        let link = dir.join("escape-dir");
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");
        let tricky = link.join("..").join("escaped.json");
        let landed = dir.parent().expect("parent").join("escaped.json");
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "cache": tricky.to_str().expect("utf8"),
        });
        let err =
            probe_model_with_route(&args, mcp_empty_route(), None, &McpServerOptions::default())
                .await
                .expect_err("dotdot through symlink");
        assert!(
            err.contains("cache path is outside the default cache directory"),
            "{err}"
        );
        assert!(!landed.exists(), "opened path must stay uncreated");
        assert!(
            !dir.join("escaped.json").exists(),
            "lexical path must stay uncreated"
        );
        assert!(
            !outside.path().join("escaped.json").exists(),
            "symlink target must stay uncreated"
        );
    }

    #[test]
    fn mcp_pinned_env_outside_allowlist_is_read() {
        let _foo = FooKeyEnv::set("canary-pinned");
        let policy = McpServerOptions {
            api_key_env: Some(" FOO_KEY ".into()),
            ..McpServerOptions::default()
        };
        let name = permitted_api_key_env(None, &policy).expect("pin");
        assert_eq!(name.as_deref(), Some("FOO_KEY"));
        let key = read_named_or_route_key(name.as_deref(), "openai", &policy);
        assert_eq!(key.as_deref(), Some("canary-pinned"));
        let mismatch = permitted_api_key_env(Some("OPENAI_API_KEY"), &policy).unwrap_err();
        assert_eq!(mismatch, "api_key_env is pinned by the server");
        let same = permitted_api_key_env(Some("FOO_KEY"), &policy).expect("echo pin");
        assert_eq!(same.as_deref(), Some("FOO_KEY"));
    }

    #[tokio::test]
    async fn mcp_pinned_unknown_env_unset_names_the_var() {
        let _home = IsolatedHome::new();
        let _foo = FooKeyEnv::set("");
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let _circuit = OauthShortCircuit::enable();
        let dir = tempfile::tempdir().expect("temp");
        let cache_path = dir.path().join("probes.json");
        let policy = McpServerOptions {
            allow_cache: true,
            api_key_env: Some("FOO_KEY".into()),
            ..McpServerOptions::default()
        };
        let args = json!({
            "model": "gpt-4o",
            "provider": "openai",
            "force": true,
            "cache": cache_path.to_str().expect("utf8"),
        });
        let err = probe_model_args(&args, &policy).await.unwrap_err();
        assert_eq!(err, "FOO_KEY is unset or empty");
        assert!(!err.contains("name not allowed"), "{err}");
        assert_eq!(oauth_counts(), (0, 0), "{err}");
        assert!(crate::adapters::openai::take_catalog_lookups().is_empty());
    }

    #[test]
    fn mcp_backslash_authority_is_not_a_loopback_tool_url() {
        let err = resolve_effective_base_url(
            Some("http://evil.com\\@127.0.0.1/v1"),
            &McpServerOptions::default(),
        )
        .expect_err("backslash authority");
        assert!(
            err.contains("base_url"),
            "dial host must be refused, got {err}"
        );
        assert!(
            !err.contains("authentication error"),
            "refusal must happen before login, got {err}"
        );
    }

    #[test]
    fn mcp_tool_url_still_reads_anthropic_env_key() {
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let _circuit = OauthShortCircuit::enable();
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-from-env");
        }
        let route = load_mcp_route("anthropic", None, true, false, &McpServerOptions::default())
            .expect("route");
        assert_eq!(route.key.as_deref(), Some("sk-ant-from-env"));
        assert!(route.from_anthropic);
        assert_eq!(
            oauth_counts(),
            (0, 0),
            "a tool base_url must not read a stored Claude login"
        );
    }

    #[tokio::test]
    async fn mcp_pinned_base_url_rejects_a_different_tool_url() {
        let policy = McpServerOptions {
            base_url: Some("http://127.0.0.1:11434/v1".into()),
            ..McpServerOptions::default()
        };
        let args = json!({
            "model": "llama",
            "provider": "ollama",
            "base_url": "https://api.openai.com/v1",
        });
        let err = probe_model_with_route(&args, mcp_empty_route(), None, &policy)
            .await
            .unwrap_err();
        assert_eq!(err, "base_url is pinned by the server");
    }

    #[tokio::test]
    async fn mcp_same_trimmed_base_url_is_not_a_conflict() {
        let policy = McpServerOptions {
            allow_cache: true,
            base_url: Some(" http://127.0.0.1:11434/v1 ".into()),
            ..McpServerOptions::default()
        };
        let (_dir, cache_str) = seed_model_cache("llama", "ollama", 333);
        let args = json!({
            "model": "llama",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "cache": cache_str,
        });
        let envelope = probe_model_with_route(&args, mcp_empty_route(), None, &policy)
            .await
            .expect("same trimmed url");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
    }

    #[test]
    fn mcp_human_pin_public_url_is_not_a_tool_url() {
        let policy = McpServerOptions {
            base_url: Some("https://api.openai.com/v1".into()),
            ..McpServerOptions::default()
        };
        let (url, tool_supplied) = resolve_effective_base_url(None, &policy).expect("pin");
        assert_eq!(url.as_deref(), Some("https://api.openai.com/v1"));
        assert!(!tool_supplied);
    }

    #[tokio::test]
    async fn mcp_human_pinned_anthropic_url_still_reads_login() {
        let _skip = crate::adapters::openai::CatalogSkipHttp::enable();
        let _home = IsolatedHome::new();
        let _env = LoginEnvGuard::clear();
        let _circuit = OauthShortCircuit::enable();
        let (_dir, cache_str) = seed_model_cache("claude-test", "anthropic", 111);
        let policy = McpServerOptions {
            allow_cache: true,
            base_url: Some("https://api.anthropic.com/v1".into()),
            ..McpServerOptions::default()
        };
        let args = json!({
            "model": "claude-test",
            "provider": "anthropic",
            "cache": cache_str,
        });
        let envelope = probe_model_args(&args, &policy)
            .await
            .expect("a server-pinned url may still read a stored login");
        assert_eq!(envelope["fromCache"], true, "{envelope}");
        assert_eq!(
            oauth_counts(),
            (1, 0),
            "named anthropic with a server-pinned url still consults Claude login"
        );
    }
}
