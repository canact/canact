//! Stdio MCP server. Tool name `probe_model` is stolen from Jwrede/llmprobe;
//! the payload is canact host-policy JSON, not TTFT.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::{
    CatalogPriors, HostPolicyMeta, OpenAiCompatClient, ProbeCache, ProbeError, ProbeRunner,
    cloud_endpoint_requires_key, default_compat_base_url, looks_cheap, provider_from_base_url,
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
        if msg.get("method").and_then(Value::as_str) == Some("notifications/initialized") {
            continue;
        }
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
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
                    "cheap": { "type": "boolean" },
                    "full": { "type": "boolean" },
                    "vision": { "type": "boolean" },
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
    let is_error = envelope.get("canUseTools").and_then(Value::as_bool) == Some(false);
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error
    }))
}

async fn probe_model_args(args: &Value) -> Result<Value, String> {
    let model = args
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "model is required".to_owned())?
        .to_owned();
    let advertised = json_u32(args.get("advertised_context"));
    let cheap = json_bool(args.get("cheap")).unwrap_or(false);
    let full = json_bool(args.get("full")).unwrap_or(false);
    let vision = json_bool(args.get("vision")).unwrap_or(false);
    let force = json_bool(args.get("force")).unwrap_or(false);
    let cache_path = args
        .get("cache")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .map(expand_tilde)
        .unwrap_or_else(default_cache_path);
    let mut cache = ProbeCache::load(&cache_path)
        .map_err(|e| format!("failed to load cache {}: {e}", cache_path.display()))?;

    let (api_key, from_openrouter) = match args.get("api_key_env").and_then(Value::as_str) {
        Some(var) if !var.is_empty() => (
            std::env::var(var).ok().filter(|s| !s.is_empty()),
            var == "OPENROUTER_API_KEY",
        ),
        _ => {
            if let Some(key) = std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
            {
                (Some(key), false)
            } else if let Some(key) = std::env::var("OPENROUTER_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
            {
                (Some(key), true)
            } else {
                (None, false)
            }
        }
    };
    let provider_given = args
        .get("provider")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let base_url = args
        .get("base_url")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            default_compat_base_url(provider_given.as_deref().unwrap_or(""), from_openrouter)
        });
    let provider = provider_given.unwrap_or_else(|| provider_from_base_url(&base_url));
    let skip_expensive = if full {
        false
    } else if cheap {
        true
    } else {
        looks_cheap(&provider, &model, &base_url)
    };
    if !force {
        if let Some(profile) =
            cache.get_with_knobs(&model, &provider, skip_expensive, vision, advertised)
        {
            return Ok(profile.host_policy_envelope_with(HostPolicyMeta {
                cacheable: true,
                skip_expensive,
                advertised_context_tokens: advertised,
            }));
        }
        if !full {
            if let Some((profile, cheap_row)) = cache.find_profile_with_cost(&model, &provider) {
                return Ok(profile.host_policy_envelope_with(HostPolicyMeta {
                    cacheable: true,
                    skip_expensive: cheap_row,
                    advertised_context_tokens: advertised,
                }));
            }
        }
    }
    if api_key.is_none() && cloud_endpoint_requires_key(&base_url) {
        return Err(
            "set api_key_env (or OPENAI_API_KEY / OPENROUTER_API_KEY), or pass base_url for a local host"
                .to_owned(),
        );
    }
    let catalog = CatalogPriors {
        advertised_context_tokens: advertised,
        supports_vision: if vision { Some(true) } else { None },
        supports_tools: None,
    };
    let client = OpenAiCompatClient::new(base_url, api_key, model, provider, catalog)
        .map_err(|e| e.to_string())?;
    let runner = if skip_expensive {
        ProbeRunner::new_throttled(client)
    } else {
        ProbeRunner::new(client)
    };
    let run = runner.run_detailed().await.map_err(|e| match e {
        ProbeError::Auth(msg) => format!("authentication error: {msg}"),
        other => other.to_string(),
    })?;
    if let Err(err) = run.persist(&mut cache, &cache_path) {
        eprintln!("warning: failed to save probe cache: {err}");
    }
    Ok(run.host_policy_envelope())
}

fn default_cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("canact")
        .join("probes.json")
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return dirs::home_dir().unwrap_or(path);
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
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

fn read_message(reader: &mut impl BufRead) -> Result<Option<Value>, String> {
    let mut headers = String::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(None);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        headers.push_str(&line);
    }
    let mut len = None;
    for header in headers.lines() {
        let header = header.trim_end_matches('\r');
        if let Some(rest) = header.strip_prefix("Content-Length:") {
            len = rest.trim().parse::<usize>().ok();
        }
    }
    let len = len.ok_or_else(|| "missing Content-Length".to_owned())?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    serde_json::from_slice(&buf)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn write_message(writer: &mut impl Write, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len()).map_err(|e| e.to_string())?;
    writer.write_all(&body).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
}
