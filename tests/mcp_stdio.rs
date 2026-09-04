//! MCP stdio: `probe_model` returns host-policy JSON, not TTFT.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use canact::{CapabilityLevel, CapabilityProfile, ProbeCache, ProbeResult};
use serde_json::{Value, json};

fn sample() -> CapabilityProfile {
    let pr = |name: &str, level: CapabilityLevel| ProbeResult {
        name: name.to_owned(),
        score: 1.0,
        max_score: 1.0,
        level,
        details: "test".to_owned(),
    };
    CapabilityProfile {
        model_id: "qwen2.5-coder".to_owned(),
        provider: "ollama".to_owned(),
        tool_calling: pr("tool_calling", CapabilityLevel::Strong),
        json_output: pr("json_output", CapabilityLevel::Strong),
        instruction_following: pr("instruction_following", CapabilityLevel::Strong),
        search_replace: pr("search_replace", CapabilityLevel::Strong),
        unified_diff: pr("unified_diff", CapabilityLevel::Medium),
        xml_tool_calling: pr("xml_tool_calling", CapabilityLevel::Medium),
        complex_tool_calling: pr("complex_tool_calling", CapabilityLevel::Strong),
        nested_arguments: pr("nested_arguments", CapabilityLevel::Strong),
        vision: pr("vision", CapabilityLevel::Weak),
        tool_selection: pr("tool_selection", CapabilityLevel::Medium),
        streaming_tool_calls: pr("streaming_tool_calls", CapabilityLevel::Strong),
        one_shot_tool_plan: pr("one_shot_tool_plan", CapabilityLevel::Strong),
        multi_turn_task_sequencing: pr("multi_turn_task_sequencing", CapabilityLevel::Strong),
        context_faithfulness: pr("context_faithfulness", CapabilityLevel::Strong),
        code_syntax: pr("code_syntax", CapabilityLevel::Strong),
        max_tokens_compliance: pr("max_tokens_compliance", CapabilityLevel::Strong),
        multi_turn_memory: pr("multi_turn_memory", CapabilityLevel::Strong),
        system_message_adherence: pr("system_message_adherence", CapabilityLevel::Strong),
        token_efficiency: pr("token_efficiency", CapabilityLevel::Strong),
        parallel_tool_scale: pr("parallel_tool_scale", CapabilityLevel::Strong),
        probed_at: 1_700_000_000,
        effective_context_tokens: Some(8192),
        probed_context_floor: Some(8192),
    }
}

fn write_rpc(stdin: &mut impl Write, value: &Value) {
    let body = serde_json::to_vec(value).expect("json");
    write!(stdin, "Content-Length: {}\r\n\r\n", body.len()).expect("hdr");
    stdin.write_all(&body).expect("body");
    stdin.flush().expect("flush");
}

fn read_rpc(stdout: &mut impl Read) -> Value {
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line).expect("ndjson reply");
        assert!(!line.is_empty(), "MCP stdout closed before JSON-RPC reply");
        if !line.trim().is_empty() {
            break;
        }
    }
    serde_json::from_str(line.trim()).expect("rpc json")
}

#[test]
fn mcp_probe_model_returns_host_policy_from_cache() {
    let dir = tempfile::tempdir().expect("temp");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(sample());
    cache.save(&cache_path).expect("save");

    let mut child = Command::new(env!("CARGO_BIN_EXE_canact"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn canact mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "canact-test", "version": "0" }
            }
        }),
    );
    let init = read_rpc(&mut stdout);
    assert_eq!(init["result"]["serverInfo"]["name"], "canact", "{init}");
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05", "{init}");

    write_rpc(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
    );
    let listed = read_rpc(&mut stdout);
    let name = listed["result"]["tools"][0]["name"].as_str().expect("name");
    let desc = listed["result"]["tools"][0]["description"]
        .as_str()
        .expect("desc");
    assert_eq!(name, "probe_model");
    assert!(desc.contains("host-policy"), "{desc}");
    assert!(
        desc.contains("Not TTFT"),
        "must steal the name without stealing TTFT semantics: {desc}"
    );

    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "probe_model",
                "arguments": {
                    "model": "qwen2.5-coder",
                    "provider": "ollama",
                    "cache": cache_path.to_str().expect("utf8")
                }
            }
        }),
    );
    let called = read_rpc(&mut stdout);
    let text = called["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    assert_eq!(called["result"]["isError"], false, "{called}");
    let envelope: Value = serde_json::from_str(text).expect("envelope");
    assert_eq!(envelope["model"], "qwen2.5-coder", "{envelope}");
    assert_eq!(envelope["maxTools"], 20, "{envelope}");
    assert_eq!(
        envelope["probeLadderEditFormat"], "search_replace",
        "{envelope}"
    );
    assert_eq!(envelope["recommendedContextTokens"], 8192, "{envelope}");
    assert!(
        envelope.get("ttft").is_none(),
        "must not emit Jwrede TTFT fields: {envelope}"
    );

    drop(stdin);
    let _ = child.wait_timeout();
}

#[test]
fn mcp_cached_weak_tools_is_not_error() {
    let dir = tempfile::tempdir().expect("temp");
    let cache_path = dir.path().join("probes.json");
    let mut profile = sample();
    profile.tool_calling = ProbeResult {
        name: "tool_calling".to_owned(),
        score: 0.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "test".to_owned(),
    };
    profile.xml_tool_calling = ProbeResult {
        name: "xml_tool_calling".to_owned(),
        score: 0.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "test".to_owned(),
    };
    let mut cache = ProbeCache::default();
    cache.put(profile);
    cache.save(&cache_path).expect("save");

    let mut child = Command::new(env!("CARGO_BIN_EXE_canact"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn canact mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "canact-test", "version": "0" }
            }
        }),
    );
    let _ = read_rpc(&mut stdout);

    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "probe_model",
                "arguments": {
                    "model": "qwen2.5-coder",
                    "provider": "ollama",
                    "cache": cache_path.to_str().expect("utf8")
                }
            }
        }),
    );
    let called = read_rpc(&mut stdout);
    let is_error = called["result"].get("isError");
    assert!(
        is_error.is_none() || is_error == Some(&json!(false)),
        "successful Weak-tools envelope must not be isError: {called}"
    );
    let text = called["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    let envelope: Value = serde_json::from_str(text).expect("envelope");
    assert_eq!(envelope["canUseTools"], false, "{envelope}");
    assert_eq!(envelope["model"], "qwen2.5-coder", "{envelope}");

    drop(stdin);
    let _ = child.wait_timeout();
}

#[test]
fn mcp_ndjson_initialize_gets_jsonrpc_reply() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_canact"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn canact mcp");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "canact-test", "version": "0" }
        }
    });
    writeln!(stdin, "{req}").expect("ndjson initialize");
    stdin.flush().expect("flush");

    let mut line = String::new();
    {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(&mut stdout);
        reader.read_line(&mut line).expect("ndjson reply");
    }
    assert!(
        !line.is_empty(),
        "single-line initialize must get a JSON-RPC reply"
    );
    let init: Value = serde_json::from_str(line.trim()).expect("jsonrpc");
    assert_eq!(init["jsonrpc"], "2.0", "{init}");
    assert_eq!(init["id"], 1, "{init}");
    assert_eq!(init["result"]["serverInfo"]["name"], "canact", "{init}");
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05", "{init}");

    drop(stdin);
    let _ = child.wait_timeout();
}

#[test]
fn mcp_full_does_not_return_cheap_cache() {
    let dir = tempfile::tempdir().expect("temp");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample(), true, false, None);
    cache.save(&cache_path).expect("save");

    let mut child = Command::new(env!("CARGO_BIN_EXE_canact"))
        .arg("mcp")
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");
    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "canact-test", "version": "0" }
            }
        }),
    );
    let _ = read_rpc(&mut stdout);
    write_rpc(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "probe_model",
                "arguments": {
                    "model": "qwen2.5-coder",
                    "provider": "ollama",
                    "cache": cache_path.to_str().expect("utf8"),
                    "full": true
                }
            }
        }),
    );
    let called = read_rpc(&mut stdout);
    let text = called["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    assert_eq!(called["result"]["isError"], true, "{called}");
    assert!(
        text.contains("failed to connect")
            || text.contains("api_key")
            || text.contains("not found"),
        "full must miss cheap cache and go live: {text}"
    );
    drop(stdin);
    let _ = child.wait_timeout();
}

trait WaitTimeout {
    fn wait_timeout(&mut self) -> std::process::ExitStatus;
}

impl WaitTimeout for std::process::Child {
    fn wait_timeout(&mut self) -> std::process::ExitStatus {
        for _ in 0..50 {
            if let Ok(Some(status)) = self.try_wait() {
                return status;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.kill();
        self.wait().expect("wait after kill")
    }
}
