//! CLI help / usage goldens for `canact` and `canact probe`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

use canact::{CapabilityLevel, CapabilityProfile, ProbeCache, ProbeResult};

fn canact() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_canact"));
    cmd.env("NO_COLOR", "1");
    cmd.env_remove("CLICOLOR_FORCE");
    cmd
}

fn stdout_of(args: &[&str]) -> String {
    let out = canact()
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn canact {args:?}: {e}"));
    assert!(
        out.status.success(),
        "canact {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn root_no_args_prints_not_ready() {
    let out = canact().output().expect("spawn canact");
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "Not ready.");
}

#[test]
fn root_help_mentions_probe() {
    let help = stdout_of(&["--help"]);
    assert!(help.contains("probe"), "{help}");
}

#[test]
fn probe_help_lists_cheap_full_vision() {
    let help = stdout_of(&["probe", "--help"]);
    assert!(help.contains("--cheap"), "{help}");
    assert!(help.contains("--full"), "{help}");
    assert!(help.contains("--vision"), "{help}");
}

#[test]
fn root_help_lists_cheap_full_vision_via_probe() {
    let help = stdout_of(&["--help"]);
    let probe = stdout_of(&["probe", "--help"]);
    assert!(
        help.contains("probe")
            && probe.contains("--cheap")
            && probe.contains("--full")
            && probe.contains("--vision"),
        "root={help}\nprobe={probe}"
    );
}

fn cached_profile(tool: CapabilityLevel, xml: CapabilityLevel) -> CapabilityProfile {
    let pr = |name: &str, level: CapabilityLevel| ProbeResult {
        name: name.to_owned(),
        score: match level {
            CapabilityLevel::Strong => 1.0,
            CapabilityLevel::Medium => 0.5,
            CapabilityLevel::Weak => 0.1,
        },
        max_score: 1.0,
        level,
        details: "test".to_owned(),
    };
    CapabilityProfile {
        model_id: "weak-tools".to_owned(),
        provider: "test".to_owned(),
        tool_calling: pr("tool_calling", tool),
        json_output: pr("json_output", CapabilityLevel::Strong),
        instruction_following: pr("instruction_following", CapabilityLevel::Strong),
        search_replace: pr("search_replace", CapabilityLevel::Strong),
        unified_diff: pr("unified_diff", CapabilityLevel::Medium),
        xml_tool_calling: pr("xml_tool_calling", xml),
        complex_tool_calling: pr("complex_tool_calling", CapabilityLevel::Weak),
        nested_arguments: pr("nested_arguments", CapabilityLevel::Weak),
        vision: pr("vision", CapabilityLevel::Weak),
        tool_selection: pr("tool_selection", CapabilityLevel::Weak),
        streaming_tool_calls: pr("streaming_tool_calls", CapabilityLevel::Weak),
        one_shot_tool_plan: pr("one_shot_tool_plan", CapabilityLevel::Weak),
        multi_turn_task_sequencing: pr("multi_turn_task_sequencing", CapabilityLevel::Weak),
        context_faithfulness: pr("context_faithfulness", CapabilityLevel::Strong),
        code_syntax: pr("code_syntax", CapabilityLevel::Strong),
        max_tokens_compliance: pr("max_tokens_compliance", CapabilityLevel::Strong),
        multi_turn_memory: pr("multi_turn_memory", CapabilityLevel::Strong),
        system_message_adherence: pr("system_message_adherence", CapabilityLevel::Strong),
        token_efficiency: pr("token_efficiency", CapabilityLevel::Strong),
        parallel_tool_scale: pr("parallel_tool_scale", CapabilityLevel::Weak),
        probed_at: 1_700_000_000,
        effective_context_tokens: Some(8192),
    }
}

#[test]
fn probe_cached_weak_tools_exits_2_and_explains() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(cached_profile(CapabilityLevel::Weak, CapabilityLevel::Weak));
    cache.save(&cache_path).expect("save cache");

    let out = canact()
        .args([
            "probe",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8 cache path"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn canact probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.contains("cannot use tools"), "stderr={stderr}");
    assert!(stdout.contains("=== Probe Results ==="), "{stdout}");
    assert!(stdout.contains("8192"), "{stdout}");
    assert!(stdout.contains("Effective context tokens:"), "{stdout}");
}

#[test]
fn probe_cheap_cache_is_not_returned_on_full() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Weak, CapabilityLevel::Weak),
        true,
        false,
    );
    cache.save(&cache_path).expect("save cheap cache");
    let cache_str = cache_path.to_str().expect("utf8 cache path");

    let cheap_hit = canact()
        .args([
            "probe",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cheap",
            "--cache",
            cache_str,
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn cheap probe");
    let cheap_stdout = String::from_utf8_lossy(&cheap_hit.stdout);
    let cheap_stderr = String::from_utf8_lossy(&cheap_hit.stderr);
    assert_eq!(
        cheap_hit.status.code(),
        Some(2),
        "cheap must hit cache; stdout={cheap_stdout}\nstderr={cheap_stderr}"
    );
    assert!(
        cheap_stdout.contains("=== Probe Results ==="),
        "{cheap_stdout}"
    );

    let base = spawn_401(br#"{"error":{"message":"full must miss cheap cache"}}"#);
    let full_miss = canact()
        .args([
            "probe",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--full",
            "--base-url",
            &base,
            "--api-key",
            "sk-cli-secret",
            "--cache",
            cache_str,
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn full probe");
    let full_stdout = String::from_utf8_lossy(&full_miss.stdout);
    let full_stderr = String::from_utf8_lossy(&full_miss.stderr);
    assert_eq!(
        full_miss.status.code(),
        Some(1),
        "full must miss cheap cache and probe; stdout={full_stdout}\nstderr={full_stderr}"
    );
    assert!(
        !full_stdout.contains("=== Probe Results ==="),
        "full must not emit the cheap-cached table; stdout={full_stdout}"
    );
    assert!(
        full_stderr.contains("authentication error:"),
        "stderr={full_stderr}"
    );
}

fn spawn_401(body: &[u8]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let body = body.to_vec();
    thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut buf = [0u8; 4096];
            let mut got = Vec::new();
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        got.extend_from_slice(&buf[..n]);
                        if got.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            let head = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        }
    });
    format!("http://{addr}/v1")
}

#[test]
fn probe_auth_prints_authentication_error_once() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let base = spawn_401(br#"{"error":{"message":"Bearer SECRET sk-live-secret"}}"#);
    let out = canact()
        .args([
            "probe",
            "--model",
            "m",
            "--provider",
            "test",
            "--base-url",
            &base,
            "--api-key",
            "sk-cli-secret",
            "--cache",
            cache_path.to_str().expect("utf8 cache path"),
            "--force",
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn canact probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert_eq!(
        stderr.matches("authentication error:").count(),
        1,
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("authentication error: authentication error:"),
        "stderr={stderr}"
    );
    assert!(!stderr.contains("SECRET"), "stderr={stderr}");
    assert!(!stderr.contains("sk-live-secret"), "stderr={stderr}");
    assert!(!stderr.contains("sk-cli-secret"), "stderr={stderr}");
}

#[test]
fn probe_auth_redacts_api_key_underscore() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let base = spawn_401(br#"{"api_key":"SECRET","error":{"message":"api_key=SECRET"}}"#);
    let out = canact()
        .args([
            "probe",
            "--model",
            "m",
            "--provider",
            "test",
            "--base-url",
            &base,
            "--api-key",
            "sk-cli-secret",
            "--cache",
            cache_path.to_str().expect("utf8 cache path"),
            "--force",
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn canact probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(!stderr.contains("SECRET"), "stderr={stderr}");
    assert!(!stdout.contains("SECRET"), "stdout={stdout}");
    assert!(stderr.contains("[REDACTED]"), "stderr={stderr}");
    assert!(stderr.contains("authentication error:"), "stderr={stderr}");
}
