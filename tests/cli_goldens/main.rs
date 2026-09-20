//! CLI help / usage goldens for `canact` and `canact probe`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

use canact::{
    CapabilityLevel, CapabilityProfile, PROBE_SUITE_VERSION, ProbeCache, ProbeResult, SuiteTier,
};

fn isolated_home() -> &'static std::path::Path {
    static HOME: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    HOME.get_or_init(|| {
        let dir = tempfile::tempdir().expect("isolated HOME");
        let path = dir.path().to_path_buf();
        std::mem::forget(dir);
        path
    })
}

fn canact() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_canact"));
    cmd.env("NO_COLOR", "1");
    cmd.env_remove("CLICOLOR_FORCE");
    cmd.env_remove("GROK_API_KEY");
    cmd.env_remove("GROQ_API_KEY");
    cmd.env_remove("AWS_BEARER_TOKEN_BEDROCK");
    cmd.env("HOME", isolated_home());
    #[cfg(windows)]
    cmd.env("USERPROFILE", isolated_home());
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
fn root_no_args_prints_help() {
    let out = canact().output().expect("spawn canact");
    assert!(!out.status.success(), "no subcommand should fail closed");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("probe"), "{text}");
    assert!(text.contains("export"), "{text}");
    assert!(text.contains("matrix"), "{text}");
}

#[test]
fn root_help_mentions_probe() {
    let help = stdout_of(&["--help"]);
    assert!(help.contains("probe"), "{help}");
    assert!(help.contains("export"), "{help}");
    assert!(help.contains("mcp"), "{help}");
    assert!(help.contains("matrix"), "{help}");
}

#[test]
fn export_help_lists_aider_and_cline() {
    let help = stdout_of(&["export", "--help"]);
    assert!(help.contains("--aider"), "{help}");
    assert!(help.contains("--cline"), "{help}");
    assert!(help.contains("--dir"), "{help}");
}

#[test]
fn mcp_help_mentions_probe_model() {
    let help = stdout_of(&["mcp", "--help"]);
    assert!(
        help.contains("probe_model") || help.contains("host-policy"),
        "{help}"
    );
}

#[test]
fn probe_advertised_context_zero_is_refused() {
    let out = canact()
        .args([
            "probe",
            "--provider",
            "ollama",
            "--model",
            "llama3.2:3b",
            "--advertised-context",
            "0",
            "--cache",
            isolated_home().join("adv0.json").to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn");
    assert!(
        !out.status.success(),
        "advertised-context 0 must fail closed"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("advertised-context") && (err.contains("0") || err.contains("invalid")),
        "{err}"
    );
}

#[test]
fn probe_padded_advertised_context_parses() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong),
        true,
        false,
        Some(4096),
    );
    cache.save(&cache_path).expect("save cache");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--cheap",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--advertised-context",
            " 4096 ",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "padded advertised-context must parse; stdout={stdout}\nstderr={stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["advertisedContextTokens"], 4096, "{value}");
}

#[test]
fn probe_help_lists_cheap_full_vision() {
    let help = stdout_of(&["probe", "--help"]);
    assert!(help.contains("--cheap"), "{help}");
    assert!(help.contains("--full"), "{help}");
    assert!(help.contains("--suite"), "{help}");
    assert!(help.contains("--vision"), "{help}");
    assert!(help.contains("--advertised-context"), "{help}");
}

#[test]
fn probe_cheap_conflicts_with_suite_full() {
    let out = canact()
        .args([
            "probe",
            "--provider",
            "ollama",
            "--model",
            "x",
            "--cheap",
            "--suite=full",
            "--cache",
            isolated_home()
                .join("cheap-suite.json")
                .to_str()
                .expect("utf8"),
        ])
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "cheap+suite=full must fail closed");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--cheap") && err.contains("--suite=full"),
        "{err}"
    );

    let out = canact()
        .args([
            "probe",
            "--provider",
            "ollama",
            "--model",
            "x",
            "--full",
            "--suite=policy",
            "--cache",
            isolated_home()
                .join("full-suite.json")
                .to_str()
                .expect("utf8"),
        ])
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "full+suite=policy must fail closed");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--full") && err.contains("--suite=policy"),
        "{err}"
    );
}

#[test]
fn probe_padded_suite_parses_full() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut policy = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong);
    policy.effective_context_tokens = Some(111);
    cache.put_with_suite(policy, SuiteTier::Policy, false, None);
    let mut full = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong);
    full.effective_context_tokens = Some(999);
    cache.put_with_suite(full, SuiteTier::Full, false, None);
    cache.save(&cache_path).expect("save cache");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--suite",
            " full ",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "padded --suite must parse Full; stdout={stdout}\nstderr={stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["suite"], "full", "{value}");
    assert_eq!(value["effectiveContextTokens"], 999, "{value}");
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
    let mut p = CapabilityProfile::unprobed("weak-tools", "test");
    p.tool_calling = pr("tool_calling", tool);
    p.json_output = pr("json_output", CapabilityLevel::Strong);
    p.instruction_following = pr("instruction_following", CapabilityLevel::Strong);
    p.search_replace = pr("search_replace", CapabilityLevel::Strong);
    p.unified_diff = pr("unified_diff", CapabilityLevel::Medium);
    p.xml_tool_calling = pr("xml_tool_calling", xml);
    p.complex_tool_calling = pr("complex_tool_calling", CapabilityLevel::Weak);
    p.nested_arguments = pr("nested_arguments", CapabilityLevel::Weak);
    p.vision = pr("vision", CapabilityLevel::Weak);
    p.tool_selection = pr("tool_selection", CapabilityLevel::Weak);
    p.streaming_tool_calls = pr("streaming_tool_calls", CapabilityLevel::Weak);
    p.one_shot_tool_plan = pr("one_shot_tool_plan", CapabilityLevel::Weak);
    p.multi_turn_task_sequencing = pr("multi_turn_task_sequencing", CapabilityLevel::Weak);
    p.context_faithfulness = pr("context_faithfulness", CapabilityLevel::Strong);
    p.code_syntax = pr("code_syntax", CapabilityLevel::Strong);
    p.max_tokens_compliance = pr("max_tokens_compliance", CapabilityLevel::Strong);
    p.multi_turn_memory = pr("multi_turn_memory", CapabilityLevel::Strong);
    p.system_message_adherence = pr("system_message_adherence", CapabilityLevel::Strong);
    p.token_efficiency = pr("token_efficiency", CapabilityLevel::Strong);
    p.parallel_tool_scale = pr("parallel_tool_scale", CapabilityLevel::Weak);
    p.probed_at = 1_700_000_000;
    p.effective_context_tokens = Some(8192);
    p.probed_context_floor = Some(8192);
    p
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
    assert!(stdout.contains("Cached (probedAt=1700000000)"), "{stdout}");
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
        None,
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
        cheap_stdout.contains("Cached (probedAt=1700000000)"),
        "{cheap_stdout}"
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

#[cfg(unix)]
fn seed_default_cache_locations(home: &std::path::Path, xdg: &std::path::Path, cache: &ProbeCache) {
    let parents = [
        xdg.join("canact"),
        home.join("Library").join("Caches").join("canact"),
        home.join(".cache").join("canact"),
    ];
    for parent in parents {
        std::fs::create_dir_all(&parent).expect("cache dir");
        cache
            .save(&parent.join("probes.json"))
            .expect("save default cache");
    }
}

#[cfg(unix)]
#[test]
fn probe_whitespace_cache_uses_default_path() {
    let home = tempfile::tempdir().expect("home");
    let xdg = home.path().join("xdg-cache");
    let mut cache = ProbeCache::default();
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong);
    profile.model_id = "whitespace-cache-model".to_owned();
    profile.effective_context_tokens = Some(4242);
    cache.put_with_knobs(profile, true, false, None);
    seed_default_cache_locations(home.path(), &xdg, &cache);
    let out = canact()
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("XDG_CACHE_HOME", &xdg)
        .args([
            "probe",
            "--json",
            "--cheap",
            "--model",
            "whitespace-cache-model",
            "--provider",
            "test",
            "--cache",
            "   ",
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "whitespace --cache must use default; stdout={stdout}\nstderr={stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["model"], "whitespace-cache-model", "{value}");
    assert_eq!(value["effectiveContextTokens"], 4242, "{value}");
}

fn probe_json_from_cache(args: &[&str], cheap: bool, advertised: Option<u32>) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong),
        cheap,
        false,
        advertised,
    );
    cache.save(&cache_path).expect("save cache");
    let cache_str = cache_path.to_str().expect("utf8 cache path");
    let mut cmd_args = vec![
        "probe",
        "--json",
        "--model",
        "weak-tools",
        "--provider",
        "test",
        "--cache",
        cache_str,
    ];
    cmd_args.extend_from_slice(args);
    let out = canact()
        .args(&cmd_args)
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn canact probe --json");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("probe --json must be an object: {e}; stdout={stdout}"))
}

#[test]
fn probe_json_trims_model_whitespace_for_cache_hit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong),
        true,
        false,
        None,
    );
    cache.save(&cache_path).expect("save cache");
    let cache_str = cache_path.to_str().expect("utf8 cache path");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--cheap",
            "--model",
            "weak-tools ",
            "--provider",
            "test",
            "--cache",
            cache_str,
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["model"], "weak-tools", "{value}");
}

#[test]
fn probe_json_trims_provider_whitespace_for_cache_hit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong),
        true,
        false,
        None,
    );
    cache.save(&cache_path).expect("save cache");
    let cache_str = cache_path.to_str().expect("utf8 cache path");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--cheap",
            "--model",
            "weak-tools",
            "--provider",
            " test",
            "--cache",
            cache_str,
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["provider"], "test", "{value}");
}

#[test]
fn probe_json_cache_hit_includes_flags() {
    let cheap = probe_json_from_cache(&["--cheap"], true, None);
    assert_eq!(cheap["cacheable"], true, "{cheap}");
    assert_eq!(cheap["fromCache"], true, "{cheap}");
    assert_eq!(cheap["skipExpensive"], true, "{cheap}");

    let full = probe_json_from_cache(&["--full"], false, None);
    assert_eq!(full["cacheable"], true, "{full}");
    assert_eq!(full["fromCache"], true, "{full}");
    assert_eq!(full["skipExpensive"], false, "{full}");
    assert_eq!(full["suite"], "full", "{full}");
    assert_eq!(cheap["suite"], "policy", "{cheap}");
    assert!(
        cheap.get("constraintPlacement").is_none(),
        "policy must omit constraintPlacement: {cheap}"
    );
    assert!(
        full.get("constraintPlacement").is_none(),
        "full must omit constraintPlacement: {full}"
    );
}

#[test]
fn probe_json_all_suite_emits_constraint_placement() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong);
    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".to_owned(),
        score: 0.1,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "ignored the system prompt".to_owned(),
    };
    cache.put_with_suite(profile, SuiteTier::All, false, None);
    cache.save(&cache_path).expect("save cache");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--suite=all",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn canact probe --json");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["suite"], "all", "{value}");
    assert_eq!(value["constraintPlacement"], "user", "{value}");
}

#[test]
fn probe_json_policy_fallback_reports_full_row_suite() {
    let value = probe_json_from_cache(&["--cheap"], false, None);
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["suite"], "full", "{value}");
    assert_eq!(value["skipExpensive"], false, "{value}");
}

#[test]
fn probe_json_policy_fallback_reports_all_row_suite() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong);
    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".to_owned(),
        score: 1.0,
        max_score: 1.0,
        level: CapabilityLevel::Strong,
        details: "followed the system prompt".to_owned(),
    };
    cache.put_with_suite(profile, SuiteTier::All, false, None);
    cache.save(&cache_path).expect("save cache");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--cheap",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn canact probe --json");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["suite"], "all", "{value}");
    assert_eq!(value["constraintPlacement"], "system", "{value}");
}

#[test]
fn probe_json_policy_only_row_omits_constraint_placement() {
    let value = probe_json_from_cache(&["--cheap"], true, None);
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["suite"], "policy", "{value}");
    assert!(
        value.get("constraintPlacement").is_none(),
        "policy-only row must omit constraintPlacement: {value}"
    );
}

#[test]
fn probe_json_reuses_catalog_filled_row_without_flags() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        cached_profile(CapabilityLevel::Strong, CapabilityLevel::Strong),
        false,
        true,
        Some(200_000),
    );
    cache.save(&cache_path).expect("save cache");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--full",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("XAI_API_KEY")
        .output()
        .expect("spawn canact probe --json");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "catalog-filled cache must hit without a key or GET /models: stdout={stdout}\nstderr={stderr}"
    );
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["advertisedContextTokens"], 200_000, "{value}");
}

#[test]
fn probe_json_cache_hit_includes_advertised_context() {
    let value = probe_json_from_cache(
        &["--cheap", "--advertised-context", "40960"],
        true,
        Some(40960),
    );
    assert_eq!(value["advertisedContextTokens"], 40960, "{value}");
    assert_eq!(value["recommendedContextTokens"], 8192, "{value}");
    assert_eq!(value["cacheable"], true, "{value}");
    assert_eq!(value["fromCache"], true, "{value}");
    assert_eq!(value["skipExpensive"], true, "{value}");
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

#[test]
fn probe_without_key_prints_missing_key_and_exits() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args([
            "probe",
            "--model",
            "gpt-4o",
            "--provider",
            "openai",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("OPENAI_API_KEY") || stderr.contains("--api-key"),
        "stderr={stderr}"
    );
}

#[test]
fn probe_ollama_uses_full_cache_row_when_cheap() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "qwen2.5-coder".to_owned();
    profile.provider = "ollama".to_owned();
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(profile, false, false, None);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "probe",
            "--json",
            "--model",
            "qwen2.5-coder",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .expect("spawn probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("qwen2.5-coder"), "{stdout}");
    assert!(
        !stderr.contains("authentication error"),
        "must not fall through to a live host: {stderr}"
    );
}

#[test]
fn export_aider_writes_cwd_when_dir_omitted() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(cached_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
    ));
    cache.save(&cache_path).expect("save");
    let out = canact()
        .current_dir(dir.path())
        .args([
            "export",
            "--aider",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    assert!(
        dir.path().join(".aider.model.settings.yml").is_file(),
        "settings missing; stderr={stderr}"
    );
    assert!(
        dir.path().join(".aider.model.metadata.json").is_file(),
        "metadata missing; stderr={stderr}"
    );
    assert!(stderr.contains("wrote"), "stderr={stderr}");
}

#[test]
fn export_whitespace_dir_writes_cwd() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(cached_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
    ));
    cache.save(&cache_path).expect("save");
    let out = canact()
        .current_dir(dir.path())
        .args([
            "export",
            "--aider",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            "   ",
        ])
        .output()
        .expect("spawn export");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    assert!(
        dir.path().join(".aider.model.settings.yml").is_file(),
        "whitespace --dir must omit to cwd; stderr={stderr}"
    );
    assert!(
        dir.path().join(".aider.model.metadata.json").is_file(),
        "whitespace --dir must omit to cwd; stderr={stderr}"
    );
}

#[test]
fn export_cline_without_advertised_window_omits_token_fields() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.effective_context_tokens = Some(8192);
    profile.probed_context_floor = Some(8192);
    let mut cache = ProbeCache::default();
    cache.put(profile);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "export",
            "--cline",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            dir.path().to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    let body = std::fs::read_to_string(dir.path().join("cline.modelinfo.json")).expect("json");
    let value: serde_json::Value = serde_json::from_str(&body).expect("parse");
    assert!(value.get("contextWindow").is_none(), "{value}");
    assert!(value.get("maxTokens").is_none(), "{value}");
}

#[test]
fn export_aider_without_advertised_window_omits_token_fields() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.effective_context_tokens = Some(8192);
    profile.probed_context_floor = Some(8192);
    let mut cache = ProbeCache::default();
    cache.put(profile);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "export",
            "--aider",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            dir.path().to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    let metadata =
        std::fs::read_to_string(dir.path().join(".aider.model.metadata.json")).expect("metadata");
    let value: serde_json::Value = serde_json::from_str(&metadata).expect("parse");
    let entry = &value["test/weak-tools"];
    assert!(entry.get("max_input_tokens").is_none(), "{value}");
    assert!(entry.get("max_output_tokens").is_none(), "{value}");
}

#[test]
fn export_dir_file_explains_not_os_error_17() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(cached_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
    ));
    cache.save(&cache_path).expect("save");
    let as_file = dir.path().join("notadir");
    std::fs::write(&as_file, b"nope").expect("file");
    let out = canact()
        .args([
            "export",
            "--aider",
            "--model",
            "weak-tools",
            "--provider",
            "test",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            as_file.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.contains("must be a directory"), "stderr={stderr}");
    assert!(
        !stderr.contains("os error 17"),
        "raw OS error leaked: {stderr}"
    );
}

#[test]
fn matrix_help_lists_provider() {
    let help = stdout_of(&["matrix", "--help"]);
    assert!(help.contains("--provider"), "{help}");
    assert!(help.contains("--cache"), "{help}");
    assert!(
        help.contains("matrix [OPTIONS]"),
        "provider must be optional: {help}"
    );
}

#[test]
fn matrix_without_provider_lists_every_cached_row() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut grok = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    grok.model_id = "grok-4-fast-non-reasoning".to_owned();
    grok.provider = "grok".to_owned();
    let mut ollama = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    ollama.model_id = "llama3.2:3b".to_owned();
    ollama.provider = "ollama".to_owned();
    cache.put(grok);
    cache.put(ollama);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args(["matrix", "--cache", cache_path.to_str().expect("utf8")])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert!(value.get("provider").is_none(), "{value}");
    let rows = value["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "{value}");
    let models: Vec<&str> = rows
        .iter()
        .map(|r| r["model"].as_str().expect("model"))
        .collect();
    assert!(models.contains(&"grok-4-fast-non-reasoning"), "{value}");
    assert!(models.contains(&"llama3.2:3b"), "{value}");
}

#[test]
fn matrix_provider_grok_filters_mixed_cache() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut grok = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    grok.model_id = "grok-4-fast-non-reasoning".to_owned();
    grok.provider = "grok".to_owned();
    let mut ollama = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    ollama.model_id = "llama3.2:3b".to_owned();
    ollama.provider = "ollama".to_owned();
    cache.put(grok);
    cache.put(ollama);
    cache.save(&cache_path).expect("save");
    let path = cache_path.to_str().expect("utf8");
    for provider_arg in ["grok", " grok "] {
        let out = canact()
            .args(["matrix", "--provider", provider_arg, "--cache", path])
            .output()
            .expect("spawn matrix");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "provider={provider_arg:?} stdout={stdout}\nstderr={stderr}"
        );
        let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(value["provider"], "grok", "{value}");
        let rows = value["rows"].as_array().expect("rows");
        assert_eq!(rows.len(), 1, "{value}");
        assert_eq!(rows[0]["model"], "grok-4-fast-non-reasoning", "{value}");
    }
}

#[test]
fn matrix_whitespace_provider_lists_every_cached_row() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut grok = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    grok.model_id = "grok-4-fast-non-reasoning".to_owned();
    grok.provider = "grok".to_owned();
    let mut ollama = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    ollama.model_id = "llama3.2:3b".to_owned();
    ollama.provider = "ollama".to_owned();
    cache.put(grok);
    cache.put(ollama);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "   ",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert!(value.get("provider").is_none(), "{value}");
    assert_eq!(value["rows"].as_array().expect("rows").len(), 2, "{value}");
}

#[test]
fn matrix_missing_provider_fails_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut grok = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    grok.model_id = "grok-4-fast-non-reasoning".to_owned();
    grok.provider = "grok".to_owned();
    cache.put(grok);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "missing",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(
        stderr.contains("no cached probes for missing"),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("required arguments"),
        "unknown provider must not be clap-required: {stderr}"
    );
}

#[test]
fn matrix_without_provider_empty_cache_fails_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args(["matrix", "--cache", cache_path.to_str().expect("utf8")])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(stderr.contains("no cached probes"), "stderr={stderr}");
    assert!(
        !stderr.contains("required arguments"),
        "omitted --provider must not be clap-required: {stderr}"
    );
}

#[test]
fn matrix_empty_cache_fails_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(stderr.contains("no cached probes"), "stderr={stderr}");
}

#[test]
fn matrix_empty_object_cache_fails_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    std::fs::write(&cache_path, "{}").expect("write empty object");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(stderr.contains("no cached probes"), "stderr={stderr}");
    assert!(
        !stderr.contains("missing field"),
        "empty object must not be a serde error: {stderr}"
    );
}

#[test]
fn matrix_prints_json_without_score() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    let mut a = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    a.model_id = "alpha".to_owned();
    a.provider = "ollama".to_owned();
    let mut b = cached_profile(CapabilityLevel::Weak, CapabilityLevel::Medium);
    b.model_id = "beta".to_owned();
    b.provider = "ollama".to_owned();
    cache.put(a);
    cache.put(b);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["provider"], "ollama", "{value}");
    assert_eq!(value["rows"].as_array().unwrap().len(), 2, "{value}");
    assert!(value.get("score").is_none(), "{value}");
    assert!(value.get("overall").is_none(), "{value}");
    assert_eq!(value["rows"][0]["model"], "alpha", "{value}");
    assert_eq!(value["rows"][0]["nativeTools"], "pass", "{value}");
    assert_eq!(value["rows"][1]["nativeTools"], "fail", "{value}");
}

#[test]
fn matrix_policy_row_skips_unmeasured_output_and_placement() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "cheap-row".to_owned();
    profile.provider = "grok".to_owned();
    profile.max_output_tokens = None;
    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".into(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Skipped: diagnostic suite (use --suite=all)".into(),
    };
    let mut cache = ProbeCache::default();
    cache.put_with_suite(profile, SuiteTier::Policy, false, None);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "grok",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["rows"][0]["maxOutputTokens"], "skipped", "{value}");
    assert_eq!(
        value["rows"][0]["constraintPlacement"], "skipped",
        "{value}"
    );
}

#[test]
fn matrix_full_row_omitted_output_cap_is_fail() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "full-row".to_owned();
    profile.provider = "grok".to_owned();
    profile.max_output_tokens = None;
    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".into(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Skipped: diagnostic suite (use --suite=all)".into(),
    };
    let mut cache = ProbeCache::default();
    cache.put_with_suite(profile, SuiteTier::Full, false, None);
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "grok",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(value["rows"][0]["maxOutputTokens"], "fail", "{value}");
    assert_eq!(
        value["rows"][0]["constraintPlacement"], "skipped",
        "{value}"
    );
}

#[test]
fn probe_named_xai_with_openai_env_asks_for_xai_key() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args([
            "probe",
            "--provider",
            "xai",
            "--model",
            "grok-test",
            "--cheap",
            "--force",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env("OPENAI_API_KEY", "sk-openai-must-not-go-to-xai")
        .env_remove("XAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .output()
        .expect("spawn probe");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(stderr.contains("XAI_API_KEY"), "stderr={stderr}");
    assert!(
        !stderr.contains("set --api-key, OPENAI_API_KEY"),
        "named xAI must not treat OPENAI_API_KEY as the fix: {stderr}"
    );
}

#[test]
fn probe_named_groq_with_openai_env_asks_for_groq_key() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args([
            "probe",
            "--provider",
            "groq",
            "--model",
            "llama-test",
            "--cheap",
            "--force",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env("OPENAI_API_KEY", "sk-openai-must-not-go-to-groq")
        .env_remove("XAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .output()
        .expect("spawn probe");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(stderr.contains("GROQ_API_KEY"), "stderr={stderr}");
    assert!(
        !stderr.contains("set --api-key, OPENAI_API_KEY"),
        "named Groq must not treat OPENAI_API_KEY as the fix: {stderr}"
    );
}

#[test]
fn probe_named_bedrock_with_openai_env_asks_for_bedrock_token() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let out = canact()
        .args([
            "probe",
            "--provider",
            "amazon-bedrock",
            "--model",
            "claude-test",
            "--cheap",
            "--force",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .env("OPENAI_API_KEY", "sk-openai-must-not-go-to-bedrock")
        .env_remove("XAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .output()
        .expect("spawn probe");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(
        stderr.contains("AWS_BEARER_TOKEN_BEDROCK"),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("set --api-key, OPENAI_API_KEY"),
        "named Bedrock must not treat OPENAI_API_KEY as the fix: {stderr}"
    );
}

#[test]
fn export_uses_cached_advertised_window() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut older = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    older.model_id = "grok-window".to_owned();
    older.provider = "xai".to_owned();
    let newer = older.clone();
    let mut cache = ProbeCache::default();
    cache.put(older);
    for entry in cache.profiles.values_mut() {
        entry.cached_at = 1;
    }
    cache.put_with_knobs(newer, true, false, Some(1_000_000));
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "export",
            "--aider",
            "--model",
            "grok-window",
            "--provider",
            "xai",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            dir.path().to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    let metadata =
        std::fs::read_to_string(dir.path().join(".aider.model.metadata.json")).expect("metadata");
    let value: serde_json::Value = serde_json::from_str(&metadata).expect("parse");
    assert_eq!(
        value["xai/grok-window"]["max_input_tokens"], 1_000_000,
        "{value}"
    );
}

#[test]
fn export_stale_suite_explains_rerun() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "llama3.2:3b".to_owned();
    profile.provider = "ollama".to_owned();
    let mut cache = ProbeCache::default();
    cache.put_with_settings(
        profile,
        "unset",
        PROBE_SUITE_VERSION - 1,
        true,
        false,
        Some(131_072),
    );
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "export",
            "--aider",
            "--model",
            "llama3.2:3b",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            dir.path().to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn export");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(
        stderr.contains(&format!("suite {}", PROBE_SUITE_VERSION - 1)),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains(&format!("need {PROBE_SUITE_VERSION}")),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("no cached probe"),
        "stale suite must not look empty: {stderr}"
    );
}

#[test]
fn matrix_stale_suite_explains_rerun() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "llama3.2:3b".to_owned();
    profile.provider = "ollama".to_owned();
    let mut cache = ProbeCache::default();
    cache.put_with_settings(
        profile,
        "unset",
        PROBE_SUITE_VERSION - 1,
        true,
        false,
        Some(131_072),
    );
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args([
            "matrix",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
        ])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(
        stderr.contains(&format!("suite {}", PROBE_SUITE_VERSION - 1)),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains(&format!("need {PROBE_SUITE_VERSION}")),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("no cached probes"),
        "stale suite must not look empty: {stderr}"
    );
}

#[test]
fn matrix_without_provider_stale_only_explains_rerun() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut profile = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    profile.model_id = "llama3.2:3b".to_owned();
    profile.provider = "ollama".to_owned();
    let mut cache = ProbeCache::default();
    cache.put_with_settings(
        profile,
        "unset",
        PROBE_SUITE_VERSION - 1,
        true,
        false,
        Some(131_072),
    );
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args(["matrix", "--cache", cache_path.to_str().expect("utf8")])
        .output()
        .expect("spawn matrix");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr={stderr}");
    assert!(
        stderr.contains(&format!("suite {}", PROBE_SUITE_VERSION - 1)),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains(&format!("need {PROBE_SUITE_VERSION}")),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("no cached probes"),
        "unfiltered stale suite must not look empty: {stderr}"
    );
}

#[test]
fn matrix_without_provider_skips_stale_and_keeps_current() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut grok = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    grok.model_id = "grok-4-fast-non-reasoning".to_owned();
    grok.provider = "grok".to_owned();
    let mut ollama = cached_profile(CapabilityLevel::Strong, CapabilityLevel::Medium);
    ollama.model_id = "llama3.2:3b".to_owned();
    ollama.provider = "ollama".to_owned();
    let mut cache = ProbeCache::default();
    cache.put(grok);
    cache.put_with_settings(
        ollama,
        "unset",
        PROBE_SUITE_VERSION - 1,
        true,
        false,
        Some(131_072),
    );
    cache.save(&cache_path).expect("save");
    let out = canact()
        .args(["matrix", "--cache", cache_path.to_str().expect("utf8")])
        .output()
        .expect("spawn matrix");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    let rows = value["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "{value}");
    assert_eq!(rows[0]["model"], "grok-4-fast-non-reasoning", "{value}");
}
