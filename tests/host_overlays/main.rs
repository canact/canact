//! Integration: canact overlays must construct Aider `ModelSettings`
//! and match Cline `ModelInfo` keys. Optional live `aider` CLI check.

use std::path::PathBuf;
use std::process::Command;

use canact::{
    AiderSettingsRow, CapabilityLevel, CapabilityProfile, HostOverlay, ProbeCache, ProbeResult,
};

fn sample(search: CapabilityLevel, unified: CapabilityLevel) -> CapabilityProfile {
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
        model_id: "qwen2.5-coder".to_owned(),
        provider: "ollama".to_owned(),
        tool_calling: pr("tool_calling", CapabilityLevel::Strong),
        json_output: pr("json_output", CapabilityLevel::Strong),
        instruction_following: pr("instruction_following", CapabilityLevel::Strong),
        search_replace: pr("search_replace", search),
        unified_diff: pr("unified_diff", unified),
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

fn python3() -> Option<Command> {
    for bin in ["python3", "python"] {
        if Command::new(bin)
            .arg("-c")
            .arg("import sys; raise SystemExit(0)")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some(Command::new(bin));
        }
    }
    None
}

fn aider_helper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/host_overlays/aider_model_settings.py")
}

#[test]
fn aider_model_settings_dataclass_accepts_export_row() {
    let Some(mut py) = python3() else {
        panic!("python3 is required for the Aider ModelSettings integration test");
    };
    let overlay = HostOverlay::aider(
        &sample(CapabilityLevel::Strong, CapabilityLevel::Medium),
        Some(40960),
    );
    let HostOverlay::Aider(aider) = overlay else {
        panic!("expected Aider overlay");
    };
    let row: &AiderSettingsRow = &aider.settings[0];
    let payload = serde_json::to_vec(row).expect("row json");
    let out = py
        .arg(aider_helper())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().expect("stdin").write_all(&payload)?;
            child.wait_with_output()
        })
        .expect("run aider ModelSettings helper");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "Aider ModelSettings rejected the row: stdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("ollama/qwen2.5-coder"), "{stdout}");
    assert!(stdout.contains("\ndiff\n"), "{stdout}");
    assert!(stdout.contains("ok"), "{stdout}");
}

#[test]
fn cline_modelinfo_json_has_only_official_keys() {
    let overlay = HostOverlay::cline(&sample(CapabilityLevel::Weak, CapabilityLevel::Weak), None);
    let files = overlay.files();
    let body = &files[0].body;
    let value: serde_json::Value = serde_json::from_str(body).expect("json");
    let official = [
        "maxTokens",
        "contextWindow",
        "supportsImages",
        "supportsPromptCache",
        "inputPrice",
        "outputPrice",
        "cacheWritesPrice",
        "cacheReadsPrice",
        "description",
    ];
    for key in value.as_object().expect("object").keys() {
        assert!(
            official.contains(&key.as_str()),
            "Cline ModelInfo does not declare {key}"
        );
    }
    assert_eq!(value["contextWindow"], 8192, "{value}");
    assert_eq!(value["supportsImages"], false, "{value}");
}

fn canact() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_canact"));
    cmd.env("NO_COLOR", "1");
    cmd
}

#[test]
fn cli_export_aider_and_cline_from_cache() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(sample(CapabilityLevel::Strong, CapabilityLevel::Medium));
    cache.save(&cache_path).expect("save");
    let out_dir = dir.path().join("overlays");

    let aider = canact()
        .args([
            "export",
            "--aider",
            "--model",
            "qwen2.5-coder",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            out_dir.to_str().expect("utf8"),
        ])
        .output()
        .expect("export aider");
    assert!(
        aider.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&aider.stderr)
    );
    let settings = std::fs::read_to_string(out_dir.join(".aider.model.settings.yml")).expect("yml");
    assert!(settings.contains("edit_format: diff"), "{settings}");

    let cline = canact()
        .args([
            "export",
            "--cline",
            "--model",
            "qwen2.5-coder",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            out_dir.to_str().expect("utf8"),
        ])
        .output()
        .expect("export cline");
    assert!(
        cline.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&cline.stderr)
    );
    assert!(out_dir.join("cline.modelinfo.json").is_file());
}

#[test]
fn cli_export_dir_expands_tilde() {
    let dir = tempfile::tempdir().expect("temp dir");
    let home = dir.path().join("home");
    std::fs::create_dir(&home).expect("home");
    let cache_path = dir.path().join("probes.json");
    let mut cache = ProbeCache::default();
    cache.put(sample(CapabilityLevel::Strong, CapabilityLevel::Medium));
    cache.save(&cache_path).expect("save");

    let export = canact()
        .env("HOME", &home)
        .args([
            "export",
            "--aider",
            "--model",
            "qwen2.5-coder",
            "--provider",
            "ollama",
            "--cache",
            cache_path.to_str().expect("utf8"),
            "--dir",
            "~/overlays",
        ])
        .output()
        .expect("export aider");
    assert!(
        export.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&export.stderr)
    );
    assert!(
        home.join("overlays/.aider.model.settings.yml").is_file(),
        "stderr={}",
        String::from_utf8_lossy(&export.stderr)
    );
}

#[test]
fn optional_aider_cli_loads_exported_settings() {
    let aider = Command::new("aider").arg("--help").output();
    let Ok(help) = aider else {
        return;
    };
    if !help.status.success() {
        return;
    }
    let help_txt = String::from_utf8_lossy(&help.stdout);
    assert!(
        help_txt.contains("model-settings-file") || help_txt.contains("model-settings"),
        "aider --help should mention model settings; got {help_txt}"
    );
    let dir = tempfile::tempdir().expect("temp dir");
    HostOverlay::aider(
        &sample(CapabilityLevel::Strong, CapabilityLevel::Medium),
        None,
    )
    .write_to(dir.path())
    .expect("write");
    let settings = dir.path().join(".aider.model.settings.yml");
    let listed = Command::new("aider")
        .current_dir(dir.path())
        .args([
            "--model-settings-file",
            settings.to_str().expect("utf8"),
            "--list-models",
            "qwen",
        ])
        .output();
    let Ok(out) = listed else {
        return;
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success()
            || stderr.contains("qwen")
            || stdout.contains("qwen")
            || !stderr.to_ascii_lowercase().contains("traceback"),
        "aider rejected the overlay: stdout={stdout}\nstderr={stderr}"
    );
}
