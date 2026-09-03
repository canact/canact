//! Overlay exporters must only emit keys the host will accept.

use canact::{
    CapabilityLevel, CapabilityProfile, ClineModelInfo, HostOverlay, ProbeCache, ProbeResult,
    overlay_model_name,
};

fn sample(
    search: CapabilityLevel,
    unified: CapabilityLevel,
    vision: CapabilityLevel,
) -> CapabilityProfile {
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
        vision: pr("vision", vision),
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

#[test]
fn cache_find_profile_sees_cheap_row() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(
        sample(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        ),
        true,
        false,
        None,
    );
    let found = cache
        .find_profile("qwen2.5-coder", "ollama")
        .expect("cheap row");
    assert_eq!(found.effective_context_tokens, Some(8192));
}

#[test]
fn host_overlay_write_aider_pair() {
    let dir = tempfile::tempdir().expect("temp dir");
    let overlay = HostOverlay::aider(
        &sample(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        ),
        Some(40960),
    );
    let paths = overlay.write_to(dir.path()).expect("write");
    assert_eq!(paths.len(), 2);
    let settings =
        std::fs::read_to_string(dir.path().join(".aider.model.settings.yml")).expect("settings");
    let metadata =
        std::fs::read_to_string(dir.path().join(".aider.model.metadata.json")).expect("metadata");
    assert!(settings.contains("edit_format: diff"), "{settings}");
    let value: serde_json::Value = serde_json::from_str(&metadata).expect("json");
    assert_eq!(
        value["ollama/qwen2.5-coder"]["max_input_tokens"], 8192,
        "{value}"
    );
}

#[test]
fn host_overlay_write_cline_json() {
    let dir = tempfile::tempdir().expect("temp dir");
    let overlay = HostOverlay::cline(
        &sample(
            CapabilityLevel::Weak,
            CapabilityLevel::Weak,
            CapabilityLevel::Strong,
        ),
        None,
    );
    overlay.write_to(dir.path()).expect("write");
    let body = std::fs::read_to_string(dir.path().join("cline.modelinfo.json")).expect("json");
    let info: ClineModelInfo = serde_json::from_str(&body).expect("parse");
    assert_eq!(info.context_window, Some(8192));
    assert!(info.supports_images);
}

#[test]
fn overlay_model_name_stable() {
    let p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    assert_eq!(overlay_model_name(&p), "ollama/qwen2.5-coder");
}
