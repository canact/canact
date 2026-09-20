//! Overlay exporters must only emit keys the host will accept.

use canact::{
    CapabilityLevel, CapabilityProfile, ClineModelInfo, HostOverlay, ProbeResult,
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
    let mut p = CapabilityProfile::unprobed("qwen2.5-coder", "ollama");
    p.tool_calling = pr("tool_calling", CapabilityLevel::Strong);
    p.json_output = pr("json_output", CapabilityLevel::Strong);
    p.instruction_following = pr("instruction_following", CapabilityLevel::Strong);
    p.search_replace = pr("search_replace", search);
    p.unified_diff = pr("unified_diff", unified);
    p.xml_tool_calling = pr("xml_tool_calling", CapabilityLevel::Medium);
    p.complex_tool_calling = pr("complex_tool_calling", CapabilityLevel::Strong);
    p.nested_arguments = pr("nested_arguments", CapabilityLevel::Strong);
    p.vision = pr("vision", vision);
    p.tool_selection = pr("tool_selection", CapabilityLevel::Medium);
    p.streaming_tool_calls = pr("streaming_tool_calls", CapabilityLevel::Strong);
    p.one_shot_tool_plan = pr("one_shot_tool_plan", CapabilityLevel::Strong);
    p.multi_turn_task_sequencing = pr("multi_turn_task_sequencing", CapabilityLevel::Strong);
    p.context_faithfulness = pr("context_faithfulness", CapabilityLevel::Strong);
    p.code_syntax = pr("code_syntax", CapabilityLevel::Strong);
    p.max_tokens_compliance = pr("max_tokens_compliance", CapabilityLevel::Strong);
    p.multi_turn_memory = pr("multi_turn_memory", CapabilityLevel::Strong);
    p.system_message_adherence = pr("system_message_adherence", CapabilityLevel::Strong);
    p.token_efficiency = pr("token_efficiency", CapabilityLevel::Strong);
    p.parallel_tool_scale = pr("parallel_tool_scale", CapabilityLevel::Strong);
    p.probed_at = 1_700_000_000;
    p.effective_context_tokens = Some(8192);
    p.probed_context_floor = Some(8192);
    p
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
        value["ollama/qwen2.5-coder"]["max_input_tokens"], 40960,
        "{value}"
    );
    assert!(
        value["ollama/qwen2.5-coder"]
            .get("max_output_tokens")
            .is_none(),
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
    assert_eq!(info.context_window, None);
    assert_eq!(info.max_tokens, None);
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

#[test]
fn overlay_model_name_trims_model_id() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "qwen2.5-coder ".into();
    assert_eq!(overlay_model_name(&p), "ollama/qwen2.5-coder");
}

#[test]
fn overlay_localhost_port_still_maps_to_ollama() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.provider = "127.0.0.1:1234".into();
    assert_eq!(
        overlay_model_name(&p),
        "ollama/qwen2.5-coder",
        "export still maps loopback host:port to ollama"
    );
    p.provider = "localhost:11434".into();
    assert_eq!(overlay_model_name(&p), "ollama/qwen2.5-coder");
}

#[test]
fn overlay_xai_host_maps_to_xai_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "grok-4".into();
    p.provider = "api.x.ai".into();
    assert_eq!(overlay_model_name(&p), "xai/grok-4");
    p.provider = "grok".into();
    assert_eq!(overlay_model_name(&p), "xai/grok-4");
}

#[test]
fn overlay_anthropic_host_maps_to_anthropic_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "claude-haiku-4-5-20251001".into();
    p.provider = "api.anthropic.com".into();
    assert_eq!(
        overlay_model_name(&p),
        "anthropic/claude-haiku-4-5-20251001"
    );
    p.provider = "claude".into();
    assert_eq!(
        overlay_model_name(&p),
        "anthropic/claude-haiku-4-5-20251001"
    );
}

#[test]
fn overlay_groq_host_maps_to_groq_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "llama-3.1-8b-instant".into();
    p.provider = "api.groq.com".into();
    assert_eq!(overlay_model_name(&p), "groq/llama-3.1-8b-instant");
    p.provider = "groq".into();
    assert_eq!(overlay_model_name(&p), "groq/llama-3.1-8b-instant");
}

#[test]
fn overlay_bedrock_host_maps_to_bedrock_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "amazon.nova-lite-v1:0".into();
    p.provider = "amazon-bedrock".into();
    assert_eq!(overlay_model_name(&p), "bedrock/amazon.nova-lite-v1:0");
    p.provider = "bedrock-runtime.us-east-1.amazonaws.com".into();
    assert_eq!(overlay_model_name(&p), "bedrock/amazon.nova-lite-v1:0");
    p.provider = "bedrock".into();
    assert_eq!(overlay_model_name(&p), "bedrock/amazon.nova-lite-v1:0");
}

#[test]
fn overlay_lmstudio_maps_to_lm_studio_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.provider = "lmstudio".into();
    assert_eq!(
        overlay_model_name(&p),
        "lm_studio/qwen2.5-coder",
        "Aider/LiteLLM expect lm_studio/, not lmstudio/"
    );
    p.provider = "lm_studio".into();
    assert_eq!(
        overlay_model_name(&p),
        "lm_studio/qwen2.5-coder",
        "already-normalized lm_studio must stay lm_studio"
    );
}

#[test]
fn overlay_vllm_maps_to_hosted_vllm_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.provider = "vllm".into();
    assert_eq!(
        overlay_model_name(&p),
        "hosted_vllm/qwen2.5-coder",
        "LiteLLM OpenAI-compat expects hosted_vllm/, not vllm/"
    );
    p.provider = "hosted_vllm".into();
    assert_eq!(overlay_model_name(&p), "hosted_vllm/qwen2.5-coder");
}

#[test]
fn overlay_grok_build_stays_off_xai_family() {
    let mut p = sample(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    p.model_id = "grok-4".into();
    p.provider = "grok-build".into();
    assert_eq!(
        overlay_model_name(&p),
        "grok-build/grok-4",
        "grok-build must not map to xai"
    );
    p.provider = "xai-grok-build".into();
    assert_eq!(overlay_model_name(&p), "xai-grok-build/grok-4");
    p.provider = "cli-chat-proxy.grok.com".into();
    assert_eq!(overlay_model_name(&p), "cli-chat-proxy.grok.com/grok-4");
    p.provider = "grok-build-messages".into();
    assert_eq!(overlay_model_name(&p), "grok-build-messages/grok-4");
}
