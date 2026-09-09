use canact::{CapabilityLevel, CapabilityProfile, ProbeResult};

fn probe(name: &str, level: CapabilityLevel) -> ProbeResult {
    ProbeResult {
        name: name.to_owned(),
        score: match level {
            CapabilityLevel::Strong => 1.0,
            CapabilityLevel::Medium => 0.5,
            CapabilityLevel::Weak => 0.1,
        },
        max_score: 1.0,
        level,
        details: "example".to_owned(),
    }
}

fn sample_profile() -> CapabilityProfile {
    let mut p = CapabilityProfile::unprobed("qwen2.5-coder", "ollama");
    p.tool_calling = probe("tool_calling", CapabilityLevel::Strong);
    p.json_output = probe("json_output", CapabilityLevel::Strong);
    p.instruction_following = probe("instruction_following", CapabilityLevel::Strong);
    p.search_replace = probe("search_replace", CapabilityLevel::Strong);
    p.unified_diff = probe("unified_diff", CapabilityLevel::Medium);
    p.xml_tool_calling = probe("xml_tool_calling", CapabilityLevel::Medium);
    p.complex_tool_calling = probe("complex_tool_calling", CapabilityLevel::Strong);
    p.nested_arguments = probe("nested_arguments", CapabilityLevel::Strong);
    p.vision = probe("vision", CapabilityLevel::Weak);
    p.tool_selection = probe("tool_selection", CapabilityLevel::Medium);
    p.streaming_tool_calls = probe("streaming_tool_calls", CapabilityLevel::Strong);
    p.one_shot_tool_plan = probe("one_shot_tool_plan", CapabilityLevel::Strong);
    p.multi_turn_task_sequencing = probe("multi_turn_task_sequencing", CapabilityLevel::Strong);
    p.context_faithfulness = probe("context_faithfulness", CapabilityLevel::Strong);
    p.code_syntax = probe("code_syntax", CapabilityLevel::Strong);
    p.max_tokens_compliance = probe("max_tokens_compliance", CapabilityLevel::Strong);
    p.multi_turn_memory = probe("multi_turn_memory", CapabilityLevel::Strong);
    p.system_message_adherence = probe("system_message_adherence", CapabilityLevel::Strong);
    p.token_efficiency = probe("token_efficiency", CapabilityLevel::Strong);
    p.parallel_tool_scale = probe("parallel_tool_scale", CapabilityLevel::Strong);
    p.probed_at = 1_700_000_000;
    p.effective_context_tokens = Some(8192);
    p.probed_context_floor = Some(8192);
    p
}
