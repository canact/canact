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
    CapabilityProfile {
        model_id: "qwen2.5-coder".to_owned(),
        provider: "ollama".to_owned(),
        tool_calling: probe("tool_calling", CapabilityLevel::Strong),
        json_output: probe("json_output", CapabilityLevel::Strong),
        instruction_following: probe("instruction_following", CapabilityLevel::Strong),
        search_replace: probe("search_replace", CapabilityLevel::Strong),
        unified_diff: probe("unified_diff", CapabilityLevel::Medium),
        xml_tool_calling: probe("xml_tool_calling", CapabilityLevel::Medium),
        complex_tool_calling: probe("complex_tool_calling", CapabilityLevel::Strong),
        nested_arguments: probe("nested_arguments", CapabilityLevel::Strong),
        vision: probe("vision", CapabilityLevel::Weak),
        tool_selection: probe("tool_selection", CapabilityLevel::Medium),
        streaming_tool_calls: probe("streaming_tool_calls", CapabilityLevel::Strong),
        one_shot_tool_plan: probe("one_shot_tool_plan", CapabilityLevel::Strong),
        multi_turn_task_sequencing: probe("multi_turn_task_sequencing", CapabilityLevel::Strong),
        context_faithfulness: probe("context_faithfulness", CapabilityLevel::Strong),
        code_syntax: probe("code_syntax", CapabilityLevel::Strong),
        max_tokens_compliance: probe("max_tokens_compliance", CapabilityLevel::Strong),
        multi_turn_memory: probe("multi_turn_memory", CapabilityLevel::Strong),
        system_message_adherence: probe("system_message_adherence", CapabilityLevel::Strong),
        token_efficiency: probe("token_efficiency", CapabilityLevel::Strong),
        parallel_tool_scale: probe("parallel_tool_scale", CapabilityLevel::Strong),
        probed_at: 1_700_000_000,
        effective_context_tokens: Some(8192),
        probed_context_floor: Some(8192),
        max_output_tokens: None,
    }
}
