use canact::{
    CORE_DIMENSION_NAMES, CapabilityLevel, CapabilityProfile, DIMENSION_NAMES,
    EditFormatRecommendation, ProbeResult, REQUIREMENT_DIMENSION_NAMES, classify,
    missing_model_message,
};

fn make_probe(name: &str, level: CapabilityLevel) -> ProbeResult {
    ProbeResult {
        name: name.to_string(),
        score: match level {
            CapabilityLevel::Strong => 1.0,
            CapabilityLevel::Medium => 0.5,
            CapabilityLevel::Weak => 0.1,
        },
        max_score: 1.0,
        level,
        details: "test".to_string(),
    }
}

fn make_profile(
    tool: CapabilityLevel,
    json: CapabilityLevel,
    instr: CapabilityLevel,
) -> CapabilityProfile {
    CapabilityProfile {
        model_id: "test-model".to_string(),
        provider: "test-provider".to_string(),
        tool_calling: make_probe("tool_calling", tool),
        json_output: make_probe("json_output", json),
        instruction_following: make_probe("instruction_following", instr),
        search_replace: make_probe("search_replace", CapabilityLevel::Strong),
        unified_diff: make_probe("unified_diff", CapabilityLevel::Medium),
        xml_tool_calling: make_probe("xml_tool_calling", CapabilityLevel::Medium),
        complex_tool_calling: make_probe("complex_tool_calling", CapabilityLevel::Strong),
        nested_arguments: make_probe("nested_arguments", CapabilityLevel::Strong),
        vision: make_probe("vision", CapabilityLevel::Medium),
        tool_selection: make_probe("tool_selection", CapabilityLevel::Strong),
        streaming_tool_calls: make_probe("streaming_tool_calls", CapabilityLevel::Strong),
        one_shot_tool_plan: make_probe("one_shot_tool_plan", CapabilityLevel::Strong),
        multi_turn_task_sequencing: make_probe(
            "multi_turn_task_sequencing",
            CapabilityLevel::Strong,
        ),
        context_faithfulness: make_probe("context_faithfulness", CapabilityLevel::Strong),
        code_syntax: make_probe("code_syntax", CapabilityLevel::Strong),
        max_tokens_compliance: make_probe("max_tokens_compliance", CapabilityLevel::Strong),
        multi_turn_memory: make_probe("multi_turn_memory", CapabilityLevel::Strong),
        system_message_adherence: make_probe("system_message_adherence", CapabilityLevel::Strong),
        token_efficiency: make_probe("token_efficiency", CapabilityLevel::Strong),
        parallel_tool_scale: make_probe("parallel_tool_scale", CapabilityLevel::Strong),
        probed_at: 1_700_000_000,
        effective_context_tokens: None,
    }
}

#[test]
fn dimension_names_count_matches_probe_fields() {
    assert_eq!(
        DIMENSION_NAMES.len(),
        20,
        "DIMENSION_NAMES should have exactly 20 entries (one per probe field)"
    );
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    for &name in DIMENSION_NAMES {
        assert!(
            profile.dimension_level(name).is_some(),
            "missing dimension_level for {name}"
        );
    }
}

#[test]
fn requirement_dimension_names_are_first_nine() {
    assert_eq!(REQUIREMENT_DIMENSION_NAMES, &DIMENSION_NAMES[..9]);
}

#[test]
fn core_dimension_names_are_not_the_requirement_zip() {
    assert_ne!(CORE_DIMENSION_NAMES, &DIMENSION_NAMES[..9]);
    assert!(CORE_DIMENSION_NAMES.contains(&"xml_tool_calling"));
    assert!(!CORE_DIMENSION_NAMES.contains(&"tool_selection"));
}

#[test]
fn dimension_result_recognises_all_dimension_names() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    for &name in DIMENSION_NAMES {
        let result = profile
            .dimension_result(name)
            .unwrap_or_else(|| panic!("dimension_result returned None for {name:?}"));
        assert_eq!(result.level, profile.dimension_level(name).unwrap());
    }
    assert!(profile.dimension_result("not_a_real_dimension").is_none());
}

#[test]
fn dimension_level_returns_none_for_unknown() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert_eq!(profile.dimension_level("nonexistent_dimension"), None);
}

#[test]
fn classify_thresholds() {
    assert_eq!(classify(1.0), CapabilityLevel::Strong);
    assert_eq!(classify(0.8), CapabilityLevel::Strong);
    assert_eq!(classify(0.79), CapabilityLevel::Medium);
    assert_eq!(classify(0.4), CapabilityLevel::Medium);
    assert_eq!(classify(0.39), CapabilityLevel::Weak);
    assert_eq!(classify(0.0), CapabilityLevel::Weak);
}

#[test]
fn overall_level_returns_minimum() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Weak,
        CapabilityLevel::Medium,
    );
    assert_eq!(profile.overall_level(), CapabilityLevel::Weak);
}

#[test]
fn overall_level_all_strong() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert_eq!(profile.overall_level(), CapabilityLevel::Strong);
}

#[test]
fn needs_xml_fallback_true_for_weak_tool_calling() {
    let profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.needs_xml_fallback());
}

#[test]
fn needs_xml_fallback_false_for_strong_tool_calling() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(!profile.needs_xml_fallback());
}

#[test]
fn needs_json_repair_true_for_medium() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    assert!(profile.needs_json_repair());
}

#[test]
fn needs_json_repair_false_for_strong() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(!profile.needs_json_repair());
}

#[test]
fn can_use_tools_true_when_native_strong() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.can_use_tools());
}

#[test]
fn can_use_tools_true_when_xml_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.xml_tool_calling = make_probe("xml_tool_calling", CapabilityLevel::Medium);
    assert!(profile.can_use_tools());
}

#[test]
fn can_use_tools_false_when_both_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.xml_tool_calling = make_probe("xml_tool_calling", CapabilityLevel::Weak);
    assert!(!profile.can_use_tools());
}

#[test]
fn can_use_tools_false_when_xml_is_transient_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.xml_tool_calling = ProbeResult {
        name: "xml_tool_calling".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    assert!(!profile.can_use_tools());
    assert!(profile.tool_gate_error().is_some());
}

#[test]
fn best_edit_format_search_replace_when_strong() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.search_replace = make_probe("search_replace", CapabilityLevel::Strong);
    assert_eq!(
        profile.best_edit_format(),
        EditFormatRecommendation::SearchReplace
    );
}

#[test]
fn best_edit_format_unified_diff_when_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.search_replace = make_probe("search_replace", CapabilityLevel::Medium);
    profile.unified_diff = make_probe("unified_diff", CapabilityLevel::Medium);
    assert_eq!(
        profile.best_edit_format(),
        EditFormatRecommendation::UnifiedDiff
    );
}

#[test]
fn best_edit_format_whole_file_when_both_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.search_replace = make_probe("search_replace", CapabilityLevel::Weak);
    profile.unified_diff = make_probe("unified_diff", CapabilityLevel::Weak);
    assert_eq!(
        profile.best_edit_format(),
        EditFormatRecommendation::WholeFile
    );
}

#[test]
fn max_tools_none_for_strong() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert_eq!(profile.max_tools(), None);
}

#[test]
fn max_tools_20_for_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.tool_selection = make_probe("tool_selection", CapabilityLevel::Medium);
    assert_eq!(profile.max_tools(), Some(20));
}

#[test]
fn max_tools_10_for_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.tool_selection = make_probe("tool_selection", CapabilityLevel::Weak);
    assert_eq!(profile.max_tools(), Some(10));
}

#[test]
fn max_tools_not_10_for_033_tool_selection() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.tool_selection = ProbeResult {
        name: "tool_selection".to_owned(),
        score: 1.0 / 3.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "1 tool call(s): task1=0.5, task2=0, task3=0.5".to_owned(),
    };
    assert_ne!(
        profile.max_tools(),
        Some(10),
        "0.33 generic-edit path must not cap at 10 tools"
    );
}

#[test]
fn supports_vision_true_for_medium() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.supports_vision());
}

#[test]
fn supports_vision_false_for_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.vision = make_probe("vision", CapabilityLevel::Weak);
    assert!(!profile.supports_vision());
}

#[test]
fn old_cache_without_edit_fields_deserializes() {
    let old_json = r#"{
            "modelId": "test-model",
            "provider": "test-provider",
            "toolCalling": {"name":"tool_calling","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "jsonOutput": {"name":"json_output","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "instructionFollowing": {"name":"instruction_following","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "probedAt": 1700000000
        }"#;
    let profile: CapabilityProfile = serde_json::from_str(old_json).unwrap();
    assert_eq!(profile.search_replace.level, CapabilityLevel::Medium);
    assert_eq!(profile.unified_diff.level, CapabilityLevel::Medium);
}

#[test]
fn one_shot_tool_plan_deserializes_from_legacy_multi_step_key() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Weak,
    );
    profile.one_shot_tool_plan = ProbeResult {
        name: "multi_step_reasoning".to_owned(),
        score: 0.3,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "legacy".to_owned(),
    };
    let mut value = serde_json::to_value(&profile).expect("serialize");
    let obj = value.as_object_mut().expect("object");
    if let Some(plan) = obj.remove("oneShotToolPlan") {
        obj.insert("multiStepReasoning".to_owned(), plan);
    }
    obj.remove("multiTurnTaskSequencing");
    let restored: CapabilityProfile =
        serde_json::from_value(value).expect("legacy multiStepReasoning must deserialize");
    assert_eq!(restored.one_shot_tool_plan.score, 0.3);
    assert_eq!(restored.one_shot_tool_plan.level, CapabilityLevel::Weak);
    assert_eq!(
        restored.multi_turn_task_sequencing.details,
        "Not probed (cached before this probe existed)"
    );
}

#[test]
fn effective_context_tokens_defaults_none() {
    let json = r#"{
            "modelId": "test-model",
            "provider": "test-provider",
            "toolCalling": {"name":"tool_calling","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "jsonOutput": {"name":"json_output","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "instructionFollowing": {"name":"instruction_following","score":1.0,"maxScore":1.0,"level":"strong","details":"test"},
            "probedAt": 1700000000
        }"#;
    let profile: CapabilityProfile = serde_json::from_str(json).unwrap();
    assert_eq!(profile.effective_context_tokens, None);
    let value = serde_json::to_value(&profile).unwrap();
    assert!(value.get("effectiveContextTokens").is_none());
}

#[test]
fn meets_empty_always_passes() {
    let profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Weak,
        CapabilityLevel::Weak,
    );
    assert!(profile.meets(&[]));
}

#[test]
fn meets_fails_on_weak_json() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
    );
    assert!(!profile.meets(&[("json_output", CapabilityLevel::Medium)]));
}

#[test]
fn meets_camel_case_tool_calling_fails_when_weak() {
    let profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(!profile.meets(&[("toolCalling", CapabilityLevel::Strong)]));
}

#[test]
fn meets_camel_case_tool_calling_passes_when_strong() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.meets(&[("toolCalling", CapabilityLevel::Strong)]));
}

#[test]
fn dimension_level_accepts_envelope_camel_case() {
    let profile = make_profile(
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
        CapabilityLevel::Weak,
    );
    assert_eq!(
        profile.dimension_level("toolCalling"),
        Some(CapabilityLevel::Medium)
    );
    assert_eq!(
        profile.dimension_level("jsonOutput"),
        Some(CapabilityLevel::Strong)
    );
    assert_eq!(
        profile
            .dimension_result("instructionFollowing")
            .map(|p| p.level),
        Some(CapabilityLevel::Weak)
    );
    for (snake, camel) in [
        ("xml_tool_calling", "xmlToolCalling"),
        ("one_shot_tool_plan", "oneShotToolPlan"),
        ("multi_turn_task_sequencing", "multiTurnTaskSequencing"),
    ] {
        assert_eq!(
            profile.dimension_level(snake),
            profile.dimension_level(camel),
            "{snake} vs {camel}"
        );
    }
}

#[test]
fn meets_unknown_camel_name_still_skips() {
    let profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Weak,
        CapabilityLevel::Weak,
    );
    assert!(profile.meets(&[("notADimension", CapabilityLevel::Strong)]));
}

#[test]
fn capability_level_default_is_weak() {
    assert_eq!(CapabilityLevel::default(), CapabilityLevel::Weak);
}

#[test]
fn host_policy_envelope_omits_bline_best_edit_format() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    let value = profile.host_policy_envelope();
    assert!(value.get("bestEditFormat").is_none(), "{value}");
    assert_eq!(value["model"], "test-model");
    assert_eq!(value["provider"], "test-provider");
    assert_eq!(value["probeLadderEditFormat"], "search_replace");
    assert_eq!(value["canUseTools"], true);
    assert_eq!(value["needsJsonRepair"], true);
    assert_eq!(value["needsXmlFallback"], false);
    assert!(value["probes"]["toolCalling"].is_object());
    assert_eq!(value["scoreScale"]["strongMin"], 0.8);
    assert_eq!(value["scoreScale"]["mediumMin"], 0.4);
}

#[test]
fn host_policy_envelope_includes_effective_context_tokens() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    profile.effective_context_tokens = Some(8192);
    let value = profile.host_policy_envelope();
    assert!(value.get("effectiveContextTokens").is_some(), "{value}");
    assert!(value["effectiveContextTokens"].is_number(), "{value}");
    assert_eq!(value["effectiveContextTokens"], 8192);
}

#[test]
fn host_policy_envelope_null_effective_context_when_unset() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    assert!(profile.effective_context_tokens.is_none());
    let value = profile.host_policy_envelope();
    assert!(value["effectiveContextTokens"].is_null(), "{value}");
}

#[test]
fn tool_gate_error_when_both_weak_explains_exit_2() {
    let mut profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.xml_tool_calling = make_probe("xml_tool_calling", CapabilityLevel::Weak);
    assert!(!profile.can_use_tools());
    let msg = profile
        .tool_gate_error()
        .expect("Weak native+XML must fail the tool gate");
    assert!(msg.contains("cannot use tools"), "{msg}");
}

#[test]
fn tool_gate_error_none_when_tools_usable() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.tool_gate_error().is_none());
}

#[test]
fn human_table_includes_effective_context_tokens_when_some() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.effective_context_tokens = Some(8192);
    let table = profile.format_human_table(false);
    assert!(table.contains("8192"), "{table}");
    assert!(table.contains("Effective context tokens:"), "{table}");
}

#[test]
fn human_table_omits_effective_context_tokens_when_none() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    let table = profile.format_human_table(false);
    assert!(
        !table.to_ascii_lowercase().contains("effective context"),
        "{table}"
    );
}

#[test]
fn missing_model_message_zero_ids_includes_count() {
    let empty: [&str; 0] = [];
    let msg = missing_model_message(&empty);
    assert!(msg.contains("got 0"), "{msg}");
    assert!(msg.contains("--model"), "{msg}");
}

#[test]
fn missing_model_message_previews_at_most_eight_ids() {
    let ids: Vec<String> = (1..=12).map(|i| format!("model-{i}")).collect();
    let msg = missing_model_message(&ids);
    assert!(msg.contains("got 12"), "{msg}");
    assert!(msg.contains("model-1"), "{msg}");
    assert!(msg.contains("model-8"), "{msg}");
    assert!(!msg.contains("model-9"), "{msg}");
    assert!(!msg.contains("model-12"), "{msg}");
    assert!(msg.contains("..."), "{msg}");
}
