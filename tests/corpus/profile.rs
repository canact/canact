use canact::{
    CORE_DIMENSION_NAMES, CapabilityLevel, CapabilityProfile, DIAGNOSTIC_DIMENSION_NAMES,
    DIMENSION_NAMES, EditFormatRecommendation, HostPolicyMeta, POLICY_DIMENSION_NAMES, ProbeResult,
    REQUIREMENT_DIMENSION_NAMES, classify, missing_model_message,
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
    let mut p = CapabilityProfile::unprobed("test-model", "test-provider");
    p.tool_calling = make_probe("tool_calling", tool);
    p.json_output = make_probe("json_output", json);
    p.instruction_following = make_probe("instruction_following", instr);
    p.search_replace = make_probe("search_replace", CapabilityLevel::Strong);
    p.unified_diff = make_probe("unified_diff", CapabilityLevel::Medium);
    p.xml_tool_calling = make_probe("xml_tool_calling", CapabilityLevel::Medium);
    p.complex_tool_calling = make_probe("complex_tool_calling", CapabilityLevel::Strong);
    p.nested_arguments = make_probe("nested_arguments", CapabilityLevel::Strong);
    p.vision = make_probe("vision", CapabilityLevel::Medium);
    p.tool_selection = make_probe("tool_selection", CapabilityLevel::Strong);
    p.streaming_tool_calls = make_probe("streaming_tool_calls", CapabilityLevel::Strong);
    p.one_shot_tool_plan = make_probe("one_shot_tool_plan", CapabilityLevel::Strong);
    p.multi_turn_task_sequencing =
        make_probe("multi_turn_task_sequencing", CapabilityLevel::Strong);
    p.context_faithfulness = make_probe("context_faithfulness", CapabilityLevel::Strong);
    p.code_syntax = make_probe("code_syntax", CapabilityLevel::Strong);
    p.max_tokens_compliance = make_probe("max_tokens_compliance", CapabilityLevel::Strong);
    p.multi_turn_memory = make_probe("multi_turn_memory", CapabilityLevel::Strong);
    p.system_message_adherence = make_probe("system_message_adherence", CapabilityLevel::Strong);
    p.token_efficiency = make_probe("token_efficiency", CapabilityLevel::Strong);
    p.parallel_tool_scale = make_probe("parallel_tool_scale", CapabilityLevel::Strong);
    p.probed_at = 1_700_000_000;
    p
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
fn dimension_names_are_policy_diagnostic_or_explicit_omit() {
    // one_shot_tool_plan is serde-only; do not derive policy from DIMENSION_NAMES.
    const OMIT_FROM_HOST_ENVELOPE: &[&str] = &["one_shot_tool_plan"];

    fn disjoint(left: &[&str], right: &[&str]) {
        for &name in left {
            assert!(
                !right.contains(&name),
                "{name} must not appear in two envelope sets"
            );
        }
    }
    disjoint(POLICY_DIMENSION_NAMES, DIAGNOSTIC_DIMENSION_NAMES);
    disjoint(POLICY_DIMENSION_NAMES, OMIT_FROM_HOST_ENVELOPE);
    disjoint(DIAGNOSTIC_DIMENSION_NAMES, OMIT_FROM_HOST_ENVELOPE);

    for &name in DIMENSION_NAMES {
        let covered = POLICY_DIMENSION_NAMES.contains(&name)
            || DIAGNOSTIC_DIMENSION_NAMES.contains(&name)
            || OMIT_FROM_HOST_ENVELOPE.contains(&name);
        assert!(
            covered,
            "{name} must be in POLICY_DIMENSION_NAMES, DIAGNOSTIC_DIMENSION_NAMES, or the explicit omit set"
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
fn transient_json_does_not_collapse_overall_or_force_repair() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.json_output = ProbeResult {
        name: "json_output".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: transient error: Upstream error from Nvidia: \
                  Service temporarily overloaded"
            .to_string(),
    };
    assert_eq!(profile.overall_level(), CapabilityLevel::Strong);
    assert!(!profile.needs_json_repair());
    assert!(profile.can_use_tools());
}

#[test]
fn skipped_json_does_not_collapse_overall_or_force_repair() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.json_output = ProbeResult {
        name: "json_output".to_string(),
        score: 0.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "Skipped: expensive probe".to_string(),
    };
    assert_eq!(profile.overall_level(), CapabilityLevel::Strong);
    assert!(!profile.needs_json_repair());
}

#[test]
fn all_core_transients_overall_is_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Medium,
        CapabilityLevel::Medium,
        CapabilityLevel::Medium,
    );
    for probe in [
        &mut profile.tool_calling,
        &mut profile.json_output,
        &mut profile.instruction_following,
    ] {
        probe.details = "Probe failed: timeout".to_string();
    }
    assert_eq!(profile.overall_level(), CapabilityLevel::Weak);
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
fn synthesized_medium_does_not_open_host_policy() {
    let mut profile = make_profile(
        CapabilityLevel::Weak,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.vision = ProbeResult {
        name: "vision".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    profile.unified_diff = ProbeResult {
        name: "unified_diff".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    profile.tool_calling = ProbeResult {
        name: "tool_calling".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    profile.search_replace = make_probe("search_replace", CapabilityLevel::Weak);
    profile.tool_selection = ProbeResult {
        name: "tool_selection".to_string(),
        score: 0.9,
        max_score: 1.0,
        level: CapabilityLevel::Strong,
        details: "Probe failed: timeout".to_string(),
    };
    profile.json_output = ProbeResult {
        name: "json_output".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: 429".to_string(),
    };
    assert!(!profile.supports_vision());
    assert!(profile.needs_xml_fallback());
    assert!(
        !profile.needs_json_repair(),
        "transient JSON must not turn repair on"
    );
    assert_eq!(
        profile.overall_level(),
        CapabilityLevel::Strong,
        "transient cores must not collapse a completed Strong instruction card"
    );
    assert_eq!(profile.max_tools(), Some(10));
    assert_eq!(
        profile.best_edit_format(),
        EditFormatRecommendation::WholeFile
    );
    assert!(!profile.meets(&[("vision", CapabilityLevel::Medium)]));
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
fn max_tools_20_for_033_medium_tool_selection() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.tool_selection = ProbeResult {
        name: "tool_selection".to_owned(),
        score: 1.0 / 3.0,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "1 tool call(s): task1=0.5, task2=0, task3=0.5".to_owned(),
    };
    assert_eq!(
        profile.max_tools(),
        Some(20),
        "generic-edit 0.33 Medium must cap at 20, not unlimited"
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
fn unprobed_default_fields_do_not_open_host_policy() {
    let old_json = r#"{
            "modelId": "m",
            "provider": "p",
            "toolCalling": {"name":"tool_calling","score":0.0,"maxScore":1.0,"level":"weak","details":"No tool call"},
            "jsonOutput": {"name":"json_output","score":1.0,"maxScore":1.0,"level":"strong","details":"ok"},
            "instructionFollowing": {"name":"instruction_following","score":1.0,"maxScore":1.0,"level":"strong","details":"ok"},
            "probedAt": 1
        }"#;
    let profile: CapabilityProfile = serde_json::from_str(old_json).unwrap();
    assert!(
        !profile.can_use_tools(),
        "missing xml_tool_calling must not open tools: envelope={}",
        profile.host_policy_envelope()
    );
    assert!(
        !profile.supports_vision(),
        "missing vision must not set supportsVision"
    );
    assert_eq!(
        profile.max_tools(),
        Some(10),
        "missing tool_selection must not default to Medium 20"
    );
    assert_eq!(
        profile.best_edit_format(),
        EditFormatRecommendation::WholeFile,
        "missing edit probes must not recommend unified_diff"
    );
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
    assert_eq!(value["useStreamingForToolCalls"], true);
    assert_eq!(value["supportsNestedToolArgs"], true);
    assert_eq!(value["verifiedParallelToolCalls"], 5);
    assert_eq!(value["agentLoop"], "full");
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
    assert!(table.contains("Recommended context tokens:"), "{table}");
}

#[test]
fn human_table_prints_probed_context_floor_when_effective_unset() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.probed_context_floor = Some(4096);
    let table = profile.format_human_table(false);
    assert!(table.contains("4096"), "{table}");
    assert!(table.contains("Probed context floor:"), "{table}");
    assert!(table.contains("Recommended context tokens:"), "{table}");
    assert!(
        !table.to_ascii_lowercase().contains("effective context"),
        "{table}"
    );
}

#[test]
fn human_table_prints_recommended_not_advertised_as_measured() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.probed_context_floor = Some(4096);
    let table = profile.format_human_table_with(false, Some(40960));
    assert!(table.contains("Recommended context tokens:"), "{table}");
    assert!(table.contains("4096"), "{table}");
    assert!(
        !table.contains("40960"),
        "advertised must not print as measured: {table}"
    );
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
fn human_table_prints_max_output_tokens_when_some() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.max_output_tokens = Some(4096);
    let table = profile.format_human_table(false);
    assert!(table.contains("Max output tokens:"), "{table}");
    assert!(table.contains("4096"), "{table}");
}

#[test]
fn human_table_omits_max_output_tokens_when_none() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert!(profile.max_output_tokens.is_none());
    let table = profile.format_human_table(false);
    assert!(
        !table.to_ascii_lowercase().contains("max output tokens"),
        "{table}"
    );
}

#[test]
fn human_table_prints_constraint_placement_when_measured() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    let table = profile.format_human_table(false);
    assert!(table.contains("Constraint placement:"), "{table}");
    assert!(table.contains("system"), "{table}");

    profile.system_message_adherence =
        make_probe("system_message_adherence", CapabilityLevel::Weak);
    let table = profile.format_human_table(false);
    assert!(table.contains("Constraint placement:"), "{table}");
    assert!(table.contains("user"), "{table}");
}

#[test]
fn human_table_omits_constraint_placement_when_unprobed_or_skipped() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Not probed (cached before this probe existed)".to_string(),
    };
    let table = profile.format_human_table(false);
    assert!(
        !table.to_ascii_lowercase().contains("constraint placement"),
        "{table}"
    );

    profile.system_message_adherence = ProbeResult {
        name: "system_message_adherence".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Skipped: diagnostic suite (use --suite=all)".to_string(),
    };
    let table = profile.format_human_table(false);
    assert!(
        !table.to_ascii_lowercase().contains("constraint placement"),
        "{table}"
    );
}

const EXPENSIVE_SKIP: &str = "Skipped: free-tier model, conserving API budget";
const XML_INFERRED: &str = "Not tested (native tool calling is Strong; XML fallback unused)";

fn cheap_skip_probe(name: &str) -> ProbeResult {
    ProbeResult {
        name: name.to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: EXPENSIVE_SKIP.to_string(),
    }
}

#[test]
fn human_table_prints_completed_weak_for_skipped_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.search_replace = cheap_skip_probe("search_replace");
    let table = profile.format_human_table(true);
    let search_line = table
        .lines()
        .find(|l| l.contains("Search Replace"))
        .unwrap_or("");
    assert!(
        search_line.contains("Weak"),
        "skipped Medium must display as Weak:\n{table}"
    );
    assert!(
        !search_line.contains("Medium"),
        "must not show stored Medium for a skip:\n{table}"
    );
}

#[test]
fn is_skipped_matches_skipped_prefix_only() {
    let skip = cheap_skip_probe("one_shot_tool_plan");
    assert!(skip.is_skipped());
    let xml = ProbeResult {
        name: "xml_tool_calling".to_string(),
        score: 1.0,
        max_score: 1.0,
        level: CapabilityLevel::Strong,
        details: XML_INFERRED.to_string(),
    };
    assert!(!xml.is_skipped());
    let failed = ProbeResult {
        name: "tool_calling".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    assert!(!failed.is_skipped());
}

#[test]
fn cheap_skip_is_not_measured_medium() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.one_shot_tool_plan = cheap_skip_probe("one_shot_tool_plan");
    assert_eq!(profile.one_shot_tool_plan.level, CapabilityLevel::Medium);
    assert_eq!(profile.one_shot_tool_plan.score, 0.5);
    assert_eq!(profile.one_shot_tool_plan.details, EXPENSIVE_SKIP);
    assert!(
        !profile.meets(&[("one_shot_tool_plan", CapabilityLevel::Medium)]),
        "cheap skip must not satisfy a Medium requirement"
    );
    assert_eq!(
        profile.dimension_level("one_shot_tool_plan"),
        Some(CapabilityLevel::Weak)
    );
    let value = profile.host_policy_envelope();
    assert!(
        value["probes"].get("oneShotToolPlan").is_none(),
        "one_shot must not appear in policy probes: {value}"
    );
}

#[test]
fn xml_inferred_strong_is_not_skipped() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.xml_tool_calling = ProbeResult {
        name: "xml_tool_calling".to_string(),
        score: 1.0,
        max_score: 1.0,
        level: CapabilityLevel::Strong,
        details: XML_INFERRED.to_string(),
    };
    assert_eq!(
        profile.dimension_level("xml_tool_calling"),
        Some(CapabilityLevel::Strong)
    );
    let value = profile.host_policy_envelope();
    assert_eq!(value["probes"]["xmlToolCalling"]["status"], "completed");
}

#[test]
fn unprobed_default_envelope_status_is_unprobed() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.vision = ProbeResult {
        name: "vision".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Not probed (cached before this probe existed)".to_string(),
    };
    let value = profile.host_policy_envelope();
    assert_eq!(value["probes"]["vision"]["status"], "unprobed");
}

#[test]
fn synthesized_error_envelope_status_is_error() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.tool_calling = ProbeResult {
        name: "tool_calling".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: timeout".to_string(),
    };
    let value = profile.host_policy_envelope();
    assert_eq!(value["probes"]["toolCalling"]["status"], "error");
}

#[test]
fn host_policy_envelope_with_emits_session_flags() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    profile.probed_context_floor = Some(4096);
    let value = profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
        false,
        false,
        canact::SuiteTier::Policy,
        Some(40960),
    ));
    assert_eq!(value["cacheable"], false, "{value}");
    assert_eq!(value["fromCache"], false, "{value}");
    assert_eq!(value["skipExpensive"], true, "{value}");
    assert_eq!(value["advertisedContextTokens"], 40960, "{value}");
    assert_eq!(value["probedContextFloor"], 4096, "{value}");
    assert_eq!(value["recommendedContextTokens"], 4096, "{value}");
    assert!(value["effectiveContextTokens"].is_null(), "{value}");
}

#[test]
fn host_policy_envelope_default_meta_is_cacheable_full() {
    let profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Medium,
        CapabilityLevel::Strong,
    );
    let value = profile.host_policy_envelope();
    assert_eq!(value["cacheable"], true, "{value}");
    assert_eq!(value["fromCache"], false, "{value}");
    assert_eq!(value["skipExpensive"], false, "{value}");
    assert_eq!(value["suite"], "full", "{value}");
    assert!(
        value["diagnostics"].as_object().unwrap().is_empty(),
        "{value}"
    );
    assert!(value["advertisedContextTokens"].is_null(), "{value}");
    assert!(value["probedContextFloor"].is_null(), "{value}");
    assert!(value["recommendedContextTokens"].is_null(), "{value}");
}

#[test]
fn human_table_vs_json_constraint_placement_on_policy() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert_eq!(
        profile.system_message_adherence.level,
        CapabilityLevel::Strong
    );
    assert!(profile.max_output_tokens.is_none());

    let table = profile.format_human_table(false);
    assert!(table.contains("Constraint placement"), "{table}");
    assert!(table.contains("system"), "{table}");
    assert!(
        !table.to_ascii_lowercase().contains("max output tokens"),
        "unmeasured cap omitted in the human table: {table}"
    );

    let policy = profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
        true,
        false,
        canact::SuiteTier::Policy,
        None,
    ));
    assert!(
        policy.get("constraintPlacement").is_none(),
        "policy --json omits placement: {policy}"
    );
    assert!(
        policy.get("maxOutputTokens").is_none(),
        "unmeasured cap omitted in --json: {policy}"
    );

    profile.max_output_tokens = Some(4096);
    let table = profile.format_human_table(false);
    assert!(table.contains("Constraint placement"), "{table}");
    assert!(table.contains("system"), "{table}");
    assert!(table.contains("Max output tokens:"), "{table}");
    assert!(table.contains("4096"), "{table}");

    let policy = profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
        true,
        false,
        canact::SuiteTier::Policy,
        None,
    ));
    assert!(
        policy.get("constraintPlacement").is_none(),
        "policy --json still omits placement: {policy}"
    );
    assert_eq!(policy["maxOutputTokens"], 4096, "{policy}");
}

#[test]
fn constraint_placement_weak_is_user_only_on_all() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.system_message_adherence =
        make_probe("system_message_adherence", CapabilityLevel::Weak);
    let all = profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
        true,
        false,
        canact::SuiteTier::All,
        None,
    ));
    assert_eq!(all["constraintPlacement"], "user", "{all}");
    assert!(
        all["constraintPlacement"].as_str().is_some(),
        "placement is an action, not a score: {all}"
    );
    let policy = profile.host_policy_envelope_with(HostPolicyMeta::for_suite(
        true,
        false,
        canact::SuiteTier::Policy,
        None,
    ));
    assert!(
        policy.get("constraintPlacement").is_none(),
        "policy isolation: {policy}"
    );
}

#[test]
fn plumbing_fields_are_independent_of_diagnostic_weak() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    let before = profile.host_policy_envelope();
    profile.one_shot_tool_plan = make_probe("one_shot_tool_plan", CapabilityLevel::Weak);
    profile.token_efficiency = make_probe("token_efficiency", CapabilityLevel::Weak);
    profile.code_syntax = make_probe("code_syntax", CapabilityLevel::Weak);
    let after = profile.host_policy_envelope();
    assert_eq!(
        after["useStreamingForToolCalls"],
        before["useStreamingForToolCalls"]
    );
    assert_eq!(
        after["supportsNestedToolArgs"],
        before["supportsNestedToolArgs"]
    );
    assert_eq!(
        after["verifiedParallelToolCalls"],
        before["verifiedParallelToolCalls"]
    );
    assert_eq!(after["agentLoop"], before["agentLoop"]);
    assert!(profile.meets(&[("nested_arguments", CapabilityLevel::Medium)]));
    assert!(profile.meets(&[("complex_tool_calling", CapabilityLevel::Medium)]));
}

#[test]
fn parallel_field_is_a_floor_not_a_max() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    profile.parallel_tool_scale.score = 0.6;
    profile.parallel_tool_scale.level = CapabilityLevel::Medium;
    assert_eq!(profile.verified_parallel_tool_calls(), Some(3));
    profile.parallel_tool_scale.details = "Skipped: --cheap".into();
    assert_eq!(profile.verified_parallel_tool_calls(), None);
}

#[test]
fn agent_loop_maps_completed_levels() {
    let mut profile = make_profile(
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
        CapabilityLevel::Strong,
    );
    assert_eq!(profile.agent_loop(), Some(canact::AgentLoop::Full));
    profile.multi_turn_task_sequencing =
        make_probe("multi_turn_task_sequencing", CapabilityLevel::Medium);
    assert_eq!(profile.agent_loop(), Some(canact::AgentLoop::Assisted));
    profile.multi_turn_task_sequencing =
        make_probe("multi_turn_task_sequencing", CapabilityLevel::Weak);
    assert_eq!(profile.agent_loop(), Some(canact::AgentLoop::Single));
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
