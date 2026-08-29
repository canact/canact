use canact::{
    CACHE_TTL_SECS, CacheEntry, CapabilityLevel, CapabilityProfile, DEFAULT_PROBE_EFFORT,
    DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION, PROBE_SUITE_VERSION, ProbeCache, ProbeResult, classify,
};

fn sample_profile() -> CapabilityProfile {
    let pr = |name: &str| ProbeResult {
        name: name.to_owned(),
        score: 0.9,
        max_score: 1.0,
        level: classify(0.9),
        details: "test".to_owned(),
    };
    CapabilityProfile {
        model_id: "m".to_owned(),
        provider: "p".to_owned(),
        tool_calling: pr("tool_calling"),
        json_output: pr("json_output"),
        instruction_following: pr("instruction_following"),
        search_replace: pr("search_replace"),
        unified_diff: pr("unified_diff"),
        complex_tool_calling: pr("complex_tool_calling"),
        nested_arguments: pr("nested_arguments"),
        vision: pr("vision"),
        tool_selection: pr("tool_selection"),
        xml_tool_calling: pr("xml_tool_calling"),
        streaming_tool_calls: pr("streaming_tool_calls"),
        one_shot_tool_plan: pr("one_shot_tool_plan"),
        multi_turn_task_sequencing: pr("multi_turn_task_sequencing"),
        context_faithfulness: pr("context_faithfulness"),
        code_syntax: pr("code_syntax"),
        max_tokens_compliance: pr("max_tokens_compliance"),
        multi_turn_memory: pr("multi_turn_memory"),
        system_message_adherence: pr("system_message_adherence"),
        token_efficiency: pr("token_efficiency"),
        parallel_tool_scale: pr("parallel_tool_scale"),
        probed_at: 1,
        effective_context_tokens: None,
    }
}

#[test]
fn cache_key_includes_effort_and_suite() {
    let k = ProbeCache::cache_key("gpt", "openai", "unset", 2);
    assert_eq!(k, "gpt|openai|unset|v2|full|novision");
}

#[test]
fn cache_key_format_is_model_provider_unset_v8() {
    assert_eq!(PROBE_SUITE_VERSION, 8);
    assert_eq!(CACHE_TTL_SECS, 30 * 24 * 60 * 60);
    let k = ProbeCache::cache_key(
        "model",
        "provider",
        DEFAULT_PROBE_EFFORT,
        PROBE_SUITE_VERSION,
    );
    assert_eq!(k, "model|provider|unset|v8|full|novision");
}

#[test]
fn cache_key_includes_cheap_and_vision_knobs() {
    let cheap = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, true, false);
    let full = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, false);
    let vision = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, true);
    assert_ne!(cheap, full);
    assert_ne!(full, vision);
    assert_eq!(cheap, "m|p|unset|v7|cheap|novision");
    assert_eq!(full, "m|p|unset|v7|full|novision");
    assert_eq!(vision, "m|p|unset|v7|full|vision");
}

#[test]
fn different_effort_is_different_key() {
    let a = ProbeCache::cache_key("m", "p", "unset", 2);
    let b = ProbeCache::cache_key("m", "p", "xhigh", 2);
    assert_ne!(a, b);
}

#[test]
fn put_get_round_trip() {
    let mut cache = ProbeCache::default();
    cache.put(sample_profile());
    let profile = cache
        .get("m", "p")
        .expect("profile under current suite/effort");
    assert_eq!(profile.model_id, "m");
    let entry = cache.get_entry("m", "p").expect("entry metadata");
    assert_eq!(entry.reasoning_effort, "unset");
    assert_eq!(entry.probe_suite_version, PROBE_SUITE_VERSION);
}

#[test]
fn get_misses_when_effort_differs() {
    let mut cache = ProbeCache::default();
    cache.put_with_settings(
        sample_profile(),
        "unset",
        PROBE_SUITE_VERSION,
        DEFAULT_SKIP_EXPENSIVE,
        DEFAULT_VISION,
    );
    assert!(
        cache
            .get_with_settings(
                "m",
                "p",
                "xhigh",
                PROBE_SUITE_VERSION,
                DEFAULT_SKIP_EXPENSIVE,
                DEFAULT_VISION,
            )
            .is_none(),
        "xhigh must not hit unset cache entry"
    );
    assert!(
        cache
            .get_with_settings(
                "m",
                "p",
                "unset",
                PROBE_SUITE_VERSION,
                DEFAULT_SKIP_EXPENSIVE,
                DEFAULT_VISION,
            )
            .is_some()
    );
}

#[test]
fn get_misses_when_suite_differs() {
    let mut cache = ProbeCache::default();
    cache.put_with_settings(
        sample_profile(),
        "unset",
        6,
        DEFAULT_SKIP_EXPENSIVE,
        DEFAULT_VISION,
    );
    assert!(
        cache
            .get_with_settings(
                "m",
                "p",
                "unset",
                PROBE_SUITE_VERSION,
                DEFAULT_SKIP_EXPENSIVE,
                DEFAULT_VISION,
            )
            .is_none(),
        "suite v8 must not hit v6 cache entry"
    );
}

#[test]
fn cheap_cache_misses_on_full_and_vision() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample_profile(), true, false);
    assert!(
        cache.get_with_knobs("m", "p", true, false).is_some(),
        "cheap/novision must hit its own entry"
    );
    assert!(
        cache.get_with_knobs("m", "p", false, false).is_none(),
        "full must not return a cheap-cached profile"
    );
    assert!(
        cache.get_with_knobs("m", "p", true, true).is_none(),
        "vision must not return a no-vision cheap-cached profile"
    );
    assert!(
        cache.get("m", "p").is_none(),
        "default get is full/novision and must miss cheap"
    );
}

#[test]
fn get_entry_returns_metadata() {
    let mut cache = ProbeCache::default();
    cache.put(sample_profile());
    let entry = cache.get_entry("m", "p").expect("entry");
    assert_eq!(entry.reasoning_effort, DEFAULT_PROBE_EFFORT);
    assert_eq!(entry.probe_suite_version, PROBE_SUITE_VERSION);
    assert_eq!(entry.profile.model_id, "m");
    assert_eq!(entry.profile.provider, "p");
    assert!(entry.cached_at > 0);
}

#[test]
fn get_misses_when_ttl_expired() {
    let mut cache = ProbeCache::default();
    let key = ProbeCache::cache_key("m", "p", DEFAULT_PROBE_EFFORT, PROBE_SUITE_VERSION);
    cache.profiles.insert(
        key,
        CacheEntry {
            profile: sample_profile(),
            cached_at: 0,
            reasoning_effort: DEFAULT_PROBE_EFFORT.to_owned(),
            probe_suite_version: PROBE_SUITE_VERSION,
        },
    );
    assert!(cache.get("m", "p").is_none(), "expired TTL must miss");
    assert!(cache.get_entry("m", "p").is_none());
}

#[test]
fn put_then_load_round_trip() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    cache.put(sample_profile());
    cache.save(&path).expect("save");
    let loaded = ProbeCache::load(&path).expect("load");
    let profile = loaded.get("m", "p").expect("hit after reload");
    assert_eq!(profile.model_id, "m");
    assert_eq!(profile.provider, "p");
    assert_eq!(profile.effective_context_tokens, None);
}

#[test]
fn put_then_load_preserves_effective_context_tokens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let mut profile = sample_profile();
    profile.effective_context_tokens = Some(8192);
    cache.put(profile);
    cache.save(&path).expect("save");
    let loaded = ProbeCache::load(&path).expect("load");
    let got = loaded.get("m", "p").expect("hit after reload");
    assert_eq!(got.effective_context_tokens, Some(8192));
}

#[test]
fn load_migrates_stale_no_tools_v7_to_weak_and_persists() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut profile = sample_profile();
    let stale = |name: &str| ProbeResult {
        name: name.to_owned(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: LLM error: does not support tools".to_owned(),
    };
    profile.tool_calling = stale("tool_calling");
    profile.one_shot_tool_plan = stale("one_shot_tool_plan");
    profile.multi_turn_task_sequencing = stale("multi_turn_task_sequencing");
    profile.json_output = stale("json_output");
    let mut cache = ProbeCache::default();
    cache.put(profile);
    cache.save(&path).expect("save stale v7");

    let loaded = ProbeCache::load(&path).expect("load migrates");
    let got = loaded.get("m", "p").expect("v7 hit after load");
    assert_eq!(got.tool_calling.level, CapabilityLevel::Weak);
    assert_eq!(got.tool_calling.score, 0.0);
    assert_eq!(got.one_shot_tool_plan.level, CapabilityLevel::Weak);
    assert_eq!(got.one_shot_tool_plan.score, 0.0);
    assert_eq!(got.multi_turn_task_sequencing.level, CapabilityLevel::Weak);
    assert_eq!(got.multi_turn_task_sequencing.score, 0.0);
    assert_eq!(got.json_output.level, CapabilityLevel::Medium);
    assert_eq!(got.json_output.score, 0.5);

    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("reread")).expect("json");
    let key = ProbeCache::cache_key("m", "p", DEFAULT_PROBE_EFFORT, PROBE_SUITE_VERSION);
    let stored = &raw["profiles"][&key]["profile"];
    assert_eq!(stored["toolCalling"]["level"], "weak");
    assert_eq!(stored["toolCalling"]["score"].as_f64(), Some(0.0));
    assert_eq!(stored["oneShotToolPlan"]["level"], "weak");
    assert_eq!(stored["multiTurnTaskSequencing"]["level"], "weak");
    assert_eq!(stored["jsonOutput"]["level"], "medium");
}

#[test]
fn load_missing_file_is_empty() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("missing.json");
    let loaded = ProbeCache::load(&path).expect("missing file");
    assert!(loaded.profiles.is_empty());
}

#[test]
fn load_keeps_migrated_profile_when_save_fails() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut profile = sample_profile();
    profile.tool_calling = ProbeResult {
        name: "tool_calling".to_owned(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Probe failed: LLM error: does not support tools".to_owned(),
    };
    let mut cache = ProbeCache::default();
    cache.put(profile);
    cache.save(&path).expect("save stale");

    // save() writes path.with_extension("tmp"); a directory there blocks rewrite.
    std::fs::create_dir(path.with_extension("tmp")).expect("block save tmp");

    let loaded = ProbeCache::load(&path).expect("load still returns migrated rows");
    let got = loaded.get("m", "p").expect("session-correct migrated hit");
    assert_eq!(got.tool_calling.level, CapabilityLevel::Weak);
    assert_eq!(got.tool_calling.score, 0.0);

    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("reread")).expect("json");
    let key = ProbeCache::cache_key("m", "p", DEFAULT_PROBE_EFFORT, PROBE_SUITE_VERSION);
    let stored = &raw["profiles"][&key]["profile"];
    assert_eq!(
        stored["toolCalling"]["level"], "medium",
        "disk stays unmigrated when save fails"
    );
    assert_eq!(stored["toolCalling"]["score"].as_f64(), Some(0.5));
}

#[test]
fn legacy_json_deserializes_with_defaults() {
    let entry_json = serde_json::json!({
        "profile": sample_profile(),
        "cachedAt": 1,
    });
    let entry: CacheEntry = serde_json::from_value(entry_json).unwrap();
    assert_eq!(entry.reasoning_effort, "unset");
    assert_eq!(entry.probe_suite_version, 1);
}
