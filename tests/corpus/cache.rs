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
        probed_context_floor: None,
    }
}

#[test]
fn cache_key_includes_effort_and_suite() {
    let k = ProbeCache::cache_key("gpt", "openai", "unset", 2);
    assert_eq!(k, "gpt|openai|unset|v2|full|novision|ctxnone");
}

#[test]
fn cache_key_format_is_model_provider_unset_v89() {
    assert_eq!(PROBE_SUITE_VERSION, 89);
    assert_eq!(CACHE_TTL_SECS, 30 * 24 * 60 * 60);
    let k = ProbeCache::cache_key(
        "model",
        "provider",
        DEFAULT_PROBE_EFFORT,
        PROBE_SUITE_VERSION,
    );
    assert_eq!(k, "model|provider|unset|v89|full|novision|ctxnone");
}

#[test]
fn cache_key_includes_cheap_and_vision_knobs() {
    let cheap = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, true, false, None);
    let full = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, false, None);
    let vision = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, true, None);
    assert_ne!(cheap, full);
    assert_ne!(full, vision);
    assert_eq!(cheap, "m|p|unset|v7|cheap|novision|ctxnone");
    assert_eq!(full, "m|p|unset|v7|full|novision|ctxnone");
    assert_eq!(vision, "m|p|unset|v7|full|vision|ctxnone");
}

#[test]
fn cache_key_includes_advertised_context() {
    let none = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, false, None);
    let cap = ProbeCache::cache_key_with_knobs("m", "p", "unset", 7, false, false, Some(2000));
    assert_ne!(none, cap);
    assert_eq!(none, "m|p|unset|v7|full|novision|ctxnone");
    assert_eq!(cap, "m|p|unset|v7|full|novision|ctx2000");
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
fn find_profile_sees_cheap_row() {
    let mut cache = ProbeCache::default();
    let mut profile = sample_profile();
    profile.effective_context_tokens = Some(8192);
    cache.put_with_knobs(profile, true, false, None);
    let found = cache.find_profile("m", "p").expect("cheap row");
    assert_eq!(found.effective_context_tokens, Some(8192));
}

#[test]
fn find_profile_prefers_newer_cached_at() {
    let mut cache = ProbeCache::default();
    let mut old = sample_profile();
    old.effective_context_tokens = Some(1024);
    let mut new = sample_profile();
    new.effective_context_tokens = Some(8192);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs();
    cache.profiles.insert(
        "old".into(),
        CacheEntry {
            profile: old,
            cached_at: now.saturating_sub(60),
            reasoning_effort: DEFAULT_PROBE_EFFORT.into(),
            probe_suite_version: PROBE_SUITE_VERSION,
        },
    );
    cache.profiles.insert(
        "new".into(),
        CacheEntry {
            profile: new,
            cached_at: now,
            reasoning_effort: DEFAULT_PROBE_EFFORT.into(),
            probe_suite_version: PROBE_SUITE_VERSION,
        },
    );
    let found = cache.find_profile("m", "p").expect("row");
    assert_eq!(found.effective_context_tokens, Some(8192));
}

#[test]
fn find_profile_with_cost_reports_cheap_row() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample_profile(), true, false, None);
    let (_, cheap) = cache.find_profile_with_cost("m", "p").expect("cheap row");
    assert!(cheap, "fallback row must report cheap");
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
        None,
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
                None,
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
                None,
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
        None,
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
                None,
            )
            .is_none(),
        "suite v73 must not hit v6 cache entry"
    );
}

#[test]
fn find_profile_with_cost_and_advertised_keeps_isolation() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample_profile(), false, false, Some(2000));
    assert!(
        cache.find_profile("m", "p").is_some(),
        "export may still use a loose advertised row"
    );
    assert!(
        cache
            .find_profile_with_cost_and_advertised("m", "p", None)
            .is_none(),
        "advertised-capped row must not satisfy an uncapped probe"
    );
    assert!(
        cache
            .find_profile_with_cost_and_advertised("m", "p", Some(2000))
            .is_some(),
        "same advertised cap must still hit"
    );

    let mut uncapped = ProbeCache::default();
    uncapped.put_with_knobs(sample_profile(), false, false, None);
    assert!(
        uncapped
            .find_profile_with_cost_and_advertised("m", "p", Some(2000))
            .is_none(),
        "uncapped row must not satisfy an advertised-capped probe"
    );
    assert!(
        uncapped
            .find_profile_with_cost_and_advertised("m", "p", None)
            .is_some(),
        "uncapped probe must still hit an uncapped row"
    );
}

#[test]
fn find_profile_accepts_overlay_provider_aliases() {
    let mut cache = ProbeCache::default();
    let mut openai = sample_profile();
    openai.model_id = "gpt-4o".into();
    openai.provider = "api.openai.com".into();
    cache.put(openai);
    assert!(
        cache.find_profile("gpt-4o", "openai").is_some(),
        "--provider openai must find a row stored as api.openai.com"
    );
    assert!(
        cache.find_profile("gpt-4o", "OpenAI").is_some(),
        "provider aliases must compare case-insensitively"
    );

    let mut openrouter = sample_profile();
    openrouter.model_id = "claude".into();
    openrouter.provider = "openrouter.ai".into();
    cache.put(openrouter);
    assert!(cache.find_profile("claude", "openrouter").is_some());

    let mut ollama = sample_profile();
    ollama.model_id = "qwen".into();
    ollama.provider = "127.0.0.1".into();
    cache.put(ollama);
    assert!(cache.find_profile("qwen", "ollama").is_some());
    assert!(cache.find_profile("qwen", "localhost").is_some());
    assert!(cache.find_profile("qwen", "::1").is_some());
}

#[test]
fn find_profile_treats_all_zeros_as_ollama() {
    let mut cache = ProbeCache::default();
    let mut zeros = sample_profile();
    zeros.model_id = "qwen".into();
    zeros.provider = "0.0.0.0".into();
    cache.put(zeros);
    assert!(
        cache.find_profile("qwen", "ollama").is_some(),
        "--provider ollama must find a row stored as 0.0.0.0"
    );
}

#[test]
fn find_profile_strips_normalized_provider_prefix() {
    let mut cache = ProbeCache::default();
    let mut profile = sample_profile();
    profile.model_id = "anthropic/claude-3.5-sonnet".into();
    profile.provider = "openrouter".into();
    cache.put(profile);
    assert!(
        cache
            .find_profile("openrouter/anthropic/claude-3.5-sonnet", "openrouter")
            .is_some(),
        "export openrouter/anthropic/claude-3.5-sonnet must hit probed anthropic/claude-3.5-sonnet"
    );
    assert!(
        cache
            .get_with_knobs(
                "openrouter/anthropic/claude-3.5-sonnet",
                "openrouter",
                false,
                false,
                None,
            )
            .is_some(),
        "MCP exact-knob lookup must retry after stripping openrouter/"
    );
}

#[test]
fn find_profile_strips_stored_host_prefix_via_stored_family() {
    let mut cache = ProbeCache::default();
    let mut ollama = sample_profile();
    ollama.model_id = "localhost/qwen".into();
    ollama.provider = "localhost".into();
    cache.put(ollama);
    assert!(
        cache.find_profile("qwen", "ollama").is_some(),
        "stored localhost/qwen under localhost must hit --model qwen --provider ollama"
    );

    let mut openai = sample_profile();
    openai.model_id = "api.openai.com/gpt-4o".into();
    openai.provider = "api.openai.com".into();
    cache.put(openai);
    assert!(
        cache.find_profile("gpt-4o", "openai").is_some(),
        "stored api.openai.com/gpt-4o under api.openai.com must hit --model gpt-4o --provider openai"
    );
}

#[test]
fn find_profile_matches_when_stored_has_provider_prefix() {
    let mut cache = ProbeCache::default();
    let mut profile = sample_profile();
    profile.model_id = "openrouter/anthropic/claude-3.5-sonnet".into();
    profile.provider = "openrouter".into();
    cache.put(profile);
    assert!(
        cache
            .find_profile("anthropic/claude-3.5-sonnet", "openrouter")
            .is_some(),
        "export anthropic/claude-3.5-sonnet must hit probed openrouter/anthropic/claude-3.5-sonnet"
    );
    assert!(
        cache
            .get_with_knobs(
                "anthropic/claude-3.5-sonnet",
                "openrouter",
                false,
                false,
                None,
            )
            .is_some(),
        "MCP exact-knob lookup must match after stripping stored openrouter/"
    );
}

#[test]
fn get_with_knobs_accepts_provider_aliases_without_cheap_fallback() {
    let mut cache = ProbeCache::default();
    let mut vision = sample_profile();
    vision.model_id = "gpt-4o".into();
    vision.provider = "api.openai.com".into();
    vision.effective_context_tokens = Some(128000);
    cache.put_with_knobs(vision, false, true, None);

    let mut cheap = sample_profile();
    cheap.model_id = "gpt-4o".into();
    cheap.provider = "api.openai.com".into();
    cheap.effective_context_tokens = Some(4096);
    cache.put_with_knobs(cheap, true, false, None);

    let hit = cache
        .get_with_knobs("gpt-4o", "openai", false, true, None)
        .expect("--vision must find api.openai.com via openai alias");
    assert_eq!(
        hit.effective_context_tokens,
        Some(128000),
        "alias-aware knob lookup must not fall through to the cheap/novision row"
    );
    assert!(
        cache
            .get_with_knobs("gpt-4o", "openai", false, false, None)
            .is_none(),
        "full/novision must not hit a vision or cheap row"
    );
}

#[test]
fn advertised_cache_misses_on_different_cap() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample_profile(), false, false, None);
    assert!(
        cache.get_with_knobs("m", "p", false, false, None).is_some(),
        "uncapped put must hit uncapped get"
    );
    assert!(
        cache
            .get_with_knobs("m", "p", false, false, Some(2000))
            .is_none(),
        "advertised 2000 must not reuse an uncapped climb"
    );
}

#[test]
fn cheap_cache_misses_on_full_and_vision() {
    let mut cache = ProbeCache::default();
    cache.put_with_knobs(sample_profile(), true, false, None);
    assert!(
        cache.get_with_knobs("m", "p", true, false, None).is_some(),
        "cheap/novision must hit its own entry"
    );
    assert!(
        cache.get_with_knobs("m", "p", false, false, None).is_none(),
        "full must not return a cheap-cached profile"
    );
    assert!(
        cache.get_with_knobs("m", "p", true, true, None).is_none(),
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
    cache.save(&path).expect("overwrite existing cache file");
    let reloaded = ProbeCache::load(&path).expect("load after overwrite");
    assert_eq!(reloaded.get("m", "p").expect("hit").model_id, "m");
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
fn put_then_load_preserves_probed_context_floor() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let mut profile = sample_profile();
    profile.effective_context_tokens = None;
    profile.probed_context_floor = Some(4096);
    cache.put(profile);
    cache.save(&path).expect("save");
    let loaded = ProbeCache::load(&path).expect("load");
    let got = loaded.get("m", "p").expect("hit after reload");
    assert_eq!(got.effective_context_tokens, None);
    assert_eq!(got.probed_context_floor, Some(4096));
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
fn save_keeps_newer_other_model_row() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probes.json");
    let mut a = ProbeCache::default();
    let mut pa = sample_profile();
    pa.model_id = "a".into();
    a.put(pa);
    a.save(&path).expect("save a");
    let mut b = ProbeCache::default();
    let mut pb = sample_profile();
    pb.model_id = "b".into();
    b.put(pb);
    b.save(&path).expect("save b merges a");
    let loaded = ProbeCache::load(&path).expect("load");
    assert!(
        loaded.find_profile("a", "p").is_some(),
        "a must survive b save"
    );
    assert!(loaded.find_profile("b", "p").is_some(), "b must be stored");
}

#[test]
fn load_rejects_oversized_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("huge.json");
    let blob = vec![b'x'; 8 * 1024 * 1024 + 1];
    std::fs::write(&path, blob).expect("write");
    let err = ProbeCache::load(&path).expect_err("oversized");
    assert!(err.to_string().contains("too large"), "{err}");
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

    // save() writes path.with_extension("tmp-<pid>"); a directory there blocks rewrite.
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::create_dir(tmp).expect("block save tmp");

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
