use canact::{
    CapabilityLevel, CapabilityProfile, MockLlm, ProbeCache, ProbeClient, ProbeError, ProbeFinish,
    ProbeRequest, ProbeResult, ProbeRun, ProbeRunner, ProbeTool, classify, resolve_probe,
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

fn tool_request() -> ProbeRequest {
    ProbeRequest {
        messages: Vec::new(),
        tools: vec![ProbeTool {
            name: "read_file".to_owned(),
            description: "read".to_owned(),
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }],
        model: "m".to_owned(),
        temperature: None,
        max_tokens: None,
    }
}

fn text_request() -> ProbeRequest {
    ProbeRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        model: "m".to_owned(),
        temperature: None,
        max_tokens: None,
    }
}

#[test]
fn resolve_probe_timeout_is_not_cacheable() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Transient("timeout".into()));
    let (result, cacheable) = resolve_probe(err, "streaming_tool_calls").expect("synthesized");
    assert_eq!(result.level, CapabilityLevel::Medium);
    assert!(
        !cacheable,
        "timeout must not be written to the 30-day cache"
    );
    assert!(result.details.contains("timeout"));
}

#[test]
fn resolve_probe_rate_limit_is_not_cacheable() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::RateLimit { retry_after: None });
    let (result, cacheable) = resolve_probe(err, "streaming_tool_calls").expect("synthesized");
    assert_eq!(result.level, CapabilityLevel::Medium);
    assert!(!cacheable, "429 must not be written to the 30-day cache");
}

#[test]
fn resolve_probe_no_tools_is_weak_and_cacheable() {
    let err: Result<ProbeResult, ProbeError> =
        Err(ProbeError::Llm("does not support tools".into()));
    let (result, cacheable) = resolve_probe(err, "tool_calling").expect("synthesized");
    assert_eq!(result.level, CapabilityLevel::Weak);
    assert_eq!(result.score, 0.0);
    assert!(
        cacheable,
        "definitive no-tools may stay in the 30-day cache"
    );
}

#[test]
fn resolve_probe_non_listed_no_tools_is_medium_and_cacheable() {
    let err: Result<ProbeResult, ProbeError> =
        Err(ProbeError::Llm("does not support tools".into()));
    let (result, cacheable) = resolve_probe(err, "json_output").expect("synthesized");
    assert_eq!(result.level, CapabilityLevel::Medium);
    assert_eq!(result.score, 0.5);
    assert!(cacheable, "non-listed no-tools is still cacheable");
}

#[test]
fn resolve_probe_auth_aborts() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Auth("bad key".into()));
    let result = resolve_probe(err, "tool_calling");
    match result {
        Err(ProbeError::Auth(msg)) => assert_eq!(msg, "bad key"),
        other => panic!("expected Auth abort, got {other:?}"),
    }
}

#[tokio::test]
async fn mock_llm_chat_with_tools_returns_tool_call() {
    let llm = MockLlm::new("m", "p");
    let resp = llm.chat(tool_request()).await.expect("chat");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "call_1");
    assert_eq!(resp.tool_calls[0].name, "read_file");
    assert!(resp.tool_calls[0].arguments.is_empty());
    assert_eq!(resp.finish, ProbeFinish::ToolCalls);
}

#[tokio::test]
async fn mock_llm_chat_without_tools_returns_text() {
    let llm = MockLlm::new("m", "p");
    let resp = llm.chat(text_request()).await.expect("chat");
    assert_eq!(resp.text, "ok");
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.finish, ProbeFinish::Stop);
}

#[test]
fn persist_does_not_write_when_cacheable_false() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let run = ProbeRun {
        profile: sample_profile(),
        cacheable: false,
    };
    let wrote = run.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "persist must skip uncacheable runs");
    assert!(
        !path.exists(),
        "cache file must not be written when cacheable=false"
    );
    assert!(cache.get("m", "p").is_none());
}

#[test]
fn runner_persist_without_run_does_not_write() {
    let runner = ProbeRunner::new(MockLlm::new("m", "p"));
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let wrote = runner.persist(&mut cache, &path).expect("persist");
    assert!(!wrote);
    assert!(!path.exists());
}
