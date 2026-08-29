use std::future::Future;
use std::sync::{Arc, Mutex};

use canact::{
    CapabilityLevel, CapabilityProfile, CatalogPriors, DIMENSION_NAMES, MockLlm, ProbeCache,
    ProbeClient, ProbeContent, ProbeContentPart, ProbeError, ProbeFinish, ProbeRequest,
    ProbeResponse, ProbeResult, ProbeRun, ProbeRunner, ProbeStreamChunk, ProbeTool, classify,
    resolve_probe,
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
    assert_eq!(
        resp.tool_calls[0]
            .arguments
            .get("path")
            .and_then(|v| v.as_str()),
        Some("/tmp/test.txt")
    );
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

const VISION_SKIP: &str = "Skipped: provider does not advertise vision support";
const EXPENSIVE_SKIP: &str = "Skipped: free-tier model, conserving API budget";

struct RecordingLlm {
    inner: MockLlm,
    requests: Arc<Mutex<Vec<ProbeRequest>>>,
}

impl RecordingLlm {
    fn wrap(inner: MockLlm) -> (Self, Arc<Mutex<Vec<ProbeRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner,
                requests: requests.clone(),
            },
            requests,
        )
    }

    fn record(&self, req: &ProbeRequest) {
        self.requests.lock().expect("lock").push(req.clone());
    }
}

impl ProbeClient for RecordingLlm {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        self.record(&req);
        self.inner.chat(req)
    }

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl futures::Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        self.record(&req);
        self.inner.stream_chat(req)
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn provider(&self) -> &str {
        self.inner.provider()
    }

    fn catalog(&self) -> CatalogPriors {
        self.inner.catalog()
    }
}

fn request_has_image(req: &ProbeRequest) -> bool {
    req.messages.iter().any(|m| match &m.content {
        ProbeContent::Parts(parts) => parts
            .iter()
            .any(|p| matches!(p, ProbeContentPart::ImageBase64 { .. })),
        ProbeContent::Text(_) => false,
    })
}

fn expensive_names() -> [&'static str; 4] {
    [
        "one_shot_tool_plan",
        "multi_turn_task_sequencing",
        "context_faithfulness",
        "multi_turn_memory",
    ]
}

#[tokio::test]
async fn vision_runs_only_when_catalog_some_true() {
    let catalog = CatalogPriors {
        supports_vision: Some(true),
        ..CatalogPriors::default()
    };
    let (llm, requests) = RecordingLlm::wrap(MockLlm::new("m", "p").with_catalog(catalog));
    let profile = ProbeRunner::new(llm).run().await.expect("run");
    assert_ne!(profile.vision.details, VISION_SKIP);
    let rec = requests.lock().expect("lock");
    assert!(
        rec.iter().any(request_has_image),
        "vision probe must send an image when catalog.supports_vision is Some(true)"
    );
}

#[tokio::test]
async fn vision_skipped_when_catalog_none() {
    let (llm, requests) = RecordingLlm::wrap(MockLlm::new("m", "p"));
    let profile = ProbeRunner::new(llm).run().await.expect("run");
    assert_eq!(profile.vision.level, CapabilityLevel::Weak);
    assert_eq!(profile.vision.score, 0.0);
    assert_eq!(profile.vision.details, VISION_SKIP);
    let rec = requests.lock().expect("lock");
    assert!(
        !rec.iter().any(request_has_image),
        "vision must not run when catalog.supports_vision is None"
    );
}

#[tokio::test]
async fn vision_skipped_when_catalog_some_false() {
    let catalog = CatalogPriors {
        supports_vision: Some(false),
        ..CatalogPriors::default()
    };
    let (llm, requests) = RecordingLlm::wrap(MockLlm::new("m", "p").with_catalog(catalog));
    let profile = ProbeRunner::new(llm).run().await.expect("run");
    assert_eq!(profile.vision.level, CapabilityLevel::Weak);
    assert_eq!(profile.vision.score, 0.0);
    assert_eq!(profile.vision.details, VISION_SKIP);
    let rec = requests.lock().expect("lock");
    assert!(
        !rec.iter().any(request_has_image),
        "vision must not run when catalog.supports_vision is Some(false)"
    );
}

#[tokio::test]
async fn supports_tools_false_does_not_skip_tool_probes() {
    let catalog = CatalogPriors {
        supports_tools: Some(false),
        ..CatalogPriors::default()
    };
    let (llm, requests) = RecordingLlm::wrap(MockLlm::new("m", "p").with_catalog(catalog));
    let profile = ProbeRunner::new(llm).run().await.expect("run");
    let rec = requests.lock().expect("lock");
    assert!(
        rec.iter().any(|r| !r.tools.is_empty()),
        "tool_calling must still send tools when catalog.supports_tools is Some(false)"
    );
    assert_eq!(
        profile.tool_calling.details,
        "Valid tool call with correct name and arguments"
    );
    assert_ne!(
        profile.tool_calling.details, VISION_SKIP,
        "catalog supports_tools must not be persisted as a skip/Strong flag"
    );
}

#[tokio::test]
async fn xml_skipped_when_native_tool_calling_is_strong() {
    let profile = ProbeRunner::new(MockLlm::new("m", "p"))
        .run()
        .await
        .expect("run");
    assert_eq!(profile.tool_calling.level, CapabilityLevel::Strong);
    assert!(
        profile.xml_tool_calling.details.contains("Not tested"),
        "XML probe should be marked not tested when native is Strong, got: {}",
        profile.xml_tool_calling.details
    );
    assert_eq!(
        profile.xml_tool_calling.details,
        "Not tested (native tool calling is Strong; XML fallback unused)"
    );
    assert_eq!(profile.xml_tool_calling.level, CapabilityLevel::Strong);
    assert_eq!(profile.xml_tool_calling.score, 1.0);
}

#[tokio::test]
async fn new_throttled_sets_expensive_dims_to_free_tier_skip() {
    let profile = ProbeRunner::new_throttled(MockLlm::new("m", "p"))
        .run()
        .await
        .expect("run");
    for name in expensive_names() {
        let result = profile
            .dimension_result(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(result.details, EXPENSIVE_SKIP, "{name}");
        assert_eq!(result.level, CapabilityLevel::Medium, "{name}");
        assert_eq!(result.score, 0.5, "{name}");
        assert_eq!(result.name, name);
    }
}

#[tokio::test]
async fn cheap_sets_expensive_dims_to_free_tier_skip() {
    let profile = ProbeRunner::new(MockLlm::new("m", "p"))
        .cheap()
        .run()
        .await
        .expect("run");
    for name in expensive_names() {
        let result = profile
            .dimension_result(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(result.details, EXPENSIVE_SKIP, "{name}");
        assert_eq!(result.score, 0.5, "{name}");
    }
}

const NOT_PROBED: &str = "Not probed (cached before this probe existed)";

#[tokio::test]
async fn runner_returns_complete_profile() {
    let profile = ProbeRunner::new(MockLlm::new("m", "p"))
        .run()
        .await
        .expect("run");
    assert_eq!(DIMENSION_NAMES.len(), 20);
    for &name in DIMENSION_NAMES {
        let result = profile
            .dimension_result(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(result.name, name, "{name}");
        assert_ne!(
            result.details, NOT_PROBED,
            "{name} must run instead of named_default"
        );
    }
}

#[tokio::test]
async fn run_fills_edit_json_instruction_probes() {
    let profile = ProbeRunner::new(MockLlm::new("m", "p"))
        .run()
        .await
        .expect("run");
    for (result, name) in [
        (&profile.search_replace, "search_replace"),
        (&profile.unified_diff, "unified_diff"),
        (&profile.json_output, "json_output"),
        (&profile.instruction_following, "instruction_following"),
    ] {
        assert_eq!(result.name, name, "{name} must keep its living probe name");
        assert_ne!(
            result.details, NOT_PROBED,
            "{name} must run instead of named_default"
        );
    }
}

#[tokio::test]
async fn auth_aborts_run_detailed_and_does_not_persist() {
    let runner =
        ProbeRunner::new(MockLlm::new("m", "p").with_error(ProbeError::Auth("bad".into())));
    let result = runner.run_detailed().await;
    match result {
        Err(ProbeError::Auth(msg)) => assert_eq!(msg, "bad"),
        other => panic!("expected Auth abort, got {other:?}"),
    }

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let wrote = runner.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "auth abort must not persist");
    assert!(!path.exists(), "auth abort must not write a cache file");
    assert!(cache.get("m", "p").is_none());
}
