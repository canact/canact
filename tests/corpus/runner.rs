use std::future::Future;
use std::sync::{Arc, Mutex};

use canact::{
    CapabilityLevel, CapabilityProfile, CatalogPriors, DIMENSION_NAMES, MockLlm, ProbeCache,
    ProbeClient, ProbeContent, ProbeContentPart, ProbeError, ProbeFinish, ProbeRequest,
    ProbeResponse, ProbeResult, ProbeRun, ProbeRunner, ProbeStreamChunk, ProbeTool,
    TOOL_PROBE_NAMES, classify, resolve_probe,
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
fn transient_xml_medium_does_not_open_can_use_tools() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Transient("timeout".into()));
    let (xml, cacheable) = resolve_probe(err, "xml_tool_calling").expect("synthesized");
    assert_eq!(xml.level, CapabilityLevel::Medium);
    assert!(!cacheable, "transient XML must not persist");
    let mut card = sample_profile();
    card.tool_calling = ProbeResult {
        name: "tool_calling".into(),
        score: 0.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: "does not support tools".into(),
    };
    card.xml_tool_calling = xml;
    assert!(!card.can_use_tools());
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
fn resolve_probe_no_tools_is_weak_for_all_tool_probe_names() {
    for name in TOOL_PROBE_NAMES {
        let err: Result<ProbeResult, ProbeError> =
            Err(ProbeError::Llm("does not support tools".into()));
        let (result, cacheable) = resolve_probe(err, name).expect("synthesized");
        assert_eq!(result.level, CapabilityLevel::Weak, "{name}");
        assert_eq!(result.score, 0.0, "{name}");
        assert!(cacheable, "{name}");
    }
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

#[test]
fn resolve_probe_not_found_aborts() {
    let err: Result<ProbeResult, ProbeError> =
        Err(ProbeError::NotFound("model 'x' not found".into()));
    let result = resolve_probe(err, "tool_calling");
    match result {
        Err(ProbeError::NotFound(msg)) => assert!(msg.contains("not found"), "{msg}"),
        other => panic!("expected NotFound abort, got {other:?}"),
    }
}

#[test]
fn resolve_probe_unreachable_host_aborts() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Transient(
        "failed to connect: error sending request for url (http://127.0.0.1:1/v1/chat/completions)"
            .into(),
    ));
    let result = resolve_probe(err, "tool_calling");
    match result {
        Err(ProbeError::Transient(msg)) => {
            assert!(msg.contains("failed to connect"), "{msg}");
        }
        other => panic!("expected Transient abort, got {other:?}"),
    }
}

#[test]
fn resolve_probe_connect_timeout_aborts() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Transient(
        "failed to connect: error sending request for url (http://192.0.2.1:11434/v1/chat/completions): timed out"
            .into(),
    ));
    let result = resolve_probe(err, "tool_calling");
    match result {
        Err(ProbeError::Transient(msg)) => {
            assert!(msg.contains("failed to connect:"), "{msg}");
        }
        other => panic!("expected connect-timeout abort, got {other:?}"),
    }
}

#[test]
fn resolve_probe_send_timeout_stays_medium() {
    let err: Result<ProbeResult, ProbeError> = Err(ProbeError::Transient(
        "error sending request for url (http://127.0.0.1:11434/v1/chat/completions): operation timed out"
            .into(),
    ));
    let (result, cacheable) = resolve_probe(err, "tool_calling").expect("synthesized");
    assert_eq!(result.level, CapabilityLevel::Medium);
    assert!(!cacheable, "timeout must not persist");
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
fn persist_cheap_run_is_not_returned_as_full() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let run = ProbeRun {
        profile: sample_profile(),
        cacheable: true,
        skip_expensive: true,
        vision: false,
        advertised_context_tokens: None,
    };
    let wrote = run.persist(&mut cache, &path).expect("persist");
    assert!(wrote);
    assert!(
        cache.get_with_knobs("m", "p", true, false, None).is_some(),
        "cheap persist must be readable with cheap knobs"
    );
    assert!(
        cache.get("m", "p").is_none(),
        "full/default get must not return a cheap-persisted profile"
    );
}

#[test]
fn persist_advertised_is_not_returned_as_uncapped() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let run = ProbeRun {
        profile: sample_profile(),
        cacheable: true,
        skip_expensive: false,
        vision: false,
        advertised_context_tokens: Some(2000),
    };
    let wrote = run.persist(&mut cache, &path).expect("persist");
    assert!(wrote);
    assert!(
        cache
            .get_with_knobs("m", "p", false, false, Some(2000))
            .is_some(),
        "advertised persist must hit the same advertised get"
    );
    assert!(
        cache.get("m", "p").is_none(),
        "uncapped get must not reuse an advertised-capped climb"
    );
}

#[test]
fn persist_does_not_write_when_cacheable_false() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let run = ProbeRun {
        profile: sample_profile(),
        cacheable: false,
        skip_expensive: false,
        vision: false,
        advertised_context_tokens: None,
    };
    let wrote = run.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "persist must skip uncacheable runs");
    assert!(
        !path.exists(),
        "cache file must not be written when cacheable=false"
    );
    assert!(cache.get("m", "p").is_none());
}

struct TruncatedToolLlm;

impl ProbeClient for TruncatedToolLlm {
    fn chat(
        &self,
        _req: ProbeRequest,
    ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        std::future::ready(Ok(ProbeResponse {
            text: "I will call read_file after this leftover reasoning.".into(),
            tool_calls: Vec::new(),
            finish: ProbeFinish::Length,
            usage: None,
        }))
    }

    fn stream_chat(
        &self,
        _req: ProbeRequest,
    ) -> impl futures::Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        futures::stream::empty()
    }

    fn model_id(&self) -> &str {
        "m"
    }

    fn provider(&self) -> &str {
        "p"
    }
}

#[tokio::test]
async fn truncated_tool_calling_is_not_written_to_cache() {
    let runner = ProbeRunner::new(TruncatedToolLlm);
    let run = runner.run_detailed().await.expect("run_detailed");
    assert_eq!(run.profile.tool_calling.level, CapabilityLevel::Medium);
    assert!(
        run.profile.tool_calling.details.contains("truncated"),
        "{}",
        run.profile.tool_calling.details
    );
    assert!(
        !run.cacheable,
        "Length + no tool call must not be a 30-day Weak card"
    );

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let wrote = run.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "persist must skip truncated tool_calling");
    assert!(
        !path.exists(),
        "cache file must not be written when tool_calling is truncated"
    );
    assert!(cache.get("m", "p").is_none());
}

#[test]
fn uncacheable_run_envelope_cacheable_false() {
    let run = ProbeRun {
        profile: sample_profile(),
        cacheable: false,
        skip_expensive: true,
        vision: false,
        advertised_context_tokens: Some(40960),
    };
    let envelope = run.host_policy_envelope();
    assert_eq!(envelope["cacheable"], false, "{envelope}");
    assert_eq!(envelope["skipExpensive"], true, "{envelope}");
    assert_eq!(envelope["advertisedContextTokens"], 40960, "{envelope}");
    assert!(envelope["recommendedContextTokens"].is_null(), "{envelope}");
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

const VISION_SKIP_NOT_REQUESTED: &str = "Skipped: vision not requested";
const VISION_SKIP_FLAG: &str = "Skipped: --no-vision";
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
    assert_ne!(profile.vision.details, VISION_SKIP_NOT_REQUESTED);
    assert_ne!(profile.vision.details, VISION_SKIP_FLAG);
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
    assert_eq!(profile.vision.details, VISION_SKIP_NOT_REQUESTED);
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
    assert_eq!(profile.vision.details, VISION_SKIP_FLAG);
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
        profile.tool_calling.details, VISION_SKIP_NOT_REQUESTED,
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
    assert!(
        !profile.meets(&[("one_shot_tool_plan", CapabilityLevel::Medium)]),
        "cheap skip must not satisfy a Medium requirement"
    );
    assert_eq!(
        profile.dimension_level("one_shot_tool_plan"),
        Some(CapabilityLevel::Weak)
    );
    let envelope = profile.host_policy_envelope();
    assert_eq!(envelope["probes"]["oneShotToolPlan"]["status"], "skipped");
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
async fn not_found_aborts_run_detailed_and_does_not_persist() {
    let runner = ProbeRunner::new(
        MockLlm::new("m", "p").with_error(ProbeError::NotFound("model 'x' not found".into())),
    );
    let result = runner.run_detailed().await;
    match result {
        Err(ProbeError::NotFound(msg)) => assert!(msg.contains("not found"), "{msg}"),
        other => panic!("expected NotFound abort, got {other:?}"),
    }

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let wrote = runner.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "not-found abort must not persist");
    assert!(
        !path.exists(),
        "not-found abort must not write a cache file"
    );
    assert!(cache.get("m", "p").is_none());
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

#[derive(Clone, Copy)]
enum LadderReply {
    Recall,
    Transient,
    Auth,
    RecallThenTransient,
    RecallThenAuth,
}

struct ContextLadderLlm {
    inner: MockLlm,
    requests: Arc<Mutex<Vec<ProbeRequest>>>,
    ladder: LadderReply,
}

impl ContextLadderLlm {
    fn wrap(inner: MockLlm, ladder: LadderReply) -> (Self, Arc<Mutex<Vec<ProbeRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner,
                requests: requests.clone(),
                ladder,
            },
            requests,
        )
    }
}

impl ProbeClient for ContextLadderLlm {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        self.requests.lock().expect("lock").push(req.clone());
        if is_ladder_request(&req) {
            let tokens = request_token_estimate(&req);
            let recall = Ok(ProbeResponse {
                text: "WH-4481\nproto-9.2.11\n2840".to_owned(),
                tool_calls: Vec::new(),
                finish: ProbeFinish::Stop,
                usage: None,
            });
            let result = match self.ladder {
                LadderReply::Recall => recall,
                LadderReply::Transient => Err(ProbeError::Transient("timeout".into())),
                LadderReply::Auth => Err(ProbeError::Auth("bad".into())),
                LadderReply::RecallThenTransient if tokens < 8192 => recall,
                LadderReply::RecallThenTransient => Err(ProbeError::Transient("timeout".into())),
                LadderReply::RecallThenAuth if tokens < 8192 => recall,
                LadderReply::RecallThenAuth => Err(ProbeError::Auth("bad".into())),
            };
            futures::future::Either::Left(std::future::ready(result))
        } else {
            futures::future::Either::Right(self.inner.chat(req))
        }
    }

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl futures::Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        self.requests.lock().expect("lock").push(req.clone());
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

fn request_plain_text(req: &ProbeRequest) -> String {
    let mut out = String::new();
    for message in &req.messages {
        match &message.content {
            ProbeContent::Text(text) => out.push_str(text),
            ProbeContent::Parts(parts) => {
                for part in parts {
                    if let ProbeContentPart::Text { text } = part {
                        out.push_str(text);
                    }
                }
            }
        }
    }
    out
}

fn is_ladder_request(req: &ProbeRequest) -> bool {
    request_plain_text(req).contains("inventory warehouse code")
}

fn request_token_estimate(req: &ProbeRequest) -> u32 {
    u32::try_from((request_plain_text(req).chars().count() / 4).max(1)).unwrap_or(u32::MAX)
}

fn ladder_requests(recorded: &[ProbeRequest]) -> Vec<&ProbeRequest> {
    recorded
        .iter()
        .filter(|req| is_ladder_request(req))
        .collect()
}

#[tokio::test]
async fn full_run_populates_effective_context_tokens() {
    let (llm, requests) = ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::Recall);
    let profile = ProbeRunner::new(llm).run().await.expect("run");
    assert_eq!(profile.effective_context_tokens, Some(16384));
    assert_eq!(profile.probed_context_floor, Some(16384));
    let envelope = profile.host_policy_envelope();
    assert_eq!(envelope["effectiveContextTokens"], 16384);
    assert_eq!(
        envelope["probedContextFloor"], 16384,
        "finished climb floor matches effective; envelope={envelope}"
    );
    let rec = requests.lock().expect("lock");
    let ladder = ladder_requests(&rec);
    assert_eq!(ladder.len(), 3);
    let t0 = request_token_estimate(ladder[0]);
    let t1 = request_token_estimate(ladder[1]);
    let t2 = request_token_estimate(ladder[2]);
    assert!((4096..8192).contains(&t0), "{t0}");
    assert!((8192..16384).contains(&t1), "{t1}");
    assert!(t2 >= 16384, "{t2}");
}

#[tokio::test]
async fn cheap_run_attempts_at_most_4k_rung() {
    for throttled in [true, false] {
        let (llm, requests) = ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::Recall);
        let profile = if throttled {
            ProbeRunner::new_throttled(llm).run().await.expect("run")
        } else {
            ProbeRunner::new(llm).cheap().run().await.expect("run")
        };
        assert_eq!(
            profile.effective_context_tokens, None,
            "cheap 4k pass is not a finished size; throttled={throttled}"
        );
        let envelope = profile.host_policy_envelope();
        assert!(
            envelope["effectiveContextTokens"].is_null(),
            "throttled={throttled} envelope={envelope}"
        );
        assert_eq!(
            profile.probed_context_floor,
            Some(4096),
            "cheap 4k pass must persist a floor; throttled={throttled}"
        );
        assert_eq!(
            envelope["probedContextFloor"], 4096,
            "cheap 4k pass must publish a floor; throttled={throttled} envelope={envelope}"
        );
        assert_eq!(
            profile.context_faithfulness.details, EXPENSIVE_SKIP,
            "cheap must still skip context_faithfulness"
        );
        let rec = requests.lock().expect("lock");
        let ladder = ladder_requests(&rec);
        assert_eq!(ladder.len(), 1, "throttled={throttled}");
        let tokens = request_token_estimate(ladder[0]);
        assert!(
            (4096..8192).contains(&tokens),
            "throttled={throttled} tokens={tokens}"
        );
    }
}

#[tokio::test]
async fn ladder_transient_leaves_none_and_uncacheable() {
    let (llm, _) = ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::Transient);
    let run = ProbeRunner::new(llm)
        .run_detailed()
        .await
        .expect("run_detailed");
    assert_eq!(run.profile.effective_context_tokens, None);
    assert!(
        !run.cacheable,
        "transient ladder must not be a 30-day cache hit"
    );
}

#[tokio::test]
async fn ladder_mid_climb_transient_keeps_4k_and_uncacheable() {
    let (llm, requests) =
        ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::RecallThenTransient);
    let run = ProbeRunner::new(llm)
        .run_detailed()
        .await
        .expect("run_detailed");
    assert_eq!(run.profile.effective_context_tokens, Some(4096));
    assert_eq!(run.profile.probed_context_floor, Some(4096));
    assert_eq!(run.host_policy_envelope()["cacheable"], false);
    assert!(
        !run.cacheable,
        "mid-climb transient must not be a 30-day cache hit"
    );
    let rec = requests.lock().expect("lock");
    let ladder = ladder_requests(&rec);
    assert_eq!(ladder.len(), 2, "4k must pass, then 8k transient");
    let t0 = request_token_estimate(ladder[0]);
    let t1 = request_token_estimate(ladder[1]);
    assert!((4096..8192).contains(&t0), "{t0}");
    assert!((8192..16384).contains(&t1), "{t1}");
}

#[tokio::test]
async fn ladder_mid_climb_auth_aborts_run_detailed() {
    let (llm, requests) =
        ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::RecallThenAuth);
    let result = ProbeRunner::new(llm).run_detailed().await;
    match result {
        Err(ProbeError::Auth(msg)) => assert_eq!(msg, "bad"),
        other => panic!("expected mid-climb Auth abort, got {other:?}"),
    }
    let rec = requests.lock().expect("lock");
    let ladder = ladder_requests(&rec);
    assert_eq!(ladder.len(), 2, "4k must pass, then 8k Auth");
}

#[tokio::test]
async fn ladder_auth_aborts_run_detailed_and_does_not_persist() {
    let (llm, requests) = ContextLadderLlm::wrap(MockLlm::new("m", "p"), LadderReply::Auth);
    let runner = ProbeRunner::new(llm);
    let result = runner.run_detailed().await;
    match result {
        Err(ProbeError::Auth(msg)) => assert_eq!(msg, "bad"),
        other => panic!("expected Auth abort, got {other:?}"),
    }

    let rec = requests.lock().expect("lock");
    assert!(
        rec.iter().any(|req| !is_ladder_request(req)),
        "non-ladder probes must still run"
    );
    assert!(
        rec.iter().any(is_ladder_request),
        "ladder chat must return Auth after non-ladder probes succeed"
    );

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe-cache.json");
    let mut cache = ProbeCache::default();
    let wrote = runner.persist(&mut cache, &path).expect("persist");
    assert!(!wrote, "ladder auth abort must not persist");
    assert!(
        !path.exists(),
        "ladder auth abort must not write a cache file"
    );
    assert!(cache.get("m", "p").is_none(), "last_run must stay empty");
}
