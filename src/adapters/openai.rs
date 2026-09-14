//! Probe adapter over `wiremux` `WireClient`.
//!
//! HTTP, SSE, catalog, and vendor error classes live in wiremux 0.3.0.
//! This module maps [`ProbeRequest`] to IR and [`wiremux::ClientError`] to
//! [`ProbeError`]. Never log `Authorization`.

#[cfg(test)]
use std::cell::{Cell, RefCell};

use futures::Stream;
use futures::StreamExt;
use serde_json::Value;
use wiremux::ir::{IrItem, IrPart, IrRequest, IrSampling, IrStreamEvent, IrTool};
use wiremux::{ClientError, ListedModel as WireListed, WireClient};
use wiremux_auth::{AnyTokenProvider, LoadOptions, StaticToken, parse_profile_str};

use crate::client::{
    CatalogPriors, ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeRequest,
    ProbeResponse, ProbeRole, ProbeStreamChunk, ProbeToolCall, ProbeUsage,
};
use crate::endpoint::is_anthropic_cloud_host;
use crate::error::ProbeError;
use crate::{finish_from_reason, strip_think_blocks};

/// OpenAI-compatible chat client backed by [`WireClient`].
#[derive(Clone)]
pub struct OpenAiCompatClient {
    wire: WireClient,
    base_url: String,
    model_id: String,
    provider: String,
    catalog: CatalogPriors,
}

impl std::fmt::Debug for OpenAiCompatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatClient")
            .field("base_url", &self.base_url)
            .field("model_id", &self.model_id)
            .field("provider", &self.provider)
            .finish()
    }
}

impl OpenAiCompatClient {
    /// Bind a wiremux client to `{base}` (OpenAI-compat `/v1` or a mock).
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model_id: impl Into<String>,
        provider: impl Into<String>,
        catalog: CatalogPriors,
    ) -> Result<Self, ProbeError> {
        let base_url = trim_slash(base_url.into());
        let model_id = model_id.into();
        let provider = provider.into();
        let wire = wire_client_for(&base_url, api_key.as_deref(), &provider)?;
        Ok(Self {
            wire,
            base_url,
            model_id,
            provider,
            catalog,
        })
    }
}

impl ProbeClient for OpenAiCompatClient {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl std::future::Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        let this = self.clone();
        async move {
            let (events, _loss) = this
                .wire
                .send(ir_request(&req))
                .await
                .map_err(map_client_error)?;
            Ok(fold_events(events))
        }
    }

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        self.wire
            .stream(ir_request(&req))
            .filter_map(|item| async move {
                match item {
                    Ok(event) => stream_chunk(event).map(Ok),
                    Err(err) => Some(Err(map_client_error(err))),
                }
            })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn provider(&self) -> &str {
        &self.provider
    }

    fn catalog(&self) -> CatalogPriors {
        self.catalog.clone()
    }
}

/// One row from `GET /models`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedModel {
    /// Provider model id.
    pub id: String,
    /// Catalog window when the host sent one.
    pub advertised_context_tokens: Option<u32>,
    /// `Some(true)` when the host lists image input. Absence stays `None`.
    pub supports_vision: Option<bool>,
}

/// Catalog hints from `/models` and, on Ollama, native `/api/show`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostCatalogHints {
    /// Catalog window when the host sent one.
    pub advertised_context_tokens: Option<u32>,
    /// `Some(true)` only when the host lists image input.
    pub supports_vision: Option<bool>,
}

/// Catalog window from a `/models` object.
pub fn advertised_context_from_model_object(model: &Value) -> Option<u32> {
    json_positive_u32(model.get("context_length"))
        .or_else(|| json_positive_u32(model.get("max_input_tokens")))
}

/// `Some(true)` when `/models` lists image input. Never `Some(false)`.
pub fn vision_from_model_object(model: &Value) -> Option<bool> {
    let modalities = model
        .pointer("/architecture/input_modalities")
        .and_then(Value::as_array)?;
    modalities
        .iter()
        .any(|m| {
            matches!(
                m.as_str().map(str::to_ascii_lowercase).as_deref(),
                Some("image" | "vision")
            )
        })
        .then_some(true)
}

/// Catalog window from Ollama `/api/show`.
pub fn advertised_context_from_ollama_show(show: &Value) -> Option<u32> {
    json_positive_u32(show.get("context_length")).or_else(|| {
        let info = show.get("model_info")?.as_object()?;
        info.iter().find_map(|(key, val)| {
            key.ends_with("context_length")
                .then(|| json_positive_u32(Some(val)))
                .flatten()
        })
    })
}

/// Vision prior from Ollama `/api/show`.
pub fn vision_from_ollama_show(show: &Value) -> Option<bool> {
    show.pointer("/details/families")
        .and_then(Value::as_array)
        .and_then(|families| {
            families
                .iter()
                .any(|f| {
                    matches!(
                        f.as_str().map(str::to_ascii_lowercase).as_deref(),
                        Some(name) if name.contains("vision") || name.contains("clip")
                    )
                })
                .then_some(true)
        })
}

/// Match a listed row to the probe model id.
pub fn advertised_context_for_model(models: &[ListedModel], model_id: &str) -> Option<u32> {
    models
        .iter()
        .find(|model| model.id == model_id)
        .and_then(|model| model.advertised_context_tokens)
}

/// `GET {base}/models`. 404 yields an empty list so the CLI can require `--model`.
pub async fn list_models(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<ListedModel>, ProbeError> {
    let client = wire_client_for(base_url, api_key, "openai-compat")?;
    let models = client.list_models().await.map_err(map_client_error)?;
    Ok(models.into_iter().map(from_wire_listed).collect())
}

/// `GET {base}/models` ids only.
pub async fn list_model_ids(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, ProbeError> {
    Ok(list_models(base_url, api_key)
        .await?
        .into_iter()
        .map(|model| model.id)
        .collect())
}

/// Catalog window for `model_id`, if `/models` listed one.
pub async fn lookup_advertised_context(
    base_url: &str,
    api_key: Option<&str>,
    model_id: &str,
) -> Result<Option<u32>, ProbeError> {
    Ok(lookup_host_catalog(base_url, api_key, model_id)
        .await?
        .advertised_context_tokens)
}

/// `/models` row plus Ollama `/api/show` when the OpenAI-compat list
/// omitted a window or vision flag.
pub async fn lookup_host_catalog(
    base_url: &str,
    api_key: Option<&str>,
    model_id: &str,
) -> Result<HostCatalogHints, ProbeError> {
    let models = list_models(base_url, api_key).await?;
    let listed = models.iter().find(|model| model.id == model_id);
    Ok(HostCatalogHints {
        advertised_context_tokens: listed.and_then(|model| model.advertised_context_tokens),
        supports_vision: listed.and_then(|model| model.supports_vision),
    })
}

/// Merge a CLI `--advertised-context` flag over a catalog lookup.
pub fn merge_advertised_context(
    advertised_flag: Option<u32>,
    catalog: Result<Option<u32>, ProbeError>,
) -> Option<u32> {
    advertised_flag.or_else(|| catalog.ok().flatten())
}

/// CLI vision flag wins over a catalog prior.
pub fn merge_vision_catalog(vision_flag: Option<bool>, catalog: Option<bool>) -> Option<bool> {
    vision_flag.or(catalog)
}

/// Resolve advertised context: flag, else catalog, else none.
pub async fn resolve_advertised_context(
    advertised_flag: Option<u32>,
    base_url: &str,
    api_key: Option<&str>,
    model_id: &str,
) -> Option<u32> {
    resolve_host_catalog(advertised_flag, None, base_url, api_key, model_id)
        .await
        .advertised_context_tokens
}

/// Resolve catalog hints. Flags win. Catalog errors stay `None`.
pub async fn resolve_host_catalog(
    advertised_flag: Option<u32>,
    vision_flag: Option<bool>,
    base_url: &str,
    api_key: Option<&str>,
    model_id: &str,
) -> HostCatalogHints {
    if advertised_flag.is_some() && vision_flag.is_some() {
        return HostCatalogHints {
            advertised_context_tokens: advertised_flag,
            supports_vision: vision_flag,
        };
    }
    #[cfg(test)]
    {
        CATALOG_LOOKUPS.with(|lookups| lookups.borrow_mut().push(base_url.to_owned()));
        if CATALOG_SKIP_HTTP.with(|flag| flag.get()) {
            return HostCatalogHints {
                advertised_context_tokens: advertised_flag,
                supports_vision: vision_flag,
            };
        }
    }
    let catalog = lookup_host_catalog(base_url, api_key, model_id)
        .await
        .unwrap_or_default();
    HostCatalogHints {
        advertised_context_tokens: merge_advertised_context(
            advertised_flag,
            Ok(catalog.advertised_context_tokens),
        ),
        supports_vision: merge_vision_catalog(vision_flag, catalog.supports_vision),
    }
}

#[cfg(test)]
thread_local! {
    static CATALOG_SKIP_HTTP: Cell<bool> = const { Cell::new(false) };
    static CATALOG_LOOKUPS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Test hook: skip catalog HTTP and record attempted bases.
#[cfg(test)]
pub(crate) struct CatalogSkipHttp;

#[cfg(test)]
impl CatalogSkipHttp {
    pub(crate) fn enable() -> Self {
        CATALOG_SKIP_HTTP.with(|flag| flag.set(true));
        CATALOG_LOOKUPS.with(|lookups| lookups.borrow_mut().clear());
        Self
    }
}

#[cfg(test)]
impl Drop for CatalogSkipHttp {
    fn drop(&mut self) {
        CATALOG_SKIP_HTTP.with(|flag| flag.set(false));
    }
}

/// Bases `resolve_host_catalog` would have called (tests).
#[cfg(test)]
pub(crate) fn take_catalog_lookups() -> Vec<String> {
    CATALOG_LOOKUPS.with(|lookups| lookups.replace(Vec::new()))
}

fn wire_client_for(
    base_url: &str,
    api_key: Option<&str>,
    provider: &str,
) -> Result<WireClient, ProbeError> {
    if is_anthropic_cloud_host(base_url) {
        return anthropic_client(api_key);
    }
    let (origin, chat_path) = split_compat_base(base_url);
    let scheme = if api_key.is_some_and(|k| !k.trim().is_empty()) {
        "bearer"
    } else {
        "none"
    };
    let toml = format!(
        "schema_version = 1\nid = \"canact-compat\"\nwire = \"chat-completions\"\nbase_url = \"{}\"\nchat_path = \"{chat_path}\"\nauth_scheme = \"{scheme}\"\n",
        escape_toml_basic(&origin),
    );
    let profile = parse_profile_str(&toml).map_err(|err| ProbeError::Internal(err.to_string()))?;
    let token = StaticToken::new(api_key.unwrap_or(""));
    let _ = provider;
    WireClient::from_resolved(profile, AnyTokenProvider::Static(token)).map_err(map_client_error)
}

fn anthropic_client(api_key: Option<&str>) -> Result<WireClient, ProbeError> {
    let opts = LoadOptions {
        include_user_config: true,
        include_shipped: true,
        ..LoadOptions::default()
    };
    let oat = api_key.is_some_and(|k| k.starts_with("sk-ant-oat"));
    let id = if oat { "anthropic-oauth" } else { "anthropic" };
    let profile =
        wiremux::load_profile(id, &opts).map_err(|err| ProbeError::Internal(err.to_string()))?;
    let provider = match api_key {
        Some(key) if !key.trim().is_empty() => AnyTokenProvider::Static(StaticToken::new(key)),
        _ => wiremux_auth::provider_for_profile_opts(id, &opts)
            .map_err(|err| ProbeError::Auth(err.to_string()))?,
    };
    WireClient::from_resolved(profile, provider).map_err(map_client_error)
}

fn split_compat_base(base_url: &str) -> (String, &'static str) {
    let trimmed = trim_slash(base_url.to_owned());
    if let Some(root) = trimmed.strip_suffix("/v1") {
        (root.to_owned(), "/v1/chat/completions")
    } else {
        (trimmed, "/chat/completions")
    }
}

fn escape_toml_basic(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn trim_slash(url: String) -> String {
    url.trim_end_matches('/').to_owned()
}

fn from_wire_listed(model: WireListed) -> ListedModel {
    ListedModel {
        id: model.id,
        advertised_context_tokens: model.context_tokens,
        // Catalog may only promote true. Wiremux `false` (no image
        // modality) must not look like the user passed `--no-vision`.
        supports_vision: model.vision.filter(|v| *v),
    }
}

fn ir_request(req: &ProbeRequest) -> IrRequest {
    let mut items = Vec::new();
    for msg in &req.messages {
        match msg.role {
            ProbeRole::System => items.push(IrItem::System {
                text: content_text(&msg.content),
            }),
            ProbeRole::User => items.push(IrItem::User {
                parts: content_parts(&msg.content),
            }),
            ProbeRole::Assistant => {
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        items.push(IrItem::FunctionCall {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: serde_json::to_string(&call.arguments)
                                .unwrap_or_else(|_| "{}".into()),
                            thought_signature: None,
                        });
                    }
                }
                let parts = content_parts(&msg.content);
                if !parts.is_empty() {
                    items.push(IrItem::Assistant { parts });
                }
            }
            ProbeRole::Tool => items.push(IrItem::FunctionOutput {
                call_id: msg.tool_call_id.clone().unwrap_or_default(),
                output: content_text(&msg.content),
            }),
        }
    }
    IrRequest {
        model: req.model.clone(),
        items,
        tools: req
            .tools
            .iter()
            .map(|tool| IrTool::Function {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect(),
        sampling: IrSampling {
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            ..IrSampling::default()
        },
    }
}

fn content_text(content: &ProbeContent) -> String {
    match content {
        ProbeContent::Text(text) => text.clone(),
        ProbeContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ProbeContentPart::Text { text } => Some(text.as_str()),
                ProbeContentPart::ImageBase64 { .. } => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn content_parts(content: &ProbeContent) -> Vec<IrPart> {
    match content {
        ProbeContent::Text(text) if text.is_empty() => Vec::new(),
        ProbeContent::Text(text) => vec![IrPart::Text(text.clone())],
        ProbeContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ProbeContentPart::Text { text } => IrPart::Text(text.clone()),
                ProbeContentPart::ImageBase64 { media_type, data } => IrPart::ImageBase64 {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
            })
            .collect(),
    }
}

fn fold_events(events: Vec<IrStreamEvent>) -> ProbeResponse {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut current: Option<(String, String, String)> = None;
    let mut finish = ProbeFinish::Stop;
    let mut usage = None;
    for event in events {
        match event {
            IrStreamEvent::TextDelta { text: delta } => text.push_str(&delta),
            IrStreamEvent::ToolCallStart { id, name, .. } => {
                if let Some(call) = current.take() {
                    push_call(&mut tool_calls, call);
                }
                current = Some((id, name, String::new()));
            }
            IrStreamEvent::ToolCallArgDelta { delta } => {
                if let Some((_, _, args)) = &mut current {
                    args.push_str(&delta);
                }
            }
            IrStreamEvent::ToolCallEnd => {
                if let Some(call) = current.take() {
                    push_call(&mut tool_calls, call);
                }
            }
            IrStreamEvent::FinishReason { reason } => {
                finish = finish_from_reason(&reason);
            }
            IrStreamEvent::Usage {
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                ..
            } => {
                usage = Some(ProbeUsage {
                    prompt_tokens: Some(prompt_tokens),
                    completion_tokens: Some(completion_tokens),
                    reasoning_tokens: Some(reasoning_tokens),
                });
            }
            IrStreamEvent::ReasoningDelta { .. }
            | IrStreamEvent::ReasoningSignature { .. }
            | IrStreamEvent::Protocol { .. }
            | IrStreamEvent::Unknown { .. }
            | IrStreamEvent::Done => {}
        }
    }
    if let Some(call) = current {
        push_call(&mut tool_calls, call);
    }
    if !tool_calls.is_empty() && matches!(finish, ProbeFinish::Stop) {
        finish = ProbeFinish::ToolCalls;
    }
    let text = strip_think_blocks(&text);
    ProbeResponse {
        text,
        tool_calls,
        finish,
        usage,
    }
}

fn push_call(out: &mut Vec<ProbeToolCall>, (id, name, args): (String, String, String)) {
    if name.trim().is_empty() {
        return;
    }
    let arguments = serde_json::from_str(&args)
        .ok()
        .and_then(|value: Value| value.as_object().cloned())
        .unwrap_or_default();
    out.push(ProbeToolCall {
        id,
        name,
        arguments,
    });
}

fn stream_chunk(event: IrStreamEvent) -> Option<ProbeStreamChunk> {
    match event {
        IrStreamEvent::TextDelta { text } if !text.is_empty() => {
            Some(ProbeStreamChunk::TextDelta { text })
        }
        IrStreamEvent::ToolCallStart { id, name, .. } => {
            Some(ProbeStreamChunk::ToolCallStart { id, name })
        }
        IrStreamEvent::ToolCallArgDelta { delta } => {
            Some(ProbeStreamChunk::ToolCallArgDelta { delta })
        }
        IrStreamEvent::ToolCallEnd => Some(ProbeStreamChunk::ToolCallEnd),
        IrStreamEvent::FinishReason { reason } => Some(ProbeStreamChunk::Finished {
            finish: finish_from_reason(&reason),
        }),
        _ => None,
    }
}

fn map_client_error(err: ClientError) -> ProbeError {
    match err {
        ClientError::Auth { message, .. } => ProbeError::Auth(redact_secrets(&message)),
        ClientError::NotFound { message, .. } => ProbeError::NotFound(redact_secrets(&message)),
        ClientError::RateLimit { retry_after, .. } => ProbeError::RateLimit { retry_after },
        ClientError::Transient { message, .. } => {
            let message = redact_secrets(&message);
            if is_connect_message(&message) && !message.starts_with("failed to connect:") {
                ProbeError::Transient(format!("failed to connect: {message}"))
            } else {
                ProbeError::Transient(message)
            }
        }
        ClientError::Vendor { message, .. } => ProbeError::Llm(redact_secrets(&message)),
        ClientError::Map(err) => ProbeError::Llm(redact_secrets(&err.to_string())),
        ClientError::Transport(message) => ProbeError::Transient(redact_secrets(&message)),
    }
}

/// Strip Bearer tokens, `sk-` / `gsk_` / `ghp_` / `xai-` keys, and values after Authorization / api-key / api_key.
fn redact_secrets(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let lower = input.to_ascii_lowercase();
    let mut i = 0;
    while i < input.len() {
        if let Some(n) = secret_header_len(&lower, input, i) {
            out.push_str(&input[i..i + n]);
            i += n;
            i = copy_separators(input, i, &mut out);
            if starts_at(&lower, i, "bearer") && is_ascii_word_at(input, i, 6) {
                continue;
            }
            if i < input.len() {
                out.push_str("[REDACTED]");
                i = skip_token(input, i);
            }
            continue;
        }
        if starts_at(&lower, i, "bearer") && is_ascii_word_at(input, i, 6) {
            out.push_str(&input[i..i + 6]);
            i += 6;
            i = copy_whitespace(input, i, &mut out);
            if i < input.len() {
                out.push_str("[REDACTED]");
                i = skip_token(input, i);
            }
            continue;
        }
        if input[i..].starts_with("sk-") {
            out.push_str("sk-[REDACTED]");
            i = skip_secret_key(input, i + 3);
            continue;
        }
        if input[i..].starts_with("gsk_") {
            out.push_str("gsk_[REDACTED]");
            i = skip_secret_key(input, i + 4);
            continue;
        }
        if input[i..].starts_with("ghp_") {
            out.push_str("ghp_[REDACTED]");
            i = skip_secret_key(input, i + 4);
            continue;
        }
        if input[i..].starts_with("xai-") {
            out.push_str("xai-[REDACTED]");
            i = skip_secret_key(input, i + 4);
            continue;
        }
        let ch = input[i..].chars().next().expect("i is a char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn secret_header_len(lower: &str, input: &str, i: usize) -> Option<usize> {
    if starts_at(lower, i, "x-api-key") && is_ascii_word_at(input, i, 9) {
        Some(9)
    } else if starts_at(lower, i, "authorization") && is_ascii_word_at(input, i, 13) {
        Some(13)
    } else if (starts_at(lower, i, "api-key") || starts_at(lower, i, "api_key"))
        && is_ascii_word_at(input, i, 7)
    {
        Some(7)
    } else if starts_at(lower, i, "apikey") && is_ascii_word_at(input, i, 6) {
        Some(6)
    } else {
        None
    }
}

fn starts_at(lower: &str, i: usize, needle: &str) -> bool {
    lower.get(i..).is_some_and(|s| s.starts_with(needle))
}

fn is_ascii_word_at(input: &str, i: usize, len: usize) -> bool {
    let before_ok = i == 0 || !input.as_bytes()[i - 1].is_ascii_alphanumeric();
    let after = i + len;
    let after_ok = input
        .as_bytes()
        .get(after)
        .is_none_or(|b| !b.is_ascii_alphanumeric());
    before_ok && after_ok
}

fn copy_separators(input: &str, mut i: usize, out: &mut String) -> usize {
    while i < input.len() {
        let ch = input[i..].chars().next().expect("i is a char boundary");
        if ch == ':'
            || ch == '='
            || ch == '\\'
            || ch == '"'
            || ch == '\''
            || ch.is_ascii_whitespace()
        {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    i
}

fn copy_whitespace(input: &str, mut i: usize, out: &mut String) -> usize {
    while i < input.len() {
        let ch = input[i..].chars().next().expect("i is a char boundary");
        if ch.is_ascii_whitespace() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    i
}

fn skip_token(input: &str, mut i: usize) -> usize {
    while i < input.len() {
        let ch = input[i..].chars().next().expect("i is a char boundary");
        if ch.is_ascii_whitespace() || matches!(ch, '"' | '\'' | ',' | '}' | ']' | '&') {
            break;
        }
        i += ch.len_utf8();
    }
    i
}

fn skip_secret_key(input: &str, mut i: usize) -> usize {
    while i < input.len() {
        let ch = input[i..].chars().next().expect("i is a char boundary");
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    i
}

fn is_connect_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    if lower.contains("connection refused")
        || lower.contains("connect error")
        || lower.contains("error trying to connect")
        || lower.contains("tcp connect error")
        || lower.contains("dns error")
    {
        return true;
    }
    // Wiremux closed-port is often only "error sending request for url (...)".
    // A hung-after-accept read timeout uses the same prefix plus "timed out".
    lower.contains("error sending request")
        && !lower.contains("timed out")
        && !lower.contains("timeout")
}

fn json_positive_u32(value: Option<&Value>) -> Option<u32> {
    let n = match value? {
        Value::Number(num) => num.as_u64().or_else(|| {
            num.as_f64()
                .filter(|f| f.is_finite() && *f > 0.0)
                .map(|f| f as u64)
        })?,
        Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    u32::try_from(n).ok().filter(|n| *n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapabilityLevel;
    use crate::client::ProbeMessage;
    use crate::client::ProbeTool;
    use crate::runner::resolve_probe;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    const SECRET: &str = "sk-test-secret";

    fn empty_req() -> ProbeRequest {
        ProbeRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            model: "m".into(),
            temperature: None,
            max_tokens: None,
        }
    }

    fn spawn_http(status: u16, reason: &'static str, body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = read_http(&mut stream);
                let head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        format!("http://{addr}/v1")
    }

    fn spawn_http_seq(replies: Vec<(u16, Vec<u8>)>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_thread = Arc::clone(&seen);
        thread::spawn(move || {
            for (status, body) in replies {
                if let Ok((mut stream, _)) = listener.accept() {
                    let req = read_http(&mut stream);
                    if let Ok(mut log) = seen_thread.lock() {
                        log.push(req.lines().next().unwrap_or("").to_owned());
                    }
                    let head = format!(
                        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body);
                }
            }
        });
        (format!("http://{addr}/v1"), seen)
    }

    fn read_http(stream: &mut std::net::TcpStream) -> String {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(_) => break,
            }
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn client(base: &str) -> OpenAiCompatClient {
        OpenAiCompatClient::new(
            base,
            Some(SECRET.into()),
            "m",
            "test",
            CatalogPriors::default(),
        )
        .expect("client")
    }

    fn chat_ok(text: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 2 }
        }))
        .expect("json")
    }

    #[tokio::test]
    async fn chat_closed_port_is_connect_abort() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(listener);
        let err = client(&format!("http://{addr}/v1"))
            .chat(empty_req())
            .await
            .expect_err("closed port");
        match &err {
            ProbeError::Transient(msg) => {
                assert!(
                    msg.starts_with("failed to connect:")
                        || msg.to_ascii_lowercase().contains("connection refused"),
                    "connect refuse must stay Transient: {msg}"
                );
            }
            other => panic!("expected Transient connect, got {other:?}"),
        }
        match resolve_probe(Err(err), "tool_calling") {
            Err(ProbeError::Transient(_)) => {}
            other => panic!("expected Transient abort, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_401_is_auth() {
        let base = spawn_http(
            401,
            "Unauthorized",
            br#"{"error":{"message":"bad key"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("401");
        match err {
            ProbeError::Auth(msg) => assert!(msg.to_ascii_lowercase().contains("bad key"), "{msg}"),
            other => panic!("expected Auth, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_400_unknown_model_is_not_found() {
        let base = spawn_http(
            400,
            "Bad Request",
            br#"{"error":{"message":"The model `foo` does not exist"}}"#.to_vec(),
        );
        match client(&base).chat(empty_req()).await.expect_err("400") {
            ProbeError::NotFound(msg) => {
                assert!(msg.contains("does not exist"), "{msg}");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_200_error_overload_is_transient() {
        let base = spawn_http(
            200,
            "OK",
            br#"{"error":{"message":"Upstream error: Service temporarily overloaded","code":502}}"#
                .to_vec(),
        );
        match client(&base).chat(empty_req()).await.expect_err("overload") {
            ProbeError::Transient(_) => {}
            other => panic!("expected Transient, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_parses_text_and_usage() {
        let base = spawn_http(200, "OK", chat_ok("hello"));
        let resp = client(&base).chat(empty_req()).await.expect("ok");
        assert_eq!(resp.text, "hello");
        assert_eq!(resp.finish, ProbeFinish::Stop);
        let usage = resp.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, Some(2));
    }

    #[tokio::test]
    async fn chat_ignores_reasoning_delta() {
        let body = serde_json::to_vec(&serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "visible",
                    "reasoning_content": "hidden chain"
                },
                "finish_reason": "stop"
            }]
        }))
        .expect("json");
        let base = spawn_http(200, "OK", body);
        let resp = client(&base).chat(empty_req()).await.expect("ok");
        assert_eq!(resp.text, "visible");
        assert!(!resp.text.contains("hidden"));
    }

    #[tokio::test]
    async fn chat_strips_think_tags() {
        let base = spawn_http(200, "OK", chat_ok("<think>hidden chain</think>Paris"));
        let resp = client(&base).chat(empty_req()).await.expect("ok");
        assert_eq!(resp.text, "Paris");
        assert!(!resp.text.contains("hidden"));

        let base = spawn_http(200, "OK", chat_ok("<think>partial"));
        let resp = client(&base).chat(empty_req()).await.expect("ok");
        assert_eq!(resp.text, "");
    }

    #[test]
    fn send_timeout_is_not_connect_abort() {
        let err = map_client_error(ClientError::Transient {
            status: None,
            message: "error sending request for url (http://127.0.0.1:11434/v1/chat/completions): operation timed out".into(),
        });
        match &err {
            ProbeError::Transient(msg) => {
                assert!(
                    !msg.starts_with("failed to connect:"),
                    "send timeout must not look like connect abort: {msg}"
                );
            }
            other => panic!("expected Transient, got {other:?}"),
        }
        let (result, cacheable) =
            resolve_probe(Err(err), "tool_calling").expect("send timeout stays scored");
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert!(!cacheable, "timeout must not persist");

        let connect = map_client_error(ClientError::Transient {
            status: None,
            message: "error trying to connect: tcp connect error: Connection refused".into(),
        });
        match &connect {
            ProbeError::Transient(msg) => {
                assert!(
                    msg.starts_with("failed to connect:"),
                    "true connect must stay prefixed: {msg}"
                );
            }
            other => panic!("expected Transient connect, got {other:?}"),
        }
        match resolve_probe(Err(connect), "tool_calling") {
            Err(ProbeError::Transient(msg)) => {
                assert!(msg.contains("failed to connect:"), "{msg}");
            }
            other => panic!("expected Transient abort, got {other:?}"),
        }

        let connect_timeout = map_client_error(ClientError::Transient {
            status: None,
            message: "error sending request for url (http://192.0.2.1:11434/v1/chat/completions): error trying to connect: tcp connect error: Operation timed out".into(),
        });
        match &connect_timeout {
            ProbeError::Transient(msg) => {
                assert!(
                    msg.starts_with("failed to connect:"),
                    "SYN timeout must still abort: {msg}"
                );
            }
            other => panic!("expected Transient connect timeout, got {other:?}"),
        }
        match resolve_probe(Err(connect_timeout), "tool_calling") {
            Err(ProbeError::Transient(msg)) => {
                assert!(msg.contains("failed to connect:"), "{msg}");
            }
            other => panic!("expected connect-timeout abort, got {other:?}"),
        }

        let closed = map_client_error(ClientError::Transient {
            status: None,
            message: "error sending request for url (http://127.0.0.1:11434/v1/chat/completions)"
                .into(),
        });
        match &closed {
            ProbeError::Transient(msg) => {
                assert!(
                    msg.starts_with("failed to connect:"),
                    "closed port without refused text must still abort: {msg}"
                );
            }
            other => panic!("expected Transient closed port, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_models_404_is_empty() {
        let base = spawn_http(404, "Not Found", b"missing".to_vec());
        let ids = list_model_ids(&base, Some(SECRET)).await.expect("404");
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn list_models_401_is_auth() {
        let base = spawn_http(
            401,
            "Unauthorized",
            br#"{"error":{"message":"bad key"}}"#.to_vec(),
        );
        match list_models(&base, Some(SECRET)).await {
            Ok(models) => panic!("401 must stay Auth, not empty catalog: {models:?}"),
            Err(ProbeError::Auth(_)) => {}
            other => panic!("expected Auth, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_models_keeps_context_length() {
        let body = serde_json::to_vec(&serde_json::json!({
            "data": [{
                "id": "grok-4",
                "context_length": 1_000_000
            }]
        }))
        .expect("json");
        let base = spawn_http(200, "OK", body);
        let models = list_models(&base, Some(SECRET)).await.expect("200");
        assert_eq!(
            advertised_context_for_model(&models, "grok-4"),
            Some(1_000_000)
        );
    }

    #[tokio::test]
    async fn lookup_host_catalog_skips_show_off_11434() {
        let body = serde_json::to_vec(&serde_json::json!({
            "data": [{ "id": "llama3.2:3b" }]
        }))
        .expect("json");
        let (base, seen) = spawn_http_seq(vec![(200, body)]);
        let hints = lookup_host_catalog(&base, None, "llama3.2:3b")
            .await
            .expect("models");
        assert_eq!(hints.advertised_context_tokens, None);
        let seen = seen.lock().expect("seen").clone();
        assert_eq!(
            seen.len(),
            1,
            "must not POST /api/show off :11434, got {seen:?}"
        );
        assert!(
            seen[0].contains("GET") && seen[0].contains("/models"),
            "first request must be GET /models, got {}",
            seen[0]
        );
    }

    #[test]
    fn advertised_context_from_anthropic_max_input() {
        let model = serde_json::json!({"id": "claude", "max_input_tokens": 200_000});
        assert_eq!(advertised_context_from_model_object(&model), Some(200_000));
    }

    #[test]
    fn merge_advertised_catalog_error_stays_none() {
        assert_eq!(
            merge_advertised_context(None, Err(ProbeError::Auth("x".into()))),
            None
        );
        assert_eq!(
            merge_advertised_context(Some(8), Err(ProbeError::Auth("x".into()))),
            Some(8)
        );
    }

    #[tokio::test]
    async fn stream_emits_text_and_finish() {
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec();
        let base = spawn_http(200, "OK", body);
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, ProbeStreamChunk::TextDelta { text } if text == "hi")),
            "missing text: {chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::Finished {
                    finish: ProbeFinish::Stop
                }
            )),
            "missing finish: {chunks:?}"
        );
    }

    #[test]
    fn ir_request_maps_image_part() {
        let req = ProbeRequest {
            messages: vec![ProbeMessage {
                role: ProbeRole::User,
                content: ProbeContent::Parts(vec![ProbeContentPart::ImageBase64 {
                    media_type: "image/png".into(),
                    data: "abc".into(),
                }]),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: vec![ProbeTool {
                name: "read_file".into(),
                description: "read".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            model: "m".into(),
            temperature: Some(0.2),
            max_tokens: Some(64),
        };
        let ir = ir_request(&req);
        assert_eq!(ir.model, "m");
        assert_eq!(ir.sampling.max_tokens, Some(64));
        assert!(matches!(
            ir.items.first(),
            Some(IrItem::User { parts }) if matches!(parts.first(), Some(IrPart::ImageBase64 { .. }))
        ));
    }

    #[test]
    fn from_wire_listed_drops_catalog_false_vision() {
        let listed = from_wire_listed(WireListed {
            id: "llama3.2:3b".into(),
            context_tokens: Some(131_072),
            vision: Some(false),
        });
        assert_eq!(listed.supports_vision, None);
        let listed = from_wire_listed(WireListed {
            id: "llava".into(),
            context_tokens: None,
            vision: Some(true),
        });
        assert_eq!(listed.supports_vision, Some(true));
    }
}
