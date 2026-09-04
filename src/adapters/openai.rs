//! Slim OpenAI-compatible HTTP + SSE adapter.
//!
//! Do not copy `bline-llm`. Never log `Authorization`.

use std::collections::HashMap;
use std::time::Duration;

use futures::Stream;
use futures::channel::mpsc::UnboundedSender;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value};

use crate::client::{
    CatalogPriors, ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeMessage,
    ProbeRequest, ProbeResponse, ProbeRole, ProbeStreamChunk, ProbeTool, ProbeToolCall, ProbeUsage,
};
use crate::error::ProbeError;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// OpenAI-compatible chat completions client (Ollama / vLLM / LM Studio / cloud).
#[derive(Clone)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
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
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}

impl OpenAiCompatClient {
    /// Build a client for `{base}/chat/completions` and `{base}/models`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model_id: impl Into<String>,
        provider: impl Into<String>,
        catalog: CatalogPriors,
    ) -> Result<Self, ProbeError> {
        Ok(Self {
            http: default_http_client()?,
            base_url: trim_slash(base_url.into()),
            api_key,
            model_id: model_id.into(),
            provider: provider.into(),
            catalog,
        })
    }

    fn endpoint(&self, suffix: &str) -> String {
        join_url(&self.base_url, suffix)
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) if !key.is_empty() => {
                builder.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"))
            }
            _ => builder,
        }
    }

    async fn stream_into(
        &self,
        req: ProbeRequest,
        tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
    ) -> Result<(), ProbeError> {
        let url = self.endpoint("chat/completions");
        let body = chat_body(&req, true);
        let resp = self
            .apply_auth(
                self.http
                    .post(url)
                    .header(reqwest::header::ACCEPT, "text/event-stream")
                    .json(&body),
            )
            .send()
            .await
            .map_err(map_transport)?;
        let resp = ensure_success(resp).await?;

        use futures::StreamExt;
        let mut bytes = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut received = 0usize;
        let mut open_tools: HashMap<u64, OpenTool> = HashMap::new();
        let mut think = ThinkFilter::default();
        while let Some(item) = bytes.next().await {
            let chunk = item.map_err(map_transport)?;
            accumulate_stream_bytes(&mut received, chunk.len())?;
            buf.extend_from_slice(&chunk);
            for line in drain_complete_sse_lines(&mut buf) {
                if let Err(err) = emit_sse_line(&line, &mut open_tools, tx, &mut think) {
                    let _ = tx.unbounded_send(Err(err));
                    return Ok(());
                }
            }
        }
        if !buf.is_empty() {
            let line = take_sse_tail(&buf);
            if sse_tail_is_truncated(&line) {
                let _ = tx.unbounded_send(Err(ProbeError::Transient(
                    "truncated stream: incomplete SSE tail".into(),
                )));
                return Ok(());
            }
            if let Err(err) = emit_sse_line(&line, &mut open_tools, tx, &mut think) {
                let _ = tx.unbounded_send(Err(err));
                return Ok(());
            }
        }
        let leftover = think.flush();
        if !leftover.is_empty() {
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::TextDelta { text: leftover }));
        }
        end_open_tools(&mut open_tools, tx);
        Ok(())
    }
}

impl ProbeClient for OpenAiCompatClient {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl std::future::Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        let this = self.clone();
        async move {
            let url = this.endpoint("chat/completions");
            let body = chat_body(&req, false);
            let resp = this
                .apply_auth(this.http.post(url).json(&body))
                .send()
                .await
                .map_err(map_transport)?;
            let resp = ensure_success(resp).await?;
            let raw = read_success_body(resp).await?;
            let value: Value = serde_json::from_slice(&raw)?;
            if let Some(err) = value.get("error") {
                return Err(ProbeError::Llm(error_message(err)));
            }
            parse_chat_response(&value)
        }
    }

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let this = self.clone();
        tokio::spawn(async move {
            if let Err(err) = this.stream_into(req, &tx).await {
                let _ = tx.unbounded_send(Err(err));
            }
        });
        rx
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

/// `GET {base}/models`. 404 yields an empty list so the CLI can require `--model`.
pub async fn list_model_ids(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, ProbeError> {
    let http = default_http_client()?;
    let url = join_url(&trim_slash(base_url.to_owned()), "models");
    let mut builder = http.get(url);
    if let Some(key) = api_key {
        if !key.is_empty() {
            builder = builder.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"));
        }
    }
    let resp = builder.send().await.map_err(map_transport)?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(Vec::new());
    }
    let resp = ensure_success(resp).await?;
    let raw = read_success_body(resp).await?;
    let value: Value = serde_json::from_slice(&raw)?;
    if let Some(err) = value.get("error") {
        return Err(ProbeError::Llm(error_message(err)));
    }
    let ids = value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    Ok(ids)
}

fn default_http_client() -> Result<reqwest::Client, ProbeError> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| ProbeError::Internal(e.to_string()))
}

fn trim_slash(url: String) -> String {
    url.trim_end_matches('/').to_owned()
}

fn join_url(base: &str, suffix: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

fn map_transport(err: reqwest::Error) -> ProbeError {
    let msg = redact_secrets(&err.to_string());
    if err.is_connect() {
        ProbeError::Transient(format!("failed to connect: {msg}"))
    } else {
        ProbeError::Transient(msg)
    }
}

async fn ensure_success(resp: reqwest::Response) -> Result<reqwest::Response, ProbeError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let retry_after = parse_retry_after(resp.headers());
    let code = status.as_u16();
    let body = read_capped_text(resp, MAX_ERROR_BODY_BYTES).await;
    Err(map_status(code, retry_after, &body))
}

fn push_response_chunk(buf: &mut Vec<u8>, chunk: &[u8], max: usize) -> Result<(), ProbeError> {
    if buf.len().saturating_add(chunk.len()) > max {
        return Err(ProbeError::Transient("response too large".into()));
    }
    buf.extend_from_slice(chunk);
    Ok(())
}

async fn read_success_body(resp: reqwest::Response) -> Result<Vec<u8>, ProbeError> {
    use futures::StreamExt;
    if resp
        .content_length()
        .is_some_and(|n| n > MAX_RESPONSE_BYTES as u64)
    {
        return Err(ProbeError::Transient("response too large".into()));
    }
    let mut buf = Vec::new();
    let mut bytes = resp.bytes_stream();
    while let Some(item) = bytes.next().await {
        let chunk = item.map_err(map_transport)?;
        push_response_chunk(&mut buf, &chunk, MAX_RESPONSE_BYTES)?;
    }
    Ok(buf)
}

async fn read_capped_text(resp: reqwest::Response, max: usize) -> String {
    use futures::StreamExt;
    let mut buf = Vec::new();
    let mut bytes = resp.bytes_stream();
    while let Some(item) = bytes.next().await {
        let Ok(chunk) = item else {
            break;
        };
        let room = max.saturating_sub(buf.len());
        if room == 0 {
            break;
        }
        if chunk.len() > room {
            buf.extend_from_slice(&chunk[..room]);
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn parse_retry_after(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

fn map_status(code: u16, retry_after: Option<u64>, body: &str) -> ProbeError {
    match code {
        401 => ProbeError::Auth(body_or_status(code, body)),
        403 if body_looks_like_auth(body) => ProbeError::Auth(body_or_status(code, body)),
        403 => ProbeError::Llm(body_or_status(code, body)),
        429 => ProbeError::RateLimit { retry_after },
        408 | 500..=599 => ProbeError::Transient(body_or_status(code, body)),
        _ => ProbeError::Llm(body_or_status(code, body)),
    }
}

fn body_looks_like_auth(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    b.contains("api key")
        || b.contains("api_key")
        || b.contains("unauthorized")
        || b.contains("invalid key")
        || b.contains("incorrect api")
}

fn body_or_status(code: u16, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        format!("HTTP {code}")
    } else {
        redact_secrets(trimmed)
    }
}

fn error_message(err: &Value) -> String {
    let raw = err
        .get("message")
        .and_then(|m| m.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| err.to_string());
    redact_secrets(&raw)
}

/// Strip Bearer tokens, `sk-` keys, and values after Authorization / api-key / api_key.
fn redact_secrets(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let lower = input.to_ascii_lowercase();
    let mut i = 0;
    while i < input.len() {
        if let Some(n) = secret_header_len(&lower, input, i) {
            out.push_str(&input[i..i + n]);
            i += n;
            i = copy_separators(input, i, &mut out);
            // Leave "Bearer" for the dedicated handler so its token is stripped.
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
        let ch = input[i..].chars().next().expect("i is a char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Longest match first so `x-api-key` is not treated as `api-key`.
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
        // camelCase apiKey lowercases to apikey, not api_key / api-key.
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

fn chat_body(req: &ProbeRequest, stream: bool) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));
    body.insert(
        "messages".into(),
        Value::Array(req.messages.iter().map(message_json).collect()),
    );
    if !req.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(req.tools.iter().map(tool_json).collect()),
        );
    }
    if let Some(t) = req.temperature {
        if !reasoning_chat_model(&req.model) {
            body.insert("temperature".into(), Value::from(t));
        }
    }
    if let Some(m) = req.max_tokens {
        let key = if reasoning_chat_model(&req.model) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        body.insert(key.into(), Value::from(m));
    }
    if stream {
        body.insert("stream".into(), Value::Bool(true));
    }
    Value::Object(body)
}

/// o-series and GPT-5 chat-completions reject `max_tokens` and often
/// reject `temperature`. Use `max_completion_tokens` and omit temperature.
fn reasoning_chat_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    let id = m.rsplit('/').next().unwrap_or(&m);
    reasoning_id_prefix(id, "o1")
        || reasoning_id_prefix(id, "o3")
        || reasoning_id_prefix(id, "o4")
        || id.starts_with("gpt-5")
}

fn reasoning_id_prefix(id: &str, prefix: &str) -> bool {
    id == prefix || id.starts_with(&format!("{prefix}-"))
}

fn message_json(msg: &ProbeMessage) -> Value {
    let mut obj = Map::new();
    obj.insert("role".into(), Value::String(role_str(msg.role).to_owned()));
    let empty_text = matches!(&msg.content, ProbeContent::Text(t) if t.is_empty());
    if empty_text && msg.tool_calls.is_some() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert("content".into(), content_json(&msg.content));
    }
    if let Some(calls) = &msg.tool_calls {
        obj.insert(
            "tool_calls".into(),
            Value::Array(calls.iter().map(tool_call_json).collect()),
        );
    }
    if let Some(id) = &msg.tool_call_id {
        obj.insert("tool_call_id".into(), Value::String(id.clone()));
    }
    Value::Object(obj)
}

fn role_str(role: ProbeRole) -> &'static str {
    match role {
        ProbeRole::System => "system",
        ProbeRole::User => "user",
        ProbeRole::Assistant => "assistant",
        ProbeRole::Tool => "tool",
    }
}

fn content_json(content: &ProbeContent) -> Value {
    match content {
        ProbeContent::Text(text) => Value::String(text.clone()),
        ProbeContent::Parts(parts) => Value::Array(parts.iter().map(part_json).collect()),
    }
}

fn part_json(part: &ProbeContentPart) -> Value {
    match part {
        ProbeContentPart::Text { text } => {
            serde_json::json!({ "type": "text", "text": text })
        }
        ProbeContentPart::ImageBase64 { media_type, data } => {
            let url = format!("data:{media_type};base64,{data}");
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url }
            })
        }
    }
}

fn tool_json(tool: &ProbeTool) -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn tool_call_json(call: &ProbeToolCall) -> Value {
    let args = Value::Object(call.arguments.clone()).to_string();
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": args,
        }
    })
}

fn parse_chat_response(value: &Value) -> Result<ProbeResponse, ProbeError> {
    let choice = value
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| ProbeError::Llm("chat completion missing choices".into()))?;
    let message = choice.get("message").unwrap_or(choice);
    let text = extract_text(message.get("content"));
    let mut tool_calls = message
        .get("tool_calls")
        .map(parse_tool_calls)
        .unwrap_or_default();
    if tool_calls.is_empty() {
        if let Some(call) = parse_legacy_function(message) {
            tool_calls.push(call);
        }
    }
    let finish = match choice.get("finish_reason").and_then(|v| v.as_str()) {
        Some(reason) => finish_from_reason(reason),
        None if !tool_calls.is_empty() => ProbeFinish::ToolCalls,
        None => ProbeFinish::Stop,
    };
    Ok(ProbeResponse {
        text,
        tool_calls,
        finish,
        usage: parse_usage(value.get("usage")),
    })
}

fn parse_usage(value: Option<&Value>) -> Option<ProbeUsage> {
    let usage = value?;
    if !usage.is_object() {
        return None;
    }
    let completion_tokens = first_u32(usage, &["completion_tokens", "output_tokens"]);
    let prompt_tokens = first_u32(usage, &["prompt_tokens", "input_tokens"]);
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|d| first_u32(d, &["reasoning_tokens"]))
        .or_else(|| first_u32(usage, &["reasoning_tokens"]));
    if completion_tokens.is_none() && prompt_tokens.is_none() && reasoning_tokens.is_none() {
        return None;
    }
    Some(ProbeUsage {
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
    })
}

fn first_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .or_else(|| {
                    v.as_f64()
                        .and_then(|n| u32::try_from(n.round() as i64).ok())
                })
                .or_else(|| {
                    v.as_str()
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .and_then(|n| u32::try_from(n.round() as i64).ok())
                })
        })
    })
}

fn extract_text(content: Option<&Value>) -> String {
    let raw = match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| !is_hidden_reasoning_part(p))
            .filter_map(visible_part_text)
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    strip_think_blocks(&raw)
}

fn is_hidden_reasoning_part(part: &Value) -> bool {
    part.get("type").and_then(|t| t.as_str()).is_some_and(|t| {
        t.eq_ignore_ascii_case("thinking")
            || t.eq_ignore_ascii_case("reasoning")
            || t.eq_ignore_ascii_case("reasoning_content")
    })
}

fn visible_part_text(part: &Value) -> Option<&str> {
    part.get("text")
        .or_else(|| part.get("output_text"))
        .and_then(|t| t.as_str())
}

fn strip_think_blocks(input: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let lower = input.to_ascii_lowercase();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let Some(open_at) = lower.get(i..).and_then(|rest| rest.find(OPEN)) else {
            out.push_str(&input[i..]);
            break;
        };
        let open_at = i + open_at;
        out.push_str(&input[i..open_at]);
        let after_open = open_at + OPEN.len();
        match lower.get(after_open..).and_then(|rest| rest.find(CLOSE)) {
            Some(rel) => i = after_open + rel + CLOSE.len(),
            None => break,
        }
    }
    out
}

fn parse_legacy_function(message: &Value) -> Option<ProbeToolCall> {
    let func = message
        .get("function")
        .filter(|f| f.is_object())
        .or_else(|| message.get("function_call").filter(|f| f.is_object()))?;
    let name = func
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    if !crate::probes::has_visible_arg_text(&name) {
        return None;
    }
    Some(ProbeToolCall {
        id: String::new(),
        name,
        arguments: parse_arguments(func.get("arguments")),
    })
}

fn parse_tool_calls(value: &Value) -> Vec<ProbeToolCall> {
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|tc| {
            let id = json_id(tc.get("id")).unwrap_or_default();
            let func = tc.get("function")?;
            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            if !crate::probes::has_visible_arg_text(&name) {
                return None;
            }
            Some(ProbeToolCall {
                id,
                name,
                arguments: parse_arguments(func.get("arguments")),
            })
        })
        .collect()
}

fn parse_arguments(value: Option<&Value>) -> Map<String, Value> {
    match value {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default(),
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    }
}

/// In-flight streamed tool call, keyed by `tool_calls` index.
struct OpenTool {
    id: String,
    name: String,
    started: bool,
    rejected: bool,
    pending_args: Vec<String>,
}

/// Split complete newline-terminated SSE lines out of `buf`.
/// Incomplete UTF-8 at the end stays in `buf` for the next HTTP chunk.
fn drain_complete_sse_lines(buf: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Some(idx) = buf.iter().position(|&b| b == b'\n') {
        let mut line = String::from_utf8_lossy(&buf[..idx]).into_owned();
        buf.drain(..=idx);
        if line.ends_with('\r') {
            line.pop();
        }
        lines.push(line);
    }
    lines
}

fn take_sse_tail(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf)
        .trim_end_matches('\r')
        .to_owned()
}

fn sse_tail_is_truncated(line: &str) -> bool {
    let data = if let Some(rest) = line.strip_prefix("data:") {
        rest.trim()
    } else if line.trim_start().starts_with('{') {
        line.trim()
    } else {
        return false;
    };
    !data.is_empty() && data != "[DONE]" && serde_json::from_str::<Value>(data).is_err()
}

fn accumulate_stream_bytes(received: &mut usize, chunk_len: usize) -> Result<(), ProbeError> {
    let next = received.saturating_add(chunk_len);
    if next > MAX_RESPONSE_BYTES {
        return Err(ProbeError::Transient("response too large".into()));
    }
    *received = next;
    Ok(())
}

fn emit_sse_line(
    line: &str,
    open_tools: &mut HashMap<u64, OpenTool>,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
    think: &mut ThinkFilter,
) -> Result<(), ProbeError> {
    let data = if let Some(rest) = line.strip_prefix("data:") {
        rest.trim()
    } else if line.trim_start().starts_with('{') {
        line.trim()
    } else {
        return Ok(());
    };
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        if data.starts_with('{') || data.starts_with('[') {
            return Err(ProbeError::Transient("malformed stream JSON".into()));
        }
        return Ok(());
    };
    if let Some(err) = value.get("error") {
        return Err(ProbeError::Llm(error_message(err)));
    }
    emit_delta(&value, open_tools, tx, think);
    Ok(())
}

fn emit_delta(
    value: &Value,
    open_tools: &mut HashMap<u64, OpenTool>,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
    think: &mut ThinkFilter,
) {
    let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
        return;
    };
    let delta = choice
        .get("delta")
        .or_else(|| choice.get("message"))
        .unwrap_or(&Value::Null);

    if let Some(text) = content_text(delta.get("content")) {
        let visible = think.feed(&text);
        if !visible.is_empty() {
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::TextDelta { text: visible }));
        }
    }

    let tool_calls = delta.get("tool_calls").and_then(|t| t.as_array());
    if tool_calls.is_none_or(Vec::is_empty) {
        if let Some(func) = delta
            .get("function")
            .filter(|f| f.is_object())
            .or_else(|| delta.get("function_call").filter(|f| f.is_object()))
        {
            emit_function(func, None, 0, open_tools, tx, false);
        }
    }

    if let Some(tcs) = tool_calls {
        for tc in tcs {
            let id = json_id(tc.get("id"));
            let index = json_u64(tc.get("index")).unwrap_or(0);
            let func = tc.get("function").unwrap_or(&Value::Null);
            emit_function(func, id.as_deref(), index, open_tools, tx, false);
        }
    }

    if let Some(reason) = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("finish_reason").and_then(|v| v.as_str()))
    {
        end_open_tools(open_tools, tx);
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::Finished {
            finish: finish_from_reason(reason),
        }));
    }
}

fn json_u64(v: Option<&Value>) -> Option<u64> {
    let v = v?;
    v.as_u64()
        .or_else(|| v.as_i64().and_then(|n| u64::try_from(n).ok()))
        .or_else(|| v.as_str()?.trim().parse().ok())
}

fn json_id(v: Option<&Value>) -> Option<String> {
    let v = v?;
    if let Some(s) = v.as_str() {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        return Some(t.to_owned());
    }
    if let Some(n) = v.as_u64() {
        return Some(n.to_string());
    }
    if let Some(n) = v.as_i64() {
        return Some(n.to_string());
    }
    None
}

fn content_text(content: Option<&Value>) -> Option<String> {
    let content = content?;
    if let Some(s) = content.as_str() {
        if s.is_empty() {
            return None;
        }
        return Some(s.to_owned());
    }
    let arr = content.as_array()?;
    let mut out = String::new();
    for part in arr {
        if is_hidden_reasoning_part(part) {
            continue;
        }
        if let Some(s) = part.as_str() {
            out.push_str(s);
            continue;
        }
        if let Some(s) = visible_part_text(part) {
            out.push_str(s);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Carry `<think>` across SSE tokens, including when the tag itself is split.
#[derive(Default)]
struct ThinkFilter {
    in_think: bool,
    hold: String,
}

impl ThinkFilter {
    fn feed(&mut self, input: &str) -> String {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";
        let mut buf = std::mem::take(&mut self.hold);
        buf.push_str(input);
        let lower = buf.to_ascii_lowercase();
        let mut out = String::new();
        let mut i = 0;
        loop {
            if self.in_think {
                if let Some(rel) = lower.get(i..).and_then(|rest| rest.find(CLOSE)) {
                    i += rel + CLOSE.len();
                    self.in_think = false;
                    continue;
                }
                let hold_len = suffix_is_tag_prefix(&lower[i..], CLOSE);
                self.hold = buf[buf.len() - hold_len..].to_owned();
                break;
            }
            if let Some(rel) = lower.get(i..).and_then(|rest| rest.find(OPEN)) {
                out.push_str(&buf[i..i + rel]);
                i += rel + OPEN.len();
                self.in_think = true;
                continue;
            }
            let hold_len = suffix_is_tag_prefix(&lower[i..], OPEN);
            out.push_str(&buf[i..buf.len() - hold_len]);
            self.hold = buf[buf.len() - hold_len..].to_owned();
            break;
        }
        out
    }

    fn flush(&mut self) -> String {
        if self.in_think {
            self.hold.clear();
            String::new()
        } else {
            std::mem::take(&mut self.hold)
        }
    }
}

fn suffix_is_tag_prefix(s: &str, tag: &str) -> usize {
    let max = s.len().min(tag.len());
    for len in (1..=max).rev() {
        let at = s.len() - len;
        if s.is_char_boundary(at) && tag.starts_with(&s[at..]) {
            return len;
        }
    }
    0
}

fn finish_from_reason(reason: &str) -> ProbeFinish {
    match reason {
        "stop" => ProbeFinish::Stop,
        "tool_calls" | "function_call" => ProbeFinish::ToolCalls,
        "length" => ProbeFinish::Length,
        _ => ProbeFinish::Other,
    }
}

fn emit_function(
    func: &Value,
    id: Option<&str>,
    index: u64,
    open_tools: &mut HashMap<u64, OpenTool>,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
    assembled: bool,
) {
    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
    {
        let tool = open_tools.entry(index).or_insert_with(|| OpenTool {
            id: id.unwrap_or("").to_owned(),
            name: String::new(),
            started: false,
            rejected: false,
            pending_args: Vec::new(),
        });
        if let Some(id) = id {
            if !id.is_empty() {
                tool.id = id.to_owned();
            }
        }
        if crate::probes::has_visible_arg_text(name) {
            tool.name = name.to_owned();
        } else if !name.is_empty() {
            tool.rejected = true;
        }
        if let Some(delta) = arg_delta(func.get("arguments")) {
            if tool.started {
                let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallArgDelta { delta }));
            } else {
                tool.pending_args.push(delta);
            }
        }
        start_named_tool(tool, tx);
    }
    if assembled {
        if let Some(tool) = open_tools.remove(&index) {
            finish_tool(tool, tx);
        }
    }
}

fn arg_delta(args: Option<&Value>) -> Option<String> {
    match args {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(obj) if obj.is_object() => Some(obj.to_string()),
        _ => None,
    }
}

fn start_named_tool(
    tool: &mut OpenTool,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
) {
    if tool.started || !crate::probes::has_visible_arg_text(&tool.name) {
        return;
    }
    let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallStart {
        id: tool.id.clone(),
        name: tool.name.clone(),
    }));
    tool.started = true;
    flush_pending_args(tool, tx);
}

fn flush_pending_args(
    tool: &mut OpenTool,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
) {
    for delta in tool.pending_args.drain(..) {
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallArgDelta { delta }));
    }
}

fn finish_tool(mut tool: OpenTool, tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>) {
    if tool.rejected {
        return;
    }
    if !tool.started {
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallStart {
            id: tool.id.clone(),
            name: tool.name.clone(),
        }));
        tool.started = true;
        flush_pending_args(&mut tool, tx);
    }
    let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallEnd));
}

fn end_open_tools(
    open_tools: &mut HashMap<u64, OpenTool>,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
) {
    let mut indexes: Vec<u64> = open_tools.keys().copied().collect();
    indexes.sort_unstable();
    for index in indexes {
        if let Some(tool) = open_tools.remove(&index) {
            finish_tool(tool, tx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeFinish, ProbeRequest, ProbeStreamChunk, ProbeTool};
    use futures::StreamExt;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    const SECRET: &str = "sk-test-secret-key";

    fn empty_req() -> ProbeRequest {
        ProbeRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            model: "m".into(),
            temperature: None,
            max_tokens: None,
        }
    }

    fn tool_req() -> ProbeRequest {
        ProbeRequest {
            messages: Vec::new(),
            tools: vec![ProbeTool {
                name: "read_file".into(),
                description: "read".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            model: "m".into(),
            temperature: None,
            max_tokens: None,
        }
    }

    fn spawn_http(
        status: u16,
        reason: &'static str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = read_http(&mut stream);
                let mut head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
                    body.len()
                );
                for (k, v) in &headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    fn spawn_http_chunked_split(
        status: u16,
        reason: &'static str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        split_at: usize,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = read_http(&mut stream);
                let split_at = split_at.min(body.len());
                let first = &body[..split_at];
                let second = &body[split_at..];
                let mut head = format!(
                    "HTTP/1.1 {status} {reason}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n"
                );
                for (k, v) in &headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let _ = stream.write_all(head.as_bytes());
                let _ = write!(stream, "{:X}\r\n", first.len());
                let _ = stream.write_all(first);
                let _ = stream.write_all(b"\r\n");
                let _ = stream.flush();
                thread::sleep(Duration::from_millis(150));
                let _ = write!(stream, "{:X}\r\n", second.len());
                let _ = stream.write_all(second);
                let _ = stream.write_all(b"\r\n0\r\n\r\n");
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
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
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let header_end = pos + 4;
                let headers = String::from_utf8_lossy(&buf[..header_end]);
                let mut content_len = 0usize;
                for line in headers.lines() {
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                }
                while buf.len() < header_end + content_len {
                    match stream.read(&mut tmp) {
                        Ok(0) => break,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        Err(_) => break,
                    }
                }
                break;
            }
            if buf.len() > 1_000_000 {
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

    #[tokio::test]
    async fn chat_401_is_auth() {
        let base = spawn_http(
            401,
            "Unauthorized",
            vec![("Content-Type".into(), "application/json".into())],
            br#"{"error":{"message":"bad key"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("401");
        assert!(matches!(err, ProbeError::Auth(_)), "{err:?}");
        let text = err.to_string();
        assert!(!text.contains(SECRET), "{text}");
    }

    #[tokio::test]
    async fn chat_403_model_forbidden_is_llm() {
        let base = spawn_http(
            403,
            "Forbidden",
            vec![("Content-Type".into(), "application/json".into())],
            br#"{"error":{"message":"model not allowed in this region"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("403");
        assert!(matches!(err, ProbeError::Llm(_)), "{err:?}");
    }

    #[test]
    fn redact_secrets_strips_bearer_sk_and_authorization() {
        let raw = "Authorization: Bearer SECRET leaked sk-live-secret";
        let redacted = redact_secrets(raw);
        assert!(!redacted.contains("SECRET"), "{redacted}");
        assert!(!redacted.contains("sk-live-secret"), "{redacted}");
        assert!(!redacted.contains("live-secret"), "{redacted}");
        assert!(redacted.contains("Authorization"), "{redacted}");
        assert!(redacted.contains("Bearer [REDACTED]"), "{redacted}");
        let groq = redact_secrets("Invalid API key: gsk_live_secret");
        assert!(!groq.contains("gsk_live_secret"), "{groq}");
        assert!(groq.contains("gsk_[REDACTED]"), "{groq}");
        assert!(redacted.contains("sk-[REDACTED]"), "{redacted}");
    }

    #[test]
    fn redact_secrets_strips_raw_authorization_and_api_keys() {
        let raw = r#"Authorization: raw-not-sk x-api-key: SECRETKEY "api-key":"SECRETKEY""#;
        let redacted = redact_secrets(raw);
        assert!(!redacted.contains("raw-not-sk"), "{redacted}");
        assert!(!redacted.contains("SECRETKEY"), "{redacted}");
        assert!(redacted.contains("Authorization: [REDACTED]"), "{redacted}");
        assert!(redacted.contains("x-api-key: [REDACTED]"), "{redacted}");
        assert!(redacted.contains(r#""api-key":"[REDACTED]""#), "{redacted}");
    }

    #[test]
    fn redact_secrets_strips_api_key_underscore() {
        let raw = r#"api_key=SECRET "api_key":"SECRET""#;
        let redacted = redact_secrets(raw);
        assert!(!redacted.contains("SECRET"), "{redacted}");
        assert!(redacted.contains("api_key=[REDACTED]"), "{redacted}");
        assert!(redacted.contains(r#""api_key":"[REDACTED]""#), "{redacted}");

        let escaped = r#"{\"api_key\":\"SECRET\"}"#;
        let redacted = redact_secrets(escaped);
        assert!(!redacted.contains("SECRET"), "{redacted}");
        assert!(redacted.contains("[REDACTED]"), "{redacted}");
    }

    #[test]
    fn redact_secrets_strips_camel_case_api_key() {
        let raw = r#"{"apiKey":"SECRET"}"#;
        let redacted = redact_secrets(raw);
        assert!(!redacted.contains("SECRET"), "{redacted}");
        assert!(
            redacted.contains(r#"{"apiKey":"[REDACTED]"}"#),
            "{redacted}"
        );
    }

    #[test]
    fn parse_chat_response_fills_usage() {
        let value = serde_json::json!({
            "choices": [{
                "message": { "content": "4" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 80,
                "completion_tokens_details": { "reasoning_tokens": 40 }
            }
        });
        let resp = parse_chat_response(&value).expect("parse");
        let usage = resp.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(11));
        assert_eq!(usage.completion_tokens, Some(80));
        assert_eq!(usage.reasoning_tokens, Some(40));
        assert_eq!(resp.text, "4");
    }

    #[test]
    fn parse_chat_response_strips_think_tags() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "<think>WH-4481 secret</think>{\"word\":\"hello\"}"
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert!(!resp.text.contains("WH-4481"), "{:?}", resp.text);
        assert!(resp.text.contains("hello"), "{:?}", resp.text);
    }

    #[test]
    fn parse_chat_response_drops_thinking_parts() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "thinking", "text": "count these words now"},
                        {"type": "text", "text": "ok"}
                    ]
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert_eq!(resp.text, "ok");
    }

    #[test]
    fn parse_chat_response_ignores_reasoning_content_field() {
        let reasoning = "long cot that restates the secret fact WH-4481. ".repeat(40);
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": reasoning,
                    "content": "final"
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert_eq!(resp.text, "final");
    }

    #[test]
    fn parse_chat_response_strips_unclosed_think() {
        let value = serde_json::json!({
            "choices": [{
                "message": { "content": "<think>partial" },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert!(!resp.text.contains("partial"), "{:?}", resp.text);
        assert_eq!(resp.text, "");
    }

    #[test]
    fn parse_chat_response_usage_absent_is_none() {
        let value = serde_json::json!({
            "choices": [{
                "message": { "content": "4" },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn parse_chat_response_legacy_function_is_tool_call() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"/tmp/a\"}"
                    }
                },
                "finish_reason": "function_call"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert_eq!(resp.finish, ProbeFinish::ToolCalls);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "read_file");
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .get("path")
                .and_then(|v| v.as_str()),
            Some("/tmp/a")
        );
    }

    #[test]
    fn parse_chat_response_function_call_field_is_tool_call() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "function_call": {
                        "name": "list_dir",
                        "arguments": "{\"path\":\"/tmp\"}"
                    }
                },
                "finish_reason": "function_call"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert_eq!(resp.tool_calls[0].name, "list_dir");
        assert_eq!(resp.finish, ProbeFinish::ToolCalls);
    }

    #[test]
    fn parse_chat_response_numeric_tool_id_is_kept() {
        let value = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": 42,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/a\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let resp = parse_chat_response(&value).expect("parse");
        assert_eq!(
            resp.tool_calls[0].id, "42",
            "numeric chat tool id must stringify: {resp:?}"
        );
    }

    #[test]
    fn push_response_chunk_rejects_over_cap() {
        let mut buf = vec![0u8; 10];
        let err = push_response_chunk(&mut buf, &[1, 2, 3, 4, 5, 6], 15).unwrap_err();
        assert!(
            matches!(err, ProbeError::Transient(ref msg) if msg.contains("too large")),
            "{err:?}"
        );
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn parse_chat_response_empty_tool_name_is_not_a_call() {
        for name in ["", "   ", "\u{200b}"] {
            let value = serde_json::json!({
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            });
            let resp = parse_chat_response(&value).expect("parse");
            assert!(
                resp.tool_calls.is_empty(),
                "empty name must not become a tool call: {name:?}"
            );
        }
    }

    fn chat_req(model: &str) -> ProbeRequest {
        ProbeRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            model: model.into(),
            temperature: Some(0.2),
            max_tokens: Some(64),
        }
    }

    #[test]
    fn chat_body_o3_mini_uses_max_completion_tokens() {
        let body = chat_body(&chat_req("o3-mini"), false);
        assert_eq!(body["max_completion_tokens"], 64);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn chat_body_provider_prefixed_o3_uses_max_completion_tokens() {
        let body = chat_body(&chat_req("openai/o3-mini"), false);
        assert_eq!(body["max_completion_tokens"], 64);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn chat_body_gpt5_uses_max_completion_tokens() {
        let body = chat_body(&chat_req("gpt-5-mini"), false);
        assert_eq!(body["max_completion_tokens"], 64);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn chat_body_gpt4o_keeps_max_tokens_and_temperature() {
        let body = chat_body(&chat_req("gpt-4o"), false);
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], serde_json::json!(0.2f32));
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn reasoning_chat_model_does_not_match_o10() {
        assert!(!reasoning_chat_model("o10"));
        assert!(reasoning_chat_model("o1"));
        assert!(reasoning_chat_model("o1-mini"));
    }

    #[tokio::test]
    async fn chat_401_redacts_raw_authorization_from_body() {
        let base = spawn_http(
            401,
            "Unauthorized",
            vec![("Content-Type".into(), "application/json".into())],
            br#"{"error":{"message":"Authorization: raw-not-sk"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("401");
        assert!(matches!(err, ProbeError::Auth(_)), "{err:?}");
        let text = err.to_string();
        assert!(!text.contains("raw-not-sk"), "{text}");
        assert!(text.contains("authentication error:"), "{text}");
        assert!(text.contains("Authorization"), "{text}");
    }

    #[tokio::test]
    async fn chat_401_redacts_bearer_and_sk_from_body() {
        let base = spawn_http(
            401,
            "Unauthorized",
            vec![("Content-Type".into(), "application/json".into())],
            br#"{"error":{"message":"Bearer SECRET sk-live-secret"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("401");
        assert!(matches!(err, ProbeError::Auth(_)), "{err:?}");
        let text = err.to_string();
        assert!(!text.contains("SECRET"), "{text}");
        assert!(!text.contains("sk-live-secret"), "{text}");
        assert!(!text.contains("live-secret"), "{text}");
        assert!(text.matches("authentication error:").count() == 1, "{text}");
    }

    #[tokio::test]
    async fn chat_429_is_rate_limit() {
        let base = spawn_http(
            429,
            "Too Many Requests",
            vec![
                ("Content-Type".into(), "application/json".into()),
                ("Retry-After".into(), "7".into()),
            ],
            br#"{"error":{"message":"slow down"}}"#.to_vec(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("429");
        match err {
            ProbeError::RateLimit { retry_after } => assert_eq!(retry_after, Some(7)),
            other => panic!("expected RateLimit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_openai_delta_tool_calls() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"/tmp/test.txt\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallStart { id, name }
                    if id == "call_1" && name == "read_file"
            )),
            "{chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallArgDelta { delta } if delta.contains("/tmp/test.txt")
            )),
            "{chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, ProbeStreamChunk::ToolCallEnd)),
            "{chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::Finished {
                    finish: ProbeFinish::ToolCalls
                }
            )),
            "{chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_length_emits_finished_length() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"I will call read_file\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::Finished {
                    finish: ProbeFinish::Length
                }
            )),
            "SSE length must surface as Finished(Length): {chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_length_flushes_partial_tool_then_finished() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallStart { name, .. } if name == "read_file"
            )),
            "{chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, ProbeStreamChunk::ToolCallEnd)),
            "length must flush the open tool before Finished: {chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::Finished {
                    finish: ProbeFinish::Length
                }
            )),
            "{chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_late_name_emits_one_named_start() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"arguments\":\"{\\\"path\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\",\"arguments\":\":\\\"/tmp/test.txt\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let starts: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![("c1", "read_file")], "{chunks:?}");
        let start_at = chunks
            .iter()
            .position(|c| matches!(c, ProbeStreamChunk::ToolCallStart { .. }))
            .expect("start");
        let first_arg = chunks
            .iter()
            .position(|c| matches!(c, ProbeStreamChunk::ToolCallArgDelta { .. }))
            .expect("args");
        assert!(start_at < first_arg, "{chunks:?}");
        let args: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallArgDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(args, r#"{"path":"/tmp/test.txt"}"#);
        let ends = chunks
            .iter()
            .filter(|c| matches!(c, ProbeStreamChunk::ToolCallEnd))
            .count();
        assert_eq!(ends, 1, "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_unnamed_args_start_once_at_end() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"arguments\":\"{\\\"path\\\":\\\"/tmp/test.txt\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let starts: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![("c1", "")], "{chunks:?}");
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallArgDelta { delta } if delta.contains("/tmp/test.txt")
            )),
            "{chunks:?}"
        );
        let ends = chunks
            .iter()
            .filter(|c| matches!(c, ProbeStreamChunk::ToolCallEnd))
            .count();
        assert_eq!(ends, 1, "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_assembled_function_call_field() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"function_call\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/test.txt\\\"}\"}}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallStart { name, .. } if name == "read_file"
            )),
            "{chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, ProbeStreamChunk::ToolCallEnd)),
            "{chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_ignores_legacy_function_call_when_tool_calls_present() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"function_call\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/test.txt\\\"}\"},\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            !chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallArgDelta { delta } if delta.contains("/tmp/test.txt")
            )),
            "leftover function_call path must not emit when tool_calls is present: {chunks:?}"
        );
        let starts = chunks
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    ProbeStreamChunk::ToolCallStart { name, .. } if name == "read_file"
                )
            })
            .count();
        assert_eq!(starts, 1, "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_assembled_function_object() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/test.txt\\\"}\"}}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallStart { name, .. } if name == "read_file"
            )),
            "{chunks:?}"
        );
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallArgDelta { delta } if delta.contains("/tmp/test.txt")
            )),
            "{chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, ProbeStreamChunk::ToolCallEnd)),
            "{chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_parallel_indexes_emit_two_starts() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"p\\\":\\\"a\\\"}\"}},{\"index\":1,\"id\":\"c1\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"p\\\":\\\"b\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let starts: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![("c0", "read_file"), ("c1", "write_file")]);
        let args: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallArgDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 2, "{chunks:?}");
        assert!(args[0].contains('a'), "{args:?}");
        assert!(args[1].contains('b'), "{args:?}");
        assert!(!args[0].contains('b'), "{args:?}");
        let ends = chunks
            .iter()
            .filter(|c| matches!(c, ProbeStreamChunk::ToolCallEnd))
            .count();
        assert_eq!(ends, 2, "{chunks:?}");
    }

    #[test]
    fn emit_sse_line_malformed_json_object_is_transient() {
        let (tx, _rx) = futures::channel::mpsc::unbounded();
        let mut open = HashMap::new();
        let mut think = ThinkFilter::default();
        let err = emit_sse_line("data: {", &mut open, &tx, &mut think).expect_err("object");
        assert!(
            matches!(err, ProbeError::Transient(ref msg) if msg.contains("malformed")),
            "{err:?}"
        );
        emit_sse_line("data: ping", &mut open, &tx, &mut think).expect("keepalive");
        let err = emit_sse_line("data: [", &mut open, &tx, &mut think).expect_err("array");
        assert!(
            matches!(err, ProbeError::Transient(ref msg) if msg.contains("malformed")),
            "{err:?}"
        );
    }

    #[test]
    fn accumulate_stream_bytes_counts_after_sse_drain() {
        let line = b"data: {\"ok\":true}\n";
        let mut buf = line.to_vec();
        let mut received = 0usize;
        accumulate_stream_bytes(&mut received, line.len()).expect("under cap");
        let _ = drain_complete_sse_lines(&mut buf);
        assert!(buf.is_empty(), "complete SSE line must drain");
        assert_eq!(received, line.len(), "budget must not reset on drain");
        let err = accumulate_stream_bytes(&mut received, MAX_RESPONSE_BYTES).unwrap_err();
        assert!(
            matches!(err, ProbeError::Transient(ref msg) if msg.contains("too large")),
            "{err:?}"
        );
        assert_eq!(received, line.len());
    }

    #[test]
    fn drain_complete_sse_lines_holds_split_utf8_until_newline() {
        let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"café!\"}}]}\n";
        let split_at = payload.find('é').expect("é") + 1;
        let bytes = payload.as_bytes();
        let mut buf = bytes[..split_at].to_vec();
        assert!(drain_complete_sse_lines(&mut buf).is_empty());
        buf.extend_from_slice(&bytes[split_at..]);
        let lines = drain_complete_sse_lines(&mut buf);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("café!"), "{lines:?}");
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn stream_utf8_split_across_http_chunks_preserves_text() {
        let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"café!\"}}]}\n\n";
        let split_at = payload.find('é').expect("é") + 1;
        let base = spawn_http_chunked_split(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            payload.as_bytes().to_vec(),
            split_at,
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let text: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            text, "café!",
            "UTF-8 split across HTTP chunks corrupted the text: {chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_string_index_keeps_parallel_tools() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":\"0\",\"id\":\"c0\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"p\\\":\\\"a\\\"}\"}},{\"index\":\"1\",\"id\":\"c1\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"p\\\":\\\"b\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let starts: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::ToolCallStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            vec![("c0", "read_file"), ("c1", "write_file")],
            "string tool_calls index must not collapse both tools onto 0: {chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_numeric_tool_id_is_kept() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":42,\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::ToolCallStart { id, name }
                    if id == "42" && name == "read_file"
            )),
            "numeric tool id must surface as a string: {chunks:?}"
        );
    }

    #[tokio::test]
    async fn stream_thinking_parts_are_not_emitted() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"thinking\",\"text\":\"WH-4481 secret\"},{\"type\":\"text\",\"text\":\"ok\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let text: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !text.contains("WH-4481"),
            "stream must drop thinking parts: {chunks:?}"
        );
        assert_eq!(text, "ok", "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_reasoning_parts_and_think_tags_are_not_emitted() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"reasoning\",\"text\":\"count these words now\"},{\"type\":\"text\",\"text\":\"<think>hidden</think>visible\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let text: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !text.contains("count these words"),
            "stream must drop reasoning parts: {chunks:?}"
        );
        assert!(
            !text.contains("hidden"),
            "stream must strip <think> in content: {chunks:?}"
        );
        assert_eq!(text, "visible", "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_think_split_across_sse_lines_is_not_emitted() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello <thi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"nk>WH-4481 secret</thi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"nk> world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let text: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !text.contains("WH-4481"),
            "split <think> across SSE lines must not leak: {chunks:?}"
        );
        assert!(
            !text.to_ascii_lowercase().contains("<think"),
            "split think tags must not leak: {chunks:?}"
        );
        assert_eq!(text, "hello  world", "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_zwsp_or_whitespace_name_does_not_start_a_tool() {
        for name in ["   ", "\u{200b}"] {
            let sse = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"c1\",\"function\":{{\"name\":{name},\"arguments\":\"{{\\\"path\\\":\\\"/tmp/test.txt\\\"}}\"}}}}]}}}}]}}\n\n\
                 data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
                 data: [DONE]\n\n",
                name = serde_json::to_string(name).expect("name")
            );
            let base = spawn_http(
                200,
                "OK",
                vec![("Content-Type".into(), "text/event-stream".into())],
                sse.as_bytes().to_vec(),
            );
            let chunks: Vec<_> = client(&base).stream_chat(tool_req()).collect().await;
            let chunks: Vec<ProbeStreamChunk> =
                chunks.into_iter().map(|c| c.expect("chunk")).collect();
            let names: Vec<&str> = chunks
                .iter()
                .filter_map(|c| match c {
                    ProbeStreamChunk::ToolCallStart { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                names.is_empty(),
                "ZWSP/whitespace name must not emit ToolCallStart: name={name:?} {chunks:?}"
            );
        }
    }

    #[tokio::test]
    async fn stream_array_content_emits_text() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        let text: String = chunks
            .iter()
            .filter_map(|c| match c {
                ProbeStreamChunk::TextDelta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "hello", "{chunks:?}");
    }

    #[tokio::test]
    async fn stream_finish_reason_on_delta_emits_finished() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\",\"finish_reason\":\"length\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let base = spawn_http(
            200,
            "OK",
            vec![("Content-Type".into(), "text/event-stream".into())],
            sse.as_bytes().to_vec(),
        );
        let chunks: Vec<_> = client(&base).stream_chat(empty_req()).collect().await;
        let chunks: Vec<ProbeStreamChunk> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
        assert!(
            chunks.iter().any(|c| matches!(
                c,
                ProbeStreamChunk::Finished {
                    finish: ProbeFinish::Length
                }
            )),
            "finish_reason on delta must emit Finished: {chunks:?}"
        );
    }

    #[tokio::test]
    async fn error_body_is_capped() {
        let huge = "x".repeat(200 * 1024);
        let base = spawn_http(
            500,
            "Internal Server Error",
            vec![("Content-Type".into(), "text/plain".into())],
            huge.into_bytes(),
        );
        let err = client(&base).chat(empty_req()).await.expect_err("500");
        let text = err.to_string();
        assert!(
            text.len() < 80 * 1024,
            "error body must be capped, got {} bytes",
            text.len()
        );
    }

    #[tokio::test]
    async fn list_models_404_is_empty() {
        let base = spawn_http(404, "Not Found", Vec::new(), b"missing".to_vec());
        let ids = list_model_ids(&base, Some(SECRET)).await.expect("404");
        assert!(ids.is_empty());
    }
}
