//! Slim OpenAI-compatible HTTP + SSE adapter.
//!
//! Do not copy `bline-llm`. Never log `Authorization`.

use std::time::Duration;

use futures::Stream;
use futures::channel::mpsc::UnboundedSender;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value};

use crate::client::{
    CatalogPriors, ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeMessage,
    ProbeRequest, ProbeResponse, ProbeRole, ProbeStreamChunk, ProbeTool, ProbeToolCall,
};
use crate::error::ProbeError;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

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
        let mut rest = String::new();
        let mut tool_open = false;
        while let Some(item) = bytes.next().await {
            let chunk = item.map_err(map_transport)?;
            rest.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(idx) = rest.find('\n') {
                let mut line = rest[..idx].to_string();
                rest = rest[idx + 1..].to_string();
                if line.ends_with('\r') {
                    line.pop();
                }
                if let Err(err) = emit_sse_line(&line, &mut tool_open, tx) {
                    let _ = tx.unbounded_send(Err(err));
                    return Ok(());
                }
            }
        }
        if !rest.is_empty() {
            let line = rest.trim_end_matches('\r');
            if let Err(err) = emit_sse_line(line, &mut tool_open, tx) {
                let _ = tx.unbounded_send(Err(err));
                return Ok(());
            }
        }
        if tool_open {
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallEnd));
        }
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
            let value: Value = resp.json().await.map_err(map_transport)?;
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
    let value: Value = resp.json().await.map_err(map_transport)?;
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
    ProbeError::Transient(err.to_string())
}

async fn ensure_success(resp: reqwest::Response) -> Result<reqwest::Response, ProbeError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let retry_after = parse_retry_after(resp.headers());
    let code = status.as_u16();
    let body = resp.text().await.unwrap_or_default();
    Err(map_status(code, retry_after, &body))
}

fn parse_retry_after(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

fn map_status(code: u16, retry_after: Option<u64>, body: &str) -> ProbeError {
    match code {
        401 | 403 => ProbeError::Auth(body_or_status(code, body)),
        429 => ProbeError::RateLimit { retry_after },
        408 | 500..=599 => ProbeError::Transient(body_or_status(code, body)),
        _ => ProbeError::Llm(body_or_status(code, body)),
    }
}

fn body_or_status(code: u16, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        format!("HTTP {code}")
    } else {
        trimmed.to_owned()
    }
}

fn error_message(err: &Value) -> String {
    err.get("message")
        .and_then(|m| m.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| err.to_string())
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
        body.insert("temperature".into(), Value::from(t));
    }
    if let Some(m) = req.max_tokens {
        body.insert("max_tokens".into(), Value::from(m));
    }
    if stream {
        body.insert("stream".into(), Value::Bool(true));
    }
    Value::Object(body)
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
    let tool_calls = message
        .get("tool_calls")
        .map(parse_tool_calls)
        .unwrap_or_default();
    let finish = match choice.get("finish_reason").and_then(|v| v.as_str()) {
        Some("stop") => ProbeFinish::Stop,
        Some("tool_calls") => ProbeFinish::ToolCalls,
        Some("length") => ProbeFinish::Length,
        Some(_) => ProbeFinish::Other,
        None if !tool_calls.is_empty() => ProbeFinish::ToolCalls,
        None => ProbeFinish::Stop,
    };
    Ok(ProbeResponse {
        text,
        tool_calls,
        finish,
    })
}

fn extract_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn parse_tool_calls(value: &Value) -> Vec<ProbeToolCall> {
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|tc| {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let func = tc.get("function")?;
            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
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

fn emit_sse_line(
    line: &str,
    tool_open: &mut bool,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
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
        return Ok(());
    };
    if let Some(err) = value.get("error") {
        return Err(ProbeError::Llm(error_message(err)));
    }
    emit_delta(&value, tool_open, tx);
    Ok(())
}

fn emit_delta(
    value: &Value,
    tool_open: &mut bool,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
) {
    let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
        return;
    };
    let delta = choice
        .get("delta")
        .or_else(|| choice.get("message"))
        .unwrap_or(&Value::Null);

    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::TextDelta {
                text: text.to_owned(),
            }));
        }
    }

    if let Some(func) = delta.get("function").filter(|f| f.is_object()) {
        emit_function(func, None, tool_open, tx, true);
    }

    if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let id = tc.get("id").and_then(|v| v.as_str());
            let func = tc.get("function").unwrap_or(&Value::Null);
            emit_function(func, id, tool_open, tx, false);
        }
    }

    let finish = choice.get("finish_reason").and_then(|v| v.as_str());
    if matches!(finish, Some("tool_calls") | Some("stop")) && *tool_open {
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallEnd));
        *tool_open = false;
    }
}

fn emit_function(
    func: &Value,
    id: Option<&str>,
    tool_open: &mut bool,
    tx: &UnboundedSender<Result<ProbeStreamChunk, ProbeError>>,
    assembled: bool,
) {
    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = func.get("arguments");

    if !name.is_empty() && !*tool_open {
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallStart {
            id: id.unwrap_or("").to_owned(),
            name: name.to_owned(),
        }));
        *tool_open = true;
    }

    match args {
        Some(Value::String(s)) if !s.is_empty() => {
            if !*tool_open {
                let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallStart {
                    id: id.unwrap_or("").to_owned(),
                    name: name.to_owned(),
                }));
                *tool_open = true;
            }
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallArgDelta { delta: s.clone() }));
        }
        Some(obj) if obj.is_object() => {
            if !*tool_open {
                let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallStart {
                    id: id.unwrap_or("").to_owned(),
                    name: name.to_owned(),
                }));
                *tool_open = true;
            }
            let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: obj.to_string(),
            }));
        }
        _ => {}
    }

    if assembled && *tool_open {
        let _ = tx.unbounded_send(Ok(ProbeStreamChunk::ToolCallEnd));
        *tool_open = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeRequest, ProbeStreamChunk, ProbeTool};
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
    async fn list_models_404_is_empty() {
        let base = spawn_http(404, "Not Found", Vec::new(), b"missing".to_vec());
        let ids = list_model_ids(&base, Some(SECRET)).await.expect("404");
        assert!(ids.is_empty());
    }
}
