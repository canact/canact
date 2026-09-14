//! Probe LLM client trait and wire types.

use std::future::Future;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::ProbeError;

/// Chat role for a probe turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProbeRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Message body: plain text or multimodal parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProbeContent {
    Text(String),
    Parts(Vec<ProbeContentPart>),
}

/// One content part (text or inline image).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProbeContentPart {
    Text { text: String },
    ImageBase64 { media_type: String, data: String },
}

/// One chat message in a probe request or replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeMessage {
    pub role: ProbeRole,
    pub content: ProbeContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ProbeToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeFinish {
    Stop,
    ToolCalls,
    Length,
    Other,
}

/// Map a provider `finish_reason` / `stop_reason` string to
/// [`ProbeFinish`].
///
/// Hosts that implement [`ProbeClient`] around their own HTTP client
/// should run this on `StreamChunk::FinishReason` (or the non-stream
/// `finish_reason` field). Do not reimplement the table. A Length
/// cut must stay Transient instead of a cached Weak card.
///
/// | Reason | [`ProbeFinish`] |
/// | --- | --- |
/// | `stop`, `end_turn`, `eos` | [`ProbeFinish::Stop`] |
/// | `tool_calls`, `tool_use`, `function_call` | [`ProbeFinish::ToolCalls`] |
/// | `length`, `max_tokens` | [`ProbeFinish::Length`] |
/// | anything else | [`ProbeFinish::Other`] |
#[must_use]
pub fn finish_from_reason(reason: &str) -> ProbeFinish {
    match reason {
        "stop" | "end_turn" | "eos" => ProbeFinish::Stop,
        "tool_calls" | "tool_use" | "function_call" => ProbeFinish::ToolCalls,
        "length" | "max_tokens" => ProbeFinish::Length,
        _ => ProbeFinish::Other,
    }
}

/// Streamed token or tool-call fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStreamChunk {
    TextDelta {
        text: String,
    },
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallArgDelta {
        delta: String,
    },
    ToolCallEnd,
    /// Terminal finish reason from the provider. Missing means Stop.
    Finished {
        finish: ProbeFinish,
    },
}

/// Host catalog priors. Never persist a catalog boolean as Strong.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogPriors {
    pub advertised_context_tokens: Option<u32>,
    pub supports_vision: Option<bool>,
    pub supports_tools: Option<bool>,
}

/// JSON-schema tool offered to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// One chat completion request from a probe.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRequest {
    pub messages: Vec<ProbeMessage>,
    pub tools: Vec<ProbeTool>,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Token usage reported by the host client, when the wire includes it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProbeUsage {
    /// Prompt / input tokens, if the provider sent them.
    pub prompt_tokens: Option<u32>,
    /// Provider completion / output count (not a char estimate).
    pub completion_tokens: Option<u32>,
    /// Hidden reasoning tokens, if the provider sent them.
    pub reasoning_tokens: Option<u32>,
}

/// One chat completion response.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeResponse {
    pub text: String,
    pub tool_calls: Vec<ProbeToolCall>,
    pub finish: ProbeFinish,
    /// Live usage from the provider. `None` when the wire omitted it.
    pub usage: Option<ProbeUsage>,
}

impl ProbeResponse {
    /// Build a host [`ProbeResponse`] from raw completion text and a
    /// provider finish string.
    ///
    /// Strips inline `<think>` blocks via [`crate::strip_think_blocks`]
    /// and maps the finish string via [`finish_from_reason`]. Usage is
    /// `None` until [`Self::with_usage`]. Tool calls stay empty; hosts
    /// that received tools still fill `tool_calls` themselves.
    ///
    /// An empty or whitespace-only finish is [`ProbeFinish::Stop`],
    /// the same as a missing stream finish
    /// ([`ProbeStreamChunk::Finished`]).
    #[must_use]
    pub fn from_host_text(text: &str, finish: &str) -> Self {
        let finish = if finish.trim().is_empty() {
            ProbeFinish::Stop
        } else {
            finish_from_reason(finish)
        };
        Self {
            text: crate::text::strip_think_blocks(text),
            tool_calls: Vec::new(),
            finish,
            usage: None,
        }
    }

    /// Attach provider usage reported on the wire.
    #[must_use]
    pub fn with_usage(mut self, usage: ProbeUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// A tool call returned by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// Host-implemented LLM used by the probe suite.
pub trait ProbeClient: Send + Sync {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send;

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send;

    fn model_id(&self) -> &str;

    fn provider(&self) -> &str;

    /// Catalog flags are priors only. Never persist a catalog boolean as Strong.
    fn catalog(&self) -> CatalogPriors {
        CatalogPriors::default()
    }
}

/// In-memory client for corpus tests and later probe modules.
#[derive(Debug)]
pub struct MockLlm {
    model_id: String,
    provider: String,
    error: Option<ProbeError>,
    catalog: CatalogPriors,
}

impl MockLlm {
    /// Construct a successful mock for `model_id` / `provider`.
    pub fn new(model_id: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            provider: provider.into(),
            error: None,
            catalog: CatalogPriors::default(),
        }
    }

    /// Fail `chat` / `stream_chat` with this error.
    pub fn with_error(mut self, error: ProbeError) -> Self {
        self.error = Some(error);
        self
    }

    /// Override catalog priors (vision/tools advertisements).
    pub fn with_catalog(mut self, catalog: CatalogPriors) -> Self {
        self.catalog = catalog;
        self
    }

    fn injected_error(&self) -> Option<ProbeError> {
        self.error.as_ref().map(clone_probe_error)
    }
}

fn clone_probe_error(err: &ProbeError) -> ProbeError {
    match err {
        ProbeError::Auth(s) => ProbeError::Auth(s.clone()),
        ProbeError::NotFound(s) => ProbeError::NotFound(s.clone()),
        ProbeError::Llm(s) => ProbeError::Llm(s.clone()),
        ProbeError::Transient(s) => ProbeError::Transient(s.clone()),
        ProbeError::RateLimit { retry_after } => ProbeError::RateLimit {
            retry_after: *retry_after,
        },
        ProbeError::Io(e) => ProbeError::Io(std::io::Error::new(e.kind(), e.to_string())),
        ProbeError::Json(e) => ProbeError::Internal(format!("JSON error: {e}")),
        ProbeError::Internal(s) => ProbeError::Internal(s.clone()),
    }
}

impl ProbeClient for MockLlm {
    fn chat(
        &self,
        req: ProbeRequest,
    ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
        let err = self.injected_error();
        let resp = if req.tools.is_empty() {
            ProbeResponse {
                text: "ok".to_owned(),
                tool_calls: Vec::new(),
                finish: ProbeFinish::Stop,
                usage: None,
            }
        } else {
            ProbeResponse {
                text: String::new(),
                tool_calls: vec![ProbeToolCall {
                    id: "call_1".to_owned(),
                    name: req.tools[0].name.clone(),
                    arguments: serde_json::json!({"path": "/tmp/test.txt"})
                        .as_object()
                        .unwrap()
                        .clone(),
                }],
                finish: ProbeFinish::ToolCalls,
                usage: None,
            }
        };
        async move {
            if let Some(err) = err {
                return Err(err);
            }
            Ok(resp)
        }
    }

    fn stream_chat(
        &self,
        req: ProbeRequest,
    ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
        let items = if let Some(err) = self.injected_error() {
            vec![Err(err)]
        } else if req.tools.is_empty() {
            vec![Ok(ProbeStreamChunk::TextDelta {
                text: "ok".to_owned(),
            })]
        } else {
            vec![
                Ok(ProbeStreamChunk::ToolCallStart {
                    id: "call_1".to_owned(),
                    name: req.tools[0].name.clone(),
                }),
                Ok(ProbeStreamChunk::ToolCallArgDelta {
                    delta: "{}".to_owned(),
                }),
                Ok(ProbeStreamChunk::ToolCallEnd),
            ]
        };
        futures::stream::iter(items)
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

#[cfg(test)]
mod tests {
    use super::{ProbeFinish, ProbeResponse, ProbeUsage, finish_from_reason};

    #[test]
    fn finish_from_reason_stop_family() {
        assert_eq!(finish_from_reason("stop"), ProbeFinish::Stop);
        assert_eq!(finish_from_reason("end_turn"), ProbeFinish::Stop);
        assert_eq!(finish_from_reason("eos"), ProbeFinish::Stop);
    }

    #[test]
    fn finish_from_reason_tool_family() {
        assert_eq!(finish_from_reason("tool_calls"), ProbeFinish::ToolCalls);
        assert_eq!(finish_from_reason("tool_use"), ProbeFinish::ToolCalls);
        assert_eq!(finish_from_reason("function_call"), ProbeFinish::ToolCalls);
    }

    #[test]
    fn finish_from_reason_length_family() {
        assert_eq!(finish_from_reason("length"), ProbeFinish::Length);
        assert_eq!(finish_from_reason("max_tokens"), ProbeFinish::Length);
    }

    #[test]
    fn finish_from_reason_other_is_exact() {
        assert_eq!(finish_from_reason("other"), ProbeFinish::Other);
        assert_eq!(finish_from_reason("STOP"), ProbeFinish::Other);
        assert_eq!(finish_from_reason(""), ProbeFinish::Other);
    }

    #[test]
    fn from_host_text_strips_think_and_maps_finish() {
        let resp = ProbeResponse::from_host_text("<think>hid</think>Paris", "length");
        assert_eq!(resp.text, "Paris");
        assert_eq!(resp.finish, ProbeFinish::Length);
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.usage, None);
    }

    #[test]
    fn from_host_text_empty_finish_is_stop() {
        let empty = ProbeResponse::from_host_text("ok", "");
        assert_eq!(empty.finish, ProbeFinish::Stop);
        let blank = ProbeResponse::from_host_text("ok", "   ");
        assert_eq!(blank.finish, ProbeFinish::Stop);
        let exact_other = ProbeResponse::from_host_text("ok", "STOP");
        assert_eq!(exact_other.finish, ProbeFinish::Other);
    }

    #[test]
    fn from_host_text_with_usage() {
        let usage = ProbeUsage {
            prompt_tokens: Some(3),
            completion_tokens: Some(7),
            reasoning_tokens: None,
        };
        let resp = ProbeResponse::from_host_text("ok", "stop").with_usage(usage.clone());
        assert_eq!(resp.finish, ProbeFinish::Stop);
        assert_eq!(resp.usage, Some(usage));
    }
}
