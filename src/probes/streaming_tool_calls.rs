//! Streaming tool-call probe.
//!
//! Tests whether the model can emit tool calls via the streaming API
//! without producing malformed argument chunks. Models that fail this
//! should use non-streaming for tool-call turns.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest, ProbeStreamChunk};
use crate::types::{ProbeResult, classify};
use futures::StreamExt;

use super::{tool, user_text};

const FORCEFUL_READ_FILE: &str = "Immediately call the read_file tool with path /tmp/test.txt. Do not describe what you would do. Do not ask for confirmation.";

/// Probe whether the model can stream tool calls reliably.
///
/// Sends a streaming request with a simple tool and checks whether
/// the streamed chunks assemble into a valid tool call.
///
/// Scoring:
/// - `1.0` - stream completed, tool call name and valid JSON arguments received
/// - `0.5` - stream completed with tool call name but malformed/missing arguments
/// - `0.0` - stream completed with no tool call chunks
///
/// Stream setup, timeout, 429, or other `stream_chat` errors are returned as
/// [`Err`], not Weak. A completed stream with no tool calls is still Weak.
pub async fn probe_streaming_tool_calls<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let tool_spec = tool(
        "read_file",
        "Read the contents of a file at the given path.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        }),
    );

    let request = ProbeRequest {
        messages: vec![user_text(FORCEFUL_READ_FILE)],
        tools: vec![tool_spec],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let mut stream = std::pin::pin!(llm.stream_chat(request));

    let mut got_tool_name = false;
    let mut args_buffer = String::new();
    let mut got_any_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                got_any_chunk = true;
                match &chunk {
                    ProbeStreamChunk::ToolCallStart { name, .. } => {
                        if !name.is_empty() {
                            got_tool_name = true;
                        }
                    }
                    ProbeStreamChunk::ToolCallArgDelta { delta } => {
                        args_buffer.push_str(delta);
                    }
                    _ => {}
                }
            }
            Err(err) => {
                // Timeout, 429, transport, or setup failure is not a
                // completed Weak capability result.
                return Err(err);
            }
        }
    }

    let (score, details) = if !got_any_chunk {
        (0.0, "Stream produced no chunks".to_string())
    } else if got_tool_name {
        let args_valid = if args_buffer.is_empty() {
            false
        } else {
            serde_json::from_str::<serde_json::Value>(&args_buffer).is_ok()
        };
        if args_valid {
            (
                1.0,
                "Streaming tool call with valid JSON arguments".to_string(),
            )
        } else {
            (
                0.5,
                format!(
                    "Tool call name streamed but arguments malformed: {:?}",
                    prefix_bytes(&args_buffer, 80)
                ),
            )
        }
    } else {
        (
            0.0,
            "Stream completed but no tool call chunks received".to_string(),
        )
    };

    Ok(ProbeResult {
        name: "streaming_tool_calls".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn prefix_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeResponse, ProbeStreamChunk};
    use futures::Stream;

    struct StreamMockLlm {
        chunks: std::sync::Mutex<Option<Vec<Result<ProbeStreamChunk, ProbeError>>>>,
    }

    impl StreamMockLlm {
        fn new(chunks: Vec<Result<ProbeStreamChunk, ProbeError>>) -> Self {
            Self {
                chunks: std::sync::Mutex::new(Some(chunks)),
            }
        }
    }

    impl ProbeClient for StreamMockLlm {
        async fn chat(&self, _req: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
            Err(ProbeError::Transient("not used".into()))
        }

        fn stream_chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            let chunks = self.chunks.lock().unwrap().take().unwrap_or_default();
            futures::stream::iter(chunks)
        }

        fn model_id(&self) -> &str {
            "test-model"
        }

        fn provider(&self) -> &str {
            "test-provider"
        }
    }

    #[test]
    fn streaming_prompt_is_forceful() {
        assert!(FORCEFUL_READ_FILE.contains("Immediately call"));
        assert!(FORCEFUL_READ_FILE.contains("Do not describe what you would do"));
        assert!(FORCEFUL_READ_FILE.contains("Do not ask for confirmation"));
    }

    #[tokio::test]
    async fn streaming_strong_for_valid_tool_call() {
        let llm = StreamMockLlm::new(vec![
            Ok(ProbeStreamChunk::ToolCallStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "{\"path\":".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "\"/tmp/test.txt\"}".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallEnd),
        ]);
        let result = probe_streaming_tool_calls(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn streaming_medium_for_malformed_args() {
        let llm = StreamMockLlm::new(vec![
            Ok(ProbeStreamChunk::ToolCallStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "{\"path\": broken".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallEnd),
        ]);
        let result = probe_streaming_tool_calls(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn streaming_weak_for_no_tool_chunks() {
        let llm = StreamMockLlm::new(vec![Ok(ProbeStreamChunk::TextDelta {
            text: "I would read the file".to_string(),
        })]);
        let result = probe_streaming_tool_calls(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn streaming_timeout_before_chunks_is_err_not_weak() {
        let llm = StreamMockLlm::new(vec![Err(ProbeError::Transient("timeout".into()))]);
        let result = probe_streaming_tool_calls(&llm).await;
        assert!(
            result.is_err(),
            "timeout before chunks must be Err, not a completed Weak score; got {result:?}"
        );
        let err = result.expect_err("timeout before chunks");
        assert!(
            err.to_string().contains("timeout"),
            "error should preserve the timeout: {err}"
        );
    }

    #[tokio::test]
    async fn streaming_rate_limit_before_chunks_is_err_not_weak() {
        let llm = StreamMockLlm::new(vec![Err(ProbeError::RateLimit { retry_after: None })]);
        let result = probe_streaming_tool_calls(&llm).await;
        assert!(
            result.is_err(),
            "429 before chunks must be Err, not a completed Weak score; got {result:?}"
        );
    }
}
