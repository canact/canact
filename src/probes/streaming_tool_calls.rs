//! Streaming tool-call probe.
//!
//! Tests whether the model can emit tool calls via the streaming API
//! without producing malformed argument chunks. Models that fail this
//! should use non-streaming for tool-call turns.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest, ProbeStreamChunk};
use crate::types::{ProbeResult, classify};
use futures::StreamExt;

use super::{nonempty_string_arg, tool, user_text};

const FORCEFUL_READ_FILE: &str = "Immediately call the read_file tool with path /tmp/test.txt. Do not describe what you would do. Do not ask for confirmation.";

/// Probe whether the model can stream tool calls reliably.
///
/// Sends a streaming request with a simple tool and checks whether
/// the streamed chunks assemble into a valid tool call.
///
/// Scoring:
/// - `1.0` - stream completed, tool call name and JSON `path` as a string
/// - `0.5` - stream completed with tool call name but malformed args or non-string `path`
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

    let mut got_read_file = false;
    let mut got_other_tool = false;
    let mut args_buffer = String::new();
    let mut got_any_chunk = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                got_any_chunk = true;
                match &chunk {
                    ProbeStreamChunk::ToolCallStart { name, .. } => {
                        if name == "read_file" {
                            got_read_file = true;
                        } else if !name.is_empty() {
                            got_other_tool = true;
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
    } else if got_read_file {
        match parsed_string_path(&args_buffer) {
            Ok(true) => (
                1.0,
                "Streaming tool call with valid JSON arguments".to_string(),
            ),
            Ok(false) => (
                0.5,
                "Tool call name streamed but path is not a string".to_string(),
            ),
            Err(_) => (
                0.5,
                format!(
                    "Tool call name streamed but arguments malformed: {:?}",
                    super::utf8_prefix(&args_buffer, 80)
                ),
            ),
        }
    } else if got_other_tool {
        (0.5, "Streamed a tool name other than read_file".to_string())
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

fn parsed_string_path(args: &str) -> Result<bool, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(args)?;
    Ok(value
        .as_object()
        .is_some_and(|o| nonempty_string_arg(o, "path")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeResponse, ProbeStreamChunk};
    use crate::probes::test_support::request_user_text;
    use crate::types::CapabilityLevel;
    use futures::Stream;

    struct StreamMockLlm {
        chunks: std::sync::Mutex<Option<Vec<Result<ProbeStreamChunk, ProbeError>>>>,
        requests: std::sync::Mutex<Vec<ProbeRequest>>,
    }

    impl StreamMockLlm {
        fn new(chunks: Vec<Result<ProbeStreamChunk, ProbeError>>) -> Self {
            Self {
                chunks: std::sync::Mutex::new(Some(chunks)),
                requests: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl ProbeClient for StreamMockLlm {
        async fn chat(&self, req: ProbeRequest) -> Result<ProbeResponse, ProbeError> {
            self.requests.lock().expect("lock").push(req);
            Err(ProbeError::Transient("not used".into()))
        }

        fn stream_chat(
            &self,
            req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            self.requests.lock().expect("lock").push(req);
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

    #[tokio::test]
    async fn streaming_prompt_is_forceful() {
        let llm = StreamMockLlm::new(vec![
            Ok(ProbeStreamChunk::ToolCallStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "{\"path\":\"/tmp/test.txt\"}".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallEnd),
        ]);
        let _ = probe_streaming_tool_calls(&llm).await.unwrap();
        let rec = llm.requests.lock().expect("lock");
        assert_eq!(rec.len(), 1);
        let user = request_user_text(&rec[0]);
        assert!(user.contains("Immediately call"), "{user}");
        assert!(user.contains("Do not describe what you would do"), "{user}");
        assert!(user.contains("Do not ask for confirmation"), "{user}");
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
    async fn streaming_wrong_name_with_string_path_is_not_strong() {
        let llm = StreamMockLlm::new(vec![
            Ok(ProbeStreamChunk::ToolCallStart {
                id: "call_1".to_string(),
                name: "write_file".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "{\"path\":\"/tmp/test.txt\"}".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallEnd),
        ]);
        let result = probe_streaming_tool_calls(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn streaming_empty_or_whitespace_path_is_not_strong() {
        for args in [r#"{"path":""}"#, r#"{"path":" "}"#, r#"{"path":"\n"}"#] {
            let llm = StreamMockLlm::new(vec![
                Ok(ProbeStreamChunk::ToolCallStart {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                }),
                Ok(ProbeStreamChunk::ToolCallArgDelta {
                    delta: args.to_string(),
                }),
                Ok(ProbeStreamChunk::ToolCallEnd),
            ]);
            let result = probe_streaming_tool_calls(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "empty/whitespace path must not be Strong: {args}"
            );
            assert_eq!(result.score, 0.5, "path args={args}");
            assert_eq!(result.level, CapabilityLevel::Medium, "path args={args}");
        }
    }

    #[tokio::test]
    async fn streaming_numeric_path_is_not_strong() {
        let llm = StreamMockLlm::new(vec![
            Ok(ProbeStreamChunk::ToolCallStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallArgDelta {
                delta: "{\"path\":1}".to_string(),
            }),
            Ok(ProbeStreamChunk::ToolCallEnd),
        ]);
        let result = probe_streaming_tool_calls(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.level, CapabilityLevel::Medium);
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
