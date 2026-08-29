//! Individual capability probes.
//!
//! Each probe sends a single lightweight request to the LLM and scores the
//! response along one capability dimension.

mod code_syntax;
mod context_faithfulness;
mod context_ladder;
mod edit_format;
mod json_output;
mod max_tokens_compliance;
mod multi_turn_memory;
mod multi_turn_task_sequencing;
mod one_shot_tool_plan;
mod parallel_tool_scale;
mod streaming_tool_calls;
mod system_message_adherence;
mod token_efficiency;
mod tool_calling;
mod vision;
mod xml_fallback;

pub use code_syntax::probe_code_syntax;
pub use context_faithfulness::probe_context_faithfulness;
pub use context_ladder::{ContextLadder, probe_effective_context_tokens};
pub use edit_format::{probe_search_replace, probe_unified_diff};
pub use json_output::{probe_instruction_following, probe_json_output};
pub use max_tokens_compliance::probe_max_tokens_compliance;
pub use multi_turn_memory::probe_multi_turn_memory;
pub use multi_turn_task_sequencing::probe_multi_turn_task_sequencing;
pub use one_shot_tool_plan::probe_one_shot_tool_plan;
pub use parallel_tool_scale::probe_parallel_tool_scale;
pub use streaming_tool_calls::probe_streaming_tool_calls;
pub use system_message_adherence::probe_system_message_adherence;
pub use token_efficiency::probe_token_efficiency;
pub use tool_calling::{
    probe_complex_tool_calling, probe_nested_arguments, probe_tool_calling, probe_tool_selection,
};
pub use vision::probe_vision;
pub use xml_fallback::probe_xml_tool_calling;

use crate::client::{ProbeContent, ProbeMessage, ProbeRole, ProbeTool, ProbeToolCall};

pub(crate) fn user_text(text: impl Into<String>) -> ProbeMessage {
    ProbeMessage {
        role: ProbeRole::User,
        content: ProbeContent::Text(text.into()),
        tool_calls: None,
        tool_call_id: None,
    }
}

pub(crate) fn system_text(text: impl Into<String>) -> ProbeMessage {
    ProbeMessage {
        role: ProbeRole::System,
        content: ProbeContent::Text(text.into()),
        tool_calls: None,
        tool_call_id: None,
    }
}

pub(crate) fn assistant_text(text: impl Into<String>) -> ProbeMessage {
    ProbeMessage {
        role: ProbeRole::Assistant,
        content: ProbeContent::Text(text.into()),
        tool_calls: None,
        tool_call_id: None,
    }
}

pub(crate) fn assistant_tool_calls(
    text: impl Into<String>,
    tool_calls: Vec<ProbeToolCall>,
) -> ProbeMessage {
    ProbeMessage {
        role: ProbeRole::Assistant,
        content: ProbeContent::Text(text.into()),
        tool_calls: Some(tool_calls),
        tool_call_id: None,
    }
}

pub(crate) fn tool_result(
    tool_call_id: impl Into<String>,
    text: impl Into<String>,
) -> ProbeMessage {
    ProbeMessage {
        role: ProbeRole::Tool,
        content: ProbeContent::Text(text.into()),
        tool_calls: None,
        tool_call_id: Some(tool_call_id.into()),
    }
}

/// Try to isolate a JSON object from text that may be wrapped in markdown
/// fences or surrounded by prose.
pub(crate) fn extract_json_from_text(text: &str) -> &str {
    let trimmed = text.trim();

    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }

    match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if end > start => return &trimmed[start..=end],
        _ => {}
    }

    trimmed
}

pub(crate) fn tool(name: &str, description: &str, parameters: serde_json::Value) -> ProbeTool {
    ProbeTool {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::Mutex;

    use crate::client::{
        ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeRequest, ProbeResponse,
        ProbeRole, ProbeStreamChunk, ProbeToolCall,
    };
    use crate::error::ProbeError;
    use futures::Stream;

    pub(crate) struct MockLlm {
        pub(crate) response: ProbeResponse,
    }

    impl ProbeClient for MockLlm {
        fn chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
            let resp = self.response.clone();
            async move { Ok(resp) }
        }

        fn stream_chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            futures::stream::empty()
        }

        fn model_id(&self) -> &str {
            "test-model"
        }

        fn provider(&self) -> &str {
            "test-provider"
        }
    }

    pub(crate) struct SequentialMock {
        responses: Mutex<VecDeque<ProbeResponse>>,
    }

    impl SequentialMock {
        pub(crate) fn new(responses: Vec<ProbeResponse>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    impl ProbeClient for SequentialMock {
        fn chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
            let next = self
                .responses
                .lock()
                .expect("sequential mock lock")
                .pop_front()
                .unwrap_or_else(|| text_response("done"));
            async move { Ok(next) }
        }

        fn stream_chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            futures::stream::empty()
        }

        fn model_id(&self) -> &str {
            "test-model"
        }

        fn provider(&self) -> &str {
            "test-provider"
        }
    }

    pub(crate) fn tool_call_response() -> ProbeResponse {
        ProbeResponse {
            text: String::new(),
            tool_calls: vec![ProbeToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "/tmp/test.txt"})
                    .as_object()
                    .unwrap()
                    .clone(),
            }],
            finish: ProbeFinish::ToolCalls,
            usage: None,
        }
    }

    pub(crate) fn text_response(text: &str) -> ProbeResponse {
        ProbeResponse {
            text: text.to_string(),
            tool_calls: Vec::new(),
            finish: ProbeFinish::Stop,
            usage: None,
        }
    }

    pub(crate) fn multi_tool_call_response(calls: Vec<ProbeToolCall>) -> ProbeResponse {
        ProbeResponse {
            text: String::new(),
            tool_calls: calls,
            finish: ProbeFinish::ToolCalls,
            usage: None,
        }
    }

    pub(crate) struct RecordingMock {
        inner: MockLlm,
        pub(crate) requests: Mutex<Vec<ProbeRequest>>,
    }

    impl RecordingMock {
        pub(crate) fn new(response: ProbeResponse) -> Self {
            Self {
                inner: MockLlm { response },
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ProbeClient for RecordingMock {
        fn chat(
            &self,
            req: ProbeRequest,
        ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
            self.requests.lock().expect("lock").push(req.clone());
            self.inner.chat(req)
        }

        fn stream_chat(
            &self,
            req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            self.requests.lock().expect("lock").push(req.clone());
            self.inner.stream_chat(req)
        }

        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        fn provider(&self) -> &str {
            self.inner.provider()
        }
    }

    pub(crate) fn request_user_text(req: &ProbeRequest) -> String {
        let mut out = String::new();
        for message in &req.messages {
            if message.role != ProbeRole::User {
                continue;
            }
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
}

#[cfg(test)]
mod extract_json_tests {
    use super::extract_json_from_text;

    #[test]
    fn extract_json_from_bare_object() {
        assert_eq!(extract_json_from_text(r#"  {"a": 1}  "#), r#"{"a": 1}"#);
    }

    #[test]
    fn extract_json_from_fenced_block() {
        let input = "```json\n{\"a\": 1}\n```";
        assert_eq!(extract_json_from_text(input), "{\"a\": 1}");
    }
}
