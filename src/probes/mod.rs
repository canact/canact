//! Individual capability probes.
//!
//! Each probe sends a single lightweight request to the LLM and scores the
//! response along one capability dimension.

mod parallel_tool_scale;
mod streaming_tool_calls;
mod tool_calling;
mod vision;
mod xml_fallback;

pub use parallel_tool_scale::probe_parallel_tool_scale;
pub use streaming_tool_calls::probe_streaming_tool_calls;
pub use tool_calling::{
    probe_complex_tool_calling, probe_nested_arguments, probe_tool_calling, probe_tool_selection,
};
pub use vision::probe_vision;
pub use xml_fallback::probe_xml_tool_calling;

use crate::client::{ProbeContent, ProbeMessage, ProbeRole, ProbeTool};

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

pub(crate) fn tool(name: &str, description: &str, parameters: serde_json::Value) -> ProbeTool {
    ProbeTool {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::future::Future;

    use crate::client::{
        ProbeClient, ProbeFinish, ProbeRequest, ProbeResponse, ProbeStreamChunk, ProbeToolCall,
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
        }
    }

    pub(crate) fn text_response(text: &str) -> ProbeResponse {
        ProbeResponse {
            text: text.to_string(),
            tool_calls: Vec::new(),
            finish: ProbeFinish::Stop,
        }
    }

    pub(crate) fn multi_tool_call_response(calls: Vec<ProbeToolCall>) -> ProbeResponse {
        ProbeResponse {
            text: String::new(),
            tool_calls: calls,
            finish: ProbeFinish::ToolCalls,
        }
    }
}
