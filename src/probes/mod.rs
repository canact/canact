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

    if let Some(body) = fenced_json_body(trimmed) {
        return body;
    }

    let obj = trimmed.find('{');
    let arr = trimmed.find('[');
    match (obj, arr) {
        (Some(o), Some(a)) if a < o => {
            if let Some(end) = trimmed.rfind(']') {
                if end > a {
                    return &trimmed[a..=end];
                }
            }
        }
        (Some(start), _) => {
            if let Some(end) = trimmed.rfind('}') {
                if end > start {
                    return &trimmed[start..=end];
                }
            }
        }
        (None, Some(start)) => {
            if let Some(end) = trimmed.rfind(']') {
                if end > start {
                    return &trimmed[start..=end];
                }
            }
        }
        _ => {}
    }

    trimmed
}

fn fenced_json_body(trimmed: &str) -> Option<&str> {
    let start = trimmed.find("```")?;
    let after = &trimmed[start + 3..];
    let (inner, _) = after.split_once("```")?;
    let inner = inner.trim_start_matches('\r');
    let body = if let Some((first, rest)) = inner.split_once('\n') {
        let tag = first.trim().trim_end_matches('\r');
        if is_fence_language_tag(tag) {
            rest
        } else {
            inner
        }
    } else {
        inner
    };
    let body = body.trim();
    if body.starts_with('{') || body.starts_with('[') {
        Some(body)
    } else {
        None
    }
}

fn is_fence_language_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '_'))
}

/// Prefix that never splits a UTF-8 code point.
pub(crate) fn utf8_prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// True when `s` has a character that is not whitespace or a format/ZWSP mark.
fn has_visible_arg_text(s: &str) -> bool {
    s.chars().any(|c| {
        !c.is_whitespace()
            && !matches!(
                c,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}'
            )
    })
}

/// True when `key` is a string with at least one non-whitespace character.
pub(crate) fn nonempty_string_arg(
    args: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> bool {
    args.get(key)
        .and_then(|v| v.as_str())
        .is_some_and(has_visible_arg_text)
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
mod nonempty_string_arg_tests {
    use super::nonempty_string_arg;

    fn args(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn nonempty_string_arg_rejects_empty_and_whitespace() {
        assert!(!nonempty_string_arg(
            &args(serde_json::json!({"path": ""})),
            "path"
        ));
        assert!(!nonempty_string_arg(
            &args(serde_json::json!({"path": " "})),
            "path"
        ));
        assert!(!nonempty_string_arg(
            &args(serde_json::json!({"path": "\n"})),
            "path"
        ));
        assert!(!nonempty_string_arg(
            &args(serde_json::json!({"path": 1})),
            "path"
        ));
        assert!(!nonempty_string_arg(&args(serde_json::json!({})), "path"));
        assert!(!nonempty_string_arg(
            &args(serde_json::json!({"path": "\u{200b}"})),
            "path"
        ));
        assert!(nonempty_string_arg(
            &args(serde_json::json!({"path": "/tmp/a"})),
            "path"
        ));
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

    #[test]
    fn extract_json_from_uppercase_json_fence() {
        let input = "```JSON\n{\"a\": 1}\n```";
        assert_eq!(extract_json_from_text(input), "{\"a\": 1}");
    }

    #[test]
    fn extract_json_from_jsonc_fence_falls_through_to_object() {
        let input = "```jsonc\n{\"a\": 1}\n```";
        assert_eq!(extract_json_from_text(input), "{\"a\": 1}");
    }

    #[test]
    fn extract_json_does_not_peel_array_wrapper() {
        let input = r#"[{"word": "hello", "length": 5, "reversed": "olleh"}]"#;
        assert_eq!(extract_json_from_text(input), input);
    }

    #[test]
    fn extract_json_keeps_array_when_prose_wraps_it() {
        let input = r#"Here: [{"word": "hello", "length": 5, "reversed": "olleh"}]"#;
        assert_eq!(
            extract_json_from_text(input),
            r#"[{"word": "hello", "length": 5, "reversed": "olleh"}]"#
        );
    }
}
