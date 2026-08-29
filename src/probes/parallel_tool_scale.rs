//! Parallel tool-call scaling probe.
//!
//! Tests whether the model can emit 5+ tool calls in a single response.
//! The existing complex_tool_calling probe only tests 2 parallel calls.
//! Many coding tasks require reading 5-10 files at once, and models
//! that can only emit 1-2 calls per turn require extra round-trips.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{nonempty_string_arg, tool, user_text};

/// Probe whether the model can produce 5 parallel tool calls.
///
/// Provides a `read_file` tool and asks the model to read 5 specific files
/// in a single response.
///
/// Scoring:
/// - `1.0` - 5 correct `read_file` calls with distinct paths
/// - `0.8` - 4 calls
/// - `0.6` - 3 calls
/// - `0.4` - 2 calls
/// - `0.2` - 1 call
/// - `0.0` - no tool calls
pub async fn probe_parallel_tool_scale<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let read_file = tool(
        "read_file",
        "Read the contents of a file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The file path to read" }
            },
            "required": ["path"]
        }),
    );

    let request = ProbeRequest {
        messages: vec![user_text(
            "Read ALL FIVE of these files in a SINGLE response by calling \
                 read_file five times:\n\
                 1. src/main.rs\n\
                 2. src/lib.rs\n\
                 3. Cargo.toml\n\
                 4. README.md\n\
                 5. tests/integration.rs",
        )],
        tools: vec![read_file],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let calls = &response.tool_calls;

    let valid_calls: Vec<&str> = calls
        .iter()
        .filter(|c| c.name == "read_file" && nonempty_string_arg(&c.arguments, "path"))
        .filter_map(|c| c.arguments.get("path").and_then(|v| v.as_str()))
        .collect();

    let mut unique_paths: Vec<&str> = valid_calls.clone();
    unique_paths.sort();
    unique_paths.dedup();
    let unique_count = unique_paths.len();

    let named_read_file = calls.iter().any(|c| c.name == "read_file");
    let score = match unique_count {
        5.. => 1.0,
        4 => 0.8,
        3 => 0.6,
        2 => 0.4,
        1 => 0.2,
        _ if named_read_file => 0.5,
        _ => 0.0,
    };

    let details = if unique_count == 0 {
        if calls.is_empty() {
            "no tool calls in one response (target 5 unique read_file)".to_string()
        } else {
            format!(
                "{} tool call(s), 0 unique read_file paths (target 5)",
                calls.len()
            )
        }
    } else if valid_calls.len() == unique_count {
        format!("{unique_count} unique read_file calls in one response (target 5)")
    } else {
        format!(
            "{unique_count} unique of {} read_file calls in one response (target 5)",
            valid_calls.len()
        )
    };

    Ok(ProbeResult {
        name: "parallel_tool_scale".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProbeToolCall;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    fn read_file_call(id: &str, path: &str) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": path})
                .as_object()
                .unwrap()
                .clone(),
        }
    }

    #[tokio::test]
    async fn strong_for_five_calls() {
        let response = multi_tool_call_response(vec![
            read_file_call("1", "src/main.rs"),
            read_file_call("2", "src/lib.rs"),
            read_file_call("3", "Cargo.toml"),
            read_file_call("4", "README.md"),
            read_file_call("5", "tests/integration.rs"),
        ]);
        let llm = MockLlm { response };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
        assert_eq!(
            result.details,
            "5 unique read_file calls in one response (target 5)"
        );
        assert!(!result.details.contains("src/main.rs"));
    }

    #[tokio::test]
    async fn medium_for_named_reads_with_numeric_paths() {
        let response = multi_tool_call_response(vec![
            ProbeToolCall {
                id: "1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": 1}).as_object().unwrap().clone(),
            },
            ProbeToolCall {
                id: "2".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": 2}).as_object().unwrap().clone(),
            },
        ]);
        let llm = MockLlm { response };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn not_strong_for_five_distinct_blank_paths() {
        let response = multi_tool_call_response(vec![
            read_file_call("1", ""),
            read_file_call("2", " "),
            read_file_call("3", "  "),
            read_file_call("4", "\t"),
            read_file_call("5", "\n"),
        ]);
        let llm = MockLlm { response };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn weak_for_text_only() {
        let llm = MockLlm {
            response: text_response("I would read those files for you."),
        };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert!(result.details.contains("no tool calls"));
        assert!(!result.details.contains('['));
    }

    #[tokio::test]
    async fn medium_for_two_calls() {
        let response = multi_tool_call_response(vec![
            read_file_call("1", "src/main.rs"),
            read_file_call("2", "src/lib.rs"),
        ]);
        let llm = MockLlm { response };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.4);
        assert_eq!(
            result.details,
            "2 unique read_file calls in one response (target 5)"
        );
    }
}
