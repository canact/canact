//! Parallel tool-call scaling probe.
//!
//! Tests whether the model can emit 5+ tool calls in a single response.
//! The existing complex_tool_calling probe only tests 2 parallel calls.
//! Many coding tasks require reading 5-10 files at once, and models
//! that can only emit 1-2 calls per turn require extra round-trips.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest, ProbeTool};
use crate::types::{ProbeResult, classify};

use super::{
    has_visible_arg_text, refuse_truncated_incomplete, refuse_truncated_tool_call, tool, user_text,
};

/// First usable `path`, otherwise `file_path`. A blank or non-string `path`
/// must not hide a real `file_path`.
fn usable_read_path(arguments: &serde_json::Map<String, serde_json::Value>) -> Option<&str> {
    for key in ["path", "file_path"] {
        let Some(text) = arguments.get(key).and_then(|value| value.as_str()) else {
            continue;
        };
        if has_visible_arg_text(text) {
            return Some(text.trim());
        }
    }
    None
}

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
/// - `0.1` - named `read_file` but no usable path
/// - `0.0` - no tool calls
pub async fn probe_parallel_tool_scale<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    probe_parallel_tool_scale_with(llm, None).await
}

/// Same probe as [`probe_parallel_tool_scale`] when `tools` is `None`.
///
/// `Some` sends that list. A call counts when its name is in the list.
/// The 5 / 4 / 3 / 2 / 1 thresholds stay the same.
pub async fn probe_parallel_tool_scale_with<C: ProbeClient>(
    llm: &C,
    tools: Option<&[ProbeTool]>,
) -> Result<ProbeResult, ProbeError> {
    let builtin_prompt = "Read ALL FIVE of these files in a SINGLE response by calling \
                 read_file five times:\n\
                 1. src/main.rs\n\
                 2. src/lib.rs\n\
                 3. Cargo.toml\n\
                 4. README.md\n\
                 5. tests/integration.rs";
    let (request_tools, accepted, prompt, label) = if let Some(list) = tools {
        let accepted: Vec<String> = list.iter().map(|tool| tool.name.clone()).collect();
        let label = if accepted.is_empty() {
            "tool".to_owned()
        } else {
            accepted.join(", ")
        };
        let prompt = format!(
            "Read ALL FIVE of these files in a SINGLE response by calling \
             {label} five times:\n\
             1. src/main.rs\n\
             2. src/lib.rs\n\
             3. Cargo.toml\n\
             4. README.md\n\
             5. tests/integration.rs"
        );
        (list.to_vec(), accepted, prompt, label)
    } else {
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
        (
            vec![read_file],
            vec!["read_file".to_owned()],
            builtin_prompt.to_owned(),
            "read_file".to_owned(),
        )
    };

    let request = ProbeRequest {
        messages: vec![user_text(prompt)],
        tools: request_tools,
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    refuse_truncated_tool_call(&response)?;
    let calls = &response.tool_calls;

    let name_ok = |name: &str| accepted.iter().any(|accepted_name| accepted_name == name);
    let valid_calls: Vec<&str> = calls
        .iter()
        .filter(|call| name_ok(&call.name))
        .filter_map(|call| usable_read_path(&call.arguments))
        .collect();

    let mut unique_paths: Vec<&str> = valid_calls.clone();
    unique_paths.sort();
    unique_paths.dedup();
    let unique_count = unique_paths.len();

    let named = calls.iter().any(|call| name_ok(&call.name));
    let score = match unique_count {
        5.. => 1.0,
        4 => 0.8,
        3 => 0.6,
        2 => 0.4,
        1 => 0.2,
        _ if named => 0.1,
        _ => 0.0,
    };

    let details = if unique_count == 0 {
        if calls.is_empty() {
            "no tool calls in one response (target 5 unique read_file)".to_string()
        } else {
            format!(
                "{} tool call(s), 0 unique {label} paths (target 5)",
                calls.len()
            )
        }
    } else if valid_calls.len() == unique_count {
        format!("{unique_count} unique {label} calls in one response (target 5)")
    } else {
        format!(
            "{unique_count} unique of {} {label} calls in one response (target 5)",
            valid_calls.len()
        )
    };

    refuse_truncated_incomplete(response.finish, score)?;
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
    use crate::client::{ProbeResponse, ProbeToolCall};
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
    async fn weak_for_named_reads_with_numeric_paths() {
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
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.1);
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
        assert_eq!(result.score, 0.1);
        assert_eq!(result.level, CapabilityLevel::Weak);
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
    async fn one_good_call_outranks_blank_path() {
        let good = MockLlm {
            response: multi_tool_call_response(vec![read_file_call("1", "src/main.rs")]),
        };
        let blank = MockLlm {
            response: multi_tool_call_response(vec![read_file_call("1", "")]),
        };
        let g = probe_parallel_tool_scale(&good).await.unwrap();
        let b = probe_parallel_tool_scale(&blank).await.unwrap();
        assert!(
            g.score > b.score,
            "valid path must outrank blank path: good={} blank={}",
            g.score,
            b.score
        );
        assert_eq!(g.score, 0.2);
        assert_eq!(b.score, 0.1);
        assert_eq!(g.level, CapabilityLevel::Weak);
        assert_eq!(b.level, CapabilityLevel::Weak);
    }

    fn read_file_call_alias(id: &str, path: &str) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"file_path": path})
                .as_object()
                .unwrap()
                .clone(),
        }
    }

    fn read_file_call_path_and_alias(
        id: &str,
        path: serde_json::Value,
        file_path: &str,
    ) -> ProbeToolCall {
        let mut arguments = serde_json::Map::new();
        arguments.insert("path".to_string(), path);
        arguments.insert(
            "file_path".to_string(),
            serde_json::Value::String(file_path.to_string()),
        );
        ProbeToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments,
        }
    }

    #[tokio::test]
    async fn strong_when_unusable_path_hides_file_path() {
        let files = [
            "src/main.rs",
            "src/lib.rs",
            "Cargo.toml",
            "README.md",
            "tests/integration.rs",
        ];
        for (label, path_of) in [
            ("blank", serde_json::Value::String(String::new())),
            ("numeric", serde_json::json!(1)),
        ] {
            let calls = files
                .iter()
                .enumerate()
                .map(|(index, file_path)| {
                    let path = if label == "numeric" {
                        serde_json::json!(index + 1)
                    } else {
                        path_of.clone()
                    };
                    read_file_call_path_and_alias(&index.to_string(), path, file_path)
                })
                .collect();
            let llm = MockLlm {
                response: multi_tool_call_response(calls),
            };
            let result = probe_parallel_tool_scale(&llm).await.unwrap();
            assert_eq!(
                result.score, 1.0,
                "{label} path must not hide a usable file_path: {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{label}");
        }
    }

    #[tokio::test]
    async fn strong_for_five_file_path_alias_calls() {
        let response = multi_tool_call_response(vec![
            read_file_call_alias("1", "src/main.rs"),
            read_file_call_alias("2", "src/lib.rs"),
            read_file_call_alias("3", "Cargo.toml"),
            read_file_call_alias("4", "README.md"),
            read_file_call_alias("5", "tests/integration.rs"),
        ]);
        let llm = MockLlm { response };
        let result = probe_parallel_tool_scale(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "file_path alias must count unique paths: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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

    fn calls_named(name: &str, count: usize) -> ProbeResponse {
        let paths = [
            "src/main.rs",
            "src/lib.rs",
            "Cargo.toml",
            "README.md",
            "tests/integration.rs",
        ];
        let made = (0..count)
            .map(|index| ProbeToolCall {
                id: index.to_string(),
                name: name.into(),
                arguments: serde_json::json!({"path": paths[index]})
                    .as_object()
                    .unwrap()
                    .clone(),
            })
            .collect();
        multi_tool_call_response(made)
    }

    #[tokio::test]
    async fn parallel_tool_scale_uses_caller_tool_names() {
        let caller = vec![super::tool(
            "fetch_blob",
            "Fetch a blob by path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];
        let llm = RecordingMock::new(calls_named("fetch_blob", 5));
        let result = probe_parallel_tool_scale_with(&llm, Some(&caller))
            .await
            .expect("caller scale");
        assert_eq!(result.score, 1.0);
        let recorded = llm.requests.lock().expect("lock").clone();
        let prompt = request_user_text(&recorded[0]);
        assert!(prompt.contains("fetch_blob"), "{prompt}");
        assert_eq!(
            recorded[0]
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["fetch_blob"]
        );
        for (count, score) in [(4, 0.8_f32), (3, 0.6), (2, 0.4), (1, 0.2)] {
            let llm = RecordingMock::new(calls_named("fetch_blob", count));
            let result = probe_parallel_tool_scale_with(&llm, Some(&caller))
                .await
                .expect("caller scale");
            assert_eq!(result.score, score, "caller count {count}");
        }
        for (count, score) in [(5, 1.0_f32), (4, 0.8), (3, 0.6), (2, 0.4), (1, 0.2)] {
            let llm = MockLlm {
                response: calls_named("read_file", count),
            };
            let result = probe_parallel_tool_scale(&llm)
                .await
                .expect("builtin scale");
            assert_eq!(result.score, score, "builtin count {count}");
        }
    }
}
