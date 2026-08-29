//! One-shot ordered multi-tool plan probe (#1336).
//!
//! Measures whether the model emits several **heterogeneous** tool calls in
//! **one** response, in a dependency-respecting order (read -> edit -> run).
//!
//! This is **not** multi-turn agent-loop sequencing (see
//! `multi_turn_task_sequencing`). Strong agent models often score weakly
//! here by correctly starting with only `read_file` - that is expected.
//! Do **not** use this probe as an auto-architect or multi-step competence
//! signal; use `multi_turn_task_sequencing` instead.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{tool, user_text};

/// Probe one-shot ordered multi-tool planning (single LLM turn).
///
/// Presents a bug-fixing scenario that requires: (1) read the file,
/// (2) edit the file, (3) run tests. The model must produce tool calls
/// in a logically correct order.
///
/// Scoring:
/// - `1.0` - 3 precise tool calls in correct logical order (read -> edit -> run)
/// - `0.7` - 3 precise tools but wrong order
/// - `0.5` - 2 of 3 precise tools, or 3 names with imprecise args
/// - `0.3` - only one tool call (did not emit a multi-tool plan)
/// - `0.0` - no tool calls or only text response
pub async fn probe_one_shot_tool_plan<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let tools = vec![
        tool(
            "read_file",
            "Read the contents of a file.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path" }
                },
                "required": ["path"]
            }),
        ),
        tool(
            "edit_file",
            "Edit a file using search-and-replace.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path" },
                    "old_text": { "type": "string", "description": "Text to find" },
                    "new_text": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        ),
        tool(
            "run_command",
            "Execute a shell command.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run" }
                },
                "required": ["command"]
            }),
        ),
        tool(
            "list_dir",
            "List directory contents.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path" }
                },
                "required": ["path"]
            }),
        ),
    ];

    let request = ProbeRequest {
        messages: vec![user_text(
            "Fix the off-by-one bug in src/parser.rs: the loop condition on line 42 \
             uses `<` but should use `<=`. After fixing it, run the tests to verify.\n\n\
             Call the appropriate tools in the right order to complete this task.",
        )],
        tools,
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let calls = &response.tool_calls;
    let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();

    let nonempty = |args: &serde_json::Map<String, serde_json::Value>, key: &str| {
        args.get(key)
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty())
    };

    let has_read = calls
        .iter()
        .any(|c| c.name == "read_file" && nonempty(&c.arguments, "path"));
    let has_edit = calls.iter().any(|c| {
        c.name == "edit_file"
            && nonempty(&c.arguments, "path")
            && nonempty(&c.arguments, "old_text")
            && nonempty(&c.arguments, "new_text")
    });
    let has_run = calls
        .iter()
        .any(|c| c.name == "run_command" && nonempty(&c.arguments, "command"));

    let step_count = u8::from(has_read) + u8::from(has_edit) + u8::from(has_run);
    let named_count = u8::from(names.contains(&"read_file"))
        + u8::from(names.contains(&"edit_file"))
        + u8::from(names.contains(&"run_command"));

    let read_pos = names.iter().position(|n| *n == "read_file");
    let edit_pos = names.iter().position(|n| *n == "edit_file");
    let run_pos = names.iter().position(|n| *n == "run_command");

    let correct_order = match (read_pos, edit_pos, run_pos) {
        (Some(r), Some(e), Some(t)) => r < e && e < t,
        _ => false,
    };

    let (score, details) = if step_count == 3 && correct_order {
        (
            1.0,
            format!(
                "3 steps in correct order (read->edit->run): [{}]",
                names.join(", ")
            ),
        )
    } else if step_count == 3 {
        (
            0.7,
            format!("3 correct tools but wrong order: [{}]", names.join(", ")),
        )
    } else if step_count == 2 {
        (
            0.5,
            format!("2 of 3 expected steps: [{}]", names.join(", ")),
        )
    } else if named_count == 3 {
        (
            0.5,
            format!(
                "3 expected tools but arguments imprecise: [{}]",
                names.join(", ")
            ),
        )
    } else if !calls.is_empty() {
        (
            0.3,
            format!(
                "Only {} tool call(s), did not emit multi-tool plan: [{}]",
                calls.len(),
                names.join(", ")
            ),
        )
    } else {
        (0.0, "No tool calls, text-only response".to_string())
    };

    Ok(ProbeResult {
        name: "one_shot_tool_plan".to_string(),
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

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.as_object().unwrap().clone(),
        }
    }

    #[tokio::test]
    async fn reasoning_strong_for_correct_order() {
        let response = multi_tool_call_response(vec![
            call(
                "1",
                "read_file",
                serde_json::json!({"path": "src/parser.rs"}),
            ),
            call(
                "2",
                "edit_file",
                serde_json::json!({"path": "src/parser.rs", "old_text": "<", "new_text": "<="}),
            ),
            call(
                "3",
                "run_command",
                serde_json::json!({"command": "cargo test"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_one_shot_tool_plan(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn reasoning_medium_for_wrong_order() {
        let response = multi_tool_call_response(vec![
            call(
                "1",
                "edit_file",
                serde_json::json!({"path": "src/parser.rs", "old_text": "<", "new_text": "<="}),
            ),
            call(
                "2",
                "read_file",
                serde_json::json!({"path": "src/parser.rs"}),
            ),
            call(
                "3",
                "run_command",
                serde_json::json!({"command": "cargo test"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_one_shot_tool_plan(&llm).await.unwrap();
        assert_eq!(result.score, 0.7);
    }

    #[tokio::test]
    async fn reasoning_medium_for_empty_or_numeric_args() {
        let empty = multi_tool_call_response(vec![
            call("1", "read_file", serde_json::json!({})),
            call("2", "edit_file", serde_json::json!({})),
            call("3", "run_command", serde_json::json!({})),
        ]);
        let empty_result = probe_one_shot_tool_plan(&MockLlm { response: empty })
            .await
            .unwrap();
        assert_eq!(empty_result.score, 0.5);
        assert_ne!(empty_result.level, CapabilityLevel::Strong);

        let numeric = multi_tool_call_response(vec![
            call("1", "read_file", serde_json::json!({"path": 1})),
            call(
                "2",
                "edit_file",
                serde_json::json!({"path": 1, "old_text": 2, "new_text": 3}),
            ),
            call("3", "run_command", serde_json::json!({"cwd": "/tmp"})),
        ]);
        let numeric_result = probe_one_shot_tool_plan(&MockLlm { response: numeric })
            .await
            .unwrap();
        assert_eq!(numeric_result.score, 0.5);
        assert_ne!(numeric_result.level, CapabilityLevel::Strong);

        let wrong_order = multi_tool_call_response(vec![
            call("1", "run_command", serde_json::json!({})),
            call("2", "edit_file", serde_json::json!({"path": 1})),
            call("3", "read_file", serde_json::json!({})),
        ]);
        let wrong_order_result = probe_one_shot_tool_plan(&MockLlm {
            response: wrong_order,
        })
        .await
        .unwrap();
        assert_eq!(wrong_order_result.score, 0.5);
        assert_ne!(wrong_order_result.score, 0.7);
    }

    #[tokio::test]
    async fn reasoning_medium_for_two_steps() {
        let response = multi_tool_call_response(vec![
            call(
                "1",
                "edit_file",
                serde_json::json!({"path": "src/parser.rs", "old_text": "<", "new_text": "<="}),
            ),
            call(
                "2",
                "run_command",
                serde_json::json!({"command": "cargo test"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_one_shot_tool_plan(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn reasoning_weak_for_single_step() {
        let response = multi_tool_call_response(vec![call(
            "1",
            "edit_file",
            serde_json::json!({"path": "src/parser.rs", "old_text": "<", "new_text": "<="}),
        )]);
        let llm = MockLlm { response };
        let result = probe_one_shot_tool_plan(&llm).await.unwrap();
        assert_eq!(result.score, 0.3);
    }

    #[tokio::test]
    async fn reasoning_weak_for_text_only() {
        let llm = MockLlm {
            response: text_response("I would read the file, fix the bug, then run tests."),
        };
        let result = probe_one_shot_tool_plan(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
    }
}
