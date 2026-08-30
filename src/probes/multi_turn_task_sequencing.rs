//! Multi-turn task sequencing probe (#1337).
//!
//! Simulates a short agent loop with synthetic tool results and scores
//! whether the model continues a dependent read -> edit -> verify chain
//! across turns. This is the decision-grade signal for multi-step agent
//! competence, unlike [`super::one_shot_tool_plan`].

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest, ProbeToolCall};
use crate::types::{ProbeResult, classify};

use super::{
    assistant_tool_calls, nonempty_string_arg, refuse_truncated_tool_call, tool, tool_result,
    user_text,
};

fn tool_specs() -> Vec<crate::client::ProbeTool> {
    vec![
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
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        ),
        tool(
            "run_command",
            "Run a shell command (e.g. tests).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        ),
        tool(
            "list_dir",
            "List a directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        ),
    ]
}

fn first_tool_name(calls: &[ProbeToolCall]) -> Option<&str> {
    calls.first().map(|t| t.name.as_str())
}

fn is_precise_read(c: &ProbeToolCall) -> bool {
    c.name == "read_file" && nonempty_string_arg(&c.arguments, "path")
}

fn is_precise_edit(c: &ProbeToolCall) -> bool {
    c.name == "edit_file"
        && nonempty_string_arg(&c.arguments, "path")
        && nonempty_string_arg(&c.arguments, "old_string")
        && nonempty_string_arg(&c.arguments, "new_string")
}

fn is_precise_run(c: &ProbeToolCall) -> bool {
    c.name == "run_command" && nonempty_string_arg(&c.arguments, "command")
}

/// Probe multi-turn dependent tool sequencing with synthetic tool results.
///
/// Loop (up to 3 model turns):
/// 1. User asks to fix a bug then run tests.
/// 2. After `read_file`, inject a synthetic file that still has the bug.
/// 3. After `edit_file`, inject success.
/// 4. After `run_command`, stop.
///
/// Scoring:
/// - `1.0` - observed read, then edit, then run across turns, each with
///   nonempty string args (`path`; `path`+`old_string`+`new_string`; `command`)
/// - `0.7` - read then edit (missing verify) or read then run (skipped explicit edit)
/// - `0.5` - full name chain with empty, whitespace, or imprecise args
/// - `0.3` - sensible first tool only (`read_file` or `edit_file`) then stops
/// - `0.0` - no tools or wrong-only start
pub async fn probe_multi_turn_task_sequencing<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let tools = tool_specs();
    let mut messages = vec![user_text(
        "There is an off-by-one bug in `src/parser.rs`: the loop on line 42 \
         uses `<` but should use `<=`. Fix the bug, then run the tests \
         (`cargo test -p parser`) to verify.\n\n\
         Use tools. You will receive tool results between turns. Do not invent \
         file contents — read the file first if you need it.",
    )];

    let mut saw_read = false;
    let mut saw_edit = false;
    let mut saw_run = false;
    let mut precise_read = false;
    let mut precise_edit = false;
    let mut precise_run = false;
    let mut first_tool: Option<String> = None;
    let mut turns = 0u32;

    for _ in 0..3 {
        turns += 1;
        let request = ProbeRequest {
            messages: messages.clone(),
            tools: tools.clone(),
            model: llm.model_id().to_string(),
            temperature: Some(0.0),
            max_tokens: Some(512),
        };
        let response = llm.chat(request).await?;
        refuse_truncated_tool_call(&response)?;
        let calls = response.tool_calls;
        if calls.is_empty() {
            break;
        }

        let name = first_tool_name(&calls).unwrap_or("").to_string();
        if first_tool.is_none() {
            first_tool = Some(name);
        }

        messages.push(assistant_tool_calls(response.text, calls.clone()));

        for call in &calls {
            let tool_name = call.name.as_str();
            let result_text = match tool_name {
                "read_file" => {
                    saw_read = true;
                    precise_read |= is_precise_read(call);
                    "fn parse(items: &[u8]) -> usize {\n    let mut i = 0;\n    while i < items.len() { // bug: should be <=\n        i += 1;\n    }\n    i\n}\n".to_string()
                }
                "edit_file" => {
                    saw_edit = true;
                    precise_edit |= is_precise_edit(call);
                    "ok: file updated".to_string()
                }
                "run_command" => {
                    saw_run = true;
                    precise_run |= is_precise_run(call);
                    "test result: ok. 3 passed".to_string()
                }
                other => format!("ok: {other}"),
            };
            messages.push(tool_result(call.id.clone(), result_text));
        }

        if saw_run {
            break;
        }
    }

    let (score, details) = match (saw_read, saw_edit, saw_run) {
        (true, true, true) if precise_read && precise_edit && precise_run => (
            1.0,
            format!("Completed read → edit → verify across {turns} turn(s)"),
        ),
        (true, true, true) => (
            0.5,
            format!("Completed read → edit → verify but arguments imprecise (turns={turns})"),
        ),
        (true, true, false) => (
            0.7,
            format!("Completed read → edit but did not verify (turns={turns})"),
        ),
        (true, false, true) => (
            0.7,
            format!("Read then verify without explicit edit (turns={turns})"),
        ),
        (true, false, false) => (
            0.3,
            format!(
                "Started with read_file but did not continue after tool result (turns={turns})"
            ),
        ),
        (false, true, true) => (
            0.7,
            format!("Edited and verified without read (turns={turns})"),
        ),
        (false, true, false) => (
            0.3,
            format!("Started with edit_file but incomplete chain (turns={turns})"),
        ),
        (false, false, true) => (
            0.3,
            format!("Ran command without prior fix steps (turns={turns})"),
        ),
        (false, false, false) => {
            let first = first_tool.as_deref().unwrap_or("(none)");
            (
                0.0,
                format!("No progress on task chain; first tool={first} turns={turns}"),
            )
        }
    };

    Ok(ProbeResult {
        name: "multi_turn_task_sequencing".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    fn tc(name: &str, id: &str) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::Map::new(),
        }
    }

    fn tc_args(name: &str, id: &str, arguments: serde_json::Value) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.as_object().unwrap().clone(),
        }
    }

    fn precise_tc(name: &str, id: &str) -> ProbeToolCall {
        match name {
            "read_file" => tc_args(name, id, serde_json::json!({"path": "src/parser.rs"})),
            "edit_file" => tc_args(
                name,
                id,
                serde_json::json!({
                    "path": "src/parser.rs",
                    "old_string": "<",
                    "new_string": "<="
                }),
            ),
            "run_command" => tc_args(
                name,
                id,
                serde_json::json!({"command": "cargo test -p parser"}),
            ),
            _ => tc(name, id),
        }
    }

    fn tool_resp(calls: Vec<ProbeToolCall>) -> crate::client::ProbeResponse {
        multi_tool_call_response(calls)
    }

    #[tokio::test]
    async fn strong_for_full_chain() {
        let llm = SequentialMock::new(vec![
            tool_resp(vec![precise_tc("read_file", "1")]),
            tool_resp(vec![precise_tc("edit_file", "2")]),
            tool_resp(vec![precise_tc("run_command", "3")]),
        ]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn empty_maps_full_chain_is_not_strong() {
        let llm = SequentialMock::new(vec![
            tool_resp(vec![tc("read_file", "1")]),
            tool_resp(vec![tc("edit_file", "2")]),
            tool_resp(vec![tc("run_command", "3")]),
        ]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert!((result.score - 0.5).abs() < f32::EPSILON);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn whitespace_args_full_chain_is_not_strong() {
        let llm = SequentialMock::new(vec![
            tool_resp(vec![tc_args(
                "read_file",
                "1",
                serde_json::json!({"path": " "}),
            )]),
            tool_resp(vec![tc_args(
                "edit_file",
                "2",
                serde_json::json!({
                    "path": "\n",
                    "old_string": " ",
                    "new_string": "\t"
                }),
            )]),
            tool_resp(vec![tc_args(
                "run_command",
                "3",
                serde_json::json!({"command": "  "}),
            )]),
        ]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert!((result.score - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn medium_for_read_only_then_stop() {
        let llm = SequentialMock::new(vec![
            tool_resp(vec![tc("read_file", "1")]),
            text_response("I need more context."),
        ]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert!((result.score - 0.3).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn weak_for_no_tools() {
        let llm = SequentialMock::new(vec![text_response("fixed it")]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert!((result.score - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn strongish_for_read_edit_without_run() {
        let llm = SequentialMock::new(vec![
            tool_resp(vec![tc("read_file", "1")]),
            tool_resp(vec![tc("edit_file", "2")]),
            text_response("done"),
        ]);
        let result = probe_multi_turn_task_sequencing(&llm).await.unwrap();
        assert!((result.score - 0.7).abs() < f32::EPSILON);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }
}
