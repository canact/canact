//! Tool-calling capability probes: basic, complex, nested, and selection.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest, ProbeToolCall};
use crate::types::{CapabilityLevel, ProbeResult, classify};

use super::{nonempty_string_arg, tool, user_text};

const FORCEFUL_READ_FILE: &str = "Immediately call the read_file tool with path /tmp/test.txt. Do not describe what you would do. Do not ask for confirmation.";

const FORCEFUL_EDIT_FILE: &str = "Immediately call the edit_file tool. Do not describe what you would do. Do not ask for confirmation.\n\
                 Use the edit_file tool to edit /tmp/app.py and make two changes:\n\
                 1. Replace \"Hello\" with \"Hi\"\n\
                 2. Replace \"World\" with \"Earth\"";

/// Probe whether the model can produce proper native tool calls.
///
/// Sends a request with a `read_file` tool spec and asks the model to use it.
/// Scoring:
/// - `1.0` - valid tool call with the correct name and `path` as a string
/// - `0.5` - tool call present but wrong name, missing `path`, or `path` not a string
/// - `0.0` - text-only response (no tool call emitted)
pub async fn probe_tool_calling<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
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

    let response = llm.chat(request).await?;

    let (score, details) = if !response.tool_calls.is_empty() {
        let tc = &response.tool_calls[0];
        let path_is_string = nonempty_string_arg(&tc.arguments, "path");
        if tc.name == "read_file" && path_is_string {
            (
                1.0,
                "Valid tool call with correct name and arguments".to_string(),
            )
        } else {
            (
                0.5,
                format!("Tool call present but imprecise: name={}", tc.name),
            )
        }
    } else {
        (0.0, "No tool call in response, text only".to_string())
    };

    Ok(ProbeResult {
        name: "tool_calling".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model can choose the right tool from multiple options
/// and produce multiple tool calls in one response.
///
/// Provides 5 tool specs and asks the model to perform two actions that
/// require two different tools.
///
/// Scoring:
/// - `1.0` - two correct tool calls with correct names and required arguments
/// - `0.8` - two tool calls, correct names, but one has extra/missing params
/// - `0.5` - only one correct tool call
/// - `0.3` - tool calls present but wrong names or merged parameters
/// - `0.0` - no tool calls or calls to hallucinated tools
pub async fn probe_complex_tool_calling<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let tools = vec![
        tool(
            "read_file",
            "Read the contents of a file at the given path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to read" }
                },
                "required": ["path"]
            }),
        ),
        tool(
            "write_file",
            "Write content to a file at the given path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to write" },
                    "content": { "type": "string", "description": "The content to write" }
                },
                "required": ["path", "content"]
            }),
        ),
        tool(
            "list_dir",
            "List the contents of a directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to list" }
                },
                "required": ["path"]
            }),
        ),
        tool(
            "run_command",
            "Run a shell command and return its output.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run" }
                },
                "required": ["command"]
            }),
        ),
        tool(
            "search",
            "Search for files matching a pattern.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "The search pattern" },
                    "directory": { "type": "string", "description": "Directory to search in" }
                },
                "required": ["pattern"]
            }),
        ),
    ];

    let request = ProbeRequest {
        messages: vec![user_text(
            "Do both of these:\n\
                 1. Read the file at /tmp/config.json\n\
                 2. List the contents of /tmp/data/",
        )],
        tools,
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let calls = &response.tool_calls;

    let path_is_string = |c: &ProbeToolCall| nonempty_string_arg(&c.arguments, "path");
    let has_read_file = calls
        .iter()
        .any(|c| c.name == "read_file" && path_is_string(c));
    let has_list_dir = calls
        .iter()
        .any(|c| c.name == "list_dir" && path_is_string(c));
    let has_read_name = calls.iter().any(|c| c.name == "read_file");
    let has_list_name = calls.iter().any(|c| c.name == "list_dir");

    let any_valid = calls.iter().any(|c| {
        [
            "read_file",
            "write_file",
            "list_dir",
            "run_command",
            "search",
        ]
        .contains(&c.name.as_str())
    });

    let (score, details) = if has_read_file && has_list_dir {
        (
            1.0,
            format!(
                "Two correct tool calls with proper arguments ({} total)",
                calls.len()
            ),
        )
    } else if has_read_name && has_list_name {
        (
            0.5,
            "Both expected tools present but arguments imprecise".to_string(),
        )
    } else if has_read_file || has_list_dir {
        let found = if has_read_file {
            "read_file"
        } else {
            "list_dir"
        };
        (
            0.5,
            format!("Only one correct tool call ({found}), expected two"),
        )
    } else if any_valid {
        (
            0.3,
            format!(
                "Tool calls present but wrong tools chosen: [{}]",
                calls
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    } else if !calls.is_empty() {
        (
            0.3,
            format!(
                "Tool calls present but names not in spec: [{}]",
                calls
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    } else {
        (0.0, "No tool calls in response".to_string())
    };

    Ok(ProbeResult {
        name: "complex_tool_calling".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model can produce nested JSON arguments (arrays of objects).
///
/// Provides a tool with a nested schema (file_path + array of edit objects)
/// and asks the model to apply two edits.
///
/// Scoring:
/// - `1.0` - `file_path` is a nonempty string and `edits` is an array of 2+
///   objects with nonempty string fields
/// - `0.5` - `edits` is a single object, only 1 valid edit, or `file_path`
///   is not a nonempty string
/// - `0.0` - missing `file_path`/`edits`, or `edits` is a string/number
pub async fn probe_nested_arguments<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let edit_file = tool(
        "edit_file",
        "Apply text edits to a file. Each edit replaces old_text with new_text.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Path to the file to edit" },
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": { "type": "string", "description": "Text to find" },
                            "new_text": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["old_text", "new_text"]
                    }
                }
            },
            "required": ["file_path", "edits"]
        }),
    );

    let request = ProbeRequest {
        messages: vec![user_text(FORCEFUL_EDIT_FILE)],
        tools: vec![edit_file],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let calls = &response.tool_calls;
    let edit_call = calls.iter().find(|c| c.name == "edit_file");

    let (score, details) = match edit_call {
        Some(call) => {
            let file_path = call.arguments.get("file_path");
            let edits = call.arguments.get("edits");

            if file_path.is_none() || edits.is_none() {
                (0.0, "Missing required key file_path or edits".to_string())
            } else {
                let file_path_is_string = nonempty_string_arg(&call.arguments, "file_path");
                match edits {
                    Some(serde_json::Value::Array(arr)) => {
                        let valid_edits = arr
                            .iter()
                            .filter(|e| {
                                e.as_object().is_some_and(|m| {
                                    nonempty_string_arg(m, "old_text")
                                        && nonempty_string_arg(m, "new_text")
                                })
                            })
                            .count();

                        if valid_edits >= 2 && file_path_is_string {
                            (
                                1.0,
                                format!(
                                    "Valid nested arguments: {valid_edits} edits with correct structure"
                                ),
                            )
                        } else if valid_edits == 1 {
                            (0.5, "Valid call but only 1 edit instead of 2".to_string())
                        } else if valid_edits >= 2 {
                            (
                                0.5,
                                format!(
                                    "Array present but file_path is not a string, {valid_edits} valid edits"
                                ),
                            )
                        } else {
                            (
                                0.5,
                                format!(
                                    "Array present but {valid_edits} valid edits, has_file_path={file_path_is_string}"
                                ),
                            )
                        }
                    }
                    Some(serde_json::Value::Object(_)) => (
                        0.5,
                        "edits is a single object instead of an array".to_string(),
                    ),
                    _ => (0.0, "edits field is missing or not structured".to_string()),
                }
            }
        }
        None => {
            if calls.is_empty() {
                (0.0, "No tool calls in response".to_string())
            } else {
                (
                    0.0,
                    format!(
                        "Tool calls present but none named edit_file: [{}]",
                        calls
                            .iter()
                            .map(|c| c.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                )
            }
        }
    };

    Ok(ProbeResult {
        name: "nested_arguments".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model picks the right tool from a larger set.
///
/// Presents 8 tool specs and asks three micro-tasks. Each task has a
/// preferred tool (1.0 points) and an acceptable alternative (0.5 points).
///
/// Scoring:
/// - Task 1 (set value in config.toml): `doc_set` with nonempty string
///   path+selector and a present non-null `value` (1.0), name-only
///   `doc_set` or `edit_file` (0.5)
/// - Task 2 (find "deprecated" in src/): `search` with string pattern (1.0),
///   name-only `search` or `run_command` (0.5)
/// - Task 3 (replace markdown section): `md_replace_section` with string
///   path+heading+content (1.0), name-only or `edit_file` (0.5)
/// - Final score: total_points / 3.0 (normalized to 0.0-1.0)
pub async fn probe_tool_selection<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let tools = vec![
        tool(
            "read_file",
            "Read the contents of a file at the given path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to read" }
                },
                "required": ["path"]
            }),
        ),
        tool(
            "edit_file",
            "Edit a file using structured search-and-replace operations.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to edit" },
                    "old_text": { "type": "string", "description": "Text to find" },
                    "new_text": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        ),
        tool(
            "doc_set",
            "Set a value in a JSON, YAML, or TOML file by selector path.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path" },
                    "selector": { "type": "string", "description": "Dot-separated path to the key" },
                    "value": { "description": "The value to set" }
                },
                "required": ["path", "selector", "value"]
            }),
        ),
        tool(
            "search",
            "Search files for a regex pattern and return matching lines.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "The regex pattern to search for" },
                    "directory": { "type": "string", "description": "Directory to search in" }
                },
                "required": ["pattern"]
            }),
        ),
        tool(
            "run_command",
            "Execute a shell command and return its output.",
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
            "List the contents of a directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The directory path to list" }
                },
                "required": ["path"]
            }),
        ),
        tool(
            "md_replace_section",
            "Replace the content under a specific markdown heading in a file.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The markdown file path" },
                    "heading": { "type": "string", "description": "The heading whose section to replace" },
                    "content": { "type": "string", "description": "New content for the section" }
                },
                "required": ["path", "heading", "content"]
            }),
        ),
        tool(
            "write_file",
            "Write or overwrite a file with the given content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path to write" },
                    "content": { "type": "string", "description": "The content to write" }
                },
                "required": ["path", "content"]
            }),
        ),
    ];

    let request = ProbeRequest {
        messages: vec![user_text(
            "You have access to the tools listed above. For each task below, call the \
                 SINGLE BEST tool.\n\n\
                 Task 1: Change the \"port\" value to 8080 in the file config.toml\n\
                 Task 2: Find all files containing the word \"deprecated\" in the src/ directory\n\
                 Task 3: Replace the \"Installation\" section in README.md with new instructions",
        )],
        tools,
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let calls = &response.tool_calls;

    let mut points = 0.0_f32;
    let mut task_details = Vec::new();

    let preferred = |name: &str, keys: &[&str], alt: &str| -> f32 {
        let precise = |c: &ProbeToolCall| {
            keys.iter()
                .all(|key| nonempty_string_arg(&c.arguments, key))
        };
        if calls.iter().any(|c| c.name == name && precise(c)) {
            1.0
        } else if calls.iter().any(|c| c.name == name || c.name == alt) {
            0.5
        } else {
            0.0
        }
    };

    let t1_precise = |c: &ProbeToolCall| {
        nonempty_string_arg(&c.arguments, "path")
            && nonempty_string_arg(&c.arguments, "selector")
            && c.arguments.get("value").is_some_and(|v| !v.is_null())
    };
    let t1_score = if calls.iter().any(|c| c.name == "doc_set" && t1_precise(c)) {
        1.0
    } else if calls
        .iter()
        .any(|c| c.name == "doc_set" || c.name == "edit_file")
    {
        0.5
    } else {
        0.0
    };
    points += t1_score;
    task_details.push(format!("task1={t1_score}"));

    let t2_score = preferred("search", &["pattern"], "run_command");
    points += t2_score;
    task_details.push(format!("task2={t2_score}"));

    let t3_score = preferred(
        "md_replace_section",
        &["path", "heading", "content"],
        "edit_file",
    );
    points += t3_score;
    task_details.push(format!("task3={t3_score}"));

    let score = points / 3.0;
    let details = format!("{} tool call(s): {}", calls.len(), task_details.join(", "));
    let used_generic = calls
        .iter()
        .any(|c| matches!(c.name.as_str(), "edit_file" | "search" | "run_command"));

    Ok(ProbeResult {
        name: "tool_selection".to_string(),
        score,
        max_score: 1.0,
        level: tool_selection_level(score, used_generic),
        details,
    })
}

/// Classify tool_selection. A generic file tool (`edit_file` / `search` /
/// `run_command`) is a miss on bline-specific names, not Weak. Weak would
/// persist `max_tools=10` for 30 days (#3315).
fn tool_selection_level(score: f32, used_generic: bool) -> CapabilityLevel {
    let level = classify(score);
    if level == CapabilityLevel::Weak && used_generic {
        CapabilityLevel::Medium
    } else {
        level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ProbeFinish, ProbeResponse, ProbeToolCall};
    use crate::probes::test_support::*;

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> ProbeToolCall {
        ProbeToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.as_object().unwrap().clone(),
        }
    }

    fn assert_forceful(prompt: &str) {
        assert!(
            prompt.contains("Immediately call"),
            "prompt must be forceful: {prompt}"
        );
        assert!(
            prompt.contains("Do not describe what you would do"),
            "prompt must forbid describing: {prompt}"
        );
    }

    #[tokio::test]
    async fn tool_calling_prompt_is_forceful() {
        let llm = RecordingMock::new(tool_call_response());
        let _ = probe_tool_calling(&llm).await.unwrap();
        let rec = llm.requests.lock().expect("lock");
        assert_eq!(rec.len(), 1);
        let user = request_user_text(&rec[0]);
        assert!(
            user.contains("Immediately call"),
            "live request must be forceful: {user}"
        );
        assert!(
            user.contains("Do not describe"),
            "live request must forbid describing: {user}"
        );
        assert!(
            user.contains("Do not ask for confirmation"),
            "live request must forbid confirmation: {user}"
        );
    }

    #[tokio::test]
    async fn nested_arguments_prompt_is_forceful() {
        let llm = RecordingMock::new(tool_call_response());
        let _ = probe_nested_arguments(&llm).await.unwrap();
        let rec = llm.requests.lock().expect("lock");
        assert_eq!(rec.len(), 1);
        let user = request_user_text(&rec[0]);
        assert_forceful(&user);
        assert!(user.contains("Do not ask for confirmation"), "{user}");
        assert!(user.contains("Replace \"Hello\" with \"Hi\""), "{user}");
        assert!(user.contains("Replace \"World\" with \"Earth\""), "{user}");
    }

    #[tokio::test]
    async fn tool_calling_strong_for_valid_tool_call() {
        let llm = MockLlm {
            response: tool_call_response(),
        };
        let result = probe_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn tool_calling_weak_for_text_only() {
        let llm = MockLlm {
            response: text_response("I would read /tmp/test.txt for you."),
        };
        let result = probe_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn tool_calling_medium_for_wrong_tool_name() {
        let response = ProbeResponse {
            text: String::new(),
            tool_calls: vec![call(
                "call_1",
                "open_file",
                serde_json::json!({"path": "/tmp/test.txt"}),
            )],
            finish: ProbeFinish::ToolCalls,
            usage: None,
        };
        let llm = MockLlm { response };
        let result = probe_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn tool_calling_medium_when_path_is_empty_or_whitespace() {
        for path in ["", " ", "\n"] {
            let response = ProbeResponse {
                text: String::new(),
                tool_calls: vec![call(
                    "call_1",
                    "read_file",
                    serde_json::json!({"path": path}),
                )],
                finish: ProbeFinish::ToolCalls,
                usage: None,
            };
            let result = probe_tool_calling(&MockLlm { response }).await.unwrap();
            assert_eq!(result.score, 0.5, "path={path:?}");
            assert_eq!(result.level, CapabilityLevel::Medium, "path={path:?}");
        }
    }

    #[tokio::test]
    async fn tool_calling_medium_when_path_is_number() {
        let response = ProbeResponse {
            text: String::new(),
            tool_calls: vec![call("call_1", "read_file", serde_json::json!({"path": 1}))],
            finish: ProbeFinish::ToolCalls,
            usage: None,
        };
        let llm = MockLlm { response };
        let result = probe_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn complex_tool_calling_strong_for_two_correct_calls() {
        let response = multi_tool_call_response(vec![
            call(
                "call_1",
                "read_file",
                serde_json::json!({"path": "/tmp/config.json"}),
            ),
            call(
                "call_2",
                "list_dir",
                serde_json::json!({"path": "/tmp/data/"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn complex_tool_calling_medium_when_both_names_have_numeric_path() {
        let response = multi_tool_call_response(vec![
            call("call_1", "read_file", serde_json::json!({"path": 1})),
            call("call_2", "list_dir", serde_json::json!({"path": 2})),
        ]);
        let llm = MockLlm { response };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn complex_tool_calling_not_strong_when_path_is_empty_or_whitespace() {
        for path in ["", " "] {
            let response = multi_tool_call_response(vec![
                call("call_1", "read_file", serde_json::json!({"path": path})),
                call("call_2", "list_dir", serde_json::json!({"path": path})),
            ]);
            let result = probe_complex_tool_calling(&MockLlm { response })
                .await
                .unwrap();
            assert_ne!(result.level, CapabilityLevel::Strong, "path={path:?}");
            assert_eq!(result.score, 0.5, "path={path:?}");
            assert_eq!(result.level, CapabilityLevel::Medium, "path={path:?}");
        }
    }

    #[tokio::test]
    async fn complex_tool_calling_not_strong_when_path_is_number() {
        let response = multi_tool_call_response(vec![
            call("call_1", "read_file", serde_json::json!({"path": 1})),
            call(
                "call_2",
                "list_dir",
                serde_json::json!({"path": "/tmp/data/"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn complex_tool_calling_medium_for_one_correct_call() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "read_file",
            serde_json::json!({"path": "/tmp/config.json"}),
        )]);
        let llm = MockLlm { response };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn complex_tool_calling_weak_for_no_calls() {
        let llm = MockLlm {
            response: text_response("I would read the file and list the directory."),
        };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn complex_tool_calling_weak_for_hallucinated_tool() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "fetch_data",
            serde_json::json!({"url": "/tmp/config.json"}),
        )]);
        let llm = MockLlm { response };
        let result = probe_complex_tool_calling(&llm).await.unwrap();
        assert_eq!(result.score, 0.3);
    }

    #[tokio::test]
    async fn nested_arguments_strong_for_array_of_edits() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "file_path": "/tmp/app.py",
                "edits": [
                    {"old_text": "Hello", "new_text": "Hi"},
                    {"old_text": "World", "new_text": "Earth"}
                ]
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn nested_arguments_weak_for_missing_file_path() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "edits": [
                    {"old_text": "Hello", "new_text": "Hi"},
                    {"old_text": "World", "new_text": "Earth"}
                ]
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn nested_arguments_not_strong_when_file_path_is_number() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "file_path": 1,
                "edits": [
                    {"old_text": "Hello", "new_text": "Hi"},
                    {"old_text": "World", "new_text": "Earth"}
                ]
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert!(result.score <= 0.5);
    }

    #[tokio::test]
    async fn nested_arguments_not_strong_when_file_path_is_empty_or_whitespace() {
        for file_path in ["", " "] {
            let response = multi_tool_call_response(vec![call(
                "call_1",
                "edit_file",
                serde_json::json!({
                    "file_path": file_path,
                    "edits": [
                        {"old_text": "Hello", "new_text": "Hi"},
                        {"old_text": "World", "new_text": "Earth"}
                    ]
                }),
            )]);
            let result = probe_nested_arguments(&MockLlm { response }).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "file_path={file_path:?}"
            );
            assert!(result.score <= 0.5, "file_path={file_path:?}");
        }
    }

    #[tokio::test]
    async fn nested_arguments_not_strong_when_edit_strings_are_empty_or_whitespace() {
        for text in ["", " "] {
            let response = multi_tool_call_response(vec![call(
                "call_1",
                "edit_file",
                serde_json::json!({
                    "file_path": "/tmp/app.py",
                    "edits": [
                        {"old_text": text, "new_text": text},
                        {"old_text": text, "new_text": text}
                    ]
                }),
            )]);
            let result = probe_nested_arguments(&MockLlm { response }).await.unwrap();
            assert_ne!(result.level, CapabilityLevel::Strong, "text={text:?}");
            assert!(result.score <= 0.5, "text={text:?}");
        }
    }

    #[tokio::test]
    async fn nested_arguments_weak_for_edits_string() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "file_path": "/tmp/app.py",
                "edits": "replace Hello with Hi"
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn nested_arguments_medium_for_single_edit() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "file_path": "/tmp/app.py",
                "edits": [
                    {"old_text": "Hello", "new_text": "Hi"}
                ]
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn nested_arguments_weak_for_flat_args() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "file_path": "/tmp/app.py",
                "old_text": "Hello",
                "new_text": "Hi"
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn nested_arguments_weak_for_no_calls() {
        let llm = MockLlm {
            response: text_response("I would edit the file."),
        };
        let result = probe_nested_arguments(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn tool_selection_strong_for_preferred_tools() {
        let response = multi_tool_call_response(vec![
            call(
                "call_1",
                "doc_set",
                serde_json::json!({"path": "config.toml", "selector": "port", "value": 8080}),
            ),
            call(
                "call_2",
                "search",
                serde_json::json!({"pattern": "deprecated", "directory": "src/"}),
            ),
            call(
                "call_3",
                "md_replace_section",
                serde_json::json!({
                    "path": "README.md",
                    "heading": "Installation",
                    "content": "new instructions"
                }),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn tool_selection_strong_for_best_same_name_call() {
        let response = multi_tool_call_response(vec![
            call("call_1", "doc_set", serde_json::json!({})),
            call(
                "call_2",
                "doc_set",
                serde_json::json!({"path": "config.toml", "selector": "port", "value": 8080}),
            ),
            call("call_3", "search", serde_json::json!({})),
            call(
                "call_4",
                "search",
                serde_json::json!({"pattern": "deprecated", "directory": "src/"}),
            ),
            call("call_5", "md_replace_section", serde_json::json!({})),
            call(
                "call_6",
                "md_replace_section",
                serde_json::json!({
                    "path": "README.md",
                    "heading": "Installation",
                    "content": "new instructions"
                }),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn tool_selection_not_strong_when_doc_set_omits_or_nulls_value() {
        let task1_args = [
            serde_json::json!({"path": "config.toml", "selector": "port"}),
            serde_json::json!({"path": "config.toml", "selector": "port", "value": null}),
        ];
        for args in task1_args {
            let response = multi_tool_call_response(vec![
                call("call_1", "doc_set", args),
                call(
                    "call_2",
                    "search",
                    serde_json::json!({"pattern": "deprecated", "directory": "src/"}),
                ),
                call(
                    "call_3",
                    "md_replace_section",
                    serde_json::json!({
                        "path": "README.md",
                        "heading": "Installation",
                        "content": "new instructions"
                    }),
                ),
            ]);
            let result = probe_tool_selection(&MockLlm { response }).await.unwrap();
            assert!(result.score < 1.0, "{}", result.details);
            assert!(
                result.details.contains("task1=0.5"),
                "doc_set without a non-null value must be 0.5: {}",
                result.details
            );
            assert!(
                (result.score - 2.5 / 3.0).abs() < 0.01,
                "precise task2+task3 with imprecise task1: {}",
                result.score
            );
        }
    }

    #[tokio::test]
    async fn tool_selection_medium_for_whitespace_preferred_args() {
        let response = multi_tool_call_response(vec![
            call(
                "call_1",
                "doc_set",
                serde_json::json!({"path": " ", "selector": "\n", "value": 8080}),
            ),
            call("call_2", "search", serde_json::json!({"pattern": " "})),
            call(
                "call_3",
                "md_replace_section",
                serde_json::json!({"path": " ", "heading": "\n", "content": " "}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn tool_selection_medium_for_preferred_names_without_string_args() {
        let response = multi_tool_call_response(vec![
            call("call_1", "doc_set", serde_json::json!({})),
            call("call_2", "search", serde_json::json!({})),
            call("call_3", "md_replace_section", serde_json::json!({})),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
        assert_ne!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn tool_selection_medium_for_acceptable_alternatives() {
        let response = multi_tool_call_response(vec![
            call(
                "call_1",
                "edit_file",
                serde_json::json!({
                    "path": "config.toml",
                    "old_text": "port = 3000",
                    "new_text": "port = 8080"
                }),
            ),
            call(
                "call_2",
                "run_command",
                serde_json::json!({"command": "grep -r deprecated src/"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn tool_selection_single_edit_file_is_033_and_not_weak() {
        let response = multi_tool_call_response(vec![call(
            "call_1",
            "edit_file",
            serde_json::json!({
                "path": "config.toml",
                "old_text": "port = 3000",
                "new_text": "port = 8080"
            }),
        )]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert!(
            (result.score - 1.0 / 3.0).abs() < 0.01,
            "single edit_file is the 0.33 path, got {}",
            result.score
        );
        assert_ne!(
            result.level,
            CapabilityLevel::Weak,
            "generic edit_file must not classify Weak: {}",
            result.details
        );
    }

    #[tokio::test]
    async fn tool_selection_weak_for_wrong_tools() {
        let response = multi_tool_call_response(vec![
            call(
                "call_1",
                "read_file",
                serde_json::json!({"path": "config.toml"}),
            ),
            call("call_2", "list_dir", serde_json::json!({"path": "src/"})),
            call(
                "call_3",
                "write_file",
                serde_json::json!({"path": "README.md", "content": "new"}),
            ),
        ]);
        let llm = MockLlm { response };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn tool_selection_weak_for_text_only() {
        let llm = MockLlm {
            response: text_response(
                "I would change the config, search for deprecated, and update the README.",
            ),
        };
        let result = probe_tool_selection(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
    }
}
