//! XML fallback tool-calling probe and XML parsing helpers.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{system_text, user_text};

const FORCEFUL_READ_FILE: &str = "Immediately call the read_file tool with path /tmp/example.txt. Do not describe what you would do. Do not ask for confirmation.";

/// Probe whether the model can produce valid XML tool-call blocks.
///
/// This tests the fallback path used when native function calling is not
/// available. The model is given XML format instructions and a `read_file`
/// tool spec, then asked to call it.
///
/// Scoring:
/// - `1.0` - valid `<tool_call>` block with correct name and JSON arguments
/// - `0.7` - valid block but wrong tool name or missing arguments
/// - `0.4` - `<tool_call>` tags present but block is not parseable
/// - `0.0` - no `<tool_call>` tags at all
pub async fn probe_xml_tool_calling<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let system = "\
To call a tool, emit an XML block in this exact format:

<tool_call>
<name>TOOL_NAME</name>
<arguments>{\"param\": \"value\"}</arguments>
</tool_call>

You may emit multiple <tool_call> blocks in a single response.

Available tools:

- **read_file**: Read the contents of a file at the given path.
  Parameters: {\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"The file path to read\"}},\"required\":[\"path\"]}";

    let user = FORCEFUL_READ_FILE;

    let request = ProbeRequest {
        messages: vec![system_text(system), user_text(user)],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(500),
    };

    let response = llm.chat(request).await?;
    let text = response.text;

    let has_tool_call_open = text.contains("<tool_call>");
    let has_tool_call_close = text.contains("</tool_call>");

    let (score, details) = if has_tool_call_open && has_tool_call_close {
        match parse_xml_tool_block(&text) {
            Some((name, args)) => {
                let correct_name = name == "read_file";
                let has_path = args
                    .as_object()
                    .and_then(|o| o.get("path"))
                    .and_then(|v| v.as_str())
                    .is_some();

                if correct_name && has_path {
                    (
                        1.0,
                        "Valid XML tool call with correct name and arguments".to_string(),
                    )
                } else {
                    (
                        0.7,
                        format!(
                            "XML tool call parsed but imprecise: name={name}, has_path={has_path}"
                        ),
                    )
                }
            }
            None => (
                0.4,
                "XML tool_call tags present but block is not parseable".to_string(),
            ),
        }
    } else if has_tool_call_open {
        (
            0.4,
            "Opening <tool_call> tag found but no closing tag".to_string(),
        )
    } else {
        (0.0, "No <tool_call> tags in response".to_string())
    };

    Ok(ProbeResult {
        name: "xml_tool_calling".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Try to extract the first `<tool_call>` block's name and parsed JSON arguments.
fn parse_xml_tool_block(text: &str) -> Option<(String, serde_json::Value)> {
    let start = text.find("<tool_call>")? + "<tool_call>".len();
    let end = text[start..].find("</tool_call>")?;
    let block = &text[start..start + end];

    let name = extract_xml_element_simple(block, "name")?;
    let args_str = extract_xml_element_simple(block, "arguments")?;
    let args: serde_json::Value = serde_json::from_str(args_str.trim()).ok()?;
    Some((name.trim().to_string(), args))
}

/// Extract the text content of a simple XML element like `<tag>content</tag>`.
fn extract_xml_element_simple<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)?;
    Some(&text[start..start + end])
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn xml_tool_calling_prompt_is_forceful() {
        let llm = RecordingMock::new(text_response(
            "<tool_call>\n<name>read_file</name>\n<arguments>{\"path\": \"/tmp/example.txt\"}</arguments>\n</tool_call>",
        ));
        let _ = probe_xml_tool_calling(&llm).await.unwrap();
        let rec = llm.requests.lock().expect("lock");
        assert_eq!(rec.len(), 1);
        let user = request_user_text(&rec[0]);
        assert!(user.contains("Immediately call"), "{user}");
        assert!(user.contains("Do not describe what you would do"), "{user}");
        assert!(user.contains("Do not ask for confirmation"), "{user}");
    }

    #[tokio::test]
    async fn xml_tool_calling_strong_for_valid_block() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"path\": \"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_weak_for_prose() {
        let llm = MockLlm {
            response: text_response("I would read /tmp/example.txt for you."),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_medium_for_malformed() {
        let response_text = "<tool_call>\n<name>read_file</name>\n";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.4);
    }

    #[tokio::test]
    async fn xml_tool_calling_medium_for_wrong_name() {
        let response_text = "\
<tool_call>
<name>open_file</name>
<arguments>{\"path\": \"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.score, 0.7);
    }

    #[tokio::test]
    async fn xml_tool_calling_medium_for_unparseable_json() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>not json</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.4);
    }

    #[test]
    fn parse_xml_tool_block_valid() {
        let text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"path\": \"/tmp/test.txt\"}</arguments>
</tool_call>";
        let (name, args) = parse_xml_tool_block(text).unwrap();
        assert_eq!(name, "read_file");
        assert_eq!(args["path"], "/tmp/test.txt");
    }

    #[test]
    fn parse_xml_tool_block_missing_close() {
        let text = "<tool_call>\n<name>read_file</name>\n";
        assert!(parse_xml_tool_block(text).is_none());
    }
}
