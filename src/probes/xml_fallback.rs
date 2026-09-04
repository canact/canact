//! XML fallback tool-calling probe and XML parsing helpers.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{
    has_visible_arg_text, nonempty_string_arg_any, refuse_truncated_incomplete,
    refuse_truncated_tool_call, system_text, user_text,
};

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
/// - `0.4` - attempted `read_file` block with `<arguments>` that is not parseable
/// - `0.0` - no tags, format-card echo (`TOOL_NAME` / `{"param":"value"}`),
///   or tag mention without a tool attempt
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
    if !response.text.contains("<tool_call>") {
        refuse_truncated_tool_call(&response)?;
    }
    let finish = response.finish;
    let text = response.text;

    let has_tool_call_open = text.contains("<tool_call>");
    let has_tool_call_close = text.contains("</tool_call>");

    let (score, details) = if has_tool_call_open && has_tool_call_close {
        match parse_xml_tool_block(&text) {
            Some((name, args)) => {
                if !has_visible_arg_text(&name) {
                    (0.0, "XML tool call has empty name".to_string())
                } else if is_xml_format_card_echo(&name, &args) {
                    (
                        0.0,
                        "Echoed the XML format card, not a tool call".to_string(),
                    )
                } else {
                    let correct_name = name == "read_file";
                    let has_path = args
                        .as_object()
                        .is_some_and(|o| nonempty_string_arg_any(o, &["path", "file_path"]));

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
            }
            None => {
                if xml_closed_block_is_read_file_attempt(&text) {
                    (
                        0.4,
                        "XML tool_call tags present but block is not parseable".to_string(),
                    )
                } else {
                    (
                        0.0,
                        "Named <tool_call> tags without a tool attempt".to_string(),
                    )
                }
            }
        }
    } else if has_tool_call_open {
        if xml_open_span_names_read_file(&text) && !xml_open_span_is_format_card(&text) {
            (
                0.4,
                "Opening <tool_call> tag found but no closing tag".to_string(),
            )
        } else {
            (
                0.0,
                "Opening <tool_call> tag without a tool attempt".to_string(),
            )
        }
    } else {
        (0.0, "No <tool_call> tags in response".to_string())
    };

    refuse_truncated_incomplete(finish, score)?;
    Ok(ProbeResult {
        name: "xml_tool_calling".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// True when the parsed block is the system-card example, not a call.
fn is_xml_format_card_echo(name: &str, args: &serde_json::Value) -> bool {
    if name == "TOOL_NAME" {
        return true;
    }
    if xml_args_have_real_path(args) {
        return false;
    }
    xml_args_are_card_tokens(args)
}

/// A nonempty `path` other than the card token `value` is a real call.
fn xml_args_have_real_path(args: &serde_json::Value) -> bool {
    match args {
        serde_json::Value::Object(o) => {
            let own = xml_map_get_ci(o, "path").is_some_and(xml_path_value_is_real);
            own || o.values().any(xml_args_have_real_path)
        }
        serde_json::Value::Array(arr) => arr.iter().any(xml_args_have_real_path),
        serde_json::Value::String(s) => {
            parse_json_wrapped_value(s).is_some_and(|v| xml_args_have_real_path(&v))
        }
        _ => false,
    }
}

/// Walk objects, arrays, and JSON-encoded strings for card tokens.
fn xml_args_are_card_tokens(args: &serde_json::Value) -> bool {
    match args {
        serde_json::Value::Array(arr) => {
            !arr.is_empty() && arr.iter().all(xml_args_are_card_tokens)
        }
        serde_json::Value::Object(o) => {
            xml_object_is_card(o) || o.values().any(xml_args_are_card_tokens)
        }
        serde_json::Value::String(s) => {
            parse_json_wrapped_value(s).is_some_and(|v| xml_args_are_card_tokens(&v))
        }
        _ => false,
    }
}

fn xml_object_is_card(o: &serde_json::Map<String, serde_json::Value>) -> bool {
    let param_value_echo = xml_map_get_ci(o, "param").is_some_and(xml_value_is_card_value);
    let path_value_echo = xml_map_get_ci(o, "path").is_some_and(xml_value_is_card_value);
    let schema_echo = xml_map_get_ci(o, "type")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("object"))
        && xml_map_get_ci(o, "properties").is_some()
        && !xml_map_get_ci(o, "path")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty() && !s.eq_ignore_ascii_case("value"));
    param_value_echo || path_value_echo || schema_echo
}

/// A string or array `path` that is not the card token `value`.
/// Schema objects under `properties.path` are not a real call.
fn xml_path_value_is_real(v: &serde_json::Value) -> bool {
    if let Some(s) = v.as_str() {
        let t = xml_visible_card_token(s);
        return !t.is_empty() && !t.eq_ignore_ascii_case("value");
    }
    v.as_array()
        .is_some_and(|arr| !arr.is_empty() && !xml_value_is_card_value(v))
}

/// `{"param":"value"}` and `{"param":["value"]}`.
fn xml_value_is_card_value(v: &serde_json::Value) -> bool {
    if v.as_str()
        .is_some_and(|s| xml_visible_card_token(s).eq_ignore_ascii_case("value"))
    {
        return true;
    }
    v.as_array().is_some_and(|arr| {
        !arr.is_empty()
            && arr.iter().all(|el| {
                el.as_str()
                    .is_some_and(|s| xml_visible_card_token(s).eq_ignore_ascii_case("value"))
            })
    })
}

/// Drop whitespace and ZWSP/format marks so `value` + U+200B is still the card.
/// Fold fullwidth ASCII letters/digits so `ｖａｌｕｅ` matches `value`.
fn xml_visible_card_token(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !c.is_whitespace()
                && !matches!(
                    c,
                    '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}'
                )
        })
        .map(fold_fullwidth_ascii)
        .collect()
}

fn fold_fullwidth_ascii(c: char) -> char {
    match c {
        '\u{FF10}'..='\u{FF19}' => char::from(b'0' + (c as u32 - 0xFF10) as u8),
        '\u{FF21}'..='\u{FF3A}' => char::from(b'A' + (c as u32 - 0xFF21) as u8),
        '\u{FF41}'..='\u{FF5A}' => char::from(b'a' + (c as u32 - 0xFF41) as u8),
        _ => c,
    }
}

fn xml_map_get_ci<'a>(
    o: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    o.iter()
        .find(|(k, _)| xml_visible_card_token(k).eq_ignore_ascii_case(key))
        .map(|(_, v)| v)
}

fn parse_json_wrapped_value(s: &str) -> Option<serde_json::Value> {
    let t = s.trim();
    if !(t.starts_with('{') || t.starts_with('[') || t.starts_with('"')) {
        return None;
    }
    serde_json::from_str(t).ok()
}

/// Open-only Medium only when `read_file` is named after `<tool_call>`.
fn xml_open_span_names_read_file(text: &str) -> bool {
    let Some(start) = text.find("<tool_call>") else {
        return false;
    };
    text[start + "<tool_call>".len()..].contains("<name>read_file</name>")
}

/// Open-only param/value card (even without `</tool_call>`) is echo.
fn xml_open_span_is_format_card(text: &str) -> bool {
    let Some(start) = text.find("<tool_call>") else {
        return false;
    };
    let after = &text[start + "<tool_call>".len()..];
    let Some(args_pos) = after.find("<arguments>") else {
        return false;
    };
    let args = &after[args_pos + "<arguments>".len()..];
    let args = match args.find("</arguments>") {
        Some(end) => &args[..end],
        None => args,
    };
    xml_arguments_text_is_format_card(args)
}

/// Closed unparseable Medium only when the first block names `read_file` and
/// opens `<arguments>` (a truncated or broken call, not a format lecture).
fn xml_closed_block_is_read_file_attempt(text: &str) -> bool {
    let Some(start) = text.find("<tool_call>") else {
        return false;
    };
    let after = &text[start + "<tool_call>".len()..];
    let span = match after.find("</tool_call>") {
        Some(end) => &after[..end],
        None => after,
    };
    span.contains("<name>read_file</name>")
        && span.contains("<arguments>")
        && (argument_span_starts_json_object(span) || argument_span_has_asked_path(span))
        && !extract_xml_element_simple(span, "arguments")
            .is_some_and(xml_arguments_text_is_format_card)
}

/// Bare `/tmp/example.txt` (or other nonempty non-card text that names it)
/// is a Weak-adjacent attempt, not a lecture.
fn argument_span_has_asked_path(span: &str) -> bool {
    extract_xml_element_simple(span, "arguments").is_some_and(|args| {
        let t = args.trim();
        !t.is_empty() && t.contains("/tmp/example.txt") && !xml_arguments_text_is_format_card(args)
    })
}

/// Unparseable `{'param':'value'}` (and extra-key siblings) is the card.
fn xml_arguments_text_is_format_card(args: &str) -> bool {
    let t = args.trim();
    let lower = xml_visible_card_token(t).to_ascii_lowercase();
    if !((lower.contains("param") || lower.contains("path")) && lower.contains("value")) {
        return false;
    }
    let normalized = t.replace('\'', "\"");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&normalized) {
        return is_xml_format_card_echo("read_file", &v);
    }
    !t.contains("/tmp/example.txt")
}

/// Closed Medium needs an arguments body that at least opened `{`.
fn argument_span_starts_json_object(span: &str) -> bool {
    extract_xml_element_simple(span, "arguments").is_some_and(|args| args.trim().starts_with('{'))
}

/// Try to extract the first `<tool_call>` block's name and parsed JSON arguments.
fn parse_xml_tool_block(text: &str) -> Option<(String, serde_json::Value)> {
    let mut search = 0;
    let mut first = None;
    let mut best_real = None;
    while let Some(rel) = text.get(search..).and_then(|s| s.find("<tool_call>")) {
        let start = search + rel + "<tool_call>".len();
        let Some(end) = text.get(start..).and_then(|s| s.find("</tool_call>")) else {
            break;
        };
        let block = &text[start..start + end];
        if let (Some(name), Some(args_str)) = (
            extract_xml_element_simple(block, "name"),
            extract_xml_element_simple(block, "arguments"),
        ) {
            if let Some(args) = parse_xml_arguments(args_str) {
                let parsed = (name.trim().to_string(), args);
                if first.is_none() {
                    first = Some(parsed.clone());
                }
                let real_read = parsed.0 == "read_file"
                    && parsed
                        .1
                        .as_object()
                        .is_some_and(|o| nonempty_string_arg_any(o, &["path", "file_path"]))
                    && !is_xml_format_card_echo(&parsed.0, &parsed.1);
                if real_read {
                    best_real = Some(parsed);
                    break;
                }
            }
        }
        search = start + end + "</tool_call>".len();
    }
    best_real.or(first)
}

fn parse_xml_arguments(args_str: &str) -> Option<serde_json::Value> {
    let trimmed = args_str.trim();
    if let Ok(args) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(args);
    }
    let mut obj = serde_json::Map::new();
    for key in ["path", "file_path"] {
        if let Some(val) = extract_xml_element_simple(trimmed, key) {
            let t = val.trim();
            if !t.is_empty() {
                obj.insert(key.to_string(), serde_json::Value::String(t.to_string()));
            }
        }
    }
    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj))
    }
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
    async fn xml_file_path_json_alias_is_strong() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"file_path\": \"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "JSON file_path alias must score Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn xml_nested_path_element_is_strong() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>
<path>/tmp/example.txt</path>
</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "XML path child must count as arguments: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn xml_bare_path_arguments_are_a_medium_attempt() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>/tmp/example.txt</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.score, 0.4,
            "bare path arguments must be a 0.4 attempt: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn xml_bare_path_does_not_reopen_card_echo() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"param\": \"value\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "param/value card must stay Weak after bare-path attempt: {result:?}"
        );
        assert_eq!(result.score, 0.0);
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
    async fn xml_tool_calling_length_unclosed_is_transient() {
        let response = crate::client::ProbeResponse {
            text: "<tool_call>\n<name>read_file</name>".into(),
            tool_calls: Vec::new(),
            finish: crate::client::ProbeFinish::Length,
            usage: None,
        };
        let err = probe_xml_tool_calling(&MockLlm { response })
            .await
            .expect_err("Length + unclosed XML must not score Medium");
        assert!(
            matches!(&err, crate::ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
        let (result, cacheable) =
            crate::runner::resolve_probe(Err(err), "xml_tool_calling").expect("synthesized Medium");
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert!(
            !cacheable,
            "Length + unclosed XML must not be a 30-day cache hit"
        );
        assert!(result.details.contains("truncated"), "{}", result.details);
    }

    #[tokio::test]
    async fn xml_tool_calling_length_without_tags_is_transient() {
        let response = crate::client::ProbeResponse {
            text: "I will emit a tool_call after leftover reasoning.".into(),
            tool_calls: Vec::new(),
            finish: crate::client::ProbeFinish::Length,
            usage: None,
        };
        let err = probe_xml_tool_calling(&MockLlm { response })
            .await
            .expect_err("truncated XML must not score Weak");
        assert!(
            matches!(&err, crate::ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn xml_tool_calling_name_before_open_tag_does_not_open_tools() {
        let llm = MockLlm {
            response: text_response(
                "Use <name>read_file</name> as shown.\n<tool_call>\nI cannot emit a tool call.\n",
            ),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "name mention before <tool_call> must not set canUseTools: {result:?}"
        );
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
    async fn xml_tool_calling_not_strong_for_empty_or_whitespace_path() {
        for args in [r#"{"path": ""}"#, r#"{"path": " "}"#, r#"{"path": "\n"}"#] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "empty/whitespace path must not be Strong: {args}"
            );
            assert_eq!(result.score, 0.7, "imprecise XML path: {args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_weak_for_empty_or_whitespace_name() {
        for name in ["", "   ", "\u{200b}"] {
            let response_text = format!(
                "<tool_call>\n<name>{name}</name>\n<arguments>{{\"path\": \"/tmp/example.txt\"}}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "empty XML name must not open can_use_tools: {name:?} {result:?}"
            );
            assert_eq!(result.score, 0.0, "name={name:?}");
        }
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
    async fn xml_tool_calling_card_echo_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>TOOL_NAME</name>
<arguments>{\"param\": \"value\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "system card echo must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert!(result.score < 0.4, "card echo score: {}", result.score);
    }

    #[tokio::test]
    async fn xml_tool_calling_array_wrapped_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>[{\"param\":\"value\"}]</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "array-wrapped param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_multi_element_card_array_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>[{\"param\":\"value\"},{\"param\":\"value\"}]</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "multi-element param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_mixed_card_and_real_path_array_is_still_attempt() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>[{\"param\":\"value\"},{\"path\":\"/tmp/example.txt\"}]</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "mixed card plus real path is still an attempt: {result:?}"
        );
        assert_eq!(result.score, 0.7);
    }

    #[tokio::test]
    async fn xml_tool_calling_unclosed_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"param\":\"value\"}
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "unclosed param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_array_wrapped_real_path_is_still_attempt() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>[{\"path\":\"/tmp/example.txt\"}]</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "array-wrapped real path is still an attempt: {result:?}"
        );
        assert_eq!(result.score, 0.7);
    }

    #[tokio::test]
    async fn xml_tool_calling_unclosed_real_path_is_still_attempt() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"path\":\"/tmp/example.txt\"}
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "unclosed real path is still an attempt: {result:?}"
        );
        assert_eq!(result.score, 0.4);
    }

    #[tokio::test]
    async fn xml_tool_calling_card_extra_keys_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"param\": \"value\", \"type\": \"object\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "param/value plus extra keys must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_single_quoted_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{'param': 'value'}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "single-quoted param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_unparseable_zwsp_key_card_is_echo() {
        for args in [
            "{p\u{200B}ath: 'value'}",
            "{p\u{200B}aram: 'value'}",
            "{\u{FF50}\u{FF41}\u{FF54}\u{FF48}: 'value'}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "unparseable ZWSP/fullwidth key card must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_single_quoted_title_case_path_card_is_echo() {
        for args in [
            "{'path': 'Value'}",
            "{'Path': 'value'}",
            "{'path':['Value']}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "title-case unparseable path card must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_single_quoted_path_card_does_not_open_tools() {
        for args in ["{'path': 'value'}", "{'path':['value']}"] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "single-quoted path card must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_single_quoted_real_path_is_still_attempt() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{'path': '/tmp/example.txt'}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "single-quoted real path is still an attempt: {result:?}"
        );
        assert_eq!(result.score, 0.4);
    }

    #[tokio::test]
    async fn xml_tool_calling_card_then_real_block_is_strong() {
        let response_text = "\
<tool_call>
<name>tool_name</name>
<arguments>{\"param\": \"value\"}</arguments>
</tool_call>
<tool_call>
<name>read_file</name>
<arguments>{\"path\": \"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn xml_tool_calling_card_extra_keys_plus_real_path_is_strong() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"param\": \"value\", \"path\": \"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Strong,
            "extra keys plus a real path is a call: {result:?}"
        );
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_nested_payload_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"payload\":{\"param\":\"value\"}}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "nested param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_json_string_wrapped_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>\"{\\\"param\\\":\\\"value\\\"}\"</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "JSON-string-wrapped param/value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_param_array_value_card_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"param\":[\"value\"]}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "param array-of-value card must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_nested_card_plus_real_path_is_strong() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"payload\":{\"param\":\"value\"},\"path\":\"/tmp/example.txt\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Strong,
            "nested card plus a real path is a call: {result:?}"
        );
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_path_value_card_mix_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"path\": \"value\"}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "path=value card mix must not be Strong: {result:?}"
        );
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "path=value card mix must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_fullwidth_value_is_echo() {
        // Fullwidth ｖａｌｕｅ (U+FF56 U+FF41 U+FF4C U+FF55 U+FF45).
        for args in [
            "{\"path\":\"\u{FF56}\u{FF41}\u{FF4C}\u{FF55}\u{FF45}\"}",
            "{\"param\":\"\u{FF56}\u{FF41}\u{FF4C}\u{FF55}\u{FF45}\"}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "fullwidth card value must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_zwsp_padded_key_is_echo() {
        for args in [
            "{\"path\\u200b\":\"value\"}",
            "{\"\\u200bparam\":\"value\"}",
            "{\"param\\u200b\":\"value\"}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "ZWSP-padded card key must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_zwsp_padded_value_is_echo() {
        for args in [
            "{\"path\":\"value\\u200b\"}",
            "{\"path\":[\"value\\u200b\"]}",
            "{\"param\":\"value\\u200b\"}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "ZWSP-padded card value must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_path_array_and_padded_value_are_echo() {
        for args in [
            "{\"path\":[\"value\"]}",
            "{\"path\":[\"Value\"]}",
            "{\"path\":\"value \"}",
            "{\"path\":\" Value\"}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "path card variants must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_title_case_card_does_not_open_tools() {
        for args in [
            "{\"Param\":\"value\"}",
            "{\"param\":\"Value\"}",
            "{\"path\":\"Value\"}",
        ] {
            let response_text = format!(
                "<tool_call>\n<name>read_file</name>\n<arguments>{args}</arguments>\n</tool_call>"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_xml_tool_calling(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "title-case card tokens must not set canUseTools: {args} {result:?}"
            );
            assert_eq!(result.score, 0.0, "args={args}");
        }
    }

    #[tokio::test]
    async fn xml_tool_calling_schema_paste_does_not_open_tools() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"The file path to read\"}},\"required\":[\"path\"]}</arguments>
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "JSON Schema paste must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_dummy_arguments_lecture_does_not_open_tools() {
        let response_text = "\
<tool_call>
The format uses <name>read_file</name> and <arguments>a JSON object</arguments>.
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "dummy <arguments> lecture must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_read_file_mention_inside_tags_does_not_open_tools() {
        let response_text = "\
<tool_call>
The format uses <name>read_file</name> and a path argument.
</tool_call>";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "format mention of read_file inside tags must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_tag_mention_does_not_open_tools() {
        let llm = MockLlm {
            response: text_response(
                "I cannot emit <tool_call></tool_call> blocks in this environment.",
            ),
        };
        let result = probe_xml_tool_calling(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "naming the tags must not set canUseTools: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn xml_tool_calling_medium_for_unparseable_json() {
        let response_text = "\
<tool_call>
<name>read_file</name>
<arguments>{path: /tmp/example.txt}</arguments>
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
