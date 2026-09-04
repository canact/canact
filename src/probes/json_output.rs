//! JSON output and instruction-following probes.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{extract_json_from_text, refuse_truncated_incomplete, user_text};

/// Probe whether the model can produce structured JSON output.
///
/// Asks the model to analyse a word and return a JSON object with specific
/// fields: `word`, `length`, and `reversed`.
///
/// Scoring:
/// - `1.0` - valid JSON with all three fields and correct types
/// - Partial credit for valid JSON with fewer fields
/// - `0.0` - response is not valid JSON
pub async fn probe_json_output<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text(
            "Analyze the word \"hello\" and respond with ONLY a JSON object (no other text) \
             containing these fields: \"word\" (the word as a string), \"length\" (the number \
             of characters as a number), \"reversed\" (the word reversed as a string). \
             Example: {\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}",
        )],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let (score, details) = score_json_text(&response.text);

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "json_output".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model follows precise instructions.
///
/// Asks for a single-word answer and scores based on response verbosity.
///
/// Scoring:
/// - `1.0` - exactly one word, the asked capital (Paris)
/// - `0.5` - one wrong word, or short but multi-word (2-4 words)
/// - `0.0` - empty or verbose (5+ words)
pub async fn probe_instruction_following<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text(
            "What is the capital of France? Reply with ONLY the single word answer, \
             no punctuation, no explanation.",
        )],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let trimmed = peel_one_fence(response.text.trim());
    let word_count = trimmed.split_whitespace().count();

    let (score, details) = if word_count == 0 {
        (0.0, "Empty response".to_string())
    } else if word_count == 1 && asked_capital(trimmed) {
        (1.0, format!("Single word response: \"{trimmed}\""))
    } else if word_count == 1 {
        (
            0.5,
            format!("Single word but not the asked answer: \"{trimmed}\""),
        )
    } else if word_count < 5 {
        (
            0.5,
            format!("Short but multi-word response ({word_count} words)"),
        )
    } else {
        (0.0, format!("Verbose response ({word_count} words)"))
    };

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "instruction_following".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn asked_capital(word: &str) -> bool {
    word.trim()
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .eq_ignore_ascii_case("paris")
}

fn peel_one_fence(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    let rest = match rest.find('\n') {
        Some(i) => &rest[i + 1..],
        None => rest,
    };
    rest.strip_suffix("```").map(str::trim).unwrap_or(t)
}

fn score_json_text(text: &str) -> (f32, String) {
    let primary = extract_json_from_text(text);
    let primary_is_array =
        serde_json::from_str::<serde_json::Value>(primary).is_ok_and(|val| val.is_array());
    // Standalone objects win. A complete hello that only lives inside an
    // array envelope stays Weak (needs_json_repair).
    let mut best: Option<(f32, String)> = None;
    let mut array_wrapped_hello = false;
    consider_standalone_json(text, &mut best, &mut array_wrapped_hello);
    if best.as_ref().is_some_and(|(s, _)| *s >= 1.0) {
        return best.expect("best is Some");
    }
    if array_wrapped_hello {
        return (0.0, "Response JSON is an array, not an object".to_string());
    }
    if let Some(best) = best {
        return best;
    }
    if primary_is_array {
        return (0.0, "Response JSON is an array, not an object".to_string());
    }
    (0.0, "Response was not valid JSON".to_string())
}

fn consider_standalone_json(
    text: &str,
    best: &mut Option<(f32, String)>,
    array_wrapped_hello: &mut bool,
) {
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with('[') {
            if let Some((val, end)) = next_json_array(text, i) {
                if value_contains_complete_hello(&val) {
                    *array_wrapped_hello = true;
                }
                i = end;
                continue;
            }
            if let Some(end) = delimited_span_end(text, i, '[', ']') {
                if inner_has_complete_hello(&text[i + 1..end - 1]) {
                    *array_wrapped_hello = true;
                }
                i = end;
                continue;
            }
        }
        if text[i..].starts_with('{') {
            if let Some((val, end)) = next_json_object(text, i) {
                collect_nested_object_scores(&val, best);
                i = end;
                continue;
            }
        }
        i += text[i..].chars().next().map_or(1, char::len_utf8);
    }
}

fn collect_nested_object_scores(val: &serde_json::Value, best: &mut Option<(f32, String)>) {
    if val.is_object() {
        let scored = score_json_object(val);
        if best.as_ref().is_none_or(|(s, _)| scored.0 > *s) {
            *best = Some(scored);
        }
        if let Some(obj) = val.as_object() {
            for nested in obj.values() {
                collect_nested_object_scores(nested, best);
            }
        }
    }
}

fn value_contains_complete_hello(val: &serde_json::Value) -> bool {
    if val.is_object() && score_json_object(val).0 >= 1.0 {
        return true;
    }
    match val {
        serde_json::Value::Array(arr) => arr.iter().any(value_contains_complete_hello),
        serde_json::Value::Object(o) => o.values().any(value_contains_complete_hello),
        _ => false,
    }
}

fn score_json_object(val: &serde_json::Value) -> (f32, String) {
    let word = val
        .get("word")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let length = val.get("length").and_then(json_length_u64);
    let reversed = val
        .get("reversed")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let has_word = word.is_some();
    let has_length = length.is_some();
    let has_reversed = reversed.is_some();
    let field_count = u32::from(has_word) + u32::from(has_length) + u32::from(has_reversed);

    if word == Some("hello") && length == Some(5) && reversed == Some("olleh") {
        (
            1.0,
            "Valid JSON with all required fields and correct types".to_string(),
        )
    } else if field_count == 3 {
        (
            0.5,
            "Valid JSON types but word/length/reversed do not match hello".to_string(),
        )
    } else {
        let partial = field_count as f32 / 6.0 + 0.1;
        (
            partial,
            format!("Valid JSON but only {field_count}/3 required fields present"),
        )
    }
}

fn next_json_object(text: &str, start: usize) -> Option<(serde_json::Value, usize)> {
    next_json_delimited(text, start, '{', '}').and_then(|(val, end)| {
        if val.is_object() {
            Some((val, end))
        } else {
            None
        }
    })
}

fn next_json_array(text: &str, start: usize) -> Option<(serde_json::Value, usize)> {
    next_json_delimited(text, start, '[', ']').and_then(|(val, end)| {
        if val.is_array() {
            Some((val, end))
        } else {
            None
        }
    })
}

fn next_json_delimited(
    text: &str,
    start: usize,
    open: char,
    close: char,
) -> Option<(serde_json::Value, usize)> {
    let end = delimited_span_end(text, start, open, close)?;
    let val: serde_json::Value = serde_json::from_str(&text[start..end]).ok()?;
    Some((val, end))
}

fn delimited_span_end(text: &str, start: usize, open: char, close: char) -> Option<usize> {
    let slice = &text[start..];
    if !slice.starts_with(open) {
        return None;
    }
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (off, c) in slice.char_indices() {
        if in_str {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            continue;
        }
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(start + off + 1);
            }
        }
    }
    None
}

fn inner_has_complete_hello(text: &str) -> bool {
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with('{') {
            if let Some((val, end)) = next_json_object(text, i) {
                if value_contains_complete_hello(&val) {
                    return true;
                }
                i = end;
                continue;
            }
        }
        i += text[i..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

fn json_length_u64(v: &serde_json::Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(s) = v.as_str() {
        let t = s.trim();
        if let Ok(n) = t.parse::<u64>() {
            return Some(n);
        }
        if let Ok(n) = t.parse::<f64>() {
            if n.is_finite() && n >= 0.0 && n.fract() == 0.0 {
                return Some(n as u64);
            }
        }
        return None;
    }
    let n = v.as_f64()?;
    if n.is_finite() && n >= 0.0 && n.fract() == 0.0 {
        Some(n as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn json_output_strong_for_valid_json_all_fields() {
        let llm = MockLlm {
            response: text_response(r#"{"word": "hello", "length": 5, "reversed": "olleh"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn json_output_weak_for_invalid_json() {
        let llm = MockLlm {
            response: text_response("The word hello has 5 letters and reversed is olleh."),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn json_output_partial_credit_for_missing_fields() {
        let llm = MockLlm {
            response: text_response(r#"{"word": "hello"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert!(result.score > 0.0);
        assert!(result.score < 1.0);
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_output_card_then_hello_object_is_strong() {
        let llm = MockLlm {
            response: text_response(
                "```json\n{\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}\n```\n\n\
                 {\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_output_utf8_prose_then_hello_object_is_strong() {
        let llm = MockLlm {
            response: text_response(
                "Voici café 日本語\n{\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_output_prompt_example_is_not_strong() {
        let llm = MockLlm {
            response: text_response(
                r#"The example was {"word": "cat", "length": 3, "reversed": "tac"} but I cannot emit JSON."#,
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "echoed prompt example must not skip JSON repair: {result:?}"
        );
    }

    #[tokio::test]
    async fn json_output_not_strong_for_empty_or_whitespace_strings() {
        for body in [
            r#"{"word":"","length":0,"reversed":""}"#,
            r#"{"word":" ","length":0,"reversed":"\t"}"#,
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_json_output(&llm).await.unwrap();
            assert!(result.score < 1.0, "{body}: {}", result.details);
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "empty json strings must not skip repair: {body}"
            );
        }
    }

    #[tokio::test]
    async fn json_nested_hello_object_is_strong() {
        let llm = MockLlm {
            response: text_response(
                r#"{"data": {"word": "hello", "length": 5, "reversed": "olleh"}}"#,
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "nested hello object must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_hello_before_fenced_array_is_strong() {
        let llm = MockLlm {
            response: text_response(
                "{\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}\n\n\
                 ```json\n[{\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}]\n```",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "hello object before a fenced cat array must stay Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_array_then_hello_object_is_strong() {
        let llm = MockLlm {
            response: text_response(
                "```json\n[{\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}]\n```\n\n\
                 {\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "fenced cat array must not abort pick-best of a later hello object: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_example_object_then_array_wrapped_hello_is_not_strong() {
        let llm = MockLlm {
            response: text_response(
                "Example: {\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}\n\
                 [{\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}]",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.score, 0.0,
            "array-wrapped hello after an example object must stay Weak: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn json_trailing_comma_array_wrapped_hello_is_not_strong() {
        let llm = MockLlm {
            response: text_response(r#"[{"word": "hello", "length": 5, "reversed": "olleh"},]"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "trailing-comma array wrap must stay Weak (needs_json_repair): {result:?}"
        );
        assert_eq!(result.score, 0.0, "{result:?}");
    }

    #[tokio::test]
    async fn json_commented_array_wrapped_hello_is_not_strong() {
        let llm = MockLlm {
            response: text_response(
                "[\n  // hello\n  {\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}\n]",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "commented array wrap must stay Weak (needs_json_repair): {result:?}"
        );
        assert_eq!(result.score, 0.0, "{result:?}");
    }

    #[tokio::test]
    async fn json_output_array_wrapped_object_is_not_strong() {
        for body in [
            r#"[{"word": "hello", "length": 5, "reversed": "olleh"}]"#,
            r#"Here: [{"word": "hello", "length": 5, "reversed": "olleh"}]"#,
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_json_output(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "array-wrapped JSON must not skip repair: {body} {result:?}"
            );
            assert!(
                result.details.to_lowercase().contains("array"),
                "{}",
                result.details
            );
        }
    }

    #[tokio::test]
    async fn json_output_object_after_citation_is_strong() {
        let llm = MockLlm {
            response: text_response(
                r#"See [1] {"word": "hello", "length": 5, "reversed": "olleh"}"#,
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn json_output_strong_for_string_length() {
        let llm = MockLlm {
            response: text_response(r#"{"word":"hello","length":"5","reversed":"olleh"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "quoted length 5 must count as 5: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_output_strong_for_float_length() {
        let llm = MockLlm {
            response: text_response(r#"{"word":"hello","length":5.0,"reversed":"olleh"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
    }

    #[tokio::test]
    async fn json_output_strong_for_markdown_wrapped_json() {
        let llm = MockLlm {
            response: text_response(
                "```json\n{\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}\n```",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn instruction_following_weak_for_empty() {
        let llm = MockLlm {
            response: text_response(""),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.details, "Empty response");
    }

    #[tokio::test]
    async fn instruction_following_fenced_paris_is_strong() {
        let llm = MockLlm {
            response: text_response("```\nParis\n```"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn instruction_following_strong_for_single_word() {
        let llm = MockLlm {
            response: text_response("Paris"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn instruction_following_wrong_single_word_is_not_strong() {
        for word in ["The", "Stop", "London", "Yes"] {
            let llm = MockLlm {
                response: text_response(word),
            };
            let result = probe_instruction_following(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "wrong one-word answer must not set overall Strong: {word} -> {result:?}"
            );
            assert_eq!(result.score, 0.5, "{word}");
            assert_eq!(result.level, CapabilityLevel::Medium, "{word}");
        }
    }

    #[tokio::test]
    async fn instruction_following_paris_is_case_and_punct_insensitive() {
        for word in ["paris", "PARIS", "Paris."] {
            let llm = MockLlm {
                response: text_response(word),
            };
            let result = probe_instruction_following(&llm).await.unwrap();
            assert_eq!(result.score, 1.0, "{word}");
            assert_eq!(result.level, CapabilityLevel::Strong, "{word}");
        }
    }

    #[tokio::test]
    async fn instruction_following_weak_for_verbose() {
        let llm = MockLlm {
            response: text_response(
                "The capital of France is Paris, which is also the largest city in the country.",
            ),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn instruction_following_medium_for_short_multiword() {
        let llm = MockLlm {
            response: text_response("It is Paris"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn json_output_length_incomplete_is_transient() {
        let llm = MockLlm {
            response: length_text_response(r#"{"word": "hello", "length":"#),
        };
        let result = probe_json_output(&llm).await;
        assert!(
            matches!(result, Err(ProbeError::Transient(_))),
            "Length plus truncated JSON must be Transient, not 30-day Weak; got {result:?}"
        );
    }

    #[tokio::test]
    async fn json_output_length_complete_stays_strong() {
        let llm = MockLlm {
            response: length_text_response(
                r#"{"word": "hello", "length": 5, "reversed": "olleh"}"#,
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn instruction_following_length_empty_is_transient() {
        let llm = MockLlm {
            response: length_text_response(""),
        };
        let result = probe_instruction_following(&llm).await;
        assert!(
            matches!(result, Err(ProbeError::Transient(_))),
            "Length plus empty instruction reply must be Transient; got {result:?}"
        );
    }

    #[tokio::test]
    async fn instruction_following_length_complete_word_stays_strong() {
        let llm = MockLlm {
            response: length_text_response("Paris"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }
}
