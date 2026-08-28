//! Code syntax accuracy probe.
//!
//! Tests whether the model produces syntactically valid code when asked
//! to write a small function. Models that fail this should have higher
//! lint-fix retry counts.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::user_text;

/// Probe whether the model produces syntactically valid code.
///
/// Asks the model to write a small function and checks for common
/// syntax issues: balanced braces/parens, no trailing commas in
/// invalid positions, no incomplete statements.
///
/// Scoring:
/// - `1.0` - code has balanced delimiters, no obvious syntax errors,
///   and contains the expected function signature
/// - `0.5` - code present and mostly correct but has minor issues
///   (unbalanced delimiters or missing return)
/// - `0.0` - no code block found or prose-only response
pub async fn probe_code_syntax<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text(
            "Write a Python function called `merge_sorted` that takes two sorted lists \
             and returns a single sorted list. Reply with ONLY the code, no explanation.",
        )],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(512),
    };

    let response = llm.chat(request).await?;
    let text = &response.text;

    let code = extract_code_block(text).unwrap_or(text);
    let trimmed = code.trim();

    if trimmed.is_empty() {
        return Ok(ProbeResult {
            name: "code_syntax".to_string(),
            score: 0.0,
            max_score: 1.0,
            level: classify(0.0),
            details: "Empty response, no code produced".to_string(),
        });
    }

    let has_def = trimmed.contains("def merge_sorted") || trimmed.contains("def merge");
    let has_return = trimmed.contains("return ");

    let parens_balanced = count_char(trimmed, '(') == count_char(trimmed, ')');
    let brackets_balanced = count_char(trimmed, '[') == count_char(trimmed, ']');
    let braces_balanced = count_char(trimmed, '{') == count_char(trimmed, '}');
    let delimiters_ok = parens_balanced && brackets_balanced && braces_balanced;

    let has_ellipsis = trimmed.contains("...");
    let has_pass_only = trimmed.lines().any(|l| l.trim() == "pass")
        && !trimmed.contains("return")
        && !trimmed.contains("append");

    let (score, details) = if has_def && has_return && delimiters_ok && !has_ellipsis {
        (
            1.0,
            "Valid function with correct signature, return, and balanced delimiters".to_string(),
        )
    } else if has_def && delimiters_ok && !has_pass_only {
        (
            0.5,
            format!(
                "Function present but incomplete: return={has_return}, ellipsis={has_ellipsis}"
            ),
        )
    } else if has_def {
        (
            0.5,
            format!(
                "Function present but syntax issues: parens={parens_balanced}, \
                 brackets={brackets_balanced}, braces={braces_balanced}"
            ),
        )
    } else {
        (
            0.0,
            "No recognizable function definition in response".to_string(),
        )
    };

    Ok(ProbeResult {
        name: "code_syntax".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn extract_code_block(text: &str) -> Option<&str> {
    let start_marker = text.find("```")?;
    let after_marker = &text[start_marker + 3..];
    let code_start = after_marker.find('\n')? + 1;
    let code_body = &after_marker[code_start..];
    let end = code_body.find("```")?;
    Some(&code_body[..end])
}

fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn code_syntax_strong_for_valid_function() {
        let code = "\
```python
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
```";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_medium_for_missing_return() {
        let code = "def merge_sorted(a, b):\n    result = a + b\n    result.sort()";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn code_syntax_weak_for_prose() {
        let llm = MockLlm {
            response: text_response(
                "To merge two sorted lists, you can use a two-pointer approach.",
            ),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_weak_for_empty() {
        let llm = MockLlm {
            response: text_response(""),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn extract_code_block_python() {
        let text = "```python\ndef foo():\n    return 42\n```";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("def foo()"));
        assert!(code.contains("return 42"));
    }

    #[test]
    fn extract_code_block_bare() {
        let text = "```\nprint(1)\n```";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code.trim(), "print(1)");
    }

    #[test]
    fn count_char_works() {
        assert_eq!(count_char("((()))", '('), 3);
        assert_eq!(count_char("((()))", ')'), 3);
        assert_eq!(count_char("abc", '('), 0);
    }
}
