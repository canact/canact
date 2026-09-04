//! Code syntax accuracy probe.
//!
//! Tests whether the model produces syntactically valid code when asked
//! to write a small function. Models that fail this should have higher
//! lint-fix retry counts.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete, user_text};

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

    let fenced = extract_code_block(text);
    let from_fence = fenced.is_some();
    let code = fenced.unwrap_or(text);
    let trimmed = code.trim();

    if trimmed.is_empty() {
        refuse_truncated_incomplete(response.finish, 0.0)?;
        return Ok(ProbeResult {
            name: "code_syntax".to_string(),
            score: 0.0,
            max_score: 1.0,
            level: classify(0.0),
            details: "Empty response, no code produced".to_string(),
        });
    }

    let code_body = strip_python_string_literals(&strip_hash_comments(trimmed));
    // Unfenced prose can name `def merge_sorted` and `return` as tokens.
    // Strong needs a real function (colon + indented body) or a fence.
    let has_def = code_body.contains("def merge_sorted")
        && (from_fence || has_indented_merge_sorted_body(trimmed));
    let has_return = code_body.contains("return ") || code_body.contains("return(");

    let parens_balanced = count_char(trimmed, '(') == count_char(trimmed, ')');
    let brackets_balanced = count_char(trimmed, '[') == count_char(trimmed, ']');
    let braces_balanced = count_char(trimmed, '{') == count_char(trimmed, '}');
    let delimiters_ok = parens_balanced && brackets_balanced && braces_balanced;

    let has_ellipsis = trimmed.lines().any(|l| {
        let t = l.trim().trim_start_matches('#').trim();
        t == "..."
            || t == "...."
            || t == "return ..."
            || t == "return..."
            || t.starts_with("return ...")
            || t.starts_with("return...")
    });
    let has_pass_only = code_body.lines().any(|l| l.trim() == "pass")
        && !code_body.contains("return")
        && !code_body.contains("append");

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

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "code_syntax".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn strip_hash_comments(text: &str) -> String {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                return None;
            }
            Some(line.split_once('#').map(|(code, _)| code).unwrap_or(line))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_python_string_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        let delim = if rest.starts_with("\"\"\"") {
            Some("\"\"\"")
        } else if rest.starts_with("'''") {
            Some("'''")
        } else if rest.starts_with('"') {
            Some("\"")
        } else if rest.starts_with('\'') {
            Some("'")
        } else {
            None
        };
        if let Some(quote) = delim {
            i += quote.len();
            if quote.len() == 1 {
                let closer = quote.as_bytes()[0];
                while i < text.len() {
                    let b = text.as_bytes()[i];
                    i += 1;
                    if b == b'\\' {
                        if i < text.len() {
                            i += 1;
                        }
                        continue;
                    }
                    if b == closer {
                        break;
                    }
                }
            } else {
                match text[i..].find(quote) {
                    Some(rel) => i += rel + quote.len(),
                    None => break,
                }
            }
            continue;
        }
        let ch = rest.chars().next().expect("char");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn has_indented_merge_sorted_body(text: &str) -> bool {
    let needle = "def merge_sorted";
    let mut search = 0;
    while let Some(rel) = text.get(search..).and_then(|s| s.find(needle)) {
        let idx = search + rel;
        let after = &text[idx + needle.len()..];
        if let Some(colon_rel) = after.find(':') {
            if !after[..colon_rel].contains('\n') {
                let rest = &after[colon_rel + 1..];
                if rest
                    .lines()
                    .next()
                    .is_some_and(|line| !line.trim().is_empty())
                {
                    return true;
                }
                for line in rest.lines().skip(1) {
                    if line.trim().is_empty() {
                        continue;
                    }
                    return line.starts_with(' ') || line.starts_with('\t');
                }
            }
        }
        search = idx + needle.len();
    }
    false
}

fn extract_code_block(text: &str) -> Option<&str> {
    let mut search = 0;
    let mut first = None;
    while let Some(rel) = text.get(search..).and_then(|s| s.find("```")) {
        let start_marker = search + rel;
        let after_marker = start_marker + 3;
        let Some(nl) = text.get(after_marker..).and_then(|s| s.find('\n')) else {
            break;
        };
        let code_start = after_marker + nl + 1;
        let Some(end_rel) = text.get(code_start..).and_then(|s| s.find("```")) else {
            break;
        };
        let body = &text[code_start..code_start + end_rel];
        if first.is_none() {
            first = Some(body);
        }
        if body.contains("def merge_sorted") {
            return Some(body);
        }
        search = code_start + end_rel + 3;
    }
    first
}

fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProbeError;
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
    async fn code_syntax_strong_with_docstring_ellipsis() {
        let code = "\
def merge_sorted(a, b):
    \"\"\"Merge two lists...\"\"\"
    return a + b
";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
    }

    #[tokio::test]
    async fn code_syntax_prefers_merge_fence_after_note() {
        let text = "\
```
two-pointer merge
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let llm = MockLlm {
            response: text_response(text),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
    }

    #[tokio::test]
    async fn code_syntax_return_ellipsis_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    return ...\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong, "{result:?}");
    }

    #[tokio::test]
    async fn code_syntax_return_in_comment_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    # return a + b\n    pass\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "return only in a comment must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_return_in_docstring_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists and return a single sorted list.\"\"\"\n    pass\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "return only in a docstring must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_return_in_regular_quotes_is_not_strong() {
        for code in [
            "def merge_sorted(a, b):\n    \"return a merged list\"\n    pass\n",
            "def merge_sorted(a, b):\n    'return a merged list'\n    pass\n",
        ] {
            let llm = MockLlm {
                response: text_response(code),
            };
            let result = probe_code_syntax(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "return only inside regular quotes must not be Strong: {code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_real_return_with_quoted_string_stays_strong() {
        let code =
            "def merge_sorted(a, b):\n    note = \"return a merged list\"\n    return a + b\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "real return a + b must stay Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_return_in_single_quote_docstring_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    '''Merge two sorted lists and return a single sorted list.'''\n    pass\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "return only in a ''' docstring must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_prefers_merge_sorted_fence_after_sketch() {
        let text = "\
```python
def merge(left, right):
    return left + right
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let llm = MockLlm {
            response: text_response(text),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "later def merge_sorted fence must win over a def merge sketch: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_def_merge_without_sorted_is_not_strong() {
        let code = "def merge(a, b):\n    return a + b\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "def merge without merge_sorted must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_strong_for_return_paren() {
        let code = "def merge_sorted(a, b):\n    return(sorted(a + b))";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
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
    async fn code_syntax_unfenced_prose_tokens_are_not_strong() {
        let llm = MockLlm {
            response: text_response(
                "I would write a function def merge_sorted that takes two lists \
                 and return a single sorted list.",
            ),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "unfenced prose that only names def/return tokens must not be Strong: {result:?}"
        );
        assert_eq!(
            result.score, 0.0,
            "tokens in a sentence are not a function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_comment_then_real_unfenced_function_is_strong() {
        let code = "# def merge_sorted merges lists\ndef merge_sorted(a, b):\n    return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "comment naming the token must not hide a real function: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_one_line_function_is_strong() {
        let code = "def merge_sorted(a, b): return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "one-line def merge_sorted must stay Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn length_empty_is_transient() {
        let llm = MockLlm {
            response: length_text_response(""),
        };
        let err = probe_code_syntax(&llm).await.expect_err("must refuse");
        assert!(
            matches!(&err, ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn length_incomplete_function_is_transient() {
        let llm = MockLlm {
            response: length_text_response("def merge_sorted(a, b):\n    result = ["),
        };
        let err = probe_code_syntax(&llm).await.expect_err("must refuse");
        assert!(
            matches!(&err, ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn length_complete_function_stays_strong() {
        let code = "\
```python
def merge_sorted(a, b):
    return a + b
```";
        let llm = MockLlm {
            response: length_text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
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
