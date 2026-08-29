//! Edit-format probes: SEARCH/REPLACE and unified diff.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{system_text, user_text};

/// Probe whether the model can produce valid SEARCH/REPLACE edit blocks.
///
/// Sends a small file as context and asks the model to rename a variable
/// using the exact SEARCH/REPLACE format.
///
/// Scoring:
/// - `1.0` - parseable block with correct markers and matching content
/// - `0.7` - markers present and parseable but old_content has drift
/// - `0.4` - markers present but block is not parseable
/// - `0.0` - no SEARCH/REPLACE markers at all
pub async fn probe_search_replace<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let system = "\
When editing files, use SEARCH/REPLACE blocks. Each block identifies the \
file, the exact lines to find, and their replacement:

<<<<<<< SEARCH
path/to/file.rs
-------
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

Rules:
- The SEARCH section must match the existing file content exactly.
- Include enough surrounding context to uniquely identify the location.
- Use one block per edit.";

    let user = "\
Here is the file `src/greet.rs`:

```rust
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
```

Rename the function `greet` to `welcome` and change the greeting from \
`Hello` to `Welcome`. Use the SEARCH/REPLACE format.";

    let request = ProbeRequest {
        messages: vec![system_text(system), user_text(user)],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(500),
    };

    let response = llm.chat(request).await?;
    let text = response.text;

    let has_search = text.contains("<<<<<<< SEARCH");
    let has_separator = text.contains("=======");
    let has_replace = text.contains(">>>>>>> REPLACE");

    let (score, details) = if has_search && has_separator && has_replace {
        let mut best = 0.4_f64;
        for block in parse_search_replace_blocks(&text) {
            let has_file_ref = block.path.contains("greet.rs");
            let search = strip_line_comments(block.search);
            let replace = strip_line_comments(block.replace);
            // Strong requires the function signatures in code, not comments.
            let has_old_content = search.contains("fn greet(") && search.contains("Hello");
            let has_new_content = replace.contains("fn welcome(") && replace.contains("Welcome");
            let block_score = if has_file_ref && has_old_content && has_new_content {
                1.0
            } else if has_old_content || has_new_content {
                0.7
            } else {
                0.4
            };
            if block_score > best {
                best = block_score;
            }
        }
        if (best - 1.0).abs() < f64::EPSILON {
            (
                1.0,
                "Valid SEARCH/REPLACE block with correct content".to_string(),
            )
        } else if best > 0.4 {
            (
                0.7,
                "Parseable SEARCH/REPLACE block but content has drift".to_string(),
            )
        } else {
            (
                0.4,
                "SEARCH/REPLACE markers present but content unclear".to_string(),
            )
        }
    } else if has_search || has_replace {
        (0.4, "Partial SEARCH/REPLACE markers found".to_string())
    } else {
        (0.0, "No SEARCH/REPLACE markers in response".to_string())
    };

    Ok(ProbeResult {
        name: "search_replace".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model can produce valid unified diff output.
///
/// Sends the same small file and asks for a unified diff format rename.
///
/// Scoring:
/// - `1.0` - parseable diff with correct headers and +/- lines
/// - `0.5` - parseable but with wrong paths or extra junk
/// - `0.0` - not a valid unified diff
pub async fn probe_unified_diff<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let system = "\
When editing files, output a unified diff. Use the standard format with \
file headers and hunk markers:

```diff
--- a/path/to/file.rs
+++ b/path/to/file.rs
@@ -10,4 +10,5 @@
 context line
-removed line
+added line
 context line
```

Rules:
- Include enough context lines for unambiguous placement.
- Use one diff per file.";

    let user = "\
Here is the file `src/greet.rs`:

```rust
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
```

Rename the function `greet` to `welcome` and change the greeting from \
`Hello` to `Welcome`. Output a unified diff.";

    let request = ProbeRequest {
        messages: vec![system_text(system), user_text(user)],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(500),
    };

    let response = llm.chat(request).await?;
    let text = response.text;

    let has_minus_header = text.lines().any(|l| l.starts_with("---"));
    let has_plus_header = text.lines().any(|l| l.starts_with("+++"));
    let has_hunk = text.lines().any(|l| l.starts_with("@@"));
    let has_minus_line = text.lines().any(|l| is_code_diff_line(l, '-'));
    let has_plus_line = text.lines().any(|l| is_code_diff_line(l, '+'));

    let (score, details) = if is_unified_diff_card_echo(&text) {
        (0.0, "Echoed the unified diff format card".to_string())
    } else if has_minus_header && has_plus_header && has_hunk && has_minus_line && has_plus_line {
        let has_file_ref = text
            .lines()
            .any(|l| (l.starts_with("---") || l.starts_with("+++")) && l.contains("greet.rs"));
        let minus_has_greet = text.lines().any(|l| {
            l.starts_with('-')
                && !l.starts_with("---")
                && strip_line_comments(&l[1..]).contains("fn greet")
        });
        let plus_has_welcome = text.lines().any(|l| {
            l.starts_with('+')
                && !l.starts_with("+++")
                && strip_line_comments(&l[1..]).contains("fn welcome")
        });
        if has_file_ref && minus_has_greet && plus_has_welcome {
            (1.0, "Valid unified diff with correct +/- lines".to_string())
        } else {
            (0.5, "Valid diff structure but content unclear".to_string())
        }
    } else if (has_hunk || (has_minus_header && has_plus_header)) && has_minus_line && has_plus_line
    {
        (
            0.5,
            "Has diff lines but missing headers or hunk markers".to_string(),
        )
    } else {
        (0.0, "Not a recognizable unified diff".to_string())
    };

    Ok(ProbeResult {
        name: "unified_diff".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn is_code_diff_line(line: &str, mark: char) -> bool {
    if mark == '-' && line.starts_with("---") {
        return false;
    }
    if mark == '+' && line.starts_with("+++") {
        return false;
    }
    if !line.starts_with(mark) {
        return false;
    }
    !strip_line_comments(&line[1..]).trim().is_empty()
}

fn strip_line_comments(text: &str) -> String {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                return None;
            }
            Some(match line.split_once("//") {
                Some((code, _)) => code,
                None => line,
            })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// True when the reply is the system format example, not an edit of greet.rs.
fn is_unified_diff_card_echo(text: &str) -> bool {
    text.contains("path/to/file.rs") && text.contains("removed line") && text.contains("added line")
}

struct SearchReplaceBlock<'a> {
    path: &'a str,
    search: &'a str,
    replace: &'a str,
}

fn parse_search_replace_blocks(text: &str) -> Vec<SearchReplaceBlock<'_>> {
    let mut blocks = Vec::new();
    let mut rest = text;
    const START: &str = "<<<<<<< SEARCH";
    const DASH: &str = "\n-------\n";
    const SEP: &str = "\n=======\n";
    const END: &str = "\n>>>>>>> REPLACE";
    while let Some(start) = rest.find(START) {
        let after_marker = &rest[start + START.len()..];
        let after_nl = after_marker.strip_prefix('\n').unwrap_or(after_marker);
        let Some(dash) = after_nl.find(DASH) else {
            rest = after_marker;
            continue;
        };
        let path = after_nl[..dash].trim();
        let after_dash = &after_nl[dash + DASH.len()..];
        let Some(sep) = after_dash.find(SEP) else {
            rest = after_dash;
            continue;
        };
        let search = &after_dash[..sep];
        let after_sep = &after_dash[sep + SEP.len()..];
        let Some(end) = after_sep.find(END) else {
            rest = after_sep;
            continue;
        };
        blocks.push(SearchReplaceBlock {
            path,
            search,
            replace: &after_sep[..end],
        });
        rest = &after_sep[end + END.len()..];
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn search_replace_strong_for_valid_block() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
=======
fn welcome(name: &str) -> String {
    format!(\"Welcome, {}\", name)
}
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn search_replace_prompt_echo_is_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
path/to/file.rs
-------
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

Rename greet to welcome and Hello to Welcome in src/greet.rs
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "echoed system markers plus user words must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn search_replace_comment_tokens_are_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
// keep fn greet( and Hello
=======
// emit fn welcome( and Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "comment tokens must not set SearchReplace: {result:?}"
        );
    }

    #[tokio::test]
    async fn search_replace_fn_tokens_without_signature_are_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
fn greet Hello
=======
fn welcome Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "fn greet without a signature must not set SearchReplace: {result:?}"
        );
    }

    #[tokio::test]
    async fn search_replace_prose_stuffing_is_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
Rename greet Hello
=======
Rename welcome Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "prose stuffing greet/Hello/welcome/Welcome must not set SearchReplace: {result:?}"
        );
    }

    #[tokio::test]
    async fn search_replace_weak_for_prose() {
        let llm = MockLlm {
            response: text_response("You should rename greet to welcome in the file."),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn search_replace_medium_for_partial_markers() {
        let response_text = "<<<<<<< SEARCH\nsome content\n>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn unified_diff_strong_for_valid_diff() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,3 +1,3 @@
-fn greet(name: &str) -> String {
-    format!(\"Hello, {}\", name)
+fn welcome(name: &str) -> String {
+    format!(\"Welcome, {}\", name)
 }";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn unified_diff_comment_tokens_are_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,3 +1,3 @@
 // context
-// drop greet Hello
+// add welcome Welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "comment +/- tokens must not set UnifiedDiff Strong: {result:?}"
        );
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "comment-only +/- must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn unified_diff_prose_markers_are_not_medium() {
        let llm = MockLlm {
            response: text_response("Docs use ---, +++, and @@.\n- greet\n+ welcome\n"),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "prose ---/+++/@@ plus markdown list must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_system_card_echo_is_not_medium() {
        let response_text = "\
--- a/path/to/file.rs
+++ b/path/to/file.rs
@@ -10,4 +10,5 @@
 context line
-removed line
+added line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "system card diff must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_prompt_echo_is_not_strong() {
        let response_text = "\
--- a/path/to/file.rs
+++ b/path/to/file.rs
@@ -10,4 +10,5 @@
 context line
-removed line
+added line
 context line

Rename greet to welcome and Hello to Welcome in src/greet.rs
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "echoed system diff plus user words must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn unified_diff_markdown_list_is_not_medium() {
        let llm = MockLlm {
            response: text_response(
                "Changes I would make:\n- greet function name\n+ welcome function name\n",
            ),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "markdown +/- list must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_weak_for_prose() {
        let llm = MockLlm {
            response: text_response("Change greet to welcome on line 1."),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }
}
