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
        let has_file_ref = text.contains("greet.rs");
        let has_old_content = text.contains("greet") && text.contains("Hello");
        let has_new_content = text.contains("welcome") && text.contains("Welcome");

        if has_file_ref && has_old_content && has_new_content {
            (
                1.0,
                "Valid SEARCH/REPLACE block with correct content".to_string(),
            )
        } else if has_old_content || has_new_content {
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

    let has_minus_header = text.contains("--- a/") || text.contains("---");
    let has_plus_header = text.contains("+++ b/") || text.contains("+++");
    let has_hunk = text.contains("@@");
    let has_minus_line = text
        .lines()
        .any(|l| l.starts_with('-') && !l.starts_with("---"));
    let has_plus_line = text
        .lines()
        .any(|l| l.starts_with('+') && !l.starts_with("+++"));

    let (score, details) =
        if has_minus_header && has_plus_header && has_hunk && has_minus_line && has_plus_line {
            let has_correct_content = text.contains("greet") && text.contains("welcome");
            if has_correct_content {
                (1.0, "Valid unified diff with correct +/- lines".to_string())
            } else {
                (0.5, "Valid diff structure but content unclear".to_string())
            }
        } else if has_minus_line && has_plus_line {
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
    async fn unified_diff_weak_for_prose() {
        let llm = MockLlm {
            response: text_response("Change greet to welcome on line 1."),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }
}
