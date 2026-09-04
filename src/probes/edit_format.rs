//! Edit-format probes: SEARCH/REPLACE and unified diff.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete, system_text, user_text};

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
        let mut any_greet = false;
        let mut old_fn = false;
        let mut old_hello = false;
        let mut new_fn = false;
        let mut new_welcome = false;
        for block in parse_search_replace_blocks(&text) {
            let has_file_ref = block.path.contains("greet.rs");
            let search = strip_comments(block.search);
            let replace = strip_comments(block.replace);
            // Strong requires function-definition lines, not lecture tokens.
            let block_old_fn = has_code_fn_token(&search, "fn greet(");
            let block_old_hello = search.contains("Hello");
            let block_new_fn = has_code_fn_token(&replace, "fn welcome(");
            let block_new_welcome = replace.contains("Welcome");
            let has_old_content = block_old_fn && block_old_hello;
            let has_new_content = block_new_fn && block_new_welcome;
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
            if has_file_ref {
                any_greet = true;
            }
            old_fn |= block_old_fn;
            old_hello |= block_old_hello;
            new_fn |= block_new_fn;
            new_welcome |= block_new_welcome;
        }
        if any_greet && old_fn && old_hello && new_fn && new_welcome {
            best = 1.0;
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

    refuse_truncated_incomplete(response.finish, score)?;
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

    let has_file_ref = text
        .lines()
        .any(|l| (l.starts_with("---") || l.starts_with("+++")) && l.contains("greet.rs"));
    let minus_has_greet = text.lines().any(|l| {
        l.starts_with('-')
            && !l.starts_with("---")
            && has_code_fn_token(&strip_comments(&l[1..]), "fn greet(")
    });
    let plus_has_welcome = text.lines().any(|l| {
        l.starts_with('+')
            && !l.starts_with("+++")
            && has_code_fn_token(&strip_comments(&l[1..]), "fn welcome(")
    });
    let minus_has_hello = text.lines().any(|l| {
        l.starts_with('-') && !l.starts_with("---") && strip_comments(&l[1..]).contains("Hello")
    });
    let plus_has_welcome_text = text.lines().any(|l| {
        l.starts_with('+') && !l.starts_with("+++") && strip_comments(&l[1..]).contains("Welcome")
    });
    let real_edit = has_file_ref
        && minus_has_greet
        && plus_has_welcome
        && minus_has_hello
        && plus_has_welcome_text;

    let (score, details) = if is_unified_diff_card_echo(&text) && !real_edit {
        (0.0, "Echoed the unified diff format card".to_string())
    } else if has_minus_header && has_plus_header && has_hunk && has_minus_line && has_plus_line {
        if real_edit {
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

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "unified_diff".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// `fn greet(` / `fn welcome(` must start a line (after indent), not sit mid-sentence.
/// The rest of the line must look like a Rust signature, not English (`should say`).
fn has_code_fn_token(text: &str, token: &str) -> bool {
    text.lines().any(|line| {
        let t = line.trim_start();
        let after = if t.starts_with(token) {
            &t[token.len()..]
        } else if let Some(rest) = t.strip_prefix("pub ") {
            let rest = rest.trim_start();
            if rest.starts_with(token) {
                &rest[token.len()..]
            } else {
                return false;
            }
        } else {
            return false;
        };
        looks_like_rust_fn_signature_rest(after)
    })
}

/// After `fn greet(`, the rest must close the params and then be a type/`{`/`;`.
fn looks_like_rust_fn_signature_rest(after_open_paren: &str) -> bool {
    let mut depth = 1i32;
    let mut close = None;
    for (i, c) in after_open_paren.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return false;
    };
    let after = after_open_paren[close + 1..].trim();
    let after = after
        .strip_suffix('{')
        .or_else(|| after.strip_suffix(';'))
        .map(str::trim)
        .unwrap_or(after);
    if after.is_empty() {
        return true;
    }
    let Some(ty) = after.strip_prefix("->") else {
        return false;
    };
    is_rust_type_tokens(ty.trim().trim_end_matches(['{', ';']).trim())
}

fn is_rust_type_tokens(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut saw_ident = false;
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if matches!(
            c,
            b':' | b'<' | b'>' | b'&' | b'\'' | b',' | b'[' | b']' | b'(' | b')' | b'+' | b'*'
        ) {
            i += 1;
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &s[start..i];
            if matches!(
                word,
                "should"
                    | "say"
                    | "the"
                    | "a"
                    | "an"
                    | "to"
                    | "and"
                    | "or"
                    | "for"
                    | "with"
                    | "from"
                    | "that"
                    | "this"
                    | "is"
                    | "be"
                    | "no"
                    | "longer"
                    | "use"
                    | "change"
                    | "rename"
            ) {
                return false;
            }
            saw_ident = true;
            continue;
        }
        return false;
    }
    saw_ident
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
    !strip_comments(&line[1..]).trim().is_empty()
}

fn strip_comments(text: &str) -> String {
    strip_line_comments(&strip_block_comments(text))
}

fn strip_block_comments(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 4 + end..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

fn is_rust_attribute(text: &str) -> bool {
    text.starts_with("#[") || text.starts_with("#!")
}

fn strip_inline_hash_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let rest = &line[i..];
            if is_rust_attribute(rest) {
                i += 1;
                continue;
            }
            return &line[..i];
        }
        i += 1;
    }
    line
}

fn strip_line_comments(text: &str) -> String {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                return None;
            }
            // `#` comments (Python/shell). Rust attributes (`#[`, `#!`) stay.
            if trimmed.starts_with('#') && !is_rust_attribute(trimmed) {
                return None;
            }
            let without_slashes = match line.split_once("//") {
                Some((code, _)) => code,
                None => line,
            };
            Some(strip_inline_hash_comment(without_slashes))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collapse ASCII whitespace, hyphens, underscores, and ZWSP/format
/// marks so `removed-line`, `removed_line`, `removed  line`, and
/// `removed\u{200B}line` match the format-card tokens. CamelCase is
/// left intact.
fn normalize_unified_diff_card_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .replace(['-', '_'], " ")
        .chars()
        .map(|c| match c {
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' => ' ',
            _ => c,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// True when the reply is the system format example, not an edit of greet.rs.
fn is_unified_diff_card_echo(text: &str) -> bool {
    let normalized = normalize_unified_diff_card_text(text);
    let removed_added = normalized.contains("removed line") && normalized.contains("added line");
    let remove_add = normalized.contains("remove line") && normalized.contains("add line");
    removed_added || remove_add
}

struct SearchReplaceBlock<'a> {
    path: &'a str,
    search: &'a str,
    replace: &'a str,
}

fn looks_like_path(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() || t.contains("fn ") {
        return false;
    }
    t.contains('/')
        || t.rsplit_once('.').is_some_and(|(_, ext)| {
            !ext.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric())
        })
}

fn is_dash_separator(line: &str) -> bool {
    let t = line.trim();
    t.len() >= 3 && t.bytes().all(|b| b == b'-')
}

fn split_path_and_search(header: &str) -> (&str, &str) {
    let Some((first, rest)) = header.split_once('\n') else {
        if looks_like_path(header) {
            return (header.trim(), "");
        }
        return ("", header);
    };
    if looks_like_path(first) {
        if let Some((maybe_dash, after_dash)) = rest.split_once('\n') {
            if is_dash_separator(maybe_dash) {
                return (first.trim(), after_dash);
            }
        }
        if is_dash_separator(rest) {
            return (first.trim(), "");
        }
        return (first.trim(), rest);
    }
    if is_dash_separator(first) {
        return ("", rest);
    }
    ("", header)
}

fn path_line_before(before: &str) -> &str {
    before
        .lines()
        .rev()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(">>>>>>>"))
        .find(|l| looks_like_path(l))
        .unwrap_or("")
}

fn parse_search_replace_blocks(text: &str) -> Vec<SearchReplaceBlock<'_>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    let mut last_path = "";
    const START: &str = "<<<<<<< SEARCH";
    const SEP: &str = "\n=======\n";
    const END: &str = "\n>>>>>>> REPLACE";
    while let Some(start) = text.get(offset..).and_then(|s| s.find(START)) {
        let abs_start = offset + start;
        let before = &text[..abs_start];
        let after_marker = &text[abs_start + START.len()..];
        let after_nl = after_marker.strip_prefix('\n').unwrap_or(after_marker);
        let Some(sep) = after_nl.find(SEP) else {
            offset = abs_start + START.len();
            continue;
        };
        let (mut path, search) = split_path_and_search(&after_nl[..sep]);
        if path.is_empty() {
            path = path_line_before(before);
        }
        if path.is_empty() {
            path = last_path;
        } else {
            last_path = path;
        }
        let after_sep = &after_nl[sep + SEP.len()..];
        let Some(end) = after_sep.find(END) else {
            offset = text.len() - after_sep.len();
            continue;
        };
        blocks.push(SearchReplaceBlock {
            path,
            search,
            replace: &after_sep[..end],
        });
        offset = text.len() - after_sep.len() + end + END.len();
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
    async fn search_replace_block_comment_tokens_are_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
/* keep fn greet( and Hello */
=======
/* emit fn welcome( and Welcome */
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "block-comment tokens must not set SearchReplace: {result:?}"
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
    async fn search_replace_hash_comment_tokens_are_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
# fn greet( ... Hello
=======
# fn welcome( ... Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "hash-comment tokens must not set SearchReplace: {result:?}"
        );
    }

    #[tokio::test]
    async fn search_replace_hash_attribute_tokens_remain_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
#[derive(Debug)]
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
=======
#[derive(Debug)]
fn welcome(name: &str) -> String {
    format!(\"Welcome, {}\", name)
}
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Strong,
            "Rust attributes must not be stripped as comments: {result:?}"
        );
        assert_eq!(result.score, 1.0);
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
    async fn search_replace_lecture_signature_line_is_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
fn greet(name: &str) -> String should say Hello
=======
fn welcome(name: &str) -> String should say Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.score, 1.0,
            "start-of-line lecture signatures must not be Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_lecture_fn_greet_paren_is_not_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
Change fn greet( so it no longer says Hello
=======
Use fn welcome( and say Welcome
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_ne!(
            result.score, 1.0,
            "lecture fn greet( mid-sentence must not be Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_two_one_edit_blocks_are_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
fn greet(name: &str) -> String {
=======
fn welcome(name: &str) -> String {
>>>>>>> REPLACE
<<<<<<< SEARCH
src/greet.rs
    format!(\"Hello, {}\", name)
=======
    format!(\"Welcome, {}\", name)
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "two correct one-block-per-edit SEARCH/REPLACE blocks must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_path_once_split_blocks_is_strong() {
        let response_text = "\
src/greet.rs
<<<<<<< SEARCH
fn greet(name: &str) -> String {
=======
fn welcome(name: &str) -> String {
>>>>>>> REPLACE
<<<<<<< SEARCH
    format!(\"Hello, {}\", name)
=======
    format!(\"Welcome, {}\", name)
>>>>>>> REPLACE";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_search_replace(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "path-once Aider split SEARCH/REPLACE must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn search_replace_without_dash_separator_is_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
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
        assert_eq!(
            result.score, 1.0,
            "Aider-style block without ------- must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_path_above_block_is_strong() {
        let response_text = "\
src/greet.rs
<<<<<<< SEARCH
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
        assert_eq!(
            result.score, 1.0,
            "filename above SEARCH must count as the path: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn unified_diff_rename_without_welcome_text_is_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,3 +1,3 @@
-fn greet(name: &str) -> String {
+fn welcome(name: &str) -> String {
     format!(\"Hello, {}\", name)
 }";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "rename without Hello/Welcome body must not be Strong: {result:?}"
        );
        assert_eq!(result.score, 0.5, "{result:?}");
    }

    #[tokio::test]
    async fn unified_diff_block_comment_tokens_are_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-/* fn greet */
+/* fn welcome */
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "block-comment +/- must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
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
    async fn unified_diff_hash_comment_tokens_are_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-# fn greet
+# fn welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "hash-comment-only +/- must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn unified_diff_hash_attribute_tokens_remain_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,4 +1,4 @@
 #[derive(Debug)]
-fn greet(name: &str) -> String {
-    format!(\"Hello, {}\", name)
+fn welcome(name: &str) -> String {
+    format!(\"Welcome, {}\", name)
 }
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Strong,
            "Rust attributes must not block UnifiedDiff Strong: {result:?}"
        );
        assert_eq!(result.score, 1.0);
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
    async fn unified_diff_card_then_real_edit_is_strong() {
        let response_text = "\
The format uses -removed line and +added line.

--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,3 +1,3 @@
-fn greet(name: &str) -> String {
-    format!(\"Hello, {}\", name)
+fn welcome(name: &str) -> String {
+    format!(\"Welcome, {}\", name)
 }
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
    }

    #[tokio::test]
    async fn unified_diff_name_only_is_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-fn greet
+fn welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(result.level, CapabilityLevel::Strong, "{result:?}");
    }

    #[tokio::test]
    async fn unified_diff_lecture_signature_line_is_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-fn greet(name: &str) -> String should say Hello
+fn welcome(name: &str) -> String should say Welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.score, 1.0,
            "start-of-line lecture signatures must not be UnifiedDiff Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn unified_diff_lecture_fn_greet_paren_is_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-Change fn greet( so it no longer says Hello
+Use fn welcome( and say Welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.score, 1.0,
            "lecture fn greet( mid-sentence must not be UnifiedDiff Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn unified_diff_card_echo_with_greet_path_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
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
            "card body with greet.rs path must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn unified_diff_title_case_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-Removed line
+Added line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "title-case card body must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_remove_add_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-remove line
+add line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "remove/add card body must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_hyphenated_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-removed-line
+added-line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "hyphenated card body must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_underscore_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-removed_line
+added_line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "underscore card body must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_camel_case_removed_line_is_not_card_echo() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-removedLine
+addedLine
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.details, "Echoed the unified diff format card",
            "camelCase removedLine must not be treated as a format card: {result:?}"
        );
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "camelCase body is a real-looking diff, not a card echo: {result:?}"
        );
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn unified_diff_double_space_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-removed  line
+added  line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "double-space card body must not set UnifiedDiff: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn unified_diff_zwsp_card_echo_is_not_medium() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -10,4 +10,5 @@
 context line
-removed\u{200B}line
+added\u{200B}line
 context line
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "ZWSP card body must not set UnifiedDiff: {result:?}"
        );
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

    #[tokio::test]
    async fn search_replace_length_partial_markers_is_transient() {
        let llm = MockLlm {
            response: length_text_response(
                "<<<<<<< SEARCH\nsrc/greet.rs\n-------\nfn greet(name: &str) -> String {\n",
            ),
        };
        let result = probe_search_replace(&llm).await;
        assert!(
            matches!(result, Err(ProbeError::Transient(_))),
            "Length plus a cut SEARCH block must be Transient, not 30-day Medium; got {result:?}"
        );
    }

    #[tokio::test]
    async fn unified_diff_length_partial_is_transient() {
        let llm = MockLlm {
            response: length_text_response(
                "--- a/src/greet.rs\n+++ b/src/greet.rs\n@@\n-fn greet\n",
            ),
        };
        let result = probe_unified_diff(&llm).await;
        assert!(
            matches!(result, Err(ProbeError::Transient(_))),
            "Length plus a cut unified diff must be Transient; got {result:?}"
        );
    }
}
