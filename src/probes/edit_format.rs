//! Edit-format probes: SEARCH/REPLACE and unified diff.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete_text, system_text, user_text};

/// Probe whether the model can produce valid SEARCH/REPLACE edit blocks.
///
/// Sends a small file as context and asks the model to rename a variable
/// using the exact SEARCH/REPLACE format.
///
/// Scoring:
/// - `1.0` - parseable block with correct markers and matching content.
///   A `#`, `//`, or `/*` inside a string, raw string, or character
///   does not hide the function or the greeting. A comment that is
///   the whole line still does not count, and `#[derive]` stays.
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

    refuse_truncated_incomplete_text(response.finish, score, &text)?;
    Ok(ProbeResult {
        name: "search_replace".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// System prompt for [`probe_unified_diff`].
///
/// Describes the format without a copyable example. A fenced card with
/// `--- a/path/to/file.rs` and `-removed line` / `+added line` is what
/// models echoed through suite v28–v62.
const UNIFIED_DIFF_SYSTEM: &str = "\
When editing files, output a unified diff. Use the standard format: \
old-file and new-file headers, a hunk header, unchanged context, \
and the requested deletion and insertion.

Rules:
- Include enough context lines for unambiguous placement.
- Use one diff per file.
- Header paths must name the file the user provided.";

/// Probe whether the model can produce valid unified diff output.
///
/// Sends the same small file and asks for a unified diff format rename.
///
/// Scoring:
/// - `1.0` - parseable diff with correct headers and +/- lines.
///   Comment marks inside a string on a `+` or `-` line do not hide
///   the function or the greeting. A `#` comment that is the whole
///   line is still ignored.
/// - `0.5` - parseable diff whose paths or body do not match the rename, or plus and minus lines with a missing header or hunk marker
/// - `0.0` - not a recognizable unified diff, or an echo of the format card
pub async fn probe_unified_diff<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let system = UNIFIED_DIFF_SYSTEM;

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

    refuse_truncated_incomplete_text(response.finish, score, &text)?;
    Ok(ProbeResult {
        name: "unified_diff".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// `fn greet(` / `fn welcome(` must sit outside a string, at an
/// identifier boundary. `fn greet<'a>(` counts too. The rest must look
/// like a Rust signature, not English (`should say`, `says`, `must print`).
/// A signature may share the line with an earlier statement
/// (`let label = "..."; fn greet(`).
///
/// Literals are masked first. `fn greet(` inside a label string is not
/// the edit. The greeting check still reads the unmasked text, because
/// `Hello` lives in `format!("Hello")`.
fn has_code_fn_token(text: &str, token: &str) -> bool {
    // Mask the whole block so a raw string or triple quote that spans
    // lines hides `fn greet(` on a later line. Per-line masking would
    // see that continuation as code.
    mask_literals(text)
        .lines()
        .any(|line| line_has_code_fn_token(line, token))
}

fn line_has_code_fn_token(line: &str, token: &str) -> bool {
    let Some(name) = token.strip_suffix('(') else {
        return false;
    };
    let bytes = line.as_bytes();
    let mut search = 0;
    while let Some(rel) = line.get(search..).and_then(|rest| rest.find(name)) {
        let idx = search + rel;
        let boundary = idx == 0 || !is_ident_byte(bytes[idx - 1]);
        if boundary
            && let Some(after_paren) = skip_generics_to_paren(&line[idx + name.len()..])
            && looks_like_rust_fn_signature_rest(after_paren)
        {
            return true;
        }
        search = idx + name.len();
    }
    false
}

/// `fn greet(` or `fn greet<'a>(`. Returns the text after the parameter `(`.
fn skip_generics_to_paren(after_name: &str) -> Option<&str> {
    let s = after_name.trim_start();
    let bytes = s.as_bytes();
    if bytes.first() == Some(&b'(') {
        return Some(&s[1..]);
    }
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut depth = 0i32;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    let rest = s[i + 1..].trim_start();
                    return rest.strip_prefix('(');
                }
            }
            _ => {}
        }
    }
    None
}

/// After `fn greet(`, the rest must close the params and then be a type,
/// `;`, or a `{ ... }` body. A body on the same line counts when it
/// looks like code (`format!(...)`, `return name`). English
/// (`{ should say Hello }`) does not.
fn looks_like_rust_fn_signature_rest(after_open_paren: &str) -> bool {
    let bytes = after_open_paren.as_bytes();
    let mut depth = 1i32;
    let mut close = None;
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = literal_end(after_open_paren, i) {
            i = end;
            continue;
        }
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let Some(close) = close else {
        return false;
    };
    let after = after_open_paren[close + 1..].trim();
    if after.is_empty() || after == "{" || after == ";" {
        return true;
    }
    if let Some(body) = after.strip_prefix('{') {
        return brace_body_is_code(body);
    }
    let Some(ty) = after.strip_prefix("->") else {
        return false;
    };
    let (ty, body) = split_type_and_brace_body(ty.trim());
    if !is_rust_type_tokens(ty.trim().trim_end_matches(';').trim()) {
        return false;
    }
    match body {
        Some(body) => brace_body_is_code(body),
        None => true,
    }
}

fn split_type_and_brace_body(after_arrow: &str) -> (&str, Option<&str>) {
    let bytes = after_arrow.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = literal_end(after_arrow, i) {
            i = end;
            continue;
        }
        if bytes[i] == b'{' {
            let body = after_arrow[i + 1..].trim().trim_end_matches('}').trim();
            return (after_arrow[..i].trim(), Some(body));
        }
        let ch = after_arrow[i..].chars().next().expect("char");
        i += ch.len_utf8();
    }
    (after_arrow.trim().trim_end_matches(';').trim(), None)
}

/// Empty braces stay code so a signature can end in `{` and put the
/// body on the next line. A same-line body needs a call, an assignment,
/// or a semicolon. `{ Hello }`, `{ doesn't print Hello }`, and
/// `{ let me say Hello }` are lecture.
fn brace_body_is_code(body: &str) -> bool {
    let t = body.trim().trim_end_matches('}').trim();
    if t.is_empty() {
        return true;
    }
    t.contains(['(', ';', '='])
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
                    | "says"
                    | "said"
                    | "saying"
                    | "will"
                    | "must"
                    | "print"
                    | "prints"
                    | "does"
                    | "doesn"
                    | "doesnt"
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

/// Drop comments, but keep the text of string and character literals.
/// The greeting lives in a string, so blanking literals would hide a
/// real edit. A `#`, `//`, or `/*` inside a literal is not a comment.
/// `#[` and `#!` stay.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut line_start = 0;
    while i < bytes.len() {
        if let Some(end) = literal_end(text, i) {
            let literal = &text[i..end];
            out.push_str(literal);
            if let Some(rel) = literal.rfind('\n') {
                line_start = out.len() - (literal.len() - rel - 1);
            }
            i = end;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            drop_rest_of_line(bytes, &mut i, &mut out, &mut line_start);
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            } else {
                i = bytes.len();
            }
            continue;
        }
        if bytes[i] == b'#' {
            let rest = &text[i..];
            if rest.starts_with("#[") || rest.starts_with("#!") {
                out.push('#');
                i += 1;
                continue;
            }
            drop_rest_of_line(bytes, &mut i, &mut out, &mut line_start);
            continue;
        }
        let ch = text[i..].chars().next().expect("char");
        out.push(ch);
        i += ch.len_utf8();
        if ch == '\n' {
            line_start = out.len();
        }
    }
    out
}

fn drop_rest_of_line(bytes: &[u8], i: &mut usize, out: &mut String, line_start: &mut usize) {
    while *i < bytes.len() && bytes[*i] != b'\n' {
        *i += 1;
    }
    if out[*line_start..].chars().all(|ch| ch == ' ' || ch == '\t') {
        out.truncate(*line_start);
    }
    if *i < bytes.len() && bytes[*i] == b'\n' {
        if out.len() == *line_start {
            *i += 1;
        } else {
            out.push('\n');
            *i += 1;
            *line_start = out.len();
        }
    }
}

fn mask_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = literal_end(text, i) {
            out.extend(text[i..end].chars().map(|_| ' '));
            i = end;
            continue;
        }
        let ch = text[i..].chars().next().expect("char");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// End index of a Rust or Python literal that starts at `i`.
/// `None` when `i` is not a literal. An unclosed raw string runs
/// through the end of `text` so its `#` delimiters are not comments.
fn literal_end(text: &str, i: usize) -> Option<usize> {
    if let Some(end) = raw_string_end(text, i) {
        return Some(end);
    }
    if let Some(end) = triple_quote_end(text, i) {
        return Some(end);
    }
    if let Some(end) = prefixed_byte_or_c_end(text, i) {
        return Some(end);
    }
    let bytes = text.as_bytes();
    if bytes.get(i) == Some(&b'"') {
        return Some(escaped_string_end(text, i));
    }
    if bytes.get(i) == Some(&b'\'') {
        if let Some(end) = char_literal_end(text, i) {
            return Some(end);
        }
        if let Some(end) = single_quote_with_comment_mark_end(text, i) {
            return Some(end);
        }
    }
    None
}

fn raw_string_end(text: &str, i: usize) -> Option<usize> {
    if i > 0 && is_ident_byte(text.as_bytes()[i - 1]) {
        return None;
    }
    let rest = text.get(i..)?;
    let prefix = if rest.starts_with("br") || rest.starts_with("cr") {
        2
    } else if rest.starts_with('r') {
        1
    } else {
        return None;
    };
    let bytes = text.as_bytes();
    let mut j = i + prefix;
    let mut hashes = 0usize;
    while j < bytes.len() && bytes[j] == b'#' {
        hashes += 1;
        j += 1;
    }
    if j >= bytes.len() || bytes[j] != b'"' {
        return None;
    }
    j += 1;
    Some(find_raw_closer(text, j, hashes).unwrap_or(text.len()))
}

fn find_raw_closer(text: &str, from: usize, hashes: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut j = from;
    while j < bytes.len() {
        if bytes[j] == b'"' {
            let mut k = 0;
            while k < hashes && j + 1 + k < bytes.len() && bytes[j + 1 + k] == b'#' {
                k += 1;
            }
            if k == hashes {
                return Some(j + 1 + hashes);
            }
        }
        j += 1;
    }
    None
}

fn triple_quote_end(text: &str, i: usize) -> Option<usize> {
    let rest = text.get(i..)?;
    let quote = if rest.starts_with("\"\"\"") {
        "\"\"\""
    } else if rest.starts_with("'''") {
        "'''"
    } else {
        return None;
    };
    let search_from = i + quote.len();
    let rel = text.get(search_from..)?.find(quote)?;
    Some(search_from + rel + quote.len())
}

fn prefixed_byte_or_c_end(text: &str, i: usize) -> Option<usize> {
    if i > 0 && is_ident_byte(text.as_bytes()[i - 1]) {
        return None;
    }
    let rest = text.get(i..)?;
    if rest.starts_with("b\"") || rest.starts_with("c\"") {
        Some(escaped_string_end(text, i + 1))
    } else if rest.starts_with("b'") || rest.starts_with("c'") {
        char_literal_end(text, i + 1)
    } else {
        None
    }
}

fn escaped_string_end(text: &str, quote_at: usize) -> usize {
    let bytes = text.as_bytes();
    let mut j = quote_at + 1;
    while j < bytes.len() {
        let b = bytes[j];
        if b == b'\\' {
            j += 1;
            if j < bytes.len() {
                j += 1;
            }
            continue;
        }
        if b == b'"' {
            return j + 1;
        }
        // A Rust string cannot cross a raw newline. Stopping here keeps
        // a real function on the next line when the closer was dropped.
        if b == b'\n' {
            return j;
        }
        j += 1;
    }
    text.len()
}

fn char_literal_end(text: &str, quote_at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(quote_at) != Some(&b'\'') {
        return None;
    }
    let mut j = quote_at + 1;
    if j >= bytes.len() || bytes[j] == b'\n' || bytes[j] == b'\'' {
        return None;
    }
    if bytes[j] == b'\\' {
        j += 1;
        if j >= bytes.len() || bytes[j] == b'\n' {
            return None;
        }
        if bytes[j] == b'x' {
            j += 1;
            if j + 1 >= bytes.len()
                || !bytes[j].is_ascii_hexdigit()
                || !bytes[j + 1].is_ascii_hexdigit()
            {
                return None;
            }
            j += 2;
        } else if bytes[j] == b'u' {
            j += 1;
            if j >= bytes.len() || bytes[j] != b'{' {
                return None;
            }
            j += 1;
            let start = j;
            while j < bytes.len() && bytes[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j == start || j >= bytes.len() || bytes[j] != b'}' {
                return None;
            }
            j += 1;
        } else {
            j += 1;
        }
    } else {
        let ch = text.get(j..)?.chars().next()?;
        j += ch.len_utf8();
    }
    if bytes.get(j) == Some(&b'\'') {
        Some(j + 1)
    } else {
        None
    }
}

/// Python `'a # b'` hides comment marks. A Rust lifetime (`'a`, `'static`)
/// has no closer, or the span to the next `'` has no comment mark, so
/// it stays code.
fn single_quote_with_comment_mark_end(text: &str, quote_at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(quote_at) != Some(&b'\'') {
        return None;
    }
    let mut j = quote_at + 1;
    let mut saw_mark = false;
    while j < bytes.len() && bytes[j] != b'\n' {
        if bytes[j] == b'\\' {
            j += 1;
            if j < bytes.len() && bytes[j] != b'\n' {
                j += 1;
            }
            continue;
        }
        if bytes[j] == b'\'' {
            return if saw_mark { Some(j + 1) } else { None };
        }
        if bytes[j] == b'#' {
            saw_mark = true;
        }
        if bytes[j] == b'/' && j + 1 < bytes.len() && (bytes[j + 1] == b'/' || bytes[j + 1] == b'*')
        {
            saw_mark = true;
        }
        j += 1;
    }
    None
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
        if let Some((maybe_dash, after_dash)) = rest.split_once('\n')
            && is_dash_separator(maybe_dash)
        {
            return (first.trim(), after_dash);
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
    async fn search_replace_mark_inside_string_stays_strong() {
        for (old_line, new_line) in [
            (
                "let label = \"a # b\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"c # d\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = \"http://old\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"http://new\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = \"a /* b\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"c /* d\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = r#\"a # b\"#; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = r#\"c # d\"#; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = '#'; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = '#'; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = 'a # b'; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = 'c # d'; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = \"\"\"a # b\"\"\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"\"\"c # d\"\"\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "pub(crate) fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "pub(crate) fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "async fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "async fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = br#\"a # b\"#; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = br#\"c # d\"#; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = b'#'; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = b'#'; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
        ] {
            let response_text = format!(
                "\
<<<<<<< SEARCH
src/greet.rs
-------
{old_line}
=======
{new_line}
>>>>>>> REPLACE"
            );
            let result = probe_search_replace(&MockLlm {
                response: text_response(&response_text),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a mark inside a string must not hide the edit: {old_line:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{old_line:?}");
        }
    }

    #[tokio::test]
    async fn search_replace_comment_after_string_is_not_strong() {
        for (old_line, new_line) in [
            (
                "let label = \"ok\"; // fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"ok\"; // fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = \"ok\"; # fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"ok\"; # fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
        ] {
            let response_text = format!(
                "\
<<<<<<< SEARCH
src/greet.rs
-------
{old_line}
=======
{new_line}
>>>>>>> REPLACE"
            );
            let result = probe_search_replace(&MockLlm {
                response: text_response(&response_text),
            })
            .await
            .unwrap();
            assert_ne!(
                result.score, 1.0,
                "a comment after a string must not count as the edit: {old_line:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{old_line:?}");
        }
    }

    #[tokio::test]
    async fn search_replace_braced_lecture_is_not_strong() {
        for (old_line, new_line) in [
            (
                "fn greet(name: &str) -> String { should say Hello }",
                "fn welcome(name: &str) -> String { should say Welcome }",
            ),
            (
                "fn greet(name: &str) -> String { Hello }",
                "fn welcome(name: &str) -> String { Welcome }",
            ),
            (
                "fn greet(name: &str) -> String { doesn't print Hello }",
                "fn welcome(name: &str) -> String { doesn't print Welcome }",
            ),
            (
                "fn greet(name: &str) -> String { should say Hello. }",
                "fn welcome(name: &str) -> String { should say Welcome. }",
            ),
            (
                "fn greet(name: &str) -> String { let me say Hello }",
                "fn welcome(name: &str) -> String { let me say Welcome }",
            ),
            (
                "fn greet(name: &str) -> String { for example Hello }",
                "fn welcome(name: &str) -> String { for example Welcome }",
            ),
        ] {
            let response_text = format!(
                "\
<<<<<<< SEARCH
src/greet.rs
{old_line}
=======
{new_line}
>>>>>>> REPLACE"
            );
            let result = probe_search_replace(&MockLlm {
                response: text_response(&response_text),
            })
            .await
            .unwrap();
            assert_ne!(
                result.score, 1.0,
                "a braced lecture must not be Strong: {old_line:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{old_line:?}");
        }
    }

    #[tokio::test]
    async fn search_replace_real_fn_after_raw_string_stays_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
let s = r#\"
note
\"#;
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
=======
let s = r#\"
note
\"#;
fn welcome(name: &str) -> String {
    format!(\"Welcome, {}\", name)
}
>>>>>>> REPLACE";
        let result = probe_search_replace(&MockLlm {
            response: text_response(response_text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a function after a closed raw string must stay Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_fn_after_unclosed_quote_stays_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
let s = \"unclosed
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
=======
let s = \"unclosed
fn welcome(name: &str) -> String {
    format!(\"Welcome, {}\", name)
}
>>>>>>> REPLACE";
        let result = probe_search_replace(&MockLlm {
            response: text_response(response_text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a dropped string closer must not hide the next line: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn search_replace_fn_inside_spanning_string_is_not_strong() {
        for (old_body, new_body) in [
            (
                "let s = r#\"\nfn greet(name: &str) -> String { format!(\"Hello, {}\", name) }\n\"#;",
                "let s = r#\"\nfn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }\n\"#;",
            ),
            (
                "let s = \"\"\"\nfn greet(name: &str) -> String { format!(\"Hello, {}\", name) }\n\"\"\";",
                "let s = \"\"\"\nfn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }\n\"\"\";",
            ),
        ] {
            let response_text = format!(
                "\
<<<<<<< SEARCH
src/greet.rs
-------
{old_body}
=======
{new_body}
>>>>>>> REPLACE"
            );
            let result = probe_search_replace(&MockLlm {
                response: text_response(&response_text),
            })
            .await
            .unwrap();
            assert_ne!(
                result.score, 1.0,
                "fn inside a spanning string must not be Strong: {old_body:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{old_body:?}");
        }
    }

    #[tokio::test]
    async fn search_replace_lifetime_signature_stays_strong() {
        let response_text = "\
<<<<<<< SEARCH
src/greet.rs
-------
fn greet<'a>(name: &'a str) -> String {
    format!(\"Hello, {}\", name)
}
=======
fn welcome<'a>(name: &'a str) -> String {
    format!(\"Welcome, {}\", name)
}
>>>>>>> REPLACE";
        let result = probe_search_replace(&MockLlm {
            response: text_response(response_text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "lifetimes must not be read as strings: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn search_replace_lecture_says_is_not_strong() {
        for (old_line, new_line) in [
            (
                "fn greet(name: &str) -> String says Hello",
                "fn welcome(name: &str) -> String says Welcome",
            ),
            (
                "fn greet(name: &str) -> String said Hello",
                "fn welcome(name: &str) -> String said Welcome",
            ),
            (
                "fn greet(name: &str) -> String saying Hello",
                "fn welcome(name: &str) -> String saying Welcome",
            ),
            (
                "fn greet(name: &str) -> String will print Hello",
                "fn welcome(name: &str) -> String will print Welcome",
            ),
            (
                "fn greet(name: &str) -> String prints Hello",
                "fn welcome(name: &str) -> String prints Welcome",
            ),
            (
                "fn greet(name: &str) -> String doesn't print Hello",
                "fn welcome(name: &str) -> String doesn't print Welcome",
            ),
        ] {
            let response_text = format!(
                "\
<<<<<<< SEARCH
src/greet.rs
{old_line}
=======
{new_line}
>>>>>>> REPLACE"
            );
            let llm = MockLlm {
                response: text_response(&response_text),
            };
            let result = probe_search_replace(&llm).await.unwrap();
            assert_ne!(
                result.score, 1.0,
                "lecture verbs after -> Type must not be SearchReplace Strong: {old_line:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{old_line:?}");
        }
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

    #[test]
    fn unified_diff_system_has_no_echoable_format_card() {
        let prompt = UNIFIED_DIFF_SYSTEM;
        for token in [
            "removed line",
            "added line",
            "path/to/file.rs",
            "```diff",
            "--- a/",
            "+++ b/",
            "@@ -",
        ] {
            assert!(
                !prompt.contains(token),
                "unified-diff system must not include copyable card token {token:?}: {prompt}"
            );
        }
        assert!(
            !is_unified_diff_card_echo(prompt),
            "the system prompt itself must not match the leftover card-echo grader: {prompt}"
        );
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
    async fn unified_diff_mark_inside_string_stays_strong() {
        for (old_line, new_line) in [
            (
                "let label = \"a # b\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"c # d\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
            (
                "let label = \"http://old\"; fn greet(name: &str) -> String { format!(\"Hello, {}\", name) }",
                "let label = \"http://new\"; fn welcome(name: &str) -> String { format!(\"Welcome, {}\", name) }",
            ),
        ] {
            let response_text = format!(
                "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-{old_line}
+{new_line}
"
            );
            let result = probe_unified_diff(&MockLlm {
                response: text_response(&response_text),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a mark inside a string on a diff line must stay Strong: {old_line:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{old_line:?}");
        }
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
    async fn unified_diff_lecture_must_print_is_not_strong() {
        let response_text = "\
--- a/src/greet.rs
+++ b/src/greet.rs
@@ -1,1 +1,1 @@
-fn greet(name: &str) -> String must print Hello
+fn welcome(name: &str) -> String must print Welcome
";
        let llm = MockLlm {
            response: text_response(response_text),
        };
        let result = probe_unified_diff(&llm).await.unwrap();
        assert_ne!(
            result.score, 1.0,
            "must print after -> Type must not be UnifiedDiff Strong: {result:?}"
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
