//! Code syntax accuracy probe.
//!
//! Tests whether the model produces syntactically valid code when asked
//! to write a small function. Models that fail this should have higher
//! lint-fix retry counts.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete_text, user_text};

/// Probe whether the model produces syntactically valid code.
///
/// Asks the model to write a small function and checks for common
/// syntax issues: balanced braces/parens, no trailing commas in
/// invalid positions, no incomplete statements.
///
/// Scoring:
/// - `1.0` - code has balanced delimiters, no obvious syntax errors,
///   and contains the expected function signature. An ellipsis in a
///   comment, a string, a slice, a type subscript, or a parameter
///   default does not count. A value ellipsis still does, including
///   `return ...`, `return (...)`, `name = ...`, an f-string
///   expression, and a call argument.
/// - `0.5` - code present and mostly correct but has minor issues
///   (unbalanced delimiters, a real ellipsis, or missing return)
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
    let code = fenced.unwrap_or(text);
    let trimmed = code.trim();

    if trimmed.is_empty() {
        refuse_truncated_incomplete_text(response.finish, 0.0, &response.text)?;
        return Ok(ProbeResult {
            name: "code_syntax".to_string(),
            score: 0.0,
            max_score: 1.0,
            level: classify(0.0),
            details: "Empty response, no code produced".to_string(),
        });
    }

    let commented = strip_hash_comments(trimmed);
    let code_body = strip_python_string_literals(&commented);
    // Fenced or unfenced prose can name `def merge_sorted(` and `return`.
    // Strong needs a colon plus a real body, not a parenthesized name-drop.
    let has_def = has_indented_merge_sorted_body(trimmed);
    let has_return = merge_sorted_body_has_return(&code_body);

    let parens_balanced = count_char(&code_body, '(') == count_char(&code_body, ')');
    let brackets_balanced = count_char(&code_body, '[') == count_char(&code_body, ']');
    let braces_balanced = count_char(&code_body, '{') == count_char(&code_body, '}');
    let delimiters_ok = parens_balanced && brackets_balanced && braces_balanced;

    // Comments are already gone. String data stays, so an f-string
    // expression can still be read. A value `...` is a stub. A slice,
    // a type subscript, and a parameter default are not.
    let has_ellipsis = has_stub_ellipsis(&commented);
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

    refuse_truncated_incomplete_text(response.finish, score, &response.text)?;
    Ok(ProbeResult {
        name: "code_syntax".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// `#` starts a comment only outside a Python string. A quote can span
/// lines, so a per-line split would treat a `#` inside that quote as a
/// comment and then swallow the rest of the function.
fn strip_hash_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(quote) = python_string_open(text, i) {
            let end = skip_python_string(text, i, quote);
            out.push_str(&text[i..end]);
            i = end;
            continue;
        }
        if bytes[i] == b'#' {
            let line_start = out.rfind('\n').map_or(0, |pos| pos + 1);
            let only_indent = out[line_start..].chars().all(|ch| ch == ' ' || ch == '\t');
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            if only_indent {
                out.truncate(line_start);
                if i < bytes.len() && bytes[i] == b'\n' {
                    i += 1;
                }
            }
            continue;
        }
        let ch = text[i..].chars().next().expect("char");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn python_string_open(text: &str, i: usize) -> Option<&'static str> {
    let rest = text.get(i..)?;
    if rest.starts_with("\"\"\"") {
        Some("\"\"\"")
    } else if rest.starts_with("'''") {
        Some("'''")
    } else if rest.starts_with('"') {
        Some("\"")
    } else if rest.starts_with('\'') {
        Some("'")
    } else {
        None
    }
}

/// Index just past the string that opens at `i`. An unclosed string
/// runs through the end of `text`.
fn skip_python_string(text: &str, i: usize, quote: &str) -> usize {
    let mut j = i + quote.len();
    let bytes = text.as_bytes();
    if quote.len() == 1 {
        let closer = quote.as_bytes()[0];
        while j < bytes.len() {
            let b = bytes[j];
            j += 1;
            if b == b'\\' {
                if j < bytes.len() {
                    j += 1;
                }
                continue;
            }
            if b == closer {
                break;
            }
        }
        return j;
    }
    match text.get(j..).and_then(|rest| rest.find(quote)) {
        Some(rel) => j + rel + quote.len(),
        None => text.len(),
    }
}

fn strip_python_string_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if let Some(quote) = python_string_open(text, i) {
            i = skip_python_string(text, i, quote);
            continue;
        }
        let ch = text[i..].chars().next().expect("char");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// A `...` used as a value. Parameter defaults, slices, and type
/// subscripts are real syntax. String text is skipped, except an
/// f-string `{expression}`.
fn has_stub_ellipsis(text: &str) -> bool {
    scan_python(text, 0, false).0
}

enum PyFrame {
    Paren,
    Bracket { subscript: bool },
    Brace,
}

/// Walk Python tokens. When `stop_expr` is set, stop at the f-string
/// expression terminator and return that index.
fn scan_python(text: &str, mut i: usize, stop_expr: bool) -> (bool, usize) {
    let bytes = text.as_bytes();
    let mut frames: Vec<PyFrame> = Vec::new();
    // `frames.len()` when `def` or `lambda` began. The matching `:` pops it.
    let mut sigs: Vec<usize> = Vec::new();
    let mut annotating = false;
    let mut prev_primary = false;
    let mut found = false;

    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            if bytes[i] == b'\n' {
                annotating = false;
            }
            i += 1;
            continue;
        }
        if bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if stop_expr
            && frames.is_empty()
            && (bytes[i] == b'}' || is_fstring_conversion(text, i) || is_debug_eq(text, i))
        {
            return (found, i);
        }
        if let Some((quote_at, quote, formatted)) = string_literal_at(text, i) {
            if formatted {
                let (stub, end) = scan_fstring(text, quote_at, quote);
                // A default or annotation is not a stub body. `return f"{...}"` is.
                if sigs.is_empty() && !annotating {
                    found |= stub;
                }
                i = end;
            } else {
                i = skip_python_string(text, quote_at, quote);
            }
            prev_primary = true;
            continue;
        }
        if is_ellipsis_at(bytes, i) {
            let mut j = i;
            while j < bytes.len() && bytes[j] == b'.' {
                j += 1;
            }
            let in_subscript = matches!(frames.last(), Some(PyFrame::Bracket { subscript: true }));
            if !in_subscript && sigs.is_empty() && !annotating {
                found = true;
            }
            i = j;
            prev_primary = true;
            continue;
        }
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let word = &text[start..i];
            if word == "def" || word == "lambda" {
                sigs.push(frames.len());
            }
            prev_primary = !is_non_primary_keyword(word);
            continue;
        }
        match bytes[i] {
            b'(' => {
                frames.push(PyFrame::Paren);
                prev_primary = false;
                i += 1;
            }
            b')' => {
                frames.pop();
                prev_primary = true;
                i += 1;
            }
            b'[' => {
                frames.push(PyFrame::Bracket {
                    subscript: prev_primary,
                });
                prev_primary = false;
                i += 1;
            }
            b']' => {
                frames.pop();
                prev_primary = true;
                i += 1;
            }
            b'{' => {
                frames.push(PyFrame::Brace);
                prev_primary = false;
                i += 1;
            }
            b'}' => {
                if stop_expr && frames.is_empty() {
                    return (found, i);
                }
                frames.pop();
                prev_primary = true;
                i += 1;
            }
            b':' => {
                if bytes.get(i + 1) == Some(&b'=') {
                    prev_primary = false;
                    i += 1;
                    continue;
                }
                if sigs.last().copied() == Some(frames.len()) {
                    sigs.pop();
                    annotating = false;
                } else if stop_expr && frames.is_empty() {
                    return (found, i);
                } else if frames.is_empty() && sigs.is_empty() {
                    annotating = true;
                }
                prev_primary = false;
                i += 1;
            }
            b'=' => {
                if !is_comparison_eq(text, i) {
                    if stop_expr && frames.is_empty() {
                        return (found, i);
                    }
                    annotating = false;
                }
                prev_primary = false;
                i += 1;
            }
            _ => {
                let ch = text[i..].chars().next().expect("char");
                prev_primary = false;
                i += ch.len_utf8();
            }
        }
    }
    (found, i)
}

fn scan_fstring(text: &str, quote_at: usize, quote: &str) -> (bool, usize) {
    let bytes = text.as_bytes();
    let mut j = quote_at + quote.len();
    let mut found = false;
    while j < bytes.len() {
        if bytes[j] == b'\\' {
            j += 1;
            if j < text.len() {
                let ch = text[j..].chars().next().expect("char");
                j += ch.len_utf8();
            }
            continue;
        }
        if text[j..].starts_with(quote) {
            return (found, j + quote.len());
        }
        if bytes[j] == b'{' && bytes.get(j + 1) == Some(&b'{') {
            j += 2;
            continue;
        }
        if bytes[j] == b'}' && bytes.get(j + 1) == Some(&b'}') {
            j += 2;
            continue;
        }
        if bytes[j] == b'{' {
            let (stub, end) = scan_python(text, j + 1, true);
            found |= stub;
            if end >= bytes.len() {
                return (found, bytes.len());
            }
            if bytes[end] == b'}' {
                j = end + 1;
                continue;
            }
            let (tail_stub, after) = skip_fstring_tail(text, end);
            found |= tail_stub;
            j = after;
            continue;
        }
        let ch = text[j..].chars().next().expect("char");
        j += ch.len_utf8();
    }
    (found, bytes.len())
}

/// `index` points at `!`, `:`, or the debug `=`. Skip through the
/// closing `}` and scan nested replacement fields.
fn skip_fstring_tail(text: &str, mut j: usize) -> (bool, usize) {
    let bytes = text.as_bytes();
    let mut found = false;
    if j < bytes.len() && bytes[j] == b'!' {
        j += 1;
        if j < bytes.len() && matches!(bytes[j], b's' | b'r' | b'a') {
            j += 1;
        }
    }
    if j < bytes.len() && bytes[j] == b'=' {
        j += 1;
    }
    while j < bytes.len() {
        if bytes[j] == b'\\' {
            j += 1;
            if j < text.len() {
                let ch = text[j..].chars().next().expect("char");
                j += ch.len_utf8();
            }
            continue;
        }
        if bytes[j] == b'{' && bytes.get(j + 1) == Some(&b'{') {
            j += 2;
            continue;
        }
        if bytes[j] == b'}' && bytes.get(j + 1) == Some(&b'}') {
            j += 2;
            continue;
        }
        if bytes[j] == b'}' {
            return (found, j + 1);
        }
        if bytes[j] == b'{' {
            let (stub, end) = scan_python(text, j + 1, true);
            found |= stub;
            if end >= bytes.len() {
                return (found, bytes.len());
            }
            if bytes[end] == b'}' {
                j = end + 1;
                continue;
            }
            let (tail_stub, after) = skip_fstring_tail(text, end);
            found |= tail_stub;
            j = after;
            continue;
        }
        let ch = text[j..].chars().next().expect("char");
        j += ch.len_utf8();
    }
    (found, bytes.len())
}

/// `(quote_index, quote, formatted)`.
///
/// A quote with no prefix is still a string when it follows a name
/// (`return"..."`). A prefix is only a prefix when it is not the tail
/// of a longer name (`myf"..."` is a name, then a string).
fn string_literal_at(text: &str, i: usize) -> Option<(usize, &'static str, bool)> {
    let bytes = text.as_bytes();
    if let Some(quote) = python_string_open(text, i) {
        return Some((i, quote, false));
    }
    if i > 0 && is_ident_continue(bytes[i - 1]) {
        return None;
    }
    let mut k = i;
    while k < bytes.len() && k - i < 3 && is_string_prefix(bytes[k]) {
        k += 1;
    }
    if k == i {
        return None;
    }
    let quote = python_string_open(text, k)?;
    let formatted = text[i..k]
        .bytes()
        .any(|b| matches!(b, b'f' | b'F' | b't' | b'T'));
    Some((k, quote, formatted))
}

fn is_string_prefix(b: u8) -> bool {
    matches!(
        b,
        b'r' | b'R' | b'u' | b'U' | b'b' | b'B' | b'f' | b'F' | b't' | b'T'
    )
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_ellipsis_at(bytes: &[u8], i: usize) -> bool {
    bytes.get(i) == Some(&b'.')
        && bytes.get(i + 1) == Some(&b'.')
        && bytes.get(i + 2) == Some(&b'.')
}

fn is_fstring_conversion(text: &str, i: usize) -> bool {
    let bytes = text.as_bytes();
    bytes.get(i) == Some(&b'!')
        && matches!(bytes.get(i + 1), Some(b's' | b'r' | b'a'))
        && !bytes.get(i + 2).is_some_and(|b| is_ident_continue(*b))
}

fn is_debug_eq(text: &str, i: usize) -> bool {
    let bytes = text.as_bytes();
    bytes.get(i) == Some(&b'=') && !is_comparison_eq(text, i)
}

fn is_comparison_eq(text: &str, i: usize) -> bool {
    let bytes = text.as_bytes();
    if bytes.get(i) != Some(&b'=') {
        return false;
    }
    if bytes.get(i + 1) == Some(&b'=') {
        return true;
    }
    matches!(
        i.checked_sub(1).and_then(|prev| bytes.get(prev)),
        Some(b'!' | b'<' | b'>' | b':' | b'=')
    )
}

fn is_non_primary_keyword(word: &str) -> bool {
    matches!(
        word,
        "return"
            | "assert"
            | "if"
            | "elif"
            | "else"
            | "for"
            | "while"
            | "yield"
            | "raise"
            | "del"
            | "pass"
            | "break"
            | "continue"
            | "and"
            | "or"
            | "not"
            | "in"
            | "is"
            | "as"
            | "with"
            | "from"
            | "import"
            | "class"
            | "try"
            | "except"
            | "finally"
            | "match"
            | "case"
            | "global"
            | "nonlocal"
            | "await"
            | "def"
            | "lambda"
    )
}

fn has_indented_merge_sorted_body(text: &str) -> bool {
    let needle = "def merge_sorted";
    let mut search = 0;
    while let Some(rel) = text.get(search..).and_then(|s| s.find(needle)) {
        let idx = search + rel;
        let after = &text[idx + needle.len()..];
        // Type-hint colons sit inside the parameter list. The def colon
        // is the first `:` after balanced parens (and optional `-> type`),
        // including when params wrap across lines.
        if let Some(colon_abs) = signature_colon_offset(after)
            && looks_like_parameter_list(&after[..colon_abs])
        {
            let rest = &after[colon_abs + 1..];
            if rest.lines().next().is_some_and(looks_like_same_line_body) {
                return true;
            }
            for line in rest.lines().skip(1) {
                if line.trim().is_empty() {
                    continue;
                }
                if !(line.starts_with(' ') || line.starts_with('\t')) {
                    return false;
                }
                if looks_like_same_line_body(line) {
                    return true;
                }
                // English return phrases, lecture lines, comments, and
                // docstrings are not a body. Keep looking for a statement.
            }
        }
        search = idx + needle.len();
    }
    false
}

/// Offset of the def colon after `def merge_sorted`.
/// Strings and `{...}` defaults do not supply that colon. A no-paren
/// signature must put its colon on the first line.
fn signature_colon_offset(after: &str) -> Option<usize> {
    let bytes = after.as_bytes();
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    let mut seen_paren = false;
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(after, i) {
            i = end;
            continue;
        }
        let Some(c) = after.get(i..).and_then(|rest| rest.chars().next()) else {
            break;
        };
        match c {
            '(' => {
                paren += 1;
                seen_paren = true;
            }
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            ':' if paren <= 0 && brack <= 0 && brace <= 0 => return Some(i),
            '\n' if !seen_paren && paren <= 0 && brace <= 0 => return None,
            _ => {}
        }
        i += c.len_utf8();
    }
    None
}

fn looks_like_parameter_list(between: &str) -> bool {
    let stripped = strip_hash_comments(between);
    let s = strip_return_annotation(stripped.trim());
    if s.starts_with('(') && s.ends_with(')') {
        // A default like `"("` must not look unbalanced.
        let visible = strip_python_string_literals(s);
        if count_char(&visible, '(') != count_char(&visible, ')') {
            return false;
        }
        let inner = s[1..s.len() - 1].trim();
        if inner.is_empty() {
            return true;
        }
        return split_top_level_commas(inner)
            .into_iter()
            .filter(|p| !p.is_empty())
            .all(is_parameter);
    }
    let parts: Vec<&str> = split_top_level_commas(s)
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect();
    parts.len() >= 2 && parts.iter().all(|p| is_parameter(p))
}

fn strip_return_annotation(s: &str) -> &str {
    let bytes = s.as_bytes();
    if !bytes.starts_with(b"(") {
        return s;
    }
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in s.char_indices() {
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
        return s;
    };
    let after = s[close + 1..].trim_start();
    if after.starts_with("->") {
        return s[..=close].trim();
    }
    s
}

fn skip_string_at(text: &str, i: usize) -> Option<usize> {
    let (quote_at, quote, _) = string_literal_at(text, i)?;
    let end = skip_python_string(text, quote_at, quote);
    (end > i).then_some(end)
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(s, i) {
            i = end;
            continue;
        }
        let Some(c) = s.get(i..).and_then(|rest| rest.chars().next()) else {
            break;
        };
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            ',' if paren <= 0 && brack <= 0 && brace <= 0 => {
                out.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
        i += c.len_utf8();
    }
    out.push(s[start..].trim());
    out
}

fn is_parameter(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    // Positional-only `/` and keyword-only `*` are slots, not names.
    if s == "/" || s == "*" {
        return true;
    }
    let name = match s.split_once('=') {
        Some((head, _)) => match head.split_once(':') {
            Some((name, _)) => name.trim(),
            None => head.trim(),
        },
        None => match s.split_once(':') {
            Some((name, _)) => name.trim(),
            None => s,
        },
    };
    let name = name
        .strip_prefix("**")
        .or_else(|| name.strip_prefix('*'))
        .unwrap_or(name);
    is_simple_ident(name)
}

fn looks_like_same_line_body(line: &str) -> bool {
    let t = line.trim_start();
    if t.is_empty() {
        return false;
    }
    if let Some(after) = t.strip_prefix("return") {
        // `returning` is an identifier. The keyword needs a boundary.
        let bounded = after.is_empty()
            || after.starts_with(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        if bounded {
            // `return`, `return(`, `return[`, and `return"..."` stay real.
            // Word operators (`and`, `or`, `not`, `lambda`, ...) and code
            // punctuation (`|`, `&`, `:`, `.`, `@`) are a body. Two or more
            // bare English words are not. A final period is not an operator.
            // An f-string still counts when the text has spaces (`f"café ..."`).
            if after.is_empty()
                || after.starts_with(['(', '['])
                || string_literal_at(after, 0).is_some()
            {
                return true;
            }
            if !after.starts_with(|c: char| c.is_whitespace()) {
                return false;
            }
            let rest = after.trim_start();
            if rest.is_empty()
                || rest.starts_with(['(', '['])
                || string_literal_at(rest, 0).is_some()
            {
                return true;
            }
            return return_rest_is_code(rest);
        }
    }
    if t == "pass"
        || t.starts_with("pass ")
        || t.starts_with("pass\t")
        || t.starts_with("pass#")
        || t.starts_with("pass;")
    {
        return true;
    }
    if t.starts_with('[') {
        return true;
    }
    let ident_end = t
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(t.len());
    if ident_end == 0 || !is_simple_ident(&t[..ident_end]) {
        return false;
    }
    let after = t[ident_end..].trim_start();
    after.starts_with('=') || after.starts_with('(')
}

fn return_rest_is_code(rest: &str) -> bool {
    let trimmed = rest.trim_end();
    let sentence = trimmed.strip_suffix('.').unwrap_or(trimmed);
    if sentence.contains([
        '+', '-', '*', '/', '%', ',', '(', '[', '<', '>', '=', '!', ':', '|', '&', '^', '~', '.',
        '@',
    ]) {
        return true;
    }
    if sentence.split_whitespace().any(is_return_operator_word) {
        return true;
    }
    sentence.split_whitespace().nth(1).is_none()
}

fn is_return_operator_word(word: &str) -> bool {
    matches!(
        word,
        "and" | "or" | "not" | "in" | "is" | "if" | "else" | "elif" | "lambda" | "await" | "yield"
    )
}

fn is_simple_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_real_merge_sorted(body: &str) -> bool {
    has_indented_merge_sorted_body(body)
}

fn fence_has_merge_sorted_return(body: &str) -> bool {
    let code_body = strip_python_string_literals(&strip_hash_comments(body));
    merge_sorted_body_has_return(&code_body)
}

fn merge_sorted_body_has_return(text: &str) -> bool {
    let needle = "def merge_sorted";
    let mut search = 0;
    while let Some(rel) = text.get(search..).and_then(|s| s.find(needle)) {
        let idx = search + rel;
        let after = &text[idx + needle.len()..];
        if let Some(colon_abs) = signature_colon_offset(after)
            && looks_like_parameter_list(&after[..colon_abs])
        {
            let rest = &after[colon_abs + 1..];
            if rest.lines().next().is_some_and(line_has_return_token) {
                return true;
            }
            for line in rest.lines().skip(1) {
                if line.trim().is_empty() {
                    continue;
                }
                // Next top-level def or any column-0 line ends the body.
                if !(line.starts_with(' ') || line.starts_with('\t')) {
                    break;
                }
                if line_has_return_token(line) {
                    return true;
                }
            }
        }
        search = idx + needle.len();
    }
    false
}

fn line_has_return_token(line: &str) -> bool {
    let t = line.trim_start();
    looks_like_same_line_body(line)
        && (t == "return"
            || t.starts_with("return ")
            || t.starts_with("return\t")
            || t.starts_with("return(")
            || t.starts_with("return\"")
            || t.starts_with("return'"))
}

fn extract_code_block(text: &str) -> Option<&str> {
    let mut best: Option<(&str, i32)> = None;
    let mut search = 0;
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
        if is_real_merge_sorted(body) {
            let rank = if fence_has_merge_sorted_return(body) {
                1
            } else {
                0
            };
            if best.as_ref().is_none_or(|(_, r)| rank > *r) {
                best = Some((body, rank));
            }
        }
        search = code_start + end_rel + 3;
    }
    best.map(|(body, _)| body)
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
    async fn code_syntax_comment_paren_stays_strong() {
        let code = "def merge_sorted(a, b):\n    # compare (a[i]\n    return a + b\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "a comment parenthesis must not drop a real function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_hash_inside_string_keeps_real_return_strong() {
        for code in [
            "def merge_sorted(a, b):\n    note = \"return a merged list # note\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    \"\"\"Merge # two lists.\"\"\"\n    return a + b\n",
        ] {
            let llm = MockLlm {
                response: text_response(code),
            };
            let result = probe_code_syntax(&llm).await.unwrap();
            assert_eq!(
                result.score, 1.0,
                "a hash inside a string must not hide the real return: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong);
        }
    }

    #[tokio::test]
    async fn code_syntax_hash_inside_string_does_not_hide_unmatched_paren() {
        let code = "def merge_sorted(a, b):\n    return a + b\n    s = \"fine # \" + (\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert!(
            result.score < 1.0,
            "an unmatched parenthesis after a hash inside a string must stay visible: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_unbalanced_call_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    return (a + b\n";
        let llm = MockLlm {
            response: text_response(code),
        };
        let result = probe_code_syntax(&llm).await.unwrap();
        assert!(
            result.score < 1.0,
            "a real unmatched parenthesis must stay visible after the comment strip: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
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
    async fn code_syntax_prefers_complete_fence_over_stub() {
        let text = "\
```python
def merge_sorted(a, b):
    result = []
    i = j = 0
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "complete merge_sorted fence must win over an earlier stub: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn code_syntax_helper_return_is_not_strong() {
        let code = "\
def merge_sorted(a, b):
    result = []
    i = j = 0

return merge_sorted([1, 3], [2, 4])
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_ne!(
            result.score, 1.0,
            "return outside merge_sorted body must not be Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
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
    async fn code_syntax_fenced_sentence_naming_def_is_not_strong() {
        let text = "\
```
I would write a function def merge_sorted that takes two lists \
and return a single sorted list.
```";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "fenced sentence that only names def merge_sorted and return must not be Strong: {result:?}"
        );
        assert_eq!(
            result.score, 0.0,
            "name-only fenced sentence is not a function: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_fenced_sentence_with_english_colon_is_not_strong() {
        let text = "\
```
I would write def merge_sorted that takes two lists: a and b and return a list
```";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "fenced English colon after def merge_sorted must not be Strong: {result:?}"
        );
        assert_eq!(
            result.score, 0.0,
            "English colon name-drop is not a function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_prefers_def_paren_fence_after_english_colon_mention() {
        let text = "\
```
I would write def merge_sorted that takes two lists: a and b and return a list
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let code = extract_code_block(text).unwrap();
        assert!(
            code.contains("def merge_sorted("),
            "must prefer the fence with def merge_sorted(: {code:?}"
        );
        assert!(
            !code.contains("I would write"),
            "must not pick the English-colon name-drop fence: {code:?}"
        );
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "later def merge_sorted( fence must win over an English-colon mention: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_prefers_def_paren_fence_after_name_mention() {
        let text = "\
```
I would write def merge_sorted and return a list
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "later def merge_sorted( fence must win over a name-only mention: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn code_syntax_sentence_with_paren_signature_is_not_strong() {
        for text in [
            "I would write def merge_sorted(a, b) and return a single sorted list.",
            "```\nI would write def merge_sorted(a, b) and return a single sorted list.\n```",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(text),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.0,
                "sentence with def merge_sorted( plus return must not be Strong: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{text:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_english_paren_contents_not_strong() {
        let code = "def merge_sorted (two lists): and return a merged list";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "English inside parens is not a parameter list: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_english_same_line_body_not_strong() {
        let code = "def merge_sorted a, b: you should return one list";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "English after a no-paren signature colon is not a body: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_lecture_line_with_return_is_not_strong() {
        let code = "\
def merge_sorted(a, b):
    Use two pointers then return a + b
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "lecture line that only mentions return must be Weak: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_english_return_phrase_is_not_strong() {
        for text in [
            "def merge_sorted(a, b): return a single sorted list",
            "```\ndef merge_sorted(a, b): return a single sorted list\n```",
            "def merge_sorted(a, b):\n    return a single sorted list\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(text),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.0,
                "English return phrase is not a body: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{text:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_word_operator_return_is_strong() {
        for code in [
            "def merge_sorted(a, b):\n    return a or b\n",
            "def merge_sorted(a, b):\n    return a and b\n",
            "def merge_sorted(a, b):\n    return not a\n",
            "def merge_sorted(a, b):\n    return a if b else a\n",
            "def merge_sorted(a, b):\n    return a | b\n",
            "def merge_sorted(a, b):\n    return a & b\n",
            "def merge_sorted(a, b):\n    return lambda: a\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "word-operator return must be Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_nested_default_is_strong() {
        for code in [
            "def merge_sorted(a, b=\", \"):\n    return a + b\n",
            "def merge_sorted(a, b={\"x\": 1, \"y\": 2}):\n    return a + b\n",
            "def merge_sorted(a, b) -> \"a:b\":\n    return a + b\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "comma or colon inside a default must stay Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_word_operator_value_ellipsis_is_not_strong() {
        for code in [
            "def merge_sorted(a, b):\n    return a or ...\n",
            "def merge_sorted(a, b):\n    return lambda: ...\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "value ellipsis on a word-operator return stays incomplete: {code:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_sentence_colon_return_phrase_is_not_strong() {
        let text = "I would write def merge_sorted(a, b): return a single sorted list.";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "sentence with colon plus English return phrase must not be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_unfenced_real_fn_after_name_fence_is_strong() {
        let text = "\
```
I would write def merge_sorted
```
def merge_sorted(a, b):
    return a + b
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(text),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "name-drop fence must not hide a later unfenced real function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    async fn code_syntax_typed_signature_is_strong() {
        for code in [
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    return a + b\n",
            "def merge_sorted(a: list, b: list):\n    return a + b\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "typed merge_sorted signature must be Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_wrapped_typed_signature_is_strong() {
        let code = "\
def merge_sorted(
    a: list[int],
    b: list[int],
) -> list[int]:
    return a + b
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "wrapped typed merge_sorted must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_wrapped_params_with_hash_comments_is_strong() {
        let code = "\
def merge_sorted(
    a: list[int],  # already sorted
    b: list[int],  # already sorted
) -> list[int]:
    return a + b
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "wrapped typed params with # comments must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_trailing_comma_signature_is_strong() {
        let code =
            "def merge_sorted(a: list[int], b: list[int],) -> list[int]:\n    return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "trailing-comma typed merge_sorted must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_star_args_signature_is_strong() {
        let code = "def merge_sorted(*lists):\n    return list(lists)\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "*lists merge_sorted must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_ellipsis_assignment_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    result = ...\n    return result\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_ne!(
            result.score, 1.0,
            "ellipsis assignment stub must not be Strong: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_ellipsis_inside_string_or_comment_stays_strong() {
        for code in [
            "def merge_sorted(a, b):\n    note = \"example = ...\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    note = 'example = ...'\n    return a + b\n",
            "def merge_sorted(a, b):\n    note = \"example=...\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    note = r\"example = ...\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    note = f\"example = ...\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    \"\"\"\n    example = ...\n    \"\"\"\n    return a + b\n",
            "def merge_sorted(a, b):\n    # result = ...\n    return a + b\n",
            "def merge_sorted(a, b):\n    return \"...\"\n",
            "def merge_sorted(a, b):\n    return a + b  # example = ...\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "ellipsis inside a string or comment must stay Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_string_ellipsis_does_not_hide_real_stub() {
        let code = "def merge_sorted(a, b):\n    note = \"example = ...\"\n    return ...\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "a real return ... stays incomplete when a string also has an ellipsis: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_default_arg_is_strong() {
        let code = "def merge_sorted(a, b=None):\n    return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "default-arg merge_sorted must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_indented_no_paren_signature_is_strong() {
        let code = "def merge_sorted a, b:\n    return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "no-paren comma params plus indented body must stay Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
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
    fn extract_code_block_prefers_def_paren_over_name_mention() {
        let text = "\
```
I would write def merge_sorted
```

```python
def merge_sorted(a, b):
    return a + b
```
";
        let code = extract_code_block(text).unwrap();
        assert!(
            code.contains("def merge_sorted("),
            "must prefer the fence with def merge_sorted(: {code:?}"
        );
        assert!(
            !code.contains("I would write"),
            "must not pick the first name-mention fence: {code:?}"
        );
    }

    #[test]
    fn extract_code_block_python() {
        let text = "```python\ndef foo():\n    return 42\n```";
        assert!(
            extract_code_block(text).is_none(),
            "a fence that is not a real merge_sorted must fall back to the full text"
        );
    }

    #[test]
    fn extract_code_block_bare() {
        let text = "```\nprint(1)\n```";
        assert!(
            extract_code_block(text).is_none(),
            "a name-less fence must fall back to the full text"
        );
    }

    #[test]
    fn count_char_works() {
        assert_eq!(count_char("((()))", '('), 3);
        assert_eq!(count_char("((()))", ')'), 3);
        assert_eq!(count_char("abc", '('), 0);
    }

    #[tokio::test]
    async fn code_syntax_value_ellipsis_is_not_strong() {
        for code in [
            "def merge_sorted(a, b):\n    return (...)\n",
            "def merge_sorted(a, b):\n    return(...)\n",
            "def merge_sorted(a, b):\n    return ( ... )\n",
            "def merge_sorted(a, b):\n    return ((...))\n",
            "def merge_sorted(a, b):\n    return (...,)\n",
            "def merge_sorted(a, b):\n    return (...)  # done\n",
            "def merge_sorted(a, b):\n    return [...]\n",
            "def merge_sorted(a, b):\n    return {...}\n",
            "def merge_sorted(a, b):\n    return ....\n",
            "def merge_sorted(a, b):\n    out = (...)\n    return out\n",
            "def merge_sorted(a, b):\n    out = [...]\n    return out\n",
            "def merge_sorted(a, b):\n    out = {...}\n    return out\n",
            "def merge_sorted(a, b):\n    out=(...)\n    return out\n",
            "def merge_sorted(a, b):\n    out: list = (...)\n    return out\n",
            "def merge_sorted(a, b):\n    a = b = (...)\n    return a\n",
            "def merge_sorted(a, b):\n    return f\"{...}\"\n",
            "def merge_sorted(a, b):\n    return f'{...}'\n",
            "def merge_sorted(a, b):\n    return f\"\"\"{...}\"\"\"\n",
            "def merge_sorted(a, b):\n    return rf\"{...}\"\n",
            "def merge_sorted(a, b):\n    return fr\"{...}\"\n",
            "def merge_sorted(a, b):\n    return F\"{...}\"\n",
            "def merge_sorted(a, b):\n    return f\"{...!r}\"\n",
            "def merge_sorted(a, b):\n    return f\"{a + ...}\"\n",
            "def merge_sorted(a, b):\n    note = f\"{...}\"\n    return note\n",
            "def merge_sorted(a, b):\n    return f\"\"\"\n{...}\n\"\"\"\n",
            "def merge_sorted(a, b):\n    return a + ...\n",
            "def merge_sorted(a, b):\n    x = a + ...\n    return x\n",
            "def merge_sorted(a, b):\n    return a // ...\n",
            "def merge_sorted(a, b):\n    return \"...\" + ...\n",
            "def merge_sorted(a, b):\n    return helper(...)\n",
            "def merge_sorted(a, b):\n    out = []\n    out.append(...)\n    return out\n",
            "def merge_sorted(a, b):\n    assert ...\n    return a + b\n",
            "def merge_sorted(a, b):\n    for x in ...:\n        pass\n    return a + b\n",
            "def merge_sorted(a, b):\n    x: int = ...\n    return x\n",
            "def merge_sorted(a, b):\n    x: int = f\"{...}\"\n    return x\n",
            "def merge_sorted(a, b):\n    return f\"\\é{...}\"\n",
            "def merge_sorted(a, b):\n    return f\"{a == ...}\"\n",
            "def merge_sorted(a, b):\n    return a == ...\n",
            "def merge_sorted(a, b):\n    ...\n    return a + b\n",
            "def merge_sorted(a, b):\n    return (\n        ...\n    )\n",
            "def merge_sorted(a, b=...):\n    return ...\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "value ellipsis must not be Strong: {code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_non_value_ellipsis_stays_strong() {
        for code in [
            "def merge_sorted(a, b=...):\n    return a + b\n",
            "def merge_sorted(a, b = ...):\n    return a + b\n",
            "def merge_sorted(a, b=f\"{...}\"):\n    return a + b\n",
            "def merge_sorted(\n    a,\n    b=...,\n):\n    return a + b\n",
            "def merge_sorted(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:\n    return a + b\n",
            "def merge_sorted(a, b):\n    x: tuple[int, ...] = (1,)\n    return a + b\n",
            "def merge_sorted(a, b):\n    return a[...]\n",
            "def merge_sorted(a, b):\n    out = a[..., 0]\n    return out\n",
            "def merge_sorted(a, b):\n    return a[1:...]\n",
            "def merge_sorted(a, b):\n    return f\"{'...'}\"\n",
            "def merge_sorted(a, b):\n    return f\"{a + '...'}\"\n",
            "def merge_sorted(a, b):\n    return f\"{a:...}\"\n",
            "def merge_sorted(a, b):\n    return f\"{{...}}\"\n",
            "def merge_sorted(a, b):\n    return\"...\"\n",
            "def merge_sorted(a, b):\n    return \"a single sorted list\"\n",
            "def merge_sorted(a, b):\n    return f\"café ...\"\n",
            "def merge_sorted(a, b):\n    return f\"{a == b}\"\n",
            "def merge_sorted(a, b):\n    if (x := a):\n        return a + b\n    return b\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "ellipsis in a slice, signature, or string must stay Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }
}
