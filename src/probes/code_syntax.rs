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
///   expression, and a call argument. An English return stays Weak
///   when a comma, hyphen, parenthesis, colon, or exclamation mark
///   sits in the phrase. A return that stops on an operator stays
///   below Strong. An augmented assignment such as `a += b` is a
///   statement, so the function is present. So are `raise`, `yield`,
///   `assert`, `del`, `global`, `nonlocal`, `import`, `from`, `break`,
///   `continue`, `await`, a `while` or `for` header, an annotated
///   assignment (`x: int = 1`), and a parenthesized statement
///   (`(a := a + b)`, `(lambda x: x)(a)`). A method call
///   (`a.extend(b)`), an attribute or subscript assignment
///   (`a.items = b`, `a[0] = b`), and an augmented assignment on
///   those targets (`a.items += b`) are statements too. A call that
///   wraps onto the next line still counts. `pass` with balanced
///   delimiters is incomplete, not a syntax error. A `return` inside a
///   nested `def`, `async def`, or `class` does not finish this function.
/// - `0.5` - code present and mostly correct but has minor issues
///   (unbalanced delimiters, a real ellipsis, a body that is only
///   `...`, or missing return)
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
    // Keep string text here. Stripping it turns `return {"a": b}` into
    // `return {: b}`. Lines inside a string are skipped below.
    let has_return = merge_sorted_body_has_return(&commented);

    let parens_balanced = count_char(&code_body, '(') == count_char(&code_body, ')');
    let brackets_balanced = count_char(&code_body, '[') == count_char(&code_body, ']');
    let braces_balanced = count_char(&code_body, '{') == count_char(&code_body, '}');
    let delimiters_ok = parens_balanced && brackets_balanced && braces_balanced;

    // Comments are already gone. String data stays, so an f-string
    // expression can still be read. A value `...` is a stub. A slice,
    // a type subscript, and a parameter default are not.
    let has_ellipsis = has_stub_ellipsis(&commented);

    let (score, details) = if has_def && has_return && delimiters_ok && !has_ellipsis {
        (
            1.0,
            "Valid function with correct signature, return, and balanced delimiters".to_string(),
        )
    } else if has_def && delimiters_ok {
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
            let rest_at = idx + needle.len() + colon_abs + 1;
            if body_lines_are_real(text, rest, rest_at) {
                return true;
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

enum TopMark {
    Colon,
    Assign,
}

/// Index of a depth-0 `:` or plain `=`, outside strings and brackets.
fn top_level_mark(s: &str, kind: TopMark) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(s, i) {
            i = end;
            continue;
        }
        let Some(c) = s[i..].chars().next() else {
            break;
        };
        let at_zero = paren <= 0 && brack <= 0 && brace <= 0;
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            ':' if at_zero
                && matches!(kind, TopMark::Colon)
                && !s[i + c.len_utf8()..].starts_with('=') =>
            {
                return Some(i);
            }
            '=' if at_zero && matches!(kind, TopMark::Assign) => {
                let prev = i.checked_sub(1).map(|p| bytes[p]);
                let next = bytes.get(i + 1).copied();
                let aug = matches!(
                    prev,
                    Some(
                        b'=' | b'!'
                            | b'<'
                            | b'>'
                            | b':'
                            | b'+'
                            | b'-'
                            | b'*'
                            | b'/'
                            | b'%'
                            | b'&'
                            | b'|'
                            | b'^'
                            | b'@'
                    )
                );
                if next != Some(b'=') && !aug {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += c.len_utf8();
    }
    None
}

fn keyword_rest<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
    let rest = line.trim_start().strip_prefix(keyword)?;
    if rest
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    Some(rest)
}

fn split_keyword_once<'a>(s: &'a str, keyword: &str) -> Option<(&'a str, &'a str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + keyword.len() <= s.len() {
        if s[i..].starts_with(keyword) {
            let before_ok =
                i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
            let after = i + keyword.len();
            let after_ok =
                after == s.len() || (!bytes[after].is_ascii_alphanumeric() && bytes[after] != b'_');
            if before_ok && after_ok {
                return Some((s[..i].trim(), s[after..].trim()));
            }
        }
        i += s[i..].chars().next().map_or(1, char::len_utf8);
    }
    None
}

fn statement_tail(after: &str) -> String {
    strip_hash_comments(after)
        .trim()
        .trim_end_matches(';')
        .trim()
        .to_string()
}

fn is_dotted_name(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty() && !s.starts_with('.') && !s.ends_with('.') && s.split('.').all(is_simple_ident)
}

fn is_from_module(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let dots = s.bytes().take_while(|b| *b == b'.').count();
    let rest = &s[dots..];
    rest.is_empty() || is_dotted_name(rest)
}

fn import_items(s: &str, dotted: bool) -> bool {
    let s = s.trim();
    let s = if s.starts_with('(') && s.ends_with(')') && s.len() >= 2 {
        s[1..s.len() - 1].trim()
    } else {
        s
    };
    if s.is_empty() {
        return false;
    }
    if !dotted && s == "*" {
        return true;
    }
    s.split(',').all(|part| {
        let part = part.trim();
        if part.is_empty() {
            return false;
        }
        let (name, alias) = match split_keyword_once(part, "as") {
            Some((name, alias)) => (name, Some(alias)),
            None => (part, None),
        };
        let name_ok = if dotted {
            is_dotted_name(name)
        } else {
            is_simple_ident(name)
        };
        name_ok && alias.is_none_or(is_simple_ident)
    })
}

fn expr_list_is_code(s: &str, max_parts: usize) -> bool {
    let parts = split_top_level_commas(s);
    !parts.is_empty()
        && parts.len() <= max_parts
        && parts
            .iter()
            .all(|part| !part.is_empty() && return_rest_is_code(part))
}

fn looks_like_raise(after: &str) -> bool {
    let tail = statement_tail(after);
    if tail.is_empty() {
        return true;
    }
    if let Some((left, right)) = split_keyword_once(&tail, "from") {
        return !left.is_empty()
            && !right.is_empty()
            && return_rest_is_code(left)
            && return_rest_is_code(right);
    }
    return_rest_is_code(&tail)
}

fn looks_like_suite(after: &str, header_ok: impl Fn(&str) -> bool) -> bool {
    let commented = strip_hash_comments(after);
    let Some(colon) = top_level_mark(&commented, TopMark::Colon) else {
        return false;
    };
    let header = commented[..colon].trim();
    let suite = commented[colon + 1..].trim();
    if header.is_empty() || !header_ok(header) {
        return false;
    }
    suite.is_empty() || looks_like_same_line_body(suite)
}

fn looks_like_keyword_statement(line: &str) -> bool {
    let t = line.trim_start();
    if let Some(after) = keyword_rest(t, "break") {
        return statement_tail(after).is_empty();
    }
    if let Some(after) = keyword_rest(t, "continue") {
        return statement_tail(after).is_empty();
    }
    if let Some(after) = keyword_rest(t, "raise") {
        return looks_like_raise(after);
    }
    if let Some(after) = keyword_rest(t, "assert") {
        return expr_list_is_code(&statement_tail(after), 2);
    }
    if let Some(after) = keyword_rest(t, "del") {
        return expr_list_is_code(&statement_tail(after), 32);
    }
    if let Some(after) = keyword_rest(t, "global") {
        let tail = statement_tail(after);
        return !tail.is_empty() && tail.split(',').all(|part| is_simple_ident(part.trim()));
    }
    if let Some(after) = keyword_rest(t, "nonlocal") {
        let tail = statement_tail(after);
        return !tail.is_empty() && tail.split(',').all(|part| is_simple_ident(part.trim()));
    }
    if let Some(after) = keyword_rest(t, "import") {
        return import_items(&statement_tail(after), true);
    }
    if let Some(after) = keyword_rest(t, "from") {
        let tail = statement_tail(after);
        let Some((module, names)) = split_keyword_once(&tail, "import") else {
            return false;
        };
        return is_from_module(module) && import_items(names, false);
    }
    if keyword_rest(t, "await").is_some() || keyword_rest(t, "yield").is_some() {
        return return_rest_is_code(t);
    }
    if let Some(after) = keyword_rest(t, "while") {
        return looks_like_suite(after, return_rest_is_code);
    }
    if let Some(after) = keyword_rest(t, "for") {
        return looks_like_suite(after, |header| {
            let Some((target, expr)) = split_keyword_once(header, "in") else {
                return false;
            };
            !target.is_empty() && return_rest_is_code(target) && return_rest_is_code(expr)
        });
    }
    false
}

fn looks_like_paren_statement(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('(') && (return_rest_is_code(t) || looks_like_trailer_statement(t))
}

/// Text before a depth-0 `;`. A semicolon inside a string or brackets
/// stays in the statement.
fn first_statement(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(s, i) {
            i = end;
            continue;
        }
        let Some(c) = s[i..].chars().next() else {
            break;
        };
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            ';' if paren <= 0 && brack <= 0 && brace <= 0 => return s[..i].trim(),
            _ => {}
        }
        i += c.len_utf8();
    }
    s.trim()
}

fn top_level_augassign(s: &str) -> Option<(usize, usize)> {
    const OPS: &[&str] = &[
        "**=", "//=", ">>=", "<<=", "+=", "-=", "*=", "@=", "/=", "%=", "&=", "|=", "^=",
    ];
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(s, i) {
            i = end;
            continue;
        }
        let Some(c) = s[i..].chars().next() else {
            break;
        };
        if paren <= 0
            && brack <= 0
            && brace <= 0
            && let Some(op) = OPS.iter().find(|op| s[i..].starts_with(*op))
        {
            return Some((i, op.len()));
        }
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        i += c.len_utf8();
    }
    None
}

fn looks_like_chained_assign(stmt: &str) -> bool {
    let Some(eq) = top_level_mark(stmt, TopMark::Assign) else {
        return false;
    };
    let left = stmt[..eq].trim();
    let right = stmt[eq + 1..].trim();
    if left.is_empty() || right.is_empty() {
        return false;
    }
    assignment_targets(left) && (return_rest_is_code(right) || looks_like_chained_assign(right))
}

/// `a.extend(b)`, `a.items = b`, `a[0] += 1`. The name is already
/// known. A dot or subscript is what the bare-name check misses.
fn looks_like_trailer_statement(line: &str) -> bool {
    let tail = statement_tail(line);
    if tail.is_empty() {
        return false;
    }
    let stmt = first_statement(&tail);
    if stmt.is_empty() {
        return false;
    }
    if return_rest_is_code(stmt) {
        return true;
    }
    if let Some((at, len)) = top_level_augassign(stmt) {
        let left = stmt[..at].trim();
        let right = stmt[at + len..].trim();
        return assignment_targets(left)
            && !right.is_empty()
            && (return_rest_is_code(right) || unclosed_delimiters(right));
    }
    if looks_like_chained_assign(stmt) {
        return true;
    }
    // `a.extend(` continues on the next line. An open bracket is enough
    // to say the function is there; the delimiter check scores the close.
    unclosed_delimiters(stmt)
}

fn assignment_targets(s: &str) -> bool {
    let parts: Vec<&str> = split_top_level_commas(s)
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect();
    !parts.is_empty() && parts.iter().copied().all(one_assign_target)
}

fn one_assign_target(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let mut parser = ExprParser { s, i: 0, group: 0 };
    parser.parse_assign_target() && {
        parser.skip_ws();
        parser.i >= parser.s.len()
    }
}

fn looks_like_ann_assign(after: &str) -> bool {
    let commented = strip_hash_comments(after);
    let Some(body) = commented.trim_start().strip_prefix(':') else {
        return false;
    };
    let body = body.trim().trim_end_matches(';').trim();
    if body.is_empty() {
        return false;
    }
    if let Some(eq) = top_level_mark(body, TopMark::Assign) {
        let ann = body[..eq].trim();
        let value = body[eq + 1..].trim();
        return !ann.is_empty()
            && !value.is_empty()
            && return_rest_is_code(ann)
            && return_rest_is_code(value);
    }
    return_rest_is_code(body)
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
            // A bare `return` is a body. Anything after it has to be a
            // finished expression: `return a + b`, `return (a + b)`,
            // `return a or b`. English with a comma, hyphen, or
            // parenthesis is not. A dangling operator is not either.
            if after.is_empty() {
                return true;
            }
            let rest = after.trim_start();
            if rest.is_empty() {
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
    // `...` is the usual stub body. A docstring line is skipped by the
    // caller, so this does not treat an ellipsis inside quotes as code.
    if strip_hash_comments(t).trim() == "..." {
        return true;
    }
    if t.starts_with('[') {
        return true;
    }
    if looks_like_paren_statement(t) || looks_like_keyword_statement(t) {
        return true;
    }
    let ident_end = t
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(t.len());
    if ident_end == 0 || !is_simple_ident(&t[..ident_end]) {
        return false;
    }
    let after = t[ident_end..].trim_start();
    if after.starts_with('=') || after.starts_with('(') || starts_augassign(after) {
        return true;
    }
    if after.starts_with(':') && looks_like_ann_assign(after) {
        return true;
    }
    // `sorted(a + b)` is the `(` arm. `a.extend(b)` and `a[0] = b`
    // continue the same name through `.` or `[`.
    (after.starts_with('.') || after.starts_with('[')) && looks_like_trailer_statement(t)
}

fn starts_augassign(after: &str) -> bool {
    const OPS: &[&str] = &[
        "+=", "-=", "*=", "@=", "/=", "%=", "&=", "|=", "^=", ">>=", "<<=", "**=", "//=",
    ];
    OPS.iter().any(|op| after.starts_with(op))
}

fn return_rest_is_code(rest: &str) -> bool {
    let commented = strip_hash_comments(rest);
    let trimmed = commented.trim();
    if trimmed.is_empty() {
        return false;
    }
    let sentence = strip_sentence_period(trimmed);
    let mut parser = ExprParser {
        s: sentence,
        i: 0,
        group: 0,
    };
    parser.parse_star_exprs() && {
        parser.skip_ws();
        // `return a + b;` is one statement. Words after `;` are not.
        if parser.rest().starts_with(';') {
            parser.i += 1;
            parser.skip_ws();
        }
        parser.i >= parser.s.len()
    }
}

fn line_starts_return(line: &str) -> bool {
    let t = line.trim_start();
    let Some(after) = t.strip_prefix("return") else {
        return false;
    };
    after.is_empty() || after.starts_with(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
}

/// Parentheses, brackets, and braces join the next indented lines.
fn unclosed_delimiters(s: &str) -> bool {
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    let mut i = 0;
    while i < s.len() {
        if let Some((quote_at, quote, _)) = string_literal_at(s, i) {
            if quote_at > i {
                i = quote_at;
            }
            let end = skip_python_string(s, quote_at, quote);
            if end <= i {
                let Some(ch) = s[i..].chars().next() else {
                    break;
                };
                i += ch.len_utf8();
            } else {
                i = end;
            }
            continue;
        }
        let Some(ch) = s[i..].chars().next() else {
            break;
        };
        match ch {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => brack += 1,
            ']' => brack -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        i += ch.len_utf8();
    }
    paren > 0 || brack > 0 || brace > 0
}

fn is_hash_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn ends_with_line_join(line: &str) -> bool {
    strip_hash_comments(line).trim_end().ends_with('\\')
}

fn join_open_return(lines: &[&str]) -> (String, usize) {
    let Some(first) = lines.first().copied() else {
        return (String::new(), 1);
    };
    if !line_starts_return(first) || !(unclosed_delimiters(first) || ends_with_line_join(first)) {
        return (first.to_string(), 1);
    }
    let mut joined = first.to_string();
    let mut n = 1;
    while n < lines.len() && (unclosed_delimiters(&joined) || ends_with_line_join(&joined)) {
        let next = lines[n];
        if is_hash_comment_line(next) {
            n += 1;
            continue;
        }
        if !(next.trim().is_empty() || next.starts_with(' ') || next.starts_with('\t')) {
            break;
        }
        if ends_with_line_join(&joined) {
            let code = strip_hash_comments(&joined);
            let trimmed = code.trim_end();
            if let Some(body) = trimmed.strip_suffix('\\') {
                joined = body.trim_end().to_string();
            }
        }
        joined.push('\n');
        joined.push_str(next);
        n += 1;
    }
    (joined, n)
}

fn body_lines_are_real(text: &str, rest: &str, rest_at: usize) -> bool {
    let lines: Vec<&str> = rest.lines().collect();
    let starts = line_starts(rest, rest_at);
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if i > 0 && line.trim().is_empty() {
            i += 1;
            continue;
        }
        if i > 0 && !(line.starts_with(' ') || line.starts_with('\t')) {
            // A column-0 `#` line is a comment, not the next function.
            if is_hash_comment_line(line) {
                i += 1;
                continue;
            }
            return false;
        }
        if !line_outside_string(text, starts[i]) {
            i += 1;
            continue;
        }
        let (joined, used) = join_open_return(&lines[i..]);
        if looks_like_same_line_body(&joined) {
            return true;
        }
        i += used.max(1);
    }
    false
}

fn line_starts(rest: &str, rest_at: usize) -> Vec<usize> {
    let lines: Vec<&str> = rest.lines().collect();
    let bytes = rest.as_bytes();
    let mut starts = Vec::with_capacity(lines.len());
    let mut at = rest_at;
    let mut b = 0;
    for line in &lines {
        starts.push(at);
        b += line.len();
        at += line.len();
        if bytes.get(b) == Some(&b'\n') {
            b += 1;
            at += 1;
        }
    }
    starts
}

fn rest_has_return_line(text: &str, rest: &str, rest_at: usize) -> bool {
    let lines: Vec<&str> = rest.lines().collect();
    let mut starts = Vec::with_capacity(lines.len());
    let bytes = rest.as_bytes();
    let mut at = rest_at;
    let mut b = 0;
    for line in &lines {
        starts.push(at);
        b += line.len();
        at += line.len();
        if bytes.get(b) == Some(&b'\n') {
            b += 1;
            at += 1;
        }
    }
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if i > 0 && line.trim().is_empty() {
            i += 1;
            continue;
        }
        if i > 0 && !(line.starts_with(' ') || line.starts_with('\t')) {
            return false;
        }
        // A nested def or class has its own return. That return does not
        // finish merge_sorted. An if or while suite still does.
        if line_outside_string(text, starts[i]) && is_suite_header(line) {
            i = skip_nested_suite(&lines, i);
            continue;
        }
        let (joined, used) = join_open_return(&lines[i..]);
        if line_outside_string(text, starts[i]) && line_has_return_token(&joined) {
            return true;
        }
        i += used.max(1);
    }
    false
}

fn leading_indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ' || *c == '\t').count()
}

fn is_suite_header(line: &str) -> bool {
    let mut t = line.trim_start();
    if let Some(rest) = t.strip_prefix("async")
        && rest.starts_with(|c: char| c.is_whitespace())
    {
        t = rest.trim_start();
    }
    for kw in ["def", "class"] {
        if let Some(rest) = t.strip_prefix(kw)
            && (rest.is_empty()
                || rest.starts_with(|c: char| !(c.is_ascii_alphanumeric() || c == '_')))
        {
            return true;
        }
    }
    false
}

/// Index of the first line after a nested `def` or `class` suite.
///
/// The header may wrap while parentheses are open. Lines indented
/// past the header belong to that suite, including a further nested
/// function. The same indent is `merge_sorted` again.
fn skip_nested_suite(lines: &[&str], header_at: usize) -> usize {
    let header_indent = leading_indent(lines[header_at]);
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    let mut header_end = header_at;
    let mut i = header_at;
    let mut closed = false;
    while i < lines.len() {
        let line = lines[i];
        if i > header_at && line.trim().is_empty() {
            i += 1;
            continue;
        }
        if i > header_at
            && paren <= 0
            && brack <= 0
            && brace <= 0
            && leading_indent(line) <= header_indent
        {
            break;
        }
        header_end = i;
        if header_colon_on_line(line, &mut paren, &mut brack, &mut brace) {
            closed = true;
            break;
        }
        i += 1;
    }
    let mut j = header_end + 1;
    if !closed {
        return j;
    }
    while j < lines.len() {
        let line = lines[j];
        if line.trim().is_empty() {
            j += 1;
            continue;
        }
        if leading_indent(line) > header_indent {
            j += 1;
            continue;
        }
        break;
    }
    j
}

fn header_colon_on_line(line: &str, paren: &mut i32, brack: &mut i32, brace: &mut i32) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = skip_string_at(line, i) {
            i = end;
            continue;
        }
        let Some(ch) = line[i..].chars().next() else {
            break;
        };
        if ch == '#' {
            return false;
        }
        match ch {
            '(' => *paren += 1,
            ')' => *paren -= 1,
            '[' => *brack += 1,
            ']' => *brack -= 1,
            '{' => *brace += 1,
            '}' => *brace -= 1,
            ':' if *paren <= 0 && *brack <= 0 && *brace <= 0 => return true,
            _ => {}
        }
        i += ch.len_utf8();
    }
    false
}

/// Drop one sentence period. Keep `1.` and a trailing attribute dot.
fn strip_sentence_period(s: &str) -> &str {
    let Some(body) = s.strip_suffix('.') else {
        return s;
    };
    if body.is_empty() || body.ends_with('.') || body.ends_with(|c: char| c.is_ascii_digit()) {
        return s;
    }
    if !body.chars().any(|c| c.is_whitespace()) {
        return s;
    }
    body
}

struct ExprParser<'a> {
    s: &'a str,
    i: usize,
    group: u32,
}

impl<'a> ExprParser<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn skip_ws(&mut self) {
        let skip = self
            .rest()
            .chars()
            .take_while(|c| c.is_whitespace())
            .map(|c| c.len_utf8())
            .sum::<usize>();
        self.i += skip;
    }

    fn bump_str(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn keyword(&mut self, word: &str) -> bool {
        self.skip_ws();
        let rest = self.rest();
        if !rest.starts_with(word) {
            return false;
        }
        let after = rest.as_bytes().get(word.len()).copied();
        if after.is_some_and(is_ident_continue) {
            return false;
        }
        self.i += word.len();
        true
    }

    fn peek_keyword(&mut self, word: &str) -> bool {
        let saved = self.i;
        let ok = self.keyword(word);
        self.i = saved;
        ok
    }

    fn bump_op(&mut self, op: &str) -> bool {
        self.skip_ws();
        let rest = self.rest();
        if !rest.starts_with(op) {
            return false;
        }
        let next = rest.as_bytes().get(op.len()).copied();
        if next == Some(b'=') {
            return false;
        }
        if op == "<" && next == Some(b'<') || op == ">" && next == Some(b'>') {
            return false;
        }
        if op == "*" && next == Some(b'*') || op == "/" && next == Some(b'/') {
            return false;
        }
        self.i += op.len();
        true
    }

    fn parse_maybe_starred(&mut self) -> Option<bool> {
        self.skip_ws();
        if self.rest().starts_with("**") {
            return None;
        }
        if self.rest().starts_with('*') {
            self.i += 1;
            return self.parse_expr().then_some(true);
        }
        self.parse_expr().then_some(false)
    }

    fn parse_star_exprs(&mut self) -> bool {
        let Some(mut saw_star) = self.parse_maybe_starred() else {
            return false;
        };
        let mut saw_comma = false;
        loop {
            self.skip_ws();
            if !self.rest().starts_with(',') {
                break;
            }
            saw_comma = true;
            self.i += 1;
            self.skip_ws();
            if self.i >= self.s.len()
                || self.rest().starts_with(')')
                || self.rest().starts_with(']')
                || self.rest().starts_with('}')
            {
                break;
            }
            let Some(star) = self.parse_maybe_starred() else {
                return false;
            };
            saw_star |= star;
        }
        // `return *a` is a syntax error. `return *a,` is a tuple.
        !saw_star || saw_comma
    }

    fn parse_expr(&mut self) -> bool {
        if !self.parse_or() {
            return false;
        }
        if self.peek_keyword("if") {
            self.keyword("if");
            if !self.parse_or() || !self.keyword("else") || !self.parse_expr() {
                return false;
            }
        }
        true
    }

    fn parse_or(&mut self) -> bool {
        if !self.parse_and() {
            return false;
        }
        while self.peek_keyword("or") {
            self.keyword("or");
            if !self.parse_and() {
                return false;
            }
        }
        true
    }

    fn parse_and(&mut self) -> bool {
        if !self.parse_not() {
            return false;
        }
        while self.peek_keyword("and") {
            self.keyword("and");
            if !self.parse_not() {
                return false;
            }
        }
        true
    }

    fn parse_not(&mut self) -> bool {
        if self.peek_keyword("not") {
            self.keyword("not");
            return self.parse_not();
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> bool {
        if !self.parse_bor() {
            return false;
        }
        while self.take_comp_op() {
            if !self.parse_bor() {
                return false;
            }
        }
        true
    }

    fn take_comp_op(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with("<<") || self.rest().starts_with(">>") {
            return false;
        }
        if self.bump_str("<=")
            || self.bump_str(">=")
            || self.bump_str("==")
            || self.bump_str("!=")
            || self.bump_str("<")
            || self.bump_str(">")
        {
            return true;
        }
        if self.peek_keyword("not") {
            let saved = self.i;
            self.keyword("not");
            if self.keyword("in") {
                return true;
            }
            self.i = saved;
            return false;
        }
        if self.peek_keyword("is") {
            self.keyword("is");
            let _ = self.keyword("not");
            return true;
        }
        if self.peek_keyword("in") {
            return self.keyword("in");
        }
        false
    }

    fn parse_bor(&mut self) -> bool {
        if !self.parse_bxor() {
            return false;
        }
        while self.bump_op("|") {
            if !self.parse_bxor() {
                return false;
            }
        }
        true
    }

    fn parse_bxor(&mut self) -> bool {
        if !self.parse_band() {
            return false;
        }
        while self.bump_op("^") {
            if !self.parse_band() {
                return false;
            }
        }
        true
    }

    fn parse_band(&mut self) -> bool {
        if !self.parse_shift() {
            return false;
        }
        while self.bump_op("&") {
            if !self.parse_shift() {
                return false;
            }
        }
        true
    }

    fn parse_shift(&mut self) -> bool {
        if !self.parse_sum() {
            return false;
        }
        loop {
            self.skip_ws();
            if self.rest().starts_with("<<") || self.rest().starts_with(">>") {
                self.i += 2;
                if self.rest().starts_with('=') {
                    return false;
                }
                if !self.parse_sum() {
                    return false;
                }
                continue;
            }
            return true;
        }
    }

    fn parse_sum(&mut self) -> bool {
        if !self.parse_term() {
            return false;
        }
        while self.bump_op("+") || self.bump_op("-") {
            if !self.parse_term() {
                return false;
            }
        }
        true
    }

    fn parse_term(&mut self) -> bool {
        if !self.parse_factor() {
            return false;
        }
        loop {
            self.skip_ws();
            if self.rest().starts_with("//") {
                self.i += 2;
                if self.rest().starts_with('=') {
                    return false;
                }
                if !self.parse_factor() {
                    return false;
                }
                continue;
            }
            if self.bump_op("*") || self.bump_op("/") || self.bump_op("%") || self.bump_op("@") {
                if !self.parse_factor() {
                    return false;
                }
                continue;
            }
            return true;
        }
    }

    fn parse_factor(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with("**") {
            return self.parse_power();
        }
        if self.bump_op("+") || self.bump_op("-") || self.bump_op("~") {
            return self.parse_factor();
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> bool {
        if !self.parse_await_primary() {
            return false;
        }
        self.skip_ws();
        if self.rest().starts_with("**") {
            self.i += 2;
            if self.rest().starts_with('=') {
                return false;
            }
            return self.parse_factor();
        }
        true
    }

    fn parse_await_primary(&mut self) -> bool {
        if self.peek_keyword("await") {
            self.keyword("await");
            return self.parse_primary();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> bool {
        self.skip_ws();
        if self.peek_keyword("lambda") {
            return self.parse_lambda();
        }
        if self.peek_keyword("yield") {
            return self.parse_yield();
        }
        if self.consume_string() || self.consume_ellipsis() || self.consume_number() {
            return self.parse_trailers();
        }
        if self.consume_name() {
            self.skip_ws();
            if self.group > 0 && self.rest().starts_with(":=") {
                self.i += 2;
                if !self.parse_expr() {
                    return false;
                }
            }
            return self.parse_trailers();
        }
        if self.rest().starts_with('(') {
            return self.parse_paren();
        }
        if self.rest().starts_with('[') {
            return self.parse_bracket();
        }
        if self.rest().starts_with('{') {
            return self.parse_brace();
        }
        false
    }

    fn parse_trailers(&mut self) -> bool {
        loop {
            self.skip_ws();
            if self.rest().starts_with('(') {
                if !self.parse_call_args() {
                    return false;
                }
                continue;
            }
            if self.rest().starts_with('[') {
                if !self.parse_subscript() {
                    return false;
                }
                continue;
            }
            if self.rest().starts_with('.') {
                let next = self.rest().as_bytes().get(1).copied();
                if next == Some(b'.') || next.is_some_and(|b| b.is_ascii_digit()) {
                    return false;
                }
                self.i += 1;
                if !self.consume_name() {
                    return false;
                }
                continue;
            }
            return true;
        }
    }

    fn parse_assign_target(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with("**") {
            return false;
        }
        let starred = if self.rest().starts_with('*') {
            self.i += 1;
            self.skip_ws();
            true
        } else {
            false
        };
        if self.rest().starts_with('(') || self.rest().starts_with('[') {
            if starred {
                return false;
            }
            return self.parse_target_group();
        }
        if !self.consume_name() {
            return false;
        }
        if starred {
            return true;
        }
        self.parse_target_trailers()
    }

    fn parse_target_trailers(&mut self) -> bool {
        loop {
            self.skip_ws();
            if self.rest().starts_with('.') {
                let next = self.rest().as_bytes().get(1).copied();
                if next == Some(b'.') || next.is_some_and(|b| b.is_ascii_digit()) {
                    return false;
                }
                self.i += 1;
                self.skip_ws();
                if !self.consume_name() {
                    return false;
                }
                continue;
            }
            if self.rest().starts_with('[') {
                if !self.parse_subscript() {
                    return false;
                }
                continue;
            }
            return true;
        }
    }

    fn parse_target_group(&mut self) -> bool {
        let close = if self.rest().starts_with('(') {
            ')'
        } else {
            ']'
        };
        self.i += 1;
        self.skip_ws();
        if self.rest().starts_with(close) {
            return false;
        }
        loop {
            if !self.parse_assign_target() {
                return false;
            }
            self.skip_ws();
            if self.rest().starts_with(',') {
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with(close) {
                    self.i += 1;
                    return true;
                }
                continue;
            }
            if self.rest().starts_with(close) {
                self.i += 1;
                return true;
            }
            return false;
        }
    }

    fn parse_lambda(&mut self) -> bool {
        if !self.keyword("lambda") {
            return false;
        }
        self.skip_ws();
        if !self.rest().starts_with(':') {
            loop {
                self.skip_ws();
                if self.rest().starts_with(':') {
                    break;
                }
                if self.rest().starts_with("**") || self.rest().starts_with('*') {
                    self.i += if self.rest().starts_with("**") { 2 } else { 1 };
                    let _ = self.consume_name();
                } else if !self.consume_name() {
                    return false;
                }
                self.skip_ws();
                if self.rest().starts_with('=') && !self.rest().starts_with("==") {
                    self.i += 1;
                    if !self.parse_expr() {
                        return false;
                    }
                }
                self.skip_ws();
                if self.rest().starts_with(',') {
                    self.i += 1;
                    continue;
                }
                break;
            }
        }
        self.skip_ws();
        if !self.bump_str(":") {
            return false;
        }
        self.parse_expr()
    }

    fn parse_yield(&mut self) -> bool {
        if !self.keyword("yield") {
            return false;
        }
        let _ = self.keyword("from");
        self.skip_ws();
        if self.i >= self.s.len() || self.rest().starts_with([')', ']', '}', ',', ':']) {
            return true;
        }
        self.parse_expr()
    }

    fn parse_paren(&mut self) -> bool {
        self.i += 1;
        self.group += 1;
        self.skip_ws();
        if self.rest().starts_with(')') {
            self.i += 1;
            self.group -= 1;
            return self.parse_trailers();
        }
        if self.peek_keyword("yield") {
            let ok = self.parse_yield();
            self.skip_ws();
            if !ok || !self.rest().starts_with(')') {
                self.group -= 1;
                return false;
            }
            self.i += 1;
            self.group -= 1;
            return self.parse_trailers();
        }
        let Some(mut saw_star) = self.parse_maybe_starred() else {
            self.group -= 1;
            return false;
        };
        let mut saw_comma = false;
        self.skip_ws();
        if self.peek_keyword("for") {
            let ok = self.parse_comp_for();
            self.skip_ws();
            if !ok || !self.rest().starts_with(')') {
                self.group -= 1;
                return false;
            }
            self.i += 1;
            self.group -= 1;
            return self.parse_trailers();
        }
        loop {
            self.skip_ws();
            if self.rest().starts_with(')') {
                if saw_star && !saw_comma {
                    self.group -= 1;
                    return false;
                }
                self.i += 1;
                self.group -= 1;
                return self.parse_trailers();
            }
            if self.rest().starts_with(',') {
                saw_comma = true;
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with(')') {
                    self.i += 1;
                    self.group -= 1;
                    return self.parse_trailers();
                }
                let Some(star) = self.parse_maybe_starred() else {
                    self.group -= 1;
                    return false;
                };
                saw_star |= star;
                continue;
            }
            self.group -= 1;
            return false;
        }
    }

    fn parse_call_args(&mut self) -> bool {
        self.i += 1;
        self.group += 1;
        self.skip_ws();
        if self.rest().starts_with(')') {
            self.i += 1;
            self.group -= 1;
            return true;
        }
        let mut saw_keyword = false;
        let mut kwargs_closed = false;
        loop {
            self.skip_ws();
            if self.rest().starts_with("**") {
                if kwargs_closed {
                    self.group -= 1;
                    return false;
                }
                self.i += 2;
                saw_keyword = true;
                kwargs_closed = true;
                if !self.parse_expr() {
                    self.group -= 1;
                    return false;
                }
            } else if self.rest().starts_with('*') {
                // `f(a=1, *xs)` and `f(**k, *xs)` are syntax errors.
                if saw_keyword || kwargs_closed {
                    self.group -= 1;
                    return false;
                }
                self.i += 1;
                if !self.parse_expr() {
                    self.group -= 1;
                    return false;
                }
            } else if self.peek_kwarg() {
                if kwargs_closed {
                    self.group -= 1;
                    return false;
                }
                let _ = self.consume_name();
                self.skip_ws();
                self.i += 1;
                saw_keyword = true;
                if !self.parse_expr() {
                    self.group -= 1;
                    return false;
                }
            } else if saw_keyword || kwargs_closed || !self.parse_expr() {
                // A positional after a keyword is not a finished call.
                self.group -= 1;
                return false;
            }
            self.skip_ws();
            if self.peek_keyword("for") && !self.parse_comp_for() {
                self.group -= 1;
                return false;
            }
            self.skip_ws();
            if self.rest().starts_with(',') {
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with(')') {
                    break;
                }
                continue;
            }
            break;
        }
        self.skip_ws();
        if !self.rest().starts_with(')') {
            self.group -= 1;
            return false;
        }
        self.i += 1;
        self.group -= 1;
        true
    }

    fn peek_kwarg(&mut self) -> bool {
        let saved = self.i;
        let ok = self.consume_name() && {
            self.skip_ws();
            self.rest().starts_with('=') && !self.rest().starts_with("==")
        };
        self.i = saved;
        ok
    }

    fn parse_bracket(&mut self) -> bool {
        self.i += 1;
        self.group += 1;
        self.skip_ws();
        if self.rest().starts_with(']') {
            self.i += 1;
            self.group -= 1;
            return self.parse_trailers();
        }
        if self.rest().starts_with('*') && !self.rest().starts_with("**") {
            self.i += 1;
        }
        if !self.parse_expr() {
            self.group -= 1;
            return false;
        }
        self.skip_ws();
        if self.peek_keyword("for") {
            if !self.parse_comp_for() {
                self.group -= 1;
                return false;
            }
        } else {
            loop {
                self.skip_ws();
                if !self.rest().starts_with(',') {
                    break;
                }
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with(']') {
                    break;
                }
                if self.rest().starts_with('*') && !self.rest().starts_with("**") {
                    self.i += 1;
                }
                if !self.parse_expr() {
                    self.group -= 1;
                    return false;
                }
            }
        }
        self.skip_ws();
        if !self.rest().starts_with(']') {
            self.group -= 1;
            return false;
        }
        self.i += 1;
        self.group -= 1;
        self.parse_trailers()
    }

    fn parse_subscript(&mut self) -> bool {
        self.i += 1;
        self.group += 1;
        if !self.parse_subscript_list() {
            self.group -= 1;
            return false;
        }
        self.skip_ws();
        if !self.rest().starts_with(']') {
            self.group -= 1;
            return false;
        }
        self.i += 1;
        self.group -= 1;
        true
    }

    fn parse_subscript_list(&mut self) -> bool {
        loop {
            if !self.parse_slice_item() {
                return false;
            }
            self.skip_ws();
            if self.rest().starts_with(',') {
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with(']') {
                    return true;
                }
                continue;
            }
            return true;
        }
    }

    fn parse_slice_item(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with(':') {
            return self.parse_slice_tail();
        }
        if !self.parse_expr() {
            return false;
        }
        self.skip_ws();
        if self.rest().starts_with(':') {
            return self.parse_slice_tail();
        }
        true
    }

    fn parse_slice_tail(&mut self) -> bool {
        let mut colons = 0;
        while colons < 2 {
            self.skip_ws();
            if !self.rest().starts_with(':') {
                break;
            }
            self.i += 1;
            colons += 1;
            self.skip_ws();
            if self.rest().starts_with(':')
                || self.rest().starts_with(',')
                || self.rest().starts_with(']')
            {
                continue;
            }
            if !self.parse_expr() {
                return false;
            }
        }
        true
    }

    fn parse_brace(&mut self) -> bool {
        self.i += 1;
        self.group += 1;
        self.skip_ws();
        if self.rest().starts_with('}') {
            self.i += 1;
            self.group -= 1;
            return self.parse_trailers();
        }
        let mut saw_star = false;
        let mut saw_comma = false;
        let dict = if self.rest().starts_with("**") {
            self.i += 2;
            if !self.parse_expr() {
                self.group -= 1;
                return false;
            }
            true
        } else if self.rest().starts_with('*') {
            saw_star = true;
            self.i += 1;
            if !self.parse_expr() {
                self.group -= 1;
                return false;
            }
            false
        } else if !self.parse_expr() {
            self.group -= 1;
            return false;
        } else {
            false
        };
        self.skip_ws();
        let dict = dict || self.rest().starts_with(':');
        if self.rest().starts_with(':') {
            self.i += 1;
            if !self.parse_expr() {
                self.group -= 1;
                return false;
            }
        }
        self.skip_ws();
        if self.peek_keyword("for") {
            if !self.parse_comp_for() {
                self.group -= 1;
                return false;
            }
        } else {
            loop {
                self.skip_ws();
                if !self.rest().starts_with(',') {
                    break;
                }
                saw_comma = true;
                self.i += 1;
                self.skip_ws();
                if self.rest().starts_with('}') {
                    break;
                }
                if self.rest().starts_with('*') && !self.rest().starts_with("**") {
                    saw_star = true;
                    self.i += 1;
                    if !self.parse_expr() {
                        self.group -= 1;
                        return false;
                    }
                    continue;
                }
                if self.rest().starts_with("**") {
                    self.i += 2;
                    if !self.parse_expr() {
                        self.group -= 1;
                        return false;
                    }
                    continue;
                }
                if !self.parse_expr() {
                    self.group -= 1;
                    return false;
                }
                self.skip_ws();
                if self.rest().starts_with(':') {
                    self.i += 1;
                    if !self.parse_expr() {
                        self.group -= 1;
                        return false;
                    }
                } else if dict {
                    self.group -= 1;
                    return false;
                }
            }
        }
        self.skip_ws();
        if saw_star && !saw_comma {
            self.group -= 1;
            return false;
        }
        if !self.rest().starts_with('}') {
            self.group -= 1;
            return false;
        }
        self.i += 1;
        self.group -= 1;
        self.parse_trailers()
    }

    fn parse_comp_target(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with('(') || self.rest().starts_with('[') {
            let close = if self.rest().starts_with('(') {
                ')'
            } else {
                ']'
            };
            self.i += 1;
            self.skip_ws();
            if self.rest().starts_with(close) {
                return false;
            }
            loop {
                if !self.parse_comp_target() {
                    return false;
                }
                self.skip_ws();
                if self.rest().starts_with(',') {
                    self.i += 1;
                    self.skip_ws();
                    if self.rest().starts_with(close) {
                        break;
                    }
                    continue;
                }
                break;
            }
            self.skip_ws();
            if !self.rest().starts_with(close) {
                return false;
            }
            self.i += 1;
            true
        } else {
            self.consume_name()
        }
    }

    fn parse_comp_for(&mut self) -> bool {
        loop {
            if !self.keyword("for") {
                return false;
            }
            loop {
                if !self.parse_comp_target() {
                    return false;
                }
                self.skip_ws();
                if self.rest().starts_with(',') {
                    self.i += 1;
                    continue;
                }
                break;
            }
            if !self.keyword("in") || !self.parse_or() {
                return false;
            }
            while self.peek_keyword("if") {
                if !self.keyword("if") || !self.parse_or() {
                    return false;
                }
            }
            if self.peek_keyword("for") {
                continue;
            }
            return true;
        }
    }

    fn consume_string(&mut self) -> bool {
        self.skip_ws();
        let Some((quote_at, quote, _)) = string_literal_at(self.s, self.i) else {
            return false;
        };
        let Some(end) = closed_string_end(self.s, quote_at, quote) else {
            return false;
        };
        self.i = end;
        true
    }

    fn consume_ellipsis(&mut self) -> bool {
        self.skip_ws();
        if self.rest().starts_with("...") {
            self.i += 3;
            true
        } else {
            false
        }
    }

    fn consume_number(&mut self) -> bool {
        self.skip_ws();
        let bytes = self.rest().as_bytes();
        if bytes.is_empty() {
            return false;
        }
        let mut n = 0;
        if bytes[0] == b'0' && matches!(bytes.get(1), Some(b'x' | b'X' | b'b' | b'B' | b'o' | b'O'))
        {
            let kind = bytes[1];
            n = 2;
            let mut digits = 0;
            while n < bytes.len() {
                let b = bytes[n];
                if b == b'_' {
                    if !bytes.get(n + 1).is_some_and(|c| number_digit(*c, kind)) {
                        break;
                    }
                    n += 1;
                    continue;
                }
                if !number_digit(b, kind) {
                    break;
                }
                digits += 1;
                n += 1;
            }
            if digits == 0 {
                return false;
            }
        } else if bytes[0] == b'.' || bytes[0].is_ascii_digit() {
            if bytes[0] == b'.' {
                if !bytes.get(1).is_some_and(|b| b.is_ascii_digit()) {
                    return false;
                }
                n = 1;
            }
            while n < bytes.len() {
                let b = bytes[n];
                if b == b'_' {
                    if !bytes.get(n + 1).is_some_and(|c| c.is_ascii_digit()) {
                        break;
                    }
                    n += 1;
                    continue;
                }
                if b.is_ascii_digit() {
                    n += 1;
                    continue;
                }
                break;
            }
            if bytes.get(n) == Some(&b'.') && bytes.get(n + 1) != Some(&b'.') {
                n += 1;
                while n < bytes.len() {
                    let b = bytes[n];
                    if b == b'_' {
                        if !bytes.get(n + 1).is_some_and(|c| c.is_ascii_digit()) {
                            break;
                        }
                        n += 1;
                        continue;
                    }
                    if b.is_ascii_digit() {
                        n += 1;
                        continue;
                    }
                    break;
                }
            }
            if matches!(bytes.get(n), Some(b'e' | b'E')) {
                let mut k = n + 1;
                if matches!(bytes.get(k), Some(b'+' | b'-')) {
                    k += 1;
                }
                let exp_at = k;
                while k < bytes.len() && (bytes[k].is_ascii_digit() || bytes[k] == b'_') {
                    k += 1;
                }
                if k > exp_at && bytes[k - 1].is_ascii_digit() {
                    n = k;
                }
            }
            if n == 0 {
                return false;
            }
            if matches!(bytes.get(n), Some(b'j' | b'J')) {
                n += 1;
            }
        } else {
            return false;
        }
        self.i += n;
        true
    }

    fn consume_name(&mut self) -> bool {
        self.skip_ws();
        const WORDS: &[&str] = &[
            "and", "or", "not", "in", "is", "if", "else", "elif", "lambda", "await", "yield",
            "for", "from", "as",
        ];
        for word in WORDS {
            if self.peek_keyword(word) {
                return false;
            }
        }
        let rest = self.rest();
        let mut chars = rest.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        if !(first.is_ascii_alphabetic()
            || first == '_'
            || (!first.is_ascii() && first.is_alphabetic()))
        {
            return false;
        }
        let mut n = first.len_utf8();
        for ch in chars {
            let ok = ch.is_ascii_alphanumeric()
                || ch == '_'
                || (!ch.is_ascii()
                    && (ch.is_alphanumeric() || ch == '\u{200c}' || ch == '\u{200d}'));
            if !ok {
                break;
            }
            n += ch.len_utf8();
        }
        self.i += n;
        true
    }
}

fn number_digit(b: u8, kind: u8) -> bool {
    match kind {
        b'x' | b'X' => b.is_ascii_hexdigit(),
        b'b' | b'B' => b == b'0' || b == b'1',
        _ => (b'0'..=b'7').contains(&b),
    }
}

fn closed_string_end(text: &str, quote_at: usize, quote: &str) -> Option<usize> {
    let end = skip_python_string(text, quote_at, quote);
    if end < quote_at + quote.len() || !text[..end].ends_with(quote) {
        return None;
    }
    if quote.len() == 1 {
        return Some(end);
    }
    if end == text.len() && !text[quote_at + quote.len()..].contains(quote) {
        return None;
    }
    Some(end)
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
    let commented = strip_hash_comments(body);
    merge_sorted_body_has_return(&commented)
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
            let rest_at = idx + needle.len() + colon_abs + 1;
            if rest_has_return_line(text, rest, rest_at) {
                return true;
            }
        }
        search = idx + needle.len();
    }
    false
}

/// `target` is outside a Python string. A `return` on its own line
/// inside a docstring is not a statement.
fn line_outside_string(text: &str, target: usize) -> bool {
    let mut i = 0;
    while i < target && i < text.len() {
        if let Some((quote_at, quote, _)) = string_literal_at(text, i) {
            if quote_at >= target {
                return true;
            }
            if quote_at > i {
                i = quote_at;
            }
            let end = skip_python_string(text, quote_at, quote);
            if quote_at < target && end > target {
                return false;
            }
            if end <= i {
                let Some(ch) = text[i..].chars().next() else {
                    break;
                };
                i += ch.len_utf8();
            } else {
                i = end;
            }
            continue;
        }
        let Some(ch) = text[i..].chars().next() else {
            break;
        };
        i += ch.len_utf8();
    }
    true
}

fn line_has_return_token(line: &str) -> bool {
    let t = line.trim_start();
    // `return[a, b]` and `return-a` are the same statement as the spaced form.
    line_starts_return(t) && looks_like_same_line_body(t)
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
    async fn code_syntax_augmented_assignment_is_incomplete() {
        for body in ["a += b", "out *= 2", "a //= b"] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "an augmented assignment is a statement with no return: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{body:?}");
            assert_eq!(
                result.details, "Function present but incomplete: return=false, ellipsis=false",
                "{body:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_augmented_assignment_then_return_is_strong() {
        let code = "def merge_sorted(a, b):\n    a += b\n    return a\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a return after an augmented assignment stays Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_method_call_and_attribute_assign_are_incomplete() {
        for body in [
            "a.extend(b)",
            "a.b.extend(c)",
            "a .extend(b)",
            "a.items + b",
            "a[0] + b",
            "a.extend(b) # in place",
            "a.items = b",
            "a.items = \"b # c\"",
            "a[\"items\"] = b",
            "a[0] = b",
            "a[0:1] = b",
            "a.items += b",
            "a[0] += 1",
            "a[i].append(b)",
            "(a.items) = b",
            "a.x, b[0] = b, a",
            "a.items = b.items = []",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "a method call or attribute assignment is a body: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{body:?}");
            assert_eq!(
                result.details, "Function present but incomplete: return=false, ellipsis=false",
                "{body:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_wrapped_method_call_is_incomplete() {
        let code = "def merge_sorted(a, b):\n    a.extend(\n        b\n    )\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.5,
            "a wrapped method call is still a function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(
            result.details, "Function present but incomplete: return=false, ellipsis=false",
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_method_call_then_return_is_strong() {
        for code in [
            "def merge_sorted(a, b):\n    a.extend(b)\n    return a\n",
            "def merge_sorted(a, b):\n    a.items = b\n    return a\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a return after a method call or attribute assignment is Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_dotted_prose_is_not_a_statement() {
        for body in [
            "a.extend the lists",
            "a.extend(the lists)",
            "a.items = the two lists",
            "a.items + = b",
            "a.",
            "a.items = b +",
            "\"a.extend(b)\"",
            "# a.extend(b)",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.0,
                "dotted prose is not a function body: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{body:?}");
            assert_eq!(
                result.details, "No recognizable function definition in response",
                "{body:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_remaining_statements_are_incomplete() {
        for code in [
            "def merge_sorted(a, b):\n    raise ValueError(\"no\")\n",
            "def merge_sorted(a, b):\n    yield a\n",
            "def merge_sorted(a, b):\n    yield from a\n",
            "def merge_sorted(a, b):\n    assert a\n",
            "def merge_sorted(a, b):\n    del a\n",
            "def merge_sorted(a, b):\n    global a\n",
            "def merge_sorted(a, b):\n    while a:\n        break\n",
            "def merge_sorted(a, b):\n    (a := a + b)\n",
            "def merge_sorted(a, b):\n    continue\n",
            "def merge_sorted(a, b):\n    nonlocal a\n",
            "def merge_sorted(a, b):\n    import json\n",
            "def merge_sorted(a, b):\n    from json import dumps\n",
            "def merge_sorted(a, b):\n    break\n",
            "def merge_sorted(a, b):\n    x: int = 1\n",
            "def merge_sorted(a, b):\n    (lambda x: x)(a)\n",
            "async def merge_sorted(a, b):\n    await a\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "a real statement with no return is incomplete: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{code:?}");
            assert_eq!(
                result.details, "Function present but incomplete: return=false, ellipsis=false",
                "{code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_pass_suite_with_balanced_delimiters_is_incomplete() {
        for code in [
            "def merge_sorted(a, b):\n    for x in a:\n        pass\n",
            "def merge_sorted(a, b):\n    with a:\n        pass\n",
            "def merge_sorted(a, b):\n    try:\n        pass\n",
            "def merge_sorted(a, b):\n    class Box:\n        pass\n",
            "def merge_sorted(a, b):\n    pass\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "pass with balanced delimiters is incomplete: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{code:?}");
            assert_eq!(
                result.details, "Function present but incomplete: return=false, ellipsis=false",
                "{code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_pass_with_unbalanced_delimiter_stays_syntax_issue() {
        let code = "def merge_sorted(a, b):\n    pass\n    result = (a\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(result.score, 0.5, "{result:?}");
        assert!(
            result.details.contains("syntax issues") && result.details.contains("parens=false"),
            "an unmatched parenthesis stays a syntax issue: {result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_spaced_augassign_is_not_a_statement() {
        let code = "def merge_sorted(a, b):\n    a + = b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(result.score, 0.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(
            result.details, "No recognizable function definition in response",
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn code_syntax_nested_return_is_not_strong() {
        for code in [
            "def merge_sorted(a, b):\n    def helper(x):\n        return x\n",
            "def merge_sorted(a, b):\n    class Box:\n        def get(self):\n            return a\n",
            "def merge_sorted(a, b):\n    def helper(\n        x,\n    ):\n        return x\n",
            "def merge_sorted(a, b):\n    if a:\n        def helper(x):\n            return x\n",
            "def merge_sorted(a, b):\n    async def helper(x):\n        return x\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "a return inside a nested def or class is not merge_sorted's return: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{code:?}");
            assert!(
                result.details.contains("return=false"),
                "{code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_nested_helper_with_outer_return_is_strong() {
        let code = "\
def merge_sorted(a, b):
    def helper(x):
        return x
    return a + b
";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a return on merge_sorted stays Strong when a helper also returns: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_if_return_stays_strong() {
        let code = "def merge_sorted(a, b):\n    if a:\n        return a\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a return inside if still belongs to merge_sorted: {result:?}"
        );
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
    async fn code_syntax_english_punctuation_return_is_not_strong() {
        for body in [
            "return a single sorted list, please",
            "return a pre-sorted list",
            "return a single list (sorted)",
            "return a single list: sorted",
            "return a single list!",
            "return a single list; please",
            "return \"a sorted list\" please",
            "return (a + b) please",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.0,
                "English return punctuation is not a body: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{body:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_truncated_return_is_not_strong() {
        for body in [
            "return a +",
            "return a -",
            "return a *",
            "return a /",
            "return a //",
            "return a %",
            "return a |",
            "return a &",
            "return a ^",
            "return a @",
            "return a <<",
            "return a >>",
            "return a **",
            "return a ==",
            "return a !=",
            "return a <=",
            "return a <",
            "return a or",
            "return a and",
            "return not",
            "return lambda:",
            "return await",
            "return a.",
            "return a[",
            "return a if b",
            "return *a",
            "return (*a)",
            "return {*a}",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert!(
                result.score < 1.0,
                "truncated return must stay below Strong: {body:?} {result:?}"
            );
            assert_ne!(result.level, CapabilityLevel::Strong, "{body:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_truncated_return_after_statement_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    out = a\n    return out +\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert!(
            result.score < 1.0,
            "a real statement does not finish a dangling operator: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_wrapped_return_is_strong() {
        for code in [
            "def merge_sorted(a, b):\n    return (\n        a + b\n    )\n",
            "def merge_sorted(a, b):\n    return sorted(\n        a + b\n    )\n",
            "def merge_sorted(a, b):\n    return [\n        a,\n        b,\n    ]\n",
            "def merge_sorted(a, b):\n    out = a\n    return (\n        out + b\n    )\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a return joined inside brackets must stay Strong: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_wrapped_english_return_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    return (\n        a single list, please\n    )\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "joined English inside parens is still not a body: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn code_syntax_finished_expression_return_is_strong() {
        for body in [
            "return a + b",
            "return (a + b)",
            "return a or b",
            "return a if b else a",
            "return a,",
            "return a, b",
            "return (a,)",
            "return a.b",
            "return -a",
            "return a[0]",
            "return a[1:2]",
            "return a[:]",
            "return [a, b]",
            "return {\"a\": b}",
            "return f(a, b)",
            "return a // b",
            "return a ** b",
            "return a << 1",
            "return (sorted)",
            "return pre-sorted",
            "return a != b",
            "return a is not b",
            "return a not in b",
            "return await a",
            "return lambda x: x",
            "return [x for x in a]",
            "return a @ b",
            "return a + b;",
            "return a;",
            "return (*a, *b)",
            "return *a, b",
            "return {*a, *b}",
            "return 0x10",
            "return 1_000",
            "return 2j",
            "return café",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "finished return expression must stay Strong: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{body:?}");
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
    async fn code_syntax_column_zero_comment_before_return_is_strong() {
        let code = "def merge_sorted(a, b):\n# two-pointer merge\n    return a + b\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a column-0 comment is not the end of the function: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_tight_return_spelling_is_strong() {
        for body in [
            "return[a, b]",
            "return{\"a\": b}",
            "return*a, b",
            "return-a",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a return with no space after the keyword is still a return: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{body:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_backslash_return_is_strong() {
        for code in [
            "def merge_sorted(a, b):\n    return \\\n        sorted(a + b)\n",
            "def merge_sorted(a, b):\n    return \\\n        a + b\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a backslash continues the return: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{code:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_parenthesized_comprehension_target_is_strong() {
        for body in [
            "return [x for (x, y) in zip(a, b)]",
            "return [x for [x, y] in zip(a, b)]",
            "return [x for x, y in zip(a, b)]",
        ] {
            let code = format!("def merge_sorted(a, b):\n    {body}\n");
            let result = probe_code_syntax(&MockLlm {
                response: text_response(&code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 1.0,
                "a comprehension target in parentheses is finished code: {body:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{body:?}");
        }
    }

    #[tokio::test]
    async fn code_syntax_positional_after_keyword_is_not_strong() {
        let code = "def merge_sorted(a, b):\n    return sorted(reverse=True, a + b)\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert!(
            result.score < 1.0,
            "a positional after a keyword is not a finished call: {result:?}"
        );
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_keyword_after_positional_stays_strong() {
        let code = "def merge_sorted(a, b):\n    return sorted(a + b, reverse=True)\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 1.0,
            "a keyword after a positional stays Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn code_syntax_ellipsis_stub_is_incomplete() {
        for code in [
            "def merge_sorted(a, b):\n    ...\n",
            "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists.\"\"\"\n    ...\n",
            "def merge_sorted(a, b): ...\n",
        ] {
            let result = probe_code_syntax(&MockLlm {
                response: text_response(code),
            })
            .await
            .unwrap();
            assert_eq!(
                result.score, 0.5,
                "an ellipsis stub is a function, not a missing one: {code:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Medium, "{code:?} {result:?}");
            assert!(
                !result.details.contains("No recognizable"),
                "{code:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn code_syntax_ellipsis_only_inside_docstring_is_not_a_function() {
        let code = "def merge_sorted(a, b):\n    \"\"\"\n    ...\n    \"\"\"\n";
        let result = probe_code_syntax(&MockLlm {
            response: text_response(code),
        })
        .await
        .unwrap();
        assert_eq!(
            result.score, 0.0,
            "an ellipsis inside a docstring is not a body: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
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
