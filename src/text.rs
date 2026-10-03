//! Host-facing text helpers used by the OpenAI-compat adapter.

/// Remove case-insensitive `<think>` / `</think>` blocks, including an
/// unclosed open tag (drop through end of string).
///
/// Hosts that implement [`crate::ProbeClient`] around their own HTTP
/// client should run this on collected assistant text before grading.
/// Typed `thinking` / `reasoning` parts are already dropped; this
/// strips the inline tag form so chain-of-thought is not scored as
/// the answer. The stream fold path must keep the same strip.
#[must_use]
pub fn strip_think_blocks(input: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let lower = input.to_ascii_lowercase();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let Some(open_at) = lower.get(i..).and_then(|rest| rest.find(OPEN)) else {
            out.push_str(&input[i..]);
            break;
        };
        let open_at = i + open_at;
        out.push_str(&input[i..open_at]);
        let after_open = open_at + OPEN.len();
        match lower.get(after_open..).and_then(|rest| rest.find(CLOSE)) {
            Some(rel) => i = after_open + rel + CLOSE.len(),
            None => break,
        }
    }
    out
}

/// `Some` when `value` contains a newline, NUL, or other control character.
///
/// Model ids and provider labels are printed into cache-list rows and
/// Aider YAML. A control character splits those records.
#[must_use]
pub fn control_character_message(field: &str, value: &str) -> Option<String> {
    value
        .chars()
        .find(|c| c.is_control())
        .map(|_| format!("{field} must not contain a control character"))
}

#[cfg(test)]
mod tests {
    use super::{control_character_message, strip_think_blocks};

    #[test]
    fn strip_think_blocks_closed_pair() {
        assert_eq!(
            strip_think_blocks("<think>hidden chain</think>Paris"),
            "Paris"
        );
    }

    #[test]
    fn strip_think_blocks_unclosed_drops_through_end() {
        assert_eq!(strip_think_blocks("<think>partial"), "");
        assert_eq!(strip_think_blocks("keep<think>partial"), "keep");
    }

    #[test]
    fn strip_think_blocks_is_case_insensitive() {
        assert_eq!(strip_think_blocks("<THINK>hid</THINK>ok"), "ok");
        assert_eq!(strip_think_blocks("<Think>hid</think>ok"), "ok");
    }

    #[test]
    fn strip_think_blocks_multiple_and_plain() {
        assert_eq!(
            strip_think_blocks("<think>a</think>X<think>b</think>Y"),
            "XY"
        );
        assert_eq!(
            strip_think_blocks("Hello <think>hid</think>world"),
            "Hello world"
        );
        assert_eq!(strip_think_blocks("no tags"), "no tags");
        assert_eq!(strip_think_blocks(""), "");
    }

    #[test]
    fn control_character_message_rejects_newline_and_nul() {
        assert_eq!(control_character_message("model", "llama3.2:3b"), None);
        assert!(
            control_character_message("model", "llama\n3")
                .unwrap()
                .contains("model")
        );
        assert!(control_character_message("provider", "ollama\0").is_some());
        assert_eq!(control_character_message("model", "  gpt-4o  "), None);
    }
}
