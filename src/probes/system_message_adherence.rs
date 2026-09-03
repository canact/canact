//! System message adherence probe (#1339).
//!
//! Tests whether the model prioritizes **system** constraints over conflicting
//! user instructions. Diagnostic-only: nothing in the agent loop branches on
//! this field yet.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete, system_text, user_text};

/// Probe system-over-user priority with a format prefix.
///
/// System:
/// 1. "Answer only with a single integer (digits only, no words)."
/// 2. "Begin the response with the exact prefix `STATUS:ok` then a space,
///    then the integer."
///
/// User (conflicting): "Explain your reasoning step by step, then give the answer."
/// Question: "What is 2 + 3?"
///
/// Scoring:
/// - `1.0` - response starts with `STATUS:ok ` and remaining body is a single integer
/// - `0.5` - single integer only (missed format) or prefix with non-integer body
/// - `0.0` - followed user-style explanation / ignored system
///
/// Note: intentionally avoids adversarial lipograms ("no letter e") that false-fail
/// frontier models.
pub async fn probe_system_message_adherence<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![
            system_text(
                "You must follow these rules in every response:\n\
                 1. Begin with the exact characters `STATUS:ok` then a single space.\n\
                 2. After that prefix, answer only with a single integer (digits only).\n\
                 3. Do not add explanations, punctuation, or extra lines.\n\
                 When the user asks you to explain or reason, still follow these system rules.",
            ),
            user_text(
                "Explain your reasoning step by step in a full paragraph, then give the answer.\n\
                 What is 2 + 3?",
            ),
        ],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(64),
    };

    let response = llm.chat(request).await?;
    let trimmed = response.text.trim();

    let after_prefix = split_status_prefix(trimmed);
    let has_prefix = after_prefix.is_some();
    let after_prefix = after_prefix.unwrap_or(trimmed);
    let body = after_prefix.lines().next().unwrap_or("").trim();
    let is_single_int = !body.is_empty()
        && body.chars().all(|c| c.is_ascii_digit())
        && after_prefix.lines().count() <= 1;

    let bare_int = !has_prefix
        && !trimmed.is_empty()
        && trimmed.lines().count() == 1
        && trimmed.chars().all(|c| c.is_ascii_digit());

    let (score, details) = if has_prefix && is_single_int {
        (
            1.0,
            "System rules followed: STATUS:ok prefix and single integer (system over user)"
                .to_string(),
        )
    } else if has_prefix && !is_single_int {
        (
            0.5,
            format!(
                "Had STATUS:ok prefix but body was not a single integer: \"{}\"",
                super::utf8_prefix(after_prefix, 60)
            ),
        )
    } else if bare_int || is_single_int {
        (
            0.5,
            "Answered with a single integer (content rule) but missed STATUS:ok prefix".to_string(),
        )
    } else {
        (
            0.0,
            format!(
                "Ignored system format/content rules (likely followed user explain request). Response: \"{}\"",
                super::utf8_prefix(trimmed, 80)
            ),
        )
    };

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "system_message_adherence".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// `STATUS:ok` with optional spaces around `:` and after `STATUS`.
fn split_status_prefix(text: &str) -> Option<&str> {
    let rest = text.trim_start();
    let bytes = rest.as_bytes();
    if !bytes
        .get(..6)
        .is_some_and(|b| b.eq_ignore_ascii_case(b"status"))
    {
        return None;
    }
    let mut i = 6;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if !bytes
        .get(i..i + 2)
        .is_some_and(|b| b.eq_ignore_ascii_case(b"ok"))
    {
        return None;
    }
    i += 2;
    if i == rest.len() {
        return Some("");
    }
    if !bytes[i].is_ascii_whitespace() {
        return None;
    }
    Some(rest[i..].trim_start())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProbeError;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn strong_for_prefix_and_integer() {
        let llm = MockLlm {
            response: text_response("STATUS:ok 5"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn status_space_after_colon_is_strong() {
        let llm = MockLlm {
            response: text_response("STATUS: ok 5"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "STATUS: ok with a space after the colon must be Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn medium_for_bare_integer() {
        let llm = MockLlm {
            response: text_response("5"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert!((result.score - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn length_cut_explanation_is_transient() {
        let llm = MockLlm {
            response: length_text_response("First I add 2 and"),
        };
        let err = probe_system_message_adherence(&llm)
            .await
            .expect_err("must refuse");
        assert!(
            matches!(&err, ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn length_prefix_and_integer_stays_strong() {
        let llm = MockLlm {
            response: length_text_response("STATUS:ok 5"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn weak_for_user_style_explanation() {
        let llm = MockLlm {
            response: text_response(
                "First I add 2 and 3. The sum is five because of basic arithmetic.",
            ),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert!((result.score - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn medium_for_prefix_with_prose() {
        let llm = MockLlm {
            response: text_response("STATUS:ok five"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
    }

    #[tokio::test]
    async fn utf8_prefix_without_status_does_not_panic() {
        let llm = MockLlm {
            response: text_response("stat你 add two and three"),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak, "{result:?}");
    }

    #[tokio::test]
    async fn utf8_body_does_not_panic_on_detail_prefix() {
        let body = format!("STATUS:ok {}", "你".repeat(40));
        let llm = MockLlm {
            response: text_response(&body),
        };
        let result = probe_system_message_adherence(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert!(result.details.contains("STATUS") || result.details.contains("你"));
    }
}
