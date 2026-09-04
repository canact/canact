//! Multi-turn memory probe.
//!
//! Tests whether the model retains information across conversation turns.
//! Models that lose context after 2 turns break agentic loops where the
//! agent must remember file contents, tool results, and user requests.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{assistant_text, refuse_truncated_incomplete, user_text};

/// Probe whether the model retains facts across a 3-turn conversation.
///
/// Turn 1: User states a unique fact ("The secret code is ZEPHYR-4829").
/// Turn 2: User asks an unrelated question (distraction).
/// Turn 3: User asks the model to recall the secret code.
///
/// Uses 2 API calls (turns 2 and 3), but the information gain is high:
/// this directly predicts whether multi-turn agentic loops will work.
///
/// Scoring:
/// - `1.0` - exact code recalled ("ZEPHYR-4829" present in response)
/// - `0.5` - partial recall (contains "ZEPHYR" or "4829" but not the full code)
/// - `0.0` - no recall or refusal
pub async fn probe_multi_turn_memory<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let model = llm.model_id().to_string();

    let request_distraction = ProbeRequest {
        messages: vec![
            user_text(
                "Remember this secret code for later: ZEPHYR-4829. \
                 Just confirm you've noted it.",
            ),
            assistant_text("Got it, I've noted the secret code ZEPHYR-4829."),
            user_text("What is the chemical symbol for gold?"),
        ],
        tools: vec![],
        model: model.clone(),
        temperature: Some(0.0),
        max_tokens: Some(100),
    };

    let distraction_resp = llm.chat(request_distraction).await?;
    let distraction_text = distraction_resp.text;

    let request_recall = ProbeRequest {
        messages: vec![
            user_text(
                "Remember this secret code for later: ZEPHYR-4829. \
                 Just confirm you've noted it.",
            ),
            assistant_text("Got it, I've noted the secret code ZEPHYR-4829."),
            user_text("What is the chemical symbol for gold?"),
            assistant_text(distraction_text),
            user_text("What was the secret code I asked you to remember earlier?"),
        ],
        tools: vec![],
        model,
        temperature: Some(0.0),
        max_tokens: Some(100),
    };

    let recall_resp = llm.chat(request_recall).await?;
    let upper = recall_resp.text.to_uppercase();

    let has_full = upper.contains("ZEPHYR-4829");
    let has_partial = upper.contains("ZEPHYR") || upper.contains("4829");
    let refused = memory_refused(&recall_resp.text);

    let (score, details) = if refused {
        (0.0, "Refused to recall the secret code".to_string())
    } else if has_full {
        (1.0, "Full code recalled: ZEPHYR-4829".to_string())
    } else if has_partial {
        (
            0.5,
            "Partial recall (ZEPHYR or 4829 but not both)".to_string(),
        )
    } else {
        (0.0, "No recall of the secret code".to_string())
    };

    refuse_truncated_incomplete(recall_resp.finish, score)?;
    Ok(ProbeResult {
        name: "multi_turn_memory".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn memory_refused(text: &str) -> bool {
    let folded: String = text
        .chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}'
            )
        })
        .collect();
    let lower = folded.to_lowercase();
    lower.contains("don't remember")
        || lower.contains("do not remember")
        || lower.contains("can't remember")
        || lower.contains("cannot remember")
        || lower.contains("can not remember")
        || lower.contains("don't recall")
        || lower.contains("do not recall")
        || lower.contains("can't recall")
        || lower.contains("cannot recall")
        || lower.contains("can not recall")
        || lower.contains("can't repeat")
        || lower.contains("cannot repeat")
        || lower.contains("won't repeat")
        || lower.contains("will not repeat")
        || lower.contains("can't share")
        || lower.contains("cannot share")
        || lower.contains("can not share")
        || lower.contains("won't share")
        || lower.contains("will not share")
        || lower.contains("can't disclose")
        || lower.contains("cannot disclose")
        || lower.contains("can not disclose")
        || lower.contains("won't disclose")
        || lower.contains("will not disclose")
        || lower.contains("can't reveal")
        || lower.contains("cannot reveal")
        || lower.contains("can not reveal")
        || lower.contains("won't reveal")
        || lower.contains("will not reveal")
        || lower.contains("unable to remember")
        || lower.contains("unable to recall")
        || lower.contains("unable to share")
        || lower.contains("unable to disclose")
        || lower.contains("unable to reveal")
        || lower.contains("unable to repeat")
        || lower.contains("not able to remember")
        || lower.contains("not able to recall")
        || lower.contains("not able to share")
        || lower.contains("not able to disclose")
        || lower.contains("not able to reveal")
        || lower.contains("not able to repeat")
        || lower.contains("can not repeat")
        || lower.contains("not allowed to share")
        || lower.contains("not allowed to disclose")
        || lower.contains("not allowed to reveal")
        || lower.contains("couldn't remember")
        || lower.contains("couldn't recall")
        || lower.contains("could not remember")
        || lower.contains("could not recall")
        || lower.contains("i've forgotten")
        || lower.contains("i have forgotten")
        || lower.contains("i forgot")
        || lower.contains("i had forgotten")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProbeError;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn refusal_zwsp_dont_remember_is_weak() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            text_response("I don\u{200B}'t remember ZEPHYR-4829"),
        ]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(
            result.score, 0.0,
            "ZWSP in don't remember must be a refusal, not Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn refusal_that_quotes_code_is_weak() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            text_response("I don't remember a secret code ZEPHYR-4829"),
        ]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.score, 0.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn refusal_cannot_share_code_is_weak() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            text_response("I can't share ZEPHYR-4829"),
        ]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(
            result.score, 0.0,
            "share/disclose refusal that quotes the code must be Weak: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn refusal_can_not_remember_reveal_forgotten_quoted_code_is_weak() {
        for text in [
            "I can not remember ZEPHYR-4829",
            "I cannot reveal ZEPHYR-4829",
            "I've forgotten ZEPHYR-4829",
        ] {
            let llm = SequentialMock::new(vec![text_response("Au"), text_response(text)]);
            let result = probe_multi_turn_memory(&llm).await.unwrap();
            assert_eq!(
                result.score, 0.0,
                "refusal that quotes ZEPHYR-4829 must be Weak: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{text:?}");
        }
    }

    #[tokio::test]
    async fn refusal_unable_to_recall_quoted_code_is_weak() {
        for text in [
            "I am unable to recall the secret code ZEPHYR-4829",
            "I'm not allowed to share ZEPHYR-4829",
        ] {
            let llm = SequentialMock::new(vec![text_response("Au"), text_response(text)]);
            let result = probe_multi_turn_memory(&llm).await.unwrap();
            assert_eq!(
                result.score, 0.0,
                "unable/not-allowed refusal that quotes the code must be Weak: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{text:?}");
        }
    }

    #[tokio::test]
    async fn refusal_not_able_can_not_repeat_forgot_quoted_code_is_weak() {
        for text in [
            "I'm not able to recall ZEPHYR-4829",
            "I can not repeat ZEPHYR-4829",
            "I forgot ZEPHYR-4829",
            "I had forgotten ZEPHYR-4829",
        ] {
            let llm = SequentialMock::new(vec![text_response("Au"), text_response(text)]);
            let result = probe_multi_turn_memory(&llm).await.unwrap();
            assert_eq!(
                result.score, 0.0,
                "not-able/can-not-repeat/forgot that quotes the code must be Weak: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Weak, "{text:?}");
        }
    }

    #[tokio::test]
    async fn strong_for_full_recall() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            text_response("The secret code is ZEPHYR-4829."),
        ]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn medium_for_partial_recall() {
        let llm = SequentialMock::new(vec![text_response("Au"), text_response("ZEPHYR")]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn length_no_recall_is_transient() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            length_text_response("Let me recall what you told me earlier"),
        ]);
        let err = probe_multi_turn_memory(&llm)
            .await
            .expect_err("must refuse");
        assert!(
            matches!(&err, ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn length_full_recall_stays_strong() {
        let llm = SequentialMock::new(vec![
            text_response("Au"),
            length_text_response("The secret code is ZEPHYR-4829."),
        ]);
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn weak_for_no_recall() {
        let llm = MockLlm {
            response: text_response("Paris"),
        };
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
    }
}
