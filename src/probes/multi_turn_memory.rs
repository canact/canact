//! Multi-turn memory probe.
//!
//! Tests whether the model retains information across conversation turns.
//! Models that lose context after 2 turns break agentic loops where the
//! agent must remember file contents, tool results, and user requests.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{assistant_text, user_text};

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

    let (score, details) = if has_full {
        (1.0, "Full code recalled: ZEPHYR-4829".to_string())
    } else if has_partial {
        (
            0.5,
            "Partial recall (ZEPHYR or 4829 but not both)".to_string(),
        )
    } else {
        (0.0, "No recall of the secret code".to_string())
    };

    Ok(ProbeResult {
        name: "multi_turn_memory".to_string(),
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
    async fn weak_for_no_recall() {
        let llm = MockLlm {
            response: text_response("Paris"),
        };
        let result = probe_multi_turn_memory(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
    }
}
