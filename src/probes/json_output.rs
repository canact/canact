//! JSON output and instruction-following probes.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{extract_json_from_text, user_text};

/// Probe whether the model can produce structured JSON output.
///
/// Asks the model to analyse a word and return a JSON object with specific
/// fields: `word`, `length`, and `reversed`.
///
/// Scoring:
/// - `1.0` - valid JSON with all three fields and correct types
/// - Partial credit for valid JSON with fewer fields
/// - `0.0` - response is not valid JSON
pub async fn probe_json_output<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text(
            "Analyze the word \"hello\" and respond with ONLY a JSON object (no other text) \
             containing these fields: \"word\" (the word as a string), \"length\" (the number \
             of characters as a number), \"reversed\" (the word reversed as a string). \
             Example: {\"word\": \"cat\", \"length\": 3, \"reversed\": \"tac\"}",
        )],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let json_str = extract_json_from_text(&response.text);

    let (score, details) = match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(val) => {
            let has_word = val.get("word").and_then(|v| v.as_str()).is_some();
            let has_length = val.get("length").and_then(|v| v.as_u64()).is_some();
            let has_reversed = val.get("reversed").and_then(|v| v.as_str()).is_some();
            let field_count = u32::from(has_word) + u32::from(has_length) + u32::from(has_reversed);

            if field_count == 3 {
                (
                    1.0,
                    "Valid JSON with all required fields and correct types".to_string(),
                )
            } else {
                let partial = field_count as f32 / 6.0 + 0.1;
                (
                    partial,
                    format!("Valid JSON but only {field_count}/3 required fields present"),
                )
            }
        }
        Err(_) => (0.0, "Response was not valid JSON".to_string()),
    };

    Ok(ProbeResult {
        name: "json_output".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// Probe whether the model follows precise instructions.
///
/// Asks for a single-word answer and scores based on response verbosity.
///
/// Scoring:
/// - `1.0` - exactly one word
/// - `0.5` - short but multi-word (2-4 words)
/// - `0.0` - verbose (5+ words)
pub async fn probe_instruction_following<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text(
            "What is the capital of France? Reply with ONLY the single word answer, \
             no punctuation, no explanation.",
        )],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let trimmed = response.text.trim();
    let word_count = trimmed.split_whitespace().count();

    let (score, details) = if word_count == 1 {
        (1.0, format!("Single word response: \"{trimmed}\""))
    } else if word_count < 5 {
        (
            0.5,
            format!("Short but multi-word response ({word_count} words)"),
        )
    } else {
        (0.0, format!("Verbose response ({word_count} words)"))
    };

    Ok(ProbeResult {
        name: "instruction_following".to_string(),
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
    async fn json_output_strong_for_valid_json_all_fields() {
        let llm = MockLlm {
            response: text_response(r#"{"word": "hello", "length": 5, "reversed": "olleh"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn json_output_weak_for_invalid_json() {
        let llm = MockLlm {
            response: text_response("The word hello has 5 letters and reversed is olleh."),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn json_output_partial_credit_for_missing_fields() {
        let llm = MockLlm {
            response: text_response(r#"{"word": "hello"}"#),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert!(result.score > 0.0);
        assert!(result.score < 1.0);
        assert_ne!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn json_output_strong_for_markdown_wrapped_json() {
        let llm = MockLlm {
            response: text_response(
                "```json\n{\"word\": \"hello\", \"length\": 5, \"reversed\": \"olleh\"}\n```",
            ),
        };
        let result = probe_json_output(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn instruction_following_strong_for_single_word() {
        let llm = MockLlm {
            response: text_response("Paris"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
    }

    #[tokio::test]
    async fn instruction_following_weak_for_verbose() {
        let llm = MockLlm {
            response: text_response(
                "The capital of France is Paris, which is also the largest city in the country.",
            ),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn instruction_following_medium_for_short_multiword() {
        let llm = MockLlm {
            response: text_response("It is Paris"),
        };
        let result = probe_instruction_following(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }
}
