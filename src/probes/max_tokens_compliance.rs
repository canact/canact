//! Max-tokens compliance probe.
//!
//! Tests whether the model honors `max_tokens` limits. Models that ignore
//! the limit cause cost blowups and context overflow.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::user_text;

/// Probe whether the model respects `max_tokens` output limits.
///
/// Asks for a long numbered list with `max_tokens=40`, then checks whether
/// the response stayed under a generous character budget. The budget is
/// generous (800 chars) because tokens != chars, but a model that produces
/// 5K+ chars with `max_tokens=40` is clearly not honoring the limit.
///
/// Scoring:
/// - `1.0` - response <= 400 characters (model respected the limit)
/// - `0.5` - response 401-800 characters (borderline)
/// - `0.0` - response > 800 characters (limit ignored)
pub async fn probe_max_tokens_compliance<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![user_text("List the numbers 1 through 200, one per line.")],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(40),
    };

    let response = llm.chat(request).await?;
    let char_count = response.text.len();

    let (score, details) = if char_count <= 400 {
        (
            1.0,
            format!("Response {char_count} chars with max_tokens=40 (compliant)"),
        )
    } else if char_count <= 800 {
        (
            0.5,
            format!("Response {char_count} chars with max_tokens=40 (borderline)"),
        )
    } else {
        (
            0.0,
            format!("Response {char_count} chars with max_tokens=40 (limit ignored)"),
        )
    };

    Ok(ProbeResult {
        name: "max_tokens_compliance".to_string(),
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
    async fn compliant_short_response() {
        let llm = MockLlm {
            response: text_response("1\n2\n3\n4\n5\n6\n7\n8"),
        };
        let result = probe_max_tokens_compliance(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn non_compliant_huge_response() {
        let long = (1..=500)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let llm = MockLlm {
            response: text_response(&long),
        };
        let result = probe_max_tokens_compliance(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
    }
}
