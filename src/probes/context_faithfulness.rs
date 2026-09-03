//! Context faithfulness probe.
//!
//! Tests whether the model accurately recalls specific details from a
//! moderately long context. Models that fail this should have their
//! effective context window reduced and compact earlier.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeRequest};
use crate::types::{ProbeResult, classify};

use super::{refuse_truncated_incomplete, system_text, user_text};

/// Filler text to pad the context. Each block is ~200 tokens.
const FILLER_BLOCK: &str = "\
The distributed cache invalidation system uses a gossip-based protocol \
to propagate changes across nodes. When a key is updated on node A, the \
change is first written to the local write-ahead log, then broadcast to \
a randomly selected subset of peers (fanout=3). Each peer verifies the \
vector clock and either applies the update or initiates a conflict \
resolution round. The system achieves eventual consistency within 500ms \
under normal network conditions. Tombstones are retained for 72 hours \
before garbage collection. The maximum key size is 256 bytes and the \
maximum value size is 1MB. Connection pooling uses a min-idle of 2 and \
max-active of 16 per node. Health checks run every 10 seconds with a \
timeout of 3 seconds.";

/// Probe whether the model faithfully recalls details from context.
///
/// Embeds 3 specific facts (a port number, a version string, and a
/// timeout value) inside ~2K tokens of filler, then asks the model
/// to recall all three.
///
/// Scoring:
/// - `1.0` - all 3 facts recalled correctly
/// - `0.67` - 2 of 3 correct
/// - `0.33` - 1 of 3 correct
/// - `0.0` - 0 correct or refused to answer
pub async fn probe_context_faithfulness<C: ProbeClient>(
    llm: &C,
) -> Result<ProbeResult, ProbeError> {
    let context = format!(
        "{FILLER_BLOCK}\n\n\
         IMPORTANT CONFIGURATION:\n\
         - The monitoring dashboard runs on port 9847.\n\
         - The current deployment uses artifact version v3.7.42-rc1.\n\
         - The circuit breaker timeout is set to 1750 milliseconds.\n\n\
         {FILLER_BLOCK}\n\n\
         {FILLER_BLOCK}\n\n\
         {FILLER_BLOCK}"
    );

    let request = ProbeRequest {
        messages: vec![
            system_text(context),
            user_text(
                "Based on the configuration above, answer these three questions. \
                 Reply with ONLY the answers, one per line, no labels:\n\
                 1. What port does the monitoring dashboard run on?\n\
                 2. What is the current artifact version?\n\
                 3. What is the circuit breaker timeout in milliseconds?",
            ),
        ],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let lower = response.text.to_lowercase();

    let has_port = lower.contains("9847");
    let has_version = lower.contains("3.7.42") || lower.contains("v3.7.42-rc1");
    let has_timeout = recalls_timeout(&lower);

    let correct = u8::from(has_port) + u8::from(has_version) + u8::from(has_timeout);
    let score = f32::from(correct) / 3.0;

    let details = format!(
        "{correct}/3 facts recalled: port={has_port}, version={has_version}, timeout={has_timeout}"
    );

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "context_faithfulness".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

/// True when the reply restates the planted 1750 ms timeout.
///
/// Accepts `1750`, `1,750 ms`, `1.75s`, and `1.75 seconds` so a single
/// phrasing miss does not drop the score.
fn recalls_timeout(lower: &str) -> bool {
    if lower.contains("1750") || lower.contains("1,750") {
        return true;
    }
    let compact: String = lower
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ',')
        .collect();
    compact.contains("1750")
        || compact.contains("1.75s")
        || compact.contains("1.75sec")
        || (lower.contains("1.75") && (lower.contains("second") || lower.contains("timeout")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProbeError;
    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn length_partial_facts_is_transient() {
        let llm = MockLlm {
            response: length_text_response("9847\n"),
        };
        let err = probe_context_faithfulness(&llm)
            .await
            .expect_err("must refuse");
        assert!(
            matches!(&err, ProbeError::Transient(msg) if msg.contains("truncated")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn length_all_facts_stays_strong() {
        let llm = MockLlm {
            response: length_text_response("9847\nv3.7.42-rc1\n1750"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn context_strong_for_all_facts() {
        let llm = MockLlm {
            response: text_response("9847\nv3.7.42-rc1\n1750"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn context_medium_for_two_facts() {
        let llm = MockLlm {
            response: text_response("9847\nv3.7.42-rc1\n2000"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert!(result.score > 0.6 && result.score < 0.7);
    }

    #[tokio::test]
    async fn context_weak_for_no_facts() {
        let llm = MockLlm {
            response: text_response("I don't have access to that configuration."),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn context_partial_for_one_fact() {
        let llm = MockLlm {
            response: text_response("Port 9847\nUnknown\nNot sure"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert!(result.score > 0.3 && result.score < 0.4);
    }

    #[tokio::test]
    async fn context_synonym_timeout_1_75s_counts() {
        let llm = MockLlm {
            response: text_response("9847\nv3.7.42-rc1\n1.75s"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "1.75s is the same timeout fact as 1750 ms: {}",
            result.details
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn context_synonym_timeout_comma_ms_counts() {
        let llm = MockLlm {
            response: text_response("9847\n3.7.42\n1,750 ms"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "1,750 ms is the same timeout fact as 1750: {}",
            result.details
        );
    }

    #[tokio::test]
    async fn context_synonym_timeout_seconds_phrase_counts() {
        let llm = MockLlm {
            response: text_response("9847\nv3.7.42-rc1\n1.75 seconds"),
        };
        let result = probe_context_faithfulness(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "1.75 seconds is the same timeout fact as 1750 ms: {}",
            result.details
        );
    }

    #[test]
    fn recalls_timeout_accepts_equivalent_phrasings() {
        assert!(recalls_timeout("1750"));
        assert!(recalls_timeout("1,750 ms"));
        assert!(recalls_timeout("1.75s"));
        assert!(recalls_timeout("1.75 seconds"));
        assert!(recalls_timeout("timeout is 1.75 sec"));
        assert!(!recalls_timeout("2000"));
        assert!(!recalls_timeout("i don't know"));
    }
}
