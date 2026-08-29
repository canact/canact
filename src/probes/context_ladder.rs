//! Effective context-token ladder.
//!
//! Separate from `context_faithfulness`. Climbs 4k / 8k / 16k and writes
//! `CapabilityProfile.effective_context_tokens`. Stop on first fail.
//! Catalog `advertised_context_tokens` may cap the max rung; it is never
//! stored as the measured value without a passing live rung.
//! Mid-climb Transient/RateLimit keeps the last passing rung.

use crate::ProbeError;
use crate::client::{ProbeClient, ProbeContent, ProbeContentPart, ProbeMessage, ProbeRequest};

use super::{system_text, user_text};

/// Token rungs, smallest first.
const RUNGS: [u32; 3] = [4096, 8192, 16384];
const FIRST_RUNG: u32 = 4096;

/// Distinct from the context_faithfulness port / version / timeout set.
const FACT_WAREHOUSE: &str = "WH-4481";
const FACT_PROTOCOL: &str = "proto-9.2.11";
const FACT_HEARTBEAT: &str = "2840";

const FACTS: &str = "\
IMPORTANT CONTEXT LADDER MARKERS:\n\
- The inventory warehouse code is WH-4481.\n\
- The telemetry channel uses protocol proto-9.2.11.\n\
- The heartbeat interval is set to 2840 milliseconds.\n";

const QUESTIONS: &str = "\
Based on the markers above, answer these three questions. \
Reply with ONLY the answers, one per line, no labels:\n\
1. What is the inventory warehouse code?\n\
2. What telemetry protocol is in use?\n\
3. What is the heartbeat interval in milliseconds?";

const FILLER_BLOCK: &str = "\
River stage telemetry is collected at fifteen-minute intervals from \
staff gauges along the north fork. Each station stores a rolling window \
of raw samples, applies a median filter to drop spike noise, then \
forwards a compacted packet to the basin office. Operators compare \
the packet against seasonal rating curves before updating flood \
bulletins. Backup radios remain on standby when the microwave link \
drops below the fade margin. Crews inspect stilling wells after each \
ice-out and replace desiccant packs in the logger housing. The office \
archives daily summaries for later model calibration.";

/// Best passing rung plus any mid-climb transport or auth error.
///
/// A recall miss is `error: Ok(())` with `tokens` set to the last pass.
/// Auth stays in `error` so the suite can abort. Other errors keep
/// `tokens` and must refuse the 30-day cache.
#[derive(Debug)]
pub struct ContextLadder {
    /// Highest rung that recalled all markers.
    pub tokens: Option<u32>,
    /// `Ok` when the climb finished or stopped on a recall miss.
    pub error: Result<(), ProbeError>,
}

/// Climb the 4k/8k/16k ladder. `skip_expensive` tries 4k only.
///
/// Advertised tokens skip rungs strictly larger than the prior, except
/// the 4k rung is always attempted. A failed 4k rung is `None` even when
/// advertised is 128k. Transient/RateLimit after a pass keeps that pass.
pub async fn probe_effective_context_tokens<C: ProbeClient>(
    llm: &C,
    skip_expensive: bool,
) -> ContextLadder {
    let advertised = llm.catalog().advertised_context_tokens;
    let mut best: Option<u32> = None;

    for &rung in &RUNGS {
        if skip_expensive && rung > FIRST_RUNG {
            break;
        }
        // Advertised may cap larger rungs; 4k is still live-tested.
        if advertised.is_some_and(|n| rung > n && rung != FIRST_RUNG) {
            break;
        }

        let request = build_rung_request(llm.model_id(), rung);
        match llm.chat(request).await {
            Ok(response) => {
                if !recalls_all_facts(&response.text) {
                    return ContextLadder {
                        tokens: best,
                        error: Ok(()),
                    };
                }
                best = Some(rung);
            }
            Err(err) => {
                return ContextLadder {
                    tokens: best,
                    error: Err(err),
                };
            }
        }
    }

    ContextLadder {
        tokens: best,
        error: Ok(()),
    }
}

fn build_rung_request(model: &str, rung: u32) -> ProbeRequest {
    let mut filler = String::new();
    loop {
        filler.push_str(FILLER_BLOCK);
        filler.push('\n');
        let system = format!("{FACTS}\n{filler}");
        let request = ProbeRequest {
            messages: vec![system_text(system), user_text(QUESTIONS)],
            tools: vec![],
            model: model.to_string(),
            temperature: Some(0.0),
            max_tokens: Some(256),
        };
        if estimate_request_tokens(&request) >= rung {
            return request;
        }
    }
}

fn recalls_all_facts(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains(&FACT_WAREHOUSE.to_lowercase())
        && lower.contains(&FACT_PROTOCOL.to_lowercase())
        && lower.contains(FACT_HEARTBEAT)
}

fn estimate_tokens(chars: usize) -> u32 {
    u32::try_from((chars / 4).max(1)).unwrap_or(u32::MAX)
}

fn estimate_request_tokens(req: &ProbeRequest) -> u32 {
    estimate_tokens(request_char_count(req))
}

fn request_char_count(req: &ProbeRequest) -> usize {
    req.messages.iter().map(message_char_count).sum()
}

fn message_char_count(msg: &ProbeMessage) -> usize {
    match &msg.content {
        ProbeContent::Text(text) => text.chars().count(),
        ProbeContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ProbeContentPart::Text { text } => text.chars().count(),
                ProbeContentPart::ImageBase64 { .. } => 0,
            })
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{CatalogPriors, ProbeClient, ProbeFinish, ProbeResponse, ProbeStreamChunk};
    use crate::error::ProbeError;
    use futures::Stream;
    use std::future::Future;
    use std::sync::Mutex;

    struct LadderMock {
        advertised: Option<u32>,
        fail_at_or_above: Option<u32>,
        transient_at_or_above: Option<u32>,
        auth_at_or_above: Option<u32>,
        calls: Mutex<Vec<u32>>,
    }

    impl LadderMock {
        fn new(advertised: Option<u32>, fail_at_or_above: Option<u32>) -> Self {
            Self {
                advertised,
                fail_at_or_above,
                transient_at_or_above: None,
                auth_at_or_above: None,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn transient_at(mut self, tokens: u32) -> Self {
            self.transient_at_or_above = Some(tokens);
            self
        }

        fn auth_at(mut self, tokens: u32) -> Self {
            self.auth_at_or_above = Some(tokens);
            self
        }

        fn recorded(&self) -> Vec<u32> {
            self.calls.lock().expect("lock").clone()
        }
    }

    fn assert_is_4k(tokens: u32) {
        assert!(
            (4096..8192).contains(&tokens),
            "expected 4k rung, got {tokens}"
        );
    }

    fn assert_is_8k(tokens: u32) {
        assert!(
            (8192..16384).contains(&tokens),
            "expected 8k rung, got {tokens}"
        );
    }

    fn assert_is_16k(tokens: u32) {
        assert!(tokens >= 16384, "expected 16k rung, got {tokens}");
    }

    impl ProbeClient for LadderMock {
        fn chat(
            &self,
            req: ProbeRequest,
        ) -> impl Future<Output = Result<ProbeResponse, ProbeError>> + Send {
            let tokens = estimate_request_tokens(&req);
            self.calls.lock().expect("lock").push(tokens);
            let result = if self.auth_at_or_above.is_some_and(|t| tokens >= t) {
                Err(ProbeError::Auth("bad".into()))
            } else if self.transient_at_or_above.is_some_and(|t| tokens >= t) {
                Err(ProbeError::Transient("timeout".into()))
            } else {
                let fail = self
                    .fail_at_or_above
                    .is_some_and(|threshold| tokens >= threshold);
                let text = if fail {
                    "I cannot find those markers.".to_owned()
                } else {
                    format!("{FACT_WAREHOUSE}\n{FACT_PROTOCOL}\n{FACT_HEARTBEAT}")
                };
                Ok(ProbeResponse {
                    text,
                    tool_calls: Vec::new(),
                    finish: ProbeFinish::Stop,
                    usage: None,
                })
            };
            async move { result }
        }

        fn stream_chat(
            &self,
            _req: ProbeRequest,
        ) -> impl Stream<Item = Result<ProbeStreamChunk, ProbeError>> + Send {
            futures::stream::empty()
        }

        fn model_id(&self) -> &str {
            "test-model"
        }

        fn provider(&self) -> &str {
            "test-provider"
        }

        fn catalog(&self) -> CatalogPriors {
            CatalogPriors {
                advertised_context_tokens: self.advertised,
                ..CatalogPriors::default()
            }
        }
    }

    #[test]
    fn estimate_tokens_is_max_1_chars_div_4() {
        assert_eq!(estimate_tokens(0), 1);
        assert_eq!(estimate_tokens(3), 1);
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 1);
        assert_eq!(estimate_tokens(8), 2);
        assert_eq!(estimate_tokens(16384), 4096);
    }

    #[test]
    fn rung_request_meets_token_target() {
        let req = build_rung_request("test-model", 4096);
        assert_is_4k(estimate_request_tokens(&req));
    }

    #[test]
    fn ladder_facts_differ_from_faithfulness() {
        let blob = format!("{FACTS}{QUESTIONS}");
        assert!(!blob.contains("9847"));
        assert!(!blob.contains("3.7.42"));
        assert!(!blob.contains("1750"));
        assert!(!FILLER_BLOCK.contains(FACT_WAREHOUSE));
        assert!(!FILLER_BLOCK.contains(FACT_PROTOCOL));
        assert!(!FILLER_BLOCK.contains(FACT_HEARTBEAT));
    }

    #[tokio::test]
    async fn pass_4k_fail_8k_is_4096() {
        let llm = LadderMock::new(None, Some(8192));
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(4096));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 2);
        assert_is_4k(calls[0]);
        assert_is_8k(calls[1]);
    }

    #[tokio::test]
    async fn pass_4k_transient_8k_keeps_4096() {
        let llm = LadderMock::new(None, None).transient_at(8192);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(matches!(got.error, Err(ProbeError::Transient(_))));
        assert_eq!(got.tokens, Some(4096));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 2);
        assert_is_4k(calls[0]);
        assert_is_8k(calls[1]);
    }

    #[tokio::test]
    async fn pass_4k_auth_8k_keeps_4096_and_auth() {
        let llm = LadderMock::new(None, None).auth_at(8192);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(matches!(got.error, Err(ProbeError::Auth(_))));
        assert_eq!(got.tokens, Some(4096));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 2);
        assert_is_4k(calls[0]);
        assert_is_8k(calls[1]);
    }

    #[tokio::test]
    async fn pass_all_is_16384() {
        let llm = LadderMock::new(None, None);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(16384));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 3);
        assert_is_4k(calls[0]);
        assert_is_8k(calls[1]);
        assert_is_16k(calls[2]);
    }

    #[tokio::test]
    async fn transient_4k_is_none_with_err() {
        let llm = LadderMock::new(None, None).transient_at(4096);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(matches!(got.error, Err(ProbeError::Transient(_))));
        assert_eq!(got.tokens, None);
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1);
        assert_is_4k(calls[0]);
    }

    #[tokio::test]
    async fn fail_4k_is_none() {
        let llm = LadderMock::new(None, Some(4096));
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, None);
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1);
        assert_is_4k(calls[0]);
    }

    #[tokio::test]
    async fn advertised_4096_does_not_issue_8k_or_16k() {
        let llm = LadderMock::new(Some(4096), None);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(4096));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1, "must not climb past advertised 4096");
        assert_is_4k(calls[0]);
    }

    #[tokio::test]
    async fn advertised_8192_tries_4k_and_8k_not_16k() {
        let llm = LadderMock::new(Some(8192), None);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(8192));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 2);
        assert_is_4k(calls[0]);
        assert_is_8k(calls[1]);
        assert!(
            calls.iter().all(|&tokens| tokens < 16384),
            "must not issue a 16k rung, got {calls:?}"
        );
    }

    #[tokio::test]
    async fn advertised_128k_with_4k_fail_is_none_not_advertised() {
        let llm = LadderMock::new(Some(128000), Some(4096));
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, None, "must not copy advertised 128000");
        assert_ne!(got.tokens, Some(128000));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1);
        assert_is_4k(calls[0]);
    }

    #[tokio::test]
    async fn advertised_2000_still_tries_4k() {
        let llm = LadderMock::new(Some(2000), None);
        let got = probe_effective_context_tokens(&llm, false).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(4096));
        assert_ne!(got.tokens, Some(2000));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1);
        assert_is_4k(calls[0]);
    }

    #[tokio::test]
    async fn cheap_tries_4k_only() {
        let llm = LadderMock::new(None, None);
        let got = probe_effective_context_tokens(&llm, true).await;
        assert!(got.error.is_ok(), "{:?}", got.error);
        assert_eq!(got.tokens, Some(4096));
        let calls = llm.recorded();
        assert_eq!(calls.len(), 1);
        assert_is_4k(calls[0]);
    }
}
