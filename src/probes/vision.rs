//! Vision capability probe.

use crate::ProbeError;
use crate::client::{
    ProbeClient, ProbeContent, ProbeContentPart, ProbeMessage, ProbeRequest, ProbeRole,
};
use crate::types::{ProbeResult, classify};

/// Minimal 16x16 PNG with "BL" text pattern (white on black), base64-encoded.
/// Used as the test image for the vision capability probe.
const PROBE_IMAGE_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAIklEQVR4nGNgYGD4jwQYkAAaF7sELj\
     bFGoaQk3BJjYIhBwBFt1ykBOQgDQAAAABJRU5ErkJggg==";

/// Probe whether the model can process image input (vision capability).
///
/// Sends a small test image containing the text "BL" and asks the model
/// to describe what it sees. This validates the minimum-viable vision path
/// using inline base64 images.
///
/// Scoring:
/// - `1.0` - response mentions "BL" or "BLINE" (case-insensitive)
/// - `0.5` - response describes image content (letters, text, pattern)
///   but does not identify the exact text
/// - `0.0` - response indicates inability to process images, or gives
///   a generic answer ignoring the image
///
/// Note: vision probes are more expensive than text probes due to image
/// token costs. The cache prevents repeated probing.
pub async fn probe_vision<C: ProbeClient>(llm: &C) -> Result<ProbeResult, ProbeError> {
    let request = ProbeRequest {
        messages: vec![ProbeMessage {
            role: ProbeRole::User,
            content: ProbeContent::Parts(vec![
                ProbeContentPart::Text {
                    text: "What text or letters appear in this image? Reply with ONLY the text \
                           you see, nothing else."
                        .to_string(),
                },
                ProbeContentPart::ImageBase64 {
                    media_type: "image/png".to_string(),
                    data: PROBE_IMAGE_BASE64.to_string(),
                },
            ]),
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: vec![],
        model: llm.model_id().to_string(),
        temperature: Some(0.0),
        max_tokens: Some(256),
    };

    let response = llm.chat(request).await?;
    let text = response.text;
    let lower = text.to_lowercase();

    // Check if the model identified the text. Be careful not to match "black",
    // "blue", "blank", etc. - only match "BL" as a standalone word or "BLINE".
    let trimmed_lower = lower.trim().trim_matches(|c: char| c == '"' || c == '\'');
    let identified_text = trimmed_lower == "bl"
        || lower.contains("bline")
        || lower.contains("\"bl\"")
        || lower.contains("'bl'")
        || lower.contains("letters bl")
        || lower.contains("text bl")
        || lower.contains("says bl")
        || lower.contains("reads bl")
        || lower.contains("bl and")
        || lower.contains(": bl");

    // User-facing details only. Ground-truth text in the probe image stays
    // internal (see PROBE_IMAGE_BASE64 / scoring above); do not echo it or
    // the raw model reply in default details.
    let refused = lower.contains("cannot")
        || lower.contains("can't")
        || lower.contains("unable")
        || lower.contains("don't")
        || lower.contains("no image");
    // "text" alone is too common in refusals ("cannot see any text").
    let processed_content = lower.contains("letter")
        || lower.contains("character")
        || lower.contains("pattern")
        || lower.contains("pixel")
        || lower.contains("black")
        || lower.contains("white");

    let (score, details) = if identified_text {
        (1.0, "Can read text from images".to_string())
    } else if processed_content {
        (
            0.5,
            "Processed the image but could not read the text clearly".to_string(),
        )
    } else if refused {
        (0.0, "Cannot process images".to_string())
    } else if lower.contains("text") {
        (
            0.5,
            "Processed the image but could not read the text clearly".to_string(),
        )
    } else {
        (0.0, "Did not use the image (generic reply)".to_string())
    };

    Ok(ProbeResult {
        name: "vision".to_string(),
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
    async fn vision_strong_when_text_identified() {
        let llm = MockLlm {
            response: text_response("BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Strong);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.details, "Can read text from images");
    }

    #[tokio::test]
    async fn vision_medium_when_image_processed_but_wrong_text() {
        let llm = MockLlm {
            response: text_response("I see some white text on a black background"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
        assert_eq!(
            result.details,
            "Processed the image but could not read the text clearly"
        );
    }

    #[tokio::test]
    async fn vision_weak_when_cannot_process() {
        let llm = MockLlm {
            response: text_response("I cannot process images in this format."),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.details, "Cannot process images");
    }

    #[tokio::test]
    async fn vision_weak_when_refusal_mentions_text() {
        let llm = MockLlm {
            response: text_response("I cannot see any text in this image"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.details, "Cannot process images");
    }

    #[tokio::test]
    async fn vision_medium_when_partial_read_also_says_cannot() {
        let llm = MockLlm {
            response: text_response("I see dark letters but cannot make them out"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Medium);
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn vision_weak_for_generic_response() {
        let llm = MockLlm {
            response: text_response("Sure, I'd be happy to help you with that."),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.details, "Did not use the image (generic reply)");
    }
}
