//! Vision capability probe.

use crate::ProbeError;
use crate::client::{
    ProbeClient, ProbeContent, ProbeContentPart, ProbeMessage, ProbeRequest, ProbeRole,
};
use crate::types::{ProbeResult, classify};

use super::refuse_truncated_incomplete;

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
    let raw = text.to_lowercase();
    let folded = fold_format_marks(&raw);
    let stripped = strip_format_marks(&raw);
    let lower = folded.as_str();

    // Check if the model identified the text. Be careful not to match "black",
    // "blue", "blank", etc. - only match "BL" as a standalone word or "BLINE".
    let trimmed_lower = lower.trim().trim_matches(|c: char| c == '"' || c == '\'');
    let letter_tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let identified_text = trimmed_lower == "bl"
        || lower.contains("bline")
        || lower.contains("\"bl\"")
        || lower.contains("'bl'")
        || letter_tokens.iter().any(|word| *word == "bl")
        || letter_tokens.as_slice() == ["b", "l"];

    // User-facing details only. Ground-truth text in the probe image stays
    // internal (see PROBE_IMAGE_BASE64 / scoring above); do not echo it or
    // the raw model reply in default details.
    let font_hedge = vision_font_hedge(&folded) || vision_font_hedge(&stripped);
    let refused = (vision_refused(&folded) || vision_refused(&stripped)) && !font_hedge;
    // Partial reads name glyphs. "no letters" is a refusal, not a read.
    // Do not gate on `refused`: "I see letters but cannot make them out"
    // is still Medium.
    let negated_glyphs = vision_negated_glyphs(&folded) || vision_negated_glyphs(&stripped);
    let make_out_negated = cannot_make_token_out(&folded) || cannot_make_token_out(&stripped);
    let saw_glyphs = !negated_glyphs
        && (has_surface_word(lower, "letter")
            || has_surface_word(lower, "letters")
            || has_surface_word(lower, "character")
            || has_surface_word(lower, "characters"));
    // Word tokens only. "whitespace" is not "white"; "context" is not "text".
    let processed_surface = has_surface_word(lower, "pattern")
        || has_surface_word(lower, "pixel")
        || has_surface_word(lower, "pixels")
        || has_surface_word(lower, "black")
        || has_surface_word(lower, "white")
        || has_surface_word(lower, "text");

    let echoed_question = lower.contains("what text or letters appear");

    // Strong is a read, not any "don't"/"cannot" hedge (font, typeface).
    // See-denial still blocks Strong via negated_glyphs ("I don't see BL").
    let (score, details) = if identified_text && !negated_glyphs && !make_out_negated {
        (1.0, "Can read text from images".to_string())
    } else if echoed_question {
        (0.0, "Did not use the image (generic reply)".to_string())
    } else if saw_glyphs {
        (
            0.5,
            "Processed the image but could not read the text clearly".to_string(),
        )
    } else if refused {
        (0.0, "Cannot process images".to_string())
    } else if processed_surface {
        (
            0.5,
            "Processed the image but could not read the text clearly".to_string(),
        )
    } else {
        (0.0, "Did not use the image (generic reply)".to_string())
    };

    refuse_truncated_incomplete(response.finish, score)?;
    Ok(ProbeResult {
        name: "vision".to_string(),
        score,
        max_score: 1.0,
        level: classify(score),
        details,
    })
}

fn fold_format_marks(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' => ' ',
            '\u{2019}' => '\'',
            _ => c,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_format_marks(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}'
            )
        })
        .map(|c| if c == '\u{2019}' { '\'' } else { c })
        .collect()
}

fn vision_refused(lower: &str) -> bool {
    lower.contains("cannot")
        || lower.contains("can't")
        || lower.contains("can not")
        || lower.contains("unable")
        || lower.contains("not able")
        || lower.contains("don't")
        || lower.contains("didn't")
        || lower.contains("did not")
        || lower.contains("couldn't")
        || lower.contains("could not")
        || lower.contains("no image")
        || lower.contains("no text")
        || lower.contains("no readable")
        || lower.contains("no visible")
        || lower.contains("no letter")
        || lower.contains("no character")
        || lower.contains("text-only")
        || lower.contains("text only")
        || lower.contains("text-based")
        || lower.contains("text model")
        || lower.contains("processes text")
        || lower.contains("process text")
        || lower.contains("work with text")
        || lower.contains("isn't any text")
        || lower.contains("is not any text")
        || lower.contains("black box")
        || lower.contains("white box")
        || lower.contains("white-box")
        || lower.contains("white space")
        || lower.contains("white-space")
        || lower.contains("no discernible text")
        || lower.contains("text-processing")
        || lower.contains("text processing")
}

fn vision_negated_glyphs(lower: &str) -> bool {
    lower.contains("no letter")
        || lower.contains("no character")
        || lower.contains("no readable")
        || lower.contains("no visible letter")
        || lower.contains("no visible character")
        || lower.contains("no discernible letter")
        || lower.contains("no discernible character")
        || lower.contains("doesn't contain character")
        || lower.contains("does not contain character")
        || lower.contains("aren't any character")
        || lower.contains("are not any character")
        || lower.contains("don't see")
        || lower.contains("do not see")
        || lower.contains("can't see")
        || lower.contains("cannot see")
        || lower.contains("can't read")
        || lower.contains("cannot read")
        || lower.contains("can't identify")
        || lower.contains("cannot identify")
        || recognize_negates_glyphs(lower)
        || lower.contains("can't decipher")
        || lower.contains("cannot decipher")
        || lower.contains("unable to read")
        || lower.contains("unable to see")
        || lower.contains("unable to identify")
        || lower.contains("unable to decipher")
        || lower.contains("not able to read")
        || lower.contains("not able to see")
        || lower.contains("not able to identify")
        || lower.contains("not able to decipher")
        || lower.contains("can not read")
        || lower.contains("can not see")
        || lower.contains("can not identify")
        || lower.contains("can not decipher")
        || lower.contains("couldn't read")
        || lower.contains("couldn't see")
        || lower.contains("couldn't identify")
        || lower.contains("could not read")
        || lower.contains("could not see")
        || lower.contains("could not identify")
        || lower.contains("didn't identify")
        || lower.contains("did not identify")
        || lower.contains("couldn't decipher")
        || lower.contains("could not decipher")
        || lower.contains("didn't decipher")
        || lower.contains("did not decipher")
        || lower.contains("wasn't able to read")
        || lower.contains("wasn't able to see")
        || lower.contains("wasn't able to identify")
        || lower.contains("wasn't able to decipher")
        || lower.contains("didn't see")
        || lower.contains("didn't read")
        || lower.contains("did not see")
        || lower.contains("did not read")
        || lower.contains("doesn't contain letter")
        || lower.contains("does not contain letter")
        || lower.contains("aren't any letter")
        || lower.contains("are not any letter")
}

fn recognize_negates_glyphs(lower: &str) -> bool {
    const STEMS: &[&str] = &[
        "don't recognize",
        "do not recognize",
        "can't recognize",
        "cannot recognize",
        "can not recognize",
        "unable to recognize",
        "not able to recognize",
        "didn't recognize",
        "did not recognize",
        "couldn't recognize",
        "could not recognize",
        "don't recognise",
        "do not recognise",
        "can't recognise",
        "cannot recognise",
        "can not recognise",
        "unable to recognise",
        "not able to recognise",
        "didn't recognise",
        "did not recognise",
        "couldn't recognise",
        "could not recognise",
        "wasn't able to recognize",
        "wasn't able to recognise",
    ];
    for stem in STEMS {
        let mut search = 0;
        while let Some(rel) = lower.get(search..).and_then(|s| s.find(stem)) {
            let after = lower[search + rel + stem.len()..].trim_start();
            if rest_is_font_hedge(after) {
                search += rel + stem.len();
                continue;
            }
            return true;
        }
    }
    false
}

fn rest_is_font_hedge(after: &str) -> bool {
    let after = after
        .strip_prefix("the ")
        .or_else(|| after.strip_prefix("this "))
        .or_else(|| after.strip_prefix("that "))
        .or_else(|| after.strip_prefix("these "))
        .or_else(|| after.strip_prefix("those "))
        .unwrap_or(after);
    after.starts_with("font") || after.starts_with("typeface")
}

fn vision_font_hedge(lower: &str) -> bool {
    let has_font = has_surface_word(lower, "font")
        || has_surface_word(lower, "fonts")
        || has_surface_word(lower, "typeface")
        || has_surface_word(lower, "typefaces");
    if !has_font {
        return false;
    }
    lower.contains("don't")
        || lower.contains("do not")
        || lower.contains("can't")
        || lower.contains("cannot")
        || lower.contains("can not")
}

fn cannot_make_token_out(lower: &str) -> bool {
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    tokens.windows(2).any(|w| w[0] == "make" && w[1] == "out")
        || tokens.windows(3).any(|w| w[0] == "make" && w[2] == "out")
}

fn has_surface_word(lower: &str, word: &str) -> bool {
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|token| token == word)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::probes::test_support::*;
    use crate::types::CapabilityLevel;

    #[tokio::test]
    async fn vision_color_words_are_not_strong() {
        for text in [
            "I see: black and white pixels only",
            "The letters blend together",
            "There is no text black",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "color/no-read must not set supportsVision Strong: {text:?} -> {result:?}"
            );
        }
    }

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
    async fn vision_text_model_does_not_set_medium() {
        for body in [
            "I am a text model",
            "This model processes text",
            "This is a text-only model.",
            "I'm a text-based model.",
            "I only work with text.",
            "There isn't any text.",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "text-model phrasing must not set supportsVision: {body} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_text_only_model_does_not_set_medium() {
        let llm = MockLlm {
            response: text_response("This is a text-only model."),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "text-only must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn vision_negated_letters_do_not_set_medium() {
        for body in [
            "No visible letters",
            "There are no discernible letters",
            "The image doesn't contain letters",
            "There aren't any letters",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "negated letters must not set supportsVision: {body} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_black_box_idiom_does_not_set_medium() {
        let llm = MockLlm {
            response: text_response("This is a black box to me."),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "black box idiom must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn vision_question_echo_does_not_set_medium() {
        let llm = MockLlm {
            response: text_response("What text or letters appear in this image?"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "question echo must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn vision_weak_when_zwsp_splits_dont_see() {
        for body in [
            "I don\u{200B}'t see letters",
            "I can\u{200B}'t see any text",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "ZWSP in don't/can't must not set supportsVision: {body} {result:?}"
            );
            assert_eq!(result.score, 0.0, "body={body}");
        }
    }

    #[tokio::test]
    async fn vision_hedged_bl_is_strong() {
        let llm = MockLlm {
            response: text_response("BL (I don't know the font)"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.score, 1.0, "{result:?}");
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn vision_dont_recognize_the_font_with_bl_is_strong() {
        let llm = MockLlm {
            response: text_response("BL (I don't recognize the font)"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.score, 1.0,
            "font hedge that uses recognize must stay Strong: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Strong);
    }

    #[tokio::test]
    async fn vision_dont_see_bl_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I don't see BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.score, 0.0,
            "refusal that names BL must not be Strong"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn vision_cannot_read_bl_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I cannot read BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "cannot-read that names BL must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_cannot_identify_bl_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I cannot identify BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "cannot-identify that names BL must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_cannot_recognize_bl_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I cannot recognize BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "cannot-recognize that names BL must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_dont_recognize_bl_is_not_strong() {
        for text in [
            "I don't recognize BL",
            "I do not recognize BL",
            "I cannot recognise BL",
            "I don\u{2019}t recognize BL",
            "I can\u{2019}t recognise BL",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "recognize/recognise refusal that names BL must not be Strong: {text:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_cannot_make_bl_out_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I cannot make BL out"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "cannot-make-BL-out must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_no_readable_text_or_letters_is_weak() {
        for text in ["no readable text", "no readable letters"] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "no-readable phrasing must not set supportsVision: {text:?} {result:?}"
            );
            assert_eq!(result.score, 0.0, "{text:?}");
        }
    }

    #[tokio::test]
    async fn vision_not_able_to_read_bl_is_not_strong() {
        let llm = MockLlm {
            response: text_response("I'm not able to read BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Strong,
            "not-able-to-read that names BL must not be Strong: {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_unable_to_read_bl_is_not_strong() {
        for text in [
            "I am unable to read BL",
            "I can not see BL",
            "I am unable to make out BL",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "unable/can-not refusal that names BL must not be Strong: {text:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_weak_when_dont_see_letters() {
        let llm = MockLlm {
            response: text_response("I don't see letters"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "don't-see-letters must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn vision_weak_when_no_letters_visible() {
        let llm = MockLlm {
            response: text_response("There are no letters visible"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "no-letters refusal must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn vision_weak_when_no_visible_text() {
        let llm = MockLlm {
            response: text_response("No visible text"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "no visible text must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn vision_weak_when_zwsp_splits_no_text() {
        let llm = MockLlm {
            response: text_response("There is no\u{200B} text visible"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "ZWSP in no-text refusal must not set supportsVision: {result:?}"
        );
        assert_eq!(result.score, 0.0);
    }

    #[tokio::test]
    async fn vision_weak_when_no_text_visible() {
        let llm = MockLlm {
            response: text_response("There is no text visible"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "no-text refusal must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
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
    async fn vision_weak_when_refusal_mentions_black_white() {
        let llm = MockLlm {
            response: text_response("I cannot process this black and white image"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.level, CapabilityLevel::Weak);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.details, "Cannot process images");
    }

    #[tokio::test]
    async fn vision_can_not_process_black_white_is_weak() {
        let llm = MockLlm {
            response: text_response("I can not process this black and white image"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Weak,
            "can-not process black-and-white must be a vision refusal: {result:?}"
        );
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
    async fn vision_letters_but_cannot_make_out_is_medium() {
        let llm = MockLlm {
            response: text_response("I see letters but cannot make out the text"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "letters plus cannot-make-out is a partial read: {result:?}"
        );
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn vision_could_not_read_bl_is_not_strong() {
        for text in ["I couldn't read BL", "I could not see BL"] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "could-not refusal that names BL must not be Strong: {text:?} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_did_not_see_letters_is_weak() {
        for text in ["I didn't see any letters", "I did not see any text"] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "did-not-see letters/text must be Weak: {text:?} {result:?}"
            );
            assert_eq!(result.score, 0.0, "{text:?}");
        }
    }

    #[tokio::test]
    async fn vision_dont_recognize_this_font_with_bl_is_strong() {
        for text in [
            "BL (I don't recognize this font)",
            "BL (I don't recognize that typeface)",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.score, 1.0,
                "this/that font hedge with BL must stay Strong: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{text:?}");
        }
    }

    #[tokio::test]
    async fn vision_dont_recognize_these_fonts_with_bl_is_strong() {
        for text in [
            "BL (I don't recognize these fonts)",
            "BL (I don't recognize those typefaces)",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.score, 1.0,
                "these/those font hedge with BL must stay Strong: {text:?} {result:?}"
            );
            assert_eq!(result.level, CapabilityLevel::Strong, "{text:?}");
        }
    }

    #[tokio::test]
    async fn vision_didnt_recognize_could_not_identify_is_not_strong() {
        for text in [
            "I didn't recognize BL",
            "I could not recognise BL",
            "I did not identify BL",
            "I couldn't decipher BL",
            "I wasn't able to read BL",
            "I wasn't able to recognize BL",
            "I wasn't able to recognise BL",
        ] {
            let llm = MockLlm {
                response: text_response(text),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_ne!(
                result.level,
                CapabilityLevel::Strong,
                "past-tense recognize/identify that names BL must not be Strong: {text:?} {result:?}"
            );
        }

        let was_not = probe_vision(&MockLlm {
            response: text_response("I was not able to read BL"),
        })
        .await
        .unwrap();
        assert_eq!(
            was_not.level,
            CapabilityLevel::Weak,
            "I was not able to read BL must stay Weak: {was_not:?}"
        );

        let make_out = probe_vision(&MockLlm {
            response: text_response("I see letters but cannot make them out"),
        })
        .await
        .unwrap();
        assert_eq!(
            make_out.level,
            CapabilityLevel::Medium,
            "cannot make them out must stay Medium: {make_out:?}"
        );

        let spaced = probe_vision(&MockLlm {
            response: text_response("B L"),
        })
        .await
        .unwrap();
        assert_eq!(
            spaced.level,
            CapabilityLevel::Strong,
            "spaced B L must stay a read: {spaced:?}"
        );
    }

    #[tokio::test]
    async fn vision_white_text_dont_know_font_is_medium() {
        let llm = MockLlm {
            response: text_response("I see white text but I don't know the font"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(
            result.level,
            CapabilityLevel::Medium,
            "processed surface plus font hedge must be Medium, not Weak: {result:?}"
        );
        assert_eq!(result.score, 0.5);
    }

    #[tokio::test]
    async fn vision_whitespace_word_does_not_set_medium() {
        let llm = MockLlm {
            response: text_response("Looks like whitespace only"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "whitespace must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
    }

    #[tokio::test]
    async fn vision_no_discernible_characters_do_not_set_medium() {
        for body in [
            "No discernible characters",
            "There aren't any characters",
            "The image doesn't contain characters",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "negated characters must not set supportsVision: {body} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_no_visible_characters_and_white_hyphen_do_not_set_medium() {
        for body in [
            "No visible characters",
            "There is no discernible text",
            "Looks like white-space only",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "blank-image phrasing must not set supportsVision: {body} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_white_space_and_text_processing_do_not_set_medium() {
        for body in [
            "Looks like white space only",
            "This is a white-box to me.",
            "This is a white box to me.",
            "I am a text-processing model",
            "This model does text processing",
        ] {
            let llm = MockLlm {
                response: text_response(body),
            };
            let result = probe_vision(&llm).await.unwrap();
            assert_eq!(
                result.level,
                CapabilityLevel::Weak,
                "blank-image or text-processing phrasing must not set supportsVision: {body} {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn vision_context_word_does_not_set_medium() {
        let llm = MockLlm {
            response: text_response("I need more context"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_ne!(
            result.level,
            CapabilityLevel::Medium,
            "context must not set supportsVision: {result:?}"
        );
        assert_eq!(result.level, CapabilityLevel::Weak);
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

    #[tokio::test]
    async fn vision_length_empty_is_transient() {
        let llm = MockLlm {
            response: length_text_response(""),
        };
        let result = probe_vision(&llm).await;
        assert!(
            matches!(result, Err(ProbeError::Transient(_))),
            "Length plus empty vision text must be Transient, not 30-day Weak; got {result:?}"
        );
    }

    #[tokio::test]
    async fn vision_length_complete_bl_stays_strong() {
        let llm = MockLlm {
            response: length_text_response("BL"),
        };
        let result = probe_vision(&llm).await.unwrap();
        assert_eq!(result.score, 1.0);
        assert_eq!(result.level, CapabilityLevel::Strong);
    }
}
