//! Cline `ModelInfo` JSON (user-side paste, not a Cline PR).
//!
//! Keys match `ModelInfo` in cline/cline (`contextWindow`, `supportsImages`,
//! `maxTokens`, `supportsPromptCache`). Do not emit fields Cline does not
//! declare.

use serde::{Deserialize, Serialize};

use super::{OverlayFiles, overlay_context_tokens};
use crate::types::CapabilityProfile;

/// Cline `ModelInfo` subset a user can attach to an OpenAI-compatible model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClineModelInfo {
    /// Measured window. Cline's default 128000 is the #13457 lie.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Same measured window as `contextWindow` (`maxInputTokens` sibling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Vision probe Medium or higher.
    pub supports_images: bool,
    /// canact does not measure prompt cache. Always false.
    pub supports_prompt_cache: bool,
}

impl ClineModelInfo {
    /// Build a Cline `ModelInfo` from a probed profile.
    pub fn from_profile(profile: &CapabilityProfile, advertised: Option<u32>) -> Self {
        let ctx = overlay_context_tokens(profile, advertised);
        Self {
            context_window: ctx,
            max_tokens: ctx,
            supports_images: profile.supports_vision(),
            supports_prompt_cache: false,
        }
    }

    /// Pretty JSON object.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_owned())
    }

    pub(crate) fn files(&self) -> Vec<OverlayFiles> {
        vec![OverlayFiles {
            name: "cline.modelinfo.json",
            body: format!("{}\n", self.to_json()),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::tests::sample_profile;
    use crate::types::CapabilityLevel;

    #[test]
    fn cline_export_uses_measured_window_not_128k() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let info = ClineModelInfo::from_profile(&p, Some(128_000));
        assert_eq!(info.context_window, Some(8192));
        assert_eq!(info.max_tokens, Some(8192));
        assert!(!info.supports_images);
        assert!(!info.supports_prompt_cache);
    }

    #[test]
    fn cline_export_sets_supports_images_from_vision() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Strong,
        );
        let info = ClineModelInfo::from_profile(&p, None);
        assert!(info.supports_images);
    }

    #[test]
    fn cline_keys_are_model_info_fields() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let value = serde_json::to_value(ClineModelInfo::from_profile(&p, None)).expect("json");
        for key in value.as_object().expect("object").keys() {
            assert!(
                CLINE_MODEL_INFO_FIELDS.contains(&key.as_str()),
                "unknown Cline ModelInfo field {key}"
            );
        }
    }

    /// Pinned from cline/cline `ModelInfo` (contextWindow / maxTokens /
    /// supportsImages / supportsPromptCache). Extra official keys exist;
    /// we only emit this subset.
    const CLINE_MODEL_INFO_FIELDS: &[&str] = &[
        "maxTokens",
        "contextWindow",
        "supportsImages",
        "supportsPromptCache",
        "inputPrice",
        "outputPrice",
        "cacheWritesPrice",
        "cacheReadsPrice",
        "description",
    ];
}
