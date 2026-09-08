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
    /// Catalog advertised window. Cline's default 128000 is the #13457 lie.
    /// Do not write the measured ladder floor here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Cline output-token budget. Omitted until a measured cap exists.
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
        Self {
            context_window: overlay_context_tokens(profile, advertised),
            max_tokens: profile.max_output_tokens,
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
    fn cline_export_uses_advertised_window_not_ladder_floor() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let info = ClineModelInfo::from_profile(&p, Some(128_000));
        assert_eq!(info.context_window, Some(128_000));
        assert_eq!(info.max_tokens, None);
        assert!(!info.supports_images);
        assert!(!info.supports_prompt_cache);
    }

    #[test]
    fn cline_export_omits_window_when_unadvertised() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let info = ClineModelInfo::from_profile(&p, None);
        assert_eq!(info.context_window, None);
        assert_eq!(info.max_tokens, None);
        let value = serde_json::to_value(&info).expect("json");
        assert!(value.get("contextWindow").is_none(), "{value}");
        assert!(value.get("maxTokens").is_none(), "{value}");
    }

    #[test]
    fn cline_export_writes_measured_output_cap() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.max_output_tokens = Some(4096);
        let info = ClineModelInfo::from_profile(&p, Some(200_000));
        assert_eq!(info.max_tokens, Some(4096));
        assert_eq!(info.context_window, Some(200_000));
        assert_ne!(info.max_tokens, info.context_window);
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
