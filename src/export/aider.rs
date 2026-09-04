//! Aider `.aider.model.settings.yml` and `.aider.model.metadata.json`.
//!
//! Field names match `ModelSettings` in Aider-AI/aider `aider/models.py`
//! (Apache-2.0, main as of 2026-09-03). Do not invent keys Aider will
//! reject in `ModelSettings(**dict)`.

use serde::Serialize;

use super::{OverlayFiles, aider_edit_format, overlay_context_tokens, overlay_model_name};
use crate::types::CapabilityProfile;

/// One Aider `ModelSettings` row (user overlay list item).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AiderSettingsRow {
    /// Model id Aider matches (`provider/model` or already-namespaced).
    pub name: String,
    /// Aider edit format token (`diff` / `udiff` / `whole` / `diff-fenced`).
    pub edit_format: String,
    /// Enable repo-map when the model can use tools.
    pub use_repo_map: bool,
}

/// LiteLLM-shaped metadata Aider loads from `--model-metadata-file`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AiderMetadataEntry {
    /// Measured (or min(advertised, measured)) input window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u32>,
    /// Same number as input when we have no separate output cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Provider label LiteLLM/Aider use for routing.
    pub litellm_provider: String,
    /// Chat completion mode.
    pub mode: String,
}

/// Pair of files an Aider user drops next to the repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiderOverlay {
    /// Settings list (one row).
    pub settings: Vec<AiderSettingsRow>,
    /// Metadata map keyed by the same `name` as the settings row.
    pub metadata: std::collections::BTreeMap<String, AiderMetadataEntry>,
}

impl AiderOverlay {
    /// Build settings + metadata from a probed profile.
    pub fn from_profile(profile: &CapabilityProfile, advertised: Option<u32>) -> Self {
        let name = overlay_model_name(profile);
        let ctx = overlay_context_tokens(profile, advertised);
        let row = AiderSettingsRow {
            name: name.clone(),
            edit_format: aider_edit_format(profile.best_edit_format()).to_owned(),
            use_repo_map: profile.can_use_tools(),
        };
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            name,
            AiderMetadataEntry {
                max_input_tokens: ctx,
                max_output_tokens: ctx,
                litellm_provider: super::normalize_overlay_provider(
                    &profile.provider.to_ascii_lowercase(),
                )
                .to_owned(),
                mode: "chat".to_owned(),
            },
        );
        Self {
            settings: vec![row],
            metadata,
        }
    }

    /// YAML list Aider loads with `yaml.safe_load` then `ModelSettings(**row)`.
    pub fn settings_yaml(&self) -> String {
        let mut out = String::new();
        for row in &self.settings {
            out.push_str("- name: ");
            out.push_str(&yaml_scalar(&row.name));
            out.push('\n');
            out.push_str("  edit_format: ");
            out.push_str(&yaml_scalar(&row.edit_format));
            out.push('\n');
            out.push_str("  use_repo_map: ");
            out.push_str(if row.use_repo_map { "true" } else { "false" });
            out.push('\n');
        }
        out
    }

    /// JSON object Aider loads as `local_model_metadata`.
    pub fn metadata_json(&self) -> String {
        serde_json::to_string_pretty(&self.metadata).unwrap_or_else(|_| "{}".to_owned())
    }

    pub(crate) fn files(&self) -> Vec<OverlayFiles> {
        vec![
            OverlayFiles {
                name: ".aider.model.settings.yml",
                body: self.settings_yaml(),
            },
            OverlayFiles {
                name: ".aider.model.metadata.json",
                body: format!("{}\n", self.metadata_json()),
            },
        ]
    }
}

fn yaml_scalar(s: &str) -> String {
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':'));
    if safe && !s.is_empty() {
        s.to_owned()
    } else {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::tests::sample_profile;
    use crate::types::CapabilityLevel;

    #[test]
    fn strong_search_replace_exports_aider_diff() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let overlay = AiderOverlay::from_profile(&p, Some(40960));
        let yaml = overlay.settings_yaml();
        assert!(yaml.contains("name: ollama/qwen2.5-coder"), "{yaml}");
        assert!(yaml.contains("edit_format: diff"), "{yaml}");
        assert!(yaml.contains("use_repo_map: true"), "{yaml}");
        let meta = overlay.metadata.get("ollama/qwen2.5-coder").expect("meta");
        assert_eq!(meta.max_input_tokens, Some(8192));
        assert_eq!(meta.litellm_provider, "ollama");
    }

    #[test]
    fn openrouter_host_provider_normalizes_litellm_id() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.provider = "openrouter.ai".to_owned();
        p.model_id = "anthropic/claude-3.5-sonnet".to_owned();
        let overlay = AiderOverlay::from_profile(&p, Some(8192));
        let name = "openrouter/anthropic/claude-3.5-sonnet";
        let meta = overlay.metadata.get(name).expect("meta");
        assert_eq!(meta.litellm_provider, "openrouter");
    }

    #[test]
    fn mixed_case_provider_normalizes_litellm_provider() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );

        p.provider = "OpenAI".to_owned();
        p.model_id = "gpt-4o".to_owned();
        let overlay = AiderOverlay::from_profile(&p, Some(8192));
        let meta = overlay.metadata.get("openai/gpt-4o").expect("openai meta");
        assert_eq!(
            meta.litellm_provider, "openai",
            "mixed-case OpenAI must match overlay_model_name family openai"
        );

        p.provider = "OpenRouter".to_owned();
        p.model_id = "anthropic/claude-3.5-sonnet".to_owned();
        let overlay = AiderOverlay::from_profile(&p, Some(8192));
        let meta = overlay
            .metadata
            .get("openrouter/anthropic/claude-3.5-sonnet")
            .expect("openrouter meta");
        assert_eq!(
            meta.litellm_provider, "openrouter",
            "mixed-case OpenRouter must match overlay_model_name family openrouter"
        );

        p.provider = "Localhost".to_owned();
        p.model_id = "qwen".to_owned();
        let overlay = AiderOverlay::from_profile(&p, Some(8192));
        let meta = overlay.metadata.get("ollama/qwen").expect("ollama meta");
        assert_eq!(
            meta.litellm_provider, "ollama",
            "mixed-case Localhost must match overlay_model_name family ollama"
        );
    }

    #[test]
    fn weak_edit_exports_aider_whole() {
        let p = sample_profile(
            CapabilityLevel::Weak,
            CapabilityLevel::Weak,
            CapabilityLevel::Weak,
        );
        let yaml = AiderOverlay::from_profile(&p, None).settings_yaml();
        assert!(yaml.contains("edit_format: whole"), "{yaml}");
    }

    #[test]
    fn medium_unified_exports_aider_udiff() {
        let p = sample_profile(
            CapabilityLevel::Medium,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let yaml = AiderOverlay::from_profile(&p, None).settings_yaml();
        assert!(yaml.contains("edit_format: udiff"), "{yaml}");
    }

    #[test]
    fn settings_keys_are_aider_model_settings_fields() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        let row = &AiderOverlay::from_profile(&p, None).settings[0];
        let value = serde_json::to_value(row).expect("row json");
        let keys: Vec<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        for key in &keys {
            assert!(
                AIDER_MODEL_SETTINGS_FIELDS.contains(key),
                "unknown Aider ModelSettings field {key}"
            );
        }
    }

    /// Pinned from Aider-AI/aider `aider/models.py` `ModelSettings`.
    const AIDER_MODEL_SETTINGS_FIELDS: &[&str] = &[
        "name",
        "edit_format",
        "weak_model_name",
        "use_repo_map",
        "send_undo_reply",
        "lazy",
        "overeager",
        "reminder",
        "examples_as_sys_msg",
        "extra_params",
        "cache_control",
        "caches_by_default",
        "use_system_prompt",
        "use_temperature",
        "streaming",
        "editor_model_name",
        "editor_edit_format",
        "reasoning_tag",
        "remove_reasoning",
        "system_prompt_prefix",
        "accepts_settings",
    ];
}
