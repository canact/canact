//! Host overlay files for products that will not embed canact.
//!
//! Aider reads `.aider.model.settings.yml` plus
//! `.aider.model.metadata.json`. Cline's `ModelInfo` is a JSON object
//! the user pastes into an OpenAI-compatible model entry.

mod aider;
mod cline;

pub use aider::{AiderMetadataEntry, AiderOverlay, AiderSettingsRow};
pub use cline::ClineModelInfo;

use crate::types::{CapabilityProfile, EditFormatRecommendation};

/// Files written by [`HostOverlay::write_to`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayFiles {
    /// Relative file name (for example `.aider.model.settings.yml`).
    pub name: &'static str,
    /// File body.
    pub body: String,
}

/// One host's overlay, built from a live [`CapabilityProfile`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostOverlay {
    /// Aider user overlay (settings YAML + metadata JSON).
    Aider(AiderOverlay),
    /// Cline `ModelInfo` JSON object.
    Cline(ClineModelInfo),
}

impl HostOverlay {
    /// Build the Aider overlay for `profile`.
    pub fn aider(profile: &CapabilityProfile, advertised: Option<u32>) -> Self {
        Self::Aider(AiderOverlay::from_profile(profile, advertised))
    }

    /// Build the Cline overlay for `profile`.
    pub fn cline(profile: &CapabilityProfile, advertised: Option<u32>) -> Self {
        Self::Cline(ClineModelInfo::from_profile(profile, advertised))
    }

    /// Serialize every file this overlay writes.
    pub fn files(&self) -> Vec<OverlayFiles> {
        match self {
            Self::Aider(o) => o.files(),
            Self::Cline(o) => o.files(),
        }
    }

    /// Write every file under `dir`. Returns the paths written.
    pub fn write_to(
        &self,
        dir: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
        std::fs::create_dir_all(dir)?;
        let mut paths = Vec::new();
        for file in self.files() {
            let path = dir.join(file.name);
            std::fs::write(&path, file.body.as_bytes())?;
            paths.push(path);
        }
        Ok(paths)
    }
}

/// Map the canact edit ladder onto Aider's `edit_format` tokens.
pub fn aider_edit_format(rec: EditFormatRecommendation) -> &'static str {
    match rec {
        EditFormatRecommendation::SearchReplace => "diff",
        EditFormatRecommendation::UnifiedDiff => "udiff",
        EditFormatRecommendation::WholeFile => "whole",
        EditFormatRecommendation::DiffFenced => "diff-fenced",
    }
}

/// Aider / LiteLLM style `provider/model` name.
pub fn overlay_model_name(profile: &CapabilityProfile) -> String {
    if profile.model_id.contains('/') {
        profile.model_id.clone()
    } else {
        format!("{}/{}", profile.provider, profile.model_id)
    }
}

/// Context tokens the host should compact against.
pub fn overlay_context_tokens(profile: &CapabilityProfile, advertised: Option<u32>) -> Option<u32> {
    profile.recommended_context_tokens(advertised)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CapabilityLevel, CapabilityProfile, ProbeResult};

    pub(crate) fn sample_profile(
        search: CapabilityLevel,
        unified: CapabilityLevel,
        vision: CapabilityLevel,
    ) -> CapabilityProfile {
        let pr = |name: &str, level: CapabilityLevel| ProbeResult {
            name: name.to_owned(),
            score: match level {
                CapabilityLevel::Strong => 1.0,
                CapabilityLevel::Medium => 0.5,
                CapabilityLevel::Weak => 0.1,
            },
            max_score: 1.0,
            level,
            details: "test".to_owned(),
        };
        CapabilityProfile {
            model_id: "qwen2.5-coder".to_owned(),
            provider: "ollama".to_owned(),
            tool_calling: pr("tool_calling", CapabilityLevel::Strong),
            json_output: pr("json_output", CapabilityLevel::Strong),
            instruction_following: pr("instruction_following", CapabilityLevel::Strong),
            search_replace: pr("search_replace", search),
            unified_diff: pr("unified_diff", unified),
            xml_tool_calling: pr("xml_tool_calling", CapabilityLevel::Medium),
            complex_tool_calling: pr("complex_tool_calling", CapabilityLevel::Strong),
            nested_arguments: pr("nested_arguments", CapabilityLevel::Strong),
            vision: pr("vision", vision),
            tool_selection: pr("tool_selection", CapabilityLevel::Medium),
            streaming_tool_calls: pr("streaming_tool_calls", CapabilityLevel::Strong),
            one_shot_tool_plan: pr("one_shot_tool_plan", CapabilityLevel::Strong),
            multi_turn_task_sequencing: pr("multi_turn_task_sequencing", CapabilityLevel::Strong),
            context_faithfulness: pr("context_faithfulness", CapabilityLevel::Strong),
            code_syntax: pr("code_syntax", CapabilityLevel::Strong),
            max_tokens_compliance: pr("max_tokens_compliance", CapabilityLevel::Strong),
            multi_turn_memory: pr("multi_turn_memory", CapabilityLevel::Strong),
            system_message_adherence: pr("system_message_adherence", CapabilityLevel::Strong),
            token_efficiency: pr("token_efficiency", CapabilityLevel::Strong),
            parallel_tool_scale: pr("parallel_tool_scale", CapabilityLevel::Strong),
            probed_at: 1_700_000_000,
            effective_context_tokens: Some(8192),
            probed_context_floor: Some(8192),
        }
    }

    #[test]
    fn overlay_name_prefixes_provider_when_model_has_no_slash() {
        let p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        assert_eq!(overlay_model_name(&p), "ollama/qwen2.5-coder");
    }

    #[test]
    fn overlay_name_keeps_already_namespaced_model() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.model_id = "openai/gpt-4o".to_owned();
        assert_eq!(overlay_model_name(&p), "openai/gpt-4o");
    }

    #[test]
    fn aider_edit_format_maps_ladder() {
        assert_eq!(
            aider_edit_format(EditFormatRecommendation::SearchReplace),
            "diff"
        );
        assert_eq!(
            aider_edit_format(EditFormatRecommendation::UnifiedDiff),
            "udiff"
        );
        assert_eq!(
            aider_edit_format(EditFormatRecommendation::WholeFile),
            "whole"
        );
        assert_eq!(
            aider_edit_format(EditFormatRecommendation::DiffFenced),
            "diff-fenced"
        );
    }
}
