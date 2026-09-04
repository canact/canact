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
        let files = self.files();
        for file in &files {
            let path = dir.join(file.name);
            if path
                .symlink_metadata()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("refusing to write through symlink {}", path.display()),
                ));
            }
        }
        let mut paths = Vec::new();
        for file in files {
            let path = dir.join(file.name);
            if path
                .symlink_metadata()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("refusing to write through symlink {}", path.display()),
                ));
            }
            let body = match self {
                Self::Aider(_) if path.exists() => merge_overlay_file(&path, &file)?,
                _ => file.body,
            };
            std::fs::write(&path, body.as_bytes())?;
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
    let lowered = profile.provider.to_ascii_lowercase();
    let provider = normalize_overlay_provider(&lowered);
    let prefix = format!("{provider}/");
    if profile.model_id.starts_with(&prefix) {
        profile.model_id.clone()
    } else {
        format!("{prefix}{}", profile.model_id)
    }
}

pub(crate) fn normalize_overlay_provider(provider: &str) -> &str {
    match provider {
        "openrouter.ai" | "openrouter" => "openrouter",
        "api.openai.com" | "openai" => "openai",
        "localhost" | "127.0.0.1" | "::1" | "[::1]" => "ollama",
        other => other,
    }
}

/// Context tokens the host should compact against.
pub fn overlay_context_tokens(profile: &CapabilityProfile, advertised: Option<u32>) -> Option<u32> {
    profile.recommended_context_tokens(advertised)
}

fn merge_overlay_file(path: &std::path::Path, file: &OverlayFiles) -> std::io::Result<String> {
    let existing = std::fs::read_to_string(path)?;
    match file.name {
        ".aider.model.settings.yml" => Ok(merge_aider_settings_yaml(&existing, &file.body)),
        ".aider.model.metadata.json" => merge_aider_metadata_json(&existing, &file.body),
        _ => Ok(file.body.clone()),
    }
}

fn merge_aider_settings_yaml(existing: &str, incoming: &str) -> String {
    let Some(name_line) = incoming.lines().find(|l| l.starts_with("- name: ")) else {
        return incoming.to_owned();
    };
    let mut out = String::new();
    let mut skipping = false;
    let mut replaced = false;
    for line in existing.lines() {
        if line.starts_with("- name: ") {
            if line == name_line {
                skipping = true;
                if !replaced {
                    out.push_str(incoming);
                    if !incoming.ends_with('\n') {
                        out.push('\n');
                    }
                    replaced = true;
                }
                continue;
            }
            skipping = false;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !replaced {
        if !out.ends_with('\n') && !out.is_empty() {
            out.push('\n');
        }
        out.push_str(incoming);
        if !incoming.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

fn merge_aider_metadata_json(existing: &str, incoming: &str) -> std::io::Result<String> {
    let mut dest: serde_json::Map<String, serde_json::Value> = serde_json::from_str(existing)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let src: serde_json::Map<String, serde_json::Value> = serde_json::from_str(incoming)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    for (k, v) in src {
        dest.insert(k, v);
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&dest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
    ))
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
    fn overlay_name_maps_loopback_host_to_ollama() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        for host in ["localhost", "127.0.0.1", "::1", "[::1]"] {
            p.provider = host.to_owned();
            assert_eq!(
                overlay_model_name(&p),
                "ollama/qwen2.5-coder",
                "loopback provider {host} must export as ollama"
            );
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
    fn overlay_name_prefixes_openrouter_for_slash_model() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.model_id = "anthropic/claude-3.5-sonnet".to_owned();
        p.provider = "openrouter.ai".to_owned();
        assert_eq!(
            overlay_model_name(&p),
            "openrouter/anthropic/claude-3.5-sonnet"
        );
    }

    #[test]
    fn overlay_name_keeps_already_namespaced_model() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.provider = "openai".to_owned();
        p.model_id = "openai/gpt-4o".to_owned();
        assert_eq!(overlay_model_name(&p), "openai/gpt-4o");
    }

    #[test]
    fn overlay_name_prefixes_slash_model_unless_already_prefixed() {
        let mut p = sample_profile(
            CapabilityLevel::Strong,
            CapabilityLevel::Medium,
            CapabilityLevel::Weak,
        );
        p.model_id = "library/qwen".to_owned();
        assert_eq!(
            overlay_model_name(&p),
            "ollama/library/qwen",
            "ollama + library/qwen must keep the provider prefix"
        );
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

    #[test]
    fn merge_aider_settings_appends_other_model() {
        let existing = "- name: other/m\n  edit_format: whole\n  use_repo_map: false\n";
        let incoming = "- name: ollama/qwen2.5-coder\n  edit_format: diff\n  use_repo_map: true\n";
        let merged = merge_aider_settings_yaml(existing, incoming);
        assert!(merged.contains("other/m"), "{merged}");
        assert!(merged.contains("ollama/qwen2.5-coder"), "{merged}");
    }

    #[test]
    fn merge_aider_settings_replaces_same_model() {
        let existing =
            "- name: ollama/qwen2.5-coder\n  edit_format: whole\n  use_repo_map: false\n";
        let incoming = "- name: ollama/qwen2.5-coder\n  edit_format: diff\n  use_repo_map: true\n";
        let merged = merge_aider_settings_yaml(existing, incoming);
        assert_eq!(merged.matches("- name:").count(), 1);
        assert!(merged.contains("edit_format: diff"), "{merged}");
        assert!(!merged.contains("whole"), "{merged}");
    }

    #[test]
    #[cfg(unix)]
    fn write_to_refuses_symlink_dest() {
        let dir = tempfile::tempdir().expect("temp");
        let target = dir.path().join("real.yml");
        std::fs::write(&target, b"keep\n").expect("target");
        let dest = dir.path().join(".aider.model.settings.yml");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &dest).expect("symlink");
            let overlay = HostOverlay::aider(
                &sample_profile(
                    CapabilityLevel::Strong,
                    CapabilityLevel::Medium,
                    CapabilityLevel::Weak,
                ),
                None,
            );
            let err = overlay.write_to(dir.path()).expect_err("symlink");
            assert!(err.to_string().contains("symlink"), "{err}");
            assert_eq!(std::fs::read_to_string(&target).expect("read"), "keep\n");
        }
    }
}
