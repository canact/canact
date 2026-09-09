//! Plumbing conformance table. Pass / degraded / fail. No composite score.

use serde::{Deserialize, Serialize};

use crate::cache::ProbeCache;
use crate::types::{CapabilityLevel, CapabilityProfile, EditFormatRecommendation};

/// One plumbing cell. Not a rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlumbingCell {
    /// Host can use the capability as-is.
    Pass,
    /// Works with a host workaround (XML, JSON repair, unified diff).
    Degraded,
    /// Missing or unusable.
    Fail,
}

/// One cached model's plumbing cells.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlumbingRow {
    /// Model id from the cache row.
    pub model: String,
    /// Provider label from the cache row.
    pub provider: String,
    /// Native tool calling completed Medium or stronger.
    pub native_tools: PlumbingCell,
    /// Native tools pass, or XML fallback when native is Weak.
    pub xml_fallback: PlumbingCell,
    /// Streaming tool-call probe completed Medium or stronger.
    pub streaming_tool_calls: PlumbingCell,
    /// Nested-argument probe completed Medium or stronger.
    pub nested_args: PlumbingCell,
    /// Complex / multi-tool probe completed Medium or stronger.
    pub complex_tools: PlumbingCell,
    /// Parallel floor: at least 2 pass, 1 degraded, else fail.
    pub parallel_floor: PlumbingCell,
    /// JSON Strong pass, Medium (repair) degraded, else fail.
    pub json_output: PlumbingCell,
    /// Search/replace pass, unified diff degraded, whole file fail.
    pub edit_format: PlumbingCell,
    /// A measured context floor exists.
    pub measured_context_floor: PlumbingCell,
    /// A measured provider output cap exists.
    pub max_output_tokens: PlumbingCell,
    /// `constraintPlacement` was measured (`system` or `user`).
    ///
    /// Fail when unprobed, skipped (policy/full), or error. Never invent
    /// `system`.
    pub constraint_placement: PlumbingCell,
}

/// Compatibility table for one provider. No overall score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlumbingMatrix {
    /// Provider filter used to build the table.
    pub provider: String,
    /// One row per model, sorted by model id.
    pub rows: Vec<PlumbingRow>,
}

impl PlumbingRow {
    /// Score one profile. Semantic probes stay out.
    pub fn from_profile(profile: &CapabilityProfile) -> Self {
        Self {
            model: profile.model_id.clone(),
            provider: profile.provider.clone(),
            native_tools: measured_medium_pass(&profile.tool_calling),
            xml_fallback: xml_cell(profile),
            streaming_tool_calls: measured_medium_pass(&profile.streaming_tool_calls),
            nested_args: measured_medium_pass(&profile.nested_arguments),
            complex_tools: measured_medium_pass(&profile.complex_tool_calling),
            parallel_floor: parallel_cell(profile),
            json_output: json_cell(profile),
            edit_format: edit_cell(profile),
            measured_context_floor: if profile.effective_context_tokens.is_some()
                || profile.probed_context_floor.is_some()
            {
                PlumbingCell::Pass
            } else {
                PlumbingCell::Fail
            },
            max_output_tokens: if profile.max_output_tokens.is_some() {
                PlumbingCell::Pass
            } else {
                PlumbingCell::Fail
            },
            constraint_placement: if profile.constraint_placement().is_some() {
                PlumbingCell::Pass
            } else {
                PlumbingCell::Fail
            },
        }
    }
}

impl PlumbingMatrix {
    /// Build a table from cached profiles for `provider`.
    ///
    /// Reads the cache only. Does not call a model. When the same model
    /// has several suite rows, the richest suite wins (all, then full,
    /// then policy).
    pub fn from_cache(cache: &ProbeCache, provider: &str) -> Self {
        let profiles = cache.matrix_profiles(provider);
        let rows = profiles
            .into_iter()
            .map(PlumbingRow::from_profile)
            .collect();
        Self {
            provider: provider.to_owned(),
            rows,
        }
    }
}

fn measured_medium_pass(pr: &crate::types::ProbeResult) -> PlumbingCell {
    match pr.measured_level() {
        Some(CapabilityLevel::Strong | CapabilityLevel::Medium) => PlumbingCell::Pass,
        Some(CapabilityLevel::Weak) | None => PlumbingCell::Fail,
    }
}

fn xml_cell(profile: &CapabilityProfile) -> PlumbingCell {
    match profile.tool_calling.measured_level() {
        Some(CapabilityLevel::Strong | CapabilityLevel::Medium) => PlumbingCell::Pass,
        _ => match profile.xml_tool_calling.measured_level() {
            Some(CapabilityLevel::Strong | CapabilityLevel::Medium) => PlumbingCell::Degraded,
            _ => PlumbingCell::Fail,
        },
    }
}

fn parallel_cell(profile: &CapabilityProfile) -> PlumbingCell {
    match profile.verified_parallel_tool_calls() {
        Some(n) if n >= 2 => PlumbingCell::Pass,
        Some(1) => PlumbingCell::Degraded,
        Some(_) | None => PlumbingCell::Fail,
    }
}

fn json_cell(profile: &CapabilityProfile) -> PlumbingCell {
    match profile.json_output.measured_level() {
        Some(CapabilityLevel::Strong) => PlumbingCell::Pass,
        Some(CapabilityLevel::Medium) => PlumbingCell::Degraded,
        Some(CapabilityLevel::Weak) | None => PlumbingCell::Fail,
    }
}

fn edit_cell(profile: &CapabilityProfile) -> PlumbingCell {
    match profile.best_edit_format() {
        EditFormatRecommendation::SearchReplace => PlumbingCell::Pass,
        EditFormatRecommendation::UnifiedDiff | EditFormatRecommendation::DiffFenced => {
            PlumbingCell::Degraded
        }
        EditFormatRecommendation::WholeFile => PlumbingCell::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProbeResult;

    fn probe(name: &str, level: CapabilityLevel) -> ProbeResult {
        ProbeResult {
            name: name.to_owned(),
            score: match level {
                CapabilityLevel::Strong => 1.0,
                CapabilityLevel::Medium => 0.5,
                CapabilityLevel::Weak => 0.1,
            },
            max_score: 1.0,
            level,
            details: "test".to_owned(),
        }
    }

    fn profile() -> CapabilityProfile {
        let mut p = CapabilityProfile::unprobed("m", "ollama");
        p.tool_calling = probe("tool_calling", CapabilityLevel::Strong);
        p.json_output = probe("json_output", CapabilityLevel::Strong);
        p.instruction_following = probe("instruction_following", CapabilityLevel::Strong);
        p.search_replace = probe("search_replace", CapabilityLevel::Strong);
        p.unified_diff = probe("unified_diff", CapabilityLevel::Medium);
        p.xml_tool_calling = probe("xml_tool_calling", CapabilityLevel::Weak);
        p.complex_tool_calling = probe("complex_tool_calling", CapabilityLevel::Strong);
        p.nested_arguments = probe("nested_arguments", CapabilityLevel::Strong);
        p.vision = probe("vision", CapabilityLevel::Weak);
        p.tool_selection = probe("tool_selection", CapabilityLevel::Medium);
        p.streaming_tool_calls = probe("streaming_tool_calls", CapabilityLevel::Strong);
        p.one_shot_tool_plan = probe("one_shot_tool_plan", CapabilityLevel::Strong);
        p.multi_turn_task_sequencing = probe("multi_turn_task_sequencing", CapabilityLevel::Strong);
        p.context_faithfulness = probe("context_faithfulness", CapabilityLevel::Strong);
        p.code_syntax = probe("code_syntax", CapabilityLevel::Strong);
        p.max_tokens_compliance = probe("max_tokens_compliance", CapabilityLevel::Strong);
        p.multi_turn_memory = probe("multi_turn_memory", CapabilityLevel::Strong);
        p.system_message_adherence = probe("system_message_adherence", CapabilityLevel::Strong);
        p.token_efficiency = probe("token_efficiency", CapabilityLevel::Strong);
        p.parallel_tool_scale = probe("parallel_tool_scale", CapabilityLevel::Strong);
        p.probed_at = 1;
        p.effective_context_tokens = Some(8192);
        p.probed_context_floor = Some(8192);
        p
    }

    #[test]
    fn strong_plumbing_is_pass() {
        let row = PlumbingRow::from_profile(&profile());
        assert_eq!(row.native_tools, PlumbingCell::Pass);
        assert_eq!(row.xml_fallback, PlumbingCell::Pass);
        assert_eq!(row.streaming_tool_calls, PlumbingCell::Pass);
        assert_eq!(row.nested_args, PlumbingCell::Pass);
        assert_eq!(row.complex_tools, PlumbingCell::Pass);
        assert_eq!(row.parallel_floor, PlumbingCell::Pass);
        assert_eq!(row.json_output, PlumbingCell::Pass);
        assert_eq!(row.edit_format, PlumbingCell::Pass);
        assert_eq!(row.measured_context_floor, PlumbingCell::Pass);
        assert_eq!(row.max_output_tokens, PlumbingCell::Fail);
        assert_eq!(row.constraint_placement, PlumbingCell::Pass);
    }

    #[test]
    fn measured_output_cap_is_pass() {
        let mut p = profile();
        p.max_output_tokens = Some(4096);
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.max_output_tokens, PlumbingCell::Pass);
    }

    #[test]
    fn omitted_output_cap_is_fail() {
        let p = profile();
        assert!(p.max_output_tokens.is_none());
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.max_output_tokens, PlumbingCell::Fail);
    }

    #[test]
    fn skipped_constraint_placement_is_fail() {
        let mut p = profile();
        p.system_message_adherence = ProbeResult {
            name: "system_message_adherence".into(),
            score: 0.5,
            max_score: 1.0,
            level: CapabilityLevel::Medium,
            details: "Skipped: policy suite (use --suite=full or --suite=all)".into(),
        };
        assert!(p.constraint_placement().is_none());
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.constraint_placement, PlumbingCell::Fail);
    }

    #[test]
    fn xml_only_is_degraded() {
        let mut p = profile();
        p.tool_calling = probe("tool_calling", CapabilityLevel::Weak);
        p.xml_tool_calling = probe("xml_tool_calling", CapabilityLevel::Medium);
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.native_tools, PlumbingCell::Fail);
        assert_eq!(row.xml_fallback, PlumbingCell::Degraded);
    }

    #[test]
    fn json_medium_is_degraded() {
        let mut p = profile();
        p.json_output = probe("json_output", CapabilityLevel::Medium);
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.json_output, PlumbingCell::Degraded);
    }

    #[test]
    fn unified_diff_only_is_degraded_edit() {
        let mut p = profile();
        p.search_replace = probe("search_replace", CapabilityLevel::Weak);
        p.unified_diff = probe("unified_diff", CapabilityLevel::Medium);
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.edit_format, PlumbingCell::Degraded);
    }

    #[test]
    fn parallel_one_is_degraded() {
        let mut p = profile();
        p.parallel_tool_scale = ProbeResult {
            name: "parallel_tool_scale".into(),
            score: 0.2,
            max_score: 1.0,
            level: CapabilityLevel::Weak,
            details: "one call".into(),
        };
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.parallel_floor, PlumbingCell::Degraded);
    }

    #[test]
    fn skipped_native_is_fail_not_pass() {
        let mut p = profile();
        p.tool_calling = ProbeResult {
            name: "tool_calling".into(),
            score: 0.5,
            max_score: 1.0,
            level: CapabilityLevel::Medium,
            details: "Skipped: policy suite (use --suite=full or --suite=all)".into(),
        };
        let row = PlumbingRow::from_profile(&p);
        assert_eq!(row.native_tools, PlumbingCell::Fail);
    }

    #[test]
    fn matrix_json_has_no_composite() {
        let mut cache = ProbeCache::default();
        cache.put(profile());
        let matrix = PlumbingMatrix::from_cache(&cache, "ollama");
        let value = serde_json::to_value(&matrix).unwrap();
        assert!(value.get("score").is_none(), "{value}");
        assert!(value.get("overall").is_none(), "{value}");
        assert!(value.get("rank").is_none(), "{value}");
        assert_eq!(value["rows"].as_array().unwrap().len(), 1);
        assert!(value["rows"][0].get("score").is_none(), "{value}");
    }

    #[test]
    fn empty_provider_has_no_rows() {
        let cache = ProbeCache::default();
        let matrix = PlumbingMatrix::from_cache(&cache, "ollama");
        assert!(matrix.rows.is_empty());
    }
}
