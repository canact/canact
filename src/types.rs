//! Capability probe results and host policy methods.
//!
//! canact owns [`CapabilityLevel`], [`CapabilityProfile`], [`classify`],
//! and the policy methods on the profile (`max_tools`, `best_edit_format`,
//! `needs_xml_fallback`, `meets`, and related getters).
//!
//! Bline will `pub use` these types from `bline-types` and add a
//! `CapabilityProfileExt` trait for `meets_requirements`,
//! `tool_eligibility`, and `tool_selection_level` if it keeps those
//! Bline-only wrappers. [`CapabilityProfile::meets`] takes caller-supplied
//! `(name, level)` pairs. Bline must zip [`REQUIREMENT_DIMENSION_NAMES`]
//! ([`DIMENSION_NAMES`]`[0..9]`), not [`CORE_DIMENSION_NAMES`].
//! Re-exporting the types is not enough for `tool_filter.rs` to compile;
//! every call site must import `CapabilityProfileExt`.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Individual probe result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResult {
    /// Name of the probe (e.g. `"tool_calling"`).
    pub name: String,
    /// Achieved score in the range `0.0..=1.0`.
    pub score: f32,
    /// Maximum achievable score (always `1.0` for the built-in probes).
    pub max_score: f32,
    /// Classified capability level derived from the score.
    pub level: CapabilityLevel,
    /// Human-readable explanation of how the score was determined.
    pub details: String,
}

impl ProbeResult {
    /// Synthesized `resolve_probe` error (timeout / 429 / 5xx), not a
    /// completed score. Details always start with `Probe failed:`.
    pub fn is_synthesized_error(&self) -> bool {
        self.details.starts_with("Probe failed:")
    }
}

/// Recommended edit format based on probe results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditFormatRecommendation {
    /// Targeted search/replace blocks. For strong models.
    SearchReplace,
    /// Unified diff format. Middle ground.
    UnifiedDiff,
    /// Rewrite the entire file. For weak models.
    WholeFile,
    /// Search/replace blocks wrapped in fenced code blocks. For Gemini models.
    DiffFenced,
}

/// Capability level for a probe dimension.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    /// Score below 0.4.
    #[default]
    Weak,
    /// Score in `0.4..0.8`.
    Medium,
    /// Score at or above 0.8.
    Strong,
}

/// Default probe result for deserialization when the field is absent.
pub(crate) fn default_probe() -> ProbeResult {
    ProbeResult {
        name: "unknown".to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: "Not probed (cached before this probe existed)".to_string(),
    }
}

/// Generates [`CapabilityProfile`], [`DIMENSION_NAMES`], and the
/// `dimension_level` / `dimension_result` lookups from one field list.
///
/// Adding a new probe dimension only requires adding one entry here.
/// Fields in the `required` group have no serde default (they must be
/// present when deserializing). Fields in the `defaulted` group get
/// `#[serde(default = "default_probe")]` so older caches that lack
/// the field still deserialize.
macro_rules! define_probe_dimensions {
    (
        required {
            $(
                $(#[$req_meta:meta])*
                $req_field:ident,
            )*
        }
        defaulted {
            $(
                $(#[$def_meta:meta])*
                $def_field:ident,
            )*
        }
    ) => {
        /// Complete capability profile for a model.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub struct CapabilityProfile {
            /// Model identifier (e.g. `"gpt-4o"`).
            pub model_id: String,
            /// Provider name (e.g. `"openai"`).
            pub provider: String,
            $(
                $(#[$req_meta])*
                pub $req_field: ProbeResult,
            )*
            $(
                $(#[$def_meta])*
                #[serde(default = "default_probe")]
                pub $def_field: ProbeResult,
            )*
            /// Unix epoch seconds when the profile was created.
            pub probed_at: u64,
            /// Measured usable context, in tokens. `None` until a suite writes it.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub effective_context_tokens: Option<u32>,
        }

        /// All probed dimension names, in the order they appear on the struct.
        ///
        /// Used by `dimension_level` and `dimension_result` so there is
        /// exactly one place to update when a new probe is added.
        pub const DIMENSION_NAMES: &[&str] = &[
            $(stringify!($req_field),)*
            $(stringify!($def_field),)*
        ];

        impl CapabilityProfile {
            /// Look up the capability level for a named probe dimension.
            ///
            /// Accepts snake_case (`tool_calling`) and host-envelope camelCase
            /// (`toolCalling`). Returns `None` for unrecognised names.
            pub fn dimension_level(&self, dimension: &str) -> Option<CapabilityLevel> {
                match normalize_dimension_name(dimension).as_ref() {
                    $(stringify!($req_field) => Some(self.$req_field.level),)*
                    $(stringify!($def_field) => Some(self.$def_field.level),)*
                    _ => None,
                }
            }

            /// Look up the full [`ProbeResult`] for a named dimension.
            ///
            /// Accepts snake_case and host-envelope camelCase. Returns `None`
            /// for unrecognised names.
            pub fn dimension_result(&self, dimension: &str) -> Option<&ProbeResult> {
                match normalize_dimension_name(dimension).as_ref() {
                    $(stringify!($req_field) => Some(&self.$req_field),)*
                    $(stringify!($def_field) => Some(&self.$def_field),)*
                    _ => None,
                }
            }

            /// Mutable lookup for a named dimension (cache migration).
            pub fn dimension_result_mut(&mut self, dimension: &str) -> Option<&mut ProbeResult> {
                match normalize_dimension_name(dimension).as_ref() {
                    $(stringify!($req_field) => Some(&mut self.$req_field),)*
                    $(stringify!($def_field) => Some(&mut self.$def_field),)*
                    _ => None,
                }
            }
        }
    };
}

define_probe_dimensions! {
    required {
        /// Result of the tool-calling probe.
        tool_calling,
        /// Result of the JSON output probe.
        json_output,
        /// Result of the instruction-following probe.
        instruction_following,
    }
    defaulted {
        /// Result of the SEARCH/REPLACE edit format probe.
        search_replace,
        /// Result of the unified diff edit format probe.
        unified_diff,
        /// Result of the complex (multi-tool) tool-calling probe.
        complex_tool_calling,
        /// Result of the nested-arguments probe.
        nested_arguments,
        /// Result of the vision capability probe.
        vision,
        /// Result of the tool-selection probe (picking the right tool from a set).
        tool_selection,
        /// Result of the XML fallback tool-calling probe.
        xml_tool_calling,
        /// Result of the streaming tool-call probe.
        streaming_tool_calls,
        /// One-shot ordered multi-tool plan (not multi-turn agent sequencing).
        /// Serde alias keeps older probe caches readable (#1336).
        #[serde(alias = "multiStepReasoning")]
        one_shot_tool_plan,
        /// Multi-turn agent-loop task sequencing (read, act, verify).
        multi_turn_task_sequencing,
        /// Result of the context faithfulness probe.
        context_faithfulness,
        /// Result of the code syntax accuracy probe.
        code_syntax,
        /// Result of the max-tokens compliance probe.
        max_tokens_compliance,
        /// Result of the multi-turn memory probe.
        multi_turn_memory,
        /// Result of the system message adherence probe.
        system_message_adherence,
        /// Result of the token efficiency probe.
        token_efficiency,
        /// Result of the parallel tool-call scaling probe.
        parallel_tool_scale,
    }
}

/// Probe names scored Weak when the provider reports "does not support tools".
/// Shared by `resolve_probe` and stale-cache migration.
pub const TOOL_PROBE_NAMES: &[&str] = &[
    "tool_calling",
    "complex_tool_calling",
    "nested_arguments",
    "tool_selection",
    "streaming_tool_calls",
    "parallel_tool_scale",
    "one_shot_tool_plan",
    "multi_turn_task_sequencing",
];

/// First 9 of [`DIMENSION_NAMES`]. Zips 1:1 with Bline `ToolRequirements::as_slice()`.
/// Do not zip [`CORE_DIMENSION_NAMES`] against that slice.
pub const REQUIREMENT_DIMENSION_NAMES: &[&str] = &[
    "tool_calling",
    "json_output",
    "instruction_following",
    "search_replace",
    "unified_diff",
    "complex_tool_calling",
    "nested_arguments",
    "vision",
    "tool_selection",
];

/// Default human table only. Includes `xml_tool_calling`, omits `tool_selection`.
/// Not a `ToolRequirements` zip.
pub const CORE_DIMENSION_NAMES: &[&str] = &[
    "tool_calling",
    "xml_tool_calling",
    "complex_tool_calling",
    "nested_arguments",
    "json_output",
    "instruction_following",
    "search_replace",
    "unified_diff",
    "vision",
];

impl CapabilityProfile {
    /// Overall capability level (minimum of the three required dimensions).
    pub fn overall_level(&self) -> CapabilityLevel {
        completed_level(&self.tool_calling)
            .min(completed_level(&self.json_output))
            .min(completed_level(&self.instruction_following))
    }

    /// Whether the host should use XML-tag fallback for tool calls.
    pub fn needs_xml_fallback(&self) -> bool {
        completed_level(&self.tool_calling) == CapabilityLevel::Weak
    }

    /// Whether JSON output should be wrapped in a repair layer.
    pub fn needs_json_repair(&self) -> bool {
        completed_level(&self.json_output) <= CapabilityLevel::Medium
    }

    /// Whether the model can be used for agentic work (tool calling).
    ///
    /// Returns `false` if both native and XML failed to complete at
    /// Medium or above. A synthesized error Medium (timeout / 429)
    /// does not count.
    pub fn can_use_tools(&self) -> bool {
        completed_usable_tools(&self.tool_calling) || completed_usable_tools(&self.xml_tool_calling)
    }

    /// How well the model picks the right tool from a set.
    pub fn tool_selection_level(&self) -> CapabilityLevel {
        completed_level(&self.tool_selection)
    }

    /// Recommended maximum number of tools to send to the model.
    ///
    /// Strong (`>= 0.8`): no limit. Medium (`>= 0.4`): 20. Weak: 10.
    /// Scores in `[1.0 / 3.0, 0.4)` return `None` (issue #3315).
    pub fn max_tools(&self) -> Option<usize> {
        // Generic-edit scores ~0.33 are not 30-day Weak (#3315).
        if !self.tool_selection.is_synthesized_error()
            && (1.0 / 3.0..0.4).contains(&self.tool_selection.score)
        {
            return None;
        }
        match completed_level(&self.tool_selection) {
            CapabilityLevel::Strong => None,
            CapabilityLevel::Medium => Some(20),
            CapabilityLevel::Weak => Some(10),
        }
    }

    /// Whether the model supports vision (image input).
    ///
    /// Returns `true` if the vision probe scored Medium or higher.
    /// A synthesized error Medium (timeout / 429) does not count.
    pub fn supports_vision(&self) -> bool {
        completed_usable_tools(&self.vision)
    }

    /// Recommend the best edit format from the probe ladder.
    ///
    /// Strong `search_replace` yields
    /// [`EditFormatRecommendation::SearchReplace`]. Otherwise `unified_diff`
    /// at Medium or above yields [`EditFormatRecommendation::UnifiedDiff`].
    /// Else [`EditFormatRecommendation::WholeFile`]. Does not apply a Gemini
    /// [`EditFormatRecommendation::DiffFenced`] override.
    pub fn best_edit_format(&self) -> EditFormatRecommendation {
        if completed_level(&self.search_replace) == CapabilityLevel::Strong {
            EditFormatRecommendation::SearchReplace
        } else if completed_level(&self.unified_diff) >= CapabilityLevel::Medium {
            EditFormatRecommendation::UnifiedDiff
        } else {
            EditFormatRecommendation::WholeFile
        }
    }

    /// Returns true when every named dimension is at least the required level.
    ///
    /// Unknown dimension names are skipped. Hosts zip their own pairs.
    /// Bline zips [`REQUIREMENT_DIMENSION_NAMES`] (the first 9 of
    /// [`DIMENSION_NAMES`]), not [`CORE_DIMENSION_NAMES`].
    pub fn meets(&self, reqs: &[(&str, CapabilityLevel)]) -> bool {
        for &(name, required) in reqs {
            match self.dimension_result(name) {
                Some(pr) if completed_level(pr) < required => return false,
                _ => {}
            }
        }
        true
    }

    /// canact CLI `--json` host-policy envelope.
    ///
    /// Not Bline `build_probe_json`. Does not emit `bestEditFormat`.
    pub fn host_policy_envelope(&self) -> serde_json::Value {
        let mut probes = serde_json::Map::new();
        for &dim in DIMENSION_NAMES {
            if let Some(probe) = self.dimension_result(dim) {
                probes.insert(snake_to_camel(dim), probe_envelope_json(probe));
            }
        }
        serde_json::json!({
            "model": self.model_id,
            "provider": self.provider,
            "overall": self.overall_level(),
            "probeLadderEditFormat": self.best_edit_format(),
            "canUseTools": self.can_use_tools(),
            "supportsVision": self.supports_vision(),
            "maxTools": self.max_tools(),
            "needsXmlFallback": self.needs_xml_fallback(),
            "needsJsonRepair": self.needs_json_repair(),
            "effectiveContextTokens": self.effective_context_tokens,
            "probedAt": self.probed_at,
            "scoreScale": {
                "min": 0.0,
                "max": 1.0,
                "strongMin": 0.8,
                "mediumMin": 0.4,
            },
            "probes": probes,
        })
    }
}

fn completed_usable_tools(pr: &ProbeResult) -> bool {
    completed_level(pr) >= CapabilityLevel::Medium
}

fn completed_level(pr: &ProbeResult) -> CapabilityLevel {
    if pr.is_synthesized_error() {
        CapabilityLevel::Weak
    } else {
        pr.level
    }
}

fn normalize_dimension_name(dimension: &str) -> Cow<'_, str> {
    if dimension.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(camel_to_snake(dimension))
    } else {
        Cow::Borrowed(dimension)
    }
}

fn camel_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn snake_to_camel(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut cap_next = false;
    for (i, ch) in s.chars().enumerate() {
        if ch == '_' {
            cap_next = true;
            continue;
        }
        if cap_next {
            out.extend(ch.to_uppercase());
            cap_next = false;
        } else if i == 0 {
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn probe_envelope_json(probe: &ProbeResult) -> serde_json::Value {
    serde_json::json!({
        "level": probe.level,
        "score": probe.score,
        "maxScore": probe.max_score,
        "details": probe.details,
    })
}

/// Classify a normalized score (`0.0..=1.0`) into a capability level.
///
/// Thresholds: `>= 0.8` → Strong, `>= 0.4` → Medium, else Weak.
pub fn classify(score: f32) -> CapabilityLevel {
    if score >= 0.8 {
        CapabilityLevel::Strong
    } else if score >= 0.4 {
        CapabilityLevel::Medium
    } else {
        CapabilityLevel::Weak
    }
}
