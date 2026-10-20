//! Human CLI presentation for a finished [`CapabilityProfile`].

use std::fmt::Write as _;

use crate::types::{CORE_DIMENSION_NAMES, CapabilityProfile, ConstraintPlacement, DIMENSION_NAMES};

impl CapabilityProfile {
    /// Non-`--json` table printed by `canact probe`.
    pub fn format_human_table(&self, verbose: bool) -> String {
        self.format_human_table_with(verbose, None)
    }

    /// Human table with an advertised context prior for the recommended window.
    pub fn format_human_table_with(&self, verbose: bool, advertised: Option<u32>) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== Probe Results ===");
        let _ = writeln!(out, "Weak < 0.4   Medium >= 0.4   Strong >= 0.8");
        if !verbose {
            let _ = writeln!(
                out,
                "(Showing core dimensions; use --verbose for the rest.)"
            );
        }
        let _ = writeln!(out);
        let dims: &[&str] = if verbose {
            DIMENSION_NAMES
        } else {
            CORE_DIMENSION_NAMES
        };
        for &dim in dims {
            if dim == "one_shot_tool_plan" {
                continue;
            }
            if let Some(probe) = self.dimension_result(dim) {
                if probe.is_skipped() || probe.is_unprobed_default() {
                    let _ = writeln!(out, "{:<28}not probed", format!("{}:", display_name(dim)),);
                } else {
                    let _ = writeln!(
                        out,
                        "{:<28}{:?}  {:.1} / {:.1}",
                        format!("{}:", display_name(dim)),
                        probe.completed_level(),
                        if probe.is_synthesized_error() {
                            0.0
                        } else {
                            probe.score
                        },
                        probe.max_score
                    );
                }
                let _ = writeln!(out, "{:<28}{}", "", probe.details);
            }
        }
        let _ = writeln!(out);
        let _ = writeln!(out, "{:<28}{:?}", "Overall:", self.overall_level());
        let _ = writeln!(
            out,
            "{:<28}{:?}",
            "Edit format (probe ladder):",
            self.best_edit_format()
        );
        let tools = if self.can_use_tools() {
            "yes"
        } else if self.tool_gate_error().is_some() {
            "no"
        } else {
            "unknown (tool probes did not finish)"
        };
        let _ = writeln!(out, "{:<28}{tools}", "Can use tools:");
        let _ = writeln!(
            out,
            "{:<28}{}",
            "Vision:",
            if self.supports_vision() {
                "supported"
            } else if self.vision.measured_level().is_none() {
                "not probed"
            } else {
                "not supported"
            }
        );
        match self.max_tools() {
            Some(n) => {
                let _ = writeln!(out, "{:<28}{n}", "Max tools:");
            }
            None => {
                let _ = writeln!(out, "{:<28}unlimited", "Max tools:");
            }
        }
        if let Some(n) = advertised {
            let _ = writeln!(out, "{:<28}{n}", "Advertised context tokens:");
        }
        if let Some(n) = self.recommended_context_tokens(advertised) {
            let _ = writeln!(out, "{:<28}{n}", "Recommended context tokens:");
        }
        if let Some(n) = self.effective_context_tokens {
            let _ = writeln!(out, "{:<28}{n}", "Effective context tokens:");
        } else if let Some(n) = self.probed_context_floor {
            let _ = writeln!(out, "{:<28}{n}", "Probed context floor:");
        }
        if let Some(n) = self.max_output_tokens {
            let _ = writeln!(out, "{:<28}{n}", "Max output tokens:");
        }
        if let Some(placement) = self.constraint_placement() {
            let value = match placement {
                ConstraintPlacement::System => "system",
                ConstraintPlacement::User => "user",
            };
            let _ = writeln!(out, "{:<28}{value}", "Constraint placement:");
        }
        if self.tool_calling.measured_level() == Some(crate::types::CapabilityLevel::Weak) {
            let _ = writeln!(out, "{:<28}XML fallback needed", "");
        }
        if self.needs_json_repair() {
            let _ = writeln!(out, "{:<28}JSON repair needed", "");
        }
        out
    }

    /// Stderr line when both tool probes completed below Medium.
    ///
    /// One finished Weak row is not enough. The other probe may still
    /// be a `Probe failed:` timeout or vendor error. Exit 2 is only
    /// for a pair of completed scores that both sit below Medium.
    pub fn tool_gate_error(&self) -> Option<&'static str> {
        if self.can_use_tools() {
            return None;
        }
        let native = self.tool_calling.measured_level();
        let xml = self.xml_tool_calling.measured_level();
        match (native, xml) {
            (Some(native), Some(xml))
                if native < crate::types::CapabilityLevel::Medium
                    && xml < crate::types::CapabilityLevel::Medium =>
            {
                Some(
                    "error: cannot use tools (native and XML tool calling both completed below Medium)",
                )
            }
            _ => None,
        }
    }
}

fn display_name(dim: &str) -> String {
    dim.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `--model` is required because `GET /models` did not return exactly one id.
///
/// Includes the count and every listed id so the user can pick `--model`.
pub fn missing_model_message<S: AsRef<str>>(ids: &[S]) -> String {
    let n = ids.len();
    let mut msg =
        format!("error: --model is required unless GET /models returns exactly one id (got {n}");
    if n == 0 {
        msg.push(')');
        return msg;
    }
    msg.push_str(": ");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            msg.push_str(", ");
        }
        msg.push_str(id.as_ref());
    }
    msg.push(')');
    msg
}

#[cfg(test)]
mod tests {
    use crate::types::{CapabilityLevel, CapabilityProfile, DIMENSION_NAMES, ProbeResult};

    fn completed(name: &str) -> ProbeResult {
        ProbeResult {
            name: name.to_owned(),
            score: 1.0,
            max_score: 1.0,
            level: CapabilityLevel::Strong,
            details: "done".to_owned(),
        }
    }

    #[test]
    fn verbose_human_table_has_19_dimension_labels() {
        let mut profile = CapabilityProfile::unprobed("m", "p");
        for &name in DIMENSION_NAMES {
            *profile
                .dimension_result_mut(name)
                .unwrap_or_else(|| panic!("missing {name}")) = completed(name);
        }
        let verbose = profile.format_human_table(true);
        let score_lines = verbose.lines().filter(|line| line.contains(" / ")).count();
        assert_eq!(score_lines, 19, "{verbose}");
        assert!(!verbose.contains("One Shot Tool Plan"), "{verbose}");
        let quiet = profile.format_human_table(false);
        assert!(quiet.contains("use --verbose for the rest."), "{quiet}");
        assert!(!quiet.contains("use --verbose for all."), "{quiet}");
    }

    fn unfinished(name: &str) -> ProbeResult {
        ProbeResult {
            name: name.to_owned(),
            score: 0.5,
            max_score: 1.0,
            level: CapabilityLevel::Medium,
            details: "Probe failed: transient error: please wait".to_owned(),
        }
    }

    fn weak(name: &str) -> ProbeResult {
        ProbeResult {
            name: name.to_owned(),
            score: 0.1,
            max_score: 1.0,
            level: CapabilityLevel::Weak,
            details: "no tool call".to_owned(),
        }
    }

    #[test]
    fn tool_gate_skips_exit_2_when_tool_probes_did_not_complete() {
        let mut profile = CapabilityProfile::unprobed("m", "p");
        profile.tool_calling = unfinished("tool_calling");
        profile.xml_tool_calling = unfinished("xml_tool_calling");
        assert!(!profile.can_use_tools());
        assert!(profile.tool_gate_error().is_none());
        let table = profile.format_human_table(false);
        assert!(
            table.contains("unknown (tool probes did not finish)"),
            "{table}"
        );
        assert!(!table.contains("XML fallback needed"), "{table}");
        assert!(!profile.needs_xml_fallback());
    }

    #[test]
    fn tool_gate_skips_exit_2_when_only_xml_finished_weak() {
        let mut profile = CapabilityProfile::unprobed("m", "p");
        profile.tool_calling = unfinished("tool_calling");
        profile.xml_tool_calling = weak("xml_tool_calling");
        assert!(!profile.can_use_tools());
        assert!(
            profile.tool_gate_error().is_none(),
            "a rate limit on native must not exit 2 because XML finished Weak"
        );
    }

    #[test]
    fn tool_gate_exit_2_when_both_tool_probes_completed_weak() {
        let mut profile = CapabilityProfile::unprobed("m", "p");
        profile.tool_calling = weak("tool_calling");
        profile.xml_tool_calling = weak("xml_tool_calling");
        assert!(!profile.can_use_tools());
        let msg = profile.tool_gate_error().expect("both weak");
        assert!(msg.contains("cannot use tools"), "{msg}");
        assert!(msg.contains("both completed below Medium"), "{msg}");
        let table = profile.format_human_table(false);
        assert!(
            table
                .lines()
                .any(|line| line.contains("Can use tools:") && line.trim_end().ends_with("no")),
            "{table}"
        );
        assert!(table.contains("XML fallback needed"), "{table}");
    }
}
