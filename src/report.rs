//! Human CLI presentation for a finished [`CapabilityProfile`].

use std::fmt::Write as _;

use crate::types::{CORE_DIMENSION_NAMES, CapabilityProfile, DIMENSION_NAMES};

/// How many `/models` ids to include in the missing-`--model` error.
const MISSING_MODEL_ID_PREVIEW: usize = 8;

impl CapabilityProfile {
    /// Non-`--json` table printed by `canact probe`.
    pub fn format_human_table(&self, verbose: bool) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== Probe Results ===");
        let _ = writeln!(out, "Weak < 0.4   Medium >= 0.4   Strong >= 0.8");
        if !verbose {
            let _ = writeln!(out, "(Showing core dimensions; use --verbose for all.)");
        }
        let _ = writeln!(out);
        let dims: &[&str] = if verbose {
            DIMENSION_NAMES
        } else {
            CORE_DIMENSION_NAMES
        };
        for &dim in dims {
            if let Some(probe) = self.dimension_result(dim) {
                let _ = writeln!(
                    out,
                    "{:<28}{:?}  {:.1} / {:.1}",
                    format!("{}:", display_name(dim)),
                    probe.level,
                    probe.score,
                    probe.max_score
                );
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
        let _ = writeln!(
            out,
            "{:<28}{}",
            "Can use tools:",
            if self.can_use_tools() { "yes" } else { "no" }
        );
        let _ = writeln!(
            out,
            "{:<28}{}",
            "Vision:",
            if self.supports_vision() {
                "supported"
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
        if let Some(n) = self.effective_context_tokens {
            let _ = writeln!(out, "{:<28}{n}", "Effective context tokens:");
        }
        if self.needs_xml_fallback() {
            let _ = writeln!(out, "{:<28}XML fallback needed", "");
        }
        if self.needs_json_repair() {
            let _ = writeln!(out, "{:<28}JSON repair needed", "");
        }
        out
    }

    /// Stderr line when a finished probe must exit 2 (`!can_use_tools`).
    pub fn tool_gate_error(&self) -> Option<&'static str> {
        if self.can_use_tools() {
            None
        } else {
            Some("error: cannot use tools (native and XML both Weak)")
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
/// Includes the count and up to [`MISSING_MODEL_ID_PREVIEW`] ids.
pub fn missing_model_message<S: AsRef<str>>(ids: &[S]) -> String {
    let n = ids.len();
    let mut msg =
        format!("error: --model is required unless GET /models returns exactly one id (got {n}");
    if n == 0 {
        msg.push(')');
        return msg;
    }
    msg.push_str(": ");
    for (i, id) in ids.iter().take(MISSING_MODEL_ID_PREVIEW).enumerate() {
        if i > 0 {
            msg.push_str(", ");
        }
        msg.push_str(id.as_ref());
    }
    if n > MISSING_MODEL_ID_PREVIEW {
        msg.push_str(", ...");
    }
    msg.push(')');
    msg
}
