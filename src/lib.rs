//! Not ready.

mod cache;
mod error;
mod report;
mod types;

pub use cache::{
    CACHE_TTL_SECS, CacheEntry, DEFAULT_PROBE_EFFORT, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION,
    PROBE_SUITE_VERSION, ProbeCache,
};
pub use error::ProbeError;
pub use report::missing_model_message;
pub use types::{
    CORE_DIMENSION_NAMES, CapabilityLevel, CapabilityProfile, DIMENSION_NAMES,
    EditFormatRecommendation, HostPolicyMeta, ProbeResult, REQUIREMENT_DIMENSION_NAMES,
    TOOL_PROBE_NAMES, classify,
};

#[cfg(feature = "runtime")]
mod client;
#[cfg(feature = "runtime")]
mod probes;
#[cfg(feature = "runtime")]
mod runner;

#[cfg(feature = "runtime")]
pub use client::{
    CatalogPriors, MockLlm, ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeMessage,
    ProbeRequest, ProbeResponse, ProbeRole, ProbeStreamChunk, ProbeTool, ProbeToolCall, ProbeUsage,
};
#[cfg(feature = "runtime")]
pub use runner::{FREE_CONCURRENCY, PAID_CONCURRENCY, ProbeRun, ProbeRunner, resolve_probe};

#[cfg(all(feature = "runtime", feature = "openai"))]
mod adapters;
#[cfg(all(feature = "runtime", feature = "openai"))]
pub use adapters::openai::{OpenAiCompatClient, list_model_ids};

// Aider/Cline overlays and MCP stay off the Bline pin
// (`default-features = false`, optional `runtime`).
#[cfg(feature = "cli")]
mod export;
#[cfg(feature = "cli")]
pub use export::{
    AiderMetadataEntry, AiderOverlay, AiderSettingsRow, ClineModelInfo, HostOverlay, OverlayFiles,
    aider_edit_format, overlay_context_tokens, overlay_model_name,
};

#[cfg(feature = "cli")]
mod mcp;
#[cfg(feature = "cli")]
pub use mcp::run_mcp_stdio;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name() {
        assert_eq!(env!("CARGO_PKG_NAME"), "canact");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn cli_exports_host_overlay() {
        let _ = crate::HostOverlay::aider;
        let _ = crate::run_mcp_stdio;
    }
}
