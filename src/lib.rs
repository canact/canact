//! Not ready.

mod cache;
mod error;
mod export;
mod report;
mod types;

pub use cache::{
    CACHE_TTL_SECS, CacheEntry, DEFAULT_PROBE_EFFORT, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION,
    PROBE_SUITE_VERSION, ProbeCache,
};
pub use error::ProbeError;
pub use export::{
    AiderMetadataEntry, AiderOverlay, AiderSettingsRow, ClineModelInfo, HostOverlay, OverlayFiles,
    aider_edit_format, overlay_context_tokens, overlay_model_name,
};
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
}
