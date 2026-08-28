//! Not ready.

mod cache;
mod error;
mod types;

pub use cache::{
    CACHE_TTL_SECS, CacheEntry, DEFAULT_PROBE_EFFORT, PROBE_SUITE_VERSION, ProbeCache,
};
pub use error::ProbeError;
pub use types::{
    CORE_DIMENSION_NAMES, CapabilityLevel, CapabilityProfile, DIMENSION_NAMES,
    EditFormatRecommendation, ProbeResult, REQUIREMENT_DIMENSION_NAMES, classify,
};

#[cfg(feature = "runtime")]
mod client;
#[cfg(feature = "runtime")]
mod runner;

#[cfg(feature = "runtime")]
pub use client::{
    CatalogPriors, MockLlm, ProbeClient, ProbeContent, ProbeContentPart, ProbeFinish, ProbeMessage,
    ProbeRequest, ProbeResponse, ProbeRole, ProbeStreamChunk, ProbeTool, ProbeToolCall,
};
#[cfg(feature = "runtime")]
pub use runner::{FREE_CONCURRENCY, PAID_CONCURRENCY, ProbeRun, ProbeRunner, resolve_probe};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name() {
        assert_eq!(env!("CARGO_PKG_NAME"), "canact");
    }
}
