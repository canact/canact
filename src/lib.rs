//! Not ready.

mod types;
pub use types::{
    CORE_DIMENSION_NAMES, CapabilityLevel, CapabilityProfile, DIMENSION_NAMES,
    EditFormatRecommendation, ProbeResult, REQUIREMENT_DIMENSION_NAMES, classify,
};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name() {
        assert_eq!(env!("CARGO_PKG_NAME"), "canact");
    }
}
