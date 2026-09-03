//! Compile surface for Bline's pin: `default-features = false`
//! plus optional `runtime`. Overlay / MCP types must not be named here.
//! `cargo test --locked` (no `cli`) is the proof.

use canact::{CapabilityProfile, ProbeCache};

#[test]
fn bline_pin_sees_profile_and_cache() {
    let cache = ProbeCache::default();
    assert!(cache.find_profile("missing", "none").is_none());
    assert_eq!(
        std::any::type_name::<CapabilityProfile>(),
        "canact::types::CapabilityProfile"
    );
}
