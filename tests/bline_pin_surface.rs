//! Compile surface for Bline's pin: `default-features = false`
//! plus optional `runtime`. Overlay / MCP types must not be named here.
//! `cargo test --locked` (no `cli`) is the proof.

use canact::{CapabilityProfile, ProbeCache, ProbeError};

#[test]
fn bline_pin_sees_profile_and_cache() {
    let cache = ProbeCache::default();
    assert!(cache.find_profile("missing", "none").is_none());
    assert_eq!(
        std::any::type_name::<CapabilityProfile>(),
        "canact::types::CapabilityProfile"
    );
    let p = CapabilityProfile::unprobed("m", "p");
    assert_eq!(p.model_id, "m");
    assert_eq!(p.max_output_tokens, None);
}

#[test]
fn bline_pin_sees_not_found_classifier() {
    assert!(matches!(
        ProbeError::from_http(404, "missing"),
        Some(ProbeError::NotFound(_))
    ));
    assert!(ProbeError::not_found_from_body("invalid json schema").is_none());
}

#[cfg(feature = "runtime")]
#[test]
fn bline_pin_sees_runtime_host_helpers() {
    let _ = canact::strip_think_blocks;
    let _ = canact::finish_from_reason;
    assert_eq!(canact::strip_think_blocks("<think>hid</think>ok"), "ok");
    assert_eq!(
        canact::finish_from_reason("length"),
        canact::ProbeFinish::Length
    );
}
