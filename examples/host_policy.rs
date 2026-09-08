//! Print a host-policy JSON envelope from an in-memory profile.
//!
//! No network and no API key.

include!("include/sample_profile.rs");

use canact::HostPolicyMeta;

fn main() {
    let profile = sample_profile();
    let meta = HostPolicyMeta::for_suite(true, false, canact::SuiteTier::Policy, Some(40_960));
    let envelope = profile.host_policy_envelope_with(meta);
    println!(
        "{}",
        serde_json::to_string_pretty(&envelope).expect("serialize host-policy JSON")
    );
}
