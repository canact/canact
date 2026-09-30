//! Print a policy-suite host-policy JSON envelope.
//!
//! Sequencing is skipped, so `agentLoop` is null. No network and no API key.

include!("include/policy_sample.rs");

use canact::HostPolicyMeta;

fn main() {
    let profile = policy_sample();
    let meta = HostPolicyMeta::for_suite(true, false, canact::SuiteTier::Policy, Some(40_960));
    let envelope = profile.host_policy_envelope_with(meta);
    println!(
        "{}",
        serde_json::to_string_pretty(&envelope).expect("serialize host-policy JSON")
    );
}
