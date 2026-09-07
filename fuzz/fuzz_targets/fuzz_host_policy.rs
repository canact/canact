#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let Ok(profile) = serde_json::from_str::<canact::CapabilityProfile>(data) else {
        return;
    };
    let _ = profile.overall_level();
    let _ = profile.max_tools();
    let _ = profile.best_edit_format();
    let _ = profile.needs_xml_fallback();
    let _ = profile.needs_json_repair();
    let _ = profile.can_use_tools();
    let _ = profile.supports_vision();
    let _ = profile.recommended_context_tokens(None);
    let _ = profile.host_policy_envelope();
});
