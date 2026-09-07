#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = canact::provider_from_base_url(data);
    let _ = canact::local_provider_base_url(data);
    let _ = canact::is_xai_provider_label(data);
    let _ = canact::is_anthropic_provider_label(data);
    let _ = canact::is_anthropic_cloud_host(data);
    let _ = canact::is_ollama_compat_base(data);
    let _ = canact::cloud_endpoint_requires_key(data);
    let _ = canact::looks_cheap(data, data, data);
});
