#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _: Result<canact::ProbeCache, _> = serde_json::from_str(data);
});
