# canact

[![CI](https://github.com/canact/canact/actions/workflows/ci.yml/badge.svg)](https://github.com/canact/canact/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/canact?logo=rust)](https://crates.io/crates/canact)
[![docs.rs](https://img.shields.io/docsrs/canact?logo=docs.rs)](https://docs.rs/canact)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/canact/canact/badge)](https://securityscorecards.dev/viewer/?uri=github.com/canact/canact)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14503/badge)](https://www.bestpractices.dev/projects/14503)
[![FOSSA](https://github.com/canact/canact/actions/workflows/fossa.yml/badge.svg)](https://github.com/canact/canact/actions/workflows/fossa.yml)

Probe an LLM against this host's tools and return a capability card the
host can use: how many tools to send, which edit format to pick, whether
to enable XML fallback, and whether to wrap JSON in a repair layer.

Catalog flags (`supports_function_calling: true`, `context: 128k`) are
priors. canact spends seconds of real prompts on this model, this
template, and this tool schema, then writes host policy.

## Install

CLI:

```bash
cargo install canact --locked --features cli
```

Prebuilt archives ship on GitHub Releases (macOS, Linux, Windows x64).
After `v0.1.1`:

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/canact/canact/releases/latest/download/canact-installer.sh | sh
```

```bash
brew trust canact/tap
brew install canact/tap/canact
```

Windows (Scoop, x64):

```powershell
scoop bucket add canact https://github.com/canact/scoop-bucket
scoop install canact/canact
```

Library (runtime only, no CLI):

```toml
canact = { version = "0.1", default-features = false, features = ["runtime"] }
```

Default features are empty so a host pin does not pull clap. MSRV is
Rust 1.85.

## Getting started

```bash
canact probe --provider ollama --model llama3.2:3b --cheap --json
```

Cloud hosts need a key before any HTTP call (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or
`--api-key`). Auth, a missing model, and connect failures abort the
suite. Timeouts and 429/5xx stay session-local and are not cached.

`--json` prints the host-policy envelope. That object is not the
on-disk cache. The cache is `probes.json` (30-day TTL, keyed by
model, provider, effort, and suite version). `fromCache` is true
only on a cache hit. `cacheable` means the result may be stored
for 30 days.

Dry runs that do not call a model live in [`examples/`](examples/).

```bash
cargo run --locked --example host_policy
cargo run --locked --features cli --example export_overlays -- /tmp/canact-overlays
```

After a cached probe:

```bash
canact export --aider --model llama3.2:3b --provider ollama --dir /tmp/overlays
```

`canact mcp` is a stdio MCP server. The tool is `probe_model`. It
returns the same host-policy JSON as `canact probe --json`. Pass
`api_key_env` (the name of an env var), never the key itself.

## Library

Hosts implement `ProbeClient` and run `ProbeRunner`:

```rust
use canact::{ProbeClient, ProbeError, ProbeRunner};

async fn card(client: impl ProbeClient) -> Result<canact::CapabilityProfile, ProbeError> {
    ProbeRunner::new_throttled(client).run().await
}
```

Then read `max_tools()`, `best_edit_format()`, `needs_xml_fallback()`,
and `needs_json_repair()` on the profile. `ProbeError::Auth` aborts
the suite. Do not persist a Transient run. `ProbeCache` writes the
on-disk `probes.json` file.

## Host policy

| Field | Meaning |
|-------|---------|
| `maxTools` | Strong tool selection: no cap. Medium: 20. Weak: 10. |
| `probeLadderEditFormat` | Search/replace, unified diff, or whole file |
| `needsXmlFallback` | Native tools were Weak |
| `needsJsonRepair` | Completed JSON score is Medium or weaker |
| `useStreamingForToolCalls` | Streaming tool-call probe completed Medium or stronger |
| `supportsNestedToolArgs` | Nested-argument probe completed Medium or stronger |
| `verifiedParallelToolCalls` | Floor: at least N parallel `read_file` calls (probe asks for 5) |
| `agentLoop` | `full` / `assisted` / `single` from sequencing |
| `recommendedContextTokens` | Verified floor: `min(advertised, measured)`. Not a host window. Advertised alone is never used. |
| `cacheable` | Safe to persist for 30 days |
| `fromCache` | This print came from disk |

Agents that only have the repo URL should start at [`llms.txt`](llms.txt).

## License

Licensed under either of:

- MIT license ([LICENSE](LICENSE))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
