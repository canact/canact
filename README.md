# canact

[![CI](https://github.com/canact/canact/actions/workflows/ci.yml/badge.svg)](https://github.com/canact/canact/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/canact?logo=rust)](https://crates.io/crates/canact)
[![docs.rs](https://img.shields.io/docsrs/canact?logo=docs.rs)](https://docs.rs/canact)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/canact/canact/badge)](https://securityscorecards.dev/viewer/?uri=github.com/canact/canact)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14503/badge)](https://www.bestpractices.dev/projects/14503)
[![FOSSA](https://github.com/canact/canact/actions/workflows/fossa.yml/badge.svg)](https://github.com/canact/canact/actions/workflows/fossa.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://canact.github.io/canact/)

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
canact = { version = "0.10", default-features = false, features = ["runtime"] }
```

Default features are empty so a host pin does not pull clap. MSRV is
Rust 1.95.

## Getting started

```bash
canact probe --provider ollama --model llama3.2:3b --cheap --json
```

The built-in Ollama URL is `http://127.0.0.1:11434/v1`. That string
is the crate constant `OLLAMA_BASE_URL`. Nothing reads an environment
variable of that name. Pass `--base-url` to use a different URL.
`openai-codex` and `codex` dial the shipped Responses profile at
`https://api.openai.com`. The grok-build messages labels dial
`https://cli-chat-proxy.grok.com`. A different `--base-url` is an
error for those labels. Amazon Bedrock dials
`bedrock-runtime.$AWS_REGION`. `--base-url` does not select that
region.

`--cheap` is `--suite=policy` (host-policy fields, 4k ladder).
`--full` adds sequencing and the 8k/16k ladder. `--suite=all`
adds diagnostics (`code_syntax`, token efficiency, system
message, memory).

Cloud hosts need a key before any HTTP call. The first match in
this table wins. `--api-key` is visible in shell history and the
process list; prefer an env var. `canact probe --no-login` skips
stored Grok and Claude Code logins. Env vars still apply.
`canact probe --dry-run` prints the provider, redacted base URL,
suite, and probe names, then exits. It does not read a login, the
cache, or the network.
`canact probe --fail-on weak` exits 2 when a finished row is Weak.
`--fail-on degraded` also exits 2 for Medium. A skipped row or a
probe error does not count. An unknown value exits 1.

| Provider label | Default URL when `--base-url` is omitted | What is tried, first match wins | Stored login |
| --- | --- | --- | --- |
| empty | `https://api.openai.com/v1`, unless a key selects the xAI, Anthropic, or OpenRouter host | `--api-key`, else `OPENAI_API_KEY`, else `XAI_API_KEY` then `GROK_API_KEY` then Grok login, else `ANTHROPIC_AUTH_TOKEN` then `ANTHROPIC_API_KEY` then Claude Code login, else `OPENROUTER_API_KEY` | Grok login unless `--api-key`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`, or `GROK_API_KEY` is set. An Anthropic env key does not skip it. Claude Code login runs only when nothing earlier matched. `--no-login` skips only those two helpers. |
| `xai`, `grok`, `api.x.ai`, `x.ai` | `https://api.x.ai/v1` | `--api-key`, else `XAI_API_KEY`, else `GROK_API_KEY`, else Grok login. Ignores `OPENAI_API_KEY`. | `~/.grok/auth.json`. A named xAI provider still loads it when `OPENAI_API_KEY` is set. |
| `grok-build`, `xai-grok-build`, `cli-chat-proxy.grok.com`, and the `-messages` pair | `https://cli-chat-proxy.grok.com/v1` | Same xAI pack | Same Grok login |
| `claude`, `anthropic`, `api.anthropic.com` | `https://api.anthropic.com/v1` | `--api-key`, else `ANTHROPIC_AUTH_TOKEN`, else `ANTHROPIC_API_KEY`, else Claude Code. Ignores OpenAI and xAI. | Claude Code login |
| `openrouter`, `openrouter.ai` | `https://openrouter.ai/api/v1` | `--api-key`, else `OPENAI_API_KEY`, else `OPENROUTER_API_KEY`. No stored login. | none |
| `groq`, `api.groq.com` | `https://api.groq.com/openai/v1` | `--api-key`, else `GROQ_API_KEY`. No stored login. | none |
| `amazon-bedrock`, `bedrock` | `https://bedrock-runtime.us-east-1.amazonaws.com` | `--api-key`, else `AWS_BEARER_TOKEN_BEDROCK`. No stored login. | none |
| `openai-codex`, `codex` | `https://api.openai.com/v1` | `--api-key`, else `OPENAI_API_KEY`. No stored login. | none |
| `ollama`, `localhost`, `127.0.0.1`, `::1`, `0.0.0.0` | `http://127.0.0.1:11434/v1` (this constant, not an environment variable) | none | none |
| `lmstudio` | `http://127.0.0.1:1234/v1` | none | none |
| `vllm` | `http://127.0.0.1:8000/v1` | none | none |

Cache list, matrix, and export treat the labels in one row as one
card. On the grok-build row, the chat labels share a card and the
messages labels share a different card.

Auth, a missing model, and connect failures abort the
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
canact export --all --model llama3.2:3b --provider ollama --dir /tmp/overlays
canact matrix --provider ollama
canact matrix
```

`canact export --aider` writes `.aider.model.settings.yml` and `.aider.model.metadata.json` into `--dir` (the current directory when `--dir` is omitted). Aider loads those two files from the repo. `canact export --cline` writes `cline.modelinfo.json` in that same directory. Paste that file into Cline. Cline does not load it from the repo. `canact export --all` writes all three files and leaves stdout empty. Each `wrote` line is on stderr. Overlay windows are the advertised context stored on the cache row, or the value passed to `--advertised-context`.

`canact matrix` prints pretty JSON from the cache. Each cell is
`pass`, `degraded`, `fail`, or `skipped`. There is no `--json` flag
and no human table. `--provider` is optional; omit it to include
every cached provider. It does not call a model and has no composite
score. `skipped` means the cell is not a pass or a fail.
`constraintPlacement` is skipped unless `--suite=all`.
`maxOutputTokens` is measured on every suite, including policy.
A policy cell is `skipped` when that probe does not find a cap.
`--full` does not add the probe. A missing cap on full or all is
`fail`.

`canact mcp` is a stdio MCP server. The tool is `probe_model`. It
returns the same host-policy JSON as `canact probe --json`. Start it
with `--api-key-env` (the env var name), `--base-url`,
`--allow-base-url`, and `--allow-cache`. The tool must not receive
a raw key.

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

`ProbeFinish` is `non_exhaustive`. Match it with a wildcard arm so a
later patch can add a finish reason. Score that arm like `Other`:
a completed reply can be cached. Handle `Safety` on its own arm
(transient, not a capability score). Handle an empty `Malformed`
on its own arm (completed tool failure).

`OpenAiCompatClient` needs `runtime` and `openai`. The pin above
stays `runtime` only, which is enough for `ProbeRunner`. The key
below is `None` for local Ollama. Do not put a raw key in source.

```toml
canact = { version = "0.10", default-features = false, features = ["runtime", "openai"] }
```

```rust
use canact::{CatalogPriors, OpenAiCompatClient, ProbeRunner};

let client = OpenAiCompatClient::new(
    "http://127.0.0.1:11434/v1",
    None,
    "llama3.2:3b",
    "ollama",
    CatalogPriors::default(),
)
.expect("client");
let _runner = ProbeRunner::new_throttled(client);
```

## Host policy

The table is the fields a host branches on. The other envelope
fields are in [Read the card](docs/read-the-card.md).

| Field | Meaning |
|-------|---------|
| `maxTools` | Strong is JSON `null` (no cap). Medium is 20. Weak or not completed is 10. Read `toolSelectionStatus` before treating 10 as measured Weak. |
| `probeLadderEditFormat` | Search/replace, unified diff, or whole file |
| `needsXmlFallback` | Native tools were Weak |
| `needsJsonRepair` | Completed JSON score is Medium or weaker |
| `useStreamingForToolCalls` | Streaming tool-call probe completed Medium or stronger |
| `supportsNestedToolArgs` | Nested-argument probe completed Medium or stronger |
| `verifiedParallelToolCalls` | Floor: at least N parallel `read_file` calls (probe asks for 5) |
| `agentLoop` | `full` / `assisted` / `single` when sequencing completed. JSON `null` when sequencing was skipped (a policy run). |
| `recommendedContextTokens` | Verified floor: `min(advertised, measured)`. Not a host window. Advertised alone is never used. |
| `maxOutputTokens` | Measured provider output cap. Never the input window. Omitted until measured. |
| `cacheable` | Safe to persist for 30 days |
| `fromCache` | This print came from disk |
| `constraintPlacement` | `--suite=all` only. `system` if the model follows the system prompt (Medium or stronger). `user` if Weak: put hard constraints in the user turn. Not a rank. |

Agents that only have the repo URL should start at [`llms.txt`](llms.txt).

## License

Licensed under either of:

- MIT license ([LICENSE](LICENSE))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
