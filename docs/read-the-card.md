# Read the card

The first command is:

```bash
canact probe --provider ollama --model llama3.2:3b --cheap --json
```

The built-in Ollama URL is `http://127.0.0.1:11434/v1`. That string
is the crate constant `OLLAMA_BASE_URL`. Nothing reads an environment
variable of that name. Pass `--base-url` to use a different URL.

## Definitions

A host is the program that will call the model. That is the CLI user,
Aider, Cline, or a `ProbeClient`. The card is for that host.

Host policy is the set of envelope fields a host branches on: how
many tools to send, which edit format to pick, whether to fall back
to XML, and whether to repair JSON. It is not a rank of the model.

The fixed probe tool set is what canact sends. `tool_selection` sends
eight names: `read_file`, `edit_file`, `doc_set`, `search`,
`run_command`, `list_dir`, `md_replace_section`, and `write_file`.
`parallel_tool_scale` sends `read_file` only. That set is not the
caller's schema. This page does not describe a host-supplied tool
list.

The context ladder climbs 4096, then 8192, then 16384. `--cheap` and
`--suite=policy` stop after 4096. `--full` and `--suite=all` continue
through 8192 and 16384.

`canact probe --json` prints the host-policy envelope. The cache file
is `probes.json`. Entries last 30 days. The cache key includes the
suite version (currently 97). `fromCache` is true only when this
print came from that file. `cacheable` means the result may be
stored. Do not paste the envelope over `probes.json`.

## Envelope fields

The [Getting started](getting-started.md) page and the README
host-policy table list the fields a host branches on. The envelope
also always includes these:

| Field | Meaning |
| --- | --- |
| `model` | Model id that was probed |
| `provider` | Provider label |
| `overall` | Lowest measured level among tool calling, JSON output, and instruction following (`weak`, `medium`, or `strong`). `weak` when none of those three were measured. |
| `canUseTools` | Native or XML tool calling completed Medium or stronger |
| `supportsVision` | The vision probe completed Medium or stronger |
| `toolSelectionStatus` | `completed`, `skipped`, `unprobed`, or `error` for `tool_selection`. Read this before treating `maxTools` 10 as measured Weak. |
| `effectiveContextTokens` | Measured usable context, in tokens, or JSON `null` until a suite writes it |
| `probedContextFloor` | Highest passing ladder rung when the climb is incomplete, or JSON `null` |
| `skipExpensive` | True on the policy suite |
| `suite` | `policy`, `full`, or `all` |
| `advertisedContextTokens` | Catalog or flag prior, or JSON `null` |
| `probedAt` | Unix seconds when the card was stored |
| `scoreScale` | `min` 0.0, `max` 1.0, `strongMin` 0.8, `mediumMin` 0.4 |
| `probes` | Policy dimensions. `multiTurnTaskSequencing` is present on policy with status `skipped`. |
| `diagnostics` | Empty on policy and full. Populated on `--suite=all`. |

`maxOutputTokens` is inserted only when it was measured. It is
omitted otherwise. `constraintPlacement` is inserted only on
`--suite=all`, and only when a placement was measured. Both fields
are already in the README host-policy table.

On a policy run, `agentLoop` is JSON `null` and
`probes.multiTurnTaskSequencing.status` is `skipped`. A full suite
that completes sequencing sets `agentLoop` to `full`, `assisted`,
or `single`.

## Exit code 2

When the model cannot use tools, canact prints the envelope on
stdout first, then this line on stderr, then exits 2:

```text
error: cannot use tools (native and XML both failed to complete)
```

JSON on stdout is still valid. A shell with `set -e` stops on that
exit code.

## Library

`ProbeRunner` stays on feature `runtime`. `ProbeRunner::new` is the
full suite and paid concurrency (64). `ProbeRunner::new_throttled`
is the policy suite and free concurrency (3). The README example
uses `new_throttled`. `.suite()` changes the suite only.
`.throttled()` changes concurrency only. `.cheap()` and `.full()`
set both.

A snippet that constructs `OpenAiCompatClient` needs features
`runtime` and `openai`. The key below is `None` because this URL is
local Ollama. Do not put a raw key in the source.

```toml
canact = { version = "0.9", default-features = false, features = ["runtime", "openai"] }
```

```rust
use canact::{CatalogPriors, OpenAiCompatClient};

let _client = OpenAiCompatClient::new(
    "http://127.0.0.1:11434/v1",
    None,
    "llama3.2:3b",
    "ollama",
    CatalogPriors::default(),
)
.expect("client");
```

## Matrix

`canact matrix` prints pretty JSON. Each cell is `pass`, `degraded`,
`fail`, or `skipped`. There is no `--json` flag and no human table.
`--provider` is optional. Omit it to include every cached provider.
The command does not call a model.

## MCP

```bash
canact mcp --api-key-env XAI_API_KEY --base-url http://127.0.0.1:11434/v1
```

`--api-key-env` is the name of an environment variable. The tool
cannot name a different variable, and it must not receive a raw key.
`--base-url` is the probe URL. The tool cannot replace it.
`--allow-base-url` lets the tool pass a base URL that is not a
loopback host. `--allow-cache` lets the tool pass a cache path
outside the default cache directory.

## Other commands

The credential table is in [Getting started](getting-started.md).
`canact probe --no-login` skips stored Grok and Claude Code logins.
Environment variables and `--api-key` still apply. See
[#288](https://github.com/canact/canact/issues/288).

`canact cache path` prints the cache file. `canact cache list`
prints the stored rows. This page does not copy that output. See
[#287](https://github.com/canact/canact/issues/287).

`canact export --all` writes the Aider pair and `cline.modelinfo.json`
and leaves stdout empty. The file list stays in the
[README](https://github.com/canact/canact/blob/main/README.md). See
[#289](https://github.com/canact/canact/issues/289).

`canact probe --dry-run` prints the provider, redacted base URL,
suite, and probe names, then exits. It does not read a login, the
cache, or the network. See
[#290](https://github.com/canact/canact/issues/290).

`canact probe --fail-on weak` exits 2 when a finished row is Weak.
`--fail-on degraded` also exits 2 for Medium. A skipped row or a
probe error does not count. An unknown value exits 1.
