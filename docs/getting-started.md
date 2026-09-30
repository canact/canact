# Getting started

## Install

```bash
cargo install canact --locked --features cli
```

Prebuilt archives ship on GitHub Releases (macOS, Linux, Windows x64):

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
canact = { version = "0.9", default-features = false, features = ["runtime"] }
```

MSRV is Rust 1.95.

## First probe

```bash
canact probe --provider ollama --model llama3.2:3b --cheap --json
```

`--cheap` is `--suite=policy` (host-policy fields, 4k ladder).
`--full` adds sequencing and the 8k/16k ladder. `--suite=all`
adds diagnostics.

Cloud hosts need a key before any HTTP call. The first match in
this table wins. `--api-key` is visible in shell history and the
process list; prefer an env var. `canact probe --no-login` skips
stored Grok and Claude Code logins. Env vars still apply.
`canact probe --dry-run` prints the provider, redacted base URL,
suite, and probe names, then exits. It does not read a login, the
cache, or the network.

| Provider label | Default URL when `--base-url` is omitted | What is tried, first match wins | Stored login |
| --- | --- | --- | --- |
| empty | `https://api.openai.com/v1`, unless a key selects the xAI, Anthropic, or OpenRouter host | `--api-key`, else `OPENAI_API_KEY`, else `XAI_API_KEY` then `GROK_API_KEY` then Grok login, else `ANTHROPIC_AUTH_TOKEN` then `ANTHROPIC_API_KEY` then Claude Code login, else `OPENROUTER_API_KEY` | Grok login, then Claude Code, when no env key and no `--api-key`. `--no-login` skips only those two helpers. |
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

Auth, a missing model, and connect failures abort
the suite.

`--json` prints the host-policy envelope. Dry runs that do not
call a model live in the repo [`examples/`](https://github.com/canact/canact/tree/main/examples)
directory.
