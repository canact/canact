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
canact = { version = "0.2", default-features = false, features = ["runtime"] }
```

MSRV is Rust 1.85.

## First probe

```bash
canact probe --provider ollama --model llama3.2:3b --cheap --json
```

`--cheap` is `--suite=policy` (host-policy fields, 4k ladder).
`--full` adds sequencing and the 8k/16k ladder. `--suite=all`
adds diagnostics.

Cloud hosts need a key before any HTTP call (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or
`--api-key`). Auth, a missing model, and connect failures abort
the suite.

`--json` prints the host-policy envelope. Dry runs that do not
call a model live in the repo [`examples/`](https://github.com/canact/canact/tree/main/examples)
directory.
