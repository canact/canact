# Architecture

canact is one crate: a library plus an optional `canact` binary.

```text
src/lib.rs          public types and re-exports
  cache             probes.json (30-day TTL, not the CLI envelope)
  endpoint          provider URL and host-family hints
  types             CapabilityProfile, levels, host-policy fields
  error             Auth / NotFound abort; Transient stays session-local
  runtime           ProbeClient, ProbeRunner, graders
  adapters/openai   OpenAI-compat, Anthropic, Ollama, xAI
src/bin/canact.rs   CLI, export, and `canact mcp` (feature `cli`)
  export            Aider / Cline overlays
```

Default features are empty. A library pin uses `default-features =
false` plus `runtime`. The binary needs `--features cli`.

The host-policy card is the product: `max_tools`, edit-format
ladder, XML fallback, and JSON repair. Graders live next to the
probes they score. Types and CLI goldens are the source of truth.

Tests: `cargo test --locked --features runtime --lib`. Local gate:
`make check`.
