# canact

Probe an LLM against this host's tools and return a capability card
the host can use: how many tools to send, which edit format to pick,
whether to enable XML fallback, and whether to wrap JSON in a repair
layer.

Catalog flags (`supports_function_calling: true`, `context: 128k`)
are priors. canact spends seconds of real prompts on this model, this
template, and this tool schema, then writes host policy.

- Repo: [canact/canact](https://github.com/canact/canact)
- Crate: [crates.io/crates/canact](https://crates.io/crates/canact)
- API docs: [docs.rs/canact](https://docs.rs/canact)
- Releases: [GitHub Releases](https://github.com/canact/canact/releases)

Default features are empty. A library pin uses
`default-features = false` plus `runtime`. The CLI and MCP server
need `--features cli`.
