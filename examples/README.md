# Examples

These runs stay on disk. They do not call a model.

## Host-policy JSON

```bash
cargo run --locked --example host_policy
```

Prints the same envelope shape as `canact probe --json` (cacheable,
`fromCache`, `maxTools`, `recommendedContextTokens`).

## Aider and Cline overlays

```bash
cargo run --locked --features cli --example export_overlays -- /tmp/canact-overlays
```

Writes `.aider.model.settings.yml`, `.aider.model.metadata.json`, and
`cline.modelinfo.json`.

## CLI cache hit

After a real `canact probe` has written a cache file:

```bash
bash examples/probe-from-cache.sh \
  "${XDG_CACHE_HOME:-$HOME/.cache}/canact/probes.json" \
  qwen2.5-coder \
  ollama
```

That path is a cache hit only. It still needs a matching model and
provider in the file. A miss tries the network.
