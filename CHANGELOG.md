# Changelog

Record user-visible changes here. Add a dated version heading when a
release ships. Until then, put work under Unreleased.

## [Unreleased]

- MCP `probe_model` does not GET a cloud catalog without an API key.
- Claude Code login token refresh tries the second host on connect failure as well as HTTP 404.
- HTTP error bodies redact `xai-` keys the same way as `sk-` / `gsk_`.
- `canact mcp` `probe_model` picks the API key from `provider`, not from
  whichever `*_API_KEY` is set first.
- MCP `api_key_env` that is unset or empty names that variable in the
  error instead of listing unused fallbacks.
- Dual-licensed as MIT OR Apache-2.0 (`LICENSE` plus `LICENSE-APACHE`).
- GitHub Release archives include Cosign `.sigstore.json` bundles and
  SLSA `.intoto.jsonl` provenance.

## [0.1.1] - 2026-09-07

- cargo-dist GitHub Release archives for macOS, Linux, and Windows x64.
- Shell and PowerShell installers from the GitHub Release.
- Homebrew formula push to `canact/homebrew-tap` after those archives exist.
- Scoop bucket rewrite in `canact/scoop-bucket` (Windows x64 zip).
- crates.io and docs.rs README badges.
- `llms.txt` install line: `cargo install canact --locked --features cli`.
- OpenSSF Best Practices listing and README badge.
- FOSSA license scan workflow and filter script.
- cargo-fuzz targets plus a CI smoke job.

## [0.1.0] - 2026-09-07

First crates.io release.

- Launch README, `llms.txt`, GitHub About, and crate keywords.
- `canact` with no subcommand prints help instead of `Not ready.`
- Dry `examples/` for host-policy JSON and Aider/Cline overlay files.
- Community files, DCO, and stealth-safe CI (CodeQL, Dependency Review,
  Scorecard, lychee, semantic PR titles, actionlint, zizmor, cargo-deny).
- Org avatar and repository social preview (source: `docs/brand/canact.svg`).
