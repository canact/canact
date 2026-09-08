# Changelog

release-please writes version headings from conventional commit
titles on `main`. Do not keep an Unreleased section.

## [0.1.2](https://github.com/canact/canact/compare/v0.1.1...v0.1.2) (2026-09-08)


### Bug Fixes

* redact xai- keys in HTTP error bodies ([#156](https://github.com/canact/canact/issues/156)) ([6591a28](https://github.com/canact/canact/commit/6591a280d739163cd8ca0957a954a454d7df3d72))
* refuse MCP catalog without a key; Claude refresh on connect fail ([#158](https://github.com/canact/canact/issues/158)) ([8e2af68](https://github.com/canact/canact/commit/8e2af6811ed616ba49bf890762598c466aa14ebe))
* route MCP keys by provider ([#155](https://github.com/canact/canact/issues/155)) ([1ad1780](https://github.com/canact/canact/commit/1ad1780b5ba556b588e96b2cc0eed5a69dd03307))
* Scoop checkver regex should match digits ([#152](https://github.com/canact/canact/issues/152)) ([3727eba](https://github.com/canact/canact/commit/3727eba5813d4c53f4dbd5adf07f3313c72f975e))

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
