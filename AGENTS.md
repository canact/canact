# AGENTS

> **Human contributors:** This file is for AI coding assistants.
> You can safely ignore it. See README.md and CONTRIBUTING.md instead.

Local gate: `make check`

Targeted probe tests:
`cargo test --locked --features runtime --lib <filter>`
`cargo test --lib` without `runtime` compiles and runs 0 probe tests.

DCO: `git commit -s`

PR titles must be conventional (`feat` / `fix` / `perf` bump a
release; `docs` / `chore` / `test` / `ci` do not). Do not merge a
release-please PR (`autorelease: pending`) without an explicit
human yes. Curated GitHub Release notes are
`docs/releases/vX.Y.Z.md` (file stays; no cleanup PR).

Keep GitHub About, topics, and README in sync with the launched
product. Do not publish to crates.io until the maintainer publishes.

Do not depend on `bline-llm`, `bline-types`, `bline-probe`, or other
`bline-*` crates.

Do not open pull requests or issues on `blineai/bline`. Bline may
consume canact; canact never drives Bline.

MSRV 1.85. Edition 2024.

See `CONSTITUTION.md`.
