# Contributing

## Where to start

- [Good first issues](https://github.com/canact/canact/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Help wanted](https://github.com/canact/canact/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

Open an issue before a large change. Small, tested fixes can go
straight to a pull request.

## Local gate

The commands in `AGENTS.md` must pass on your workspace before you
open a pull request. In short:

```bash
make check
```

Copy-paste runs that do not call a model live in `examples/`.

Every commit needs a Developer Certificate of Origin trailer:

```bash
git commit -s
```

The sign-off email is `git config user.email`. The DCO workflow skips
bot commits and merge commits.

## Pull requests

Use the pull request template. Commits on `main` squash through the
required checks (Lint, Test, Stealth, Workflow sanity, DCO).

PR titles must be a conventional type (`feat`, `fix`, `docs`, `ci`,
`chore`, `test`, `refactor`, `perf`, `build`, `style`, `revert`).
The Semantic PR Title check enforces that. After squash-merge, that
title is what release-please reads:

| Title prefix | Next version |
|--------------|--------------|
| `feat` / `feat!` | minor (0.x while pre-1.0) |
| `fix` / `perf` | patch |
| `docs` / `chore` / `test` / `ci` / `refactor` | changelog only, no bump |

release-please opens a `chore(main): release X.Y.Z` PR and writes
`CHANGELOG.md`. That PR is labeled `autorelease: pending`. Do not
auto-merge it. Merging it creates the git tag and starts cargo-dist
(GitHub Release archives, Homebrew, Scoop). crates.io is still a
manual `cargo publish` by the maintainer.

Optional curated GitHub Release notes: add `docs/releases/vX.Y.Z.md`
on `main` (a `docs:` PR is enough). The file stays. There is no
cleanup PR and no `RELEASE_NOTES.md` at the repo root. cargo-dist
host copies that file onto the Release page if it exists; otherwise
the auto changelog stays. To change notes after the tag without
rebuilding archives:

```bash
gh workflow run "Apply release notes" -f tag=vX.Y.Z
```

A `docs:` notes PR does not rewrite the release-please changelog.
Merge the notes PR, then merge the release PR when you want the cut.
Do not put notes on the `release-please--*` branch (that head is
force-pushed).

## License

This project is dual-licensed under MIT or Apache-2.0. You may choose
either. See `LICENSE` (MIT) and `LICENSE-APACHE`.

## Conduct

See `CODE_OF_CONDUCT.md`. Security reports go to `SECURITY.md`, not
a public issue.
