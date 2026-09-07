# Contributing

## Where to start

- [Good first issues](https://github.com/canact/canact/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Help wanted](https://github.com/canact/canact/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)

Open an issue before a large change. Small, tested fixes can go
straight to a pull request.

## Local gate

The commands in `AGENTS.md` must pass on your workspace before you
open a pull request. Put user-visible changes under Unreleased in
`CHANGELOG.md`. In short:

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
The Semantic PR Title check enforces that.

This repository stays quiet until a maintainer says launch. Do not
add GitHub topics, a repo description, README badges, or a product
pitch in the README.

## License

This project is licensed under Apache-2.0. See `LICENSE`.

## Conduct

See `CODE_OF_CONDUCT.md`. Security reports go to `SECURITY.md`, not
a public issue.
