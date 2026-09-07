# Security

Report vulnerabilities privately through GitHub Security Advisories:

https://github.com/canact/canact/security/advisories/new

Do not open a public issue for a security report.

We aim to acknowledge a report within 7 days. Include enough detail
to reproduce the issue (version or commit, steps, and impact). Credit
in `CHANGELOG.md` when a fix ships.

## Design notes

canact is a probe client. It talks to LLM HTTP APIs the operator
already chose. It does not implement custom cryptography. TLS is
rustls via reqwest. Secrets stay in the environment or in a local
login store; they are not written to `probes.json`.

| Claim | How we check it |
|---|---|
| No custom crypto | rustls + reqwest; no project cipher code |
| Auth failures abort | `ProbeError::Auth` and missing-model abort the suite |
| Cache is not a secret store | `probes.json` is host-policy JSON, 30-day TTL |
| Supply chain | cargo-deny, Dependabot, CodeQL, Scorecard, FOSSA |
| Private reports | GitHub Security Advisories, not public issues |
| Release archives | Cosign `.sigstore.json` plus SLSA `.intoto.jsonl` on the GitHub Release |

## Verifying a GitHub Release archive

GitHub Release archives include two extra files next to each asset:

- `ASSET.sigstore.json` is a Cosign keyless signature bundle
- `ASSET.intoto.jsonl` is SLSA build provenance from GitHub Attestations

Download the archive and both files, then:

```bash
gh attestation verify ./canact-x86_64-unknown-linux-gnu.tar.xz \
  --repo canact/canact

cosign verify-blob \
  --bundle ./canact-x86_64-unknown-linux-gnu.tar.xz.sigstore.json \
  --certificate-identity-regexp \
    '^https://github.com/canact/canact/.github/workflows/(release|sign-release)\.yml@refs/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ./canact-x86_64-unknown-linux-gnu.tar.xz
```

Git tags are not GPG-signed. These signatures cover Release assets, not
the git tag object.
