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
