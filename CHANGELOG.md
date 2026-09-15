# Changelog

release-please writes version headings from conventional commit
titles on `main`. Do not keep an Unreleased section.

## [0.6.0](https://github.com/canact/canact/compare/v0.5.0...v0.6.0) (2026-09-15)


### Features

* consume wiremux 0.3.0 ([#226](https://github.com/canact/canact/issues/226)) ([0b2ee16](https://github.com/canact/canact/commit/0b2ee16ff8befee3cd42087ded38bd5650870f85))
* consume wiremux 0.4.0 ([#228](https://github.com/canact/canact/issues/228)) ([79ce145](https://github.com/canact/canact/commit/79ce145d73f8c1f88cbfe0c80710ac18e3200d03))
* use Grok login when XAI_API_KEY is unset ([#227](https://github.com/canact/canact/issues/227)) ([e401cbd](https://github.com/canact/canact/commit/e401cbdba2a7dd890088d379188777c2ba854c5b))


### Bug Fixes

* print advertised window and unprobed vision in human table ([#225](https://github.com/canact/canact/issues/225)) ([f568108](https://github.com/canact/canact/commit/f568108f51b041b035894f96fb7d470d612814e2))
* read USER Claude keychain and ignore catalog vision false ([#224](https://github.com/canact/canact/issues/224)) ([adb30f2](https://github.com/canact/canact/commit/adb30f2a86d8a5dd27c584fcafa97e3ada52a409))
* recover advertised window and name missing cloud keys ([#222](https://github.com/canact/canact/issues/222)) ([dd0f44c](https://github.com/canact/canact/commit/dd0f44c630e3e1e2694787ae7b91650bff748e72))
* treat MCP GROK_API_KEY as the xAI family ([#229](https://github.com/canact/canact/issues/229)) ([afab08b](https://github.com/canact/canact/commit/afab08b359215118560fedd8d0a74eb780927c67))

## [0.5.0](https://github.com/canact/canact/compare/v0.4.0...v0.5.0) (2026-09-14)


### Features

* add typed NotFound and ProbeResponse host constructors ([#221](https://github.com/canact/canact/issues/221)) ([41cb978](https://github.com/canact/canact/commit/41cb9786b74c54cbb37a5c06aa9135e9681767c2)), closes [#219](https://github.com/canact/canact/issues/219) [#220](https://github.com/canact/canact/issues/220)
* export NotFound classifier and think-strip helpers ([#216](https://github.com/canact/canact/issues/216)) ([9c270c4](https://github.com/canact/canact/commit/9c270c4b51ea0fe288bde9796bc12bea4e4fb4dd))

## [0.4.0](https://github.com/canact/canact/compare/v0.3.0...v0.4.0) (2026-09-14)


### Features

* consume wiremux 0.2.1 for LLM connect ([#211](https://github.com/canact/canact/issues/211)) ([63fc98a](https://github.com/canact/canact/commit/63fc98abb8f4eb8615a21ab3308a02e3a2bdbba9))


### Bug Fixes

* abort unknown-model 400 and log refresh merge errors ([#208](https://github.com/canact/canact/issues/208)) ([93564d5](https://github.com/canact/canact/commit/93564d5d33134bbf90de3b3653dfdfc10d1c31c8))
* do not send OPENAI_API_KEY to xAI or Anthropic hosts ([#213](https://github.com/canact/canact/issues/213)) ([1fd7c93](https://github.com/canact/canact/commit/1fd7c9327ed8f59b0b5c80b0a0547b399e7b06a2))
* keep send-timeouts session-local and restore think-tag strip ([#212](https://github.com/canact/canact/issues/212)) ([65ce753](https://github.com/canact/canact/commit/65ce753fe7f7f55ceb1fbafe62c7291f9f871762))

## [0.3.0](https://github.com/canact/canact/compare/v0.2.0...v0.3.0) (2026-09-09)


### Features

* public CapabilityProfile constructor and 0.2 matrix cells ([#204](https://github.com/canact/canact/issues/204)) ([aef5c9c](https://github.com/canact/canact/commit/aef5c9c90623b2ef417780655d40ea1c77794810))


### Bug Fixes

* keep 0.2 host-policy fields honest ([#206](https://github.com/canact/canact/issues/206)) ([3628e46](https://github.com/canact/canact/commit/3628e46b4b2e12df43e726f55e6ca4dc99b2c289))

## [0.2.0](https://github.com/canact/canact/compare/v0.1.2...v0.2.0) (2026-09-08)


### Features

* add canact matrix plumbing table ([#199](https://github.com/canact/canact/issues/199)) ([22fcaee](https://github.com/canact/canact/commit/22fcaee3545ad485dcf71c21f24318934ca685ee)), closes [#183](https://github.com/canact/canact/issues/183)
* add constraintPlacement from system-message probe ([#198](https://github.com/canact/canact/issues/198)) ([b85f442](https://github.com/canact/canact/commit/b85f442557de4a1561eabd9c049031f51c5050cd)), closes [#179](https://github.com/canact/canact/issues/179)
* constrain the vision reply to two letters or NONE ([#196](https://github.com/canact/canact/issues/196)) ([e26a748](https://github.com/canact/canact/commit/e26a7485a67115e0958613a32df1964142f1d597)), closes [#181](https://github.com/canact/canact/issues/181)
* derive context_faithfulness from ladder recall ([#194](https://github.com/canact/canact/issues/194)) ([5a7405b](https://github.com/canact/canact/commit/5a7405be09740bb796c82583a6789124b966f3a5)), closes [#178](https://github.com/canact/canact/issues/178)
* drop echoable unified-diff format card ([#197](https://github.com/canact/canact/issues/197)) ([36b67c4](https://github.com/canact/canact/commit/36b67c48a0b64a5c9290d20c44ad623b1a2e1944)), closes [#181](https://github.com/canact/canact/issues/181)
* measure maxOutputTokens from provider rejects ([#193](https://github.com/canact/canact/issues/193)) ([9a2f3a4](https://github.com/canact/canact/commit/9a2f3a4e130eb01164d5f31759605e7a3c5cd73e))
* promote plumbing dimensions into host-policy fields ([#191](https://github.com/canact/canact/issues/191)) ([2ea1144](https://github.com/canact/canact/commit/2ea1144e85b270ac8ca57edc4b20f8f1f9744edc)), closes [#175](https://github.com/canact/canact/issues/175)
* suite tiers and per-dimension cache keys ([#192](https://github.com/canact/canact/issues/192)) ([f608d76](https://github.com/canact/canact/commit/f608d76523d33f43f7186c25a5dea9a8f345ced6))


### Bug Fixes

* collapse matrix rows on provider-prefixed model ids ([#200](https://github.com/canact/canact/issues/200)) ([da20065](https://github.com/canact/canact/commit/da20065b3fa730e0e659cf110428d009944c7f4a))
* reuse catalog-filled cache rows on the probe fast path ([#189](https://github.com/canact/canact/issues/189)) ([7cedff6](https://github.com/canact/canact/commit/7cedff60d474f831448760f8f8181ae324795a28)), closes [#173](https://github.com/canact/canact/issues/173)
* stop writing the ladder floor into overlay windows ([#186](https://github.com/canact/canact/issues/186)) ([f3063d6](https://github.com/canact/canact/commit/f3063d635c519721981e660dad0655f6f7cb4eea)), closes [#171](https://github.com/canact/canact/issues/171)

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
