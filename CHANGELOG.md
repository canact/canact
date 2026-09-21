# Changelog

release-please writes version headings from conventional commit
titles on `main`. Do not keep an Unreleased section.

## [0.8.0](https://github.com/canact/canact/compare/v0.7.0...v0.8.0) (2026-09-21)


### Features

* consume wiremux 0.8.0 ([#273](https://github.com/canact/canact/issues/273)) ([ad59fa4](https://github.com/canact/canact/commit/ad59fa47b804e03b946992aa78cf015dfe10d71e))


### Bug Fixes

* map lmstudio and vllm overlay providers to LiteLLM families ([#266](https://github.com/canact/canact/issues/266)) ([5cff9ea](https://github.com/canact/canact/commit/5cff9ea011fce4d83423f4d4dfe129894b2b7d9f))
* map oauth refresh failure to Auth ([#272](https://github.com/canact/canact/issues/272)) ([cb55345](https://github.com/canact/canact/commit/cb553450386587bf76094029ede6b9a05c818bcb))
* refresh Claude oat and matrix plumbing leftovers ([#271](https://github.com/canact/canact/issues/271)) ([994b14b](https://github.com/canact/canact/commit/994b14be721581d0675bcbb85bd7bb3e7593eb85))
* refuse invalid MCP cheap/suite flags and empty cache ([#264](https://github.com/canact/canact/issues/264)) ([1d4b4e9](https://github.com/canact/canact/commit/1d4b4e929e5c95bee83d8da4a9629ad8dbd77cfb))
* treat openai-codex and codex as one cache family ([#274](https://github.com/canact/canact/issues/274)) ([bb0e096](https://github.com/canact/canact/commit/bb0e096ab046d4ca3ce55be944d2ca7ff738c5b7))
* treat whitespace-only base_url as omitted for oauth skip ([#267](https://github.com/canact/canact/issues/267)) ([e123169](https://github.com/canact/canact/commit/e123169847caf69e131bc2dfa89961e765e6060c))
* trim CLI suite, advertised-context, and path flags ([#265](https://github.com/canact/canact/issues/265)) ([321dd86](https://github.com/canact/canact/commit/321dd86f506a49988994c2a698a487e2f851ee80))
* trim MCP api_key_env and refuse invalid advertised_context ([#262](https://github.com/canact/canact/issues/262)) ([b1ae10b](https://github.com/canact/canact/commit/b1ae10b066143bfc63474edf9cf5d8b5c39220a7))

## [0.7.0](https://github.com/canact/canact/compare/v0.6.0...v0.7.0) (2026-09-18)


### Features

* consume wiremux 0.7.0 ([#258](https://github.com/canact/canact/issues/258)) ([f1b44d4](https://github.com/canact/canact/commit/f1b44d4357835307b7e8688e3df89efb7fbb5191))
* route groq and bedrock and classify Ollama Display ([#239](https://github.com/canact/canact/issues/239)) ([4a5fd55](https://github.com/canact/canact/commit/4a5fd553ba102de092ef44721d254bb76983ea0b))


### Bug Fixes

* abort when vendor chat says model not found ([#245](https://github.com/canact/canact/issues/245)) ([9423da2](https://github.com/canact/canact/commit/9423da2670084eb8b4d6b9c95f5ca949d4c1f49c))
* append /v1 to Ollama listen URLs ([#255](https://github.com/canact/canact/issues/255)) ([b6d0dba](https://github.com/canact/canact/commit/b6d0dbaa2b43198706e32d471de72351f21dfb42))
* collapse whitespace-equivalent model ids in cache lookup ([#256](https://github.com/canact/canact/issues/256)) ([f6ca459](https://github.com/canact/canact/commit/f6ca459d419b2e4adf648c52ccd70cf140f1f004))
* do not call a directory cache an internal error ([#252](https://github.com/canact/canact/issues/252)) ([6d05729](https://github.com/canact/canact/commit/6d0572950f92dbba36e2cd3369c9b701823251ad))
* flush folded tool calls on ToolCallEnd ([#259](https://github.com/canact/canact/issues/259)) ([a8e4a8c](https://github.com/canact/canact/commit/a8e4a8cc960a82b9f70daa92d811bc323e865bcd))
* honor explicit groq base-url and tighten folded 404 ([#241](https://github.com/canact/canact/issues/241)) ([da31a08](https://github.com/canact/canact/commit/da31a08566793f62d737a20ae374624572d43aff))
* list every GET /models id when --model is missing ([#244](https://github.com/canact/canact/issues/244)) ([12f5c3f](https://github.com/canact/canact/commit/12f5c3f19ef7371c10eca6edad37bb389b824bed))
* map groq and bedrock hosts to overlay families ([#242](https://github.com/canact/canact/issues/242)) ([ebb2a69](https://github.com/canact/canact/commit/ebb2a69a2a805c72add5dcb549de66d262d95550))
* print not probed for skipped human-table rows ([#246](https://github.com/canact/canact/issues/246)) ([516d4c3](https://github.com/canact/canact/commit/516d4c35489faf102179078acf31ac1df1d10b5e))
* refuse --cheap when --suite is not policy ([#248](https://github.com/canact/canact/issues/248)) ([660b8c2](https://github.com/canact/canact/commit/660b8c295ca83ac018c15a2f4a8271bcecdbfe3b))
* refuse a directory as the probe cache path ([#251](https://github.com/canact/canact/issues/251)) ([ba8408f](https://github.com/canact/canact/commit/ba8408f6f9829e07e7678cf1aa7323c8b81cd06d))
* refuse advertised-context 0 ([#250](https://github.com/canact/canact/issues/250)) ([2f8afe6](https://github.com/canact/canact/commit/2f8afe6980a286b4ffb1bf83ba9000c9cdab82ad))
* replace alias Aider overlay names on re-export ([#243](https://github.com/canact/canact/issues/243)) ([bdc4817](https://github.com/canact/canact/commit/bdc481797dfd5dce2d6f2e0462b9718a6be8074c))
* treat an empty probe cache file as empty ([#247](https://github.com/canact/canact/issues/247)) ([1986ace](https://github.com/canact/canact/commit/1986acef8c2c33142d08f8ab96cb9942bf6f99c7))
* treat multimodal refusal as measured Weak vision ([#249](https://github.com/canact/canact/issues/249)) ([68eb017](https://github.com/canact/canact/commit/68eb017f50f7b3aa47000844bd24089dc4f4c370))
* trim whitespace on --model before cache lookup ([#253](https://github.com/canact/canact/issues/253)) ([58189ed](https://github.com/canact/canact/commit/58189ed3c951329d088bcec100914ec6144ce922))
* trim whitespace on --provider before routing ([#254](https://github.com/canact/canact/issues/254)) ([8620272](https://github.com/canact/canact/commit/8620272ba2b17ca1f8afb60d5b684d62f77ddf81))
* trim whitespace on explicit --base-url before host routing ([#257](https://github.com/canact/canact/issues/257)) ([e029bbf](https://github.com/canact/canact/commit/e029bbf954b52876ed4d7d0b7a02c74014235760))

## [0.6.0](https://github.com/canact/canact/compare/v0.5.0...v0.6.0) (2026-09-16)


### Features

* consume wiremux 0.3.0 ([#226](https://github.com/canact/canact/issues/226)) ([0b2ee16](https://github.com/canact/canact/commit/0b2ee16ff8befee3cd42087ded38bd5650870f85))
* consume wiremux 0.4.0 ([#228](https://github.com/canact/canact/issues/228)) ([79ce145](https://github.com/canact/canact/commit/79ce145d73f8c1f88cbfe0c80710ac18e3200d03))
* consume wiremux 0.5.0 ([#230](https://github.com/canact/canact/issues/230)) ([faf2c76](https://github.com/canact/canact/commit/faf2c761de90da44042660934c395dc364ff3997))
* consume wiremux 0.6.0 ([#233](https://github.com/canact/canact/issues/233)) ([85bab8a](https://github.com/canact/canact/commit/85bab8a6601cf94a71bc6a6eec4f1121ee75c208))
* use Grok login when XAI_API_KEY is unset ([#227](https://github.com/canact/canact/issues/227)) ([e401cbd](https://github.com/canact/canact/commit/e401cbdba2a7dd890088d379188777c2ba854c5b))


### Bug Fixes

* print advertised window and unprobed vision in human table ([#225](https://github.com/canact/canact/issues/225)) ([f568108](https://github.com/canact/canact/commit/f568108f51b041b035894f96fb7d470d612814e2))
* read Grok Build context_window in catalog helper ([#231](https://github.com/canact/canact/issues/231)) ([038e151](https://github.com/canact/canact/commit/038e1517a1143f0d9e30e929ad82194d3958883e))
* read USER Claude keychain and ignore catalog vision false ([#224](https://github.com/canact/canact/issues/224)) ([adb30f2](https://github.com/canact/canact/commit/adb30f2a86d8a5dd27c584fcafa97e3ada52a409))
* recover advertised window and name missing cloud keys ([#222](https://github.com/canact/canact/issues/222)) ([dd0f44c](https://github.com/canact/canact/commit/dd0f44c630e3e1e2694787ae7b91650bff748e72))
* route streamed tool-call arg deltas by index ([#235](https://github.com/canact/canact/issues/235)) ([09243b2](https://github.com/canact/canact/commit/09243b227fc06e523ee9266d7b7390f6eded5b82))
* route wiremux tool-call arg deltas by index ([#234](https://github.com/canact/canact/issues/234)) ([e8a14aa](https://github.com/canact/canact/commit/e8a14aa3f5ea53a6acf0a6dbad1528fd5fb0df19))
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
