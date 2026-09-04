//! File-based probe cache.
//!
//! Persists [`CapabilityProfile`] results to disk so that probing is only
//! performed once per model+provider+settings combination (with a 30-day
//! TTL). Cache keys include reasoning effort, probe suite version, the
//! cheap/full plus vision suite knobs, and the advertised context cap.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::ProbeError;
use crate::types::{CapabilityLevel, CapabilityProfile, TOOL_PROBE_NAMES};

/// How long a cached entry remains valid (30 days in seconds).
pub const CACHE_TTL_SECS: u64 = 30 * 24 * 60 * 60;

/// Bump when probe identity/scoring changes enough to invalidate old entries.
/// v2: #1336 rename, #1337 multi-turn probe, #1339 system adherence redesign.
/// v3: transient stream/tool probe errors are not 30-day Weak/Medium.
/// v4: context-faithfulness timeout synonyms (#3317).
/// v5: generic edit_file on tool_selection is not 30-day Weak/max_tools=10 (#3315).
/// v6: forceful tool-call prompts (Goose #6281) and stricter arg schema.
/// v7: persisted effective_context_tokens ladder.
/// v8: token_efficiency prefers live ProbeResponse usage when present.
/// v9: cheap ladder is not a finished size; vision refusal beats "text";
///     synthesized error Medium does not open can_use_tools.
/// v10: empty instruction-following is Weak; vision color-only refusal
///      is Weak; SSE late tool name still emits ToolCallStart.
/// v11: synthesized error Medium does not open host policy (vision,
///      XML fallback, edit format, meets); o-series chat uses
///      max_completion_tokens and omits temperature.
/// v12: empty token_efficiency is Weak; SSE tool_calls indexes
///      each emit ToolCallStart.
/// v13: chat parse accepts legacy function/function_call; complex
///      two-name imprecise args are Medium; ladder heartbeat
///      accepts 2,840.
/// v14: SSE function_call deltas emit starts; empty max_tokens
///      compliance is Weak; parallel numeric paths are Medium.
/// v15: JSON fence language tags; streaming Strong requires
///      read_file; system-adherence details are UTF-8 safe.
/// v16: one_shot_tool_plan and tool_selection Strong require
///      non-empty string args on preferred tools.
/// v17: whitespace-only args are not Strong; tool_selection scores
///      the best same-name call; one_shot order uses first precise
///      call; tool_calling/xml/streaming reject empty path.
/// v18: remaining tool probes reject empty/whitespace string args;
///      tool_selection doc_set Strong requires a present non-null value.
/// v19: multi_turn_task_sequencing Strong requires nonempty string args;
///      token_efficiency empty text is Weak even when usage is 1-10.
/// v20: tool_selection Strong requires all three preferred tools
///      precise; empty/whitespace `doc_set` value is not precise;
///      json_output Strong requires nonempty word and reversed.
/// v21: streaming_tool_calls Strong requires a string `path` on the
///      `read_file` start that produced the args (no other-tool merge);
///      search_replace / unified_diff Strong require the edit bodies
///      (not whole-response contains) to hold greet/Hello and
///      welcome/Welcome.
/// v22: json_output Strong requires word=hello, length=5,
///      reversed=olleh (prompt example is not Strong);
///      stream ignores leftover function_call when tool_calls is
///      present; vision "no text" is Weak.
/// v23: vision Strong requires a standalone `bl` token (not `black`);
///      tool-arg strings reject ZWSP-only values; unprobed default
///      dimensions do not open host policy.
/// v24: XML format-card echo and tag mention do not open canUseTools;
///      SEARCH/REPLACE Strong requires `fn greet` / `fn welcome`;
///      unified_diff Medium requires hunk or file headers (not a
///      markdown +/- list); vision Medium surface words are tokens
///      (not `whitespace` / `context`).
/// v25: vision "no letters" / "no characters" is Weak; a closed XML
///      block that only mentions `<name>read_file</name>` without
///      `<arguments>` does not open canUseTools.
/// v26: closed XML Medium requires arguments that open `{`;
///      vision "don't see letters" is Weak; SEARCH/REPLACE Strong
///      requires `fn greet(` / `fn welcome(`.
/// v27: vision question echo, unified-diff format-card echo, and
///      XML JSON Schema paste do not open host policy.
/// v28: unified_diff headers and hunks must start a line
///      (prose `---` / `+++` / `@@` is Weak).
/// v29: vision "text-only" / "black box" is Weak; XML
///      `{"path":"value"}` is format-card echo.
/// v30: vision "no visible" is Weak; an unclosed `<tool_call>`
///      does not open tools from a `read_file` name before the tag.
/// v31: json_output does not peel an array wrapper to reach a
///      valid object (array-wrapped JSON must not skip repair).
/// v32: tool_calling empty or whitespace `function.name` is Weak
///      (does not open can_use_tools).
/// v33: SEARCH/REPLACE Strong ignores `//` comment tokens
///      (`fn greet(` / Hello in comments is not a rename).
/// v34: xml_tool_calling empty or whitespace `<name>` is Weak
///      (does not open can_use_tools).
/// v35: unified_diff Strong requires `fn greet` / `fn welcome` in
///      code +/- lines; comment-only +/- is Weak (not UnifiedDiff).
/// v36: SEARCH/REPLACE and unified_diff ignore `/* */` comment
///      tokens the same way as `//` comments.
/// v37: tool names that are only ZWSP/format marks are Weak
///      (same as empty names for can_use_tools).
/// v38: vision "text model" / "processes text" is Weak
///      (does not set supportsVision).
/// v39: vision "text-based" / "work with text" / "isn't any text"
///      is Weak.
/// v40: unified_diff format-card body (`removed line` / `added line`)
///      is Weak even when the path is greet.rs.
/// v41: vision "no visible letters" / "doesn't contain letters" is Weak
///      (does not set supportsVision).
/// v42: unified_diff title-case format-card body (`Removed line` /
///      `Added line`) is Weak.
/// v43: vision "white space" / "white-box" / "text-processing" is Weak
///      (does not set supportsVision).
/// v44: SEARCH/REPLACE and unified_diff ignore `#` comment tokens
///      the same way as `//` (`# fn greet` is not a rename).
///      Rust attributes (`#[derive]`) are not comments.
/// v45: XML `{"param":"value"}` plus extra keys, or `{'param':'value'}`,
///      is format-card echo (does not set canUseTools).
/// v46: vision "no visible characters" / "no discernible text" /
///      "white-space" is Weak (does not set supportsVision).
/// v47: XML array-wrapped `[{"param":"value"}]` and unclosed
///      param/value cards are format-card echo (do not set canUseTools).
/// v48: vision "no discernible characters" / "aren't any characters"
///      is Weak (does not set supportsVision).
/// v49: unified_diff format-card body (`remove line` / `add line`)
///      is Weak.
/// v50: XML multi-element `[{"param":"value"},{"param":"value"}]`
///      is format-card echo (does not set canUseTools).
/// v51: unified_diff hyphenated or double-space format-card body
///      (`removed-line` / `added-line`, `removed  line` / `added  line`)
///      is Weak.
/// v52: XML nested `{"payload":{"param":"value"}}`, JSON-string
///      wrap `"{\"param\":\"value\"}"`, and `{"param":["value"]}`
///      are format-card echo (do not set canUseTools).
/// v53: XML card keys and `value` compare case-insensitively
///      (`Param`/`Value` / `path":"Value"` is echo, not canUseTools).
/// v54: unified_diff underscore format-card body
///      (`removed_line` / `added_line`) is Weak.
/// v55: XML `{"path":["value"]}` and padded `"value "` are format-card
///      echo (do not set canUseTools).
/// v56: unparseable `{'path':'value'}` is format-card echo
///      (does not set canUseTools).
/// v57: XML card `value` ignores ZWSP/format marks
///      (`"value\\u200b"` is echo, not canUseTools).
/// v58: unparseable `{'path':'Value'}` is format-card echo
///      (does not set canUseTools).
/// v59: XML card keys ignore ZWSP/format marks
///      (`"path\\u200b"` is echo, not canUseTools).
/// v60: XML card `ｖａｌｕｅ` (fullwidth) is echo
///      (does not set canUseTools).
/// v61: unparseable `{p\\u200bath:'value'}` is format-card echo
///      (does not set canUseTools).
/// v62: unified-diff card `removed\\u{200B}line` is echo
///      (does not set UnifiedDiff).
/// v63: vision `no\\u{200B} text` is a refusal
///      (does not set supportsVision).
/// v64: vision `don\\u{200B}'t` / `can\\u{200B}'t` is a refusal
///      (does not set supportsVision).
/// v65: cheap skip is not a measured Medium (envelope status + completed_level);
///      persist probed_context_floor; CLI envelope cacheable/skipExpensive.
/// v66: adapter strips <think> / thinking parts so CoT is not graded.
/// v67: Length + no tool call is truncated (uncacheable), not 30-day Weak.
/// v68: recommendedContextTokens = min(advertised, measured); advertised in cache key.
/// v69: Length + incomplete tool/XML is truncated, not 30-day Medium.
/// v70: stream Length is truncated (uncacheable), not 30-day Weak/Medium.
/// v71: Length on JSON / edit / vision / instruction / context ladder
///      is truncated, not a 30-day host-policy card.
/// v72: instruction_following Strong requires the asked capital.
/// v73: blank-path parallel reads no longer outrank valid paths;
///      Length on code/faithfulness/memory/system is uncacheable;
///      SSE buffers bytes so split UTF-8 is not corrupted.
/// v74: vision Strong requires no refusal; code_syntax ignores
///      docstring `...`; empty Length on max_tokens is Transient.
/// v75: memory refusals are Weak; code_syntax uses the merge fence;
///      JSON length 5.0; XML/unified-diff pick a real edit over a card;
///      fenced Paris; ladder 2.84s; Length-empty token_efficiency Transient.
/// v76: tool_calling Strong if any call is read_file + path (not only first).
/// v77: json_output scores the best object, not only the first fence.
/// v78: SEARCH/REPLACE parses without -------; unified-diff Strong
///      requires Hello/Welcome in the +/- body.
/// v79: stream string indexes, numeric tool ids, array content,
///      finish_reason on delta, capped error bodies.
/// v80: JSON length "5"; port 9,847; old_text aliases; XML path child.
/// v81: STATUS: ok space-fold; code_syntax ignores comment return
///      and requires def merge_sorted.
/// v82: nested edit path alias; XML file_path JSON alias.
/// v83: vision cannot-read BL; memory cannot-share; numeric chat tool id;
///      incremental success-body cap.
pub const PROBE_SUITE_VERSION: u32 = 83;

/// Default effort label when probes leave `reasoning_effort` unset.
pub const DEFAULT_PROBE_EFFORT: &str = "unset";

/// Default cost knob: paid/full suite (`skip_expensive = false`).
pub const DEFAULT_SKIP_EXPENSIVE: bool = false;

/// Default vision knob: vision probe not requested.
pub const DEFAULT_VISION: bool = false;

/// A cached probe result together with the time it was stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheEntry {
    /// The cached capability profile.
    pub profile: CapabilityProfile,
    /// Unix epoch seconds when this entry was written.
    pub cached_at: u64,
    /// Effective reasoning effort used for probe requests (e.g. `unset`, `low`).
    #[serde(default = "default_effort_label")]
    pub reasoning_effort: String,
    /// Probe suite version used when this entry was written.
    #[serde(default = "default_suite_v1")]
    pub probe_suite_version: u32,
}

fn default_effort_label() -> String {
    DEFAULT_PROBE_EFFORT.to_owned()
}

fn default_suite_v1() -> u32 {
    1
}

/// File-based probe cache keyed by model|provider|effort|suite|cost|vision|ctx.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProbeCache {
    /// All cached entries.
    pub profiles: HashMap<String, CacheEntry>,
}

impl ProbeCache {
    /// Load cache from disk. Returns an empty cache if the file does not exist.
    ///
    /// Applies migrations to fix stale probe scores from older versions.
    pub fn load(path: &Path) -> Result<Self, ProbeError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let len = std::fs::metadata(path)?.len();
        if len > 8 * 1024 * 1024 {
            return Err(ProbeError::Internal(format!(
                "probe cache is too large ({} bytes): {}",
                len,
                path.display()
            )));
        }
        let mut cache = Self::read_disk(path)?;
        if cache.migrate_stale_tool_scores() {
            // Keep the migrated rows in memory even when rewrite fails.
            // The next process retries migrate against the stale file.
            if let Err(err) = cache.save(path) {
                eprintln!(
                    "warning: failed to persist migrated probe cache ({}): {err}",
                    path.display()
                );
            }
        }
        Ok(cache)
    }

    fn read_disk(path: &Path) -> Result<Self, ProbeError> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }

    /// Save cache to disk, creating parent directories if necessary.
    pub fn save(&self, path: &Path) -> Result<(), ProbeError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut profiles = self.profiles.clone();
        if path.exists() {
            if let Ok(disk) = Self::read_disk(path) {
                for (key, theirs) in disk.profiles {
                    match profiles.get(&key) {
                        Some(ours) if ours.cached_at >= theirs.cached_at => {}
                        _ => {
                            profiles.insert(key, theirs);
                        }
                    }
                }
            }
        }
        let outgoing = Self { profiles };
        let contents = serde_json::to_string_pretty(&outgoing)?;
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        std::fs::write(&tmp, contents)?;
        // Windows rename cannot replace an existing file. Move the dest
        // aside so a failed rename can restore it.
        #[cfg(windows)]
        {
            let bak = path.with_extension("bak");
            if path.exists() {
                let _ = std::fs::remove_file(&bak);
                std::fs::rename(path, &bak)?;
            }
            if let Err(err) = std::fs::rename(&tmp, path) {
                let _ = std::fs::rename(&bak, path);
                return Err(err.into());
            }
            let _ = std::fs::remove_file(&bak);
        }
        #[cfg(not(windows))]
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Newest valid current-suite profile for `model_id` + `provider`.
    ///
    /// Tries the default knob key, then the newest matching row. Export and MCP
    /// use this so a cheap-suite cache still produces an overlay.
    pub fn find_profile(&self, model_id: &str, provider: &str) -> Option<&CapabilityProfile> {
        if let Some(profile) = self.get(model_id, provider) {
            return Some(profile);
        }
        self.find_profile_with_cost(model_id, provider)
            .map(|(profile, _)| profile)
    }

    /// Newest matching row and whether that row was stored as cheap.
    pub fn find_profile_with_cost(
        &self,
        model_id: &str,
        provider: &str,
    ) -> Option<(&CapabilityProfile, bool)> {
        if let Some(profile) = self.get(model_id, provider) {
            return Some((profile, DEFAULT_SKIP_EXPENSIVE));
        }
        self.profiles
            .iter()
            .filter(|(_, entry)| {
                Self::is_valid(entry)
                    && entry.probe_suite_version == PROBE_SUITE_VERSION
                    && entry.profile.model_id == model_id
                    && entry.profile.provider == provider
            })
            .max_by_key(|(_, entry)| entry.cached_at)
            .map(|(key, entry)| (&entry.profile, key.split('|').nth(4) == Some("cheap")))
    }

    /// Get a cached profile for the current suite, default effort, and default knobs.
    pub fn get(&self, model_id: &str, provider: &str) -> Option<&CapabilityProfile> {
        self.get_with_knobs(
            model_id,
            provider,
            DEFAULT_SKIP_EXPENSIVE,
            DEFAULT_VISION,
            None,
        )
    }

    /// Get a cached profile for the current suite and explicit cheap/vision/context knobs.
    pub fn get_with_knobs(
        &self,
        model_id: &str,
        provider: &str,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) -> Option<&CapabilityProfile> {
        self.get_with_settings(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
            advertised,
        )
    }

    /// Get a cached profile for explicit effort/suite/knob settings.
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_settings(
        &self,
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) -> Option<&CapabilityProfile> {
        let key = Self::cache_key_with_knobs(
            model_id,
            provider,
            reasoning_effort,
            suite_version,
            skip_expensive,
            vision,
            advertised,
        );
        self.profiles.get(&key).and_then(|entry| {
            if Self::is_valid(entry) {
                Some(&entry.profile)
            } else {
                None
            }
        })
    }

    /// Look up the full cache entry (for doctor/probe display metadata).
    pub fn get_entry(&self, model_id: &str, provider: &str) -> Option<&CacheEntry> {
        self.get_entry_with_knobs(
            model_id,
            provider,
            DEFAULT_SKIP_EXPENSIVE,
            DEFAULT_VISION,
            None,
        )
    }

    /// Look up the full cache entry for explicit cheap/vision/context knobs.
    pub fn get_entry_with_knobs(
        &self,
        model_id: &str,
        provider: &str,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) -> Option<&CacheEntry> {
        let key = Self::cache_key_with_knobs(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
            advertised,
        );
        self.profiles.get(&key).filter(|e| Self::is_valid(e))
    }

    /// Store a profile under the current suite, default effort, and default knobs.
    pub fn put(&mut self, profile: CapabilityProfile) {
        self.put_with_knobs(profile, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION, None);
    }

    /// Store a profile under the current suite with explicit cheap/vision/context knobs.
    pub fn put_with_knobs(
        &mut self,
        profile: CapabilityProfile,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) {
        self.put_with_settings(
            profile,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
            advertised,
        );
    }

    /// Store a profile with explicit effort/suite/knob metadata.
    pub fn put_with_settings(
        &mut self,
        profile: CapabilityProfile,
        reasoning_effort: &str,
        suite_version: u32,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) {
        let key = Self::cache_key_with_knobs(
            &profile.model_id,
            &profile.provider,
            reasoning_effort,
            suite_version,
            skip_expensive,
            vision,
            advertised,
        );
        let entry = CacheEntry {
            cached_at: unix_now(),
            profile,
            reasoning_effort: reasoning_effort.to_owned(),
            probe_suite_version: suite_version,
        };
        self.profiles.insert(key, entry);
    }

    /// Fix stale probe scores from before the "does not support tools"
    /// scoring fix. Tool-related probes that failed because the provider
    /// returned "does not support tools" were incorrectly scored as Medium
    /// (0.5) instead of Weak (0.0) by the old `probe_or_default`.
    ///
    /// Returns true when any row was rewritten so [`Self::load`] can persist.
    fn migrate_stale_tool_scores(&mut self) -> bool {
        let mut changed = false;
        for entry in self.profiles.values_mut() {
            for name in TOOL_PROBE_NAMES {
                let Some(probe) = entry.profile.dimension_result_mut(name) else {
                    continue;
                };
                if probe.details.contains("does not support tools")
                    && probe.level != CapabilityLevel::Weak
                {
                    probe.level = CapabilityLevel::Weak;
                    probe.score = 0.0;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Cache key for model+provider+effort+suite and default cheap/vision knobs.
    pub fn cache_key(
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
    ) -> String {
        Self::cache_key_with_knobs(
            model_id,
            provider,
            reasoning_effort,
            suite_version,
            DEFAULT_SKIP_EXPENSIVE,
            DEFAULT_VISION,
            None,
        )
    }

    /// Cache key including cheap/full, vision, and advertised context knobs.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_key_with_knobs(
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
        skip_expensive: bool,
        vision: bool,
        advertised: Option<u32>,
    ) -> String {
        let cost = if skip_expensive { "cheap" } else { "full" };
        let vis = if vision { "vision" } else { "novision" };
        let ctx = match advertised {
            Some(n) => format!("ctx{n}"),
            None => "ctxnone".to_owned(),
        };
        format!("{model_id}|{provider}|{reasoning_effort}|v{suite_version}|{cost}|{vis}|{ctx}")
    }

    /// Check whether a cache entry is still valid (less than 30 days old).
    fn is_valid(entry: &CacheEntry) -> bool {
        let now = unix_now();
        now.saturating_sub(entry.cached_at) < CACHE_TTL_SECS
    }
}

/// Current time as Unix epoch seconds.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
