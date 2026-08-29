//! File-based probe cache.
//!
//! Persists [`CapabilityProfile`] results to disk so that probing is only
//! performed once per model+provider+settings combination (with a 30-day
//! TTL). Cache keys include reasoning effort, probe suite version, and the
//! cheap/full plus vision suite knobs.

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
pub const PROBE_SUITE_VERSION: u32 = 19;

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

/// File-based probe cache keyed by model|provider|effort|suite|cost|vision.
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
        let contents = std::fs::read_to_string(path)?;
        let mut cache: Self = serde_json::from_str(&contents)?;
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

    /// Save cache to disk, creating parent directories if necessary.
    pub fn save(&self, path: &Path) -> Result<(), ProbeError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, contents)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Get a cached profile for the current suite, default effort, and default knobs.
    pub fn get(&self, model_id: &str, provider: &str) -> Option<&CapabilityProfile> {
        self.get_with_knobs(model_id, provider, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION)
    }

    /// Get a cached profile for the current suite and explicit cheap/vision knobs.
    pub fn get_with_knobs(
        &self,
        model_id: &str,
        provider: &str,
        skip_expensive: bool,
        vision: bool,
    ) -> Option<&CapabilityProfile> {
        self.get_with_settings(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
        )
    }

    /// Get a cached profile for explicit effort/suite/knob settings.
    pub fn get_with_settings(
        &self,
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
        skip_expensive: bool,
        vision: bool,
    ) -> Option<&CapabilityProfile> {
        let key = Self::cache_key_with_knobs(
            model_id,
            provider,
            reasoning_effort,
            suite_version,
            skip_expensive,
            vision,
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
        self.get_entry_with_knobs(model_id, provider, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION)
    }

    /// Look up the full cache entry for explicit cheap/vision knobs.
    pub fn get_entry_with_knobs(
        &self,
        model_id: &str,
        provider: &str,
        skip_expensive: bool,
        vision: bool,
    ) -> Option<&CacheEntry> {
        let key = Self::cache_key_with_knobs(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
        );
        self.profiles.get(&key).filter(|e| Self::is_valid(e))
    }

    /// Store a profile under the current suite, default effort, and default knobs.
    pub fn put(&mut self, profile: CapabilityProfile) {
        self.put_with_knobs(profile, DEFAULT_SKIP_EXPENSIVE, DEFAULT_VISION);
    }

    /// Store a profile under the current suite with explicit cheap/vision knobs.
    pub fn put_with_knobs(
        &mut self,
        profile: CapabilityProfile,
        skip_expensive: bool,
        vision: bool,
    ) {
        self.put_with_settings(
            profile,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
            skip_expensive,
            vision,
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
    ) {
        let key = Self::cache_key_with_knobs(
            &profile.model_id,
            &profile.provider,
            reasoning_effort,
            suite_version,
            skip_expensive,
            vision,
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
        )
    }

    /// Cache key including cheap/full and vision suite knobs.
    pub fn cache_key_with_knobs(
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
        skip_expensive: bool,
        vision: bool,
    ) -> String {
        let cost = if skip_expensive { "cheap" } else { "full" };
        let vis = if vision { "vision" } else { "novision" };
        format!("{model_id}|{provider}|{reasoning_effort}|v{suite_version}|{cost}|{vis}")
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
