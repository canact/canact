//! File-based probe cache.
//!
//! Persists [`CapabilityProfile`] results to disk so that probing is only
//! performed once per model+provider+settings combination (with a 30-day
//! TTL). Cache keys include reasoning effort and probe suite version.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::ProbeError;
use crate::types::{CapabilityLevel, CapabilityProfile};

/// How long a cached entry remains valid (30 days in seconds).
pub const CACHE_TTL_SECS: u64 = 30 * 24 * 60 * 60;

/// Bump when probe identity/scoring changes enough to invalidate old entries.
/// v2: #1336 rename, #1337 multi-turn probe, #1339 system adherence redesign.
/// v3: transient stream/tool probe errors are not 30-day Weak/Medium.
/// v4: context-faithfulness timeout synonyms (#3317).
/// v5: generic edit_file on tool_selection is not 30-day Weak/max_tools=10 (#3315).
/// v6: forceful tool-call prompts (Goose #6281) and stricter arg schema.
pub const PROBE_SUITE_VERSION: u32 = 6;

/// Default effort label when probes leave `reasoning_effort` unset.
pub const DEFAULT_PROBE_EFFORT: &str = "unset";

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

/// File-based probe cache keyed by model|provider|effort|suite.
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
        cache.migrate_stale_tool_scores();
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

    /// Get a cached profile for the current suite and default probe effort.
    pub fn get(&self, model_id: &str, provider: &str) -> Option<&CapabilityProfile> {
        self.get_with_settings(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
        )
    }

    /// Get a cached profile for explicit effort/suite settings.
    pub fn get_with_settings(
        &self,
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
    ) -> Option<&CapabilityProfile> {
        let key = Self::cache_key(model_id, provider, reasoning_effort, suite_version);
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
        let key = Self::cache_key(
            model_id,
            provider,
            DEFAULT_PROBE_EFFORT,
            PROBE_SUITE_VERSION,
        );
        self.profiles.get(&key).filter(|e| Self::is_valid(e))
    }

    /// Store a profile under the current suite and default probe effort.
    pub fn put(&mut self, profile: CapabilityProfile) {
        self.put_with_settings(profile, DEFAULT_PROBE_EFFORT, PROBE_SUITE_VERSION);
    }

    /// Store a profile with explicit effort/suite metadata.
    pub fn put_with_settings(
        &mut self,
        profile: CapabilityProfile,
        reasoning_effort: &str,
        suite_version: u32,
    ) {
        let key = Self::cache_key(
            &profile.model_id,
            &profile.provider,
            reasoning_effort,
            suite_version,
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
    fn migrate_stale_tool_scores(&mut self) {
        for entry in self.profiles.values_mut() {
            let p = &mut entry.profile;
            let tool_probes = [
                &mut p.tool_calling,
                &mut p.complex_tool_calling,
                &mut p.nested_arguments,
                &mut p.tool_selection,
                &mut p.streaming_tool_calls,
                &mut p.parallel_tool_scale,
            ];
            for probe in tool_probes {
                if probe.details.contains("does not support tools")
                    && probe.level != CapabilityLevel::Weak
                {
                    probe.level = CapabilityLevel::Weak;
                    probe.score = 0.0;
                }
            }
        }
    }

    /// Cache key for model+provider+effort+suite.
    pub fn cache_key(
        model_id: &str,
        provider: &str,
        reasoning_effort: &str,
        suite_version: u32,
    ) -> String {
        format!("{model_id}|{provider}|{reasoning_effort}|v{suite_version}")
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
