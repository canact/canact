//! Probe orchestrator.

use std::path::Path;
use std::sync::Mutex;

use tracing::warn;

use crate::cache::ProbeCache;
use crate::client::ProbeClient;
use crate::error::ProbeError;
use crate::types::{CapabilityLevel, CapabilityProfile, ProbeResult};

/// Default concurrency for paid providers (effectively unlimited).
pub const PAID_CONCURRENCY: usize = 64;

/// Concurrency limit for free-tier models to avoid rate limit exhaustion.
pub const FREE_CONCURRENCY: usize = 3;

/// Outcome of a probe suite run, including whether the profile is cacheable.
#[derive(Debug, Clone)]
pub struct ProbeRun {
    /// Assembled capability profile (session-local even when uncacheable).
    pub profile: CapabilityProfile,
    /// False when any required probe hit a transient error (timeout, 429,
    /// network, 5xx). Callers must not persist this profile for 30 days.
    pub cacheable: bool,
}

impl ProbeRun {
    /// Persist the profile when [`Self::cacheable`] is true.
    ///
    /// Returns `Ok(true)` when the 30-day cache was written, `Ok(false)`
    /// when persist was skipped because a required probe was transient.
    pub fn persist(&self, cache: &mut ProbeCache, path: &Path) -> Result<bool, ProbeError> {
        if !self.cacheable {
            warn!("skipping probe cache persist: transient probe error");
            return Ok(false);
        }
        cache.put(self.profile.clone());
        cache.save(path)?;
        Ok(true)
    }
}

/// Orchestrates running capability probes against an LLM.
pub struct ProbeRunner<C: ProbeClient> {
    #[allow(dead_code)]
    client: C,
    #[allow(dead_code)]
    concurrency: usize,
    #[allow(dead_code)]
    skip_expensive: bool,
    last_run: Mutex<Option<ProbeRun>>,
}

impl<C: ProbeClient> ProbeRunner<C> {
    /// Create a new probe runner targeting the given client (paid knobs).
    pub fn new(client: C) -> Self {
        Self {
            client,
            concurrency: PAID_CONCURRENCY,
            skip_expensive: false,
            last_run: Mutex::new(None),
        }
    }

    /// Create a probe runner with throttled concurrency for free-tier models.
    pub fn new_throttled(client: C) -> Self {
        Self {
            client,
            concurrency: FREE_CONCURRENCY,
            skip_expensive: true,
            last_run: Mutex::new(None),
        }
    }

    /// Builder alias of [`Self::new_throttled`] knobs.
    pub fn cheap(self) -> Self {
        Self {
            client: self.client,
            concurrency: FREE_CONCURRENCY,
            skip_expensive: true,
            last_run: self.last_run,
        }
    }

    /// Builder alias of [`Self::new`] knobs.
    pub fn full(self) -> Self {
        Self {
            client: self.client,
            concurrency: PAID_CONCURRENCY,
            skip_expensive: false,
            last_run: self.last_run,
        }
    }

    /// Persist the last run when it was cacheable.
    ///
    /// Returns `Ok(false)` when there is no last run or it was not cacheable.
    /// Transient results must not be written.
    pub fn persist(&self, cache: &mut ProbeCache, path: &Path) -> Result<bool, ProbeError> {
        let guard = self
            .last_run
            .lock()
            .map_err(|_| ProbeError::Internal("probe runner lock poisoned".into()))?;
        match guard.as_ref() {
            Some(run) => run.persist(cache, path),
            None => Ok(false),
        }
    }

    /// Run all probes and return a [`CapabilityProfile`].
    ///
    /// Err only on Auth (suite abort). Other probe errors still synthesize.
    pub async fn run(&self) -> Result<CapabilityProfile, ProbeError> {
        Ok(self.run_detailed().await?.profile)
    }

    /// Run all probes and report whether the profile is safe to cache.
    pub async fn run_detailed(&self) -> Result<ProbeRun, ProbeError> {
        Err(ProbeError::Internal("probes not wired".into()))
    }
}

/// Resolve a probe result and whether it is safe to write into the 30-day cache.
///
/// Auth aborts the suite (`Err`). Definitive "does not support tools" is Weak
/// (tool-named) or Medium (other) and cacheable. Transient errors (timeout,
/// 429, network, 5xx, other Err) stay Medium for this session and must not
/// be persisted as a capability score.
pub fn resolve_probe(
    result: Result<ProbeResult, ProbeError>,
    name: &str,
) -> Result<(ProbeResult, bool), ProbeError> {
    match result {
        Ok(pr) => Ok((pr, true)),
        Err(err @ ProbeError::Auth(_)) => Err(err),
        Err(err) => {
            let err_msg = err.to_string();
            let is_tool_probe = matches!(
                name,
                "tool_calling"
                    | "complex_tool_calling"
                    | "nested_arguments"
                    | "tool_selection"
                    | "streaming_tool_calls"
                    | "parallel_tool_scale"
            );
            let tools_not_supported = err_msg.contains("does not support tools");

            if is_tool_probe && tools_not_supported {
                warn!(probe = name, error = %err, "Model does not support tools, scoring as Weak");
                Ok((
                    ProbeResult {
                        name: name.to_string(),
                        score: 0.0,
                        max_score: 1.0,
                        level: CapabilityLevel::Weak,
                        details: format!("Probe failed: {err}"),
                    },
                    true,
                ))
            } else {
                warn!(probe = name, error = %err, "Probe failed, defaulting to Medium");
                Ok((
                    ProbeResult {
                        name: name.to_string(),
                        score: 0.5,
                        max_score: 1.0,
                        level: CapabilityLevel::Medium,
                        details: format!("Probe failed: {err}"),
                    },
                    tools_not_supported,
                ))
            }
        }
    }
}
