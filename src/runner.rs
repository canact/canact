//! Probe orchestrator.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::Semaphore;
use tracing::warn;

use crate::cache::ProbeCache;
use crate::client::ProbeClient;
use crate::error::ProbeError;
use crate::probes;
use crate::types::{
    CapabilityLevel, CapabilityProfile, HostPolicyMeta, ProbeResult, SuiteTier, TOOL_PROBE_NAMES,
};

/// Default concurrency for paid providers (effectively unlimited).
pub const PAID_CONCURRENCY: usize = 64;

/// Concurrency limit for free-tier models to avoid rate limit exhaustion.
pub const FREE_CONCURRENCY: usize = 3;

const VISION_SKIP_NOT_REQUESTED: &str = "Skipped: vision not requested";
const VISION_SKIP_FLAG: &str = "Skipped: --no-vision";
const XML_SKIP: &str = "Not tested (native tool calling is Strong; XML fallback unused)";
const PROMOTED_SKIP: &str = "Skipped: policy suite (use --suite=full or --suite=all)";
const DIAGNOSTIC_SKIP: &str = "Skipped: diagnostic suite (use --suite=all)";
const ONE_SHOT_SKIP: &str = "Skipped: one_shot_tool_plan is not a host-policy signal";

/// Outcome of a probe suite run, including whether the profile is cacheable.
#[derive(Debug, Clone)]
pub struct ProbeRun {
    /// Assembled capability profile (session-local even when uncacheable).
    pub profile: CapabilityProfile,
    /// False when any required probe hit a transient error (timeout, 429,
    /// network, 5xx). Callers must not persist this profile for 30 days.
    pub cacheable: bool,
    /// Whether this run skipped the expensive suite (`--cheap` / policy).
    pub skip_expensive: bool,
    /// Suite tier that produced this run.
    pub suite: SuiteTier,
    /// Whether this run requested the vision probe (`--vision`).
    pub vision: bool,
    /// Catalog advertised context prior for this run, if any.
    pub advertised_context_tokens: Option<u32>,
}

impl ProbeRun {
    /// Host-policy JSON using this run's cacheable / cheap / advertised knobs.
    pub fn host_policy_envelope(&self) -> serde_json::Value {
        self.profile
            .host_policy_envelope_with(HostPolicyMeta::for_suite(
                self.cacheable,
                false,
                self.suite,
                self.advertised_context_tokens,
            ))
    }

    /// Persist the profile when [`Self::cacheable`] is true.
    ///
    /// Returns `Ok(true)` when the 30-day cache was written, `Ok(false)`
    /// when persist was skipped because a required probe was transient.
    pub fn persist(&self, cache: &mut ProbeCache, path: &Path) -> Result<bool, ProbeError> {
        if !self.cacheable {
            warn!("skipping probe cache persist: transient probe error");
            return Ok(false);
        }
        cache.put_with_suite(
            self.profile.clone(),
            self.suite,
            self.vision,
            self.advertised_context_tokens,
        );
        cache.save(path)?;
        Ok(true)
    }
}

/// Orchestrates running capability probes against an LLM.
pub struct ProbeRunner<C: ProbeClient> {
    client: C,
    concurrency: usize,
    suite: SuiteTier,
    last_run: Mutex<Option<ProbeRun>>,
}

impl<C: ProbeClient> ProbeRunner<C> {
    /// Create a new probe runner targeting the given client (paid knobs).
    pub fn new(client: C) -> Self {
        Self {
            client,
            concurrency: PAID_CONCURRENCY,
            suite: SuiteTier::Full,
            last_run: Mutex::new(None),
        }
    }

    /// Create a probe runner with throttled concurrency for free-tier models.
    pub fn new_throttled(client: C) -> Self {
        Self {
            client,
            concurrency: FREE_CONCURRENCY,
            suite: SuiteTier::Policy,
            last_run: Mutex::new(None),
        }
    }

    /// Builder alias of [`Self::new_throttled`] knobs (`--cheap` / policy).
    pub fn cheap(self) -> Self {
        Self {
            client: self.client,
            concurrency: FREE_CONCURRENCY,
            suite: SuiteTier::Policy,
            last_run: self.last_run,
        }
    }

    /// Builder alias of [`Self::new`] knobs (`--full`).
    pub fn full(self) -> Self {
        Self {
            client: self.client,
            concurrency: PAID_CONCURRENCY,
            suite: SuiteTier::Full,
            last_run: self.last_run,
        }
    }

    /// Set the suite tier. Does not change concurrency.
    pub fn suite(self, suite: SuiteTier) -> Self {
        Self {
            client: self.client,
            concurrency: self.concurrency,
            suite,
            last_run: self.last_run,
        }
    }

    /// Throttle concurrency for local or free-tier hosts.
    pub fn throttled(self) -> Self {
        Self {
            client: self.client,
            concurrency: FREE_CONCURRENCY,
            suite: self.suite,
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
        let sem = Arc::new(Semaphore::new(self.concurrency));

        let tool_fut = Self::gated(&sem, probes::probe_tool_calling(&self.client));
        let json_fut = Self::gated(&sem, probes::probe_json_output(&self.client));
        let instr_fut = Self::gated(&sem, probes::probe_instruction_following(&self.client));
        let sr_fut = Self::gated(&sem, probes::probe_search_replace(&self.client));
        let diff_fut = Self::gated(&sem, probes::probe_unified_diff(&self.client));
        let complex_fut = Self::gated(&sem, probes::probe_complex_tool_calling(&self.client));
        let nested_fut = Self::gated(&sem, probes::probe_nested_arguments(&self.client));
        let tool_sel_fut = Self::gated(&sem, probes::probe_tool_selection(&self.client));
        let streaming_fut = Self::gated(&sem, probes::probe_streaming_tool_calls(&self.client));
        let code_syntax_fut = Self::gated_or_skip_named(
            !self.suite.run_diagnostics(),
            &sem,
            "code_syntax",
            DIAGNOSTIC_SKIP,
            probes::probe_code_syntax(&self.client),
        );
        let max_tok_fut = Self::gated_or_skip_named(
            !self.suite.run_diagnostics(),
            &sem,
            "max_tokens_compliance",
            DIAGNOSTIC_SKIP,
            probes::probe_max_tokens_compliance(&self.client),
        );
        let sys_msg_fut = Self::gated_or_skip_named(
            !self.suite.run_diagnostics(),
            &sem,
            "system_message_adherence",
            DIAGNOSTIC_SKIP,
            probes::probe_system_message_adherence(&self.client),
        );
        let efficiency_fut = Self::gated_or_skip_named(
            !self.suite.run_diagnostics(),
            &sem,
            "token_efficiency",
            DIAGNOSTIC_SKIP,
            probes::probe_token_efficiency(&self.client),
        );
        let par_scale_fut = Self::gated(&sem, probes::probe_parallel_tool_scale(&self.client));
        let vision_flag = self.client.catalog().supports_vision;
        let vision_enabled = vision_flag == Some(true);
        let vision_fut = async {
            match vision_flag {
                Some(true) => Self::gated(&sem, probes::probe_vision(&self.client)).await,
                Some(false) => Ok(vision_skip(VISION_SKIP_FLAG)),
                None => Ok(vision_skip(VISION_SKIP_NOT_REQUESTED)),
            }
        };

        let (
            tool_result,
            json_result,
            instr_result,
            sr_result,
            diff_result,
            complex_result,
            nested_result,
            tool_sel_result,
            streaming_tc_result,
            code_syntax_result,
            max_tok_result,
            sys_msg_result,
            efficiency_result,
            par_scale_result,
            vision_result,
        ) = tokio::join!(
            tool_fut,
            json_fut,
            instr_fut,
            sr_fut,
            diff_fut,
            complex_fut,
            nested_fut,
            tool_sel_fut,
            streaming_fut,
            code_syntax_fut,
            max_tok_fut,
            sys_msg_fut,
            efficiency_fut,
            par_scale_fut,
            vision_fut,
        );

        let mut cacheable = true;
        let tool_calling = take_probe(&mut cacheable, tool_result, "tool_calling")?;
        let json_output = take_probe(&mut cacheable, json_result, "json_output")?;
        let instruction_following =
            take_probe(&mut cacheable, instr_result, "instruction_following")?;
        let search_replace = take_probe(&mut cacheable, sr_result, "search_replace")?;
        let unified_diff = take_probe(&mut cacheable, diff_result, "unified_diff")?;
        let complex_tool_calling =
            take_probe(&mut cacheable, complex_result, "complex_tool_calling")?;
        let nested_arguments = take_probe(&mut cacheable, nested_result, "nested_arguments")?;
        let tool_selection = take_probe(&mut cacheable, tool_sel_result, "tool_selection")?;
        let streaming_tool_calls =
            take_probe(&mut cacheable, streaming_tc_result, "streaming_tool_calls")?;
        let code_syntax = take_probe(&mut cacheable, code_syntax_result, "code_syntax")?;
        let max_tokens_compliance =
            take_probe(&mut cacheable, max_tok_result, "max_tokens_compliance")?;
        let system_message_adherence =
            take_probe(&mut cacheable, sys_msg_result, "system_message_adherence")?;
        let token_efficiency = take_probe(&mut cacheable, efficiency_result, "token_efficiency")?;
        let parallel_tool_scale =
            take_probe(&mut cacheable, par_scale_result, "parallel_tool_scale")?;
        let vision = take_probe(&mut cacheable, vision_result, "vision")?;

        let skip_promoted = !self.suite.run_promoted_expensive();
        let skip_diag = !self.suite.run_diagnostics();
        let (plan_r, seq_r, mem_r) = tokio::join!(
            async { Ok(named_skip("one_shot_tool_plan", ONE_SHOT_SKIP)) },
            Self::gated_or_skip_named(
                skip_promoted,
                &sem,
                "multi_turn_task_sequencing",
                PROMOTED_SKIP,
                probes::probe_multi_turn_task_sequencing(&self.client),
            ),
            Self::gated_or_skip_named(
                skip_diag,
                &sem,
                "multi_turn_memory",
                DIAGNOSTIC_SKIP,
                probes::probe_multi_turn_memory(&self.client),
            ),
        );
        let one_shot_tool_plan = take_probe(&mut cacheable, plan_r, "one_shot_tool_plan")?;
        let multi_turn_task_sequencing =
            take_probe(&mut cacheable, seq_r, "multi_turn_task_sequencing")?;
        let multi_turn_memory = take_probe(&mut cacheable, mem_r, "multi_turn_memory")?;

        let ladder = {
            let _permit = sem
                .acquire()
                .await
                .map_err(|_| ProbeError::Internal("probe semaphore closed unexpectedly".into()))?;
            probes::probe_effective_context_tokens(&self.client, self.suite.skip_expensive()).await
        };
        let context_faithfulness = probes::faithfulness_from_ladder(&ladder);
        let ladder_tokens = take_ladder(&mut cacheable, ladder)?;
        // Cheap / auto-cheap stops after 4k. That pass is not a finished
        // 4k/8k/16k climb, so do not publish it as effectiveContextTokens.
        let effective_context_tokens = if self.suite.skip_expensive() {
            None
        } else {
            ladder_tokens
        };
        let probed_context_floor = ladder_tokens;

        let max_output_tokens = {
            let _permit = sem
                .acquire()
                .await
                .map_err(|_| ProbeError::Internal("probe semaphore closed unexpectedly".into()))?;
            take_max_output(probes::probe_max_output_tokens(&self.client).await)?
        };

        let xml_tool_calling = if tool_calling.level == CapabilityLevel::Strong {
            ProbeResult {
                name: "xml_tool_calling".to_string(),
                score: 1.0,
                max_score: 1.0,
                level: CapabilityLevel::Strong,
                details: XML_SKIP.to_string(),
            }
        } else {
            take_probe(
                &mut cacheable,
                Self::gated(&sem, probes::probe_xml_tool_calling(&self.client)).await,
                "xml_tool_calling",
            )?
        };

        let run = ProbeRun {
            profile: CapabilityProfile {
                model_id: self.client.model_id().to_string(),
                provider: self.client.provider().to_string(),
                tool_calling,
                json_output,
                instruction_following,
                search_replace,
                unified_diff,
                complex_tool_calling,
                nested_arguments,
                vision,
                tool_selection,
                xml_tool_calling,
                streaming_tool_calls,
                one_shot_tool_plan,
                multi_turn_task_sequencing,
                context_faithfulness,
                code_syntax,
                max_tokens_compliance,
                multi_turn_memory,
                system_message_adherence,
                token_efficiency,
                parallel_tool_scale,
                probed_at: unix_now(),
                effective_context_tokens,
                probed_context_floor,
                max_output_tokens,
            },
            cacheable,
            skip_expensive: self.suite.skip_expensive(),
            suite: self.suite,
            vision: vision_enabled,
            advertised_context_tokens: self.client.catalog().advertised_context_tokens,
        };

        {
            let mut guard = self
                .last_run
                .lock()
                .map_err(|_| ProbeError::Internal("probe runner lock poisoned".into()))?;
            *guard = Some(run.clone());
        }

        Ok(run)
    }

    async fn gated<F>(sem: &Arc<Semaphore>, fut: F) -> Result<ProbeResult, ProbeError>
    where
        F: std::future::Future<Output = Result<ProbeResult, ProbeError>>,
    {
        let _permit = sem
            .acquire()
            .await
            .map_err(|_| ProbeError::Internal("probe semaphore closed unexpectedly".into()))?;
        fut.await
    }

    async fn gated_or_skip_named<F>(
        skip: bool,
        sem: &Arc<Semaphore>,
        name: &'static str,
        details: &'static str,
        fut: F,
    ) -> Result<ProbeResult, ProbeError>
    where
        F: std::future::Future<Output = Result<ProbeResult, ProbeError>>,
    {
        if skip {
            Ok(named_skip(name, details))
        } else {
            Self::gated(sem, fut).await
        }
    }
}

fn take_probe(
    cacheable: &mut bool,
    result: Result<ProbeResult, ProbeError>,
    name: &str,
) -> Result<ProbeResult, ProbeError> {
    let (probe, ok_to_cache) = resolve_probe(result, name)?;
    *cacheable &= ok_to_cache;
    Ok(probe)
}

fn take_max_output(result: Result<Option<u32>, ProbeError>) -> Result<Option<u32>, ProbeError> {
    match result {
        Ok(n) => Ok(n),
        Err(err @ ProbeError::Auth(_)) | Err(err @ ProbeError::NotFound(_)) => Err(err),
        Err(err) if is_unreachable_host(&err) => Err(err),
        Err(_) => Ok(None),
    }
}

fn take_ladder(
    cacheable: &mut bool,
    ladder: probes::ContextLadder,
) -> Result<Option<u32>, ProbeError> {
    match ladder.error {
        Ok(()) => Ok(ladder.tokens),
        Err(err) => {
            let (_, ok_to_cache) = resolve_probe(Err(err), "effective_context_tokens")?;
            *cacheable &= ok_to_cache;
            Ok(ladder.tokens)
        }
    }
}

fn named_skip(name: &str, details: &str) -> ProbeResult {
    ProbeResult {
        name: name.to_string(),
        score: 0.5,
        max_score: 1.0,
        level: CapabilityLevel::Medium,
        details: details.to_string(),
    }
}

fn vision_skip(details: &'static str) -> ProbeResult {
    ProbeResult {
        name: "vision".to_string(),
        score: 0.0,
        max_score: 1.0,
        level: CapabilityLevel::Weak,
        details: details.to_string(),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True when the host never answered (TCP/DNS/connect), not a scored reply.
pub fn is_unreachable_host(err: &ProbeError) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    if msg.contains("failed to connect:") {
        return true;
    }
    if msg.contains("timed out") || msg.contains("timeout") {
        return false;
    }
    msg.contains("connection refused")
        || msg.contains("connect error")
        || msg.contains("dns error")
        || msg.contains("error trying to connect")
        || msg.contains("tcp connect error")
}

/// Resolve a probe result and whether it is safe to write into the 30-day cache.
///
/// Auth and unreachable hosts abort the suite (`Err`). Definitive "does
/// not support tools" on a tool-named probe is Weak and cacheable. Other
/// errors store Medium with `Probe failed:` details. Policy uses
/// [`crate::ProbeResult::completed_level`], which treats that prefix as
/// Weak. Transient errors must not be persisted as a capability score.
pub fn resolve_probe(
    result: Result<ProbeResult, ProbeError>,
    name: &str,
) -> Result<(ProbeResult, bool), ProbeError> {
    match result {
        Ok(pr) => Ok((pr, true)),
        Err(err @ ProbeError::Auth(_)) | Err(err @ ProbeError::NotFound(_)) => Err(err),
        Err(err) if is_unreachable_host(&err) => Err(err),
        Err(err) => {
            let err_msg = err.to_string();
            let is_tool_probe = TOOL_PROBE_NAMES.contains(&name);
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
