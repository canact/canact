//! Probe and cache errors.

/// Errors that can occur during probing or cache I/O.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    /// Authentication failed. Suite abort; do not synthesize a score.
    #[error("authentication error: {0}")]
    Auth(String),
    /// Provider or model error (including "does not support tools").
    #[error("LLM error: {0}")]
    Llm(String),
    /// Timeout, network reset, or other transient failure.
    #[error("transient error: {0}")]
    Transient(String),
    /// HTTP 429. Do not persist a 30-day score.
    #[error("rate limited")]
    RateLimit { retry_after: Option<u64> },
    /// Filesystem I/O error (cache read/write).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Internal runtime error (e.g. poisoned lock, probes not wired).
    #[error("internal error: {0}")]
    Internal(String),
}
