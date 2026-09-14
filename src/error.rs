//! Probe and cache errors.

use serde_json::Value;

/// Errors that can occur during probing or cache I/O.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    /// Authentication failed. Suite abort; do not synthesize a score.
    #[error("authentication error: {0}")]
    Auth(String),
    /// Host says the model or chat route does not exist. Suite abort.
    #[error("not found: {0}")]
    NotFound(String),
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

impl ProbeError {
    /// Classify HTTP status plus body as [`ProbeError::NotFound`] when the
    /// host says the model or chat route is missing.
    ///
    /// Hosts that implement `ProbeClient` around their own HTTP client
    /// should call this before mapping the rest of the status table. A
    /// `Some` result is a suite abort, the same as Auth. Do not
    /// reimplement the needle table; this is the same classifier the
    /// shipped OpenAI-compat adapter uses.
    ///
    /// Returns `Some` for:
    /// - HTTP 404 (Ollama missing model)
    /// - HTTP 400 or 403 whose body looks like a missing model
    ///   (`does not exist`, `unknown model`, `model_not_found`)
    /// - HTTP 2xx with an `error` object whose body or message matches
    ///   those needles (a numeric `code` 400 is not enough on its own)
    ///
    /// Validation 400 and region-forbidden 403 return `None` so the
    /// host can keep its own Auth / Transient / Llm mapping.
    #[must_use]
    pub fn from_http(status: u16, body: &str) -> Option<Self> {
        let message = http_error_message(body);
        if status == 404 {
            return Some(Self::NotFound(message));
        }
        let wrapped_error = (200..300).contains(&status) && json_has_error_object(body);
        if (status == 400 || status == 403 || wrapped_error)
            && (looks_like_model_not_found(body) || looks_like_model_not_found(&message))
        {
            return Some(Self::NotFound(message));
        }
        None
    }

    /// Classify a response body as [`ProbeError::NotFound`] without a
    /// status (SSE payloads or a 200-wrapped `error` object).
    ///
    /// Use [`Self::from_http`] when the HTTP status is known. This
    /// helper is only the body needles.
    #[must_use]
    pub fn not_found_from_body(body: &str) -> Option<Self> {
        let message = http_error_message(body);
        if looks_like_model_not_found(body) || looks_like_model_not_found(&message) {
            Some(Self::NotFound(message))
        } else {
            None
        }
    }
}

fn looks_like_model_not_found(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("does not exist") || t.contains("model_not_found") || t.contains("unknown model")
}

fn json_has_error_object(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    value.get("error").is_some_and(Value::is_object)
}

fn http_error_message(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return msg.to_string();
        }
        if let Some(msg) = value
            .get("message")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return msg.to_string();
        }
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        "request failed".into()
    } else {
        trimmed.chars().take(512).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeError;

    fn not_found_msg(err: Option<ProbeError>) -> String {
        match err {
            Some(ProbeError::NotFound(msg)) => msg,
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn from_http_404_is_not_found() {
        let msg = not_found_msg(ProbeError::from_http(404, "model 'x' not found"));
        assert!(msg.contains("not found"), "{msg}");
    }

    #[test]
    fn from_http_400_does_not_exist_is_not_found() {
        let body = r#"{"error":{"message":"The model `foo` does not exist"}}"#;
        let msg = not_found_msg(ProbeError::from_http(400, body));
        assert!(msg.contains("does not exist"), "{msg}");
    }

    #[test]
    fn from_http_400_unknown_model_is_not_found() {
        let body = r#"{"error":{"message":"unknown model: bar"}}"#;
        let msg = not_found_msg(ProbeError::from_http(400, body));
        assert!(msg.contains("unknown model"), "{msg}");
    }

    #[test]
    fn from_http_400_model_not_found_code_is_not_found() {
        let body = r#"{"error":{"message":"no such id","code":"model_not_found"}}"#;
        let msg = not_found_msg(ProbeError::from_http(400, body));
        assert!(msg.contains("no such id"), "{msg}");
    }

    #[test]
    fn from_http_200_wrapped_400_unknown_model_is_not_found() {
        let body = r#"{"error":{"message":"The model `foo` does not exist","code":400}}"#;
        let msg = not_found_msg(ProbeError::from_http(200, body));
        assert!(msg.contains("does not exist"), "{msg}");
    }

    #[test]
    fn from_http_403_unknown_model_is_not_found() {
        let body = r#"{"error":{"message":"unknown model"}}"#;
        assert!(matches!(
            ProbeError::from_http(403, body),
            Some(ProbeError::NotFound(_))
        ));
    }

    #[test]
    fn from_http_400_validation_stays_none() {
        let body = r#"{"error":{"message":"invalid json schema for tools"}}"#;
        assert!(ProbeError::from_http(400, body).is_none());
    }

    #[test]
    fn from_http_403_region_forbidden_stays_none() {
        let body = r#"{"error":{"message":"this model is not available in your region"}}"#;
        assert!(ProbeError::from_http(403, body).is_none());
    }

    #[test]
    fn from_http_401_is_none() {
        let body = r#"{"error":{"message":"Incorrect API key provided"}}"#;
        assert!(ProbeError::from_http(401, body).is_none());
    }

    #[test]
    fn from_http_200_without_error_object_is_none() {
        assert!(ProbeError::from_http(200, r#"{"choices":[]}"#).is_none());
        assert!(ProbeError::from_http(200, "the model does not exist").is_none());
    }

    #[test]
    fn from_http_200_numeric_code_400_without_needles_is_none() {
        assert!(ProbeError::from_http(200, r#"{"error":{"code":400}}"#).is_none());
    }

    #[test]
    fn from_http_empty_404_uses_fallback_message() {
        let msg = not_found_msg(ProbeError::from_http(404, "   "));
        assert_eq!(msg, "request failed");
    }

    #[test]
    fn not_found_from_body_matches_needles() {
        let msg = not_found_msg(ProbeError::not_found_from_body(
            r#"{"error":{"message":"unknown model"}}"#,
        ));
        assert!(msg.contains("unknown model"), "{msg}");
        assert!(ProbeError::not_found_from_body("invalid json schema").is_none());
    }
}
