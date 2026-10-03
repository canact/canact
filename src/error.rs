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
    ///
    /// A closed port or DNS failure is [`Self::Unreachable`], not this
    /// variant. Do not prefix the message with `failed to connect:`.
    #[error("transient error: {0}")]
    Transient(String),
    /// TCP, DNS, or connect never reached the host. Suite abort.
    ///
    /// Display stays `transient error: failed to connect: {0}` so CLI
    /// text that searches that prefix still matches. Hosts match this
    /// variant or [`Self::is_connect`]. Do not scrape Display for
    /// `refused` or `dns`.
    #[error("transient error: failed to connect: {0}")]
    Unreachable(String),
    /// HTTP 429. Do not persist a 30-day score.
    ///
    /// `message` is redacted vendor text. Empty when the caller only
    /// has `Retry-After`.
    #[error("{}", rate_limit_text(*retry_after, message))]
    RateLimit {
        retry_after: Option<u64>,
        message: String,
    },
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

fn rate_limit_text(retry_after: Option<u64>, message: &str) -> String {
    let mut out = String::from("rate limited");
    if let Some(secs) = retry_after {
        out.push_str(&format!("; retry after {secs}s"));
    }
    let message = message.trim();
    if !message.is_empty() && !message.eq_ignore_ascii_case("rate limited") {
        out.push_str(": ");
        out.push_str(message);
    }
    out
}

impl ProbeError {
    /// Closed port, DNS, or connect failure. Strips a leading
    /// `failed to connect:` so Display does not repeat the prefix.
    #[must_use]
    pub fn unreachable(message: impl Into<String>) -> Self {
        let message = message.into();
        let body = message
            .strip_prefix("failed to connect:")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or(message);
        Self::Unreachable(body)
    }

    /// True for [`Self::Unreachable`]. Timeout and reset stay false.
    #[must_use]
    pub fn is_connect(&self) -> bool {
        matches!(self, Self::Unreachable(_))
    }

    /// Classify HTTP status plus body as [`ProbeError::NotFound`] when the
    /// host says the model or chat route is missing.
    ///
    /// Hosts that implement `ProbeClient` around their own HTTP client
    /// should call this before mapping the rest of the status table. A
    /// `Some` result is a suite abort, the same as Auth. Do not
    /// reimplement the needle table. This matches the shipped
    /// OpenAI-compat adapter's NotFound arm (wiremux `ClientError::NotFound`).
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

    /// Classify a typed host error Display / provider message as
    /// [`ProbeError::NotFound`].
    ///
    /// Use this when the host already mapped HTTP into something like
    /// `LlmError::Provider(msg)` and has no status code.
    /// [`Self::from_http`] and [`Self::not_found_from_body`] stay for
    /// raw HTTP. Do not reimplement the needle table.
    ///
    /// Needles (case-insensitive): the [`Self::from_http`] body needles
    /// plus folded Display copy a host already mapped (`model` plus
    /// `not found`, `try pulling`, `http 404`, or `404` plus
    /// `not found`). Validation 400 and region-forbidden copy return
    /// `None` so the host can keep Auth / Transient / Llm.
    #[must_use]
    pub fn not_found_from_message(display: &str) -> Option<Self> {
        if looks_like_model_not_found(display) || looks_like_folded_not_found_display(display) {
            Some(Self::NotFound(display.trim().chars().take(512).collect()))
        } else {
            None
        }
    }
}

fn looks_like_model_not_found(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("does not exist") || t.contains("model_not_found") || t.contains("unknown model")
}

/// Extra needles for a host Display that already folded HTTP status.
/// Keep these off `looks_like_model_not_found` so `from_http` stays
/// status-scoped.
fn looks_like_folded_not_found_display(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    (t.contains("model") && t.contains("not found"))
        || t.contains("try pulling")
        || t.contains("http 404")
        || (t.contains("404") && t.contains("not found"))
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
    fn rate_limit_text_keeps_retry_after_and_vendor_message() {
        let err = ProbeError::RateLimit {
            retry_after: Some(12),
            message: "slow down".to_owned(),
        };
        assert_eq!(err.to_string(), "rate limited; retry after 12s: slow down");
        let bare = ProbeError::RateLimit {
            retry_after: None,
            message: String::new(),
        };
        assert_eq!(bare.to_string(), "rate limited");
    }

    #[test]
    fn unreachable_strips_prefix_and_is_connect() {
        let err = ProbeError::unreachable("failed to connect: tcp reset");
        match &err {
            ProbeError::Unreachable(msg) => assert_eq!(msg, "tcp reset"),
            other => panic!("expected Unreachable, got {other:?}"),
        }
        assert!(err.is_connect());
        assert!(err.to_string().contains("failed to connect: tcp reset"));
        let plain = ProbeError::unreachable("connection refused");
        assert!(matches!(plain, ProbeError::Unreachable(msg) if msg == "connection refused"));
        assert!(!ProbeError::Transient("failed to connect: tcp".into()).is_connect());
        assert!(
            !ProbeError::Transient("connection error: error trying to connect".into()).is_connect()
        );
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
    fn from_http_needles_are_case_insensitive() {
        let body = r#"{"error":{"message":"The model `foo` Does Not Exist"}}"#;
        assert!(matches!(
            ProbeError::from_http(400, body),
            Some(ProbeError::NotFound(_))
        ));
    }

    #[test]
    fn from_http_429_with_needles_stays_none() {
        let body = r#"{"error":{"message":"the model does not exist"}}"#;
        assert!(ProbeError::from_http(429, body).is_none());
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

    #[test]
    fn not_found_from_message_matches_display_needles() {
        let msg = not_found_msg(ProbeError::not_found_from_message(
            "provider: The model `foo` does not exist",
        ));
        assert_eq!(msg, "provider: The model `foo` does not exist");
        let msg = not_found_msg(ProbeError::not_found_from_message("unknown model: bar"));
        assert_eq!(msg, "unknown model: bar");
        let msg = not_found_msg(ProbeError::not_found_from_message(
            "code=model_not_found no such id",
        ));
        assert!(msg.contains("model_not_found"), "{msg}");
    }

    #[test]
    fn not_found_from_message_needles_are_case_insensitive() {
        let msg = not_found_msg(ProbeError::not_found_from_message(
            "Provider: The model `foo` Does Not Exist",
        ));
        assert!(msg.contains("Does Not Exist"), "{msg}");
    }

    #[test]
    fn not_found_from_message_validation_and_region_stay_none() {
        assert!(ProbeError::not_found_from_message("invalid json schema for tools").is_none());
        assert!(
            ProbeError::not_found_from_message("this model is not available in your region")
                .is_none()
        );
        assert!(ProbeError::not_found_from_message("rate limited, retry later").is_none());
        assert!(ProbeError::not_found_from_message("").is_none());
        assert!(
            ProbeError::not_found_from_message("HTTP 400 Bad Request: bad json").is_none(),
            "validation 400 Display must stay host-owned"
        );
        assert!(
            ProbeError::not_found_from_message("invalid json at position 404").is_none(),
            "a bare 404 offset is not a missing-model Display"
        );
        assert!(
            ProbeError::not_found_from_message(
                "HTTP 400 Bad Request: invalid json at position 404"
            )
            .is_none(),
            "folded validation 400 plus a 404 offset must stay host-owned"
        );
    }

    #[test]
    fn not_found_from_message_matches_ollama_display() {
        let pull = "Model not found. Pull it first with: ollama pull llama3";
        let msg = not_found_msg(ProbeError::not_found_from_message(pull));
        assert_eq!(msg, pull);
        let folded = "HTTP 404 Not Found: not found";
        let msg = not_found_msg(ProbeError::not_found_from_message(folded));
        assert_eq!(msg, folded);
        let try_pulling = "try pulling the weights with ollama pull";
        let msg = not_found_msg(ProbeError::not_found_from_message(try_pulling));
        assert_eq!(msg, try_pulling);
    }

    #[test]
    fn from_http_needles_stay_status_scoped() {
        assert!(
            ProbeError::from_http(
                400,
                "Model not found. Pull it first with: ollama pull llama3"
            )
            .is_none(),
            "from_http 400 must not grow Display-only needles"
        );
        assert!(
            ProbeError::from_http(400, "HTTP 404 Not Found: not found").is_none(),
            "from_http 400 must not treat a folded 404 Display as NotFound"
        );
        assert!(ProbeError::from_http(400, "HTTP 400 Bad Request: bad json").is_none());
        let msg = not_found_msg(ProbeError::from_http(404, "HTTP 404 Not Found: not found"));
        assert!(msg.contains("404"), "{msg}");
    }
}
