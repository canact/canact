//! canact CLI. Default (no subcommand) prints `Not ready.` and exits 0.

use std::path::PathBuf;
use std::process::ExitCode;

use canact::{
    CapabilityProfile, CatalogPriors, HostPolicyMeta, OpenAiCompatClient, ProbeCache, ProbeError,
    ProbeRun, ProbeRunner, list_model_ids, missing_model_message,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "canact", version, about = "Reserved.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Probe a model and print host-policy results
    Probe(ProbeArgs),
}

#[derive(clap::Args)]
struct ProbeArgs {
    /// Model id (required unless GET /v1/models returns exactly one id)
    #[arg(long)]
    model: Option<String>,

    /// Provider name [default: URL host or openai-compat]
    #[arg(long)]
    provider: Option<String>,

    /// OpenAI-compatible base URL
    #[arg(long)]
    base_url: Option<String>,

    /// API key (else OPENAI_API_KEY / OPENROUTER_API_KEY)
    #[arg(long)]
    api_key: Option<String>,

    /// Catalog prior: advertise vision support
    #[arg(long, conflicts_with = "no_vision")]
    vision: bool,

    /// Catalog prior: do not advertise vision
    #[arg(long = "no-vision")]
    no_vision: bool,

    /// Print canact host-policy JSON envelope
    #[arg(long)]
    json: bool,

    /// Print all 20 dimensions (human table)
    #[arg(long)]
    verbose: bool,

    /// Ignore cache
    #[arg(long)]
    force: bool,

    /// Alias of new_throttled
    #[arg(long, conflicts_with = "full")]
    cheap: bool,

    /// Alias of new (paid suite even on local/free)
    #[arg(long)]
    full: bool,

    /// Cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Catalog prior: advertised context window in tokens
    #[arg(long, value_name = "N")]
    advertised_context: Option<u32>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => {
            println!("Not ready.");
            ExitCode::SUCCESS
        }
        Some(Command::Probe(args)) => match run_probe(args).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
    }
}

async fn run_probe(args: ProbeArgs) -> Result<(), u8> {
    let (api_key, from_openrouter) = resolve_api_key(args.api_key.clone());
    let base_url = args.base_url.clone().unwrap_or_else(|| {
        if from_openrouter {
            "https://openrouter.ai/api/v1".to_owned()
        } else {
            "https://api.openai.com/v1".to_owned()
        }
    });
    let model = resolve_model(&args, &base_url, api_key.as_deref()).await?;
    let provider = args
        .provider
        .clone()
        .unwrap_or_else(|| provider_from_url(&base_url));
    let catalog = CatalogPriors {
        advertised_context_tokens: args.advertised_context,
        supports_vision: if args.vision {
            Some(true)
        } else if args.no_vision {
            Some(false)
        } else {
            None
        },
        supports_tools: None,
    };
    let cache_path = args.cache.clone().unwrap_or_else(default_cache_path);
    let mut cache = ProbeCache::load(&cache_path).unwrap_or_default();
    let cheap = if args.full {
        false
    } else if args.cheap {
        true
    } else {
        looks_cheap(&provider, &model, &base_url)
    };
    let vision = args.vision;

    if !args.force {
        if let Some(profile) = cache
            .get_with_knobs(&model, &provider, cheap, vision)
            .cloned()
        {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                HostPolicyMeta {
                    cacheable: true,
                    skip_expensive: cheap,
                    advertised_context_tokens: args.advertised_context,
                },
            );
        }
    }

    let client = OpenAiCompatClient::new(
        base_url.clone(),
        api_key,
        model.clone(),
        provider.clone(),
        catalog,
    )
    .map_err(|e| {
        eprintln!("error: {e}");
        1u8
    })?;

    let runner = if cheap {
        ProbeRunner::new_throttled(client)
    } else {
        ProbeRunner::new(client)
    };

    if !args.json {
        println!("Probing {model} ({provider})...");
        println!();
    }

    let run = match runner.run_detailed().await {
        Ok(run) => run,
        Err(err) => {
            eprintln!("error: {err}");
            return Err(1);
        }
    };

    if let Err(err) = run.persist(&mut cache, &cache_path) {
        eprintln!("warning: failed to save probe cache: {err}");
    }

    emit_run(&run, args.json, args.verbose)
}

fn emit_run(run: &ProbeRun, json: bool, verbose: bool) -> Result<(), u8> {
    emit_envelope(&run.profile, json, verbose, run.host_policy_envelope())
}

fn emit_profile(
    profile: &CapabilityProfile,
    json: bool,
    verbose: bool,
    meta: HostPolicyMeta,
) -> Result<(), u8> {
    emit_envelope(
        profile,
        json,
        verbose,
        profile.host_policy_envelope_with(meta),
    )
}

fn emit_envelope(
    profile: &CapabilityProfile,
    json: bool,
    verbose: bool,
    envelope: serde_json::Value,
) -> Result<(), u8> {
    if json {
        match serde_json::to_string_pretty(&envelope) {
            Ok(s) => println!("{s}"),
            Err(err) => {
                eprintln!("error: failed to serialize probe JSON: {err}");
                return Err(1);
            }
        }
    } else {
        print!("{}", profile.format_human_table(verbose));
    }
    if let Some(msg) = profile.tool_gate_error() {
        eprintln!("{msg}");
        Err(2)
    } else {
        Ok(())
    }
}

fn resolve_api_key(cli: Option<String>) -> (Option<String>, bool) {
    if let Some(key) = cli {
        if !key.is_empty() {
            return (Some(key), false);
        }
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if !key.is_empty() {
            return (Some(key), false);
        }
    }
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        if !key.is_empty() {
            return (Some(key), true);
        }
    }
    (None, false)
}

async fn resolve_model(
    args: &ProbeArgs,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<String, u8> {
    if let Some(model) = args.model.as_deref() {
        if !model.is_empty() {
            return Ok(model.to_owned());
        }
    }
    match list_model_ids(base_url, api_key).await {
        Ok(ids) if ids.len() == 1 => Ok(ids[0].clone()),
        Ok(ids) => {
            eprintln!("{}", missing_model_message(&ids));
            Err(1)
        }
        Err(err @ ProbeError::Auth(_)) => {
            eprintln!("error: {err}");
            Err(1)
        }
        Err(err) => {
            eprintln!("error: --model is required ({err})");
            Err(1)
        }
    }
}

fn provider_from_url(base_url: &str) -> String {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_owned))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "openai-compat".to_owned())
}

fn looks_cheap(provider: &str, model: &str, base_url: &str) -> bool {
    let provider = provider.to_ascii_lowercase();
    let url = base_url.to_ascii_lowercase();
    model.contains(":free")
        || provider == "ollama"
        || provider == "lmstudio"
        || provider == "vllm"
        || provider == "localhost"
        || provider == "127.0.0.1"
        || is_ipv6_loopback_label(&provider)
        || url.contains("localhost")
        || url.contains("127.0.0.1")
        || url.contains("0.0.0.0")
        || url.contains("[::1]")
        || ipv6_loopback_host(base_url)
}

fn is_ipv6_loopback_label(host: &str) -> bool {
    let host = host.trim();
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    host == "::1"
}

fn ipv6_loopback_host(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|h| is_ipv6_loopback_label(&h))
}

fn default_cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("canact")
        .join("probes.json")
}

#[cfg(test)]
mod tests {
    use super::{ipv6_loopback_host, looks_cheap};

    #[test]
    fn looks_cheap_treats_ipv6_loopback_like_localhost() {
        assert!(looks_cheap("openai-compat", "llama3", "http://[::1]:11434"));
        assert!(looks_cheap("::1", "llama3", "http://example.invalid/v1"));
        assert!(looks_cheap("[::1]", "llama3", "http://example.invalid/v1"));
        assert!(looks_cheap("localhost", "llama3", "http://localhost:11434"));
        assert!(
            ipv6_loopback_host("http://[::1]:11434"),
            "parsed [::1] host must be cheap"
        );
        assert!(!looks_cheap(
            "openai",
            "gpt-4o",
            "https://api.openai.com/v1"
        ));
        assert!(
            !looks_cheap("openai", "gpt-4o", "https://[2001:db8::1]/v1"),
            "non-loopback IPv6 must not match via a naive ::1 substring"
        );
        assert!(!ipv6_loopback_host("https://[2001:db8::1]/v1"));
    }
}
