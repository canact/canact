//! canact CLI. Default (no subcommand) prints `Not ready.` and exits 0.

use std::path::PathBuf;
use std::process::ExitCode;

use canact::{
    ANTHROPIC_BASE_URL, CapabilityProfile, CatalogPriors, HostOverlay, HostPolicyMeta,
    OpenAiCompatClient, ProbeCache, ProbeError, ProbeRun, ProbeRunner, XAI_BASE_URL,
    claude_code_access_token, cloud_endpoint_requires_key, default_compat_base_url,
    is_anthropic_provider_label, is_xai_provider_label, list_model_ids, looks_cheap,
    missing_model_message, overlay_context_tokens, provider_from_base_url, resolve_host_catalog,
    run_mcp_stdio,
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
    /// Write Aider or Cline overlay files from a cached probe
    Export(ExportArgs),
    /// Serve MCP stdio (`probe_model` returns host-policy JSON, not TTFT)
    Mcp,
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

    /// API key (else OPENAI_API_KEY / OPENROUTER_API_KEY / XAI_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY)
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

#[derive(clap::Args)]
struct ExportArgs {
    /// Write `.aider.model.settings.yml` and `.aider.model.metadata.json`
    #[arg(long, conflicts_with = "cline")]
    aider: bool,

    /// Write `cline.modelinfo.json`
    #[arg(long)]
    cline: bool,

    /// Model id stored in the probe cache
    #[arg(long)]
    model: String,

    /// Provider name stored in the probe cache
    #[arg(long)]
    provider: String,

    /// Probe cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Directory to write overlay files [default: current directory]
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Catalog advertised context used for min(advertised, measured)
    #[arg(long, value_name = "N")]
    advertised_context: Option<u32>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => {
            println!("Not ready.");
            ExitCode::SUCCESS
        }
        Some(Command::Probe(args)) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            match rt.block_on(run_probe(args)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(code) => ExitCode::from(code),
            }
        }
        Some(Command::Export(args)) => match run_export(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
        Some(Command::Mcp) => ExitCode::from(run_mcp_stdio()),
    }
}

async fn run_probe(args: ProbeArgs) -> Result<(), u8> {
    let provider_hint = args.provider.clone().unwrap_or_default();
    let route = resolve_api_key(args.api_key.clone(), &provider_hint);
    let api_key = route.key;
    let base_url = args.base_url.clone().unwrap_or_else(|| {
        if route.from_xai && provider_hint.is_empty() {
            XAI_BASE_URL.to_owned()
        } else if route.from_anthropic && provider_hint.is_empty() {
            ANTHROPIC_BASE_URL.to_owned()
        } else {
            default_compat_base_url(&provider_hint, route.from_openrouter)
        }
    });
    let provider = if provider_hint.is_empty() {
        provider_from_base_url(&base_url)
    } else {
        provider_hint
    };
    let cache_path = expand_tilde(args.cache.clone().unwrap_or_else(default_cache_path));
    let mut cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let vision = args.vision;

    if !args.force {
        if let Some(model) = args.model.as_deref().filter(|s| !s.is_empty()) {
            let cheap = if args.full {
                false
            } else if args.cheap {
                true
            } else {
                looks_cheap(&provider, model, &base_url)
            };
            if let Some((profile, skip_expensive)) = cached_probe(
                &cache,
                model,
                &provider,
                cheap,
                vision,
                args.advertised_context,
                args.full,
            ) {
                return emit_profile(
                    &profile,
                    args.json,
                    args.verbose,
                    HostPolicyMeta {
                        cacheable: true,
                        skip_expensive,
                        advertised_context_tokens: args.advertised_context,
                    },
                );
            }
        }
    }

    if api_key.is_none() && cloud_endpoint_requires_key(&base_url) {
        eprintln!(
            "error: set --api-key, OPENAI_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY (or pass --base-url for a local host)"
        );
        return Err(1);
    }
    let model = resolve_model(&args, &base_url, api_key.as_deref()).await?;
    let hints = resolve_host_catalog(
        args.advertised_context,
        vision_catalog_flag(&args),
        &base_url,
        api_key.as_deref(),
        &model,
    )
    .await;
    let advertised = hints.advertised_context_tokens;
    let vision = hints.supports_vision == Some(true);
    let cheap = if args.full {
        false
    } else if args.cheap {
        true
    } else {
        looks_cheap(&provider, &model, &base_url)
    };
    if !args.force {
        if let Some((profile, skip_expensive)) = cached_probe(
            &cache, &model, &provider, cheap, vision, advertised, args.full,
        ) {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                HostPolicyMeta {
                    cacheable: true,
                    skip_expensive,
                    advertised_context_tokens: advertised,
                },
            );
        }
    }
    let catalog = CatalogPriors {
        advertised_context_tokens: advertised,
        supports_vision: hints.supports_vision,
        supports_tools: None,
    };

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

fn run_export(args: ExportArgs) -> Result<(), u8> {
    if !args.aider && !args.cline {
        eprintln!("error: specify --aider or --cline");
        return Err(1);
    }
    let cache_path = expand_tilde(args.cache.clone().unwrap_or_else(default_cache_path));
    let cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let profile = match args.advertised_context {
        Some(n) => cache
            .find_profile_with_cost_and_advertised(&args.model, &args.provider, Some(n))
            .map(|(p, _)| p)
            .or_else(|| cache.find_profile(&args.model, &args.provider)),
        None => cache.find_profile(&args.model, &args.provider),
    }
    .cloned()
    .ok_or_else(|| {
        eprintln!(
            "error: no cached probe for {} / {} (run `canact probe` first)",
            args.model, args.provider
        );
        1u8
    })?;
    let overlay = if args.aider {
        HostOverlay::aider(&profile, args.advertised_context)
    } else {
        HostOverlay::cline(&profile, args.advertised_context)
    };
    let missing_window = match &overlay {
        HostOverlay::Cline(info) => info.context_window.is_none(),
        HostOverlay::Aider(_) => {
            overlay_context_tokens(&profile, args.advertised_context).is_none()
        }
    };
    if missing_window {
        eprintln!("error: no measured context window; re-run `canact probe` without --cheap");
        return Err(1);
    }
    let files = overlay.files();
    let dir = expand_tilde(args.dir.clone().unwrap_or_else(|| PathBuf::from(".")));
    if dir.exists() && !dir.is_dir() {
        eprintln!(
            "error: --dir must be a directory (got a file: {})",
            dir.display()
        );
        return Err(1);
    }
    match overlay.write_to(&dir) {
        Ok(paths) => {
            for path in paths {
                eprintln!("wrote {}", path.display());
            }
        }
        Err(err) => {
            eprintln!("error: failed to write overlays: {err}");
            return Err(1);
        }
    }
    if let Some(first) = files.first() {
        print!("{}", first.body);
    }
    Ok(())
}

fn emit_run(run: &ProbeRun, json: bool, verbose: bool) -> Result<(), u8> {
    emit_envelope(
        &run.profile,
        json,
        verbose,
        run.host_policy_envelope(),
        run.advertised_context_tokens,
    )
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
        meta.advertised_context_tokens,
    )
}

fn emit_envelope(
    profile: &CapabilityProfile,
    json: bool,
    verbose: bool,
    envelope: serde_json::Value,
    advertised: Option<u32>,
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
        print!("{}", profile.format_human_table_with(verbose, advertised));
    }
    if let Some(msg) = profile.tool_gate_error() {
        eprintln!("{msg}");
        Err(2)
    } else {
        Ok(())
    }
}

struct KeyRoute {
    key: Option<String>,
    from_openrouter: bool,
    from_xai: bool,
    from_anthropic: bool,
}

fn resolve_api_key(cli: Option<String>, provider: &str) -> KeyRoute {
    let anthropic = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .or_else(claude_code_access_token);
    resolve_api_key_from(
        cli,
        std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        std::env::var("XAI_API_KEY").ok().filter(|s| !s.is_empty()),
        anthropic,
        provider,
    )
}

fn openrouter_default_ok(provider: &str) -> bool {
    let p = provider.to_ascii_lowercase();
    p.is_empty() || p == "openrouter" || p == "openrouter.ai"
}

fn xai_default_ok(provider: &str) -> bool {
    provider.is_empty() || is_xai_provider_label(provider)
}

fn anthropic_default_ok(provider: &str) -> bool {
    provider.is_empty() || is_anthropic_provider_label(provider)
}

fn resolve_api_key_from(
    cli: Option<String>,
    openai: Option<String>,
    openrouter: Option<String>,
    xai: Option<String>,
    anthropic: Option<String>,
    provider: &str,
) -> KeyRoute {
    let from_openrouter =
        openrouter.is_some() && openai.is_none() && openrouter_default_ok(provider);
    let from_xai = xai.is_some() && openai.is_none() && xai_default_ok(provider);
    let from_anthropic = anthropic.is_some() && openai.is_none() && anthropic_default_ok(provider);
    if let Some(key) = cli {
        if !key.is_empty() {
            return KeyRoute {
                key: Some(key),
                from_openrouter,
                from_xai: from_xai && !from_openrouter,
                from_anthropic: from_anthropic && !from_openrouter && !from_xai,
            };
        }
    }
    if let Some(key) = openai {
        return KeyRoute {
            key: Some(key),
            from_openrouter: false,
            from_xai: false,
            from_anthropic: false,
        };
    }
    if from_xai {
        return KeyRoute {
            key: xai,
            from_openrouter: false,
            from_xai: true,
            from_anthropic: false,
        };
    }
    if from_anthropic {
        return KeyRoute {
            key: anthropic,
            from_openrouter: false,
            from_xai: false,
            from_anthropic: true,
        };
    }
    if let Some(key) = openrouter {
        return KeyRoute {
            key: Some(key),
            from_openrouter,
            from_xai: false,
            from_anthropic: false,
        };
    }
    KeyRoute {
        key: None,
        from_openrouter: false,
        from_xai: false,
        from_anthropic: false,
    }
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

fn default_cache_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("canact")
        .join("probes.json")
}

fn vision_catalog_flag(args: &ProbeArgs) -> Option<bool> {
    if args.vision {
        Some(true)
    } else if args.no_vision {
        Some(false)
    } else {
        None
    }
}

fn cached_probe(
    cache: &ProbeCache,
    model: &str,
    provider: &str,
    skip_expensive: bool,
    vision: bool,
    advertised: Option<u32>,
    full: bool,
) -> Option<(CapabilityProfile, bool)> {
    if let Some(profile) = cache.get_with_knobs(model, provider, skip_expensive, vision, advertised)
    {
        return Some((profile.clone(), skip_expensive));
    }
    if full || vision {
        return None;
    }
    cache
        .find_profile_with_cost_and_advertised(model, provider, advertised)
        .map(|(profile, cheap_row)| (profile.clone(), cheap_row))
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return dirs::home_dir().unwrap_or(path);
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::{expand_tilde, resolve_api_key_from};
    use canact::looks_cheap;
    use std::path::PathBuf;

    #[test]
    fn api_key_flag_plus_openrouter_env_routes_to_openrouter() {
        let route = resolve_api_key_from(
            Some("sk-or-cli".to_owned()),
            None,
            Some("sk-or-env".to_owned()),
            None,
            None,
            "",
        );
        assert_eq!(route.key.as_deref(), Some("sk-or-cli"));
        assert!(
            route.from_openrouter,
            "--api-key with OPENROUTER_API_KEY set must not default to OpenAI"
        );
        assert!(!route.from_xai);
        assert_eq!(
            canact::default_compat_base_url("", route.from_openrouter),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn api_key_flag_plus_openai_provider_stays_on_openai() {
        let route = resolve_api_key_from(
            Some("sk-proj-cli".to_owned()),
            None,
            Some("sk-or-env".to_owned()),
            None,
            None,
            "openai",
        );
        assert_eq!(route.key.as_deref(), Some("sk-proj-cli"));
        assert!(
            !route.from_openrouter,
            "--provider openai plus a CLI key must not select OpenRouter"
        );
        assert_eq!(
            canact::default_compat_base_url("openai", route.from_openrouter),
            "https://api.openai.com/v1",
            "--provider openai plus a CLI key must not hit OpenRouter"
        );
        let host = resolve_api_key_from(
            Some("sk-proj-cli".to_owned()),
            None,
            Some("sk-or-env".to_owned()),
            None,
            None,
            "api.openai.com",
        );
        assert!(!host.from_openrouter);
        assert_eq!(
            canact::default_compat_base_url("api.openai.com", host.from_openrouter),
            "https://api.openai.com/v1",
            "--provider api.openai.com plus a CLI key must not hit OpenRouter"
        );
        let env_only = resolve_api_key_from(
            None,
            None,
            Some("sk-or-env".to_owned()),
            None,
            None,
            "openai",
        );
        assert_eq!(env_only.key.as_deref(), Some("sk-or-env"));
        assert!(
            !env_only.from_openrouter,
            "OPENROUTER_API_KEY alone must not reroute --provider openai"
        );
    }

    #[test]
    fn xai_api_key_alone_routes_to_api_x_ai() {
        let route = resolve_api_key_from(None, None, None, Some("xai-env".to_owned()), None, "");
        assert_eq!(route.key.as_deref(), Some("xai-env"));
        assert!(!route.from_openrouter);
        assert!(route.from_xai);
        assert_eq!(
            if route.from_xai {
                canact::XAI_BASE_URL.to_owned()
            } else {
                canact::default_compat_base_url("", route.from_openrouter)
            },
            canact::XAI_BASE_URL
        );
        let named =
            resolve_api_key_from(None, None, None, Some("xai-env".to_owned()), None, "grok");
        assert!(named.from_xai);
        let openai = resolve_api_key_from(
            None,
            None,
            Some("sk-or-env".to_owned()),
            Some("xai-env".to_owned()),
            None,
            "openai",
        );
        assert!(!openai.from_openrouter);
        assert!(!openai.from_xai, "--provider openai must not select xAI");
    }

    #[test]
    fn anthropic_api_key_alone_routes_to_api_anthropic() {
        let route = resolve_api_key_from(None, None, None, None, Some("sk-ant-env".to_owned()), "");
        assert_eq!(route.key.as_deref(), Some("sk-ant-env"));
        assert!(route.from_anthropic);
        assert!(!route.from_xai);
        assert_eq!(
            canact::default_compat_base_url("claude", false),
            canact::ANTHROPIC_BASE_URL
        );
        let named = resolve_api_key_from(
            None,
            None,
            None,
            None,
            Some("sk-ant-env".to_owned()),
            "claude",
        );
        assert!(named.from_anthropic);
        let openai = resolve_api_key_from(
            None,
            None,
            None,
            None,
            Some("sk-ant-env".to_owned()),
            "openai",
        );
        assert!(
            !openai.from_anthropic,
            "--provider openai must not select Anthropic"
        );
        let xai_only =
            resolve_api_key_from(None, None, None, Some("xai-env".to_owned()), None, "claude");
        assert!(
            xai_only.key.is_none(),
            "--provider claude must not reuse XAI_API_KEY"
        );
        assert!(!xai_only.from_xai);
        assert!(!xai_only.from_anthropic);
        let both_cli = resolve_api_key_from(
            Some("sk-cli".to_owned()),
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "",
        );
        assert!(
            both_cli.from_xai,
            "--api-key plus XAI_API_KEY and ANTHROPIC_* must keep the xAI default"
        );
        assert!(!both_cli.from_anthropic);
        assert!(!both_cli.from_openrouter);
        let both_env = resolve_api_key_from(
            None,
            None,
            None,
            Some("xai-env".to_owned()),
            Some("sk-ant-env".to_owned()),
            "",
        );
        assert!(both_env.from_xai);
        assert!(!both_env.from_anthropic);
        let anthropic_or = resolve_api_key_from(
            None,
            None,
            Some("sk-or-env".to_owned()),
            None,
            Some("sk-ant-env".to_owned()),
            "",
        );
        assert!(anthropic_or.from_anthropic);
        assert!(!anthropic_or.from_openrouter);
    }

    #[test]
    fn expand_tilde_joins_home_for_export_dir() {
        let home = dirs::home_dir().expect("home");
        assert_eq!(
            expand_tilde(PathBuf::from("~/overlays")),
            home.join("overlays")
        );
        assert_eq!(expand_tilde(PathBuf::from("~")), home);
        assert_eq!(
            expand_tilde(PathBuf::from("/tmp/overlays")),
            PathBuf::from("/tmp/overlays")
        );
    }

    #[test]
    fn looks_cheap_treats_ipv6_loopback_like_localhost() {
        assert!(looks_cheap("openai-compat", "llama3", "http://[::1]:11434"));
        assert!(looks_cheap("::1", "llama3", "http://example.invalid/v1"));
        assert!(looks_cheap("[::1]", "llama3", "http://example.invalid/v1"));
        assert!(looks_cheap("localhost", "llama3", "http://localhost:11434"));
        assert!(!looks_cheap(
            "openai",
            "gpt-4o",
            "https://api.openai.com/v1"
        ));
        assert!(
            !looks_cheap("openai", "gpt-4o", "https://[2001:db8::1]/v1"),
            "non-loopback IPv6 must not match via a naive ::1 substring"
        );
        assert!(looks_cheap(
            "127.0.0.1:1234",
            "llama3",
            "http://example.invalid/v1"
        ));
        assert!(looks_cheap(
            "localhost:11434",
            "llama3",
            "http://example.invalid/v1"
        ));
        assert_eq!(
            canact::default_compat_base_url("127.0.0.1:1234", false),
            "http://127.0.0.1:1234/v1",
            "--provider 127.0.0.1:1234 without --base-url must stay on loopback"
        );
    }
}
