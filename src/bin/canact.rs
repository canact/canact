//! canact CLI. No subcommand prints help.

use std::path::PathBuf;
use std::process::ExitCode;

use canact::{
    CapabilityProfile, CatalogPriors, HostOverlay, HostPolicyMeta, OpenAiCompatClient,
    PlumbingMatrix, ProbeCache, ProbeError, ProbeRun, ProbeRunner, SuiteTier,
    claude_code_access_token, finalize_key_route, is_bedrock_provider_label,
    is_groq_provider_label, list_model_ids, looks_cheap, missing_cloud_key_message,
    missing_model_message, present_base_url, refuse_cloud_without_key, resolve_api_key_from,
    resolve_host_catalog, run_mcp_stdio, should_load_claude_code_login, should_load_xai_oauth,
    xai_oauth_access_token,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "canact",
    version,
    about = "Probe a model and print host-policy results",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Probe a model and print host-policy results
    Probe(ProbeArgs),
    /// Write Aider or Cline overlay files from a cached probe
    Export(ExportArgs),
    /// Plumbing table from cached probes (pass / degraded / fail, no rank)
    Matrix(MatrixArgs),
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

    /// Suite tier: policy (host-policy only), full (+ sequencing/ladder), all (+ diagnostics)
    #[arg(long, value_name = "policy|full|all")]
    suite: Option<String>,

    /// Alias of `--suite=policy`
    #[arg(long, conflicts_with = "full")]
    cheap: bool,

    /// Alias of `--suite=full`
    #[arg(long)]
    full: bool,

    /// Cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Catalog prior: advertised context window in tokens
    #[arg(long, value_name = "N", value_parser = parse_advertised_context)]
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

    /// Catalog advertised context: cache-row key and overlay window
    #[arg(long, value_name = "N", value_parser = parse_advertised_context)]
    advertised_context: Option<u32>,
}

#[derive(clap::Args)]
struct MatrixArgs {
    /// Provider whose cached models appear in the table
    #[arg(long)]
    provider: String,

    /// Probe cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Probe(args) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            match rt.block_on(run_probe(args)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(code) => ExitCode::from(code),
            }
        }
        Command::Export(args) => match run_export(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
        Command::Matrix(args) => match run_matrix(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
        Command::Mcp => ExitCode::from(run_mcp_stdio()),
    }
}

fn cli_explicit_base_url(raw: Option<&str>) -> bool {
    present_base_url(raw).is_some()
}

async fn run_probe(args: ProbeArgs) -> Result<(), u8> {
    let provider_hint = args
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("")
        .to_owned();
    let first = resolve_api_key(
        args.api_key.clone(),
        &provider_hint,
        cli_explicit_base_url(args.base_url.as_deref()),
    );
    let (route, base_url, provider) =
        finalize_key_route(&provider_hint, args.base_url.clone(), first, |provider| {
            resolve_api_key(args.api_key.clone(), provider, false)
        });
    let api_key = route.key.clone();
    let cache_path = resolve_user_path(args.cache.clone(), default_cache_path());
    let mut cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let vision = args.vision;
    let suite = match resolve_suite(&args) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("error: {msg}");
            return Err(1);
        }
    };

    if !args.force
        && let Some(model) = args
            .model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    {
        if vision_catalog_flag(&args).is_none()
            && args.advertised_context.is_none()
            && let Some((profile, _skip_expensive, advertised)) = cache
                .find_profile_unspecified_catalog_suite(model, &provider, suite)
                .map(|(p, c, a)| (p.clone(), c, a))
        {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                HostPolicyMeta::for_suite(true, true, suite, advertised),
            );
        }
        if let Some((profile, hit_suite, advertised)) = cached_probe(
            &cache,
            model,
            &provider,
            suite,
            vision,
            args.advertised_context,
        ) {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                HostPolicyMeta::for_suite(true, true, hit_suite, advertised),
            );
        }
    }

    if refuse_cloud_without_key(api_key.as_deref(), &base_url) {
        eprintln!("{}", missing_cloud_key_message(&provider, &base_url));
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
    if !args.force
        && let Some((profile, hit_suite, advertised)) =
            cached_probe(&cache, &model, &provider, suite, vision, advertised)
    {
        return emit_profile(
            &profile,
            args.json,
            args.verbose,
            HostPolicyMeta::for_suite(true, true, hit_suite, advertised),
        );
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

    let mut runner = ProbeRunner::new(client).suite(suite);
    if looks_cheap(&provider, &model, &base_url) {
        runner = runner.throttled();
    }

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
    let cache_path = resolve_user_path(args.cache.clone(), default_cache_path());
    let cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let model = args.model.trim();
    let provider = args.provider.trim();
    let (profile, cached_advertised) = match args.advertised_context {
        Some(n) => cache
            .find_profile_with_cost_and_advertised(model, provider, Some(n))
            .map(|(p, _)| (p, Some(n)))
            .or_else(|| cache.find_profile_and_advertised(model, provider)),
        None => cache.find_profile_and_advertised(model, provider),
    }
    .map(|(p, advertised)| (p.clone(), advertised))
    .ok_or_else(|| {
        if let Some(stale) = cache.stale_suite_version(model, provider) {
            eprintln!(
                "error: cached probe for {model} / {provider} is suite {stale} (need {}); run `canact probe` again",
                canact::PROBE_SUITE_VERSION
            );
        } else {
            eprintln!(
                "error: no cached probe for {model} / {provider} (run `canact probe` first)",
            );
        }
        1u8
    })?;
    let advertised = args.advertised_context.or(cached_advertised);
    let overlay = if args.aider {
        HostOverlay::aider(&profile, advertised)
    } else {
        HostOverlay::cline(&profile, advertised)
    };
    let files = overlay.files();
    let dir = resolve_user_path(args.dir.clone(), PathBuf::from("."));
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

fn run_matrix(args: MatrixArgs) -> Result<(), u8> {
    let provider = args.provider.trim();
    if provider.is_empty() {
        eprintln!("error: --provider is required");
        return Err(1);
    }
    let cache_path = resolve_user_path(args.cache.clone(), default_cache_path());
    let cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let matrix = PlumbingMatrix::from_cache(&cache, provider);
    if matrix.rows.is_empty() {
        if let Some(stale) = cache.stale_suite_version_for_provider(provider) {
            eprintln!(
                "error: cached {provider} probes are suite {stale} (need {}); run `canact probe` again",
                canact::PROBE_SUITE_VERSION
            );
        } else {
            eprintln!("error: no cached probes for {provider} (run `canact probe` first)",);
        }
        return Err(1);
    }
    match serde_json::to_string_pretty(&matrix) {
        Ok(s) => {
            println!("{s}");
            Ok(())
        }
        Err(err) => {
            eprintln!("error: failed to serialize matrix JSON: {err}");
            Err(1)
        }
    }
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
        if envelope.get("fromCache").and_then(|v| v.as_bool()) == Some(true) {
            println!("Cached (probedAt={})", profile.probed_at);
        }
        print!("{}", profile.format_human_table_with(verbose, advertised));
    }
    if let Some(msg) = profile.tool_gate_error() {
        eprintln!("{msg}");
        Err(2)
    } else {
        Ok(())
    }
}

fn resolve_api_key(
    cli: Option<String>,
    provider: &str,
    explicit_base_url: bool,
) -> canact::KeyRoute {
    if is_groq_provider_label(provider) {
        let groq = std::env::var("GROQ_API_KEY").ok().filter(|s| !s.is_empty());
        return resolve_api_key_from(cli.or(groq), None, None, None, None, provider);
    }
    if is_bedrock_provider_label(provider) {
        let bedrock = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
            .ok()
            .filter(|s| !s.is_empty());
        return resolve_api_key_from(cli.or(bedrock), None, None, None, None, provider);
    }
    let openai = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let openrouter = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty());
    let xai_env = std::env::var("XAI_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("GROK_API_KEY").ok().filter(|s| !s.is_empty()));
    let other_before_xai_oauth = openai.is_some()
        || openrouter.is_some()
        || xai_env.is_some()
        || cli.as_ref().is_some_and(|s| !s.is_empty());
    let xai = xai_env.or_else(|| {
        if should_load_xai_oauth(provider, other_before_xai_oauth, explicit_base_url) {
            xai_oauth_access_token()
        } else {
            None
        }
    });
    let other_cloud_keys = openai.is_some()
        || openrouter.is_some()
        || xai.is_some()
        || cli.as_ref().is_some_and(|s| !s.is_empty());
    let anthropic = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            if should_load_claude_code_login(provider, other_cloud_keys, explicit_base_url) {
                claude_code_access_token()
            } else {
                None
            }
        });
    resolve_api_key_from(cli, openai, openrouter, xai, anthropic, provider)
}

async fn resolve_model(
    args: &ProbeArgs,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<String, u8> {
    if let Some(model) = args
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Ok(model.to_owned());
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

fn parse_advertised_context(raw: &str) -> Result<u32, String> {
    let trimmed = raw.trim();
    let n: u32 = trimmed.parse().map_err(|_| {
        format!(
            "invalid value '{raw}' for '--advertised-context <N>': invalid digit found in string"
        )
    })?;
    if n < 1 {
        return Err(format!(
            "invalid value '{raw}' for '--advertised-context <N>': {n} is not in 1.."
        ));
    }
    Ok(n)
}

fn resolve_user_path(raw: Option<PathBuf>, default: PathBuf) -> PathBuf {
    match raw {
        Some(p) if !p.to_string_lossy().trim().is_empty() => expand_tilde(p),
        _ => default,
    }
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
    suite: SuiteTier,
    vision: bool,
    advertised: Option<u32>,
) -> Option<(CapabilityProfile, SuiteTier, Option<u32>)> {
    if let Some(profile) = cache.get_with_suite(model, provider, suite, vision, advertised) {
        return Some((profile.clone(), suite, advertised));
    }
    if !matches!(suite, SuiteTier::Policy) || vision {
        return None;
    }
    cache
        .find_profile_with_cost_and_advertised(model, provider, advertised)
        .map(|(profile, hit_suite)| (profile.clone(), hit_suite, advertised))
}

fn resolve_suite(args: &ProbeArgs) -> Result<SuiteTier, String> {
    if let Some(raw) = args.suite.as_deref() {
        let parsed = SuiteTier::parse(raw)
            .ok_or_else(|| format!("unknown --suite={raw} (expected policy, full, or all)"))?;
        if args.cheap && parsed != SuiteTier::Policy {
            return Err(format!("--cheap conflicts with --suite={raw}"));
        }
        if args.full && parsed != SuiteTier::Full {
            return Err(format!("--full conflicts with --suite={raw}"));
        }
        return Ok(parsed);
    }
    if args.full {
        Ok(SuiteTier::Full)
    } else {
        // `--cheap` and the no-flag default are the policy tier.
        Ok(SuiteTier::Policy)
    }
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(home) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(home));
        }
    }
    dirs::home_dir()
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let owned = path.to_string_lossy();
    let raw = owned.trim();
    if raw == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    if raw == owned.as_ref() {
        path
    } else {
        PathBuf::from(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::{cli_explicit_base_url, expand_tilde};
    use canact::{looks_cheap, resolve_api_key_from, should_load_xai_oauth};
    use std::path::PathBuf;

    #[test]
    fn whitespace_only_base_url_does_not_skip_oauth() {
        assert!(!cli_explicit_base_url(Some("  ")));
        assert!(!cli_explicit_base_url(Some("")));
        assert!(!cli_explicit_base_url(None));
        assert!(cli_explicit_base_url(Some(" http://127.0.0.1:11434 ")));
        assert!(should_load_xai_oauth(
            "",
            false,
            cli_explicit_base_url(Some("  "))
        ));
        assert!(!should_load_xai_oauth(
            "",
            false,
            cli_explicit_base_url(Some("http://127.0.0.1:11434"))
        ));
    }

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
        assert_eq!(
            expand_tilde(PathBuf::from(" ~/overlays ")),
            home.join("overlays")
        );
        assert_eq!(expand_tilde(PathBuf::from(" ~ ")), home);
    }

    #[test]
    fn resolve_user_path_whitespace_uses_default() {
        let default = PathBuf::from("/tmp/canact-default");
        assert_eq!(super::resolve_user_path(None, default.clone()), default);
        assert_eq!(
            super::resolve_user_path(Some(PathBuf::from("   ")), default.clone()),
            default
        );
        assert_eq!(
            super::resolve_user_path(Some(PathBuf::from("")), default.clone()),
            default
        );
        assert_eq!(
            super::resolve_user_path(Some(PathBuf::from("/tmp/overlays")), default),
            PathBuf::from("/tmp/overlays")
        );
    }

    #[test]
    fn default_cache_path_joins_dirs_cache() {
        let expected = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("canact")
            .join("probes.json");
        assert_eq!(super::default_cache_path(), expected);
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
