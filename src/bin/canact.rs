//! canact CLI. No subcommand prints help.

use std::path::PathBuf;
use std::process::ExitCode;

use canact::{
    CacheListRow, CapabilityLevel, CapabilityProfile, CatalogPriors, FailOn, HostOverlay,
    HostPolicyMeta, McpServerOptions, OpenAiCompatClient, PlumbingMatrix, ProbeCache, ProbeError,
    ProbeRun, ProbeRunner, ProbeTool, SuiteTier, claude_code_access_token, finalize_key_route,
    invalid_explicit_base_url, is_bedrock_provider_label, is_groq_provider_label, list_model_ids,
    looks_cheap, missing_cloud_key_message, missing_model_message, planned_probe_names,
    present_base_url, present_secret, probe_endpoint_without_key, probe_tools_digest,
    redact_base_url, refuse_cloud_without_key, resolve_api_key_from, resolve_host_catalog,
    run_mcp_stdio_with, shipped_profile_base_conflict, should_load_claude_code_login,
    should_load_xai_oauth, with_route_error_label, xai_oauth_access_token,
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
    /// Plumbing table from cached probes (pass / degraded / fail / skipped, no rank)
    Matrix(MatrixArgs),
    /// Print the cache path or list cached models
    Cache(CacheArgs),
    /// Serve MCP stdio (`probe_model` returns host-policy JSON, not TTFT)
    Mcp(McpArgs),
}

#[derive(clap::Args)]
struct CacheArgs {
    #[command(subcommand)]
    command: CacheCommand,
}

#[derive(Subcommand)]
enum CacheCommand {
    /// Print the cache file path. Does not create the file.
    Path(CachePathArgs),
    /// List cached probes, including an older suite version.
    List(CacheListArgs),
}

#[derive(clap::Args)]
struct CachePathArgs {
    /// Probe cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,
}

#[derive(clap::Args)]
struct CacheListArgs {
    /// Provider whose cached models are listed (omit for every provider)
    #[arg(long)]
    provider: Option<String>,

    /// Probe cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,
}

#[derive(clap::Args)]
struct McpArgs {
    /// Env var the tool may read. The tool cannot name a different variable.
    #[arg(long)]
    api_key_env: Option<String>,

    /// Base URL for probes. The tool cannot replace it.
    #[arg(long)]
    base_url: Option<String>,

    /// Let the tool pass a base URL that is not a loopback host.
    #[arg(long)]
    allow_base_url: bool,

    /// Let the tool pass a cache path outside the default cache directory.
    #[arg(long)]
    allow_cache: bool,
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

    /// API key (else OPENAI_API_KEY / OPENROUTER_API_KEY / XAI_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY). The value is visible in shell history and the process list; prefer an env var.
    #[arg(long)]
    api_key: Option<String>,

    /// Skip stored Grok and Claude Code logins. Env vars and --api-key still apply.
    #[arg(long)]
    no_login: bool,

    /// Catalog prior: advertise vision support
    #[arg(long, conflicts_with = "no_vision")]
    vision: bool,

    /// Catalog prior: do not advertise vision
    #[arg(long = "no-vision")]
    no_vision: bool,

    /// Print canact host-policy JSON envelope
    #[arg(long)]
    json: bool,

    /// Print the provider, base URL, suite, and probe names, then exit. Skips login, cache, and HTTP.
    #[arg(long)]
    dry_run: bool,

    /// Print every dimension in the human table. Omits one_shot_tool_plan.
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

    /// Exit 2 when a completed dimension is below the bar. Skipped and failed probes do not count.
    #[arg(long, value_name = "weak|degraded")]
    fail_on: Option<String>,

    /// Cache file [default: platform cache dir / canact / probes.json]
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Catalog prior: advertised context window in tokens
    #[arg(long, value_name = "N", value_parser = parse_advertised_context)]
    advertised_context: Option<u32>,

    /// JSON file of caller tools. The file is a JSON array of objects with name, description, and parameters. Omitted means builtin probe tools.
    #[arg(long)]
    tools: Option<PathBuf>,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Write the Aider repo files and `cline.modelinfo.json`. Stdout stays empty. Overlay windows are the advertised context.
    #[arg(long, conflicts_with_all = ["aider", "cline"])]
    all: bool,

    /// Write `.aider.model.settings.yml` and `.aider.model.metadata.json`. Aider loads both from the repo.
    #[arg(long, conflicts_with = "cline")]
    aider: bool,

    /// Write `cline.modelinfo.json`. Paste this file into Cline. Cline does not load it from the repo.
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
    /// Provider whose cached models appear in the table (omit for every provider)
    #[arg(long)]
    provider: Option<String>,

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
        Command::Cache(args) => match run_cache(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
        Command::Mcp(args) => ExitCode::from(run_mcp_stdio_with(McpServerOptions {
            allow_cache: args.allow_cache,
            allow_base_url: args.allow_base_url,
            api_key_env: args.api_key_env,
            base_url: args.base_url,
        })),
    }
}

fn cli_explicit_base_url(raw: Option<&str>) -> bool {
    present_base_url(raw).is_some()
}

async fn run_probe(args: ProbeArgs) -> Result<(), u8> {
    let fail_on = match parse_fail_on(args.fail_on.as_deref()) {
        Ok(bar) => bar,
        Err(msg) => {
            eprintln!("{msg}");
            return Err(1);
        }
    };
    let caller_tools = match load_caller_tools(args.tools.as_deref()) {
        Ok(tools) => tools,
        Err(msg) => {
            eprintln!("error: {msg}");
            return Err(1);
        }
    };
    if args.dry_run {
        return run_dry_run(&args);
    }
    if let Some(msg) = invalid_explicit_base_url(args.base_url.as_deref()) {
        eprintln!("error: {msg}");
        return Err(1);
    }
    let provider_hint = args
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("")
        .to_owned();
    let first = match resolve_api_key(
        args.api_key.clone(),
        &provider_hint,
        cli_explicit_base_url(args.base_url.as_deref()),
        args.no_login,
    ) {
        Ok(route) => route,
        Err(msg) => {
            eprintln!("error: authentication error: {msg}");
            return Err(1);
        }
    };
    let (route, base_url, provider) =
        match finalize_key_route(&provider_hint, args.base_url.clone(), first, |provider| {
            resolve_api_key(args.api_key.clone(), provider, false, args.no_login)
        }) {
            Ok(resolved) => resolved,
            Err(msg) => {
                let labeled = with_route_error_label(msg);
                eprintln!("error: {labeled}");
                return Err(1);
            }
        };
    let api_key = route.key.clone();
    if let Some(msg) = shipped_profile_base_conflict(&provider, &base_url) {
        eprintln!("error: {msg}");
        return Err(1);
    }
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
        if caller_tools.is_none()
            && vision_catalog_flag(&args).is_none()
            && args.advertised_context.is_none()
            && let Some((profile, _skip_expensive, advertised)) = cache
                .find_profile_unspecified_catalog_suite(model, &provider, suite)
                .map(|(p, c, a)| (p.clone(), c, a))
        {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                probe_meta(true, suite, advertised, caller_tools.as_deref()),
                fail_on,
            );
        }
        if let Some((profile, hit_suite, advertised)) = cached_probe(
            &cache,
            model,
            &provider,
            suite,
            vision,
            args.advertised_context,
            caller_tools.as_deref(),
        ) {
            return emit_profile(
                &profile,
                args.json,
                args.verbose,
                probe_meta(true, hit_suite, advertised, caller_tools.as_deref()),
                fail_on,
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
        && let Some((profile, hit_suite, advertised)) = cached_probe(
            &cache,
            &model,
            &provider,
            suite,
            vision,
            advertised,
            caller_tools.as_deref(),
        )
    {
        return emit_profile(
            &profile,
            args.json,
            args.verbose,
            probe_meta(true, hit_suite, advertised, caller_tools.as_deref()),
            fail_on,
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
    if let Some(list) = caller_tools {
        runner = runner.with_tools(list);
    }

    if !args.json {
        println!("Probing {model} ({provider})...");
        println!();
    }
    for name in planned_probe_names(suite, vision) {
        eprintln!("{name}");
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

    emit_run(&run, args.json, args.verbose, fail_on)
}

fn run_export(args: ExportArgs) -> Result<(), u8> {
    if !args.all && !args.aider && !args.cline {
        eprintln!("error: specify --aider or --cline, or pass --all");
        return Err(1);
    }
    let model = args.model.trim();
    let provider = args.provider.trim();
    if model.is_empty() {
        eprintln!("error: --model is empty");
        return Err(1);
    }
    if provider.is_empty() {
        eprintln!("error: --provider is empty");
        return Err(1);
    }
    let cache_path = resolve_user_path(args.cache.clone(), default_cache_path());
    let cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
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
    let dir = resolve_user_path(args.dir.clone(), PathBuf::from("."));
    reject_export_dir(&dir)?;
    let mut stdout_body = None;
    if args.all || args.aider {
        let overlay = HostOverlay::aider(&profile, advertised);
        if !args.all {
            stdout_body = overlay.files().into_iter().next().map(|file| file.body);
        }
        write_one_overlay(&overlay, &dir)?;
    }
    if args.all || args.cline {
        let overlay = HostOverlay::cline(&profile, advertised);
        if !args.all && stdout_body.is_none() {
            stdout_body = overlay.files().into_iter().next().map(|file| file.body);
        }
        write_one_overlay(&overlay, &dir)?;
    }
    if let Some(body) = stdout_body {
        print!("{body}");
    }
    Ok(())
}

fn reject_export_dir(dir: &std::path::Path) -> Result<(), u8> {
    let mut path = dir;
    loop {
        if path.exists() && !path.is_dir() {
            eprintln!(
                "error: --dir must be a directory (got a file: {})",
                path.display()
            );
            return Err(1);
        }
        if path.exists() {
            return Ok(());
        }
        match path.parent() {
            Some(parent) if parent != path => path = parent,
            _ => return Ok(()),
        }
    }
}

fn write_one_overlay(overlay: &HostOverlay, dir: &std::path::Path) -> Result<(), u8> {
    match overlay.write_to(dir) {
        Ok(paths) => {
            for path in paths {
                eprintln!("wrote {}", path.display());
            }
            Ok(())
        }
        Err(err) => {
            eprintln!("error: failed to write overlays: {err}");
            Err(1)
        }
    }
}

fn run_cache(args: CacheArgs) -> Result<(), u8> {
    match args.command {
        CacheCommand::Path(args) => {
            let path = resolve_user_path(args.cache, default_cache_path());
            println!("{}", path.display());
            Ok(())
        }
        CacheCommand::List(args) => run_cache_list(args),
    }
}

fn run_cache_list(args: CacheListArgs) -> Result<(), u8> {
    let path = resolve_user_path(args.cache, default_cache_path());
    if !path.exists() {
        eprintln!("no cache file: {}", path.display());
        return Ok(());
    }
    let cache = ProbeCache::load(&path).map_err(|err| {
        eprintln!("error: failed to load cache {}: {err}", path.display());
        1u8
    })?;
    for row in cache.list_rows(args.provider.as_deref()) {
        println!("{}", format_cache_list_line(&row));
    }
    Ok(())
}

fn format_cache_list_line(row: &CacheListRow) -> String {
    let mut line = format!(
        "{}\t{}\t{}\t{}",
        row.model_id,
        row.provider,
        row.suite.as_str(),
        row.probed_at
    );
    if row.stale {
        line.push_str("\tstale");
    }
    line
}

fn run_matrix(args: MatrixArgs) -> Result<(), u8> {
    let provider = args
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let cache_path = resolve_user_path(args.cache.clone(), default_cache_path());
    let cache = ProbeCache::load(&cache_path).map_err(|e| {
        eprintln!("error: failed to load cache {}: {e}", cache_path.display());
        1u8
    })?;
    let matrix = PlumbingMatrix::from_cache_filter(&cache, provider);
    if matrix.rows.is_empty() {
        let stale = match provider {
            Some(p) => cache.stale_suite_version_for_provider(p),
            None => cache.stale_suite_version_any(),
        };
        if let Some(stale) = stale {
            if let Some(p) = provider {
                eprintln!(
                    "error: cached {p} probes are suite {stale} (need {}); run `canact probe` again",
                    canact::PROBE_SUITE_VERSION
                );
            } else {
                eprintln!(
                    "error: cached probes are suite {stale} (need {}); run `canact probe` again",
                    canact::PROBE_SUITE_VERSION
                );
            }
        } else if let Some(p) = provider {
            eprintln!("error: no cached probes for {p} (run `canact probe` first)");
        } else {
            eprintln!("error: no cached probes (run `canact probe` first)");
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

fn emit_run(run: &ProbeRun, json: bool, verbose: bool, fail_on: Option<FailOn>) -> Result<(), u8> {
    emit_envelope(
        &run.profile,
        json,
        verbose,
        run.host_policy_envelope(),
        run.advertised_context_tokens,
        fail_on,
    )
}

fn run_dry_run(args: &ProbeArgs) -> Result<(), u8> {
    let suite = match resolve_suite(args) {
        Ok(suite) => suite,
        Err(msg) => {
            eprintln!("error: {msg}");
            return Err(1);
        }
    };
    let provider_hint = args
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("");
    if let Some(msg) = invalid_explicit_base_url(args.base_url.as_deref()) {
        eprintln!("error: {msg}");
        return Err(1);
    }
    let (provider, base_url) = probe_endpoint_without_key(provider_hint, args.base_url.as_deref());
    if let Some(msg) = invalid_explicit_base_url(Some(&base_url)) {
        eprintln!("error: {msg}");
        return Err(1);
    }
    if let Some(msg) = shipped_profile_base_conflict(&provider, &base_url) {
        eprintln!("error: {msg}");
        return Err(1);
    }
    let base_url = redact_base_url(&base_url);
    let probes = planned_probe_names(suite, args.vision);
    if args.json {
        let plan = serde_json::json!({
            "provider": provider,
            "baseUrl": base_url,
            "suite": suite.as_str(),
            "probes": probes,
        });
        match serde_json::to_string_pretty(&plan) {
            Ok(text) => {
                println!("{text}");
                Ok(())
            }
            Err(err) => {
                eprintln!("error: failed to serialize probe JSON: {err}");
                Err(1)
            }
        }
    } else {
        let ladder = if suite.skip_expensive() {
            "4096"
        } else {
            "4096, 8192, 16384"
        };
        println!("provider: {provider}");
        println!("baseUrl: {base_url}");
        println!("suite: {}", suite.as_str());
        println!("context ladder: {ladder}");
        println!("xml_tool_calling runs unless native tool_calling is Strong");
        println!("probe_max_output_tokens is measured and is not listed below");
        for name in probes {
            println!("{name}");
        }
        Ok(())
    }
}

fn emit_profile(
    profile: &CapabilityProfile,
    json: bool,
    verbose: bool,
    meta: HostPolicyMeta,
    fail_on: Option<FailOn>,
) -> Result<(), u8> {
    eprintln!("cache hit");
    emit_envelope(
        profile,
        json,
        verbose,
        profile.host_policy_envelope_with(meta),
        meta.advertised_context_tokens,
        fail_on,
    )
}

fn emit_envelope(
    profile: &CapabilityProfile,
    json: bool,
    verbose: bool,
    envelope: serde_json::Value,
    advertised: Option<u32>,
    fail_on: Option<FailOn>,
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
        return Err(2);
    }
    if let Some(bar) = fail_on {
        let hits = profile.fail_on_hits(bar);
        if !hits.is_empty() {
            eprintln!("{}", fail_on_stderr(bar, &hits));
            return Err(2);
        }
    }
    Ok(())
}

fn parse_fail_on(raw: Option<&str>) -> Result<Option<FailOn>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    FailOn::parse(raw)
        .map(Some)
        .ok_or_else(|| format!("error: unknown --fail-on={raw} (expected weak or degraded)"))
}

fn fail_on_stderr(bar: FailOn, hits: &[(&str, CapabilityLevel)]) -> String {
    let parts = hits
        .iter()
        .map(|(name, level)| format!("{name} is {}", level_word(*level)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("error: --fail-on {}: {parts}", bar.as_str())
}

fn level_word(level: CapabilityLevel) -> &'static str {
    match level {
        CapabilityLevel::Weak => "weak",
        CapabilityLevel::Medium => "medium",
        CapabilityLevel::Strong => "strong",
    }
}

fn resolve_api_key(
    cli: Option<String>,
    provider: &str,
    explicit_base_url: bool,
    no_login: bool,
) -> Result<canact::KeyRoute, String> {
    let cli = present_secret(cli);
    if is_groq_provider_label(provider) {
        let groq = present_secret(std::env::var("GROQ_API_KEY").ok());
        return Ok(resolve_api_key_from(
            cli.or(groq),
            None,
            None,
            None,
            None,
            provider,
        ));
    }
    if is_bedrock_provider_label(provider) {
        let bedrock = present_secret(std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok());
        return Ok(resolve_api_key_from(
            cli.or(bedrock),
            None,
            None,
            None,
            None,
            provider,
        ));
    }
    let openai = present_secret(std::env::var("OPENAI_API_KEY").ok());
    let openrouter = present_secret(std::env::var("OPENROUTER_API_KEY").ok());
    let xai_env = present_secret(std::env::var("XAI_API_KEY").ok())
        .or_else(|| present_secret(std::env::var("GROK_API_KEY").ok()));
    let other_before_xai_oauth =
        openai.is_some() || openrouter.is_some() || xai_env.is_some() || cli.is_some();
    let xai = match xai_env {
        Some(key) => Some(key),
        None if !no_login
            && should_load_xai_oauth(provider, other_before_xai_oauth, explicit_base_url) =>
        {
            xai_oauth_access_token()?
        }
        None => None,
    };
    let other_cloud_keys =
        openai.is_some() || openrouter.is_some() || xai.is_some() || cli.is_some();
    let anthropic = match present_secret(std::env::var("ANTHROPIC_AUTH_TOKEN").ok())
        .or_else(|| present_secret(std::env::var("ANTHROPIC_API_KEY").ok()))
    {
        Some(key) => Some(key),
        None if !no_login
            && should_load_claude_code_login(provider, other_cloud_keys, explicit_base_url) =>
        {
            claude_code_access_token()?
        }
        None => None,
    };
    Ok(resolve_api_key_from(
        cli, openai, openrouter, xai, anthropic, provider,
    ))
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
    // Clap already prefixes "invalid value '…' for '--advertised-context <N>'".
    let n: u32 = trimmed
        .parse()
        .map_err(|_| "invalid digit found in string".to_owned())?;
    if n < 1 {
        return Err(format!("{n} must be at least 1"));
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
    tools: Option<&[ProbeTool]>,
) -> Option<(CapabilityProfile, SuiteTier, Option<u32>)> {
    if let Some(profile) =
        cache.get_with_suite_tools(model, provider, suite, vision, advertised, tools)
    {
        return Some((profile.clone(), suite, advertised));
    }
    if tools.is_some() || !matches!(suite, SuiteTier::Policy) || vision {
        return None;
    }
    cache
        .find_profile_with_cost_and_advertised(model, provider, advertised)
        .map(|(profile, hit_suite)| (profile.clone(), hit_suite, advertised))
}

#[derive(serde::Deserialize)]
struct CallerToolFile {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

fn parent_file_blocking(path: &std::path::Path) -> Option<&std::path::Path> {
    let mut cursor = path.parent()?;
    loop {
        if cursor.as_os_str().is_empty() {
            return None;
        }
        if cursor.exists() && !cursor.is_dir() {
            return Some(cursor);
        }
        if cursor.exists() {
            return None;
        }
        match cursor.parent() {
            Some(parent) if parent != cursor => cursor = parent,
            _ => return None,
        }
    }
}

fn load_caller_tools(path: Option<&std::path::Path>) -> Result<Option<Vec<ProbeTool>>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_dir() {
        return Err(format!("tools file {} must be a file", path.display()));
    }
    if let Some(blocker) = parent_file_blocking(path) {
        return Err(format!(
            "tools file {} parent must be a directory (got a file: {})",
            path.display(),
            blocker.display()
        ));
    }
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read tools file {}: {err}", path.display()))?;
    // A Vec sees `{` and stops, so invalid JSON that starts with `{`
    // would be reported as the wrong type.
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse tools file {}: {err}", path.display()))?;
    let Some(items) = value.as_array() else {
        return Err(format!(
            "failed to parse tools file {}: expected a JSON array of tools",
            path.display()
        ));
    };
    for (index, item) in items.iter().enumerate() {
        if !item.is_object() {
            return Err(format!(
                "failed to parse tools file {}: item {index} must be an object with name, description, and parameters",
                path.display()
            ));
        }
    }
    let rows: Vec<CallerToolFile> = serde_json::from_value(value)
        .map_err(|err| format!("failed to parse tools file {}: {err}", path.display()))?;
    Ok(Some(
        rows.into_iter()
            .map(|row| ProbeTool {
                name: row.name,
                description: row.description,
                parameters: row.parameters,
            })
            .collect(),
    ))
}

fn probe_meta(
    from_cache: bool,
    suite: SuiteTier,
    advertised: Option<u32>,
    tools: Option<&[ProbeTool]>,
) -> HostPolicyMeta {
    let meta = HostPolicyMeta::for_suite(true, from_cache, suite, advertised);
    match tools {
        Some(list) => meta.with_tool_digest(probe_tools_digest(list)),
        None => meta,
    }
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

    #[test]
    fn broken_tools_object_reports_json_syntax() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tools.json");
        std::fs::write(&path, "{not-json\n").expect("write");
        let err = super::load_caller_tools(Some(&path)).expect_err("broken object");
        assert!(err.contains("key must be a string"), "{err}");
        assert!(!err.contains("expected a sequence"), "{err}");
    }

    #[test]
    fn tools_json_object_asks_for_an_array() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tools.json");
        std::fs::write(&path, "{}\n").expect("write");
        let err = super::load_caller_tools(Some(&path)).expect_err("object");
        assert!(err.contains("expected a JSON array of tools"), "{err}");
    }

    #[test]
    fn tools_array_string_asks_for_an_object() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tools.json");
        std::fs::write(&path, "[\"read_file\"]\n").expect("write");
        let err = super::load_caller_tools(Some(&path)).expect_err("string item");
        assert!(
            err.contains("item 0 must be an object with name, description, and parameters"),
            "{err}"
        );
        assert!(!err.contains("CallerToolFile"), "{err}");
    }

    #[test]
    fn tools_file_object_missing_description() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tools.json");
        std::fs::write(&path, "[{\"name\":\"lookup_issue\"}]").expect("write");
        let err = super::load_caller_tools(Some(&path)).expect_err("missing description");
        assert!(err.contains("description"), "{err}");
    }

    #[test]
    fn empty_tools_array_loads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tools.json");
        std::fs::write(&path, "[]\n").expect("write");
        let tools = super::load_caller_tools(Some(&path))
            .expect("empty array")
            .expect("some");
        assert!(tools.is_empty());
    }

    #[test]
    fn tools_directory_must_be_a_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = super::load_caller_tools(Some(dir.path())).expect_err("directory");
        assert!(err.contains("must be a file"), "{err}");
        assert!(!err.contains("os error"), "{err}");
    }

    #[test]
    fn tools_parent_file_names_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let as_file = dir.path().join("notadir");
        std::fs::write(&as_file, b"nope").expect("file");
        let nested = as_file.join("tools.json");
        let err = super::load_caller_tools(Some(&nested)).expect_err("parent file");
        assert!(err.contains("got a file"), "{err}");
        assert!(err.contains("notadir"), "{err}");
        assert!(!err.contains("os error"), "{err}");
    }
    use canact::{looks_cheap, resolve_api_key_from, should_load_xai_oauth};
    use std::path::PathBuf;

    #[test]
    fn advertised_context_parser_omits_the_flag_prefix() {
        let err = super::parse_advertised_context("0").expect_err("zero");
        assert_eq!(err, "0 must be at least 1");
        let err = super::parse_advertised_context("-1").expect_err("negative");
        assert_eq!(err, "invalid digit found in string");
        assert!(!err.contains("--advertised-context"), "{err}");
        let err = super::parse_advertised_context("abc").expect_err("text");
        assert_eq!(err, "invalid digit found in string");
        assert_eq!(
            super::parse_advertised_context(" 4096 ").expect("pad"),
            4096
        );
    }

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
    fn unknown_fail_on_is_exit_1_message() {
        let err = super::parse_fail_on(Some("foo")).expect_err("unknown");
        assert_eq!(
            err,
            "error: unknown --fail-on=foo (expected weak or degraded)"
        );
        assert_eq!(super::parse_fail_on(None).expect("omitted"), None);
        assert_eq!(
            super::parse_fail_on(Some(" degraded ")).expect("padded"),
            Some(canact::FailOn::Degraded)
        );
        assert_eq!(
            super::fail_on_stderr(
                canact::FailOn::Degraded,
                &[
                    ("json_output", canact::CapabilityLevel::Medium),
                    ("instruction_following", canact::CapabilityLevel::Weak),
                ],
            ),
            "error: --fail-on degraded: json_output is medium, instruction_following is weak"
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
