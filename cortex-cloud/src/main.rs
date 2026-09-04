//! cortex-server — OpenAI-compatible HTTP inference server.
//!
//! Loads a GGUF model and serves the OpenAI `/v1/chat/completions` wire format
//! so that any OpenAI-protocol client (AgentOS, curl, etc.) can use cortex
//! as a drop-in inference backend.
//!
//! ```text
//! POST /v1/chat/completions   — chat completion (text + tool calls)
//! GET  /v1/models             — list loaded model
//! GET  /health                — readiness probe
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use clap::Parser;
use tokio::sync::Mutex;
use tracing::info;


mod metrics;
mod api;
mod cache;
mod chat;
mod shims;
mod state;

use self::api::*;
use self::cache::*;
use self::chat::*;
use self::shims::*;
use self::state::*;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "cortex-server", about = "OpenAI-compatible cortex inference server")]
struct Cli {
    /// Path to the GGUF model file.
    #[arg(long, short)]
    model: String,

    /// Port to listen on.
    #[arg(long, short, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,

    /// Maximum sequence length for KV cache.
    #[arg(long, default_value = "4096")]
    max_seq_len: usize,

    /// Maximum number of resident cache shards. A cache/load beyond this
    /// returns 507 unless it replaces an existing id. Bounds the VRAM the
    /// cache pool can consume (each shard reserves a full max-seq-len cache).
    #[arg(long, default_value = "32")]
    max_cache_shards: usize,

    /// Enable cache endpoints (/v1/cache/*) and cache_shards support on
    /// /v1/chat/completions. Use this for the librarian deployment.
    /// When disabled (default), the server is a stateless generation
    /// endpoint only — appropriate for the 32B Bob deployment.
    #[arg(long)]
    enable_cache: bool,

    /// Enable retrieval mode (mode: "retrieve" on /v1/chat/completions).
    /// Implies --enable-cache since retrieval operates over cached shards.
    #[arg(long)]
    enable_retrieve: bool,

    /// Enable PolarQuant-compressed KV cache for retrieval. When set,
    /// every shard built via /v1/cache/load is compressed to polar
    /// format on the GPU after the f32 prefill. Single-shard
    /// /v1/retrieve queries then run against the polar cache via
    /// `forward_full_gpu_polar_traced`. Multi-shard composition and
    /// chat-mode generation continue to use the f32 cache for now.
    /// Trades polar quantization noise (per-row cosine ~0.95+ at
    /// realistic head_dim) for ~7x KV VRAM reduction.
    #[arg(long)]
    enable_polar_cache: bool,

    /// Number of QJL (Quantized Johnson-Lindenstrauss) projections for
    /// K-residual correction when a shard is loaded with `qjl: true`.
    /// 32 is the standard tradeoff (matches CPU default); brings polar
    /// attention output cosine from ~0.84 to ~0.95 vs f32 at ~32 KB
    /// extra per layer at Qwen 3B / max_seq=4096. Max 32 in current
    /// shader (single u32 sign word per entry).
    #[arg(long, default_value_t = 32)]
    qjl_projections: usize,

    /// Seed base for per-layer QJL projection matrices. Layer i uses
    /// `qjl_seed + i`. Kept separate from `--polar-rotation-seed`
    /// because rotation matrices are square orthonormal and QJL
    /// projections are rectangular Gaussian-normalized.
    #[arg(long, default_value_t = 0xDEADBEEFCAFEF00Du64)]
    qjl_seed: u64,

    /// Enable shim registry endpoints (PUT/GET/DELETE /v1/shims/{id},
    /// GET /v1/shims/, POST /v1/shims/infer). Shims are small ONNX
    /// modules used as classifiers / gates / steers per the v1 shim API
    /// (project_cortex_v1_shim_api.md). When disabled, the routes are
    /// not mounted (404).
    #[arg(long)]
    enable_shims: bool,
}























// ---------------------------------------------------------------------------
// OpenAI wire types
// ---------------------------------------------------------------------------
















// ---------------------------------------------------------------------------
// Cache endpoint wire types
// ---------------------------------------------------------------------------





// ---------------------------------------------------------------------------
// Chat template — converts messages[] to a token sequence
// ---------------------------------------------------------------------------




// ---------------------------------------------------------------------------
// Retrieval response types
// ---------------------------------------------------------------------------






// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------





// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------







// Suppress unused-import warning for Stream when only used via concrete types.
#[allow(dead_code)]
fn _stream_marker(_: impl Stream) {}




// ---------------------------------------------------------------------------
// Tokenize endpoint
// ---------------------------------------------------------------------------



/// POST /v1/tokenize — convert text to token IDs using the loaded model's tokenizer.
///
/// This ensures memex and cortex use the same tokenizer, so token IDs from
/// memex's ingestion pipeline match what cortex's cache endpoints expect.
async fn tokenize(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<TokenizeRequest>,
) -> Json<TokenizeResponse> {
    let add_bos = req.add_bos.unwrap_or(state.tokenizer.add_bos_default());
    let tokens = state.tokenizer.encode(&req.text, add_bos);
    let count = tokens.len();
    Json(TokenizeResponse { tokens, count })
}



/// POST /v1/detokenize — convert token IDs back to text.
///
/// Useful for resolving retrieval hits — given a hit at (offset, length),
/// the caller can pull tokens[offset-k .. offset+length+k] from the original
/// shard token list and POST them here to get a human-readable context window.
async fn detokenize(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    // Review #11: an out-of-range id indexed the tokenizer tables directly.
    crate::chat::check_token_ids(&req.tokens, state.engine.vocab_size())?;
    let text = state.tokenizer.decode(&req.tokens);
    Ok(Json(DetokenizeResponse { text }))
}

// ---------------------------------------------------------------------------
// Other handlers
// ---------------------------------------------------------------------------

async fn list_models(
    State(state): State<Arc<ServerState>>,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        data: vec![ModelEntry {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "cortex".to_string(),
        }],
    })
}

/// Prometheus text-format metrics. Served at `GET /metrics`.
/// Pulls counters/histograms from `state.metrics`. New metrics belong
/// in `metrics.rs` — see the module doc for the no-phantom rule.
async fn metrics_endpoint(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let body = state.metrics.render_prometheus();
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        .body(body.into())
        .expect("static response builder cannot fail")
}

async fn health(
    State(state): State<Arc<ServerState>>,
) -> Json<HealthResponse> {
    // Use try_lock so /health stays responsive even when cache_append /
    // cache_load are holding the pool mutex during a long forward pass.
    // Reporting "(busy)" is much better than letting health probes hang
    // — orchestrators (memex, AgentOS) need timely liveness signals.
    let (pool_size, total_tokens) = match state.cache_pool.try_lock() {
        Ok(pool) => (
            pool.len(),
            pool.values().map(|e| {
                // Polar-only shards report seq_len from polar; f32 shards from cache.
                e.cache.as_ref().map(|c| c.seq_len())
                    .or_else(|| e.polar.as_ref().map(|p| p.seq_len()))
                    .unwrap_or(0)
            }).sum(),
        ),
        Err(_) => (0, 0), // pool busy; report zeros rather than hang
    };
    Json(HealthResponse {
        status: "ready".to_string(),
        model: state.model_name.clone(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        memory: HealthMemory {
            cache_pool_size: pool_size,
            cache_pool_total_tokens: total_tokens,
            max_seq_len: state.max_seq_len,
        },
    })
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!(model = %cli.model, "loading model");
    let loaded = cortex::load_model(&cli.model)?;

    let model_name = loaded
        .config
        .model_name
        .clone()
        .unwrap_or_else(|| "cortex-model".to_string());

    // --enable-retrieve implies --enable-cache
    let cache_enabled = cli.enable_cache || cli.enable_retrieve;
    let retrieve_enabled = cli.enable_retrieve;

    // Wrap the loaded model in a GpuEngine. Reuses the GpuDevice the loader
    // built (the layers' resident weights are tied to it — building a second
    // device produces cross-device buffer-binding errors, see #16).
    let gpu = loaded.gpu.clone()
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cortex-server requires a discrete GPU; none detected",
        ))?;
    let engine = cortex::layers::gpu_engine::GpuEngine::with_max_seq(
        loaded.model, gpu, cli.max_seq_len,
    );

    // Polar caching only meaningful when retrieval is enabled. Refuse the
    // confusing combination at startup rather than silently ignoring.
    if cli.enable_polar_cache && !retrieve_enabled {
        return Err("--enable-polar-cache requires --enable-retrieve".into());
    }
    let polar_cache_enabled = cli.enable_polar_cache;
    // Fixed deterministic seed base for this server's polar caches. All
    // shards loaded by this process share it (so future multi-shard polar
    // composition can use a single rotation scheme). Different runs are
    // free to pick different seeds; persistence across restarts is not
    // a property anything in cortex-cloud relies on yet.
    let polar_rotation_seed: u64 = 0x9E37_79B9_7F4A_7C15;

    let state = Arc::new(ServerState {
        engine,
        tokenizer: loaded.tokenizer,
        config: loaded.config,
        cache_pool: Mutex::new(HashMap::new()),
        composition: Mutex::new(None),
        model_name: model_name.clone(),
        start_time: Instant::now(),
        max_seq_len: cli.max_seq_len,
        max_cache_shards: cli.max_cache_shards,
        cache_enabled,
        retrieve_enabled,
        polar_cache_enabled,
        polar_rotation_seed,
        qjl_projections: cli.qjl_projections,
        qjl_seed: cli.qjl_seed,
        shims: Mutex::new(HashMap::new()),
        shims_enabled: cli.enable_shims,
        metrics: Arc::new(metrics::Metrics::new(
            model_name.clone(),
            env!("CARGO_PKG_VERSION").to_string(),
        )),
    });

    // Phase K: periodic metrics sampler. Snapshots the read-only
    // gauges (cache pool depth/tokens, vram-heap usage across all 5
    // heaps, ParamsBufferPool cumulative acquire count) into the
    // Metrics struct so `/metrics` exposes them. Push-style metrics
    // (concurrent_requests, request histograms, token counters) go
    // through RequestTimer / handler code paths and don't need the
    // sampler.
    //
    // Interval tunable via CORTEX_METRICS_SAMPLE_INTERVAL_MS
    // (default 1000 ms). The locked region under cache_pool is
    // tiny: just `len()` and a sum of `tokens.len()` per entry,
    // then release. vram-heap stats() each hold the heap's
    // internal lock for microseconds; safe at 1 Hz.
    let sampler_state = state.clone();
    let sampler_interval_ms: u64 = std::env::var("CORTEX_METRICS_SAMPLE_INTERVAL_MS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(
            std::time::Duration::from_millis(sampler_interval_ms),
        );
        // Skip the immediate first tick so we don't read empty
        // state before the engine is warm.
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;

            // Cache pool: lock briefly, copy out size + token-sum,
            // release.
            let (pool_size, pool_tokens) = {
                let pool = sampler_state.cache_pool.lock().await;
                let size = pool.len() as u64;
                let tokens: u64 = pool.values()
                    .map(|e| e.tokens.len() as u64).sum();
                (size, tokens)
            };
            sampler_state.metrics.record_cache_pool(pool_size, pool_tokens);

            // vram-heap usage across all 5 heaps.
            let gpu = sampler_state.engine.gpu();
            for (id, used_bytes) in gpu.vram_heap_usage() {
                let label = match id {
                    cortex::compute::wgpu_backend::VramHeapId::TransientA =>
                        metrics::VramHeapLabel::TransientA,
                    cortex::compute::wgpu_backend::VramHeapId::TransientB =>
                        metrics::VramHeapLabel::TransientB,
                    cortex::compute::wgpu_backend::VramHeapId::TransientC =>
                        metrics::VramHeapLabel::TransientC,
                    cortex::compute::wgpu_backend::VramHeapId::Weights =>
                        metrics::VramHeapLabel::Weights,
                    cortex::compute::wgpu_backend::VramHeapId::HostReadback =>
                        metrics::VramHeapLabel::HostReadback,
                };
                sampler_state.metrics.record_vram_heap(label, used_bytes);
            }

            // ParamsBufferPool cumulative acquire count.
            let p = gpu.params_pool.stats();
            sampler_state.metrics.record_params_pool(p.total_acquired as u64);

            // Device VRAM budget (Phase M): total + committed across all
            // live DeviceLocal heaps (globals + per-cache).
            let (budget_total, budget_committed) = gpu.vram_budget_snapshot();
            sampler_state.metrics.record_vram_budget(budget_total, budget_committed);
        }
    });

    // Build router: always include completions, models, health, metrics.
    // Cache and retrieve endpoints are conditional on startup flags.
    let mut app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint));

    if cache_enabled {
        app = app
            .route("/v1/cache/load", post(cache_load))
            .route("/v1/cache/append", post(cache_append))
            .route("/v1/cache/{id}", get(cache_get).delete(cache_delete));
    }

    if cli.enable_shims {
        app = app
            .route("/v1/shims/", get(shims_list))
            .route("/v1/shims/infer", post(shim_infer))
            .route("/v1/shims/embed", post(shim_embed))
            .route("/v1/shims/{id}", get(shim_get).put(shim_put).delete(shim_delete));
    }

    let app = app.with_state(state);

    let addr = format!("{}:{}", cli.bind, cli.port);
    info!(
        addr = %addr,
        model = %model_name,
        cache = cache_enabled,
        retrieve = retrieve_enabled,
        polar = polar_cache_enabled,
        shims = cli.enable_shims,
        "cortex-server ready",
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// ort smoke
// ---------------------------------------------------------------------------

/// The `ort` crate is wired in for the v1 shim runtime
/// (see `project_cortex_v1_shim_api.md`). #5 lands the registry +
/// `/v1/shims/infer`; this file is the link surface.
#[cfg(test)]
mod ort_smoke {
    /// Construct an ort `SessionBuilder` to confirm the crate is linked
    /// and the ONNX runtime native library loads. We don't load a model
    /// — that's #5. If this test fails to even start (DLL missing),
    /// that's a build-environment problem, not a code problem.
    #[test]
    fn ort_session_builder_constructs() {
        // Result<SessionBuilder, ort::Error>; either branch is fine for
        // a link smoke. The point is symbols resolve and the ORT
        // initializer runs.
        let _ = ort::session::Session::builder();
    }
}
