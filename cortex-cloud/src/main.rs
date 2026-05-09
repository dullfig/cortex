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

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use cortex::layers::gpu_engine::GpuEngine;
use cortex::layers::gpu_kv_cache::GpuKvCache;
use cortex::layers::sampler::{Sampler, SamplerConfig};
use cortex::{ForwardTrace, ModelConfig, Tokenizer};

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

    /// Enable shim registry endpoints (PUT/GET/DELETE /v1/shims/{id},
    /// GET /v1/shims/, POST /v1/shims/infer). Shims are small ONNX
    /// modules used as classifiers / gates / steers per the v1 shim API
    /// (project_cortex_v1_shim_api.md). When disabled, the routes are
    /// not mounted (404).
    #[arg(long)]
    enable_shims: bool,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Number of dummy "sink" tokens prepended to every shard at load time.
/// These absorb the position-0 attention sink artifact (see POSITION-
/// addendum.md section 15 on the structural cause) so real content tokens
/// aren't contaminated. Retrieval scoring skips the first SINK_TOKENS
/// positions per shard.
const SINK_TOKENS: usize = 4;

/// Per-cache metadata stored alongside the KV cache in the pool.
struct CacheEntry {
    cache: GpuKvCache,
    /// Optional PolarQuant-compressed K/V parallel to `cache`. Populated
    /// once at cache_load time (via `populate_from_f32_cache_gpu`) when
    /// `--enable-polar-cache` is set. Single-shard /v1/retrieve queries
    /// against this entry use the polar trace forward; chat and
    /// multi-shard paths continue to use `cache`. Memory cost: parallel
    /// (both caches resident); a future refinement can drop the f32
    /// cache once the chat path supports polar too.
    polar: Option<cortex::layers::gpu_polar_kv_cache::GpuPolarKvCache>,
    /// Token history that built this cache. Stored so shards can be composed
    /// by replaying tokens in sequence (which gives correct RoPE positions).
    tokens: Vec<u32>,
    /// Bumps any time the shard's K/V content changes (load replaces, append
    /// extends). Used as the staleness witness for the multi-shard
    /// retrieval `composition` cache below.
    version: u64,
    #[allow(dead_code)]
    created_at: Instant,
    last_used: Instant,
}

/// One composed-cache slot, reused across multi-shard retrieve requests so
/// each query doesn't re-allocate and re-prefill ~85 MiB of K/V buffers.
/// Populated lazily on first multi-shard retrieve and reused while the
/// cached `(shard_name, version)` key keeps matching incoming requests.
struct ComposedEntry {
    /// Ordered list of `(shard_name, version_at_compose_time)`. Matches
    /// the request key exactly: same shards, same order, same versions.
    /// Order matters because RoPE positions depend on token order.
    key: Vec<(String, u64)>,
    /// The composed cache itself.
    cache: GpuKvCache,
}

/// Shim manifest as defined in `project_cortex_v1_shim_api.md`. Wire-
/// compatible with what AgentOS's shim-management control plane pushes
/// via PUT /v1/shims/{id}.
///
/// `input_shape` and `output_shape` are kept as `serde_json::Value` so
/// the schema can grow without breaking older clients — the v1 shapes
/// (`{"hidden_dim": N}` for input, `{"kind": "scalar"|"category:N"|"hidden_delta"}`
/// for output) are recognized at infer time, not at registration.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ShimManifest {
    id: String,
    version: String,
    /// "injection" | "gate" | "steer"
    phase: String,
    attachment: ShimAttachment,
    input_shape: serde_json::Value,
    output_shape: serde_json::Value,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ShimAttachment {
    /// "final" | "entrance:N" | "entrance:all"
    layer: String,
    /// "last_token" | "mean" | "attention" | "none"
    pooling: String,
}

/// One registered shim. Holds the manifest plus the loaded ort Session.
/// Wrapped in `Arc` so handlers can take a clone (cheap) without
/// holding the registry lock through inference.
struct RegisteredShim {
    manifest: ShimManifest,
    session: Mutex<ort::session::Session>,
}

struct ServerState {
    /// GPU-resident inference engine. Owns the underlying TransformerModel
    /// and the GPU device. CPU-side calls go through `engine.cpu()`; the
    /// GPU-native retrieve path goes through `engine.forward_full_gpu_traced()`.
    engine: cortex::layers::gpu_engine::GpuEngine,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: ModelConfig,
    /// Pool of named KV caches. Only used when cache_enabled is true
    /// (librarian deployment). When false (32B Bob deployment), the pool
    /// is empty and cache_shards on requests are ignored.
    cache_pool: Mutex<HashMap<String, CacheEntry>>,
    /// Single-slot composition cache for multi-shard retrieve. Holds at most
    /// one composed `GpuKvCache`; reused when the next request's
    /// `(shard, version)` key matches; rebuilt in place (clear + re-prefill,
    /// no buffer alloc) when it differs. Critical for stability: rapid
    /// per-request alloc-and-drop of ~85 MiB buffer arrays hangs the wgpu
    /// driver after ~3 requests.
    composition: Mutex<Option<ComposedEntry>>,
    model_name: String,
    start_time: Instant,
    max_seq_len: usize,
    /// Whether cache endpoints and cache_shards are enabled.
    cache_enabled: bool,
    /// Whether retrieval mode is enabled.
    retrieve_enabled: bool,
    /// Whether to build a parallel polar-compressed cache on cache_load
    /// (and use it for single-shard retrieve when present).
    polar_cache_enabled: bool,
    /// Per-layer rotation seed base for any polar caches built by this
    /// server. Stored on ServerState so all polar caches share the same
    /// seeding scheme — required for cross-cache compatibility (e.g.
    /// multi-shard polar composition, future).
    polar_rotation_seed: u64,
    /// Shim registry: hot-resident ONNX shims keyed by id. Empty unless
    /// `shims_enabled` is true. `Arc` so handlers can clone-into-handler
    /// without holding the registry lock through inference.
    shims: Mutex<HashMap<String, Arc<RegisteredShim>>>,
    /// Whether shim endpoints are enabled.
    shims_enabled: bool,
}

// ---------------------------------------------------------------------------
// OpenAI wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    tools: Option<Vec<Tool>>,
    /// Ordered list of cache shard names to compose for this request.
    /// Cortex looks up each shard in the pool, composes them in the given
    /// order (with correct RoPE positions via sequential token replay),
    /// and runs inference over the composed context.
    ///
    /// If any shard is not in the pool, cortex returns 404. Use
    /// POST /v1/cache/load to create shards before referencing them.
    /// If absent (or empty), the request runs stateless with a fresh
    /// temporary cache.
    #[serde(default)]
    cache_shards: Option<Vec<String>>,

    /// Backward-compatible single cache ID. If present and cache_shards
    /// is absent, treated as a one-element shard list. Deprecated in
    /// favor of cache_shards.
    #[serde(default)]
    cache_id: Option<String>,

    /// Inference mode. "generate" (default) produces tokens. "retrieve"
    /// computes attention from query positions over the cached corpus
    /// and returns top-K positions with scores instead of generating.
    #[serde(default)]
    #[allow(dead_code)]
    mode: Option<String>,

    /// For mode: "retrieve" — number of top-scoring positions to return.
    #[serde(default = "default_top_k")]
    #[allow(dead_code)]
    top_k: usize,

    /// OpenAI-compatible streaming flag. When true, the response is an
    /// SSE stream of `chat.completion.chunk` events terminated by a
    /// `data: [DONE]` event. First wire supports stateless mode only
    /// (no cache_shards, mode=generate); cached + streaming is a
    /// follow-up because it requires holding the cache pool lock across
    /// the entire generation, which serializes all other requests.
    #[serde(default)]
    stream: bool,
}

fn default_top_k() -> usize { 10 }

fn default_max_tokens() -> u32 { 2048 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Serialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    uptime_secs: u64,
    memory: HealthMemory,
}

#[derive(Serialize)]
struct HealthMemory {
    cache_pool_size: usize,
    cache_pool_total_tokens: usize,
    max_seq_len: usize,
}

// ---------------------------------------------------------------------------
// Cache endpoint wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CacheLoadRequest {
    cache_id: String,
    /// Token IDs to replay through the model to build the KV cache.
    /// For a brand-new user this is empty []. For a returning user after
    /// eviction, this is the full conversation history from sled.
    #[serde(default)]
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CacheAppendRequest {
    cache_id: String,
    tokens: Vec<u32>,
}

#[derive(Serialize)]
struct CacheInfoResponse {
    cache_id: String,
    seq_len: usize,
    max_seq_len: usize,
}

#[derive(Serialize)]
struct CacheLoadResponse {
    cache_id: String,
    seq_len: usize,
    status: String,
}

// ---------------------------------------------------------------------------
// Chat template — converts messages[] to a token sequence
// ---------------------------------------------------------------------------

/// Apply ChatML-style template (works for Qwen, many HF models).
///
/// ```text
/// <|im_start|>system\n{content}<|im_end|>\n
/// <|im_start|>user\n{content}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
fn apply_chat_template(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');

        if let Some(ref content) = msg.content {
            prompt.push_str(content);
        }

        // For tool result messages, include the tool_call_id context
        if msg.role == "tool" {
            if let Some(ref id) = msg.tool_call_id {
                prompt.push_str(&format!("\n[tool_call_id: {id}]"));
            }
        }

        prompt.push_str("<|im_end|>\n");
    }

    // If tools are provided, inject their definitions into the prompt
    // so the model knows what's available.
    if let Some(tools) = tools {
        if !tools.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str("You have access to the following tools. To call a tool, respond with a JSON object in this exact format:\n");
            prompt.push_str("{\"tool_call\": {\"name\": \"<function_name>\", \"arguments\": {<args>}}}\n\n");
            prompt.push_str("Available tools:\n");
            for tool in tools {
                if let Ok(json) = serde_json::to_string_pretty(&tool.function) {
                    prompt.push_str(&json);
                    prompt.push('\n');
                }
            }
            prompt.push_str("<|im_end|>\n");
        }
    }

    // Start the assistant turn
    prompt.push_str("<|im_start|>assistant\n");

    tokenizer.encode(&prompt, tokenizer.add_bos_default())
}

/// Try to parse tool calls from generated text.
///
/// Looks for `{"tool_call": {"name": "...", "arguments": {...}}}` patterns.
fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    // Try to find a tool_call JSON object in the output
    if let Some(start) = text.find("{\"tool_call\"") {
        if let Some(obj) = extract_json_object(&text[start..]) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&obj) {
                if let Some(tc) = parsed.get("tool_call") {
                    let name = tc.get("name")?.as_str()?.to_string();
                    let arguments = tc.get("arguments")
                        .map(|a| serde_json::to_string(a).unwrap_or_default())
                        .unwrap_or_default();
                    return Some(vec![ToolCall {
                        id: format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                        call_type: "function".to_string(),
                        function: ToolCallFunction { name, arguments },
                    }]);
                }
            }
        }
    }
    None
}

/// Extract a balanced JSON object starting from the first `{`.
fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Retrieval response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct RetrievalResponse {
    hits: Vec<RetrievalHit>,
    metadata: RetrievalMetadata,
}

#[derive(Serialize)]
struct RetrievalHit {
    shard_id: String,
    offset: usize,
    length: u32,
    score: f32,
}

#[derive(Serialize)]
struct RetrievalMetadata {
    retrieval_ms: u64,
    query_tokens: u32,
    corpus_tokens: u32,
    layers_used: Vec<usize>,
}

/// Maps a position in a composed token sequence back to its shard + offset.
struct ShardMap {
    /// Sorted by start position: (shard_name, start, end)
    entries: Vec<(String, usize, usize)>,
}

impl ShardMap {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn add(&mut self, shard_name: String, start: usize, end: usize) {
        self.entries.push((shard_name, start, end));
    }

    /// Resolve an absolute position in the composed sequence to (shard_name, offset_within_shard).
    fn resolve(&self, pos: usize) -> Option<(&str, usize)> {
        for (name, start, end) in &self.entries {
            if pos >= *start && pos < *end {
                return Some((name, pos - start));
            }
        }
        None
    }

    /// Total corpus positions (sum of all shard lengths).
    fn corpus_len(&self) -> usize {
        self.entries.last().map(|(_, _, end)| *end).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate tokens with an existing KV cache. Prefills the prompt tokens
/// into the cache, then samples autoregressively up to max_tokens.
fn generate_with_cache(
    engine: &GpuEngine,
    prompt_tokens: &[u32],
    cache: &mut GpuKvCache,
    sampler_config: SamplerConfig,
    seed: u64,
    eos: u32,
    max_tokens: usize,
) -> Vec<u32> {
    let mut sampler = Sampler::new(sampler_config, seed);

    let prefill_logits = engine.forward_full_gpu_with_cache(prompt_tokens, cache);
    let vocab = engine.vocab_size();
    let last_logits_start = (prompt_tokens.len() - 1) * vocab;
    let last_logits = &prefill_logits[last_logits_start..last_logits_start + vocab];
    let mut next_token = sampler.sample(last_logits);

    let mut out = Vec::new();
    if next_token == eos {
        return out;
    }
    out.push(next_token);

    for _ in 1..max_tokens {
        let logits = engine.forward_full_gpu_with_cache(&[next_token], cache);
        next_token = sampler.sample(&logits);
        if next_token == eos {
            break;
        }
        out.push(next_token);
    }
    out
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let prompt_tokens = apply_chat_template(
        &req.messages,
        req.tools.as_deref(),
        &state.tokenizer,
    );

    // Streaming dispatch. Stateless (no cache_shards / cache_id), no tools,
    // not in retrieve mode — those combinations are explicit follow-ups.
    if req.stream {
        let stream_uses_cache = req.cache_shards.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || req.cache_id.is_some();
        let mode_is_retrieve = req.mode.as_deref() == Some("retrieve");
        if stream_uses_cache || mode_is_retrieve || req.tools.is_some() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": "stream=true currently supports stateless generate only \
                                    (no cache_shards / cache_id, no tools, not retrieve mode). \
                                    Cached + streaming is a planned follow-up.",
                    }
                })),
            ));
        }
        return chat_completions_stream(state, req, prompt_tokens).await;
    }

    let prompt_len = prompt_tokens.len() as u32;

    let sampler_config = if req.temperature <= 0.0 {
        SamplerConfig::greedy()
    } else {
        SamplerConfig {
            temperature: req.temperature,
            top_k: 40,
            top_p: 0.95,
            ..Default::default()
        }
    };

    let eos = state.tokenizer.eos_token_id();
    let seed = rand_seed();
    let max_tokens = req.max_tokens as usize;

    // Resolve the shard list: cache_shards takes priority, then cache_id
    // for backward compat, then empty (stateless).
    let shards: Vec<String> = req
        .cache_shards
        .or_else(|| req.cache_id.map(|id| vec![id]))
        .unwrap_or_default();

    // Gate: if cache_shards provided but cache is not enabled, reject.
    if !shards.is_empty() && !state.cache_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "feature_disabled",
                    "message": "cache_shards requires --enable-cache. This deployment is stateless only.",
                }
            })),
        ));
    }

    // ---------------------------------------------------------------
    // RETRIEVAL MODE: return top-K attention positions instead of
    // generating tokens. Uses forward_traced over composed shard
    // tokens + prompt tokens. Returns early with a RetrievalResponse.
    // ---------------------------------------------------------------
    let is_retrieve = req.mode.as_deref() == Some("retrieve");

    if is_retrieve && !state.retrieve_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "feature_disabled",
                    "message": "mode 'retrieve' requires --enable-retrieve. This deployment is generation only.",
                }
            })),
        ));
    }

    if is_retrieve {
        if shards.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "invalid_request",
                        "message": "mode 'retrieve' requires cache_shards to be set",
                    }
                })),
            ));
        }

        // Phase 1: under the pool lock, verify every requested shard exists,
        // snapshot the bits we need (name, version, tokens, length), and
        // build `shard_map`. After this block we drop the pool lock so the
        // long forward(s) below don't block other handlers.
        let snapshot: Vec<(String, u64, Vec<u32>)>;
        let mut shard_map = ShardMap::new();
        let mut corpus_len = 0usize;
        {
            let pool = state.cache_pool.lock().await;
            for shard_name in &shards {
                if !pool.contains_key(shard_name) {
                    return Err((
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "error": {
                                "type": "cache_not_found",
                                "message": format!("shard '{}' not found", shard_name),
                                "cache_id": shard_name,
                            }
                        })),
                    ));
                }
            }
            snapshot = shards.iter().map(|s| {
                let e = pool.get(s).unwrap();
                (s.clone(), e.version, e.tokens.clone())
            }).collect();
            for (name, _, tokens) in &snapshot {
                let start = corpus_len;
                corpus_len += tokens.len();
                shard_map.add(name.clone(), start, corpus_len);
            }
        }

        let retrieve_start = Instant::now();

        // Capture the last 4 layers' pre-softmax scores (memex architecture:
        // "last few layers carry the retrieval signal").
        let n_layers_total = state.engine.n_layers();
        let n_heads = state.engine.cpu().blocks()[0].attention().n_heads();
        let capture_start = n_layers_total.saturating_sub(4);
        let capture_layers: Vec<usize> = (capture_start..n_layers_total).collect();

        let n_query = prompt_tokens.len();
        let bos = state.tokenizer.bos_token_id();
        let baseline_tokens = vec![bos];
        let _ = corpus_len; // already captured in shard_map

        // Phase 2: pick a cache to score against. Single-shard borrows from
        // the pool's resident cache (no composition needed). Multi-shard
        // goes through the single-slot `composition` cache: reuse if the
        // request key matches the cached one, otherwise clear the existing
        // buffer and re-prefill in place. Critically, multi-shard must NOT
        // alloc a fresh GpuKvCache per request — that's what hangs the
        // wgpu driver after a few consecutive requests.
        //
        // Each branch returns the same shape so scoring can stay shared.
        let (per_layer_scores, baseline_per_layer, cache_seq) = if shards.len() == 1 {
            // Re-acquire the pool lock briefly to borrow the resident cache
            // for the trace forwards. Holding the lock through the forwards
            // is fine here — trace forwards are ~250ms and other handlers
            // can wait. Composition is not touched on this path.
            let pool = state.cache_pool.lock().await;
            let entry = pool.get(&shards[0]).unwrap();
            // Polar fast path: when the entry has a polar cache populated
            // (server started with --enable-polar-cache), use the polar
            // trace forward; ~7x less KV VRAM read per token.
            if let Some(polar_ref) = entry.polar.as_ref() {
                let cache_seq = polar_ref.seq_len();
                info!(
                    shard = %shards[0],
                    corpus_tokens = corpus_len,
                    query_tokens = n_query,
                    backend = "polar",
                    "retrieval mode: single-shard polar trace forward",
                );
                let (q, b) = tokio::task::block_in_place(|| {
                    let q = state.engine.forward_full_gpu_polar_traced(
                        &prompt_tokens, polar_ref, &capture_layers,
                    );
                    let b = state.engine.forward_full_gpu_polar_traced(
                        &baseline_tokens, polar_ref, &capture_layers,
                    );
                    (q, b)
                });
                (q, b, cache_seq)
            } else {
                let cache_ref = &entry.cache;
                let cache_seq = cache_ref.seq_len();
                info!(
                    shard = %shards[0],
                    corpus_tokens = corpus_len,
                    query_tokens = n_query,
                    backend = "f32",
                    "retrieval mode: single-shard cached forward",
                );
                let (q, b) = tokio::task::block_in_place(|| {
                    let q = state.engine.forward_full_gpu_with_cache_traced(
                        &prompt_tokens, cache_ref, &capture_layers,
                    );
                    let b = state.engine.forward_full_gpu_with_cache_traced(
                        &baseline_tokens, cache_ref, &capture_layers,
                    );
                    (q, b)
                });
                (q, b, cache_seq)
            }
        } else {
            // Build the request key from the snapshot. Order-preserving so
            // shards=[A,B] and shards=[B,A] are different keys (RoPE
            // positions depend on order).
            let key: Vec<(String, u64)> = snapshot.iter()
                .map(|(s, v, _)| (s.clone(), *v))
                .collect();
            let total_tokens_len: usize = snapshot.iter().map(|(_, _, t)| t.len()).sum();

            let mut composition = state.composition.lock().await;
            let reused = composition.as_ref().map(|e| e.key == key).unwrap_or(false);
            if !reused {
                // Reuse the existing buffer if its allocation is large enough
                // for our composition. Otherwise allocate fresh (rare: only
                // when total_tokens_len > current buffer's max_seq_len).
                let mut cache_buf = match composition.take() {
                    Some(e) if e.cache.max_seq_len() >= total_tokens_len => {
                        let mut c = e.cache;
                        c.clear();
                        c
                    }
                    _ => state.engine.create_gpu_kv_cache(state.max_seq_len),
                };
                let all_tokens: Vec<u32> = snapshot.iter()
                    .flat_map(|(_, _, t)| t.iter().copied())
                    .collect();
                tokio::task::block_in_place(|| {
                    if !all_tokens.is_empty() {
                        let _ = state.engine.forward_full_gpu_with_cache(&all_tokens, &mut cache_buf);
                    }
                });
                *composition = Some(ComposedEntry {
                    key,
                    cache: cache_buf,
                });
            }
            let entry_ref = composition.as_ref().unwrap();
            let cache_ref = &entry_ref.cache;
            let cache_seq = cache_ref.seq_len();
            info!(
                shards = ?shards,
                composed_tokens = cache_seq,
                corpus_tokens = corpus_len,
                query_tokens = n_query,
                composition = if reused { "reused" } else { "rebuilt" },
                "retrieval mode: multi-shard composed cached forward",
            );
            let (q, b) = tokio::task::block_in_place(|| {
                let q = state.engine.forward_full_gpu_with_cache_traced(
                    &prompt_tokens, cache_ref, &capture_layers,
                );
                let b = state.engine.forward_full_gpu_with_cache_traced(
                    &baseline_tokens, cache_ref, &capture_layers,
                );
                (q, b)
            });
            (q, b, cache_seq)
        };

        let attn_max_seq = cache_seq + n_query;
        let baseline_attn_max = cache_seq + baseline_tokens.len();

        // Closure: compute per-corpus-position MAX score from a captured
        // per-layer attention tensor (layout [n_q, n_heads, attn_max]).
        // Aggregates across (layers x heads x LAST query position only).
        // Using last-position-only keeps query and baseline comparable
        // (both aggregate over the same number of values: layers x heads).
        let aggregate_max = |per_layer: &[Vec<f32>], n_q: usize, attn_max: usize| -> Vec<f32> {
            let q_last = n_q - 1; // n_q >= 1 (asserted by forward_full_gpu_with_cache_traced)
            let mut out = vec![f32::NEG_INFINITY; corpus_len];
            for k in 0..corpus_len {
                let mut m = f32::NEG_INFINITY;
                for layer_scores in per_layer {
                    for h in 0..n_heads {
                        let idx = q_last * n_heads * attn_max + h * attn_max + k;
                        let v = layer_scores[idx];
                        if v > m { m = v; }
                    }
                }
                if m.is_finite() { out[k] = m; }
            }
            out
        };

        let query_max = aggregate_max(&per_layer_scores, n_query, attn_max_seq);
        let baseline_max = aggregate_max(&baseline_per_layer, baseline_tokens.len(), baseline_attn_max);

        // Differential score: query attention - baseline attention. Positions
        // that are "always hot" (high in both) drop to zero; positions that
        // are query-specific stay high.
        let mut scores = vec![f32::NEG_INFINITY; corpus_len];
        for k in 0..corpus_len {
            if query_max[k].is_finite() && baseline_max[k].is_finite() {
                scores[k] = query_max[k] - baseline_max[k];
            }
        }
        let selected_layers = capture_layers;

        // Rank and take top-K
        let mut ranked: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s.is_finite())
            .map(|(i, &s)| (i, s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k = req.top_k.min(ranked.len());
        let hits: Vec<RetrievalHit> = ranked
            .iter()
            .filter_map(|&(pos, score)| {
                let (shard_name, offset) = shard_map.resolve(pos)?;
                // Skip sink tokens at the start of each shard
                if offset < SINK_TOKENS {
                    return None;
                }
                Some(RetrievalHit {
                    shard_id: shard_name.to_string(),
                    // Report offset relative to real content (after sinks)
                    offset: offset - SINK_TOKENS,
                    length: 1,
                    score,
                })
            })
            .take(top_k)
            .collect();

        let retrieval_ms = retrieve_start.elapsed().as_millis() as u64;

        return Ok(Json(serde_json::to_value(RetrievalResponse {
            hits,
            metadata: RetrievalMetadata {
                retrieval_ms,
                query_tokens: prompt_tokens.len() as u32,
                corpus_tokens: corpus_len as u32,
                layers_used: selected_layers,
            },
        }).unwrap()).into_response());
    }

    // ---------------------------------------------------------------
    // GENERATE MODE (default): produce tokens.
    // ---------------------------------------------------------------
    let (generated_tokens, completion_len) = if !shards.is_empty() {
        let mut pool = state.cache_pool.lock().await;

        // Verify all shards exist before doing any work.
        for shard_name in &shards {
            if !pool.contains_key(shard_name) {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": {
                            "type": "cache_not_found",
                            "message": format!("shard '{}' not found in pool. Use POST /v1/cache/load to create it.", shard_name),
                            "cache_id": shard_name,
                        }
                    })),
                ));
            }
        }

        if shards.len() == 1 {
            // Single shard: use the existing cache directly (fast path,
            // no copying or replaying). This is the common case.
            let entry = pool.get_mut(&shards[0]).unwrap();
            let generated = tokio::task::block_in_place(|| {
                generate_with_cache(
                    &state.engine,
                    &prompt_tokens,
                    &mut entry.cache,
                    sampler_config,
                    seed,
                    eos,
                    max_tokens,
                )
            });
            entry.tokens.extend_from_slice(&prompt_tokens);
            entry.tokens.extend_from_slice(&generated);
            entry.version += 1;
            // Chat extends the f32 cache with prompt + generated K/V.
            // Any polar snapshot is now stale.
            entry.polar = None;
            entry.last_used = Instant::now();
            let len = generated.len() as u32;
            (generated, len)
        } else {
            // Multi-shard: compose by replaying all shards' tokens into
            // a fresh temporary cache in the given order. This gives
            // correct contiguous RoPE positions across shards.
            let mut all_tokens: Vec<u32> = Vec::new();
            for shard_name in &shards {
                let entry = pool.get(shard_name).unwrap();
                all_tokens.extend_from_slice(&entry.tokens);
            }

            let mut composed_cache = state.engine.create_gpu_kv_cache(state.max_seq_len);
            tokio::task::block_in_place(|| {
                if !all_tokens.is_empty() {
                    let _ = state.engine.forward_full_gpu_with_cache(&all_tokens, &mut composed_cache);
                }
            });

            // Now generate with the composed cache
            let generated = tokio::task::block_in_place(|| {
                generate_with_cache(
                    &state.engine,
                    &prompt_tokens,
                    &mut composed_cache,
                    sampler_config,
                    seed,
                    eos,
                    max_tokens,
                )
            });

            // Update the LAST shard with the new tokens (the user's shard
            // is conventionally the last in the list). The shared shards
            // don't change.
            if let Some(last_shard) = shards.last() {
                if let Some(entry) = pool.get_mut(last_shard) {
                    entry.tokens.extend_from_slice(&prompt_tokens);
                    entry.tokens.extend_from_slice(&generated);
                    entry.version += 1;
                    // The polar snapshot of this shard is stale after the
                    // chat append (multi-shard path).
                    entry.polar = None;
                    entry.last_used = Instant::now();
                }
            }

            // The composed cache is temporary and gets dropped.
            let len = generated.len() as u32;
            (generated, len)
        }
    } else {
        // Stateless: create a temporary cache, generate, discard.
        let output_tokens = tokio::task::block_in_place(|| {
            state.engine.generate(&prompt_tokens, max_tokens, sampler_config, seed, Some(eos))
        });
        let generated = output_tokens[prompt_tokens.len()..].to_vec();
        let len = generated.len() as u32;
        (generated, len)
    };

    let text = state.tokenizer.decode(&generated_tokens);

    // Determine finish reason and check for tool calls
    let finish_reason;
    let mut response_msg = ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: None,
        tool_call_id: None,
    };

    if req.tools.is_some() {
        if let Some(tool_calls) = parse_tool_calls(&text) {
            finish_reason = "tool_calls".to_string();
            response_msg.tool_calls = Some(tool_calls);
        } else {
            finish_reason = if completion_len >= req.max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            };
            response_msg.content = Some(text);
        }
    } else {
        finish_reason = if completion_len >= req.max_tokens {
            "length".to_string()
        } else {
            "stop".to_string()
        };
        response_msg.content = Some(text);
    }

    let response = ChatResponse {
        id: format!("cortex-{}", &uuid::Uuid::new_v4().to_string()[..12]),
        model: state.model_name.clone(),
        choices: vec![Choice {
            message: response_msg,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
        },
    };

    Ok(Json(serde_json::to_value(response).unwrap()).into_response())
}

/// Streaming variant of `chat_completions`. Stateless mode only.
///
/// Generation runs in a `spawn_blocking` task so the GPU calls don't
/// block the tokio reactor. Each new sampled token gets detokenized
/// incrementally (delta = decode(all) - decode(all_minus_last)) and
/// pushed through an mpsc channel; the SSE stream emits one
/// `chat.completion.chunk` event per delta. The OpenAI shape is:
///
/// ```text
/// data: {"id":"...","object":"chat.completion.chunk","choices":[
///   {"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
/// data: {"choices":[{"index":0,"delta":{"content":"Hello"},...}]}
/// ...
/// data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
/// data: [DONE]
/// ```
///
/// On client disconnect (channel send fails), generation aborts on the
/// next token boundary and the spawned task exits cleanly. The temporary
/// cache is dropped on task exit.
async fn chat_completions_stream(
    state: Arc<ServerState>,
    req: ChatRequest,
    prompt_tokens: Vec<u32>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let sampler_config = if req.temperature <= 0.0 {
        SamplerConfig::greedy()
    } else {
        SamplerConfig {
            temperature: req.temperature,
            top_k: 40,
            top_p: 0.95,
            ..Default::default()
        }
    };
    let eos = state.tokenizer.eos_token_id();
    let seed = rand_seed();
    let max_tokens = req.max_tokens as usize;

    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Bounded channel: small buffer so a slow client back-pressures the
    // GPU rather than letting tokens pile up unboundedly. Each item is
    // an Option<String>: Some(delta) for content, None for "we're done,
    // emit the final finish_reason chunk".
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamMessage>(8);

    // Spawn the generation. block_in_place isn't an option from inside a
    // spawn_blocking, so we keep this in a blocking thread and use the
    // channel's blocking send path.
    let state_for_gen = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut cache = state_for_gen.engine.create_gpu_kv_cache(state_for_gen.max_seq_len);
        let mut sampler = Sampler::new(sampler_config, seed);

        // Prefill + first token.
        let prefill_logits = state_for_gen.engine.forward_full_gpu_with_cache(&prompt_tokens, &mut cache);
        let vocab = state_for_gen.engine.vocab_size();
        let last_off = (prompt_tokens.len() - 1) * vocab;
        let mut next_token = sampler.sample(&prefill_logits[last_off..last_off + vocab]);

        let mut generated: Vec<u32> = Vec::new();
        let mut emitted_text = String::new();

        let push_delta = |generated: &Vec<u32>, emitted_text: &mut String,
                          tx: &tokio::sync::mpsc::Sender<StreamMessage>| -> bool {
            let full = state_for_gen.tokenizer.decode(generated);
            if full.len() > emitted_text.len() && full.starts_with(emitted_text.as_str()) {
                let delta = full[emitted_text.len()..].to_string();
                *emitted_text = full;
                tx.blocking_send(StreamMessage::Delta(delta)).is_ok()
            } else {
                // Decode shrank or diverged (rare; happens with some BPE
                // edge cases when a new token reshapes earlier output).
                // Reset baseline; don't emit a delta this round.
                *emitted_text = full;
                true
            }
        };

        if next_token != eos {
            generated.push(next_token);
            if !push_delta(&generated, &mut emitted_text, &tx) {
                return; // client gone
            }

            for _ in 1..max_tokens {
                let logits = state_for_gen.engine.forward_full_gpu_with_cache(&[next_token], &mut cache);
                next_token = sampler.sample(&logits);
                if next_token == eos {
                    break;
                }
                generated.push(next_token);
                if !push_delta(&generated, &mut emitted_text, &tx) {
                    return; // client gone
                }
            }
        }

        let finish = if generated.len() >= max_tokens { "length" } else { "stop" };
        let _ = tx.blocking_send(StreamMessage::Finish(finish.to_string()));
        // Cache dropped on scope exit.
    });

    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let chunk_id_for_stream = chunk_id.clone();
    let model_for_stream = model_name.clone();

    // Initial chunk: role only, no content. OpenAI clients expect this.
    let role_event = stream_chunk_event(
        &chunk_id, created, &model_name,
        Some(serde_json::json!({"role": "assistant"})),
        None,
    );

    // Subsequent chunks come from the generation task.
    let body_stream = ReceiverStream::new(rx).map(move |msg| {
        Ok::<_, std::convert::Infallible>(match msg {
            StreamMessage::Delta(text) => stream_chunk_event(
                &chunk_id_for_stream, created, &model_for_stream,
                Some(serde_json::json!({"content": text})),
                None,
            ),
            StreamMessage::Finish(reason) => stream_chunk_event(
                &chunk_id_for_stream, created, &model_for_stream,
                Some(serde_json::json!({})),
                Some(reason),
            ),
        })
    });

    // Final [DONE] sentinel per OpenAI SSE spec.
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });

    let initial = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(role_event)
    });

    let combined = initial.chain(body_stream).chain(done_event);

    Ok(Sse::new(combined)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// One unit of work crossing the gen-thread → SSE-stream boundary.
enum StreamMessage {
    /// Newly-detokenized text since the last chunk.
    Delta(String),
    /// Generation finished; emit the final chunk with this finish_reason
    /// ("stop" or "length"). Always the last message before the channel closes.
    Finish(String),
}

/// Build one `chat.completion.chunk` SSE event with the given delta and
/// optional finish_reason. Mirrors OpenAI's wire shape exactly.
fn stream_chunk_event(
    id: &str,
    created: u64,
    model: &str,
    delta: Option<serde_json::Value>,
    finish_reason: Option<String>,
) -> Event {
    let payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta.unwrap_or(serde_json::json!({})),
            "finish_reason": finish_reason,
        }],
    });
    Event::default().data(payload.to_string())
}

// Suppress unused-import warning for Stream when only used via concrete types.
#[allow(dead_code)]
fn _stream_marker(_: impl Stream) {}

// ---------------------------------------------------------------------------
// Cache endpoints
// ---------------------------------------------------------------------------

/// POST /v1/cache/load — create or replace a cache by replaying tokens.
///
/// If tokens is empty, creates an empty cache (cold start for a new user).
/// If tokens is non-empty, runs forward_cached to build the KV cache from
/// the token history (reawaken after eviction).
///
/// "load replaces" — if the cache_id already exists, it's overwritten.
async fn cache_load(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CacheLoadRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut cache = state.engine.create_gpu_kv_cache(state.max_seq_len);

    // Prepend sink tokens (BOS repeated) to absorb position-0 attention
    // sink artifact. Real content starts at position SINK_TOKENS.
    let bos = state.tokenizer.bos_token_id();
    let sink_tokens: Vec<u32> = vec![bos; SINK_TOKENS];
    let mut all_tokens = sink_tokens;
    all_tokens.extend_from_slice(&req.tokens);

    if !all_tokens.is_empty() {
        tokio::task::block_in_place(|| {
            let _ = state.engine.forward_full_gpu_with_cache(&all_tokens, &mut cache);
        });
    }

    // If polar caching is enabled, build a parallel polar cache from the
    // f32 prefill output via the GPU compress shader. No CPU round-trip.
    let polar = if state.polar_cache_enabled {
        tokio::task::block_in_place(|| {
            let mut p = state.engine.create_gpu_polar_kv_cache(
                state.max_seq_len, state.polar_rotation_seed,
            );
            p.populate_from_f32_cache_gpu(&cache);
            Some(p)
        })
    } else {
        None
    };

    let seq_len = cache.seq_len();
    let now = Instant::now();

    let mut pool = state.cache_pool.lock().await;
    // If overwriting an existing shard, bump from its current version so the
    // composition cache's staleness check sees the change. New shards start
    // at version 0; any subsequent insert / append bumps it monotonically.
    let next_version = pool.get(&req.cache_id).map(|e| e.version + 1).unwrap_or(0);
    pool.insert(
        req.cache_id.clone(),
        CacheEntry {
            cache,
            polar,
            tokens: all_tokens,
            version: next_version,
            created_at: now,
            last_used: now,
        },
    );
    let pool_size = pool.len();
    drop(pool);
    // Drop any stale composition that referenced the old (or absent) version
    // of this shard. Cheap: a single buffer-array drop on the GPU.
    *state.composition.lock().await = None;

    info!(
        cache_id = %req.cache_id,
        seq_len = seq_len,
        tokens_replayed = req.tokens.len(),
        pool_size = pool_size,
        "cache loaded",
    );

    Ok((
        StatusCode::CREATED,
        Json(CacheLoadResponse {
            cache_id: req.cache_id,
            seq_len,
            status: "loaded".to_string(),
        }),
    ))
}

/// POST /v1/cache/append — extend an existing cache with new tokens.
///
/// Runs forward_cached on the new tokens against the existing cache.
/// "append extends" — the cache grows by the new tokens.
async fn cache_append(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CacheAppendRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut pool = state.cache_pool.lock().await;
    let entry = pool.get_mut(&req.cache_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "cache_not_found",
                    "message": format!("cache_id '{}' not found", req.cache_id),
                    "cache_id": req.cache_id,
                }
            })),
        )
    })?;

    if !req.tokens.is_empty() {
        tokio::task::block_in_place(|| {
            let _ = state.engine.forward_full_gpu_with_cache(&req.tokens, &mut entry.cache);
        });
        entry.tokens.extend_from_slice(&req.tokens);
        entry.version += 1;
        // The polar cache (if any) was a snapshot of the f32 cache at
        // load time. After append, it's stale. Drop it; the next /v1/retrieve
        // against this shard will fall back to the f32 path. A future
        // refinement can rebuild the polar cache here, but for first wiring
        // we keep it conservative.
        entry.polar = None;
    }

    entry.last_used = Instant::now();
    let seq_len = entry.cache.seq_len();
    drop(pool);
    // Invalidate composition cache; any composition that included this
    // shard is now stale (version bumped above).
    if !req.tokens.is_empty() {
        *state.composition.lock().await = None;
    }

    info!(
        cache_id = %req.cache_id,
        seq_len = seq_len,
        tokens_appended = req.tokens.len(),
        "cache appended",
    );

    Ok(Json(CacheInfoResponse {
        cache_id: req.cache_id,
        seq_len,
        max_seq_len: state.max_seq_len,
    }))
}

/// GET /v1/cache/{id} — get cache info.
async fn cache_get(
    State(state): State<Arc<ServerState>>,
    Path(cache_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let pool = state.cache_pool.lock().await;
    let entry = pool.get(&cache_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "cache_not_found",
                    "message": format!("cache_id '{}' not found", cache_id),
                    "cache_id": cache_id,
                }
            })),
        )
    })?;

    Ok(Json(CacheInfoResponse {
        cache_id,
        seq_len: entry.cache.seq_len(),
        max_seq_len: state.max_seq_len,
    }))
}

/// DELETE /v1/cache/{id} — evict a cache from the pool.
async fn cache_delete(
    State(state): State<Arc<ServerState>>,
    Path(cache_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut pool = state.cache_pool.lock().await;
    if pool.remove(&cache_id).is_some() {
        let pool_size = pool.len();
        drop(pool);
        // Composition might reference the evicted shard; safest to drop.
        *state.composition.lock().await = None;
        info!(cache_id = %cache_id, pool_size = pool_size, "cache evicted");
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "cache_not_found",
                    "message": format!("cache_id '{}' not found", cache_id),
                    "cache_id": cache_id,
                }
            })),
        ))
    }
}

// ---------------------------------------------------------------------------
// Shim registry endpoints (5a)
// ---------------------------------------------------------------------------

/// PUT body shape: JSON envelope with the manifest and the ONNX bytes
/// as base64. Multipart-free so AgentOS's HTTP client can use the same
/// JSON pipeline as every other endpoint.
#[derive(Deserialize)]
struct ShimPutRequest {
    manifest: ShimManifest,
    /// Base64-encoded ONNX model bytes.
    onnx_base64: String,
}

#[derive(Serialize)]
struct ShimRegistryEntry {
    manifest: ShimManifest,
}

#[derive(Serialize)]
struct ShimsListResponse {
    shims: Vec<ShimManifest>,
}

/// PUT /v1/shims/{id} — register a shim. Body decodes the ONNX bytes,
/// loads them via ort, and stores the (manifest, session) pair in the
/// registry. If `id` already exists, replaces it.
///
/// 400 on base64 decode failure or ONNX load failure (the bytes were
/// junk or not a well-formed ONNX model). 400 on id mismatch between
/// URL path and manifest body — refuse to register an ambiguous shim.
async fn shim_put(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<String>,
    Json(req): Json<ShimPutRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if req.manifest.id != id {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": format!("manifest id '{}' does not match URL path id '{}'",
                                       req.manifest.id, id),
                }
            })),
        ));
    }

    use base64::Engine;
    let onnx_bytes = base64::engine::general_purpose::STANDARD
        .decode(&req.onnx_base64)
        .map_err(|e| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": format!("onnx_base64 decode failed: {e}"),
                }
            })),
        ))?;

    // Build the ort session from in-memory bytes. Runs synchronously
    // (ORT init is fast) — wrapping in spawn_blocking would be
    // overkill for typical shim sizes (~28k params, < 100 KB ONNX).
    let session = ort::session::Session::builder()
        .and_then(|mut b| b.commit_from_memory(&onnx_bytes))
        .map_err(|e| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": format!("ort session load failed: {e}"),
                }
            })),
        ))?;

    let registered = Arc::new(RegisteredShim {
        manifest: req.manifest.clone(),
        session: Mutex::new(session),
    });

    let mut shims = state.shims.lock().await;
    let existed = shims.insert(id.clone(), registered).is_some();
    let count = shims.len();
    drop(shims);

    info!(
        shim_id = %id,
        version = %req.manifest.version,
        phase = %req.manifest.phase,
        replaced = existed,
        registered_count = count,
        "shim registered",
    );

    Ok((
        if existed { StatusCode::OK } else { StatusCode::CREATED },
        Json(ShimRegistryEntry { manifest: req.manifest }),
    ))
}

/// GET /v1/shims/{id} — return one shim's manifest.
async fn shim_get(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let shims = state.shims.lock().await;
    let entry = shims.get(&id).ok_or_else(|| (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "type": "shim_not_found",
                "message": format!("shim '{id}' not registered"),
                "shim_id": id,
            }
        })),
    ))?;
    Ok(Json(ShimRegistryEntry { manifest: entry.manifest.clone() }))
}

/// GET /v1/shims/ — list all registered shim manifests.
async fn shims_list(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let shims = state.shims.lock().await;
    let manifests: Vec<ShimManifest> = shims.values()
        .map(|s| s.manifest.clone())
        .collect();
    Json(ShimsListResponse { shims: manifests })
}

/// POST /v1/shims/infer — standalone classification.
///
/// Pipeline (v1, attachment.layer = "final" only):
///   1. Tokenize `context` with the model tokenizer
///   2. Forward through the model, capture final post-norm hidden state
///      ([n_tokens, embed_dim] f32)
///   3. Pool per manifest.attachment.pooling
///      ("last_token" | "mean"; "attention" / "none" return 400 for v1)
///   4. Validate pooled length == manifest.input_shape.hidden_dim
///   5. Run the registered ort Session over the pooled vector
///   6. Format the output per manifest.output_shape.kind
///      ("scalar" → first f32; "category" → argmax + probs;
///       "hidden_delta" → raw vector; others 400)
///
/// "entrance:N" attachments + "attention" pooling are explicit v2.
#[derive(Deserialize)]
struct ShimInferRequest {
    shim_id: String,
    context: String,
}

#[allow(clippy::too_many_lines)]
async fn shim_infer(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ShimInferRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let shim = {
        let shims = state.shims.lock().await;
        shims.get(&req.shim_id).cloned().ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "shim_not_found",
                    "message": format!("shim '{}' not registered", req.shim_id),
                    "shim_id": req.shim_id,
                }
            })),
        ))?
    };
    let manifest = &shim.manifest;

    // v1 supports only attachment.layer = "final".
    if manifest.attachment.layer != "final" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": format!(
                        "v1 supports only attachment.layer='final'; got '{}'",
                        manifest.attachment.layer),
                }
            })),
        ));
    }

    // 1. Tokenize.
    let tokens = state.tokenizer.encode(&req.context, /*add_bos*/ true);
    if tokens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "type": "invalid_request", "message": "empty context after tokenization" }
            })),
        ));
    }

    // 2. Forward through cortex; capture final post-norm hidden only.
    let infer_start = Instant::now();
    let hc = tokio::task::block_in_place(|| {
        state.engine.forward_full_gpu_with_hidden_capture(&tokens, &[])
    });
    let cortex_ms = infer_start.elapsed().as_millis() as u64;

    // 3. Pool.
    let pooled: Vec<f32> = match manifest.attachment.pooling.as_str() {
        "last_token" => hc.final_last_token().to_vec(),
        "mean" => {
            let mut sum = vec![0.0f32; hc.embed_dim];
            for t in 0..hc.n_tokens {
                let off = t * hc.embed_dim;
                for d in 0..hc.embed_dim {
                    sum[d] += hc.final_post_norm_hidden[off + d];
                }
            }
            let n = hc.n_tokens as f32;
            for v in sum.iter_mut() { *v /= n; }
            sum
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!("v1 pooling supports last_token | mean; got '{other}'"),
                    }
                })),
            ));
        }
    };

    // 4. Validate hidden_dim against manifest.
    let expected_dim = manifest.input_shape.get("hidden_dim")
        .and_then(|v| v.as_u64()).map(|n| n as usize)
        .ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "invalid_manifest",
                    "message": "manifest.input_shape must include integer 'hidden_dim'",
                }
            })),
        ))?;
    if pooled.len() != expected_dim {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "shape_mismatch",
                    "message": format!(
                        "model embed_dim={} != manifest input_shape.hidden_dim={}",
                        pooled.len(), expected_dim),
                }
            })),
        ));
    }

    // 5. Run the ort session. Lock the per-shim session mutex (ort
    //    Sessions aren't Sync); v1 serializes inference per shim, fine
    //    for a server that's not yet fronting many concurrent users.
    //    Scoped so all borrows of `session` end (outputs / extracted
    //    tensor refs) before the function continues with owned data.
    let ort_start = Instant::now();
    let (out_vec, out_shape_vec) = {
        let mut session = shim.session.lock().await;
        let input_name = session.inputs().first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".to_string());
        let tensor = ort::value::TensorRef::from_array_view((
            vec![1_i64, pooled.len() as i64],
            pooled.as_slice(),
        )).map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "type": "ort_error",
                    "message": format!("input tensor construction failed: {e}"),
                }
            })),
        ))?;
        let outputs = session.run(ort::inputs![input_name.as_str() => tensor]).map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "type": "ort_error",
                    "message": format!("session run failed: {e}"),
                }
            })),
        ))?;
        let first_out = outputs.iter().next().ok_or_else(|| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "type": "ort_error",
                    "message": "session produced no outputs",
                }
            })),
        ))?.1;
        let (out_shape, out_data) = first_out.try_extract_tensor::<f32>().map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "type": "ort_error",
                    "message": format!("output extraction failed: {e}"),
                }
            })),
        ))?;
        (out_data.to_vec(), out_shape.iter().copied().collect::<Vec<i64>>())
    };
    let ort_ms = ort_start.elapsed().as_millis() as u64;

    // 6. Format per output_shape.kind. v1 accepted: "scalar",
    //    "category:N" (N read from the shape if needed), "hidden_delta".
    let kind = manifest.output_shape.get("kind").and_then(|v| v.as_str()).unwrap_or("scalar");
    let decision: serde_json::Value = if kind == "scalar" {
        if out_vec.is_empty() {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": { "type": "ort_error", "message": "scalar output empty" }
            }))));
        }
        serde_json::json!(out_vec[0])
    } else if kind.starts_with("category") {
        let argmax = out_vec.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
            .0;
        serde_json::json!(argmax)
    } else if kind == "hidden_delta" {
        serde_json::json!(out_vec)
    } else {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": { "type": "unsupported", "message": format!("output_shape.kind '{kind}' not supported") }
        }))));
    };

    let response = serde_json::json!({
        "decision": decision,
        "metadata": {
            "shim_id": req.shim_id,
            "shim_version": manifest.version,
            "kind": kind,
            "output_shape": out_shape_vec,
            "context_tokens": tokens.len(),
            "cortex_ms": cortex_ms,
            "ort_ms": ort_ms,
            "raw_output": out_vec,
        }
    });

    info!(
        shim_id = %req.shim_id,
        kind = %kind,
        context_tokens = tokens.len(),
        cortex_ms,
        ort_ms,
        "shim infer",
    );

    Ok(Json(response))
}

/// DELETE /v1/shims/{id} — unregister a shim.
async fn shim_delete(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut shims = state.shims.lock().await;
    let removed = shims.remove(&id).is_some();
    let count = shims.len();
    drop(shims);
    if removed {
        info!(shim_id = %id, registered_count = count, "shim unregistered");
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "shim_not_found",
                    "message": format!("shim '{id}' not registered"),
                    "shim_id": id,
                }
            })),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tokenize endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    text: String,
    #[serde(default)]
    add_bos: Option<bool>,
}

#[derive(Serialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
    count: usize,
}

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

#[derive(Deserialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
}

#[derive(Serialize)]
struct DetokenizeResponse {
    text: String,
}

/// POST /v1/detokenize — convert token IDs back to text.
///
/// Useful for resolving retrieval hits — given a hit at (offset, length),
/// the caller can pull tokens[offset-k .. offset+length+k] from the original
/// shard token list and POST them here to get a human-readable context window.
async fn detokenize(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Json<DetokenizeResponse> {
    let text = state.tokenizer.decode(&req.tokens);
    Json(DetokenizeResponse { text })
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

async fn health(
    State(state): State<Arc<ServerState>>,
) -> Json<HealthResponse> {
    let pool = state.cache_pool.lock().await;
    let total_tokens: usize = pool.values().map(|e| e.cache.seq_len()).sum();
    Json(HealthResponse {
        status: "ready".to_string(),
        model: state.model_name.clone(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        memory: HealthMemory {
            cache_pool_size: pool.len(),
            cache_pool_total_tokens: total_tokens,
            max_seq_len: state.max_seq_len,
        },
    })
}

/// Simple non-crypto RNG seed from system time.
fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
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
        cache_enabled,
        retrieve_enabled,
        polar_cache_enabled,
        polar_rotation_seed,
        shims: Mutex::new(HashMap::new()),
        shims_enabled: cli.enable_shims,
    });

    // Build router: always include completions, models, health.
    // Cache and retrieve endpoints are conditional on startup flags.
    let mut app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/models", get(list_models))
        .route("/health", get(health));

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
