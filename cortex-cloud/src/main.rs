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
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
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
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let prompt_tokens = apply_chat_template(
        &req.messages,
        req.tools.as_deref(),
        &state.tokenizer,
    );

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
            let cache_ref = &pool.get(&shards[0]).unwrap().cache;
            let cache_seq = cache_ref.seq_len();
            info!(
                shard = %shards[0],
                corpus_tokens = corpus_len,
                query_tokens = n_query,
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
        }).unwrap()));
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

    Ok(Json(serde_json::to_value(response).unwrap()))
}

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

    let app = app.with_state(state);

    let addr = format!("{}:{}", cli.bind, cli.port);
    info!(
        addr = %addr,
        model = %model_name,
        cache = cache_enabled,
        retrieve = retrieve_enabled,
        "cortex-server ready",
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
