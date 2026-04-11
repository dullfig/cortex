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

use cortex::layers::model::TransformerModel;
use cortex::layers::sampler::{Sampler, SamplerConfig};
use cortex::layers::kv_cache::ModelKvCache;
use cortex::{ModelConfig, Tokenizer};

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
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Per-cache metadata stored alongside the KV cache in the pool.
struct CacheEntry {
    cache: ModelKvCache,
    created_at: Instant,
    last_used: Instant,
}

struct ServerState {
    model: TransformerModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: ModelConfig,
    /// Pool of named KV caches. Key is the opaque cache_id assigned by agentos.
    /// Cortex never invents a cache_id — every cache in this pool got here via
    /// an explicit POST /v1/cache/load. The invariant "cortex never invents state"
    /// is load-bearing for the eviction-recovery lifecycle (see CACHE-LIFECYCLE.md).
    cache_pool: Mutex<HashMap<String, CacheEntry>>,
    model_name: String,
    start_time: Instant,
    max_seq_len: usize,
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
    /// Opaque cache identifier assigned by the caller (agentos). If present,
    /// cortex looks up the cache in the pool and uses it for this request.
    /// If the cache_id is not in the pool, cortex returns 404 — it never
    /// creates a cache implicitly. Use POST /v1/cache/load to create one.
    /// If absent, the request runs stateless with a fresh temporary cache.
    #[serde(default)]
    cache_id: Option<String>,
}

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

    // If cache_id is provided, use the pool cache; otherwise stateless.
    let (generated_tokens, completion_len) = if let Some(ref cache_id) = req.cache_id {
        let mut pool = state.cache_pool.lock().await;
        let entry = pool.get_mut(cache_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "type": "cache_not_found",
                        "message": format!("cache_id '{}' not found in pool. Use POST /v1/cache/load to create it.", cache_id),
                        "cache_id": cache_id,
                    }
                })),
            )
        })?;

        // Generate with the existing cache — prefill the new prompt
        // tokens then sample autoregressively.
        let generated = tokio::task::block_in_place(|| {
            let model = &state.model;
            let cache = &mut entry.cache;
            let mut sampler = Sampler::new(sampler_config, seed);

            // Prefill the prompt into the existing cache
            let prefill_logits = model.forward_cached(&prompt_tokens, cache);
            let last_logits_start = (prompt_tokens.len() - 1) * model.vocab_size();
            let last_logits = &prefill_logits[last_logits_start..last_logits_start + model.vocab_size()];
            let mut next_token = sampler.sample(last_logits);

            let mut out = Vec::new();
            if Some(next_token) == Some(eos) {
                return out;
            }
            out.push(next_token);

            for _ in 1..max_tokens {
                let logits = model.forward_cached(&[next_token], cache);
                next_token = sampler.sample(&logits);
                if next_token == eos {
                    break;
                }
                out.push(next_token);
            }
            out
        });

        entry.last_used = Instant::now();
        let len = generated.len() as u32;
        (generated, len)
    } else {
        // Stateless: create a temporary cache, generate, discard.
        let output_tokens = tokio::task::block_in_place(|| {
            state.model.generate(&prompt_tokens, max_tokens, sampler_config, seed, Some(eos))
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

    Ok(Json(response))
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
    let mut cache = state.model.create_kv_cache(state.max_seq_len);

    if !req.tokens.is_empty() {
        // Replay the token history to build the KV cache
        tokio::task::block_in_place(|| {
            let _ = state.model.forward_cached(&req.tokens, &mut cache);
        });
    }

    let seq_len = cache.seq_len();
    let now = Instant::now();

    let mut pool = state.cache_pool.lock().await;
    pool.insert(
        req.cache_id.clone(),
        CacheEntry {
            cache,
            created_at: now,
            last_used: now,
        },
    );

    info!(
        cache_id = %req.cache_id,
        seq_len = seq_len,
        tokens_replayed = req.tokens.len(),
        pool_size = pool.len(),
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
            let _ = state.model.forward_cached(&req.tokens, &mut entry.cache);
        });
    }

    entry.last_used = Instant::now();
    let seq_len = entry.cache.seq_len();

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
        info!(cache_id = %cache_id, pool_size = pool.len(), "cache evicted");
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

    let state = Arc::new(ServerState {
        model: loaded.model,
        tokenizer: loaded.tokenizer,
        config: loaded.config,
        cache_pool: Mutex::new(HashMap::new()),
        model_name: model_name.clone(),
        start_time: Instant::now(),
        max_seq_len: cli.max_seq_len,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/cache/load", post(cache_load))
        .route("/v1/cache/append", post(cache_append))
        .route("/v1/cache/{id}", get(cache_get).delete(cache_delete))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", cli.bind, cli.port);
    info!(addr = %addr, model = %model_name, "cortex-server ready");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
