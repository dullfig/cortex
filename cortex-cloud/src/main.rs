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

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use cortex::layers::model::TransformerModel;
use cortex::layers::sampler::SamplerConfig;
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

struct ServerState {
    model: TransformerModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: ModelConfig,
    cache: Mutex<ModelKvCache>,
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
    kv_cache_entries: usize,
    max_seq_len: usize,
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

    // Generate — this is blocking CPU work, run on a blocking thread
    let model = &state.model;
    let max_tokens = req.max_tokens as usize;

    let output_tokens = tokio::task::block_in_place(|| {
        model.generate(&prompt_tokens, max_tokens, sampler_config, seed, Some(eos))
    });

    // Decode only the generated part (skip prompt)
    let generated = &output_tokens[prompt_tokens.len()..];
    let completion_len = generated.len() as u32;
    let text = state.tokenizer.decode(generated);

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
    let cache = state.cache.lock().await;
    Json(HealthResponse {
        status: "ready".to_string(),
        model: state.model_name.clone(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        memory: HealthMemory {
            kv_cache_entries: cache.seq_len(),
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

    let cache = loaded.model.create_kv_cache(cli.max_seq_len);

    let state = Arc::new(ServerState {
        model: loaded.model,
        tokenizer: loaded.tokenizer,
        config: loaded.config,
        cache: Mutex::new(cache),
        model_name: model_name.clone(),
        start_time: Instant::now(),
        max_seq_len: cli.max_seq_len,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", cli.bind, cli.port);
    info!(addr = %addr, model = %model_name, "cortex-server ready");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
