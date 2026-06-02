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

use cortex::layers::gpu_engine::{GpuEngine, HiddenCaptures};
use cortex::wgpu;
use cortex::layers::gpu_kv_cache::GpuKvCache;
use cortex::layers::sampler::{Sampler, SamplerConfig};
use cortex::{ForwardTrace, ModelConfig, Tokenizer};

mod metrics;

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
    /// f32 KV cache. `None` for polar-only shards (loaded with
    /// `polar_only=true` so the f32 copy is dropped after the polar
    /// cache is materialized — ~7x VRAM win per shard). Polar-only
    /// shards reject chat and append operations with 409 since both
    /// require the f32 cache today; only `/v1/retrieve` is supported.
    cache: Option<GpuKvCache>,
    /// Optional PolarQuant-compressed K/V. Populated once at cache_load
    /// time (via `populate_from_f32_cache_gpu`) when `--enable-polar-cache`
    /// is set. Single-shard `/v1/retrieve` queries route through this
    /// when present; multi-shard composition replays from `tokens`
    /// regardless. When `cache` is `None`, this is the only KV storage.
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

/// One declarative gate-rule. Per `project_cortex_v1_shim_api.md`, the
/// rule table is intentionally bounded — match-and-dispatch only, no
/// scripting. Wire shape:
///
/// ```json
/// {"if": {"gate": "is_crisis", "gt": 0.8}, "then": {"silent": true, "signal": "escalate"}}
/// {"if": {"gate": "should_respond", "gt": 0.7}, "then": {"activate": ["voice_bob"]}}
/// {"else": {"silent": true}}
/// ```
///
/// We accept `else: <action>` as an alias for `then: <action>` — both
/// produce the same Rust field. An absent `if` means "always fires
/// when reached" (the conventional fallthrough).
#[derive(Debug, Deserialize, Clone)]
struct ShimRule {
    #[serde(rename = "if", default)]
    cond: Option<RuleCondition>,
    #[serde(alias = "else", default)]
    then: RuleAction,
}

/// One gate's value tested against a single comparison op. Exactly one
/// of `gt` / `lt` / `eq` should be set; if multiple are set, the
/// condition matches only when ALL set ops are satisfied (AND). If
/// none are set, the condition never matches — that's a request bug,
/// not a fallthrough.
#[derive(Debug, Deserialize, Clone)]
struct RuleCondition {
    /// shim_id whose decision is being tested
    gate: String,
    #[serde(default)]
    gt: Option<f64>,
    #[serde(default)]
    lt: Option<f64>,
    #[serde(default)]
    eq: Option<f64>,
}

impl RuleCondition {
    /// Match the gate's decision JSON against this condition. Decisions
    /// must be coercible to f64 (scalar shims emit numbers; category
    /// shims emit integer argmax). Anything else (hidden_delta arrays,
    /// nulls) never matches.
    fn matches(&self, decision: &serde_json::Value) -> bool {
        let val = match decision.as_f64() {
            Some(v) => v,
            None => return false,
        };
        let mut any_op = false;
        if let Some(t) = self.gt { any_op = true; if !(val > t) { return false; } }
        if let Some(t) = self.lt { any_op = true; if !(val < t) { return false; } }
        if let Some(t) = self.eq { any_op = true; if (val - t).abs() > 1e-9 { return false; } }
        any_op
    }
}

/// Action taken when a rule fires. Composed of three independent
/// vocabulary slots — silent, activate (steers), signal — per the v1
/// spec. The first matching rule wins (no multi-rule composition in
/// v1; complex logic stays in AgentOS via multi-call orchestration).
#[derive(Debug, Deserialize, Clone, Default)]
struct RuleAction {
    /// Mark this completion as silent — emit a `done` event with
    /// `finish_reason: "silent"` and zero generated content.
    #[serde(default)]
    silent: bool,
    /// Steers to activate for the decode path. v1 records these in
    /// response metadata; actual per-token application lands in 6b.
    #[serde(default)]
    activate: Vec<String>,
    /// Free-form signal string surfaced in `done.metadata.signals`.
    /// AgentOS uses this for downstream routing (e.g., "escalate").
    #[serde(default)]
    signal: Option<String>,
}

/// Evaluate the shim_rules table top-to-bottom against the gate
/// decisions. First matching rule wins; an `else` rule (no `if`)
/// fires unconditionally when reached, so it should appear last.
/// Returns `None` when no rule matched (or rules is empty) — the
/// caller distinguishes "no rule routed" from "rule explicitly chose
/// the default action" so it can fall back to `req.steer_shims` in
/// the former case (per the v1 spec: steer_shims is the default,
/// rules override).
fn evaluate_shim_rules(
    rules: &[ShimRule],
    decisions: &HashMap<String, serde_json::Value>,
) -> Option<RuleAction> {
    for rule in rules {
        let fires = match &rule.cond {
            Some(c) => decisions.get(&c.gate).map(|d| c.matches(d)).unwrap_or(false),
            None => true,
        };
        if fires {
            return Some(rule.then.clone());
        }
    }
    None
}

/// Outcome of running gate shims and evaluating shim_rules. Both the
/// streaming and non-streaming chat paths consume this to decide
/// whether to short-circuit silent or proceed to generation.
struct GateOutcome {
    /// shim_id → JSON decision. Surfaced in response metadata under
    /// `gate_decisions` for observability.
    decisions: HashMap<String, serde_json::Value>,
    /// One shared cortex prefill cost across all gate shims.
    gate_prefill_ms: u64,
    /// shim_id → ort run time in ms (per-shim variable cost).
    ort_ms_per_shim: HashMap<String, u64>,
    /// `Some(action)` when a rule matched and dictates dispatch;
    /// `None` when no rule matched (rules empty, or no condition fired).
    /// In the `None` case, callers fall back to `req.steer_shims` for
    /// the active-steer set and treat silent as false / signal as None.
    matched: Option<RuleAction>,
}

impl GateOutcome {
    /// True if a matched rule routed to silent. Default-action when no
    /// rule matched is "proceed normally" — silent must be explicit.
    fn is_silent(&self) -> bool {
        self.matched.as_ref().map(|a| a.silent).unwrap_or(false)
    }

    /// Signal string from the matched rule, if any.
    fn signal(&self) -> Option<&String> {
        self.matched.as_ref().and_then(|a| a.signal.as_ref())
    }

    /// Build the response-metadata JSON value emitted on `done`.
    /// `active_steers` and `active_inject` are filled by the caller
    /// so metadata reflects what *actually* shaped generation.
    fn to_metadata_with_inject(
        &self,
        active_steers: &[String],
        active_inject: &[String],
    ) -> serde_json::Value {
        let signals: Vec<&String> = self.signal().into_iter().collect();
        serde_json::json!({
            "gate_decisions": self.decisions,
            "active_steers": active_steers,
            "active_inject": active_inject,
            "signals": signals,
            "gate_prefill_ms": self.gate_prefill_ms,
            "ort_ms_per_shim": self.ort_ms_per_shim,
        })
    }
}

/// Resolve gate shim IDs against the registry and validate phase=gate.
/// Returns the resolved shims paired with their original IDs (for
/// metadata reporting). Errors map to 404/400. Empty input returns
/// empty result.
async fn resolve_gate_shims(
    state: &Arc<ServerState>,
    gate_ids: &[String],
) -> Result<Vec<(String, Arc<RegisteredShim>)>, (StatusCode, Json<serde_json::Value>)> {
    if gate_ids.is_empty() { return Ok(Vec::new()); }
    let shims = state.shims.lock().await;
    let mut out = Vec::with_capacity(gate_ids.len());
    for id in gate_ids {
        let shim = shims.get(id).cloned().ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "shim_not_found",
                    "message": format!("gate shim '{id}' not registered"),
                    "shim_id": id,
                }
            })),
        ))?;
        if shim.manifest.phase != "gate" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "wrong_phase",
                        "message": format!(
                            "shim '{}' has phase='{}', not 'gate' (gate_shims requires gate-phase shims)",
                            id, shim.manifest.phase),
                        "shim_id": id,
                    }
                })),
            ));
        }
        // v1 gate shims read the final post-norm pooled hidden — no
        // per-block intermediate capture is wired for gate dispatch.
        if shim.manifest.attachment.layer != "final" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!(
                            "v1 gate requires attachment.layer='final'; shim '{}' has '{}'",
                            id, shim.manifest.attachment.layer),
                        "shim_id": id,
                    }
                })),
            ));
        }
        out.push((id.clone(), shim));
    }
    Ok(out)
}

/// Run gate shims against an already-computed hidden capture and
/// evaluate shim_rules. Pulled out of `run_gate_shims_and_rules` so the
/// prefill can be shared with injection (#6c) — both gate and inject
/// shims read the same prompt-final hidden.
async fn run_gate_shims_against_hc(
    resolved: &[(String, Arc<RegisteredShim>)],
    hc: &HiddenCaptures,
    shim_rules: &[ShimRule],
    gate_prefill_ms: u64,
) -> Result<GateOutcome, (StatusCode, Json<serde_json::Value>)> {
    let mut decisions = HashMap::new();
    let mut ort_ms_per_shim = HashMap::new();
    for (id, shim) in resolved {
        let result = run_shim_against_hidden(shim, hc).await?;
        ort_ms_per_shim.insert(id.clone(), result.ort_ms);
        decisions.insert(id.clone(), result.decision);
    }
    let matched = evaluate_shim_rules(shim_rules, &decisions);
    Ok(GateOutcome {
        decisions,
        gate_prefill_ms,
        ort_ms_per_shim,
        matched,
    })
}

/// Resolve a list of steer shim IDs against the registry and validate
/// each is shaped for v1 steer dispatch. Errors map to HTTP responses
/// (404 / 400) suitable for the chat handler.
///
/// v1 constraints (rejected with 400 if violated):
/// - phase must be "steer"
/// - attachment.layer must be "final" (entrance:N is #6c)
/// - attachment.pooling must be "last_token" (per-token steer reads
///   exactly one hidden vector; mean / attention don't make sense
///   in the decode loop)
/// - input_shape.hidden_dim must equal embed_dim
/// - output_shape.kind must be "hidden_delta" with the same dim
async fn resolve_steer_shims(
    state: &Arc<ServerState>,
    steer_ids: &[String],
) -> Result<Vec<Arc<RegisteredShim>>, (StatusCode, Json<serde_json::Value>)> {
    if steer_ids.is_empty() {
        return Ok(Vec::new());
    }
    let embed_dim = state.engine.embed_dim();
    let shims = state.shims.lock().await;
    let mut out = Vec::with_capacity(steer_ids.len());
    for id in steer_ids {
        let shim = shims.get(id).cloned().ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "shim_not_found",
                    "message": format!("steer shim '{id}' not registered"),
                    "shim_id": id,
                }
            })),
        ))?;
        let m = &shim.manifest;
        if m.phase != "steer" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "wrong_phase",
                        "message": format!(
                            "shim '{id}' has phase='{}', not 'steer'", m.phase),
                        "shim_id": id,
                    }
                })),
            ));
        }
        if m.attachment.layer != "final" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!(
                            "v1 steer requires attachment.layer='final'; shim '{id}' has '{}'",
                            m.attachment.layer),
                        "shim_id": id,
                    }
                })),
            ));
        }
        if m.attachment.pooling != "last_token" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!(
                            "v1 steer requires attachment.pooling='last_token'; shim '{id}' has '{}'",
                            m.attachment.pooling),
                        "shim_id": id,
                    }
                })),
            ));
        }
        let in_dim = m.input_shape.get("hidden_dim")
            .and_then(|v| v.as_u64()).map(|n| n as usize)
            .ok_or_else(|| (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "invalid_manifest",
                        "message": format!("steer shim '{id}': input_shape.hidden_dim missing"),
                        "shim_id": id,
                    }
                })),
            ))?;
        if in_dim != embed_dim {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "shape_mismatch",
                        "message": format!(
                            "steer shim '{id}': input_shape.hidden_dim={in_dim} != model embed_dim={embed_dim}"),
                        "shim_id": id,
                    }
                })),
            ));
        }
        let kind = m.output_shape.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "hidden_delta" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!(
                            "v1 steer requires output_shape.kind='hidden_delta'; shim '{id}' has '{kind}'"),
                        "shim_id": id,
                    }
                })),
            ));
        }
        out.push(shim);
    }
    Ok(out)
}

/// Apply a sequence of steer shims to a single token's hidden state in
/// place. Each steer's ort session is run on the current hidden, and
/// its `hidden_delta` output is added back into hidden — the spec's
/// sequential-chain composition (instruct-then-voice ≠ voice-then-
/// instruct, so order matters and is preserved as declared).
///
/// Called from `tokio::task::block_in_place` and `spawn_blocking`
/// contexts — uses `blocking_lock()` on the per-shim mutex (works
/// because we're synchronous at the call site). All-at-once
/// validation in `resolve_steer_shims` means runtime failures here
/// would be transient (e.g. an ort internal allocation hiccup);
/// panics are acceptable in v1 because they bubble up to the
/// spawn_blocking JoinError or block_in_place caller cleanly.
/// Where an injection-phase shim's `hidden_delta` output should be
/// broadcast-added during forward. v1 supports two attachment forms.
#[derive(Debug, Clone, Copy)]
enum InjectAttachment {
    /// Add at this single layer's entrance, every forward step.
    EntranceN(usize),
    /// Add at every layer's entrance, every forward step.
    EntranceAll,
}

impl InjectAttachment {
    fn parse(layer: &str, n_layers: usize) -> Option<Self> {
        if layer == "entrance:all" {
            return Some(Self::EntranceAll);
        }
        if let Some(rest) = layer.strip_prefix("entrance:") {
            if let Ok(n) = rest.parse::<usize>() {
                if n < n_layers {
                    return Some(Self::EntranceN(n));
                }
            }
        }
        None
    }
}

/// Resolve injection shim IDs against the registry and validate v1
/// shape (phase=injection, attachment.layer parses as entrance:N or
/// entrance:all with N < n_layers, output_shape.kind=hidden_delta,
/// hidden_dim matches embed_dim). Returns each shim paired with its
/// parsed attachment so the caller can route the delta to the right
/// layer(s).
async fn resolve_inject_shims(
    state: &Arc<ServerState>,
    inject_ids: &[String],
) -> Result<Vec<(Arc<RegisteredShim>, InjectAttachment)>, (StatusCode, Json<serde_json::Value>)> {
    if inject_ids.is_empty() { return Ok(Vec::new()); }
    let n_layers = state.engine.n_layers();
    let embed_dim = state.engine.embed_dim();
    let shims = state.shims.lock().await;
    let mut out = Vec::with_capacity(inject_ids.len());
    for id in inject_ids {
        let shim = shims.get(id).cloned().ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "type": "shim_not_found",
                    "message": format!("inject shim '{id}' not registered"),
                    "shim_id": id,
                }
            })),
        ))?;
        let m = &shim.manifest;
        if m.phase != "injection" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "wrong_phase",
                        "message": format!("shim '{id}' has phase='{}', not 'injection'", m.phase),
                        "shim_id": id,
                    }
                })),
            ));
        }
        let attach = InjectAttachment::parse(&m.attachment.layer, n_layers).ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": format!(
                        "v1 injection requires attachment.layer='entrance:N' (N<{n_layers}) or 'entrance:all'; shim '{id}' has '{}'",
                        m.attachment.layer),
                    "shim_id": id,
                }
            })),
        ))?;
        let in_dim = m.input_shape.get("hidden_dim")
            .and_then(|v| v.as_u64()).map(|n| n as usize)
            .ok_or_else(|| (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "invalid_manifest",
                        "message": format!("inject shim '{id}': input_shape.hidden_dim missing"),
                        "shim_id": id,
                    }
                })),
            ))?;
        if in_dim != embed_dim {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "shape_mismatch",
                        "message": format!(
                            "inject shim '{id}': input_shape.hidden_dim={in_dim} != model embed_dim={embed_dim}"),
                        "shim_id": id,
                    }
                })),
            ));
        }
        let kind = m.output_shape.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "hidden_delta" {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "unsupported",
                        "message": format!(
                            "v1 injection requires output_shape.kind='hidden_delta'; shim '{id}' has '{kind}'"),
                        "shim_id": id,
                    }
                })),
            ));
        }
        out.push((shim, attach));
    }
    Ok(out)
}

/// Run each injection shim against the shared prompt prefill hidden,
/// sum deltas per layer (composition is sum — commutative,
/// order-independent per spec), upload one wgpu buffer per non-empty
/// layer. Returns a Vec<Option<wgpu::Buffer>> of length n_layers
/// suitable for `forward_full_gpu_with_cache_inject_returning_hidden`,
/// or empty Vec when no shims resolved.
async fn compute_inject_deltas(
    state: &Arc<ServerState>,
    resolved: &[(Arc<RegisteredShim>, InjectAttachment)],
    hc: &HiddenCaptures,
) -> Result<Vec<Option<wgpu::Buffer>>, (StatusCode, Json<serde_json::Value>)> {
    if resolved.is_empty() { return Ok(Vec::new()); }
    let n_layers = state.engine.n_layers();
    let embed_dim = state.engine.embed_dim();

    let mut sums: Vec<Option<Vec<f32>>> = vec![None; n_layers];
    for (shim, attach) in resolved {
        let result = run_shim_against_hidden(shim, hc).await?;
        if result.raw_output.len() != embed_dim {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "type": "shape_mismatch",
                        "message": format!(
                            "inject shim '{}' produced {} values, expected embed_dim={embed_dim}",
                            shim.manifest.id, result.raw_output.len()),
                    }
                })),
            ));
        }
        let target_layers: Vec<usize> = match attach {
            InjectAttachment::EntranceN(n) => vec![*n],
            InjectAttachment::EntranceAll => (0..n_layers).collect(),
        };
        for layer in target_layers {
            match sums[layer].as_mut() {
                Some(existing) => {
                    for (a, b) in existing.iter_mut().zip(result.raw_output.iter()) {
                        *a += *b;
                    }
                }
                None => sums[layer] = Some(result.raw_output.clone()),
            }
        }
    }

    let buffers: Vec<Option<wgpu::Buffer>> = sums.into_iter().enumerate().map(|(i, opt)| {
        opt.map(|v| state.engine.upload_f32_to_storage(
            &v, &format!("inject.layer{i}"),
        ))
    }).collect();
    Ok(buffers)
}

fn apply_steers_inplace(
    steers: &[Arc<RegisteredShim>],
    hidden: &mut [f32],
) {
    if steers.is_empty() { return; }
    let embed_dim = hidden.len();
    for shim in steers {
        let mut session = shim.session.blocking_lock();
        let input_name = session.inputs().first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".to_string());
        // Snapshot hidden so the input borrow ends before we mutate it
        // via the output. Avoids any aliasing surprise inside ort's
        // tensor view machinery.
        let input_snapshot: Vec<f32> = hidden.to_vec();
        let tensor = ort::value::TensorRef::from_array_view((
            vec![1_i64, embed_dim as i64],
            input_snapshot.as_slice(),
        )).expect("steer input tensor construction");
        let outputs = session.run(ort::inputs![input_name.as_str() => tensor])
            .expect("steer ort run failed");
        let first_out = outputs.iter().next().expect("steer produced no outputs").1;
        let (_shape, out_data) = first_out.try_extract_tensor::<f32>()
            .expect("steer output extraction failed");
        assert_eq!(out_data.len(), embed_dim,
            "steer output length {} != embed_dim {}", out_data.len(), embed_dim);
        for (h, &d) in hidden.iter_mut().zip(out_data.iter()) {
            *h += d;
        }
    }
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
    /// Prometheus telemetry. Recorded by chat_completions / cache_load /
    /// cache_append handlers; rendered via `GET /metrics`.
    metrics: Arc<crate::metrics::Metrics>,
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

    /// Gate shims to evaluate on the prefilled prompt's final hidden
    /// state. Each shim must be registered and its phase must be
    /// "gate" (otherwise the request is rejected before generation).
    /// Decisions are evaluated against `shim_rules`; if a rule routes
    /// to silent, generation is skipped and a `done{silent:true}`
    /// response is emitted. Per `project_cortex_v1_shim_api.md`, the
    /// gate fires once after prefill — the single prefill is what made
    /// the decision possible. Empty / absent = no gate dispatch.
    #[serde(default)]
    gate_shims: Vec<String>,

    /// Steer shims to apply during decode (per-token hidden
    /// modification). v1 records the active set in response metadata
    /// but does not yet apply it — the steer-phase forward path is
    /// scheduled for #6b. Listed here so callers (AgentOS) can pin
    /// the wire shape now and v1 servers accept the field without
    /// erroring.
    #[serde(default)]
    #[allow(dead_code)]
    steer_shims: Vec<String>,

    /// Injection shims attached to layer entrances (residual add to
    /// hidden state during forward). v1 records the requested set in
    /// response metadata but does not yet apply it — the injection
    /// hook lands in #6c. Field present for forward-compat.
    #[serde(default)]
    #[allow(dead_code)]
    inject_shims: Vec<String>,

    /// Declarative dispatch rules evaluated against gate decisions.
    /// First matching rule wins; an `else` arm (rule with no `if`)
    /// is the conventional fallthrough. See `ShimRule` for the wire
    /// shape. Empty / absent = run all gates for observability but
    /// take no action (generation proceeds normally).
    #[serde(default)]
    shim_rules: Vec<ShimRule>,
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
    /// Shim phase-dispatch metadata. Present only when gate_shims fired
    /// for this request: includes per-shim decisions, active steers, and
    /// any signals emitted by matched rules. Omitted (not `null`) when
    /// no shim work happened for this completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
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
    /// When true, build the polar-compressed cache and drop the f32 KV
    /// copy. The resulting shard is read-only: only `/v1/retrieve`
    /// works against it; chat (`cache_shards`) and `cache_append`
    /// reject it with 409 Conflict. Requires the server to be started
    /// with `--enable-polar-cache`; otherwise the request is rejected
    /// with 400. ~7x VRAM reduction per shard for retrieval-only use
    /// cases (RAG corpora, knowledge bases).
    #[serde(default)]
    polar_only: bool,
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
    let mut sampler = Sampler::new(sampler_config.clone(), seed);
    let greedy_gpu = lm_head_greedy_eligible(&sampler_config);

    let mut next_token = if greedy_gpu {
        if let Some(tok) = engine.forward_full_gpu_with_cache_inject_argmax_greedy(
            prompt_tokens, cache, &[],
        ) {
            tok
        } else {
            let prefill_logits = engine.forward_full_gpu_with_cache(prompt_tokens, cache);
            let vocab = engine.vocab_size();
            let last_logits_start = (prompt_tokens.len() - 1) * vocab;
            sampler.sample(&prefill_logits[last_logits_start..last_logits_start + vocab])
        }
    } else {
        let prefill_logits = engine.forward_full_gpu_with_cache(prompt_tokens, cache);
        let vocab = engine.vocab_size();
        let last_logits_start = (prompt_tokens.len() - 1) * vocab;
        sampler.sample(&prefill_logits[last_logits_start..last_logits_start + vocab])
    };

    let mut out = Vec::new();
    if next_token == eos {
        return out;
    }
    out.push(next_token);

    for _ in 1..max_tokens {
        next_token = if greedy_gpu {
            if let Some(tok) = engine.forward_full_gpu_with_cache_inject_argmax_greedy(
                &[next_token], cache, &[],
            ) {
                tok
            } else {
                let logits = engine.forward_full_gpu_with_cache(&[next_token], cache);
                sampler.sample(&logits)
            }
        } else {
            let logits = engine.forward_full_gpu_with_cache(&[next_token], cache);
            sampler.sample(&logits)
        };
        if next_token == eos {
            break;
        }
        out.push(next_token);
    }
    out
}

/// Greedy fast-path eligibility: the GPU LM-head + argmax shader can
/// supply the next token directly (4-byte readback) only when the
/// sampler config reduces to plain argmax — no temperature scaling,
/// no top-k/p, no repetition penalty. `CORTEX_LM_HEAD=cpu` forces the
/// legacy CPU path even for greedy requests (rollback switch).
fn lm_head_greedy_eligible(cfg: &SamplerConfig) -> bool {
    if std::env::var("CORTEX_LM_HEAD").as_deref() == Ok("cpu") {
        return false;
    }
    let greedy = cfg.temperature <= 0.0 || cfg.top_k == 1;
    let no_rep_penalty = cfg.repetition_penalty <= 1.0;
    greedy && no_rep_penalty
}

/// Stateless GPU generation with optional steer + inject support.
/// Used whenever either set is non-empty — pure plain chat (no shims)
/// stays on the cheaper `engine.generate()` CPU path.
///
/// Per token: forward (with inject deltas applied at chosen layer
/// entrances), apply steers in declared order to the last token's
/// hidden, re-project on CPU, sample. Steers modify hidden after
/// inject has shaped the forward — order matches the v1 spec's
/// inject-then-gate-then-steer pipeline.
fn generate_stateless_gpu(
    engine: &GpuEngine,
    prompt_tokens: &[u32],
    sampler_config: SamplerConfig,
    seed: u64,
    eos: u32,
    max_tokens: usize,
    max_seq_len: usize,
    steers: &[Arc<RegisteredShim>],
    inject_deltas: &[Option<wgpu::Buffer>],
) -> Vec<u32> {
    let mut cache = engine.create_gpu_kv_cache(max_seq_len);
    let mut sampler = Sampler::new(sampler_config.clone(), seed);
    let embed_dim = engine.embed_dim();
    let has_steers = !steers.is_empty();
    // Greedy GPU LM-head fast path is incompatible with steers (steers
    // mutate hidden BEFORE the projection, so we still need the full
    // hidden readback for them).
    let greedy_gpu = !has_steers && lm_head_greedy_eligible(&sampler_config);

    // Prefill: get [n_prompt * embed_dim] hidden (with inject), take
    // last token's slice, apply steers (if any), project, sample.
    let mut next_token = if has_steers {
        let mut prefill_hidden = engine.forward_full_gpu_with_cache_inject_returning_hidden(
            prompt_tokens, &mut cache, inject_deltas,
        );
        let last_off = (prompt_tokens.len() - 1) * embed_dim;
        let last_slice = &mut prefill_hidden[last_off..last_off + embed_dim];
        apply_steers_inplace(steers, last_slice);
        let last_logits = engine.cpu().finalize_logits(last_slice, 1);
        sampler.sample(&last_logits)
    } else if greedy_gpu {
        // GPU greedy: forward + LM head + argmax all on device, 4-byte
        // readback. Falls back to the CPU path if lm_head isn't resident.
        if let Some(tok) = engine.forward_full_gpu_with_cache_inject_argmax_greedy(
            prompt_tokens, &mut cache, inject_deltas,
        ) {
            tok
        } else {
            let prefill_hidden = engine.forward_full_gpu_with_cache_inject_returning_hidden(
                prompt_tokens, &mut cache, inject_deltas,
            );
            let last_off = (prompt_tokens.len() - 1) * embed_dim;
            let last_slice = &prefill_hidden[last_off..last_off + embed_dim];
            let last_logits = engine.cpu().finalize_logits(last_slice, 1);
            sampler.sample(&last_logits)
        }
    } else {
        // Read hidden, project only the LAST token's slice through the
        // LM head. The previous version called forward_full_gpu_with_cache_inject
        // which runs finalize_logits on ALL n_tokens of prefill hidden
        // — wasteful since only the last token feeds the sampler.
        let prefill_hidden = engine.forward_full_gpu_with_cache_inject_returning_hidden(
            prompt_tokens, &mut cache, inject_deltas,
        );
        let last_off = (prompt_tokens.len() - 1) * embed_dim;
        let last_slice = &prefill_hidden[last_off..last_off + embed_dim];
        let last_logits = engine.cpu().finalize_logits(last_slice, 1);
        sampler.sample(&last_logits)
    };

    let mut out: Vec<u32> = Vec::new();
    if next_token == eos {
        return out;
    }
    out.push(next_token);

    for _ in 1..max_tokens {
        next_token = if has_steers {
            let mut hidden = engine.forward_full_gpu_with_cache_inject_returning_hidden(
                &[next_token], &mut cache, inject_deltas,
            );
            apply_steers_inplace(steers, &mut hidden);
            let logits = engine.cpu().finalize_logits(&hidden, 1);
            sampler.sample(&logits)
        } else if greedy_gpu {
            if let Some(tok) = engine.forward_full_gpu_with_cache_inject_argmax_greedy(
                &[next_token], &mut cache, inject_deltas,
            ) {
                tok
            } else {
                let logits = engine.forward_full_gpu_with_cache_inject(
                    &[next_token], &mut cache, inject_deltas,
                );
                sampler.sample(&logits)
            }
        } else {
            // Decode: n_tokens=1, so forward_full_gpu_with_cache_inject's
            // internal finalize_logits is already only one token — no
            // wasted projection here. Keep the simple direct-logits path.
            let logits = engine.forward_full_gpu_with_cache_inject(
                &[next_token], &mut cache, inject_deltas,
            );
            sampler.sample(&logits)
        };
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
    // Telemetry: stamp on entry, mark success at each Ok return path.
    // Drop records duration + endpoint counter regardless. Streaming
    // responses record duration up to handoff (the SSE stream itself
    // is recorded by TTFT inside chat_completions_stream).
    let mut _telemetry = metrics::RequestTimer::new(state.metrics.clone(), metrics::Endpoint::ChatCompletions);

    let prompt_tokens = apply_chat_template(
        &req.messages,
        req.tools.as_deref(),
        &state.tokenizer,
    );
    state.metrics.record_tokens(prompt_tokens.len() as u64, 0);

    // Gate-phase dispatch (#6a). Per `project_cortex_v1_shim_api.md`, the
    // gate fires once after a prefill that *is* the work computing its
    // input. v1 constraints (rejected before any GPU work):
    //  - gate_shims requires --enable-shims
    //  - gate_shims is incompatible with cache_shards/cache_id and with
    //    mode=retrieve. Cached gating + retrieve gating are follow-ups
    //    that need a cached forward variant of the hidden-capture path.
    //  - inject_shims and steer_shims are accepted as wire fields but
    //    not applied in v1 (recorded in metadata for forward-compat).
    if !req.gate_shims.is_empty() && !state.shims_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "feature_disabled",
                    "message": "gate_shims requires --enable-shims",
                }
            })),
        ));
    }
    let gate_with_cache = !req.gate_shims.is_empty()
        && (req.cache_shards.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || req.cache_id.is_some());
    if gate_with_cache {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": "gate_shims with cache_shards / cache_id is a planned follow-up; \
                                v1 only gates against the prompt itself (stateless prefill).",
                }
            })),
        ));
    }
    if !req.gate_shims.is_empty() && req.mode.as_deref() == Some("retrieve") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": "gate_shims is not supported with mode='retrieve'.",
                }
            })),
        ));
    }

    // Inject-phase v1 limits (rejected before any GPU work):
    //  - inject_shims requires --enable-shims (same as gate)
    //  - inject_shims is incompatible with cache_shards/cache_id and
    //    with mode=retrieve (cached / retrieve injection variants
    //    are follow-ups)
    let inject_with_cache = !req.inject_shims.is_empty()
        && (req.cache_shards.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || req.cache_id.is_some());
    if inject_with_cache {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": "inject_shims with cache_shards / cache_id is a planned follow-up.",
                }
            })),
        ));
    }
    if !req.inject_shims.is_empty() && req.mode.as_deref() == Some("retrieve") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": "inject_shims is not supported with mode='retrieve'.",
                }
            })),
        ));
    }
    if !req.inject_shims.is_empty() && !state.shims_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "feature_disabled",
                    "message": "inject_shims requires --enable-shims",
                }
            })),
        ));
    }

    // Resolve gate + inject shims against the registry up front so a
    // bad shim_id 404s before we burn a prefill.
    let resolved_gate_shims = resolve_gate_shims(&state, &req.gate_shims).await?;
    let resolved_inject_shims = resolve_inject_shims(&state, &req.inject_shims).await?;

    // Shared prefill: gate and inject both read the prompt's final
    // post-norm hidden. Skip the prefill entirely when neither set
    // has shims (preserves the existing fast path for plain chat).
    let need_hc = !resolved_gate_shims.is_empty() || !resolved_inject_shims.is_empty();
    let (hc, gate_prefill_ms) = if need_hc {
        let prefill_start = Instant::now();
        let hc = tokio::task::block_in_place(|| {
            state.engine.forward_full_gpu_with_hidden_capture(&prompt_tokens, &[])
        });
        (Some(hc), prefill_start.elapsed().as_millis() as u64)
    } else {
        (None, 0)
    };

    let gate_outcome = match hc.as_ref() {
        Some(hc_ref) if !resolved_gate_shims.is_empty() => {
            run_gate_shims_against_hc(
                &resolved_gate_shims, hc_ref, &req.shim_rules, gate_prefill_ms,
            ).await?
        }
        _ => GateOutcome {
            decisions: HashMap::new(),
            gate_prefill_ms,
            ort_ms_per_shim: HashMap::new(),
            matched: None,
        },
    };

    // Compute the active steer set: a matched rule's `activate` overrides
    // the request's `steer_shims` baseline; if no rule matched we fall
    // back to the baseline. This is the v1 spec's "steer_shims is the
    // default; rules may override" semantics.
    let active_steer_ids: Vec<String> = match &gate_outcome.matched {
        Some(action) => action.activate.clone(),
        None => req.steer_shims.clone(),
    };

    // Silent short-circuit. Skip generation entirely and emit either a
    // silent SSE stream (if streaming was requested) or a silent
    // ChatResponse. Either way the metadata captures what the gate saw.
    if gate_outcome.is_silent() {
        let metadata = gate_outcome.to_metadata_with_inject(&[], &req.inject_shims);
        info!(
            gate_shims = ?req.gate_shims,
            silent = true,
            signal = ?gate_outcome.signal(),
            gate_prefill_ms = gate_outcome.gate_prefill_ms,
            "gate dispatch: silent",
        );
        if req.stream {
            _telemetry.mark_success();
            return Ok(silent_stream_response(&state.model_name, metadata));
        }
        _telemetry.mark_success();
        return Ok(Json(serde_json::to_value(ChatResponse {
            id: format!("cortex-{}", &uuid::Uuid::new_v4().to_string()[..12]),
            model: state.model_name.clone(),
            choices: vec![Choice {
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(String::new()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "silent".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_tokens.len() as u32,
                completion_tokens: 0,
            },
            metadata: Some(metadata),
        }).unwrap()).into_response());
    }

    // Resolve and validate active steers up front. v1 limit: steers are
    // not yet supported with cache_shards (the steer apply path needs a
    // returning_hidden cached forward variant — straightforward but a
    // follow-up). Reject early so the user gets a clear 400.
    let steers_with_cache = !active_steer_ids.is_empty()
        && (req.cache_shards.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || req.cache_id.is_some());
    if steers_with_cache {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": "active steers with cache_shards / cache_id is a planned follow-up; \
                                v1 only steers in stateless generation.",
                }
            })),
        ));
    }
    if !active_steer_ids.is_empty() && !state.shims_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "feature_disabled",
                    "message": "steer_shims requires --enable-shims",
                }
            })),
        ));
    }
    let active_steers = resolve_steer_shims(&state, &active_steer_ids).await?;

    // Compute injection deltas now that we've passed the silent check.
    // Built per-layer wgpu buffers ready for forward_full_gpu_with_cache_inject*.
    let inject_deltas: Vec<Option<wgpu::Buffer>> = match hc.as_ref() {
        Some(hc_ref) => compute_inject_deltas(&state, &resolved_inject_shims, hc_ref).await?,
        None => Vec::new(),
    };

    let gate_metadata: Option<serde_json::Value> = if !req.gate_shims.is_empty()
        || !active_steer_ids.is_empty()
        || !req.inject_shims.is_empty()
    {
        Some(gate_outcome.to_metadata_with_inject(&active_steer_ids, &req.inject_shims))
    } else {
        None
    };
    if !req.gate_shims.is_empty() || !req.inject_shims.is_empty() {
        info!(
            gate_shims = ?req.gate_shims,
            inject_shims = ?req.inject_shims,
            silent = false,
            active_steers = ?active_steer_ids,
            signal = ?gate_outcome.signal(),
            gate_prefill_ms = gate_outcome.gate_prefill_ms,
            "shim dispatch: proceed",
        );
    }

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
        _telemetry.mark_success();
        return chat_completions_stream(state, req, prompt_tokens, gate_metadata, active_steers, inject_deltas).await;
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
                // Polar absent → f32 cache must be present. Polar-only
                // shards are validated to require --enable-polar-cache,
                // which guarantees polar=Some, so this branch is only
                // reached for shards that still have the f32 cache.
                let cache_ref = entry.cache.as_ref()
                    .expect("shard with no polar must have f32 cache");
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

        _telemetry.mark_success();
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

        // Polar-only shards can't be chat targets — the chat path needs
        // the f32 cache that those shards dropped at load. Reject early.
        for shard_name in &shards {
            if pool.get(shard_name).map(|e| e.cache.is_none()).unwrap_or(false) {
                return Err((
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": {
                            "type": "shard_is_polar_only",
                            "message": format!(
                                "shard '{}' was loaded with polar_only=true; only /v1/retrieve is supported for these shards.",
                                shard_name,
                            ),
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
                    entry.cache.as_mut().expect("polar-only rejected above"),
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
    } else if !active_steers.is_empty() || !inject_deltas.is_empty() {
        // Stateless with active steers and/or inject: GPU generation
        // path that threads inject deltas through every forward and
        // applies steer deltas to last-token hidden each step.
        let generated = tokio::task::block_in_place(|| {
            generate_stateless_gpu(
                &state.engine, &prompt_tokens, sampler_config, seed, eos,
                max_tokens, state.max_seq_len, &active_steers, &inject_deltas,
            )
        });
        let len = generated.len() as u32;
        (generated, len)
    } else {
        // Stateless, no shims: route through the GPU batch path so bitnet
        // (and any ternary GGUF) gets the same per-token throughput as the
        // streaming wire. The old fallback called engine.generate() which
        // delegates to cpu.generate() and pays a CPU↔GPU sync per layer
        // per token — ~2 t/s on Qwen-sized bitnet vs ~18 t/s through the
        // batch shaders. generate_stateless_gpu with empty steers/inject
        // is equivalent semantically to the prior CPU path (greedy/temp=0
        // matches; sampling differs only in float-order accumulation).
        let generated = tokio::task::block_in_place(|| {
            generate_stateless_gpu(
                &state.engine, &prompt_tokens, sampler_config, seed, eos,
                max_tokens, state.max_seq_len, &[], &[],
            )
        });
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
        metadata: gate_metadata,
    };

    state.metrics.record_tokens(0, completion_len as u64);
    _telemetry.mark_success();
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
    gate_metadata: Option<serde_json::Value>,
    active_steers: Vec<Arc<RegisteredShim>>,
    inject_deltas: Vec<Option<wgpu::Buffer>>,
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

    // Telemetry: stamp start so the streaming TTFT histogram records
    // request-arrival to first content delta. (record_request for the
    // streaming response itself fires in the parent chat_completions
    // handler on handoff — see _telemetry.mark_success before this is
    // called.)
    let ttft_start = Instant::now();

    // Spawn the generation. block_in_place isn't an option from inside a
    // spawn_blocking, so we keep this in a blocking thread and use the
    // channel's blocking send path.
    let state_for_gen = state.clone();
    let steers_for_gen = active_steers;
    let inject_for_gen = inject_deltas;
    tokio::task::spawn_blocking(move || {
        let mut cache = state_for_gen.engine.create_gpu_kv_cache(state_for_gen.max_seq_len);
        let mut sampler = Sampler::new(sampler_config.clone(), seed);
        let embed_dim = state_for_gen.engine.embed_dim();
        let has_steers = !steers_for_gen.is_empty();
        // Greedy GPU LM-head fast path is incompatible with steers
        // (which mutate hidden BEFORE the projection — they still need
        // the full hidden readback).
        let greedy_gpu = !has_steers && lm_head_greedy_eligible(&sampler_config);

        // Prefill + first token. Four modes:
        //  - steers (with or without inject): forward returns hidden,
        //    apply each steer's hidden_delta to last token, re-project,
        //    sample.
        //  - no steers, greedy: GPU LM-head + argmax, 4-byte readback.
        //  - no steers, no inject: existing direct-projection fast path.
        //  - no steers, has inject: inject-aware forward returns logits
        //    directly (one projection inside), sample.
        let mut next_token = if has_steers {
            let mut hidden = state_for_gen.engine.forward_full_gpu_with_cache_inject_returning_hidden(
                &prompt_tokens, &mut cache, &inject_for_gen,
            );
            let last_off = (prompt_tokens.len() - 1) * embed_dim;
            let last_slice = &mut hidden[last_off..last_off + embed_dim];
            apply_steers_inplace(&steers_for_gen, last_slice);
            let logits = state_for_gen.engine.cpu().finalize_logits(last_slice, 1);
            sampler.sample(&logits)
        } else if greedy_gpu {
            if let Some(tok) = state_for_gen.engine.forward_full_gpu_with_cache_inject_argmax_greedy(
                &prompt_tokens, &mut cache, &inject_for_gen,
            ) {
                tok
            } else {
                let prefill_hidden = state_for_gen.engine.forward_full_gpu_with_cache_inject_returning_hidden(
                    &prompt_tokens, &mut cache, &inject_for_gen,
                );
                let last_off = (prompt_tokens.len() - 1) * embed_dim;
                let last_slice = &prefill_hidden[last_off..last_off + embed_dim];
                let last_logits = state_for_gen.engine.cpu().finalize_logits(last_slice, 1);
                sampler.sample(&last_logits)
            }
        } else {
            // Read hidden, only project the LAST token's slice through
            // the LM head. The previous version called
            // forward_full_gpu_with_cache_inject which runs
            // finalize_logits on ALL n_tokens of prefill hidden — for
            // a 1525-token prompt × 152k vocab × 2048 embed that's
            // ~475B CPU ops (~30 seconds on this hardware), and the
            // first-token sampler only ever reads the last vocab slot.
            let prefill_hidden = state_for_gen.engine.forward_full_gpu_with_cache_inject_returning_hidden(
                &prompt_tokens, &mut cache, &inject_for_gen,
            );
            let last_off = (prompt_tokens.len() - 1) * embed_dim;
            let last_slice = &prefill_hidden[last_off..last_off + embed_dim];
            let last_logits = state_for_gen.engine.cpu().finalize_logits(last_slice, 1);
            sampler.sample(&last_logits)
        };

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
            // Record TTFT now: the first sampled token is about to land in
            // the stream channel. Even if the receiver hasn't read yet,
            // this is the GPU-side definition of "first token ready".
            state_for_gen.metrics.record_ttft(ttft_start.elapsed().as_secs_f64());
            if !push_delta(&generated, &mut emitted_text, &tx) {
                return; // client gone
            }

            for _ in 1..max_tokens {
                next_token = if has_steers {
                    let mut hidden = state_for_gen.engine.forward_full_gpu_with_cache_inject_returning_hidden(
                        &[next_token], &mut cache, &inject_for_gen,
                    );
                    apply_steers_inplace(&steers_for_gen, &mut hidden);
                    let logits = state_for_gen.engine.cpu().finalize_logits(&hidden, 1);
                    sampler.sample(&logits)
                } else if greedy_gpu {
                    if let Some(tok) = state_for_gen.engine.forward_full_gpu_with_cache_inject_argmax_greedy(
                        &[next_token], &mut cache, &inject_for_gen,
                    ) {
                        tok
                    } else {
                        let logits = state_for_gen.engine.forward_full_gpu_with_cache_inject(
                            &[next_token], &mut cache, &inject_for_gen,
                        );
                        sampler.sample(&logits)
                    }
                } else {
                    let logits = state_for_gen.engine.forward_full_gpu_with_cache_inject(
                        &[next_token], &mut cache, &inject_for_gen,
                    );
                    sampler.sample(&logits)
                };
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
        state_for_gen.metrics.record_tokens(0, generated.len() as u64);
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

    // Subsequent chunks come from the generation task. The Finish chunk
    // carries the gate metadata (if any) at the chunk root so callers
    // see `gate_decisions` / `signals` alongside the OpenAI fields.
    let body_stream = ReceiverStream::new(rx).map(move |msg| {
        Ok::<_, std::convert::Infallible>(match msg {
            StreamMessage::Delta(text) => stream_chunk_event(
                &chunk_id_for_stream, created, &model_for_stream,
                Some(serde_json::json!({"content": text})),
                None,
            ),
            StreamMessage::Finish(reason) => stream_finish_event(
                &chunk_id_for_stream, created, &model_for_stream,
                reason, gate_metadata.clone(),
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

/// Build the terminal `chat.completion.chunk` SSE event with an optional
/// metadata object at the chunk root. Used for both the silent-gate
/// short-circuit (`finish_reason: "silent"`) and the normal end-of-
/// generation path when gate_shims fired (so callers see the gate
/// decisions alongside the finish reason).
fn stream_finish_event(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: String,
    metadata: Option<serde_json::Value>,
) -> Event {
    let mut payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    });
    if let Some(meta) = metadata {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("metadata".to_string(), meta);
        }
    }
    Event::default().data(payload.to_string())
}

/// Silent-gate short-circuit SSE response. Emits the role chunk for
/// OpenAI client compatibility, then a terminal chunk with
/// `finish_reason: "silent"` and the gate metadata, then the [DONE]
/// sentinel. Zero content events — silence is first-class
/// (`project_silence_as_first_class.md`).
fn silent_stream_response(
    model_name: &str,
    metadata: serde_json::Value,
) -> axum::response::Response {
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let role_event = stream_chunk_event(
        &chunk_id, created, model_name,
        Some(serde_json::json!({"role": "assistant"})),
        None,
    );
    let finish_event = stream_finish_event(
        &chunk_id, created, model_name,
        "silent".to_string(), Some(metadata),
    );
    let done_event = Event::default().data("[DONE]");
    let stream = futures::stream::iter(vec![
        Ok::<_, std::convert::Infallible>(role_event),
        Ok::<_, std::convert::Infallible>(finish_event),
        Ok::<_, std::convert::Infallible>(done_event),
    ]);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
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
    let mut _telemetry = metrics::RequestTimer::new(state.metrics.clone(), metrics::Endpoint::CacheLoad);

    // polar_only requires the server to have polar caching enabled —
    // otherwise the shard would have NO storage at all after the drop.
    if req.polar_only && !state.polar_cache_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "polar_cache_disabled",
                    "message": "polar_only=true requires the server to be started with --enable-polar-cache",
                    "cache_id": req.cache_id,
                }
            })),
        ));
    }

    // Pre-flight overflow check (Bug 1): return a structured 400 instead
    // of letting `forward_full_gpu_with_cache_returning_hidden` panic
    // inside `block_in_place`.
    if req.tokens.len() + SINK_TOKENS > state.max_seq_len {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "cache_overflow",
                    "message": format!(
                        "cache_load would overflow: {} tokens + {} sinks > {} (--max-seq-len). \
                         Increase --max-seq-len at server startup or shard the corpus.",
                        req.tokens.len(), SINK_TOKENS, state.max_seq_len),
                    "cache_id": req.cache_id,
                    "tokens_to_load": req.tokens.len(),
                    "max_seq_len": state.max_seq_len,
                }
            })),
        ));
    }

    let mut cache = state.engine.create_gpu_kv_cache(state.max_seq_len);

    // Prepend sink tokens (BOS repeated) to absorb position-0 attention
    // sink artifact. Real content starts at position SINK_TOKENS.
    let bos = state.tokenizer.bos_token_id();
    let sink_tokens: Vec<u32> = vec![bos; SINK_TOKENS];
    let mut all_tokens = sink_tokens;
    all_tokens.extend_from_slice(&req.tokens);

    if !all_tokens.is_empty() {
        // Chunk to keep BlockScratch.scores under the safe budget — same
        // wedge protection as cache_append. Uses the no-LM-head forward
        // variant since cache_load discards logits.
        let n_heads = state.engine.cpu().blocks()[0].attention().n_heads();
        let cache_id_for_log = req.cache_id.clone();
        tokio::task::block_in_place(|| {
            forward_chunked_into_cache(
                &state.engine, &all_tokens, &mut cache, n_heads,
                |chunk_idx, chunk_size, new_seq_len, chunk_ms| {
                    tracing::debug!(
                        cache_id = %cache_id_for_log,
                        chunk = chunk_idx,
                        chunk_tokens = chunk_size,
                        new_seq_len,
                        chunk_ms,
                        "cache_load chunk",
                    );
                },
            );
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
    // polar_only: drop the f32 cache now that polar is populated. The
    // GpuKvCache's wgpu::Buffers release on drop, freeing ~150 MB per
    // Qwen 3B shard at max_seq=4096. Reject at the validation step
    // above guarantees polar is Some here, so the shard still has
    // storage.
    let cache_opt = if req.polar_only {
        debug_assert!(polar.is_some(), "polar_only validated to require polar_cache_enabled");
        drop(cache);
        None
    } else {
        Some(cache)
    };
    let now = Instant::now();

    let mut pool = state.cache_pool.lock().await;
    // If overwriting an existing shard, bump from its current version so the
    // composition cache's staleness check sees the change. New shards start
    // at version 0; any subsequent insert / append bumps it monotonically.
    let next_version = pool.get(&req.cache_id).map(|e| e.version + 1).unwrap_or(0);
    pool.insert(
        req.cache_id.clone(),
        CacheEntry {
            cache: cache_opt,
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

    state.metrics.record_tokens(req.tokens.len() as u64, 0);
    _telemetry.mark_success();
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
/// Largest n_tokens to feed into a single `forward_full_gpu_with_cache`
/// call given the existing cache size. The attention scratch buffer
/// (`BlockScratch.scores`) is sized as `n_tokens * n_heads * (start_pos
/// + n_tokens) * 4` bytes. wgpu/Vulkan begins to silently hang the
/// forward at scratch sizes around 1 GB on typical hardware (4080
/// Laptop, 12 GB VRAM, Vulkan), so we budget 512 MB and chunk
/// accordingly.
///
/// Solving `n * (start_pos + n) * n_heads * 4 < BUDGET`:
///   n^2 + start_pos * n - BUDGET / (n_heads * 4) < 0
///   n < (-start_pos + sqrt(start_pos^2 + BUDGET / n_heads)) / 2
fn safe_chunk_size(start_pos: usize, n_heads: usize) -> usize {
    const SCORES_BUDGET_BYTES: usize = 512 * 1024 * 1024;
    let term = SCORES_BUDGET_BYTES / (n_heads * std::mem::size_of::<f32>());
    let sp = start_pos as f64;
    let disc = (sp * sp + term as f64).sqrt();
    let n_max = ((-sp + disc) * 0.5).floor() as usize;
    n_max.max(1)
}

/// Run `forward_full_gpu_with_cache_returning_hidden` on `tokens` in
/// safe-sized chunks against `cache`. Used by both `cache_load` and
/// `cache_append` to prevent wgpu from wedging on huge BlockScratch.scores
/// allocations at large cumulative seq_len. Caller is responsible for
/// ensuring the cache has capacity (`cache.seq_len() + tokens.len() <=
/// cache.max_seq_len()`); this fn does NOT validate.
///
/// The `progress` callback fires after each chunk with the chunk index,
/// chunk size, and new cumulative seq_len. Use it for tracing or to
/// report streaming progress to the client.
fn forward_chunked_into_cache<F>(
    engine: &GpuEngine,
    tokens: &[u32],
    cache: &mut GpuKvCache,
    n_heads: usize,
    mut progress: F,
)
where
    F: FnMut(usize, usize, usize, u64),
{
    let mut tokens_remaining = tokens;
    let mut chunk_idx = 0usize;
    while !tokens_remaining.is_empty() {
        let start = cache.seq_len();
        let chunk_size = safe_chunk_size(start, n_heads).min(tokens_remaining.len());
        let chunk = &tokens_remaining[..chunk_size];
        chunk_idx += 1;
        let t0 = Instant::now();
        // Use the no-readback forward: cache_load/cache_append discard
        // the hidden state anyway. Skipping the device.poll(Maintain::Wait)
        // round-trip drops per-call wall time substantially on small
        // chunks (where the sync round-trip otherwise dominates).
        // Correctness is preserved because wgpu's queue executes in
        // submission order, so subsequent forwards observe the K/V
        // writes via cache buffer storage.
        engine.forward_full_gpu_with_cache_advance_only(chunk, cache);
        progress(chunk_idx, chunk_size, cache.seq_len(), t0.elapsed().as_millis() as u64);
        tokens_remaining = &tokens_remaining[chunk_size..];
    }
}

async fn cache_append(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CacheAppendRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut _telemetry = metrics::RequestTimer::new(state.metrics.clone(), metrics::Endpoint::CacheAppend);

    // Verify the cache exists before kicking off any GPU work. Snapshot
    // metadata (current seq_len for chunk sizing, max_seq_len for the
    // overflow error message), then drop the lock so /health stays
    // responsive while the forward runs.
    {
        let pool = state.cache_pool.lock().await;
        match pool.get(&req.cache_id) {
            None => {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": {
                            "type": "cache_not_found",
                            "message": format!("cache_id '{}' not found", req.cache_id),
                            "cache_id": req.cache_id,
                        }
                    })),
                ));
            }
            Some(e) if e.cache.is_none() => {
                return Err((
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": {
                            "type": "shard_is_polar_only",
                            "message": format!(
                                "cache '{}' was loaded with polar_only=true; cannot append. Reload the shard with polar_only=false.",
                                req.cache_id,
                            ),
                            "cache_id": req.cache_id,
                        }
                    })),
                ));
            }
            Some(_) => {}
        }
    }

    let final_seq_len = if req.tokens.is_empty() {
        // Re-acquire briefly to read current seq_len for the response.
        let pool = state.cache_pool.lock().await;
        pool.get(&req.cache_id)
            .and_then(|e| e.cache.as_ref().map(|c| c.seq_len()))
            .unwrap_or(0)
    } else {
        // Pre-flight overflow check using a snapshot of current seq_len.
        // The actual cache may grow concurrently if other appends race
        // (no protection here in v1; cache_append is single-writer per
        // shard by convention), but this catches the common case.
        let n_heads = state.engine.cpu().blocks()[0].attention().n_heads();
        let (start_seq, max_seq) = {
            let pool = state.cache_pool.lock().await;
            let e = pool.get(&req.cache_id).unwrap();
            let c = e.cache.as_ref().expect("polar-only rejected above");
            (c.seq_len(), c.max_seq_len())
        };
        if start_seq + req.tokens.len() > max_seq {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "type": "cache_overflow",
                        "message": format!(
                            "cache '{}' would overflow: {} + {} > {} (--max-seq-len). \
                             Increase --max-seq-len at server startup or shard the corpus.",
                            req.cache_id, start_seq, req.tokens.len(), max_seq),
                        "cache_id": req.cache_id,
                        "current_seq_len": start_seq,
                        "tokens_to_append": req.tokens.len(),
                        "max_seq_len": max_seq,
                    }
                })),
            ));
        }

        // Chunk the tokens to keep BlockScratch.scores under the safe
        // budget. Each chunk runs the GPU forward, advances the cache,
        // and updates entry state. The pool lock is acquired BRIEFLY
        // around each chunk's mutation (to allow /health and other
        // handlers a chance to schedule between chunks), held DURING
        // the GPU work for that chunk only (memex's sequential append
        // pattern means no other writer is racing for this shard).
        //
        // We use forward_full_gpu_with_cache_advance_only — submits GPU
        // work, advances the cache cursor, doesn't readback or sync.
        // cache_append discards the hidden anyway. Skipping the
        // device.poll(Maintain::Wait) round-trip saves ~300ms per chunk
        // on a 4080 Laptop / Vulkan (empirical; the sync cost is
        // roughly constant regardless of token count). Correctness is
        // preserved because wgpu's queue is in-order: subsequent
        // forwards see the K/V writes via cache buffer storage.
        let mut current_start = start_seq;
        let mut tokens_remaining = &req.tokens[..];
        while !tokens_remaining.is_empty() {
            let next_start = current_start;
            let chunk_size = safe_chunk_size(next_start, n_heads).min(tokens_remaining.len());
            let chunk = &tokens_remaining[..chunk_size];

            // Hold the pool lock only across the GPU work for THIS chunk.
            // Between chunks the lock is released so /health can land.
            let mut pool = state.cache_pool.lock().await;
            let entry = pool.get_mut(&req.cache_id).ok_or_else(|| (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "type": "cache_not_found",
                        "message": format!("cache_id '{}' evicted mid-append", req.cache_id),
                    }
                })),
            ))?;
            let chunk_t0 = Instant::now();
            // Polar-only was rejected at handler entry, so cache must be Some.
            let entry_cache = entry.cache.as_mut().expect("polar-only rejected above");
            tokio::task::block_in_place(|| {
                state.engine.forward_full_gpu_with_cache_advance_only(
                    chunk, entry_cache,
                );
            });
            entry.tokens.extend_from_slice(chunk);
            entry.version += 1;
            entry.polar = None;
            entry.last_used = Instant::now();
            current_start = entry.cache.as_ref().expect("polar-only rejected above").seq_len();
            let chunk_ms = chunk_t0.elapsed().as_millis() as u64;
            drop(pool);

            tracing::debug!(
                cache_id = %req.cache_id,
                chunk_tokens = chunk_size,
                start_pos = next_start,
                new_seq_len = current_start,
                chunk_ms,
                "cache_append chunk",
            );

            tokens_remaining = &tokens_remaining[chunk_size..];
        }

        // Invalidate composition cache once at the end; any composition
        // that included this shard is now stale (version bumped per chunk).
        *state.composition.lock().await = None;

        current_start
    };

    info!(
        cache_id = %req.cache_id,
        seq_len = final_seq_len,
        tokens_appended = req.tokens.len(),
        "cache appended",
    );

    state.metrics.record_tokens(req.tokens.len() as u64, 0);
    _telemetry.mark_success();
    Ok(Json(CacheInfoResponse {
        cache_id: req.cache_id,
        seq_len: final_seq_len,
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

    // seq_len from whichever storage the shard has. Polar-only shards
    // dropped the f32 cache but retain the polar one with the same seq_len.
    let seq_len = entry.cache.as_ref().map(|c| c.seq_len())
        .or_else(|| entry.polar.as_ref().map(|p| p.seq_len()))
        .unwrap_or(0);
    Ok(Json(CacheInfoResponse {
        cache_id,
        seq_len,
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

/// Output of one shim run against an already-captured hidden state.
/// Shared by `/v1/shims/infer` (standalone) and the chat-handler gate
/// dispatch (6a) — both pool the same way, hit ort the same way, and
/// produce the same `decision` payload from the same `output_shape.kind`
/// vocabulary.
struct ShimRunResult {
    decision: serde_json::Value,
    raw_output: Vec<f32>,
    output_shape: Vec<i64>,
    ort_ms: u64,
    /// Manifest's `output_shape.kind` ("scalar" | "category:N" | "hidden_delta").
    kind: String,
}

/// Pool a `[n_tokens, embed_dim]` row-major hidden buffer to a single
/// `[embed_dim]` vector per the v1 pooling vocabulary
/// (`"last_token"` | `"mean"`). Errors map to a 400 the caller can
/// pass straight to the HTTP layer.
fn pool_layer_hidden(
    layer_data: &[f32],
    n_tokens: usize,
    embed_dim: usize,
    pooling: &str,
) -> Result<Vec<f32>, (StatusCode, Json<serde_json::Value>)> {
    debug_assert_eq!(layer_data.len(), n_tokens * embed_dim);
    match pooling {
        "last_token" => {
            let off = (n_tokens - 1) * embed_dim;
            Ok(layer_data[off..off + embed_dim].to_vec())
        }
        "mean" => {
            let mut sum = vec![0.0f32; embed_dim];
            for t in 0..n_tokens {
                let off = t * embed_dim;
                for d in 0..embed_dim {
                    sum[d] += layer_data[off + d];
                }
            }
            let n = n_tokens as f32;
            for v in sum.iter_mut() { *v /= n; }
            Ok(sum)
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": format!("v1 pooling supports last_token | mean; got '{other}'"),
                }
            })),
        )),
    }
}

/// Convenience wrapper around `pool_layer_hidden` for the common case
/// of pooling the final post-norm hidden out of a `HiddenCaptures`.
fn pool_final_hidden(
    hc: &HiddenCaptures,
    pooling: &str,
) -> Result<Vec<f32>, (StatusCode, Json<serde_json::Value>)> {
    pool_layer_hidden(&hc.final_post_norm_hidden, hc.n_tokens, hc.embed_dim, pooling)
}

/// Pool the captured hidden state per the shim's manifest, run the
/// shim's ort session, and format the output per `output_shape.kind`.
///
/// `attachment.layer` is NOT checked here — it controls where the
/// OUTPUT goes (decision payload for gate, hidden_delta for steer/
/// inject), not where the input comes from. Callers enforce
/// phase-appropriate layer constraints in their resolve_* helpers.
///
/// v1 input-side constraints (rejected with 400 if violated):
/// - `attachment.pooling` must be `"last_token"` or `"mean"` —
///   `attention` and `none` need extra plumbing beyond v1.
/// - `input_shape.hidden_dim` must equal `hc.embed_dim`.
async fn run_shim_against_hidden(
    shim: &Arc<RegisteredShim>,
    hc: &HiddenCaptures,
) -> Result<ShimRunResult, (StatusCode, Json<serde_json::Value>)> {
    let manifest = &shim.manifest;

    let pooled = pool_final_hidden(hc, &manifest.attachment.pooling)?;

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

    // Lock the per-shim session mutex (ort Sessions aren't Sync); v1
    // serializes inference per shim. Scoped so all borrows of `session`
    // (outputs + extracted tensor refs) end before we move out of it.
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

    let kind = manifest.output_shape.get("kind")
        .and_then(|v| v.as_str()).unwrap_or("scalar").to_string();
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

    Ok(ShimRunResult {
        decision,
        raw_output: out_vec,
        output_shape: out_shape_vec,
        ort_ms,
        kind,
    })
}

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

    // /v1/shims/infer is for standalone classification — only "final"
    // attachment is meaningful (the standalone caller has nowhere to
    // route entrance:N output). Reject other layers up front.
    if shim.manifest.attachment.layer != "final" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "unsupported",
                    "message": format!(
                        "/v1/shims/infer supports only attachment.layer='final'; got '{}'",
                        shim.manifest.attachment.layer),
                }
            })),
        ));
    }

    let tokens = state.tokenizer.encode(&req.context, /*add_bos*/ true);
    if tokens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "type": "invalid_request", "message": "empty context after tokenization" }
            })),
        ));
    }

    let infer_start = Instant::now();
    let hc = tokio::task::block_in_place(|| {
        state.engine.forward_full_gpu_with_hidden_capture(&tokens, &[])
    });
    let cortex_ms = infer_start.elapsed().as_millis() as u64;

    let result = run_shim_against_hidden(&shim, &hc).await?;

    let response = serde_json::json!({
        "decision": result.decision,
        "metadata": {
            "shim_id": req.shim_id,
            "shim_version": shim.manifest.version,
            "kind": result.kind,
            "output_shape": result.output_shape,
            "context_tokens": tokens.len(),
            "cortex_ms": cortex_ms,
            "ort_ms": result.ort_ms,
            "raw_output": result.raw_output,
        }
    });

    info!(
        shim_id = %req.shim_id,
        kind = %result.kind,
        context_tokens = tokens.len(),
        cortex_ms,
        ort_ms = result.ort_ms,
        "shim infer",
    );

    Ok(Json(response))
}

/// Which hidden state `/v1/shims/embed` returns. The vocabulary
/// mirrors the shim manifest's `attachment.layer` field so a training
/// pipeline can request the exact embedding the manifest will receive
/// at inference time.
#[derive(Debug, Clone, Copy)]
enum EmbedLayer {
    /// Final post-norm hidden (LM head input). What gate / steer / inject
    /// shims read in v1 — the canonical embedding for shim training.
    Final,
    /// Hidden at entrance to block N (== output of block N-1). Carries
    /// the index `N-1` so the caller can directly select
    /// `per_layer_hidden[idx]` from the capture result.
    EntranceN(usize),
}

impl EmbedLayer {
    /// Parse `"final"` or `"entrance:N"` (with `1 <= N <= n_layers`).
    /// `entrance:0` is rejected — that's the embedding-lookup output
    /// which is not captured by `forward_full_gpu_with_hidden_capture`
    /// in v1 (would need a new capture point).
    fn parse(s: &str, n_layers: usize) -> Option<Self> {
        if s == "final" { return Some(Self::Final); }
        if let Some(rest) = s.strip_prefix("entrance:") {
            if let Ok(n) = rest.parse::<usize>() {
                if n >= 1 && n <= n_layers {
                    return Some(Self::EntranceN(n - 1));
                }
            }
        }
        None
    }
}

/// `POST /v1/shims/embed` — return a pooled hidden-state embedding
/// for arbitrary text. Used by AgentOS to train shim classifiers
/// against the cortex substrate (the alternative was stubbing with
/// MiniLM, which would have produced throwaway training data once
/// trained shims started reading actual cortex hidden states at
/// inference time).
#[derive(Debug, Deserialize)]
struct ShimEmbedRequest {
    text: String,
    /// `"final"` (default) or `"entrance:N"` for `N` in `1..=n_layers`.
    /// Field name + vocabulary intentionally match the shim manifest's
    /// `attachment.layer` so training-time and inference-time
    /// signatures are identical.
    #[serde(default = "default_embed_layer")]
    layer: String,
    /// `"last_token"` (default) or `"mean"`. Same vocabulary as the
    /// manifest's `attachment.pooling`.
    #[serde(default = "default_embed_pooling")]
    pooling: String,
}

fn default_embed_layer() -> String { "final".to_string() }
fn default_embed_pooling() -> String { "last_token".to_string() }

async fn shim_embed(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ShimEmbedRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let n_layers = state.engine.n_layers();

    // Parse + validate the layer specifier.
    let layer = EmbedLayer::parse(&req.layer, n_layers).ok_or_else(|| (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "type": "unsupported",
                "message": format!(
                    "v1 layer must be 'final' or 'entrance:N' for 1<=N<={n_layers}; got '{}'",
                    req.layer),
            }
        })),
    ))?;

    if req.text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "type": "invalid_request", "message": "text must be non-empty" }
            })),
        ));
    }
    let tokens = state.tokenizer.encode(&req.text, /*add_bos*/ true);
    if tokens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": { "type": "invalid_request", "message": "empty text after tokenization" }
            })),
        ));
    }

    // Build capture_layers: empty for Final (only need final post-norm),
    // single-element for EntranceN(idx).
    let capture_layers: Vec<usize> = match layer {
        EmbedLayer::Final => Vec::new(),
        EmbedLayer::EntranceN(idx) => vec![idx],
    };

    let infer_start = Instant::now();
    let hc = tokio::task::block_in_place(|| {
        state.engine.forward_full_gpu_with_hidden_capture(&tokens, &capture_layers)
    });
    let cortex_ms = infer_start.elapsed().as_millis() as u64;

    let embedding = match layer {
        EmbedLayer::Final => pool_final_hidden(&hc, &req.pooling)?,
        EmbedLayer::EntranceN(_) => {
            // capture_layers had one element; per_layer_hidden[0] holds it.
            pool_layer_hidden(
                &hc.per_layer_hidden[0], hc.n_tokens, hc.embed_dim, &req.pooling,
            )?
        }
    };

    let response = serde_json::json!({
        "embedding": embedding,
        "metadata": {
            "model": state.model_name,
            "layer": req.layer,
            "pooling": req.pooling,
            "n_tokens": hc.n_tokens,
            "embed_dim": hc.embed_dim,
            "cortex_ms": cortex_ms,
        }
    });

    info!(
        layer = %req.layer,
        pooling = %req.pooling,
        n_tokens = hc.n_tokens,
        cortex_ms,
        "shim embed",
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
        metrics: Arc::new(metrics::Metrics::new(
            model_name.clone(),
            env!("CARGO_PKG_VERSION").to_string(),
        )),
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
