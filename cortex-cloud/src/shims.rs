//! ONNX shim subsystem: types, gate/steer/inject dispatch, registry handlers (split from main.rs, Phase N).
#![allow(unused_imports)]

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

use crate::metrics;
use crate::state::*;

pub(crate) fn default_embed_layer() -> String { "final".to_string() }
pub(crate) fn default_embed_pooling() -> String { "last_token".to_string() }
use crate::api::*;
use crate::chat::*;
use crate::cache::*;

/// Shim manifest as defined in `project_cortex_v1_shim_api.md`. Wire-
/// compatible with what AgentOS's shim-management control plane pushes
/// via PUT /v1/shims/{id}.
///
/// `input_shape` and `output_shape` are kept as `serde_json::Value` so
/// the schema can grow without breaking older clients — the v1 shapes
/// (`{"hidden_dim": N}` for input, `{"kind": "scalar"|"category:N"|"hidden_delta"}`
/// for output) are recognized at infer time, not at registration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ShimManifest {
    pub(crate) id: String,
    pub(crate) version: String,
    /// "injection" | "gate" | "steer"
    pub(crate) phase: String,
    pub(crate) attachment: ShimAttachment,
    pub(crate) input_shape: serde_json::Value,
    pub(crate) output_shape: serde_json::Value,
    #[serde(default)]
    pub(crate) description: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ShimAttachment {
    /// "final" | "entrance:N" | "entrance:all"
    pub(crate) layer: String,
    /// "last_token" | "mean" | "attention" | "none"
    pub(crate) pooling: String,
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
pub(crate) struct ShimRule {
    #[serde(rename = "if", default)]
    pub(crate) cond: Option<RuleCondition>,
    #[serde(alias = "else", default)]
    pub(crate) then: RuleAction,
}

/// One gate's value tested against a single comparison op. Exactly one
/// of `gt` / `lt` / `eq` should be set; if multiple are set, the
/// condition matches only when ALL set ops are satisfied (AND). If
/// none are set, the condition never matches — that's a request bug,
/// not a fallthrough.
#[derive(Debug, Deserialize, Clone)]
pub(crate) struct RuleCondition {
    /// shim_id whose decision is being tested
    pub(crate) gate: String,
    #[serde(default)]
    pub(crate) gt: Option<f64>,
    #[serde(default)]
    pub(crate) lt: Option<f64>,
    #[serde(default)]
    pub(crate) eq: Option<f64>,
}

impl RuleCondition {
    /// Match the gate's decision JSON against this condition. Decisions
    /// must be coercible to f64 (scalar shims emit numbers; category
    /// shims emit integer argmax). Anything else (hidden_delta arrays,
    /// nulls) never matches.
    pub(crate) fn matches(&self, decision: &serde_json::Value) -> bool {
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
pub(crate) struct RuleAction {
    /// Mark this completion as silent — emit a `done` event with
    /// `finish_reason: "silent"` and zero generated content.
    #[serde(default)]
    pub(crate) silent: bool,
    /// Steers to activate for the decode path. v1 records these in
    /// response metadata; actual per-token application lands in 6b.
    #[serde(default)]
    pub(crate) activate: Vec<String>,
    /// Free-form signal string surfaced in `done.metadata.signals`.
    /// AgentOS uses this for downstream routing (e.g., "escalate").
    #[serde(default)]
    pub(crate) signal: Option<String>,
}

/// Evaluate the shim_rules table top-to-bottom against the gate
/// decisions. First matching rule wins; an `else` rule (no `if`)
/// fires unconditionally when reached, so it should appear last.
/// Returns `None` when no rule matched (or rules is empty) — the
/// caller distinguishes "no rule routed" from "rule explicitly chose
/// the default action" so it can fall back to `req.steer_shims` in
/// the former case (per the v1 spec: steer_shims is the default,
/// rules override).
pub(crate) fn evaluate_shim_rules(
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
pub(crate) struct GateOutcome {
    /// shim_id → JSON decision. Surfaced in response metadata under
    /// `gate_decisions` for observability.
    pub(crate) decisions: HashMap<String, serde_json::Value>,
    /// One shared cortex prefill cost across all gate shims.
    pub(crate) gate_prefill_ms: u64,
    /// shim_id → ort run time in ms (per-shim variable cost).
    pub(crate) ort_ms_per_shim: HashMap<String, u64>,
    /// `Some(action)` when a rule matched and dictates dispatch;
    /// `None` when no rule matched (rules empty, or no condition fired).
    /// In the `None` case, callers fall back to `req.steer_shims` for
    /// the active-steer set and treat silent as false / signal as None.
    pub(crate) matched: Option<RuleAction>,
}

impl GateOutcome {
    /// True if a matched rule routed to silent. Default-action when no
    /// rule matched is "proceed normally" — silent must be explicit.
    pub(crate) fn is_silent(&self) -> bool {
        self.matched.as_ref().map(|a| a.silent).unwrap_or(false)
    }

    /// Signal string from the matched rule, if any.
    pub(crate) fn signal(&self) -> Option<&String> {
        self.matched.as_ref().and_then(|a| a.signal.as_ref())
    }

    /// Build the response-metadata JSON value emitted on `done`.
    /// `active_steers` and `active_inject` are filled by the caller
    /// so metadata reflects what *actually* shaped generation.
    pub(crate) fn to_metadata_with_inject(
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
pub(crate) async fn resolve_gate_shims(
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
pub(crate) async fn run_gate_shims_against_hc(
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
pub(crate) async fn resolve_steer_shims(
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
pub(crate) enum InjectAttachment {
    /// Add at this single layer's entrance, every forward step.
    EntranceN(usize),
    /// Add at every layer's entrance, every forward step.
    EntranceAll,
}

impl InjectAttachment {
    pub(crate) fn parse(layer: &str, n_layers: usize) -> Option<Self> {
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
pub(crate) async fn resolve_inject_shims(
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
pub(crate) async fn compute_inject_deltas(
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

pub(crate) fn apply_steers_inplace(
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
pub(crate) struct RegisteredShim {
    pub(crate) manifest: ShimManifest,
    pub(crate) session: Mutex<ort::session::Session>,
}

/// PUT body shape: JSON envelope with the manifest and the ONNX bytes
/// as base64. Multipart-free so AgentOS's HTTP client can use the same
/// JSON pipeline as every other endpoint.
#[derive(Deserialize)]
pub(crate) struct ShimPutRequest {
    pub(crate) manifest: ShimManifest,
    /// Base64-encoded ONNX model bytes.
    pub(crate) onnx_base64: String,
}

#[derive(Serialize)]
pub(crate) struct ShimRegistryEntry {
    pub(crate) manifest: ShimManifest,
}

#[derive(Serialize)]
pub(crate) struct ShimsListResponse {
    pub(crate) shims: Vec<ShimManifest>,
}

/// PUT /v1/shims/{id} — register a shim. Body decodes the ONNX bytes,
/// loads them via ort, and stores the (manifest, session) pair in the
/// registry. If `id` already exists, replaces it.
///
/// 400 on base64 decode failure or ONNX load failure (the bytes were
/// junk or not a well-formed ONNX model). 400 on id mismatch between
/// URL path and manifest body — refuse to register an ambiguous shim.
pub(crate) async fn shim_put(
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
pub(crate) async fn shim_get(
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
pub(crate) async fn shims_list(
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
pub(crate) struct ShimInferRequest {
    pub(crate) shim_id: String,
    pub(crate) context: String,
}

/// Output of one shim run against an already-captured hidden state.
/// Shared by `/v1/shims/infer` (standalone) and the chat-handler gate
/// dispatch (6a) — both pool the same way, hit ort the same way, and
/// produce the same `decision` payload from the same `output_shape.kind`
/// vocabulary.
pub(crate) struct ShimRunResult {
    pub(crate) decision: serde_json::Value,
    pub(crate) raw_output: Vec<f32>,
    pub(crate) output_shape: Vec<i64>,
    pub(crate) ort_ms: u64,
    /// Manifest's `output_shape.kind` ("scalar" | "category:N" | "hidden_delta").
    pub(crate) kind: String,
}

/// Pool a `[n_tokens, embed_dim]` row-major hidden buffer to a single
/// `[embed_dim]` vector per the v1 pooling vocabulary
/// (`"last_token"` | `"mean"`). Errors map to a 400 the caller can
/// pass straight to the HTTP layer.
pub(crate) fn pool_layer_hidden(
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
pub(crate) fn pool_final_hidden(
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
pub(crate) async fn run_shim_against_hidden(
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

pub(crate) async fn shim_infer(
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
pub(crate) enum EmbedLayer {
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
    pub(crate) fn parse(s: &str, n_layers: usize) -> Option<Self> {
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
pub(crate) struct ShimEmbedRequest {
    pub(crate) text: String,
    /// `"final"` (default) or `"entrance:N"` for `N` in `1..=n_layers`.
    /// Field name + vocabulary intentionally match the shim manifest's
    /// `attachment.layer` so training-time and inference-time
    /// signatures are identical.
    #[serde(default = "default_embed_layer")]
    pub(crate) layer: String,
    /// `"last_token"` (default) or `"mean"`. Same vocabulary as the
    /// manifest's `attachment.pooling`.
    #[serde(default = "default_embed_pooling")]
    pub(crate) pooling: String,
}

pub(crate) async fn shim_embed(
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
pub(crate) async fn shim_delete(
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

