//! /v1/chat/completions — generation + retrieval + streaming (split from main.rs, Phase N).
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

/// Simple non-crypto RNG seed from system time.
pub(crate) fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
use crate::api::*;
use crate::shims::*;
use crate::cache::*;

/// Apply ChatML-style template (works for Qwen, many HF models).
///
/// ```text
/// <|im_start|>system\n{content}<|im_end|>\n
/// <|im_start|>user\n{content}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
pub(crate) fn apply_chat_template(
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
pub(crate) fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
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
pub(crate) fn extract_json_object(s: &str) -> Option<String> {
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

/// Generate tokens with an existing KV cache. Prefills the prompt tokens
/// into the cache, then samples autoregressively up to max_tokens.
pub(crate) fn generate_with_cache(
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

/// Polar chat generation: greedy-only prefill + decode loop against a
/// `GpuPolarKvCache`. Returns `None` if non-greedy (caller falls back
/// to the f32 path) or if the engine's LM-head isn't GPU-resident.
/// `polar_cache.set_len(...)` advances inside the engine on each
/// call, so successive token generations see the growing prefix.
pub(crate) fn generate_with_polar_cache(
    engine: &GpuEngine,
    prompt_tokens: &[u32],
    polar_cache: &mut cortex::layers::gpu_polar_kv_cache::GpuPolarKvCache,
    sampler_config: SamplerConfig,
    _seed: u64,
    eos: u32,
    max_tokens: usize,
) -> Option<Vec<u32>> {
    if !lm_head_greedy_eligible(&sampler_config) {
        return None;
    }
    let mut next_token = engine.forward_full_gpu_polar_with_cache_inject_argmax_greedy(
        prompt_tokens, polar_cache, &[],
    )?;

    let mut out: Vec<u32> = Vec::new();
    if next_token == eos {
        return Some(out);
    }
    out.push(next_token);

    for _ in 1..max_tokens {
        next_token = engine.forward_full_gpu_polar_with_cache_inject_argmax_greedy(
            &[next_token], polar_cache, &[],
        )?;
        if next_token == eos {
            break;
        }
        out.push(next_token);
    }
    Some(out)
}

/// Greedy fast-path eligibility: the GPU LM-head + argmax shader can
/// supply the next token directly (4-byte readback) only when the
/// sampler config reduces to plain argmax — no temperature scaling,
/// no top-k/p, no repetition penalty. `CORTEX_LM_HEAD=cpu` forces the
/// legacy CPU path even for greedy requests (rollback switch).
pub(crate) fn lm_head_greedy_eligible(cfg: &SamplerConfig) -> bool {
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
pub(crate) fn generate_stateless_gpu(
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

pub(crate) async fn chat_completions(
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
                // Allocator-state probe around the polar retrieve, opt-in via
                // CORTEX_POLAR_TRACE_DIAG=1. Pairs with the per-submit
                // tracing inside forward_full_gpu_polar_traced. Use to
                // localize the 2200-token retrieve device-lost ceiling:
                // if total_allocated_mb climbs sharply across retrieves,
                // it's cumulative; if a specific top-N alloc balloons at
                // big seq_len, that's the per-call buffer to attack.
                let diag = std::env::var("CORTEX_POLAR_TRACE_DIAG").as_deref() == Ok("1");
                if diag {
                    state.engine.log_allocator_report("before_polar_retrieve");
                    state.engine.log_vram_heap_stats("before_polar_retrieve");
                }
                let (q, b) = tokio::task::block_in_place(|| {
                    let q = state.engine.forward_full_gpu_polar_traced(
                        &prompt_tokens, polar_ref, &capture_layers,
                    );
                    let b = state.engine.forward_full_gpu_polar_traced(
                        &baseline_tokens, polar_ref, &capture_layers,
                    );
                    (q, b)
                });
                if diag {
                    state.engine.log_allocator_report("after_polar_retrieve");
                    state.engine.log_vram_heap_stats("after_polar_retrieve");
                }
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

        // Closure: compute per-corpus-position score from a captured
        // per-layer attention tensor (layout [n_q, n_heads, attn_max]).
        // Aggregation strategy is env-var gated for retrieval-quality A/B:
        //
        //   CORTEX_RETRIEVE_AGG=last  (default) — MAX over (layers, heads)
        //       at the LAST query position. Original behavior; keeps query
        //       and baseline aggregating over the same count of values.
        //   CORTEX_RETRIEVE_AGG=all   — MAX over (layers, heads, all query
        //       positions). Tests whether earlier query tokens (the actual
        //       content nouns/keywords) attend to more retrieval-relevant
        //       corpus positions than the final assistant-marker token does.
        //   CORTEX_RETRIEVE_AGG=mean  — MEAN over (layers, heads) at the
        //       LAST query position. Tests whether MAX is being dominated
        //       by outlier sink-attending heads.
        //   CORTEX_RETRIEVE_AGG=all_mean — MEAN over (layers, heads, all
        //       query positions). Both fixes combined.
        let agg_mode: String = std::env::var("CORTEX_RETRIEVE_AGG")
            .unwrap_or_else(|_| "last".to_string());
        let agg_all_q = agg_mode == "all" || agg_mode == "all_mean";
        let agg_mean = agg_mode == "mean" || agg_mode == "all_mean";
        // Phase P: optional (layer, head) mask for the aggregation —
        // CORTEX_RETRIEVE_HEADS="35:7,34:2,..." (absolute layer indices
        // from `layers_used`). When set, only listed pairs contribute to
        // BOTH query and baseline aggregation. Retrieval-heads
        // experiment knob (Wu et al. 2404.15574): aggregate attention
        // mass is mostly plumbing; recall lives in a sparse head subset.
        let head_mask: Option<std::collections::HashSet<(usize, usize)>> =
            std::env::var("CORTEX_RETRIEVE_HEADS").ok().map(|spec| {
                spec.split(',')
                    .filter_map(|pair| {
                        let (l, h) = pair.trim().split_once(':')?;
                        Some((l.trim().parse().ok()?, h.trim().parse().ok()?))
                    })
                    .collect()
            });
        let capture_layers_ref = &capture_layers;
        let head_mask_ref = &head_mask;
        let aggregate_score = move |per_layer: &[Vec<f32>], n_q: usize, attn_max: usize| -> Vec<f32> {
            let q_range_start = if agg_all_q { 0 } else { n_q - 1 };
            let q_range_end = n_q;
            let mut out = vec![f32::NEG_INFINITY; corpus_len];
            for k in 0..corpus_len {
                let mut accum = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for q in q_range_start..q_range_end {
                    for (li, layer_scores) in per_layer.iter().enumerate() {
                        for h in 0..n_heads {
                            if let Some(mask) = head_mask_ref {
                                if !mask.contains(&(capture_layers_ref[li], h)) {
                                    continue;
                                }
                            }
                            let idx = q * n_heads * attn_max + h * attn_max + k;
                            let v = layer_scores[idx];
                            if v.is_finite() {
                                if v > accum { accum = v; }
                                sum += v;
                                count += 1;
                            }
                        }
                    }
                }
                let agg = if agg_mean {
                    if count > 0 { sum / (count as f32) } else { f32::NEG_INFINITY }
                } else {
                    accum
                };
                if agg.is_finite() { out[k] = agg; }
            }
            out
        };
        let aggregate_max = aggregate_score;

        // Phase P: head-resolved dump for the per-head retrieval sweep.
        // CORTEX_RETRIEVE_HEAD_DUMP=<dir> writes one JSON per retrieve
        // request with the RAW last-query-row scores per (layer, head,
        // corpus position) for query AND baseline — the offline script
        // (pinky/retrieval-heads/sweep.py) recomputes any aggregation /
        // differential variant from these without further server runs.
        if let Ok(dump_dir) = std::env::var("CORTEX_RETRIEVE_HEAD_DUMP") {
            static DUMP_SEQ: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let seq = DUMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let last_row = |per_layer: &[Vec<f32>], n_q: usize, attn_max: usize| -> Vec<Vec<Vec<f32>>> {
                per_layer.iter().map(|layer_scores| {
                    (0..n_heads).map(|h| {
                        let base = (n_q - 1) * n_heads * attn_max + h * attn_max;
                        layer_scores[base..base + corpus_len].to_vec()
                    }).collect()
                }).collect()
            };
            let dump = serde_json::json!({
                "n_heads": n_heads,
                "corpus_len": corpus_len,
                "layers": capture_layers,
                "sink_tokens": SINK_TOKENS,
                "shards": shard_map.entries,
                "query": last_row(&per_layer_scores, n_query, attn_max_seq),
                "baseline": last_row(&baseline_per_layer, baseline_tokens.len(), baseline_attn_max),
            });
            let path = format!("{dump_dir}/headdump-{seq:04}.json");
            if let Err(e) = std::fs::write(&path, dump.to_string()) {
                tracing::warn!(path, error = %e, "head dump write failed");
            }
        }

        // Differential score: query attention - baseline attention. Positions
        // that are "always hot" (high in both) drop to zero; positions that
        // are query-specific stay high.
        //
        // Phase P: CORTEX_RETRIEVE_AGG=perhead computes the differential
        // PER (layer, head) and THEN takes the max — each head calibrated
        // against its own baseline before combining. The per-head sweep
        // showed this is the form under which sparse retrieval heads
        // dominate (single head L35:H13 R@10 0.50 vs 0.12 for
        // diff-of-maxes over all heads); diff-of-maxes lets plumbing
        // heads (sinks, syntax) saturate both maxes and cancel the
        // signal. Respects CORTEX_RETRIEVE_HEADS. Uses the last query /
        // baseline row (the "last" q-convention).
        let mut scores = vec![f32::NEG_INFINITY; corpus_len];
        if agg_mode == "perhead" {
            let qn = n_query;
            let bn = baseline_tokens.len();
            for k in 0..corpus_len {
                let mut best = f32::NEG_INFINITY;
                for (li, (q_layer, b_layer)) in per_layer_scores.iter()
                    .zip(baseline_per_layer.iter()).enumerate()
                {
                    for h in 0..n_heads {
                        if let Some(mask) = &head_mask {
                            if !mask.contains(&(capture_layers[li], h)) {
                                continue;
                            }
                        }
                        let qv = q_layer[(qn - 1) * n_heads * attn_max_seq + h * attn_max_seq + k];
                        let bv = b_layer[(bn - 1) * n_heads * baseline_attn_max + h * baseline_attn_max + k];
                        if qv.is_finite() && bv.is_finite() {
                            let d = qv - bv;
                            if d > best { best = d; }
                        }
                    }
                }
                if best.is_finite() { scores[k] = best; }
            }
        } else {
            let query_max = aggregate_max(&per_layer_scores, n_query, attn_max_seq);
            let baseline_max = aggregate_max(&baseline_per_layer, baseline_tokens.len(), baseline_attn_max);
            for k in 0..corpus_len {
                if query_max[k].is_finite() && baseline_max[k].is_finite() {
                    scores[k] = query_max[k] - baseline_max[k];
                }
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

        // Flush wgpu's deferred-destroy queue: each traced forward drops
        // per-layer capture buffers + stagings at exit, and repeated
        // retrieves accumulate them until wgpu-29 panics with a delayed
        // Validation Error at a later Device::poll (observed at ~8-10
        // load/retrieve/delete cycles). Mirrors the cache_delete flush.
        tokio::task::block_in_place(|| state.engine.poll_wait());

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

        // Polar-only shards can't be chat targets via the f32 path.
        // BUT: polar_only + polar_chat together is allowed — chat goes
        // via the polar orchestrator which doesn't need entry.cache.
        // Only reject when neither polar_chat nor cache is available.
        for shard_name in &shards {
            if let Some(e) = pool.get(shard_name) {
                if e.cache.is_none() && !e.polar_chat {
                    return Err((
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({
                            "error": {
                                "type": "shard_is_polar_only",
                                "message": format!(
                                    "shard '{}' was loaded with polar_only=true and polar_chat=false; only /v1/retrieve is supported.",
                                    shard_name,
                                ),
                                "cache_id": shard_name,
                            }
                        })),
                    ));
                }
            }
        }

        if shards.len() == 1 {
            // Single shard: use the existing cache directly (fast path,
            // no copying or replaying). This is the common case.
            let entry = pool.get_mut(&shards[0]).unwrap();
            // Polar-chat fast path: if the shard was loaded with
            // polar_chat=true AND the request is greedy, route through
            // the polar orchestrator. New K/V land directly in the
            // polar cache; f32 cache stays unchanged (Phase 2 keeps
            // it for non-greedy fallback only).
            let polar_generated: Option<Vec<u32>> = if entry.polar_chat {
                entry.polar.as_mut().and_then(|polar| {
                    tokio::task::block_in_place(|| {
                        generate_with_polar_cache(
                            &state.engine,
                            &prompt_tokens,
                            polar,
                            sampler_config.clone(),
                            seed,
                            eos,
                            max_tokens,
                        )
                    })
                })
            } else {
                None
            };
            let generated = if let Some(g) = polar_generated {
                g
            } else if entry.cache.is_some() {
                // Either not polar_chat, or non-greedy with f32 fallback
                // available: use the f32 path. (For polar_chat + non-greedy
                // this diverges semantics across turns — see plan Phase 2
                // notes.)
                tokio::task::block_in_place(|| {
                    generate_with_cache(
                        &state.engine,
                        &prompt_tokens,
                        entry.cache.as_mut().expect("checked is_some above"),
                        sampler_config,
                        seed,
                        eos,
                        max_tokens,
                    )
                })
            } else {
                // polar_chat + polar_only + non-greedy: no path exists.
                // Polar orchestrator only supports greedy today; f32
                // fallback unavailable since polar_only dropped it.
                return Err((
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": {
                            "type": "polar_chat_non_greedy_unsupported",
                            "message": format!(
                                "shard '{}' was loaded with polar_chat=true + polar_only=true. Non-greedy sampling against it isn't supported yet (the f32 fallback was dropped, and the polar orchestrator is greedy-only). Use temperature=0 / top_k=1, or reload with polar_only=false to keep the f32 fallback.",
                                shards[0],
                            ),
                            "cache_id": shards[0],
                        }
                    })),
                ));
            };
            entry.tokens.extend_from_slice(&prompt_tokens);
            entry.tokens.extend_from_slice(&generated);
            entry.version += 1;
            // Chat extends the f32 cache with prompt + generated K/V
            // (the f32 path) OR the polar cache (the polar_chat path).
            // For non-polar-chat shards, any polar snapshot is now
            // stale; clear it. For polar_chat shards, polar IS the
            // canonical state — keep it.
            if !entry.polar_chat {
                entry.polar = None;
            }
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
pub(crate) async fn chat_completions_stream(
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
pub(crate) enum StreamMessage {
    /// Newly-detokenized text since the last chunk.
    Delta(String),
    /// Generation finished; emit the final chunk with this finish_reason
    /// ("stop" or "length"). Always the last message before the channel closes.
    Finish(String),
}

/// Build one `chat.completion.chunk` SSE event with the given delta and
/// optional finish_reason. Mirrors OpenAI's wire shape exactly.
pub(crate) fn stream_chunk_event(
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
pub(crate) fn stream_finish_event(
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
pub(crate) fn silent_stream_response(
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

