//! /v1/cache/* handlers + chunked prefill (split from main.rs, Phase N).
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

/// Run `forward_full_gpu_with_cache_returning_hidden` on `tokens` in
/// safe-sized chunks against `cache`. Used by both `cache_load` and
/// `cache_append` to prevent wgpu from wedging on huge BlockScratch.scores
/// allocations at large cumulative seq_len. Caller is responsible for
/// ensuring the cache has capacity (`cache.seq_len() + tokens.len() <=
/// cache.max_seq_len()`); this fn does NOT validate.
///
/// Chunk sizing comes from [`GpuEngine::safe_prefill_chunk_size`], which
/// reads the real Lane B (`transient_heap_b`) capacity so each chunk's
/// `scores` (+ `normed` + `activated`) fits — auto-adapting to whatever
/// the heap is sized to, on any GPU.
///
/// The `progress` callback fires after each chunk with the chunk index,
/// chunk size, and new cumulative seq_len. Use it for tracing or to
/// report streaming progress to the client.
pub(crate) fn forward_chunked_into_cache<F>(
    engine: &GpuEngine,
    tokens: &[u32],
    cache: &mut GpuKvCache,
    mut progress: F,
)
where
    F: FnMut(usize, usize, usize, u64),
{
    let mut tokens_remaining = tokens;
    let mut chunk_idx = 0usize;
    while !tokens_remaining.is_empty() {
        let start = cache.seq_len();
        let chunk_size = engine.safe_prefill_chunk_size(start).min(tokens_remaining.len());
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
use crate::api::*;
use crate::shims::*;
use crate::chat::*;

/// POST /v1/cache/load — create or replace a cache by replaying tokens.
///
/// If tokens is empty, creates an empty cache (cold start for a new user).
/// If tokens is non-empty, runs forward_cached to build the KV cache from
/// the token history (reawaken after eviction).
///
/// "load replaces" — if the cache_id already exists, it's overwritten.
pub(crate) async fn cache_load(
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
    // polar_chat needs the polar cache to actually exist for the chat
    // path to route through it.
    if req.polar_chat && !state.polar_cache_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "polar_cache_disabled",
                    "message": "polar_chat=true requires the server to be started with --enable-polar-cache",
                    "cache_id": req.cache_id,
                }
            })),
        ));
    }
    // qjl only meaningful when polar is enabled.
    if req.qjl && !state.polar_cache_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "type": "polar_cache_disabled",
                    "message": "qjl=true requires the server to be started with --enable-polar-cache",
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

    // Review #11: token ids must be < vocab, or the engine asserts inside
    // block_in_place AFTER a full-size cache was reserved.
    crate::chat::check_token_ids(&req.tokens, state.engine.vocab_size())?;

    // Review #6: soft pool cap — cheap early reject before the expensive
    // allocation + prefill. Replacing an existing id is always allowed.
    // (The authoritative, race-safe check is under the insert lock below.)
    {
        let pool = state.cache_pool.lock().await;
        if pool.len() >= state.max_cache_shards && !pool.contains_key(&req.cache_id) {
            return Err(crate::chat::cache_pool_full_err(pool.len(), state.max_cache_shards));
        }
    }

    // Review #6: a refused VRAM budget is a 503, not a panic.
    let mut cache = state
        .engine
        .try_create_gpu_kv_cache(state.max_seq_len)
        .map_err(crate::chat::vram_exhausted_err)?;

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
        let cache_id_for_log = req.cache_id.clone();
        tokio::task::block_in_place(|| {
            forward_chunked_into_cache(
                &state.engine, &all_tokens, &mut cache,
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
    // When qjl=true is requested, construct via the QJL variant so the
    // populate path also writes K residual signs (handled inside
    // populate_from_f32_cache_gpu via qjl_encode_k_layer).
    let polar = if state.polar_cache_enabled {
        tokio::task::block_in_place(|| {
            let mut p = if req.qjl {
                state.engine.create_gpu_polar_kv_cache_with_qjl(
                    state.max_seq_len,
                    state.polar_rotation_seed,
                    state.qjl_projections,
                    state.qjl_seed,
                )
            } else {
                state.engine.create_gpu_polar_kv_cache(
                    state.max_seq_len, state.polar_rotation_seed,
                )
            };
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
        // Flush wgpu's deferred-destroy queue so the 300MB f32 cache we
        // just dropped is actually freed before the next cache_load
        // tries to allocate a fresh one. Without this, multiple back-
        // to-back polar_only loads accumulate destroyed-but-not-yet-
        // freed buffers in wgpu-29's allocator until allocation fails
        // with a delayed validation error ("Buffer X is invalid")
        // that surfaces at the NEXT poll/get_mapped_range — usually a
        // subsequent retrieve, with a misleading buffer label.
        tokio::task::block_in_place(|| state.engine.poll_wait());
        None
    } else {
        Some(cache)
    };
    let now = Instant::now();

    let mut pool = state.cache_pool.lock().await;
    // Review #6: authoritative pool-cap check under the lock (race-safe).
    // If the pool filled while we were prefilling, drop what we built.
    if pool.len() >= state.max_cache_shards && !pool.contains_key(&req.cache_id) {
        let n = pool.len();
        drop(pool);
        return Err(crate::chat::cache_pool_full_err(n, state.max_cache_shards));
    }
    // If overwriting an existing shard, bump from its current version so the
    // composition cache's staleness check sees the change. New shards start
    // at version 0; any subsequent insert / append bumps it monotonically.
    let next_version = pool.get(&req.cache_id).map(|e| e.version + 1).unwrap_or(0);
    pool.insert(
        req.cache_id.clone(),
        CacheEntry {
            cache: cache_opt,
            polar,
            polar_chat: req.polar_chat,
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
pub(crate) async fn cache_append(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CacheAppendRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut _telemetry = metrics::RequestTimer::new(state.metrics.clone(), metrics::Endpoint::CacheAppend);

    // Review #11: reject out-of-range token ids before any GPU work.
    crate::chat::check_token_ids(&req.tokens, state.engine.vocab_size())?;

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
            // Retrieval-only shards (polar_only=true AND polar_chat=false)
            // still 409. Everything with polar_chat=true now supports
            // append via the polar advance path (Phase 4b).
            Some(e) if e.cache.is_none() && !e.polar_chat => {
                return Err((
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": {
                            "type": "shard_is_polar_only",
                            "message": format!(
                                "cache '{}' was loaded with polar_only=true and polar_chat=false; cannot append. Reload the shard with polar_chat=true (and optionally polar_only=true) to enable appendable polar mode.",
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
        // Prefer the f32 cache when present; fall back to polar (Phase 4b
        // polar-only-appendable shards have cache=None but polar populated).
        let pool = state.cache_pool.lock().await;
        pool.get(&req.cache_id)
            .map(|e| {
                e.cache.as_ref().map(|c| c.seq_len())
                    .or_else(|| e.polar.as_ref().map(|p| p.seq_len()))
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    } else {
        // Pre-flight overflow check using a snapshot of current seq_len.
        // Uses f32 cache when present, else polar.
        let (start_seq, max_seq) = {
            let pool = state.cache_pool.lock().await;
            let e = pool.get(&req.cache_id).unwrap();
            match (e.cache.as_ref(), e.polar.as_ref()) {
                (Some(c), _) => (c.seq_len(), c.max_seq_len()),
                (None, Some(p)) => (p.seq_len(), p.max_seq_len()),
                (None, None) => unreachable!("entry validated to have at least one cache above"),
            }
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
            let chunk_size = state.engine.safe_prefill_chunk_size(next_start).min(tokens_remaining.len());
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
            // Three modes:
            //  (cache=Some, polar_chat=false): legacy f32-only path.
            //    polar (if present) is invalidated since the f32 cache
            //    is the canonical state and polar is now stale.
            //  (cache=Some, polar_chat=true):  update BOTH f32 and polar.
            //    Keeps the f32 fallback in sync for non-greedy chat;
            //    polar stays canonical for greedy chat.
            //  (cache=None,  polar_chat=true): polar-only path. f32 was
            //    dropped at load (polar_only=true).
            tokio::task::block_in_place(|| {
                if let Some(c) = entry.cache.as_mut() {
                    state.engine.forward_full_gpu_with_cache_advance_only(chunk, c);
                }
                if entry.polar_chat {
                    if let Some(p) = entry.polar.as_mut() {
                        state.engine.forward_full_gpu_polar_with_cache_advance_only(chunk, p);
                    }
                }
            });
            entry.tokens.extend_from_slice(chunk);
            entry.version += 1;
            // Invalidate polar only when this shard isn't polar-chat-tracked
            // (the existing path that nukes polar on every append). For
            // polar_chat shards, polar IS being kept in sync above.
            if !entry.polar_chat {
                entry.polar = None;
            }
            entry.last_used = Instant::now();
            current_start = match (entry.cache.as_ref(), entry.polar.as_ref()) {
                (Some(c), _) => c.seq_len(),
                (None, Some(p)) => p.seq_len(),
                (None, None) => unreachable!("validated at handler entry"),
            };
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
pub(crate) async fn cache_get(
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
pub(crate) async fn cache_delete(
    State(state): State<Arc<ServerState>>,
    Path(cache_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut pool = state.cache_pool.lock().await;
    if pool.remove(&cache_id).is_some() {
        let pool_size = pool.len();
        drop(pool);
        // Composition might reference the evicted shard; safest to drop.
        *state.composition.lock().await = None;
        // Flush wgpu's deferred-destroy queue so the evicted cache's
        // backing buffers (f32 kv_heap + polar const/data/signs heaps)
        // are actually freed NOW. Without this, repeated
        // load/retrieve/delete cycles accumulate destroyed-but-unfreed
        // buffers until wgpu-29 surfaces a delayed "Validation Error"
        // panic at the next Device::poll (observed at ~8-10 cycles).
        // Same mitigation as the polar_only drop in cache_load.
        tokio::task::block_in_place(|| state.engine.poll_wait());
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

