//! Shared server state (split from main.rs, Phase N).
#![allow(unused_imports)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
use crate::api::*;
use crate::shims::*;
use crate::chat::*;
use crate::cache::*;

/// Number of dummy "sink" tokens prepended to every shard at load time.
/// These absorb the position-0 attention sink artifact (see POSITION-
/// addendum.md section 15 on the structural cause) so real content tokens
/// aren't contaminated. Retrieval scoring skips the first SINK_TOKENS
/// positions per shard.
pub(crate) const SINK_TOKENS: usize = 4;

/// Per-cache metadata stored alongside the KV cache in the pool.
pub(crate) struct CacheEntry {
    /// f32 KV cache. `None` for polar-only shards (loaded with
    /// `polar_only=true` so the f32 copy is dropped after the polar
    /// cache is materialized — ~7x VRAM win per shard). Polar-only
    /// shards reject chat and append operations with 409 since both
    /// require the f32 cache today; only `/v1/retrieve` is supported.
    pub(crate) cache: Option<GpuKvCache>,
    /// Optional PolarQuant-compressed K/V. Populated once at cache_load
    /// time (via `populate_from_f32_cache_gpu`) when `--enable-polar-cache`
    /// is set. Single-shard `/v1/retrieve` queries route through this
    /// when present; multi-shard composition replays from `tokens`
    /// regardless. When `cache` is `None`, this is the only KV storage.
    pub(crate) polar: Option<cortex::layers::gpu_polar_kv_cache::GpuPolarKvCache>,
    /// When true (set via `polar_chat=true` at cache_load), greedy
    /// chat against this shard routes through the polar orchestrator
    /// (compresses new K/V into the polar cache as it generates).
    /// Non-greedy / steered chat falls through to the f32 path during
    /// Phase 2. Append still 409s.
    pub(crate) polar_chat: bool,
    /// Token history that built this cache. Stored so shards can be composed
    /// by replaying tokens in sequence (which gives correct RoPE positions).
    pub(crate) tokens: Vec<u32>,
    /// Changes any time the shard's K/V content changes (load replaces, append
    /// extends, chat writes back). Drawn from `ServerState::next_cache_version`
    /// so it is unique across the whole process lifetime — a DELETE followed
    /// by a same-id load can never reproduce an earlier value. Used as the
    /// staleness witness for the multi-shard `composition` cache and, via
    /// `EntryWitness`, for every lock re-acquisition (review #9).
    pub(crate) version: u64,
    #[allow(dead_code)]
    pub(crate) created_at: Instant,
    pub(crate) last_used: Instant,
}

/// What a handler remembers about an entry across a gap in holding the
/// pool lock. Review #9: after the lock is dropped and re-taken, the entry
/// may be gone (`DELETE`), replaced (same-id `cache/load`) or mutated
/// (`cache/append`, chat write-back); the witness lets `relookup` tell the
/// difference instead of `unwrap()`ing a stale assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryWitness {
    pub(crate) version: u64,
    pub(crate) tokens_len: usize,
}

impl CacheEntry {
    pub(crate) fn witness(&self) -> EntryWitness {
        EntryWitness { version: self.version, tokens_len: self.tokens.len() }
    }

    /// Review #8 — THE invariant of a resident shard: `tokens` describes
    /// exactly what every resident cache holds, i.e.
    /// `tokens.len() == cache.seq_len() == polar.seq_len()` for whichever
    /// caches are present (sink tokens are counted on both sides — they are
    /// prepended into `tokens` at load and prefilled). Retrieve derives its
    /// score-row width from the cache and its corpus width from `tokens`;
    /// any drift is either silent mis-scoring or an out-of-bounds index.
    /// Every mutation path must leave this true; `check_lockstep` is the
    /// runtime backstop that turns a wedged entry into a 409 instead.
    pub(crate) fn check_lockstep(&self) -> Result<(), LockstepError> {
        let tokens_len = self.tokens.len();
        let f32_len = self.cache.as_ref().map(|c| c.seq_len());
        let polar_len = self.polar.as_ref().map(|p| p.seq_len());
        let ok = f32_len.map_or(true, |l| l == tokens_len)
            && polar_len.map_or(true, |l| l == tokens_len);
        if ok {
            Ok(())
        } else {
            Err(LockstepError { tokens_len, f32_len, polar_len })
        }
    }
}

/// What `CacheEntry::check_lockstep` found when the invariant is broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LockstepError {
    pub(crate) tokens_len: usize,
    pub(crate) f32_len: Option<usize>,
    pub(crate) polar_len: Option<usize>,
}

pub(crate) type CachePool = HashMap<String, CacheEntry>;

/// Review #7 — admission control for the GPU. The engine shares three
/// transient lanes, the params ring and its timers between all callers and
/// was written for one forward at a time; nothing in the server enforced
/// that (9 of 20 GPU regions ran with no lock at all, the rest happened to
/// hold the pool mutex). `admit()` hands out the single permit in FIFO
/// order; a handler holds it across its whole `block_in_place` /
/// `spawn_blocking` region — cache allocation, prefill, polar populate,
/// `poll_wait` included. Lock order, enforced by placement: **gpu_gate →
/// cache_pool → composition** (never wait for the gate while holding a
/// pool lock). `waiting` is the queue depth, exported as
/// `cortex_gpu_gate_waiting` — the Stage-2 (batching) trigger signal.
pub(crate) struct GpuGate {
    sem: Arc<tokio::sync::Semaphore>,
    waiting: AtomicU64,
}

impl GpuGate {
    pub(crate) fn new() -> Self {
        Self { sem: Arc::new(tokio::sync::Semaphore::new(1)), waiting: AtomicU64::new(0) }
    }

    /// Wait for the GPU. The returned permit is owned so it can be moved
    /// into a `spawn_blocking` closure; it is released on drop.
    pub(crate) async fn admit(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let permit = self.sem.clone().acquire_owned().await
            .expect("gpu_gate semaphore is never closed");
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        permit
    }

    pub(crate) fn waiting(&self) -> u64 {
        self.waiting.load(Ordering::Relaxed)
    }
}

/// Re-find `id` after a lock gap and check it is still the entry the caller
/// snapshotted. 404 `cache_not_found` if it is gone, 409 `cache_changed` if
/// it was replaced or mutated in the meantime.
pub(crate) fn relookup<'a>(
    pool: &'a CachePool,
    id: &str,
    expected: EntryWitness,
) -> Result<&'a CacheEntry, (StatusCode, Json<serde_json::Value>)> {
    match pool.get(id) {
        None => Err(cache_not_found_err(id)),
        Some(e) if e.witness() != expected => Err(cache_changed_err(id, expected, e.witness())),
        Some(e) => Ok(e),
    }
}

/// Mutable twin of [`relookup`].
pub(crate) fn relookup_mut<'a>(
    pool: &'a mut CachePool,
    id: &str,
    expected: EntryWitness,
) -> Result<&'a mut CacheEntry, (StatusCode, Json<serde_json::Value>)> {
    match pool.get_mut(id) {
        None => Err(cache_not_found_err(id)),
        Some(e) if e.witness() != expected => Err(cache_changed_err(id, expected, e.witness())),
        Some(e) => Ok(e),
    }
}

/// One composed-cache slot, reused across multi-shard retrieve requests so
/// each query doesn't re-allocate and re-prefill ~85 MiB of K/V buffers.
/// Populated lazily on first multi-shard retrieve and reused while the
/// cached `(shard_name, version)` key keeps matching incoming requests.
pub(crate) struct ComposedEntry {
    /// Ordered list of `(shard_name, version_at_compose_time)`. Matches
    /// the request key exactly: same shards, same order, same versions.
    /// Order matters because RoPE positions depend on token order.
    pub(crate) key: Vec<(String, u64)>,
    /// The composed cache itself.
    pub(crate) cache: GpuKvCache,
}

pub(crate) struct ServerState {
    /// GPU-resident inference engine. Owns the underlying TransformerModel
    /// and the GPU device. CPU-side calls go through `engine.cpu()`; the
    /// GPU-native retrieve path goes through `engine.forward_full_gpu_with_cache_traced()`
    /// (f32 shards) / `forward_full_gpu_polar_traced()` (polar shards).
    pub(crate) engine: cortex::layers::gpu_engine::GpuEngine,
    pub(crate) tokenizer: Tokenizer,
    #[allow(dead_code)]
    pub(crate) config: ModelConfig,
    /// Pool of named KV caches. Only used when cache_enabled is true
    /// (librarian deployment). When false (32B Bob deployment), the pool
    /// is empty and cache_shards on requests are ignored.
    pub(crate) cache_pool: Mutex<HashMap<String, CacheEntry>>,
    /// Source of `CacheEntry::version` values: a process-wide monotonic
    /// counter, so no two entries (or two states of one entry) ever share a
    /// version. See `EntryWitness`.
    pub(crate) next_cache_version: AtomicU64,
    /// Review #7: one GPU region at a time. See `GpuGate`.
    pub(crate) gpu_gate: GpuGate,
    /// Single-slot composition cache for multi-shard retrieve. Holds at most
    /// one composed `GpuKvCache`; reused when the next request's
    /// `(shard, version)` key matches; rebuilt in place (clear + re-prefill,
    /// no buffer alloc) when it differs. Critical for stability: rapid
    /// per-request alloc-and-drop of ~85 MiB buffer arrays hangs the wgpu
    /// driver after ~3 requests.
    pub(crate) composition: Mutex<Option<ComposedEntry>>,
    pub(crate) model_name: String,
    pub(crate) start_time: Instant,
    pub(crate) max_seq_len: usize,
    /// Review #6: cap on resident cache shards (see --max-cache-shards).
    pub(crate) max_cache_shards: usize,
    /// Whether cache endpoints and cache_shards are enabled.
    pub(crate) cache_enabled: bool,
    /// Whether retrieval mode is enabled.
    pub(crate) retrieve_enabled: bool,
    /// Whether to build a parallel polar-compressed cache on cache_load
    /// (and use it for single-shard retrieve when present).
    pub(crate) polar_cache_enabled: bool,
    /// Per-layer rotation seed base for any polar caches built by this
    /// server. Stored on ServerState so all polar caches share the same
    /// seeding scheme — required for cross-cache compatibility (e.g.
    /// multi-shard polar composition, future).
    pub(crate) polar_rotation_seed: u64,
    /// Number of QJL projections per K residual when a polar cache is
    /// loaded with `qjl: true`. Comes from `--qjl-projections` (default
    /// 32). 0 disables QJL even if the request asks for it (not
    /// currently used — CLI default is 32).
    pub(crate) qjl_projections: usize,
    /// Seed base for per-layer QJL projection matrices. From `--qjl-seed`.
    /// Independent of `polar_rotation_seed` — see CLI docs.
    pub(crate) qjl_seed: u64,
    /// Shim registry: hot-resident ONNX shims keyed by id. Empty unless
    /// `shims_enabled` is true. `Arc` so handlers can clone-into-handler
    /// without holding the registry lock through inference.
    pub(crate) shims: Mutex<HashMap<String, Arc<RegisteredShim>>>,
    /// Whether shim endpoints are enabled.
    pub(crate) shims_enabled: bool,
    /// Prometheus telemetry. Recorded by chat_completions / cache_load /
    /// cache_append handlers; rendered via `GET /metrics`.
    pub(crate) metrics: Arc<crate::metrics::Metrics>,
}

impl ServerState {
    /// Next unique `CacheEntry::version`. Starts at 1 so 0 never appears.
    pub(crate) fn next_cache_version(&self) -> u64 {
        self.next_cache_version.fetch_add(1, Ordering::Relaxed) + 1
    }
}

#[cfg(test)]
mod witness_tests {
    use super::*;

    fn entry(version: u64, n_tokens: usize) -> CacheEntry {
        let now = Instant::now();
        CacheEntry {
            cache: None,
            polar: None,
            polar_chat: false,
            tokens: vec![0; n_tokens],
            version,
            created_at: now,
            last_used: now,
        }
    }

    fn err_type(e: &(StatusCode, Json<serde_json::Value>)) -> (StatusCode, String) {
        (e.0, e.1["error"]["type"].as_str().unwrap_or("").to_string())
    }

    #[test]
    fn relookup_accepts_an_unchanged_entry() {
        let mut pool = CachePool::new();
        pool.insert("a".into(), entry(7, 10));
        let w = pool["a"].witness();
        assert!(relookup(&pool, "a", w).is_ok());
        assert!(relookup_mut(&mut pool, "a", w).is_ok());
    }

    #[test]
    fn relookup_404s_when_the_entry_was_deleted() {
        let mut pool = CachePool::new();
        pool.insert("a".into(), entry(7, 10));
        let w = pool["a"].witness();
        pool.remove("a");
        let e = relookup(&pool, "a", w).err().unwrap();
        assert_eq!(err_type(&e), (StatusCode::NOT_FOUND, "cache_not_found".into()));
        let e = relookup_mut(&mut pool, "a", w).err().unwrap();
        assert_eq!(err_type(&e), (StatusCode::NOT_FOUND, "cache_not_found".into()));
    }

    #[test]
    fn relookup_409s_when_the_entry_was_replaced_or_mutated() {
        let mut pool = CachePool::new();
        pool.insert("a".into(), entry(7, 10));
        let w = pool["a"].witness();
        // Same-id reload: different version, same length.
        pool.insert("a".into(), entry(8, 10));
        let e = relookup(&pool, "a", w).err().unwrap();
        assert_eq!(err_type(&e), (StatusCode::CONFLICT, "cache_changed".into()));
        // Append that forgot to bump the version: same version, longer.
        pool.insert("a".into(), entry(7, 11));
        let e = relookup_mut(&mut pool, "a", w).err().unwrap();
        assert_eq!(err_type(&e), (StatusCode::CONFLICT, "cache_changed".into()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gpu_gate_admits_one_at_a_time_and_counts_waiters() {
        let gate = Arc::new(GpuGate::new());
        let first = gate.admit().await;
        assert_eq!(gate.waiting(), 0);

        let g2 = gate.clone();
        let second = tokio::spawn(async move {
            let _p = g2.admit().await;
            g2.waiting()
        });
        // Give the second task time to park on the semaphore.
        for _ in 0..50 {
            if gate.waiting() == 1 { break; }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(gate.waiting(), 1, "second admit should be queued");
        assert!(!second.is_finished(), "second admit ran while the first permit was held");

        drop(first);
        let waiting_seen_inside = tokio::time::timeout(
            std::time::Duration::from_secs(2), second,
        ).await.expect("second admit never completed").unwrap();
        assert_eq!(waiting_seen_inside, 0);
        assert_eq!(gate.waiting(), 0);
    }

    #[test]
    fn lockstep_holds_vacuously_without_resident_caches() {
        // cache: None, polar: None is the only shape constructible without a
        // GPU; the invariant is vacuous there. The populated shapes are
        // covered end-to-end (e2e_lockstep.py).
        assert!(entry(1, 10).check_lockstep().is_ok());
    }

    #[test]
    fn a_refreshed_witness_tracks_the_callers_own_mutation() {
        let mut pool = CachePool::new();
        pool.insert("a".into(), entry(1, 10));
        let mut w = pool["a"].witness();
        {
            let e = relookup_mut(&mut pool, "a", w).unwrap();
            e.tokens.push(0);
            e.version = 2;
            w = e.witness();
        }
        assert!(relookup(&pool, "a", w).is_ok());
    }
}

