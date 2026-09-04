//! Shared server state (split from main.rs, Phase N).
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
    /// Bumps any time the shard's K/V content changes (load replaces, append
    /// extends). Used as the staleness witness for the multi-shard
    /// retrieval `composition` cache below.
    pub(crate) version: u64,
    #[allow(dead_code)]
    pub(crate) created_at: Instant,
    pub(crate) last_used: Instant,
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
    /// GPU-native retrieve path goes through `engine.forward_full_gpu_traced()`.
    pub(crate) engine: cortex::layers::gpu_engine::GpuEngine,
    pub(crate) tokenizer: Tokenizer,
    #[allow(dead_code)]
    pub(crate) config: ModelConfig,
    /// Pool of named KV caches. Only used when cache_enabled is true
    /// (librarian deployment). When false (32B Bob deployment), the pool
    /// is empty and cache_shards on requests are ignored.
    pub(crate) cache_pool: Mutex<HashMap<String, CacheEntry>>,
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

