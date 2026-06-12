//! Wire types — OpenAI-compatible + cache/tokenize request/response (split from main.rs, Phase N).
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
use crate::shims::*;
use crate::chat::*;
use crate::cache::*;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub(crate) max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub(crate) temperature: f32,
    #[serde(default)]
    pub(crate) tools: Option<Vec<Tool>>,
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
    pub(crate) cache_shards: Option<Vec<String>>,

    /// Backward-compatible single cache ID. If present and cache_shards
    /// is absent, treated as a one-element shard list. Deprecated in
    /// favor of cache_shards.
    #[serde(default)]
    pub(crate) cache_id: Option<String>,

    /// Inference mode. "generate" (default) produces tokens. "retrieve"
    /// computes attention from query positions over the cached corpus
    /// and returns top-K positions with scores instead of generating.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) mode: Option<String>,

    /// For mode: "retrieve" — number of top-scoring positions to return.
    #[serde(default = "default_top_k")]
    #[allow(dead_code)]
    pub(crate) top_k: usize,

    /// OpenAI-compatible streaming flag. When true, the response is an
    /// SSE stream of `chat.completion.chunk` events terminated by a
    /// `data: [DONE]` event. First wire supports stateless mode only
    /// (no cache_shards, mode=generate); cached + streaming is a
    /// follow-up because it requires holding the cache pool lock across
    /// the entire generation, which serializes all other requests.
    #[serde(default)]
    pub(crate) stream: bool,

    /// Gate shims to evaluate on the prefilled prompt's final hidden
    /// state. Each shim must be registered and its phase must be
    /// "gate" (otherwise the request is rejected before generation).
    /// Decisions are evaluated against `shim_rules`; if a rule routes
    /// to silent, generation is skipped and a `done{silent:true}`
    /// response is emitted. Per `project_cortex_v1_shim_api.md`, the
    /// gate fires once after prefill — the single prefill is what made
    /// the decision possible. Empty / absent = no gate dispatch.
    #[serde(default)]
    pub(crate) gate_shims: Vec<String>,

    /// Steer shims to apply during decode (per-token hidden
    /// modification). v1 records the active set in response metadata
    /// but does not yet apply it — the steer-phase forward path is
    /// scheduled for #6b. Listed here so callers (AgentOS) can pin
    /// the wire shape now and v1 servers accept the field without
    /// erroring.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) steer_shims: Vec<String>,

    /// Injection shims attached to layer entrances (residual add to
    /// hidden state during forward). v1 records the requested set in
    /// response metadata but does not yet apply it — the injection
    /// hook lands in #6c. Field present for forward-compat.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) inject_shims: Vec<String>,

    /// Declarative dispatch rules evaluated against gate decisions.
    /// First matching rule wins; an `else` arm (rule with no `if`)
    /// is the conventional fallthrough. See `ShimRule` for the wire
    /// shape. Empty / absent = run all gates for observability but
    /// take no action (generation proceeds normally).
    #[serde(default)]
    pub(crate) shim_rules: Vec<ShimRule>,
}

pub(crate) fn default_top_k() -> usize { 10 }

pub(crate) fn default_max_tokens() -> u32 { 2048 }

pub(crate) fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub(crate) tool_type: String,
    pub(crate) function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct ToolFunction {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) call_type: String,
    pub(crate) function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct ToolCallFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Serialize)]
pub(crate) struct ChatResponse {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) choices: Vec<Choice>,
    pub(crate) usage: Usage,
    /// Shim phase-dispatch metadata. Present only when gate_shims fired
    /// for this request: includes per-shim decisions, active steers, and
    /// any signals emitted by matched rules. Omitted (not `null`) when
    /// no shim work happened for this completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub(crate) struct Choice {
    pub(crate) message: ChatMessage,
    pub(crate) finish_reason: String,
}

#[derive(Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
}

#[derive(Serialize)]
pub(crate) struct ModelsResponse {
    pub(crate) data: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub(crate) struct ModelEntry {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: u64,
    pub(crate) owned_by: String,
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    pub(crate) status: String,
    pub(crate) model: String,
    pub(crate) uptime_secs: u64,
    pub(crate) memory: HealthMemory,
}

#[derive(Serialize)]
pub(crate) struct HealthMemory {
    pub(crate) cache_pool_size: usize,
    pub(crate) cache_pool_total_tokens: usize,
    pub(crate) max_seq_len: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CacheLoadRequest {
    pub(crate) cache_id: String,
    /// Token IDs to replay through the model to build the KV cache.
    /// For a brand-new user this is empty []. For a returning user after
    /// eviction, this is the full conversation history from sled.
    #[serde(default)]
    pub(crate) tokens: Vec<u32>,
    /// When true, build the polar-compressed cache and drop the f32 KV
    /// copy. The resulting shard is read-only: only `/v1/retrieve`
    /// works against it; chat (`cache_shards`) and `cache_append`
    /// reject it with 409 Conflict. Requires the server to be started
    /// with `--enable-polar-cache`; otherwise the request is rejected
    /// with 400. ~7x VRAM reduction per shard for retrieval-only use
    /// cases (RAG corpora, knowledge bases).
    #[serde(default)]
    pub(crate) polar_only: bool,
    /// When true, mark the shard as chattable via the polar path:
    /// greedy chat against this shard routes through the polar
    /// orchestrator (compresses new K/V into the polar cache as it
    /// generates). Requires `--enable-polar-cache`. Append remains
    /// 409 — Phase 4b will add a polar-aware append helper. Can be
    /// combined with `polar_only` to drop the f32 cache entirely
    /// (chat goes via polar, retrieve goes via polar — minimum VRAM).
    #[serde(default)]
    pub(crate) polar_chat: bool,
    /// When true, the polar cache is built with K-residual QJL
    /// correction enabled (`--qjl-projections` projections per layer,
    /// seeded from `--qjl-seed`). Brings polar attention output
    /// cosine from ~0.84 to ~0.95 vs f32 at small storage cost (~1.1
    /// MB extra for Qwen 3B). Requires `--enable-polar-cache`. Has
    /// no effect when neither `polar_only` nor `polar_chat` is set.
    #[serde(default)]
    pub(crate) qjl: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CacheAppendRequest {
    pub(crate) cache_id: String,
    pub(crate) tokens: Vec<u32>,
}

#[derive(Serialize)]
pub(crate) struct CacheInfoResponse {
    pub(crate) cache_id: String,
    pub(crate) seq_len: usize,
    pub(crate) max_seq_len: usize,
}

#[derive(Serialize)]
pub(crate) struct CacheLoadResponse {
    pub(crate) cache_id: String,
    pub(crate) seq_len: usize,
    pub(crate) status: String,
}

#[derive(Serialize)]
pub(crate) struct RetrievalResponse {
    pub(crate) hits: Vec<RetrievalHit>,
    pub(crate) metadata: RetrievalMetadata,
}

#[derive(Serialize)]
pub(crate) struct RetrievalHit {
    pub(crate) shard_id: String,
    pub(crate) offset: usize,
    pub(crate) length: u32,
    pub(crate) score: f32,
}

#[derive(Serialize)]
pub(crate) struct RetrievalMetadata {
    pub(crate) retrieval_ms: u64,
    pub(crate) query_tokens: u32,
    pub(crate) corpus_tokens: u32,
    pub(crate) layers_used: Vec<usize>,
}

/// Maps a position in a composed token sequence back to its shard + offset.
pub(crate) struct ShardMap {
    /// Sorted by start position: (shard_name, start, end)
    pub(crate) entries: Vec<(String, usize, usize)>,
}

impl ShardMap {
    pub(crate) fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub(crate) fn add(&mut self, shard_name: String, start: usize, end: usize) {
        self.entries.push((shard_name, start, end));
    }

    /// Resolve an absolute position in the composed sequence to (shard_name, offset_within_shard).
    pub(crate) fn resolve(&self, pos: usize) -> Option<(&str, usize)> {
        for (name, start, end) in &self.entries {
            if pos >= *start && pos < *end {
                return Some((name, pos - start));
            }
        }
        None
    }

    /// Total corpus positions (sum of all shard lengths).
    pub(crate) fn corpus_len(&self) -> usize {
        self.entries.last().map(|(_, _, end)| *end).unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenizeRequest {
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) add_bos: Option<bool>,
}

#[derive(Serialize)]
pub(crate) struct TokenizeResponse {
    pub(crate) tokens: Vec<u32>,
    pub(crate) count: usize,
}

#[derive(Deserialize)]
pub(crate) struct DetokenizeRequest {
    pub(crate) tokens: Vec<u32>,
}

#[derive(Serialize)]
pub(crate) struct DetokenizeResponse {
    pub(crate) text: String,
}

