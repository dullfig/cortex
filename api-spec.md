# Cortex Inference Server — API Specification

**Date:** 2026-04-05
**Consumer:** AgentOS `agentos-llm` crate (OpenAI wire format client)
**First deployment:** RunPod A100 via `agentos-cloud` provisioning

## Overview

Cortex needs an HTTP server that speaks the OpenAI `/v1/chat/completions` wire
format. This is the de facto standard for local inference engines (vLLM, llama.cpp
server, Ollama). AgentOS's `OpenAiClient` is already built and tested against
this format — cortex just needs to serve it.

The server is a thin wrapper around `TransformerModel::generate()`. The persistent
KV memory system is *internal* to cortex — the API is stateless request/response
from the caller's perspective, but cortex remembers across calls.

## Required Endpoints

### `POST /v1/chat/completions`

The only endpoint that matters for AgentOS integration.

**Request:**
```json
{
  "model": "qwen-30b",
  "messages": [
    {"role": "system", "content": "You are the ringhub concierge."},
    {"role": "user", "content": "What events are happening tonight?"}
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_events",
        "description": "Search ringhub events",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
      }
    }
  ]
}
```

**Response (text):**
```json
{
  "id": "cortex-abc123",
  "model": "qwen-30b",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Let me check what's happening tonight!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 10
  }
}
```

**Response (tool call):**
```json
{
  "id": "cortex-def456",
  "model": "qwen-30b",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_001",
            "type": "function",
            "function": {
              "name": "search_events",
              "arguments": "{\"query\": \"tonight\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 20
  }
}
```

**Extended field — `cache_id`:**

The chat completions endpoint accepts an optional `cache_id` field that references
a resident per-user KV cache. When present, cortex uses the cached attention state
instead of recomputing from scratch. This is the mechanism that gives the concierge
persistent memory across conversations.

```json
{
  "model": "qwen-30b",
  "cache_id": "user-alice",
  "messages": [...]
}
```

If `cache_id` is provided but not resident, cortex returns 404 with:
```json
{"error": "cache_not_found", "cache_id": "user-alice"}
```
The caller (AgentOS) should then `POST /v1/cache/load` to restore from its sled
backup, then retry the completion.

The response includes new KV entries generated during this turn:
```json
{
  "id": "cortex-abc123",
  "model": "qwen-30b",
  "choices": [...],
  "usage": {...},
  "new_cache_entries": [
    {"layer": 0, "seq_start": 2048, "seq_len": 128, "format": "turboquant", "data": "<base64>"},
    {"layer": 1, "seq_start": 2048, "seq_len": 128, "format": "turboquant", "data": "<base64>"}
  ]
}
```

AgentOS appends these to both cortex (via `/v1/cache/append`) and its local sled
store (for durability). If `new_cache_entries` is absent or empty, no new entries
were generated (e.g., very short response).

**Other field notes:**
- `model` in the request is informational — cortex serves whatever model it loaded at startup
- `temperature`, `max_tokens` map directly to sampler config
- `tools` support is needed for the concierge to use AgentOS tools (search, calendar, etc.)
- `tool_calls[].function.arguments` is a **JSON string**, not an object (OpenAI convention)
- `finish_reason`: `"stop"` for normal completion, `"tool_calls"` when the model invokes tools, `"length"` if max_tokens hit
- `id` can be any unique string, `"cortex-"` + uuid is fine

### `GET /v1/models`

Optional but nice — lets the pipeline discover what's loaded.

```json
{
  "data": [
    {
      "id": "qwen-30b",
      "object": "model",
      "created": 1712300000,
      "owned_by": "cortex"
    }
  ]
}
```

### `POST /v1/cache/load`

Full cache restore — called on cold start when a user's cache is not resident.
AgentOS sends the compressed KV entries from its sled store; cortex loads them
into GPU memory so subsequent requests can use `cache_id`.

**Request:**
```json
{
  "cache_id": "user-alice",
  "entries": [
    {"layer": 0, "seq_start": 0, "seq_len": 128, "format": "turboquant", "data": "<base64>"},
    {"layer": 0, "seq_start": 128, "seq_len": 128, "format": "turboquant", "data": "<base64>"},
    {"layer": 1, "seq_start": 0, "seq_len": 128, "format": "turboquant", "data": "<base64>"}
  ]
}
```

**Response:**
```json
{
  "cache_id": "user-alice",
  "status": "loaded",
  "entries_loaded": 3,
  "layers": 2,
  "seq_range": [0, 256],
  "memory_mb": 42.5
}
```

**Notes:**
- This is the only request that sends bulk KV data. It happens once per session
  (when a user first talks after being idle) — not on every message.
- Transfer size: ~42MB for a 2000-token history with TurboQuant at 12x compression.
- Cortex should decompress and inject entries into the model's KV cache layers.
- If `cache_id` already exists, the old cache is replaced (not merged).

### `POST /v1/cache/append`

Incremental update — adds new KV entries to an existing resident cache.
Called after each inference turn to keep the cache current.

**Request:**
```json
{
  "cache_id": "user-alice",
  "entries": [
    {"layer": 0, "seq_start": 2048, "seq_len": 128, "format": "turboquant", "data": "<base64>"},
    {"layer": 1, "seq_start": 2048, "seq_len": 128, "format": "turboquant", "data": "<base64>"}
  ]
}
```

**Response:**
```json
{
  "cache_id": "user-alice",
  "status": "appended",
  "entries_added": 2,
  "total_entries": 50,
  "memory_mb": 43.2
}
```

**Notes:**
- Tiny payload — only the new entries from the latest turn (~0.5-2MB).
- Returns 404 if `cache_id` not resident (caller needs to do a full load first).
- AgentOS also writes these same entries to its local sled store for durability.

### `GET /v1/cache/{cache_id}`

Check if a user's cache is resident and get stats. Used by AgentOS to decide
whether to do a full load or go straight to inference.

**Response (resident):**
```json
{
  "cache_id": "user-alice",
  "status": "resident",
  "entries": 48,
  "layers": 32,
  "seq_range": [0, 2048],
  "memory_mb": 42.5,
  "last_accessed_secs_ago": 30
}
```

**Response (not resident):**
```json
HTTP 404
{"error": "cache_not_found", "cache_id": "user-alice"}
```

### `DELETE /v1/cache/{cache_id}`

Evict a user's cache from GPU memory. Called when a user goes idle or when
cortex needs to free memory for other users.

**Response:**
```json
{
  "cache_id": "user-alice",
  "status": "evicted",
  "memory_freed_mb": 42.5
}
```

**Notes:**
- AgentOS should call this proactively when a user goes idle (e.g., 5 min timeout).
- Cortex may also auto-evict LRU caches when GPU memory pressure is high.
  In that case, subsequent requests with that `cache_id` get 404 and AgentOS
  does a full load — graceful degradation.

### `GET /health`

For cloud-expert to poll during provisioning.

```json
{
  "status": "ready",
  "model": "qwen-30b",
  "uptime_secs": 3600,
  "cache_pool": {
    "resident_users": 347,
    "total_entries": 16384,
    "total_memory_mb": 14700.0,
    "gpu_memory_used_mb": 58000.0,
    "gpu_memory_total_mb": 81920.0
  }
}
```

`status` values: `"loading"` (model still loading), `"ready"` (accepting requests), `"error"`.

## Implementation Notes

### Server framework

Suggest `axum` — already tokio-native, minimal, plays well with Rust async.
Add to Cargo.toml behind a feature flag:

```toml
[features]
server = ["axum", "tokio"]

[dependencies]
axum = { version = "0.8", optional = true }
tokio = { version = "1", features = ["full"], optional = true }
```

### Binary

`src/bin/cortex-server.rs` — loads model, starts axum, serves the API:

```rust
// Pseudocode
let model = cortex::load_model(&args.model_path)?;
let cache_pool = Arc::new(CachePool::new(gpu_memory_budget));
let app = Router::new()
    .route("/v1/chat/completions", post(chat_completions))
    .route("/v1/cache/load", post(cache_load))
    .route("/v1/cache/append", post(cache_append))
    .route("/v1/cache/{cache_id}", get(cache_status))
    .route("/v1/cache/{cache_id}", delete(cache_evict))
    .route("/v1/models", get(list_models))
    .route("/health", get(health))
    .with_state((model, cache_pool));
axum::serve(listener, app).await?;
```

### Chat template

The server needs to convert `messages[]` → token sequence using the model's
chat template (ChatML for Qwen, Llama-style for others). The tokenizer already
knows the template from GGUF metadata — just apply it.

### Tool calling

For a 30B model to reliably produce tool calls, cortex should:
1. Inject tool definitions into the system prompt (as structured text)
2. Use grammar-constrained decoding to force valid JSON in tool call output
3. Parse the output to detect tool call patterns and structure the response

This is where cortex's grammar constraints + the Code-Savant approach pay off.
The model doesn't need to "know" the tool call format — the grammar forces it.

### Memory integration — incremental cache protocol

Cortex manages a **cache pool** — a collection of per-user KV caches resident
in GPU memory. The protocol is designed so the A100 is semi-stateful (caches
live on GPU for performance) but the source of truth is AgentOS's sled store.

**Request flow (warm user — cache resident):**
```
AgentOS: GET  /v1/cache/alice            → 200, resident
AgentOS: POST /v1/chat/completions       → {cache_id: "alice", messages: [...]}
Cortex: runs inference with resident cache, returns response + new_cache_entries
AgentOS: POST /v1/cache/append           → appends new entries to cortex
AgentOS: kv_store.append("alice", ...)   → appends to local sled (durability)
```

**Request flow (cold user — cache not resident):**
```
AgentOS: GET  /v1/cache/alice            → 404
AgentOS: loads entries from sled
AgentOS: POST /v1/cache/load             → full restore (~42MB one-time)
AgentOS: POST /v1/chat/completions       → {cache_id: "alice", messages: [...]}
... (same as warm from here)
```

**Idle eviction:**
```
AgentOS: user idle 5 min
AgentOS: DELETE /v1/cache/alice           → frees GPU memory
(sled still has everything — next message triggers cold load)
```

**Pod death recovery:**
```
Pod dies → all resident caches lost
New pod starts → all users are cold
First message from each user → full load from sled → back to warm
```

Key points:
- Bulk transfer (full load) happens once per session, not per message
- Per-turn transfer (append) is tiny: ~0.5-2MB of new entries
- Cortex may auto-evict LRU caches under memory pressure — AgentOS handles 404 gracefully
- AgentOS's sled store is the source of truth, cortex's GPU memory is a hot cache
- The `/health` endpoint reports cache pool stats for monitoring

### Docker image

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server,gpu

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
COPY --from=builder /app/target/release/cortex-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["cortex-server", "--port", "8080", "--model", "/models/model.gguf"]
```

Model weights mount as a volume at `/models/` — RunPod network volumes
persist across pod restarts, so you download once.

### Startup sequence (on RunPod)

1. Pod starts with `dullfig/cortex:latest`
2. cortex-server loads model from `/models/` volume
3. Hits `/health` → returns `"loading"` while model loads
4. Model ready → `/health` returns `"ready"`
5. cloud-expert sees `"ready"` → calls `register_cloud_endpoint()`
6. LlmPool picks up the new provider → pipeline routes inference to cortex

## What AgentOS Already Has

These are built and tested — cortex just needs to serve the format:

- `OpenAiClient` (`crates/llm/src/openai.rs`) — translates AgentOS messages ↔ OpenAI format
- `LlmPool` auto-detects OpenAI protocol for provider names containing "cortex"
- `register_cloud_endpoint()` — writes the provider entry into ModelsConfig
- `cloud-expert` organism — provisions RunPod, polls health, registers endpoint
- `runpod.rhai` — RunPod GraphQL adapter for provisioning

## Testing the Integration

Before deploying to RunPod, test locally:

```bash
# Terminal 1: start cortex with a small model
cortex-server --port 8080 --model path/to/qwen-0.5b.gguf

# Terminal 2: test with curl
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Terminal 3: test through AgentOS
# Add to ~/.agentos/models.yaml:
#   cortex-local:
#     base_url: http://localhost:8080/v1
#     models:
#       local: qwen-0.5b
# Then: /model local
```
