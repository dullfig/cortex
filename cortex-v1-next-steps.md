> **Cross-session coordination:** This doc assumes you have read `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\MEMORY.md` and its referenced files. Several architectural decisions from 2026-04-19's sessions change what cortex needs to be — read memory first, then this. If you haven't, start there.

# Cortex v1 Next Steps

**Purpose:** scope what cortex needs to do for v1, aligned to today's (2026-04-19) architectural decisions pinned in the integration memory.

**Audience:** the Claude session working in `C:\src\cortex\`.

**Status:** v1 scope document. Not exhaustive. Implementation details live in the code.

---

## The foundational framing (read this first — it changes everything)

**Memex is a classifier, not a generator.** See `project_memex_architecture.md` in integration memory.

> "We're not running a language model; we're running a relevance classifier whose internals happen to be transformer-shaped."

This means cortex has two distinct operating modes, not one:

| Mode | Purpose | Output |
|---|---|---|
| **Retrieval** | Memex librarian — attention-as-classifier over a cache | Ranked cache positions with scores |
| **Generation** | Bob's responses — standard autoregressive generation | Token stream |

Some cortex deployments run only retrieval (the memex librarian). Some run only generation (the 32B Qwen Instruct serving Bob). The codebase is shared; the startup flags differ.

**Critical:** retrieval mode does NOT generate tokens. It computes attention scores across the composed cache and returns ranked (shard_id, offset, length, score) tuples. The output is structured data, not text.

---

## What cortex needs to build / finish for v1

### 1. Retrieval mode — `forward_traced` and position ranking

Priority: **highest.** This is what memex needs to work on Monday when the 4090 arrives.

**Capability:**

- Load one or more shards into the compute context (compose the cache)
- Run a query through the composed cache
- Extract attention scores from the last N layers (configurable; probably last 3-5)
- Rank all cache positions by aggregated attention weight
- Return top-K as `(shard_id, offset, length, score)` tuples

**Interface (rough — refine as needed):**

```rust
// Pseudocode
struct RetrievalQuery {
    shards: Vec<ShardId>,     // which shards to compose
    text: String,              // the query
    top_k: usize,              // how many positions to return
}

struct RetrievalHit {
    shard_id: ShardId,
    offset: usize,             // byte offset within the shard's source content
    length: usize,             // length of the span
    score: f32,                // aggregated attention score
}

fn retrieve(query: RetrievalQuery) -> Vec<RetrievalHit>;
```

**Implementation notes:**
- Activate retrieval mode via startup flag `--enable-retrieve` (see `api-spec.md`)
- Use the existing `TransformerModel::forward` but intercept attention scores rather than emitting tokens
- For the v1 model (Qwen 7B base with softmax attention), **prepend 4 "sink" tokens** to each shard at ingest to absorb the attention-sink pathology — see `project_memex_architecture.md`
- Attention aggregation across layers: start with "max score across last 3 layers per position" — simple, works well in practice; refine if needed

### 2. Compressed KV cache support (TurboQuant)

Priority: **high.** Memex's whole scale story depends on this.

**What to do:**
- Move `QuantizedKvCache` from `engram` into cortex (or implement if not present)
- TurboQuant compression target: ~12× reduction vs FP16 uncompressed
- Shard files on sled contain compressed KV entries; cortex decompresses on load
- Per-layer compression is fine; per-token compression is what the paper does

The roadmap in cortex/CLAUDE.md already lists this as "TODO" — today's decisions confirm it's a v1 item, not a v2 item. Memex at Harmonizer scale (13M tokens) requires this compression to fit in GPU memory.

### 3. Cache management HTTP API

Priority: **high.** Needed for AgentOS / memex caller to manage resident shards.

The existing `api-spec.md` already describes these endpoints (with "AgentOS" as the caller, per today's rename from "Donna"). Confirm they're implemented:

- `POST /v1/cache/load` — load a compressed shard from provided bytes into GPU pool
- `POST /v1/cache/append` — append new KV entries to a resident shard
- `GET /v1/cache/{id}` — check residency and stats
- `DELETE /v1/cache/{id}` — evict shard from GPU

**404 on missing shard is the protocol** (never implicit creation) per `api-spec.md`. The caller (AgentOS) handles cold-start by loading from sled and retrying.

### 4. Retrieval endpoint (new)

Priority: **high.** The caller needs a way to invoke retrieval mode.

**Shape (sketch):**

```
POST /v1/retrieve
Authorization: Bearer <token>
Content-Type: application/json

{
  "cache_shards": ["ringhub.shared.public", "ringhub.users.alice", ...],
  "query": "what was the 'keep it barbershop' backlash all about?",
  "top_k": 20
}

Response:
{
  "hits": [
    {"shard_id": "ringhub.shared.public", "offset": 4521, "length": 166, "score": 0.87},
    ...
  ],
  "metadata": {
    "model": "qwen-7b-base",
    "retrieval_ms": 142
  }
}
```

**Attribution-stripping invariant** (from `project_who_says_problem.md` in integration memory): retrieval response metadata must never include shard paths that contain user identifiers. Cortex stores opaque bytes; attribution policy is enforced at shard construction by the caller. Cortex just reports back what it retrieved from what it was given.

### 5. Generation mode — SSE streaming per API contract

Priority: **high** for the generative deployment (32B Qwen Instruct).

See `project_agentos_api_contract.md` in integration memory for the full contract.

Key requirements:
- `POST /v1/chat/completions` returns SSE stream (`Content-Type: text/event-stream`)
- Event sequence: `ack → [text]* → done`
- **Silence is first-class** (see `project_silence_as_first_class.md`): empty text stream between `ack` and `done` is a valid, non-error outcome
- `done` event includes `silent: bool`, `metadata` with shim decisions, corpora queried, generation_ms

**The silence requirement is architectural.** Cortex must support "the shim said don't respond → emit ack then done with silent=true, zero text events in between." Don't treat this as an error path; it's a first-class outcome.

### 6. Shim loading and invocation

Priority: **medium.** Initial shim (`should_respond`) doesn't need to exist at cortex-launch-time (AgentOS ships v1 shim later), but cortex needs the MECHANISM to load and invoke shims when they're provided.

See `project_agentos_shim_management.md` in integration memory for the full design. Cortex's role:

- Load ONNX shim files at startup (path provided via config)
- Invoke shim at the right point in the forward pass (typically: hidden state at the final layer, before logits)
- Return shim decisions to the caller via `done` event metadata

**Shim management is AgentOS's job**, not cortex's. Cortex is the execution runtime. When a shim file exists at the configured path, cortex loads and runs it; when not, cortex runs without gating.

---

## What cortex is NOT responsible for

Important clarifications so scope doesn't creep:

1. **Access control.** Cortex stores opaque bytes. Whether a given shard should be loaded for a given query is the caller's decision. Memex caller enforces privacy invariants (DM exclusion, per-user shard isolation, attribution stripping at ingest). Cortex just loads what it's told.

2. **Shim training.** AgentOS's shim-management subsystem (`project_agentos_shim_management.md`) trains shims. Cortex just loads the ONNX output.

3. **Agent orchestration.** Conversations, turn management, tool invocation, retries — all AgentOS's kernel. Cortex is stateless per request from the agent's perspective.

4. **Bob's voice / persona / register.** System prompts, voice calibration, illustrated states — those live in RingHub and AgentOS layers.

5. **Attribution policy.** Whether a shard contains user-identifying content or attribution-stripped content is determined at shard *construction* (outside cortex). Cortex doesn't parse shard content; it just attends over cache entries.

6. **Retrieval result resolution.** Memex converts `(shard_id, offset, length)` tuples to source text via sidecar. Cortex doesn't do this — it returns the tuples, caller resolves.

---

## Future (v2+) — not for v1, but know they're coming

Documented in integration memory for context; no v1 implementation needed:

- **Sigmoid attention model** — v1 uses Qwen 7B base with softmax + 4-token sink workaround. A native sigmoid-attention model would eliminate the sink hack. See `project_memex_architecture.md`.
- **1.58-bit BitNet support** — for the FPGA deployment path (`project_zynqberry_bitnet_memex.md`). Distant v2+.
- **Per-corpus memex fleet coordination** — multiple cortex instances serving different memex corpora (`project_memex_per_corpus.md`). Each instance stands alone; coordination happens at AgentOS layer.
- **Confidence calibrator** — an optional post-processing shim on retrieval scores to produce calibrated relevance probabilities. Quality optimization, not v1 requirement.

---

## Handoff checklist for cortex-Claude

Before writing code:

- [ ] Read integration memory `MEMORY.md` index, especially: `project_memex_architecture.md`, `project_agentos_api_contract.md`, `project_who_says_problem.md`, `project_silence_as_first_class.md`, `project_cortex_ffn_shims.md`
- [ ] Skim existing `api-spec.md` in cortex/ — much of it is correct, just confirm nothing got orphaned
- [ ] Inspect existing code: what's implemented, what's stubbed, what's TODO
- [ ] Confirm which engram pieces (QuantizedKvCache, retrieval) need porting
- [ ] Decide v1 deployment target: same binary serves both retrieval and generation via flags, or separate binaries? Probably same binary; simpler ops.

For v1 implementation:

- [ ] Retrieval mode — `forward_traced` + position ranking
- [ ] TurboQuant KV cache compression (port from engram)
- [ ] `POST /v1/retrieve` endpoint
- [ ] `POST /v1/cache/*` endpoints (confirm implementation matches `api-spec.md`)
- [ ] 4-token sink padding for shard ingest
- [ ] Generation mode SSE streaming per API contract
- [ ] Silence-as-first-class handling (empty text stream, silent=true in done event)
- [ ] Shim loading mechanism (even if v1 runs with no shim configured)
- [ ] Request-id propagation, observability per API contract

## Sequence recommendation

If building fresh, roughly:

1. **Retrieval mode first** — this is the novel piece, highest risk, and the Monday 4090 arrival is the event that validates it. Get a minimal working `forward_traced` + attention ranking against a small test cache. Smoke-test against the bhs-corpus ingested at small scale.

2. **Then TurboQuant** — once retrieval works, add the compression layer. This lets retrieval scale from small-test-cache to Harmonizer-scale.

3. **Then cache HTTP API** — wrap the in-process retrieval behind the HTTP endpoints. AgentOS can start calling cortex over HTTP instead of in-process.

4. **Then generation SSE** — enhance existing `/v1/chat/completions` to support SSE streaming with silence-as-first-class.

5. **Then shim integration** — wire in the shim loader. Start with a stub that always returns `should_respond=true` to unblock end-to-end flow; real shim training happens in AgentOS-land.

Each step proves a piece of the stack end-to-end. No big bang; stepwise validation.

---

## References (integration memory)

- `project_memex_architecture.md` — memex-as-classifier, 4-token sink, TurboQuant
- `project_agentos_api_contract.md` — the HTTP+SSE contract AgentOS serves (cortex is downstream but must be consistent)
- `project_memex_per_corpus.md` — per-corpus memex instance pattern
- `project_who_says_problem.md` — attribution-stripping-at-ingest, retrieval metadata safety
- `project_silence_as_first_class.md` — why empty text stream is a first-class outcome
- `project_cortex_ffn_shims.md` — shim architecture cortex loads and invokes
- `project_agentos_shim_management.md` — AgentOS is the control plane; cortex is the runtime
- `project_do_justice_to_the_medium.md` — the animating principle; retrieval quality is community-facing and must be right
