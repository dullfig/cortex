# Cortex / Memex / RingHub Roadmap

**Date**: 2026-04-12
**Status**: Living document. Updated as decisions are made.

## The stack

```
RingHub (web platform, members, posts, events)
     │
     ▼
Memex (orchestration: ingestion, shard management, position-to-source mapping)
     │                              │
     ▼                              ▼
Librarian (cortex-server)     Bob (cortex-server)
1B/7B model, 4090             32B model, A100/rented
--enable-cache --enable-retrieve    (stateless, default flags)
Port 8081                     Port 8080
```

## What's done

| Component | Status | Location |
|---|---|---|
| cortex inference engine | ✓ Complete | `cortex/` |
| forward_traced (attention capture) | ✓ Complete | `cortex/src/layers/trace.rs` |
| Pre-softmax Q·K^T capture | ✓ Complete | `cortex/src/layers/attention.rs` |
| Q5_1 / Q4_1 dequantization | ✓ Complete | `cortex/src/ops/dequant.rs` |
| cortex-cloud HTTP server | ✓ Complete | `cortex-cloud/src/main.rs` |
| Cache pool + 4 endpoints | ✓ Complete | `cortex-cloud/src/main.rs` |
| Multi-shard composition | ✓ Complete | `cortex-cloud/src/main.rs` |
| Retrieval mode | ✓ Complete | `cortex-cloud/src/main.rs` |
| Sink token padding | ✓ Complete | `cortex-cloud/src/main.rs` |
| Startup flags (--enable-cache/retrieve) | ✓ Complete | `cortex-cloud/src/main.rs` |
| cortex-local (in-process provider) | ✓ Complete | `cortex-local/` |
| Boundary classifier (f1=0.824) | ✓ Validated | `pinky/concept-boundaries*` |
| Morsel retrieval (3/3 connections) | ✓ Validated | `pinky/morsel-retrieval/` |

## What's left — in dependency order

### Tier 0: Blocking decisions (Daniel)

- [ ] **Choose the librarian model**. Softmax Qwen 1.5B (available now,
  sink padding handles the artifact) vs sigmoid-attention model (cleaner
  retrieval, need to find GGUF, cortex needs ~20 lines of sigmoid
  support). Everything downstream of this decision depends on it because
  KV vectors are model-specific and re-ingestion is required on model
  switch.

- [ ] **Buy or rent GPU**. 4090 ($1,600 buy) or A4000/A5000 ($87-160/mo
  rent). The librarian needs a GPU for real-time retrieval; CPU at 5
  tok/s is fine for development but not for production. This unblocks
  all deployment work.

### Tier 1: Cortex work (this Claude)

- [ ] **Sigmoid attention support** (if sigmoid model chosen). Add
  `sigmoid_inplace` alongside `softmax_inplace` in `attention.rs`,
  gated by a model-config flag read from GGUF metadata. ~20 lines,
  ~1 hour. Only needed if a sigmoid model is selected.

- [ ] **Grammar-constrained tool calling** on the 32B Bob deployment.
  Force valid JSON output when the model invokes tools. This is the
  gate between "demo" and "deployable" for the concierge. Larger
  effort — needs research into grammar-guided decoding techniques
  compatible with cortex's sampler. ~2-3 evenings.

### Tier 2: Memex (new project, needs a new Claude session)

- [ ] **Create C:\src\memex** project. Depends on cortex (inference
  engine) and engram (memory engine). The orchestration layer that
  makes the librarian useful.

- [ ] **Ingestion pipeline**. New content published on ringhub →
  tokenize → forward pass through librarian model → cache/append to
  the appropriate shard. Runs in background. Records position-to-source
  mapping in sled.

- [ ] **Shard management**. Create/manage shared shards (platform
  knowledge, events, wiki) and per-user shards. Load/evict based on
  usage. The lifecycle semantics from CACHE-LIFECYCLE.md.

- [ ] **Query routing**. @Bob in a conversation → determine which
  shards to compose (shared + participant caches) → call librarian
  retrieval → fetch original text → call 32B Bob with retrieved context.

- [ ] **Position-to-source sidecar**. The lookup table that maps
  (shard, offset) → source-id. Binary search over sorted spans.
  Memex owns this, cortex returns raw positions.

### Tier 3: Bootstrap corpus (Daniel + Memex)

- [ ] **Curate barbershop knowledge**. The Harmonizer archives, BHS
  glossary, coaching material, famous quartet history, contest results.
  This is Daniel's domain expertise applied to content curation.
  Bob's personality comes from what he's been fed.

- [ ] **Ingest bootstrap corpus** through the librarian model into
  shared shards. First real test of the ingestion pipeline at scale.

### Tier 4: AgentOS integration (agentos Claude)

- [ ] **Lifecycle manager** in AgentOS. Spawn, route to, and tear down
  per-user Bob instances with the right context. This is the multi-user
  gate. Daniel is working on this with agentos-Claude.

- [ ] **models.yaml config** for cortex. Already defined by
  agentos-Claude (see `cortex-cloud/models-example.yaml`). Wire it
  into the AgentOS startup.

- [ ] **Tool definitions** for the concierge. search_events,
  get_calendar, get_member_profile, etc. These are ringhub-specific
  and live in AgentOS.

### Tier 5: Deployment

- [ ] **Deploy librarian** on rented GPU (or local 4090). Run
  cortex-server with `--enable-cache --enable-retrieve`. Load the
  librarian model. Verify retrieval works at production latency.

- [ ] **Deploy Bob** on rented A100 (or second GPU). Run cortex-server
  with default flags (stateless). Load the 32B model. Verify generation
  quality.

- [ ] **End-to-end test**. Member posts on ringhub → ingested by memex
  → @Bob in a conversation → librarian retrieves relevant content →
  Bob responds with context. The first time the full stack works.

## What can happen in parallel

```
Daniel: curate bootstrap corpus ──────────────────────┐
Daniel: research sigmoid models ─────────────────┐    │
                                                 │    │
cortex-Claude: sigmoid support (if chosen) ──────┤    │
cortex-Claude: grammar-constrained decoding ─────┤    │
                                                 │    │
agentos-Claude: lifecycle manager ───────────────┤    │
                                                 │    │
memex-Claude (new session): ingestion pipeline ──┼────┘
                            shard management     │
                            query routing        │
                            position-to-source   │
                                                 │
                           ALL CONVERGE HERE ────┘
                                 │
                           End-to-end test
                                 │
                              SHIP IT
```

## Timeline estimate (evening-sized chunks)

| Work | Evenings | Who |
|---|---|---|
| Sigmoid attention in cortex (if needed) | 1 | cortex-Claude |
| Grammar-constrained tool calling | 2-3 | cortex-Claude |
| Memex ingestion pipeline | 2-3 | memex-Claude |
| Memex shard management | 1-2 | memex-Claude |
| Memex query routing | 1-2 | memex-Claude |
| Position-to-source sidecar | 1 | memex-Claude |
| Bootstrap corpus curation | 3-5 | Daniel |
| Bootstrap corpus ingestion | 1 | memex-Claude + Daniel |
| AgentOS lifecycle manager | 2-3 | agentos-Claude |
| AgentOS tool definitions | 1-2 | agentos-Claude |
| Deploy + end-to-end test | 1-2 | all |
| **Total** | **~15-25 evenings** | |

At 4-5 evenings per week with parallel work across Claude instances:
**~3-5 weeks to ship.** The critical path runs through memex (the new
project that doesn't exist yet) and the bootstrap corpus (Daniel's
curation work that no one else can do).

## The decisions that shorten the timeline

1. **Pick softmax Qwen 1.5B now** instead of waiting for sigmoid.
   Saves ~1 week of research + validation. Accept re-ingestion later
   if sigmoid is worth it. Cost of switching: ~1 hour of GPU time
   for re-ingestion at launch scale.

2. **Start memex tomorrow** rather than building more cortex features.
   Cortex is feature-complete for v0. The bottleneck moved to memex
   and AgentOS weeks ago and every cortex refinement since then
   (while valuable) has been off the critical path.

3. **Curate the bootstrap corpus in parallel** while memex is being
   built. Daniel's domain expertise is the one resource that can't
   be parallelized with Claude instances. Starting the curation now
   means it's ready when the ingestion pipeline lands.
