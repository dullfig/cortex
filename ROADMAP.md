# Cortex Roadmap — Stages

**Date:** 2026-05-18
**Status:** Living document. Replaces the 2026-04-12 stack-overview roadmap (archived in git history at commit before this one).

## How to use this doc

**Stages are sequential capacity tiers.** Each stage exits when its target user-load is sustained at a defined quality bar.

**Concurrent work principle:** while Stage N is SHIPPING (focused human + Claude execution), Stage N+1 is PLANNING (design decisions, sketches), Stage N+2 is RESEARCHING (read-only, can be delegated to a separate Claude session without coordination — different files, different concerns).

The pattern: when Stage N exits, Stage N+1's planning has already converged and shipping starts immediately. No "what do we do now" gap.

See `STATUS.md` for the snapshot of what cortex provides today. See `pinky/cubecl-migration-plan-2026-05-18.md` for Stage 1's detailed file-by-file plan.

---

## Stage 0 — Foundation (DONE 2026-05-17/18)

- Phantom-work audit: deleted ~15 unused shaders / pipelines / dispatchers / tests
- Telemetry MVP: Prometheus `GET /metrics` (request counts, tokens, TTFT histogram)
- Status board (`STATUS.md`) in cortex-claude / agentos-claude / memex-claude format
- TTFT cliff diagnosed (wgpu/NVIDIA `vkFreeMemory` accumulated-state cost, not GPU work)
- Baseline bench harness (`pinky/tools/bench_baseline.py`, `pinky/tools/probe_ttft_stages.sh`)
- CubeCL spike: reading + technical Go/No-Go, both GREEN. Migration sequenced.
- M1 scaffold on `cubecl-migration` branch: dep added, feature flag wired, 394/394 tests green
- **Exit:** CubeCL migration planned + branch live + Go decision committed

## Stage 1 — Engine (SHIPPING, ~3-5 weeks)

**Goal:** ~100 concurrent users sustained with competitive single-stream perf on commodity GPU.

**Work items (in flight on `cubecl-migration` branch):**
- M2: resolve wgpu 24→29 version collision, wrap `BlockScratch::allocate` via CubeCL `MemoryManagement` → TTFT cliff dies
- M3: swap `dispatch_attention_inner` for `cubek-attention::launch_ref` → 3 kernels become 1, scores scratch goes away
- M4: swap float matmul for `cubek-matmul` + custom `MatmulPrecision` impl for (f16 weight, f32 activation)
- M5: port RMSNorm / RoPE / SiLU / ReLU² / KV-write / bias-add to `#[cube]` macro DSL
- M6: install CUDA Toolkit + MSVC → flip runtime to `CudaRuntime` → tensor cores (cmma) → decode 21 → 40-60+ t/s
- M7: port BitNet ternary path (custom — `cubek-quant` has 2-bit but not ternary semantics)
- M-final: merge `cubecl-migration` → `master`, delete legacy wgpu paths, tag release

**Production hardening (parallel with M2-M7):**
- Panic recovery: cortex panics on wgpu device-loss today (observed twice on 2026-05-17). Need graceful restart-on-failure, request retry.
- Concurrency gauge in `/metrics` (closes the 3-of-3 threshold-metric gap from `project_cortex_v1_perf_threshold.md`)
- `cortex_local::CortexLocal::complete()` over GpuEngine (currently slow CPU `model.generate()` path — direct AgentOS integration win)

**Exit:** 100 concurrent users sustained on a single beefy GPU (24 GB+ VRAM), TTFT <5s for typical prompts, decode ≥40 t/s on CUDA backend, no panics under sustained load.

**Concurrent research-in-parallel for Stage 2** (can be a separate Claude session):
- Read `vllm/core/scheduler.py` — how do they pick which requests to batch each iteration?
- Read flash-attention's variable-length API + `cubek-attention`'s strategy enum — can we batch sequences of different lengths in one kernel call?
- Sketch cortex's request-batching API surface — what does `chat_completions` look like internally when 8 requests share a forward pass?
- Decide consistent-hashing scheme for multi-instance sharding (Karger? Rendezvous? Jump?)

## Stage 2 — Batched serving (PLANNING, ~4-6 weeks after Stage 1)

**Goal:** ~1000 concurrent users via continuous batching + multi-instance sharding.

**Work items:**
- Continuous batching scheduler in cortex-cloud — pack N requests into one forward pass, each at their own position in their own KV cache
- Variable-length attention via `cubek-attention` (or custom variant if cubek doesn't expose it)
- Multi-instance: cortex-cloud runs as a fleet of N processes, user_id consistent-hashed to one instance
- Load balancer in front (could be nginx, could be a tiny Rust router — TBD during Stage 2 planning)
- Auth + per-user rate limiting (was a `[?]` in STATUS.md)
- Multi-tenant isolation: per-tenant cache pool keys, per-tenant rate limits

**Exit:** 1000 concurrent users sustained across 2-4 cortex-cloud instances, p99 TTFT <10s, p95 decode rate >30 t/s.

**Concurrent research-in-parallel for Stage 3:**
- Tiered cache designs — Redis LFU eviction, PostgreSQL buffer pool clock-sweep
- Reawaken latency budget — what's an acceptable UX for a returning user? 1s? 5s? Affects tier promotion/demotion policy
- Cache placement strategies — replication vs sharding vs hierarchical
- Read PagedAttention paper for what NOT to do (cortex's persistent-cache + reawaken is the alternative)

## Stage 3 — Cortex + AgentOS cache coordination (RESEARCHING, ~3-5 weeks after Stage 2)

**Reframed 2026-05-19** from "cortex internal tiering" to "cortex + AgentOS coordination." Daniel observed that AgentOS already has the semantic knowledge (which Bob, which user, what's likely next) that cortex doesn't have. Letting cortex invent its own LRU heuristics + disk store is reinventing what AgentOS already does well. The cleaner separation:

- **Cortex** = hot VRAM cache + eviction policy + notification hooks (the *mechanism*)
- **AgentOS** = cold-tier owner + prefetch policy + working-set knowledge (the *policy*)

This makes cortex *smaller* than the original Stage 3 plan, not bigger.

**Goal:** ~5000 concurrent users via cortex/AgentOS protocol. Cortex stays hot-only with a sane default LRU; AgentOS owns warm and cold tiers (system RAM + memex/byte-store + disk).

**Cortex-side work items (small):**
- Configurable per-instance cache size budget (cap VRAM by total bytes or shard count)
- Default LRU eviction policy when budget exceeded
- **Eviction notification hook**: when cortex self-evicts shard X, notify AgentOS so its shadow doesn't drift (webhook or SSE channel — TBD)
- Bulk inspect endpoint: `GET /v1/cache/` already lists; may add timing/access metadata
- Tests for eviction-under-pressure + notification delivery

**AgentOS-side work items** (lives in agentos repo, not cortex):
- Cortex cache shadow — AgentOS tracks which shards cortex currently holds
- Eviction policy — when to evict from cortex (idle Bob, day boundary, etc.)
- Prefetch heuristics — warm a shard *before* it's needed based on Bob's planned action
- Token persistence per shard (probably already exists via memex/agentos-byte-store)

**Exit:** 5000 concurrent across the fleet. AgentOS handles cache-miss reawaken in <5s p95 (push tokens to cortex via `cache_load`, cortex prefills, ready for chat_completions). Cortex never OOMs — eviction kicks in before VRAM exhausted.

**Why this is cleaner than internal tiering:**
- Cortex doesn't need a disk-tier subsystem (AgentOS already persists tokens)
- Cortex doesn't need a warm-RAM tier subsystem (AgentOS can re-feed quickly enough)
- AgentOS's semantic knowledge beats cortex's blind LRU
- The protocol becomes a stable contract both projects can iterate against
- vLLM does this internally because it serves anonymous users and has no upper-layer with intent knowledge; AgentOS has that knowledge, so we exploit it

**Concurrent research-in-parallel for Stage 4:**
- Fleet ops patterns (Kubernetes vs Nomad vs systemd, blue/green deploy strategies)
- Backup/restore for the persistent cache pool (or: skip — AgentOS persists tokens; cortex cache is purely an acceleration layer that can be rebuilt)
- Multi-region replication patterns
- Cortex-AgentOS protocol spec (would have to be designed in Stage 3 anyway; refining for cross-region is a Stage 4 task)

## Stage 4 — Production fleet (PARKED until Stage 3, ongoing thereafter)

**Goal:** RingHub-scale operations (20k members → 1-3k concurrent at peak).

**Work items:**
- Multi-region deployment
- Zero-downtime config reload, blue/green deploys
- Backup/restore for cache pool (currently in-memory only; restart loses everything)
- On-call playbook, alerting hooks beyond `/metrics`
- Capacity planning automation
- Cost monitoring (per-tenant compute attribution)

**Exit:** Cortex runs as a real production service, not a research engine.

## Stage 5+ — Exploratory (LOW-PRIORITY PARALLEL, anytime)

These are real possibilities, not roadmap commitments. Pick up when there's slack or when a use case forces the issue.

- **BitNet + FPGA pipeline** — the Zynqberry play. Pipeline-parallel BitNet across one-FPGA-per-layer. Genuinely elegant algorithm-hardware fit; multi-quarter hardware project. Cortex's BitNet kernels already exist (shipped 2026-05-10); missing piece is FPGA toolchain. Likely 6-12 months dedicated work IF it becomes a priority.
- **Tensor parallelism** — split a single model across N GPUs. Only needed if cortex grows toward 70B+ models on consumer hardware. ~6-8 weeks when needed.
- **Polar/QJL on V dequant** (currently K-only) — closes the cosine-similarity gap on retrieval attention. Was a `[ ]` in STATUS.md. Small work, ~1 week.
- **Flash-attention variants for polar cache path** — cubek-attention may not natively handle the compressed-K shape. If retrieval path needs more perf, custom kernel.
- **Bit-packed 3-bit angle representation** for PolarQuant — u8 → 3-bits, ~12× compression vs current ~7.5×. Listed in STATUS.md as `[ ]`.

---

## What this roadmap deliberately does NOT include

- **High-throughput public serving** (vLLM-style, millions of users, thousands of concurrent). Different design point. Cortex is optimized for stateful conversations with curated context — Bob/Librarian shape. If RingHub explodes to a public service, the architecture stays the same up through Stage 4 and we lean on horizontal scaling. We don't try to match vLLM's continuous-batching+PagedAttention combo because our shim/cache primitives don't fit that abstraction.
- **PagedAttention** — wrong cost/benefit for our access pattern.
- **Training** — cortex is inference-only. Training-related shim work (e.g., training a "should-I-reply" classifier) happens outside cortex.

---

## How a Claude session uses this doc

> "I'm picking up cortex work for a session. Where do I start?"

1. Read `STATUS.md` for what cortex provides today.
2. Find which Stage is currently SHIPPING in this doc.
3. Open the linked plan doc (e.g. `pinky/cubecl-migration-plan-2026-05-18.md` for Stage 1) for file-by-file detail.
4. If the user wants help on the SHIPPING stage → focus implementation work.
5. If the user wants to free your time for parallel research → pick a "research-in-parallel" bullet from the next stage and produce a structured report.
