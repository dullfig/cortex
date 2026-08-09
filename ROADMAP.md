# Cortex Roadmap — Stages

**Date:** 2026-08-09
**Status:** Living document. Replaces the 2026-05-18 CubeCL-oriented roadmap
(archived in git history) — the CubeCL migration it planned was **never
taken**; the engine substrate was rebuilt with `vram-heap` + `device-probe`
instead (see Stage E). BitNet references are gone (un-merged 2026-05-29 to
`ternary-rs`).

> **⚠️ PARKED posture (read first).** Per the integration pin
> `state_of_project_2026-07-24`, **cortex is the next *code* phase, not the
> next *project* phase.** The engine substrate is done and stable; the
> project foreground is the mission/corpus track (which needs zero cortex).
> So **no stage below is actively SHIPPING right now** — this doc is the plan
> for when cortex resumes as the code critical path, plus the near-term code
> items that are worth doing whenever a session picks cortex up.

## How to use this doc

**Stages are sequential capacity tiers.** Each stage exits when its target
user-load is sustained at a defined quality bar.

**Concurrent work principle:** while Stage N is SHIPPING, Stage N+1 is
PLANNING, Stage N+2 is RESEARCHING (read-only, delegable to a separate
Claude session — different files, different concerns).

See `STATUS.md` for the snapshot of what cortex provides today. The
migration/phase history (vram-heap A–K, L chunker, M sizing, N refactor, O
QJL-V, P retrieve diagnostics, Q deploy, device-probe) lives in the git log
on `wgpu-29` and the `## Roadmap` checklist in `CLAUDE.md`.

---

## Stage E — Engine substrate (DONE 2026-06-13, via vram-heap not CubeCL)

The old roadmap's Stage 0/1 planned to fix cortex's engine problems — chiefly
the ~17s TTFT cliff — by migrating to CubeCL. That migration was **not
taken.** The problems were solved a different way, and the engine reached a
stable, deployable state on 2026-06-13 (device-probe consumption followed
2026-06-25).

**What shipped (all on `wgpu-29`):**
- **vram-heap free-list substrate** (Phases A–I) — RAII + coalescing
  allocator over pre-allocated heaps; 3-lane scheme for wgpu-29's
  same-backing R+RW rule; static weights heap; per-cache heaps. **Killed the
  TTFT cliff** (it was wgpu allocator churn, not GPU work — exactly what
  CubeCL was going to fix; vram-heap was the answer instead). This is the
  "mini-OS for GPU memory."
- **ParamsBufferPool** ring bump + `stats()` (Phase J).
- **Scaling-stage meters** (Phase K) — concurrent/pool/heap/GPU-busy +
  VRAM-budget gauges in `/metrics`. These are the Stage 2 trigger (below).
- **Device-aware heap sizing + binding-clamped chunker** (Phase M).
- **Source refactor** (Phase N) — `gpu_engine/` module dir; cortex-cloud
  split into modules.
- **QJL-256 V-side residual correction** (Phase O) — closed the polar
  attention-output cosine gap (was a future roadmap item; now done).
- **Retrieve diagnostics** (Phase P.1–P.3) — per-head sweep, offset-zero
  probe, by-shard holdout. Verdict: attention-score-readout retrieval is
  **method-limited** (see Stage M below).
- **Secure deploy path** (Phase Q) — Docker + Caddy TLS/auth; structurally
  validated, GPU-in-container unverified on a real Linux box.
- **device-probe boot integration** — device selection + VRAM budget +
  measured f16/bandwidth from the `device-probe → vram-heap → cortex` stack.

**Still open from the original Stage 1 *perf* goal (competitive single-stream):**
- **Tensor-core / cooperative-matrix matmul** — the old M6 CUDA target
  (~40–60 t/s decode). Matmul remains the bottleneck; device-probe now
  *measures* f16 speedup so the future precision-kernel switch has its input.
- **C3 packed-perf restoration** — `hidden_buf`/`projected` still f32 from
  the old BitNet "Option E" revert (no longer load-bearing); restoring
  packed-f16 recovers ~9% Qwen prefill.

**Exit (met for stability; perf partial):** engine is stable, deployable, no
TTFT cliff, no panics under normal load. The tensor-core decode win is the
one unfinished piece of the original perf bar.

## Stage M — Memory / retrieval (the real next code phase)

Not in the old roadmap, but this is what "cortex resumes" actually means
next — because the retrieve path's recall is **method-limited, not
quantization-limited** (f32 control R@10 ≈ 0.10; retrieval-heads do **not**
generalize on holdout, R@10 = 0.00, Phase P.3). Attention-score readout is a
dead end for recall.

**Direction (pins: `project_memex_architecture_direction`,
`_retrieval_method_bottleneck`, `_retrieval_heads_overfit`):**
- **Generation-as-index / synopsis-routing** — two-step retrieval (route on
  synopsis, generate grounded from real chunk text), not attention readout.
- **Memex foundation experiment** — synopsis/grep step-1 bake-off on the
  52-query holdout + no-answer adversarial queries; validate generation-as-
  index on Qwen-2.5-3B-Instruct before building machinery.
- **DEFECT to fix here:** large one-shot `cache/load` (~6K tokens) panics on
  wgpu's 65535 workgroup-dim limit (memex report, STATUS §4). Chunk the
  prefill dispatch + structured error + verify heap-free-on-failure.
- Move retrieval (bidirectional attention) + HierarchicalCache +
  consolidation from engram; `project_qk()` on TransformerModel.

## Stage 2 — Batched serving (PLANNING; start when Phase-K meters say so)

**Goal:** ~1000 concurrent users via continuous batching + multi-instance
sharding. **Trigger:** watch the Phase-K gauges (`cortex_concurrent_requests`,
GPU-busy %) — start when load approaches the single-instance ceiling, not
before.

**Work items:**
- Continuous-batching scheduler in cortex-cloud — pack N requests into one
  forward pass, each at its own position in its own KV cache.
- Variable-length attention (custom batch variant of the existing shaders).
- Multi-instance: fleet of N processes, user_id consistent-hashed to one.
- Load balancer in front (nginx or a tiny Rust router — TBD).
- Auth + per-user rate limiting; multi-tenant isolation (per-tenant cache
  keys + limits).

**Exit:** 1000 concurrent sustained across 2–4 instances, p99 TTFT <10s, p95
decode >30 t/s.

## Stage 3 — Cortex + AgentOS cache coordination (RESEARCHING)

Cortex stays hot-VRAM-only with a sane default LRU; AgentOS owns warm/cold
tiers + prefetch policy (it has the semantic knowledge cortex doesn't).
Cortex = *mechanism*; AgentOS = *policy*. Makes cortex *smaller*, not bigger.

**Cortex-side (small):**
- Configurable per-instance cache budget; default LRU eviction on overflow.
- **Eviction notification hook** so AgentOS's shadow doesn't drift.
- Bulk inspect endpoint with access/timing metadata; eviction-under-pressure
  tests.

**Exit:** ~5000 concurrent across the fleet; AgentOS reawaken (`cache_load`
→ prefill) <5s p95; cortex never OOMs (eviction before VRAM exhaustion).

## Stage 4 — Production fleet (PARKED until Stage 3)

**Goal:** RingHub-scale (20k members → 1–3k concurrent peak).
Multi-region deploy; zero-downtime config reload / blue-green; cache-pool
backup/restore; on-call playbook + alerting beyond `/metrics`; capacity
planning; per-tenant cost attribution.

**Exit:** cortex runs as a real production service, not a research engine.

## Stage 5+ — Exploratory (low-priority, anytime)

Real possibilities, not commitments — pick up on slack or when a use case
forces it.

- **RetrievalAttention / ANN-over-KV** (arXiv 2409.10516) — validate on
  integration's stack first; 4-phase roadmap to a swappable
  `AttentionBackend` (CPU index + GPU compute). Pin:
  `project_retrieval_attention_modularization`.
- **Device-probe-driven pipeline specialization** — tile/workgroup sizes as
  wgpu pipeline-overridable constants from the `DeviceProfile` (a
  37-pipeline `const`→`override` refactor); and a native-f16-arith matmul
  variant gated on measured f16 speedup (cortex is f16-storage/f32-compute
  today, so this is the tensor-core win from Stage E).
- **Bit-packed 3-bit angle representation** for PolarQuant — ~12× vs ~7.5×.
- **Tensor parallelism** — split one model across N GPUs; only if cortex
  grows toward 70B+ on consumer hardware.
- **Ternary / BitNet + FPGA** — moved out with the 2026-05-29 un-merge; lives
  in `ternary-rs` now. The Zynqberry play is that repo's concern, not cortex's.

---

## What this roadmap deliberately does NOT include

- **High-throughput anonymous public serving** (vLLM-style). Different design
  point. Cortex is optimized for stateful conversations with curated context
  (Bob/Librarian shape); we scale horizontally through Stage 4 rather than
  match continuous-batching+PagedAttention, which our shim/cache primitives
  don't fit.
- **PagedAttention** — wrong cost/benefit for our persistent-cache + reawaken
  access pattern.
- **Training** — cortex is inference-only. Training-adjacent shim work (e.g.
  a "should-I-reply" classifier) happens outside cortex.

---

## How a Claude session uses this doc

> "I'm picking up cortex work for a session. Where do I start?"

1. Read `STATUS.md` for what cortex provides today (and the parked posture).
2. The engine substrate (Stage E) is done; the live next-code-phase is
   **Stage M** (memory/retrieval) — that's where net-new cortex work goes.
3. Stages 2–4 are load-gated: don't start Stage 2 until the Phase-K meters
   say the single-instance ceiling is near.
4. For a parallel research session, pick a Stage 5+ item (e.g.
   RetrievalAttention on integration's stack) and produce a structured report.
