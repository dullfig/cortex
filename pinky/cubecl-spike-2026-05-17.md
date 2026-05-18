# CubeCL Spike — Go/No-Go Report

**Date:** 2026-05-17 (late session)
**Author:** cortex-claude
**Status:** Reading-only spike complete. Decision deferred for fresh-mind review.

## Why we considered CubeCL

Two perf walls hit on 2026-05-17 both pointed at the abstraction layer
beneath us:

1. **Matmul ceiling.** Bench showed cortex Qwen 3B Q4_K_M at ~21 t/s
   decode on a 4080 Laptop, vs llama.cpp ~60-80 t/s same hardware. Bisect
   (`pinky/tools/probe_ttft_stages.sh` and per-stage skip flags in
   `dispatch_attention_inner`) confirmed matmul is the bottleneck, not
   attention. Closing this needs cooperative_matrix / WMMA / tensor cores.

2. **wgpu/NVIDIA vkFreeMemory cliff.** Every `chat_completions` request 2+
   pays ~17s in buffer drops (proven not GPU work — explicit
   `device.poll(Wait)` after readback = 19µs). Pool-the-cache attempt:
   regressed decode 21 → 12 t/s. Pool-the-scratch attempt: cliff moved
   from `BlockScratch::drop` to the remaining `hidden_buf/normed_buf/staging`
   drops (same 17s, different drop site). The cost is essentially fixed
   per-request regardless of which buffer drops first.

The hope: CubeCL solves both — it owns a device-side memory pool AND
ships an optimized GEMM with cooperative_matrix on CUDA.

## Findings

### Q1: Memory pool — GO

CubeCL exposes a first-class `MemoryManagement` in
`cubecl-runtime/src/memory_management/memory_manage.rs`:
- `MemoryConfiguration` with strategies: `SubSlices`, `ExclusivePages`,
  `Custom { pool_options: Vec<MemoryPoolOptions> }`
- `MemoryAllocationMode::{Auto, Persistent}` — **Persistent mode holds
  pages and never returns memory to the driver during normal operation**
  (per maintainer). Exactly the "mini-OS for GPU memory" pattern.
- `cleanup(explicit)` is opt-in; default behavior is to keep allocations
  alive.

Caveat: issue #90 ("memory strategy too greedy") shows OOM on 6 GB cards
with default tuning under Llama 3.1. Tuning is real work but the knobs
exist.

### Q2: Cooperative matrix — GO on CUDA only

- **CUDA backend** uses cmma / PTX WMMA. Their `simple_sync_mma` kernel
  is the reference; their SOTA-matmul blog reports "nearly always
  outperforming cuBLAS/CUTLASS" on RTX 4080. **This is the matmul fix.**
- **WGPU backend**: explicitly NOT supported per README ("Tensor Cores
  acceleration isn't supported on WebGPU yet").
- **Vulkan backend** (separate from WGPU, uses `cubecl-spirv`): no
  `VK_KHR_cooperative_matrix` path visible. Blog notes Vulkan compiler
  constraints force line-size 4, suggesting they haven't enabled WMMA-class
  ops on Vulkan yet.

Migrating to CubeCL → CUDA-only on the 4080 Laptop. CUDA is available on
that hardware, so this is fine for development. We'd lose Vulkan/Metal
portability we never used and AMD support drops.

### Q3: DSL maturity — Conditional

`#[cube]` macro supports arithmetic, conditionals, vector ops, indexing,
generics over `Float`/`Numeric`, plus shipped libs `cubecl-std`,
`cubecl-matmul`, `cubecl-linalg`.

What we'd port from cortex (15 shaders):
- RMSNorm, RoPE, SiLU, ReLU² × up, KV write, f32 matmul (with f16 weights)
- 3 attention kernels (score, softmax, value) + batched variants
- Polar/QJL compressed-KV attention (we'd keep these as custom)

What's net-new R&D:
- **Ternary 2-bit-packed matmul** (BitNet 1.58b path) — no precedent in
  CubeCL ecosystem. Integer bit-ops supported in IR but no int8/ternary
  GEMM in `cubecl-matmul`.
- **f16 weights × f32 activations mixed-precision** — undocumented in
  `cubecl-matmul` (only ~42% docs coverage).

Framework state:
- v0.10.0 (May 2026), ~2.1k stars, 3-6 month release cadence
- Self-described "alpha … still a lot of rough edges"
- UB / memory-mgmt fixes landing each release
- Real-world OOM bugs filed

## Migration cost

| Phase | Effort |
|---|---|
| f32/f16 parity (Q4_K_M Qwen): 15 shaders + ~5 engine integration points | 3-5 weeks |
| BitNet ternary path port | +2 weeks |
| **Total** | **5-7 weeks** |

## Recommendation

**GO with CubeCL on the CUDA backend.** Both walls fall in one move:
cooperative_matrix closes the 21 → 60+ t/s decode gap, and `Persistent`
pool kills the 17s `vkFreeMemory` cliff.

Comparable alternative: stay on wgpu, build pool-everything ourselves
(~1-2 weeks). Kills the cliff. Zero matmul win — stuck at 21 t/s.

## Risks unanswered by reading-only spike

Need a **1-day technical spike** before committing 5-7 weeks:

1. **Does CUDA backend compile on Windows?** Spike didn't install; CubeCL
   primarily targets Linux. Windows CUDA toolchain interaction unverified.
2. **Does `Persistent` mode actually kill the cliff?** Need a 10-line
   alloc-free hot-loop benchmark to confirm. If yes, validates the whole
   thesis. If no, CubeCL gives us only the matmul win — we'd still need
   our own pool layer on top.
3. **f16 × f32 mixed-precision in `cubecl-matmul`** — undocumented.
4. **Alpha-stage stability for a production server** — UB / memory fixes
   per release. Could double the porting window.
5. **API churn during the 5-7 week port** — recent Burn 0.20 refactor
   (May 2026) shows the API is moving.

## Suggested decision tree

```
1-day technical spike → answers risks 1, 2, 3
  ├─ green → commit to 5-7 week migration
  └─ blocker → fall back to wgpu pool-everything (1-2 weeks, 21 t/s ceiling)
```

## Sources

- [tracel-ai/cubecl GitHub](https://github.com/tracel-ai/cubecl)
- [CubeCL SOTA Matmul blog](https://burn.dev/blog/sota-multiplatform-matmul/)
- [Burn 0.20 release notes](https://burn.dev/blog/release-0.20.0/)
- [memory_management/memory_manage.rs](https://raw.githubusercontent.com/tracel-ai/cubecl/main/crates/cubecl-runtime/src/memory_management/memory_manage.rs)
- [Issue #90 wgpu memory greedy](https://github.com/tracel-ai/cubecl/issues/90)
- [Issue #1275 simple_sync_mma PTX](https://github.com/tracel-ai/cubecl/issues/1275)
- [cubecl-matmul docs](https://docs.rs/cubecl-matmul)
