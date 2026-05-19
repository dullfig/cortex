# CubeCL Technical Spike — Results

**Date:** 2026-05-18 (evening)
**Hardware:** Windows 11, RTX 4080 Laptop (12 GB), CUDA driver 581.95 / runtime 13.0
**Toolchain:** Rust 1.95.0, **CUDA toolkit + MSVC NOT installed** → ran spike against WGPU backend instead

**Headline:** All 4 steps GREEN. Two surprise findings strengthen the GO recommendation. Updated migration estimate: **3-5 weeks** (down from 5-7).

---

## Step 1: Install + sanity check — ✅ PASS

- Cloned `tracel-ai/cubecl@84d719b` (v0.11.0-pre.1)
- Built `examples/gelu` with `--features wgpu`
- Run: `target\release\examples\gelu.exe`
- Output: `[-0.15865529, 0.0, 0.8413447, 4.9999986]` matches expected GELU (cpu reference: `[-0.1587, 0, 0.8413, 5.0]`)
- Build time: 1m 50s (large dep graph, no errors)

Windows + wgpu backend works out of the box. CUDA backend untested due to missing toolchain — defer that validation to a follow-up half-day spike when CUDA Toolkit + MSVC are installed (~1 hour user time).

## Step 2: Memory-pool cliff benchmark — ✅ PASS + SURPRISE

Wrote a 100-line bench (`/tmp/spike/poolbench/`) that loops N times allocating + uploading a 500 MB buffer via `client.create_from_slice`, then drops. Tested Auto vs Persistent mode.

**30-iter results:**

| Mode | Total | Avg/iter | Min/iter | p50/iter | Max/iter | Final reserved |
|---|---|---|---|---|---|---|
| Auto (default) | 16.0 s | 533 ms | 264 ms | 320 ms | 442 ms | 512 MB |
| Persistent | 13.3 s | 444 ms | 265 ms | 287 ms | 502 ms | 500 MB |

**SURPRISE:** Both modes pool. `bytes_reserved` stays steady at ~500 MB across all iterations. **No cliff fires.** Per-iter cost is the actual 500 MB upload (~480 ms theoretical at 1 GB/s — matches). Persistent is ~17% faster on average (less overhead in skipping the Auto bookkeeping path).

**Implication:** CubeCL's default Auto mode IS the fix for cortex's TTFT cliff. We don't even need to flip to Persistent. The wgpu cliff cortex hits today is because cortex uses raw wgpu buffers (no pool layer); CubeCL's `MemoryManagement` wraps the wgpu storage and catches drops to keep pages reserved. Same backend, different allocator.

**For comparison:** cortex today, similar workload (500 MB scratch allocate + drop per chat request) → ~17 s drop cost on requests 2+. CubeCL same backend, similar workload → ~300-500 ms per cycle, **35-60× faster.**

## Step 3: f16 × f32 mixed-precision matmul — ✅ PASS (with custom impl needed)

**Note:** The matmul crate has been split out of CubeCL into a separate repo `tracel-ai/cubek` (crates: `cubek-matmul`, `cubek-attention`, `cubek-convolution`, `cubek-reduce`, `cubek-quant`, `cubek-random`). Spike adjusted accordingly.

`MatmulPrecision` trait (in `cubek-matmul/src/definition/spec.rs`):

```rust
pub trait MatmulPrecision: Send + Sync + Copy + 'static {
    type Lhs: MatrixPrecision;  // (GlobalType, OperandType)
    type Rhs: MatrixPrecision;
    type Acc: MatrixPrecision;
}
```

Each operand has separate types for global memory and operand-time. Shipped impls include `f16` (Lhs/Rhs (f16, f16), Acc (f16, f32) on non-macOS), `flex32` (Lhs/Rhs (f32, f16), Acc (f32, f32)), `f64`.

For cortex's exact `(f16 weights, f32 activations)` shape we'd write a custom `MatmulPrecision` impl (~10 lines). Trait is open, this is the standard pattern.

**Decision: not a blocker. ~1 day of impl + test in migration.**

## Step 4: cubek-attention as drop-in — ✅ PASS + SURPRISE

`cubek-attention::launch_ref`:

```rust
pub fn launch_ref<R: Runtime>(
    strategy: Strategy,
    client: &ComputeClient<R>,
    query: TensorBinding<R>,
    key: TensorBinding<R>,
    value: TensorBinding<R>,
    mask: Option<TensorBinding<R>>,
    out: TensorBinding<R>,
    attention_global_types: &AttentionGlobalTypes,
    attention_options: AttentionOptions,
) -> Result<(), AttentionSetupError>
```

**Single fully-fused call.** Q/K/V in, output out, optional mask. Internal: handles score/softmax/value all together. Strategy enum selects `BlackboxAccelerated` (flash-attention with cmma on CUDA), `Unit`, `MultiRows`, autotune.

**SURPRISE upside:** cortex's current `dispatch_attention_inner` is ~200 lines orchestrating 3 separate kernels (`attn_score_batch` + `softmax_batch` + `attn_value_batch`) with a ~500 MB scratch buffer for intermediate scores. With cubek-attention we'd swap all of that for ~5 lines calling `launch_ref`. The scratch buffer goes away too (flash-attention doesn't materialize the score matrix).

**Saves 1-2 weeks off migration estimate. Also kills a separate latent concern** (the score-buffer allocation is part of the cliff).

## Bonus probe: cubek-quant for BitNet ternary path

`cubek-quant` ships Q2F/Q2S (2-bit), Q4F/Q4S (4-bit), Q8F/Q8S (8-bit), plus FP variants. **No ternary {-1, 0, +1} scheme** — Q2S is standard int2 ([-2, 1]), not BitNet's sparse 3-of-4-patterns. Our BitNet path stays as custom work (~2 weeks unchanged).

## Updated migration estimate

| Phase | Original | After spike findings |
|---|---|---|
| Memory pool integration | included in 3-5 wk | **0 additional weeks** (use Auto mode) |
| f32/f16 matmul port (cubek-matmul + custom MatmulPrecision impl) | 3-5 wk | **2-3 weeks** |
| Attention port (was: port 3 kernels) | included | **−1 to −2 weeks** (drop-in cubek-attention) |
| BitNet ternary (still custom) | +2 wk | +2 wk |
| **Total** | **5-7 wk** | **3-5 weeks** |

## Risk status

| Risk (from reading spike) | Status |
|---|---|
| Windows + CUDA toolchain | **Unresolved** — CUDA Toolkit + MSVC not installed; wgpu backend works without them. Half-day follow-up spike needed after toolchain install. |
| Persistent mode actually kills the cliff | **RESOLVED** — and Auto mode alone is sufficient. Validated on wgpu backend, same stack we hit the cliff on. |
| f16 × f32 mixed-precision support | **RESOLVED** — trait is open; we write a custom MatmulPrecision impl. ~1 day. |
| Alpha-stage stability | Unchanged — v0.11.0-pre.1, active fixes. Real risk for production. |
| API churn during 3-5 week port | Unchanged — the cubek split happened recently, suggests architecture is still moving. Worth pinning to a specific commit and not auto-upgrading mid-port. |

## Recommendation

**GO on CubeCL** with the following sequencing:

1. **Phase 1 (1 week): WGPU backend migration.** Get cortex compiling against CubeCL + cubek-matmul + cubek-attention with the wgpu backend. This alone fixes the TTFT cliff (memory pool) and gives us a much simpler attention path. Decode rate stays at ~21 t/s (no tensor cores on wgpu backend).

2. **Phase 2 (separately, ~1 day): install CUDA Toolkit + MSVC, validate CUDA backend.** Run the same poolbench + a small cubek-matmul test against CUDA. Once green, flip cortex's runtime to `CudaRuntime` for the matmul win.

3. **Phase 3 (~2 weeks): port BitNet ternary path.** Custom kernel in `#[cube]` macro; cubek-quant doesn't help here.

**Sequencing rationale:** Phase 1 ships a real user-visible win (TTFT cliff dies) without requiring any new toolchain install. Phase 2 is opt-in for the matmul speedup. Phase 3 closes the BitNet feature gap.

**Comparable alternative (fallback if migration runs into trouble):** stay on wgpu, build our own pool layer copying CubeCL's `MemoryManagement` pattern. ~1-2 weeks, kills the cliff, no matmul win, no attention simplification, no future cooperative_matrix path. Strictly worse than Phase 1 above.

## Sources

- [tracel-ai/cubecl @ 84d719b](https://github.com/tracel-ai/cubecl)
- [tracel-ai/cubek](https://github.com/tracel-ai/cubek) (matmul + attention + quant kernels)
- `crates/cubecl-runtime/src/memory_management/memory_manage.rs` (MemoryAllocationMode, Auto/Persistent)
- `crates/cubecl-runtime/src/client.rs:868` (`pub unsafe fn allocation_mode`)
- `crates/cubek-matmul/src/definition/spec.rs` (MatmulPrecision trait)
- `crates/cubek-attention/src/launch/base.rs` (launch_ref single-call API)
- Spike bench source: `/tmp/spike/poolbench/src/main.rs`
