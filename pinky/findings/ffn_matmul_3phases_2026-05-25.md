# FFN matmul optimization — Phases 1-3 all landed (2026-05-25 evening)

Plan was `~/.claude/plans/giggly-chasing-melody.md`. All three phases
hit their gates and committed cleanly.

## Headline numbers (Qwen 0.5B Q4_K_M, 500w prompt, warm-state)

| | Legacy | Phase 1 | Phase 2 | Phase 3 | vs legacy |
|---|---|---|---|---|---|
| Wall (warm) | 1.07s | 0.74s | 0.53s | **0.51s** | **2.10x** |
| block.pass1 (norm+Q/K/V+RoPE+kv_write) | 68 ms | 34 ms | 21 ms | 21 ms | 3.24x |
| **block.pass2 (o_proj+FFN+residuals)** | **709 ms** | 403 ms | 203 ms | **195 ms** | **3.64x** |
| FFN microbench (standalone, 4864×896×526) | 14.0 ms | 5.4 ms | 6.0 ms | — | 2.31x→4.30x |

Per-pass timestamp data has block.pass2 down from **87% of prefill
GPU time → ~38%**. attn_score (~22ms cumulative) is now the
next-biggest single component.

## Commits on wgpu-29 branch

```
6bb4676 matmul: Phase 3 — fused gate+up shader (halves input bandwidth)
f89b58e matmul: Phase 2 — 2-per-thread output tile + hand-unrolled MADD
21e8d1f matmul: Phase 1 — vec4/vec2 packed loads in matmul_shared.wgsl
659230f matmul: env-var threshold + per-block GPU timestamp tracing
c055c3e matmul: shared-memory tiled GEMM shader, routes for n_tokens >= 16
0fa7ea0 wgpu-29: TTFT diagnosis — real bottleneck is naive matmul shader
9ed89bf wgpu 29: poll device between block loads to flush staging belt
894a0e6 M2 step 1: wgpu 24 → 29 upgrade (cubecl runtime addition still blocked)
```

## Key lessons (so we don't relearn them)

1. **Naga doesn't unroll loops, and neither does NVIDIA's driver.**
   Phase 2's first try (loop-based 2-per-thread MADD) was 2.31x;
   the hand-unrolled 32-MADD version of the same shader was 4.30x.
   Always unroll explicitly for hot inner loops in WGSL.

2. **Per-pass GPU timestamps are the diagnostic that worked.**
   Earlier diagnoses ("vkFreeMemory", "wgpu validation", etc.) all
   turned out wrong because we were measuring CPU-side wall time.
   `device.poll(Wait)` is genuinely the GPU running. Adding
   `encoder.write_timestamp` markers between blocks and
   `ComputePassTimestampWrites` on each pass gave us the ground
   truth that pointed straight at FFN matmul.

3. **NVIDIA L2 cache lines are 128 bytes — vec4 loads matter.**
   Scalar f32 loads pay per-instruction front-end cost even when
   the L2 transaction shape is the same. Phase 1's vec4
   cooperative-load pattern delivered 1.18x at small shapes and
   2.58x at FFN shape — the larger shape has more headroom for
   bandwidth optimization.

4. **Fusion savings are bounded by the size of the saved load.**
   Phase 3 saved one input-from-HBM load (the up_proj's). Input is
   ~1/6 of FFN-block memory traffic; the fused gate+up shader gives
   ~4% end-to-end. Worth doing but not the breakthrough.

## What's still slow

Looking at the warm prefill GPU breakdown (~510 ms total):

| Pass | Time | % |
|---|---|---|
| **block.pass2 (post-Phase-3)** | **195 ms** | **38%** |
| attn_score | 22 ms | 4% |
| block.pass1 | 21 ms | 4% |
| attn_value | 5 ms | 1% |
| softmax | 1.6 ms | <1% |
| (rest — pass-begin overhead, dispatch barriers, etc.) | ~265 ms | 52% |

block.pass2 still dominates, but a large slice of the remaining time
is in the "rest" — likely pipeline-barrier / pass-begin overhead from
the 5 passes per block × 24 blocks. May or may not be worth attacking.

The matmul shader itself is now ~4× faster on FFN shape, but the
*block.pass2* end-to-end only improved 3.64x because the o_proj,
norms, silu_mul, and residuals (the non-FFN-matmul parts of pass2)
are unchanged.

## Next-iteration options

- **Phase 4 — f16 activations**: cuts activation HBM traffic in half
  again. Bigger scope (touches every shader that reads/writes
  activations). Could be another 1.3-1.5x on the bandwidth-bound
  passes.
- **Tile larger matmul shapes**: 64×16 output tile (4-per-thread)
  could push FFN microbench past 4.30x if naga handles 4
  accumulators well. Cheap experiment, might or might not work.
- **Fuse silu_mul into the gate output**: write silu(gate)*up into
  the gate output buffer in the same shader. Eliminates one full
  intermediate (gate_out) and one dispatch (silu_mul). Probably
  worth 5-10% more.
- **Ternary matmul path** (for BitNet): same kind of optimization
  applies but in `ternary_matmul_batch.wgsl`. Currently untouched.
- **wgpu 29 Qwen 3B load OOM**: still outstanding. The optimizations
  above all apply once it's loadable.

## Verification commands

```bash
# Run all parity tests:
cargo test -p cortex --lib -- --test-threads=1 \
  matmul_shared_matches matmul_gate_up_fused \
  forward_block_gpu_matches_cpu_block forward_block_gpu_matches_cpu_bitnet_block

# Run FFN microbench:
cargo test -p cortex --lib --release -- --test-threads=1 \
  --ignored matmul_shared_vs_legacy_bench_ffn_shape --nocapture

# A/B legacy vs full stack end-to-end:
CORTEX_MATMUL_SHARED_THRESHOLD=999999 cortex-server ...  # legacy baseline
CORTEX_MATMUL_SHARED_THRESHOLD=16     cortex-server ...  # full stack (default)
```

`fwd_cache GPU per-pass cumulative` log lines show the per-pass
waterfall on every chat request when `RUST_LOG=cortex=info`.
