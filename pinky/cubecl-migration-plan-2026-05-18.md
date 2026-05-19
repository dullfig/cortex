# cortex → CubeCL Migration Plan

**Date:** 2026-05-18
**Branch:** `cubecl-migration` (off `master` at commit `2e193bc`)
**Estimate:** 3-5 weeks, 7 milestones, each with rollback point
**Scope:** "combined" sequencing — design for both wgpu AND CUDA backends from day 1; develop primarily on wgpu (no toolchain blocker); flip CUDA on at any point during port; ship matmul win as part of the same migration.

## Scope at a glance

| Module | Lines today | Migration verdict |
|---|---|---|
| `cortex/src/layers/gpu_engine.rs` | 4853 | Refactor — most complex piece |
| `cortex/src/layers/gpu_polar.rs` | 1929 | Refactor — keep semantics, swap backend |
| `cortex/src/compute/wgpu_backend.rs` | 1255 | Mostly delete — CubeCL replaces |
| `cortex/src/layers/gpu_polar_kv_cache.rs` | 701 | Refactor — swap buffer types |
| `cortex/src/layers/gpu_bitlinear.rs` | 263 | Refactor for `R: Runtime` |
| `cortex/src/layers/gpu_floatlinear.rs` | 260 | Refactor for `R: Runtime` |
| `cortex/src/layers/gpu_kv_cache.rs` | 251 | Refactor — swap `wgpu::Buffer` for CubeCL `Handle` |
| 22 `.wgsl` shaders (1130 lines) | 1130 | Most port to `#[cube]` macro; a few delete (cubek replaces) |
| `cortex-cloud/src/main.rs` | ~3600 | Touch only the engine-facing call sites; rest unchanged |

**~10K lines of Rust + WGSL touched, with ~5-6 distinct integration points.**

## Sequencing principle

**Develop primarily against wgpu backend** (no CUDA toolchain install needed).
**Design every abstraction as `R: Runtime` generic** from day 1, so flipping to CUDA at the end is a feature-flag change, not a rewrite.
**Ship one milestone at a time** — each merges back to `master` as an independently-correct step, even if migration isn't complete. Avoids the "3-week branch that never merges" failure mode.

## Milestones

### M1 — Foundation scaffold (~3 days)

**Goal:** cortex compiles with CubeCL as a dependency. No behavior change.

- Add `cubecl`, `cubecl-runtime`, `cubek-matmul`, `cubek-attention` to `cortex/Cargo.toml` (pinned to specific commits — see Risks below)
- Add feature flags: `cortex-cloud --features cubecl-wgpu` (default), `--features cubecl-cuda` (opt-in)
- Create `cortex/src/compute/cubecl_backend.rs` skeleton — empty `pub struct CubeClBackend<R: Runtime>` with `pub fn new() -> Self`
- Verify build: `cargo build --workspace --release`
- Verify tests still pass (394/394)
- **Rollback point:** revert single commit if dep upgrade breaks anything

**Pass criterion:** clean build + all tests green + CubeCL crates appear in `Cargo.lock`.

### M2 — Memory-pool swap (~5-7 days, expanded from original 3-4)

**Goal:** Kill the TTFT cliff. Smallest possible change that ships user-visible value.

**M1 surfaced a complication that lives in M2:** cortex uses wgpu 24, CubeCL uses wgpu 29 (git rev). Adding cubecl's `wgpu` feature flag to cortex's Cargo.toml causes the two wgpu versions to coexist in the dep tree; wgpu-hal's DX12 backend has incompatible windows-rs subcrate deps between the versions and fails to compile on Windows. M1 scoped down to `cubecl-core` only (DSL macros, no runtime) to land the scaffold without this collision.

**M2 resolves the wgpu collision** before doing the memory-pool swap. Three options:

1. **Upgrade cortex to wgpu 29** — touches all of `gpu_engine.rs`, `gpu_floatlinear.rs`, `gpu_bitlinear.rs`, `gpu_kv_cache.rs`, `wgpu_backend.rs`. wgpu 24→29 has API changes (renamed types, restructured submodules). Estimate ~2 days for the upgrade alone. Recommended path — clean.
2. **Vendor a CubeCL fork patched to wgpu 24** — keep cortex on wgpu 24 but maintain a patched cubecl. Adds permanent maintenance burden. Not recommended.
3. **Extract cubecl-using code to a separate workspace member** — `cortex-cubecl/` crate has wgpu 29 + cubecl; calls into cortex via traits. Architecturally clean but doubles the code surface. Defer to a future refactor if option 1 is painful.

**Then the memory-pool swap** (the original M2 content):

- Wrap `BlockScratch::allocate` to allocate via CubeCL's `MemoryManagement` instead of raw `wgpu::Buffer`
- Keep all existing wgpu shaders untouched (they bind buffers; CubeCL handle → wgpu buffer interop is supported by cubecl-wgpu)
- Run `pinky/tools/probe_ttft_stages.sh` — confirm cliff is gone
- Run `pinky/tools/bench_baseline.py` — confirm decode rate unchanged (no regression)
- Run all 394 cortex tests
- **Rollback point:** revert the BlockScratch::allocate change; everything else stays as-is

**Pass criterion:** 500w prompt TTFT drops from ~21s → ~3-5s (just actual GPU work). Decode rate unchanged. All tests green.

**This is the single biggest user-visible win in the entire migration.**

### M3 — Attention drop-in (~3-5 days)

**Goal:** Replace cortex's 3-kernel attention dispatch with `cubek_attention::launch_ref`. Eliminates ~700 lines of code (3 wgsl shaders + dispatch logic) and the 500MB scores scratch buffer.

- Add `cubek-attention` integration to `gpu_engine::dispatch_attention_inner`
- Validate numerical correctness: existing `dispatch_attention_matches_cpu_gqa` test must still pass (within float tolerance)
- Validate end-to-end: `forward_block_gpu_matches_cpu_block` must still pass
- Delete `attn_score_batch.wgsl`, `softmax_batch.wgsl`, `attn_value_batch.wgsl` (or keep behind a fallback flag for one milestone, delete in M4)
- Decide what to do with polar attention (`attn_score_polar_batch.wgsl` etc.) — likely keep as-is since cubek-attention probably doesn't handle the polar/compressed-K case. Document this in the row.
- **Rollback point:** keep old kernels alive behind `cfg!(feature = "legacy-attention")`; flip back if cubek-attention breaks parity

**Pass criterion:** all 394 tests green + decode of "Hello, how are you?" produces same text as before (to within sampler determinism).

### M4 — Float matmul via cubek-matmul (~5-7 days)

**Goal:** Replace `dispatch_matmul_into` (and its 3 variants) with `cubek_matmul::launch_ref` + custom `MatmulPrecision` impl for cortex's (f16 weights, f32 activations) shape.

- Write `impl MatmulPrecision for CortexF16Weights` (~10 lines)
- Route `dispatch_matmul_into` through cubek-matmul
- Validate numerical correctness against existing CPU-reference matmul tests
- Delete `matmul.wgsl`, `matvec.wgsl`
- On the wgpu backend, perf should be roughly equal to current (no tensor cores)
- **Rollback point:** keep `dispatch_matmul_legacy_inner_in_pass` alive behind `cfg`; route off cubek-matmul if it regresses

**Pass criterion:** all tests green + decode rate within ±10% of M2 baseline on wgpu backend (no regression).

### M5 — Remaining primitives port (~5-7 days)

**Goal:** Port RMSNorm, RoPE, SiLU, ReLU², KV-write, bias_add, add_broadcast shaders from WGSL to `#[cube]` macro DSL. These are small (10-100 lines each) and have CPU reference implementations to validate against.

- One commit per primitive (easy review, easy revert)
- Read Burn's implementations for DSL idioms (per spike findings)
- Delete the corresponding `.wgsl` files as each lands
- Run cortex test suite after each commit

**Pass criterion:** all tests green after each primitive lands. End of M5: ~8 of cortex's 22 wgsl files deleted (the non-polar batch ones).

### M6 — CUDA backend activation (~1 day)

**Goal:** Install CUDA Toolkit + MSVC on dev machine. Flip cortex to CUDA runtime. Validate the matmul win.

- Install CUDA Toolkit 13.x + MSVC Build Tools (~1 hour user time)
- Run `pinky/cubecl-poolbench` on CUDA backend — confirm same pool behavior
- Build cortex with `--features cubecl-cuda` — confirm clean build
- Run `pinky/tools/bench_baseline.py` — confirm decode rate jumps from ~21 t/s to 40-60+ t/s (matmul win from cmma)
- Run cortex test suite on CUDA backend
- **Rollback point:** `--features cubecl-wgpu` still works; both backends coexist behind feature flags

**Pass criterion:** all tests green on CUDA + bench shows 2-3× decode rate improvement vs wgpu.

### M7 — BitNet ternary path port (~10-14 days, longest pole)

**Goal:** Port the BitNet 1.58b ternary matmul path. cubek-quant doesn't have native ternary support — this is custom work in `#[cube]` macro.

- Implement ternary 2-bit-packed activations + i32 accumulation kernel
- Validate against existing `models/ggml-model-i2_s.gguf` end-to-end test (BitNet 2B inference)
- Delete `quantize_absmax_batch.wgsl` + `ternary_matmul_batch.wgsl`
- Validate prefill + decode produces coherent text on BitNet 2B
- **Rollback point:** keep old ternary kernels alive behind `cfg` until ternary port is validated end-to-end

**Pass criterion:** BitNet 2B inference produces same text as before, decode rate on CUDA backend matches or beats current ~20 t/s.

### M-final — Merge back to master + tag release

- Merge `cubecl-migration` into `master`
- Delete legacy `cfg!` flags that gated rollback
- Delete `cortex/src/compute/wgpu_backend.rs` (replaced by CubeCL)
- Update STATUS.md: mark CubeCL migration `[x]`, update perf numbers
- Tag release (e.g., `v0.2.0-cubecl`)

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| **CubeCL API churn during 3-5 wk port** (recent cubek split happened in May 2026) | Pin to specific commit hash in Cargo.toml. Don't auto-upgrade mid-migration. After M-final, plan a separate "upgrade CubeCL" PR. |
| **Alpha-stage stability** (v0.11.0-pre.1) | Each milestone is independently mergeable. If a milestone reveals a CubeCL bug we can't work around, halt at that milestone — partial progress still ships value. |
| **Polar/QJL attention doesn't fit cubek-attention** | Keep polar path on existing wgpu shaders. Document as `[x] cortex's polar attention stays on wgpu shaders` in STATUS.md. Worst case it's a feature gap for retrieval workflows on CUDA backend — fix in a follow-up. |
| **BitNet ternary port (M7) takes longer than estimated** | M7 is the last milestone — if it slips, the rest of the migration still ships. Worst case BitNet stays on wgpu temporarily. |
| **CUDA backend Windows build fails** | wgpu backend is functional throughout. Worst case: ship migration without M6, decode stays at ~21 t/s but cliff is gone + code is CubeCL-ready. M6 becomes a follow-up when toolchain lands. |
| **Numerical regression vs CPU reference** | Every milestone has a parity test gate. Existing 394 cortex tests act as the regression suite. New CubeCL-side tests as needed. |

## What NOT to touch in this migration

- Shim API (3-phase dispatch) — orthogonal to GPU backend. Stays as-is.
- Telemetry MVP — `/metrics` endpoint. Stays as-is.
- Cache pool API on cortex-cloud — `GpuKvCache` interface stays; only its internal storage swaps from wgpu to CubeCL.
- HTTP routes — no API changes.
- gguf loader — CPU-side, no GPU dependency.

## Phase 1 / Phase 2 / Phase 3 mapping (Daniel's earlier framing)

| Phase | Milestones | User-visible win |
|---|---|---|
| 1 (wgpu, ~1 week) | M1 + M2 + M3 | TTFT cliff dies, attention simplified |
| 2 (~1 day) | M4 + M6 | Decode rate 21 → 40-60+ t/s on CUDA |
| 3 (~2 weeks) | M5 + M7 | All shaders ported, BitNet on new path |

## Total estimate

3-5 weeks active dev work, 7 milestones, each independently mergeable + rollback-able.

## Tonight's scope

M1 only — get cortex compiling with CubeCL as a dependency. No behavior change.
