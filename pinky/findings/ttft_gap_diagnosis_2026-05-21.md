# TTFT gap diagnosis — 2026-05-21 morning

## Question

After pooling `BlockScratch` (commit bd477cc) drove `scratch_us` from 5551µs (cold) to 1µs (hot), curl TTFT for `chat_completions` with a ~500w prompt was still 21s. GPU forward itself was only 3.1s. Where did the other ~17s go?

## Method

Added `gen.*` tracing in `cortex-cloud::main::generate_stateless_gpu`: `Instant::now()` before/after the prefill `forward_full_gpu_with_cache_inject_returning_hidden` call, before/after `finalize_logits`, before/after `sampler.sample`, plus explicit `drop(prefill_hidden)` timing. Ran `probe_ttft_stages.sh` at 50w/500w/2000w prompts, two runs each.

## Result

The 17s is **inside `forward_full_gpu_with_cache_inject_returning_hidden`, after the stage-timings log line but before the function returns**. Cross-referenced wall-clock timestamps between the GPU-side `fwd_cache stage timings` line (logged last inside the GPU function before return) and the wrapper-side `gen.prefill` line (logged immediately after the call returns):

| Prompt | GPU `total_us` (inside fn) | Wrapper `prefill_fwd_us` (observed) | Implied gap |
|---|---|---|---|
| 50w (76 tokens) | 0.60s | 0.60s | <1ms |
| 500w (526 tokens) | 3.07s | 20.43s | **17.36s** |
| 2000w (2026 tokens) | 12.76s | 20.51s | **7.75s** |

The gap shrinks as GPU work grows. At 50w the GPU function is fast and buffers must be recycled/small enough not to trip vkFreeMemory — no cliff. At 500w/2000w the gap is roughly constant absolute cost (~17s ceiling) that the longer GPU compute eats into.

`prefill_finalize_us` is ~17ms, `prefill_sample_us` is ~200µs, `prefill_drop_hidden_us` (Vec<f32> drop on CPU heap) is <500µs. The wrapper-side stuff is fine.

`scratch_us` stays at 1-2µs across all three sizes — `BlockScratch` pool is doing its job.

## What's in that gap

Between the `tracing::info!("fwd_cache stage timings")` line and the function's `return Ok(...)` only one thing happens: function-local variables drop. The drops are:
- `BlockScratch` — now pooled (`PooledBuffer` fields), confirmed cost 1µs each
- `hidden_buf` — raw `wgpu::Buffer` from `gpu.device.create_buffer_init` at line 1631/1744 of `gpu_engine.rs`. ~4MB for 500w, ~16MB for 2000w
- `normed_buf` — raw `wgpu::Buffer` from `gpu.device.create_buffer` at line 1636/1749. Same size as `hidden_buf`
- `staging` — raw `wgpu::Buffer` from `gpu.create_staging_buffer` at line 1642/1755. `MAP_READ | COPY_DST`. Same size

Three NVIDIA `vkFreeMemory` calls per forward × ~5s each = ~15-17s. Matches the gap exactly.

## Fix

Pool the three. `hidden_buf` and `normed_buf` use `STORAGE | COPY_SRC` — same `ScratchPool` already wired for `BlockScratch`. `staging` uses `MAP_READ | COPY_DST` — needs a separate `ScratchPool` instance with that usage flag.

Caveats:
- `hidden_buf` is currently allocated via `create_buffer_init` (write data at create time). With a pool, pattern becomes acquire → `queue.write_buffer` to populate. One extra queue submit per forward, negligible.
- `staging` is mapped for readback; need to confirm `PooledBuffer`'s `Deref` works across `staging.slice(..)` / `unmap()` correctly. Should — `Deref<Target=wgpu::Buffer>` covers method calls.

Call sites to wire: 6 `hidden_buf` + 4 `normed_buf` + 2 `staging` across `gpu_engine.rs` (greps from yesterday).

## Two-cliff confirmation, restated

1. **Cliff #1**: `BlockScratch` field drops. **Fixed** (commit bd477cc, `ScratchPool` + `PooledBuffer`).
2. **Cliff #2**: `hidden_buf` / `normed_buf` / `staging` drops. **Not yet fixed**. This is what the 17s curl wall represents.

Both cliffs are the same root cause (NVIDIA `vkFreeMemory` cost). Same fix shape. Expected wall-time after fix: ~3.5s for 500w (3.1s GPU + finalize + decode), ~13.5s for 2000w. Decode-only requests already fast (50w is 2s end-to-end, including model overhead).

## Orthogonal but related: the compute-pass collapse plan

`giggly-chasing-melody.md` targets the 3.1s GPU compute itself (510 compute-passes/forward → ~150 via single-pass-per-block). Independent win, ~2-3x on the GPU side. Different code paths from the pool work.

Order to attack: kill cliff #2 first (drops curl TTFT from 21s → 3.5s — the user-visible win), then collapse passes (drops it from 3.5s → ~1.5s).
