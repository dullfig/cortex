# TTFT diagnosis — wgpu 29 lands, but real bottleneck is matmul (2026-05-25)

## Path so far

Three diagnoses, each overturned by the next:

1. **vkFreeMemory cliff** (the original 2026-05-18 diagnosis on commit 5a5ca89).
   Fix: pool buffers. Built BlockScratch pool, ParamsArena, extended pool
   to hidden/normed/staging. *Didn't move user-visible TTFT.* Pool/arena
   foundation committed regardless — it's correct, just not the cliff.
2. **wgpu 24 validation/build cliff** (2026-05-24 — based on
   nvidia-smi showing 0% GPU util the entire 22s of a 500w probe on
   wgpu 24). Hypothesis: wgpu was doing CPU-side command buffer build
   for 20+ seconds, GPU never engaged. Migration to wgpu 29 was the
   fix.
3. **Naive matmul kernel** (2026-05-25, today). After wgpu 29 landed
   and BitNet 2B loaded successfully, the stage timings show the real
   pattern.

## What stage timings show on wgpu 29 + BitNet 2B (500w prompt)

```
embed_us       = 852       (CPU embedding lookup)
io_alloc_us    = 7,805     (hidden/normed/staging buffer create)
scratch_us     = 25,300    (BlockScratch::allocate)
record_us      = 3,286     (CPU command buffer build — wgpu 29 records
                            36 blocks × ~15 dispatches in 3.3 ms)
submit_us      = 4,601     (queue.submit returned in 4.6 ms)
readback_us    = 20,407,856 (device.poll(Wait) blocking on GPU fence
                            for 20.4 SECONDS)
total_us       = 20,449,716
```

CPU prep is ~40 ms. The 20s is **inside the GPU fence wait**.
`device.poll(Wait)` is `vkWaitForFences` with infinite timeout —
the thread sleeps until the GPU signals done. It returns at 20s
because the GPU genuinely takes 20s to finish the submitted work.

## What was wrong with the wgpu 24 hypothesis

nvidia-smi sampled at 500 ms intervals. cortex's prefill submits
~600 tiny dispatches each completing in microseconds. Between
dispatches the GPU is briefly idle (pipeline barriers); sampling
catches those windows and reports 0% util. The GPU IS doing real
work — just bursty enough that 500ms-resolution polling sees mostly
idle samples followed by a sustained-busy phase as work backs up.

The "10s idle + 10s boost" pattern wasn't "10s of CPU build" — it
was sampling artifact + the GPU's clock-up latency on the heavier
backed-up phase.

## What's actually slow

The current matmul shader (`shaders/matmul.wgsl`) is naive
per-row-per-token. No tiled GEMM, no shared-memory blocking, no
register accumulation. For Qwen 3B / BitNet 2B prefill, the
matmul is **~95% of the GPU time**. llama.cpp does the same work
on the same hardware in **~1-2 seconds**. We take **~20s**. That's
the 10-20x gap.

`giggly-chasing-melody.md` had this as "Fix 2 — out of scope of
pass-collapse, deferred." We pivot to Fix 2.

## What got committed today

- wgpu-pool branch: BlockScratch pool, ParamsArena, hidden/normed/staging
  pool extension (commits bd477cc, 58f2ca5, b605126, 19b0db1). Correct
  foundation for the future PagedAttention work even though it didn't
  move TTFT.
- wgpu-29 branch: cherry-picked aaac056 (full wgpu 24 → 29 migration
  from cubecl-migration) + 9ed89bf (mitigation poll for Qwen load OOM).
  This branch is what production should be on going forward — it's
  not faster than wgpu 24 in the cliff sense, but it's the future
  cortex needs to be on for everything else (better validation,
  better adapter handling, supports newer GPUs).

## What's next

Tiled GEMM matmul shader rewrite. Standard 16×16 tile blocking
with shared-memory caching of A and B tiles, per-thread register
accumulation. The shader exists today as a single-purpose dispatch
inside `dispatch_matmul_in_pass`. The tile shader needs to be
written from scratch (or adapted from a known-good reference like
naga's wgsl examples / wonnx).

**Open question for tomorrow**: do we write it ourselves, or do we
pull in a known-good wgsl GEMM (e.g. from `wgpu_gemm`, `gpgpu-rs`,
or the candle wgpu backend)? Writing from scratch is ~1 day of
careful indexing + shared memory management. Adapting an existing
shader is ~half day of plumbing. Either way it's bounded.

## Qwen still OOMs on wgpu 29 load

Separate issue from TTFT. wgpu 29's allocator trips OOM around
block 34/36 even with 12 GB VRAM free, with or without polling
between block loads. BitNet 2B (1.2GB GGUF) loads fine, which is
how we measured the rest. To debug Qwen: try `MemoryHints::MemoryUsage`
instead of `Performance`, or instrument wgpu's hal layer to see
which specific allocation it's rejecting.
