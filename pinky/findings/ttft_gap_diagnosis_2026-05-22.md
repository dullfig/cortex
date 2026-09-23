# TTFT cliff — third diagnosis pass (2026-05-22)

## What we built today

1. **`ParamsArena`** (`cortex/src/compute/params_arena.rs`) — frame-allocator
   for the per-dispatch uniform buffers. One slab + bump cursor + reset per
   forward. 4 unit tests pass: 1000 acquires sub-second.
2. **`BindEntry` enum + `make_bind_group_with_bindings`** in
   `wgpu_backend.rs` — lets a bind group mix whole-buffer storage entries
   with slab-offset uniform entries.
3. **`bind_with_params` helper on `GpuEngine`** — universal pattern:
   acquire params slot, build bind group with storages + arena slot.
4. **Refactored 14 dispatch helpers in `gpu_engine.rs`** to use
   `bind_with_params` instead of `create_params_buffer`. (8 in
   `gpu_polar.rs` deferred — retrieve path, not on the chat-completion
   cliff.)
5. **Pooled `hidden_buf` / `normed_buf` / `staging`** in the two
   `*_returning_hidden` paths and in `forward_full_gpu`. Added a separate
   `staging_pool` (MAP_READ | COPY_DST) and added COPY_DST to the
   scratch_pool usage (`queue.write_buffer` needs it).
6. **`params_arena.reset()`** called at the top of each leaf forward
   entry point (7 sites).
7. Numerical-correctness tests pass: 5/5 on the critical forward and
   block-parity tests (Qwen-shape no-crash, attention vs CPU GQA,
   block vs CPU block including BitNet).

## What the cliff is

Same 17s wall on 500w probe as before. But now we know where it lives.

| Stage | n_prompt=526 timing (post-fix) | n_prompt=2026 (post-fix) |
|---|---|---|
| Prefill `fwd_cache total_us` | 3.07s | 12.83s |
| Wrapper `prefill_fwd_us` (gap to call) | 3.07s (gap = 160µs ✓) | 12.83s (gap = 506µs ✓) |
| Decode iter=1 `submit_us` | **17.19s** | **8.33s** |
| Decode iter=2 `submit_us` | 6.5ms (normal) | 1.1ms (normal) |
| Total wall | 20.5s | 21.3s |

The 17s previously hidden in the prefill function exit is now visibly
inside `queue.submit()` of the **first subsequent decode**, where wgpu
blocks for cleanup of the prior command buffer. Decode iter=2 onward is
fast — the cleanup is one-shot per heavy prefill.

The cliff scales **inversely** with prefill compute time: bigger prefill
→ longer GPU compute → smaller `submit_us` cleanup. The pattern is
consistent with "cleanup cost = constant ~17s overlapped with prior
GPU work, paid at next submit if cleanup hadn't finished by then."

Cleanup is most likely **descriptor-set teardown** for the ~600 bind
groups created during prefill (36 blocks × ~17 dispatches/block). Each
bind group is a Vulkan descriptor set. Free-on-retire is per-call
expensive on NVIDIA the same way `vkFreeMemory` is. Pooling the
underlying buffers doesn't help because the bind groups themselves are
new objects each forward.

## Why the bigger picture didn't change

The pool/arena infrastructure is **necessary** but **not sufficient**
for the user-visible TTFT cliff. Three real wins, none of which
the curl saw:

- BlockScratch pool: `scratch_us` 5551 µs → 1 µs (committed bd477cc)
- hidden/normed/staging pool: prefill function-exit drop ~17s → <1ms
  (this branch)
- ParamsArena: 600 vkAllocate/vkFree pairs → bump-cursor slab
  (this branch)

All three were real wgpu/NVIDIA cliffs that needed pooling. The
remaining cliff is one level deeper — descriptor sets.

## Two paths forward

### A) Bind group cache
Keep the current dispatch shape (510 dispatches per forward). Build
a cache keyed on (pipeline, [buffer_ids, offsets]) that returns a
shared bind group when the same configuration repeats. Most cortex
bind groups DO repeat — each block has the same shape (attn_norm,
Q/K/V, RoPE, score/softmax/value, gate/up/down, residuals). The 36
blocks differ only in *which* weight buffers they reference, but
those are stable across requests.

Risk: bind groups bind specific offset+size for params slabs, which
rotates per dispatch. So params bindings can't be cached; only
storage bindings can. Caching requires a hybrid where the storage
part of the bind group is reused and params get rebound dynamically
— this is `set_bind_group(group_idx, &cached_storage, &[dynamic_offset])`
with the params at a separate bind group index using dynamic offset.
Requires bind-group-layout changes in every shader.

### B) Pass collapse (`giggly-chasing-melody.md`)
Collapse 510 compute passes → ~36 (one per block). This also reduces
**bind groups** because each pass-begin needs its own. With pass
collapse, each block's bind groups go from ~17 to ~1-2 (one big
all-bindings group per pass). 17x reduction in descriptor set
churn at the same time as 17x reduction in pipeline-barrier
overhead.

Both fix the same root cause from different angles. (B) is also a
GPU-compute-speedup (2-3x estimated). (A) preserves the current
dispatch shape but requires shader-side bind group layout changes
(multiple bind groups per pipeline).

## Recommended next move

**B** (pass collapse). Plan is already written in
`~/.claude/plans/giggly-chasing-melody.md`. Three concrete wins:

1. Descriptor set churn drops 17x → cliff dies
2. GPU compute speeds 2-3x → 3.1s prefill becomes ~1.5s
3. No shader changes; mechanical refactor of dispatch helpers (already
   have `*_in_pass` variants for each)

Estimated effort: half a day. Risk: bounded — if data-hazard barriers
get missed in a merged pass, integration tests will catch (numerical
mismatch vs CPU reference), and we fall back to grouped passes split
at known hazards (attention triple).

## What ships with this work

Even though TTFT didn't move, the pool/arena foundation is correct
and necessary for any production path. ParamsArena will be how
cortex manages per-dispatch state forever — building it now means we
don't have to redo it under deadline pressure later. Pass-collapse
will then ride on top.

Suggest committing pool/arena + refactor to the `wgpu-pool` branch,
keeping it separate from main until pass-collapse lands and the
combined TTFT win is provable end-to-end.
