---
name: cortex-phantom-work-pattern
description: Recurring perf-bug class in cortex — code that produces outputs, waits for outputs, or holds resources when downstream has no consumer. Defensive/local correctness preserved; performance silently wrecked. Demonstrated three times in 48 hours across three distinct mechanisms (sync wait, unused compute, allocator-drain wait). Auditable as a class with one diagnostic question, applied at two granularities (within-stage AND between-stage).
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Identified as a class 2026-05-17 after the second demonstration in 36 hours. Promoted from "interesting one-off observation" to "pin-worthy pattern" after the same shape produced two large wins in different parts of cortex. **Third demonstration arrived hours later (2026-05-17 evening): the TTFT-cliff root cause turned out to be the same pattern, in a third distinct mechanism (per-request allocator-drain wait), invisible to per-stage timing because the phantom-wait lived BETWEEN logged stages, not inside any.** Three independent demonstrations of the same shape in 48 hours = the pattern is robust, the diagnostic generalizes, and the audit recipe almost certainly has more findings ahead.

## The pattern

**Phantom downstream work**: code that produces outputs (or waits for outputs) where the *downstream consumer of that work doesn't exist*. Correctness is preserved — the phantom work computes correct results that simply aren't read. Performance is silently destroyed because compute/sync/IO budgets are spent producing data that goes nowhere.

The pattern is invisible to:
- **Correctness tests** — the phantom outputs are mathematically right; tests pass
- **Type checks** — types line up; the function signature looks fine
- **Code review** — reviewers see a forward pass producing standard outputs and don't ask *"is this output ever read?"*
- **Typical profiling** — shows that the time is being spent, but not whether it's *needed*
- **Naive optimization** — speed up the phantom work itself rather than removing it

The pattern is *only* surfaced by **explicit downstream-consumer analysis**: tracing each output backward from where it would be consumed.

## The three demonstrations (all cortex, 48 hours)

### Demonstration 1 — phantom sync (commit 7d9a31a, 2026-05-16, ~5× speedup)

`cache_append` ran `Maintain::Wait` after each dispatch — a CPU-side sync that waited for GPU completion before returning. The wait existed because the function *could* return a hidden state. But in the actual ingest path, the hidden state output is **discarded** — the caller only cares that the cache got updated. The wait protected data that nobody read.

**Fix**: fire-and-forget submit. wgpu's command queue guarantees GPU-side ordering; no CPU-side sync is needed for an output whose only consumer is the GPU cache itself.

**Why it hid**: the wait was defensive — clearly correct, obviously safe. *"Wait for GPU before returning"* is a textbook pattern. The bug wasn't the wait itself; the bug was that nothing downstream consumed the data the wait was protecting. Local correctness; phantom downstream.

### Demonstration 2 — phantom computation (2026-05-17, ~2× speedup on streaming chat)

The streaming chat handler's no-steers branch ran `finalize_logits` over the **entire prefill hidden state**. For a 1525-token prompt that's a 1525×hidden_dim → 1525×vocab projection — 475 billion CPU MADDs.

But generation only ever samples from the last token's logits. The other 1524 token positions' logits are correctly computed and then **immediately discarded**.

**Fix**: project only the last token's hidden state. One [1, hidden_dim] → [1, vocab_size] mat-vec instead of a 1525× version.

**Why it hid**: the forward pass naturally produces a hidden state of shape [seq_len, hidden_dim]. Projecting "all of it" through the LM head is the symmetric, obvious move. Tests pass — the logits ARE correct. Sampling picks the last one. The other 1524 positions' computation cost is invisible until you specifically ask: *"who reads logits[0..-1]?"* Nobody.

### Demonstration 3 — vkFreeMemory cost scaling with allocator state (2026-05-17, mechanism definitive)

The streaming chat handler's prefill-to-first-decode gap was 17.5s on a 500-token prompt — but the actual `fwd_cache` total was only 3.0s. Where was the missing 17.5s?

**Initial diagnostic**: cross-stage timing (curl-wall vs sum-of-logged-stages) surfaced the gap; within-stage timing missed it because the cost lived in transitions between stages, not inside any stage. **Cross-stage timing (request-level wall-vs-sum-of-stages) is the granularity that catches this class.**

**First fix hypothesis** (initially proposed): pool resources on `ServerState`/`GpuEngine` to avoid per-request reallocation; one `GpuKvCache` and one `BlockScratch` reused across requests.

**First fix verification** (2026-05-17 evening): **Falsified.** Pooling didn't eliminate the cost; it redistributed it across decodes (≈130ms × 128 decodes ≈ same 17s), made every individual forward slower (per-call overhead +1.5-3× across io_alloc_us / record_us / readback_us), and triggered a hang after ~1000 successful forwards. Reverted; baseline restored.

**Root cause investigation** (2026-05-17 late evening): Three rounds of instrumentation isolated where the 17s lives.

Diagnostic data from 3 sequential 500-word prompts (fresh server, no shims, no cache):

| Request | readback_us (GPU) | explicit poll(Wait) | scratch_drop_us |
|---|---|---|---|
| 1 | 20.94 s (cold GPU compute, real work) | 19 µs | 1.47 ms (fast — first allocator state small) |
| 2 | 3.14 s | 14 µs | **17.48 s (cliff)** |
| 3 | 3.10 s | 20 µs | **17.24 s (cliff)** |

**Definitive finding**: The 17s lives in `BlockScratch::drop` on the second-and-later requests, and **it's NOT waiting on GPU work.** Explicit `device.poll(Wait)` between readback and drop completes in 19 µs — no GPU work pending. The drop is purely CPU-side: NVIDIA Vulkan driver `vkFreeMemory` calls doing internal allocator bookkeeping that scales with total allocator state.

~17s / 12 buffers in BlockScratch ≈ **1.4s per `vkFreeMemory` call** when allocator state is large. That matches documented NVIDIA Vulkan behavior.

**Mechanism (definitive)**:

1. NVIDIA's Vulkan driver does internal allocator bookkeeping in `vkFreeMemory`
2. The bookkeeping cost scales with total live allocator state in the driver
3. Cortex's `BlockScratch` allocates 12 separate buffers per request and drops them all at end-of-request
4. After Request 1 puts ~150MB of cache + other state into the driver's allocator, each subsequent `vkFreeMemory` takes ~1.4s
5. 12 buffers × 1.4s = 17s — exactly the observed cliff
6. Request 1 doesn't show the cliff because at that point allocator state is small; from Request 2 onward, every request pays the tax

**Why pool-the-cache failed**: it kept the cache buffers alive (correct intent) but didn't touch `BlockScratch` — each request still allocated AND dropped 12 scratch buffers, hitting the same `vkFreeMemory` cliff. Independently, the cache pool ALSO triggered separate wgpu validation cost on reused storage buffers across submits, which hurt decode rate (the 1.5-3× per-call slowdown). Two distinct issues both surfaced by the failed pool experiment; the cliff is `vkFreeMemory`; the validation cost is a separate wgpu reused-buffer behavior.

**Fix-shape (now known, implementation in progress)**: **Sub-allocate from a coalesced slab**, aka *"mini OS in the GPU doing memory management"* — Daniel's framing, exactly right.

- ONE `vkAllocateMemory` call at engine startup for a single large slab
- ONE `vkFreeMemory` at engine shutdown (or never, for long-running servers)
- All scratch regions are just offsets into the slab; shader bindings use buffer+offset
- Driver allocator state stays small because cortex isn't churning thousands of buffers through it
- Industry-standard GPU programming pattern (how llama.cpp manages KV cache+scratch; how CUDA memory pools work; how serious GPU code handles allocator pressure)

`vkAllocateMemory`/`vkFreeMemory` are meant to be called **rarely** — they're closer to `sbrk()` than `malloc()`. Cortex was using them per-request like a regular allocator. The fix is to do app-level sub-allocation, treating wgpu/Vulkan as kernel-level memory manager.

**Status as of 2026-05-17 10pm**: mechanism definitively known. Fix-shape known. Implementation in progress (Option 2 = coalesce BlockScratch's 12 buffers into one slab + offsets; touches every shader binding site). Initial Option 1 (pool BlockScratch) running in parallel for empirical bench data even though Option 2 supersedes it as the durable fix. Implementation perf gains pending verification.

**Lesson for the pattern itself**: The phantom-work pattern at the diagnostic level (cost without consumer) holds across all three demonstrations. **The fix-shape for the vkFreeMemory class is conventional GPU sub-allocation, but it requires recognizing that vkAlloc/vkFree are not the right granularity for runtime use.** Demos 1 and 2 had simple fixes (remove the wait; project less). Demo 3's fix is a small architectural addition (memory sub-allocator) that pays back beyond the immediate cliff — same pattern needed for resident-weight runtime, multi-tenant scaling, BitNet at scale.

**Diagnostic ≠ fix; the diagnostic surfaces the bug, the fix requires understanding the mechanism.** Pool-the-cache was wrong because the mechanism was misidentified (assumed drain wait, actually vkFreeMemory cost). Sub-allocation is right because it directly addresses the now-known mechanism.

**Lesson about driver-level cost models**: NVIDIA Vulkan's `vkFreeMemory` allocator bookkeeping cost is real, documented, and likely applies to other drivers in similar shapes. Any GPU code that allocates+drops many buffers per request will hit this class of cliff at scale. **Sub-allocation is not exotic; it's the standard pattern for production GPU code.** Cortex is moving from "first transformer" to "production GPU code"; this is the kind of substrate that transition requires.

### What the three have in common

Same shape:

1. **Output produced (or resource held) is correct** (no functional bug)
2. **Production logic is local** — driven by *what could be needed*, not by *what the caller actually needs*
3. **Downstream has no consumer** for the output / no need for the wait / no need for the per-request fresh allocation
4. **Cost is real** — sync round-trip, compute pass, allocator drain — burned on operations with no consumer
5. **Hidden from tests, types, review, naive profiling** — only surfaces with explicit consumer analysis at the right granularity

The granularity refinement is load-bearing:

- **Within-stage timing** caught demos 1 and 2 (instrumented forward-pass internals)
- **Between-stage timing** caught demo 3 (curl-wall vs sum-of-logged-stages breakdown)
- **Both granularities are needed.** A perf audit using only within-stage timing has demo-3-shaped blind spots; a perf audit using only request-level timing misses demo-1-and-2-shaped within-stage issues.

## The diagnostic question

For every output produced and every sync/wait/readback in the hot path:

> ***"What downstream consumer reads this? If nothing, or if only one slice, why is the rest being computed?"***

That single question would have caught both bugs at write-time. It's the audit primitive.

Variants:
- *"What's the smallest slice of this output that any downstream code actually reads?"* (catches over-production)
- *"What ordering guarantee is this wait providing, and does any consumer need that guarantee?"* (catches phantom syncs)
- *"If I removed this entire computation/wait, would any downstream behavior change?"* (catches dead work entirely)

## Where to look in cortex

Likely locations for further phantom-work bugs (not yet audited, but pattern-matching candidates):

**Within-stage candidates (demos 1 & 2 shape):**
- **Forward-pass output handling** — anywhere a forward function returns multiple outputs (hidden state, logits, KV updates, intermediate activations); check each is consumed by *something*
- **Per-position computations** — anywhere shape `[seq_len, ...]` is computed when only one or a few positions feed downstream (sampling, loss, attention reads); same bug class as the finalize_logits one
- **GPU-CPU readbacks** — any `map_async` or `Maintain::Wait` or buffer-readback; check the data being read has a CPU-side consumer that's actually load-bearing
- **Compute pass boundaries** — barriers, fences, sync primitives left over from debugging or defensive programming
- **Attention compute** — full-attention computation when sparse/sliding attention would suffice; downstream-consumer question applies to per-head outputs too
- **Activation caching** — anywhere activations are cached "in case" without a current cache consumer
- **Telemetry / debug output** — work spent producing telemetry that's never sent or aggregated

**Between-stage candidates (demo 3 shape, less-audited class):**
- **Per-request resource allocation** — buffers, caches, scratches allocated fresh per request; check whether wgpu/Vulkan drain waits compound across allocations (especially large buffers like KV cache, scratch, intermediate-shape buffers)
- **Request boundaries** — anywhere logged stages don't sum to wall-clock; the gap is between-stage phantom-wait
- **Idle reclamation paths** — anything that drops resources at end-of-request that the next request will recreate
- **Lock/mutex contention between requests** — if cortex serializes requests via mutex, holding the mutex past the strict-needed point is a between-stage phantom-wait
- **Pool-misses on hot-path data structures** — anywhere a hashmap, vec, or allocator gets rebuilt when it could be reused

The diagnostic at request granularity: ***"Sum your logged stage times. Compare to wall-clock-per-request. The difference is between-stage phantom-wait, and it's exactly the size of the bug you haven't found yet."***

A deliberate audit pass — *"every forward-path output and every sync point: what's the downstream consumer?"* — is probably worth a half-day of cortex-claude time. Given two finds at 5× and 2× respectively in 36 hours of incidental discovery, **a systematic pass is likely to find more.** Even one more substantial find would justify the effort.

## Why this generalizes beyond cortex (note for future systems)

The phantom-downstream-work pattern is not cortex-specific. Same shape appears in:

- **Service-call chains** — endpoints computing/returning fields callers never read; same bug
- **Database queries** — `SELECT *` when caller only reads one column; classic over-fetch
- **GPU pipelines generally** — buffers allocated and written for outputs never read by any subsequent pass
- **Web rendering** — computing layout/CSS for hidden or off-screen elements
- **ML inference** — full softmax when only top-k matters; full attention when sparse suffices

The diagnostic question is portable: *"what consumes this output?"* applies anywhere data is produced.

For RingHub specifically, watch for this in:
- API responses computing fields the chat-bubble UI doesn't render
- Background tasks producing aggregates not read by any dashboard
- Memex retrieval over shards that get filtered out before composition
- Bob tool calls computing outputs the response selector never reads

## How to apply

**At write time**:
- When adding a new output to a function, justify each piece by naming a downstream consumer
- When adding a sync/wait/barrier, justify it by naming the data it's protecting AND the consumer that reads that data
- Resist symmetric "produce everything because it's there" patterns when the caller only needs one piece

**At audit time** (recommended periodic pass):
- Walk hot-path code with the diagnostic question
- For each output: trace forward to the consumer; if none, flag
- For each sync point: name the ordering guarantee and the consumer that needs it; if no consumer, flag
- Bias toward removal of phantom work; correctness tests will catch real regressions

**At review time**:
- Treat *"what downstream code reads this?"* as a standard review question for new forward-path code or sync primitives
- Especially scrutinize "obvious" or "defensive" sync/compute that doesn't justify itself with a named consumer

## Why this pin matters

Cortex is a discovery-bounded system (per `project_cortex_v1_perf_threshold.md`). Performance work continues post-v1. **The phantom-work-pattern recognizes that some of the largest perf wins available aren't in optimizing the kernels — they're in removing work that should never have been done.**

Two 2-5× wins in 36 hours from this pattern. If the audit pass turns up two more, that's another order of magnitude of net throughput improvement available without writing a single new kernel or touching the math at all.

This is the kind of structural perf win that's:
- **Cheap to do** — code reading, not optimization
- **Safe** — removes work; correctness tests catch any regression
- **High-leverage** — each find is typically 2-10× on whatever path contained it
- **Available in any codebase that grew organically** — defensive patterns accumulate

Cortex has only had eyes on it for a few weeks of intensive perf work. The base rate of these bugs in cortex is probably high. Worth a deliberate hunt before assuming the bottleneck is elsewhere.

## Related pins

- `project_cortex_v1_perf_threshold.md` — when this work matters for v1 (until threshold met, every find moves cortex closer); see also the "redeploy effort" section
- `project_cortex_runtime_state.md` — runtime context this pattern shows up against
- `project_cortex_v1_shim_api.md` — the shim API forward path is fertile ground for this pattern (per-block/per-token outputs with selective consumption)
- `feedback_doing_then_learning.md` — phantom work accumulates from doing-then-learning; the doing produces working code that's also overproductive; deliberate audit is the *learning* phase that pays back
- `project_modular_cognition_architecture.md` — Loop 1/Loop 2 compilation patterns are themselves a defense against phantom work (aging-out + hierarchical composition remove what's not load-bearing)

## The phrase to remember

> *Correctness preserved; downstream consumer absent; performance silently wrecked. Audit by asking "what reads this?" for every output and every wait.*

Plus the three-incident proof with empirical state through 2026-05-17 evening:

> *2026-05-16: phantom sync, ~5× win on cache_append (validated, shipped). 2026-05-17 morning: phantom computation, ~2× win on streaming chat (validated, shipped). 2026-05-17 evening: phantom allocator-drain wait — diagnostic confirmed (17.5s between stages, per-request), fix hypothesis (pooling) falsified, root cause and fix-shape both unknown, investigation ongoing. Pattern at the diagnostic level is robust. Pattern at the fix-shape level varies by class.*

Plus the granularity diagnostic:

> *Within-stage timing catches sync + compute classes. Request-level wall-vs-sum-of-stages catches the allocator-state class. Both are needed.*

Plus the diagnostic-vs-fix discipline:

> *The diagnostic surfaces the bug. The fix requires understanding the mechanism. They are not the same step. When the fix-shape is obvious (remove the wait; project less), ship it. When the fix-shape is non-obvious (pooling failed for allocator class), keep investigating before throwing more fixes at it.*
