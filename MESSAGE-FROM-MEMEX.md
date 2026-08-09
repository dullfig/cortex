# Cortex bug report from memex-claude — 2026-05-14

The first live end-to-end ingest run from memex against cortex hit
two distinct cortex bugs. The first has a config workaround; the
second is a hard blocker for memex's ingest pipeline.

## Bug 1 — KV cache default is too small (workaround found)

**Symptom:** Ingest panicked on the 5th file with
`cache overflow: 4019 + 366 > 4096` at `cortex/src/layers/gpu_engine.rs:1386`.

**Cause:** Default `--max-seq-len 4096` on `cortex-server`. After 4
modest markdown files (4019 content tokens + 4 sink tokens), the 5th
file's 366 tokens overflow the GPU KV cache allocation.

**Workaround:** `--max-seq-len 32768` (Qwen2.5-3B's trained context).
Memex now starts cortex with this flag for ingest.

**Worth considering:**
- Default could be larger (16K?) — 4K is too tight for any non-toy
  corpus.
- The overflow path panics on a tokio-rt-worker thread instead of
  returning a structured error. The HTTP response is a generic 500
  with no breadcrumb; only the cortex stderr revealed the cause.
  Replacing the `panic!` with a returned `WasmError` (or a typed
  cortex error mapped to a 4xx) would make this much easier to
  diagnose from the client side.

## Bug 2 — Incremental cache_append wedges around seq_len ≈ 9.7K (BLOCKER)

**Symptom:** With `--max-seq-len 32768`, ingest gets through 10 files
cleanly. Cumulative `cache.seq_len()` after 10 files = 9733 content
tokens (+ 4 sinks = 9737 total). The 11th call to
`POST /v1/cache/append` never returns.

**Cortex process state when wedged:**
- CPU: ~4× saturation (`Get-Process` shows 4000s CPU in 16 min
  wall-clock).
- `GET /health` times out — the axum listener can't get scheduled.
- No further log output from `cortex_server::cache appended`.
- Process is alive (responding=True per PowerShell) but functionally
  dead.

**Reproducible and content-independent:**

Two distinct runs produced the same wedge. After file 10 in the second
run, I renamed `conventions/convention-guide.md` (the file that hung
in the first run) to `.skip` so the wasm driver wouldn't emit it.
Re-ran with `--limit 11`. Files 1–10 completed identically, cumulative
seq_len reached 9733 again, and `education/arranging-guide.md` (the
new 11th file) wedged at the same point.

So the trigger is **cumulative `cache.seq_len()` crossing ~9.7K, not
file content**. Both files that wedged are ~3KB markdown / ~750 tokens
each — well below the 32K max.

**Likely failure mode (educated guess from outside the cortex codebase):**

The forward pass path used by `cache_append` —
`forward_full_gpu_with_cache(tokens, &mut cache)` — enters a hot loop
or deadlock when the existing cache's seq_len passes some threshold
around 8–10K. The fact that CPU is saturated rather than the process
being idle suggests it's a spin (e.g., bounded retry, geometric
backoff that grows unboundedly) rather than a literal mutex deadlock.

Possible smoking guns to check first:
- Any code path that does an O(N²) sweep over cache positions during
  incremental append (would explain CPU saturation that compounds
  per token).
- A wgpu shader dispatch that grows in cost more than linearly with
  cache length.
- A spin-loop on a "shard ready" or "memory aligned" check that
  doesn't yield to the runtime.
- `tokio::task::block_in_place` inside `cache_append` taking over the
  current worker for the forward pass — combined with a small worker
  pool (default tokio = num_cpus), enough concurrent appends could
  block the listener thread.

The wedge being content-independent and reproducible at the same
seq_len makes this look like an internal cortex condition (likely
size-keyed) rather than anything in the input.

## Reproduction recipe

1. Start cortex-server:
   `cortex-server --model C:/src/cortex/models/Qwen2.5-3B-Q4_K_M.gguf --port 8080 --enable-retrieve --max-seq-len 32768`
2. Start memex-api with `CORTEX_URL=http://127.0.0.1:8080`.
3. Build the bhs-corpus wasm driver:
   `cd C:/src/memex/memex-ingest/drivers/bhs-corpus && cargo build --release --target wasm32-wasip2`
4. Run the ingest:
   ```
   cargo run -p memex-cli -- ingest \
       --driver memex-ingest/drivers/bhs-corpus/target/wasm32-wasip2/release/bhs_corpus_driver.wasm \
       --corpus C:/src/bhs-corpus/sources \
       --shard bhs.corpus.all
   ```
5. Watch cortex's log: 10 successful `cache appended` events, then
   silence. CPU climbs.
6. Try `curl http://127.0.0.1:8080/health` — times out.

## What I tried

- Confirmed two distinct files trigger the wedge at the same seq_len
  (eliminating content as the cause).
- Confirmed memex-api and memex-cli are healthy and waiting on cortex
  via mutual `/healthz` probes (eliminating those layers).
- Memex's audit log + sled writes were fine through file 10
  (eliminating memex storage as a suspect).

What I did NOT try (out of scope from memex's side):
- Reducing batch size per `cache_append` call (memex submits one
  file's worth of tokens per call — file 11's was ~750 tokens, well
  below file 10's 1976).
- Forcing GPU sync between appends.
- Inspecting wgpu state with a debugger.
- Cortex source-level changes.

## Memex state during the run

Memex-side everything is structurally working: the WASM+WIT driver
pipeline executes cleanly, the host POSTs are well-formed, files 1–10
land in cortex's cache in deterministic order, and memex's position
sidecar tracks offsets correctly. The bug is downstream of memex's
HTTP boundary.

Once cortex stops wedging at seq_len ≈ 10K, the next concrete
milestone is ingesting the full 57-file bhs-corpus. Based on the
first 10 files averaging ~970 tokens, the full corpus is roughly
50–60K tokens cumulative, which is over the current 32K
`--max-seq-len`. So once Bug 2 is resolved, memex will need either
`--max-seq-len 65536+` (Qwen 2.5 supports up to 128K with YaRN
extensions) or to split into multiple shards. That decision is
downstream and not something cortex needs to do anything about.

— memex-claude

---

# Addendum 2026-07-15 — one-shot load_cache of a large history panics wgpu

**Found while exercising memex's cold-start replay path (first live
exercise ever, during the erasure e2e).**

**Symptom:** `POST /v1/cache/load` with ~6K tokens in one call panics
a tokio worker:

```
wgpu error: Validation Error
  In a CommandEncoder, label = 'forward_advance_only.encoder'
    In a dispatch command, indirect:false
      Each current dispatch group size dimension ([66848, 1, 1])
      must be less or equal to 65535
```

**Cause:** the prefill forward pass for the whole token vector
dispatches one workgroup-grid dimension proportional to token count
(~11 groups/token at Qwen 3B) — 5988 tokens → 66848 groups > wgpu's
65535 per-dimension limit. Incremental `cache/append` calls (a few
hundred tokens each, as ingest does) never get near it, which is why
this stayed latent until something replayed a big history in one call.

**Secondary concern:** each failed load attempt logged a fresh
`vram_heap created label="gpu_kv.heap" capacity_mb=576` — repeated
failures may leak heap allocations toward BudgetExceeded. Worth
checking the error path frees the heap.

**Memex-side workaround (shipped):** `ShardManager::ensure_resident`
now replays via `load_cache(empty)` + `append_tokens` in 1024-token
batches, mirroring the known-good ingest path. So memex no longer
exercises the one-shot path — but any other client that does will hit
the panic. Suggested fix: chunk the prefill loop internally over the
dispatch limit (or split into multiple dispatches), and return a
structured error instead of panicking a worker thread.

— memex-claude
