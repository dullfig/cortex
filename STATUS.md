# cortex — Status Board

**Last updated:** 2026-09-08

> **Milestone banner (read first).** cortex is now a **float-only,
> Qwen-class GPU transformer inference substrate** (Q4_K_M / F16 / BF16 /
> F32) on wgpu→Vulkan. The ternary/BitNet path was **un-merged 2026-05-29**
> to the sibling `ternary-rs` crate (tag `bitnet-archive-2026-05-29`,
> commit e1e2dc1) — so the CPU scalar/AVX2 kernels and BitNet GPU prefill
> that older docs describe are **gone from this repo.**
>
> **GPU-substrate milestone reached 2026-06-13** (secure deploy path,
> Phase Q); most recent code landing is the **adversarial-review hardening
> pass 2026-09-04..08** (`docs/adversarial-review-2026-09-02.md`: 29
> findings; all five P0s + P1 #6/#11 closed in b2dfeb2, ed7790c, 5744f51 —
> request data is validated at the HTTP boundary, the prefill chunker knows
> wgpu's 65535 dispatch cap, cache allocation is fallible; then #22 revived
> the CPU-vs-GPU parity suite, which on its first run caught **#27 — every
> `/v1/shims/embed` / gate-shim vector since 2026-05-29 was garbage**, #28
> f32 traced forward on garbage, #29 untied-head models panic at init; all
> three fixed. **Shim vectors produced before 2026-09-08 are invalid.**
> 2026-09-09: the P1 concurrency / cache-integrity cluster closed — #9
> lock re-acquisition witnesses, #8/#20 `tokens`-vs-KV lockstep invariant,
> #7 `gpu_gate` one-GPU-region-at-a-time admission (+ `cortex_gpu_gate_waiting`
> gauge, the Stage-2 trigger), #12 traced-query bound, #30 reload flush.
> 2026-09-10: the GGUF / model-load trust boundary closed — a hostile or
> corrupt model file is always a clean `Err` from `load_model` naming the
> field (never an abort, panic, silent wrap or first-request assert): #13–#16
> parser bounds + checked arithmetic, #17/#19 `ModelConfig::validate` + every
> tensor shape and norm length checked, #18 RoPE layout per architecture
> (`rope.scaling.type` was read as the wrong type), #21 tokenizer arrays
> validated + UTF-8 byte fallback; release profile now has
> `overflow-checks`. 2026-09-11: **the review is closed, 31 of 31** — #23/#31
> every per-request GPU buffer on a budgeted lane / readback span (the
> intermittent "buffer is invalid" panic was a raw `create_buffer_init`
> mapping the buffer it failed to allocate), #24 panics are structured
> errors + disconnect cancels stateless/streaming/shim generation, #10
> bigram-heap BPE + input caps, #25/#26 shim ONNX validated at PUT,
> `max_tokens: 0` → 400, `PollFlush` guard, `kv_write` y-stride
> (a9d39fc, a9f91f1, ac748ac, a6a25c5). Filed: #32 (cached-path cancellation is a
> design question — see the review doc).)
> cortex is currently **PARKED as a stable inference substrate** — per the
> integration pin `state_of_project_2026-07-24`, "cortex is the next *code*
> phase, not the next *project* phase"; the project foreground is the
> mission/corpus track, which needs zero cortex. Nothing here is
> broken-and-blocking; the memex one-shot `cache/load` defect (§4) is fixed.

## Branch & working-tree state

- **Active branch: `wgpu-29`** — carries the entire post-BitNet GPU
  substrate. **136 commits ahead of `origin/main`, 0 behind** — *not yet
  merged to main.* Treat `wgpu-29` as the de-facto trunk for cortex code.
- **Working tree: clean** (keep/discard resolved 2026-08-09). The 4-day-old
  dirty tree was cleared:
  - `MESSAGE-FROM-MEMEX.md` — committed (memex-claude inbound bug report,
    Addendum 2026-07-15 — the §4 defect).
  - `docs/` (11 files) — committed. Design pins **migrated 2026-07-06 from
    the integration shared brain to their owning repo** (each shared-brain
    copy is now a pointer-stub → "/src/cortex/docs/…"); these are the
    canonical full copies, now tracked here.
  - `pinky/chat.py` — committed (interactive OpenAI chat REPL test tool).
  - `**/__pycache__/` + `*.pyc` — gitignored (junk).

## Legend

Four states. Strict on `[x]` — same diagnostic as the phantom-work audit:
*what production HTTP request actually exercises this?* If only tests
touch the code, it's not `[x]`. Aggressive on `[?]` — better to
over-mark than leave assumptions invisible to other Claude sessions.

| | Meaning |
|---|---|
| `[x]` | Built and in current production path |
| `[~]` | In flight (started, not landed / not production-verified) |
| `[ ]` | Next (specced, hasn't started) |
| `[?]` | Discussed only (no spec, possibly phantom) |

This board exists so integration-claude, AgentOS-claude, memex-claude and
other cross-repo sessions can see what cortex actually provides today —
not what design docs claim. The `[?]` rows are the antidote rows.

---

## 1. Compute backend & GPU memory substrate

- [x] **WGPU / Vulkan is the only backend** — device selected + bound at
  boot; discrete GPU required (no CPU fallback path since the BitNet
  un-merge).
- [x] **device-probe boot integration** (commit 985594d, 2026-06-25) —
  cortex sources its live `wgpu::Device`/`Queue`, VRAM budget, and *device
  selection* from the `device-probe` crate (`device-probe → vram-heap →
  cortex`). Enumerates all adapters, refuses software/CPU fallback, picks
  best-for-workload (bandwidth-bound), logs the full `DeviceProfile` incl.
  measured f16 speedup + bandwidth. Verified live: on the dev box it binds
  the discrete GPU over the iGPU; VRAM budget byte-parity with the old
  detector. `CORTEX_VRAM_TOTAL_MB` overrides; `CORTEX_PROBE_NO_MEASURE=1`
  skips the measure pass.
- [x] **vram-heap free-list allocator substrate** (migration Phases A–I,
  2026-06-08→06-09) — every GPU allocation outside `ParamsBufferPool` flows
  through `vram-heap` (RAII + coalescing free-list over pre-allocated
  heaps). 3-lane scheme (`transient_heap_a/b/c`) enforces wgpu-29's
  same-backing R+RW rule; static `weights_heap` (`allocate_static`);
  per-cache `gpu_kv` / `polar_kv` heaps. **This replaced wgpu's
  per-dispatch allocator churn and RESOLVED the old ~17s TTFT cliff** (the
  "mini-OS for GPU memory"). The CubeCL-migration spike that old STATUS
  tracked was **not taken** — vram-heap was the answer instead.
- [x] **Device-aware heap sizing + multi-lane, binding-clamped chunker**
  (Phase M, 7ee497d) — `DeviceBudget` derives lane sizes from real VRAM;
  `safe_prefill_chunk_size` / `ChunkLimits` clamp prefill chunks against
  Lane A/B/C capacity **and** `max_storage_buffer_binding_size` (~2 GB).
  Auto-scales to bigger cards; loud `BudgetExceeded` on over-commit.
- [x] `ParamsBufferPool` — 16384-slot ring (Phase J, 9570cd6); bumped to
  move the wrap-around-race threshold to ~120 concurrent forwards; `stats()`.
- [x] GPU polar attention chain — rotate_q → score_polar_batch → softmax →
  value_polar_batch → derotate; entered via `/v1/retrieve` when the shard
  has a polar cache.
- [x] Forward variants: `_with_cache_inject_returning_hidden` (chat hot
  path), `_advance_only` (cache_append), `_polar_traced` (retrieve).
- [x] Source layout refactor (Phase N) — `gpu_engine.rs` → `gpu_engine/`
  module dir; `cortex-cloud/main.rs` → modules (`chat/shims/cache/api/state`).
- [ ] **C3 packed-perf restoration** — `hidden_buf` + `scratch.projected`
  are still f32 (reverted for the old BitNet "Option E" NaN fix). BitNet is
  gone, so the revert is no longer load-bearing; restoring packed-f16
  recovers a ~9% Qwen prefill regression. Not started.
- [ ] Polar `forward_block_gpu_polar_inner` C3 port — rotate_q / derotate
  need packed variants (currently panic-guarded).
- [?] `VK_NV_cooperative_matrix` / WMMA matmul path — the H100 ceiling;
  matmul is the confirmed bottleneck but no work yet. device-probe now
  *measures* f16 speedup so the future precision-kernel switch has its input.

## 2. Quantization & weights

- [x] GGUF parser — Q4_K, Q5_K, Q6_K, F16, F32, BF16 (float types only now).
- [x] `load_model()` → `FloatLinear` / `GpuFloatLinear` (LinearLayer trait
  kept as the single-impl seam pinkies depend on).
- [x] Q4_K_M end-to-end on Qwen 2.5-3B — current default test model.
- [x] PolarQuant compression — random orthogonal rotation + 3-bit polar
  angle quant; CPU + GPU paths; resident `GpuPolarKvCache`.
- [x] **QJL residual correction on BOTH K and V** — K-side (sign-of-
  projection) + **V-side QJL-256 with Γ-scaled estimator (Phase O,
  d441f7d)**. Closed the attention-output cosine gap (CPU dequant cosine
  0.797 → 0.914); polar_qjl retrieve R@10 0.125 → 0.300 + first R@1 hit.
  (Old STATUS listed "QJL on V" as a pending roadmap item — now done.)
- [x] QuantizedKvCache (CPU) + resident `GpuPolarKvCache` — byte-layout
  compatible; ~7.5× VRAM reduction vs f32; GPU prefill compress shader
  (`kv_compress_polar.wgsl`) rotates + quantizes f32 K/V directly into
  resident polar buffers (no CPU round-trip).
- [ ] Bit-packed 3-bit angle representation — u8 → 3-bit, ~12× vs current
  ~7.5×. Not started.
- [ ] Wire `QuantizedKvCache` into cortex-cloud as the cache-pool backing
  store — still a roadmap item.

## 3. Shim API

- [x] Three-phase spec (gate / steer / inject) — all three compose
  end-to-end on Qwen 2.5-3B.
- [x] Gate / Steer / Inject phases production-callable via `chat_completions`
  (`gate_shims` + `shim_rules`; `apply_steers_inplace`; per-block inject
  delta). Gated by `--enable-shims`.
- [x] `POST /v1/shims/embed` — pooled hidden-state vector (`layer ∈
  {final, entrance:N}`, `pooling ∈ {last_token, mean}`).
- [x] Shim registry CRUD — `/v1/shims/` (list), `/v1/shims/{id}`
  (get/put/delete); hot-resident ONNX shims.
- [~] "Should-I-reply" classifier shim — gate framework live; trained
  classifier itself not packaged.
- [~] Cached + steers / cached + inject — rejected at request validation as
  v1 limits; both need a cached returning_hidden variant.
- [?] Shim manifest schema frozen for cross-team use — design doc only.

## 4. Inference paths

- [x] `POST /v1/chat/completions` — non-streaming + streaming (SSE,
  stateless mode); OpenAI-compatible; tool-call JSON parsing.
- [x] `POST /v1/cache/load` + `/v1/cache/append`, `GET`/`DELETE /v1/cache/{id}`
  — gated by `--enable-cache`. Incremental append is the known-good ingest
  path (chunked, ≤~1K tokens/call).
- [x] **Retrieve mode** (`mode=retrieve`) — multi-shard composition +
  bidirectional-attention ranking; `--enable-retrieve` (+ `--enable-polar-cache`).
  ⚠️ **Antidote note for memex-claude:** retrieve *works* but its **recall
  is METHOD-limited, not quantization-limited** — f32 control R@10 ≈ 0.10;
  attention-score readout under-recalls, and selected retrieval-heads do
  **not generalize** (by-shard holdout R@10 = 0.00, Phase P.3). Diagnostics
  P.1/P.2/P.3 (per-head sweep, offset-zero probe, trustworthy holdout) all
  agree. **Generation-as-index / synopsis-routing is the intended path**,
  not attention-readout — don't build assuming high retrieve recall. See
  memory pins `project_retrieval_method_bottleneck`, `_retrieval_heads_overfit`,
  `_memex_architecture_direction`.
- [x] **FIXED 2026-09-07 (commit 5744f51) — large one-shot `cache/load`
  panicked wgpu** (memex-claude report, `MESSAGE-FROM-MEMEX.md` Addendum
  2026-07-15; reply 2026-09-04). Root cause: the prefill chunker modelled
  bytes only; softmax dispatches `n_tokens · n_heads` in one dimension
  (16 × 4178 = 66848 > 65535). `prefill_chunk_size` now carries the
  dispatch-dimension constraint (Qwen 3B chunks ≤ 4095), the few unchunked
  forwards (shim hidden-capture, traced retrieve query, polar chat prompt)
  are bounded and return `400 context_length_exceeded`, and the stateless /
  streaming / composition prefills go through the chunker. Verified live:
  6000 tokens in one `cache/load` → 201, `seq_len 6004`. Heap-leak concern
  checked: RAII frees on unwind, nothing accumulates. Companion fixes from
  the same review (b2dfeb2, ed7790c): `max_tokens` clamped to shard room,
  control-token forgery blocked in the chat template, temperature floor +
  NaN-safe sampler, fallible cache alloc (`503 vram_exhausted`), pool cap
  (`--max-cache-shards`, `507 cache_pool_full`), token-id validation.
- [x] **Adversarial review closed 31/31 (2026-09-11)** — see
  `docs/adversarial-review-2026-09-02.md` (fix-status table). One design
  question filed as #32: cancelling cached-path generation on client
  disconnect needs the pool entry checked out for the duration of
  generation (wait vs. 409 `shard_busy` for concurrent callers).
- [ ] `cortex_local::CortexLocal::complete()` over `GpuEngine` — still uses
  the slow CPU `model.generate()` path; needs the GPU-engine wrapping.
- [?] Per-request sampling override beyond temperature (top-k/top-p/rep) —
  in `SamplerConfig`, not wired to the wire schema.

## 5. Observability & perf

- [x] `GET /metrics` Prometheus text — request/token counters, duration +
  TTFT histograms, uptime/model/build gauges.
- [x] **Scaling-stage gauges (Phase K, 4e6f9bb)** — `cortex_concurrent_requests`,
  `cortex_cache_pool_size` / `_tokens_total`, `cortex_vram_heap_bytes{heap}`,
  `cortex_params_pool_acquired_total`, `cortex_gpu_busy_micros_total`
  (+ `cortex_vram_budget_bytes{kind}` from Phase M). The dashboard that
  gives weeks of warning before the next scaling stage.
- [x] Full `DeviceProfile` logged at boot (device-probe) — first thing to
  check when something's mysteriously slow: what cortex thought it ran on.
- [x] **TTFT cliff: RESOLVED** — the old ~17s per-request `vkFreeMemory`
  cliff was an allocator-churn artifact; the vram-heap substrate (§1)
  eliminated it. (Old STATUS tracked this as an open `[~]` known issue.)
- [x] Baseline bench harness under `pinky/` — Qwen 3B Q4_K_M ~21 t/s decode
  (pre-C3-restore); healthy GPU 2271-token retrieve ~0.23s post-reboot.
- [ ] Per-endpoint metrics on tokenize/detokenize/shims/models/health/retrieve.
- [?] Continuous batching (Stage-2 scaling) — don't start until the Phase K
  meters say it's needed.

## 6. Plugin architecture

- [x] Shim registry as plugin substrate — hot-resident ONNX shims, manifest
  CRUD, three-phase dispatch.
- [?] Federated plugin API / WASM extensions / BYO-WIT — design docs only;
  no `/v1/plugins/*`, no `wasmtime` in cortex source.
- [?] memex as a cortex plugin (vs current HTTP coupling) — discussed only.
- [?] cache-pool eviction as a plugin point — hard-coded LRU today.

## 7. Integration boundaries

- [x] OpenAI-compatible HTTP API — any OpenAI-protocol client works.
- [x] `cortex-local` in-process provider — `CortexLocal::load()` + `complete()`
  (note the `complete()` GPU-path gap in §4).
- [x] **device-probe** consumed as the leaf of `device-probe → vram-heap →
  cortex` (see §1). Companion: device-probe now requests
  `TIMESTAMP_QUERY_INSIDE_ENCODERS` so cortex keeps its PASS timers.
- [~] **Secure remote deploy path (Phase Q, cf5612c, 2026-06-13)** —
  `deploy/`: multistage Dockerfile (Vulkan + onnxruntime), Caddy TLS +
  Bearer-key gate, compose stack (cortex private, no published port; the
  `graphics` NVIDIA capability is the load-bearing gotcha). **Structurally
  validated; GPU/Vulkan-in-container NOT yet verified on a real Linux NVIDIA
  box** — that's the decisive test, deferred to the H100/A40 box.
- [~] memex retrieval coupling — memex hits `/v1/cache/append` + `/v1/retrieve`
  over HTTP (live; see the §4 defect it surfaced). Deeper zero-copy coupling
  discussed only.
- [?] AgentOS kernel `LlmClient::Local(CortexLocal)` variant — not landed in
  AgentOS.
- [?] **RetrievalAttention / ANN-over-KV** (arXiv 2409.10516) — evaluate on
  integration's stack first; 4-phase roadmap to a swappable
  `AttentionBackend` (CPU index + GPU compute). Pin:
  `project_retrieval_attention_modularization`. Not started in cortex.
- [?] Multi-tenant request handling — single-tenant v1 (global cache pool +
  shim registry).

---

## Footer

- `STATUS.md` (this doc) = "what does cortex provide today?" snapshot. The
  integration pin `state_of_project_2026-07-24` points cross-repo sessions
  here as cortex's code-truth authority — keep it current.
- The migration/phase history (vram-heap A–K, L chunker, M sizing, N
  refactor, O QJL-V, P retrieve diagnostics, Q deploy, device-probe) lives
  in the git log on `wgpu-29` and the `## Roadmap` checklist in `CLAUDE.md`.
- `docs/` (once committed) holds the full design pins migrated from the
  shared brain 2026-07-06.
- When a row changes state, update the row and bump `Last updated`. Re-apply
  the `[x]` diagnostic when in doubt: *which production request exercises it?*
