# cortex — Status Board

**Last updated:** 2026-05-18

## Legend

Four states. Strict on `[x]` — same diagnostic as the phantom-work audit:
*what production HTTP request actually exercises this?* If only tests
touch the code, it's not `[x]`. Aggressive on `[?]` — better to
over-mark than leave assumptions invisible to other Claude sessions.

| | Meaning |
|---|---|
| `[x]` | Built and in current production path |
| `[~]` | In flight (started, not landed) |
| `[ ]` | Next (specced, hasn't started) |
| `[?]` | Discussed only (no spec, possibly phantom) |

This board exists so integration-claude, AgentOS-claude, memex-claude
and other cross-repo sessions can see what cortex actually provides
today — not what design docs claim it provides. The `[?]` rows are
the antidote rows: capabilities other repos may be coding against on
the assumption they exist.

---

## 1. Compute backends

- [x] CPU scalar + AVX2 ternary kernels — `cortex/src/compute/scalar.rs`, `avx2.rs`; auto-fallback when GPU unavailable
- [x] WGPU compute backend (Vulkan / DX12 / Metal) — auto-selected at boot when discrete GPU present
- [x] Resident GPU weight buffers — `GpuFloatLinear` and `GpuBitLinear` both upload once at construction and hold across requests (boot log: "Resident-weights runtime: enabled")
- [x] BitNet 1.58b GPU batched prefill — `quantize_absmax_batch` + `ternary_matmul_batch`; commit e998934 unblocked the path end-to-end on `models/ggml-model-i2_s.gguf`
- [x] Per-block compute-pass merge — `forward_block_gpu_inner` uses 2 passes per block (down from 17); commit fea6078
- [x] Forward variants: `_with_cache_inject_returning_hidden` (chat hot path), `_advance_only` (cache_append, no readback), `_polar_traced` (retrieve)
- [x] GPU polar attention chain — rotate_q → score_polar_batch → softmax → value_polar_batch → derotate; entered via single-shard `/v1/retrieve` when `entry.polar.is_some()`
- [x] Cache_append chunking with `safe_chunk_size` — fixed the wedge at ~9.7K cumulative seq_len; ~5x ingest speedup
- [?] `VK_NV_cooperative_matrix` / WMMA path — discussed today as the H100 ceiling; matmul is confirmed the bottleneck (per attn-3 bisect) but no investigation yet
- [~] **CubeCL migration: spike GREEN, awaiting commit decision** — both spikes complete (reading: `pinky/cubecl-spike-2026-05-17.md`; technical: `pinky/cubecl-spike-results-2026-05-18.md`). All 4 technical risks validated against wgpu backend (CUDA backend deferred pending toolchain install). **Two surprise findings**: (1) CubeCL's default Auto mode already pools memory — empirically validated 35-60× faster per drop cycle on the same wgpu backend cortex hits the 17s cliff on; (2) cubek-attention is a single fully-fused call (Q/K/V → out), eliminates cortex's 3-kernel attention + 500MB scores scratch. Migration estimate revised: 5-7 wks → **3-5 wks** (Phase 1: wgpu-backend swap + cliff fix + simpler attention, ~1 wk; Phase 2: install CUDA toolchain, flip to CudaRuntime for matmul win, ~1 day; Phase 3: port BitNet ternary, ~2 wks). Decision: GO recommended, awaiting commit. Regression bench at `pinky/cubecl-poolbench/`
- [?] Flash-attention kernel — referenced in design discussions, no code; would be the path forward if attention ever becomes the bottleneck (today it's <10% per per-stage skip bisect)

## 2. Quantization & weights

- [x] GGUF parser — TQ1_0, TQ2_0, I2_S, Q4_K, F16, F32, BF16
- [x] `load_model()` auto-detect — ternary GGUFs → `BitLinear`, float → `FloatLinear`; LinearLayer trait is the universal seam
- [x] Q4_K_M end-to-end on Qwen 2.5-3B — current default test model
- [x] BitNet 1.58b end-to-end on the 1.2 GB Qwen ternary GGUF — streaming chat returns coherent text
- [x] PolarQuant compression — random orthogonal rotation + 3-bit polar angle quantization; CPU + GPU paths
- [x] QJL sign-of-projection residual correction (K only) — CPU path; CLAUDE.md notes ~0.84 cosine for K alone before QJL on V
- [x] QuantizedKvCache (CPU) + resident `GpuPolarKvCache` — both byte-layout compatible; ~7x VRAM vs f32
- [x] GPU prefill compress shader — `kv_compress_polar.wgsl` rotates + polar-quantizes f32 K/V directly into resident polar buffers (no CPU round-trip)
- [ ] QJL on V dequant — CLAUDE.md roadmap item; closes cosine-similarity gap on attention output
- [ ] Bit-packed 3-bit angle representation — u8 → 3-bit, ~12x compression vs current ~7.5x
- [?] Per-layer different quantization (Q4_K for embed, BitNet for blocks, etc.) — discussed; no plan

## 3. Shim API

- [x] Three-phase spec (gate / steer / inject) converged 2026-04-19 — `project_cortex_v1_shim_api.md` (in ringhub-integration memory)
- [x] Gate phase (#6a) production-callable — `chat_completions` reads `gate_shims` + `shim_rules`, branches `silent` vs `activate` (smoke: `pinky/tools/gate_smoke_test.sh`)
- [x] Steer phase (#6b) production-callable — `apply_steers_inplace` on last-token hidden + CPU re-projection (smoke: `pinky/tools/steer_smoke_test.sh`)
- [x] Injection phase (#6c) production-callable — `pre_block_hidden_inject` + `add_broadcast_batch.wgsl` per-block delta (smoke: `pinky/tools/inject_smoke_test.sh`)
- [x] All three phases compose end-to-end on Qwen 2.5-3B (validated this session)
- [x] Declarative `shim_rules` (match-and-dispatch, no scripting) — silent short-circuit + activate proceed-to-generation
- [x] `POST /v1/shims/embed` — pooled hidden-state vector for shim training (vocabulary mirrors manifest: `layer ∈ {final, entrance:N}`, `pooling ∈ {last_token, mean}`)
- [x] Shim registry CRUD endpoints — `/v1/shims/` (list), `/v1/shims/{id}` (get/put/delete); hot-resident ONNX shims; gated by `--enable-shims`
- [~] "Should-I-reply" classifier shim (per `project_cortex_ffn_shims.md`) — gate framework live, the trained classifier itself not packaged
- [~] Cached + steers / cached + inject — rejected at request validation as v1 limits; both need a cached returning_hidden variant
- [?] Shim manifest schema frozen for cross-team use — design doc exists, no checked-in schema file or validator

## 4. Inference paths

- [x] `POST /v1/chat/completions` non-streaming — OpenAI-compatible
- [x] `POST /v1/chat/completions` streaming (SSE) — stateless mode only; rejects cache_shards + tools + retrieve combinations
- [x] Tool-calling JSON output parsing — extracted from final response when `tools` present
- [x] `POST /v1/cache/load` + `/v1/cache/append` — gated by `--enable-cache` (librarian deployment)
- [x] `GET /v1/cache/{id}` + `DELETE` — cache pool inspection and eviction
- [x] Retrieve mode (`mode=retrieve`) — multi-shard composition + bidirectional attention ranking; gated by `--enable-retrieve`
- [x] LM head over LAST token only in prefill — commit 0cab678 saved ~30s on 1525-token prompts (2x speedup)
- [x] Non-streaming chat routed through `generate_stateless_gpu` (was per-layer-CPU-sync) — BitNet 6-7x speedup on non-streaming
- [ ] `cortex_local::CortexLocal::complete()` over `GpuEngine` — still uses slow CPU `model.generate()` path; needs analogous wrapping
- [ ] Cached + inject (forward variant exists, request handler rejects as v1 limit)
- [ ] Cached + steers (same — needs cached returning_hidden hook)
- [?] Per-request sampling override beyond temperature (top-k, top-p, repetition penalty) — supported in `SamplerConfig` but not wired into the wire schema

## 5. Observability & perf

- [x] `GET /metrics` Prometheus text format — commit c9fd38c (this session)
- [x] `cortex_requests_total{endpoint, status}` counter — chat_completions / cache_load / cache_append × ok|err
- [x] `cortex_tokens_total{kind=prompt|completion}` counter
- [x] `cortex_request_duration_seconds{endpoint}` histogram — 10 buckets from 5ms to 60s
- [x] `cortex_ttft_seconds` histogram — chat streaming first-token boundary
- [x] `cortex_uptime_seconds`, `cortex_model_info`, `cortex_build_info` gauges
- [x] Stage-timing instrumentation — `forward_full_gpu_with_cache_advance_only` logs alloc/record/submit/poll_us
- [x] Debug-bisect env flags — `CORTEX_SKIP_SCORE/SOFTMAX/VALUE/BLOCK_FORWARD/SYNC_AFTER_ADVANCE`; documented as perf bisect tools, off by default
- [x] Baseline TTFT + decode bench — `pinky/tools/bench_baseline.py` (commit 1133550); current Qwen 3B Q4_K_M numbers: ~21 t/s decode, 2.8s TTFT short / 22s TTFT long; ~3-4x slower than llama.cpp on same hardware
- [x] cache_append wedge reproducer — `pinky/tools/reproduce_append_wedge.py`
- [x] TTFT cliff repro + per-stage diagnostic — `pinky/tools/probe_ttft_stages.sh` (3 sequential chat curls + server-side stage timings localize the cliff)
- [~] **Known issue: chat_completions TTFT cliff** — every chat request 2+ pays ~17s in wgpu buffer drops (`vkFreeMemory` accumulated-state cost on NVIDIA Vulkan). Proven NOT to be GPU work (explicit `device.poll(Wait)` = 19µs after readback). Pooling `BlockScratch` alone moves the cost to other buffers (hidden/normed/staging) — same 17s, different drop site. Real fix needs allocator-pool-everything OR replace wgpu's allocator. `cache_append` workflow does NOT hit this (uses pooled cache, different drop pattern); only chat_completions hits it. Tracking under "mini-OS for GPU memory" investigation (CubeCL spike in progress)
- [~] Concurrency gauge — third of three threshold metrics in `project_cortex_v1_perf_threshold.md`; TTFT + decode rate are `[x]`, concurrency isn't emitted yet
- [ ] Per-endpoint metrics on `/v1/tokenize`, `/v1/detokenize`, `/v1/shims/*`, `/v1/models`, `/health`, `/v1/retrieve` — phantom-strict rule says each lands when handler + render line ship together
- [ ] Per-stage attention timing breakdown in /metrics — would expose which of score/softmax/value dominates at any given shape
- [?] GPU utilization / VRAM gauges — not in MVP scope; would need wgpu introspection that may not exist

## 6. Plugin architecture

- [x] Shim registry as plugin substrate — hot-resident ONNX shims, manifest CRUD endpoints, three-phase dispatch
- [?] Federated plugin API per `project_cortex_plugin_architecture.md` — design doc exists; no `/v1/plugins/list` endpoint, no plugin discovery, no `register_plugin`/`unregister_plugin` machinery
- [?] WASM extensions / `wasmtime` hosting — zero `wasmtime` / `wasm-runtime` references in cortex source; AgentOS has parallel `WasmSession` infrastructure but cortex doesn't link to it
- [?] BYO-WIT contract — AgentOS-side type definitions exist; cortex side untouched
- [?] memex as a cortex plugin (vs the current HTTP coupling where memex hits `/v1/cache/append`) — direction discussed; no design
- [?] cache_pool eviction policy as a plugin point — current eviction is hard-coded LRU; pluggable strategy discussed only

## 7. Integration boundaries

- [x] OpenAI-compatible HTTP API — any OpenAI-protocol client (AgentOS `OpenAiClient`, curl, etc.) works as a drop-in
- [x] `cortex-local` in-process provider — `CortexLocal::load()` + `complete()`; same wire types as cortex-cloud, no HTTP hop
- [x] Two deployment modes — stateless 32B Bob (no `--enable-cache`) vs librarian with cache pool (`--enable-cache --enable-retrieve`)
- [x] Polar cache backend on retrieve — `--enable-polar-cache` flag; cache_load builds parallel polar cache; validated against Qwen 3B + 1941-token Harmonizer corpus
- [?] `shim_store` integration with AgentOS kernel — zero `shim_store` code in cortex; integration-claude / AgentOS-claude may assume this; it does not exist
- [?] AgentOS kernel direct-linking — `cortex-local` parallels what direct-link would do, but the kernel `LlmClient::Local(CortexLocal)` variant has not landed in AgentOS
- [?] memex retrieval coupling beyond HTTP — memex currently uses `/v1/cache/append` via HTTP; deeper coupling (shared polar cache, zero-copy hand-off) discussed only
- [?] Multi-tenant request handling — single-tenant v1; no per-tenant isolation in any handler (cache pool keys are global, shim registry is global)
- [?] Future Zynqberry / FPGA path — discussed; no spec, no investigation

---

## Footer

- This board supersedes `cortex/ROADMAP.md` (last updated 2026-04-12; now ~5 weeks stale, predates all shim phase work + polar backend + BitNet GPU + telemetry MVP).
- The `## Roadmap` section in `CLAUDE.md` is current as of 2026-05-17 but is shaped as a chronological checklist, not a state board — it answers "what did we do?" not "what does cortex provide?". Use this board for the latter, CLAUDE.md for the former.
- When a row changes state, update the row and bump the `Last updated` date at the top. When in doubt about whether a capability deserves `[x]`, re-apply the diagnostic: *which production request exercises it?* If the answer is "a test does", it's not `[x]`.
