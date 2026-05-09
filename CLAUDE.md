> **Cross-session coordination:** Before making any design/scope decision, read `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\MEMORY.md` first — that folder is the shared brain across all Claude sessions on this project. Decisions pinned there supersede anything in this repo's older docs.
>
> **If something has happened to Daniel:** read `C:\src\CARETAKER.md` — the project's caretaker-handoff document.

# cortex

Universal local transformer engine with persistent memory. Runs any GGUF model — ternary, quantized, or float.

## Workspace

Cargo workspace with three crates:

| Crate | Path | Purpose |
|-------|------|---------|
| `cortex` | `cortex/` | Core engine — tensor ops, GGUF loader, transformer stack, memory |
| `cortex-cloud` | `cortex-cloud/` | HTTP server: OpenAI-compatible `/v1/chat/completions` (axum) |
| `cortex-local` | `cortex-local/` | In-process provider for AgentOS — same types, no HTTP hop |

### cortex-cloud

Serves the OpenAI wire format so any client (AgentOS `OpenAiClient`, curl, etc.) can hit cortex as a drop-in backend. Binary: `cortex-server`. See `api-spec.md` for the full API contract.

```bash
cortex-server --model path/to/model.gguf --port 8080
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`

### cortex-local

Library crate that AgentOS can depend on directly. `CortexLocal::load()` returns a provider with `complete()` / `complete_with_tools()` — same semantics as the HTTP API but in-process.

```rust
let provider = cortex_local::CortexLocal::load("model.gguf", 4096)?;
let response = provider.complete(&request)?;
```

AgentOS integration: add as `LlmClient::Local(CortexLocal)` variant in `agentos-llm`.

## Lineage

cortex absorbs and generalizes three projects:
- **ternary-rs** → ternary kernels, BitLinear, GGUF loader, full transformer stack (DONE)
- **engram** → compressed KV cache (PolarQuant + QJL CPU side, DONE);
  tiered memory + bidirectional-attention retrieval + consolidation (TODO)
- **neuralkv-core** (GPU path) → WGPU shaders for matmul, attention, FFN (TODO)

## Architecture (cortex core)

### Core (from ternary-rs)
- **Tensor** (`cortex/src/tensor.rs`) — 2-bit packed ternary, 8-bit quantized activations, float tensors
- **I2S Kernel** (`cortex/src/ops/matmul.rs`) — Ternary matvec via conditional add/sub/skip
- **LUT Kernel** (`cortex/src/ops/lut.rs`) — Lookup table kernel, zero arithmetic in hot loop
- **GGUF** (`cortex/src/gguf.rs`) — Parser for TQ1_0, TQ2_0, I2S, Q4_K, F16, F32, BF16
- **Loader** (`cortex/src/loader.rs`) — `load_model()`: GGUF → auto-detect → right LinearLayer → go

### Layers
- **LinearLayer trait** (`cortex/src/layers/linear.rs`) — the universal seam: BitLinear | FloatLinear | WgpuLinear
- **BitLinear** (`cortex/src/layers/bitlinear.rs`) — ternary linear: quantize → ternary matmul → rescale
- **FloatLinear** (`cortex/src/layers/floatlinear.rs`) — dequantized float linear (Q4_K, F16, F32)
- **Attention** (`cortex/src/layers/attention.rs`) — GQA with RoPE, causal mask, KV cache
- **SwiGLU** (`cortex/src/layers/swiglu.rs`) — gated FFN (SiLU or ReLU²)
- **TransformerModel** (`cortex/src/layers/model.rs`) — full forward pass, generate, forward_cached
- **Memory trait** (`cortex/src/layers/memory.rs`) — TransformerMemory: ingest, retrieve, consolidate

### Compute Backends
- **Scalar** (`cortex/src/compute/scalar.rs`) — portable fallback
- **AVX2** (`cortex/src/compute/avx2.rs`) — x86-64 SIMD
- **WGPU** (`cortex/src/compute/wgpu_backend.rs`) — GPU via Vulkan/DX12/Metal

### TurboQuant KV compression (from engram)
- **PolarQuant** (`cortex/src/ops/polar.rs`) — random orthogonal rotation +
  3-bit polar angle quantization. Stage 1: ~7.5x reduction (u8-per-angle;
  ~12x with future bit-packing).
- **QJL** (`cortex/src/ops/qjl.rs`) — 1-bit sign-of-projection residual
  correction. Stage 2: refines attention dot products.
- **QuantizedKvCache** (`cortex/src/layers/quantized_kv_cache.rs`) —
  per-layer compressed cache: append, dot in compressed domain, dequant
  on demand, lossless tier migration via `CompressedEntry`.

### Memory (further engram ports, TODO)
- **HierarchicalCache** — L1 working / L2 session / L3 archive
- **Retrieval** — bidirectional attention (no causal mask), returns ranked spans
- **Consolidation** — entropy-driven sleep: evict noise, migrate summaries L1→L2→L3

## Key Invariants

- `LinearLayer` is the abstraction point: ternary or float, the transformer doesn't care
- Memory uses the SAME model's Q/K projections — one embedding space
- GGUF auto-detection: TQ1_0/TQ2_0 → BitLinear, Q4_K/F16/F32 → FloatLinear
- All f32 at layer boundaries — no custom tensor framework lock-in
- Zero unsafe

## Public API

```rust
// Core engine
let model = cortex::load_model("model.gguf")?;
let tokens = model.tokenizer.encode("Hello");
let logits = model.model.forward(&tokens, 0);
let generated = model.model.generate(&tokens, &sampler, 256);

// In-process provider (AgentOS)
let provider = cortex_local::CortexLocal::load("model.gguf", 4096)?;
let response = provider.complete(&request)?;
```

## Testing

393 tests covering: ternary packing, matmul kernels, quantization, GGUF parsing,
layer forward passes, attention, RoPE, SwiGLU, full model forward, sampler,
retrieval (forward_traced + attention-score ranking), TurboQuant compression
(PolarQuant + QJL + QuantizedKvCache + GPU score/value/derotate shaders +
algorithm-quality cosine pinning + resident GpuPolarKvCache storage +
resident dispatchers byte-equal to oneshot + GPU prefill compress shader +
GPU-only f32→polar conversion + multi-token causal-masked batch shaders +
polar trace forward through full model), hidden-state extraction hooks
(per-block + final post-norm) for shim runtime, ort link smoke.

For GPU-heavy tests, prefer `cargo test --workspace -- --test-threads=1`
to avoid VRAM contention between concurrently-running GPU tests on a
shared discrete GPU.

Run all: `cargo test --workspace`

## Roadmap

- [x] Full transformer forward pass (ternary + float)
- [x] KV cache for autoregressive generation
- [x] Token sampler (top-k, top-p, temperature)
- [x] TransformerMemory trait definition
- [x] cortex-cloud: OpenAI-compatible HTTP server
- [x] cortex-local: in-process provider for AgentOS
- [x] Move QuantizedKvCache from engram into cortex (CPU side)
- [x] GPU score shader for compressed K (`attn_score_polar.wgsl` +
      `gpu_polar::attn_score_polar_oneshot`); matches CPU `dot_key` within 1e-5
- [x] GPU value/derotate shaders for compressed V
      (`attn_value_polar.wgsl` + `derotate.wgsl`); matches CPU dequant+aggregate
      at seq_len=4096 within 1e-3 (float-order error scales O(seq_len))
- [x] Resident `GpuPolarKvCache` storage (per-layer K/V angle+radius
      buffers + per-layer rotation matrices); ~7x VRAM vs f32 GpuKvCache
      on Qwen 3B shape; byte-layout compatible with the oneshot dispatchers
- [x] Resident dispatchers (`attn_score_polar_resident` /
      `attn_value_polar_resident`) that take `&GpuPolarKvCache` + a
      layer index; output byte-equal to the oneshot path
- [x] GPU prefill compress shader (`kv_compress_polar.wgsl` +
      `compress_layer_into_polar`); rotates + polar-quantizes f32 K/V
      directly into the resident polar buffers, no CPU round-trip;
      angles byte-equal to CPU `append_one`, radius matches within 1
      ULP (FMA tolerance)
- [x] `GpuPolarKvCache::populate_from_f32_cache_gpu` — convert a
      populated f32 `GpuKvCache` to a polar cache via the compress
      shader (per-layer, all-on-GPU). Unblocks the cortex-cloud
      retrieval path: prefill stays f32, then a one-time conversion
      hands the polar cache to subsequent retrieve queries
- [x] Multi-token causal-masked batch shaders: `rotate_q.wgsl` +
      `attn_score_polar_batch.wgsl` + `attn_value_polar_batch.wgsl`
      (derotate.wgsl reused for the multi-token output by treating
      `n_tokens * n_query_heads` as effective head count). Full
      5-stage GPU pipeline matches CPU reference within 1e-5
- [x] `forward_full_gpu_polar_traced` — polar-aware traced forward
      that runs every block through `forward_block_gpu_polar_inner`:
      f32 RMSNorm + Q/K/V projections + RoPE, then GPU compress writes
      query K/V into the polar cache, then the polar attention chain
      (rotate_q → score_polar_batch → softmax_batch → value_polar_batch
      → derotate). Pre-softmax score capture via the same Option<&Buffer>
      hook as the f32 path
- [x] cortex-cloud retrieve cache_load → polar backend wiring
      (`--enable-polar-cache` flag; `cache_load` builds a parallel
      polar cache via `populate_from_f32_cache_gpu`; single-shard
      `/v1/retrieve` dispatches on `entry.polar.is_some()` to use
      `forward_full_gpu_polar_traced`. Validated end-to-end against
      Qwen 3B + 1941 Harmonizer corpus: polar trace returns
      semantically-correct hits including offset 324 = "Bluejacket")
- [x] Shim phase dispatch — gate (#6a): `gate_shims` /
      `steer_shims` / `inject_shims` / `shim_rules` on
      `/v1/chat/completions`. Gate fires once after a shared prefill;
      `shim_rules` (declarative match-and-dispatch, no scripting)
      route to `silent` (short-circuit, `finish_reason: "silent"`,
      zero content) or `activate` (proceeds to generation). Streaming
      and non-streaming both supported; metadata (`gate_decisions`,
      `active_steers`, `signals`, timings) lands on the final chunk /
      response. Steers (#6b) and injection (#6c) record the requested
      sets in metadata for forward-compat but are not yet applied.
      Validated end-to-end on Qwen 2.5-3B + a squared-norm gate shim
      (`pinky/tools/gate_smoke_shim.onnx`): silent + proceed paths
      pass for both streaming and non-streaming wires
- [ ] cortex-cloud retrieval-cache config flag → polar backend
- [ ] Shim phase dispatch — steer (#6b): per-token hidden modification
      via `forward_full_gpu_with_cache_returning_hidden`; sequential
      composition in declared order
- [ ] Shim phase dispatch — injection (#6c): residual-add at
      `entrance:N` via `pre_block_hidden_inject` parameter on
      `forward_block_gpu_inner`; sum composition (commutative)
- [ ] QJL correction on V dequant (currently K-only) to close the cosine-
      similarity gap on attention output (PolarQuant alone hits ~0.84)
- [ ] Bit-pack 3-bit angle representation (u8 → 3-bits, ~12x compression)
- [ ] Wire QuantizedKvCache into cortex-cloud as the cache_pool backing store
- [ ] Move retrieval (bidirectional attention) from engram into cortex
- [ ] Move HierarchicalCache + consolidation from engram
- [ ] WgpuLinear from neuralkv-core shaders
- [ ] project_qk() method on TransformerModel for memory integration
- [ ] Wire into AgentOS as `handler: cortex` organism listener
