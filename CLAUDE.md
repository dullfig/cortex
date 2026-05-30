> **Cross-session coordination:** Before making any design/scope decision, read `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\MEMORY.md` first — that folder is the shared brain across all Claude sessions on this project. Decisions pinned there supersede anything in this repo's older docs.
>
> **If something has happened to Daniel:** read `C:\src\CARETAKER.md` — the project's caretaker-handoff document.
>
> **2026-05-29: BitNet un-merge.** The ternary/BitNet inference path moved out of cortex into the sibling `ternary-rs` crate. Cortex is now a float-only Qwen-class GPU transformer (Q4_K_M, F16, BF16, F32). Architectural rationale: `project_training_time_representation.md` (pinned in integration repo) — representation choices belong at training time, not runtime polymorphism. Pre-excision cortex is preserved at git tag `bitnet-archive-2026-05-29` (commit e1e2dc1); GPU port artifacts went to `ternary-rs/incoming-cortex-gpu/` for ternary-rs's Stage 1.

# cortex

Float transformer inference engine with persistent memory. Targets Qwen-class GGUF models on GPU via wgpu.

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

cortex absorbs and generalizes:
- **engram** → compressed KV cache (PolarQuant + QJL CPU side, DONE);
  tiered memory + bidirectional-attention retrieval + consolidation (TODO)
- **neuralkv-core** (GPU path) → WGPU shaders for matmul, attention, FFN (DONE for matmul/attention/FFN)

Ternary/BitNet inference: see `ternary-rs` (un-merged 2026-05-29).

## Architecture (cortex core)

### Core
- **Tensor** (`cortex/src/tensor.rs`) — `FloatTensor` (f32)
- **GGUF** (`cortex/src/gguf.rs`) — Parser for Q4_K, Q5_K, Q6_K, F16, F32, BF16
- **Loader** (`cortex/src/loader.rs`) — `load_model()`: GGUF → FloatLinear → go

### Layers
- **LinearLayer trait** (`cortex/src/layers/linear.rs`) — single-impl seam (effectively `FloatLinear` only); kept for the trait shape pinkies depend on
- **FloatLinear** (`cortex/src/layers/floatlinear.rs`) — dequantized float linear (Q4_K, F16, F32)
- **GpuFloatLinear** (`cortex/src/layers/gpu_floatlinear.rs`) — GPU-resident f16-packed weights, GPU matmul
- **Attention** (`cortex/src/layers/attention.rs`) — GQA with RoPE, causal mask, KV cache
- **SwiGLU** (`cortex/src/layers/swiglu.rs`) — gated FFN (SiLU)
- **TransformerModel** (`cortex/src/layers/model.rs`) — full forward pass, generate, forward_cached
- **Memory trait** (`cortex/src/layers/memory.rs`) — TransformerMemory: ingest, retrieve, consolidate

### Compute Backends
- **WGPU** (`cortex/src/compute/wgpu_backend.rs`) — GPU via Vulkan/DX12/Metal (the only backend now; CPU scalar/AVX2 backends were ternary-only and left with the BitNet un-merge)

### TurboQuant KV compression (from engram)
- **PolarQuant** (`cortex/src/ops/polar.rs`) — random orthogonal rotation +
  3-bit polar angle quantization. Stage 1: ~7.5x reduction.
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

- Memory uses the SAME model's Q/K projections — one embedding space
- GGUF: Q4_K/Q5_K/Q6_K/F16/BF16/F32 → FloatLinear (dequantized at load)
- All f32 at layer boundaries (activations may be packed-f16 internally for bandwidth)
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

Workspace tests cover: GGUF parsing, layer forward passes, attention,
RoPE, SwiGLU, full model forward, sampler, retrieval (forward_traced +
attention-score ranking), TurboQuant compression (PolarQuant + QJL +
QuantizedKvCache + GPU score/value/derotate shaders + algorithm-quality
cosine pinning + resident GpuPolarKvCache storage + resident dispatchers
byte-equal to oneshot + GPU prefill compress shader + GPU-only f32→polar
conversion + multi-token causal-masked batch shaders + polar trace
forward through full model), hidden-state extraction hooks (per-block +
final post-norm) for shim runtime, ort link smoke.

For GPU-heavy tests, prefer `cargo test --workspace -- --test-threads=1`
to avoid VRAM contention between concurrently-running GPU tests on a
shared discrete GPU.

Run all: `cargo test --workspace`

## Roadmap

- [x] Full transformer forward pass (float)
- [x] KV cache for autoregressive generation
- [x] Token sampler (top-k, top-p, temperature)
- [x] TransformerMemory trait definition
- [x] cortex-cloud: OpenAI-compatible HTTP server
- [x] cortex-local: in-process provider for AgentOS
- [x] Move QuantizedKvCache from engram into cortex (CPU side)
- [x] GPU polar attention pipeline (compress / score / value / derotate)
- [x] cortex-cloud retrieve cache_load → polar backend wiring
      (`--enable-polar-cache`)
- [x] Shim phase dispatch — gate / steer / inject (all three phases compose)
- [x] `POST /v1/shims/embed` — text → pooled hidden-state vector
- [x] f16-activations rollout (Phases A → C3): KV cache + hidden_buf
      + per-block scratch packed f16
- [x] BitNet un-merge: ternary path moved to ternary-rs (2026-05-29).
      Tag: bitnet-archive-2026-05-29 (commit e1e2dc1).
- [ ] Restore C3 packed perf for Qwen now that BitNet's gone (currently
      hidden_buf + scratch.projected reverted to f32 for the Option E
      BitNet fix; ~9% Qwen prefill regression vs C3 baseline)
- [ ] Polar variant `forward_block_gpu_polar_inner` C3 port (rotate_q /
      derotate need packed variants; currently panic-guarded)
- [ ] QJL correction on V dequant (currently K-only) to close the cosine-
      similarity gap on attention output (PolarQuant alone hits ~0.84)
- [ ] Bit-pack 3-bit angle representation (u8 → 3-bits, ~12x compression)
- [ ] Wire QuantizedKvCache into cortex-cloud as the cache_pool backing store
- [ ] Move retrieval (bidirectional attention) from engram into cortex
- [ ] Move HierarchicalCache + consolidation from engram
- [ ] project_qk() method on TransformerModel for memory integration
- [ ] Wire into AgentOS as `handler: cortex` organism listener
