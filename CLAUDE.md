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

369 tests covering: ternary packing, matmul kernels, quantization, GGUF parsing,
layer forward passes, attention, RoPE, SwiGLU, full model forward, sampler,
retrieval (forward_traced + attention-score ranking), TurboQuant
compression (PolarQuant + QJL + QuantizedKvCache).

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
- [ ] GPU shader for compressed KV (rotate Q + dot against angle/radius)
- [ ] Bit-pack 3-bit angle representation (u8 → 3-bits, ~12x compression)
- [ ] Wire QuantizedKvCache into cortex-cloud as the cache_pool backing store
- [ ] Move retrieval (bidirectional attention) from engram into cortex
- [ ] Move HierarchicalCache + consolidation from engram
- [ ] WgpuLinear from neuralkv-core shaders
- [ ] project_qk() method on TransformerModel for memory integration
- [ ] Wire into AgentOS as `handler: cortex` organism listener
