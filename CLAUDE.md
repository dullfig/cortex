# cortex

Universal local transformer engine with persistent memory. Runs any GGUF model — ternary, quantized, or float.

## Lineage

cortex absorbs and generalizes three projects:
- **ternary-rs** → ternary kernels, BitLinear, GGUF loader, full transformer stack (DONE)
- **engram** → compressed KV cache (PolarQuant), tiered memory, retrieval, consolidation (TODO)
- **neuralkv-core** (GPU path) → WGPU shaders for matmul, attention, FFN (TODO)

## Architecture

### Core (from ternary-rs)
- **Tensor** (`tensor.rs`) — 2-bit packed ternary, 8-bit quantized activations, float tensors
- **I2S Kernel** (`ops/matmul.rs`) — Ternary matvec via conditional add/sub/skip
- **LUT Kernel** (`ops/lut.rs`) — Lookup table kernel, zero arithmetic in hot loop
- **GGUF** (`gguf.rs`) — Parser for TQ1_0, TQ2_0, I2S, Q4_K, F16, F32, BF16
- **Loader** (`loader.rs`) — `load_model()`: GGUF → auto-detect → right LinearLayer → go

### Layers
- **LinearLayer trait** (`layers/linear.rs`) — the universal seam: BitLinear | FloatLinear | WgpuLinear
- **BitLinear** (`layers/bitlinear.rs`) — ternary linear: quantize → ternary matmul → rescale
- **FloatLinear** (`layers/floatlinear.rs`) — dequantized float linear (Q4_K, F16, F32)
- **Attention** (`layers/attention.rs`) — GQA with RoPE, causal mask, KV cache
- **SwiGLU** (`layers/swiglu.rs`) — gated FFN (SiLU or ReLU²)
- **TransformerModel** (`layers/model.rs`) — full forward pass, generate, forward_cached
- **Memory trait** (`layers/memory.rs`) — TransformerMemory: ingest, retrieve, consolidate

### Compute Backends
- **Scalar** (`compute/scalar.rs`) — portable fallback
- **AVX2** (`compute/avx2.rs`) — x86-64 SIMD
- **WGPU** (`compute/wgpu_backend.rs`) — GPU via Vulkan/DX12/Metal

### Memory (from engram, TODO)
- **QuantizedKvCache** — PolarQuant 3-bit compressed KV storage (12x reduction)
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
let model = cortex::load_model("model.gguf")?;
let tokens = model.tokenizer.encode("Hello");
let logits = model.model.forward(&tokens, 0);
let generated = model.model.generate(&tokens, &sampler, 256);
```

## Testing

282 tests covering: ternary packing, matmul kernels, quantization, GGUF parsing,
layer forward passes, attention, RoPE, SwiGLU, full model forward, sampler.

## Roadmap

- [x] Full transformer forward pass (ternary + float)
- [x] KV cache for autoregressive generation
- [x] Token sampler (top-k, top-p, temperature)
- [x] TransformerMemory trait definition
- [ ] Move QuantizedKvCache from engram into cortex
- [ ] Move retrieval (bidirectional attention) from engram into cortex
- [ ] Move HierarchicalCache + consolidation from engram
- [ ] WgpuLinear from neuralkv-core shaders
- [ ] project_qk() method on TransformerModel for memory integration
- [ ] Wire into AgentOS as `handler: cortex` organism listener
