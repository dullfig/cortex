//! # cortex
//!
//! Universal local transformer engine with persistent memory.
//!
//! Runs any GGUF model — ternary {-1,0,+1}, quantized (Q4_K, Q8), or float —
//! through the same transformer stack. The `LinearLayer` trait abstracts over
//! weight formats; the `ComputeBackend` trait abstracts over hardware.
//!
//! ## Architecture
//!
//! ```text
//! cortex
//! ├── tensor        — weight storage: ternary (2-bit), quantized, float
//! ├── ops           — kernels: ternary matmul, LUT, quantization, dequantization
//! ├── compute       — backends: scalar, AVX2, wgpu (GPU)
//! ├── layers        — transformer stack: embedding → attention → FFN → output
//! │   ├── linear    — trait: BitLinear | FloatLinear | WgpuLinear
//! │   ├── attention — GQA with RoPE, causal mask, KV cache
//! │   ├── swiglu    — gated FFN (SiLU or ReLU²)
//! │   ├── model     — full TransformerModel: forward, generate, retrieve
//! │   └── memory    — trait: persistent compressed memory (engram implements)
//! ├── gguf          — GGUF v3 parser: TQ1_0, TQ2_0, I2S, Q4_K, F16, F32
//! ├── tokenizer     — BPE tokenizer from GGUF metadata
//! └── loader        — load_model(): GGUF → detect weights → pick LinearLayer → go
//! ```
//!
//! ## Model support
//!
//! The GGUF loader auto-detects weight types and selects the right `LinearLayer`:
//!
//! | Weight type        | LinearLayer   | Compute          |
//! |--------------------|---------------|------------------|
//! | TQ1_0, TQ2_0, I2S | `BitLinear`   | CPU (AVX2/scalar) |
//! | Q4_K, Q5_K, Q6_K  | `FloatLinear` | CPU (dequant+f32) |
//! | F16, BF16, F32     | `FloatLinear` | CPU or GPU        |
//! | Any (with GPU)     | `WgpuLinear`  | GPU (WGPU shaders) |
//!
//! ## Memory (optional)
//!
//! When enabled, the transformer gains persistent associative memory via
//! the `TransformerMemory` trait. The same model's Q/K projections encode
//! memories — one embedding space, one mind.
//!
//! ```text
//! model.generate()  → causal attention, KV cache, logits (standard)
//! model.retrieve()  → bidirectional attention, compressed cache, scores (memory)
//! ```
//!
//! ## Lineage
//!
//! cortex absorbs and generalizes:
//! - **ternary-rs**: ternary kernels, BitLinear, GGUF loader, full transformer stack
//! - **engram**: compressed KV cache (PolarQuant), tiered memory, retrieval, consolidation
//! - **neuralkv-core** (GPU path): WGPU shaders for matmul, attention, FFN

pub mod tensor;
pub mod ops;
pub mod compute;
pub mod layers;
pub mod gguf;
pub mod tokenizer;
pub mod loader;

pub use tensor::{TernaryTensor, ActivationTensor, Ternary};
pub use gguf::{GgufFile, GgufError, GgmlType, TensorInfo, ModelConfig, MetadataValue};
pub use tokenizer::Tokenizer;
pub use loader::{load_model, LoadedModel};
pub use layers::memory::{TransformerMemory, MemoryConfig, MemoryResult, MemoryRole, MemoryTier};
pub use layers::ffn::FeedForward;
pub use layers::transformer::FfnInjector;
#[cfg(feature = "memory")]
pub use layers::engram_memory::EngramMemory;
