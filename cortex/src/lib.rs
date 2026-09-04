//! # cortex
//!
//! Float transformer inference engine with persistent memory. Targets
//! Qwen-class GGUF models (Q4_K_M, F16, BF16, F32) on GPU via wgpu.
//!
//! Note: ternary/BitNet inference moved to the sibling `ternary-rs`
//! crate on 2026-05-29 (see git tag `bitnet-archive-2026-05-29` for
//! the last cortex commit with the ternary path).
//!
//! ## Architecture
//!
//! ```text
//! cortex
//! ├── tensor        — float weight storage (FloatTensor)
//! ├── ops           — kernels: dequant (Q4_K → f32), polar quant, QJL
//! ├── compute       — backends: scalar, AVX2, wgpu (GPU)
//! ├── layers        — transformer stack: embedding → attention → FFN → output
//! │   ├── linear    — FloatLinear (Q4_K, F16, F32 weights, f32 matmul)
//! │   ├── attention — GQA with RoPE, causal mask, KV cache
//! │   ├── swiglu    — gated FFN (SiLU activation)
//! │   ├── model     — full TransformerModel: forward, generate, retrieve
//! │   └── memory    — trait: persistent compressed memory (engram implements)
//! ├── gguf          — GGUF v3 parser: Q4_K, Q5_K, Q6_K, F16, BF16, F32
//! ├── tokenizer     — BPE tokenizer from GGUF metadata
//! └── loader        — load_model(): GGUF → FloatLinear → go
//! ```
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
//! - **engram**: compressed KV cache (PolarQuant), tiered memory, retrieval, consolidation
//! - **neuralkv-core** (GPU path): WGPU shaders for matmul, attention, FFN

pub mod tensor;
pub mod ops;
pub mod compute;
pub mod layers;
pub mod gguf;
pub mod tokenizer;
pub mod loader;

pub use tensor::FloatTensor;
pub use gguf::{GgufFile, GgufError, GgmlType, TensorInfo, ModelConfig, MetadataValue};
pub use tokenizer::Tokenizer;
pub use loader::{load_model, LoadedModel};
pub use layers::memory::{TransformerMemory, MemoryConfig, MemoryResult, MemoryRole, MemoryTier};
pub use layers::ffn::FeedForward;
pub use layers::trace::ForwardTrace;
pub use layers::transformer::FfnInjector;
#[cfg(feature = "memory")]
pub use layers::engram_memory::EngramMemory;

// Re-export the wgpu crate so downstream crates (cortex-cloud) can
// reference `wgpu::Buffer` and friends without a separate dependency
// declaration. Only available when the gpu feature is enabled (which
// it is by default).
#[cfg(feature = "gpu")]
pub use wgpu;
/// Re-exported so serving layers can name vram-heap types (e.g. the
/// `Error` returned by fallible cache allocation) without a direct dep.
#[cfg(feature = "gpu")]
pub use vram_heap;
