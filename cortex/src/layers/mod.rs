//! Neural network layers for cortex transformer inference.
//!
//! Each layer operates on the tensor types from `tensor.rs` using the
//! kernels from `ops/`. The layers compose to build a full transformer
//! forward pass in `transformer.rs`.
//!
//! The `linear` trait abstracts over float weight formats (Q4_K, F16, F32).
//! The `memory` trait adds optional persistent associative memory.

pub mod attention;
#[cfg(feature = "memory")]
pub mod engram_memory;
pub mod ffn;
pub mod floatlinear;
#[cfg(feature = "gpu")]
pub mod gpu_engine;
#[cfg(feature = "gpu")]
pub mod gpu_floatlinear;
#[cfg(feature = "gpu")]
pub mod gpu_kv_cache;
#[cfg(feature = "gpu")]
pub mod gpu_polar;
#[cfg(feature = "gpu")]
pub mod gpu_polar_kv_cache;
pub mod kv_cache;
pub mod linear;
pub mod quantized_kv_cache;
pub mod memory;
pub mod model;
pub mod moe;
pub mod rmsnorm;
pub mod rope;
pub mod sampler;
pub mod swiglu;
pub mod trace;
pub mod transformer;
