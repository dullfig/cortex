//! Neural network layers for universal transformer inference.
//!
//! Each layer operates on the tensor types from `tensor.rs` using the
//! kernels from `ops/`. The layers compose to build a full transformer
//! forward pass in `transformer.rs`.
//!
//! The `linear` trait abstracts over weight formats (ternary, quantized, float).
//! The `memory` trait adds optional persistent associative memory.

pub mod attention;
pub mod bitlinear;
#[cfg(feature = "memory")]
pub mod engram_memory;
pub mod ffn;
pub mod floatlinear;
#[cfg(feature = "gpu")]
pub mod gpu_bitlinear;
#[cfg(feature = "gpu")]
pub mod gpu_floatlinear;
pub mod kv_cache;
pub mod linear;
pub mod memory;
pub mod model;
pub mod moe;
pub mod rmsnorm;
pub mod rope;
pub mod sampler;
pub mod swiglu;
pub mod trace;
pub mod transformer;
