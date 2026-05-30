//! Computational kernels for cortex inference.
//!
//! Weight kernels:
//! - **Dequant** (`dequant`): GGUF Q4_K / Q5_K / Q6_K dequantization
//!   into f32 tensors for float matmul.
//!
//! TurboQuant KV compression (used by `layers::quantized_kv_cache`):
//! - **PolarQuant** (`polar`): random orthogonal rotation + 3-bit polar
//!   angle quantization. Stage 1, ~11x compression on f32 KV.
//! - **QJL** (`qjl`): 1-bit sign-of-projection residual correction. Stage 2,
//!   adds ~32 bits per (position, head) to refine attention dot products.

pub mod dequant;
pub mod polar;
pub mod qjl;
