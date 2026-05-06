//! Computational kernels for 1.58-bit inference and KV-cache compression.
//!
//! Weight kernels:
//! - **I2S** (`matmul`): Unpack 2-bit weights, conditional add/sub/skip.
//!   Simple, vectorization-friendly, good baseline.
//! - **LUT** (`lut`): Group 2 weights → 4-bit index into 9-entry precomputed
//!   table. Replaces all arithmetic with table lookups.
//! - **Quantize** (`quantize`): Absmax 8-bit activation quantization.
//!
//! TurboQuant KV compression (used by `layers::quantized_kv_cache`):
//! - **PolarQuant** (`polar`): random orthogonal rotation + 3-bit polar
//!   angle quantization. Stage 1, ~11x compression on f32 KV.
//! - **QJL** (`qjl`): 1-bit sign-of-projection residual correction. Stage 2,
//!   adds ~32 bits per (position, head) to refine attention dot products.

pub mod matmul;
pub mod lut;
pub mod quantize;
pub mod dequant;
pub mod polar;
pub mod qjl;
