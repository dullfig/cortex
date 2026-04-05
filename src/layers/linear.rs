//! LinearLayer trait — abstracts over ternary (BitLinear) and float linear layers.
//!
//! This enables the transformer stack to work with any weight format:
//! ternary {-1,0,+1}, standard quantized (Q4_K, Q8_0, etc.), or plain float.
//! All layer types are behind `Box<dyn LinearLayer>` for object-safe dispatch.

/// A linear projection layer: y = W · x (no bias).
///
/// Implemented by `BitLinear` (ternary) and `FloatLinear` (dequantized float).
/// Object-safe for use as `Box<dyn LinearLayer>`.
pub trait LinearLayer: Send + Sync + std::fmt::Debug {
    /// Forward pass for a single vector.
    ///
    /// Input: f32 vector of length `in_features`.
    /// Output: f32 vector of length `out_features`.
    fn forward(&self, input: &[f32]) -> Vec<f32>;

    /// Number of input features (columns).
    fn in_features(&self) -> usize;

    /// Number of output features (rows).
    fn out_features(&self) -> usize;
}
