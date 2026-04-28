//! Feed-forward network trait — the abstraction point for dense vs MoE FFN.
//!
//! Any FFN block in a transformer (SwiGLU, MoE, etc.) implements this trait.
//! `TransformerBlock` holds a `Box<dyn FeedForward>` and doesn't care
//! whether it's a single dense FFN or a mixture of experts.

/// A feed-forward network block in a transformer layer.
///
/// Implementations:
/// - [`SwiGLU`](super::swiglu::SwiGLU) — dense gated FFN (standard LLaMA)
/// - [`MoELayer`](super::moe::MoELayer) — mixture of experts (Mixtral, DeepSeek)
pub trait FeedForward: Send + Sync + std::fmt::Debug {
    /// Forward pass for a single token vector.
    ///
    /// Input: f32 slice of length `in_features()`.
    /// Output: f32 vec of length `out_features()`.
    fn forward(&self, input: &[f32]) -> Vec<f32>;

    /// Forward pass over a sequence of tokens.
    ///
    /// `input`: flat f32 slice of shape `[seq_len, in_features()]`.
    /// Returns: flat f32 vec of shape `[seq_len, out_features()]`.
    fn forward_sequence(&self, input: &[f32], seq_len: usize) -> Vec<f32>;

    /// Input dimension (embed_dim).
    fn in_features(&self) -> usize;

    /// Output dimension (embed_dim).
    fn out_features(&self) -> usize;

    /// Downcast hook so orchestrators (`GpuEngine`) can reach the concrete
    /// type to access weight buffers. Each impl provides:
    /// `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn std::any::Any;
}
