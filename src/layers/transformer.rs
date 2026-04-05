//! Transformer block — one layer of a LLaMA-style decoder.
//!
//! Each block applies:
//!   1. RMSNorm on input (attention norm)
//!   2. Multi-head attention (with residual connection)
//!   3. RMSNorm on attention output (FFN norm)
//!   4. SwiGLU feed-forward (with residual connection)
//!
//! Pre-norm architecture (norm before sublayer, not after):
//!
//!   h = x + Attention(attn_norm(x))
//!   out = h + FFN(ffn_norm(h))

use crate::layers::attention::MultiHeadAttention;
use crate::layers::ffn::FeedForward;
use crate::layers::kv_cache::KvCache;
use crate::layers::rmsnorm::RmsNorm;

/// A single transformer decoder block.
///
/// Supports optional Block Attention Residuals (MoonshotAI): learned
/// per-block scaling of attention and FFN residual streams. When set,
/// the residual connection becomes `h = x + α * sublayer(x)` instead
/// of the standard `h = x + sublayer(x)`.
pub struct TransformerBlock {
    /// Pre-attention normalization.
    attn_norm: RmsNorm,
    /// Multi-head (grouped query) attention.
    /// In BitNet b1.58, the attention layer includes an internal sub-norm
    /// before the O projection (set via `MultiHeadAttention::set_o_sub_norm`).
    attention: MultiHeadAttention,
    /// Pre-FFN normalization.
    ffn_norm: RmsNorm,
    /// Feed-forward network (dense SwiGLU or MoE).
    ffn: Box<dyn FeedForward>,
    /// Attention residual scale (Block Attention Residuals).
    /// 1.0 = standard residual connection.
    attn_residual_scale: f32,
    /// FFN residual scale (Block Attention Residuals).
    /// 1.0 = standard residual connection.
    ffn_residual_scale: f32,
}

impl TransformerBlock {
    /// Create a transformer block from its components.
    pub fn new(
        attn_norm: RmsNorm,
        attention: MultiHeadAttention,
        ffn_norm: RmsNorm,
        ffn: Box<dyn FeedForward>,
    ) -> Self {
        assert_eq!(
            attention.embed_dim(),
            ffn.in_features(),
            "attention embed_dim must match FFN input dim"
        );
        assert_eq!(
            ffn.in_features(),
            ffn.out_features(),
            "FFN must preserve embed_dim"
        );

        Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            attn_residual_scale: 1.0,
            ffn_residual_scale: 1.0,
        }
    }

    /// Create a transformer block with optional sub-normalization (BitNet b1.58).
    ///
    /// `attn_sub_norm` is applied inside the attention layer, before the O projection.
    /// This normalizes the concatenated head outputs before quantization in the
    /// output projection, matching the BitNet b1.58 architecture where each
    /// BitLinear has an internal activation norm.
    pub fn with_sub_norms(
        attn_norm: RmsNorm,
        mut attention: MultiHeadAttention,
        attn_sub_norm: Option<RmsNorm>,
        ffn_norm: RmsNorm,
        ffn: Box<dyn FeedForward>,
    ) -> Self {
        assert_eq!(
            attention.embed_dim(),
            ffn.in_features(),
            "attention embed_dim must match FFN input dim"
        );
        assert_eq!(
            ffn.in_features(),
            ffn.out_features(),
            "FFN must preserve embed_dim"
        );

        // Move attn_sub_norm into the attention layer (before O projection)
        if let Some(norm) = attn_sub_norm {
            attention.set_o_sub_norm(norm);
        }

        Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
            attn_residual_scale: 1.0,
            ffn_residual_scale: 1.0,
        }
    }

    /// Set Block Attention Residual scales (MoonshotAI).
    ///
    /// `attn_scale` modulates the attention sublayer output before
    /// adding to the residual stream. `ffn_scale` does the same for FFN.
    /// Values of 1.0 give standard residual connections.
    pub fn set_residual_scales(&mut self, attn_scale: f32, ffn_scale: f32) {
        self.attn_residual_scale = attn_scale;
        self.ffn_residual_scale = ffn_scale;
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.attention.embed_dim()
    }

    /// Access the attention layer (for querying head config).
    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }

    /// Forward pass over a sequence.
    ///
    /// `input`: flat f32 slice of shape `[seq_len, embed_dim]`.
    /// `start_pos`: sequence position of the first token (for RoPE).
    /// Returns: flat f32 vec of shape `[seq_len, embed_dim]`.
    pub fn forward(&self, input: &[f32], seq_len: usize, start_pos: usize) -> Vec<f32> {
        let embed_dim = self.embed_dim();
        assert_eq!(input.len(), seq_len * embed_dim, "input shape mismatch");

        // 1. Attention sub-block with residual
        //    Note: attn_sub_norm (if present) is applied inside attention,
        //    before the O projection — see MultiHeadAttention::set_o_sub_norm().
        //    Block Attention Residuals: h = x + α_attn * attn(norm(x))
        let normed_for_attn = self.norm_sequence(&self.attn_norm, input, seq_len);
        let attn_out = self.attention.forward(&normed_for_attn, seq_len, start_pos);

        let attn_scale = self.attn_residual_scale;
        let mut h = Vec::with_capacity(input.len());
        for (x, a) in input.iter().zip(attn_out.iter()) {
            h.push(x + attn_scale * a);
        }

        // 2. FFN sub-block with residual
        //    Block Attention Residuals: out = h + α_ffn * ffn(norm(h))
        let normed_for_ffn = self.norm_sequence(&self.ffn_norm, &h, seq_len);
        let ffn_out = self.ffn.forward_sequence(&normed_for_ffn, seq_len);

        let ffn_scale = self.ffn_residual_scale;
        for (h_val, f_val) in h.iter_mut().zip(ffn_out.iter()) {
            *h_val += ffn_scale * f_val;
        }

        h
    }

    /// Forward pass with KV cache for incremental generation.
    ///
    /// Same as `forward()` but uses the cache for attention K/V storage.
    pub fn forward_cached(
        &self,
        input: &[f32],
        seq_len: usize,
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let embed_dim = self.embed_dim();
        assert_eq!(input.len(), seq_len * embed_dim, "input shape mismatch");

        // 1. Attention sub-block with residual (Block Attention Residuals)
        let normed_for_attn = self.norm_sequence(&self.attn_norm, input, seq_len);
        let attn_out = self.attention.forward_cached(&normed_for_attn, seq_len, cache);

        let attn_scale = self.attn_residual_scale;
        let mut h = Vec::with_capacity(input.len());
        for (x, a) in input.iter().zip(attn_out.iter()) {
            h.push(x + attn_scale * a);
        }

        // 2. FFN sub-block with residual (Block Attention Residuals)
        let normed_for_ffn = self.norm_sequence(&self.ffn_norm, &h, seq_len);
        let ffn_out = self.ffn.forward_sequence(&normed_for_ffn, seq_len);

        let ffn_scale = self.ffn_residual_scale;
        for (h_val, f_val) in h.iter_mut().zip(ffn_out.iter()) {
            *h_val += ffn_scale * f_val;
        }

        h
    }

    /// Apply RmsNorm to each token in a sequence.
    fn norm_sequence(&self, norm: &RmsNorm, input: &[f32], seq_len: usize) -> Vec<f32> {
        let embed_dim = self.embed_dim();
        let mut output = Vec::with_capacity(input.len());
        for t in 0..seq_len {
            let start = t * embed_dim;
            let token = &input[start..start + embed_dim];
            output.extend_from_slice(&norm.forward(token));
        }
        output
    }
}

impl std::fmt::Debug for TransformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerBlock(embed={}, {:?}, {:?})",
            self.embed_dim(),
            self.attention,
            self.ffn,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::bitlinear::BitLinear;
    use crate::layers::linear::LinearLayer;
    use crate::layers::swiglu::SwiGLU;
    use crate::tensor::{Ternary, TernaryTensor};

    fn make_bitlinear(weights: &[i8], rows: usize, cols: usize, scale: f32) -> BitLinear {
        let ternary: Vec<Ternary> = weights
            .iter()
            .map(|&v| match v {
                -1 => Ternary::Neg,
                0 => Ternary::Zero,
                1 => Ternary::Pos,
                _ => panic!("not ternary"),
            })
            .collect();
        BitLinear::new(TernaryTensor::pack(&ternary, rows, cols), scale)
    }

    fn make_identity_proj(out_dim: usize, in_dim: usize) -> Box<dyn LinearLayer> {
        let mut weights = vec![0i8; out_dim * in_dim];
        for i in 0..out_dim.min(in_dim) {
            weights[i * in_dim + i] = 1;
        }
        Box::new(make_bitlinear(&weights, out_dim, in_dim, 1.0))
    }

    fn make_test_block(embed_dim: usize, n_heads: usize, n_kv_heads: usize, intermediate: usize) -> TransformerBlock {
        let head_dim = embed_dim / n_heads;
        let kv_dim = n_kv_heads * head_dim;

        // Attention projections
        let q_proj = make_identity_proj(embed_dim, embed_dim);
        let k_proj: Box<dyn LinearLayer> = {
            let mut w = vec![0i8; kv_dim * embed_dim];
            for i in 0..kv_dim.min(embed_dim) {
                w[i * embed_dim + i] = 1;
            }
            Box::new(make_bitlinear(&w, kv_dim, embed_dim, 1.0))
        };
        let v_proj: Box<dyn LinearLayer> = {
            let mut w = vec![0i8; kv_dim * embed_dim];
            for i in 0..kv_dim.min(embed_dim) {
                w[i * embed_dim + i] = 1;
            }
            Box::new(make_bitlinear(&w, kv_dim, embed_dim, 1.0))
        };
        let o_proj = make_identity_proj(embed_dim, embed_dim);

        let attention = MultiHeadAttention::new(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, 10000.0,
        );

        // FFN projections
        let gate = make_identity_proj(intermediate, embed_dim);
        let up = make_identity_proj(intermediate, embed_dim);
        let down = make_identity_proj(embed_dim, intermediate);
        let ffn = SwiGLU::new(gate, up, down);

        // Norms (unit weights)
        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-5);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-5);

        TransformerBlock::new(attn_norm, attention, ffn_norm, Box::new(ffn))
    }

    #[test]
    fn construction() {
        let block = make_test_block(8, 2, 2, 16);
        assert_eq!(block.embed_dim(), 8);
    }

    #[test]
    fn forward_single_token_shape() {
        let block = make_test_block(8, 2, 2, 16);
        let input = vec![1.0f32; 8];
        let output = block.forward(&input, 1, 0);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn forward_sequence_shape() {
        let block = make_test_block(8, 2, 2, 16);
        let seq_len = 4;
        let input = vec![0.5f32; seq_len * 8];
        let output = block.forward(&input, seq_len, 0);
        assert_eq!(output.len(), seq_len * 8);
    }

    #[test]
    fn forward_finite() {
        let block = make_test_block(8, 2, 2, 16);
        let input = vec![0.3f32; 8];
        let output = block.forward(&input, 1, 0);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_gqa_block() {
        // 4 Q heads, 2 KV heads
        let block = make_test_block(16, 4, 2, 32);
        assert_eq!(block.embed_dim(), 16);
        let input = vec![0.5f32; 16];
        let output = block.forward(&input, 1, 0);
        assert_eq!(output.len(), 16);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "gqa output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn residual_connection_adds() {
        // With identity-ish projections, the residual should make
        // the output different from (and generally larger than) the
        // sublayer output alone.
        let block = make_test_block(8, 2, 2, 16);
        let input = vec![1.0f32; 8];
        let output = block.forward(&input, 1, 0);

        // Output should not be identical to input (sublayers add something)
        assert!(
            input.iter().zip(output.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "residual should modify the output"
        );
    }

    #[test]
    fn different_inputs_different_outputs() {
        let block = make_test_block(8, 2, 2, 16);
        let out_a = block.forward(&vec![1.0f32; 8], 1, 0);
        let out_b = block.forward(&vec![-1.0f32; 8], 1, 0);

        assert!(
            out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "different inputs should produce different outputs"
        );
    }

    #[test]
    fn multi_token_sequence_runs() {
        // Verify a multi-token sequence runs end-to-end through
        // norm → attention (with causal mask) → residual → norm → FFN → residual.
        // With synthetic identity weights, detailed numerical assertions
        // aren't meaningful (quantization noise dominates), but the full
        // pipeline must produce finite output of correct shape.
        let block = make_test_block(8, 2, 2, 16);
        let seq_len = 5;
        let mut input = Vec::with_capacity(seq_len * 8);
        for t in 0..seq_len {
            for d in 0..8 {
                input.push((t as f32 * 0.3) + (d as f32 * 0.1));
            }
        }

        let output = block.forward(&input, seq_len, 0);
        assert_eq!(output.len(), seq_len * 8);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn debug_format() {
        let block = make_test_block(8, 2, 2, 16);
        let debug = format!("{:?}", block);
        assert!(debug.contains("TransformerBlock"));
        assert!(debug.contains("embed=8"));
    }

    #[test]
    fn residual_scales_default_to_one() {
        let block = make_test_block(8, 2, 2, 16);
        // Default scales = 1.0 means standard residual
        let input = vec![1.0f32; 8];
        let output_default = block.forward(&input, 1, 0);

        // Explicitly set to 1.0 — should give identical output
        let mut block2 = make_test_block(8, 2, 2, 16);
        block2.set_residual_scales(1.0, 1.0);
        let output_explicit = block2.forward(&input, 1, 0);

        for (a, b) in output_default.iter().zip(output_explicit.iter()) {
            assert!((a - b).abs() < 1e-7, "scale=1.0 should match default");
        }
    }

    #[test]
    fn residual_scales_zero_skips_sublayers() {
        // With scale=0.0, the sublayer output is zeroed out.
        // The residual connection becomes h = x + 0*sublayer(x) = x for each sub-block.
        let mut block = make_test_block(8, 2, 2, 16);
        block.set_residual_scales(0.0, 0.0);

        let input = vec![0.5f32; 8];
        let output = block.forward(&input, 1, 0);

        // Output should equal the input (both sublayers are suppressed)
        for (i, (x, y)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "with zero scales, output[{i}] should match input: {x} vs {y}"
            );
        }
    }

    #[test]
    fn residual_scales_affect_output() {
        let input = vec![1.0f32; 8];

        let block_full = make_test_block(8, 2, 2, 16);
        let out_full = block_full.forward(&input, 1, 0);

        let mut block_half = make_test_block(8, 2, 2, 16);
        block_half.set_residual_scales(0.5, 0.5);
        let out_half = block_half.forward(&input, 1, 0);

        // Different scales should produce different outputs
        assert!(
            out_full.iter().zip(out_half.iter()).any(|(a, b)| (a - b).abs() > 1e-5),
            "different residual scales should produce different outputs"
        );
    }
}
