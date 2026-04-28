//! Mixture of Experts (MoE) feed-forward layer.
//!
//! Replaces the single dense SwiGLU FFN with N expert FFNs and a learned
//! router that selects top-k experts per token. Used in Mixtral, DeepSeek,
//! Qwen MoE, and similar architectures.
//!
//! Architecture:
//!
//!   router_logits = router(input)              // embed_dim → n_experts
//!   top_k_indices = top_k(router_logits, k)
//!   gate_weights  = softmax(router_logits[top_k_indices])
//!   output = Σ gate_weights[i] * experts[top_k_indices[i]].forward(input)

use crate::layers::ffn::FeedForward;
use crate::layers::linear::LinearLayer;
use crate::layers::swiglu::SwiGLU;

/// Mixture of Experts layer.
///
/// Each token is routed to `top_k` out of `n_experts` expert FFNs.
/// The router is a learned linear projection from embed_dim to n_experts.
pub struct MoELayer {
    /// Expert FFN networks (all same architecture, different weights).
    experts: Vec<SwiGLU>,
    /// Router: embed_dim → n_experts (logits for expert selection).
    router: Box<dyn LinearLayer>,
    /// Number of experts to activate per token.
    top_k: usize,
}

impl MoELayer {
    /// Create a MoE layer.
    ///
    /// - `experts`: Vec of SwiGLU experts (all must have same dimensions).
    /// - `router`: Linear projection from embed_dim to n_experts.
    /// - `top_k`: Number of experts activated per token (typically 2).
    pub fn new(
        experts: Vec<SwiGLU>,
        router: Box<dyn LinearLayer>,
        top_k: usize,
    ) -> Self {
        assert!(!experts.is_empty(), "MoE must have at least one expert");
        assert!(
            top_k > 0 && top_k <= experts.len(),
            "top_k ({top_k}) must be in 1..=n_experts ({})",
            experts.len()
        );

        // Verify all experts have the same dimensions
        let in_f = experts[0].in_features();
        let out_f = experts[0].out_features();
        for (i, expert) in experts.iter().enumerate() {
            assert_eq!(
                expert.in_features(), in_f,
                "expert {i} in_features ({}) != expected ({in_f})",
                expert.in_features()
            );
            assert_eq!(
                expert.out_features(), out_f,
                "expert {i} out_features ({}) != expected ({out_f})",
                expert.out_features()
            );
        }

        // Router must map from embed_dim to n_experts
        assert_eq!(
            router.in_features(), in_f,
            "router in_features ({}) must match expert in_features ({in_f})",
            router.in_features()
        );
        assert_eq!(
            router.out_features(), experts.len(),
            "router out_features ({}) must match n_experts ({})",
            router.out_features(), experts.len()
        );

        Self { experts, router, top_k }
    }

    /// Number of experts.
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    /// Number of experts activated per token.
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

impl FeedForward for MoELayer {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let n_experts = self.experts.len();
        let k = self.top_k;

        // 1. Route: compute expert logits
        let logits = self.router.forward(input);
        assert_eq!(logits.len(), n_experts);

        // 2. Top-k selection
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
        let top_k_logits: Vec<f32> = indexed[..k].iter().map(|&(_, v)| v).collect();

        // 3. Softmax over selected expert logits (gate weights)
        let max_logit = top_k_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = top_k_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        let gate_weights: Vec<f32> = exp_values.iter().map(|&e| e / sum_exp).collect();

        // 4. Weighted sum of expert outputs
        let out_dim = self.experts[0].out_features();
        let mut output = vec![0.0f32; out_dim];

        for (i, &expert_idx) in top_k_indices.iter().enumerate() {
            let expert_out = self.experts[expert_idx].forward(input);
            let w = gate_weights[i];
            for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
                *o += w * e;
            }
        }

        output
    }

    fn forward_sequence(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let in_dim = self.in_features();
        assert_eq!(input.len(), seq_len * in_dim, "input shape mismatch");

        let out_dim = self.out_features();
        let mut output = Vec::with_capacity(seq_len * out_dim);
        for t in 0..seq_len {
            let start = t * in_dim;
            let token = &input[start..start + in_dim];
            output.extend_from_slice(&self.forward(token));
        }
        output
    }

    fn in_features(&self) -> usize {
        self.experts[0].in_features()
    }

    fn out_features(&self) -> usize {
        self.experts[0].out_features()
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl std::fmt::Debug for MoELayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MoE({}→{}→{}, {}/{} experts)",
            self.in_features(),
            self.experts[0].intermediate_size(),
            self.out_features(),
            self.top_k,
            self.experts.len(),
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

    fn make_test_expert(embed_dim: usize, intermediate: usize) -> SwiGLU {
        let gate = make_identity_proj(intermediate, embed_dim);
        let up = make_identity_proj(intermediate, embed_dim);
        let down = make_identity_proj(embed_dim, intermediate);
        SwiGLU::new(gate, up, down)
    }

    fn make_test_moe(embed_dim: usize, intermediate: usize, n_experts: usize, top_k: usize) -> MoELayer {
        let experts: Vec<SwiGLU> = (0..n_experts)
            .map(|_| make_test_expert(embed_dim, intermediate))
            .collect();

        // Router: embed_dim → n_experts (identity-ish, first n_experts dims map to experts)
        let router = make_identity_proj(n_experts, embed_dim);

        MoELayer::new(experts, router, top_k)
    }

    #[test]
    fn construction() {
        let moe = make_test_moe(8, 16, 4, 2);
        assert_eq!(moe.n_experts(), 4);
        assert_eq!(moe.top_k(), 2);
        assert_eq!(moe.in_features(), 8);
        assert_eq!(moe.out_features(), 8);
    }

    #[test]
    #[should_panic(expected = "at least one expert")]
    fn empty_experts_panics() {
        let router = make_identity_proj(4, 8);
        MoELayer::new(vec![], router, 2);
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn top_k_zero_panics() {
        make_test_moe(8, 16, 4, 0);
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn top_k_exceeds_experts_panics() {
        make_test_moe(8, 16, 4, 5);
    }

    #[test]
    fn forward_output_shape() {
        let moe = make_test_moe(8, 16, 4, 2);
        let input = vec![1.0f32; 8];
        let output = moe.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn forward_finite() {
        let moe = make_test_moe(8, 16, 4, 2);
        let input = vec![0.5f32; 8];
        let output = moe.forward(&input);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_sequence_shape() {
        let moe = make_test_moe(8, 16, 4, 2);
        let seq_len = 3;
        let input = vec![0.5f32; seq_len * 8];
        let output = moe.forward_sequence(&input, seq_len);
        assert_eq!(output.len(), seq_len * 8);
    }

    #[test]
    fn forward_sequence_matches_individual() {
        let moe = make_test_moe(4, 8, 2, 1);
        let tok0 = vec![1.0f32, 0.5, -0.3, 0.8];
        let tok1 = vec![-0.2f32, 0.7, 0.1, -0.5];

        let out0 = moe.forward(&tok0);
        let out1 = moe.forward(&tok1);

        let mut seq_input = tok0;
        seq_input.extend_from_slice(&tok1);
        let seq_output = moe.forward_sequence(&seq_input, 2);

        for (a, b) in out0.iter().chain(out1.iter()).zip(seq_output.iter()) {
            assert!((a - b).abs() < 1e-7, "sequence should match individual: {a} vs {b}");
        }
    }

    #[test]
    fn single_expert_moe_matches_dense() {
        // With 1 expert and top_k=1, MoE should produce identical output
        // to the single expert (gate weight = 1.0 after softmax of single element).
        let embed_dim = 4;
        let intermediate = 8;

        let expert = make_test_expert(embed_dim, intermediate);
        let moe = make_test_moe(embed_dim, intermediate, 1, 1);

        let input = vec![0.5f32, -0.3, 0.8, 0.1];
        let expert_out = expert.forward(&input);
        let moe_out = moe.forward(&input);

        for (a, b) in expert_out.iter().zip(moe_out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "single-expert MoE should match dense: {a} vs {b}"
            );
        }
    }

    #[test]
    fn gate_weights_sum_to_one() {
        // Indirectly verify: with all identical experts and top_k = n_experts,
        // the output should match any single expert (all gates sum to 1,
        // all expert outputs are identical).
        let moe = make_test_moe(4, 8, 3, 3);
        let expert = make_test_expert(4, 8);

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let moe_out = moe.forward(&input);
        let expert_out = expert.forward(&input);

        for (a, b) in moe_out.iter().zip(expert_out.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "identical experts with all gates should match single expert: {a} vs {b}"
            );
        }
    }

    #[test]
    fn debug_format() {
        let moe = make_test_moe(8, 16, 4, 2);
        let debug = format!("{:?}", moe);
        assert!(debug.contains("MoE"));
        assert!(debug.contains("2/4"));
    }
}
