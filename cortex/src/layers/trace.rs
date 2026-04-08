//! Forward-pass tracing — captured intermediate state for analysis.
//!
//! `ForwardTrace` holds per-layer attention scores and hidden states from a
//! `TransformerModel::forward_traced` call. It exists to support pinky
//! experiments on attention-based concept boundary discovery (and any other
//! analysis that needs to look inside the model rather than at its outputs).
//!
//! ## Layout
//!
//! - **`attention_scores[layer]`** is a flat `Vec<f32>` of shape
//!   `[n_heads * seq_len * seq_len]`, indexed as
//!   `scores[h * seq_len * seq_len + q * seq_len + k]`. The matrix is causal:
//!   for `k > q` the entry is 0.0 (the causal mask was applied before softmax,
//!   and softmax assigns zero weight to masked positions).
//! - **`hidden_states[i]`** is a flat `Vec<f32>` of shape `[seq_len * embed_dim]`.
//!   Index `i = 0` is the post-embedding hidden state (input to block 0).
//!   Index `i = 1..=n_layers` is the output of block `i - 1` (== input to block `i`).
//!   Index `i = n_layers + 1` is the post-final-norm state (input to the output projection).
//!
//! ## Cost
//!
//! Tracing is opt-in. The non-traced forward path (`forward`, `forward_cached`,
//! `forward_last`) is unchanged and pays nothing for trace support. The traced
//! path allocates `O(n_layers * (n_heads * seq_len^2 + seq_len * embed_dim))`
//! additional memory and clones a small amount of state per layer. For a
//! 24-layer 14-head 896-embed model on a 64-token input that's roughly
//! `24 * (14 * 4096 + 64 * 896) * 4` bytes ≈ 11 MB.

/// Captured intermediate state from a single forward pass.
///
/// Returned by `TransformerModel::forward_traced` alongside the final logits.
/// All fields are owned `Vec`s — the trace can outlive the model that produced
/// it (useful for offline analysis, serialization, etc.).
#[derive(Debug, Clone, Default)]
pub struct ForwardTrace {
    /// Per-layer post-softmax attention weights ("what attention decided").
    ///
    /// `attention_scores[layer]` has length `n_heads * seq_len * seq_len`.
    /// Entry at `[h, q, k]` (laid out as `h * seq_len * seq_len + q * seq_len + k`)
    /// is the attention weight from query position `q` to key position `k` in
    /// head `h`. The matrix is lower-triangular (causal): entries with `k > q`
    /// are 0.0. Each row over allowed keys sums to ~1.0 (softmax output).
    ///
    /// Note: in grouped-query attention (GQA) multiple query heads share a
    /// single KV head, but each query head still has its own row of softmax
    /// weights — so this captures per-query-head attention, which is the
    /// observable signal even when the underlying K/V is shared.
    pub attention_scores: Vec<Vec<f32>>,

    /// Per-layer pre-softmax attention scores ("what attention measured").
    ///
    /// Same shape and indexing as `attention_scores`, but holds the raw
    /// scaled `Q·K^T / sqrt(d)` values BEFORE softmax is applied. These
    /// are unbounded (can be any real number) and do not sum to 1.0 over
    /// the row.
    ///
    /// This buffer exists because **dilution is a property of softmax,
    /// not of attention**. For retrieval-style aggregations (e.g., top-K
    /// of raw scores over a large stored substrate), the pre-softmax
    /// values are what you want — softmax forces weights to sum to 1
    /// which makes per-position weights shrink as the cache grows, but
    /// the raw dot products do not have this property and produce
    /// dilution-free relevance signals at any cache size.
    ///
    /// See POSITION.md "Softmax is for inference, not for retrieval"
    /// for the architectural argument.
    pub pre_softmax_scores: Vec<Vec<f32>>,

    /// Per-layer hidden states (the residual stream).
    ///
    /// `hidden_states[0]` is the post-embedding state (input to block 0).
    /// `hidden_states[i]` for `1 <= i <= n_layers` is the output of block `i-1`,
    /// which equals the input to block `i` (or the input to the final norm if
    /// `i == n_layers`).
    /// `hidden_states[n_layers + 1]` is the post-final-norm state, the input
    /// to the output projection.
    ///
    /// Each entry has length `seq_len * embed_dim`.
    pub hidden_states: Vec<Vec<f32>>,

    /// Sequence length of the captured pass.
    pub seq_len: usize,

    /// Number of transformer layers.
    pub n_layers: usize,

    /// Number of (query) attention heads per layer.
    pub n_heads: usize,

    /// Embedding dimension.
    pub embed_dim: usize,
}

impl ForwardTrace {
    /// Create an empty trace sized for a model with the given shape.
    ///
    /// The internal `Vec`s are pre-allocated to the right outer length but
    /// the inner buffers are left empty — they're filled by the forward
    /// pass as it runs.
    pub fn new(n_layers: usize, n_heads: usize, embed_dim: usize, seq_len: usize) -> Self {
        Self {
            attention_scores: Vec::with_capacity(n_layers),
            pre_softmax_scores: Vec::with_capacity(n_layers),
            hidden_states: Vec::with_capacity(n_layers + 2),
            seq_len,
            n_layers,
            n_heads,
            embed_dim,
        }
    }

    /// Look up the attention weight from query position `q` to key position `k`
    /// in head `h` of layer `layer`.
    ///
    /// Returns 0.0 for causally-masked positions (`k > q`). Panics if any
    /// index is out of range.
    pub fn attention(&self, layer: usize, head: usize, q: usize, k: usize) -> f32 {
        let scores = &self.attention_scores[layer];
        let s = self.seq_len;
        scores[head * s * s + q * s + k]
    }

    /// Look up the full attention vector for query position `q` in head `h` of
    /// layer `layer`. Returns a slice of length `seq_len`.
    pub fn attention_row(&self, layer: usize, head: usize, q: usize) -> &[f32] {
        let scores = &self.attention_scores[layer];
        let s = self.seq_len;
        let start = head * s * s + q * s;
        &scores[start..start + s]
    }

    /// Look up the pre-softmax (raw scaled Q·K^T) score from query position
    /// `q` to key position `k` in head `h` of layer `layer`.
    pub fn pre_score(&self, layer: usize, head: usize, q: usize, k: usize) -> f32 {
        let scores = &self.pre_softmax_scores[layer];
        let s = self.seq_len;
        scores[head * s * s + q * s + k]
    }

    /// Look up the full pre-softmax score row for query position `q` in head
    /// `h` of layer `layer`. Returns a slice of length `seq_len`.
    /// These are the raw "what attention measured" values, useful for
    /// retrieval-style aggregations that don't need probability normalization.
    pub fn pre_score_row(&self, layer: usize, head: usize, q: usize) -> &[f32] {
        let scores = &self.pre_softmax_scores[layer];
        let s = self.seq_len;
        let start = head * s * s + q * s;
        &scores[start..start + s]
    }

    /// Look up the hidden state at sequence position `t` after layer `layer_idx`
    /// (using the indexing convention described in the struct doc). Returns
    /// a slice of length `embed_dim`.
    pub fn hidden(&self, layer_idx: usize, t: usize) -> &[f32] {
        let h = &self.hidden_states[layer_idx];
        let d = self.embed_dim;
        &h[t * d..(t + 1) * d]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trace_construction() {
        let trace = ForwardTrace::new(4, 8, 64, 16);
        assert_eq!(trace.n_layers, 4);
        assert_eq!(trace.n_heads, 8);
        assert_eq!(trace.embed_dim, 64);
        assert_eq!(trace.seq_len, 16);
        assert!(trace.attention_scores.is_empty());
        assert!(trace.hidden_states.is_empty());
    }

    #[test]
    fn attention_indexing() {
        let mut trace = ForwardTrace::new(1, 2, 4, 3);
        // Build a synthetic 2-head, 3-query, 3-key score tensor
        // Head 0: row 0 = [1.0, 0, 0], row 1 = [0.5, 0.5, 0], row 2 = [0.3, 0.3, 0.4]
        // Head 1: row 0 = [1.0, 0, 0], row 1 = [0.2, 0.8, 0], row 2 = [0.1, 0.1, 0.8]
        let scores: Vec<f32> = vec![
            // head 0
            1.0, 0.0, 0.0,
            0.5, 0.5, 0.0,
            0.3, 0.3, 0.4,
            // head 1
            1.0, 0.0, 0.0,
            0.2, 0.8, 0.0,
            0.1, 0.1, 0.8,
        ];
        trace.attention_scores.push(scores);

        assert_eq!(trace.attention(0, 0, 0, 0), 1.0);
        assert_eq!(trace.attention(0, 0, 1, 1), 0.5);
        assert_eq!(trace.attention(0, 0, 2, 2), 0.4);
        assert_eq!(trace.attention(0, 1, 1, 1), 0.8);
        assert_eq!(trace.attention(0, 1, 2, 0), 0.1);

        let row = trace.attention_row(0, 1, 2);
        assert_eq!(row, &[0.1, 0.1, 0.8]);
    }

    #[test]
    fn hidden_state_indexing() {
        let mut trace = ForwardTrace::new(1, 1, 4, 2);
        // 2 tokens, embed_dim=4
        let h: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        trace.hidden_states.push(h);

        assert_eq!(trace.hidden(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(trace.hidden(0, 1), &[5.0, 6.0, 7.0, 8.0]);
    }
}
