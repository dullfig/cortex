# concept-boundaries

Pinky experiment 0. The goal is to test two hypotheses, in order:

1. **Per-token provenance is structurally sound.** Every token can carry a
   source tag from the moment it leaves the tokenizer, the tag rides
   through the model as a parallel read-only channel, and the tag survives
   to the output without ever being a target of computation. If this works,
   it's the substrate for everything downstream.

2. **Attention discovers concept boundaries on raw input.** Given the
   per-token states from a small LLM, dot-product attention patterns
   between adjacent positions and a small set of learned (or hand-built)
   "concept query" vectors will reveal boundary candidates that don't
   require sentence segmentation. Boundaries should fall *across* trust
   regions when the input is mixed-trust, not blur them.

The end-state of this experiment is a binary that takes a mixed-trust
fixture, runs a small model over it, and emits both the per-token
provenance trace and the discovered concept boundaries — and we get to
visually compare those boundaries to the trust region boundaries to see
how well they align.

## Status

| Hypothesis | Status |
|------------|--------|
| Per-token provenance pipeline | scaffold ready, demonstrable today |
| Attention-based boundary discovery | **blocked on cortex instrumentation** (see below) |

## What works today

```bash
cd pinky/concept-boundaries
cargo run -- --model /path/to/model.gguf --fixture fixtures/mixed-trust.txt
```

Output: a per-token table showing `idx | id | source | text` for every
token in the encoded fixture, plus the top-1 prediction from a single
forward pass to confirm the pipeline runs end-to-end.

The point of this output is the **provenance column**. Every byte of the
prompt-injection payload in the fixture's `doc:` line will show up as
`source=doc` in the trace, even though the bytes themselves *say*
"ignore all previous instructions." That's the entire point: the trust
label is structural, not content-derived, and an attacker who controls
the bytes does not control the tag.

## What's blocked on cortex instrumentation

The boundary-discovery half of the experiment needs cortex to expose
internals it currently hides. Specifically:

### 1. Per-layer hidden state capture

`TransformerModel::forward()` and `forward_cached()` currently return
only the final logits. For boundary discovery we need the per-layer
hidden states `h_l[t]` for `l = 0..n_layers, t = 0..seq_len`.

**Proposed API** (to add in cortex):

```rust
pub struct ForwardTrace {
    pub hidden: Vec<Vec<f32>>, // [n_layers + 1][seq_len * embed_dim]
    pub attn_scores: Vec<Vec<f32>>, // [n_layers][n_heads * seq_len * seq_len]
}

impl TransformerModel {
    pub fn forward_traced(&self, tokens: &[u32], start_pos: usize)
        -> (Vec<f32>, ForwardTrace);
}
```

The trace is opt-in (regular `forward` stays unchanged) so production
inference paths pay nothing. The implementation would clone the hidden
state at the entry to each transformer block, plus the raw attention
scores from `MultiHeadAttention::forward` before they're projected
through O.

### 2. Attention score capture

`MultiHeadAttention::forward()` computes `softmax(QK^T / sqrt(d))` and
immediately uses it to weight V — the score matrix is never returned.
We need an opt-in path that returns it. This is the smaller change of
the two; the score matrix is `[n_heads, seq_len, seq_len]` per layer
and is already computed in the existing forward pass.

### Cost / risk

Both changes are additive: new methods alongside the existing ones, no
behavior changes to the production path. The trace allocates more
memory (O(layers × seq_len × embed_dim) for hidden states) but only
when explicitly requested. No tests should break.

## What experiment 1 will look like once cortex is instrumented

```bash
cargo run -- --model qwen-0.5b.gguf --fixture fixtures/mixed-trust.txt --trace
```

Output:
- Per-token provenance trace (already works)
- Per-layer attention pattern dump (post-instrumentation)
- Boundary candidate report: positions where the dominant attention
  target shifts by more than a threshold, ranked by shift magnitude
- Alignment table: do discovered boundaries fall on trust-region edges?
- Visual: ASCII heatmap of attention scores in the last few layers

The **success criterion** is loose at this stage: do the discovered
boundaries fall on the trust-region boundaries from the fixture more
often than chance? If yes, the hypothesis has signal and we promote
the work toward a real cortex feature. If no, we learn something
interesting about why and either iterate or abandon.

## Why this matters

This is the smallest possible test of the architectural argument that
prompt injection is fundamentally a *unit-of-representation* problem
rather than a training problem. If concept boundaries are discoverable
and they correlate with trust regions, then the architecture supports
a kind of policy expression that token-level models structurally
cannot:

> *Refuse any concept whose per-token source trace crosses a trust
> boundary, or whose dominant source is not in the trusted set.*

That's a rule, not a reflex. Rules generalize. Reflexes don't.

See the root of this repo for the broader cortex / engram / NeuralKV
context, and `concept-level-provenance-model.md` at the repo root for
the full architectural argument.
