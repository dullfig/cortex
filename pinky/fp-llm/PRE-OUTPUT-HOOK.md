# Pre-Output Reasoning Hook — Design Document

**Date**: 2026-04-13
**Status**: Research direction. Not on the concierge critical path.
**Prerequisite**: GPU (4090) for meaningful iteration speed.
**Key insight**: The non-iterative version already works via
`set_block_injector(23, ...)` — no new code needed for single-pass
pre-output modification. Only the iterative loop needs new code (~20 lines).

## The observation

A standard transformer gets one pass from input to output. Every layer
simultaneously tries to understand the input AND produce the output.
There's no separation between "reading" and "reasoning about what I
read." Daniel's observation: **the transformer is missing a section** —
a reflection/reasoning stage between the model's computed understanding
and the final token prediction.

## The architecture

```
tokens → embed → 24 layers (understanding) → final_norm
    → [pre-output reasoning hook] ↔ iterates
    → output projection → token
```

The hook sits between the final norm and the output projection. It
operates in the model's richest representation space (post-24-layers,
post-norm) and can iterate within that space before committing to a
token.

### Already-working version (no new code)

`set_block_injector(23, injector)` attaches an FfnInjector to the
last transformer block. This fires after the final FFN, one step
before the final norm + output projection. For single-pass pre-output
modification (inject knowledge, nudge behavior, gate output), this
is functionally equivalent to the pre-output hook and requires zero
new code. Validated in the FP-LLM experiment (2026-04-12) where
delta injection at layers 16-23 smoothly steered base model behavior.

### Why this placement is better than Coconut

Coconut (arXiv:2412.06769) feeds the last hidden state back as the
NEXT position's input embedding. This crosses a representation
boundary: layer 1 expects token embeddings, layer 24 produces
contextual states. Requires retraining to bridge the gap.

The pre-output hook iterates within the SAME representation space.
No type mismatch, no retraining needed for the loop mechanism itself.

### The preview mechanism (needs ~20 lines of new code)

Because the output projection is just a linear map, the hook can
**preview** what token the model would pick at any iteration step:

```
loop {
    modified = reasoning_step(modified);
    preview_logits = output_proj(modified);
    if confident_enough(preview_logits) { break; }
}
final_logits = output_proj(modified);
```

Natural stopping criterion: the model thinks until it's confident,
then commits. Confidence = low entropy in the preview logits.

### What the hook can compose with

| Module | What it does | Exists? |
|---|---|---|
| NeuralKV retrieval | Inject domain knowledge as residual | Yes |
| FP-LLM delta injection | Nudge toward instruction-following | Yes |
| Iterative refinement | Think for N steps before committing | ~20 lines |
| Classifier gate | Check for policy violations before output | Yes |
| Confidence check | Preview logits, loop if entropy too high | ~10 lines |

### What this could enable

- **Multi-step reasoning** without chain-of-thought token emission
- **Confidence-gated output** (easy tokens fast, hard tokens get more compute)
- **Planning** via one-step lookahead in the preview loop
- **Self-correction** before output (policy violations caught pre-emission)

### When to build the iterative version

When the 4090 arrives AND the concierge is shipped AND there's a
concrete use case that needs iterative reasoning. Until then, the
single-pass version via `set_block_injector(23, ...)` covers the
FP-LLM use cases we've validated.
