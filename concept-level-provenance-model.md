# Concept-Level Provenance Model via LLM Distillation into SONAR Space

**Date:** 2025-04-05  
**Status:** Research direction — early ideation

## Core Insight

Humans think in concepts, not tokens. Large Concept Models (Meta/FAIR) and SONAR embeddings operate at the sentence/concept level rather than the token level. This might be the right granularity for both reasoning *and* provenance tracking.

## The Idea: Piggyback on Existing LLMs

Instead of training a concept-level model from scratch, use an existing LLM as an oracle to generate training data:

1. **Feed inputs to a standard LLM**, collect its outputs
2. **Encode both sides into SONAR embeddings** — producing (concept_in → concept_out) pairs
3. **Train a concept-level model** that learns the mapping entirely in embedding space

The LLM has already learned the hard part (language understanding, reasoning, world knowledge). The concept model learns the **semantic transfer function** the LLM implicitly computes — what it *means*, not what it *says*.

## Why This Works

- **Infinite training data.** Generate as many input/output pairs from the LLM as you need. No human annotation.
- **Massively reduced dimensionality.** Predicting a dense vector in SONAR space (regression) instead of a probability distribution over 50k+ tokens at each step (classification).
- **Surface-form invariant.** Semantically equivalent outputs that differ in wording map to the same SONAR neighborhood. The concept model learns the invariant.
- **Multilingual for free.** SONAR is a massively multilingual embedding space. A concept learned from English data is already positioned near its equivalent in other languages.

## Why This Matters for Provenance

This connects directly to the provenance-aware transformer architecture:

- **Tractable provenance tracking.** In token-space, tracing which inputs influenced which outputs means navigating thousands of attention interactions. In concept-space, you're tracing influence across a smaller number of semantic units with clear boundaries.
- **Decomposable transformations.** In continuous semantic space, the concept-level transformation might be decomposable — closer to a linear map you can inspect, rather than an opaque neural computation.
- **Natural granularity for provenance tags.** Each concept vector could carry provenance metadata (source, trust level, domain). This is the right level — coarser than tokens, finer than whole documents.
- **Drift detection.** SONAR's geometric structure means you could measure "how far did this output concept drift from the input concepts?" as an integrity/faithfulness signal.

## Where Meta Went Wrong — and Why Engram Points the Way

Meta's LCM takes the input stream, segments it into sentences, and encodes each sentence into a SONAR embedding. That's the wrong decomposition. Sentence boundaries are *syntactic* artifacts — they don't correspond to concept boundaries. A single sentence can convey multiple concepts; a single concept can span multiple sentences. By chopping on syntax, LCM loses the actual conceptual structure of the input before the model even begins.

**The fix is exactly what engram already does.** In `src/engram/src/retrieve/mod.rs`, the `retrieve()` function computes bidirectional attention over the entire KV cache — it uses dot-product similarity between current query projections and all cached positions, applies softmax, and aggregates across heads and tokens to produce a relevance score per position. It doesn't pre-segment the cache into chunks and hope the boundaries are meaningful. It lets attention *discover* what's relevant.

The same principle should apply to concept extraction:

1. **Don't pre-segment.** Don't chop the input into sentences and embed them. Instead, treat the full input as a continuous signal.
2. **Compute attention to discover concepts.** Use learned attention over the input to identify *which semantic units are actually being conveyed* — let the model find concept boundaries the way engram finds relevant cache positions.
3. **Then embed the discovered concepts.** Once attention has identified the conceptual units, project them into SONAR space (or a similar embedding space). Now each embedding corresponds to an actual concept, not an arbitrary sentence.

This is the difference between "embed what the syntax gives you" and "attend to what the semantics requires." Meta chose the former. The architecture should do the latter.

### You Need an LLM to Extract Concepts

This leads to a critical architectural realization: **concept extraction requires an LLM.** You cannot shortcut it.

Think about how humans process speech. You don't segment on pauses and then separately interpret each chunk. You *listen* — continuously building a mental model of what is being conveyed, integrating context, resolving ambiguity, recognizing when a new idea starts and an old one concludes. The "concept extraction" is inseparable from the language understanding itself.

The same is true for text. Determining which concepts a passage conveys *is* the hard problem of language understanding. A sentence tokenizer can't do it. A lightweight encoder can't do it. You need the full depth of an LLM's learned representations to:

- **Resolve what's being said vs. what's being meant.** Sarcasm, implication, rhetorical questions — the concept conveyed often diverges from the literal tokens.
- **Track what's new vs. what's repeated.** A good concept extractor recognizes when a passage is elaborating on an existing concept vs. introducing a new one.
- **Weigh what matters.** Not every clause conveys a concept worth tracking. An LLM understands salience — which parts carry the actual payload vs. which are syntactic scaffolding.

This reframes the architecture. The LLM isn't just the "oracle" that generates training data for a concept model (as described above). The LLM is the **tokens-to-concepts translator** — a necessary front-end that reads the token stream and emits a stream of concept embeddings. The concept-level model then reasons over those embeddings. You need both stages:

```
Token stream
      │
  LLM (tokens → concept embeddings)    ← understands language
      │
  Concept-level model (concepts → concepts)  ← reasons over meaning
      │
  Decoder (concepts → tokens)           ← produces output
```

This is why Meta's approach falls short: SONAR encodes surface form, not understood meaning. Encoding a sentence into SONAR space tells you what the sentence *says*, not what it *conveys in context*. The LLM is what bridges that gap — it's the component that actually *understands* before handing off to the concept-level reasoning stage.

### Implications for the Concept Model

- **Concept extraction is not a preprocessing step — it's a learned operation.** The boundaries between concepts should emerge from attention, not from a sentence tokenizer.
- **Variable-granularity concepts.** Some concepts are a single word ("fire!"), some are a paragraph-long argument. Attention naturally handles this — high-relevance regions can be narrow or wide.
- **Engram as proof of concept.** Engram already demonstrates that attention-based relevance discovery over a compressed representation works at scale (3-bit quantized keys, ~12x compression). The same approach can work for concept discovery over input text.

## Open Questions

- **Multi-concept reasoning.** LLMs don't just map single concepts 1:1. They combine, conditionalize, and synthesize across concepts. The concept model needs a sequence-to-sequence structure over concept vectors, not just pairwise mapping.
- **SONAR's limitations.** SONAR embeddings are trained on meaning equivalence, not on distinguishing trusted vs. untrusted sources. The provenance channel would still need to be a parallel signal, not something SONAR provides natively.
- **Concept boundary learning.** How do you train the attention mechanism that discovers concept boundaries? Possible approaches: distill from an LLM's internal attention patterns, use contrastive learning (nearby concepts should embed differently), or use reconstruction loss (discovered concepts must be sufficient to reconstruct the original).
- **Information loss.** Sentence-level embeddings compress away details that might matter for reasoning. Attention-discovered concepts might preserve more, but this needs validation.
- **Evaluation.** How do you evaluate a concept-level model? Token-level metrics (perplexity, BLEU) don't apply directly. Need concept-level metrics — maybe cosine similarity in SONAR space, or downstream task performance after decoding.

## Relationship to Provenance-Aware Architecture

The provenance channel idea (parallel read-only metadata channel) could operate at the concept level rather than the token level:

```
Input concepts + provenance tags
        │
  Concept-level model (trained via LLM distillation)
        │
Output concepts + provenance lineage
        │
  Decoder (concept → tokens, for human-readable output)
```

The concept level might be where provenance *naturally lives* — it's the level at which we can meaningfully say "this output idea came from that input idea."

## Next Steps

1. **Literature review:** Meta's LCM paper, SONAR embedding documentation, any work on concept-level distillation
2. **Feasibility check:** Can SONAR embeddings preserve enough information for round-trip (encode → transform → decode) to produce coherent text?
3. **Prototype:** Generate (input, output) pairs from an LLM, encode to SONAR, train a simple mapping model (even linear), see what happens
4. **Provenance experiment:** With the mapping model, can you trace which input concepts most influenced each output concept? Is this more interpretable than token-level attribution?

## References

- Large Concept Models (Meta FAIR, Dec 2024)
- SONAR: Massively multilingual sentence embeddings
- Provenance-aware transformer architecture (Cortex project)
