---
name: representation-emitting-models
description: "Meta-architectural recognition that the project repeatedly reaches for the same class of models — small-to-medium models whose output is a hidden-state / vector representation rather than text tokens, designed for consumption by another component. Memex's ingestion encoder, the per-layer injection auxiliary, the dual-head voice sub-heads, and various adapter approaches all instantiate this class. The meta-principle: 'the head of a model is interchangeable; the substrate is what matters' — token head produces text, representation head produces vectors for downstream use, audio head produces phonemes, steering head produces per-layer biases. Composes with training-time representation principle (head choice is a training-time architectural commitment) and explains why the project's architectural endgame has multiple coordinated heads emitting different representations rather than 'one big LLM does everything.' Specific realization for memex: encoder model produces compact retrieval representations for storage; per-layer-injection auxiliary projects compact stored representations back up to per-layer K vectors for cortex's substrate-attended generation. Articulated 2026-06-08 evening when Daniel recognized 'the adapter is yet another example of a model that instead of a token head, emits representations for storage' and walked the recognition to its conclusions."
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-08 evening during a deep-dive conversation about memex's KV cache substrate. Daniel was working through "what does the memex encoder actually store" and recognized that the adapter-based retrieval approach is structurally the same class of model as the per-layer-injection auxiliary and the dual-head voice sub-heads — they all emit representations rather than tokens. This pin captures the meta-architectural recognition and the specific realization for memex.

## The class: representation-emitting models

**Definition**: a model whose output is a hidden-state / vector representation rather than text tokens, designed for consumption by another component.

Three structural properties distinguish this class from token-head LLMs:

1. **Output type**: continuous vectors (typically activations or embeddings) rather than discrete tokens over a vocabulary
2. **Downstream consumer**: another model or component (cortex's attention, vocoder, storage substrate, etc.) rather than a human-readable output
3. **Training objective**: optimize for the downstream component's quality, not for next-token prediction

The token-head LLM is one specific commitment within a much larger design space. Representation-emitting models are the broader class; the project keeps reaching for them at different layers of the architecture.

## The project's existing instances

| Component | What it emits | Consumed by | Pin |
|---|---|---|---|
| **Memex ingestion encoder** | Compact retrieval-shaped K/V vectors per token | Memex retrieval (attention-as-classifier at query time) | `project_memex_identity.md` |
| **Per-layer injection auxiliary** | Per-layer steering vectors | Main LLM's residual stream at every layer | `project_per_layer_injection_auxiliary.md` |
| **Dual-head voice audio sub-heads** | Phoneme tokens / duration / pitch features | Streaming vocoder | `project_dual_head_voice.md` |
| **Future continuous-output sentence/topic encoders** | Topic embeddings, summary vectors | Curation, routing, classification | Not yet pinned |

Plus the broader literature, where this class is everywhere:

- **Sentence-BERT / E5 / GTE** (text → embedding for similarity)
- **CLIP text/image encoders** (text or image → joint embedding space)
- **DPR / contrastive retrieval encoders** (query → embedding for ANN search)
- **Q-Former in BLIP-2** (image → learned query tokens for LLM consumption)
- **Memorizing Transformers' memory writer** (hidden state → external memory)
- **Speculative decoding draft models** sometimes emit logits over hidden states
- **Various adapter / LoRA approaches** that produce activations downstream of which generation happens

This pattern is huge in the literature; the project's recognition of it as a class is what makes it usable as an architectural primitive rather than a research curiosity.

## The meta-principle

> ***The head of a model is interchangeable; the substrate is what matters. Token head produces text; representation head produces vectors for downstream use; audio head produces phonemes; steering head produces per-layer biases. Same backbone substrate; different output commitments; different consumers downstream.***

This is a *meta-architectural commitment* the project has been making implicitly. Making it explicit:

- **You don't choose between "text LLM" and "embedding model"** — you choose what head to attach to a transformer backbone
- **The substrate (the backbone) is what carries capability**; the head is what specializes it for a downstream consumer
- **Different downstream consumers want different heads**; same backbone can serve multiple by adding heads in parallel (the dual-head voice pattern)
- **Heads are training-time commitments** (per `project_training_time_representation.md`) — you can't easily retrofit a head onto a backbone trained for a different head

## Why this matters for the project specifically

Several pieces of the architecture become clearer when this class is recognized:

### Cortex remains text-generation grounded

Because Bob speaks, cortex needs a text head. That commitment is non-negotiable for the conversational use case. But cortex is *one* model in a fleet of representation-emitting models, not the only model the project needs.

### Memex's substrate is representations, not text

Memex's consumer is attention-as-classifier (not a text-generating model), so memex's encoder doesn't produce text — it produces K/V vectors that attention can score against. Per `project_memex_identity.md`'s "what memex IS NOT" item 4, memex isn't standard RAG; this is the load-bearing reason — the substrate emits representations, not text.

### The dual-head voice path

Per `project_dual_head_voice.md`, cortex grows audio heads in parallel with its text head. The shared backbone serves both. The audio heads are representation-emitting; the text head is token-emitting. Same substrate; different downstream consumers.

### The per-layer-injection auxiliary

Per `project_per_layer_injection_auxiliary.md`, a small BitNet auxiliary produces per-layer steering vectors for cortex's residual stream. The auxiliary is representation-emitting by construction; its consumer is cortex's attention computation at every layer.

### The architectural endgame

The project's v3+/v4+ endgame has **multiple heads emitting different representations from related substrates**, all coordinating to produce Bob:

- Cortex backbone (text head + audio heads)
- Memex encoder (representation head)
- Per-layer injection auxiliary (steering head + potentially K/V reconstruction head)
- Possibly: a curation encoder, a friction-detector head, a Columbo-pattern signal head

This is structurally distinct from "one big LLM does everything." It's part of why the project is buildable without frontier-scale resources — specialized heads on shared substrates are individually small and economically tractable; the synthesis is the architectural commitment.

## The specific realization for memex (the encoder + auxiliary + cortex trinity)

The architectural breakthrough Daniel walked himself to 2026-06-08: **memex's storage substrate can be much more compact than full multi-layer K/V if you use representation-emitting models for both ingestion and retrieval-time projection.**

### The trinity

```
Corpus tokens
     ↓
[ Encoder ]                      ← representation-emitting; small; bidirectional
     ↓
Compact per-token representations (much smaller than multi-layer K/V)
     ↓
[ Memex storage ]                ← TurboQuant / polar+QJL compressed substrate
     ↓
At retrieval time:
     ↓
[ Per-layer projection auxiliary ]  ← representation-emitting; produces K vectors
     ↓
Per-layer K/V vectors compatible with cortex's attention
     ↓
[ Cortex ]                       ← attends to reconstructed K/V; generates text
     ↓
Text output (Bob speaks)
```

### What each component does

- **Encoder**: small bidirectional model (could be hundreds of MB rather than GB) trained to produce compact retrieval-shaped representations from corpus tokens. The encoder's output is *one vector per token* (or a small fixed number), not L vectors. The encoder embodies the corpus-side contextual processing per the "fingerprint of understanding" framing (see `project_memex_identity.md`).

- **Memex storage**: persistent compressed cache of encoder outputs. TurboQuant or polar+QJL compression on top of the already-compact encoder output. Total storage: much less than naive multi-layer K/V (~256-512 bytes/token after compression vs ~3 KB/token for all-layer compressed K/V).

- **Per-layer projection auxiliary**: at retrieval time, takes the compact stored representations + cortex's current state → produces per-layer K/V vectors compatible with cortex's attention computation. This is the same per-layer-injection auxiliary committed to in `project_per_layer_injection_auxiliary.md`, extended with the K/V reconstruction role.

- **Cortex**: attends to the produced per-layer K/V as if it were the full multi-layer K/V cache. Generates text as before.

### The storage cost win

| Storage approach | Bytes per token | For 10M-token corpus |
|---|---|---|
| Full multi-layer K/V (FP16) | ~36 KB | ~360 GB |
| Full multi-layer K/V (TurboQuant compressed) | ~3 KB | ~30 GB |
| **Encoder representations (compressed)** | **~256-512 bytes** | **~2.5-5 GB** |

A 70-100x reduction in storage cost over even compressed full-layer K/V, with retrieval quality preserved (modulo training quality of the encoder + auxiliary).

### The quality preservation argument

Per the synergy-of-the-whole-model insight (Daniel 2026-06-08): no single layer holds the semantic content; meaning is the journey. A naive single-layer K/V loses the multi-layer synergy.

But: a *trained encoder* can capture the synergy in a compact representation. The encoder is optimized end-to-end to produce representations that, after projection via the auxiliary, behave equivalently to multi-layer K/V from cortex's perspective. The synergy is preserved; the storage is compressed.

This is why the trinity works: the encoder + auxiliary together do the multi-layer synthesis that naive single-layer storage couldn't. The two representation-emitting models collaborate to compress and reconstruct the contextual processing.

## Composition with related pins

| Pin | How representation-emitting models compose |
|---|---|
| `project_memex_identity.md` | Memex encoder is the canonical project instance; "attention over a compressed cache" implicitly requires representation-emitting ingestion |
| `project_per_layer_injection_auxiliary.md` | The auxiliary is representation-emitting; this pin generalizes its role to K/V reconstruction in addition to steering |
| `project_dual_head_voice.md` | Audio heads are representation-emitting (phoneme/duration/pitch features); same class as the auxiliary |
| `project_unified_memory_architecture.md` | The v2 bias-attention path requires per-layer K/V compatibility; the auxiliary's reconstruction role enables it without storing full multi-layer K/V |
| `project_training_time_representation.md` | Head choice is a training-time architectural commitment; you can't retrofit a representation head onto a token-trained model |
| `project_compression_substrate_quality_bar.md` | Bar 1 (retrieval ranking) is the quality bar the encoder must meet; Bar 2 (autoregressive generation against compressed substrate) is what the auxiliary's reconstructed K/V must meet |
| `project_zynqberry_bitnet_memex.md` | Small representation-emitting models map well to FPGA deployment; the encoder + auxiliary could be FPGA-resident |
| `project_eggroll_and_ut.md` | EGGROLL's gradient-free integer training + UT's parameter-tied iterative depth both apply to representation-emitting models; the substrate generalizes |
| `feedback_no_camouflage.md` | Representation-emitting models are honest about being substrate components, not human-facing personas |

## How to apply

### When designing a new component

Ask the head-vs-substrate question: *"What is this model's output type, and what consumes it?"*

- If output is text and consumer is a human: token head (cortex's text head)
- If output is audio and consumer is a vocoder: audio head (dual-head voice's sub-heads)
- If output is vectors and consumer is another model's attention: representation head (memex encoder)
- If output is per-layer biases and consumer is cortex's residual stream: steering head (per-layer injection auxiliary)

Designing the head choice consciously beats inheriting the "everyone uses token-head LLMs" default.

### When evaluating new research papers

Filter by which axis the paper improves:

- **New backbone architectures** (Mamba, RWKV, Hyena, etc.): substrate-level work; affects what heads can be attached but doesn't change the heads themselves
- **New embedding / encoder models** (Sentence-BERT successors, contrastive variants, etc.): representation-head improvements; directly applicable to memex's encoder
- **New retrieval-augmented architectures** (RETRO, Memorizing Transformers successors): representation-head + integration patterns; relevant to the trinity
- **New multimodal models with shared backbone + multiple heads** (Qwen-VL, BLIP, etc.): direct examples of the multi-head pattern this pin describes; transferable techniques

### When making composition decisions

The "shared substrate + multiple heads" pattern composes cleanly with the project's other architectural commitments:

- **Substrate ownership** (per `project_five_systems.md`) — own the backbone; the heads come along
- **Training-time representation** (per `project_training_time_representation.md`) — head choices are training-time commitments, baked into the training run
- **Hybrid serving** (per `project_hybrid_serving_pseudonymization.md`) — heads can be tier-routed (text head on local cortex; high-stakes representation heads on more powerful infrastructure)

### When evaluating "should we use a token-head LLM here?"

The default in the AI community is yes; for the project, the question should be active rather than reflexive. *"Would a representation-emitting model serve this consumer better?"* If the consumer isn't a human (or a text-output-needing pipeline), the answer is often yes.

## What this is NOT

To prevent scope creep:

- **NOT a claim that all models should be representation-emitting.** Cortex's text head is essential for Bob's voice; some heads must produce tokens. The point is to *choose* the head consciously, not default to tokens.
- **NOT a replacement for text generation.** Token-head LLMs and representation-emitting models compose; they don't compete.
- **NOT a new training method.** Representation-emitting models are trained with standard techniques (distillation, contrastive learning, joint training); the pin recognizes the architectural class, not a training innovation.
- **NOT a substitute for substrate ownership.** Using off-the-shelf embedding models (Sentence-BERT, OpenAI embeddings, etc.) is the standard-RAG-shape default; the project's commitment is to own the encoder as part of its substrate.

## The phrase to remember

> ***The head of a model is interchangeable; the substrate is what matters. Token head produces text; representation head produces vectors for downstream use; audio head produces phonemes; steering head produces per-layer biases. Same backbone substrate; different output commitments; different consumers downstream.***

Plus the specific realization for memex:

> ***Memex's substrate doesn't need to be full multi-layer K/V. A trained encoder + per-layer projection auxiliary + cortex form a trinity where: encoder produces compact representations for storage; auxiliary projects them back up to per-layer K/V at retrieval time; cortex attends as if the full multi-layer K/V were present. Storage cost drops 70-100x; quality preserved by end-to-end training of the encoder + auxiliary together.***

Plus the design discipline:

> ***Ask the head-vs-substrate question consciously. The default in the AI community is token-head LLM; for the project, the question should be active. When the consumer isn't a human, the answer is often a representation-emitting model. Choose the head; share the substrate.***

## Related pins

- `project_memex_identity.md` — memex's encoder is the canonical project instance
- `project_per_layer_injection_auxiliary.md` — the auxiliary is the K/V reconstruction half of the memex trinity
- `project_dual_head_voice.md` — audio sub-heads are representation-emitting; same class
- `project_unified_memory_architecture.md` — the v2 bias-attention path uses the auxiliary's reconstructed K/V
- `project_training_time_representation.md` — head choice is a training-time commitment
- `project_compression_substrate_quality_bar.md` — Bar 1 measures encoder quality; Bar 2 measures auxiliary reconstruction quality
- `project_zynqberry_bitnet_memex.md` — small representation-emitting models suit FPGA deployment
- `project_eggroll_and_ut.md` — training infrastructure applies to representation-emitting models
- `project_five_systems.md` — substrate ownership extends to the encoder + auxiliary
