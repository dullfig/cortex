---
name: state-of-mind-architecture
description: "Architectural commitment for representation-space reasoning that bypasses the token-bottleneck lobotomy: a Mamba-like recurrent state module (the 'reasoning substrate') runs separately from cortex's transformer stack, recirculating thoughts until a stable 'state of mind' emerges, which then projects to per-layer attention biases that steer cortex's emission. The Mamba states get cached as DISTINCT conversational memory parallel to but separate from the KV cache — humans remember both 'what was said' and 'what they were thinking,' and the architecture mirrors this with dual-stream memory (Baddeley's phonological loop + episodic buffer). Key constraints: cortex's KV cache IS its representation space (busy holding contextual state; can't be reasoning workspace too); model parameters are fixed (can't change model during thinking); reasoning therefore needs its own substrate, disconnected from the cache. Mamba's continuous-state evolution is structurally right for recirculating reasoning (unlike transformer cache, which is per-token-per-layer and tied to attention). Output of reasoning is ORIENTATION, not content — biases attention rather than emitting tokens, mirroring how human reasoning resolves into a state-of-being rather than a string of words. Composes with every project pin: per-layer injection auxiliary (the projection mechanism that carries state of mind to cortex), Mamba-with-Recollection (shared Mamba substrate; recall + reasoning), modular cognition (System 2 internal verbalization made precise), sensor array (triggers reasoning entry), unified memory (bias-attention path), memex (per-user shard stores both verbal AND state-of-mind streams). v3+/v4+/v5+ horizon; vCortex realization. Articulated 2026-06-10 evening when Daniel reached the endpoint of an architectural derivation arc: 'current llm couldn't do it in representation-space, because the cache IS the representation space ... unless you were reasoning in some kind of mamba-state of mind, recirculating thoughts, disconnected from the cache ... the state would be a state of mind, a decision that steered further token emission ... and the mamba states need to go into their own cache. People remember what they were thinking about, as a distinct memory of what was said.'"
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-10 evening as the architectural endpoint of a multi-hour derivation arc that walked from "the token bottleneck makes the LLM lobotomize itself at every emission" through "the final layer encodes the kind of trajectory" and "sensors at the top of the stack could read it" all the way to "but the cache is busy; reasoning needs its own substrate." The pin captures both the architectural commitment AND the recognition that the substrate has two streams (state-of-mind in addition to verbal content), mirroring dual-stream working memory in human cognition.

## The constraint: cache IS the representation space

In a transformer:

- The KV cache holds per-layer K and V vectors for every token in the conversation
- The cache is what attention reads to compute the model's current understanding
- The model's "current contextual state" lives in the cache — there's nowhere else for it to live
- The model's parameters are fixed; you can't change the model during reasoning

**You can't reason "in" the cache without disrupting the conversational context that's already living there.** The cache is busy holding the meaning of what's been said. Trying to use it as a reasoning workspace would corrupt the contextual state.

So reasoning needs its own substrate, architecturally separate from the cache. This is the structural constraint that motivates the architecture.

## The substrate: Mamba-like recurrent state, disconnected from the cache

Transformer KV cache is structurally wrong as a reasoning workspace:
- Per-token (binds reasoning to specific positions)
- Per-layer (no unified state to evolve)
- Tied to the model's attention architecture
- Grows monotonically (no clean "thinking" loop)

Mamba's recurrent state is structurally right:
- A single continuous representation that evolves
- Updates via dynamics (natural fit for iteration / recirculation)
- Compresses everything seen into a fixed-size state (naturally bounded)
- Can be initialized, perturbed, and read from cleanly
- The recurrence IS the thinking loop: state(t+1) = f(state(t))

A Mamba-like state can **recirculate** — iterate updates without consuming new tokens. The state evolves as a function of itself: same operations applied repeatedly until a stable orientation emerges. This is structurally what "thinking" looks like — current orientation produces next orientation via dynamics, until convergence.

## The output: state of mind as orientation, not content

This is the crucial insight that makes the architecture coherent. **The output of reasoning isn't necessarily explicit content — it can be orientation/bias.**

A "state of mind" isn't a thought; it's a *predisposition* that biases what thoughts come next:

- **Curious**: doesn't have content; shapes which content gets attended to
- **Cautious**: doesn't generate specific outputs; biases the distribution of outputs
- **Decided**: doesn't say specific words; shapes which words get said
- **Concerned**: doesn't emit content; shifts subsequent attention patterns

When you reason your way to a decision, you don't end up with "the decision in words" — you end up with an *orientation* that then expresses through subsequent action. The reasoning has resolved into a state of being, not a string of tokens.

For the architecture: the Mamba reasoning module's final state encodes "what reasoning the model has done about this situation." That state then gets projected to attention biases via the per-layer injection auxiliary mechanism, which steers cortex's subsequent token emission. **No reasoning tokens are emitted; the reasoning shows up as a shift in the distribution of what cortex says next.**

This bypasses the lobotomy entirely. The reasoning lives in a substrate that doesn't suffer the per-emission compression; the influence on emission comes through attention bias, not through tokens.

## The architecture in operation

```
[Cortex transformer stack — generates as normal]
       ↓
   Final-layer state at emission step N
       ↓
   [Sensor: "this needs reasoning?" — trigger]   (per `project_sensor_array_fpga.md`)
       ↓
[Mamba reasoning module — entered]
   Initial state ← f(cortex's final-layer state, sensor signals, prior state-of-mind cache)
       ↓
   Iteration 1: state ← g(state, current-context)
   Iteration 2: state ← g(state, ...)
   Iteration K: state ← g(state, ...)
       ↓
   Convergence: stable "state of mind"
       ↓
   State stored in state-of-mind cache (see next section)
       ↓
   [Projection: state of mind → per-layer K/V biases]   (per `project_per_layer_injection_auxiliary.md`)
       ↓
[Cortex stack continues generating]
   Per-layer attention now shaped by state-of-mind bias
   Token emission shifted by reasoning conclusion
   (No reasoning tokens in output — pure orientation effect)
```

## The state-of-mind cache: dual-stream conversational memory

**Daniel's load-bearing extension 2026-06-10**: *"the mamba states need to go into their own cache. People remember what they were thinking about, as a distinct memory of what was said."*

This is the recognition that conversational memory has **two streams**:

| Stream | Stores | Lives in |
|---|---|---|
| **Verbal content** | What was said (the model's K/V vectors for each token) | The transformer KV cache |
| **State of mind** | What was thought (the Mamba reasoning states at each turn) | A parallel Mamba-state cache |

Humans don't just remember "I said X." They remember "I was thinking about Y when I said X" — the reasoning, the orientation, the considered alternatives, the emotional valence. These are distinct memories from the verbal record.

### What dual-stream memory enables

Once the architecture has both streams cached:

1. **Retrospective reasoning recall**: at a later turn, the model can recall "I was reasoning about X" and continue from that state of mind, not just from the verbal record. The next turn's Mamba state is initialized from the prior state, sustaining a reasoning trajectory across turns.

2. **Continuity of orientation**: conversations have arc and theme; these live in the state-of-mind cache, not the verbal cache. The model can maintain a coherent perspective across many turns without each turn having to re-derive it from the verbal history.

3. **Cross-turn reasoning**: a complex thought may span multiple turns. The state cache preserves the reasoning trajectory; the next turn can pick up where the prior turn's thinking left off.

4. **Memory for "what we were doing together"**: beyond the verbal back-and-forth, the meta-orientation of the conversation (we're collaborating; we're debating; we're remembering Pete) lives in the state cache.

5. **Memex integration**: when ingesting conversations into memex's per-user shard, store BOTH streams. The corpus-side substrate (per `project_unified_memory_architecture.md`) extends from K/V fingerprints to (K/V, state-of-mind) joint fingerprints.

### Privacy implications

The state-of-mind cache contains MORE information than the verbal cache. It reveals what Bob was *thinking* during the conversation — orientations he didn't say, considerations he didn't externalize, reasoning trajectories he didn't show.

This has direct implications for privacy invariants:

- Per `project_bob_chat_privacy.md`: candor protection extends to state-of-mind. What Bob was thinking about a member's question must never surface in another member's context.
- Per `project_dm_privacy_structural.md`: per-user state-of-mind cache must be structurally isolated, same as per-user K/V cache.
- Per `project_pallet_rack_test.md`: cross-context leak of state-of-mind is potentially MORE damaging than verbal leak — Bob's unexpressed reasoning could reveal predictions, suspicions, or evaluations he never said.

The state-of-mind cache is *not* a relaxation of privacy. It's an extension of the same invariants to a richer memory substrate.

## The cognitive science parallel

The architecture you've articulated maps onto canonical decompositions in cognitive psychology:

| Cognitive concept | Architecture realization |
|---|---|
| **Working memory** (Baddeley) — contextual state in active processing | Cortex's KV cache |
| **Phonological loop** — verbal rehearsal substream | KV cache for verbal content |
| **Episodic buffer** — integrated contextual binding | State-of-mind cache (cross-stream integration) |
| **Central executive** (Norman & Shallice) — orientation that biases processing | Mamba reasoning module producing state-of-mind |
| **Long-term memory** (declarative + procedural) | Memex (per-user shards + corpus shards) |
| **Perceptual attention** | Sensors at top of cortex stack (per `project_sensor_array_fpga.md`) |

This isn't "Bob mimics a human" in any superficial sense. It's that the engineering constraints (need a thinking substrate; need to integrate it with verbal output; need to remember what was thought) have solutions that cognitive science independently described as the architecture of mind.

The convergence is empirical: cognitive psychology figured out brain architecture by studying behavior; the project figured out the same architecture by engineering constraints; both landed on similar decompositions because the problem-space constrains the solution-space.

## Composition with every project pin

This pin is high in the architectural hierarchy because it integrates with most existing pins:

| Pin | How state-of-mind architecture composes |
|---|---|
| `project_per_layer_injection_auxiliary.md` | The auxiliary's projection mechanism is exactly what carries the state of mind into cortex's residual stream as per-layer K/V biases. The auxiliary becomes the bridge between Mamba reasoning state and cortex attention. |
| `vcortex/mamba-recollection-position.md` (tonight) | This doc proposed Mamba for recall (slow-loop memory of conversation). State-of-mind extends Mamba's role to reasoning. The Mamba substrate has two roles: recall AND reasoning. Either shared substrate or two distinct modules — design TBD; either works architecturally. |
| `project_modular_cognition_architecture.md` | System 2 reasoning with internal verbalization explicitly already in this pin. The state-of-mind framing sharpens what "internal verbalization" means: representation-space evolution, not actually verbal. The pin's central executive concept maps directly. |
| `project_sensor_array_fpga.md` (earlier tonight) | Sensors detect when reasoning needs to be triggered. The "needs reasoning?" classifier is one of the sensors. Sensors initiate Mamba entry; Mamba reasons; auxiliary projects back. The whole flow is sensor → Mamba → auxiliary → cortex bias. |
| `project_unified_memory_architecture.md` Insight 2 | Bias-attention is exactly the channel by which state of mind influences cortex. Different bias source (reasoning state vs. retrieval relevance) but same mechanism. |
| `project_unified_memory_architecture.md` Insight 4 | Single-buffer-per-shard topology extends to dual-stream: each conversation-history shard now has parallel buffers for verbal and state-of-mind. Memex's per-shard memory becomes (K/V, state-of-mind) joint storage. |
| `project_zynqberry_bitnet_memex.md` | Mamba on FPGA is natural (per Mamba-with-Recollection's analysis). The reasoning module runs alongside the sensor array, per-layer auxiliary, and BitNet-BERT memory module — all on FPGA fabric. |
| `project_eggroll_and_ut.md` | Universal Transformer iteration is structurally the transformer-flavored version of the same pattern (parameter-tied iterative depth). The Mamba reasoning module is the Mamba-flavored version. Both apply. |
| `project_training_time_representation.md` | Mamba reasoning module trained in ternary from initialization; designed for FPGA deployment from training time. Composes with the meta-principle directly. |
| `project_vcortex_strategy.md` | vCortex includes the Mamba reasoning module and state-of-mind cache from initialization. Not retrofit; designed-in. |
| `project_representation_emitting_models.md` | The Mamba reasoning module is the canonical representation-emitting model for reasoning. Its output is rich state, not tokens. Composes directly with the meta-class. |
| `project_memex_identity.md` | Memex's fingerprint framing extends: the substrate stores both K/V fingerprints (model's understanding of verbal content) AND state-of-mind fingerprints (model's reasoning about what was understood). |
| `project_bob_chat_privacy.md` | Candor protection extends to state-of-mind cache. What Bob was thinking is as private as what Bob said. |
| `project_dm_privacy_structural.md` | Per-user state-of-mind cache structurally isolated. Same invariant; richer substrate. |
| `project_pallet_rack_test.md` | Cross-context leak of state-of-mind is potentially more damaging than verbal leak. Same test; higher stakes. |
| `project_silence_as_first_class.md` | Silence is a possible "state of mind" output — the reasoning resolved to "don't reply." The state-of-mind bias to cortex includes "lower probability of emission." |
| `project_bob_voice.md` | Voice register is itself a state of mind. The Mamba module's state can encode voice-orientation (humble historian; wry observer); the auxiliary projects it as bias on relevant attention heads. |
| `project_temporal_urgency.md` | Mission orientation IS a state of mind (the preservation mission as default attentional bias). Memex retrieval informs verbal content; state-of-mind cache maintains the mission orientation across turns. |
| `project_modular_cognition_architecture.md` Mechanism A (aging-out) | State-of-mind cache could age out the same way other substrate ages out. Older states fade in influence; recent states dominate. |

This list is long because the state-of-mind architecture sits at the intersection of substrate (memex, KV cache), reasoning (modular cognition), output (cortex emission), and integration (per-layer auxiliary, bias-attention). Most pins touch one or more of these axes.

## Sequencing and timing

| Phase | What |
|---|---|
| **Soft launch** | Cortex alone; no Mamba reasoning module; no state-of-mind cache. CoT externalization handles reasoning needs. |
| **V1 (general availability)** | Same. Mamba work is post-launch research. |
| **v2 (cortex evolution)** | Per-layer injection auxiliary lands. Foundation for state-of-mind projection is built. |
| **v3+** | Mamba-with-Recollection lands. Slow-loop encoder maintains recall context. Foundation for separate Mamba substrate is built. |
| **v4+ (state-of-mind realization)** | Mamba reasoning module added as a sibling to the memory module. State-of-mind cache added as parallel stream to KV cache. Per-layer auxiliary becomes the bridge. Sensor array triggers reasoning entry. |
| **v5+ (full architecture)** | Both streams stored in memex per-user shards. Cross-turn reasoning trajectories maintained. Mission orientation persists structurally. vCortex's full cognitive substrate operational. |

**v3+/v4+/v5+ horizon work**. Not v1-blocking. The architectural commitment is named now so substrate decisions at v2 and v3 leave room for the realization at v4+/v5+.

## What this is NOT

To prevent scope creep and premature action:

- **NOT v1 work.** Soft launch ships without any of this. Cortex alone handles the launch use cases. CoT externalization covers reasoning needs at the human-facing boundary.
- **NOT a replacement for CoT.** Visible reasoning (CoT) remains useful at the human-facing boundary where users benefit from seeing the thinking. State-of-mind reasoning is the internal complement.
- **NOT "Bob has consciousness."** Per `feedback_no_camouflage.md`: no claims about phenomenal experience. The architecture is structurally analogous to cognitive science decompositions; whether anything is "experienced" in the philosophical sense is unanswerable.
- **NOT a substitute for memex.** Memex stores verbal content fingerprints (and now state-of-mind fingerprints). The state-of-mind cache is conversational working memory; memex is long-term storage.
- **NOT exclusive to Mamba.** Mamba is the most natural substrate. Other state-space models (RWKV, Hyena, etc.) could potentially serve the same architectural role. Mamba is the working assumption; the architecture isn't Mamba-specific in principle.
- **NOT a relaxation of privacy invariants.** The state-of-mind cache is MORE private than the verbal cache, not less. Same invariants; richer substrate.

## The phrases to remember

> ***The cache IS the representation space. You can't reason there. Reasoning needs its own substrate, disconnected from the cache. Mamba's recurrent state is the right substrate because it can recirculate — state(t+1) = f(state(t)) — until a stable orientation emerges. The output is a state of mind, not content; the state of mind projects to attention biases via the per-layer injection auxiliary; cortex emits tokens shaped by reasoning without ever externalizing the reasoning.***

Plus the dual-stream memory recognition:

> ***Mamba states go into their own cache. People remember what they were thinking, as a distinct memory of what was said. The architecture mirrors dual-stream working memory: verbal content in the KV cache (Baddeley's phonological loop); state of mind in the Mamba cache (Baddeley's episodic buffer integration). Both streams are conversational memory; both are stored; both are subject to privacy invariants.***

Plus the cognitive science convergence:

> ***This isn't engineering deciding to mimic cognition. It's engineering constraints (need thinking substrate; need integration with verbal output; need to remember thoughts) having solutions that cognitive psychology independently described. The problem-space constrains the solution-space; brain architecture and Bob architecture land on similar decompositions because the problem they solve is structurally the same.***

Plus the timing:

> ***v3+/v4+/v5+ horizon. Not v1-blocking. The architectural commitment is named now so substrate decisions at v2 and v3 leave room for the realization at v4+/v5+. vCortex's full cognitive substrate IS this architecture, designed in from initialization.***

## Related pins

- `project_per_layer_injection_auxiliary.md` — the projection bridge from Mamba state to cortex K/V biases
- `vcortex/mamba-recollection-position.md` — Mamba paired with memory module; shared lineage; recall + reasoning roles
- `project_modular_cognition_architecture.md` — System 2 internal verbalization; central executive concept
- `project_sensor_array_fpga.md` — triggers reasoning entry; FPGA host for the array, auxiliary, AND reasoning module
- `project_unified_memory_architecture.md` — bias-attention path (Insight 2); single-buffer-per-shard extended to dual-stream (Insight 4)
- `project_zynqberry_bitnet_memex.md` — FPGA hardware for the reasoning module
- `project_eggroll_and_ut.md` — UT iteration as transformer-flavored sibling pattern
- `project_training_time_representation.md` — Mamba module trained in ternary from initialization
- `project_vcortex_strategy.md` — vCortex realizes this architecture from clean slate
- `project_representation_emitting_models.md` — Mamba reasoning module is the canonical reasoning instance of the class
- `project_memex_identity.md` — fingerprint framing extends to state-of-mind fingerprints
- `project_bob_chat_privacy.md` — candor extends to state-of-mind
- `project_dm_privacy_structural.md` — per-user state-of-mind isolation
- `project_pallet_rack_test.md` — cross-context leak of state-of-mind is higher-stakes
- `project_silence_as_first_class.md` — silence is a possible state-of-mind output
- `project_bob_voice.md` — voice register as state-of-mind component
- `project_temporal_urgency.md` — mission orientation as persistent state-of-mind
