---
name: unified-memory-architecture
description: "Two-part architectural shift for cortex+memex (v2+/vCortex territory). (1) Memex stores conversation history at full fidelity; retrieval relevance replaces compressed running summaries (no decision needed about \"what's worth remembering\" — store everything, defer abstraction to query time). (2) Retrieval relevance scores become attention biases — the \"importance\" signal flows continuously from memex storage through to attention weighting, making the attention mechanism architecturally sensitive to prior relevance. Together these solve the LLM memory problem structurally rather than via prompting tricks. Composes with vCortex session-bound inference and the modular cognition architecture. Significantly sharpens vCortex's distinctive value vs vLLM (vLLM does retrieval-augmented inference via prompt injection; vCortex via attention bias)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Two insights from Daniel 2026-05-26 morning. Pinned same day (override of the usual "let it settle" discipline) because the composition is tight enough that it reads as one architectural commitment rather than two separate ideas. Composes with vCortex (session-bound KV cache + persistent context, from 2026-05-24 conversation), and with the modular cognition architecture's substrate.

## Insight 1 — Memex as unified episodic memory

**The original "running summary" problem**: how do you keep conversational context manageable as turns accumulate? Standard answer: compress old context into a summary; periodic re-summarize; lose detail at compression time.

**Daniel's reframe**: don't compress at write time. Store everything in memex. Defer abstraction to query time.

| | Running-summary approach | Memex-as-conversation-memory |
|---|---|---|
| **What's stored** | Compressed summary + recent turns | Everything, indexed for retrieval |
| **Loss point** | At compression time (forward-lossy, unrecoverable) | At retrieval time (recoverable — re-query with different terms) |
| **Decision required** | "What's worth remembering?" — taste-dependent | "What's relevant to current turn?" — computed by retrieval |
| **Failure mode** | Lost detail can't be recovered | Bad retrieval can be re-queried |
| **Calibration target** | Tune the summarizer (taste-laden) | Tune retrieval (known problem with known mechanisms) |

The taste problem dissolves. *"What's worth remembering"* becomes *"what's relevant to retrieve right now"* — a different kind of question, solvable with calibration rather than judgment.

### What this means for the memex-fleet

Per `project_memex_per_corpus.md`, the fleet is currently `bhs-corpus`, `harmonizer`, `wire-recordings`, `ringhub-living`. **Add: `conversation-history`** — possibly per-user, possibly per-tenant. Same memex substrate, different shard, different ingestion source.

The substrate that retrieves over BHS corpus is the same substrate that retrieves over conversation history. **One mechanism handles episodic memory, semantic memory, and corpus retrieval** — at different shard granularities. Memex isn't a "vector DB for documents"; it's the unified memory infrastructure for everything in the system that needs to be remembered.

### Composition with vCortex (session-bound inference)

The memory hierarchy from earlier discussions (2026-05-24 / "v200.0" musing) collapses cleanly to two tiers:

| Tier | Substrate | Role |
|---|---|---|
| **Hot (active)** | vCortex session-bound KV cache | Full-fidelity current conversation |
| **Cold (everything else)** | Memex of conversation history | Retrieval over all prior turns + corpus |

No "warm summary" middle tier needed. Cache holds active context losslessly; memex holds historical content losslessly; retrieval bridges them. **Auxiliary summary-machine LLM goes away entirely** — the architecture I sketched in the morning brainstorm becomes unnecessary because storage is no longer a compression problem.

### What this does to pin discipline

Interesting question raised by the architecture: do explicit pins still matter when memex covers everything? Answer: yes, but for different reasons than before.

| Memex of conversation handles | Pins still handle |
|---|---|
| Episodic memory within and across sessions | Cross-session structural commitments (decisions, principles, invariants) |
| "What was said when" | "These are the rules we operate under" |
| Detail recall by relevance | Taste/judgment that retrieval can't reconstruct |
| Facts | Rules |

**Pins shift role**: from "important episodic facts" to "rules + cross-session commitments + judgment that retrieval can't natively perform." Retrieval finds facts; rules and meta-judgments need to be present in context, not retrieved when relevant.

The two are complementary, not redundant. Some current pins that are essentially episodic might be candidates for migration into the conversation-history memex over time; structural pins (no-camouflage, DM privacy structural, pallet-rack test, audit timing, threshold framework) stay as pins because they're rules, not retrievable facts.

### Privacy invariants strengthen by construction

The conversation-history-memex architecture *structurally enforces* invariants that currently rely on policy + careful pipeline construction:

| Privacy invariant pin | How conversation memex enforces it |
|---|---|
| `project_dm_privacy_structural.md` | DMs live in the user's own memex shard; not in Bob's training corpus or Bob's retrieval substrate |
| `project_pallet_rack_test.md` | Per-user memex instances; cross-context leak impossible at the storage level |
| `project_bob_chat_privacy.md` | User's Bob-chat history is their memex shard; doesn't aggregate across users |
| `project_training_vs_retrieval_substrate.md` | Conversation memex is retrieval-only; doesn't feed training weights |

These move from policy-enforcement to data-shape-enforcement, which is the trust-by-construction posture the project commits to elsewhere. Architecture does the work that policy currently does.

## Insight 2 — Retrieval relevance as attention bias

**The mechanism**: store a per-position bias term alongside K and V in the cache. At attention time, the attention score for each position becomes the dot-product plus the bias:

```
attention_score = Q · K^T / sqrt(d) + bias
attention_weight = activation(attention_score)
```

Bias is set per cache item when content is injected. Memex-retrieved content gets bias proportional to its relevance score. Pinned content (rules, system prompt, anchor commitments) gets high static bias. Ephemeral routine content gets zero or slightly negative bias.

**The activation function matters.** Standard LLMs use softmax (positions compete for fixed attention budget); under softmax math, bias of +1.0 boosts attention probability by ~2.7×, +3.0 by ~20×. But **softmax is not the natural fit for this architecture** — see "Sigmoid attention" section below for why this matters and why sigmoid is the right answer.

### Why this matters

Standard RAG inserts retrieved text into the prompt and hopes the model attends to it (it sort of does, with varying reliability). **This architecture makes the attention mechanism itself sensitive to relevance.** The model doesn't just *see* retrieved content; it's *biased toward* it proportional to retrieval confidence.

The relevance signal flows continuously from storage → retrieval → attention. No discrete "pin vs not pin" decision at the model level; relevance score continuously modulates attention weight. **Pin-ness becomes a number on a spectrum, not a category.**

### Where this sits in known research

Closest cousins:

| Technique | What it does | Difference |
|---|---|---|
| **ALiBi** (Press et al. 2022) | Position-distance attention bias | Distance-based, not importance-based |
| **Attention Sinks** (Streaming-LLM, Han et al.) | Specific positions absorb attention | Model learns; not externally controlled |
| **Global tokens** (BigBird) | Designated tokens always-attended | Pre-architecture, not per-cache-item |
| **Activation steering** | Modify residual stream | Different layer, different mechanism |

The proposed mechanism — *externally specified per-cache-item importance bias, set at injection time, persists across generation* — isn't a well-trodden path in published research. May be novel; worth treating as an architectural primitive in its own right rather than a refinement of existing techniques.

### Composition with the three-phase shim API

Cortex's current shim API (`project_cortex_v1_shim_api.md`) is **inject / gate / steer** — operating on FFN activations or output logits. **Attention bias is a fourth phase: bias-attention**, operating on attention scores rather than FFN.

Same manifest pattern, same gate-then-swap flow, just a different intervention point. The plumbing is mostly already there; the new bit is the per-cache-item bias buffer and the attention-shader modification.

This is naturally a v2+ shim capability. Cortex v1 ships without it; vCortex v2 adds it as the first new shim phase beyond the original three.

## Insight 3 — Eviction as ingestion; retrieval as recall (added 2026-06-06)

Originated 2026-06-06 evening when Daniel re-derived the unified-memory architecture from a different angle (small-context-window LLM with memex overflow) and sharpened the conversation-memory cycle into an explicit two-way flow. The architecture had captured this implicitly; this insight makes the cycle and its implications load-bearing.

### The full conversation-memory cycle

```
[user turn arrives]
       ↓
[appended to LLM context window]
       ↓ (window fills)
[oldest tokens evict from context]
       ↓
[evicted tokens flow into memex] ← THE INGESTION PATH
       ↓
[memex stores at full fidelity]
       ↓ (conversation continues, topic shifts back or relevant cue arises)
[memex retrieval scores past content as relevant]
       ↓
[retrieved content gets bias-injected into attention] ← THE RECALL PATH
       ↓
[evicted content is once again available to the model's attention,
 as if it had never left]
```

**Two key clauses make this cycle work**:

1. **Context-window overflow IS the memex ingestion path for conversation.** When the LLM's context window fills, oldest tokens are evicted; the eviction path is *into memex*, not into a void. Conversation never disappears from the system — only from the active context. The earlier pin language ("memex stores everything") gestured at this but didn't name the eviction-as-ingestion mechanism explicitly. It matters because the alternative architectures (separate "long-term memory write" steps) introduce decisions and overhead that this approach doesn't need.

2. **Evicted content is offline, not destroyed.** From the conversation's experience, the content is still accessible — just via retrieval rather than via in-context. Conversational cues trigger retrieval; bias-injection makes the content available to attention again. **"Forgotten" content isn't gone — it's offline until cued.** This is the canonical recall-on-cue pattern of human episodic memory, not the gradual-fade pattern of state-space compression.

### Why this is structurally stronger than a Mamba-style continuous-state approach

Mamba and similar state-space models compress older context gradually into a continuous internal state. Properties:

- Fade is one-way; once compressed, no full-fidelity recovery
- "Forgetting" is *graceful degradation* — older content is partially-faded in the state
- No clean separation between "in working memory" and "in long-term memory"

LLM (small context window) + memex with the eviction-as-ingestion + retrieval-as-recall cycle is two-way:

- Eviction is clean (no compromised intermediate state)
- Memex stores at full fidelity
- Retrieval surfaces evicted content at full fidelity when relevance signals fire
- Content comes back as if it had never left, modulated by bias injection

**Mamba gives graceful degradation. LLM + memex gives clean separation + recall-on-demand.** Different shapes; the second is a stronger memory model for conversational use cases where topic-switching is the norm.

This is also a more cognitively grounded model — per `project_modular_cognition_architecture.md`, episodic memory in humans behaves as recall-on-cue, not as gradual-fade. The architecture mirrors how human conversation memory actually works.

### Variable-resolution retrieval as a property of recall

The "variable resolution" framing — recent past returned at high resolution, distant past at lower resolution — is a useful refinement worth designing for. Not all retrieved content needs to come back at full fidelity.

Two implementation options for variable-resolution retrieval:

1. **Tiered storage**. Recent evicted content stored raw; medium-age content distilled to key facts; distant content distilled to themes/character-arcs. Memex shards have age-dependent compression. Cleaner semantics, more architectural complexity.

2. **Recency-weighted retrieval depth**. Memex stores uniformly at full fidelity; retrieval surfaces wider context windows for recent content, narrower for older. The "resolution" is in how much surrounding content gets injected when a hit is found. Simpler implementation, same effective behavior.

Option (2) probably wins for v2+. The variable-resolution effect emerges from retrieval policy rather than requiring storage tiers. Memex's uniform-fidelity storage commitment (per `project_memex_identity.md`) is preserved.

The cognitive analog is also instructive: humans don't seem to literally store memories at different resolutions. We store rich content; what varies is what gets *retrieved* from the rich storage when cued. Specific recent details surface easily; old memories surface only their salient features unless deeply cued.

### Rationale: LLM substrate chosen over Mamba

The unified-memory architecture can be built on either substrate:

- **Mamba + memex**: state-space model with continuous fade; memex augments via retrieval; less clean separation
- **LLM (small context) + memex**: discrete window with overflow; memex absorbs evictions; recall via bias injection

The project chose the LLM path for three reasons, now worth making explicit:

1. **LLM substrate is production-mature.** Mamba and other state-space models are research-stage; LLMs ship in production everywhere. The unified-memory architecture is ambitious enough on its own; layering it on top of a research-stage substrate would compound risk.

2. **The eviction/recall cycle gives LLM + memex an advantage on the conversational memory axis**, not just equivalence. Mamba's continuous fade is a worse fit for conversation than the recall-on-cue cycle.

3. **The shim architecture and per-layer injection work the project has already pinned (`project_cortex_v1_shim_api.md`, `project_per_layer_injection_auxiliary.md`) are built for text-token-based models**, not for state-space models. Switching to Mamba would invalidate the existing architectural commitments at multiple layers.

The maturity argument validates the substrate; the eviction/recall argument validates the *shape*; the existing-architecture argument validates the *path-dependence*. Three independent reasons to commit to the LLM + memex direction over the Mamba direction, made explicit.

### Implementation surface

The new mechanisms required by this insight (beyond what Insight 1 and Insight 2 already covered):

- **Eviction → ingestion plumbing**: when context window evicts tokens, they flow into the conversation-history memex shard. Cortex and memex need a shared eviction-write path. Simple eviction policy: FIFO. Smarter policy (importance-weighted): v2+ optimization.
- **Retrieval → bias buffer for evicted content**: when memex returns relevant evicted content, bias-injection makes it available to attention again. Same mechanism as the bias-attention path from Insight 2; just sourced from conversation-history shard rather than corpus shards.
- **Variable-resolution retrieval policy**: probably Option (2) above — recency-weighted retrieval window width. Tunable.

None of these are v1-critical. They're v2/vCortex extensions to what Insight 1 and Insight 2 already pin. But naming them now keeps the v1/v2 boundary clear.

### What this insight does NOT change

- v1 launch still ships text-only chat with traditional attention; no memex-overflow-eviction cycle (per `project_road_to_launch.md`)
- The memex compressed-KV-cache substrate commitment is unchanged
- The sigmoid-attention recommendation from later in this pin still applies
- The corpus retrieval use case is unchanged; this insight just specifies the *conversation-history* use case more fully

## Insight 4 — Single-buffer-per-shard topology with quantization-coupled memex (added 2026-06-06)

Articulated 2026-06-06 evening when Daniel proposed a substrate topology that tightens the v2 vision from semantic commitments to a specific physical shape, then immediately flagged the two genuine problems the framing has to solve. The insight has three load-bearing claims and two real open problems; both belong in the architecture explicitly.

### The conceptual unification (the elegant part)

**Cortex pushes generated tokens into the end of a buffer; cortex's attention operates over the last N positions of that buffer; memex attends to the whole buffer at the same time.** Two attention patterns on one substrate rather than two substrates with a connecting policy.

Properties this gives the architecture:

- **No eviction event.** Tokens never "leave" cortex's view; they exit the *window* cortex attends to while remaining in the buffer. Insight 3's eviction-as-ingestion path collapses to a no-op — there's no transition between stores, just a sliding window.
- **"No seam" becomes architecturally enforced, not just semantically true.** The buffer IS the substrate. Cortex's working memory IS the recent slice of memex's all-time view.
- **One memory model handles everything.** Cortex's working attention and memex's retrieval attention are different access patterns on the same data, not different data with bridging logic.

### The genuine problem 1: there is no "one context" — chunks everywhere

Daniel's first flag: the unified-buffer framing is conceptually elegant but doesn't match the multi-shard reality. Per `project_memex_per_corpus.md`, memex is a fleet (`bhs-corpus`, `harmonizer`, `wire-recordings`, `ringhub-living`, plus eventual per-user `conversation-history` shards). Each shard has its own topology, update pattern, ingestion source, and access scope:

| Shard | Update pattern | Scope | Topology |
|---|---|---|---|
| `bhs-corpus` | Rare ingestion (curated documents) | Shared across users | Set-at-ingestion; mostly static |
| `harmonizer` | Slow growth (new issues added) | Shared across users | Set-at-ingestion; grows slowly |
| `wire-recordings` | Rare ingestion (transcripts added when produced) | Shared across users | Set-at-ingestion; mostly static |
| `ringhub-living` | Continuous (every member post) | Shared across users | Append-only; high update rate |
| `conversation-history` | Continuous (every turn) | Per-user | Append-only; per-user buffer |

There is no "one context" that all these collapse into. There are buffers everywhere — per-shard, per-user where applicable, with different update semantics.

### What the unified-buffer framing actually means under multi-shard reality

The architecture is **single-buffer-per-shard**, not single-buffer-overall. The unification happens at memex's *retrieval-classifier output*, not at the *storage layer*:

- Each shard is its own buffer (its own KV cache substrate)
- Cortex's "working memory" is one specific buffer's recent slice — typically the per-user `conversation-history` shard plus the active session's ephemeral state
- Memex attends across ALL shards in parallel — each shard gets its own attention computation; outputs combine via curation (per `project_memex_curation.md`)
- The "no seam" property holds within the conversation-history shard (cortex's window is a slice of that shard's buffer); for corpus shards, there's a meaningful seam (corpus content is set at ingestion, not produced by cortex)

So Insight 4 sharpens the v2 vision to: **per-shard substrate topology where each shard is a buffer; cortex's working memory is the recent slice of the conversation-history buffer; memex attends across all shards in parallel; the unification is at the retrieval output, not the storage layer.**

The "chunks everywhere" reality is genuine. The elegant single-buffer mental model is useful as a *conceptual frame*; the implementation reality is multi-buffer with parallel attention. Don't pretend the elegance survives implementation unchanged — that's exactly the kind of architectural drift the project's pin discipline exists to prevent.

### The genuine problem 2: memex must read Bob's quantization natively

Daniel's second flag: **memex has to learn to read Bob's quantization.** This is profound and load-bearing.

If cortex (Bob) writes K and V vectors into the buffer in some quantization regime (currently FP16; eventually TurboQuant / polar+QJL / BitNet ternary / sigmoid-attention-trained / whatever the project's substrate evolves to), and memex reads those vectors to do attention-as-classifier, then memex must either:

1. **Operate natively in Bob's quantization** (no seam — memex's attention computation runs on whatever representation Bob produces)
2. **Convert at the read boundary** (introduces a seam — Bob writes in one representation, memex reads after conversion)

Option 2 violates `project_training_time_representation.md`'s meta-principle: *don't compress at the seams; build architecture around the representation.* The "no seam" claim of the unified-memory architecture *only holds* if memex operates in Bob's representation natively.

### What this requires of memex going forward

**Memex's quantization-handling capability must follow cortex's evolution.** They're coupled at the substrate. Concretely:

- **Today** (FP16 cortex): memex retrieval works on FP16 K/V. The polar+QJL Bar 1 result from today demonstrates this — memex CAN operate in polar+QJL because that's exactly the substrate that passes Bar 1 for retrieval ranking.
- **v2** (polar+QJL substrate): memex's attention-as-classifier runs on polar+QJL K/V directly. This is what today's experiment validated. Bob writes polar+QJL; memex reads polar+QJL; no conversion at the seam.
- **v3+** (BitNet ternary weights + TurboQuant K/V + sigmoid attention substrate): memex's operations have to keep up. Sigmoid-attention-trained model means memex's attention semantics matches; ternary weights mean memex's K/V representations match; the project's substrate evolution is constrained by the requirement that memex follow.

### Composition with `project_training_time_representation.md`

This insight is a direct application of the training-time-representation meta-principle. The principle says: commit to representation choices at training/design time; don't compress at the seams. Applied here: **the seam between cortex's write and memex's read must not introduce a representation change.** Both must operate in the same representation, committed at the project's substrate-architecture layer.

This means: when cortex's substrate representation evolves, memex's implementation must evolve in lockstep. **The two systems are jointly-trained at the substrate layer**, not independently trainable components glued together at the API.

The implication for cross-Claude coordination: cortex-claude and memex-claude must agree on representation choices. Substrate-representation decisions are joint decisions, not single-Claude decisions. This composes with the cross-system coordination concerns named in this pin's own "Cross-Claude coordination implications" section.

### What this DOES change vs the existing pin content

- **Insight 1** (memex stores everything): reframes to "each shard is a buffer; memex attends across shards in parallel"
- **Insight 2** (retrieval relevance as attention bias): preserved; the bias mechanism operates on the per-shard K/V; the multi-shard composition is at the bias-aggregation layer
- **Insight 3** (eviction-as-ingestion + retrieval-as-recall): the eviction event collapses to no-op within the conversation-history shard; recall mechanism is unchanged

### What this raises as open design problems

1. **Compression strategy across a buffer's length** — recent positions at full fidelity (cortex's working slice), older positions at TurboQuant compression. Graduated quality vs uniform compression with two-tier storage; both functionally similar but architecturally different. Decision deferred but worth holding.

2. **Per-shard attention composition** — memex attends to N shards in parallel; how do per-shard outputs combine into a unified retrieval result? Probably the curation mechanisms from `project_memex_curation.md`, weighted per-shard. Open implementation question.

3. **Generation-against-compressed substrate (Bar 2)** — if cortex's bias-attention path requires generation against memex's compressed K/V, Bar 2 quality applies. The Bar 2 failure of polar at Qwen-3B/36-layer suggests either: (a) a different compression scheme, (b) a different model architecture, (c) the "promotion to full-fidelity recent slice" integration option (novel — relevant old content gets fetched into cortex's recent window at full fidelity, sidestepping Bar 2 entirely).

4. **Conversation-history shard ingestion rate** — append-only, every turn; how is this stored efficiently as it grows unbounded? Per-user buffers may need their own compression policy.

5. **Cross-shard attention cost** — memex attends across all shards on every cortex generation step (in the strict reading of the unified framing). That's expensive. Alternatives: cache memex retrievals across N steps; trigger memex retrieval only on topic-shift; defer to the curation layer to bound work per step.

### What this is NOT

- **NOT a literal single buffer.** The single-buffer mental model is useful as a frame but the implementation is multi-buffer.
- **NOT a claim that cortex and memex are the same system.** They remain separate components; what they share is the substrate-representation commitment.
- **NOT v1 work.** This sharpens the v2 vision; v1 ships with the existing implementation.
- **NOT a justification for retrofit conversion.** The training-time-representation principle is in force; conversion at the seam is what we're rejecting, not what we're permitting.

### The phrase to remember (for this insight)

> ***Two attention patterns on one substrate per shard. Cortex's working memory is the recent slice of the conversation-history buffer; memex attends across all shards in parallel; the unification is at the retrieval output, not the storage layer. Memex must read Bob's quantization natively — the seam between Bob's writes and memex's reads must not introduce a representation change. The two systems are jointly-trained at the substrate layer.***

Plus the constraint:

> ***There is no "one context" — chunks everywhere. The elegant single-buffer mental model is useful for reasoning; the implementation reality is per-shard buffers with parallel memex attention. Don't pretend the elegance survives implementation unchanged.***

Plus the coupling claim:

> ***Memex's quantization-handling capability must follow cortex's evolution. They're coupled at the substrate. Substrate-representation decisions are joint cortex+memex decisions, not single-Claude decisions.***

## Sigmoid attention — the natural attention regime

Daniel's follow-on observation 2026-05-26: memex already uses **sigmoid attention** (per `project_memex_architecture.md`'s "sigmoid-attention librarian" framing). Memex computes raw scores precisely because it's sigmoid-based. If cortex's bias-attention mechanism also uses sigmoid (or composes via sigmoid), the architecture becomes uniform end-to-end.

### Softmax vs sigmoid attention

| | Softmax attention | Sigmoid attention |
|---|---|---|
| **Math** | `attention_weight = softmax(scores)` | `attention_weight = sigmoid(scores)` |
| **Constraint** | Sum to 1 across positions | Each position independent (0-1) |
| **Semantics** | "Distribute fixed attention budget across positions" | "Include each position with weight X, independently" |
| **Production use** | Universal in modern LLMs | Existed in literature, fell out of fashion, revived recently in a few papers; not in mainstream production stacks |
| **Behavior under variable retrieval-set size** | 5 items vs 50 items reshapes the entire distribution (each gets less attention as set grows) | 5 items vs 50 items doesn't matter — each gets its own inclusion weight |
| **Behavior under bias** | Bias creates relative shifts; competing with all other tokens | Bias directly shifts that item's inclusion weight; no competition |
| **Compatibility with raw memex scores** | Need normalization step before injection | Direct compatibility — raw memex scores ARE sigmoid-shape signals |

### Why sigmoid is the natural fit

Three reasons composability favors sigmoid:

1. **Memex computes raw scores, not softmaxed logits.** Those raw scores are sigmoid-attention scores directly. Under sigmoid attention in cortex, those scores flow straight through without any translation. Under softmax, they need normalization (and the renormalization changes the meaning).

2. **Retrieval is "include each item if relevant," not "distribute attention across items."** When memex returns 5 items vs 50, the system shouldn't behave differently in attention budget per item — each item is independently worth attending to or not. Sigmoid's per-position independence matches this semantics; softmax's fixed-budget competition fights it.

3. **Importance bias composes naturally.** Under sigmoid, adding bias to score directly increases that position's inclusion weight. No zero-sum trade-off with other positions. Pin bias of +3.0 makes that item's attention weight saturate toward 1.0; doesn't make others lose attention.

### Implications for cortex's attention regime

This raises a real architectural question for vCortex v2/v3: **does cortex itself adopt sigmoid attention, or stay softmax with a sigmoid-flavored bias mechanism?**

Three options:

1. **Stay softmax in cortex; add bias term as proposed.** Simplest. Math works (softmax + bias is well-defined). But composing with memex's sigmoid scores requires translation; mismatch between memex retrieval semantics and cortex attention semantics.

2. **Sigmoid attention everywhere.** Architecturally uniform; matches memex; bias mechanism becomes a natural extension. Substantial change from mainstream practice — most pretrained models are softmax-trained, so cortex would need either a softmax-trained model that's robust to sigmoid swap (probably degrades) or a sigmoid-trained model (specialized fine-tune).

3. **Hybrid: softmax for standard self-attention, sigmoid specifically for retrieved-memex content.** Some attention heads (or some layers) use sigmoid for retrieved-content attention; rest use softmax. Compositionally interesting; adds complexity; requires architectural decision about which heads/layers.

### Recommended sequencing

| Phase | Approach |
|---|---|
| **v1 / soft launch** | Softmax (status quo); no bias mechanism yet |
| **v2 / first bias mechanism** | Option (1) — softmax + bias; gets the mechanism working without requiring model retraining |
| **v3+ with corpus fine-tune** | Option (2) or (3) — train (or fine-tune) Bob on sigmoid attention if it composes better with memex than the softmax+bias hybrid does. Corpus fine-tune is happening anyway per `feedback_no_camouflage.md`-adjacent work; attention-regime change is a natural addition to the same training run |

The corpus fine-tune is the leverage point. Cortex is already going to be fine-tuned (continual pretraining on barbershop + instruction tuning for Bob's voice). Adding "sigmoid attention" or "importance-bias awareness" as additional training objectives in the same run is cheap relative to running it standalone.

### Philosophical alignment

Sigmoid attention is "include this if relevant" semantics; softmax is "distribute fixed budget across positions" semantics. For retrieval-augmented generation, **include-if-relevant matches the actual task better.** Retrieved items are independently worth attending to or not; they're not competing for budget; each item's inclusion is a yes/no with weight, not a slice of a finite pie.

This is also philosophically aligned with the project's broader patterns:

- **Per-user memex shards** — each user's history is independent; no zero-sum across users
- **Per-corpus memex fleet** — each corpus is independent; no zero-sum across corpora
- **Pin discipline** — pins are individually load-bearing; no zero-sum across pins
- **No-camouflage principle** — each interaction is honestly itself; no competition for "looking-human-ness budget"

Independence over competition. Sigmoid over softmax.

## How the two insights compose

The unified picture:

```
Conversation turn arrives
       ↓
Memex retrieval (over conversation history + corpus)
       ↓
Retrieved content + relevance scores
       ↓
Injected into KV cache with bias = f(relevance_score)
       ↓
Attention weighs by Q·K + bias
       ↓
Generation continues with bias-modulated attention
       ↓
New turn (input + response) added to memex
```

The system flows continuously from storage → retrieval → attention. Same mechanism handles short-term context (cache) and long-term memory (memex via retrieval into cache with bias). **One memory architecture for the whole cognitive system.**

### What this enables

- **Lossless conversational memory** at any scale — memex stores all turns
- **Continuous importance signaling** — relevance score → attention bias, no discrete pin/no-pin decision at the model layer
- **Structural privacy enforcement** — per-user memex shards make cross-user leak impossible
- **Unified retrieval substrate** — corpus and conversation share the same library
- **vCortex differentiation** — vLLM does retrieval-augmented inference via prompt injection; vCortex does it via attention bias (architecturally different; potentially better tail behavior)

## Implementation thoughts

### For memex (conversation-history shard)

- New shard type alongside existing corpus shards
- Per-user or per-tenant granularity (for privacy isolation)
- Ingestion source: chat turns rather than corpus documents
- Same retrieval API; relevance scoring already exists per `project_memex_architecture.md`
- Probably needs scaling considerations (millions of turns vs tens of thousands of corpus documents)

### For cortex (attention-bias mechanism)

- KV cache extends to (K, V, bias) per position
- Bias buffer: one float per position, additive to attention scores
- Attention shader modification: trivial (one add before softmax)
- Bias source: set when content is injected; from memex retrieval scores or static for pinned content

### For the shim API

- New phase: **bias-attention** — alongside inject/gate/steer
- Manifest declares what positions to bias and by how much
- Same gate-then-swap discipline
- Composes with other shim phases (FFN-side shims operate independently)

### Fine-tuning implications

For the model to genuinely *use* the bias signal (not just tolerate it), continual pretraining on (text, importance) pairs would help. This composes with the corpus fine-tuning already planned per dragnet and Bob's voice work:

- Phase 1: continual pretraining on barbershop corpus (makes BHS content in-distribution per yesterday's discussion)
- Phase 2: instruction tuning for Bob's voice
- **Phase 3: importance-bias-awareness training** — teach the model that biased items carry more weight as a learned signal, not just a math distortion

Phase 3 would happen post-v1; not soft-launch-critical.

## Where this sits in the roadmap

| Phase | Status |
|---|---|
| **v1 / soft launch** | NOT NEEDED. Current attention + standard RAG suffice for Bob-as-concierge over static corpus. |
| **v2 / vCortex** | This is a v2 feature. Session-bound KV cache + memex retrieval + attention bias composes into the vCortex distinctive architecture. |
| **v3+ / modular cognition** | Importance-bias-awareness training joins the System-1 multi-shim composition story per `project_modular_cognition_architecture.md`. |

Per `project_road_to_launch.md`: this is post-V1 work. Don't let it gravitate into v1 scope. Architecturally important; not launch-critical.

## Open questions worth chewing on (later)

1. **Bias magnitude calibration.** What range works empirically? Probably 0.5-3.0; needs testing.
2. **Off-distribution behavior.** Model wasn't trained with bias. Mostly works at small magnitudes; cliffs likely at large ones.
3. **Per-token vs per-segment bias.** Inject a chunk of retrieved text — does every token get the same bias, or does relevance decay across the chunk? Probably segment-level is fine.
4. **Bias decay over generation.** Does the bias on retrieved content persist for the whole generation, or fade as generation progresses? Different behaviors are reasonable.
5. **Interaction with position-dependent attention** (RoPE, ALiBi). Composes mechanically; behavior worth verifying.
6. **Memex calibration for retrieval relevance scores.** If relevance scores feed bias directly, the calibration of those scores becomes load-bearing for attention behavior. Strengthens the argument for memex task #4 (score calibration).
7. **What about NEGATIVE bias?** Could be useful to actively de-weight known-uninteresting content. Composes with dragnet's "this is adversarial" verdict — flagged content could get strong negative bias rather than (or in addition to) being rejected outright.

## Cross-Claude coordination implications

- **memex-claude** — new shard type (conversation-history); calibration becomes more load-bearing because retrieval scores feed attention bias
- **cortex-claude** — new shim phase (bias-attention); KV cache extension; attention shader modification
- **agentos-claude** — multi-tenant primitives must support per-user memex shard isolation
- **integration-claude** — pin discipline updates (rules vs episodic distinction)
- **dragnet-claude** — possible integration: adversarial verdict → negative bias on content

When cross-repo design conversations begin on vCortex implementation, all five repos touch this pin.

## Related pins

- `project_memex_architecture.md` — base memex; this pin extends to conversation-history shard
- `project_memex_per_corpus.md` — adds conversation-history to the fleet (per-user or per-tenant)
- `project_memex_as_platform.md` — substrate generalizes; this is one more dimension of that
- `project_memex_identity.md` — what memex IS; this pin sharpens the identity to "unified memory infrastructure"
- `project_cortex_v1_shim_api.md` — extended with attention-bias phase as v2 work
- `project_cortex_plugin_architecture.md` — the bias mechanism is a natural plugin
- `project_modular_cognition_architecture.md` — composes with System 1 multi-shim composition
- `project_dm_privacy_structural.md`, `project_pallet_rack_test.md`, `project_bob_chat_privacy.md`, `project_training_vs_retrieval_substrate.md` — privacy invariants strengthen under this architecture
- `project_road_to_launch.md` — v2+ work; do NOT promote into v1 scope
- `project_meta_semantic_memory.md` — v3+ patterns-over-patterns layer composes on top of this

## The phrase to remember

> *Memex stores everything; retrieval decides relevance; relevance becomes attention bias. The "what should I remember" problem dissolves because storage is lossless and the importance signal flows continuously through to attention. Pin discipline shifts from "remember this fact" to "enforce this rule."*

Plus the architectural identity claim:

> *Cortex + memex isn't "LLM + retrieval bolt-on." It's an LLM with integrated memory hierarchy where corpus and conversation share one retrieval substrate and one attention mechanism. Same library handles episodic and semantic memory. The architecture has no seam.*

Plus the vCortex differentiation:

> *vLLM does retrieval-augmented inference via prompt injection. vCortex does it via attention bias. The model doesn't just see retrieved content; it's biased toward it proportional to retrieval confidence. Architecturally different; potentially better tail behavior.*

Plus the attention-regime claim:

> *Sigmoid attention is the natural attention regime for retrieval-augmented inference. Memex already uses it. Cortex composing on sigmoid (or sigmoid-flavored bias) gives uniform semantics: "include this if relevant" beats "distribute attention budget" for this task. The attention nobody uses now is the attention this architecture wants.*
