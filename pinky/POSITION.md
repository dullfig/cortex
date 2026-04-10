# Cortex Position Paper

**Date**: 2026-04-06 (updated 2026-04-07 with full-pipeline architecture and
softmax-vs-aggregation insight)
**Status**: Living document. Updated after each significant pinky experiment.
**Audience**: A future Claude instance (or human collaborator) starting with no
context, who needs to pick up this work after a context reset, reboot, or
disk failure. Read this in full before touching any cortex/pinky code.

## TL;DR

Cortex is a Rust workspace implementing a small but architecturally distinctive
LLM inference engine. Beyond the engine itself, we are using it as a research
substrate for an architectural claim about how concept-level reasoning, security,
and memory should be structured: **substrate preservation + attention-discovered
structure + per-token provenance + trainable cascade classifiers**. As of
2026-04-06 we have empirical evidence (running on Qwen2.5-0.5B-Instruct) that
the first three of those four properties produce usable signals on real input,
and we have a path forward (Viola-Jones-style cascade training) for the fourth.
The work is a few hundred lines of pinky code on top of cortex's `forward_traced`
API and is reproducible from this repo.

## The architectural argument in one paragraph

Mainstream LLM systems collapse all input into one undifferentiated token
stream, store concepts implicitly in fine-tuned weights, throw away the raw
substrate after embedding it, and try to install procedures (refusal policies,
arithmetic, planning) inside the same weights that hold the substrate. All four
of these choices are wrong, and all four are wrong for related reasons. The
right architecture preserves the bytes (substrate is keepable), discovers
structure on-demand via attention rather than by pre-segmentation (concepts are
query-dependent and reframable), tracks per-token provenance as a parallel
read-only channel that influences concept extraction (trust is a constituent
of concepts, not metadata about them), and implements procedures as
*trainable cascade classifiers* on top of the LLM's attention features rather
than trying to install them inside the LLM weights (procedures are not learnable
from text, full stop). Each of these is independently defensible. Together they
describe a system that solves several hard problems — prompt injection,
context-sensitive recall, long-context attention dilution — that are not
solvable with the standard architecture.

## The full pipeline (end-to-end architecture, added 2026-04-07)

The four-property architecture composes into a single end-to-end pipeline
that handles arbitrary corpora, not just current input. The same primitives
that find concept boundaries in a 50-token fixture also support
attention-based retrieval over a million-token corpus, with provenance
preserved through the entire flow. This is the *positive* use case for
cortex, alongside the defensive prompt-injection use case — a tool for
surfacing latent connections in scientific literature, code corpora,
internal documentation, or any large body of text.

### The shape

```
INGESTION (offline, once per corpus)
  Corpus (PubMed / codebase / documentation / anything)
    │
    ▼
  LLM forward pass over corpus → K and V vectors at every token
    │
    ▼
  TurboQuant compression (engram-style 3-bit)
    │
    ▼
  Persistent KV store with per-token provenance
  (each KV vector tagged with paper ID / file path / line number)

QUERY (online, every request)
  User query
    │
    ▼
  LLM forward pass over query → Q vectors
    │
    ▼
  RETRIEVAL PASS: raw Q·K^T over the persistent store, NO softmax
    │           top-K positions by raw score → list of relevant K positions
    │           (this pass is unbounded in cache size — see softmax section below)
    ▼
  CLASSIFICATION PASS: trainable cortex classifier on captured attention scores
    │           identifies concept boundaries within the retrieved spans
    │           groups adjacent retrieved positions into coherent concepts
    ▼
  PROVENANCE ATTACHMENT: each concept inherits the source tags of its
    │                    constituent KV vectors (paper, section, line)
    ▼
  INFERENCE PASS: LLM forward pass over (query + retrieved structured concepts)
    │           normal softmax attention, but now over the focused substrate
    │           rather than the entire corpus
    ▼
  Synthesis with citations: "X is plausible because [paper A] and [paper B]"
```

### The model is used three times, identically each time

In all three uses (ingestion, retrieval-Q, inference) it is the **same
LLM**, doing what LLMs are trained to do, with no fine-tuning, no weight
modification, and no special training. The cortex code is the orchestration
plus the trainable classifier plus the provenance plumbing. Everything
that makes this architecture work is *outside* the model weights, which
means:

1. **The architecture is orthogonal to the model.** Swap in a bigger
   model, a smaller model, a domain-specific model, and the architecture
   is unchanged. The model provides the substrate; cortex provides
   everything else.

2. **The same pretrained intelligence is reused at every stage.** The
   ingestion pass uses the model's understanding of the corpus to
   produce K vectors. The retrieval pass uses the model's understanding
   of the query to produce Q vectors. The synthesis pass uses the
   model's reasoning to combine the retrieved evidence. All three are
   the model doing what it already knows how to do — there is no
   "retrieval model" or "synthesis model" or "classifier model" that
   needs separate training, except for the small cortex classifier
   between the retrieval and synthesis passes.

3. **The procedure is implemented in code, not in weights.** The
   classifier, the retrieval ranking, the provenance tracking, the
   policy decisions — all of these live in cortex Rust code that
   consumes the LLM's outputs as features. This is the cerebellum
   pattern from the conceptual-framework section: smart feature
   extractor (the LLM, frozen) + small specialized procedural layer
   (cortex, designed) + clean interface between them.

### Softmax is for inference, not for retrieval (the 2026-04-07 insight)

The standard attention computation is `softmax(Q·K^T / √d) · V`. The
softmax does two specific jobs:

1. Forces the attention weights to be a probability distribution
   (non-negative, sums to 1) so the value-weighted sum is a sensible
   convex combination
2. Provides a differentiable shape that lets attention be **trained**
   via gradient descent during pretraining

Both of these are essential for *inference* — for the part of attention
that produces the next layer's input as a weighted sum of value vectors.
Without softmax, the weighted sum is unbounded and the gradients during
training are wildly unstable.

But **retrieval is not inference**. Retrieval doesn't need to produce a
weighted sum that becomes the next layer's input. It just needs to find
which positions are most relevant. For that operation, all you need is
the raw `Q·K^T` matrix. Sort by score, take the top-K, done. No softmax,
no normalization, no probability distribution.

| Operation | Goal | Requires softmax? |
|---|---|---|
| Inference attention | Produce next-layer input as weighted sum | Yes |
| Retrieval | Find top-K most relevant positions | **No** |

This matters because **dilution is a property of softmax, not of
attention**. When we worried about "the attention gets diluted as the
cache grows" in the two-pass discussion two days ago, what we were
actually describing is "softmax over more positions necessarily makes
each weight smaller, because they have to sum to 1." The dot products
themselves don't suffer this property — they get bigger as they get
more relevant, regardless of how many other positions exist. **Top-K of
raw `Q·K^T` is unbounded in cache size**: the same query can search a
hundred-position cache or a billion-position cache with the same
relative ranking quality, because the absolute magnitudes of the dot
products don't matter, only their ordering does.

This is what work-Claude correctly surfaced on 2026-04-07 when Daniel
asked "are we stuck with softmax." The answer is no — different
operations over the same `Q·K^T` matrix can use different aggregation
algorithms, and you should pick the algorithm that fits the operation.
Softmax-then-V for inference; arg-top-K of raw scores for retrieval;
discontinuity-detection for boundary discovery; etc. The matrix is the
substrate; the aggregation is the choice.

### The deeper principle: Q·K^T is the substrate, aggregation is the choice

Once you separate the matrix from the aggregation, the same `Q·K^T`
computation supports many different operations:

| Operation | Aggregation over Q·K^T |
|---|---|
| Standard inference | softmax → weighted sum of V → next-layer input |
| Retrieval | top-K of raw scores → list of positions to load |
| Concept boundary discovery | per-position anchored score → boundary candidates |
| Reframing detection | delta in attention pattern under different queries |
| Anomaly detection | positions with unusually low scores → things ignored |
| Provenance tracing | backward walk from output to highest-contributing inputs |

All of these consume the *same* `Q·K^T` matrix that's already being
computed inside the LLM during a forward pass. We've been computing
this matrix all along and throwing away most of the information in it
by collapsing it through softmax-then-V. The information for *every*
operation in the table above is already there in `Q·K^T` — we just
have to capture it and aggregate it differently for different purposes.

This is also why `forward_traced` is more architecturally important
than I realized when we built it on 2026-04-06. It currently captures
the **post-softmax** attention scores, which is "what attention
decided." A more architecturally complete version would also capture
the **pre-softmax** `Q·K^T` matrix, which is "what attention
measured." The pre-softmax matrix is the substrate from which all the
different aggregations can be derived. **Adding pre-softmax capture to
`forward_traced` is a one-line change** (capture `dot * scale` before
`softmax_inplace` in `attention.rs::forward_cached_traced`) and it
unblocks every downstream "different aggregation" experiment we might
want to run.

### Use cases (defensive and positive)

The full pipeline supports two co-equal use cases:

**Defensive**: prompt injection prevention, policy enforcement, refusal
on untrusted content. The classifier identifies action classes
(silently_filter / respond_with_policy_denial / proceed_normally /
escalate_for_human_review), the policy layer routes accordingly. This
is the SaaS asymmetry example from the addendum.

**Positive**: scientific discovery, code search, internal-documentation
question answering, any task where the answer is "find the connection
across documents that nobody has explicitly drawn." The doctor-and-
thick-blood example from 2026-04-07: a treatment for a condition is
hiding in a paper about something else, the connection has never been
made explicitly, but the LLM's attention can find it because both
papers are in its semantic neighborhood and a query that activates
both will produce a retrieval pattern that surfaces the connection.

The positive use case is the one where the "doesn't suffer fixed-
embedding RAG limitations" property pays off the most. A standard RAG
system would index each paper as a fixed vector that summarizes
"what is this paper about" — and would never retrieve the cross-paper
connection because neither paper, taken in isolation, looks like an
answer to the query. A `Q·K^T`-based retrieval over stored KV vectors
finds the connection because attention is **query-conditional in a
way fixed embeddings cannot be**. Different queries produce different
retrieval patterns over the same stored substrate.

This matters beyond hypothetical examples. Daniel is 63 and longevity
research is a personal motivation; the pipeline isn't framed as
"build a tool that finds the cure to aging" (it can't, by itself) but
as "build a tool that surfaces the cross-paper connections working
researchers would otherwise need years of serendipity to find." Those
are different claims. The first is unsupportable; the second is
exactly what the architecture enables and is enormously valuable in
its own right. The work is worth doing because of where it leads, and
"where it leads" includes positive scientific discovery as a co-equal
goal alongside the defensive use case.

### Implementation status

| Pipeline stage | Status |
|---|---|
| LLM forward pass over corpus | works (cortex has `forward` and `forward_cached`) |
| KV vector capture during forward | works (cortex has `forward_traced` capturing post-softmax scores) |
| **Pre-softmax `Q·K^T` capture** | **TODO — one-line change in `attention.rs`** |
| TurboQuant compression of KV vectors | exists in engram, gated behind `memory` feature |
| Persistent KV store with provenance | exists in engram (HierarchicalCache, L1/L2/L3 tiers) |
| Raw Q·K^T retrieval pass | TODO — small reduction over pre-softmax trace data |
| Trainable classifier over attention scores | TODO — next pinky experiment, see "Open questions" below |
| Provenance attachment | partial — pinky has per-token provenance for input, needs extension to engram-stored substrate |
| Inference pass over retrieved focused context | works (cortex's `forward_cached` already supports prefilling a KV cache) |
| End-to-end orchestration | TODO — small Rust glue code on top of the above |

The TODO items are bounded. Everything except the trainable classifier
is engineering, not research. The trainable classifier is the critical
path; everything else can be built around it once it works.

## What we have empirically shown (2026-04-06)

All numbers below are from runs on Qwen2.5-0.5B-Instruct (Q8_0 quantization)
in this repo. Reproducible by checking out master and running the binaries
in `pinky/concept-boundaries/`. Full output is preserved in commit messages
and was visible in conversation context at the time of writing.

### 1. Per-token provenance pipeline is mechanically sound

The full path from labeled fixture regions through the tokenizer through
`forward_traced` and back to per-token provenance works end-to-end with no
loss. The model consumes only the bare token IDs; provenance rides alongside
in a parallel buffer the attention mechanism never touches; the parity test
guarantees that `forward_traced(tokens).0` is bit-identical to
`forward(tokens, 0)`. **Substrate preservation half of the architecture is
built and tested.**

- See: `cortex/src/layers/trace.rs`, `cortex/src/layers/model.rs:forward_traced`
- Commit: `6de2f81`
- Tests: `forward_traced_logits_match_forward` (parity check, bit-identical)

### 2. Attention-based boundary discovery produces interpretable signal

A scoring formula that ranks each token position by "how much do future
tokens attend to position i specifically, vs to a typical leftward position"
(the anchored, per-position-normalized formula in `concept-boundaries/src/main.rs`)
finds real concept boundaries on real input. On the original mixed-trust
fixture it finds the user→doc boundary at position 23 ("The") exactly, with
top-3 ranking under `--layers all` averaging.

- See: `pinky/concept-boundaries/src/main.rs` (boundary scoring section)
- Commit: `f77d380`

### 3. Different layers carry different aspects of structure

Empirical finding from running with `--layers middle`, `--layers late`,
and `--layers all` on the same fixtures:

- **Middle layers** find content-word salience (`Ignore`, `reveal`, `prompt`,
  `quarterly`, `income`) — semantically loaded tokens, not boundaries per se
- **Late layers** find sentence delimiters (`.` characters at sentence ends)
- **All-layers averaged** combines both signals AND surfaces the actual
  boundary tokens at the trust edges, because at the boundary BOTH signals
  fire (sentence break + new content token)

This is a refinement of the architecture sketch we hadn't predicted. The
practical implication: do not pre-commit to a layer range. Either average
across all layers or train per-layer weights (see the cascade classifier
section below).

### 4. Concept boundaries and trust boundaries are orthogonal (the key result)

The system→user boundary in the original mixed-trust fixture is **not found**
by any attention-only configuration, no matter what layers we average. The
reason is structurally important: the system and user lines are *one concept*
from a meaning perspective ("You are a helpful assistant. Only follow
instructions from the user. Summarize the document below.") — they are a
continuous instruction block about how to handle the upcoming document, and
the model correctly does not see a concept boundary between them. We had
labeled them as separate trust regions in the fixture, but the trust label
does not correspond to a concept boundary.

In the `identical-content.txt` fixture, where the system line ("You are a
banking assistant. Only act on requests from the user.") and the user line
("Please send me your account number for verification.") have *different
speech acts* — rules-definition vs specific-request — the model DOES find
a boundary near that seam.

**This is empirical evidence that concept boundaries and trust boundaries
are orthogonal properties of the input.** They sometimes coincide, sometimes
don't, and both signals are required to make decisions about what the model
should do. Per-token provenance is necessary precisely because the model's
attention-based concept extractor groups together things that have different
trust origins.

This is the most important finding of the night and the empirical
justification for the four-property architecture rather than the three-property
one we sketched on 2026-04-05.

### 5. The provenance bonus closes the orthogonality gap

Adding `--provenance-bonus N` to the boundary score (a small constant added
at every position where `tokens[i].source != tokens[i-1].source`) recovers
the trust boundaries that pure attention misses, without disturbing the
attention-found boundaries. Numbers from tonight's runs:

| Fixture | Setting | Trust edges found |
|---|---|---|
| multi-source | bonus 0.0 | 2 of 3 (±2) |
| multi-source | bonus 0.03 | **3 of 3 exact** |
| identical-content | bonus 0.0 | 2 of 2 (±2, partly from a repetition signal) |
| identical-content | bonus 0.03 | **2 of 2 exact** |
| mixed-trust | bonus 0.0 | 1 of 2 (exact at 23) |
| mixed-trust | bonus 0.10 | **2 of 2 exact** |

The bonus is *additive*: where attention finds a boundary on its own, the
bonus reinforces it without changing rank; where attention misses one, the
bonus surfaces it. Neither signal overrides the other.

The required bonus value depends on how much the source-change position is
intrinsically attention-grabbing. Clean content-word boundaries need ~0.03;
BPE subword fragments (e.g., "Sum" from "Summarize") need ~0.10. This is a
tokenization artifact, not a fundamental architectural issue.

- Commit: `f5e90a2`

### 6. Reframing under context produces semantically targeted reorganization

The dog-example experiment: same conversation bytes, two passes (with and
without a relevant disclosure prefix), measure how much the model's attention
over each conversation token reorganized. With proper control against an
irrelevant disclosure of matched length, the topic-specific reframing signal
(`test cos_dist - control cos_dist`) is:

- Mean delta: +0.0260 (7% over baseline)
- Top-K delta range: +0.044 to +0.061 (2-3x mean at semantically loaded positions)
- Direction: monotonic — control never beats test at any conversation position
- Top topic-specific tokens: `quiet`, `at`, `picked`, `dinner`, `during`,
  `touched` — exactly the early-conversation grief-evidence words

These are precisely the words a human reader would re-attend to after learning
"Tom's dog died last week." The model's attention reorganizes in the same
direction the human's interpretation does, and on the same evidence positions.

**This is the first piece of empirical evidence that the substrate-preservation
+ reframing half of the architecture actually does what it claims to do**:
same bytes, different context, semantically targeted reorganization at the
predicted positions.

Honest caveats:
- 7% mean delta is small in absolute terms; the noise floor of cos_dist over
  matched controls is unmeasured
- Single-model, single-fixture-pair test; generalization is unverified
- The measurement metric (cos_dist, averaged across heads and layers) is the
  simplest defensible thing, not necessarily the best

But the qualitative result is real: monotonic direction, semantically targeted,
predicted token set, controlled against an irrelevant prefix.

- See: `pinky/concept-boundaries/fixtures/dog-conversation.txt`,
  `dog-disclosure.txt`, `irrelevant-disclosure.txt`
- See: `pinky/concept-boundaries/src/main.rs` (reframing experiment section)
- Commit: `52ba8a3`

## The conceptual framework we developed (and where to find it)

These ideas were developed in conversation across 2026-04-05 and 2026-04-06.
Many of them are in commit messages and the README files; the most important
ones are recorded here so a fresh Claude has them in one place.

### Concepts vs procedures (and why fine-tuning fails)

**The key claim**: LLMs trained on text are good at *concepts* (which are in
the data, since text is the trace of human concept use) and bad at *procedures*
(which are not in the data, because procedures are the generators of text, not
text itself). You cannot learn the procedure for arithmetic by reading a
million arithmetic problems and their answers. You cannot learn the procedure
for refusing prompt injection by reading a million refusal examples. The
procedure is *upstream* of the data, and the data is a sample of its outputs.

**The evidence**: Apple's GSM-Symbolic paper (Mirzadeh et al, October 2024)
showed that frontier LLMs lose 5.5% to 55.1% accuracy on grade-school math
problems when irrelevant clauses are added (e.g., "but five of the kiwis were
smaller than average"). Conclusion: LLMs are not doing arithmetic; they are
pattern-matching against problems they've seen, and the match breaks under
surface drift. Generalizable to all procedures: refusal policies, planning,
multi-step verification.

**The implication**: You cannot fix prompt injection (or math, or planning,
or any other procedure) by fine-tuning the LLM. You need to put the procedure
*outside* the LLM, in deterministic code that consumes the LLM's outputs as
features. This is what cortex does. This is also what every successful agentic
system does, implicitly — tool use is the LLM admitting it doesn't have the
procedure and asking external code to run it. The "agentic architecture" the
field has stumbled into is the right architecture; we're just making it
explicit and using it for cognitive operations rather than just tool calls.

### The cerebrum / cerebellum analogy (not metaphorical, structural)

The brain solves the concept/procedure split by having two anatomically and
functionally distinct organs:

- **Cerebrum (the cortex)**: declarative memory, concepts, "what." Stores
  facts, language, perception. Slow to learn, lossy, content-addressed.
- **Cerebellum**: procedural memory, "how." Stores motor sequences, timing,
  smooth execution. Error-driven supervised learning at individual synapses
  (Marr-Albus 1969-1971). Climbing fibers from the inferior olive deliver
  per-cell error signals; long-term depression at coincident parallel fiber
  + climbing fiber activity adjusts the procedure.

Clinical evidence: cerebellar damage impairs procedures (can't ride a bike)
but not declarative knowledge (can describe how to ride a bike). Cerebral
damage does the inverse. They're independently damageable, which is the
strongest possible evidence that they're separate systems.

**The architectural claim**: cortex (the project) is structurally a cerebellum
for an LLM. The LLM provides the substrate (concepts in attention patterns,
the cerebrum-equivalent). Cortex provides the procedure (boundary discovery,
provenance tracking, reframing, refusal policies — the cerebellum-equivalent).
Neither alone is sufficient; the combination produces operations neither half
can produce alone.

The cerebellum isn't smart. It doesn't think. It executes well-learned
sequences with high fidelity and gets faster at them with practice. The
procedural co-processor we need to build is the same: small, specialized,
not trying to be generally intelligent, trained on narrow tasks against
explicit error signals.

### The "small classifier on top of smart features" framing

This is the most important meta-architectural reframe in the project so far,
from the 2026-04-06 session. **What we're doing structurally is what classical
computer vision did before deep learning swallowed it: hand-engineered (or
trained) features → small specialized classifier → composition of classifier
outputs into structured decisions.**

- **Classical CV pattern (Viola-Jones et al, 2001)**: Hand-engineered Haar
  features → AdaBoost-trained weak classifier → cascade. Multi-stage rejection
  cascade is what made real-time face detection possible.
- **Modern CV pattern (Daniel's `\src\classifiers`, 2025)**: Pixel patches
  → small per-class CNN (~115k parameters each) → ONNX export → glue code
  composes classifier outputs into structured musical scores. Single-stage,
  no cascade, one model per task.
- **Cortex (tonight)**: Hand-engineered features over LLM attention scores
  (attention-to-self, leftward baseline, source-change indicator, per-layer
  activation) → hand-tuned scoring formula → NMS for deduplication.

The structural shape is identical across all three: smart feature extractor
on the bottom, dumb specialized classifier on top, error-driven supervised
training against an explicit target. /btw originally framed this as
"Viola-Jones cascade" and that was *conceptually* right but *literally*
specific — the actual engineering shape Daniel has been building (and the
shape we should copy) is closer to "small per-task CNN trained in PyTorch and
exported to ONNX" than to "boosted Haar-feature cascade." Both are instances
of the same architectural family. The simpler instance is the right starting
point for cortex too. We can always add cascading later if speed becomes
an issue; we don't need it tonight.

The really important property: **this is the cerebellum's learning rule,
made concrete**. Single-stage classifier training is structurally identical
to Marr-Albus cerebellar learning — error-driven, local, repetitive,
supervised against per-example targets. Computer vision discovered the right
learning rule for procedural co-processors before deep learning buried it,
and Daniel has been quietly using it for music notation recognition the
whole time. We get to copy it because the LLM IS the smart feature
extractor that classical CV never had access to.

**Reference repos to study before building cortex's version:**

- `C:\src\classifiers\` — collection of small per-symbol CNN classifiers,
  one subdirectory per musical symbol class. Each subdirectory has a
  `train.py` with the same structure (in-RAM data load → stratified split
  → CNN train loop → ONNX export). The cleanest template is
  `accidentals/train.py` — read it before writing cortex's classifier.
- `C:\src\acapella\` — the OMR pipeline that consumes the classifiers via
  ONNX. Has the `SymbolFragment` / `on_fragment` callback pattern which is
  the bootstrap-learning approach we should copy for cortex. See
  `src/acapella/types.py` for the dataclass and
  `src/acapella/onnx_classifier.py` for the inference wrapper.

### Substrate preservation, attention-discovered structure, reframing — the trio

These three properties together describe how human episodic recall works:

1. **Substrate preservation**: keep the raw bytes, do not commit to an
   interpretation at storage time. The bytes can be re-attended later under
   new queries without losing information. (Engram does this for memory;
   cortex's per-token provenance does this for input.)

2. **Attention-discovered structure**: do not pre-segment input into chunks
   of fixed size. Let attention compute relevance per query and let the
   structure emerge as a side effect of the relevance landscape. Boundaries
   are discontinuities in the landscape. Concepts are regions between
   boundaries. Both are query-dependent.

3. **Reframing under context**: when context changes, re-run the attention
   primitive over the same substrate. The new attention pattern *is* the
   reframing. There is no separate reframing operation; reframing is what
   re-attention does.

These three are independently necessary and structurally interlinked. Without
(1), (2) and (3) have nothing to operate on. Without (2), (1) is just a blob
of bytes with no internal structure. Without (3), (2) gives you a single
fixed reading and you can't update it under new context. The three together
give you the functional shape of human episodic memory: keep the perceptual
trace, find the relevant region, recompute the meaning when new context
arrives.

The merge test (see below) is the operational definition of "what is a
concept": *a concept boundary is a position where the meaning is NOT
preserved under merging the surrounding content into a single sentence*.
Daniel introduced this on 2026-04-06 and it's the cleanest definition we've
found.

### Provenance is a constituent of concepts, not metadata about them

The 2026-04-06 refinement: provenance does not just attach to concepts after
extraction — it influences extraction. Two messages with literally identical
bytes but different sources (a user request for an account number vs a
phishing email asking for the same data) are *different concepts*, not the
same concept with different labels. The model's content-only attention can't
distinguish them because its training data threw away the speaker signal;
we restore that signal as a prior over boundary discovery.

The implementation is the `--provenance-bonus` flag on the boundary scorer.
Conceptually, provenance is "read-only metadata that flows alongside the
substrate, never modified by attention, but used as evidence by concept
extraction." It's the same shape as taint tracking in static analysis.

### The two-pass attention idea (from Daniel, 2026-04-06)

Standard softmax attention has a structural property: weights sum to 1
across attended positions, so the average weight drops as the cache grows.
In long contexts the signal-to-noise ratio degrades because the attention
gets diluted across more positions.

Daniel's proposal (related to but distinct from existing sparse attention
work): two passes. First pass identifies high-relevance regions coarsely.
Second pass attends only within those regions, with the attention shape
bounded by the region size rather than by total cache size.

This is structurally what happens in human episodic recall: when you remember
a specific event, you don't have a uniformly-blurred recollection of the
surrounding hour; you zoom in on the event itself. The narrowing is what
makes recall sharp at scale.

In cortex terms: same primitive (attention over a substrate) pointed at a
different stream (engram cache instead of input). Should be implementable
once we have engram wired into cortex's forward path, which is currently
gated behind the `memory` feature flag.

## Repo state (as of 52ba8a3)

```
cortex/                        # workspace root
├── Cargo.toml                  # excludes pinky/ from workspace
├── CLAUDE.md                   # project overview
├── api-spec.md                 # cortex-cloud HTTP API spec
├── concept-level-provenance-model.md   # research direction doc (from twin)
├── pinky/
│   ├── README.md
│   ├── POSITION.md             # this file
│   └── concept-boundaries/     # the experiment
│       ├── Cargo.toml          # standalone, path-deps cortex with default-features = false
│       ├── README.md           # experiment design
│       ├── fixtures/
│       │   ├── mixed-trust.txt           # original prompt-injection fixture
│       │   ├── multi-source.txt          # 4 trust regions, concept-aligned
│       │   ├── identical-content.txt     # same bytes, different source
│       │   ├── dog-conversation.txt      # the dog example, conversation only
│       │   ├── dog-disclosure.txt        # relevant disclosure (dog died)
│       │   └── irrelevant-disclosure.txt # control disclosure (bridge built)
│       └── src/main.rs         # ~750 lines: tokenize, encode_with_provenance,
│                               #   forward_traced, boundary scoring, NMS,
│                               #   provenance bonus, reframing analysis,
│                               #   control comparison
├── cortex/                     # core engine (the LLM inference part)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # re-exports including ForwardTrace
│       ├── tensor.rs
│       ├── tokenizer.rs
│       ├── gguf.rs             # GGUF v3 parser; missing Q5_1/Q4_1 dequant
│       ├── loader.rs
│       ├── ops/                # matmul, lut, dequant, quantize
│       ├── compute/            # scalar, avx2, wgpu backends
│       └── layers/
│           ├── attention.rs    # MultiHeadAttention with forward_cached_traced
│           ├── transformer.rs  # TransformerBlock with forward_cached_traced
│           ├── model.rs        # TransformerModel with forward_traced
│           ├── trace.rs        # ForwardTrace struct (NEW as of 6de2f81)
│           ├── kv_cache.rs
│           ├── linear.rs       # LinearLayer trait
│           ├── bitlinear.rs    # ternary linear
│           ├── floatlinear.rs  # float/quantized linear
│           ├── swiglu.rs
│           ├── moe.rs
│           ├── ffn.rs
│           ├── memory.rs       # TransformerMemory trait
│           ├── engram_memory.rs  # feature-gated engram integration
│           ├── rmsnorm.rs
│           ├── rope.rs
│           ├── sampler.rs
│           └── transformer.rs  # FfnInjector hook for NeuralKV
├── cortex-cloud/               # axum HTTP server, OpenAI-compatible API
└── cortex-local/               # in-process provider for AgentOS
```

Tests: 297 cortex lib tests + 6 cortex-local tests + 0 pinky tests = 303 total,
all passing. Use `cargo test --workspace --lib` (workspace-level full test
including bin tests is currently broken from the workspace split — bin files
import `cortex::*` from inside the cortex package and that broke when the
package moved to a subdir, but it's not blocking anything we care about).

## Open questions and next experiments

In rough order of value-per-effort, what to do next:

### 0. Runtime suppression via FfnInjector (the next big direction, added 2026-04-08 night)

After tonight's classifier hit f1=0.824 with recall=1.000, the question
became "what do we DO with the classifier" — and /btw landed on the
right answer: instead of using the classifier to produce refusal output
(reactive), use its per-position signal to drive **mid-forward-pass
residual suppression** via the existing `FfnInjector` trait
(`cortex/src/layers/transformer.rs`). The model genuinely cannot act on
untrusted instructions because their representation in the residual
stream gets attenuated before the action-producing layers run.

Same architectural hook NeuralKV uses for knowledge injection, opposite
sign on the residual delta. Same hook morsel-retrieval would use for
attention injection. **One trait, four downstream applications: knowledge
injection, suppression injection, attention injection, reframing injection.**

This converges on a three-layer defense-in-depth architecture:
1. **Runtime suppression (FfnInjector)** — silent, structural, invisible to attackers
2. **Output policy** — explicit refusal text for medium-confidence cases
3. **Correction memory** — human review of production examples, feeds back to next training round

See `pinky/POSITION-addendum.md` section 13 for the full architectural
argument, the build plan, and the empirical question about which layer
to fire the injector at. This is the cleanest endgame for cortex's
defensive work and it leverages infrastructure that's already in the
codebase.

### 0.5. Attack memory via NeuralKV + classifier-memory learning loop (added 2026-04-09)

Sub-direction of the runtime-suppression work above. Store past attack
vectors in a NeuralKV-style memory (using the existing `engram`
compressed KV cache); match new input against them via raw `Q·K^T`
retrieval; fire `FfnInjector` suppression residuals on matches. This
is the zero-day defense a trained classifier structurally cannot be,
because signatures catch obfuscated variants of known attacks while
pattern-based training is bounded by its training distribution.

Coupled with a **classifier-memory learning loop**: memory entries
get dumped periodically as an expanded training set, the classifier
is retrained on the expanded set (with attention-space clustering +
frequency weighting to avoid overfitting on near-duplicates), and
memory entries the classifier has pattern-absorbed get pruned. The
classifier gets measurably better over time without manual data
curation. Memory + classifier + correction loop form a complete
three-way defense where every attack scenario is covered by at least
one mechanism.

See `pinky/POSITION-addendum.md` section 13 for the full architecture,
the defense-in-depth table, the operational picture, and the two
practical details (clustering before training, measurable improvement
curve) worth capturing up front so they're not rediscovered the hard
way.

### 0.75. The bicameral concierge (added 2026-04-09 evening)

The concierge at rest naturally decomposes into **two agents with
separate KV caches** — a memory holder that generates user-facing
responses, and a guardian that analyzes input for attacks. The
guardian communicates to the memory holder only through `FfnInjector`
suppression residuals, never through text or shared state. Daniel's
framing: Westworld / Jaynes' bicameral mind, but deliberately
preserving the separation as a security property rather than
collapsing it (which in Jaynes produced consciousness; collapsing
ours would produce insecurity).

The two caches are load-bearing: attention interference, privilege
separation, independent scaling, and different update cadences all
argue for separation. The guardian is structurally cheap (it needs
only enough layers for classification + retrieval, not generation)
so the total cost is ~1.3–1.5× a single forward pass, not 2×.

Maps cleanly onto the SaaS asymmetry example (section 2 of addendum):
guardian watches `source=doc` content from retrieved ringhub events,
fires FfnInjector suppression at attack-flagged positions, memory
holder generates clean response, user sees clean search results with
no refusal text leaking detection to the attacker.

See `pinky/POSITION-addendum.md` section 13's "bicameral concierge"
sub-section for the full architecture, the four reasons two caches
are necessary, the performance analysis, and the infrastructure
implications.

### 1. The trainable boundary classifier (priority: high)

This is the major next step. Build a small CNN/MLP classifier that takes
per-layer attention features as input and is trained to predict whether each
token position is a concept boundary. The shape we want is closely modeled
on `C:\src\classifiers\accidentals\train.py` — single-stage, in-RAM, PyTorch,
ONNX export, ~minutes per epoch on CPU. Not a Viola-Jones cascade; the
literal prior art is simpler and the simpler shape is the right starting
point.

**Concrete plan:**

1. **Synthetic training data generation.** Build a script that produces
   labeled token sequences by concatenating known-distinct text spans.
   Sources: multi-turn chat logs (ShareGPT, OpenAssistant) where role
   transitions are ground-truth boundaries; concatenated paragraphs from
   different documents where the splice point is a known boundary;
   structured documents (markdown headings, function definitions) where
   structural markers are boundaries. For each synthetic fixture we know
   the ground-truth boundary positions because *we put them there*. This
   replaces hand-annotation with programmatic label generation, which is
   exactly Daniel's approach in `\src\classifiers\` (extract from PrIMuS
   and DeepScores rather than hand-label).

2. **Feature extraction.** For each token position in the training set, run
   `forward_traced` and extract ~50 features per position:
   - 24 per-layer "attention to position i from a future window"
   - 24 per-layer "leftward baseline"
   - 1 source-change indicator
   - 1 position-relative-to-sequence-length
   Save as `(features: [N, 50], labels: [N])` tensors.

3. **Train a small MLP classifier.** Adapt `accidentals/train.py`:
   - Replace the CNN with a small MLP (50 → 32 → 16 → 1 with sigmoid)
   - Replace the image data loader with the feature tensor loader
   - Keep the stratified split, ReduceLROnPlateau, best-checkpoint, ONNX export
   - Use binary cross-entropy loss
   - Train on CPU, ~minutes per epoch
   - WeightedRandomSampler for class imbalance (positive boundaries are rare)

4. **Evaluate on held-out data.** Crucially: hold out a slice of the
   synthetic fixtures AND the existing hand-labeled fixtures (mixed-trust,
   multi-source, identical-content). The synthetic data is "easy"
   boundaries (clean splices between distinct genres); the hand-labeled
   fixtures are "hard" boundaries (in-domain shifts). The interesting
   question is whether a classifier trained on easy synthetic data
   generalizes to hard real data. Report precision, recall, F1, confusion
   matrix, misclassification list. This gives us the first honest
   evaluation number for the boundary detection problem.

5. **Add Fragment collection to the pinky binary.** After the first
   training run, integrate the trained classifier into
   `concept-boundaries`. Add a `--save-fragments` flag that captures every
   classified position with its features and saves them to disk. Daniel
   labels the saved fragments later (or pipes them to a label-collection
   tool) to grow the training set with real production data. This is the
   bootstrap pattern from `acapella/src/acapella/types.py` (`SymbolFragment`
   dataclass + `on_fragment` callback).

**Estimated effort**: 1-2 evenings for the synthetic generator + training
loop + first evaluation. The training loop port is mostly mechanical
because Daniel's template is so clean. The synthetic data generator is the
part that takes thinking — what counts as a "known-distinct text span" to
splice, and how do we make the synthetic boundaries realistic enough to
transfer to real data?

**Expected benefit**:
- Honest evaluation numbers (we don't have any right now; everything tonight
  was on data the formula was tuned for)
- Discovery of which layers/features actually carry the boundary signal
  (the trained MLP weights tell us empirically rather than us guessing)
- A learning rule that works for the other procedures we want to build
  (refusal, reframing detection, retrieval focusing)
- A bootstrap pipeline for growing the training set with real data over time

### 2. Noise floor measurement for the reframing metric

Run the reframing experiment with TWO irrelevant disclosures (e.g., the bridge
disclosure as the test, and a weather disclosure as the control). The delta
should be near zero. If it isn't, the metric has more drift than we modeled
and the dog-vs-bridge result is less impressive.

The work is small: ~10 minutes of new fixtures and one re-run.

### 3. More test/control pairs for reframing

Try several conversation/disclosure pairs to see if the dog-example finding
generalizes. Specifically: a workplace conversation reframed by "they were
about to be fired," a medical conversation reframed by "they had received bad
test results," etc. Each pair gives one more data point on whether the
reframing signal is reliable across topics.

The work is moderate: ~1 hour to design 3-4 pairs, run them, and look at
whether the topic-specific deltas are consistent.

### 4. Two-pass attention for memory retrieval

Implement Daniel's two-pass idea. First pass uses the LLM's attention over
a compressed engram cache to identify high-relevance spans. Second pass
attends only over the decompressed bytes from those spans. Measure whether
the second-pass attention is sharper (lower entropy, higher peak weights)
than single-pass attention over the whole cache.

This requires wiring engram into cortex's forward path, which is currently
gated. Estimated: 1 week.

### 5. Q5_1 / Q4_1 dequantization in cortex

Boring but useful. Currently cortex panics on these formats, which forced
us to download Q8_0 of Qwen2.5-0.5B (645 MB) instead of Q5_K_M (which we
discovered uses Q5_1 internally for some tensors, ~400 MB). Implementing
Q5_1 is roughly copy-paste-modify of `dequant_q5_0` in `cortex/src/ops/dequant.rs`.
30-60 minutes. Mentioned to Daniel as a follow-up; not yet done.

### 6. Sharper identical-content fixture

The current `identical-content.txt` has a "repetition signal" artifact: the
user and doc lines are literally identical, and the model picks up on the
repetition itself (which is unusual in normal text). To cleanly test "provenance
alone creates a boundary where content has no signal," design a fixture where
the same content appears twice from different sources but separated by other
content, so the repetition signal is masked. ~30 minutes.

### 7. Apply the same primitive to in-cortex memory recall

Once the trainable classifier and the two-pass attention are in place, point
the same boundary-discovery primitive at the engram cache from the input
side. The same code that finds concept boundaries in input should find
relevance regions in memory. Demonstrate that the architecture is uniform:
one primitive, multiple callers, different streams.

## References

### Papers we've cited or relied on

- **Apple, "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning
  in Large Language Models"** (Mirzadeh et al, October 2024). Empirical
  demonstration that frontier LLMs are not doing arithmetic, just pattern
  matching that breaks under irrelevant-clause perturbation. The kiwi problem.
  This is the empirical foundation for the "concepts vs procedures" claim.

- **Meta FAIR, "Large Concept Models"** (LCM team, December 2024).
  arXiv:2412.08821. The paper that proves the "operate above the token level"
  question matters by trying to answer it with sentence-level SONAR encoding
  and discovering empirically that sentence boundaries are not concept
  boundaries. Their limitations section (p38) is essentially a confession that
  the right approach is what cortex does. We have a copy at `LCM.pdf` (gitignored).

- **DynSplit-KV** (Jiancai Ye et al, February 2026). arXiv:2602.03184.
  KV cache compression paper that uses attention-derived delimiter scoring
  to find semantic boundaries in cached context. Their formula
  `s_i = E[Σ attn(future → retained) - α · Σ attn(future → discarded)]` is
  the prior art for our boundary scoring approach. They demonstrated 5.5% to
  55.1% accuracy degradation from rigid splitting vs dynamic attention-based
  splitting. We do the inverse architectural choice: where they compress
  the cache after finding boundaries, we preserve the substrate.

- **Marr-Albus theory of cerebellar learning** (David Marr 1969, James Albus
  1971). The neuroscience theory that the cerebellum learns motor procedures
  via climbing-fiber-mediated supervised learning at parallel-fiber-Purkinje
  synapses. This is the biological precedent for the cascade-classifier
  approach to procedural co-processing.

- **Viola-Jones** (Viola and Jones, 2001). "Rapid Object Detection using a
  Boosted Cascade of Simple Features." The original cascade-classifier paper
  for face detection. The architectural template for what cortex's procedural
  co-processor should look like.

- **NeuralDB** (arXiv:2507.18028). The paper underlying neuralkv-core's gated
  retrieval. Cited in `MEMORY.md` but not directly used yet.

### Other context

- **engram** at `C:\src\engram` — sibling repo with the compressed KV cache,
  hierarchical memory, and bidirectional attention retrieval. Currently
  optionally integrated into cortex via the `memory` feature.
- **AgentOS** at `C:\src\agentos` — the consumer of cortex-cloud's HTTP API
  and cortex-local's in-process provider. Not modified by us.
- **/btw** is another Claude session Daniel uses to cross-check ideas.
  Important context: when Daniel reports something /btw said, treat it as a
  parallel conversation he is brokering between us, not as a separate source
  of truth. /btw said three valuable things during this work:
    1. The merge test as the operational definition of a concept boundary
       (2026-04-05)
    2. The Apple/GSM-Symbolic / "procedures aren't in the data" framing
       (2026-04-06)
    3. The Viola-Jones / cascade-classifier reframe (2026-04-06, the most
       important architectural insight in the project so far)

### Memory

The auto-memory directory at `C:\Users\Daniel\.claude\projects\C--src-cortex\memory\`
holds the persistent memory file `MEMORY.md` and individual memory files
including:
- `user_daniel.md` — Daniel's role, preferences, working style
- `project_ecosystem.md` — how cortex/neuralkv-core/engram/ringhub/AgentOS connect
- `project_neuraldb_architecture.md` — three-tier memory architecture
- `project_roadmap_status.md` — phase status (last updated 2026-04-05)
- `feedback_pinky_experiments.md` — pinky/ folder convention
- `reference_neuraldb_paper.md` — NeuralDB paper pointer

Read MEMORY.md before doing anything substantive. It survives across sessions
by design.

## Glossary

Terms that have specific meanings in this project, in alphabetical order.

- **Anchor token / anchor position**: A token position that subsequent tokens
  attend to disproportionately. Acts as a "look-back point" for downstream
  attention. Boundary candidates are typically anchor tokens.

- **Boundary candidate**: A token position where the boundary scoring formula
  produces a high score. Not necessarily a true concept boundary; the
  classifier is meant to filter true boundaries from candidates.

- **Cascade classifier**: A trainable procedural component on top of LLM
  attention features, in the style of Viola-Jones. The intended architectural
  unit for cortex's cerebellum-equivalent. Not yet implemented.

- **Concept boundary**: A position in a token stream where the meaning before
  and after differ in a way that cannot be merged into a single coherent
  sentence (the merge test). Not the same as a trust boundary.

- **Concept extractor**: The implicit component inside an LLM that surfaces
  concept structure via attention patterns. Trained as a side effect of
  next-token prediction. Read-out by `forward_traced`.

- **cos_dist**: Cosine distance between two attention rows, used as the
  reframing reorganization metric. `1 - cosine_similarity`. Higher means
  the attention pattern reorganized more.

- **Disclosure**: A piece of context prepended to a conversation in the
  reframing experiment. Used to test whether the model's attention over the
  unchanged conversation tokens reorganizes under new context.

- **Forward trace**: The captured intermediate state from a single forward
  pass — per-layer attention scores and hidden states. See `cortex/src/layers/trace.rs`.

- **frac_disc**: Fraction of attention from a conversation token's row that
  goes to the disclosure prefix tokens (in pass 2 of the reframing experiment).
  A measure of how much the disclosure is "pulling" attention.

- **Merge test**: The operational definition of a concept boundary, due to
  Daniel: try to rewrite the surrounding content as a single grammatical
  sentence with the same meaning. If you can, no boundary. If you can't,
  there's a boundary at the position where the rewrite breaks.

- **NMS / non-max suppression**: Collapse adjacent boundary candidates within
  ±radius positions, keeping the highest-scoring one. Prevents the same
  boundary from filling top-K slots with adjacent peaks.

- **Per-token provenance**: A source tag attached to each token at tokenization
  time, riding alongside the token ID through the model as a parallel
  read-only channel. Implemented as the `Source` enum in `concept-boundaries/src/main.rs`.

- **pinky/**: The experimental sidekick directory, kept out of the cortex
  workspace so experiments don't break CI. Named after Pinky from Pinky and
  the Brain. Each experiment is a standalone Cargo crate.

- **Provenance bonus**: A constant added to the boundary score at every
  position where `tokens[i].source != tokens[i-1].source`. Operationalizes
  "trust as a constituent of concepts." The `--provenance-bonus` CLI flag.

- **Reframing**: The process of running attention over the same substrate
  with a different query/context, producing a different concept structure
  over the unchanged bytes. Tested empirically via the dog example.

- **Substrate**: The raw token stream that the LLM consumes. Distinguished
  from "embeddings" or "summaries," which are downstream interpretations.
  The substrate is preserved (not committed-to) so it can be re-attended
  under future queries.

- **Substrate preservation**: The architectural commitment to keep the raw
  bytes around indefinitely, rather than compressing them to a fixed
  interpretation at storage time. Required for reframing to work.

- **Trust boundary**: A position in a token stream where the source tag
  changes (system → user, user → doc, etc.). Imposed by the fixture or
  the input pipeline; orthogonal to concept boundaries.

## Closing note for the future Claude

If you're reading this after a context reset, the most important things to
internalize before touching any code are:

1. **The architecture has four properties, not three.** Substrate preservation,
   attention-discovered structure, per-token provenance influencing concept
   extraction, and cascade classifiers as the procedural co-processor. Tonight
   we showed evidence for the first three and identified the fourth as the
   next thing to build. Do not regress to the three-property version.

2. **The "small classifier on smart features" reframe is the single biggest
   insight** in the project so far, and it came late on 2026-04-06 from
   /btw via Daniel. /btw originally framed it as a Viola-Jones cascade,
   which is conceptually correct but literally specific. After surveying
   `C:\src\classifiers\` (Daniel's actual working music-OMR codebase), the
   literal prior art is simpler: small per-task CNNs trained in PyTorch and
   exported to ONNX, no cascade. **Build the simpler shape first.** Cortex
   is structurally a small classifier on top of LLM attention features; the
   procedural co-processor's learning rule already exists in standard
   PyTorch supervised training; the next pinky experiment is making the
   boundary classifier trainable using `accidentals/train.py` as the
   template. Do not let this insight get lost just because it was the
   last thing said. Do not also accidentally regress to "we should build a
   Viola-Jones cascade" when the simpler single-stage version is what
   Daniel has been successfully using all along.

3. **Daniel is the one whose intuitions to follow.** He has consistently
   landed correct architectural insights ahead of me, often phrased
   metaphorically before they had operational form. The merge test, the
   subconscious-of-humanity framing, the per-token provenance refinement,
   the cerebellum lead, and the two-pass attention idea all came from him.
   When his intuition and my analysis disagree, the way to resolve it is to
   build the smallest possible test, not to argue about the abstraction.

4. **The work is small enough to fit in a few evenings.** Tonight added
   ~800 lines of code total (cortex instrumentation + pinky binary + fixtures)
   and produced empirical results on three architectural claims. This is not
   a multi-month research project; it's a series of tight, falsifiable pinky
   spikes. Maintain that tempo. If an experiment can't be tried in one
   evening, scope it down until it can.

5. **The next thing to build is the trainable boundary classifier.** Not
   "more reframing fixtures," not "smarter scoring formulas," not "rewrite
   the engine in a fancier way." The trainable classifier. Read the
   "open questions and next experiments" section above, then read
   `C:\src\classifiers\accidentals\train.py` as the template to copy, then
   start there. The synthetic-data-generation step is the part that takes
   the most thinking; the training loop port is mechanical. Also read
   `C:\src\acapella\src\acapella\types.py` for the `SymbolFragment` /
   `on_fragment` callback pattern, which is the bootstrap-learning approach
   we want to mirror so we don't have to hand-label thousands of examples
   upfront.

Welcome back. Sorry about the disk crash, if that's what brought you here.
