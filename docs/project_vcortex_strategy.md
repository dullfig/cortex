---
name: vcortex-strategy
description: "Strategic commitment to declare cortex 'finished' at a quality threshold post-launch, freeze it in production, and start `/src/vcortex` as a clean architectural rewrite that inherits all the accumulated learnings the project's pin discipline has preserved. The Brooks 'plan to throw one away' move applied deliberately rather than reactively. vCortex inherits commitments cortex can't fully reach incrementally: memory planning as first-class, trinity (encoder + auxiliary + cortex) from initialization, batched serving native (PagedAttention from start, not bolted on), sigmoid attention native, TurboQuant/polar+QJL native operations, shim architecture cleanly integrated, BitNet weights from training-time, vram-heap as native substrate (not retrofit). Parallel-development pattern: cortex stays in production handling V1 traffic; vCortex grows on the side without launch pressure; eventually replaces cortex via parallel-run verification. Naming: `/src/vcortex` as sibling repo, not branch — clean codebase, no retrofit attempts. Composes with `feedback_doing_then_learning.md` at the project-architecture scale: cortex was the doing; vCortex is the learning made codified. Articulated 2026-06-09 evening after the tsunami serving architecture pin landed; Daniel: 'once the dust settled from the real launch, and all the little problems have been ironed out, i think you call cortex a finished product, start /src/vcortex on the side.'"
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-09 evening. After the tsunami serving architecture pin landed (capturing the vLLM/cortex/frontier tiering for V1+ scale), Daniel named the longer-horizon move: declare cortex finished post-launch; start `/src/vcortex` fresh. This is Brooks' "plan to throw one away" executed strategically rather than reactively. The pin captures the strategic commitment so future Claude sessions inherit the architectural intent, not just the architectural debt.

## The Brooks invocation

Fred Brooks, *The Mythical Man-Month* (1975):

> ***"Plan to throw one away; you will, whether or not you plan to."***

The first version of any system is wrong. You learn what's actually needed by building it wrong first; the second version is shaped by the discoveries the first version generated. This has been true for fifty years across every software project that ever shipped.

The project's pin discipline has been accumulating those discoveries. The architectural endgame is *articulable now* because cortex was built. Without the doing, the learning wouldn't exist. The pins are the codified second-version design that emerged from the first version's empirical work.

vCortex is the act of saying "the second version is now buildable; let's build it."

## Trigger conditions

vCortex starts when these are all true:

1. **Cortex has shipped V1 successfully** (general availability working; BHS partnership formalized; per `project_road_to_launch.md`'s V1 gates)
2. **The post-launch dust has settled** — initial bugs found and fixed; user feedback absorbed; capacity is stable; operations are routine
3. **Cortex's quality has reached "finished" threshold** — the bar isn't "perfect" but "good enough that further refinement has diminishing returns and the limiting factor is architectural, not implementation"
4. **The tsunami serving architecture is in place** (per `project_tsunami_serving_architecture.md`) — vLLM tier carrying the high-volume load means cortex's throughput isn't the bottleneck; vCortex development doesn't gate launch capacity

What "cortex is finished" looks like operationally:

- Bug rate has stabilized below some threshold (no major architectural-level surprises remaining)
- Operational issues are well-understood and have runbooks
- Performance is at or above `project_cortex_v1_perf_threshold.md`'s targets
- The retrofit work (vram-heap migration; memory planning discipline; etc.) has landed
- New features for cortex aren't on the roadmap — anything substantial goes to vCortex

The moment is recognized rather than scheduled: *"there's nothing more we should add to cortex; the limiting factor for what we want is architectural."*

## What vCortex inherits from cortex's learning

Every pin describing architectural endgames cortex can't fully reach is a commitment vCortex inherits **from initialization**:

| Pin | What vCortex inherits |
|---|---|
| `project_unified_memory_architecture.md` Insight 4 | Single-buffer-per-shard substrate topology; memex reads cortex's quantization natively; substrate-representation joint with memex |
| `project_per_layer_injection_auxiliary.md` | Auxiliary as first-class component, not retrofit; residual-stream distillation from training time |
| `project_training_time_representation.md` | Commit at training time, not as post-hoc optimization; BitNet ternary from initialization (not FP16-quantized later) |
| `project_representation_emitting_models.md` | Trinity (encoder + auxiliary + cortex) from initialization; head choice as architectural commitment |
| `project_compression_substrate_quality_bar.md` | Compression representation passes both Bar 1 (retrieval ranking) and Bar 2 (autoregressive generation) by design; not "ship and iterate" |
| `project_tsunami_serving_architecture.md` | Batched serving (PagedAttention; continuous batching) as native architecture, not bolted-on layer; vLLM-compatible at the protocol level |
| `project_dual_head_voice.md` | Multi-head architecture from start; text + audio heads parallel; not text-only with audio bolted on later |
| `project_eggroll_and_ut.md` | Gradient-free integer training methodology; UT iteration as native architecture if research lands |
| `project_zynqberry_bitnet_memex.md` | FPGA-deployable by design; memory layout compatible with FPGA constraints |
| `project_cortex_v1_shim_api.md` + `project_cortex_ffn_shims.md` | Shim architecture cleanly integrated, not retrofit; three-phase (inject/gate/steer) plus bias-attention plus per-layer injection as native |
| `project_silence_as_first_class.md` | Silence-as-output capability native, not gated through retrofitted shim |
| (Future) memory-planning pin from morning's discussion | Memory plan computed at startup; compute graph doesn't allocate; allocation logic centralized |

None of these can land cleanly in cortex incrementally. All can land in vCortex from a clean slate.

## What vCortex does differently from cortex

The architectural deltas:

### Memory as first-class architecture

Per this morning's "memory allocation is happening everywhere" discussion: cortex's memory management is retroactive. vCortex computes a memory plan at startup, allocates everything from one centralized arena, compute logic operates on slices (never allocates). vram-heap is the native substrate, not a retrofit target.

### Trinity from initialization

The encoder + per-layer-projection auxiliary + cortex trinity is the native architecture, not a v2/v3 add-on. Training infrastructure expects all three components. Shared representation space designed in.

### Batched serving native

PagedAttention-style memory layout from the start. Continuous batching as the default scheduler. Per-request serving is the *special case*, not the default. vLLM-compatible at the protocol level so tiering composes cleanly.

### BitNet ternary weights from training-time

Per `project_training_time_representation.md`: the model is trained in ternary representation, not FP16-quantized afterward. EGGROLL-style gradient-free integer training methodology applies. FPGA deployment is the natural target.

### Sigmoid attention as the default

Per `project_unified_memory_architecture.md`'s sigmoid attention section: vCortex uses sigmoid attention throughout, not softmax with sigmoid-flavored bias retrofitted. The attention computation is uniform with memex's sigmoid-attention librarian.

### Shim architecture clean

Three-phase shims (inject/gate/steer) plus bias-attention plus per-layer injection auxiliary all as native architecture. Shim API is designed in, not extended post-hoc.

### TurboQuant/polar+QJL as native operations

Operations that work directly on compressed representations. No decode-compute-encode cycles at the substrate boundaries. Per `project_training_time_representation.md`: representation choice at training time; compute kernels match the representation.

### Memex as substrate, not bolt-on

vCortex's attention reads memex's compressed KV directly (bias-attention path from Insight 2 of unified memory architecture). The seam between conversation and corpus is architecturally enforced, not policy-stated. Cortex-memex coupling at the substrate-representation layer is joint by design.

## The parallel-development pattern

The pattern is **continue running cortex in production while vCortex grows on the side**:

| Phase | Cortex | vCortex |
|---|---|---|
| **V1 stable** | Production; serving substrate-attended queries; routine ops | Doesn't exist yet |
| **vCortex initiated** | Production; no new substantive features | `/src/vcortex` repo created; clean codebase begins; architectural pins consulted |
| **vCortex early development** | Production; bug fixes only | Substrate work; trinity training infrastructure; memory planning; batched serving from scratch |
| **vCortex maturing** | Production; some operational pressure to migrate (substrate evolution wanted) | Feature-parity for soft launch use cases; testing harness; corpus re-ingestion if encoder evolved |
| **vCortex production-tested** | Production; comparison runs against vCortex | Side-by-side serving; quality comparison; gradual traffic shift |
| **Migration window** | Reducing traffic share | Increasing traffic share; vCortex absorbs more queries as confidence grows |
| **Post-migration** | Eventually deprecated and removed | Production; cortex's role assumed |

**Key property**: vCortex doesn't have launch pressure. It can be built calmly, with all the architectural commitments designed in, without the "we have to ship by date X" forcing function that produced cortex's organic-growth pattern.

This is structurally Brooks' point: the second version, designed with the first's discoveries in hand and without the first's time pressure, lands cleaner.

## Naming convention

**`/src/vcortex` as sibling repo, not branch of cortex.** This matters:

- A branch of cortex inherits cortex's code structure; vCortex is meant to be architecturally different
- A sibling repo signals "this is its own thing" not "cortex's improvement"
- The clean codebase invites architectural design rather than retrofit
- The shared parent (`/src/`) means both repos are first-class within the project's substrate

The "v" prefix can be read multiple ways:
- "version 2" (Brooks-style second version)
- "vortex" (the architectural reorganization gathering all the learnings)
- "verified" (designed with empirical learnings from cortex)

Any reading works; the substance is the clean rewrite.

## Composition with existing pins

This pin sits high in the architectural hierarchy because it shapes how every other pin's commitments eventually realize:

| Pin | Composition |
|---|---|
| `feedback_doing_then_learning.md` | Cortex was the doing; vCortex is the learning made architectural; this pin captures the project-scale instance of that pattern |
| `project_road_to_launch.md` | V1 ships on cortex; vCortex begins post-launch; pin clarifies that cortex isn't forever |
| `project_tsunami_serving_architecture.md` | Tsunami architecture solves V1 scale via vLLM tier; vCortex eventually replaces the cortex tier; both pins describe parts of the same V1+ operational story |
| `project_unified_memory_architecture.md` | Insights 1-4 describe what vCortex implements natively |
| `project_per_layer_injection_auxiliary.md` | vCortex includes the auxiliary from initialization |
| `project_training_time_representation.md` | vCortex is the canonical project-scale instance of the meta-principle (commit at training/design time, not post-hoc) |
| `project_representation_emitting_models.md` | vCortex's architecture is the trinity from initialization |
| `project_compression_substrate_quality_bar.md` | vCortex's substrate passes both bars by design |
| `project_dual_head_voice.md` | vCortex has multi-head architecture native |
| `project_eggroll_and_ut.md` | EGGROLL training methodology applies to vCortex's training; UT iteration is a candidate native architecture |
| `project_zynqberry_bitnet_memex.md` | vCortex is FPGA-deployable by design |
| `project_cortex_v1_shim_api.md` + `project_cortex_ffn_shims.md` | Shim API is native to vCortex, including bias-attention and per-layer injection |
| `project_memex_identity.md` | vCortex couples with memex at the substrate-representation level (joint design) |
| `project_encoder_fine_tuning_priority.md` | Encoder fine-tuning is part of vCortex's launch preparation; the trinity training infrastructure makes encoder iteration cheaper than cortex's encoder iteration was |
| `project_five_systems.md` | vCortex replaces cortex's slot in the five-systems framing |
| `project_zynqberry_bitnet_memex.md` | vCortex's FPGA-deployable design is what makes the chapter-appliance / FPGA-resident vision shippable |
| `feedback_no_camouflage.md` | vCortex is honest about being a second version; cortex isn't called "cortex2" or pretending to be the same system |

The pin discipline's accumulated insight comes home in vCortex. Every architectural pin describes something vCortex can realize natively.

## What this is NOT

To prevent scope creep and premature action:

- **NOT a green light to start vCortex now.** Cortex must finish first. Launch must happen on cortex. Post-launch dust must settle. The timing is non-negotiable.
- **NOT a license to abandon cortex.** Cortex stays in production. The retrofit work (vram-heap, memory planning, tsunami serving integration) continues until cortex is genuinely finished.
- **NOT a rewrite to "fix bugs."** vCortex is an architectural realization of accumulated learnings, not a bug-fix campaign. Bugs that surface get fixed in cortex; architectural commitments get realized in vCortex.
- **NOT a "we'll just port the code over" plan.** vCortex is designed-in. Code-level porting would inherit cortex's structure, which is what we're moving away from.
- **NOT a replacement for further pin work.** vCortex is the realization of pins, not the end of pin work. New pins will continue to accumulate.
- **NOT a admission that cortex was wrong.** Cortex was right *for the path traveled*. The doing was necessary to make the learning visible. vCortex would not be designable without cortex's empirical existence.
- **NOT a guarantee vCortex ships.** Like all multi-year architectural commitments, vCortex is the planned next architecture. Empirical surprises could shift the plan. The commitment is "after cortex finishes, the next thing is vCortex unless we learn otherwise."

## Recognition criteria — when has cortex earned its retirement?

The transition point is recognized rather than scheduled. Signals that cortex is "finished":

- **Architectural ceiling reached**: further substantial improvements require structural changes the codebase can't gracefully accommodate
- **Operational stability**: bug rate stable; no architectural-level surprises remaining; ops runbooks comprehensive
- **Quality threshold**: meets performance, capacity, and reliability targets; users not blocked by cortex-quality issues
- **Substrate work landed**: vram-heap migration complete; memory planning discipline in place; tsunami architecture integrated; pins reflect cortex's actual state
- **Roadmap exhausted**: cortex-specific features aren't being added; new architectural ideas land in vCortex's pin set instead

When all of these are true, cortex has earned its retirement. The clean-slate next chapter begins.

## How to apply

### When prioritizing cortex's remaining work

After the tsunami architecture lands, the question for any new cortex work is: *"is this making cortex finishable, or is it adding something that should land in vCortex instead?"*

- Finishability work (vram-heap completion; memory planning; tsunami integration; performance polish; bug fixes): cortex
- Architectural endgame work (trinity training; batched-from-start; sigmoid attention native; BitNet from initialization): vCortex

This filter reduces cortex's roadmap pressure dramatically. Anything that's "make cortex better in a structural way" is vCortex work.

### When AgentOS or other systems need cortex changes

Same filter applies: is the change finishability work or architectural endgame? Communicate accordingly. The AgentOS-cortex API contract (per `project_agentos_api_contract.md`) should be stable enough that vCortex can speak the same protocol; both cortex and vCortex implement the same interface from AgentOS's side.

### When the path-traveled has produced new architectural insights

Pin them. They become commitments vCortex inherits. The pin discipline that produced this strategic shape continues; vCortex's design IS the pin set.

### When tempted to start vCortex before cortex is finished

Resist. The Brooks discipline is explicit: the second version comes *after* the first is done. Starting vCortex while cortex is still in active development means:

- Split engineering attention across two architectures simultaneously
- vCortex's design is shaped by cortex's still-evolving state (less learning, worse design)
- Cortex's finishability work gets deprioritized (cortex stays half-finished forever)
- The clean slate becomes contaminated with "what we wish we'd done in cortex"

Wait for the trigger conditions. Calm completion is the discipline.

### When cortex-claude or other sessions push for cortex refactors

The right response is often: *"is this finishability work, or is this trying to retrofit a vCortex commitment onto cortex?"* If the latter, defer to vCortex.

This protects cortex from death-by-architectural-retrofit and protects vCortex's clean-slate quality.

## The phrases to remember

> ***Brooks' plan to throw one away applied strategically: cortex was the doing; vCortex is the learning made codified. Once cortex has earned its retirement (post-launch dust settled, quality threshold met, finishability work landed), `/src/vcortex` begins as a clean architectural realization of every endgame pin the project has accumulated. Sibling repo, not branch; calm completion, not pressure-driven launch.***

Plus the inheritance claim:

> ***vCortex inherits from initialization every commitment cortex can't fully reach incrementally: memory planning first-class; trinity from training-time; batched serving native; sigmoid attention default; BitNet ternary from start; vram-heap native substrate; shim architecture clean; TurboQuant operations native; memex as substrate not bolt-on. Designed-in, not retrofit.***

Plus the parallel-development pattern:

> ***Cortex stays in production while vCortex grows on the side. No launch pressure on vCortex; cortex carries the V1+ traffic. Eventually parallel-run for verification; migration when confidence is earned; cortex deprecated when vCortex's substrate is proven. Both repos are first-class in `/src/`; both serve the project; the transition is gradual and verified.***

Plus the recognition criterion:

> ***"Cortex is finished" is a recognized moment, not a scheduled one. Architectural ceiling reached; operational stability; quality threshold met; substrate work landed; cortex-specific roadmap exhausted. When all are true, cortex has earned its retirement. The clean-slate next chapter begins.***

Plus the discipline:

> ***Don't start vCortex until cortex is finished. Don't retrofit vCortex commitments into cortex. Don't pretend cortex was wrong (it was right for the path traveled). Don't pretend vCortex is just a code port (it's a designed-in architectural realization). The pin discipline that produced this strategic shape continues; vCortex's design IS the pin set realized.***

## Related pins

- `feedback_doing_then_learning.md` — cortex was the doing; vCortex is the learning made architectural
- `project_road_to_launch.md` — V1 ships on cortex; vCortex begins post-launch
- `project_tsunami_serving_architecture.md` — V1 scale via vLLM tier; vCortex eventually replaces cortex tier
- `project_unified_memory_architecture.md` — Insights 1-4 are what vCortex implements natively
- `project_per_layer_injection_auxiliary.md` — auxiliary as first-class in vCortex
- `project_training_time_representation.md` — meta-principle vCortex realizes at the project scale
- `project_representation_emitting_models.md` — trinity from initialization in vCortex
- `project_compression_substrate_quality_bar.md` — vCortex substrate passes both bars by design
- `project_dual_head_voice.md` — multi-head native in vCortex
- `project_eggroll_and_ut.md` — training methodology + candidate native architecture for vCortex
- `project_zynqberry_bitnet_memex.md` — vCortex is FPGA-deployable by design
- `project_cortex_v1_shim_api.md` + `project_cortex_ffn_shims.md` — shim architecture native in vCortex
- `project_memex_identity.md` — vCortex couples with memex at substrate-representation layer
- `project_encoder_fine_tuning_priority.md` — encoder fine-tuning is part of vCortex's launch preparation
- `project_five_systems.md` — vCortex eventually assumes cortex's slot
- `feedback_no_camouflage.md` — vCortex is honestly a second version, not pretending to be cortex
- (Future) memory-planning pin — vCortex implements memory planning from start
