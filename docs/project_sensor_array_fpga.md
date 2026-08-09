---
name: sensor-array-fpga
description: "Architectural pattern for the project's agentic-behavior substrate: a population of small (~0.5 MB BitNet-ternary) classifier sensors running in parallel at the top of cortex's stack, each detecting a specific 'kind' of trajectory the model is about to embark on (CoT entry, deviation from prompt, friction detected, notable-member trigger, confidence-low, voice-register-drift, tier-routing decision, etc.). The workload pattern: many small sensors, every emission step, hard latency requirement (must not slow the LLM). This pattern is FPGA-natural in a way it's GPU-hostile — FPGAs instantiate many parallel pipelines in fabric with deterministic latency; GPUs pay per-kernel-launch overhead that dominates for tiny networks. As more 'kinds' get added (CoT, friction, deviation, mission-triggers, etc.), the array grows additively without restructuring existing sensors. This is the structural substrate underneath the project's shim architecture (`project_cortex_ffn_shims.md`) — the shims ARE sensors — and the natural hardware target is the BitNet FPGA path (`project_zynqberry_bitnet_memex.md`). Enables Bob to be deeply agentic without big-iron infrastructure: cortex's heavy generation runs on H100; situational-awareness sensor array runs on modest Artix/Zynq-class FPGA. Surfaced 2026-06-10 evening when Daniel walked from 'the final layer encodes the kind of next token' to 'you could put a sensor at the top' to 'sensors compose additively' to 'this fits FPGA naturally.'"
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-10 evening as the natural conclusion of a derivation arc through transformer inference mechanics. Daniel walked, in four steps, from "the final layer encodes the kind of next token" → "you could put a sensor at the top to read kinds" → "sensors compose additively" → "this fits FPGA naturally." The pin captures the workload pattern, the hardware fit, and the strategic implication.

## The workload pattern

At every emission step, cortex computes a rich final-layer state that encodes not just "the next token" but "the kind of next token within an upcoming trajectory" (per the fingerprint-of-understanding framing in `project_memex_identity.md` applied to output tokens). A sensor at the top of the stack reads that rich state and classifies a specific property.

But you don't have ONE sensor. You have **a population of sensors**, each detecting a different "kind":

- *Is this a deviation from the prompt's stated intent?*
- *Is the model about to enter CoT / reasoning mode?*
- *Does Bob actually have something useful to say about this?* (per `project_silence_as_first_class.md`)
- *Is the conversation touching mission-relevant territory?* (notable-member, memorial topic, etc.)
- *Is the user expressing friction Bob could help bridge?* (per `project_bob_friction_detector.md`)
- *Is upcoming emission low-confidence and likely to hallucinate?*
- *Is voice register drifting away from Bob's?* (per `project_bob_voice.md`)
- *Is this query substrate-light enough for vLLM tier or does cortex need to handle it?* (per `project_tsunami_serving_architecture.md`)
- *Is the model about to commit to an agentic action that requires pseudonymization?* (per `project_hybrid_serving_pseudonymization.md`)
- *Is this an interview-mode trigger?* (per `project_bob_interview_mode.md`)

The list grows over time as the project identifies more kinds worth detecting. Each addition is a new small sensor; the existing sensors don't change.

**The workload property**: many tests, every emission step, must be fast enough not to slow the LLM down.

This is exactly the workload pattern FPGAs are designed for.

## Why this is FPGA-natural

Sensor sizes are tiny. A 0.5 MB BitNet-ternary sensor has ~2.5M parameters; an FP16 sensor of the same size has ~250K parameters. Either fits entirely in FPGA BRAM/URAM with bandwidth to spare.

The architectural fit, point by point:

| Property | Why it suits FPGA |
|---|---|
| **Small, fixed-shape networks** | Fit in on-chip memory; no DRAM traffic; locality optimal |
| **Many parallel sensors** | FPGAs instantiate many independent pipelines in fabric simultaneously |
| **Inference-only at production** | Fixed compute graph = optimal FPGA case; no dynamic dispatch overhead |
| **Real-time per-emission latency** | FPGAs have deterministic sub-millisecond latency; no scheduler unpredictability |
| **Ternary-friendly** | Per BitNet: every MAC becomes add/sub/skip; LUT-fabric wide adder trees; bypasses DSP slices entirely |
| **Additive scaling** | Each new sensor = more fabric occupied; existing pipelines unchanged; no per-call coordination overhead |

The throughput math: at cortex generating 15-30 tokens/sec, each emission has ~33-66 ms of latency budget. The sensor array's job is well under 1 ms — far inside the budget. Adding more sensors doesn't extend the latency; it just occupies more fabric in parallel.

### Why this is GPU-hostile

Compare on GPU:

- N sensors on GPU = N separate kernel launches per emission step
- Per-kernel-launch overhead (~10s of microseconds) dominates for tiny networks
- The actual computation might be 1 microsecond; the launch overhead is 10x that
- GPU schedulers are optimized for large monolithic workloads, not many tiny parallel ones
- At 10 sensors: GPU is paying 10x kernel-launch overhead per emission; adds up
- At 50 sensors: GPU is impractical; FPGA is still humming

**The key architectural asymmetry**: on GPU, adding a sensor has *multiplicative* cost (more kernel overhead per step). On FPGA, adding a sensor has *additive* cost (more fabric occupied, no per-step overhead).

This asymmetry is structural — it's about how each substrate handles many-small-parallel workloads. GPU is designed for few-big-sequential. FPGA is designed for many-small-parallel.

## The architecture in operation

```
GPU (cortex inference):
  Forward pass → final-layer hidden state (per emission step)
       ↓
       (state transferred via PCIe/Ethernet to FPGA — small payload, ~10-50 KB)
       ↓
FPGA (sensor array):
  Sensor 1: CoT-entry detector       ─┐
  Sensor 2: Friction detector        ─┤
  Sensor 3: Notable-member trigger   ─┤
  Sensor 4: Confidence calibration   ─┼──→ All run in parallel
  Sensor 5: Voice-register drift     ─┤    All return decisions per step
  Sensor 6: Tier-routing classifier  ─┤
  ... (additional sensors)           ─┘
       ↓
       (decisions transferred back to GPU — tiny payload)
       ↓
GPU (cortex inference continues):
  Routing layer integrates sensor decisions
  Per-emission contextual behavior:
    - Activate reasoning shims (CoT detected)
    - Switch memex shards (topic-class detected)
    - Route to operator-Bob (friction detected)
    - Queue interview-mode capture (notable-member detected)
    - Adjust sampling temperature (confidence detected)
    - Pseudonymize before frontier dispatch (agentic-action detected)
    - Etc.
```

The PCIe round-trip per emission is feasible — modern PCIe handles state transfer in tens of microseconds. Total overhead: well under a millisecond. Negligible relative to cortex's ~33-66 ms per emission budget.

## What this enables strategically

This is the substrate underneath **agentic behavior without big-iron infrastructure**. Concretely:

- **Cortex's heavy generation** runs on rented H100 (or local 4090/5090 once owned)
- **Sensor array** runs on modest FPGA — $200-500 Artix/Zynq-class (per `project_zynqberry_bitnet_memex.md`)
- **Combined**: Bob can be situationally aware about CoT entry, friction, mission-triggers, voice register, tier routing, confidence — all per-emission, without slowing the model

The asymmetric-silicon shape: GPU runs cortex flat out; FPGA runs the situational awareness; PCIe connects them. **Different workloads on different substrates, each optimized for its workload.**

This also composes with the chapter-appliance vision: a barbershop chapter could conceivably host their own local Bob instance with a modest Zynq-class FPGA (for sensors + auxiliary) plus a GPU subscription (for cortex inference). Big-iron isn't sole-sourced; chapters can have their own.

## What the sensor architecture IS (relation to shims)

This pin **explicitly names what the project's shim architecture has been implicitly building**. Per `project_cortex_ffn_shims.md`: shims are small (~28k-param) "steering/gating modules." Per `project_cortex_v1_shim_api.md`: three-phase API (inject / gate / steer).

The **gate phase shims ARE sensors**. The "should I reply?" shim reads cortex's state and classifies whether to emit a response (per `project_silence_as_first_class.md`). The friction-detector shim (per `project_bob_friction_detector.md`) is a sensor for the friction kind. The notable-member trigger (per `project_bob_interview_mode.md`) is a sensor for the historical-figure-mention kind.

This pin **generalizes the shim pattern** into a workload-and-hardware shape:
- Shims = the individual sensor implementations
- Sensor array = the population deployment pattern
- FPGA = the natural hardware target for the array

So the pin doesn't introduce new architecture — it names the workload pattern explicitly and identifies why FPGA is its natural home. Existing shim work is the implementation; this pin is the deployment architecture.

## Composition with existing pins

| Pin | How sensor-array-on-FPGA composes |
|---|---|
| `project_cortex_ffn_shims.md` | Shims are the individual sensors; this pin is their array deployment |
| `project_cortex_v1_shim_api.md` | Gate-phase shims are sensors; inject/steer composition unchanged |
| `project_silence_as_first_class.md` | The "should I reply?" shim is the canonical first sensor |
| `project_bob_friction_detector.md` | Friction detection is a sensor instance |
| `project_bob_interview_mode.md` | Notable-member trigger is a sensor instance |
| `project_bob_voice.md` | Voice-register-drift detector is a sensor instance |
| `project_per_layer_injection_auxiliary.md` | The auxiliary is the *write-side* representation-emitting model; sensors are the *read-side*; both are FPGA-natural and compose on the same hardware |
| `project_zynqberry_bitnet_memex.md` | This pin describes a specific workload that runs on the same FPGA hardware path |
| `project_training_time_representation.md` | Sensors trained in ternary from initialization; designed for FPGA from training time |
| `project_eggroll_and_ut.md` | EGGROLL gradient-free integer training methodology applies to sensor training |
| `project_representation_emitting_models.md` | Sensors are the *representation-consuming* complement to representation-emitting models; same class of small specialized networks, opposite direction |
| `project_modular_cognition_architecture.md` | Sensor array is part of System 1 multi-shim composition; agentic behavior emerges from sensor decisions composing |
| `project_tsunami_serving_architecture.md` | Tier-routing decision is a sensor instance; routing shim from tsunami pin is this pattern |
| `project_hybrid_serving_pseudonymization.md` | Agentic-action-trigger detector is a sensor that determines when to invoke pseudonymization + frontier dispatch |
| `vcortex/mamba-recollection-position.md` | "Always-hot-probe watching a frozen base" — sensor array is the population version of this pattern; same lineage |
| `project_vcortex_strategy.md` | vCortex includes the sensor array architecture from initialization, not as retrofit |

## What this is NOT

To prevent scope creep:

- **NOT a replacement for cortex's text generation.** Sensors classify; cortex generates. Different roles.
- **NOT a new training framework.** Sensors trained via standard supervised classification on (hidden_state, label) pairs, ideally with QAT for ternary deployment.
- **NOT a substitute for the per-layer-injection auxiliary.** Sensors read; auxiliary writes. Complementary roles.
- **NOT GPU-incompatible.** A modest number of sensors can run on GPU at acceptable overhead; the FPGA case becomes compelling at 10+ sensors and dominant at 50+.
- **NOT v1 work.** This is v3+/v4+ horizon, composing with the FPGA path. Soft launch ships on GPU-only; FPGA deployment is post-launch.
- **NOT a research project requiring novel methodology.** Probe classifiers on transformer hidden states are well-established practice (mechanistic interpretability lineage); the architectural arrangement is the contribution, not the training technique.

## Sensors as a category — what to detect

The "list of kinds" worth detecting is open-ended; here are the priority instances:

| Kind to detect | Mission relevance |
|---|---|
| **Should-respond gate** (silence as first-class output) | Daily / per-turn; load-bearing for Bob's ambient register |
| **CoT-entry / reasoning mode** | Per-emission; routes to reasoning shims; allocates compute |
| **Topic-class classifier** (corpus-aware) | Per-emission; routes to appropriate memex shard |
| **Notable-member trigger** | Mission-critical; interview-mode capture |
| **Friction detector** | Per-conversation; operator-Bob bridge |
| **Confidence / hallucination** | Per-emission; aggressive retrieval; voice shifts |
| **Voice-register drift** | Per-emission; voice shim activation |
| **Tier-routing classifier** | Per-query; vLLM vs cortex vs frontier dispatch |
| **Agentic-action trigger** | Per-emission; pseudonymization + frontier dispatch |
| **Deviation-from-prompt detector** | Per-emission; prompt-injection defense; dragnet adjacent |
| **Memorial-topic classifier** | Per-conversation; voice + retrieval shift |
| **Pete-mention / specific-VIP detector** | Mission-critical; interview-mode + Columbo signal |
| **Quartet/chorus-name detector** | Mission-critical; ringhub graph enrichment |
| **Confabulation risk classifier** | Per-emission; abstention or aggressive retrieval |
| **Voice-input transcription confidence** | Per-utterance; affects response confidence |

This list will grow. The architecture is designed for additive growth — each new kind detected is another small sensor, FPGA fabric absorbs it, no restructuring required.

## How to apply

### When designing a new behavioral feature

Ask: *"Is this feature triggered by a property of cortex's internal state at emission time?"* If yes, the implementation is likely a sensor. The architecture provides:

- Standardized hidden-state reading via the shim API surface
- Standardized decision output format
- Routing infrastructure that integrates sensor decisions into cortex's flow
- FPGA deployment target for the population

You don't have to design the whole infrastructure each time; just specify the new sensor's training data and label, train it, deploy it as another pipeline in the FPGA fabric.

### When evaluating mechanistic interpretability research

Filter by sensor relevance:

- **Linear probes on hidden states**: directly applicable as sensor implementations
- **Sparse autoencoders identifying concept neurons**: potentially valuable as sensor input features
- **Lookahead heads / planning representations**: relevant for "what kind of trajectory is committed" sensors
- **Adversarial probe defenses**: relevant for sensor robustness to prompt injection

### When sizing the FPGA target

Per `project_zynqberry_bitnet_memex.md`: $200-500 Artix/Zynq-class is the design target. Conservatively, that supports 10-50 ternary sensors of 0.5 MB each in parallel pipelines. Plenty of headroom for the array to grow as the project identifies more kinds worth detecting.

### When evaluating asymmetric deployment

The sensor array + per-layer auxiliary together fit on the same FPGA. The combined load:

- Sensor array: classifies trajectory kinds per emission
- Auxiliary: writes per-layer steering vectors per emission
- Both run in parallel pipelines in the same fabric
- Same PCIe round-trip serves both
- Same training-time-representation discipline applies

This is what `vcortex/mamba-recollection-position.md` calls "asymmetric workloads on asymmetric silicon." GPU runs cortex flat out; FPGA hosts both the reading-side (sensor array) and writing-side (auxiliary) infrastructure.

## The phrase to remember

> ***At every emission step, cortex's final-layer state encodes 'the kind of next token within an upcoming trajectory.' A population of small ternary sensors reads that state and classifies specific properties — CoT entry, friction, notable-member trigger, voice drift, etc. The workload is many-small-parallel-fast-fixed-shape, which is FPGA-natural and GPU-hostile. As more 'kinds' get added over time, the array grows additively in fabric. The whole thing runs in parallel pipelines on $200-500 Artix/Zynq-class FPGA, well within sub-millisecond per emission. Bob becomes deeply agentic without big-iron infrastructure.***

Plus the architectural recognition:

> ***Shims are sensors. The project's existing shim architecture has been implicitly building this population deployment pattern; this pin names it explicitly. Sensors are the read-side complement to representation-emitting models (write-side). Both compose on the same FPGA fabric; both are designed at training time for ternary deployment.***

Plus the strategic claim:

> ***GPU does heavy generation; FPGA does situational awareness. Different workloads on different substrates, each optimized for its workload. PCIe connects them. This is what 'asymmetric workloads on asymmetric silicon' actually means in deployment terms. Chapter-appliance Bob — one FPGA plus GPU rental — becomes a coherent deployment option.***

## Related pins

- `project_cortex_ffn_shims.md` — shims are the individual sensors
- `project_cortex_v1_shim_api.md` — gate-phase shims are sensors
- `project_silence_as_first_class.md` — should-respond is the canonical first sensor
- `project_bob_friction_detector.md` — friction detection sensor instance
- `project_bob_interview_mode.md` — notable-member trigger sensor instance
- `project_bob_voice.md` — voice-register drift sensor instance
- `project_per_layer_injection_auxiliary.md` — write-side complement; both on FPGA
- `project_zynqberry_bitnet_memex.md` — FPGA path; this pin's natural hardware home
- `project_training_time_representation.md` — sensors trained in ternary from initialization
- `project_eggroll_and_ut.md` — training methodology applies to sensors
- `project_representation_emitting_models.md` — sensors are the consuming-side category
- `project_modular_cognition_architecture.md` — sensor array part of System 1 multi-shim composition
- `project_tsunami_serving_architecture.md` — tier-routing is a sensor instance
- `project_hybrid_serving_pseudonymization.md` — agentic-trigger is a sensor instance
- `project_vcortex_strategy.md` — vCortex includes sensor array from initialization
- `vcortex/mamba-recollection-position.md` — "always-hot probe watching a frozen base" lineage
