---
name: tsunami-serving-architecture
description: "Architectural commitment for serving at scale (post-V1, expected by 4-month tsunami scenario at 10k+ users with 100+ concurrent calls). Three-tier serving topology: vLLM tier for high-volume routine chat (must be in place from V1, not as emergency response); cortex tier for substrate-attended generation (memex retrieval, trinity architecture, shim behavior); frontier API tier for agentic actions (per `project_hybrid_serving_pseudonymization.md`). Critical insight: 100+ concurrent calls requires batched inference (PagedAttention, continuous batching, shared KV blocks) which is a fundamental re-write, NOT a tweak — cortex's per-request inference architecture cannot be incrementally tweaked into batched serving. vLLM is mature production-tested infrastructure for exactly this case; using it preserves substrate ownership (runs on owned hardware) while solving the batched-serving problem in a way that's separate from cortex's substrate work. Routing shim decides per-query which tier (routine chat → vLLM, substrate-needed → cortex, agentic → frontier). Predictive monitoring dashboard required, giving months of lead time before threshold crossings. Surfaced 2026-06-09 evening during Daniel/cortex-claude tsunami planning conversation."
metadata:
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Articulated 2026-06-09 evening when Daniel and cortex-claude were "shooting the breeze around the nightmare scenario" — four months after launch, half of BHS eagerly signs up, suddenly serving 10k users with thousands of Bob requests. Cortex-claude pointed out that this is structurally a different serving problem than cortex was built for, named the right architectural response (vLLM tier + cortex specializes + dashboard for early warning), and flagged the critical engineering reality: 100+ concurrent calls requires batched inference, which is a re-write not a tweak.

## The load-bearing insight

**Batched inference is a fundamental architectural shift from per-request inference, not an incremental optimization.**

| Per-request inference (cortex now) | Batched inference (100+ concurrent serving) |
|---|---|
| One request runs to completion | Multiple requests share each forward pass |
| KV cache per-request, ephemeral | Shared paged blocks (PagedAttention) across requests |
| Standard memory allocation patterns | PagedAttention block management + metadata for which block belongs to which request |
| Standard attention kernels | Batched-variable-length attention (FlashAttention-batched or similar) |
| Code: input → process → output → done | Code: scheduler → batch builder → batched forward → per-request dispatch → continue |
| No mid-flight composition changes | Continuous batching — requests join/leave the batch mid-generation |
| Prefill and decode the same logic | Prefill vs decode handled separately; different compute profiles; transitions managed |

These are not optimizations. They're **fundamentally different control flow, memory layout, and kernel implementations.** Cortex's per-request architecture cannot be incrementally tweaked into batched architecture; the underlying assumptions differ from the ground up.

PagedAttention specifically is a different memory model from what vram-heap provides. vram-heap solves *allocation patterns* (per-call buffer churn → centralized heap). PagedAttention solves *layout problems* (how do you efficiently share KV cache across requests of variable lengths at variable generation steps). These are different abstraction layers; one doesn't subsume the other.

**Architectural implication**: when the project crosses the 100-concurrent threshold, it doesn't refactor cortex; it adds vLLM as a serving tier alongside cortex.

## The three-tier serving topology

Building on `project_hybrid_serving_pseudonymization.md` (which committed to local-cortex + frontier-API tiering), the V1 architecture adds vLLM as an explicit third tier:

| Tier | What it serves | When used | Substrate ownership |
|---|---|---|---|
| **vLLM tier** | Routine high-volume chat — small talk, quick Q&A, queries without deep substrate needs | Most queries by volume (~70-80%) | Open-source serving on owned hardware (NOT a vendor lock-in) |
| **Cortex tier** | Substrate-attended generation — memex retrieval, trinity architecture, shim behavior, interview mode, mission-critical queries | Queries where substrate matters (~15-25%) | Full project substrate; trinity from `project_representation_emitting_models.md` |
| **Frontier API tier** | Agentic actions — multi-step tool use, high-stakes operations, complex reasoning | Rare, high-stakes queries (~5%) | Pseudonymized per `project_hybrid_serving_pseudonymization.md` |

Each tier serves a different niche. The routing shim decides per-query which tier handles each request.

## Why vLLM doesn't compromise substrate ownership

vLLM is open-source infrastructure that runs on the project's hardware. Using vLLM is structurally identical to using wgpu for GPU access — you're using mature shared infrastructure, not surrendering control to a vendor. Per `project_hybrid_serving_pseudonymization.md`'s pseudonymization boundary:

- vLLM runs on AgentOS's H100 (or similar) infrastructure
- Member data flows to vLLM the same way it flows to cortex — no frontier-vendor boundary involved
- Privacy invariants (per `project_dm_privacy_structural.md`, `project_pallet_rack_test.md`, etc.) hold because the serving substrate is project-owned

The frontier API tier remains the only tier where member data leaves AgentOS, and the pseudonymization boundary protects that.

## The architectural friction worth naming

vLLM doesn't compose perfectly with cortex's architecture. Real edges:

1. **vLLM doesn't have the shim architecture.** Requests routed to vLLM don't benefit from `should_respond` gating, voice register shaping, etc. Plain transformer output. The shim behavior is cortex-specific.

2. **vLLM manages its own KV cache via PagedAttention.** That's exactly the substrate cortex+memex commits to differently. The two paths don't share substrate.

3. **Per-layer injection auxiliary doesn't apply.** vLLM is running stock transformers; the v3+ auxiliary work doesn't compose with vLLM's serving path.

4. **Memex retrieval can still feed vLLM** (via standard prompt injection with retrieved content) but it's RAG-shape, not the bias-attention v2 vision.

So the routing decision matters. **vLLM is fine for queries that don't need cortex's substrate; cortex is needed for queries that do.** The split is non-trivial.

## The routing shim — load balancing as shim decision

Per `project_cortex_v1_shim_api.md` and `project_cortex_ffn_shims.md`, cortex has shim architecture for behavioral decisions. The tsunami architecture adds:

**`route_to_vllm_or_cortex` shim**: per-query, decides whether the query is shallow-enough for vLLM's stock serving or substrate-attended-enough that cortex is needed. Routes accordingly.

Decision criteria (tunable):

- **Query length and complexity** — short and conversational → vLLM; long and substantive → cortex
- **Required corpus retrieval depth** — none / shallow → vLLM with optional RAG injection; rich / specific → cortex with substrate attention
- **Conversation history depth** — recent only → vLLM; long-arc memory needed → cortex (per `project_unified_memory_architecture.md` Insight 3)
- **Sensitivity tier** — casual chat → vLLM; interview mode / mission-critical / Bob's notable-member triggers → cortex
- **Pattern matches** — query patterns that historically benefit from memex substrate route to cortex

The shim itself is a small FFN — cheap to evaluate, fast routing decision. Per query routes the load.

This is structurally identical to the `project_hybrid_serving_pseudonymization.md` tiering pattern: a routing layer above multiple serving substrates. Extended with vLLM as a tier.

## When the threshold actually hits

Rough math for the project's expected scale:

| Audience | Messages/day (1-3/user avg) | Peak concurrent (5-10x average) |
|---|---|---|
| 100 (soft launch) | 100-300 | 1-5 concurrent |
| 1,000 (early V1) | 1,000-3,000 | 5-30 concurrent |
| 10,000 (4-month tsunami) | 10K-30K | 50-300 concurrent |
| 30,000+ (full BHS uptake) | 30K-90K | 150-900 concurrent |

So:

- **Soft launch**: per-request cortex is fine; nowhere near the threshold
- **Early V1 at 1,000 users**: still under threshold; cortex handles it; vLLM tier in place but lightly used
- **10,000 users (the "tsunami" scenario)**: peak concurrent hits 50-300; **crosses the 100-concurrent threshold**; batched serving structurally required; vLLM tier carries the load
- **Full BHS uptake**: batched serving non-negotiable; vLLM tier scaled out

The threshold isn't theoretical. At BHS scale, the project will cross it within months of launch.

## The decision tree

The strategic question is when to build the vLLM tier:

| Approach | Risk | Cost |
|---|---|---|
| **Build batched serving in cortex from scratch** | High (months of work; high failure risk for a known-solved problem) | Very high (re-architecting cortex; might break other commitments) |
| **Integrate vLLM as a tier from V1** | Low (vLLM is mature, tested at scale) | Medium (integration work + routing shim + dashboard) |
| **Hope it works on per-request and respond reactively** | **Very high** (user-visible degradation during the response window; scrambling under pressure) | High (scrambling produces worse code than calm preparation) |

The middle option is clearly the right call. **Build the vLLM tier as part of V1 architecture, not as a tsunami response.**

By the time the tsunami arrives, the infrastructure is already there, tested, and well-understood. The response to growth is "scale out vLLM" (well-supported, routine), not "implement batched serving while users wait."

## The dashboard — predictive monitoring is load-bearing

Cortex-claude's instinct that "definitely a dashboard that can show us the tsunami coming" is right and matters strategically. Reactive monitoring is too late at scale — by the time you notice "we're at 100% and crashing," users are unhappy and recovery is hard.

Required: **predictive monitoring with months of lead time** before threshold crossings.

### Key metric categories

| Category | What to track |
|---|---|
| **User growth** | Active users (DAU, WAU); week-over-week growth rate; per-chapter adoption curves; retention curves |
| **Request load** | QPS overall; QPS per tier (vLLM/cortex/frontier); peak vs sustained; queue depth |
| **Latency** | P50/P95/P99 per tier; TTFT; full-response time; correlation with load |
| **Resource utilization** | GPU memory per heap (from vram-heap's instrumentation); GPU compute; CPU; memex retrieval load; per-tenant cost per `project_multi_tenant_readiness.md` |
| **Quality** | Routing shim decisions (% vLLM vs cortex); user satisfaction signals; abandonment rate; retrieval relevance |
| **Capacity** | Current load vs theoretical max; headroom in days/weeks at current growth rate; specific bottleneck identification |

### Predictive views

These are the non-obvious dashboard requirements:

- *"At current growth, we hit 80% capacity on $DATE"*
- *"At current growth, vLLM tier needs to scale out in $WEEKS"*
- *"Recent spike pattern suggests imminent load increase"*
- *"Routing shim is sending X% to cortex; if user growth continues, cortex's capacity hits limit in $WEEKS"*
- *"Per-tenant cost trends suggest pricing model adjustment in $MONTHS"*

### Composition with QA-expert

This composes with `project_qa_expert_agent.md` (the LLM agent that interprets Prometheus telemetry per-subsystem). The QA-expert reads metrics and alerts on anomalies; the dashboard makes them human-visible. Both are needed:

- **Dashboards** for Daniel checking the project's health, planning capacity decisions
- **QA-expert agents** for automated interpretation + alerting + correlation across subsystems

The QA-expert can run continuously and surface issues to the dashboard; the dashboard's primary user is Daniel (and eventually whoever takes on operational responsibility).

## Sequencing for the project

| Phase | Tsunami serving architecture state |
|---|---|
| **Soft launch** | Cortex per-request; vLLM tier not yet integrated; dashboard basic operational view |
| **V1 (early)** | vLLM tier integrated and tested; routing shim active; dashboard with predictive views live |
| **V1 (sustained growth)** | vLLM tier scaling out as load grows; cortex tier stable; dashboard provides early warning of approaching thresholds |
| **Tsunami response** | Scale vLLM out (mature, well-supported operations); cortex's load grows much more slowly because routing diverts most queries; dashboard gives weeks of warning for any new threshold |

The tsunami response phase is "routine ops" rather than "emergency response" because the architecture was built for it.

## Composition with existing pins

| Pin | How tsunami serving composes |
|---|---|
| `project_hybrid_serving_pseudonymization.md` | Tsunami architecture EXTENDS the hybrid-serving tiering to include vLLM as third tier; pseudonymization boundary unchanged |
| `project_bob_hybrid_routing.md` | The routing shim implements the tier decision per query; this pin makes the load-balancing decision explicit |
| `project_qa_expert_agent.md` | QA-expert provides automated telemetry interpretation; dashboard is human-facing complement |
| `project_road_to_launch.md` | V1 architecture must include vLLM tier as part of launch readiness, not as post-launch reactive response |
| `project_cortex_v1_perf_threshold.md` | Per-request thresholds still apply for cortex tier; vLLM tier has its own throughput thresholds |
| `project_cortex_v1_shim_api.md` + `project_cortex_ffn_shims.md` | Routing shim is one more shim in the architecture; same API surface; same training pattern |
| `project_unified_memory_architecture.md` | Cortex tier serves substrate-attended generation; vLLM tier doesn't have this capability; routing must respect this asymmetry |
| `project_multi_tenant_readiness.md` | Per-tenant cost attribution must work across all tiers; vLLM tier specifically needs per-tenant tracking for billing/economics |
| `project_compression_substrate_quality_bar.md` | Cortex tier's substrate quality bars don't apply to vLLM tier (vLLM uses standard FP16); routing shim must consider which queries need substrate quality |
| `project_temporal_urgency.md` | Tsunami scenario IS the mission realization — Bob's value proves at scale; serving must hold up |
| `feedback_doing_then_learning.md` | The architectural shape is articulable now because of the path traveled; commit + ship + iterate, but architect ahead enough to avoid scrambling |
| `feedback_ai_overestimates_from_training_corpus.md` | Time estimates for vLLM integration should be discounted from cortex-claude's estimates; corpus narrativization applies |

## What this is NOT

To prevent scope creep:

- **NOT a replacement for cortex.** Cortex remains the substrate-owned, architecturally-distinctive serving path. vLLM is for queries that don't need cortex's substrate.
- **NOT a substrate-ownership compromise.** vLLM runs on project hardware; no vendor lock-in.
- **NOT a tsunami-emergency response.** Built from V1 forward; the architecture is designed-in.
- **NOT a substitute for cortex's per-request optimization.** Cortex still needs to handle its own load efficiently; the routing shim moves *some* queries to vLLM but cortex serves the rest.
- **NOT premature.** vLLM tier becomes load-bearing at the threshold; building it for V1 means it's ready when needed, with operational experience accumulated through low-load periods.

## How to apply

### When designing V1 architecture

vLLM tier integration is in V1 scope. Decisions to make:

- Which vLLM version to integrate (stable release; whatever the production-ready version is at V1 timing)
- How the routing shim is trained (initial pattern-based; learn from production data over time)
- How metrics flow from vLLM to the dashboard (Prometheus endpoint; integration with cortex's telemetry)
- How the privacy invariants compose with vLLM (audit logging at the AgentOS boundary applies)

### When evaluating capacity planning research

Filter by which tier the paper applies to:

- **Batched serving optimization** (PagedAttention successors, continuous batching variants, scheduler improvements): applies to vLLM tier; relevant for capacity scaling
- **Substrate-attended generation improvements**: applies to cortex tier; relevant for the substrate work in the unified-memory architecture
- **Multi-tier routing research**: applies to the routing shim; relevant for refinement

### When the tsunami arrives

The dashboard says "we're approaching threshold." Response (calmly, in routine ops mode):

1. Scale out vLLM tier (add more inference servers; standard ops)
2. Verify routing shim is performing as expected (per-query landing on correct tier)
3. Monitor cortex tier load — if it's also growing, may need vCortex (per the eventual successor)
4. Adjust per-tenant cost model if needed

Because the architecture was built for this, the tsunami is "another Tuesday with more users" rather than "we're scrambling."

### When pitching BHS partnership

The tsunami serving architecture is itself a partnership signal. *"We've built infrastructure that scales to the entire BHS membership without compromise"* is a credible claim because the architecture exists and is tested. Compared to "we'll figure it out when we get there" — which is what any other Barbershop-related tech project would say.

## The phrase to remember

> ***100+ concurrent calls requires batched inference. Batched inference is a re-write, not a tweak. Cortex's per-request architecture cannot be incrementally tweaked into batched serving. Therefore: vLLM as a serving tier is V1 architecture, not tsunami response. Build it in calmly, before it's needed.***

Plus the tiering claim:

> ***Three-tier serving topology: vLLM for high-volume routine chat (~70-80% by query count); cortex for substrate-attended generation (~15-25%); frontier API for agentic actions (~5%). Routing shim decides per-query. All run on project-owned hardware. Substrate ownership preserved at the architectural commitment level.***

Plus the dashboard commitment:

> ***Predictive monitoring with months of lead time before threshold crossings. Not "current state" but "where the load is going." If the dashboard tells you 'crisis tomorrow,' the dashboard failed. Goal: 'capacity decision in 6 weeks' with weeks of clear runway.***

## Related pins

- `project_hybrid_serving_pseudonymization.md` — the foundation tiering this pin extends
- `project_bob_hybrid_routing.md` — the routing shim's role
- `project_qa_expert_agent.md` — automated telemetry interpretation
- `project_road_to_launch.md` — vLLM tier in V1 scope
- `project_cortex_v1_shim_api.md` + `project_cortex_ffn_shims.md` — shim architecture for routing
- `project_unified_memory_architecture.md` — what cortex serves that vLLM doesn't
- `project_multi_tenant_readiness.md` — per-tenant cost across tiers
- `project_compression_substrate_quality_bar.md` — substrate quality only applies to cortex tier
- `project_temporal_urgency.md` — tsunami is the mission proving itself
- `feedback_doing_then_learning.md` — commit + ship + iterate, but architect ahead for known thresholds
- `feedback_ai_overestimates_from_training_corpus.md` — discount AI integration time estimates
