---
name: hybrid-serving-pseudonymization-architecture
description: "Hybrid serving architecture for Bob — local 32B + memex + cortex persistent KV cache on rented H100 for latency-sensitive chat (90% of volume); frontier API (Claude initially) for agentic actions (10% of volume, multi-step tool use). AgentOS becomes the pseudonymization boundary: real member data stays local; opaque tokens (member_a8f3) get forwarded to frontier; tools execute on AgentOS side and resolve tokens back to real data. Frontier LLM reasons over the task without ever seeing the user graph. Reframes AgentOS from \"orchestration runtime\" to \"privacy + vendor-abstraction + policy-enforcement layer for AI applications\" — strategic positioning differentiator that frontier-vendor SDKs structurally cannot match. Originated 2026-05-30 evening from cortex-claude conversation; cross-repo architectural pattern."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Originated 2026-05-30 evening during a cortex-claude serving-topology conversation that started in performance work and drifted into deployment architecture for Bob. The pseudonymization boundary insight is the load-bearing new contribution; the hybrid topology that surrounds it is the architectural shape that makes the privacy commitment economically viable.

## The architectural pattern

### Workload decomposition

Bob has two distinct interaction modes with different requirements:

| Mode | % of volume | Latency sensitivity | Context per turn | Quality requirement |
|---|---|---|---|---|
| **Chat** (concierge mode, conversation, retrieval) | ~90% | High — turns feel instant | ~2-5K tokens | Solid mid-tier model (32B class) sufficient |
| **Agentic actions** (booking, multi-step tool use, complex reasoning) | ~10% | Tolerable — async OK | Variable | Frontier-tier quality matters; cost per request is high but volume is low |

These two modes want different model tiers. Forcing them onto one inference substrate is a mismatch: a frontier model is overkill (and expensive) for routine chat; a mid-tier local model is underpowered for high-stakes agentic work.

### Hybrid topology

**Chat path**: Local 32B model + memex + cortex's persistent KV cache, hosted on rented H100 (or similar). Per-conversation cache lives on GPU; each turn computes only the new ~1-2K tokens of prefill instead of replaying full context. Turns feel instant because expensive prefill work isn't repeated.

**Agentic path**: Frontier API (Claude initially) for multi-step tool use. User gets immediate acknowledgment (*"let me look into hotels, give me a minute"*); result returns 30-60s later via async callback (per `project_bob_outbound_api.md` outbound push path).

Rough economics:
- Frontier API tier for agentic: ~$1K-6K/month depending on volume
- Running 70B locally for same agentic quality: ~$3K-9K extra/month in GPU rental
- **Hybrid is cheaper AND higher quality on the agentic mode** because frontier models genuinely outperform 70B on complex multi-step reasoning

### The pseudonymization boundary (the load-bearing insight)

**Real member data never leaves AgentOS.** When Bob's agentic path sends a task to the frontier API, AgentOS substitutes opaque tokens for real identifiers:

```
Pre-substitution (AgentOS-local):
  "Pete Neushul wants to book the Anaheim Hilton for the chorus retreat May 15-17"

Post-substitution (forwarded to frontier):
  "Member member_a8f3 wants to book venue venue_e2d1 for date_range_b4c7"
```

The frontier LLM reasons over the task using the opaque tokens. It produces tool calls like `book_venue(member_a8f3, venue_e2d1, date_range_b4c7)`. **Tool implementations live on AgentOS side**, where:

1. Opaque tokens resolve back to real data
2. AgentOS performs the actual booking (calls the venue's API, processes payment, etc.)
3. Result references the same opaque tokens when sent back to frontier
4. AgentOS de-opaques before showing to the member

**The frontier model never sees the user graph.** It sees opaque identifiers; it reasons about relationships and tasks; it produces tool calls; tools execute locally; results flow back tokenized. The frontier LLM has zero ability to learn about, leak, or cross-reference real member data.

### The analog: payment-processor tokenization

This is structurally identical to how modern payments work. Real PANs (primary account numbers) never reach merchant servers; payment processors substitute `pm_4eC39Hq...` tokens; merchants reason about transactions using tokens; the processor handles authorization with real card data. PCI compliance becomes tractable because the sensitive data lives in one place.

For AI applications: real member data never reaches frontier APIs; AgentOS substitutes `member_a8f3` tokens; the LLM reasons about tasks using tokens; AgentOS handles execution with real data. **The privacy invariant becomes architecturally enforceable rather than policy-dependent.**

## Why this works structurally

The frontier LLM is **doing reasoning**, not **executing side effects**. It produces tool calls; it doesn't actually book hotels. As long as:

1. **All side effects route through tools** (AgentOS-controlled)
2. **All tool implementations live on AgentOS** (not the frontier provider's infrastructure)
3. **All inputs to the frontier are pseudonymized** (real data stripped before forwarding)
4. **All outputs from the frontier reference only tokens** (no real data smuggled through)

...the frontier model is structurally unable to leak member data, because **it never had access to it.** Privacy by data shape, not policy. Same trust-by-construction pattern as `project_dm_privacy_structural.md`, `project_pallet_rack_test.md`, and `project_bob_chat_privacy.md`, but applied at the inter-system boundary.

## Strategic positioning for AgentOS

This **reframes AgentOS** from how it's been described:

| Old framing | New framing |
|---|---|
| "Multi-tenant agent runtime" | "Privacy + vendor-abstraction + policy-enforcement layer for AI applications" |
| "Hosts agents and tools" | "Mediates between apps and frontier providers; enforces invariants; substitutes real data for tokens" |
| "Orchestrates tool calls" | "Owns the trust boundary between app data and frontier APIs" |

What this delivers to application developers building on AgentOS:

1. **Vendor independence** — swap Claude → GPT-5 → Gemini → local-70B in one config. The pseudonymization layer is what makes vendor swap trivial because the apps never depended on vendor-specific data handling.
2. **Audit trail for free** — every frontier call is logged at the AgentOS layer with what tokens were sent, what tool calls came back, what real data was resolved. SOC 2, HIPAA, GDPR posture follows naturally.
3. **Centralized policy enforcement** — rate limits, content filters, cost controls, prompt-injection defenses all apply at the AgentOS boundary regardless of which frontier model is in use. dragnet's adversarial-content detection (per `C:\src\dragnet\`) attaches here.
4. **"Your member data never leaves our infrastructure"** as a real, technically-verifiable sales differentiator. Not "we promise"; **architecturally enforced.**
5. **Cost arbitrage** — apps can route requests between vendors based on quality/cost tradeoffs without app code changes. AgentOS decides; app stays unchanged.

### Why frontier-vendor SDKs structurally can't provide this

The value of the pseudonymization boundary **depends on the orchestration layer being independent of the frontier provider.** Anthropic's SDK cannot offer "your data never goes to Anthropic" as a feature, because the SDK exists to talk to Anthropic. OpenAI's SDK cannot offer "switch effortlessly to Anthropic." These value-propositions are only available to an independent middle layer.

**AgentOS is structurally positioned to offer them** in a way no frontier provider can match — because the value depends on AgentOS *not* being affiliated with any one provider.

This is the kind of strategic positioning that compounds over time: as more AI applications adopt the hybrid pattern, the "you need a vendor-independent privacy layer" requirement becomes industry-standard, and AgentOS becomes the natural answer.

## Quality necessity (the hidden load-bearing argument)

Beyond cost, the hybrid architecture is **quality-mandated**, not just cost-optimized. Research consistently shows that even 70B-class open-weight models make significant errors on multi-step tool use that frontier models don't:

- **Berkeley Function Calling Leaderboard (BFCL)** consistently ranks Claude Sonnet/Opus and GPT-4 family at top; 70B open-weight models (Llama 3.1 70B, Qwen 72B, Mistral) typically 10-30% below frontier on complex multi-step chains
- **Multi-step error compounding**: if each step is 90% reliable, 5-step chains are ~59% reliable. Frontier maintains 95%+ per-step in many cases; local 70B doesn't
- **Schema adherence** (producing valid JSON for tool calls) remains surprisingly unreliable at 70B scale
- **Recovery from tool errors** (when an API returns unexpected) widens the gap further

For Bob's agentic actions specifically — high-stakes operations where mistakes cost real money, trust, and member logistics — **10-30% error rates are not viable**. Per `project_temporal_urgency.md` and `project_bob_as_social_infrastructure.md`, Bob's role is community-trust infrastructure; agentic unreliability undermines the mission directly.

Examples of unacceptable failure modes if Bob runs agentic actions on local 70B:
- Booking the wrong hotel for a 50-member chapter retreat
- Sending member contact info to the wrong recipient (privacy violation per `project_dm_privacy_structural.md`)
- Missing contest registration deadlines
- Calendar coordination errors causing missed rehearsals

**The frontier-API path isn't cost-optimized; it's quality-mandated.** Local models genuinely cannot do agentic actions reliably enough to ship in Bob's mission context. The privacy boundary (pseudonymization) is what makes the frontier-API path shippable while preserving member data sovereignty; the quality gap is what makes the frontier-API path required in the first place.

This argument is just as load-bearing as the cost case. **Even if local 70B were free, you'd still want frontier for agentic mode**, because the alternative is an unreliable agent.

The gap is narrowing slowly — research is improving function-calling capability at the open-weight tier — but the gap is real enough today that the hybrid commitment is the architecturally honest answer rather than a temporary workaround.

## Economics

For Bob specifically (BHS-scale, ~100-1000 active members during early V1):

| Component | Monthly cost (rough order) |
|---|---|
| **Chat tier**: rented H100 (24/7) for 32B + memex + persistent cache | $1.5K-3K |
| **Agentic tier**: Anthropic Claude API for ~10% of message volume | $0.3K-1K (scales with usage) |
| **AgentOS hosting**: small CPU server for orchestration + pseudonymization layer | $0.1K-0.3K |
| **Storage**: corpus + member data + audit logs | $0.05K-0.2K |
| **Total monthly** | **~$2K-4.5K** at BHS scale |

Compare to all-local-70B path:
- 70B model on A100-class GPU 24/7: $3K-9K/month
- Plus quality tradeoff (70B local doesn't match Claude on complex agentic work)
- Plus capex for owning hardware if going that route

**Hybrid is cheaper AND higher-quality** for the agentic mode specifically. The chat-mode cost dominates (you're paying for the H100 24/7 regardless of volume); the agentic-mode marginal cost is low (you only pay frontier API for the 10% of requests that need it).

As scale grows past BHS (memex-as-platform commercial extensions per `project_memex_as_platform.md`), the chat-tier cost amortizes across more customers; the agentic-tier cost scales linearly with volume but stays small relative to chat infrastructure.

## Account-takeover is the real soft-launch threat (added 2026-05-31)

Surfaced via cortex `/btw` 2026-05-31: at soft launch, the more important threat is not a *curated member* attempting prompt injection (audience is hand-picked, social cost is high, attack is unlikely), but an *attacker who has compromised a curated member's account*. The audience is curated; the audience-credentials are not.

This matters specifically for the hybrid-serving architecture because **the pseudonymization boundary protects the frontier provider from seeing member data, but a compromised account is operating from inside AgentOS as the real member.** Once authenticated as Pete Neushul, the attacker:

- Issues conversational requests that AgentOS de-pseudonymizes back to real member data
- Triggers agentic tool calls (booking, messaging, payment) that execute against the real member's identity and entitlements
- Reads Pete's Bob-chats (private candor surface per `project_bob_chat_privacy.md`)
- Triggers the friction-detector pattern (`project_bob_friction_detector.md`) with fabricated friction, potentially looping operator-Bob into adversary-coordinated DMs

**Pseudonymization does nothing against this threat.** It's a frontier-side protection, not a credential-side one. The threat lives entirely on AgentOS's authenticated-session side.

### What this implies for the architecture

The hybrid-serving architecture must compose with credential-side and tool-side defenses that don't rely on dragnet:

1. **2FA / WebAuthn at the RingHub auth layer** — load-bearing for soft launch even with curated audience. SMS-2FA is insufficient (SIM-swap is a real attack on senior demographic); prefer WebAuthn / authenticator app.
2. **High-stakes tools require explicit confirmation** — per `project_agent_correctness_architecture.md`, dry-run + confirmation pattern for any tool that spends money, sends messages externally, or reveals identity. Confirmation goes via a *separate* trust path (email, push notification, second device) that an attacker controlling the chat session doesn't automatically control.
3. **Tool-level invariants enforce damage limits** — daily transaction limits, recipient whitelists, irreversibility checks. An attacker with one compromised account should be unable to do unbounded damage.
4. **Audit + reversibility** — every agentic action is logged at the AgentOS pseudonymization boundary; reversible actions stay reversible long enough that anomaly detection has time to catch them.
5. **Bob-chat candor protection survives account takeover** — even if attacker reads Pete's prior Bob-chats, the candor protection's structural promise is "Bob never surfaces these publicly" (per `project_bob_chat_privacy.md`); the attacker reading-only doesn't break the promise. But the *write* side matters more: attacker prompting Bob to "remember I said X" could poison Pete's future Bob-context. Defense: per-session candor isolation; conversation rewrites require step-up auth.

### What this implies for the road to launch

The pseudonymization boundary is necessary but not sufficient for soft-launch privacy. The credential-side defenses (2FA, step-up auth, tool-level invariants) are also load-bearing **before** dragnet ships, not after. This is part of why soft-launch is not gated on dragnet specifically (per `project_road_to_launch.md` and `project_audit_timing.md`) — dragnet defends against adversarial *content*, not adversarial *credentials*. Different threats; different defenses; both required.

### Composition with cross-Claude review

The cross-Claude review pattern (per `project_agent_correctness_architecture.md`) gains additional weight here. A second Claude instance asked to review the *threat model* of a proposed agentic action — explicitly looking for account-takeover failure modes — catches credential-side threats that single-shot reasoning misses. The pattern composes: cross-Claude verifies the action; pseudonymization protects the frontier from data; tool invariants bound the damage; audit makes recovery possible.

### Composition with no-camouflage

Per `feedback_no_camouflage.md`: members must be told honestly that 2FA matters because account-takeover is the real threat. Not "we need extra security because reasons" — but "we've designed the privacy architecture against frontier providers, but if someone gets into your account they're inside the trust boundary; 2FA is the thing that keeps the trust boundary intact." Honest disclosure that 2FA isn't theater.

### The phrase to remember (for this threat)

> *The audience is curated; the audience-credentials are not. Pseudonymization protects the frontier provider; it does nothing against a compromised account operating from inside AgentOS as the real member. Credential-side defenses (2FA, step-up auth, tool-level invariants, audit + reversibility) are required for soft launch independent of dragnet.*

---

## Friction modes to acknowledge

The architecture has real edge cases that need design:

### Pseudonymization works well for structured fields, less well for content

Easy to pseudonymize:
- Names, emails, phone numbers, member IDs
- Venue names, dates, locations
- Quartet names, chorus names, district names
- Anything that's a discrete identifier

Hard to pseudonymize:
- Chat message content ("I've been struggling with depression and my doctor recommended...")
- Medical details, personal stories, oral history captures
- Anything where the content itself is inherently identifying

**Fallback for content**: don't forward verbatim. **Summarize locally first** via a smaller local model, then forward the summary. The frontier reasons about an abstracted version of the content; specifics never leave AgentOS.

This composes with the `project_unified_memory_architecture.md` auxiliary-summarization pattern. The auxiliary model (could be ternary-rs, eventually) does local pseudonymizing summarization; only the summary reaches the frontier.

### Tool implementations have to be careful

If tool implementations make outbound calls that re-introduce real data (e.g., calling a venue's API with the real member's name), the privacy boundary leaks at the tool layer. **Tools should be audited for the same privacy invariants as the frontier boundary.**

This is the natural extension of the principle: **every system boundary where real data could exit needs the pseudonymization discipline.** AgentOS is the central enforcement point, but tools are extensions of it.

### Audit log granularity vs cost

Per-request audit logs at the pseudonymization boundary are valuable for compliance but accumulate quickly. Need retention policies that balance compliance needs against storage cost. Probably: full logs for 30-90 days; summarized for compliance reporting; archived to cold storage thereafter.

### Pseudonym stability across sessions

If `member_a8f3` means Pete Neushul in session 1 and `member_b2d4` in session 2, the frontier can't build patterns across Pete's interactions. **This is the right behavior for chat privacy** (per `project_bob_chat_privacy.md` candor protection) but breaks if you want the frontier to maintain context across sessions about the same member.

Resolution: **session-scoped pseudonyms**, not persistent. Each agentic task gets its own opaque token namespace. The frontier reasons within one task at a time; cross-task pattern matching is intentionally blocked.

## Implications for cortex's roadmap

This architecture commits cortex to several near-term priorities:

| Cortex capability | Priority shift | Reason |
|---|---|---|
| **PagedAttention** | Up (load-bearing) | Multiple concurrent chat conversations on one H100 needs efficient KV memory sharing; shared-prefix CoW alone justifies it for prefix-heavy concurrent chats |
| **Continuous batching** | Up (load-bearing) | Same — multi-tenant chat serving on H100 needs request-level batching |
| **Persistent KV cache** | Already pinned (per `project_unified_memory_architecture.md`) | Becomes the *core* of the chat-tier value proposition; each member's conversation persists across turns |
| **Flash attention** | Up (relevant) | Tonight's flash-attention experiment relevant on tensor-core hardware (H100) where FA actually wins |
| **Pseudonymization shims** (new) | Add to roadmap | AgentOS calls cortex; cortex needs awareness of pseudonymized inputs; some shim logic at the cortex boundary for token resolution efficiency |

The chat-tier deployment (H100 + 32B local + memex + persistent cache) is essentially the v2/v3 cortex deployment target this pin makes explicit. **Cortex's perf work has the right target.**

## How this composes with other project pins

This pin is strategic-scope; it descends multiple specific commitments:

| Existing pin | How this composes |
|---|---|
| `project_dm_privacy_structural.md` | Same principle (privacy by data shape) applied at inter-system boundary |
| `project_pallet_rack_test.md` | Cross-context leak prevention extended to AgentOS↔frontier boundary |
| `project_bob_chat_privacy.md` | Main-street test for candor — pseudonymization is the structural mechanism that protects candor against frontier-side leak |
| `project_training_vs_retrieval_substrate.md` | Content separation principle — what reaches the frontier doesn't enter frontier-side training (which AgentOS can't control); content separation prevents the issue |
| `project_multi_tenant_readiness.md` | Multi-tenant primitives now have a strategic narrative attached (privacy boundary is the differentiator) |
| `project_agentos_topology.md` | Reframes AgentOS-as-platform identity |
| `project_agentos_api_contract.md` | HTTP+SSE contract carries pseudonymized data; resolution happens at AgentOS layer |
| `project_bob_outbound_api.md` | Async agentic responses use the outbound push path; pseudonymization applies to those too |
| `project_unified_memory_architecture.md` | Memex retrieval can happen with pseudonyms (memex shards are per-member; queries are local) |
| `project_who_says_problem.md` | Pseudonymization at the boundary structurally prevents the frontier from learning attribution |
| `project_training_time_representation.md` (meta-pin 2026-05-28) | Privacy commitment is a *structural architectural decision*, not a post-hoc filter — same shape as the meta-principle |
| `feedback_no_camouflage.md` | Honest disclosure to members: *"your data never leaves our infrastructure"* is structurally true, not a marketing claim |

The pin sits high in the architectural hierarchy because it shapes design decisions across many subsystems simultaneously.

## What changes operationally for the project

A few practical implications worth holding:

1. **Soft launch path**: For BHS soft launch (invited audience), the agentic tier might not even be needed — chat-only Bob handles introductions, retrieval, casual conversation. Frontier API integration becomes V1 work, not soft-launch work.

2. **BHS partnership pitch**: This architecture is a much stronger pitch than "we have AI for your members." It's *"member data lives in your control; frontier APIs we use never see your members' identities or messages."* Per `project_bhs_partnership_approach.md` and Pete's strategic advice — this is the kind of capability differentiator that lands.

3. **Multi-tenant story**: When memex-as-platform expands beyond BHS (per `project_memex_as_platform.md`), the pseudonymization boundary travels with the platform. Every tenant gets the same privacy invariants. This is *the* sales differentiator for the platform extension.

4. **dragnet integration**: dragnet (adversarial input classifier) sits at the AgentOS pseudonymization layer naturally. Pseudonymize first; then dragnet checks for adversarial patterns; then forward to frontier. Privacy + adversarial-defense compose cleanly.

5. **Audit posture for the wiz-kid Rust audit** (per `project_audit_timing.md`): the pseudonymization boundary becomes a major audit surface. Good auditors will appreciate structurally enforced invariants over policy-stated ones.

## The strategic claim worth making out loud

> ***AgentOS's value proposition isn't "agent runtime." It's "the privacy + vendor-abstraction + policy-enforcement layer that makes AI applications shippable in trust-sensitive contexts." Real member data lives on the AgentOS side; frontier APIs see only pseudonyms. Apps swap models in one config; audit trails are structural; policy enforcement is centralized. This is capability frontier vendors structurally cannot match — because it depends on AgentOS being independent of them.***

This is the kind of strategic claim that justifies the project's entire investment in self-hosting and substrate ownership. The project isn't building AgentOS because hosting is fun; it's building AgentOS because hosting is the lever that delivers privacy-by-architecture, vendor-independence, and policy-enforcement — capabilities that frontier vendors can't provide and standalone apps can't build.

## The phrase to remember

> *Real data never leaves AgentOS. Frontier LLM reasons over opaque tokens. Tool implementations live on AgentOS side. Privacy by data shape, not policy. Vendor abstraction follows naturally. This is the strategic positioning that makes the project's substrate-ownership commitments pay off.*

Plus the analogy:

> *Same shape as payment-processor tokenization — frontier sees `member_a8f3` the way merchants see `pm_4eC39Hq...`. Sensitive data lives in one architecturally-isolated place. PCI compliance becomes tractable because of this; AI compliance for member-data-sensitive contexts becomes tractable for the same reason.*

Plus the strategic reframe:

> *AgentOS is not an orchestration runtime. AgentOS is the privacy + vendor-abstraction + policy-enforcement layer for AI applications. The orchestration is a feature; the trust boundary is the product.*
