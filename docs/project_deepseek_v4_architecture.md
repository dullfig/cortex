---
name: deepseek-v4-architecture
description: "Verified architecture of DeepSeek-V4-Pro (read from the actual technical-report PDF 2026-06-13, post-cutoff, not reconstructed from memory). Key innovations relevant to the project: (1) HYBRID ATTENTION — Compressed Sparse Attention (CSA: compresses KV along sequence to 1/m, then DeepSeek Sparse Attention via a 'lightning indexer' that scores compressed blocks and top-k selects, plus a sliding-window branch for local detail) + Heavily Compressed Attention (HCA: more aggressive compression, dense attention). Result: '27% of single-token inference FLOPs and 10% of KV cache' vs V3.2 at 1M context. (2) PARTIAL RoPE with a relative-position countermeasure — RoPE on last 64 dims only; apply RoPE with position −i to attention outputs so they carry RELATIVE (not absolute) position — fixes the compression-breaks-RoPE 'mangling' problem. (3) mHC (Manifold-Constrained Hyper-Connections) — strengthens residuals, decouples residual width from hidden size; possible Bar-2 drift-control hypothesis (UNCONFIRMED). (4) MIXED-PRECISION KV — BF16 for RoPE dims, FP8 for the rest, FP4 for indexer QK → the dimension-aware-precision lesson (don't quantize uniformly; protect positional dims) is ACTIONABLE NOW for polar+QJL. (5) ON-DISK KV cache storage with shared-prefix reuse = validated virtualized-memex / two-tier source-tier pattern. MoE 1.6T total / 49B activated, MTP, Muon optimizer, 32T training tokens, 1M context, MIT license. Attention compression is ORTHOGONAL to MoE/param-count (attention layers vs FFN layers) → liftable to smaller scale, but TRAINED-IN not boltable. Daniel's point: adopting this means implementing MLA/CSA + MoE in cortex — which is a vCortex-from-initialization decision (per `project_vcortex_strategy.md`), NOT a cortex retrofit (death-by-retrofit risk). The two-tier memory architecture (`project_two_tier_memory.md`) lets cortex SHIP without it. Adopt techniques (not necessarily weights) for clean provenance in regulated verticals."
metadata:
  node_type: memory
  type: reference
  originSessionId: 0c8cbcaa-2244-408f-a2b3-9c792cfc2a2a
---

Grounded 2026-06-13 by reading the actual DeepSeek-V4 technical-report PDF (downloaded from the HuggingFace card `deepseek-ai/DeepSeek-V4-Pro`, extracted via pypdf, 58 pages, 173K chars). This is post-my-cutoff (references "Xie et al., 2026," benchmarks vs "GPT-5.4, Gemini-3.1-Pro, Claude Opus 4.5"), so the details below are quoted from the report, NOT reconstructed from prior knowledge of V2/V3. Where the model card alone was ambiguous (especially RoPE), the PDF resolved it.

## Verified architecture (quotes from the technical report)

### Hybrid attention: CSA + HCA

> *"We design a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA). CSA compresses the KV caches along the sequence dimension and then performs DeepSeek Sparse Attention (DSA), whereas HCA applies more aggressive compression to the KV caches but keeps dense attention."*

**CSA mechanism** (from the report's Figure 3 + text):
1. Compress KV entries along sequence dimension to 1/m (a token-level compressor)
2. **Lightning Indexer for Sparse Selection**: compute index scores between the query token and each preceding compressed block; a **top-k selector** retains only the relevant compressed KV entries for core attention
3. A small set of **sliding-window KV entries** is combined with the selected compressed entries to preserve local fine-grained dependencies

**HCA**: more aggressive compression but keeps dense attention (no selection step).

**The headline efficiency number**:
> *"In the 1M-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2."*

### Partial RoPE with the relative-position countermeasure (the "RoPE mangling fix")

This is the mechanism Daniel asked about and the card didn't confirm — the PDF does:

> *"For both CSA and HCA, we partially employ the Rotary Positional Embedding (RoPE) to the attention queries, KV entries, and the core attention outputs. To be specific, for each query vector and KV entry vector used in CSA and HCA, we apply RoPE to its last 64 dimensions. Since the KV entries serve as both attention keys and values, the naive core attention outputs will carry absolute position embeddings... As a countermeasure, we also apply RoPE with position −i on the last 64 dimensions of each [output]. In this way, the output of the core attention will also carry relative position embeddings."*

The compression makes attention outputs leak *absolute* position (the mangling). The fix: apply RoPE with position −i to the outputs, canceling the absolute leakage and recovering *relative* positioning. A real, specific fix to the compression-breaks-RoPE problem. Partial (last 64 dims only) keeps most dimensions position-free for the compressed latent.

### mHC — Manifold-Constrained Hyper-Connections

> *"DeepSeek-V4 series incorporate Manifold-Constrained Hyper-Connections (mHC) to strengthen the conventional residual connections between adjacent Transformer blocks... HC decouples the residual width from the actual hidden size, offering a complementary scaling axis with minimal computational overhead."*

A manifold-constrained version of Hyper-Connections (cites Zhu et al. 2025, Xie et al. 2026). Strengthens residuals; decouples residual width from hidden size for signal-propagation stability across layers. **Possible relevance to the project's Bar 2 problem** (per-layer drift): if mHC constrains residuals to stay on a valid manifold across layers, it may bound the compounding drift that killed polar Bar 2. **This is a hypothesis to investigate, NOT a confirmed link** — the report frames mHC as training-stability + capacity, not explicitly as inference-drift control.

### Mixed-precision KV — the dimension-aware-precision lesson (ACTIONABLE NOW)

> *"We adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision."*

**This is a direct, actionable lesson for the project's polar+QJL work, independent of whether the project adopts any DeepSeek architecture.** DeepSeek empirically found you *cannot* uniformly quantize the KV — the RoPE/positional dimensions need higher precision (BF16) while the rest tolerate FP8. This strongly suggests *why* polar's uniform quantization hit Bar 2 trouble: **uniform quantization over dimensions that have non-uniform precision requirements.** The lesson: the project's quantization should be **dimension-aware** — protect positional dimensions at higher precision, quantize the rest harder. Apply this to polar+QJL now.

### On-disk KV cache storage = validated virtualized-memex / two-tier source tier

> *"When serving DeepSeek-V4, we leverage an on-disk KV cache storage mechanism to eliminate repeated prefilling for shared-prefix requests. For the compressed KV entries in CSA/HCA... we simply store all of the compressed KV entries to the disk. When a request hits a stored prefix, we read and reuse the compressed KV entries."*

Plus three eviction strategies for the uncompressed sliding-window entries (Full caching / Periodic checkpointing / Zero caching) — explicit storage-vs-recompute trade-offs.

**This validates two project architectures at frontier scale**: (1) the virtualized-memex multi-tier storage hierarchy (hot VRAM / cold disk + index) discussed for memex; (2) the two-tier memory source tier (`project_two_tier_memory.md`) — compressed index hot, source on disk, read on hit. The project's version differs by storing *lossless text* as the truth (DeepSeek stores compressed KV); but the storage-hierarchy *shape* is validated and shipping at 1M context.

### Other verified specs

- **MoE**: DeepSeekMoE for FFN layers; 1.6T total params / 49B activated
- **MTP**: Multi-Token Prediction modules (inherited from V3, unchanged)
- **Optimizer**: Muon (Jordan et al. 2024, Liu et al. 2025) — "faster convergence and improved training stability"
- **Training**: 32T tokens; **context length 1M**; FP4 (MoE experts) + FP8 (most) mixed precision
- **License**: MIT
- **Benchmark positioning**: internal eval — "DeepSeek-V4-Pro-Max outperforms Claude Sonnet 4.5 and approaches the level of Opus 4.5"; "trails state-of-the-art frontier models [GPT-5.4, Gemini-3.1-Pro] by approximately 3 to 6 months"

## What's portable, what's tied to scale

**Attention compression is ORTHOGONAL to the MoE/param-count.** The report is explicit: *"hybrid CSA/HCA for attention layers, DeepSeekMoE for feed-forward layers."* Attention compression operates on the attention computation; the 1.6T params live in the FFN/MoE. Different axes.

So the attention innovations (CSA, HCA, lightning indexer, partial RoPE + countermeasure, mHC) are **scale-independent and liftable to a smaller model** — BUT they are **trained-in, not boltable.** The lightning indexer's QK projections are learned; there's FP4 QAT for the indexer path. Per `project_training_time_representation.md`: "lift and use independently" means "implement the technique in your own training at your chosen scale," not "extract a module and drop it onto a pretrained dense model."

## Daniel's point: implementing MLA + MoE in cortex → this is a vCortex decision

Daniel 2026-06-13: *"we would have to implement MLA and MoE in cortex. Which may not be a bad idea in the big scheme of things."*

Correct that adopting DeepSeek's innovations requires cortex to have MLA/CSA-style compressed attention AND MoE FFN — neither of which current cortex (dense, standard attention, Qwen-3B-class) has. And correct that it "may not be a bad idea." But the **right place to do it is vCortex, not a cortex retrofit:**

- Per `project_vcortex_strategy.md`: cortex is to be declared "finished" post-launch; vCortex is the clean rewrite that inherits accumulated learnings *from initialization*. Implementing MLA/CSA + MoE + mHC + partial-RoPE is **exactly the kind of architectural commitment that belongs in vCortex-from-initialization**, not retrofitted onto cortex. Retrofitting cortex with MoE + compressed-attention is the death-by-retrofit move the vCortex strategy specifically warns against.
- Per `project_two_tier_memory.md`: the two-tier memory architecture lets cortex **ship without** MLA/MoE. Cortex stays small + dense; memex's two-tier (synopsis + lossless source) handles the heavy lifting; Bar 2 is sidestepped. So MLA/MoE is NOT a launch blocker — it's a vCortex-tier capability improvement.

**The clean sequencing**: ship on current cortex + two-tier memory (no MLA/MoE needed) → declare cortex finished post-launch → vCortex implements CSA/HCA + MoE + mHC + partial-RoPE from initialization at the project's chosen scale. The "big scheme of things" payoff Daniel intuits is real, and its home is vCortex.

**Scale note**: V4-Pro is 1.6T params (big-iron, multi-GPU). The project would implement the *techniques* at its own much smaller scale (a small MoE, compressed attention at cortex-class param counts). The techniques are scale-independent; the project does NOT need to run 1.6T params to benefit from CSA/HCA/mHC/partial-RoPE.

## Provenance + weights-vs-techniques for the verticals

- **MIT license** removes legal friction entirely — self-host, commercial use, modify, all permitted. Fine for BHS.
- **Chinese-origin weights** remain a *security-review* consideration for the defense-manufacturing / CMMC / ITAR verticals (per the "no happy asking panda" discussion), even MIT and self-hosted. Open weights run locally is far better than an API, but a defense customer's security team may still flag origin.
- **The clean answer for regulated verticals**: vCortex implements the *published techniques* (CSA, HCA, mHC, partial RoPE, MoE routing) with *project-trained weights* → clean provenance + designed-in BitNet + trinity integration. Adopt the ideas, not necessarily the weights, where origin matters.

So DeepSeek V4 is best read as a **research gift** — published, MIT, validated-at-scale innovations that map onto memex's and cortex's needs — more than a model to deploy directly. Use weights for BHS/prototyping if useful; adopt techniques into vCortex for the regulated path.

## What to adopt, and when

| Innovation | Adopt? | When | Notes |
|---|---|---|---|
| **Dimension-aware KV precision** (protect RoPE dims) | YES | NOW | Actionable lesson for polar+QJL; don't quantize uniformly |
| **On-disk KV storage hierarchy** | YES (shape) | v1/v2 | Validates two-tier source tier + virtualized memex; project stores lossless text as truth |
| **CSA / lightning indexer** | Adopt technique | vCortex | Maps to two-tier synopsis selection; trained-in |
| **HCA** | Consider | vCortex | More aggressive compression option |
| **Partial RoPE + −i countermeasure** | Adopt technique | vCortex | The RoPE-mangling fix; needed if vCortex uses compressed attention |
| **mHC** | Investigate | vCortex (research) | Possible Bar-2 drift control — UNCONFIRMED; test the hypothesis |
| **MoE FFN** | Consider | vCortex | Capacity without proportional inference cost; small MoE at project scale |
| **Muon optimizer** | Consider | vCortex training | Faster convergence; training-infra choice |
| **The actual weights** | Maybe (BHS only) | Prototyping | MIT; fine for BHS; provenance issue for defense verticals |

## Composition with project pins

| Pin | Composition |
|---|---|
| `project_two_tier_memory.md` | Lightning indexer = synopsis selection; on-disk KV = source tier; the architecture that lets cortex ship WITHOUT needing V4's attention |
| `project_compression_substrate_quality_bar.md` | Dimension-aware precision explains the polar Bar 2 failure (uniform quant over non-uniform-precision dims); mHC is a Bar-2 hypothesis |
| `project_vcortex_strategy.md` | CSA/HCA + MoE + mHC + partial-RoPE are vCortex-from-initialization adoptions, NOT cortex retrofits |
| `project_training_time_representation.md` | The techniques are trained-in, not boltable; commit at training time |
| `project_encoder_fine_tuning_priority.md` | The synopsis/encoder model could adopt compressed-attention techniques |
| `project_unified_memory_architecture.md` | On-disk KV validates the storage-hierarchy thinking; bias-attention path still demoted per two-tier |
| `project_zynqberry_bitnet_memex.md` | The appliance/FPGA tier does NOT use V4 (too big); adopts techniques at small scale via vCortex |
| `project_tsunami_serving_architecture.md` | Big-iron tier could use V4-class weights or architecture; appliance tier adopts techniques |
| `feedback_no_camouflage.md` | Read directly from PDF rather than confabulating from memory — honest about what's verified vs recollected |
| `feedback_ai_overestimates_from_training_corpus.md` | The post-cutoff details required actual document reading, not corpus-narrative reconstruction |

## Honesty note on method

This pin's details come from reading the actual PDF (extracted text searched for RoPE/CSA/HCA/mHC/indexer/on-disk terms, quotes pulled verbatim), NOT from my prior knowledge of DeepSeek V2/V3. The model card alone was insufficient (it didn't confirm the RoPE mechanism); the PDF resolved it. Per `feedback_no_camouflage.md` and the Fable-5-rumor discipline: when asked about post-cutoff releases, read the source rather than confabulate familiarity. The extracted text was saved to `C:\Users\danu\dsv4.txt` during this session (cleanup-eligible).

## The phrases to remember

> *DeepSeek V4: hybrid attention (CSA compresses+sparse-selects via lightning indexer; HCA compresses+dense) → 27% FLOPs, 10% KV cache at 1M. Partial RoPE with a position-−i countermeasure fixes compression-breaks-RoPE. mHC strengthens residuals (possible Bar-2 help, unconfirmed). Mixed-precision KV (BF16 RoPE dims, FP8 rest) = the dimension-aware-precision lesson for polar. On-disk KV storage = validated two-tier source tier. MoE 1.6T/49B, MIT license.*

Plus the adoption strategy:

> *Attention compression is orthogonal to MoE/param-count → liftable to small scale, but trained-in not boltable. Implementing MLA/CSA + MoE is a vCortex-from-initialization decision, NOT a cortex retrofit (death-by-retrofit risk). Two-tier memory lets cortex ship without it. Adopt techniques (not weights) for clean provenance in regulated verticals.*

Plus the immediately actionable item:

> *Dimension-aware precision: don't quantize the KV uniformly. DeepSeek keeps RoPE/positional dims at BF16 and quantizes the rest to FP8. This likely explains polar's Bar 2 failure (uniform quant over non-uniform-precision dims). Apply dimension-aware precision to polar+QJL now — independent of any architecture adoption.*

## Related pins

- `project_two_tier_memory.md` — lightning indexer = synopsis selection; on-disk KV = source tier; cortex ships without V4 attention
- `project_compression_substrate_quality_bar.md` — dimension-aware precision explains Bar 2 failure; mHC as Bar-2 hypothesis
- `project_vcortex_strategy.md` — CSA/HCA/MoE/mHC are vCortex-from-initialization, not cortex retrofit
- `project_training_time_representation.md` — techniques are trained-in, not boltable
- `project_encoder_fine_tuning_priority.md` — synopsis/encoder model could adopt compressed attention
- `project_unified_memory_architecture.md` — on-disk KV validates storage hierarchy
- `project_zynqberry_bitnet_memex.md` — appliance tier adopts techniques at small scale, doesn't use V4 weights
- `project_tsunami_serving_architecture.md` — big-iron tier vs appliance tier
- `feedback_no_camouflage.md` — read the PDF, didn't confabulate
- `feedback_ai_overestimates_from_training_corpus.md` — post-cutoff required document reading not memory
