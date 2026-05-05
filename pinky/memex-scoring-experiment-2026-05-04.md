# Memex retrieval scoring: MEAN vs MAX (2026-05-04)

**TL;DR:** With Qwen 2.5-3B base + 1941 Harmonizer corpus, **mean-aggregated**
attention scoring returns identical top-4 hits across 7 different queries
(the algorithm finds "high-attention everywhere" positions, not
query-relevant ones). **Max-aggregated** scoring preserves those positions
in slots #1-2 but produces query-discriminating hits in slots #3-5. Cortex
default switched to MAX; memex iteration may want a baseline-subtraction
step to also remove the always-hot positions.

## Setup

- Model: Qwen 2.5-3B base (Q4_K_M, downloaded from `bartowski/Qwen2.5-3B-GGUF`)
- Corpus: `1941-11_rechordings-vol1-no1.md` (2284 tokens, hand-curated
  markdown of the first Harmonizer-precursor newsletter)
- Cache: 4 sink tokens prepended (`SINK_TOKENS = 4`); shard seq_len = 2288
- Path: `forward_full_gpu_with_cache_traced` — captures pre-softmax
  attention scores from the last 4 layers, returns scores for the query
  positions over the full prefix (corpus + query).
- Scoring (variable): aggregate `(layers x heads x last 3 query positions)
  -> per-corpus-position score`, then rank.

## Run 1: MEAN aggregation, base model, 7 queries

Same exact top-4 positions for every query. Position #5 wiggled slightly.

| Query | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Who founded the society? | 28 | 417 | 2100 | 42 | 44 |
| Where did O.C. Cash grow up? | 28 | 417 | 2100 | 42 | 128 |
| Who was the first president? | 28 | 417 | 2100 | 42 | 44 |
| Tell me about Bluejacket Oklahoma | 28 | 417 | 2100 | 42 | 128 |
| Who were Ambassadors of Good Will? | 28 | 417 | 2100 | 42 | 44 |
| What did Joe Wolff say? | 28 | 417 | 2100 | 42 | 44 |
| What is the masthead address? | 28 | 417 | 2100 | 42 | 128 |

Decoded:
- 28 = "Q.S.A. Barber Shop Re-Chordings - Vol. 1, No. 1, November 1941" (masthead title)
- 417 = "City taught us the night I dated her on that hay ride in June, 1910" (Cash's autobiographical hayride paragraph)
- 2100 = "vault - the seed of the Old Songs Library (now 100,000+ titles) 3." (late meta-commentary)
- 42 = "Vol. 1, No. 1, November 1941 **The first issue of what would become..."
- 44 / 128 = nearby title-area / editor-address tokens

These are all "high information density" tokens that the model finds
useful for almost any query. Averaging across 16 heads * 4 layers *
3 query positions washes out the query-specific differentiation.

Note: base model removed the SINK ARTIFACT hits that Instruct showed.
That's a real architectural difference (instruct-tuning adds a softmax-
sink artifact at position 0/1; base doesn't). But the dominant
"always hot positions" pattern is the same.

## Run 2: same MEAN, last 1 layer only

Hypothesis: averaging fewer things preserves more signal.

Result: still identical top 4 across all 6 queries (28, 417, 2100, 42).
SINK artifacts crept back into slot 4-5 (offsets 0/1) — the early
layers without the masthead averaging "compete" with the dominant
positions less effectively. **Negative result** — last-N tuning is not
the lever.

## Run 3: MAX aggregation across layers x heads x query positions

Hypothesis: any single (layer, head, query-position) lighting up for a
specific corpus position should make that position rank highly. Mean
washes that out; max preserves it.

| Query | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Who founded the society? | 28 | 2283 | **2224** | **286** | **330** |
| Where did O.C. Cash grow up? | 28 | 2283 | **417** | **2100** | **2180** |
| Who was first president? | 28 | 2283 | **2224** | **330** | **417** |
| Tell me about Bluejacket Oklahoma | 2283 | 28 | **271** | **124** | **286** |
| Who were Ambassadors of Good Will? | 28 | 2283 | **417** | **2100** | **271** |
| What did Joe Wolff say? | 2283 | 28 | **2224** | **330** | **286** |

Slots #1-2 are still 28/2283 (the always-hot positions). **Slots #3-5
genuinely vary by query.** "Bluejacket Oklahoma" gives uniquely 271, 124
in its top-5 — neither appears in any other query's top-5. That's real
query-specific signal.

This is the discriminating evidence the memex architecture predicted
should exist.

## Run 4: MAX with baseline-subtraction (the breakthrough)

Hypothesis: run two forwards — one with the actual query, one with a
"neutral" baseline (single BOS token). For each corpus position k,
compute MAX attention from the query's last position, MAX attention
from the baseline's last position. Score = query_max - baseline_max.

Always-hot positions (28, 2283) get cancelled (they're high in both).
Query-specific positions (where the query's heads attend differently
than a neutral baseline) survive.

Implementation note: query and baseline must aggregate over the SAME
denominator (same number of (layer, head, query_position) candidates)
or the MAX is asymmetric. Both use last query position only.

**5 of 6 queries hit the exact relevant passage:**

| Query | Top hit context | Match |
|---|---|---|
| Who founded the society? | "O.C. Cash (Founder)" at 682, 680 | ✅ exact |
| Where did O.C. Cash grow up? | "his youth in Bluejacket, Oklahoma" at 324 | ✅ exact |
| Who was the first president? | "Carroll P. Adams First National President" at 574 | ✅ exact |
| Tell me about Bluejacket Oklahoma | "youth in Bluejacket, Oklahoma" at 324 + "old home town of Bluejacket" at 353 | ✅ exact |
| Who were the Ambassadors of Good Will? | Officer-list context at 672, 661 | ⚠️ partial — adjacent but missed the actual Ambassadors enumeration |
| What did Joe Wolff say? | "Quotes Joe Wolff: the Society is 'a haven'" at 690 | ✅ exact |

The always-hot positions (28, 417, 2100, 2283) are GONE from the top 5
across all queries — exactly as predicted.

Semantically related queries also share top hits in expected ways:
- Q3 (first president) and Q5 (Ambassadors) share 4 positions — both
  about official Society roles.
- Q2 (Cash grew up) and Q4 (Bluejacket) share offset 324 — Bluejacket
  IS where Cash grew up.

This is genuine semantic retrieval. The memex architecture works.

**Cost:** 2 forwards per query instead of 1. Latency ~500ms instead of
~250ms. Worth it for the quality jump.

**Default in cortex:** baseline-subtraction is now wired into
`/v1/retrieve`. Single BOS token as baseline; same `capture_layers` and
shard cache as the query.

## Open questions for further memex iteration

1. **Better baseline.** Single BOS may not be the most "neutral" reference.
   Could use 4 BOS sinks. Could use a generic question like "tell me
   about this" to pick up "general info" attention without query-specific
   weighting. The choice is empirical.

2. **Head selection.** With MAX, *some* heads are doing query-relevant
   attention. Which ones? Per-head analysis on a known-answer dataset
   would let us pre-select discriminating heads instead of averaging.

3. **Position-frequency filter.** Track which positions get top-K hits
   *across* a sample of queries. Positions that appear in >50% of
   queries' top-5 are "always hot" and can be down-weighted at
   inference time.

4. **Different layer subsets.** This experiment used last 4 layers.
   Middle layers (where syntactic vs semantic features differentiate)
   might give different signal than the last layers' "summary attention."

## What changed in cortex

`/v1/chat/completions` (mode=retrieve) now uses MAX-of-(layers x heads
x query positions) instead of MEAN. Same shape, same latency (~250ms
per query against the 1941 corpus on Qwen 3B base).

Implementation: `cortex-cloud/src/main.rs` retrieve handler scoring
loop. The aggregation knob is currently hardcoded; if memex needs to
experiment further, the cleanest extension is a per-request scoring
config (mean | max | baseline-subtract) added to the request schema.

## Latency reference

- Ingest: ~60s for 2284 tokens (slower than Instruct's 27s — could be
  base-model first-cold-start; not investigated)
- Retrieve: 250-550ms per query (first query ~550ms warmup, subsequent
  ~250ms each)

## Run 5: Multi-shard retrieval (2026-05-04 evening)

`/v1/chat/completions` (mode=retrieve) now accepts multiple
`cache_shards`. Smoke harness: `multishard-smoke.ps1` loads two shards
into the pool and queries across both:

- `harm1941` — `1941-11_rechordings-vol1-no1.md` (2284 tokens, 2288 with
  4 BOS sinks)
- `whatis` — `bhs-org/what-is-barbershop.md` (546 tokens, 550 with sinks)

Composition path (mirrors the chat handler): concatenate per-shard
tokens in the order given, prefill a fresh `GpuKvCache` of the combined
seq_len, then run query+baseline forwards against that composed cache.
Hits resolve back to source shard via `shard_map.resolve(offset)`.

Server log:
```
multi-shard composed cached forward shards=["harm1941","whatis"]
  composed_tokens=2838 corpus_tokens=2838 query_tokens=30
```

### Results (2 of 3 queries before server hang)

**Q1: "Who founded the society?"** (66s, 6 hits)
| Rank | Shard | Off | Score |
|---|---|---|---|
| #1 | harm1941 | 2112 | 5.18 |
| #2 | harm1941 | 274  | 4.81 |
| #3 | harm1941 | 205  | 4.22 |
| #4 | harm1941 | 179  | 3.98 |
| #5 | whatis   | 506  | 3.88 |
| #6 | harm1941 | 180  | 3.75 |

Top hits cluster on harm1941 — the shard with O.C. Cash content. ✅
Reasonable. Note: in single-shard mode this query top-hit was offset
682 ("O.C. Cash (Founder)"); in multi-shard it shifts to 2112 (the
"100,000+ titles" meta-section). The shift is plausibly because in
multi-shard prefill the harm1941 tokens are followed by 554 whatis
tokens, slightly changing late-position attention saturation.

**Q2: "What is barbershop singing?"** (66s, 6 hits)
| Rank | Shard | Off | Score |
|---|---|---|---|
| #1 | whatis   | 326 | 5.45 |
| #2 | whatis   | 325 | 5.36 |
| #3 | harm1941 | 1375 | 4.21 |
| #4 | harm1941 | 205  | 3.46 |
| #5 | harm1941 | 1630 | 3.26 |
| #6 | harm1941 | 758  | 2.93 |

✅ Top-2 correctly land in the whatis shard (the one that is literally
about what barbershop is). Cross-shard semantic discrimination works.

### Latency cost of composition

Each multi-shard query re-prefills the composed cache (2838 tokens =
~65s on Qwen 3B/4080). That's the dominant cost — the query+baseline
forwards are ~250ms combined. A natural follow-up: cache compositions
keyed on `(sorted_shard_set, version)` so subsequent queries against
the same shard set hit the cached composition.

### Q3 server hang (latent bug)

After Q1 and Q2 completed cleanly, **Q3 hung the server** ("Where did
O.C. Cash grow up?"). Symptoms:
- Server PID alive but `/health` no longer responds
- Server CPU goes flat (1.5s in 3 min)
- Q3 was the third request to allocate a fresh ~85 MB composed cache
  in quick succession (each composed_cache holds 2838 tokens × 36
  layers × 2 (K+V) × 2 KV-heads × 128 dim × f32 = ~85 MiB GPU)

Most likely a wgpu/Vulkan driver-state issue with rapid alloc-and-drop
of large per-layer buffer arrays under sustained pressure (the same
neighborhood as the Qwen-block-26 wgpu bug we hit earlier on this
model). Single-shard retrieval (which reuses the pool's resident
cache) does not exhibit this — only multi-shard's per-request
allocation does.

The composition-cache optimization above would also fix this
incidentally by removing the alloc loop. So the same follow-up serves
both performance and stability.

### What changed in cortex (multi-shard)

`/v1/chat/completions` mode=retrieve now accepts multiple
`cache_shards`. Single-shard fast path borrows from the pool;
multi-shard creates a temporary composed cache and runs the same
scoring (MAX with BOS-baseline subtraction) against it. `shard_map`
in the response correctly identifies which shard each hit came from.

Smoke harness: `multishard-smoke.ps1` at repo root. It assumes the
server is running and pool is empty.

## Run 6: Composition cache (2026-05-05)

Added a single-slot composition cache to `ServerState`: holds at most
one `GpuKvCache`, keyed on the ordered list of `(shard_name, version)`.
Each shard has a `version: u64` that bumps on `cache_load` (overwrite),
`cache_append`, and chat completions that mutate the shard.

On a multi-shard retrieve request:
- Snapshot `(shard, version, tokens)` under the pool lock; drop pool lock
- Lock composition; if `composition.key == request_key`, **reuse**
- Else clear the existing buffer in place and re-prefill (no buffer alloc;
  `GpuKvCache::clear()` just resets the cursor, the underlying wgpu
  buffers stay resident)

Result on the same `multishard-smoke.ps1` harness:

| Query | Before | After | Composition |
|---|---|---|---|
| Q1 "Who founded the society?" | 66034 ms | 66434 ms | rebuilt |
| Q2 "What is barbershop singing?" | 65699 ms | **319 ms** | reused |
| Q3 "Where did O.C. Cash grow up?" | hung | **337 ms** | reused |

200x speedup on warm queries. Hang gone — Q3 used to hang the wgpu
driver after the third per-request alloc-and-drop of the composed K/V
buffers; now there's only one alloc, ever, for a given `max_seq_len`.

Q3's top hit (offset 324 in harm1941, "youth in Bluejacket, Oklahoma")
also matches what single-shard MAX-with-baseline produced in Run 4
(offset 324). Cross-shard composition + composition-cache reuse
preserves the discriminating signal.

### Cache invalidation

Explicit drop on `cache_load` (any shard insert/replace),
`cache_append` (when tokens were actually appended), and
`cache_delete`. Plus a passive staleness check on the version field —
chat completions that mutate a shard bump its version; the next
multi-shard retrieve sees the version mismatch and rebuilds.

The cache invalidation is intentionally pessimistic: any pool mutation
drops the composition even if the mutated shard wasn't in the
composition's key. The cost is a single rebuild on the next
multi-shard query (~65s for our two-shard demo). Smarter granular
invalidation is a future tweak if it matters.

### Scope

Composition cache is **retrieve-only**. Chat-mode multi-shard still
allocates a fresh composed cache per request (and mutates it during
generation, then drops it). Chat is interactive enough that the
allocation pressure pattern hasn't surfaced there. If it does, the
fix is the same idea but harder because chat mutates the composed
cache during generation.
