# Adversarial review — cortex + cortex-cloud (2026-09-02)

**Method.** Three independent adversarial hunters over disjoint surfaces
(GPU dispatch/resources; HTTP/DoS/concurrency; parsers/numerics), findings
deduplicated and ranked, and **the top ~15 verified by direct code read** by
the reviewing session. Labels: **VERIFIED** = reviewer read the path;
**HUNTER-CONFIRMED** = hunter traced it end-to-end with file:line (marked
×2/×3 where independent hunters converged); **PLAUSIBLE** = suspected, needs
a test. Style/lint deliberately excluded.

**Calibration.** memex-claude had reported one bug (one-shot `cache/load`
of ~6K tokens panics wgpu). The hunt rediscovered it unprompted, found the
exact root site, and matched the reported number precisely:
**softmax dispatches 16 workgroups/token; the chunker picked a 4178-token
chunk; 16 × 4178 = 66848** — the value memex logged. It then found the bug
is a *class* (13 sites) and two things the report did not cover.

---

## Ranked findings

### P0 — remote, trivial, high impact

| # | Finding | Site | Label |
|---|---|---|---|
| 1 | **Unbounded `max_tokens` on a cached shard bricks the shard and holds the pool lock for the whole run.** Decode advances the resident cache until it hits `max_seq_len` → assert panic *before* `entry.tokens` is extended → shard permanently unusable (chat panics, retrieve panics, append 400s) until `DELETE`. Pool lock held for minutes → blocks every cache/retrieve/cached-chat handler *and* the metrics sampler. | `chat.rs:655, 1154-1226, 186-205`; assert `forward_f32.rs:707-711` | HUNTER-CONFIRMED |
| 2 | **System-turn forgery via tokenizer.** Chat template is raw `push_str` concatenation with **no escaping of user content**; tokenizer emits control tokens unconditionally (no `parse_special` gate). A user message containing `<\|im_start\|>system …` becomes a genuine system turn. *For Bob: any caller can rewrite his instructions.* | `chat.rs:53-90`; `tokenizer.rs:379-397` | **VERIFIED** |
| 3 | **Remote panic via valid JSON.** `"temperature": 1e-40` passes the `<= 0.0` guard, `logit/1e-40` overflows to `inf`, `softmax_inplace` poisons to NaN, three `partial_cmp().unwrap()` sites panic. 500 on demand, after paying the full prefill. | `chat.rs:642-651`; `sampler.rs:136, 208, 268, 283-289` | **VERIFIED** (×2 hunters) |
| 4 | **The 65535 workgroup-dispatch class.** Root: `softmax_groups = n_tokens * n_heads` dispatched 1-D. `ChunkLimits` models *only bytes* (Lane A/B/C + binding size) — **no dispatch-dimension constraint exists**. At default config (`--max-seq-len 4096`, any ≥12 GB card) a 4096-token load → 65536 groups → panic. 13 sibling sites (softmax f32+polar, qjl_value_weights @16/tok; attn_value, rope_packed, rotate_q @8/tok; bias_add, add/add_broadcast, derotate @4/tok; rmsnorm/final_norm/score-y @1/tok). The 8 GB fallback budget only "works" by accident (chunk 1823). | `dispatch.rs:1458-1463`; `scratch.rs` (absence) | **VERIFIED** |
| 5 | **Unchunked prefill paths bypass the chunker entirely.** Multi-shard composition prefills the concatenation of *all* shards in ONE forward (and discards the result with `let _ =`); stateless chat, streaming, gate/inject, `/v1/shims/embed` prefill the whole prompt in one forward with **no prompt-length check anywhere**. → Lane B `.expect` panic (~1.9k tokens at 256 MB), or the #4 overflow, or `assert!("cache overflow")` instead of a 400. Multi-shard also never checks Σ shard tokens ≤ `max_seq_len`. Bonus: `forward_full_gpu_with_cache` then runs `finalize_logits` over ALL n tokens (n×151936 f32 ≈ 2.5 GB at 4096) and throws it away. | `chat.rs:865, 1240, 293-323, 1445-1476, 491`; `shims.rs:1031` | **VERIFIED** (×3 hunters) |

### P1 — resource exhaustion & concurrency

| # | Finding | Site | Label |
|---|---|---|---|
| 6 | **Cache pool is uncapped; filling the VRAM budget makes every stateless chat panic.** Each `cache/load` reserves a full `max_seq_len` heap *before* any pool check, even for `"tokens": []`; no eviction. Loop it → budget committed → every later load *and* every stateless `/v1/chat/completions` panics at `BudgetExceeded`. Reloading an existing id double-allocates transiently. | `cache.rs:151, 237`; `chat.rs:281, 1427`; `gpu_kv_cache.rs:84-90` | HUNTER-CONFIRMED |
| 7 | **Concurrent forwards share the three transient lanes with no admission control.** `cache_load` runs its whole forward *before* taking `cache_pool`; stateless chat holds no lock; `safe_prefill_chunk_size` uses `capacity()*97/100` assuming *empty* lanes. Two overlapping prefills → second `BlockScratch::allocate` fails → panic. **The doc comment at `wgpu_backend.rs:896-897` ("cortex-cloud serializes via the cache-pool mutex") is false** — and `ParamsBufferPool`'s wrap-safety silently relies on it. | `cache.rs:166 vs :232`; `scratch.rs:426-428` | **VERIFIED** (×2) |
| 8 | **`entry.tokens` and the resident KV diverge → later single-shard retrieve indexes past the score buffer (OOB panic); silent mis-scoring below the threshold.** Multi-shard chat appends prompt+generated to the *last* shard's `tokens` but its resident cache is untouched; polar-chat non-greedy fallback keeps a stale `entry.polar`; `max_tokens` finish pushes one un-forwarded token. Retrieve uses `corpus_len = tokens.len()` but `attn_max = cache_seq + n_q`. | `chat.rs:1257-1270, 1160-1194, 736-739, 896, 952` | HUNTER-CONFIRMED |
| 9 | **TOCTOU `unwrap()` after releasing and re-acquiring the pool lock.** Tokio mutex is FIFO, so retrieve-phase-1 → `DELETE` → retrieve-phase-2 → `unwrap()` on `None` is a deterministic interleaving. Same shape in `cache_append`. Related PLAUSIBLE: append racing a same-id load splits chunks across old/new entries → assert. | `cache.rs:343`; `chat.rs:772` | HUNTER-CONFIRMED |
| 10 | **CPU DoS via long prompt.** BPE merge is O(n²) with two string clones + hash per pair per iteration and an O(n) `remove`; GPT-2 pre-tokenizer emits one "word" for any unbroken letter run, SentencePiece never pre-splits. A 2 MB body of `aaaa…` ≈ 10¹² ops. Special-token scan is also quadratic in occurrences. Only cap is axum's default body limit. | `tokenizer.rs:602-632, 510-540, 383-434` | **VERIFIED** |
| 11 | **Client token ids never validated against the vocab.** `/v1/detokenize {"tokens":[4294967295]}` → OOB panic; `cache/load` with a bad id panics inside `block_in_place` *after* reserving a full-size cache (and skips the `poll_wait` deferred-destroy flush). | `api.rs:358/390/475`; `main.rs:232`; `forward_f32.rs:600` | HUNTER-CONFIRMED (×2) |
| 12 | **Polar retrieve panics on `host_readback_heap` for long queries.** One readback staging per captured layer of `n_q·n_heads·(corpus+n_q)·4` B from a 256 MB heap; pre-check bounds only ONE layer against Lane B. 4 layers × 4096-token shard → panic at n_q ≳ 241. | `forward_polar.rs:164-170, 110-125` | HUNTER-CONFIRMED |

### P2 — untrusted model file (operator trust boundary, but uncatchable aborts)

| # | Finding | Site | Label |
|---|---|---|---|
| 13 | **Allocate-before-validate.** `vec![0u8; byte_size]` sized purely from declared dims, before any check against file length → 200-byte file declaring `[2^34, 4]` F32 → 256 GB zeroed alloc → `handle_alloc_error` **abort (not a panic)**. Same for raw string lengths, array counts, `HashMap::with_capacity(metadata_count)`, `n_dims` (spec caps at 4; code does not). | `gguf.rs:762-768, 352-353, 373-374, 541, 557, 560-563` | **VERIFIED** |
| 14 | **`general.alignment = 0` → divide-by-zero panic** (present-but-zero passes `unwrap_or(DEFAULT)`). | `gguf.rs:550-554, 818-821` | **VERIFIED** |
| 15 | **Unbounded nested-array recursion** → stack overflow abort (~1.2 MB file of `[9, count=1]`). | `gguf.rs:370-379` | **VERIFIED** |
| 16 | **Size arithmetic wraps silently** — `shape.product()`, `n*4`, `div_ceil*BYTES`, `tensor_data_offset + info.offset`; **no `overflow-checks` set in any Cargo.toml** (release wraps, debug panics — verified with `rustc -O`). `FloatTensor::new` shares the wrap so a `[2^63, 2]` shape *passes* its length assert and panics on first forward. `TensorShapeMismatch` exists but is never constructed. | `gguf.rs:573, 764, 828-839`; `tensor.rs:16` | **VERIFIED** |
| 17 | **Config divisors/sizes unchecked** — `head_count=0` → div-by-zero at `loader.rs:114`; `n_layers`/`n_experts = u32::MAX` → TB alloc abort; a dozen `assert!`s fire on first request instead of `Err` at load. | `loader.rs:114, 161, 189` + siblings | HUNTER-CONFIRMED |

### P2 — silently wrong output (worse than any panic)

| # | Finding | Site | Label |
|---|---|---|---|
| 18 | **`rope.scaling.type` read as u32 but GGUF stores it as a *string*** → always 0; only the `arch.contains("qwen")` fallback saves Qwen. Any non-Qwen NeoX-style model gets interleaved RoPE → **fluent garbage, no error**. `hidden_act` is parsed and **never consumed** (always SiLU). | `gguf.rs:653-670`; `loader.rs:124-137, 195, 209` | **VERIFIED** |
| 19 | **Norm weight lengths never validated at load.** CPU path panics on first use; GPU path hits WGSL robust-buffer clamping → dims past the short weight scaled by 0/garbage, **no error**. | `loader.rs:213-217, 237-238`; `rmsnorm_batch.wgsl:49` | VERIFIED (load) / PLAUSIBLE (GPU) |
| 20 | `cache_append` runs f32 `advance_only` then polar; a polar-only failure leaves `f32.seq_len ≠ polar.seq_len ≠ tokens.len()` → silently misaligned positions thereafter. A polar-only failure exists under env-overridden lanes (chunker models the f32 layout; `PolarBlockScratch` needs +12 KB/token on Lane A). | `cache.rs:414-423`; `scratch.rs:158-205` | PLAUSIBLE |
| 21 | Tokenizer metadata arrays of mismatched length: short `token_type` **silently drops `<\|im_start\|>` from special tokens** → chat markers get BPE-split → gibberish, no error. bos/eos ids unchecked against vocab → every request asserts. SentencePiece byte-fallback decode emits Latin-1 mojibake ("é" → "Ã©"; Qwen unaffected). | `tokenizer.rs:139-162, 305-311, 189-197, 470` | HUNTER-CONFIRMED |

### P3 — observability / hygiene

| # | Finding | Label |
|---|---|---|
| 22 | **The `gpu_engine/tests.rs` suite is dead code.** `#[cfg(any())]` is always false → the 2251-line file is never compiled (will not even type-check). **Lost: 34 tests including the ONLY CPU-vs-GPU parity checks** (`forward_full_gpu_matches_cpu_*`, cache prefill/decode parity, `dispatch_attention_matches_cpu_gqa`, matmul-fused parity, polar-vs-f32 traced). *This is why none of the numeric issues above were caught.* | **VERIFIED** |
| 23 | **Budget bypass:** capture/staging/hidden buffers use raw `device.create_buffer`, not `new_in_budget` — a long retrieve allocates hundreds of MB the `DeviceBudget` never sees; the "loud BudgetExceeded instead of driver OOM" guarantee does not hold on those paths. | PLAUSIBLE |
| 24 | **Streaming swallows panics:** the `spawn_blocking` `JoinHandle` is dropped, so a panic yields a role chunk then `[DONE]` with no error. Non-streaming generation **cannot be cancelled** on client disconnect (`block_in_place`); streaming stops within ~8 tokens except when `push_delta` never sends. | HUNTER-CONFIRMED |
| 25 | Steer shims: ONNX graph never validated against the manifest at registration; a `[1,512]` input on a 2048-dim model is accepted and panics (`expect`/`assert_eq!`) mid-decode after full prefill. Inject shims are handled correctly (500). | HUNTER-CONFIRMED |
| 26 | `max_tokens: 0` still generates one token (`for _ in 1..max_tokens`, mirrored in 3 places). Panic paths skip the `poll_wait` deferred-destroy flush the code relies on elsewhere. `kv_write` clamps dx/dy but the shader reads only `gid.x` (silent tail loss >65535 tokens — latent). | HUNTER-CONFIRMED |

**Checked and found sound:** `pack_f16` asserts even length; all three KV caches assert on overflow (no wraparound); GPU gathers assert `token < vocab`; attention with `seq_len=0` returns empty; RoPE asserts even dim; `atan2(0,0)=0` so zero-norm polar vectors do not NaN; MoE sort uses `unwrap_or(Equal)`; `prefill_chunk_size` always ≥1 and chunk loops process the final partial chunk; `CORTEX_POLAR_CHUNK_LAYERS=0`/garbage is safe; env-var parsers and `parse_tool_calls` are panic-free on adversarial input; `ParamsBufferPool` — no single forward can wrap (~800-940 slots vs 16384) and `Relaxed` is sound for slot uniqueness; `cache_load` leaks no VRAM on panic (RAII) and inserts nothing half-built; shim delete mid-request is safe (`Arc` clone); `/metrics` sampler has no panic path.

---

## Five root causes (explain ~all 26)

1. **Panic is the error-handling strategy for request-controlled conditions.** `assert!`/`.expect()` on `max_seq`, token ids, lane capacity, shim dims, temperature-derived NaN. Every one turns a *validation* failure into a 500, and several corrupt state on the way (#1, #8) or get swallowed (#24). **Fix theme: validate at the HTTP boundary → 400; request data must never reach an assert.**
2. **The chunker models bytes, not dispatch dimensions.** One missing constraint in `ChunkLimits` (`n ≤ 65535 / max_workgroups_per_token`) explains the entire #4 class — and #5 shows several paths never consult the chunker at all.
3. **"Serialized by the cache-pool mutex" is a false invariant** that lane slack (97%), `ParamsBufferPool` wrap-safety, and the chunker's empty-lane assumption all silently depend on (#7). Either make it true (an engine-level admission semaphore) or stop depending on it.
4. **The GGUF trust boundary is undeclared.** Treating an operator-supplied model as trusted is defensible — but it should be *stated*, and the uncatchable aborts (#13, #15) and silent wraps (#16) are bad regardless of trust.
5. **The parity test suite is dead** (#22). The numeric/silent-wrong findings (#18, #19, #21) are exactly what those 34 tests exist to catch.

## Suggested fix order

1. **Boundary validation (one PR; closes #1, #3, #5, #6, #11, and most of #17's exposure):** cap `max_tokens` against `max_seq_len - cache.seq_len()`; reject prompts/loads/compositions over `max_seq_len` with 400; validate token ids < vocab; clamp `temperature` to a floor (and make the sampler NaN-tolerant with `total_cmp`); cap `cache_pool` size or add eviction; escape user content in the chat template (#2 — tokenize user segments with `parse_special=false`).
2. **Add the dispatch-dimension constraint to `ChunkLimits`** (#4) and route every prefill path through the chunker (#5). Longer term: make softmax/attn_value/rope/qjl_value_weights 2-D like `attn_score` already is.
3. **An engine-level forward semaphore** (#7) — or lower the lane slack and document the real concurrency model. Fix the false comment either way.
4. **Reconcile `entry.tokens` with the resident cache** on every mutation path (#8) — or derive `corpus_len` from `cache.seq_len()`, never from `tokens.len()`.
5. **Un-gate `gpu_engine/tests.rs`** (#22) — it will not compile today; fix the drift and get the parity tests running.
6. GGUF hardening (#13-#17): `checked_add(byte_size) <= file_len` before every `vec!`, `n_dims ≤ 4`, depth limit, reject `alignment == 0`, set `overflow-checks = true` in release. Fix the `rope.scaling.type` key type (#18) and validate norm lengths (#19).
