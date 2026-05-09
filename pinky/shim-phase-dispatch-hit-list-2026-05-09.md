# Shim phase dispatch (task #6) — hit list for next session

**Why this matters most (per integration-claude):** without the gate
phase, Bob cannot choose to remain silent. Every chat completion
always generates. The entire "ambient assistant / first-class silence"
product framing is broken until #6.6a lands. This is load-bearing,
not feature creep.

## Already shipped (don't redo)
- `forward_full_gpu_with_hidden_capture(tokens, capture_layers) -> HiddenCaptures`
  — read-side hidden state extraction, both per-block and
  final-post-norm. (#4)
- `forward_block_gpu_inner` + polar variant accept
  `post_block_hidden_capture: Option<&wgpu::Buffer>` (3rd trailing
  optional). (#4)
- Shim registry: `ServerState.shims: Mutex<HashMap<String, Arc<RegisteredShim>>>`,
  CRUD endpoints, ort Session loaded from in-memory ONNX bytes. (#5a)
- `POST /v1/shims/infer` end-to-end: text → tokenize → cortex hidden
  → pool (last_token | mean) → ort run → format per output_shape.kind.
  Validated against Qwen 2.5-3B + Identity ONNX. (#5b)

## Order of work (priority-ordered, smallest first)

### 6a — GATE phase (HIGHEST priority — unlocks silence)

The fused gate-and-generate flow from `project_cortex_v1_shim_api.md`:

```
1. Parse request; collect gate_shims / steer_shims / inject_shims / shim_rules
2. (inject prep — defer to 6c, just no-op for now)
3. Prefill (existing path, possibly with hidden capture)
4. Run gate shims on pooled final hidden — reuse /v1/shims/infer's
   pooling helper (extract into a shared fn first)
5. Evaluate shim_rules against gate decisions
6. If silent: emit done{silent:true, gate_decisions, signals}; return
   If activate: swap steers (defer apply to 6b); start decode
7. Decode normally (steers are 6b)
8. On completion: emit done{silent:false, ...}
```

**ChatRequest additions:**
```rust
#[serde(default)] gate_shims: Vec<String>,
#[serde(default)] steer_shims: Vec<String>,
#[serde(default)] inject_shims: Vec<String>,
#[serde(default)] shim_rules: Vec<ShimRule>,
```

**ShimRule (declarative match-and-dispatch, no scripting):**
```rust
struct ShimRule {
    #[serde(rename = "if")] cond: Option<RuleCondition>,
    #[serde(rename = "else", default)] else_branch: bool,
    then: RuleAction,
}
struct RuleCondition {
    gate: String,
    #[serde(flatten)] op: RuleOp,  // {"gt": 0.8} | {"lt": ...} | {"eq": ...}
}
struct RuleAction {
    #[serde(default)] silent: bool,
    #[serde(default)] activate: Vec<String>,
    #[serde(default)] signal: Option<String>,
}
```

Vocabulary is intentionally bounded — no loops, no variables, no
arithmetic. The spec is firm on this: "cortex does not embed a
scripting runtime." When complex logic is needed, AgentOS orchestrates
multiple cortex calls.

**Refactor first:** extract pooling + ort-run from `shim_infer` into
a `run_shim_against_hidden(state, shim, hc) -> serde_json::Value`
helper. Both `/v1/shims/infer` and the chat-handler gate dispatch
will call this.

**Streaming response shape (silent):**
```
data: {"choices":[{"delta":{"role":"assistant"},...}]}
data: {"choices":[{"delta":{},"finish_reason":"silent"}],"metadata":{"gate_decisions":{...},"signals":[...]}}
data: [DONE]
```

**Non-streaming response shape (silent):**
```json
{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"silent"}],
 "metadata":{"gate_decisions":{...},"signals":[...]}}
```

**Smoke validation plan:** register `should_respond` shim wrapping a
mean-pool that returns a scalar. Use `shim_rules: [{"if":
{"gate":"should_respond","gt":0.7}, "then":{"activate":[]}},
{"else":{"silent":true}}]`. Send a chat request → server returns
silent. Lower threshold → server generates.

### 6b — STEER phase (per-token hidden modification)

After each decode step, before sampling logits, run each active steer
shim on the last hidden and modify it. Per the spec: sequential chain
in declared order — instruct-then-voice ≠ voice-then-instruct.

**Implementation surface:** new variant of `generate_with_cache` (or
`generate_with_cache_and_steers`) that, after the per-token forward
but before `sample()`:

1. For each active steer shim (in order):
   - Build pooled hidden (= last token's final hidden, the same shape
     gate uses)
   - Run ort session
   - Apply per `output_shape.kind`:
     - `hidden_delta` → add to hidden, recompute logits via
       `output_proj` (CPU OR a new GPU dispatch)
     - other → reject as unsupported for steer in v1
2. Sample from (possibly modified) logits

**Tricky part:** `forward_full_gpu_with_cache` returns logits already
computed. To re-compute logits after modifying hidden, we either need
a public `output_projection_only(hidden) -> logits` entry on
`GpuEngine`, OR we change the forward to return the hidden + logits
pair so the caller can branch.

Recommended: add `forward_full_gpu_with_cache_returning_hidden`
that yields both. Steer wrapper applies modifications then runs
output proj.

**Streaming compat:** the streaming chat handler in #3 is stateless
only. Steers + cached chat are two open architectural items
(streaming-cached, steers-on-stream). Recommend: 6b ships steers in
NON-streaming mode first; streaming integration is a follow-up.

**Smoke validation:** register a steer shim that's Identity (returns
hidden unchanged). With and without the shim active, generated tokens
should match exactly (modulo float precision). This proves the
plumbing without testing the actual modification semantics.

### 6c — INJECTION phase (residual-add at entrance:N)

Lowest priority — content-adding (FP-LLM style); fully optional. The
hook surface:

- `forward_block_gpu_inner` + polar gain
  `pre_block_hidden_inject: Option<&wgpu::Buffer>` (4th trailing
  optional). At block entrance, if Some, dispatch_add_into the
  injection delta into hidden_buf BEFORE rmsnorm.
- Caller (chat handler) builds per-(layer, shim) deltas: run each
  injection shim, place its `hidden_delta` output into a resident
  buffer, pass the buffer ref for that layer.
- For `entrance:all`, register the same delta at every block.
- Composition: sum (commutative, order-independent).

**Smoke validation:** zero-delta injection shim — model output
byte-identical to no-shim baseline.

## Out of scope for #6 (intentional)

- Per-token gates (the "four score and seven years ago…" abort case) — v2
- Pooling = "attention" — v2 (last_token / mean cover the v1 use cases)
- Dynamic shim loading (.so/.dll) — v1 is compile-time static
- Per-tenant shim configs — single-tenant for v1
- Streaming gate shims on partial prefills — perf opt
- Multi-shim composition rules beyond what shim_rules provides — by
  design, complex composition is AgentOS's concern (multi-call)

## Files that will change

- `cortex-cloud/src/main.rs`:
  - `ChatRequest` gains 4 fields
  - New types: `ShimRule`, `RuleCondition`, `RuleOp`, `RuleAction`
  - New helper: `run_shim_against_hidden(state, shim, &HiddenCaptures)`
    refactored out of `shim_infer`
  - `chat_completions` (and `chat_completions_stream`) gain gate
    dispatch before generate path
- `cortex/src/layers/gpu_engine.rs`:
  - 6b: `forward_full_gpu_with_cache_returning_hidden` (new public fn)
  - 6c: `forward_block_gpu_inner` + polar variant gain a 4th optional
    `pre_block_hidden_inject: Option<&wgpu::Buffer>`

## Validation cadence

After each sub-step, smoke against Qwen 2.5-3B + Identity ONNX shims
sized to embed_dim=2048. Existing test count baseline: 393 workspace
tests, all green --test-threads=1.

## Open question to settle empirically (during 6b)

Per the v1 doc's "open questions" section, item 5: should multiple
steers compose by sequential chain or by sum? Spec recommends
sequential chain — easier to debug, semantics depend on order
(instruct→voice vs voice→instruct should differ). v1 ships sequential
chain; revisit if behavior is surprising.
