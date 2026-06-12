// Pass A of the V-side QJL correction (Phase O): accumulate per-
// (query, head, projection) sign-weighted softmax mass.
//
//   C[tok, head, j] = Σ_t softmax[tok, head, t] · rnorm[t, kv_h] · s_{t,j}
//
// where s_{t,j} = ±1 from the V sign bits. This is the sum-swap that
// makes the vector correction affordable: pass B then adds
// (Γ/n_proj)·Σ_j C_j·p_{j,d} to each output element — 256 MACs instead
// of seq×256 per element. rnorm folds in HERE (it weights each cached
// position's residual estimate; see ops/qjl.rs::v_correction_gamma for
// the Γ-scaled estimator derivation).
//
// One thread per (tok, head, j). Causal mask via seq_len bound, same
// as the value shader.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    n_proj: u32,         // V QJL projections (multiple of 32)
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_tokens: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       softmax: array<f32>;
@group(0) @binding(1) var<storage, read>       v_signs: array<u32>;   // [max_seq * n_kv_heads * n_proj/32]
@group(0) @binding(2) var<storage, read>       rnorm:   array<f32>;   // [max_seq * n_kv_heads]
@group(0) @binding(3) var<storage, read_write> c_out:   array<f32>;   // [n_tokens, n_heads, n_proj]
@group(0) @binding(4) var<uniform>             params:  Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let per_tok = params.n_heads * params.n_proj;
    let tok = idx / per_tok;
    let rem = idx % per_tok;
    let head = rem / params.n_proj;
    let j = rem % params.n_proj;

    if (tok >= params.n_tokens) { return; }

    let kv_h = head / params.heads_per_kv;
    let seq_len = params.start_pos + tok + 1u;
    let score_base = tok * params.n_heads * params.max_seq + head * params.max_seq;
    let sign_words = params.n_proj / 32u;
    let w_idx = j / 32u;
    let bit = j % 32u;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let entry = t * params.n_kv_heads + kv_h;
        let word = v_signs[entry * sign_words + w_idx];
        let s = select(-1.0, 1.0, ((word >> bit) & 1u) == 1u);
        acc = acc + softmax[score_base + t] * rnorm[entry] * s;
    }

    c_out[tok * per_tok + head * params.n_proj + j] = acc;
}
