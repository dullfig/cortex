// Batch attention value: output[tok, head, d] = sum_t(scores[tok, head, t] * V[t, kv_h, d])
// scores: f32 (kept f32 for softmax precision)
// v_cache: [max_seq, kv_dim/2] u32 (packed f16, Phase A)
// output:  [n_tokens, n_heads, head_dim/2] u32 (packed f16, Phase C3)
//
// One thread per (tok, head, d_pair) — d_pair = d/2. Each thread
// reads one packed V u32 per t (= 2 V values) and writes one packed
// output u32 (= 2 accumulators). Halves thread count AND eliminates
// the per-thread redundant unpack from the Phase A version.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    n_tokens: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim_half = params.head_dim / 2u;
    let out_dim_half = params.n_heads * head_dim_half;
    let idx = gid.x;
    let tok = idx / out_dim_half;
    let rem = idx % out_dim_half;
    let head = rem / head_dim_half;
    let d_pair = rem % head_dim_half;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let seq_len = params.start_pos + tok + 1u;
    let score_base = tok * params.n_heads * params.max_seq + head * params.max_seq;

    let kv_dim_half = params.kv_dim / 2u;
    let v_off = kv_h * head_dim_half + d_pair;

    var acc_lo: f32 = 0.0;
    var acc_hi: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let w = scores[score_base + t];
        let v_pair = unpack2x16float(v_cache[t * kv_dim_half + v_off]);
        acc_lo = acc_lo + w * v_pair.x;
        acc_hi = acc_hi + w * v_pair.y;
    }

    output[tok * out_dim_half + head * head_dim_half + d_pair] =
        pack2x16float(vec2<f32>(acc_lo, acc_hi));
}
