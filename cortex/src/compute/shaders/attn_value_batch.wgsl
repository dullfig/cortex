// Batch attention value: output[tok, head*head_dim + d] = sum_t(scores[tok, head, t] * V[t, kv_h, d])
// scores: f32 (Phase A keeps scores f32 — softmax precision)
// v_cache: [max_seq, kv_dim/2] u32 (packed f16, Phase A — 2 V-values per u32)
// output: f32 (Phase A keeps attn_out f32; Phase C will pack it)
// One thread per (tok, head, d) triple.
//
// Unpacking: each thread reads ONE f16 value out of a u32 that holds 2.
// Consecutive d threads share a u32 — wasteful unpacking but correct;
// Phase C will re-shape to thread-per-d-pair to eliminate the
// redundancy. For now, correctness first.

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
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_dim = params.n_heads * params.head_dim;
    let tok = idx / out_dim;
    let rem = idx % out_dim;
    let head = rem / params.head_dim;
    let d = rem % params.head_dim;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let seq_len = params.start_pos + tok + 1u;
    let score_base = tok * params.n_heads * params.max_seq + head * params.max_seq;

    // V cache packed: f32 index N → u32 index N/2, lane = N & 1.
    let kv_dim_half = params.kv_dim / 2u;
    let head_dim_half_offset = (kv_h * params.head_dim + d) / 2u;
    let d_is_high = (d & 1u) == 1u;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let w = scores[score_base + t];
        let packed = v_cache[t * kv_dim_half + head_dim_half_offset];
        let v_pair = unpack2x16float(packed);
        let v_val = select(v_pair.x, v_pair.y, d_is_high);
        acc = acc + w * v_val;
    }

    output[tok * out_dim + head * params.head_dim + d] = acc;
}
