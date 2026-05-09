// Batch attention value from compressed V cache. Mirror of
// attn_value_batch.wgsl but reads packed (V_angles + V_radius) and
// outputs in rotated/compressed space — a separate `derotate` pass
// applies R^T at the end (linearity of the weighted sum lets us
// derotate once per (tok, head), not once per cached position).
//
// output[tok, head, d_rot] = Σ_t softmax[tok, head, t] *
//                            V_rotated[t, kv_h, d_rot]
//
// where V_rotated[t, kv_h, 2i]   = V_radius[t, kv_h] * cos(bucket(t, kv_h, i))
//       V_rotated[t, kv_h, 2i+1] = V_radius[t, kv_h] * sin(bucket(t, kv_h, i))
//
// One thread per (tok, head, d_rot). Causal mask via seq_len bound.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    n_tokens: u32,
}

@group(0) @binding(0) var<storage, read>       softmax:   array<f32>;
@group(0) @binding(1) var<storage, read>       v_angles:  array<u32>;
@group(0) @binding(2) var<storage, read>       v_radius:  array<f32>;
@group(0) @binding(3) var<storage, read_write> output:    array<f32>;
@group(0) @binding(4) var<uniform>             params:    Params;
@group(0) @binding(5) var<uniform>             angle_lut: array<vec4<f32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_dim = params.n_heads * params.head_dim;
    let tok = idx / out_dim;
    let rem = idx % out_dim;
    let head = rem / params.head_dim;
    let d_rot = rem % params.head_dim;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let pair_idx = d_rot / 2u;
    let pair_lane = d_rot & 1u;
    let seq_len = params.start_pos + tok + 1u;
    let score_base = tok * params.n_heads * params.max_seq + head * params.max_seq;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
        let w = softmax[score_base + t];
        let bi = (t * params.n_kv_heads + kv_h) * params.n_pairs + pair_idx;
        let word = v_angles[bi >> 2u];
        let shift = (bi & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;
        let cs = angle_lut[bucket];
        let r = v_radius[t * params.n_kv_heads + kv_h];
        let v_lane = select(cs.y, cs.x, pair_lane == 0u) * r;
        acc = acc + w * v_lane;
    }

    output[tok * out_dim + head * params.head_dim + d_rot] = acc;
}
