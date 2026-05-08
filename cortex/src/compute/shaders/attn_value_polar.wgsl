// Attention value accumulation from PolarQuant-compressed V cache.
//
// Pass A of the compressed-V output. For each (head, d_rot) thread,
// computes one component of the weighted sum in rotated/compressed space:
//
//   weighted_rotated_V[head, d_rot] = Σ_p softmax[head, p] * V_rotated[p, kv_h, d_rot]
//
// Where V_rotated is reconstructed inline from (V_angles, V_radius):
//   V_rotated[p, kv_h, 2i]   = V_radius[p, kv_h] * cos(bucket(p, kv_h, i))
//   V_rotated[p, kv_h, 2i+1] = V_radius[p, kv_h] * sin(bucket(p, kv_h, i))
//
// A separate de-rotation pass applies R^T to weighted_rotated_V to get
// the final attention output in original space. Doing R^T once at the
// end (vs once per cached position) saves a matvec per p — the
// linearity of weighted sum makes this exact, not approximate.
//
// One thread per (head, d_rot). Mirrors attn_value.wgsl shape/dispatch.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,        // head_dim / 2
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       softmax:   array<f32>;       // [n_heads * max_seq]
@group(0) @binding(1) var<storage, read>       v_angles:  array<u32>;       // packed 4 buckets per u32
@group(0) @binding(2) var<storage, read>       v_radius:  array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(3) var<storage, read_write> out:       array<f32>;       // [n_heads * head_dim]
@group(0) @binding(4) var<uniform>             params:    Params;
@group(0) @binding(5) var<uniform>             angle_lut: array<vec4<f32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.head_dim;
    let d_rot = idx % params.head_dim;
    if (head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let pair_idx = d_rot / 2u;        // which polar pair this dim belongs to
    let pair_lane = d_rot & 1u;       // 0 = cos lane, 1 = sin lane

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < params.seq_len; t = t + 1u) {
        let w = softmax[head * params.max_seq + t];

        // Read the polar bucket for (t, kv_h, pair_idx).
        let bi = (t * params.n_kv_heads + kv_h) * params.n_pairs + pair_idx;
        let word = v_angles[bi >> 2u];
        let shift = (bi & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;

        let cs = angle_lut[bucket];
        let r = v_radius[t * params.n_kv_heads + kv_h];

        // V_rotated component on this lane: cos for even d_rot, sin for odd.
        let v_lane = select(cs.y, cs.x, pair_lane == 0u) * r;

        acc = acc + w * v_lane;
    }

    out[head * params.head_dim + d_rot] = acc;
}
