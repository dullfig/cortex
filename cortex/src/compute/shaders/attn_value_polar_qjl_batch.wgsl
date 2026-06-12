// QJL-corrected batch attention value from compressed V cache
// (Phase O). Same weighted polar sum as attn_value_polar_batch.wgsl,
// plus the Γ-scaled residual correction computed from pass A's
// C accumulator (qjl_value_weights.wgsl):
//
//   output[tok, head, d] = Σ_t w_t·V_rotated[t, kv_h, d]
//                        + (Γ/n_proj)·Σ_j C[tok, head, j]·proj[j, d]
//
// The correction approximates Σ_t w_t·residual_t — the part of the
// attention output PolarQuant's 3-bit angles dropped. Output stays in
// rotated space; derotate applies R^T afterwards exactly as in the
// uncorrected path (the correction is linear, so it derotates with
// the rest).
//
// One thread per (tok, head, d_rot): seq-loop + 256 correction MACs.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,
    n_tokens: u32,
    n_proj: u32,         // V QJL projections (multiple of 32)
    gamma: f32,          // v_correction_gamma(n_proj, head_dim)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       softmax:     array<f32>;
@group(0) @binding(1) var<storage, read>       v_angles:    array<u32>;
@group(0) @binding(2) var<storage, read>       v_radius:    array<f32>;
@group(0) @binding(3) var<storage, read>       c_weights:   array<f32>;   // [n_tokens, n_heads, n_proj]
@group(0) @binding(4) var<storage, read>       projections: array<f32>;   // [n_proj, head_dim]
@group(0) @binding(5) var<storage, read_write> output:      array<f32>;
@group(0) @binding(6) var<uniform>             params:      Params;
@group(0) @binding(7) var<uniform>             angle_lut:   array<vec4<f32>, 8>;

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

    // ---- 1. Weighted polar sum (identical to the uncorrected shader) ----
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

    // ---- 2. QJL residual correction ----
    let cj_base = (tok * params.n_heads + head) * params.n_proj;
    var corr: f32 = 0.0;
    for (var j: u32 = 0u; j < params.n_proj; j = j + 1u) {
        corr = corr + c_weights[cj_base + j] * projections[j * params.head_dim + d_rot];
    }
    acc = acc + params.gamma * corr / f32(params.n_proj);

    output[tok * out_dim + head * params.head_dim + d_rot] = acc;
}
