// Phase C3 polar: packed-f16 output variant of derotate.
//
// x stays f32 (it's the rotated-weighted-V output from
// attn_value_polar_batch — that buffer stays f32 in the polar pipeline).
// out is packed f16 (scratch.attn_out is packed in C3).
//
// One thread per (head, d_pair) — half the original thread count.
// Each thread computes two adjacent d outputs and packs them into one
// u32 slot.

struct Params {
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read>       x:        array<f32>;   // [n_heads * head_dim], rotated f32
@group(0) @binding(1) var<storage, read>       rotation: array<f32>;   // [head_dim * head_dim], row-major R
@group(0) @binding(2) var<storage, read_write> out:      array<u32>;   // packed f16, [n_heads * head_dim/2]
@group(0) @binding(3) var<uniform>             params:   Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim_half = params.head_dim / 2u;
    let idx = gid.x;
    let head = idx / head_dim_half;
    let d_pair = idx % head_dim_half;
    if (head >= params.n_heads) { return; }

    let x_base = head * params.head_dim;
    let d_lo = d_pair * 2u;
    let d_hi = d_lo + 1u;

    var sum_lo: f32 = 0.0;
    var sum_hi: f32 = 0.0;
    for (var d_rot: u32 = 0u; d_rot < params.head_dim; d_rot = d_rot + 1u) {
        // R is row-major [d_rot, d]: R[d_rot, d] = rotation[d_rot * head_dim + d].
        let r_off = d_rot * params.head_dim;
        let xv = x[x_base + d_rot];
        sum_lo = sum_lo + rotation[r_off + d_lo] * xv;
        sum_hi = sum_hi + rotation[r_off + d_hi] * xv;
    }

    out[head * head_dim_half + d_pair] = pack2x16float(vec2<f32>(sum_lo, sum_hi));
}
