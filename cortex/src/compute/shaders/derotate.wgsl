// De-rotation: out = R^T @ x, applied per-head.
//
// Pass B of the compressed-V output. Takes the rotated-space weighted V
// from `attn_value_polar` and projects back into original space using
// the transpose of the same orthogonal rotation R that was applied to K
// and V on append. R is shape [head_dim, head_dim], row-major.
//
// out[head, d] = Σ_d_rot R[d_rot, d] * x[head, d_rot]
//             = Σ_d_rot R^T[d, d_rot] * x[head, d_rot]
//
// One thread per (head, d). Pure float matvec, no compression involved
// here — kept as its own shader so the value path stays small and the
// rotation matrix doesn't get wired into the value-accumulation shader.

struct Params {
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read>       x:        array<f32>;   // [n_heads * head_dim], rotated
@group(0) @binding(1) var<storage, read>       rotation: array<f32>;   // [head_dim * head_dim], row-major R
@group(0) @binding(2) var<storage, read_write> out:      array<f32>;   // [n_heads * head_dim], de-rotated
@group(0) @binding(3) var<uniform>             params:   Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.head_dim;
    let d = idx % params.head_dim;
    if (head >= params.n_heads) { return; }

    let x_base = head * params.head_dim;

    var sum: f32 = 0.0;
    for (var d_rot: u32 = 0u; d_rot < params.head_dim; d_rot = d_rot + 1u) {
        // R is row-major [d_rot, d], so R[d_rot, d] = rotation[d_rot * head_dim + d].
        sum = sum + rotation[d_rot * params.head_dim + d] * x[x_base + d_rot];
    }

    out[x_base + d] = sum;
}
