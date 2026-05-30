// Phase C3 polar: packed-f16 input variant of rotate_q.
//
// q is packed f16 (2 per u32) from scratch.q after RoPE. rq stays f32
// because it feeds attn_score_polar_batch which reads it as f32.
//
// Inner loop iterates over head_dim/2 packed slots, unpacks each pair,
// and multiplies by the two corresponding R[row, col] entries. Same
// dispatch shape as the f32 rotate_q (one thread per output element).

struct Params {
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       q:        array<u32>;       // packed f16
@group(0) @binding(1) var<storage, read>       rotation: array<f32>;       // [head_dim, head_dim] row-major
@group(0) @binding(2) var<storage, read_write> rq:       array<f32>;       // f32 output
@group(0) @binding(3) var<uniform>             params:   Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row_total = params.n_tokens * params.n_heads * params.head_dim;
    if (idx >= row_total) { return; }

    let tok_head = idx / params.head_dim;
    let row = idx % params.head_dim;
    let head_dim_half = params.head_dim / 2u;
    let q_u32_base = tok_head * head_dim_half;
    let row_off = row * params.head_dim;

    var sum: f32 = 0.0;
    for (var pair: u32 = 0u; pair < head_dim_half; pair = pair + 1u) {
        let qp = unpack2x16float(q[q_u32_base + pair]);
        let col_lo = pair * 2u;
        let col_hi = col_lo + 1u;
        sum = sum + rotation[row_off + col_lo] * qp.x
                  + rotation[row_off + col_hi] * qp.y;
    }
    rq[tok_head * params.head_dim + row] = sum;
}
