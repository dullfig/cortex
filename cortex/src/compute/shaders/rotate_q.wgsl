// Rotate Q from original space into the polar/rotated domain.
//
// rq[tok, head, row] = Σ_col R[row, col] * q[tok, head, col]
//
// Same R as compressed K/V in the polar cache for this layer. Rotation
// preserves dot products, so attn_score_polar's RQ·K_rotated math
// equals the original-space Q·K math modulo polar quantization noise.
//
// Layout: q is `[n_tokens, n_heads, head_dim]` flat (matches the f32
// path's Q buffer post-RoPE); rq is the same shape. R is row-major
// `[head_dim, head_dim]`. One thread per (tok, head, row).

struct Params {
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       q:        array<f32>;       // [n_tokens, n_heads, head_dim]
@group(0) @binding(1) var<storage, read>       rotation: array<f32>;       // [head_dim, head_dim] row-major
@group(0) @binding(2) var<storage, read_write> rq:       array<f32>;       // [n_tokens, n_heads, head_dim]
@group(0) @binding(3) var<uniform>             params:   Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row_total = params.n_tokens * params.n_heads * params.head_dim;
    if (idx >= row_total) { return; }

    let tok_head = idx / params.head_dim;
    let row = idx % params.head_dim;
    let q_base = tok_head * params.head_dim;
    let row_off = row * params.head_dim;

    var sum: f32 = 0.0;
    for (var col: u32 = 0u; col < params.head_dim; col = col + 1u) {
        sum = sum + rotation[row_off + col] * q[q_base + col];
    }
    rq[q_base + row] = sum;
}
