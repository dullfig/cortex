// Attention scores from PolarQuant-compressed K cache.
//
// Computes scores[h * max_seq + t] = dot(rq[h], reconstruct_K[t, kv_h]) * scale
// where reconstruct_K reads (angle bucket, radius) at (t, kv_h) and projects
// onto the unit circle via the angle LUT, scaled by radius.
//
// `rq` is the *already-rotated* query (Q @ R, computed upstream). The dot
// product is computed in the rotated/compressed domain — rotation preserves
// dot products, so this equals the score that an uncompressed K attention
// would produce, modulo 3-bit angle quantization error.
//
// Storage layout:
//   k_angles: u32 array, 4 buckets packed per word (LSB-first)
//   k_radius: f32 array, [max_seq * n_kv_heads]
//   angle_lut: vec2<f32>[8] (cos, sin), uniform
//
// One thread per (head, t) pair. Mirrors attn_score.wgsl shape.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,        // head_dim / 2
    scale: f32,
}

@group(0) @binding(0) var<storage, read>       rq:        array<f32>;       // [n_heads * head_dim]
@group(0) @binding(1) var<storage, read>       k_angles:  array<u32>;       // packed 4 buckets per u32
@group(0) @binding(2) var<storage, read>       k_radius:  array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(3) var<storage, read_write> scores:    array<f32>;       // [n_heads * max_seq]
@group(0) @binding(4) var<uniform>             params:    Params;
@group(0) @binding(5) var<uniform>             angle_lut: array<vec4<f32>, 8>; // packed (cos, sin, _, _) per bucket

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.seq_len;
    let t = idx % params.seq_len;
    if (head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let q_base = head * params.head_dim;

    // Per-pair offset into the packed angle stream:
    //   bucket_index = (t * n_kv_heads + kv_h) * n_pairs + i
    let angle_base = (t * params.n_kv_heads + kv_h) * params.n_pairs;
    let radius = k_radius[t * params.n_kv_heads + kv_h];

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let bi = angle_base + i;
        let word = k_angles[bi >> 2u];
        let shift = (bi & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;

        let cs = angle_lut[bucket];
        sum = sum + rq[q_base + 2u * i] * cs.x + rq[q_base + 2u * i + 1u] * cs.y;
    }

    scores[head * params.max_seq + t] = sum * radius * params.scale;
}
