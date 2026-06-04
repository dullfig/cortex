// Encode QJL (Quantized Johnson-Lindenstrauss) sign bits for the K-side
// residual of a PolarQuant-compressed cache. Runs after kv_compress_polar
// (which produced angles + radius); reads the same K input plus the
// freshly-written angles/radius to compute residual = rotated - dequantized,
// then writes 1-bit signs of projections.
//
// One thread per (t, h). Each thread:
//   1. Recompute rotated K (same math as kv_compress_polar).
//   2. Dequantize via LUT and radius: dq[d] from angles.
//   3. residual[d] = rotated[d] - dq[d].
//   4. For each projection j in 0..n_proj: sign_j = (residual · proj[j]) >= 0.
//   5. Pack sign bits into one u32 word (n_proj <= 32) and write.
//
// Bit-pack layout matches `cortex::ops::qjl::QjlProjection::encode_signs`:
//   - signs[j/8] |= 1 << (j%8) on CPU side
//   - We pack into a single u32 word per (pos, head) entry on GPU side
//   - For little-endian u32 storage, bit j of the u32 ↔ CPU sign bit j
//     (byte j/8 of u32 holds bits j..j+8 of the word)
//
// Requires head_dim <= 128 (MAX_HEAD_DIM, register array size). Bump if
// a model with head_dim > 128 lands.

struct Params {
    n_tokens: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_pairs: u32,        // head_dim / 2
    n_proj: u32,         // QJL projections, <= 32
    max_seq: u32,        // unused at this layer; kept for parity
    _pad: u32,
}

const MAX_HEAD_DIM: u32 = 128u;

@group(0) @binding(0) var<storage, read>       k_in:        array<u32>;       // packed f16 [n_tokens, n_kv_heads, head_dim/2]
@group(0) @binding(1) var<storage, read>       rotation:    array<f32>;       // [head_dim, head_dim]
@group(0) @binding(2) var<storage, read>       k_angles:    array<u32>;       // packed 4 buckets per word (from kv_compress_polar)
@group(0) @binding(3) var<storage, read>       k_radius:    array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(4) var<storage, read>       projections: array<f32>;       // [n_proj, head_dim]
@group(0) @binding(5) var<storage, read_write> signs:       array<u32>;       // [max_seq * n_kv_heads]
@group(0) @binding(6) var<uniform>             angle_lut:   array<vec4<f32>, 8>;
@group(0) @binding(7) var<uniform>             params:      Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / params.n_kv_heads;
    let h = idx % params.n_kv_heads;
    if (t >= params.n_tokens) { return; }

    let head_dim_half = params.head_dim / 2u;
    let k_base = (t * params.n_kv_heads + h) * head_dim_half;

    // ---- 1. Rotated K: rotated[d] = Σ_c R[d, c] * K[c] ----
    // Naive O(head_dim^2) inline mat-vec. For head_dim=128: 16384
    // multiplies per thread — well under 1ms at Qwen 3B prefill rates.
    var rotated: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
        let row_off = d * params.head_dim;
        var sum: f32 = 0.0;
        for (var slot: u32 = 0u; slot < head_dim_half; slot = slot + 1u) {
            let pair = unpack2x16float(k_in[k_base + slot]);
            let col_lo = slot * 2u;
            sum = sum
                + rotation[row_off + col_lo] * pair.x
                + rotation[row_off + col_lo + 1u] * pair.y;
        }
        rotated[d] = sum;
    }

    // ---- 2-3. Dequantize and compute residual ----
    let pos = params.start_pos + t;
    let words_per_th = params.n_pairs / 4u;
    let angle_word_base = (pos * params.n_kv_heads + h) * words_per_th;
    let radius = k_radius[pos * params.n_kv_heads + h];

    var residual: array<f32, MAX_HEAD_DIM>;
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        // Angles are packed 4 per word; lane = i % 4 picks byte within word.
        let word = k_angles[angle_word_base + (i >> 2u)];
        let shift = (i & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;
        let cs = angle_lut[bucket];
        let dq_lo = radius * cs.x;
        let dq_hi = radius * cs.y;
        residual[2u * i]      = rotated[2u * i]      - dq_lo;
        residual[2u * i + 1u] = rotated[2u * i + 1u] - dq_hi;
    }

    // ---- 4. Sign bits per projection ----
    // CPU encode_signs: `if dot >= 0.0 { signs[j/8] |= 1 << (j%8); }`
    // For one u32 word holding bits 0..32, bit j of the word == byte j/8
    // bit j%8 in little-endian, which exactly matches.
    var local_word: u32 = 0u;
    for (var j: u32 = 0u; j < params.n_proj; j = j + 1u) {
        let proj_off = j * params.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + residual[d] * projections[proj_off + d];
        }
        if (dot >= 0.0) {
            local_word = local_word | (1u << j);
        }
    }

    signs[pos * params.n_kv_heads + h] = local_word;
}
