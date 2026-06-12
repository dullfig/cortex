// Encode QJL sign bits + residual norm for the V side of a PolarQuant
// cache (Phase O). Mirror of kv_qjl_encode.wgsl with two differences:
//
//   1. Multi-word sign output: V uses n_proj = 256 (a full residual
//      VECTOR needs far more bits than K's effectively-scalar score
//      correction), so each (pos, head) entry stores
//      sign_words = n_proj/32 consecutive u32 words. Word w bit b is
//      the sign for projection w*32 + b — matching the CPU
//      `QjlProjection::encode_signs` byte/bit packing little-endian.
//   2. Residual norm output: ||residual|| per entry, consumed by the
//      Γ-scaled value correction (see ops/qjl.rs::v_correction_gamma).
//
// One thread per (t, h), same as the K encoder. n_proj=256 ×
// head_dim=128 = 32k MACs per thread on top of the rotate — fine at
// prefill rates.
//
// Requires head_dim <= 128 (MAX_HEAD_DIM, register array size).

struct Params {
    n_tokens: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_pairs: u32,        // head_dim / 2
    n_proj: u32,         // V QJL projections (multiple of 32, e.g. 256)
    max_seq: u32,        // unused at this layer; kept for parity
    _pad: u32,
}

const MAX_HEAD_DIM: u32 = 128u;

@group(0) @binding(0) var<storage, read>       v_in:        array<u32>;       // packed f16 [n_tokens, n_kv_heads, head_dim/2]
@group(0) @binding(1) var<storage, read>       rotation:    array<f32>;       // [head_dim, head_dim]
@group(0) @binding(2) var<storage, read>       v_angles:    array<u32>;       // packed 4 buckets per word (from kv_compress_polar)
@group(0) @binding(3) var<storage, read>       v_radius:    array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(4) var<storage, read>       projections: array<f32>;       // [n_proj, head_dim]
@group(0) @binding(5) var<storage, read_write> signs:       array<u32>;       // [max_seq * n_kv_heads * n_proj/32]
@group(0) @binding(6) var<storage, read_write> rnorm:       array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(7) var<uniform>             angle_lut:   array<vec4<f32>, 8>;
@group(0) @binding(8) var<uniform>             params:      Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / params.n_kv_heads;
    let h = idx % params.n_kv_heads;
    if (t >= params.n_tokens) { return; }

    let head_dim_half = params.head_dim / 2u;
    let v_base = (t * params.n_kv_heads + h) * head_dim_half;

    // ---- 1. Rotated V: rotated[d] = Σ_c R[d, c] * V[c] ----
    var rotated: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
        let row_off = d * params.head_dim;
        var sum: f32 = 0.0;
        for (var slot: u32 = 0u; slot < head_dim_half; slot = slot + 1u) {
            let pair = unpack2x16float(v_in[v_base + slot]);
            let col_lo = slot * 2u;
            sum = sum
                + rotation[row_off + col_lo] * pair.x
                + rotation[row_off + col_lo + 1u] * pair.y;
        }
        rotated[d] = sum;
    }

    // ---- 2-3. Dequantize and compute residual + its norm ----
    let pos = params.start_pos + t;
    let words_per_th = params.n_pairs / 4u;
    let angle_word_base = (pos * params.n_kv_heads + h) * words_per_th;
    let radius = v_radius[pos * params.n_kv_heads + h];

    var residual: array<f32, MAX_HEAD_DIM>;
    var norm_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let word = v_angles[angle_word_base + (i >> 2u)];
        let shift = (i & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;
        let cs = angle_lut[bucket];
        let r_lo = rotated[2u * i]      - radius * cs.x;
        let r_hi = rotated[2u * i + 1u] - radius * cs.y;
        residual[2u * i]      = r_lo;
        residual[2u * i + 1u] = r_hi;
        norm_sq = norm_sq + r_lo * r_lo + r_hi * r_hi;
    }
    rnorm[pos * params.n_kv_heads + h] = sqrt(norm_sq);

    // ---- 4. Sign bits per projection, multi-word ----
    let sign_words = params.n_proj / 32u;
    let signs_base = (pos * params.n_kv_heads + h) * sign_words;
    for (var w: u32 = 0u; w < sign_words; w = w + 1u) {
        var local_word: u32 = 0u;
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let j = w * 32u + b;
            let proj_off = j * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
                dot = dot + residual[d] * projections[proj_off + d];
            }
            if (dot >= 0.0) {
                local_word = local_word | (1u << b);
            }
        }
        signs[signs_base + w] = local_word;
    }
}
