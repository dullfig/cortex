// Compress an f32 K (or V) tensor into PolarQuant-compressed cache buffers.
//
// Input:  [n_tokens, n_kv_heads, head_dim] f32 — output of the K (or V)
//         projection (post-RoPE for K). Source for one layer, one kind
//         (K or V).
// Output: angles[(start_pos+t) * n_kv_heads + h, :] packed 4 buckets per u32
//         radius[(start_pos+t) * n_kv_heads + h] = mean polar magnitude
//
// One thread per (t, h). Each thread:
//   1. Reads head_dim f32 values
//   2. For each polar pair i:
//        x = R[2i, :]   · k
//        y = R[2i+1, :] · k
//        r_i = sqrt(x^2 + y^2)
//        bucket = quantize(atan2(y, x))
//   3. radius = mean(r_i)
//   4. Writes n_pairs/4 packed u32 words for angles + 1 f32 for radius
//
// Each thread owns a contiguous run of n_pairs angles starting at a
// 4-aligned bucket index, so packing produces complete u32 words with
// no cross-thread contention. Requires n_pairs % 4 == 0
// (head_dim % 8 == 0); enforced by the Rust dispatcher.

struct Params {
    n_tokens: u32,
    start_pos: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_pairs: u32,        // head_dim / 2
    max_seq: u32,        // unused at this layer; kept for parity with kv_write
    _pad0: u32,
    _pad1: u32,
}

// Maximum supported n_pairs (head_dim / 2). 64 → covers head_dim up to 128
// (Qwen, Llama). Bump to 128 if a model with head_dim=256 lands.
const MAX_PAIRS: u32 = 64u;
const MAX_WORDS: u32 = 16u; // MAX_PAIRS / 4

@group(0) @binding(0) var<storage, read>       k_in:     array<f32>;       // [n_tokens, n_kv_heads, head_dim]
@group(0) @binding(1) var<storage, read>       rotation: array<f32>;       // [head_dim, head_dim] row-major
@group(0) @binding(2) var<storage, read_write> angles:   array<u32>;       // packed 4 per word
@group(0) @binding(3) var<storage, read_write> radius:   array<f32>;       // [max_seq * n_kv_heads]
@group(0) @binding(4) var<uniform>             params:   Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / params.n_kv_heads;
    let h = idx % params.n_kv_heads;
    if (t >= params.n_tokens) { return; }

    let k_base = (t * params.n_kv_heads + h) * params.head_dim;

    // Local packed-word accumulator. Initialized lane-by-lane below.
    var local_words: array<u32, MAX_WORDS>;
    for (var w: u32 = 0u; w < MAX_WORDS; w = w + 1u) {
        local_words[w] = 0u;
    }

    var radius_sum: f32 = 0.0;
    let two_pi = 6.283185307179586;
    let pi = 3.141592653589793;

    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        // Inline-compute the two rotated lanes for pair i. R is row-major
        // [head_dim, head_dim]; row 2i lives at offset (2i * head_dim).
        let row_x = (2u * i) * params.head_dim;
        let row_y = (2u * i + 1u) * params.head_dim;
        var x: f32 = 0.0;
        var y: f32 = 0.0;
        for (var col: u32 = 0u; col < params.head_dim; col = col + 1u) {
            let k_v = k_in[k_base + col];
            x = x + rotation[row_x + col] * k_v;
            y = y + rotation[row_y + col] * k_v;
        }

        let r_i = sqrt(x * x + y * y);
        let theta = atan2(y, x); // [-pi, pi]
        let normalized = (theta + pi) / two_pi;
        // Same quantization as polar::to_polar_quantized: bucket = floor(norm*8) % 8.
        // (theta == pi maps to normalized == 1.0 → bucket 8 before mod, 0 after.)
        let bucket = u32(normalized * 8.0) % 8u;
        radius_sum = radius_sum + r_i;

        let word = i / 4u;
        let shift = (i & 3u) * 8u;
        local_words[word] = local_words[word] | (bucket << shift);
    }

    let pos = params.start_pos + t;
    let radius_avg = radius_sum / f32(params.n_pairs);
    radius[pos * params.n_kv_heads + h] = radius_avg;

    let words_per_th = params.n_pairs / 4u;
    let word_base = (pos * params.n_kv_heads + h) * words_per_th;
    for (var w: u32 = 0u; w < words_per_th; w = w + 1u) {
        angles[word_base + w] = local_words[w];
    }
}
