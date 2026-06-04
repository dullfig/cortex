// Batch attention scores with QJL correction. Drop-in replacement for
// `attn_score_polar_batch.wgsl` when the polar cache has QJL signs
// (`n_qjl_proj > 0`). Same workgroup/dispatch shape; adds two bindings
// (signs, projections) and one param (n_proj).
//
// Score formula (matches CPU `quantized_kv_cache::dot_key`):
//
//   polar_dot      = Σ_i [rq[2i] * lut.cos[bucket_i] + rq[2i+1] * lut.sin[bucket_i]]
//   qjl_correction = (Σ_j sign_j * (rq · projection_j)) / n_proj
//   score          = (polar_dot * radius + qjl_correction) * scale
//
// The QJL term is added in the dequantized-dot domain (after radius),
// NOT inside the polar dot — matches CPU exactly. Sign bit j of the
// signs u32 word maps 1→+1, 0→−1 (consistent with CPU
// `QjlProjection::correction_dot`).
//
// Per-thread cost: polar dot O(n_pairs) + QJL correction O(n_proj * head_dim).
// For Qwen 3B (head_dim=128, n_proj=32, n_pairs=64): 64 + 4096 = ~17x
// compute vs non-QJL polar. Still well under 1ms at typical workloads.
// A future optimization can hoist `q · projection_j` out of the t-loop
// since it doesn't depend on t (see plan Phase 5).

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    n_pairs: u32,        // head_dim / 2
    scale: f32,
    n_tokens: u32,
    n_proj: u32,         // QJL projections, <= 32
    _p2: u32, _p3: u32,
}

@group(0) @binding(0) var<storage, read>       rq:          array<f32>;
@group(0) @binding(1) var<storage, read>       k_angles:    array<u32>;
@group(0) @binding(2) var<storage, read>       k_radius:    array<f32>;
@group(0) @binding(3) var<storage, read_write> scores:      array<f32>;
@group(0) @binding(4) var<storage, read>       k_qjl_signs: array<u32>;
@group(0) @binding(5) var<storage, read>       projections: array<f32>;
@group(0) @binding(6) var<uniform>             params:      Params;
@group(0) @binding(7) var<uniform>             angle_lut:   array<vec4<f32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.y;
    let max_total_seq = params.start_pos + params.n_tokens;
    let inner = gid.x;
    let head = inner / max_total_seq;
    let t = inner % max_total_seq;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    // Causal mask.
    let seq_len = params.start_pos + tok + 1u;
    if (t >= seq_len) {
        scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t] = -1e30;
        return;
    }

    let kv_h = head / params.heads_per_kv;
    let q_dim = params.n_heads * params.head_dim;
    let q_base = tok * q_dim + head * params.head_dim;

    let angle_base = (t * params.n_kv_heads + kv_h) * params.n_pairs;
    let radius = k_radius[t * params.n_kv_heads + kv_h];

    // ---- 1. Polar dot ----
    var polar_dot: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let bi = angle_base + i;
        let word = k_angles[bi >> 2u];
        let shift = (bi & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;

        let cs = angle_lut[bucket];
        polar_dot = polar_dot
            + rq[q_base + 2u * i] * cs.x
            + rq[q_base + 2u * i + 1u] * cs.y;
    }

    // ---- 2. QJL correction ----
    let signs_idx = t * params.n_kv_heads + kv_h;
    let signs_word = k_qjl_signs[signs_idx];
    var correction: f32 = 0.0;
    for (var j: u32 = 0u; j < params.n_proj; j = j + 1u) {
        let proj_off = j * params.head_dim;
        var qp: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            qp = qp + rq[q_base + d] * projections[proj_off + d];
        }
        let sign_bit = (signs_word >> j) & 1u;
        let signed_qp = select(-qp, qp, sign_bit == 1u);
        correction = correction + signed_qp;
    }
    correction = correction / f32(params.n_proj);

    // ---- 3. Final score: (polar_dot * radius + correction) * scale ----
    scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t]
        = (polar_dot * radius + correction) * params.scale;
}
