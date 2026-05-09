// Batch attention scores with causal mask, reading from a PolarQuant-
// compressed K cache. Mirror of attn_score_batch.wgsl but reads packed
// (angles + radius) instead of f32 K.
//
// Q is in the rotated/compressed domain (apply rotate_q upstream).
// scores[tok, head, t] = dot(rq[tok, head], reconstruct_K[t, kv_h]) * scale
// for t <= start_pos + tok; -1e30 (effective -inf) for masked positions.
//
// 2D dispatch: gid.x = (head, t), gid.y = tok. Same shape as the f32
// batch shader so the existing softmax_batch + dispatch wiring works.

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
    _p1: u32, _p2: u32, _p3: u32,
}

@group(0) @binding(0) var<storage, read>       rq:        array<f32>;
@group(0) @binding(1) var<storage, read>       k_angles:  array<u32>;
@group(0) @binding(2) var<storage, read>       k_radius:  array<f32>;
@group(0) @binding(3) var<storage, read_write> scores:    array<f32>;
@group(0) @binding(4) var<uniform>             params:    Params;
@group(0) @binding(5) var<uniform>             angle_lut: array<vec4<f32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.y;
    let max_total_seq = params.start_pos + params.n_tokens;
    let inner = gid.x;
    let head = inner / max_total_seq;
    let t = inner % max_total_seq;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    // Causal mask: token at position (start_pos + tok) attends to [0, start_pos + tok].
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

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let bi = angle_base + i;
        let word = k_angles[bi >> 2u];
        let shift = (bi & 3u) * 8u;
        let bucket = (word >> shift) & 0xFFu;

        let cs = angle_lut[bucket];
        sum = sum + rq[q_base + 2u * i] * cs.x + rq[q_base + 2u * i + 1u] * cs.y;
    }

    scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t] = sum * radius * params.scale;
}
