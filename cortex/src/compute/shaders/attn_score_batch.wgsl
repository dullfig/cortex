// Batch attention scores with causal mask.
// Q: [n_tokens, n_heads * head_dim], K_cache: [max_seq, kv_dim]
// scores[tok, head, t] = Q[tok, head] · K[t, kv_h] * scale  (for t <= start_pos + tok)
// One thread per (tok, head, t) triple.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    scale: f32,
    n_tokens: u32,
    _p1: u32, _p2: u32, _p3: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch: gid.x covers (head, t); gid.y is tok.
    // 1D over n_tokens × n_heads × max_seq trips wgpu's 65535 per-dim
    // workgroup limit at long-corpus scales (e.g. Qwen 3B + 2300 tokens).
    let tok = gid.y;
    let max_total_seq = params.start_pos + params.n_tokens;
    let inner = gid.x;
    let head = inner / max_total_seq;
    let t = inner % max_total_seq;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    // Causal mask: token at position (start_pos + tok) can attend to [0, start_pos + tok]
    let seq_len = params.start_pos + tok + 1u;
    if (t >= seq_len) {
        // Write -inf for masked positions
        scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t] = -1e30;
        return;
    }

    let kv_h = head / params.heads_per_kv;
    let q_dim = params.n_heads * params.head_dim;
    let q_base = tok * q_dim + head * params.head_dim;
    let k_base = t * params.kv_dim + kv_h * params.head_dim;

    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < params.head_dim; d++) {
        dot += q[q_base + d] * k_cache[k_base + d];
    }

    scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t] = dot * params.scale;
}
