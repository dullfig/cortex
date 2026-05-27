// Batch attention scores with causal mask.
// Q: [n_tokens, n_heads * head_dim] f32
// K_cache: [max_seq, kv_dim/2] u32 (packed f16, Phase A — 2 K-values
//          per u32). Semantic kv_dim in Params is the f32 count;
//          shader does /2 internally for u32 indexing.
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
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
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
    // K cache is packed f16: linear f32 index N → u32 index N/2.
    // k_base in f32 terms (kv_h * head_dim is even since head_dim is even;
    // t * kv_dim is even since kv_dim is even).
    let kv_dim_half = params.kv_dim / 2u;
    let k_u32_base = t * kv_dim_half + kv_h * (params.head_dim / 2u);

    // Iterate over head_dim in pairs (each u32 holds 2 K values).
    // head_dim is guaranteed even.
    let head_dim_half = params.head_dim / 2u;
    var dot: f32 = 0.0;
    for (var dp: u32 = 0u; dp < head_dim_half; dp = dp + 1u) {
        let packed = k_cache[k_u32_base + dp];
        let k_pair = unpack2x16float(packed);
        let q_lo = q[q_base + dp * 2u];
        let q_hi = q[q_base + dp * 2u + 1u];
        dot = dot + q_lo * k_pair.x + q_hi * k_pair.y;
    }

    scores[tok * params.n_heads * params.max_seq + head * params.max_seq + t] = dot * params.scale;
}
