// FlashAttention-1 fused score + softmax + value (no scores buffer).
//
// One workgroup per (token, q_head). WG=64 threads — each thread owns
// ONE j position in the K/V tile (for parallel s_j = Q·K_j dot
// products) AND TWO output dims (d_lo = tid*2, d_hi = tid*2 + 1) for
// register-resident o accumulators that pack directly into the
// packed-f16 output u32. Requires head_dim = 2 * WG = 128 (Qwen
// 0.5B/3B both). For head_dim ≠ 128, fall back to the legacy
// 3-shader path.
//
// Online softmax (Milakov & Gimelshein / Dao et al.):
//   For each K/V tile:
//     s_j = Q · K_j
//     m_new = max(m_old, max_j s_j)
//     p_j = exp(s_j - m_new)
//     o[d] = o[d] * exp(m_old - m_new) + sum_j p_j * V_j[d]
//     ℓ_new = ℓ_old * exp(m_old - m_new) + sum_j p_j
//   Final: o[d] /= ℓ
//
// Causal mask: tiles entirely past seq_len break early; the boundary
// tile uses per-element `select(s, -1e30, t > seq_len-1)`.
//
// Bindings:
//   0: q (packed f16, [n_tokens, n_heads, head_dim/2])
//   1: k_cache (packed f16, [max_seq, n_kv_heads, head_dim/2])
//   2: v_cache (packed f16, [max_seq, n_kv_heads, head_dim/2])
//   3: output (packed f16, [n_tokens, n_heads, head_dim/2])
//   4: params (uniform)

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

@group(0) @binding(0) var<storage, read>       q:       array<u32>;
@group(0) @binding(1) var<storage, read>       k_cache: array<u32>;
@group(0) @binding(2) var<storage, read>       v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> output:  array<u32>;
@group(0) @binding(4) var<uniform>             params:  Params;

// Compile-time tile dims (must match dispatcher).
// WG = HEAD_DIM_HALF = 64 → one thread per (j, d_pair).
const HEAD_DIM:      u32 = 128u;
const HEAD_DIM_HALF: u32 = 64u;
const B_K:           u32 = 64u;

// WG-shared arrays. WGSL requires constant-expression sizes —
// can't use `B_K * HEAD_DIM_HALF`; substitute the literal 4096.
var<workgroup> q_shared:  array<u32, 64>;     // HEAD_DIM_HALF
var<workgroup> k_tile:    array<u32, 4096>;   // B_K * HEAD_DIM_HALF
var<workgroup> v_tile:    array<u32, 4096>;   // B_K * HEAD_DIM_HALF
var<workgroup> s_shared:  array<f32, 64>;     // B_K
var<workgroup> p_shared:  array<f32, 64>;     // B_K
var<workgroup> stats:     array<f32, 4>;      // m, ℓ, m_old, _pad

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let tok = wid.y;
    let head = wid.x;

    // Bounds + shape guard (head_dim must be 128 for this shader).
    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let head_dim_half = params.head_dim / 2u;
    let kv_dim_half   = params.kv_dim / 2u;
    let q_dim_half    = params.n_heads * head_dim_half;
    let seq_len       = params.start_pos + tok + 1u;
    let max_total_seq = params.start_pos + params.n_tokens;

    // ---- Load Q[tok, head, :] into shared (packed) ----
    let q_u32_base = tok * q_dim_half + head * head_dim_half;
    if (tid < head_dim_half) {
        q_shared[tid] = q[q_u32_base + tid];
    }

    // Init running stats and per-thread output accumulators.
    if (tid == 0u) {
        stats[0] = -1.0e30; // m
        stats[1] = 0.0;     // ℓ
    }
    var o_lo: f32 = 0.0;
    var o_hi: f32 = 0.0;

    workgroupBarrier();

    // ---- Iterate K/V tiles over the causal seq prefix ----
    // Each tile covers B_K positions. Tiles entirely past seq_len
    // contribute nothing; break early. The boundary tile masks per-j.
    let n_tiles = (max_total_seq + B_K - 1u) / B_K;
    for (var ti: u32 = 0u; ti < n_tiles; ti = ti + 1u) {
        let t_start = ti * B_K;
        if (t_start >= seq_len) { break; }

        // Cooperative tile load: each of 64 threads loads one COLUMN
        // of the [B_K, head_dim_half] packed tile across all B_K rows
        // (64 u32 per thread, simple stride). Same for V.
        for (var jj: u32 = 0u; jj < B_K; jj = jj + 1u) {
            let t = t_start + jj;
            // Clamp out-of-range positions to a safe slot (0). They'll
            // be masked to -1e30 below so the values never reach
            // softmax.
            let t_clamped = min(t, params.max_seq - 1u);
            let kv_row_u32 = t_clamped * kv_dim_half + kv_h * head_dim_half;
            k_tile[jj * HEAD_DIM_HALF + tid] = k_cache[kv_row_u32 + tid];
            v_tile[jj * HEAD_DIM_HALF + tid] = v_cache[kv_row_u32 + tid];
        }
        workgroupBarrier();

        // ---- Score: s_j = Q · K_j  (one thread per j) ----
        // Each thread j does the full head_dim/2 packed dot product.
        let j: u32 = tid;
        let t = t_start + j;
        var dot: f32 = 0.0;
        for (var dp: u32 = 0u; dp < HEAD_DIM_HALF; dp = dp + 1u) {
            let q_pair = unpack2x16float(q_shared[dp]);
            let k_pair = unpack2x16float(k_tile[j * HEAD_DIM_HALF + dp]);
            dot = dot + q_pair.x * k_pair.x + q_pair.y * k_pair.y;
        }
        // Causal mask: t > seq_len-1 (i.e. t >= seq_len) → -1e30.
        // Use select (NOT branch) to keep barrier-uniformity invariants.
        let masked = (t >= seq_len);
        let s_j = select(dot * params.scale, -1.0e30, masked);
        s_shared[j] = s_j;
        workgroupBarrier();

        // ---- Tile-local max reduction over s_shared ----
        // Tree reduction across 64 lanes → s_shared[0] = max.
        var stride: u32 = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) {
                let a = s_shared[tid];
                let b = s_shared[tid + stride];
                s_shared[tid] = max(a, b);
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let m_tile = s_shared[0];

        // Restore s_shared after the reduction clobbered it (we still
        // need the per-j s values for the softmax exp).
        // Re-compute s_j: same as above but no need to redo the dot —
        // we kept it in a local. But `dot` and `masked` are still in
        // scope per-thread. Re-write s_shared[j].
        s_shared[tid] = s_j;
        workgroupBarrier();

        if (tid == 0u) {
            stats[2] = stats[0]; // m_old
            stats[0] = max(stats[0], m_tile); // m_new
        }
        workgroupBarrier();
        let m_old = stats[2];
        let m_new = stats[0];

        // ---- p_j = exp(s_j - m_new) ----
        let p_j = exp(s_shared[j] - m_new);
        p_shared[j] = p_j;
        workgroupBarrier();

        // ---- Sum p_j across tile (tree reduction). Reuses
        // p_shared, so we save p_j locally first.
        // After this loop, p_shared[0] = sum_p_tile.
        var sum_stride: u32 = 32u;
        // Restore p_shared after reduction (need per-j values for
        // accumulator update below); save a private copy.
        let p_local = p_j;
        loop {
            if (sum_stride == 0u) { break; }
            if (tid < sum_stride) {
                p_shared[tid] = p_shared[tid] + p_shared[tid + sum_stride];
            }
            workgroupBarrier();
            sum_stride = sum_stride / 2u;
        }
        let sum_p_tile = p_shared[0];
        // Restore per-j p values.
        p_shared[tid] = p_local;
        workgroupBarrier();

        // ---- Rescale running output and ℓ ----
        let o_scale = exp(m_old - m_new);
        o_lo = o_lo * o_scale;
        o_hi = o_hi * o_scale;
        if (tid == 0u) {
            stats[1] = stats[1] * o_scale + sum_p_tile;
        }

        // ---- o[d] += sum_j p_j * V[j, d] ----
        // Each thread owns d_pair = tid (= dims 2*tid, 2*tid+1).
        // Sum across j ∈ [0, B_K).
        for (var jj: u32 = 0u; jj < B_K; jj = jj + 1u) {
            let p = p_shared[jj];
            // V[jj, tid] is a packed pair (V[jj, 2*tid], V[jj, 2*tid+1]).
            let v_pair = unpack2x16float(v_tile[jj * HEAD_DIM_HALF + tid]);
            o_lo = o_lo + p * v_pair.x;
            o_hi = o_hi + p * v_pair.y;
        }
        workgroupBarrier();
    }

    // ---- Final: o[d] /= ℓ, pack and write ----
    let final_l = stats[1];
    // Guard against zero ℓ (only possible if seq_len == 0, which can't
    // happen for a real forward — but defend against div-by-zero NaN).
    let inv_l = select(1.0 / final_l, 0.0, final_l == 0.0);
    let out_lo = o_lo * inv_l;
    let out_hi = o_hi * inv_l;

    // Output layout matches attn_value_batch.wgsl: packed [n_tokens, n_heads, head_dim/2].
    let out_u32_base = tok * params.n_heads * head_dim_half + head * head_dim_half;
    output[out_u32_base + tid] = pack2x16float(vec2<f32>(out_lo, out_hi));
}
