// Write K and V vectors for multiple tokens into KV cache.
// Phase C3: k_src/v_src are now packed f16 too (scratch.k/v packed).
// k_src/v_src: [n_tokens, kv_dim/2] u32 (packed f16)
// k_cache/v_cache: [max_seq, kv_dim/2] u32 (packed f16, Phase A)
// One thread copies one u32 slot per (tok, dim_pair).

struct Params { kv_dim: u32, start_pos: u32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> k_src: array<u32>;
@group(0) @binding(1) var<storage, read> v_src: array<u32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kv_dim_half = params.kv_dim / 2u;
    let flat_u32 = gid.x;
    let total_u32 = kv_dim_half * params.n_tokens;
    if (flat_u32 >= total_u32) { return; }

    let tok = flat_u32 / kv_dim_half;
    let dim_pair = flat_u32 % kv_dim_half;

    // Source already packed: one u32 = one pair.
    let src_u32 = tok * kv_dim_half + dim_pair;
    let cache_u32 = (params.start_pos + tok) * kv_dim_half + dim_pair;
    k_cache[cache_u32] = k_src[src_u32];
    v_cache[cache_u32] = v_src[src_u32];
}
