// Write K and V vectors for multiple tokens into KV cache.
// k_src/v_src: [n_tokens, kv_dim] f32 (scratch from RoPE / V proj)
// k_cache/v_cache: [max_seq, kv_dim/2] u32 (packed f16, Phase A)
// Pack 2 adjacent f32 K-values into one u32 via pack2x16float.
// Dispatch: one thread per (token, kv_dim_pair). Total threads =
// n_tokens × (kv_dim / 2). kv_dim is guaranteed even.

struct Params { kv_dim: u32, start_pos: u32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
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

    // Read two adjacent f32 source values per K and V.
    let src_base = tok * params.kv_dim + dim_pair * 2u;
    let k_lo = k_src[src_base];
    let k_hi = k_src[src_base + 1u];
    let v_lo = v_src[src_base];
    let v_hi = v_src[src_base + 1u];

    // Pack to f16 pair and write to cache at packed u32 index.
    let cache_u32 = (params.start_pos + tok) * kv_dim_half + dim_pair;
    k_cache[cache_u32] = pack2x16float(vec2<f32>(k_lo, k_hi));
    v_cache[cache_u32] = pack2x16float(vec2<f32>(v_lo, v_hi));
}
