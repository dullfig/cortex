// Phase C3: packed-f16 RoPE. x is packed (2 f16 per u32). Each thread
// handles TWO adjacent rotation pairs (pair_pair=0 → pairs 0+1) so
// the two outputs at offsets (pair, pair+half_dim) and (pair+1,
// pair+1+half_dim) land in exactly two u32 slots that the thread owns
// — no atomic/race needed.
//
// Requires half_dim % 2 == 0 (all supported models satisfy: head_dim
// 128 → half_dim 64).

struct Params { n_heads: u32, head_dim: u32, start_pos: u32, half_dim: u32, n_tokens: u32, _p1: u32, _p2: u32, _p3: u32 }

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim_half = params.half_dim / 2u;
    let total_pair_pairs = params.n_heads * half_dim_half;
    let flat_idx = gid.x;
    let tok = flat_idx / total_pair_pairs;
    let pp_in_tok = flat_idx % total_pair_pairs;
    let head = pp_in_tok / half_dim_half;
    let pair_pair = pp_in_tok % half_dim_half;
    let pair0 = pair_pair * 2u;
    let pair1 = pair0 + 1u;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let pos = params.start_pos + tok;
    let c0 = cos_table[pos * params.half_dim + pair0];
    let s0 = sin_table[pos * params.half_dim + pair0];
    let c1 = cos_table[pos * params.half_dim + pair1];
    let s1 = sin_table[pos * params.half_dim + pair1];

    // Packed u32 indices. base is even (head_dim even, head*head_dim even).
    let head_dim_half = params.head_dim / 2u;
    let base_u32 = (tok * params.n_heads + head) * head_dim_half;
    let lo_slot = base_u32 + pair_pair;            // covers pair0, pair1
    let hi_slot = base_u32 + (params.half_dim + pair0) / 2u;  // covers pair0+half_dim, pair1+half_dim

    let lo_pair = unpack2x16float(x[lo_slot]);
    let hi_pair = unpack2x16float(x[hi_slot]);
    let x0_a = lo_pair.x; // x[pair0]
    let x0_b = lo_pair.y; // x[pair1]
    let x1_a = hi_pair.x; // x[pair0+half_dim]
    let x1_b = hi_pair.y; // x[pair1+half_dim]

    let lo_a = x0_a * c0 - x1_a * s0;
    let lo_b = x0_b * c1 - x1_b * s1;
    let hi_a = x0_a * s0 + x1_a * c0;
    let hi_b = x0_b * s1 + x1_b * c1;

    x[lo_slot] = pack2x16float(vec2<f32>(lo_a, lo_b));
    x[hi_slot] = pack2x16float(vec2<f32>(hi_a, hi_b));
}
