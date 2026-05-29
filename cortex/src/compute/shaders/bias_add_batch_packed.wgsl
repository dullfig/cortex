// Phase C3: packed-f16 bias add. `a` is packed (2 f16 per u32).
// Each thread processes one u32 slot = 2 adjacent column positions.
// Bias stays f32 (small, scarcely-touched).

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let total_slots = (params.n * params.n_tokens) / 2u;
    if (slot >= total_slots) { return; }
    let n_half = params.n / 2u;
    let col_pair = slot % n_half;
    let col_lo = col_pair * 2u;
    let col_hi = col_lo + 1u;
    let pair = unpack2x16float(a[slot]);
    let new_lo = pair.x + bias[col_lo];
    let new_hi = pair.y + bias[col_hi];
    a[slot] = pack2x16float(vec2<f32>(new_lo, new_hi));
}
