// Phase C2: ReLU²(gate) * up with packed-f16 buffers (BitNet b1.58
// activation). Mirrors silu_mul_batch_packed layout.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read> gate: array<u32>;
@group(0) @binding(1) var<storage, read> up: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 256u;
    let total_packed = (params.n * params.n_tokens) / 2u;
    if (i >= total_packed) { return; }
    let g = unpack2x16float(gate[i]);
    let u = unpack2x16float(up[i]);
    let r_lo_g = max(0.0, g.x);
    let r_hi_g = max(0.0, g.y);
    let r_lo = r_lo_g * r_lo_g * u.x;
    let r_hi = r_hi_g * r_hi_g * u.y;
    output[i] = pack2x16float(vec2<f32>(r_lo, r_hi));
}
