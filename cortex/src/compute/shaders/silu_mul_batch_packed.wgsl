// Phase C2: SiLU(gate) * up with packed-f16 buffers throughout.
// gate, up, output are all packed f16 (2 per u32). Halves dispatch
// math: each thread processes 2 adjacent elements (one u32 slot).

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
    let r_lo = (g.x / (1.0 + exp(-g.x))) * u.x;
    let r_hi = (g.y / (1.0 + exp(-g.y))) * u.y;
    output[i] = pack2x16float(vec2<f32>(r_lo, r_hi));
}
