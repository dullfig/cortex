// Phase C3: both `a` and `b` packed f16. Used for hidden += projected
// when scratch.projected becomes packed in C3.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = gid.x;
    let total_u32 = (params.n * params.n_tokens) / 2u;
    if (ip >= total_u32) { return; }
    let a_pair = unpack2x16float(a[ip]);
    let b_pair = unpack2x16float(b[ip]);
    a[ip] = pack2x16float(vec2<f32>(a_pair.x + b_pair.x, a_pair.y + b_pair.y));
}
