// Fully-f32 residual add (both `a` and `b` f32). Used when scratch.projected
// reverts to f32 alongside hidden_buf (BitNet matmul outputs saturate the
// f16 ceiling, so projected can't stay packed).

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.n_tokens;
    if (i >= total) { return; }
    a[i] = a[i] + b[i];
}
