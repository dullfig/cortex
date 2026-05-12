// Batch ReLU²(gate) * up for multiple tokens. The squared-ReLU
// activation used by BitNet b1.58 SwiGLU: out = max(0, g)² * u.
// Layout matches silu_mul_batch.wgsl exactly so callers can swap
// pipelines without changing buffer bindings.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 256u;
    let total = params.n * params.n_tokens;
    if (i >= total) { return; }
    let g = gate[i];
    let relu_g = max(0.0, g);
    output[i] = relu_g * relu_g * up[i];
}
