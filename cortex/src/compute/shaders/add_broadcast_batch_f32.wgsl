// Broadcast in-place add (f32 hidden + f32 delta) for the injection
// shim hook. Pre-Phase-B style: every row of `a` (shape [n_tokens, n])
// gets the same `b` (shape [n]) added. Used when hidden_buf reverts
// to f32 (Option E). Delta was always f32; this just unwinds the
// Phase B packed-`a` variant.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.n_tokens;
    if (i >= total) { return; }
    let j = i % params.n;
    a[i] = a[i] + b[j];
}
