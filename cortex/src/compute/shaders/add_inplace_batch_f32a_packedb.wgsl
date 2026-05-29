// Residual add: `a` is f32 (hidden_buf, Option E revert), `b` is
// packed f16 (scratch.projected — still packed in C3). Each thread
// processes one u32 slot of `b` (= 2 f16 values) and updates the
// corresponding two f32 positions of `a`. Total threads = n*n_tokens/2.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = gid.x;
    let total_u32 = (params.n * params.n_tokens) / 2u;
    if (ip >= total_u32) { return; }
    let b_pair = unpack2x16float(b[ip]);
    let i_lo = ip * 2u;
    let i_hi = i_lo + 1u;
    a[i_lo] = a[i_lo] + b_pair.x;
    a[i_hi] = a[i_hi] + b_pair.y;
}
