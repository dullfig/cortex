// Batch in-place add: a[i] += b[i] for n * n_tokens elements.
//
// Phase B mixed-types: `a` is packed f16 (hidden_buf — 2 per u32);
// `b` is f32 (scratch.projected — still f32 in Phase B). Each thread
// processes ONE u32 of `a` (= 2 f16 values), reads the 2 corresponding
// f32 from `b`, accumulates in f32, packs back to `a`.
// Total dispatch threads = (n * n_tokens) / 2.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = gid.x;  // u32 index into a; 2 f32 in b per ip
    let total_u32 = (params.n * params.n_tokens) / 2u;
    if (ip >= total_u32) { return; }
    let a_pair = unpack2x16float(a[ip]);
    let i_lo = ip * 2u;
    let i_hi = i_lo + 1u;
    let sum_lo = a_pair.x + b[i_lo];
    let sum_hi = a_pair.y + b[i_hi];
    a[ip] = pack2x16float(vec2<f32>(sum_lo, sum_hi));
}
