// Broadcast in-place add: every row of a (shape [n_tokens, n]) gets the
// same delta vector b (shape [n]) added. For injection-phase shims
// (#6c) — one [embed_dim] hidden_delta is computed per request and
// applied at the chosen layer's entrance during EVERY forward step.
//
// At decode time n_tokens=1 (delta added to one row); at prefill
// n_tokens=prompt_len (same delta added to every row). One shader
// handles both without forcing the caller to tile the buffer.
//
// Phase B: `a` is packed f16 (hidden_buf — 2 per u32). `b` (the delta)
// stays f32 — it's a small per-layer vector, no bandwidth pressure.
// One thread per u32 of `a` (= 2 elements). Total threads halve.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ip = gid.x;
    let n_half = params.n / 2u;
    let total_u32 = n_half * params.n_tokens;
    if (ip >= total_u32) { return; }
    let j_pair = ip % n_half;
    let pair = unpack2x16float(a[ip]);
    let sum_lo = pair.x + b[j_pair * 2u];
    let sum_hi = pair.y + b[j_pair * 2u + 1u];
    a[ip] = pack2x16float(vec2<f32>(sum_lo, sum_hi));
}
