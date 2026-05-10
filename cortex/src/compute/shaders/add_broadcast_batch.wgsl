// Broadcast in-place add: every row of a (shape [n_tokens, n]) gets the
// same delta vector b (shape [n]) added. For injection-phase shims
// (#6c) — one [embed_dim] hidden_delta is computed per request and
// applied at the chosen layer's entrance during EVERY forward step.
//
// At decode time n_tokens=1 (delta added to one row); at prefill
// n_tokens=prompt_len (same delta added to every row). One shader
// handles both without forcing the caller to tile the buffer.

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
    a[i] += b[j];
}
