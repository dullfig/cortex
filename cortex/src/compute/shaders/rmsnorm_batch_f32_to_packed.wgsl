// RMSNorm variant: f32 input, packed f16 output.
// Used by the BitNet attention sub-norm (o_sub_norm) in Phase C1+:
// scratch.attn_out is f32 in C1, but its sub-norm output target
// scratch.normed is now packed.

struct Params { n: u32, eps: f32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_sum: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tok = wid.x;
    if (tok >= params.n_tokens) { return; }
    let tid = lid.x;
    let n = params.n;
    let base = tok * n;

    var sq: f32 = 0.0;
    var i = tid;
    while (i < n) {
        let v = input[base + i];
        sq = sq + v * v;
        i += WG;
    }
    wg_sum[tid] = sq;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    let rms = 1.0 / sqrt(wg_sum[0u] / f32(n) + params.eps);

    // Write packed pairs.
    let n_half = n / 2u;
    let base_u32 = tok * n_half;
    var ip = tid;
    while (ip < n_half) {
        let i_lo = ip * 2u;
        let i_hi = i_lo + 1u;
        let lo = input[base + i_lo] * rms * weight[i_lo];
        let hi = input[base + i_hi] * rms * weight[i_hi];
        output[base_u32 + ip] = pack2x16float(vec2<f32>(lo, hi));
        ip += WG;
    }
}
