// RMSNorm variant: both input AND output are packed f16 (2 per u32).
// Used by the FINAL norm (hidden_buf packed → normed_buf packed) at
// the end of forward, where both buffers are packed in Phase B.

struct Params { n: u32, eps: f32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
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
    let n_half = params.n / 2u;
    let base_u32 = tok * n_half;

    var sq: f32 = 0.0;
    var ip = tid;
    while (ip < n_half) {
        let pair = unpack2x16float(input[base_u32 + ip]);
        sq = sq + pair.x * pair.x + pair.y * pair.y;
        ip += WG;
    }
    wg_sum[tid] = sq;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    let rms = 1.0 / sqrt(wg_sum[0u] / f32(params.n) + params.eps);

    ip = tid;
    while (ip < n_half) {
        let pair = unpack2x16float(input[base_u32 + ip]);
        let i_lo = ip * 2u;
        let i_hi = i_lo + 1u;
        let out_lo = pair.x * rms * weight[i_lo];
        let out_hi = pair.y * rms * weight[i_hi];
        output[base_u32 + ip] = pack2x16float(vec2<f32>(out_lo, out_hi));
        ip += WG;
    }
}
