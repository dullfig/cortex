// Per-block RMSNorm variant: reads packed-f16 input (hidden_buf, Phase B)
// and writes f32 output (scratch.normed, still f32 in Phase B until
// Phase C also packs the per-block scratch).
//
// Used by attn_norm and ffn_norm inside forward_block_gpu_inner. The
// "all-packed" variant (rmsnorm_batch.wgsl) is used for the final norm
// where input AND output are packed (hidden_buf → normed_buf at end
// of forward).
//
// Bitnet sub-norms (o_sub_norm, ffn_sub_norm) still take f32 input
// (scratch.attn_out, scratch.activated) and write f32 output (scratch.normed,
// scratch.up) — those keep using the legacy f32-in/f32-out shader
// available via a separate path in Phase C. For now Phase B doesn't
// route through this shader for sub_norms (sub_norms are still f32→f32).

struct Params { n: u32, eps: f32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
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
    let in_base_u32 = tok * n_half;
    let out_base_f32 = tok * params.n;

    var sq: f32 = 0.0;
    var ip = tid;
    while (ip < n_half) {
        let pair = unpack2x16float(input[in_base_u32 + ip]);
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
        let pair = unpack2x16float(input[in_base_u32 + ip]);
        let i_lo = ip * 2u;
        let i_hi = i_lo + 1u;
        output[out_base_f32 + i_lo] = pair.x * rms * weight[i_lo];
        output[out_base_f32 + i_hi] = pair.y * rms * weight[i_hi];
        ip += WG;
    }
}
