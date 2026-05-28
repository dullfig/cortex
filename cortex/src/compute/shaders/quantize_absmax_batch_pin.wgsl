// Per-token absmax quantization — PACKED f16 input variant (Phase C1).
// Input: [n_tokens, cols/2] u32 packed f16 (scratch.normed Phase C1+).
// Output (i8 packed + scales): same layout as quantize_absmax_batch.wgsl.

struct Params { cols: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_max: array<f32, 256>;
var<workgroup> wg_scale: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let token = wid.x;
    if (token >= params.n_tokens) { return; }
    let tid = lid.x;
    let cols = params.cols;
    let cols_half = cols / 2u;
    let in_base_u32 = token * cols_half;

    var local_max: f32 = 0.0;
    var ip = tid;
    while (ip < cols_half) {
        let pair = unpack2x16float(input[in_base_u32 + ip]);
        let m = max(abs(pair.x), abs(pair.y));
        if (m > local_max) { local_max = m; }
        ip += WG;
    }
    wg_max[tid] = local_max;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) {
            let other = wg_max[tid + s];
            if (other > wg_max[tid]) { wg_max[tid] = other; }
        }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        let max_abs = wg_max[0u];
        var scale: f32 = max_abs / 127.0;
        if (scale == 0.0) { scale = 1.0; }
        output_scales[token] = scale;
        wg_scale = scale;
    }
    workgroupBarrier();
    let scale = wg_scale;

    let n_u32_per_token = (cols + 3u) / 4u;
    let out_base = token * n_u32_per_token;

    var u = tid;
    while (u < n_u32_per_token) {
        var packed: u32 = 0u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let col = u * 4u + k;
            var qi: i32 = 0;
            if (col < cols) {
                let u32_idx = in_base_u32 + col / 2u;
                let pair = unpack2x16float(input[u32_idx]);
                let v = select(pair.x, pair.y, (col & 1u) == 1u);
                var q = i32(round(v / scale));
                if (q < -127) { q = -127; }
                if (q > 127) { q = 127; }
                qi = q;
            }
            let byte = u32(qi & 0xFF);
            packed |= byte << (k * 8u);
        }
        output_q[out_base + u] = packed;
        u += WG;
    }
}
