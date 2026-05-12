// Batched ternary matmul (#bn-2). Multi-token analog of the existing
// single-token `ternary_matvec` shader in compute/wgpu_backend.rs. The
// 2-bit weight unpacking is byte-identical to that shader so parity
// tests can compare row-by-row against `ternary_matvec` × scales.
//
// Inputs:
//   - weights         : 2-bit packed ternary [rows, cols] (resident from GpuBitLinear)
//   - activations     : i8 packed (4 per u32) [n_tokens, ceil(cols / 4)]
//                       — output of quantize_absmax_batch
//   - act_scales      : f32 [n_tokens] — per-token absmax scales
//   - params.weight_scale_bits : f32 bitcast to u32 (uniform packing)
//
// Output:
//   - output_mat      : f32 [n_tokens, rows] row-major. Scaling applied
//                       inline: out[tok, row] = i32_acc * act_scale[tok] * weight_scale.
//
// Workgroup layout matches matmul.wgsl: one workgroup per (row, token)
// pair, 256 threads doing strided column accumulation + tree reduction.

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    weight_scale_bits: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read> act_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_mat: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_acc: array<i32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    let tok = wid.z;
    if (row >= params.rows || tok >= params.n_tokens) { return; }
    let tid = lid.x;

    let cols = params.cols;
    let n_u32_per_token = (cols + 3u) / 4u;
    let act_base = tok * n_u32_per_token;

    var acc: i32 = 0;
    var col = tid;
    while (col < cols) {
        // Decode ternary weight at (row, col). Byte-for-byte match with
        // the single-token ternary_matvec decode (wgpu_backend.rs:514-519).
        let flat = row * cols + col;
        let w_byte_idx = flat / 4u;
        let w_bit_shift = (flat % 4u) * 2u;
        let w_u32 = weights[w_byte_idx / 4u];
        let w_byte = (w_u32 >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
        let w_bits = (w_byte >> w_bit_shift) & 3u;

        // Decode i8 activation at (tok, col) with sign extension.
        let act_u32 = activations[act_base + col / 4u];
        let act_byte = (act_u32 >> ((col % 4u) * 8u)) & 0xFFu;
        var act_val: i32 = i32(act_byte);
        if (act_val > 127) { act_val = act_val - 256; }

        // {-1, 0, +1} via conditional add/sub/skip. Encoding (also from
        // ternary_matvec): 0 → -1, 1 → 0, 2 → +1, 3 → 0.
        if (w_bits == 0u) {
            acc -= act_val;
        } else if (w_bits == 2u) {
            acc += act_val;
        }

        col += WG;
    }

    wg_acc[tid] = acc;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_acc[tid] += wg_acc[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        let weight_scale = bitcast<f32>(params.weight_scale_bits);
        let act_scale = act_scales[tok];
        output_mat[tok * params.rows + row] = f32(wg_acc[0]) * act_scale * weight_scale;
    }
}
