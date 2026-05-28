// Decode-path matmul (one workgroup per (row, token)) — packed-f16 input.
// Mirror of matmul.wgsl with input as array<u32> instead of array<f32>.
// Used when n_tokens < 16 (matmul_shared's tile doesn't amortize) and
// the input buffer is packed (Phase C1+ scratch.normed).

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_mat: array<u32>;  // packed f16
@group(0) @binding(2) var<storage, read_write> output_mat: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_sum: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    let tok = wid.z;
    if (row >= params.rows || tok >= params.n_tokens) { return; }
    let tid = lid.x;

    let half_cols = params.cols / 2u;
    let w_base = row * half_cols;
    let in_base_u32 = tok * half_cols;

    var sum: f32 = 0.0;
    var i = tid;
    while (i < half_cols) {
        let w = unpack2x16float(weights[w_base + i]);
        let a = unpack2x16float(input_mat[in_base_u32 + i]);
        sum += w.x * a.x + w.y * a.y;
        i += WG;
    }

    wg_sum[tid] = sum;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        output_mat[tok * params.rows + row] = wg_sum[0u];
    }
}
