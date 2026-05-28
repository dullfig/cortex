// Phase C2 decode-path matmul: packed-f16 input, packed-f16 output.
// Used when scratch.normed is packed (C1) AND the output buffer is
// packed (C2: scratch.gate/up/activated). Two threads compute adjacent
// output rows (row, row+1) and one of them packs both into the shared
// output u32 slot via atomicOr-style splitting — but here we keep it
// simple: each workgroup computes ONE output element (per-output
// dispatch shape from matmul_pin), then thread 0 reads its
// partner-row from an adjacent workgroup via a shared temp slot.
//
// Actually simpler: replicate matmul_pin (one WG per (row, tok)) but
// pair adjacent rows by having even-row WGs write both halves of the
// u32 after reading the odd-row's sum via a small scratch buffer.
// That cross-WG sync is fragile.
//
// Cleanest: change the WG layout so each WG computes a (row_pair, tok)
// pair — TWO output rows per WG with TWO accumulators per thread. WG
// dispatch shape becomes (rows/2 × n_tokens) instead of (rows × n_tokens).
// 256 threads, each accumulating sum_a and sum_b for the same K-stride.
// Pack at the end.

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_mat: array<u32>;  // packed f16
@group(0) @binding(2) var<storage, read_write> output_mat: array<u32>; // packed f16
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_sum_a: array<f32, WG>;
var<workgroup> wg_sum_b: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row_pair = wid.x + wid.y * 65535u;  // index into row pairs
    let tok = wid.z;
    let rows = params.rows;
    let row_a = row_pair * 2u;
    let row_b = row_pair * 2u + 1u;
    if (row_a >= rows || tok >= params.n_tokens) { return; }
    let tid = lid.x;

    let half_cols = params.cols / 2u;
    let w_base_a = row_a * half_cols;
    let w_base_b = row_b * half_cols;
    let in_base_u32 = tok * half_cols;

    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    var i = tid;
    while (i < half_cols) {
        let a = unpack2x16float(input_mat[in_base_u32 + i]);
        let wa = unpack2x16float(weights[w_base_a + i]);
        sum_a += wa.x * a.x + wa.y * a.y;
        if (row_b < rows) {
            let wb = unpack2x16float(weights[w_base_b + i]);
            sum_b += wb.x * a.x + wb.y * a.y;
        }
        i += WG;
    }

    wg_sum_a[tid] = sum_a;
    wg_sum_b[tid] = sum_b;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) {
            wg_sum_a[tid] += wg_sum_a[tid + s];
            wg_sum_b[tid] += wg_sum_b[tid + s];
        }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        let rows_half = rows / 2u;
        let pair_idx = tok * rows_half + row_pair;
        let final_a = wg_sum_a[0u];
        let final_b = select(0.0, wg_sum_b[0u], row_b < rows);
        output_mat[pair_idx] = pack2x16float(vec2<f32>(final_a, final_b));
    }
}
