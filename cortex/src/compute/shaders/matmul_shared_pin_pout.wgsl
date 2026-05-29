// Phase C3 matmul: packed-f16 input AND packed-f16 output. Prefill
// variant. Composes the C1 packed-input pattern (from
// matmul_shared_pin_fout) with the C2 adjacent-pair output packing
// (from matmul_gate_up_shared), plus the C2 bank-conflict padding
// (TILE_K_P=17). Used for Q/K/V/O/down projections when scratch.q/k/v/
// attn_out/projected are packed (C3) and n_tokens >= 16.

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

const TILE_M: u32 = 32u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;
// Padded stride (17) to dodge shared-memory bank conflicts on the
// adjacent-pair (li*2, li*2+1) MADD access pattern. See
// matmul_gate_up_shared.wgsl for the detailed analysis.
const TILE_K_P: u32 = 17u;

var<workgroup> a_tile: array<array<f32, TILE_K_P>, TILE_M>;
var<workgroup> b_tile: array<array<f32, TILE_K_P>, TILE_N>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let row_base: u32 = wid.x * TILE_M;
    let tok_base: u32 = wid.y * TILE_N;

    let li: u32 = lid.x;
    let lj: u32 = lid.y;

    let rows = params.rows;
    let cols = params.cols;
    let n_tokens = params.n_tokens;
    let half_cols = cols / 2u;

    // Adjacent-pair row mapping (li*2, li*2+1).
    let out_row_a: u32 = row_base + li * 2u;
    let out_row_b: u32 = row_base + li * 2u + 1u;
    let out_tok:   u32 = tok_base + lj;
    let a_in_bounds: bool = (out_row_a < rows) && (out_tok < n_tokens);
    let b_in_bounds: bool = (out_row_b < rows) && (out_tok < n_tokens);

    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;

    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // Cooperative load: A tile (weights, packed f16). 32 rows ×
        // 16 K-cols / 4-per-load = 128 logical loads.
        if (lidx < 128u) {
            let load_row: u32 = lidx / 4u;
            let load_k4:  u32 = lidx & 3u;
            let load_k_base: u32 = load_k4 * 4u;
            let w_row = row_base + load_row;
            let w_col_lo = k_base + load_k_base;

            var v0: f32 = 0.0;
            var v1: f32 = 0.0;
            var v2: f32 = 0.0;
            var v3: f32 = 0.0;
            if (w_row < rows && w_col_lo < cols) {
                let pair_idx = w_col_lo / 2u;
                let w_off = w_row * half_cols + pair_idx;
                let p0 = unpack2x16float(weights[w_off]);
                v0 = p0.x;
                v1 = p0.y;
                if (w_col_lo + 2u < cols) {
                    let p1 = unpack2x16float(weights[w_off + 1u]);
                    v2 = p1.x;
                    v3 = p1.y;
                }
            }
            a_tile[load_row][load_k_base + 0u] = v0;
            a_tile[load_row][load_k_base + 1u] = v1;
            a_tile[load_row][load_k_base + 2u] = v2;
            a_tile[load_row][load_k_base + 3u] = v3;
        }

        // Cooperative load: B tile (input, packed f16). 16 tokens ×
        // 16 K-cols / 4-per-load = 64 logical loads.
        if (lidx < 64u) {
            let load_tok: u32 = lidx / 4u;
            let load_k4:  u32 = lidx & 3u;
            let load_k_base: u32 = load_k4 * 4u;
            let in_tok = tok_base + load_tok;
            let in_col = k_base + load_k_base;

            var v0: f32 = 0.0;
            var v1: f32 = 0.0;
            var v2: f32 = 0.0;
            var v3: f32 = 0.0;
            if (in_tok < n_tokens && in_col < cols) {
                let base_u32 = (in_tok * cols + in_col) / 2u;
                let p0 = unpack2x16float(input_mat[base_u32]);
                v0 = p0.x;
                v1 = p0.y;
                if (in_col + 2u < cols) {
                    let p1 = unpack2x16float(input_mat[base_u32 + 1u]);
                    v2 = p1.x;
                    v3 = p1.y;
                }
            }
            b_tile[load_tok][load_k_base + 0u] = v0;
            b_tile[load_tok][load_k_base + 1u] = v1;
            b_tile[load_tok][load_k_base + 2u] = v2;
            b_tile[load_tok][load_k_base + 3u] = v3;
        }

        workgroupBarrier();

        // Adjacent-pair MADDs.
        let li_a: u32 = li * 2u;
        let li_b: u32 = li * 2u + 1u;
        sum_a = sum_a + a_tile[li_a][0u]  * b_tile[lj][0u];
        sum_b = sum_b + a_tile[li_b][0u]  * b_tile[lj][0u];
        sum_a = sum_a + a_tile[li_a][1u]  * b_tile[lj][1u];
        sum_b = sum_b + a_tile[li_b][1u]  * b_tile[lj][1u];
        sum_a = sum_a + a_tile[li_a][2u]  * b_tile[lj][2u];
        sum_b = sum_b + a_tile[li_b][2u]  * b_tile[lj][2u];
        sum_a = sum_a + a_tile[li_a][3u]  * b_tile[lj][3u];
        sum_b = sum_b + a_tile[li_b][3u]  * b_tile[lj][3u];
        sum_a = sum_a + a_tile[li_a][4u]  * b_tile[lj][4u];
        sum_b = sum_b + a_tile[li_b][4u]  * b_tile[lj][4u];
        sum_a = sum_a + a_tile[li_a][5u]  * b_tile[lj][5u];
        sum_b = sum_b + a_tile[li_b][5u]  * b_tile[lj][5u];
        sum_a = sum_a + a_tile[li_a][6u]  * b_tile[lj][6u];
        sum_b = sum_b + a_tile[li_b][6u]  * b_tile[lj][6u];
        sum_a = sum_a + a_tile[li_a][7u]  * b_tile[lj][7u];
        sum_b = sum_b + a_tile[li_b][7u]  * b_tile[lj][7u];
        sum_a = sum_a + a_tile[li_a][8u]  * b_tile[lj][8u];
        sum_b = sum_b + a_tile[li_b][8u]  * b_tile[lj][8u];
        sum_a = sum_a + a_tile[li_a][9u]  * b_tile[lj][9u];
        sum_b = sum_b + a_tile[li_b][9u]  * b_tile[lj][9u];
        sum_a = sum_a + a_tile[li_a][10u] * b_tile[lj][10u];
        sum_b = sum_b + a_tile[li_b][10u] * b_tile[lj][10u];
        sum_a = sum_a + a_tile[li_a][11u] * b_tile[lj][11u];
        sum_b = sum_b + a_tile[li_b][11u] * b_tile[lj][11u];
        sum_a = sum_a + a_tile[li_a][12u] * b_tile[lj][12u];
        sum_b = sum_b + a_tile[li_b][12u] * b_tile[lj][12u];
        sum_a = sum_a + a_tile[li_a][13u] * b_tile[lj][13u];
        sum_b = sum_b + a_tile[li_b][13u] * b_tile[lj][13u];
        sum_a = sum_a + a_tile[li_a][14u] * b_tile[lj][14u];
        sum_b = sum_b + a_tile[li_b][14u] * b_tile[lj][14u];
        sum_a = sum_a + a_tile[li_a][15u] * b_tile[lj][15u];
        sum_b = sum_b + a_tile[li_b][15u] * b_tile[lj][15u];

        workgroupBarrier();
    }

    // Pack adjacent (row, row+1) into one u32 output slot.
    let rows_half = rows / 2u;
    let pair_idx = out_tok * rows_half + (row_base / 2u + li);
    if (a_in_bounds && b_in_bounds) {
        output_mat[pair_idx] = pack2x16float(vec2<f32>(sum_a, sum_b));
    } else if (a_in_bounds) {
        output_mat[pair_idx] = pack2x16float(vec2<f32>(sum_a, 0.0));
    }
}
