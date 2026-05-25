// Batch matrix multiply with f16-packed weights — SHARED-MEMORY TILED variant.
//
// Computes output[tok, row] = dot(weights[row], input[tok]) for all (tok, row).
// weights: f16-packed [rows, cols/2] as array<u32>
// input:   f32 [n_tokens, cols] (row-major)
// output:  f32 [n_tokens, rows] (row-major)
//
// Design (textbook shared-memory tiled GEMM with vec4 cooperative loads,
// 2-per-thread register block — Phases 1+2 of giggly-chasing-melody).
// For each 32×16 output tile, the workgroup cooperatively loads a
// 32×TILE_K weight tile and a 16×TILE_K input tile into shared
// memory once per K-step, then every thread accumulates TWO output
// elements (stride-16 apart in the M dimension) from shared memory.
//
// **Phase 2 vs Phase 1**: TILE_M doubled from 16 to 32. Each of the 256
// threads now computes 2 output elements (sum_a + sum_b) instead of 1,
// halving the number of dispatched workgroups (for Qwen 0.5B FFN at
// 4864 rows × 526 tokens: 152 × 33 = 5016 WGs Phase-1 vs 76 × 33 =
// 2508 WGs Phase-2). Each K-step still loads 512 weights + 256 inputs
// into shared memory, but those serve 512 outputs instead of 256 —
// 33% less memory traffic per output. Two scalar f32 accumulators
// per thread stays well inside naga's register budget (the failed
// earlier attempt at register blocking used an 8×8 array which naga
// spilled; 2 scalars are fine).
//
// **Tile sizes.** TILE_M = 32, TILE_N = 16, TILE_K = 16. Workgroup is
// 16×16 = 256 threads; each thread computes 2 output rows at
// (row_base + li, row_base + 16 + li) for token (tok_base + lj).
//
// **Dispatch shape.** workgroup_id.x = M tile (0..ceil(rows/32))
//                    workgroup_id.y = N tile (0..ceil(n_tokens/16))
// Caller dispatches ceil(rows/32) × ceil(n_tokens/16) workgroups.

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_mat: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_mat: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

// Shared-memory tiles for the current K-step.
//   a_tile[i][k] = weights[row_base + i, k_base + k]    (dequantized f32, i in [0, 32))
//   b_tile[j][k] = input_mat[tok_base + j, k_base + k]  (f32, j in [0, 16))
var<workgroup> a_tile: array<array<f32, TILE_K>, TILE_M>;
var<workgroup> b_tile: array<array<f32, TILE_K>, TILE_N>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let row_base: u32 = wid.x * TILE_M;
    let tok_base: u32 = wid.y * TILE_N;

    let li: u32 = lid.x;   // row offset within first half of tile [0, 16)
    let lj: u32 = lid.y;   // tok within tile [0, TILE_N)

    let rows = params.rows;
    let cols = params.cols;
    let n_tokens = params.n_tokens;
    let half_cols = cols / 2u;

    // This thread's TWO output elements (stride-16 in M).
    let out_row_a: u32 = row_base + li;
    let out_row_b: u32 = row_base + 16u + li;
    let out_tok:   u32 = tok_base + lj;
    let a_in_bounds: bool = (out_row_a < rows) && (out_tok < n_tokens);
    let b_in_bounds: bool = (out_row_b < rows) && (out_tok < n_tokens);

    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;

    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // ---- Cooperative load: A tile (weights) ----
        // 32 rows × 16 K-cols = 512 elements per K-step. Each load
        // reads 2 packed u32s (= 4 f16 weights = 4 K-elements per
        // packed pair). 32 rows × 4 chunks = 128 logical loads,
        // distributed across the first 128 threads (lidx < 128).
        if (lidx < 128u) {
            let load_row: u32 = lidx / 4u;          // [0, 32)
            let load_k4:  u32 = lidx & 3u;          // [0, 4)
            let load_k_base: u32 = load_k4 * 4u;    // 0, 4, 8, 12
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

        // ---- Cooperative load: B tile (input) ----
        // 16 tokens × 16 K-cols = 256 elements. 4 f32s per load = 64
        // logical loads. Distributed across the first 64 threads.
        if (lidx < 64u) {
            let load_tok: u32 = lidx / 4u;          // [0, 16)
            let load_k4:  u32 = lidx & 3u;          // [0, 4)
            let load_k_base: u32 = load_k4 * 4u;    // 0, 4, 8, 12
            let in_tok = tok_base + load_tok;
            let in_col = k_base + load_k_base;

            var v0: f32 = 0.0;
            var v1: f32 = 0.0;
            var v2: f32 = 0.0;
            var v3: f32 = 0.0;
            if (in_tok < n_tokens && in_col < cols) {
                let base = in_tok * cols + in_col;
                v0 = input_mat[base + 0u];
                if (in_col + 1u < cols) { v1 = input_mat[base + 1u]; }
                if (in_col + 2u < cols) { v2 = input_mat[base + 2u]; }
                if (in_col + 3u < cols) { v3 = input_mat[base + 3u]; }
            }
            b_tile[load_tok][load_k_base + 0u] = v0;
            b_tile[load_tok][load_k_base + 1u] = v1;
            b_tile[load_tok][load_k_base + 2u] = v2;
            b_tile[load_tok][load_k_base + 3u] = v3;
        }

        workgroupBarrier();

        // ---- Per-thread MADDs against shared memory ----
        // 16 explicit MADD pairs (hand-unrolled to keep naga from
        // generating a loop the driver may not unroll). Each iteration
        // does 2 multiplies sharing one b_val and accumulates into two
        // scalar f32 registers.
        let li_b: u32 = 16u + li;
        sum_a = sum_a + a_tile[li][0u]  * b_tile[lj][0u];
        sum_b = sum_b + a_tile[li_b][0u] * b_tile[lj][0u];
        sum_a = sum_a + a_tile[li][1u]  * b_tile[lj][1u];
        sum_b = sum_b + a_tile[li_b][1u] * b_tile[lj][1u];
        sum_a = sum_a + a_tile[li][2u]  * b_tile[lj][2u];
        sum_b = sum_b + a_tile[li_b][2u] * b_tile[lj][2u];
        sum_a = sum_a + a_tile[li][3u]  * b_tile[lj][3u];
        sum_b = sum_b + a_tile[li_b][3u] * b_tile[lj][3u];
        sum_a = sum_a + a_tile[li][4u]  * b_tile[lj][4u];
        sum_b = sum_b + a_tile[li_b][4u] * b_tile[lj][4u];
        sum_a = sum_a + a_tile[li][5u]  * b_tile[lj][5u];
        sum_b = sum_b + a_tile[li_b][5u] * b_tile[lj][5u];
        sum_a = sum_a + a_tile[li][6u]  * b_tile[lj][6u];
        sum_b = sum_b + a_tile[li_b][6u] * b_tile[lj][6u];
        sum_a = sum_a + a_tile[li][7u]  * b_tile[lj][7u];
        sum_b = sum_b + a_tile[li_b][7u] * b_tile[lj][7u];
        sum_a = sum_a + a_tile[li][8u]  * b_tile[lj][8u];
        sum_b = sum_b + a_tile[li_b][8u] * b_tile[lj][8u];
        sum_a = sum_a + a_tile[li][9u]  * b_tile[lj][9u];
        sum_b = sum_b + a_tile[li_b][9u] * b_tile[lj][9u];
        sum_a = sum_a + a_tile[li][10u] * b_tile[lj][10u];
        sum_b = sum_b + a_tile[li_b][10u] * b_tile[lj][10u];
        sum_a = sum_a + a_tile[li][11u] * b_tile[lj][11u];
        sum_b = sum_b + a_tile[li_b][11u] * b_tile[lj][11u];
        sum_a = sum_a + a_tile[li][12u] * b_tile[lj][12u];
        sum_b = sum_b + a_tile[li_b][12u] * b_tile[lj][12u];
        sum_a = sum_a + a_tile[li][13u] * b_tile[lj][13u];
        sum_b = sum_b + a_tile[li_b][13u] * b_tile[lj][13u];
        sum_a = sum_a + a_tile[li][14u] * b_tile[lj][14u];
        sum_b = sum_b + a_tile[li_b][14u] * b_tile[lj][14u];
        sum_a = sum_a + a_tile[li][15u] * b_tile[lj][15u];
        sum_b = sum_b + a_tile[li_b][15u] * b_tile[lj][15u];

        workgroupBarrier();
    }

    // Writeback. Output is row-major [n_tokens, rows]:
    //   output_mat[out_tok * rows + out_row] = sum
    if (a_in_bounds) {
        output_mat[out_tok * rows + out_row_a] = sum_a;
    }
    if (b_in_bounds) {
        output_mat[out_tok * rows + out_row_b] = sum_b;
    }
}
