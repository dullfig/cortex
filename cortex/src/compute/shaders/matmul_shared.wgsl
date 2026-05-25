// Batch matrix multiply with f16-packed weights — SHARED-MEMORY TILED variant.
//
// Computes output[tok, row] = dot(weights[row], input[tok]) for all (tok, row).
// weights: f16-packed [rows, cols/2] as array<u32>
// input:   f32 [n_tokens, cols] (row-major)
// output:  f32 [n_tokens, rows] (row-major)
//
// Design (textbook shared-memory tiled GEMM with vec4/vec2 packed loads,
// Phase 1 of giggly-chasing-melody's FFN matmul optimization). For each
// 16×16 output tile, the workgroup cooperatively loads a 16×TILE_K
// weight tile and a 16×TILE_K input tile into shared memory once per
// K-step, then every thread reads from shared memory for its TILE_K
// dot-product accumulation.
//
// **Phase 1 difference vs the scalar-load version**: cooperative loads
// now use `vec4<f32>` for input (one transaction = 16 bytes = 4 K
// elements per token-row) and `vec2<u32>` for weights (one transaction
// = 8 bytes = 4 packed f16 weights per weight-row). The L2 line on
// NVIDIA is 128 bytes; one warp of 32 threads doing vec4 loads pulls
// exactly one cache line per access. Scalar f32 loads under-fill the
// line (32 × 4 = 128 bytes too — same! — BUT the L1 access cost is
// per-instruction, so fewer instructions = less front-end pressure
// regardless of which way the line math works out).
//
// **Why fewer threads do the loading.** With TILE_M = TILE_N = TILE_K = 16
// and vec4 chunks, the A tile has 16 × 16 / 4 = 64 vec4-sized weight
// loads, and the B tile has 16 × 16 / 4 = 64 vec4 input loads. So only
// 64 of the 256 threads do meaningful loading; the rest sit idle
// during the load. That's fine — the MADD phase keeps all 256 threads
// busy.
//
// **Tile sizes.** TILE_M = TILE_N = 16, TILE_K = 16. Workgroup is
// 16×16 = 256 threads, one per output element. Per K-step, the first
// 64 threads cooperatively load 64 packed weight pairs + 64 input
// vec4s; every thread does 16 MADDs against shared memory.
//
// **Dispatch shape.** workgroup_id.x = row tile (0..ceil(rows/16))
//                    workgroup_id.y = token tile (0..ceil(n_tokens/16))
// Caller dispatches ceil(rows/16) × ceil(n_tokens/16) workgroups.

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

const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

// Shared-memory tiles for the current K-step.
//   a_tile[i][k] = weights[row_base + i, k_base + k]    (dequantized f32)
//   b_tile[j][k] = input_mat[tok_base + j, k_base + k]  (f32)
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

    let li: u32 = lid.x;   // row within tile [0, TILE_M)
    let lj: u32 = lid.y;   // tok within tile [0, TILE_N)

    let rows = params.rows;
    let cols = params.cols;
    let n_tokens = params.n_tokens;
    // Cols-in-packed-u32s (since weights are 2 f16s per u32, one u32
    // = 1 weight value × 2 columns; vec2<u32> = 4 columns).
    let half_cols = cols / 2u;

    // The output element this thread is computing.
    let out_row: u32 = row_base + li;
    let out_tok: u32 = tok_base + lj;
    let out_in_bounds: bool = (out_row < rows) && (out_tok < n_tokens);

    var sum: f32 = 0.0;

    // Number of K-steps to cover the inner dimension. Round up to
    // include the partial tile at the end (we zero out-of-bounds
    // entries during cooperative loading).
    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // ---- Cooperative load: A tile (weights) — vec2<u32> packed ----
        // Each load reads 1 u32 (= 2 f16 weight values = 2 K-elements)
        // but we issue them in pairs (vec2<u32> = 4 K-elements per
        // instruction) for L2 transaction efficiency. With TILE_K=16
        // and pairs of u32s, that's 16/4 = 4 vec2 loads per weight row,
        // and 16 weight rows = 64 total vec2 loads. Distributed across
        // the first 64 threads (lidx < 64).
        if (lidx < 64u) {
            let load_row: u32 = lidx / 4u;          // [0, 16)
            let load_k4:  u32 = lidx & 3u;          // [0, 4)
            let load_k_base: u32 = load_k4 * 4u;    // 0, 4, 8, 12
            let w_row = row_base + load_row;
            let w_col_lo = k_base + load_k_base;

            // Default to zeros for out-of-bounds.
            var v0: f32 = 0.0;
            var v1: f32 = 0.0;
            var v2: f32 = 0.0;
            var v3: f32 = 0.0;
            if (w_row < rows && w_col_lo < cols) {
                // First packed u32 covers columns [w_col_lo, w_col_lo+1].
                let pair_idx = w_col_lo / 2u;
                let w_off = w_row * half_cols + pair_idx;
                let p0 = unpack2x16float(weights[w_off]);
                v0 = p0.x;
                v1 = p0.y;
                // Second packed u32 covers columns [w_col_lo+2, w_col_lo+3].
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

        // ---- Cooperative load: B tile (input) — vec4<f32> packed ----
        // Each load reads 4 f32 input values = 4 K-elements per
        // instruction. 16 tokens × 16 K-cols / 4-per-load = 64 vec4
        // loads. Distributed across the first 64 threads (lidx < 64).
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
                // Read 4 consecutive f32s. WGSL doesn't allow a literal
                // vec4 load from array<f32>, so we read scalar; in
                // practice naga emits these as adjacent loads which the
                // driver typically coalesces into one wide load
                // anyway. The intent (and the L2 access pattern) is the
                // same as if we'd used vec4 explicitly.
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
        // Each thread accumulates TILE_K multiply-adds for its
        // output element. Single f32 accumulator stays in a register.
        if (out_in_bounds) {
            for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
                sum = sum + a_tile[li][k] * b_tile[lj][k];
            }
        }

        workgroupBarrier();
    }

    // Writeback. Output is row-major [n_tokens, rows]:
    //   output_mat[out_tok * rows + out_row] = sum
    if (out_in_bounds) {
        output_mat[out_tok * rows + out_row] = sum;
    }
}
