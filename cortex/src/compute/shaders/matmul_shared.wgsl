// Batch matrix multiply with f16-packed weights — SHARED-MEMORY TILED variant.
//
// Computes output[tok, row] = dot(weights[row], input[tok]) for all (tok, row).
// weights: f16-packed [rows, cols/2] as array<u32>
// input:   f32 [n_tokens, cols] (row-major)
// output:  f32 [n_tokens, rows] (row-major)
//
// Design (textbook shared-memory tiled GEMM, the next chapter after the
// register-blocked variant in matmul_tiled.wgsl). For each 16×16 output
// tile, the workgroup cooperatively loads a 16×TILE_K weight tile and a
// 16×TILE_K input tile into shared memory once per K-step, then every
// thread reads from shared memory for its TILE_K dot-product
// accumulation. This is the optimization Zach's reference post deferred;
// llama.cpp's CUDA/Metal kernels use this exact pattern.
//
// **Why this is the right pattern on naga.** The previous tiled attempt
// (matmul_tiled.wgsl) kept an 8×8 accumulator per thread but naga
// didn't auto-unroll the inner MADD loop, so the accumulator lived in
// scratch memory and every MADD paid a memory round-trip. Shared-memory
// tiling moves the reuse from per-thread registers (which naga
// mismanaged) to workgroup-shared storage (which is explicit and
// language-level, not a codegen optimization). Each thread keeps only
// ONE accumulator (its single output element) in private memory — naga
// reliably keeps a single f32 in a register.
//
// **Tile sizes.** TILE_M = TILE_N = 16, TILE_K = 16. Workgroup is
// 16×16 = 256 threads, one per output element. Per K-step, all 256
// threads cooperatively load 16×16 = 256 elements of weights (one per
// thread) and 256 elements of input. With TILE_K = 16, each thread
// then does 16 MADDs against shared memory before the next K-step.
//
// **Dispatch shape.** workgroup_id.x = row tile (0..ceil(rows/16))
//                    workgroup_id.y = token tile (0..ceil(n_tokens/16))
// Caller dispatches ceil(rows/16) × ceil(n_tokens/16) workgroups.
// (Compare: legacy matmul.wgsl dispatches rows × n_tokens workgroups
// — for a 2048×526 matmul that's 1.07M vs our 33 × 128 = 4,224.)

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

// Shared-memory tiles for the current K-step. Loaded cooperatively
// by the workgroup at the start of each K-step; reused by every
// thread for the TILE_K MADDs that follow.
//   a_tile[i][k] = weights[row_base + i, k_base + k]    (dequantized f32)
//   b_tile[j][k] = input_mat[tok_base + j, k_base + k]  (f32)
var<workgroup> a_tile: array<array<f32, TILE_K>, TILE_M>;
var<workgroup> b_tile: array<array<f32, TILE_K>, TILE_N>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row_base: u32 = wid.x * TILE_M;
    let tok_base: u32 = wid.y * TILE_N;

    let li: u32 = lid.x;   // row within tile [0, TILE_M)
    let lj: u32 = lid.y;   // tok within tile [0, TILE_N)

    let rows = params.rows;
    let cols = params.cols;
    let n_tokens = params.n_tokens;
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

        // ---- Cooperative load: A tile (weights) ----
        // 256 threads load 256 weight elements (TILE_M × TILE_K).
        // Each thread (li, lj) loads a_tile[li][lj] from weight
        // row (row_base + li), column (k_base + lj).
        //
        // Weights are f16-packed: each u32 holds 2 f16 values for
        // adjacent columns. We dequantize on load.
        {
            let w_row = row_base + li;
            let w_col = k_base + lj;
            var val: f32 = 0.0;
            if (w_row < rows && w_col < cols) {
                let pair_idx = w_col / 2u;
                let w_off = w_row * half_cols + pair_idx;
                let packed = weights[w_off];
                let upacked = unpack2x16float(packed);
                if ((w_col & 1u) == 0u) {
                    val = upacked.x;
                } else {
                    val = upacked.y;
                }
            }
            a_tile[li][lj] = val;
        }

        // ---- Cooperative load: B tile (input) ----
        // 256 threads load 256 input elements (TILE_N × TILE_K).
        // Each thread (li, lj) loads b_tile[lj][li] from input row
        // (tok_base + lj), column (k_base + li). Using li as the
        // K-index keeps the load coalesced across threads (16
        // adjacent f32s read together).
        {
            let in_tok = tok_base + lj;
            let in_col = k_base + li;
            var val: f32 = 0.0;
            if (in_tok < n_tokens && in_col < cols) {
                val = input_mat[in_tok * cols + in_col];
            }
            b_tile[lj][li] = val;
        }

        workgroupBarrier();

        // ---- Per-thread MADDs against shared memory ----
        // Each thread accumulates TILE_K multiply-adds for its
        // output element. Single f32 accumulator stays in a register
        // (naga doesn't mismanage scalars).
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
