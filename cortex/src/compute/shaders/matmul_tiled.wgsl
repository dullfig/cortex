// Batch matrix multiply with f16-packed weights — TILED variant.
// Computes output[tok, row] = dot(weights[row], input[tok]) for all (tok, row) at once.
// weights: f16-packed [rows, cols/2] as array<u32>
// input:   f32 [n_tokens, cols] (row-major)
// output:  f32 [n_tokens, rows] (row-major)
//
// Design (Zach Nussbaum's WebGPU matmul pattern, adapted for cortex's
// f16-packed weights): each workgroup is 16×16=256 threads. Each thread
// holds an 8×8 register-blocked accumulator covering 8 tokens × 8 output
// rows. One workgroup therefore produces a 128×128 block of the output
// (16 threads × 8 tile per axis on each side). The inner K loop strides
// over column pairs (one u32 of packed weights = 2 f32 cols), unrolling
// the 8×8 multiply-add naturally because the bounds are compile-time
// constants — WGSL/naga compilers (Vulkan / Metal both verified by Zach)
// emit straight-line code with no loop-control branches in the inner
// quadratic.
//
// Tile sizes chosen for the 4080 Laptop's SM occupancy at f32 register
// pressure. Hardcoded for now; future work will autotune per-shape /
// per-GPU and ship a small lookup table at startup.
//
// Decode path (n_tokens < TILE_M=8): caller routes back to the legacy
// matmul.wgsl which is one-output-per-workgroup with tree reduction —
// fine for n_tokens=1, where register blocking has nothing to amortize.

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

const BLOCKSIZE: u32 = 16u;
const TILE_M: u32 = 8u;
const TILE_N: u32 = 8u;

// Each workgroup spans 16 threads × 16 threads × (8×8 tile each)
// = 128 tokens × 128 output rows.
// global_invocation_id.x in [0, rows / TILE_N)
// global_invocation_id.y in [0, n_tokens / TILE_M)
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tok_base: u32 = gid.y * TILE_M;
    let row_base: u32 = gid.x * TILE_N;

    let cols = params.cols;
    let half_cols = cols / 2u;
    let rows = params.rows;
    let n_tokens = params.n_tokens;

    // Register-blocked accumulator: TILE_M tokens × TILE_N rows = 64 f32s
    // held in private memory. Compiler keeps these in registers (verified
    // on Vulkan/Metal in the source post; on naga we trust the same since
    // bounds are compile-time constants).
    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
            sums[i][j] = 0.0;
        }
    }

    // Inner K loop: stride by column pairs. Each iteration reads
    // TILE_M activation pairs (2 f32 each) and TILE_N weight u32s
    // (2 f16 each via unpack2x16float), then accumulates TILE_M ×
    // TILE_N × 2 multiply-adds. With cols = 2048 (Qwen Q proj) →
    // 1024 outer iterations, each doing 128 MADDs → ~130k MADDs per
    // thread → ~8M per workgroup of 64 outputs → matches GPU SIMD width.
    for (var kp: u32 = 0u; kp < half_cols; kp = kp + 1u) {
        // Activations: 2 f32 values per token at column-pair kp.
        var a_lo: array<f32, TILE_M>;
        var a_hi: array<f32, TILE_M>;
        for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
            let tok = tok_base + i;
            // Bounds-clamp via select to keep dispatch simple — out-of-bounds
            // tokens contribute zero (no effect on in-bounds output rows).
            let in_off = tok * cols + kp * 2u;
            if (tok < n_tokens) {
                a_lo[i] = input_mat[in_off];
                a_hi[i] = input_mat[in_off + 1u];
            } else {
                a_lo[i] = 0.0;
                a_hi[i] = 0.0;
            }
        }

        // Weights: one u32 per output row at column-pair kp, holds 2
        // f16 values in (.x, .y).
        for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
            let row = row_base + j;
            if (row >= rows) { continue; }
            let w_off = row * half_cols + kp;
            let w = unpack2x16float(weights[w_off]);

            // 8 multiply-adds (one per token in this tile) per
            // (output_row, column_pair). Inner unroll is compile-time
            // because TILE_M is a const u32.
            for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
                sums[i][j] = sums[i][j] + a_lo[i] * w.x + a_hi[i] * w.y;
            }
        }
    }

    // Writeback. Bounds-check per element since the workgroup grid
    // is rounded up — partial tiles at the edges write only the
    // valid (token, row) entries.
    for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        let tok = tok_base + i;
        if (tok >= n_tokens) { continue; }
        let out_row_base = tok * rows;
        for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
            let row = row_base + j;
            if (row < rows) {
                output_mat[out_row_base + row] = sums[i][j];
            }
        }
    }
}
