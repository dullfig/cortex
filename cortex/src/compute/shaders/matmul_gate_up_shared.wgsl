// Fused gate + up SwiGLU projection — SHARED-MEMORY TILED variant.
//
// Computes BOTH gate_out and up_out from the same input in one
// dispatch:
//   gate_out[tok, row] = dot(gate_weights[row], input[tok])
//   up_out[tok, row]   = dot(up_weights[row],   input[tok])
//
// gate_weights, up_weights: f16-packed [rows, cols/2] as array<u32>
// input: f32 [n_tokens, cols] (row-major)
// gate_out, up_out: f32 [n_tokens, rows] (row-major)
//
// **Why fuse.** In SwiGLU FFN, gate_proj and up_proj read the SAME
// input (post-ffn_norm hidden) and produce parallel outputs that get
// element-wise combined as `silu(gate) * up`. Running them as two
// separate matmul dispatches loads the input from HBM TWICE — once
// for each. The fused shader loads input into shared memory once per
// K-step and accumulates into FOUR outputs per thread:
//   sum_gate_a, sum_gate_b — two rows of gate_out
//   sum_up_a,   sum_up_b   — two rows of up_out
// Halves the input HBM bandwidth for gate+up combined.
//
// **Design.** Same 32×16 output tile as the Phase 2 matmul_shared
// (TILE_M=32 stride-16 in M, TILE_N=16, TILE_K=16). 256 threads,
// each producing 4 output elements. Two `a_tile`s in shared memory
// (one for gate weights, one for up), one `b_tile` for input.
// Manually unrolled 16-MADD inner loop (32 explicit MADDs per
// projection × 2 projections = 64 MADDs). Naga's loop generation
// is what cost us the unroll in Phase 2 — keeping it explicit here.
//
// **Tile sizes.** TILE_M = 32, TILE_N = 16, TILE_K = 16. Workgroup
// is 16×16 = 256 threads.
//
// **Dispatch shape.** workgroup_id.x = M tile (0..ceil(rows/32))
//                    workgroup_id.y = N tile (0..ceil(n_tokens/16))
// Caller dispatches ceil(rows/32) × ceil(n_tokens/16) workgroups.
//
// **Shared memory budget.** 2 × (32×16 f32) + 1 × (16×16 f32) = 5 KB
// per WG. NVIDIA SMs have 100+ KB shared; this leaves plenty of
// occupancy headroom.

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(1) var<storage, read> up_weights:   array<u32>;
@group(0) @binding(2) var<storage, read> input_mat:    array<f32>;
@group(0) @binding(3) var<storage, read_write> gate_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> up_out:   array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

var<workgroup> a_gate_tile: array<array<f32, TILE_K>, TILE_M>;
var<workgroup> a_up_tile:   array<array<f32, TILE_K>, TILE_M>;
var<workgroup> b_tile:      array<array<f32, TILE_K>, TILE_N>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let row_base: u32 = wid.x * TILE_M;
    let tok_base: u32 = wid.y * TILE_N;

    let li: u32 = lid.x;   // row offset within first half [0, 16)
    let lj: u32 = lid.y;   // tok within tile [0, TILE_N)

    let rows = params.rows;
    let cols = params.cols;
    let n_tokens = params.n_tokens;
    let half_cols = cols / 2u;

    let out_row_a: u32 = row_base + li;
    let out_row_b: u32 = row_base + 16u + li;
    let out_tok:   u32 = tok_base + lj;
    let a_in_bounds: bool = (out_row_a < rows) && (out_tok < n_tokens);
    let b_in_bounds: bool = (out_row_b < rows) && (out_tok < n_tokens);

    var sum_gate_a: f32 = 0.0;
    var sum_gate_b: f32 = 0.0;
    var sum_up_a:   f32 = 0.0;
    var sum_up_b:   f32 = 0.0;

    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // ---- Cooperative load: gate-weight A tile ----
        // 32 rows × 16 K-cols / 4-per-load = 128 logical loads,
        // distributed across the first 128 threads.
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
                let p0 = unpack2x16float(gate_weights[w_off]);
                v0 = p0.x;
                v1 = p0.y;
                if (w_col_lo + 2u < cols) {
                    let p1 = unpack2x16float(gate_weights[w_off + 1u]);
                    v2 = p1.x;
                    v3 = p1.y;
                }
            }
            a_gate_tile[load_row][load_k_base + 0u] = v0;
            a_gate_tile[load_row][load_k_base + 1u] = v1;
            a_gate_tile[load_row][load_k_base + 2u] = v2;
            a_gate_tile[load_row][load_k_base + 3u] = v3;
        }

        // ---- Cooperative load: up-weight A tile ----
        // Same shape as gate; threads 128..255 do these 128 loads.
        if (lidx >= 128u && lidx < 256u) {
            let local_idx = lidx - 128u;
            let load_row: u32 = local_idx / 4u;
            let load_k4:  u32 = local_idx & 3u;
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
                let p0 = unpack2x16float(up_weights[w_off]);
                v0 = p0.x;
                v1 = p0.y;
                if (w_col_lo + 2u < cols) {
                    let p1 = unpack2x16float(up_weights[w_off + 1u]);
                    v2 = p1.x;
                    v3 = p1.y;
                }
            }
            a_up_tile[load_row][load_k_base + 0u] = v0;
            a_up_tile[load_row][load_k_base + 1u] = v1;
            a_up_tile[load_row][load_k_base + 2u] = v2;
            a_up_tile[load_row][load_k_base + 3u] = v3;
        }

        // ---- Cooperative load: B tile (input) ----
        // 16 tokens × 16 K-cols / 4-per-load = 64 logical loads.
        // First 64 threads do these (they ALSO did gate loads but
        // it's a separate sync point in the shader; both phases
        // complete before the barrier below).
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
        // 16 K-positions × (2 gate + 2 up) = 64 MADDs per thread,
        // sharing b_val[k] across all four accumulators per K.
        // Hand-unrolled (Phase 2's lesson: naga loops don't unroll
        // and the driver doesn't either, costing ~2x throughput).
        let li_b: u32 = 16u + li;

        let b0 = b_tile[lj][0u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][0u]   * b0;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][0u] * b0;
        sum_up_a   = sum_up_a   + a_up_tile[li][0u]     * b0;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][0u]   * b0;

        let b1 = b_tile[lj][1u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][1u]   * b1;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][1u] * b1;
        sum_up_a   = sum_up_a   + a_up_tile[li][1u]     * b1;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][1u]   * b1;

        let b2 = b_tile[lj][2u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][2u]   * b2;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][2u] * b2;
        sum_up_a   = sum_up_a   + a_up_tile[li][2u]     * b2;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][2u]   * b2;

        let b3 = b_tile[lj][3u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][3u]   * b3;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][3u] * b3;
        sum_up_a   = sum_up_a   + a_up_tile[li][3u]     * b3;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][3u]   * b3;

        let b4 = b_tile[lj][4u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][4u]   * b4;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][4u] * b4;
        sum_up_a   = sum_up_a   + a_up_tile[li][4u]     * b4;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][4u]   * b4;

        let b5 = b_tile[lj][5u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][5u]   * b5;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][5u] * b5;
        sum_up_a   = sum_up_a   + a_up_tile[li][5u]     * b5;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][5u]   * b5;

        let b6 = b_tile[lj][6u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][6u]   * b6;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][6u] * b6;
        sum_up_a   = sum_up_a   + a_up_tile[li][6u]     * b6;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][6u]   * b6;

        let b7 = b_tile[lj][7u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][7u]   * b7;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][7u] * b7;
        sum_up_a   = sum_up_a   + a_up_tile[li][7u]     * b7;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][7u]   * b7;

        let b8 = b_tile[lj][8u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][8u]   * b8;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][8u] * b8;
        sum_up_a   = sum_up_a   + a_up_tile[li][8u]     * b8;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][8u]   * b8;

        let b9 = b_tile[lj][9u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][9u]   * b9;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][9u] * b9;
        sum_up_a   = sum_up_a   + a_up_tile[li][9u]     * b9;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][9u]   * b9;

        let b10 = b_tile[lj][10u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][10u]   * b10;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][10u] * b10;
        sum_up_a   = sum_up_a   + a_up_tile[li][10u]     * b10;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][10u]   * b10;

        let b11 = b_tile[lj][11u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][11u]   * b11;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][11u] * b11;
        sum_up_a   = sum_up_a   + a_up_tile[li][11u]     * b11;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][11u]   * b11;

        let b12 = b_tile[lj][12u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][12u]   * b12;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][12u] * b12;
        sum_up_a   = sum_up_a   + a_up_tile[li][12u]     * b12;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][12u]   * b12;

        let b13 = b_tile[lj][13u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][13u]   * b13;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][13u] * b13;
        sum_up_a   = sum_up_a   + a_up_tile[li][13u]     * b13;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][13u]   * b13;

        let b14 = b_tile[lj][14u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][14u]   * b14;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][14u] * b14;
        sum_up_a   = sum_up_a   + a_up_tile[li][14u]     * b14;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][14u]   * b14;

        let b15 = b_tile[lj][15u];
        sum_gate_a = sum_gate_a + a_gate_tile[li][15u]   * b15;
        sum_gate_b = sum_gate_b + a_gate_tile[li_b][15u] * b15;
        sum_up_a   = sum_up_a   + a_up_tile[li][15u]     * b15;
        sum_up_b   = sum_up_b   + a_up_tile[li_b][15u]   * b15;

        workgroupBarrier();
    }

    if (a_in_bounds) {
        gate_out[out_tok * rows + out_row_a] = sum_gate_a;
        up_out[out_tok * rows + out_row_a]   = sum_up_a;
    }
    if (b_in_bounds) {
        gate_out[out_tok * rows + out_row_b] = sum_gate_b;
        up_out[out_tok * rows + out_row_b]   = sum_up_b;
    }
}
