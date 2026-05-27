// Batched ternary matmul — SHARED-MEMORY TILED + 2-per-thread variant.
// Phase 5 of giggly-chasing-melody (applies the float-matmul Phase 1+2
// pattern to the ternary path).
//
// Inputs:
//   - weights         : 2-bit packed ternary [rows, cols] (resident from GpuBitLinear)
//                       16 weights per u32 in flat row-major order.
//   - activations     : i8 packed (4 per u32) [n_tokens, ceil(cols / 4)]
//                       — output of quantize_absmax_batch
//   - act_scales      : f32 [n_tokens] — per-token absmax scales
//   - params.weight_scale_bits : f32 bitcast to u32 (uniform packing)
//
// Output:
//   - output_mat      : f32 [n_tokens, rows] row-major. Scaling applied
//                       inline: out[tok, row] = i32_acc * act_scale[tok] * weight_scale.
//
// **Design (Phase 5).** For each 32×16 output tile, the workgroup
// cooperatively decodes a 32×TILE_K weight tile and unpacks a
// 16×TILE_K activation tile into shared memory as i32 (one value per
// MADD position, branchless). Each of the 256 threads then accumulates
// TWO output elements (stride-16 in the M dimension) by reading from
// shared memory — same shape as the f16 matmul_shared shader, but with
// integer accumulators throughout (i32 × i32 → i32; convert to f32 +
// apply scales only at writeback).
//
// **Why pre-decode weights into i32 shared instead of decoding in the
// MADD loop.** The 2-bit decode requires a branch (bits == 0 ⇒ -1,
// bits == 2 ⇒ +1, else 0). Doing that 32× per K-step per thread
// (once per MADD) costs us the hand-unroll's straight-line speedup.
// Pre-decoding once into shared turns the MADD into a plain
// i32 × i32 multiply that the compiler turns into a MAD.
//
// **Tile sizes.** TILE_M = 32, TILE_N = 16, TILE_K = 16. Workgroup is
// 16×16 = 256 threads; each thread computes 2 output rows.
//
// **Dispatch shape.** workgroup_id.x = M tile (0..ceil(rows/32))
//                    workgroup_id.y = N tile (0..ceil(n_tokens/16))
//
// **Shared memory budget.** 32×16 i32 a_tile + 16×16 i32 b_tile =
// 2 KB + 1 KB = 3 KB per WG. Plenty of occupancy headroom.
//
// **K-tile load efficiency.** For TILE_K=16 starting at a 16-aligned
// k_base, the 16 K-positions for one weight row are exactly one u32
// (16 weights/u32 in flat layout); 4 u32s for one activation row
// (4 i8/u32). Cooperative loads: 32 weight rows × 1 u32 = 32 loaders,
// 16 act rows × 4 u32s = 64 loaders.

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

const TILE_M: u32 = 32u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

var<workgroup> a_tile: array<array<i32, TILE_K>, TILE_M>;
var<workgroup> b_tile: array<array<i32, TILE_K>, TILE_N>;

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
    let n_u32_per_token = (cols + 3u) / 4u;

    let out_row_a: u32 = row_base + li;
    let out_row_b: u32 = row_base + 16u + li;
    let out_tok:   u32 = tok_base + lj;
    let a_in: bool = (out_row_a < rows) && (out_tok < n_tokens);
    let b_in: bool = (out_row_b < rows) && (out_tok < n_tokens);

    var acc_a: i32 = 0;
    var acc_b: i32 = 0;

    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // ---- Cooperative load: A tile (weights) ----
        // 32 weight rows × 16 K-cols = one u32 per row (since 16 weights
        // per u32 in flat layout, and k_base is 16-aligned). First 32
        // threads each load 1 u32 and decode 16 ternary weights into i32
        // in shared memory.
        if (lidx < 32u) {
            let load_row = lidx;
            let w_row = row_base + load_row;
            if (w_row < rows && k_base < cols) {
                let u32_idx = (w_row * cols + k_base) / 16u;
                let packed = weights[u32_idx];
                // Hand-unrolled 16 ternary decodes. Each pair of bits:
                //   00 → -1, 01 → 0, 10 → +1, 11 → 0.
                // Branchless: v = (bits == 2) - (bits == 0). WGSL
                // operator precedence requires parens around (& 3u)
                // — == binds tighter than &.
                let b00  = (packed >> 0u)  & 3u;
                let b01  = (packed >> 2u)  & 3u;
                let b02  = (packed >> 4u)  & 3u;
                let b03  = (packed >> 6u)  & 3u;
                let b04  = (packed >> 8u)  & 3u;
                let b05  = (packed >> 10u) & 3u;
                let b06  = (packed >> 12u) & 3u;
                let b07  = (packed >> 14u) & 3u;
                let b08  = (packed >> 16u) & 3u;
                let b09  = (packed >> 18u) & 3u;
                let b10  = (packed >> 20u) & 3u;
                let b11  = (packed >> 22u) & 3u;
                let b12  = (packed >> 24u) & 3u;
                let b13  = (packed >> 26u) & 3u;
                let b14  = (packed >> 28u) & 3u;
                let b15  = (packed >> 30u) & 3u;
                a_tile[load_row][0u]  = i32(b00 == 2u) - i32(b00 == 0u);
                a_tile[load_row][1u]  = i32(b01 == 2u) - i32(b01 == 0u);
                a_tile[load_row][2u]  = i32(b02 == 2u) - i32(b02 == 0u);
                a_tile[load_row][3u]  = i32(b03 == 2u) - i32(b03 == 0u);
                a_tile[load_row][4u]  = i32(b04 == 2u) - i32(b04 == 0u);
                a_tile[load_row][5u]  = i32(b05 == 2u) - i32(b05 == 0u);
                a_tile[load_row][6u]  = i32(b06 == 2u) - i32(b06 == 0u);
                a_tile[load_row][7u]  = i32(b07 == 2u) - i32(b07 == 0u);
                a_tile[load_row][8u]  = i32(b08 == 2u) - i32(b08 == 0u);
                a_tile[load_row][9u]  = i32(b09 == 2u) - i32(b09 == 0u);
                a_tile[load_row][10u] = i32(b10 == 2u) - i32(b10 == 0u);
                a_tile[load_row][11u] = i32(b11 == 2u) - i32(b11 == 0u);
                a_tile[load_row][12u] = i32(b12 == 2u) - i32(b12 == 0u);
                a_tile[load_row][13u] = i32(b13 == 2u) - i32(b13 == 0u);
                a_tile[load_row][14u] = i32(b14 == 2u) - i32(b14 == 0u);
                a_tile[load_row][15u] = i32(b15 == 2u) - i32(b15 == 0u);
            } else {
                for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                    a_tile[load_row][k] = 0;
                }
            }
        }

        // ---- Cooperative load: B tile (activations) ----
        // 16 tokens × 16 K-cols = 4 u32s per token (4 i8 per u32).
        // First 64 threads each load 1 u32 (4 i8) and sign-extend into i32.
        if (lidx < 64u) {
            let load_tok = lidx / 4u;
            let load_k4 = lidx & 3u;
            let load_k_base = load_k4 * 4u;
            let in_tok = tok_base + load_tok;
            let in_col_base = k_base + load_k_base;

            if (in_tok < n_tokens && in_col_base < cols) {
                let act_base = in_tok * n_u32_per_token;
                let act_u32 = activations[act_base + in_col_base / 4u];
                // Unpack 4 i8 with sign extension.
                var v0: i32 = i32((act_u32 >> 0u)  & 0xFFu);
                var v1: i32 = i32((act_u32 >> 8u)  & 0xFFu);
                var v2: i32 = i32((act_u32 >> 16u) & 0xFFu);
                var v3: i32 = i32((act_u32 >> 24u) & 0xFFu);
                if (v0 > 127) { v0 = v0 - 256; }
                if (v1 > 127) { v1 = v1 - 256; }
                if (v2 > 127) { v2 = v2 - 256; }
                if (v3 > 127) { v3 = v3 - 256; }
                // Zero out activations that overflow the col bound (partial tile edge).
                if (in_col_base + 1u >= cols) { v1 = 0; }
                if (in_col_base + 2u >= cols) { v2 = 0; }
                if (in_col_base + 3u >= cols) { v3 = 0; }
                b_tile[load_tok][load_k_base + 0u] = v0;
                b_tile[load_tok][load_k_base + 1u] = v1;
                b_tile[load_tok][load_k_base + 2u] = v2;
                b_tile[load_tok][load_k_base + 3u] = v3;
            } else {
                b_tile[load_tok][load_k_base + 0u] = 0;
                b_tile[load_tok][load_k_base + 1u] = 0;
                b_tile[load_tok][load_k_base + 2u] = 0;
                b_tile[load_tok][load_k_base + 3u] = 0;
            }
        }

        workgroupBarrier();

        // ---- Per-thread MADDs (i32, hand-unrolled) ----
        // 16 K-positions × 2 outputs = 32 MADDs per thread. b_val shared
        // across the two accumulators per K. Branchless: shared mem holds
        // pre-decoded {-1, 0, +1} i32s.
        let li_b: u32 = 16u + li;
        let b0  = b_tile[lj][0u];
        acc_a = acc_a + a_tile[li][0u]   * b0;
        acc_b = acc_b + a_tile[li_b][0u] * b0;
        let b1  = b_tile[lj][1u];
        acc_a = acc_a + a_tile[li][1u]   * b1;
        acc_b = acc_b + a_tile[li_b][1u] * b1;
        let b2  = b_tile[lj][2u];
        acc_a = acc_a + a_tile[li][2u]   * b2;
        acc_b = acc_b + a_tile[li_b][2u] * b2;
        let b3  = b_tile[lj][3u];
        acc_a = acc_a + a_tile[li][3u]   * b3;
        acc_b = acc_b + a_tile[li_b][3u] * b3;
        let b4  = b_tile[lj][4u];
        acc_a = acc_a + a_tile[li][4u]   * b4;
        acc_b = acc_b + a_tile[li_b][4u] * b4;
        let b5  = b_tile[lj][5u];
        acc_a = acc_a + a_tile[li][5u]   * b5;
        acc_b = acc_b + a_tile[li_b][5u] * b5;
        let b6  = b_tile[lj][6u];
        acc_a = acc_a + a_tile[li][6u]   * b6;
        acc_b = acc_b + a_tile[li_b][6u] * b6;
        let b7  = b_tile[lj][7u];
        acc_a = acc_a + a_tile[li][7u]   * b7;
        acc_b = acc_b + a_tile[li_b][7u] * b7;
        let b8  = b_tile[lj][8u];
        acc_a = acc_a + a_tile[li][8u]   * b8;
        acc_b = acc_b + a_tile[li_b][8u] * b8;
        let b9  = b_tile[lj][9u];
        acc_a = acc_a + a_tile[li][9u]   * b9;
        acc_b = acc_b + a_tile[li_b][9u] * b9;
        let b10 = b_tile[lj][10u];
        acc_a = acc_a + a_tile[li][10u]   * b10;
        acc_b = acc_b + a_tile[li_b][10u] * b10;
        let b11 = b_tile[lj][11u];
        acc_a = acc_a + a_tile[li][11u]   * b11;
        acc_b = acc_b + a_tile[li_b][11u] * b11;
        let b12 = b_tile[lj][12u];
        acc_a = acc_a + a_tile[li][12u]   * b12;
        acc_b = acc_b + a_tile[li_b][12u] * b12;
        let b13 = b_tile[lj][13u];
        acc_a = acc_a + a_tile[li][13u]   * b13;
        acc_b = acc_b + a_tile[li_b][13u] * b13;
        let b14 = b_tile[lj][14u];
        acc_a = acc_a + a_tile[li][14u]   * b14;
        acc_b = acc_b + a_tile[li_b][14u] * b14;
        let b15 = b_tile[lj][15u];
        acc_a = acc_a + a_tile[li][15u]   * b15;
        acc_b = acc_b + a_tile[li_b][15u] * b15;

        workgroupBarrier();
    }

    // Writeback. Apply per-token activation scale and uniform weight scale.
    let weight_scale = bitcast<f32>(params.weight_scale_bits);
    if (a_in) {
        let act_scale = act_scales[out_tok];
        output_mat[out_tok * rows + out_row_a] = f32(acc_a) * act_scale * weight_scale;
    }
    if (b_in) {
        let act_scale = act_scales[out_tok];
        output_mat[out_tok * rows + out_row_b] = f32(acc_b) * act_scale * weight_scale;
    }
}
