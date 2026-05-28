// Phase C2: packed-f16 output variant of ternary_matmul_batch.
// Same i8-input + ternary-weight math as the parent shader, but
// writes packed f16 output (2 adjacent rows per u32). Used for the
// BitNet gate/up projections, where C2 packs scratch.gate/up.
//
// Thread-row mapping changed from stride-16 (li, 16+li) to adjacent
// pairs (li*2, li*2+1) so the per-thread (acc_a, acc_b) writes pack
// cleanly into one u32. Cooperative weight + activation loads still
// cover all 32 rows / 16 tokens of the tile — only the compute
// indexing changes.

struct Params {
    rows: u32,
    cols: u32,
    n_tokens: u32,
    weight_scale_bits: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read> act_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_mat: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;
// Phase C2: pad to 17 to dodge bank conflicts on adjacent-pair row
// access (li*2, li*2+1). See matmul_gate_up_shared.wgsl for the
// detailed bank-conflict analysis.
const TILE_K_P: u32 = 17u;

var<workgroup> a_tile: array<array<i32, TILE_K_P>, TILE_M>;
var<workgroup> b_tile: array<array<i32, TILE_K_P>, TILE_N>;

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
    let n_u32_per_token = (cols + 3u) / 4u;

    // Phase C2: adjacent-pair row mapping (li*2, li*2+1).
    let out_row_a: u32 = row_base + li * 2u;
    let out_row_b: u32 = row_base + li * 2u + 1u;
    let out_tok:   u32 = tok_base + lj;
    let a_in: bool = (out_row_a < rows) && (out_tok < n_tokens);
    let b_in: bool = (out_row_b < rows) && (out_tok < n_tokens);

    var acc_a: i32 = 0;
    var acc_b: i32 = 0;

    let num_k_steps: u32 = (cols + TILE_K - 1u) / TILE_K;

    for (var ks: u32 = 0u; ks < num_k_steps; ks = ks + 1u) {
        let k_base: u32 = ks * TILE_K;

        // Cooperative load: 32 weight rows × 16 K-cols = 1 u32 per row.
        if (lidx < 32u) {
            let load_row = lidx;
            let w_row = row_base + load_row;
            if (w_row < rows && k_base < cols) {
                let u32_idx = (w_row * cols + k_base) / 16u;
                let packed = weights[u32_idx];
                let b00 = (packed >> 0u)  & 3u;
                let b01 = (packed >> 2u)  & 3u;
                let b02 = (packed >> 4u)  & 3u;
                let b03 = (packed >> 6u)  & 3u;
                let b04 = (packed >> 8u)  & 3u;
                let b05 = (packed >> 10u) & 3u;
                let b06 = (packed >> 12u) & 3u;
                let b07 = (packed >> 14u) & 3u;
                let b08 = (packed >> 16u) & 3u;
                let b09 = (packed >> 18u) & 3u;
                let b10 = (packed >> 20u) & 3u;
                let b11 = (packed >> 22u) & 3u;
                let b12 = (packed >> 24u) & 3u;
                let b13 = (packed >> 26u) & 3u;
                let b14 = (packed >> 28u) & 3u;
                let b15 = (packed >> 30u) & 3u;
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

        // Cooperative load: 16 tokens × 16 K-cols = 4 u32s per token.
        if (lidx < 64u) {
            let load_tok = lidx / 4u;
            let load_k4 = lidx & 3u;
            let load_k_base = load_k4 * 4u;
            let in_tok = tok_base + load_tok;
            let in_col_base = k_base + load_k_base;

            if (in_tok < n_tokens && in_col_base < cols) {
                let act_base = in_tok * n_u32_per_token;
                let act_u32 = activations[act_base + in_col_base / 4u];
                var v0: i32 = i32((act_u32 >> 0u)  & 0xFFu);
                var v1: i32 = i32((act_u32 >> 8u)  & 0xFFu);
                var v2: i32 = i32((act_u32 >> 16u) & 0xFFu);
                var v3: i32 = i32((act_u32 >> 24u) & 0xFFu);
                if (v0 > 127) { v0 = v0 - 256; }
                if (v1 > 127) { v1 = v1 - 256; }
                if (v2 > 127) { v2 = v2 - 256; }
                if (v3 > 127) { v3 = v3 - 256; }
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

        // Phase C2 adjacent-pair MADDs: rows (li*2, li*2+1).
        let li_a: u32 = li * 2u;
        let li_b: u32 = li * 2u + 1u;
        let b0  = b_tile[lj][0u];
        acc_a = acc_a + a_tile[li_a][0u] * b0;
        acc_b = acc_b + a_tile[li_b][0u] * b0;
        let b1  = b_tile[lj][1u];
        acc_a = acc_a + a_tile[li_a][1u] * b1;
        acc_b = acc_b + a_tile[li_b][1u] * b1;
        let b2  = b_tile[lj][2u];
        acc_a = acc_a + a_tile[li_a][2u] * b2;
        acc_b = acc_b + a_tile[li_b][2u] * b2;
        let b3  = b_tile[lj][3u];
        acc_a = acc_a + a_tile[li_a][3u] * b3;
        acc_b = acc_b + a_tile[li_b][3u] * b3;
        let b4  = b_tile[lj][4u];
        acc_a = acc_a + a_tile[li_a][4u] * b4;
        acc_b = acc_b + a_tile[li_b][4u] * b4;
        let b5  = b_tile[lj][5u];
        acc_a = acc_a + a_tile[li_a][5u] * b5;
        acc_b = acc_b + a_tile[li_b][5u] * b5;
        let b6  = b_tile[lj][6u];
        acc_a = acc_a + a_tile[li_a][6u] * b6;
        acc_b = acc_b + a_tile[li_b][6u] * b6;
        let b7  = b_tile[lj][7u];
        acc_a = acc_a + a_tile[li_a][7u] * b7;
        acc_b = acc_b + a_tile[li_b][7u] * b7;
        let b8  = b_tile[lj][8u];
        acc_a = acc_a + a_tile[li_a][8u] * b8;
        acc_b = acc_b + a_tile[li_b][8u] * b8;
        let b9  = b_tile[lj][9u];
        acc_a = acc_a + a_tile[li_a][9u] * b9;
        acc_b = acc_b + a_tile[li_b][9u] * b9;
        let b10 = b_tile[lj][10u];
        acc_a = acc_a + a_tile[li_a][10u] * b10;
        acc_b = acc_b + a_tile[li_b][10u] * b10;
        let b11 = b_tile[lj][11u];
        acc_a = acc_a + a_tile[li_a][11u] * b11;
        acc_b = acc_b + a_tile[li_b][11u] * b11;
        let b12 = b_tile[lj][12u];
        acc_a = acc_a + a_tile[li_a][12u] * b12;
        acc_b = acc_b + a_tile[li_b][12u] * b12;
        let b13 = b_tile[lj][13u];
        acc_a = acc_a + a_tile[li_a][13u] * b13;
        acc_b = acc_b + a_tile[li_b][13u] * b13;
        let b14 = b_tile[lj][14u];
        acc_a = acc_a + a_tile[li_a][14u] * b14;
        acc_b = acc_b + a_tile[li_b][14u] * b14;
        let b15 = b_tile[lj][15u];
        acc_a = acc_a + a_tile[li_a][15u] * b15;
        acc_b = acc_b + a_tile[li_b][15u] * b15;

        workgroupBarrier();
    }

    // Writeback: scale + pack adjacent rows into one u32.
    // pair_idx must include the M-tile offset (row_base/2) so threads
    // in different M-tiles write to non-overlapping output slots.
    let weight_scale = bitcast<f32>(params.weight_scale_bits);
    let rows_half = rows / 2u;
    let pair_idx = out_tok * rows_half + (row_base / 2u + li);
    if (a_in && b_in) {
        let act_scale = act_scales[out_tok];
        let va = f32(acc_a) * act_scale * weight_scale;
        let vb = f32(acc_b) * act_scale * weight_scale;
        output_mat[pair_idx] = pack2x16float(vec2<f32>(va, vb));
    } else if (a_in) {
        let act_scale = act_scales[out_tok];
        let va = f32(acc_a) * act_scale * weight_scale;
        output_mat[pair_idx] = pack2x16float(vec2<f32>(va, 0.0));
    }
}
