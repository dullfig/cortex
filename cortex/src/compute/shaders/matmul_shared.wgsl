// Batched matmul with workgroup-shared-memory tiling.
//
// Computes output[tok, row] = dot(weights[row], input[tok]) for batched
// prefill. Weights are f16-packed in u32 (2 weights per u32, .x then .y).
//
// Design — textbook tiled SGEMM, adapted for cortex's layout:
//
//   - Workgroup: 16×16 = 256 threads
//   - Output tile per workgroup: BM × BN = 64 rows × 64 tokens = 4096 outputs
//   - K-tile size: BK = 16
//   - Per-thread register accumulator: TM × TN = 4 rows × 4 tokens = 16 floats
//
// Per outer K-tile iteration:
//   1. 256 threads cooperatively load BM×BK weights (1024 floats, 4 per
//      thread) into shared memory `sm_b`, unpacking f16 on the way in.
//   2. 256 threads cooperatively load BN×BK activations (1024 floats, 4
//      per thread) into shared memory `sm_a`.
//   3. Barrier.
//   4. Each thread reads its TM×BK weight slice and TN×BK activation
//      slice from shared memory, and accumulates TM×TN×BK MADDs into
//      its register tile. **The TM×TN inner accumulator update is
//      hand-unrolled** so naga keeps `s00..s33` in registers instead of
//      spilling a 2D array to scratch memory.
//   5. Barrier (before next iteration's loads can overwrite shared mem).
//
// At the end, each thread writes its 4×4 register tile to output_mat
// with bounds checking on the partial-edge case.
//
// Why these tile dims:
//   - BM=BN=64, BK=16 → 16 KB total shared memory (8 KB A + 8 KB B),
//     well within Vulkan's 32-48 KB typical limit.
//   - TM=TN=4 → 16 register accumulators per thread, comfortably fitting
//     in a thread's register budget on any modern GPU. (The previous
//     8×8 attempt likely spilled `array<array<f32, 8>, 8>` to scratch
//     under naga; 4×4 is the conservative starting point.)
//   - 16×16 workgroup × 4×4 tile = 64×64 output block — round-up dispatch
//     handles edge cases.
//
// Edge handling: out-of-bounds rows/tokens load zero into shared memory
// and the writeback skips them. Workgroups whose output block partially
// overlaps the matrix still do the same work; only the writeback
// changes.

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

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;
const WG_X: u32 = 16u;  // BN / TN
const WG_Y: u32 = 16u;  // BM / TM
const THREADS: u32 = 256u;

// sm_a indexed [token, k]: row-major [BN][BK]
// sm_b indexed [row,   k]: row-major [BM][BK]
var<workgroup> sm_a: array<f32, 1024>; // BN * BK
var<workgroup> sm_b: array<f32, 1024>; // BM * BK

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // wid.x selects the row-block; wid.y selects the token-block.
    let block_row_base: u32 = wid.x * BM;
    let block_tok_base: u32 = wid.y * BN;
    let tid: u32 = lid.y * WG_X + lid.x;  // 0..255 linear thread id

    // This thread's per-output position within the workgroup tile.
    let thread_row: u32 = lid.x * TM;  // 0..63 stride TM=4
    let thread_tok: u32 = lid.y * TN;  // 0..63 stride TN=4

    let cols = params.cols;
    let half_cols = cols / 2u;
    let rows = params.rows;
    let n_tokens = params.n_tokens;

    // Register accumulator (4×4 = 16 f32 per thread). Declared as flat
    // scalars to guarantee register-resident — the previous attempt with
    // a 2D `array<array<f32, 8>, 8>` likely spilled under naga.
    var s00: f32 = 0.0; var s01: f32 = 0.0; var s02: f32 = 0.0; var s03: f32 = 0.0;
    var s10: f32 = 0.0; var s11: f32 = 0.0; var s12: f32 = 0.0; var s13: f32 = 0.0;
    var s20: f32 = 0.0; var s21: f32 = 0.0; var s22: f32 = 0.0; var s23: f32 = 0.0;
    var s30: f32 = 0.0; var s31: f32 = 0.0; var s32: f32 = 0.0; var s33: f32 = 0.0;

    // Outer K loop walks `cols` in chunks of BK.
    var k_base: u32 = 0u;
    loop {
        if (k_base >= cols) { break; }

        // ---- Cooperative load: BN tokens × BK k-values of activations ----
        // Layout for coalescing: adjacent threads (in tid) hit adjacent
        // global addresses. Index pattern is `li * THREADS + tid` (NOT
        // `tid * 4 + li`) so the inner stride across the warp is 1.
        for (var li: u32 = 0u; li < 4u; li = li + 1u) {
            let idx: u32 = li * THREADS + tid;
            let tt: u32 = idx / BK;       // 0..63 (token within block)
            let kk: u32 = idx % BK;       // 0..15
            let token: u32 = block_tok_base + tt;
            let k: u32 = k_base + kk;
            var v: f32 = 0.0;
            if (token < n_tokens && k < cols) {
                v = input_mat[token * cols + k];
            }
            sm_a[idx] = v;
        }

        // ---- Cooperative load: BM rows × BK k-values of weights ----
        // Same coalesced index pattern. Unpack f16 → f32 inline.
        for (var li: u32 = 0u; li < 4u; li = li + 1u) {
            let idx: u32 = li * THREADS + tid;
            let rr: u32 = idx / BK;       // 0..63 (row within block)
            let kk: u32 = idx % BK;       // 0..15
            let row: u32 = block_row_base + rr;
            let k: u32 = k_base + kk;
            var v: f32 = 0.0;
            if (row < rows && k < cols) {
                let w_off: u32 = row * half_cols + (k / 2u);
                let pair = unpack2x16float(weights[w_off]);
                if ((k & 1u) == 0u) { v = pair.x; } else { v = pair.y; }
            }
            sm_b[idx] = v;
        }

        workgroupBarrier();

        // ---- Accumulate: BK MADDs per (i, j) output element ----
        // Loop over kk = 0..BK. For each kk:
        //   Load TM weight values (this thread's 4 rows at column kk)
        //   Load TN activation values (this thread's 4 tokens at column kk)
        //   Update 16 accumulators (hand-unrolled below).
        for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {
            // 4 weights for this thread's row range, fixed at column kk.
            let b0: f32 = sm_b[(thread_row + 0u) * BK + kk];
            let b1: f32 = sm_b[(thread_row + 1u) * BK + kk];
            let b2: f32 = sm_b[(thread_row + 2u) * BK + kk];
            let b3: f32 = sm_b[(thread_row + 3u) * BK + kk];
            // 4 activations for this thread's token range, fixed at column kk.
            let a0: f32 = sm_a[(thread_tok + 0u) * BK + kk];
            let a1: f32 = sm_a[(thread_tok + 1u) * BK + kk];
            let a2: f32 = sm_a[(thread_tok + 2u) * BK + kk];
            let a3: f32 = sm_a[(thread_tok + 3u) * BK + kk];
            // Hand-unrolled 4×4 outer product → 16 MADDs. naga sees scalar
            // ops only; no array indexing; accumulators stay in registers.
            s00 = s00 + b0 * a0; s01 = s01 + b0 * a1; s02 = s02 + b0 * a2; s03 = s03 + b0 * a3;
            s10 = s10 + b1 * a0; s11 = s11 + b1 * a1; s12 = s12 + b1 * a2; s13 = s13 + b1 * a3;
            s20 = s20 + b2 * a0; s21 = s21 + b2 * a1; s22 = s22 + b2 * a2; s23 = s23 + b2 * a3;
            s30 = s30 + b3 * a0; s31 = s31 + b3 * a1; s32 = s32 + b3 * a2; s33 = s33 + b3 * a3;
        }

        workgroupBarrier();

        k_base = k_base + BK;
    }

    // ---- Writeback: 4×4 = 16 outputs per thread, bounds-checked ----
    // output_mat layout: [n_tokens, rows] flat, output[tok * rows + row].
    let r0: u32 = block_row_base + thread_row + 0u;
    let r1: u32 = block_row_base + thread_row + 1u;
    let r2: u32 = block_row_base + thread_row + 2u;
    let r3: u32 = block_row_base + thread_row + 3u;
    let t0: u32 = block_tok_base + thread_tok + 0u;
    let t1: u32 = block_tok_base + thread_tok + 1u;
    let t2: u32 = block_tok_base + thread_tok + 2u;
    let t3: u32 = block_tok_base + thread_tok + 3u;

    if (t0 < n_tokens) {
        let base = t0 * rows;
        if (r0 < rows) { output_mat[base + r0] = s00; }
        if (r1 < rows) { output_mat[base + r1] = s10; }
        if (r2 < rows) { output_mat[base + r2] = s20; }
        if (r3 < rows) { output_mat[base + r3] = s30; }
    }
    if (t1 < n_tokens) {
        let base = t1 * rows;
        if (r0 < rows) { output_mat[base + r0] = s01; }
        if (r1 < rows) { output_mat[base + r1] = s11; }
        if (r2 < rows) { output_mat[base + r2] = s21; }
        if (r3 < rows) { output_mat[base + r3] = s31; }
    }
    if (t2 < n_tokens) {
        let base = t2 * rows;
        if (r0 < rows) { output_mat[base + r0] = s02; }
        if (r1 < rows) { output_mat[base + r1] = s12; }
        if (r2 < rows) { output_mat[base + r2] = s22; }
        if (r3 < rows) { output_mat[base + r3] = s32; }
    }
    if (t3 < n_tokens) {
        let base = t3 * rows;
        if (r0 < rows) { output_mat[base + r0] = s03; }
        if (r1 < rows) { output_mat[base + r1] = s13; }
        if (r2 < rows) { output_mat[base + r2] = s23; }
        if (r3 < rows) { output_mat[base + r3] = s33; }
    }
}
