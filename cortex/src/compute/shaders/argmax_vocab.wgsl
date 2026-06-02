// Single-workgroup argmax over an f32 logits vector. Used by the GPU
// LM-head greedy decode path: input is [vocab_size] logits (from
// dispatch_matmul_pin), output is a single u32 token id at output[0].
//
// One workgroup of WG=256 threads. Each thread serially scans
// vocab/WG entries tracking its local (max, argmax). Then a tree
// reduction in shared memory picks the global argmax. Thread 0 writes.
//
// Tie-break: matches CPU `Iterator::max_by` which keeps the LAST equal
// element. Per-thread scan uses `>=` and tree reduction uses `>=` (right
// side has a higher stride-start, hence higher local_idx on ties), so on
// equal logits the LARGEST index wins — same as the CPU path.

struct Params {
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_id: array<u32>;
@group(0) @binding(2) var<uniform>             params: Params;

const WG: u32 = 256u;

var<workgroup> wg_max: array<f32, WG>;
var<workgroup> wg_idx: array<u32, WG>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Per-thread serial scan: stride WG over vocab.
    var local_max: f32 = -3.4028235e38;  // ~f32::MIN
    var local_idx: u32 = 0u;
    var i: u32 = tid;
    while (i < params.n) {
        let v = logits[i];
        if (v >= local_max) {
            local_max = v;
            local_idx = i;
        }
        i = i + WG;
    }

    wg_max[tid] = local_max;
    wg_idx[tid] = local_idx;
    workgroupBarrier();

    // Tree reduction. Right (higher tid) wins on ties to match CPU
    // last-wins semantics — wg_idx[tid + s] is always strictly greater
    // than wg_idx[tid] when their stride starting points were equal-max,
    // so `>=` here yields the largest argmax index overall.
    var s: u32 = WG / 2u;
    while (s > 0u) {
        if (tid < s) {
            let a_max = wg_max[tid];
            let b_max = wg_max[tid + s];
            if (b_max >= a_max) {
                wg_max[tid] = b_max;
                wg_idx[tid] = wg_idx[tid + s];
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out_id[0] = wg_idx[0];
    }
}
