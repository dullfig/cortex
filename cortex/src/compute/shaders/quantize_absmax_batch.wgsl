// Per-token absmax quantization for batched ternary matmul (#bn-1).
//
// Input:  f32 [n_tokens, cols] row-major (the hidden state at a layer boundary).
// Output:
//   - i8 packed (4 per u32) activations [n_tokens, ceil(cols / 4)] —
//     same layout the single-token `ternary_matvec` shader expects, just
//     extended with a leading token dim.
//   - f32 per-token scales [n_tokens].
//
// One workgroup per token. WG=256 threads.
//   Phase 1: parallel max(|input[token, j]|) via tree reduction in shared mem.
//   Phase 2: thread 0 derives scale = max_abs / 127 (or 1.0 if zero) and
//            broadcasts via a shared scalar.
//   Phase 3: each thread quantizes 4 elements at a time and packs into a
//            single u32 (one thread = one output u32, stride-WG).

struct Params {
    cols: u32,
    n_tokens: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_max: array<f32, 256>;
var<workgroup> wg_scale: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let token = wid.x;
    if (token >= params.n_tokens) { return; }
    let tid = lid.x;
    let cols = params.cols;
    let in_base = token * cols;

    // Phase 1: each thread scans a stride of input elements, tracking
    // its local absmax. Then tree-reduce in shared memory.
    var local_max: f32 = 0.0;
    var i = tid;
    while (i < cols) {
        let av = abs(input[in_base + i]);
        if (av > local_max) { local_max = av; }
        i += WG;
    }
    wg_max[tid] = local_max;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) {
            let other = wg_max[tid + s];
            if (other > wg_max[tid]) { wg_max[tid] = other; }
        }
        workgroupBarrier();
        s >>= 1u;
    }

    // Phase 2: thread 0 publishes scale; everyone reads it back.
    if (tid == 0u) {
        let max_abs = wg_max[0u];
        var scale: f32 = max_abs / 127.0;
        if (scale == 0.0) { scale = 1.0; }
        output_scales[token] = scale;
        wg_scale = scale;
    }
    workgroupBarrier();
    let scale = wg_scale;

    // Phase 3: pack 4 quantized i8 values per output u32. Each thread
    // owns u32s at stride WG (so thread t handles u32 indices t, t+WG, ...).
    let n_u32_per_token = (cols + 3u) / 4u;
    let out_base = token * n_u32_per_token;

    var u = tid;
    while (u < n_u32_per_token) {
        var packed: u32 = 0u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let col = u * 4u + k;
            var qi: i32 = 0;
            if (col < cols) {
                let v = input[in_base + col];
                var q = i32(round(v / scale));
                if (q < -127) { q = -127; }
                if (q > 127) { q = 127; }
                qi = q;
            }
            let byte = u32(qi & 0xFF);
            packed |= byte << (k * 8u);
        }
        output_q[out_base + u] = packed;
        u += WG;
    }
}
