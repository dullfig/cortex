//! Prefill scratch buffers + chunk-size math (split from gpu_engine.rs, Phase N).
//!
//! `BlockScratch` / `PolarBlockScratch` lane-colored per-forward scratch,
//! `ChunkLimits` + the pure chunk-size solvers, and
//! `GpuEngine::safe_prefill_chunk_size`.

use super::*;

/// All sizing inputs for one prefill chunk decision. Filled by
/// [`GpuEngine::safe_prefill_chunk_size`] from model dims, the three
/// transient-lane capacities, and the device's storage-binding limit;
/// kept as a plain struct so the math is unit-testable without a GPU.
pub struct ChunkLimits {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub embed: usize,
    pub intermediate: usize,
    /// Usable Lane A budget in bytes (post-slack). Holds `attn_out`,
    /// `gate`, `up` — all linear in n_tokens.
    pub lane_a: u64,
    /// Usable Lane B budget in bytes (post-slack). Holds `normed`,
    /// `activated` (linear) and `scores` (quadratic via the attention
    /// window `start_pos + n`).
    pub lane_b: u64,
    /// Usable Lane C budget in bytes (post-slack). Holds `q`, `k`, `v`,
    /// `projected` — all linear in n_tokens.
    pub lane_c: u64,
    /// `wgpu::Limits::max_storage_buffer_binding_size` — the hard cap on
    /// any single storage binding (~2 GB on typical hardware). `scores`
    /// is bound as ONE storage buffer, so it must fit this regardless of
    /// how large Lane B is. (`max_buffer_size` is effectively unbounded
    /// on this hardware; the binding limit is the real ceiling.)
    pub binding_max: u64,
}

/// Largest `n` satisfying `A·n² + (A·start + lin_coeff)·n ≤ budget`,
/// where `A = n_heads·4` is the scores quadratic coefficient (scores =
/// n · n_heads · (start + n) · 4 bytes). With `lin_coeff = 0` this
/// degenerates to "scores alone ≤ budget" — used for the binding clamp.
fn scores_quad_max_n(start_pos: usize, n_heads: usize, lin_coeff: usize, budget: u64) -> usize {
    let a = (n_heads * 4) as f64;
    let lin = a * start_pos as f64 + lin_coeff as f64;
    let n = ((-lin + (lin * lin + 4.0 * a * budget as f64).sqrt()) / (2.0 * a)).floor();
    (n as usize).max(1)
}

/// Pure prefill chunk-size math: the largest token count whose f32
/// `BlockScratch` fits EVERY constraint at attention `start_pos`:
///
/// ```text
///   Lane B (quadratic): normed(n·embed·2) + activated(n·intermediate·2)
///                       + scores(n·n_heads·(start+n)·4) ≤ lane_b
///   Binding (quadratic): scores ≤ binding_max          (single binding)
///   Lane A (linear): attn_out(n·n_heads·head_dim·2)
///                    + gate(n·intermediate·2) + up(n·intermediate·2) ≤ lane_a
///   Lane C (linear): q(n·n_heads·head_dim·2) + k + v (n·n_kv_heads·head_dim·2 each)
///                    + projected(n·embed·2) ≤ lane_c
/// ```
///
/// Phase L checked Lane B only — safe while all three lanes shared the
/// same 128 MB default (B always bound first). Phase M sizes lanes
/// independently from the device budget, so each constraint must be
/// checked; the chunk is the min. Always returns at least 1.
fn prefill_chunk_size(start_pos: usize, lim: &ChunkLimits) -> usize {
    // Lane B: scores + normed + activated (Phase L quadratic).
    let n_b = scores_quad_max_n(
        start_pos, lim.n_heads, (lim.embed + lim.intermediate) * 2, lim.lane_b,
    );
    // Binding cap: scores alone, no linear term.
    let n_bind = scores_quad_max_n(start_pos, lim.n_heads, 0, lim.binding_max);
    // Lane A: attn_out + gate + up, all linear.
    let ca = (lim.n_heads * lim.head_dim * 2 + lim.intermediate * 4) as u64;
    let n_a = (lim.lane_a / ca).max(1) as usize;
    // Lane C: q + k + v + projected, all linear.
    let cc = (lim.n_heads * lim.head_dim * 2
        + lim.n_kv_heads * lim.head_dim * 4
        + lim.embed * 2) as u64;
    let n_c = (lim.lane_c / cc).max(1) as usize;
    n_b.min(n_bind).min(n_a).min(n_c).max(1)
}

/// Per-block scratch buffers reused across all dispatches inside a single
/// `forward_block_gpu` call.
///
/// Phase I (vram-heap): the f32 path's BlockScratch fields are sub-
/// allocations of three lane-colored device-local heaps so dispatches
/// inside `forward_block_gpu_inner` never bind R and RW on the same
/// backing buffer.
///
/// Lane assignment differs from `PolarBlockScratch` in one place:
/// `attn_out` lives on Lane A here, not Lane B. The f32 path's
/// `attn_value_batch` dispatch reads scores and writes attn_out
/// directly (the polar path goes through a rotated_buf intermediate),
/// so if both were on Lane B that would be a same-backing R+RW
/// conflict. Moving attn_out to Lane A keeps the conflict graph
/// resolvable.
///
/// - **Lane A (`transient_heap_a`)**: `attn_out`, `gate`, `up`.
/// - **Lane B (`transient_heap_b`)**: `normed`, `activated`, `scores`.
/// - **Lane C (`transient_heap_c`)**: `q`, `k`, `v`, `projected`.
pub struct BlockScratch {
    pub normed: ::vram_heap::VramAllocation,    // [n_tokens, embed_dim] post-rmsnorm scratch
    pub q: ::vram_heap::VramAllocation,         // [n_tokens, n_heads * head_dim]
    pub k: ::vram_heap::VramAllocation,         // [n_tokens, n_kv_heads * head_dim]
    pub v: ::vram_heap::VramAllocation,         // [n_tokens, n_kv_heads * head_dim]
    pub attn_out: ::vram_heap::VramAllocation,  // [n_tokens, n_heads * head_dim]
    pub scores: ::vram_heap::VramAllocation,    // [n_tokens, n_heads, max_seq] attention scores
    pub gate: ::vram_heap::VramAllocation,      // [n_tokens, intermediate]
    pub up: ::vram_heap::VramAllocation,        // [n_tokens, intermediate]
    pub activated: ::vram_heap::VramAllocation, // [n_tokens, intermediate] SiLU(gate)*up
    pub projected: ::vram_heap::VramAllocation, // [n_tokens, embed_dim] both attn-out-proj and FFN-down output reuse this
}

/// Phase D vram-heap variant of `BlockScratch`. Same 10 fields, but
/// each is a sub-allocation of a lane-colored device-local heap so
/// dispatches inside `forward_block_gpu_polar_inner` never bind R and
/// RW on the same backing buffer.
///
/// Lane assignment (lane = which heap the field lives on):
///
/// - **Lane A (`transient_heap_a`)**: shares with `hidden_buf`,
///   `rotated_buf` (Phase C). Fields: `gate`, `up`.
/// - **Lane B (`transient_heap_b`)**: `normed`, `activated`,
///   `attn_out`, `scores`.
/// - **Lane C (`transient_heap_c`, added in Phase D)**: `q`, `k`,
///   `v`, `projected`.
///
/// The coloring solves the conflict graph where each edge is a
/// (R buffer, RW buffer) pair appearing in one dispatch. With
/// hidden + rotated pinned to lane A from Phase C, the graph is not
/// 2-colorable (normed ↔ hidden forces normed off A; q ↔ rotated
/// forces q off A; normed ↔ q forces them apart — three colors
/// needed). See the Phase D plan section in
/// `~/.claude/plans/giggly-chasing-melody.md` for the full derivation.
///
/// RAII Drop on each `VramAllocation` returns the range to its
/// heap's free-list at function exit; coalesce restores each lane to
/// a single full span between polar forward calls.
pub struct PolarBlockScratch {
    pub normed: ::vram_heap::VramAllocation,
    pub q: ::vram_heap::VramAllocation,
    pub k: ::vram_heap::VramAllocation,
    pub v: ::vram_heap::VramAllocation,
    pub attn_out: ::vram_heap::VramAllocation,
    pub scores: ::vram_heap::VramAllocation,
    pub gate: ::vram_heap::VramAllocation,
    pub up: ::vram_heap::VramAllocation,
    pub activated: ::vram_heap::VramAllocation,
    pub projected: ::vram_heap::VramAllocation,
}

impl PolarBlockScratch {
    /// Allocate the 10 polar-path scratch buffers across the 3-lane
    /// heap scheme. See struct docs for lane assignment. Sizes mirror
    /// `BlockScratch::allocate` exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        gpu: &GpuDevice,
        n_tokens: usize,
        embed_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
        max_seq: usize,
    ) -> Self {
        let align = ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        let gate = gpu.transient_heap_a.allocate(
            (n_tokens * intermediate * 2) as u64, align, "polar_scratch.gate",
        ).expect("transient_heap_a capacity for polar_scratch.gate");
        let up = gpu.transient_heap_a.allocate(
            (n_tokens * intermediate * 2) as u64, align, "polar_scratch.up",
        ).expect("transient_heap_a capacity for polar_scratch.up");

        let normed = gpu.transient_heap_b.allocate(
            (n_tokens * embed_dim * 2) as u64, align, "polar_scratch.normed",
        ).expect("transient_heap_b capacity for polar_scratch.normed");
        let activated = gpu.transient_heap_b.allocate(
            (n_tokens * intermediate * 2) as u64, align, "polar_scratch.activated",
        ).expect("transient_heap_b capacity for polar_scratch.activated");
        let attn_out = gpu.transient_heap_b.allocate(
            (n_tokens * n_heads * head_dim * 2) as u64, align, "polar_scratch.attn_out",
        ).expect("transient_heap_b capacity for polar_scratch.attn_out");
        let scores = gpu.transient_heap_b.allocate(
            (n_tokens * n_heads * max_seq) as u64 * f32_bytes, align, "polar_scratch.scores",
        ).expect("transient_heap_b capacity for polar_scratch.scores");

        let q = gpu.transient_heap_c.allocate(
            (n_tokens * n_heads * head_dim * 2) as u64, align, "polar_scratch.q",
        ).expect("transient_heap_c capacity for polar_scratch.q");
        let k = gpu.transient_heap_c.allocate(
            (n_tokens * n_kv_heads * head_dim * 2) as u64, align, "polar_scratch.k",
        ).expect("transient_heap_c capacity for polar_scratch.k");
        let v = gpu.transient_heap_c.allocate(
            (n_tokens * n_kv_heads * head_dim * 2) as u64, align, "polar_scratch.v",
        ).expect("transient_heap_c capacity for polar_scratch.v");
        let projected = gpu.transient_heap_c.allocate(
            (n_tokens * embed_dim * 2) as u64, align, "polar_scratch.projected",
        ).expect("transient_heap_c capacity for polar_scratch.projected");

        Self { normed, q, k, v, attn_out, scores, gate, up, activated, projected }
    }
}

impl BlockScratch {
    /// Allocate scratch buffers sized for a single forward of `n_tokens`.
    pub fn allocate(
        gpu: &GpuDevice,
        n_tokens: usize,
        embed_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
        max_seq: usize,
    ) -> Self {
        let align = ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        // Lane A: attn_out, gate, up
        let attn_out = gpu.transient_heap_a.allocate(
            (n_tokens * n_heads * head_dim * 2) as u64, align, "scratch.attn_out",
        ).expect("transient_heap_a capacity for scratch.attn_out");
        let gate = gpu.transient_heap_a.allocate(
            (n_tokens * intermediate * 2) as u64, align, "scratch.gate",
        ).expect("transient_heap_a capacity for scratch.gate");
        let up = gpu.transient_heap_a.allocate(
            (n_tokens * intermediate * 2) as u64, align, "scratch.up",
        ).expect("transient_heap_a capacity for scratch.up");

        // Lane B: normed, activated, scores
        let normed = gpu.transient_heap_b.allocate(
            (n_tokens * embed_dim * 2) as u64, align, "scratch.normed",
        ).expect("transient_heap_b capacity for scratch.normed");
        let activated = gpu.transient_heap_b.allocate(
            (n_tokens * intermediate * 2) as u64, align, "scratch.activated",
        ).expect("transient_heap_b capacity for scratch.activated");
        let scores = gpu.transient_heap_b.allocate(
            (n_tokens * n_heads * max_seq) as u64 * f32_bytes, align, "scratch.scores",
        ).expect("transient_heap_b capacity for scratch.scores");

        // Lane C: q, k, v, projected
        let q = gpu.transient_heap_c.allocate(
            (n_tokens * n_heads * head_dim * 2) as u64, align, "scratch.q",
        ).expect("transient_heap_c capacity for scratch.q");
        let k = gpu.transient_heap_c.allocate(
            (n_tokens * n_kv_heads * head_dim * 2) as u64, align, "scratch.k",
        ).expect("transient_heap_c capacity for scratch.k");
        let v = gpu.transient_heap_c.allocate(
            (n_tokens * n_kv_heads * head_dim * 2) as u64, align, "scratch.v",
        ).expect("transient_heap_c capacity for scratch.v");
        let projected = gpu.transient_heap_c.allocate(
            (n_tokens * embed_dim * 2) as u64, align, "scratch.projected",
        ).expect("transient_heap_c capacity for scratch.projected");

        Self { normed, q, k, v, attn_out, scores, gate, up, activated, projected }
    }
}

/// Pure-math tests for the Lane-B prefill chunker. GPU-free — they
/// exercise `lane_b_chunk_size` directly, so they run on CI without a
/// device (unlike the gated parity tests below).
#[cfg(test)]
mod chunk_size_tests {
    use super::{prefill_chunk_size, ChunkLimits};

    /// Qwen 2.5 3B dims with all lanes at the Phase L 128 MB default and
    /// a typical ~2 GB storage-binding cap.
    fn qwen_limits() -> ChunkLimits {
        ChunkLimits {
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 128,
            embed: 2048,
            intermediate: 11008,
            lane_a: 128 * 1024 * 1024 * 97 / 100,
            lane_b: 128 * 1024 * 1024 * 97 / 100,
            lane_c: 128 * 1024 * 1024 * 97 / 100,
            binding_max: 2 * 1024 * 1024 * 1024 - 4096, // ~2 GiB, typical NVIDIA
        }
    }

    /// Exact per-lane footprints (bytes) of an f32 BlockScratch for `n`
    /// tokens at attention `start`, mirroring `BlockScratch::allocate`.
    fn lane_a_bytes(n: usize, lim: &ChunkLimits) -> u64 {
        let attn_out = (n * lim.n_heads * lim.head_dim * 2) as u64;
        let gate_up = (n * lim.intermediate * 2 * 2) as u64;
        attn_out + gate_up
    }
    fn lane_b_bytes(n: usize, start: usize, lim: &ChunkLimits) -> u64 {
        let normed = (n * lim.embed * 2) as u64;
        let activated = (n * lim.intermediate * 2) as u64;
        scores_bytes(n, start, lim) + normed + activated
    }
    fn lane_c_bytes(n: usize, lim: &ChunkLimits) -> u64 {
        let q = (n * lim.n_heads * lim.head_dim * 2) as u64;
        let kv = (n * lim.n_kv_heads * lim.head_dim * 2 * 2) as u64;
        let projected = (n * lim.embed * 2) as u64;
        q + kv + projected
    }
    fn scores_bytes(n: usize, start: usize, lim: &ChunkLimits) -> u64 {
        (n * lim.n_heads * (start + n) * 4) as u64
    }

    /// True iff a chunk of `n` tokens at `start` violates no constraint.
    fn fits_all(n: usize, start: usize, lim: &ChunkLimits) -> bool {
        lane_a_bytes(n, lim) <= lim.lane_a
            && lane_b_bytes(n, start, lim) <= lim.lane_b
            && lane_c_bytes(n, lim) <= lim.lane_c
            && scores_bytes(n, start, lim) <= lim.binding_max
    }

    #[test]
    fn fits_all_constraints_is_maximal_and_shrinks() {
        let lim = qwen_limits();
        let mut prev = usize::MAX;
        for &start in &[0usize, 256, 1000, 2000, 4000, 50_000] {
            let n = prefill_chunk_size(start, &lim);

            // (a) never zero.
            assert!(n >= 1, "start={start}: chunk size starved to 0");

            // (b) the chosen chunk fits EVERY constraint...
            assert!(fits_all(n, start, &lim), "start={start}: chunk n={n} overflows");
            // ...and is maximal — one more token violates something
            // (above the n>=1 floor).
            if n > 1 {
                assert!(!fits_all(n + 1, start, &lim), "start={start}: chunk n={n} not maximal");
            }

            // (c) monotonically non-increasing as the attention window grows.
            assert!(n <= prev, "start={start}: chunk n={n} grew vs prev {prev}");
            prev = n;
        }
    }

    #[test]
    fn scales_with_lane_b_budget() {
        let mut small = qwen_limits();
        small.lane_b = 128 * 1024 * 1024;
        let mut big = qwen_limits();
        big.lane_b = 512 * 1024 * 1024;
        // Open up A/C so Lane B is the binding constraint in both cases.
        big.lane_a = u64::MAX;
        big.lane_c = u64::MAX;
        small.lane_a = u64::MAX;
        small.lane_c = u64::MAX;
        let n_small = prefill_chunk_size(0, &small);
        let n_big = prefill_chunk_size(0, &big);
        assert!(n_big > n_small, "bigger Lane B should allow a bigger chunk: {n_big} vs {n_small}");
    }

    #[test]
    fn binding_cap_clamps_when_lane_b_is_huge() {
        // Phase M scenario: device-derived Lane B larger than the ~2 GB
        // single-binding limit. The scores binding must still fit it.
        let mut lim = qwen_limits();
        lim.lane_b = 16 * 1024 * 1024 * 1024; // 16 GB lane
        lim.lane_a = u64::MAX;
        lim.lane_c = u64::MAX;
        for &start in &[0usize, 1000, 10_000] {
            let n = prefill_chunk_size(start, &lim);
            assert!(
                scores_bytes(n, start, &lim) <= lim.binding_max,
                "start={start}: scores binding {} exceeds cap {}",
                scores_bytes(n, start, &lim),
                lim.binding_max,
            );
            // Maximality against the binding cap specifically.
            assert!(
                scores_bytes(n + 1, start, &lim) > lim.binding_max,
                "start={start}: n={n} not maximal against binding cap",
            );
        }
    }

    #[test]
    fn small_lane_a_binds_before_lane_b() {
        // Lane A holds gate+up (~44 KB/token at Qwen 3B) — with a small
        // Lane A and a huge Lane B, the linear Lane A constraint must win.
        let mut lim = qwen_limits();
        lim.lane_a = 64 * 1024 * 1024;
        lim.lane_b = 4 * 1024 * 1024 * 1024;
        let n = prefill_chunk_size(0, &lim);
        assert!(lane_a_bytes(n, &lim) <= lim.lane_a, "chunk n={n} overflows Lane A");
        assert!(
            lane_a_bytes(n + 1, &lim) > lim.lane_a,
            "n={n} not maximal against Lane A",
        );
    }
}

impl GpuEngine {
    /// Largest token count whose f32 `BlockScratch` fits ALL transient
    /// lanes AND the device's single-binding limit at the given attention
    /// start position. See [`prefill_chunk_size`] for the four constraints.
    ///
    /// Phase L replaced the stale 512 MB `SCORES_BUDGET_BYTES` constant
    /// (cortex-cloud's old `safe_chunk_size`) with a live read of Lane B
    /// capacity. Phase M extends it: lanes are now sized independently
    /// from the device budget, so Lane A/C linear footprints and
    /// `max_storage_buffer_binding_size` (~2 GB; the cap on the single
    /// `scores` binding) are checked too. The chunker auto-adapts to
    /// whatever the heaps are sized to, on any GPU.
    pub fn safe_prefill_chunk_size(&self, start_pos: usize) -> usize {
        let attn0 = self.cpu.blocks()[0].attention();
        let embed = self.embed_dim();
        let intermediate = self.cpu.blocks()[0]
            .ffn()
            .as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .map(|f| f.intermediate_size())
            .unwrap_or(embed * 4);
        // ~3% slack against alignment padding / fragmentation. The lanes
        // are freshly empty each prefill forward (RAII drops the prior
        // scratch), so nearly the full capacity is available.
        let lim = ChunkLimits {
            n_heads: attn0.n_heads(),
            n_kv_heads: attn0.n_kv_heads(),
            head_dim: attn0.head_dim(),
            embed,
            intermediate,
            lane_a: self.gpu.transient_heap_a.capacity() * 97 / 100,
            lane_b: self.gpu.transient_heap_b.capacity() * 97 / 100,
            lane_c: self.gpu.transient_heap_c.capacity() * 97 / 100,
            binding_max: self.gpu.device.limits().max_storage_buffer_binding_size as u64,
        };
        prefill_chunk_size(start_pos, &lim)
    }

}
