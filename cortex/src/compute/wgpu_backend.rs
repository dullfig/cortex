//! WGPU compute backend — GPU inference via compute shaders.
//!
//! `GpuEngine` runs the full transformer forward pass in a single
//! command buffer, with f16-packed weights, precomputed RoPE, KV caches
//! on GPU, and optional NeuralKV injection. Only 4 bytes read back per
//! generated token.
//!
//! The shaders in `src/compute/shaders/` handle both single-token
//! (decode) and batch (prefill) paths. Weights are stored as f16 pairs
//! packed into u32.

#[cfg(feature = "gpu")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Param structs — repr(C) uniforms passed to shaders
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatvecParams {
    pub rows: u32,
    pub cols: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    pub n: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeParams {
    pub n_heads: u32,
    pub head_dim: u32,
    pub position: u32,
    pub half_dim: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SiluMulParams {
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AddInplaceParams {
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KvWriteParams {
    pub kv_dim: u32,
    pub position: u32,
    pub max_seq: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnScoreParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub heads_per_kv: u32,
    pub kv_dim: u32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxParams {
    pub n_heads: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnValueParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub heads_per_kv: u32,
    pub kv_dim: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArgmaxParams {
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulParams {
    pub rows: u32,
    pub cols: u32,
    pub n_tokens: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormBatchParams {
    pub n: u32,
    pub eps: f32,
    pub n_tokens: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeBatchParams {
    pub n_heads: u32,
    pub head_dim: u32,
    pub start_pos: u32,
    pub half_dim: u32,
    pub n_tokens: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KvWriteBatchParams {
    pub kv_dim: u32,
    pub start_pos: u32,
    pub max_seq: u32,
    pub n_tokens: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnScoreBatchParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub heads_per_kv: u32,
    pub kv_dim: u32,
    pub scale: f32,
    pub n_tokens: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxBatchParams {
    pub n_heads: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub n_tokens: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnValueBatchParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub max_seq: u32,
    pub heads_per_kv: u32,
    pub kv_dim: u32,
    pub n_tokens: u32,
}

// ---------------------------------------------------------------------------
// Pipelines — all compiled compute pipelines
// ---------------------------------------------------------------------------

/// All compiled compute pipelines for GPU inference.
pub struct Pipelines {
    // Single-token (decode). Only matvec + the polar oneshots survive —
    // others are dispatched only via *_batch variants in production.
    pub matvec: wgpu::ComputePipeline,
    pub attn_score_polar: wgpu::ComputePipeline,
    pub attn_value_polar: wgpu::ComputePipeline,
    pub derotate: wgpu::ComputePipeline,
    /// Phase C3 polar: packed-f16 output variant of derotate (writes
    /// packed scratch.attn_out).
    pub derotate_packed: wgpu::ComputePipeline,
    pub kv_compress_polar: wgpu::ComputePipeline,
    pub rotate_q: wgpu::ComputePipeline,
    /// Phase C3 polar: packed-f16 input variant of rotate_q (reads
    /// packed scratch.q). Output rq stays f32.
    pub rotate_q_packed: wgpu::ComputePipeline,
    pub attn_score_polar_batch: wgpu::ComputePipeline,
    pub attn_value_polar_batch: wgpu::ComputePipeline,
    // Batch (prefill)
    pub matmul: wgpu::ComputePipeline,
    /// Shared-memory tiled matmul (16×16 output tile, TILE_K=16). Used
    /// for prefill (n_tokens >= 16); decode falls back to legacy matmul.
    /// F32 input, F32 output. See `shaders/matmul_shared.wgsl`.
    pub matmul_shared: wgpu::ComputePipeline,
    /// Phase C1: packed-input, f32-output variant of matmul_shared.
    /// Used for matmuls that read packed scratch.normed (Q/K/V/O/gate/up
    /// projections — i.e. everything except down_proj which still reads
    /// f32 scratch.activated in C1). See `matmul_shared_pin_fout.wgsl`.
    pub matmul_shared_pin_fout: wgpu::ComputePipeline,
    /// Phase C1: packed-input variant of the per-output legacy matmul,
    /// used for decode (n_tokens < TILE_N) reading packed scratch.normed.
    pub matmul_pin: wgpu::ComputePipeline,
    /// Phase C2: packed-input + packed-output decode-path matmul.
    pub matmul_pin_pout: wgpu::ComputePipeline,
    /// Phase C2: SiLU(gate)*up with packed gate/up/output buffers.
    pub silu_mul_batch_packed: wgpu::ComputePipeline,
    /// Phase C3: prefill matmul with packed input + packed output.
    /// Used for Q/K/V/O/down projections when scratch.q/k/v/attn_out/
    /// projected are packed.
    pub matmul_shared_pin_pout: wgpu::ComputePipeline,
    /// Phase C3: packed RoPE (in-place on packed q/k scratch).
    pub rope_batch_packed: wgpu::ComputePipeline,
    /// Phase C3: packed bias add (Qwen Q/K/V biases on packed scratch).
    pub bias_add_batch_packed: wgpu::ComputePipeline,
    /// Phase C3: packed residual add (both sides packed —
    /// hidden_buf += scratch.projected when projected is packed).
    pub add_inplace_batch_packed: wgpu::ComputePipeline,
    /// Fused gate + up SwiGLU projection. One dispatch reads input once,
    /// computes both gate and up outputs (2 rows × 2 projections per
    /// thread). Halves the input HBM bandwidth for the gate/up pair.
    /// Phase C1: input is packed f16; outputs still f32 in C1.
    /// See `shaders/matmul_gate_up_shared.wgsl`.
    pub matmul_gate_up_shared: wgpu::ComputePipeline,
    pub rmsnorm_batch: wgpu::ComputePipeline,
    /// Phase B variant: reads packed-f16 input (hidden_buf), writes f32
    /// output (scratch.normed). Used by per-block attn_norm / ffn_norm.
    pub rmsnorm_batch_packed_to_f32: wgpu::ComputePipeline,
    /// Phase B variant: reads packed-f16 input AND writes packed-f16
    /// output. Used by the FINAL norm (hidden_buf → normed_buf), both
    /// packed in Phase B.
    pub rmsnorm_batch_packed_to_packed: wgpu::ComputePipeline,
    /// Phase C1: f32 input, packed f16 output. Used by BitNet
    /// o_sub_norm (scratch.attn_out f32 → scratch.normed packed).
    pub rmsnorm_batch_f32_to_packed: wgpu::ComputePipeline,
    pub rope_batch: wgpu::ComputePipeline,
    pub silu_mul_batch: wgpu::ComputePipeline,
    pub add_inplace_batch: wgpu::ComputePipeline,
    pub add_broadcast_batch: wgpu::ComputePipeline,
    pub kv_write_batch: wgpu::ComputePipeline,
    pub attn_score_batch: wgpu::ComputePipeline,
    pub softmax_batch: wgpu::ComputePipeline,
    pub attn_value_batch: wgpu::ComputePipeline,
    /// Fused score+softmax+value (FlashAttention-1, online softmax).
    /// Replaces the 3-shader path for production chat completion.
    /// The 3-shader path stays compiled because it's needed by the
    /// retrieval trace mode (pre_softmax_capture).
    pub attn_fused_batch: wgpu::ComputePipeline,
    // Broadcast bias add (Q/K/V biases for Qwen-family)
    pub bias_add_batch: wgpu::ComputePipeline,
    /// Single-workgroup argmax reduction over an f32 logits buffer.
    /// Used by the GPU LM-head greedy decode path to produce a 4-byte
    /// token id without reading back the full vocab logits.
    pub argmax_vocab: wgpu::ComputePipeline,
    /// QJL (Quantized Johnson-Lindenstrauss) encoder for K residuals.
    /// One u32 word of sign bits per (pos, head) entry. Dispatched
    /// after `kv_compress_polar` when a polar cache has QJL enabled.
    pub kv_qjl_encode: wgpu::ComputePipeline,
    /// Batch attention scores against polar K WITH QJL correction.
    /// Drop-in for `attn_score_polar_batch` when the polar cache has
    /// QJL signs. Brings ~0.84 → ~0.95 cosine vs f32 attention.
    pub attn_score_polar_qjl_batch: wgpu::ComputePipeline,
    /// Phase O: QJL encoder for V residuals — multi-word signs
    /// (n_proj=256 → 8 u32/entry) + residual norm output for the
    /// Γ-scaled value correction.
    pub kv_qjl_encode_v: wgpu::ComputePipeline,
    /// Phase O pass A: per-(query, head, projection) sign-weighted
    /// softmax mass `C_j = Σ_t w_t·rnorm_t·s_tj` — the sum-swap that
    /// makes the V vector correction affordable.
    pub qjl_value_weights: wgpu::ComputePipeline,
    /// Phase O pass B: attn_value_polar_batch + the Γ-scaled residual
    /// correction from pass A's C accumulator. Drop-in for
    /// `attn_value_polar_batch` when the polar cache has QJL.
    pub attn_value_polar_qjl_batch: wgpu::ComputePipeline,
}

impl Pipelines {
    /// Compile all compute pipelines from embedded shader sources.
    pub fn compile(device: &wgpu::Device) -> Self {
        let make = |src: &str, label: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-derive from shader
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let make_with_entry = |src: &str, label: &str, entry: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            // Single-token
            matvec: make(include_str!("shaders/matvec.wgsl"), "matvec"),
            attn_score_polar: make(include_str!("shaders/attn_score_polar.wgsl"), "attn_score_polar"),
            attn_value_polar: make(include_str!("shaders/attn_value_polar.wgsl"), "attn_value_polar"),
            derotate: make(include_str!("shaders/derotate.wgsl"), "derotate"),
            derotate_packed: make(include_str!("shaders/derotate_packed.wgsl"), "derotate_packed"),
            kv_compress_polar: make(include_str!("shaders/kv_compress_polar.wgsl"), "kv_compress_polar"),
            rotate_q: make(include_str!("shaders/rotate_q.wgsl"), "rotate_q"),
            rotate_q_packed: make(include_str!("shaders/rotate_q_packed.wgsl"), "rotate_q_packed"),
            attn_score_polar_batch: make(include_str!("shaders/attn_score_polar_batch.wgsl"), "attn_score_polar_batch"),
            attn_value_polar_batch: make(include_str!("shaders/attn_value_polar_batch.wgsl"), "attn_value_polar_batch"),
            // Batch
            matmul: make(include_str!("shaders/matmul.wgsl"), "matmul"),
            matmul_shared: make(include_str!("shaders/matmul_shared.wgsl"), "matmul_shared"),
            matmul_shared_pin_fout: make(include_str!("shaders/matmul_shared_pin_fout.wgsl"), "matmul_shared_pin_fout"),
            matmul_pin: make(include_str!("shaders/matmul_pin.wgsl"), "matmul_pin"),
            matmul_pin_pout: make(include_str!("shaders/matmul_pin_pout.wgsl"), "matmul_pin_pout"),
            silu_mul_batch_packed: make(include_str!("shaders/silu_mul_batch_packed.wgsl"), "silu_mul_batch_packed"),
            matmul_shared_pin_pout: make(include_str!("shaders/matmul_shared_pin_pout.wgsl"), "matmul_shared_pin_pout"),
            rope_batch_packed: make(include_str!("shaders/rope_batch_packed.wgsl"), "rope_batch_packed"),
            bias_add_batch_packed: make(include_str!("shaders/bias_add_batch_packed.wgsl"), "bias_add_batch_packed"),
            add_inplace_batch_packed: make(include_str!("shaders/add_inplace_batch_packed.wgsl"), "add_inplace_batch_packed"),
            matmul_gate_up_shared: make(include_str!("shaders/matmul_gate_up_shared.wgsl"), "matmul_gate_up_shared"),
            rmsnorm_batch: make(include_str!("shaders/rmsnorm_batch.wgsl"), "rmsnorm_batch"),
            rmsnorm_batch_packed_to_f32: make(include_str!("shaders/rmsnorm_batch_packed_to_f32.wgsl"), "rmsnorm_batch_packed_to_f32"),
            rmsnorm_batch_packed_to_packed: make(include_str!("shaders/rmsnorm_batch_packed_to_packed.wgsl"), "rmsnorm_batch_packed_to_packed"),
            rmsnorm_batch_f32_to_packed: make(include_str!("shaders/rmsnorm_batch_f32_to_packed.wgsl"), "rmsnorm_batch_f32_to_packed"),
            rope_batch: make(include_str!("shaders/rope_batch.wgsl"), "rope_batch"),
            silu_mul_batch: make(include_str!("shaders/silu_mul_batch.wgsl"), "silu_mul_batch"),
            add_inplace_batch: make(include_str!("shaders/add_inplace_batch.wgsl"), "add_inplace_batch"),
            add_broadcast_batch: make(include_str!("shaders/add_broadcast_batch.wgsl"), "add_broadcast_batch"),
            kv_write_batch: make(include_str!("shaders/kv_write_batch.wgsl"), "kv_write_batch"),
            attn_score_batch: make(include_str!("shaders/attn_score_batch.wgsl"), "attn_score_batch"),
            softmax_batch: make(include_str!("shaders/softmax_batch.wgsl"), "softmax_batch"),
            attn_value_batch: make(include_str!("shaders/attn_value_batch.wgsl"), "attn_value_batch"),
            attn_fused_batch: make(include_str!("shaders/attn_fused_batch.wgsl"), "attn_fused_batch"),
            bias_add_batch: make(include_str!("shaders/bias_add_batch.wgsl"), "bias_add_batch"),
            argmax_vocab: make(include_str!("shaders/argmax_vocab.wgsl"), "argmax_vocab"),
            kv_qjl_encode: make(include_str!("shaders/kv_qjl_encode.wgsl"), "kv_qjl_encode"),
            attn_score_polar_qjl_batch: make(include_str!("shaders/attn_score_polar_qjl_batch.wgsl"), "attn_score_polar_qjl_batch"),
            kv_qjl_encode_v: make(include_str!("shaders/kv_qjl_encode_v.wgsl"), "kv_qjl_encode_v"),
            qjl_value_weights: make(include_str!("shaders/qjl_value_weights.wgsl"), "qjl_value_weights"),
            attn_value_polar_qjl_batch: make(include_str!("shaders/attn_value_polar_qjl_batch.wgsl"), "attn_value_polar_qjl_batch"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDevice — shared device + queue + pipelines
// ---------------------------------------------------------------------------

/// Slot size (bytes) for the params buffer ring pool. Sized to fit any
/// per-dispatch params struct cortex creates today (largest seen is well
/// under 128B). Asserted at `create_params_buffer` call time.
pub const PARAMS_POOL_SLOT_BYTES: u64 = 256;

/// Number of slots in the params buffer ring pool. Must comfortably
/// exceed the LARGEST aggregate in-flight slot count across ALL
/// concurrent forwards so the ring's natural wrap-around can't
/// reuse a slot while the previous occupant is still bound in flight.
///
/// Single-forward demand at Qwen-3B polar retrieve: ~360 slots
/// (36 layers × ~10 dispatches per layer). cortex-cloud's tokio
/// handlers can run multiple forwards concurrently (`Arc<GpuEngine>`
/// with `&self` forward methods), so the operating envelope is
/// (concurrent forwards) × (in-flight slots per forward between
/// chunked submits, ~135 with the default 9-layer chunking).
///
/// Phase J (2026-06-09) bumped this from 2048 to 16384 to close a
/// latent wrap-around race: at 2048 slots, ~15 concurrent forwards
/// could wrap the counter back to a slot still bound in another
/// forward's not-yet-submitted command buffer, causing silent params
/// corruption (queue.write_buffer happens before the previous
/// dispatch's submit reads the slot). At 16384 the threshold moves
/// to ~120 concurrent forwards — well outside any plausible
/// operating envelope, even adversarial stress tests on the H100
/// deployment. Memory cost: 16384 × 256 B = 4 MB; trivial against
/// ~6 GB of weights + transient heaps.
///
/// The bug stays *statistically* possible at the new bound, not
/// eliminated. The eliminate-by-construction fix would be a
/// `VramPool` migration with `PoolSlot::drop`-after-submit
/// keepalive plumbing through 44 dispatch sites — see the
/// `ParamsBufferPool` doc comment for why that's the wrong
/// tradeoff for cortex's actual concurrency profile.
pub const PARAMS_POOL_SLOT_COUNT: usize = 16384;

/// Ring of pre-allocated uniform buffers used by `create_params_buffer`.
/// See that method's doc comment for why this exists.
///
/// # Why this is NOT a `vram_heap::VramPool`
///
/// vram-heap's `VramPool` has an *explicit* slot lifecycle: `acquire()`
/// pops an index from a free-list, `PoolSlot::drop` pushes it back.
/// `ParamsBufferPool` has an *implicit* slot lifecycle: `next_slot()`
/// is an atomic round-robin over the ring and the slot is "available
/// again" the instant it returns.
///
/// cortex's dispatch helpers do `acquire → write → bind → record →
/// return`. The returned `wgpu::Buffer` (an Arc handle to the slot)
/// drops at function return, but the GPU's actual `queue.submit` +
/// dispatch read happens later — often hundreds of microseconds later,
/// batched at the end of the forward function. The ring tolerates this
/// because nothing overwrites a slot until `PARAMS_POOL_SLOT_COUNT`
/// more allocations later, by which point the GPU has long since
/// finished reading the in-flight slots — PROVIDED total in-flight
/// slot count across all concurrent forwards stays below the ring
/// size. Phase J sized the ring (16384 slots) for ~120 concurrent
/// forwards' worth of headroom; see `PARAMS_POOL_SLOT_COUNT` for the
/// math.
///
/// Migrating to `VramPool` would force every dispatch helper to grow a
/// `&mut Vec<PoolSlot>` keepalive parameter threaded from the forward
/// function down through every dispatch — ~44 call sites, all-or-
/// nothing plumbing. The bigger-ring approach (Phase J) plus the
/// `stats()` method below gives us both bug closure under any
/// realistic concurrency AND the observability a vram-heap-backed
/// pool would have provided, at the cost of being statistically
/// (not constructively) safe at the bound. cortex doesn't see
/// adversarial concurrency profiles; the tradeoff is right.
///
/// `VramPool` is the right answer for callers that DO want explicit
/// per-slot lifetime tracking — e.g. memex's KV-cache slot reuse or
/// ternary-rs's quantized-tensor allocation patterns where slots
/// outlive a single function-scope.
pub struct ParamsBufferPool {
    slots: Vec<wgpu::Buffer>,
    next: std::sync::atomic::AtomicUsize,
}

/// Pool usage snapshot. Same shape as `vram_heap::HeapStats` so future
/// diagnostic helpers can dump it uniformly alongside the vram-heap
/// stats. `next_slot()` returns `total_acquired % slot_count`, so
/// `wrap_count` answers "how stressed is the ring under current load".
#[derive(Debug, Clone, Copy)]
pub struct ParamsPoolStats {
    /// Lifetime acquire count — every `next_slot()` call bumps this.
    pub total_acquired: usize,
    /// `total_acquired / slot_count`. A wrap_count growing absurdly
    /// fast (per unit wall time) under sustained load is the signal
    /// to alarm: the ring is being stressed beyond its safety margin.
    pub wrap_count: usize,
    pub slot_count: usize,
    pub slot_bytes: u64,
}

impl ParamsBufferPool {
    pub fn new(device: &wgpu::Device) -> Self {
        let slots = (0..PARAMS_POOL_SLOT_COUNT)
            .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("params_pool[{i}]")),
                size: PARAMS_POOL_SLOT_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();
        Self {
            slots,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Hand out the next slot in round-robin order. Returned wgpu::Buffer
    /// is a refcounted handle to a pool-resident allocation — Drop just
    /// decrements the refcount; the underlying buffer lives until the
    /// pool itself drops (which is at GpuDevice tear-down).
    pub fn next_slot(&self) -> wgpu::Buffer {
        let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.slots.len();
        self.slots[idx].clone()
    }

    /// Pool usage snapshot. Reads the atomic counter only (no
    /// per-slot tracking — see the struct doc for why the implicit
    /// lifecycle precludes a `live_count` field).
    pub fn stats(&self) -> ParamsPoolStats {
        let total = self.next.load(std::sync::atomic::Ordering::Relaxed);
        let slot_count = self.slots.len();
        ParamsPoolStats {
            total_acquired: total,
            wrap_count: total / slot_count,
            slot_count,
            slot_bytes: PARAMS_POOL_SLOT_BYTES,
        }
    }
}

/// Shared GPU context: device, queue, compiled pipelines, and the
/// vram-heap arenas for per-call transient allocations.
///
/// Created once at startup, shared across all layers via `Arc<GpuDevice>`.
///
/// # vram-heap usage
///
/// `transient_heap_a`, `transient_heap_b`, and `transient_heap_c` are
/// THREE device-local heaps that callers use as lanes for dispatches
/// that need disjoint R and RW bindings: wgpu/Vulkan rejects binding
/// `STORAGE_READ_ONLY` and `STORAGE_READ_WRITE` to the same backing
/// buffer within a single dispatch (even at disjoint sub-ranges). By
/// allocating R inputs and RW outputs from different heaps, both
/// bindings reference *different* backings, so the buffer-level
/// tracker is happy.
///
/// The 3-lane scheme (added in Phase D for the BlockScratch
/// migration) carries `hidden_buf`, `rotated_buf`, and a subset of
/// BlockScratch on heap A; another subset on heap B; the remainder
/// on heap C. See `PolarBlockScratch` in `gpu_engine.rs` for the
/// per-buffer lane assignment and the conflict-graph rationale.
///
/// All heaps free their transient regions via RAII Drop on
/// `VramAllocation`; coalesce returns each lane to a single full
/// span between forward passes.
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Pipelines,
    pub params_pool: ParamsBufferPool,
    /// Device VRAM budget (Phase M). Detected through the Vulkan
    /// backend at startup (fallback: CORTEX_VRAM_TOTAL_MB, default
    /// 8 GB). Every DeviceLocal heap — the four globals below plus the
    /// per-cache `gpu_kv` / `polar_kv` heaps — reserves against it via
    /// `new_in_budget`, so VRAM over-commit fails loudly at heap
    /// construction (`BudgetExceeded` naming label/requested/committed/
    /// total) instead of as a driver-level OOM mid-inference. Heap drop
    /// auto-releases its reservation.
    pub vram_budget: Arc<::vram_heap::DeviceBudget>,
    /// Lane-A device-local heap. Holds `hidden_buf`, `rotated_buf`,
    /// `PolarBlockScratch::{gate, up}`.
    pub transient_heap_a: Arc<::vram_heap::VramHeap>,
    /// Lane-B device-local heap. Holds
    /// `PolarBlockScratch::{normed, activated, attn_out, scores}`.
    pub transient_heap_b: Arc<::vram_heap::VramHeap>,
    /// Lane-C device-local heap. Holds
    /// `PolarBlockScratch::{q, k, v, projected}`.
    pub transient_heap_c: Arc<::vram_heap::VramHeap>,
    /// Static weights heap (Phase G). Holds every never-freed
    /// allocation: GpuFloatLinear weights, GpuBlock norm + bias
    /// weights, RoPE cos/sin tables, LM head weights, final norm
    /// weight. All allocations made via `allocate_static` so heap
    /// drop emits no leak warning at process exit. Default capacity
    /// 7 GB (covers Qwen 3B's ~6 GB at packed f16 with slack);
    /// tunable via CORTEX_VRAM_HEAP_WEIGHTS_MB.
    pub weights_heap: Arc<::vram_heap::VramHeap>,
    /// Host-visible readback heap. Use for staging buffers that
    /// receive `copy_buffer_to_buffer` destinations and then get
    /// host-mapped for read. Capture-staging buffers in retrieval
    /// trace forwards land here.
    pub host_readback_heap: Arc<::vram_heap::VramHeap>,
    /// The device-probe `DeviceProfile` cortex was built on: which
    /// adapter was selected and its queried + measured capabilities
    /// (VRAM, binding/workgroup limits, supports-f16, and the measured
    /// f16/bandwidth numbers). Logged at startup; read by future phases
    /// (tile/workgroup specialization, f16-arith kernel selection) that
    /// aren't wired yet.
    pub profile: device_probe::DeviceProfile,
}

/// Identifier for one of the five vram-heaps cortex constructs on a
/// `GpuDevice`. The label matches the boot-log string and the
/// metrics gauge label (Phase K dashboard).
#[derive(Copy, Clone, Debug)]
pub enum VramHeapId {
    TransientA,
    TransientB,
    TransientC,
    Weights,
    HostReadback,
}

impl VramHeapId {
    /// Stable string label for log lines and Prometheus metric labels.
    pub fn label(self) -> &'static str {
        match self {
            VramHeapId::TransientA => "transient_a",
            VramHeapId::TransientB => "transient_b",
            VramHeapId::TransientC => "transient_c",
            VramHeapId::Weights => "weights",
            VramHeapId::HostReadback => "host_readback",
        }
    }
}

impl GpuDevice {
    /// Try to create a GPU device with all pipelines compiled.
    /// Returns `None` if no usable adapter is found.
    ///
    /// Device selection, the live `wgpu::Device`/`Queue`, and the
    /// authoritative VRAM number come from the `device-probe` crate — the
    /// leaf of the `device-probe → vram-heap → cortex` stack. It
    /// enumerates every adapter, profiles each, refuses software/CPU
    /// fallbacks (silently running on llvmpipe at ~1000× slow is the
    /// worst failure mode), and picks the best for cortex's
    /// bandwidth-bound workload (the V100 over a T4 on a heterogeneous
    /// box — never just adapter 0), then measures what it is actually
    /// fast at. [`Self::from_selection`] builds everything cortex-specific
    /// on that warm device. Set `CORTEX_PROBE_NO_MEASURE=1` to skip the
    /// few-seconds measure pass on ephemeral cold starts.
    pub fn try_new() -> Option<Self> {
        let hint = device_probe::WorkloadHint::BandwidthBound;
        let sel = if std::env::var("CORTEX_PROBE_NO_MEASURE").as_deref() == Ok("1") {
            device_probe::select_best(hint)
        } else {
            device_probe::select_best_measured(hint)
        };
        let sel = match sel {
            Ok(sel) => sel,
            Err(e) => {
                tracing::error!(error = %e, "device-probe found no usable GPU adapter");
                return None;
            }
        };
        Self::from_selection(sel)
    }

    /// Build a `GpuDevice` on a device-probe [`device_probe::Selection`] —
    /// the warm, already-selected `wgpu::Device`/`Queue` plus its
    /// `DeviceProfile`. Owns everything cortex-specific: the VRAM budget
    /// (fed by the profile's authoritative `vram_total_bytes`), the five
    /// vram-heaps, the compiled pipelines, and the params pool.
    pub fn from_selection(sel: device_probe::Selection) -> Option<Self> {
        let device_probe::Selection { profile, device, queue } = sel;

        // Log the WHOLE profile at startup (device-probe CLAUDE.md
        // mandate): the first thing you want when something is
        // mysteriously slow is what cortex THOUGHT it was running on —
        // including the measured f16 speedup (a P40 reads ~0.016 here
        // despite reporting shader-f16 supported).
        tracing::info!(
            adapter = %profile.adapter_name,
            backend = ?profile.backend,
            device_type = ?profile.device_type,
            vram_total_mb = ?profile.vram_total_bytes.map(|b| b / (1024 * 1024)),
            max_storage_binding_mb = profile.max_storage_buffer_binding_size / (1024 * 1024),
            workgroup_storage_bytes = profile.max_compute_workgroup_storage_size,
            supports_f16 = profile.supports_shader_f16,
            f16_matmul_speedup = ?profile.f16_matmul_speedup,
            measured_bandwidth_gbps = ?profile.measured_bandwidth_gbps,
            "device-probe DeviceProfile",
        );

        // Timestamp queries: recompute support from the device-probe-
        // created device's ACTUAL enabled features (not an adapter).
        // device-probe enables TIMESTAMP_QUERY (and, with the companion
        // fix, TIMESTAMP_QUERY_INSIDE_ENCODERS) when the adapter supports
        // them; if INSIDE_ENCODERS isn't enabled this is simply false and
        // the PASS-bundle timers self-disable (timestamp_us telemetry
        // stays zero) — no validation error, no breakage.
        let dev_features = device.features();
        let timestamp_supported = dev_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && dev_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        if timestamp_supported {
            tracing::info!(
                period_ns = queue.get_timestamp_period(),
                "GPU timestamp queries enabled",
            );
        } else {
            tracing::info!("GPU timestamp queries NOT enabled on the device");
        }

        let pipelines = Pipelines::compile(&device);
        tracing::info!("compiled 37 GPU compute pipelines");

        let params_pool = ParamsBufferPool::new(&device);
        tracing::info!(
            slot_count = PARAMS_POOL_SLOT_COUNT,
            slot_bytes = PARAMS_POOL_SLOT_BYTES,
            total_kb = (PARAMS_POOL_SLOT_COUNT as u64 * PARAMS_POOL_SLOT_BYTES) / 1024,
            "params buffer pool allocated",
        );

        // Phase M device VRAM budget — now SOURCED FROM device-probe.
        // The profile carries the authoritative device-local VRAM (the
        // same Vulkan/ash sum vram-heap used; byte-identical per
        // HANDSHAKE.md §Parity). CORTEX_VRAM_TOTAL_MB still wins as an
        // explicit operator override (leave VRAM for other processes on a
        // shared card, or force budget-exceeded behavior); otherwise feed
        // the profile number into DeviceBudget::explicit. All DeviceLocal
        // heaps reserve against this budget, so over-committing the card
        // is a loud BudgetExceeded at construction instead of a
        // driver-level OOM mid-inference. (vram-heap keeps its own
        // detect_or as a fallback for non-device-probe callers — this is
        // the coordinated migration's step 2; do not strand it.)
        let vram_budget = match std::env::var("CORTEX_VRAM_TOTAL_MB")
            .ok().and_then(|s| s.parse::<u64>().ok())
        {
            Some(mb) => ::vram_heap::DeviceBudget::explicit(mb * 1024 * 1024),
            None => ::vram_heap::DeviceBudget::explicit(
                profile.vram_budget_or(8192 * 1024 * 1024),
            ),
        };
        tracing::info!(
            total_mb = vram_budget.total() / (1024 * 1024),
            source = ?vram_budget.source(),
            "device VRAM budget",
        );

        // vram-heap arenas. Three device-local heaps form the 3-lane
        // scheme that the polar forward path uses to keep R inputs
        // and RW outputs on different backings within one dispatch
        // (wgpu/Vulkan rejects same-buffer R+RW even at disjoint
        // sub-ranges). Phase D required a 3rd lane because the
        // hidden_buf/rotated_buf + BlockScratch conflict graph is
        // not 2-colorable.
        //
        // Phase M sizing: env vars always win; otherwise lane sizes
        // derive from the budget headroom left after the weights heap.
        // Floors are the old 128 MB defaults (never regress); caps are
        // where extra capacity stops buying prefill throughput — Lane B
        // past ~2.3 GB is wasted because the single `scores` storage
        // binding is capped at max_storage_buffer_binding_size (~2 GB),
        // and the chunker (`safe_prefill_chunk_size`) enforces all
        // lane + binding constraints, so any sizing here is *correct*;
        // bigger lanes just mean fewer, larger prefill chunks. The
        // remaining headroom is deliberately left unreserved for the
        // per-cache heaps (gpu_kv / polar_kv) that come and go with
        // the cache pool.
        //
        // Phase G note (weights): sized for Qwen 3B-class models
        // (~6 GB at packed f16) with ~1 GB slack; model size isn't
        // known at device init, so this stays env-driven rather than
        // derived. Set CORTEX_VRAM_HEAP_WEIGHTS_MB smaller for
        // TinyLlama-class models or bigger for 7B+.
        let heap_weights_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_WEIGHTS_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(7168);
        let heap_readback_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_READBACK_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let total_mb = vram_budget.total() / (1024 * 1024);
        let headroom_mb = total_mb.saturating_sub(heap_weights_mb);
        let derived = |frac_of: u64, floor: u64, cap: u64| frac_of.clamp(floor, cap);
        let heap_a_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_A_MB")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or_else(|| derived(headroom_mb / 8, 128, 1152));
        let heap_b_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_B_MB")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or_else(|| derived(headroom_mb / 4, 128, 2304));
        let heap_c_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_C_MB")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or_else(|| derived(headroom_mb / 16, 128, 576));
        tracing::info!(
            lane_a_mb = heap_a_mb, lane_b_mb = heap_b_mb, lane_c_mb = heap_c_mb,
            weights_mb = heap_weights_mb, readback_mb = heap_readback_mb,
            "vram heap sizes (env override or budget-derived)",
        );
        let transient_heap_a = ::vram_heap::VramHeap::new_in_budget(
            &device,
            &vram_budget,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_a_mb * 1024 * 1024,
            "cortex.transient.a",
        ).expect("vram-heap A construction failed");
        let transient_heap_b = ::vram_heap::VramHeap::new_in_budget(
            &device,
            &vram_budget,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_b_mb * 1024 * 1024,
            "cortex.transient.b",
        ).expect("vram-heap B construction failed");
        let transient_heap_c = ::vram_heap::VramHeap::new_in_budget(
            &device,
            &vram_budget,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_c_mb * 1024 * 1024,
            "cortex.transient.c",
        ).expect("vram-heap C construction failed");
        let weights_heap = ::vram_heap::VramHeap::new_in_budget(
            &device,
            &vram_budget,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_weights_mb * 1024 * 1024,
            "cortex.weights",
        ).expect("vram-heap weights construction failed");
        // HostReadback is host-visible system RAM, not device VRAM —
        // deliberately NOT budgeted.
        let host_readback_heap = ::vram_heap::VramHeap::new(
            &device,
            ::vram_heap::MemoryTier::HostReadback,
            heap_readback_mb * 1024 * 1024,
            "cortex.host_readback",
        ).expect("vram-heap readback construction failed");

        Some(Self {
            device, queue, pipelines, params_pool, vram_budget,
            transient_heap_a, transient_heap_b, transient_heap_c, weights_heap, host_readback_heap,
            profile,
        })
    }

    /// Create a bind group from a pipeline and a list of buffers.
    ///
    /// Buffers are bound to `@binding(0)`, `@binding(1)`, etc.
    pub fn make_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        // Delegate to the BindingResource variant. Bindings come out
        // as `buf.as_entire_binding()` per the original semantics
        // (no offset/size; the whole buffer).
        let resources: Vec<wgpu::BindingResource<'_>> = buffers
            .iter()
            .map(|b| b.as_entire_binding())
            .collect();
        self.make_bind_group_with(pipeline, resources)
    }

    /// Build a bind group from an explicit list of `BindingResource`s.
    /// Bindings are numbered by their index in `resources` — same
    /// convention as [`Self::make_bind_group`], just with a flexible
    /// per-binding resource type.
    ///
    /// Use this when at least one binding is a sub-range of some
    /// larger buffer (typically a vram-heap backing buffer). For a
    /// `VramAllocation` `alloc`, pass `alloc.binding()`. For a whole
    /// wgpu::Buffer `buf`, pass `buf.as_entire_binding()`. The list
    /// can mix the two freely.
    ///
    /// `make_bind_group_with` is the keystone that unblocks
    /// substrate migrations: future per-call buffers (hidden_buf,
    /// rotated_buf, BlockScratch, KV caches, weights, the f32 cache)
    /// will be sub-ranges of one of the three vram-heap arenas, and
    /// their bind groups go through this helper.
    pub fn make_bind_group_with(
        &self,
        pipeline: &wgpu::ComputePipeline,
        resources: Vec<wgpu::BindingResource<'_>>,
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .into_iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    /// Create a uniform buffer from a bytemuck-able params struct.
    ///
    /// Routes through the pre-allocated `params_pool` (a ring of
    /// PARAMS_POOL_SLOT_COUNT pre-allocated uniform buffers of
    /// PARAMS_POOL_SLOT_BYTES each). This avoids cortex's per-dispatch
    /// `device.create_buffer` churn — which under wgpu-29's stricter
    /// resource accounting was reliably exhausting some internal
    /// allocator limit during multi-shard polar workloads and
    /// triggering `DeviceError::Lost` from the next `queue.submit`.
    ///
    /// Validation diagnosis 2026-06-08: enabling Vulkan validation
    /// layers (CORTEX_WGPU_VALIDATE=1) caught the symptom — a
    /// `Queue::write_buffer` on a `Buffer with '' label is invalid`,
    /// triggered from `create_params_buffer`'s anonymous create_buffer
    /// call eventually returning a `Fallible::Invalid` sentinel after
    /// the device was marked lost by a prior submission. Stack
    /// pointed at dispatch_rope_packed_in_pass /
    /// dispatch_rmsnorm_packed_to_packed_in_pass — the hot per-layer
    /// dispatches in `forward_full_gpu_polar_traced`.
    ///
    /// `wgpu::Buffer` is internally Arc'd so handing out owned clones
    /// of pool slots is cheap. Caller's existing pattern (pass the
    /// returned Buffer into `make_bind_group`'s
    /// `as_entire_binding()`) stays unchanged because each slot IS a
    /// whole standalone wgpu Buffer — we just never let them get
    /// dropped.
    ///
    /// Ring size (2048) is comfortably larger than any single
    /// synchronously-issued forward pass needs (polar retrieve =
    /// ~36 layers × ~10 dispatches per layer = ~360 slots). Caller
    /// returns are single-threaded per-device (cortex-cloud
    /// serializes via the cache-pool mutex), so the ring's natural
    /// wrap-around is safe after each function's terminal poll.
    pub fn create_params_buffer<T: bytemuck::Pod>(&self, params: &T) -> wgpu::Buffer {
        let size = std::mem::size_of::<T>() as u64;
        assert!(
            size <= PARAMS_POOL_SLOT_BYTES,
            "params struct ({} bytes) exceeds PARAMS_POOL_SLOT_BYTES ({}); \
             bump the constant or split the struct",
            size, PARAMS_POOL_SLOT_BYTES,
        );
        let buf = self.params_pool.next_slot();
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(params));
        buf
    }

    /// Create a storage buffer with initial data.
    pub fn create_storage_buffer(&self, data: &[u8], label: &str) -> wgpu::Buffer {
        wgpu::util::DeviceExt::create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        )
    }

    /// Create an empty storage buffer of a given size.
    pub fn create_empty_buffer(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for GPU→CPU readback.
    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Pack f32 values into f16 pairs stored as u32.
    ///
    /// Each u32 holds two f16 values: `(f16[2i] | f16[2i+1] << 16)`.
    /// The input length must be even.
    /// Snapshot of all five vram-heaps as (id, used_bytes) pairs.
    /// Used by the Phase K metrics sampler to update per-heap usage
    /// gauges. Each `stats()` call holds the heap's internal lock for
    /// microseconds; safe to call from a periodic task.
    pub fn vram_heap_usage(&self) -> [(VramHeapId, u64); 5] {
        [
            (VramHeapId::TransientA, self.transient_heap_a.stats().used_payload),
            (VramHeapId::TransientB, self.transient_heap_b.stats().used_payload),
            (VramHeapId::TransientC, self.transient_heap_c.stats().used_payload),
            (VramHeapId::Weights, self.weights_heap.stats().used_payload),
            (VramHeapId::HostReadback, self.host_readback_heap.stats().used_payload),
        ]
    }

    /// Snapshot of the device VRAM budget: `(total, committed)` bytes.
    /// Committed counts every live DeviceLocal heap's reservation — the
    /// four globals plus per-cache gpu_kv/polar_kv heaps. The Phase K
    /// metrics sampler exports these as `cortex_vram_budget_bytes`;
    /// `committed / total` is the "how close is the card to full"
    /// capacity meter for the scaling dashboard.
    pub fn vram_budget_snapshot(&self) -> (u64, u64) {
        (self.vram_budget.total(), self.vram_budget.committed())
    }

    pub fn pack_f16(data: &[f32]) -> Vec<u32> {
        assert!(data.len() % 2 == 0, "f16 packing requires even length");
        data.chunks_exact(2)
            .map(|pair| {
                let lo = half::f16::from_f32(pair[0]).to_bits() as u32;
                let hi = half::f16::from_f32(pair[1]).to_bits() as u32;
                lo | (hi << 16)
            })
            .collect()
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuDevice")
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_device_creation() {
        let Some(_gpu) = GpuDevice::try_new() else { return };
    }

    #[test]
    fn f16_packing() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let packed = GpuDevice::pack_f16(&data);
        assert_eq!(packed.len(), 2);
        let lo0 = half::f16::from_bits((packed[0] & 0xFFFF) as u16).to_f32();
        let hi0 = half::f16::from_bits((packed[0] >> 16) as u16).to_f32();
        assert!((lo0 - 1.0).abs() < 0.01);
        assert!((hi0 - 2.0).abs() < 0.01);
    }

    // --- device-probe consumption contract (no GPU required) ----------------
    //
    // Proves cortex reads the VRAM number device-probe publishes correctly,
    // against the synthetic profiles, with no hardware present — the
    // synthetic-profile payoff (device-probe CLAUDE.md "Testability"). These
    // mirror `from_selection`'s budget seam exactly:
    //   DeviceBudget::explicit(profile.vram_budget_or(FALLBACK))

    const PROBE_FALLBACK: u64 = 8192 * 1024 * 1024;

    #[test]
    fn budget_uses_detected_vram_from_profile() {
        // H100 reports 80 GB → the budget total is the detected number, not
        // the fallback. This is the whole point of the handshake.
        let p = device_probe::DeviceProfile::synthetic_h100();
        let budget = ::vram_heap::DeviceBudget::explicit(p.vram_budget_or(PROBE_FALLBACK));
        assert_eq!(budget.total(), 80 * 1024 * 1024 * 1024);
    }

    #[test]
    fn budget_p40_reports_24gb() {
        let p = device_probe::DeviceProfile::synthetic_p40();
        assert_eq!(p.vram_budget_or(0), 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn budget_falls_back_when_vram_undetectable() {
        // Non-Vulkan backend → vram_total_bytes is None → fallback applies
        // (a 0 here would be the dangerous misread the Option exists to stop).
        let p = device_probe::DeviceProfile {
            vram_total_bytes: None,
            ..Default::default()
        };
        assert_eq!(p.vram_budget_or(PROBE_FALLBACK), PROBE_FALLBACK);
    }
}
