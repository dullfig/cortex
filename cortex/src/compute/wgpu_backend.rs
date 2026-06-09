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
/// exceed the largest single synchronous forward pass's per-dispatch
/// params allocation count, so the ring's natural wrap-around can't
/// reuse a slot while the previous occupant is still bound in flight.
/// Polar retrieve at Qwen-3B (36 layers × ~10 dispatches per layer)
/// uses ~360 slots; 2048 leaves 5x headroom for future expansion.
/// Memory cost: 2048 × 256 B = 512 KB. Trivial vs the ~6 GB cortex
/// uses for weights+caches.
pub const PARAMS_POOL_SLOT_COUNT: usize = 2048;

/// Ring of pre-allocated uniform buffers used by `create_params_buffer`.
/// See that method's doc comment for why this exists.
///
/// # Why this is NOT a `vram_heap::VramPool`
///
/// vram-heap's `VramPool` has an *explicit* slot lifecycle: `acquire()`
/// pops an index from a free-list, `PoolSlot::drop` pushes it back.
/// `ParamsBufferPool` has an *implicit* slot lifecycle: `next_slot()`
/// is an atomic round-robin over 2048 slots and the slot is "available
/// again" the instant it returns.
///
/// cortex's dispatch helpers do `acquire → write → bind → record →
/// return`. The returned `wgpu::Buffer` (an Arc handle to the slot)
/// drops at function return, but the GPU's actual `queue.submit` +
/// dispatch read happens later — often hundreds of microseconds later,
/// batched at the end of the forward function. The current ring
/// tolerates this because nothing overwrites a slot until 2048 more
/// allocations later, by which point the GPU has long since finished
/// reading the in-flight slots.
///
/// Migrating to `VramPool` would force every dispatch helper to grow a
/// `&mut Vec<PoolSlot>` keepalive parameter threaded from the forward
/// function down through every dispatch — ~44 call sites, all-or-
/// nothing plumbing, no substrate benefit. The ring is already
/// solving the wgpu-29 allocator-pressure ceiling that motivated the
/// vram-heap work in the first place. Leave it.
///
/// `VramPool` is the right answer for callers that DO want explicit
/// per-slot lifetime tracking — e.g. memex's KV-cache slot reuse or
/// ternary-rs's quantized-tensor allocation patterns where slots
/// outlive a single function-scope.
pub struct ParamsBufferPool {
    slots: Vec<wgpu::Buffer>,
    next: std::sync::atomic::AtomicUsize,
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
}

/// Shared GPU context: device, queue, compiled pipelines, and the
/// vram-heap arenas for per-call transient allocations.
///
/// Created once at startup, shared across all layers via `Arc<GpuDevice>`.
///
/// # vram-heap usage
///
/// `transient_heap_a` and `transient_heap_b` are TWO device-local
/// heaps that callers use as "input-lane" and "output-lane" for
/// dispatches that need both R and RW bindings: wgpu/Vulkan rejects
/// binding `STORAGE_READ_ONLY` and `STORAGE_READ_WRITE` to the same
/// backing buffer within a single dispatch (even at disjoint
/// sub-ranges). By allocating reads from one heap and writes from
/// another, both bindings are sub-ranges of *different* backings, so
/// the buffer-level tracker is happy.
///
/// Both heaps reset their transient regions at the end of each
/// forward pass (see `forward_full_gpu_polar_traced` and siblings).
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Pipelines,
    pub params_pool: ParamsBufferPool,
    /// Input-lane device-local heap. Use for buffers that may be bound
    /// as `STORAGE_READ_ONLY` in a dispatch (rmsnorm inputs, matmul
    /// LHS, etc.). Also fine for write-only allocations that won't
    /// share a dispatch with the output lane.
    pub transient_heap_a: Arc<::vram_heap::VramHeap>,
    /// Output-lane device-local heap. Use for buffers that will be
    /// bound as `STORAGE_READ_WRITE` (matmul outputs, residual-add
    /// targets, etc.).
    pub transient_heap_b: Arc<::vram_heap::VramHeap>,
    /// Host-visible readback heap. Use for staging buffers that
    /// receive `copy_buffer_to_buffer` destinations and then get
    /// host-mapped for read. Capture-staging buffers in retrieval
    /// trace forwards land here.
    pub host_readback_heap: Arc<::vram_heap::VramHeap>,
}

impl GpuDevice {
    /// Try to create a GPU device with all pipelines compiled.
    /// Returns `None` if no suitable adapter is found.
    pub fn try_new() -> Option<Self> {
        // wgpu 29: InstanceDescriptor no Default; use new_without_display_handle.
        // Instance::new takes desc by value.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        // DIAG: enable Vulkan validation + debug labels so the driver
        // prints warnings/errors before the device-lost panic. Gated on
        // CORTEX_WGPU_VALIDATE=1 so it's opt-in (validation has nonzero
        // overhead and shouldn't ship). Reveal whatever the Vulkan
        // loader sees before our device-lost mystery: buffer overruns,
        // memory budget warnings, fence misuse, etc.
        if std::env::var("CORTEX_WGPU_VALIDATE").as_deref() == Ok("1") {
            desc.flags = wgpu::InstanceFlags::debugging();
            eprintln!("[cortex] wgpu validation layers ENABLED via CORTEX_WGPU_VALIDATE=1");
        }
        let instance = wgpu::Instance::new(desc);

        // wgpu 29: request_adapter returns Future<Result>, not Future<Option>.
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok()?;

        let info = adapter.get_info();
        tracing::info!(
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            "GPU adapter selected"
        );

        // Use the adapter's actual limits instead of wgpu's conservative
        // defaults. wgpu::Limits::default() caps max_buffer_size at 256 MB,
        // which is smaller than a 7B model's vocab projection (~600 MB f16).
        // The 4080 reports 4 GB+; requesting adapter limits keeps resident
        // weights practical for vocab-sized tensors.
        let adapter_limits = adapter.limits();
        // TIMESTAMP_QUERY + TIMESTAMP_QUERY_INSIDE_ENCODERS let us
        // place encoder-level write_timestamp markers anywhere in the
        // command stream. Both are widely supported on Vulkan; if the
        // adapter doesn't expose them, we drop down to no timestamp
        // tracing (the timestamp_us telemetry just stays zero).
        let adapter_features = adapter.features();
        let timestamp_supported = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let mut required_features = wgpu::Features::empty();
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        // wgpu 29: request_device takes 1 arg (desc only — trace param moved
        // INTO DeviceDescriptor as the `trace` field). DeviceDescriptor also
        // gained `experimental_features`.
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cortex"),
                required_features,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .ok()?;
        if timestamp_supported {
            tracing::info!(
                period_ns = queue.get_timestamp_period(),
                "GPU timestamp queries enabled",
            );
        } else {
            tracing::info!("GPU timestamp queries NOT supported by adapter");
        }

        let pipelines = Pipelines::compile(&device);
        tracing::info!("compiled 34 GPU compute pipelines");

        let params_pool = ParamsBufferPool::new(&device);
        tracing::info!(
            slot_count = PARAMS_POOL_SLOT_COUNT,
            slot_bytes = PARAMS_POOL_SLOT_BYTES,
            total_kb = (PARAMS_POOL_SLOT_COUNT as u64 * PARAMS_POOL_SLOT_BYTES) / 1024,
            "params buffer pool allocated",
        );

        // vram-heap arenas. Two device-local heaps so callers can
        // place R inputs and RW outputs on different backings within
        // one dispatch (wgpu/Vulkan rejects same-buffer R+RW even at
        // disjoint sub-ranges). Sizes default to 128 MB each — enough
        // for hidden_buf + rotated_buf + future per-forward transients
        // on Qwen 3B at typical query sizes; tunable via env vars.
        let heap_a_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_A_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let heap_b_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_B_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let heap_readback_mb: u64 = std::env::var("CORTEX_VRAM_HEAP_READBACK_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let transient_heap_a = ::vram_heap::VramHeap::new(
            &device,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_a_mb * 1024 * 1024,
            "cortex.transient.a",
        ).expect("vram-heap A construction failed");
        let transient_heap_b = ::vram_heap::VramHeap::new(
            &device,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_b_mb * 1024 * 1024,
            "cortex.transient.b",
        ).expect("vram-heap B construction failed");
        let host_readback_heap = ::vram_heap::VramHeap::new(
            &device,
            ::vram_heap::MemoryTier::HostReadback,
            heap_readback_mb * 1024 * 1024,
            "cortex.host_readback",
        ).expect("vram-heap readback construction failed");

        Some(Self {
            device, queue, pipelines, params_pool,
            transient_heap_a, transient_heap_b, host_readback_heap,
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
}
