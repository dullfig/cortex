//! WGPU compute backend — GPU inference via compute shaders.
//!
//! Two levels of GPU support:
//!
//! 1. **ComputeBackend impl** — drop-in ternary matvec on GPU (existing API).
//! 2. **GpuEngine** — full transformer forward pass in a single command buffer,
//!    with f16-packed weights, precomputed RoPE, KV caches on GPU, and optional
//!    NeuralKV injection. Only 4 bytes read back per generated token.
//!
//! The shaders in `src/compute/shaders/` handle both single-token (decode) and
//! batch (prefill) paths. Weights are stored as f16 pairs packed into u32.

use crate::tensor::TernaryTensor;
use super::ComputeBackend;

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
    pub kv_compress_polar: wgpu::ComputePipeline,
    pub rotate_q: wgpu::ComputePipeline,
    pub attn_score_polar_batch: wgpu::ComputePipeline,
    pub attn_value_polar_batch: wgpu::ComputePipeline,
    // Batch (prefill)
    pub matmul: wgpu::ComputePipeline,
    pub rmsnorm_batch: wgpu::ComputePipeline,
    pub rope_batch: wgpu::ComputePipeline,
    pub silu_mul_batch: wgpu::ComputePipeline,
    /// Batched ReLU²(gate) * up for BitNet b1.58 SwiGLU activations.
    pub relu2_mul_batch: wgpu::ComputePipeline,
    pub add_inplace_batch: wgpu::ComputePipeline,
    pub add_broadcast_batch: wgpu::ComputePipeline,
    pub kv_write_batch: wgpu::ComputePipeline,
    pub attn_score_batch: wgpu::ComputePipeline,
    pub softmax_batch: wgpu::ComputePipeline,
    pub attn_value_batch: wgpu::ComputePipeline,
    // Resident-weight ternary path (GpuBitLinear)
    pub ternary_matvec: wgpu::ComputePipeline,
    // Batched ternary matmul: f32 hidden → quantize_absmax_batch (i8 + scales)
    // → ternary_matmul_batch (f32 out). Used for prefill on bitnet models.
    pub quantize_absmax_batch: wgpu::ComputePipeline,
    pub ternary_matmul_batch: wgpu::ComputePipeline,
    // Broadcast bias add (Q/K/V biases for Qwen-family)
    pub bias_add_batch: wgpu::ComputePipeline,
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
            kv_compress_polar: make(include_str!("shaders/kv_compress_polar.wgsl"), "kv_compress_polar"),
            rotate_q: make(include_str!("shaders/rotate_q.wgsl"), "rotate_q"),
            attn_score_polar_batch: make(include_str!("shaders/attn_score_polar_batch.wgsl"), "attn_score_polar_batch"),
            attn_value_polar_batch: make(include_str!("shaders/attn_value_polar_batch.wgsl"), "attn_value_polar_batch"),
            // Batch
            matmul: make(include_str!("shaders/matmul.wgsl"), "matmul"),
            rmsnorm_batch: make(include_str!("shaders/rmsnorm_batch.wgsl"), "rmsnorm_batch"),
            rope_batch: make(include_str!("shaders/rope_batch.wgsl"), "rope_batch"),
            silu_mul_batch: make(include_str!("shaders/silu_mul_batch.wgsl"), "silu_mul_batch"),
            relu2_mul_batch: make(include_str!("shaders/relu2_mul_batch.wgsl"), "relu2_mul_batch"),
            add_inplace_batch: make(include_str!("shaders/add_inplace_batch.wgsl"), "add_inplace_batch"),
            add_broadcast_batch: make(include_str!("shaders/add_broadcast_batch.wgsl"), "add_broadcast_batch"),
            kv_write_batch: make(include_str!("shaders/kv_write_batch.wgsl"), "kv_write_batch"),
            attn_score_batch: make(include_str!("shaders/attn_score_batch.wgsl"), "attn_score_batch"),
            softmax_batch: make(include_str!("shaders/softmax_batch.wgsl"), "softmax_batch"),
            attn_value_batch: make(include_str!("shaders/attn_value_batch.wgsl"), "attn_value_batch"),
            // Resident-weight ternary path — entry point differs from "main"
            ternary_matvec: make_with_entry(TERNARY_SHADER, "ternary_matvec_resident", "ternary_matvec"),
            quantize_absmax_batch: make(include_str!("shaders/quantize_absmax_batch.wgsl"), "quantize_absmax_batch"),
            ternary_matmul_batch: make(include_str!("shaders/ternary_matmul_batch.wgsl"), "ternary_matmul_batch"),
            bias_add_batch: make(include_str!("shaders/bias_add_batch.wgsl"), "bias_add_batch"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDevice — shared device + queue + pipelines
// ---------------------------------------------------------------------------

/// Shared GPU context: device, queue, and compiled pipelines.
///
/// Created once at startup, shared across all layers via `Arc<GpuDevice>`.
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Pipelines,
}

impl GpuDevice {
    /// Try to create a GPU device with all pipelines compiled.
    /// Returns `None` if no suitable adapter is found.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

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
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cortex"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let pipelines = Pipelines::compile(&device);
        tracing::info!("compiled 31 GPU compute pipelines");

        Some(Self { device, queue, pipelines })
    }

    /// Create a bind group from a pipeline and a list of buffers.
    ///
    /// Buffers are bound to `@binding(0)`, `@binding(1)`, etc.
    pub fn make_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    /// Create a uniform buffer from a bytemuck-able params struct.
    ///
    /// Uses `queue.write_buffer` to populate the data rather than
    /// `create_buffer_init`. The latter uses an internal staging belt that
    /// could not recycle reliably across hundreds of per-dispatch params
    /// buffers — we hit a "staging buffer in bind group" validation error
    /// around the 200th call. `queue.write_buffer` manages its own staging
    /// at the queue level and is the wgpu-recommended pattern for frequent
    /// small writes.
    pub fn create_params_buffer<T: bytemuck::Pod>(&self, params: &T) -> wgpu::Buffer {
        let size = std::mem::size_of::<T>() as u64;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
// WgpuBackend — ComputeBackend impl for ternary matvec (legacy API)
// ---------------------------------------------------------------------------

/// GPU compute backend for the ternary matvec hot path.
///
/// This implements the `ComputeBackend` trait for drop-in use with
/// `BitLinear` layers. For full GPU inference, use `GpuDevice` directly.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend")
    }
}

/// Ternary matvec shader — separate from the f16 shaders above.
/// Unpacks 2-bit packed weights and i8 activations on GPU.
///
/// Inner-loop layout: each thread tiles 16 columns per outer iteration,
/// since one u32 of weights packs exactly 16 ternary values (2 bits each)
/// and four u32s of activations cover 16 i8 entries. The unrolled
/// 16-step decode amortizes the bit-shift overhead vs the previous
/// per-element loop (~6x fewer global memory ops). A scalar tail handles
/// the residual columns when `cols % 16 != 0` (toy test fixtures hit
/// this; production models all have cols divisible by 16+).
///
/// Decode is branchless: `sign = i32(w_bits == 2) - i32(w_bits == 0)`,
/// then `acc += act * sign`. Encoding: 0 → -1, 1 → 0, 2 → +1, 3 → 0
/// (same as the previous shader and matches CPU `ternary_matvec`).
const TERNARY_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> shared_acc: array<i32, 256>;

// Decode one i8 from a u32 holding 4 packed acts at byte `k mod 4`.
fn unpack_i8(packed: u32, k: u32) -> i32 {
    let byte = (packed >> ((k & 3u) * 8u)) & 0xFFu;
    var v: i32 = i32(byte);
    if (v > 127) { v = v - 256; }
    return v;
}

@compute @workgroup_size(256)
fn ternary_matvec(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) { return; }

    let cols = params.cols;
    var acc: i32 = 0;

    // Bulk path: process 16 columns per thread per outer iteration.
    // Each iter reads one weight u32 (= 16 packed ternary weights for
    // flat positions [row*cols + col_base, row*cols + col_base + 15])
    // and four activation u32s (= 16 i8 values at cols [col_base, +15]).
    let aligned_cols: u32 = (cols / 16u) * 16u;
    var col_base: u32 = lid * 16u;
    let stride: u32 = WG_SIZE * 16u;

    while (col_base < aligned_cols) {
        let w_flat: u32 = row * cols + col_base;
        let w_u32: u32 = weights[w_flat / 16u];

        let a_base: u32 = col_base / 4u;
        let a0: u32 = activations[a_base];
        let a1: u32 = activations[a_base + 1u];
        let a2: u32 = activations[a_base + 2u];
        let a3: u32 = activations[a_base + 3u];

        // Unrolled 16-step decode. The inner branch chooses which act u32
        // to pull from based on the (compile-time-constant) tile index.
        // Branchless decode: contribution = act * sign.
        // k = 0..3   → a0   (act bytes 0..3 of a0)
        // k = 4..7   → a1
        // k = 8..11  → a2
        // k = 12..15 → a3
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let w_bits = (w_u32 >> (k * 2u)) & 3u;
            let sign = i32(w_bits == 2u) - i32(w_bits == 0u);
            acc += unpack_i8(a0, k) * sign;
        }
        for (var k: u32 = 4u; k < 8u; k = k + 1u) {
            let w_bits = (w_u32 >> (k * 2u)) & 3u;
            let sign = i32(w_bits == 2u) - i32(w_bits == 0u);
            acc += unpack_i8(a1, k) * sign;
        }
        for (var k: u32 = 8u; k < 12u; k = k + 1u) {
            let w_bits = (w_u32 >> (k * 2u)) & 3u;
            let sign = i32(w_bits == 2u) - i32(w_bits == 0u);
            acc += unpack_i8(a2, k) * sign;
        }
        for (var k: u32 = 12u; k < 16u; k = k + 1u) {
            let w_bits = (w_u32 >> (k * 2u)) & 3u;
            let sign = i32(w_bits == 2u) - i32(w_bits == 0u);
            acc += unpack_i8(a3, k) * sign;
        }

        col_base += stride;
    }

    // Tail path: any cols in [aligned_cols, cols). Per-element with
    // stride-WG_SIZE. Mirrors the original shader's semantics so the
    // small-cols test fixtures (cols=2, cols=4) still pass.
    var col: u32 = aligned_cols + lid;
    while (col < cols) {
        let flat = row * cols + col;
        let w_byte_idx = flat / 4u;
        let w_bit_shift = (flat % 4u) * 2u;
        let w_u32 = weights[w_byte_idx / 4u];
        let w_byte = (w_u32 >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
        let w_bits = (w_byte >> w_bit_shift) & 3u;

        let act_u32 = activations[col / 4u];
        let act_val = unpack_i8(act_u32, col);
        let sign = i32(w_bits == 2u) - i32(w_bits == 0u);
        acc += act_val * sign;

        col += WG_SIZE;
    }

    shared_acc[lid] = acc;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            shared_acc[lid] += shared_acc[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        output[row] = shared_acc[0];
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct TernaryParams {
    rows: u32,
    cols: u32,
}

impl TernaryParams {
    fn as_bytes(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<Self>()) }
    }
}

impl WgpuBackend {
    /// Try to create a wgpu backend. Returns `None` if no suitable GPU is found.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cortex-ternary"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ternary_matvec"),
            source: wgpu::ShaderSource::Wgsl(TERNARY_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ternary_matvec_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ternary_matvec_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ternary_matvec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ternary_matvec"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self { device, queue, pipeline, bind_group_layout })
    }

    fn pad_to_u32(data: &[u8]) -> Vec<u8> {
        let remainder = data.len() % 4;
        if remainder == 0 {
            data.to_vec()
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len() + (4 - remainder), 0);
            padded
        }
    }

    fn pack_activations(input: &[i8]) -> Vec<u8> {
        let mut bytes: Vec<u8> = input.iter().map(|&v| v as u8).collect();
        let remainder = bytes.len() % 4;
        if remainder != 0 {
            bytes.resize(bytes.len() + (4 - remainder), 0);
        }
        bytes
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &str { "wgpu" }

    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
        assert_eq!(weights.cols(), input.len(), "dimension mismatch");

        let rows = weights.rows();
        let cols = weights.cols();

        if rows == 0 || cols == 0 {
            return vec![0i32; rows];
        }

        let weight_bytes = Self::pad_to_u32(weights.packed_data());
        let act_bytes = Self::pack_activations(input);
        let output_size = (rows * std::mem::size_of::<i32>()) as u64;

        let weight_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights"),
            size: weight_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&weight_buf, 0, &weight_bytes);

        let act_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activations"),
            size: act_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&act_buf, 0, &act_bytes);

        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = TernaryParams { rows: rows as u32, cols: cols as u32 };
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<TernaryParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, params.as_bytes());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ternary_matvec_bind"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: weight_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: act_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ternary_matvec_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ternary_matvec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("GPU readback failed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let result: Vec<i32> = data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        drop(data);
        staging_buf.unmap();

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

    fn weights_from_i8(values: &[i8], rows: usize, cols: usize) -> TernaryTensor {
        let ternary: Vec<Ternary> = values.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        TernaryTensor::pack(&ternary, rows, cols)
    }

    fn get_backend() -> Option<WgpuBackend> {
        WgpuBackend::try_new()
    }

    #[test]
    fn identity_matvec() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[1, 0, 0, 1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![42, -17]);
    }

    #[test]
    fn negation_matvec() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[-1, 0, 0, -1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![-42, 17]);
    }

    #[test]
    fn mixed_weights() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[1, -1, 0, 1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![30]);
    }

    #[test]
    fn matches_scalar_random() {
        let Some(backend) = get_backend() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        let rows = 64;
        let cols = 128;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xDEAD_BEEF;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 3) as i8 - 1;
            weights_i8.push(v);
        }
        let w = weights_from_i8(&weights_i8, rows, cols);

        let mut activations = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 255) as i8;
            activations.push(v);
        }

        let gpu_result = backend.ternary_matvec(&w, &activations);
        let cpu_result = scalar.ternary_matvec(&w, &activations);

        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(g, c, "mismatch at row {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn gpu_device_creation() {
        // Test that GpuDevice compiles all 24 pipelines
        let Some(_gpu) = GpuDevice::try_new() else { return };
        // If we got here, all 24 shaders compiled successfully
    }

    #[test]
    fn f16_packing() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let packed = GpuDevice::pack_f16(&data);
        assert_eq!(packed.len(), 2);

        // Unpack and verify
        let lo0 = half::f16::from_bits((packed[0] & 0xFFFF) as u16).to_f32();
        let hi0 = half::f16::from_bits((packed[0] >> 16) as u16).to_f32();
        assert!((lo0 - 1.0).abs() < 0.01);
        assert!((hi0 - 2.0).abs() < 0.01);
    }

    // ===== Batched ternary matmul (#bn-7) =====

    /// Read N u32s back from a GPU buffer via a staging copy.
    fn readback_u32(gpu: &GpuDevice, staging: &wgpu::Buffer, n: usize) -> Vec<u32> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<u32> = data[..n * 4].chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging.unmap();
        out
    }

    fn readback_f32(gpu: &GpuDevice, staging: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let raw = readback_u32(gpu, staging, n);
        raw.iter().map(|&u| f32::from_bits(u)).collect()
    }

    fn readback_i32(gpu: &GpuDevice, staging: &wgpu::Buffer, n: usize) -> Vec<i32> {
        let raw = readback_u32(gpu, staging, n);
        raw.iter().map(|&u| u as i32).collect()
    }
    // Suppress an unused-function warning when readback_i32 isn't exercised
    // in every test build.
    #[allow(dead_code)]
    fn _force_use(_: fn(&GpuDevice, &wgpu::Buffer, usize) -> Vec<i32>) {}

    /// Pack i8 → u32 (4 per u32) using the same layout the ternary_matmul_batch
    /// shader reads. Caller validates round-trip when comparing shader vs CPU.
    fn pack_i8_to_u32(values: &[i8]) -> Vec<u32> {
        let n_u32 = (values.len() + 3) / 4;
        let mut out = vec![0u32; n_u32];
        for (i, &v) in values.iter().enumerate() {
            let byte = (v as i32 & 0xFF) as u32;
            out[i / 4] |= byte << ((i % 4) * 8);
        }
        out
    }

    #[test]
    fn quantize_absmax_batch_parity() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        use crate::ops::quantize::quantize_per_token;

        let n_tokens = 4usize;
        let cols = 64usize;
        let mut rng: u64 = 0xCAFE_F00D;
        let mut input: Vec<f32> = Vec::with_capacity(n_tokens * cols);
        for _ in 0..(n_tokens * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((rng >> 33) as f32) / (i32::MAX as f32);
            input.push(frac * 5.0 - 2.5);
        }
        let (cpu_q, cpu_scales) = quantize_per_token(&input, cols);

        // Build GPU buffers.
        let in_bytes = bytemuck::cast_slice(&input);
        let input_buf = gpu.create_storage_buffer(in_bytes, "test.input");
        let n_u32_per_token = (cols + 3) / 4;
        let act_q_buf = gpu.create_empty_buffer(
            (n_tokens * n_u32_per_token * 4) as u64, "test.act_q",
        );
        let scales_buf = gpu.create_empty_buffer((n_tokens * 4) as u64, "test.scales");

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { cols: u32, n_tokens: u32 }
        let params_buf = gpu.create_params_buffer(&P {
            cols: cols as u32, n_tokens: n_tokens as u32,
        });

        let pipeline = &gpu.pipelines.quantize_absmax_batch;
        let bind = gpu.make_bind_group(pipeline, &[&input_buf, &act_q_buf, &scales_buf, &params_buf]);

        let act_q_staging = gpu.create_staging_buffer((n_tokens * n_u32_per_token * 4) as u64);
        let scales_staging = gpu.create_staging_buffer((n_tokens * 4) as u64);

        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n_tokens as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&act_q_buf, 0, &act_q_staging, 0, (n_tokens * n_u32_per_token * 4) as u64);
        enc.copy_buffer_to_buffer(&scales_buf, 0, &scales_staging, 0, (n_tokens * 4) as u64);
        gpu.queue.submit(Some(enc.finish()));

        let gpu_q_packed = readback_u32(&gpu, &act_q_staging, n_tokens * n_u32_per_token);
        let gpu_scales = readback_f32(&gpu, &scales_staging, n_tokens);

        // Compare scales.
        for (i, (g, c)) in gpu_scales.iter().zip(cpu_scales.iter()).enumerate() {
            assert!((g - c).abs() < 1e-6, "scale mismatch at token {i}: gpu={g}, cpu={c}");
        }

        // Compare quantized values element-by-element (unpack the u32 packing).
        let mut gpu_q_i8: Vec<i8> = Vec::with_capacity(n_tokens * cols);
        for tok in 0..n_tokens {
            for col in 0..cols {
                let u = gpu_q_packed[tok * n_u32_per_token + col / 4];
                let byte = (u >> ((col % 4) * 8)) & 0xFF;
                let signed = if byte > 127 { byte as i32 - 256 } else { byte as i32 };
                gpu_q_i8.push(signed as i8);
            }
        }
        for (i, (g, c)) in gpu_q_i8.iter().zip(cpu_q.iter()).enumerate() {
            assert_eq!(g, c, "quantized mismatch at {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn ternary_matmul_batch_parity() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        let rows = 32usize;
        let cols = 64usize;
        let n_tokens = 4usize;
        let weight_scale = 0.0123f32;

        // Random ternary weights + i8 activations + per-token scales.
        let mut rng: u64 = 0xBEEF_CAFE;
        let mut w_i8: Vec<i8> = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            w_i8.push(((rng >> 33) % 3) as i8 - 1);
        }
        let w = weights_from_i8(&w_i8, rows, cols);

        let mut acts: Vec<i8> = Vec::with_capacity(n_tokens * cols);
        for _ in 0..(n_tokens * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 255) as i32 - 127;
            acts.push(v as i8);
        }
        let scales: Vec<f32> = (0..n_tokens)
            .map(|i| 0.001 * (i as f32 + 1.0))
            .collect();

        // Build a GpuBitLinear so we can reuse the resident-weight buffer
        // path (matches the production dispatch site).
        let gpu_arc = std::sync::Arc::new(gpu);
        let bit_layer = crate::layers::gpu_bitlinear::GpuBitLinear::from_weights(
            gpu_arc.clone(), w.clone(), weight_scale,
        );

        // Pack activations + upload scales as separate buffers.
        let act_u32 = pack_i8_to_u32(&acts);
        let act_bytes: Vec<u8> = act_u32.iter().flat_map(|u| u.to_le_bytes()).collect();
        let act_buf = gpu_arc.create_storage_buffer(&act_bytes, "test.acts");
        let scales_bytes: Vec<u8> = scales.iter().flat_map(|f| f.to_le_bytes()).collect();
        let scales_buf = gpu_arc.create_storage_buffer(&scales_bytes, "test.scales");

        let out_bytes = (n_tokens * rows * 4) as u64;
        let out_buf = gpu_arc.create_empty_buffer(out_bytes, "test.out");
        let out_staging = gpu_arc.create_staging_buffer(out_bytes);

        // Dispatch via raw pipeline (mirrors what dispatch_ternary_matmul_batch_into does).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { rows: u32, cols: u32, n_tokens: u32, weight_scale_bits: u32 }
        let params_buf = gpu_arc.create_params_buffer(&P {
            rows: rows as u32, cols: cols as u32, n_tokens: n_tokens as u32,
            weight_scale_bits: weight_scale.to_bits(),
        });
        let pipeline = &gpu_arc.pipelines.ternary_matmul_batch;
        let bind = gpu_arc.make_bind_group(
            pipeline,
            &[bit_layer.weight_buffer(), &act_buf, &scales_buf, &out_buf, &params_buf],
        );

        let mut enc = gpu_arc.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let dx = rows.min(65535) as u32;
            let dy = ((rows + 65534) / 65535) as u32;
            pass.dispatch_workgroups(dx, dy, n_tokens as u32);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &out_staging, 0, out_bytes);
        gpu_arc.queue.submit(Some(enc.finish()));

        let gpu_out = readback_f32(&gpu_arc, &out_staging, n_tokens * rows);

        // CPU reference: for each token, ternary_matvec(weights, acts[tok]) * scale[tok] * weight_scale.
        for tok in 0..n_tokens {
            let acts_t = &acts[tok * cols..(tok + 1) * cols];
            let cpu_i32 = scalar.ternary_matvec(&w, acts_t);
            for row in 0..rows {
                let expected = cpu_i32[row] as f32 * scales[tok] * weight_scale;
                let actual = gpu_out[tok * rows + row];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "row {row} tok {tok}: gpu={actual}, cpu={expected}",
                );
            }
        }
    }

    #[test]
    fn ternary_pipeline_end_to_end() {
        // f32 hidden → quantize_absmax_batch → ternary_matmul_batch
        // Compared against BitLinear::forward looped per-token.
        let Some(gpu) = GpuDevice::try_new() else { return };
        use crate::layers::bitlinear::BitLinear;
        use crate::layers::linear::LinearLayer;

        let rows = 16usize;
        let cols = 32usize;
        let n_tokens = 3usize;
        let weight_scale = 0.05f32;

        let mut rng: u64 = 0x1234_5678_DEAD_BEEF;
        let mut w_i8: Vec<i8> = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            w_i8.push(((rng >> 33) % 3) as i8 - 1);
        }
        let w_tensor = weights_from_i8(&w_i8, rows, cols);

        let mut hidden_f32: Vec<f32> = Vec::with_capacity(n_tokens * cols);
        for _ in 0..(n_tokens * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = ((rng >> 33) as f32) / (i32::MAX as f32);
            hidden_f32.push(frac * 2.0 - 1.0);
        }

        // CPU reference: BitLinear forward looped per token (this is the
        // path the prefill currently degenerates to before #bn lands).
        let cpu_layer = BitLinear::new(w_tensor.clone(), weight_scale);
        let mut cpu_out: Vec<f32> = Vec::with_capacity(n_tokens * rows);
        for tok in 0..n_tokens {
            let row_in = &hidden_f32[tok * cols..(tok + 1) * cols];
            cpu_out.extend(cpu_layer.forward(row_in));
        }

        // GPU pipeline: f32 → quantize → ternary matmul → f32.
        let gpu_arc = std::sync::Arc::new(gpu);
        let bit_layer = crate::layers::gpu_bitlinear::GpuBitLinear::from_weights(
            gpu_arc.clone(), w_tensor, weight_scale,
        );

        let in_bytes: Vec<u8> = hidden_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let in_buf = gpu_arc.create_storage_buffer(&in_bytes, "e2e.in");

        let n_u32_per_token = (cols + 3) / 4;
        let act_q_buf = gpu_arc.create_empty_buffer((n_tokens * n_u32_per_token * 4) as u64, "e2e.act_q");
        let scales_buf = gpu_arc.create_empty_buffer((n_tokens * 4) as u64, "e2e.scales");
        let out_bytes = (n_tokens * rows * 4) as u64;
        let out_buf = gpu_arc.create_empty_buffer(out_bytes, "e2e.out");
        let out_staging = gpu_arc.create_staging_buffer(out_bytes);

        // Stage 1: quantize.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct QP { cols: u32, n_tokens: u32 }
        let qp_buf = gpu_arc.create_params_buffer(&QP {
            cols: cols as u32, n_tokens: n_tokens as u32,
        });
        let q_pipe = &gpu_arc.pipelines.quantize_absmax_batch;
        let q_bind = gpu_arc.make_bind_group(q_pipe, &[&in_buf, &act_q_buf, &scales_buf, &qp_buf]);

        // Stage 2: ternary matmul.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MP { rows: u32, cols: u32, n_tokens: u32, ws_bits: u32 }
        let mp_buf = gpu_arc.create_params_buffer(&MP {
            rows: rows as u32, cols: cols as u32, n_tokens: n_tokens as u32,
            ws_bits: weight_scale.to_bits(),
        });
        let m_pipe = &gpu_arc.pipelines.ternary_matmul_batch;
        let m_bind = gpu_arc.make_bind_group(
            m_pipe,
            &[bit_layer.weight_buffer(), &act_q_buf, &scales_buf, &out_buf, &mp_buf],
        );

        let mut enc = gpu_arc.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(q_pipe);
            pass.set_bind_group(0, &q_bind, &[]);
            pass.dispatch_workgroups(n_tokens as u32, 1, 1);

            pass.set_pipeline(m_pipe);
            pass.set_bind_group(0, &m_bind, &[]);
            let dx = rows.min(65535) as u32;
            let dy = ((rows + 65534) / 65535) as u32;
            pass.dispatch_workgroups(dx, dy, n_tokens as u32);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &out_staging, 0, out_bytes);
        gpu_arc.queue.submit(Some(enc.finish()));

        let gpu_out = readback_f32(&gpu_arc, &out_staging, n_tokens * rows);

        // Tolerance: per-token quantization rounding noise. With cols=32 and
        // small weight magnitudes this is well under 1e-3 in practice.
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (g - c).abs() < 5e-4,
                "mismatch at {i} (tok={}, row={}): gpu={g}, cpu={c}, diff={}",
                i / rows, i % rows, (g - c).abs(),
            );
        }
    }
}
