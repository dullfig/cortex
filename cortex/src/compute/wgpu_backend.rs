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
    // Single-token (decode)
    pub matvec: wgpu::ComputePipeline,
    pub matvec_bias: wgpu::ComputePipeline,
    pub matvec_q4k: wgpu::ComputePipeline,
    pub rmsnorm: wgpu::ComputePipeline,
    pub rope: wgpu::ComputePipeline,
    pub silu_mul: wgpu::ComputePipeline,
    pub add_inplace: wgpu::ComputePipeline,
    pub kv_write: wgpu::ComputePipeline,
    pub attn_score: wgpu::ComputePipeline,
    pub softmax: wgpu::ComputePipeline,
    pub attn_value: wgpu::ComputePipeline,
    pub argmax: wgpu::ComputePipeline,
    // Batch (prefill)
    pub matmul: wgpu::ComputePipeline,
    pub matmul_bias: wgpu::ComputePipeline,
    pub rmsnorm_batch: wgpu::ComputePipeline,
    pub rope_batch: wgpu::ComputePipeline,
    pub silu_mul_batch: wgpu::ComputePipeline,
    pub add_inplace_batch: wgpu::ComputePipeline,
    pub kv_write_batch: wgpu::ComputePipeline,
    pub attn_score_batch: wgpu::ComputePipeline,
    pub softmax_batch: wgpu::ComputePipeline,
    pub attn_value_batch: wgpu::ComputePipeline,
    // Resident-weight ternary path (GpuBitLinear)
    pub ternary_matvec: wgpu::ComputePipeline,
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
            matvec_bias: make(include_str!("shaders/matvec_bias.wgsl"), "matvec_bias"),
            matvec_q4k: make(include_str!("shaders/matvec_q4k.wgsl"), "matvec_q4k"),
            rmsnorm: make(include_str!("shaders/rmsnorm.wgsl"), "rmsnorm"),
            rope: make(include_str!("shaders/rope.wgsl"), "rope"),
            silu_mul: make(include_str!("shaders/silu_mul.wgsl"), "silu_mul"),
            add_inplace: make(include_str!("shaders/add_inplace.wgsl"), "add_inplace"),
            kv_write: make(include_str!("shaders/kv_write.wgsl"), "kv_write"),
            attn_score: make(include_str!("shaders/attn_score.wgsl"), "attn_score"),
            softmax: make(include_str!("shaders/softmax.wgsl"), "softmax"),
            attn_value: make(include_str!("shaders/attn_value.wgsl"), "attn_value"),
            argmax: make(include_str!("shaders/argmax.wgsl"), "argmax"),
            // Batch
            matmul: make(include_str!("shaders/matmul.wgsl"), "matmul"),
            matmul_bias: make(include_str!("shaders/matmul_bias.wgsl"), "matmul_bias"),
            rmsnorm_batch: make(include_str!("shaders/rmsnorm_batch.wgsl"), "rmsnorm_batch"),
            rope_batch: make(include_str!("shaders/rope_batch.wgsl"), "rope_batch"),
            silu_mul_batch: make(include_str!("shaders/silu_mul_batch.wgsl"), "silu_mul_batch"),
            add_inplace_batch: make(include_str!("shaders/add_inplace_batch.wgsl"), "add_inplace_batch"),
            kv_write_batch: make(include_str!("shaders/kv_write_batch.wgsl"), "kv_write_batch"),
            attn_score_batch: make(include_str!("shaders/attn_score_batch.wgsl"), "attn_score_batch"),
            softmax_batch: make(include_str!("shaders/softmax_batch.wgsl"), "softmax_batch"),
            attn_value_batch: make(include_str!("shaders/attn_value_batch.wgsl"), "attn_value_batch"),
            // Resident-weight ternary path — entry point differs from "main"
            ternary_matvec: make_with_entry(TERNARY_SHADER, "ternary_matvec_resident", "ternary_matvec"),
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
        tracing::info!("compiled 24 GPU compute pipelines");

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

@compute @workgroup_size(256)
fn ternary_matvec(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) { return; }

    let cols = params.cols;
    var acc: i32 = 0;

    var col = lid;
    while (col < cols) {
        let flat = row * cols + col;
        let w_byte_idx = flat / 4u;
        let w_bit_shift = (flat % 4u) * 2u;
        let w_u32 = weights[w_byte_idx / 4u];
        let w_byte = (w_u32 >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
        let w_bits = (w_byte >> w_bit_shift) & 3u;

        let act_u32 = activations[col / 4u];
        let act_byte = (act_u32 >> ((col % 4u) * 8u)) & 0xFFu;
        var act_val: i32 = i32(act_byte);
        if (act_val > 127) { act_val = act_val - 256; }

        if (w_bits == 0u) {
            acc -= act_val;
        } else if (w_bits == 2u) {
            acc += act_val;
        }

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
}
