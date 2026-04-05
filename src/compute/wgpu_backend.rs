//! wgpu compute backend — GPU ternary matvec via WGSL compute shaders.
//!
//! Runs the ternary matmul hot path on Vulkan/DX12/Metal via wgpu.
//! The kernel unpacks 2-bit packed weights and performs conditional
//! add/sub/skip on 8-bit activations entirely on the GPU.
//!
//! ## Strategy
//!
//! One workgroup per output row (256 threads). Each thread strides
//! across the column dimension accumulating partial sums, then a
//! tree reduction in shared memory produces the final i32 result.
//!
//! ## Limitations
//!
//! - Buffers are created fresh per call (no weight caching yet).
//!   This is correct but leaves bandwidth on the table for repeated
//!   calls with the same weight matrix.
//! - Softmax, rmsnorm, elementwise_mul use the default (CPU) impls.
//!   Only ternary_matvec runs on GPU — it's 90%+ of inference time.

use crate::tensor::TernaryTensor;
use super::ComputeBackend;

/// WGSL compute shader source for ternary matvec.
///
/// Encoding: 0b00 = -1 (Neg), 0b01 = 0 (Zero), 0b10 = +1 (Pos).
/// Activations are i8 packed 4-per-u32 (little-endian byte order).
const SHADER_SOURCE: &str = r#"
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

    // Each thread strides across columns by WG_SIZE
    var col = lid;
    while (col < cols) {
        // --- Unpack 2-bit weight ---
        // Global flat index in the ternary tensor
        let flat = row * cols + col;
        // 4 ternary values per byte, so byte index = flat / 4
        let w_byte_idx = flat / 4u;
        let w_bit_shift = (flat % 4u) * 2u;
        // Weights buffer is array<u32>, extract the byte
        let w_u32 = weights[w_byte_idx / 4u];
        let w_byte = (w_u32 >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
        let w_bits = (w_byte >> w_bit_shift) & 3u;

        // --- Unpack i8 activation ---
        let act_u32 = activations[col / 4u];
        let act_byte = (act_u32 >> ((col % 4u) * 8u)) & 0xFFu;
        // Sign-extend i8 → i32
        var act_val: i32 = i32(act_byte);
        if (act_val > 127) { act_val = act_val - 256; }

        // --- Conditional add/sub/skip ---
        if (w_bits == 0u) {        // Neg (-1)
            acc -= act_val;
        } else if (w_bits == 2u) { // Pos (+1)
            acc += act_val;
        }
        // 1u (Zero) and 3u (unused) → skip

        col += WG_SIZE;
    }

    // --- Workgroup tree reduction ---
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

/// GPU compute backend using wgpu.
///
/// Holds the device, queue, and precompiled pipeline. Created once at
/// startup via `WgpuBackend::try_new()` and shared across all layers.
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

/// Params uniform: rows (u32) + cols (u32) = 8 bytes.
#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    rows: u32,
    cols: u32,
}

// bytemuck Pod/Zeroable is unavailable without the dep, so we implement
// the unsafe conversions manually — these are plain u32 fields with
// repr(C), so the transmute is always valid.

impl Params {
    fn as_bytes(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        // SAFETY: Params is repr(C) with only u32 fields, no padding.
        unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<Self>()) }
    }
}

impl WgpuBackend {
    /// Try to create a wgpu backend. Returns `None` if no suitable GPU is found
    /// or if device creation fails.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request a high-performance adapter
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
            "wgpu adapter selected"
        );

        // Request device with default limits
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ternary-rs"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        // Compile shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ternary_matvec"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Bind group layout: weights, activations, output, params
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ternary_matvec_layout"),
            entries: &[
                // binding 0: weights (storage, read)
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
                // binding 1: activations (storage, read)
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
                // binding 2: output (storage, read_write)
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
                // binding 3: params (uniform)
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

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Pad a byte slice to 4-byte alignment (wgpu requirement).
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

    /// Pack i8 activations into a byte buffer (they're already byte-sized,
    /// but we need to ensure u32 alignment for the GPU buffer).
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

        // Pad weight data to u32 alignment
        let weight_bytes = Self::pad_to_u32(weights.packed_data());
        let act_bytes = Self::pack_activations(input);
        let output_size = (rows * std::mem::size_of::<i32>()) as u64;

        // Create GPU buffers
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

        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
        };
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, params.as_bytes());

        // Bind group
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

        // Dispatch
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

        // Read back
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

    /// Skip tests if no GPU is available (CI, headless, etc.).
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
        assert_eq!(y, vec![30]); // 10 - 20 + 0 + 40
    }

    #[test]
    fn all_zeros() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[0, 0, 0, 0], 1, 4);
        let x = vec![100i8, -50, 25, -12];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![0]);
    }

    #[test]
    fn non_aligned_cols() {
        let Some(backend) = get_backend() else { return };
        // 5 columns — not a multiple of 4
        let w = weights_from_i8(&[1, -1, 1, 0, -1], 1, 5);
        let x = vec![10i8, 20, 30, 40, 50];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![-30]); // 10 - 20 + 30 + 0 - 50
    }

    #[test]
    fn multi_row() {
        let Some(backend) = get_backend() else { return };
        let w = weights_from_i8(&[
             1,  0,  0,  0,
             0,  1,  0,  0,
            -1, -1, -1, -1,
        ], 3, 4);
        let x = vec![5i8, 10, 15, 20];
        let y = backend.ternary_matvec(&w, &x);
        assert_eq!(y, vec![5, 10, -50]);
    }

    #[test]
    fn matches_scalar_random() {
        let Some(backend) = get_backend() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        // Pseudo-random weights and activations (deterministic)
        let rows = 64;
        let cols = 128;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xDEAD_BEEF;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((rng >> 33) % 3) as i8 - 1; // -1, 0, 1
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

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(g, c, "mismatch at row {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn large_matrix() {
        let Some(backend) = get_backend() else { return };
        let scalar = crate::compute::scalar::ScalarBackend;

        // Realistic size: 2048 cols (embed_dim), 256 rows
        let rows = 256;
        let cols = 2048;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 42;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            weights_i8.push(((rng >> 33) % 3) as i8 - 1);
        }
        let w = weights_from_i8(&weights_i8, rows, cols);

        let mut activations = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            activations.push(((rng >> 33) % 127) as i8);
        }

        let gpu_result = backend.ternary_matvec(&w, &activations);
        let cpu_result = scalar.ternary_matvec(&w, &activations);

        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(g, c, "mismatch at row {i}: gpu={g}, cpu={c}");
        }
    }

    #[test]
    fn backend_name() {
        let Some(backend) = get_backend() else { return };
        assert_eq!(backend.name(), "wgpu");
    }
}
