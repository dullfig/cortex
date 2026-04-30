//! GpuBitLinear — ternary linear layer with weights resident in GPU memory.
//!
//! Counterpart to `BitLinear` (CPU). Weights are uploaded to a `wgpu::Buffer`
//! once at load time and stay there for the model's lifetime. Each forward
//! pass uploads only the activation vector (small) and reads back the output
//! vector (also small) — the multi-MB weight tensor never moves.
//!
//! All `GpuBitLinear` instances in a model share one `Arc<GpuDevice>`, which
//! owns the `wgpu::Device`, queue, and compiled `ternary_matvec` pipeline.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::compute::wgpu_backend::GpuDevice;
use crate::layers::linear::LinearLayer;
use crate::ops::quantize::quantize_absmax;
use crate::tensor::TernaryTensor;

/// Params struct passed to the ternary matvec shader. Layout must match
/// `TERNARY_SHADER` in `compute::wgpu_backend`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TernaryParams {
    rows: u32,
    cols: u32,
}

/// A ternary linear layer with weights resident on the GPU.
pub struct GpuBitLinear {
    gpu: Arc<GpuDevice>,
    /// Resident packed ternary weights (out_features × in_features, 2 bits each).
    weight_buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
    weight_scale: f32,
}

impl GpuBitLinear {
    /// Upload ternary weights to the GPU and build a layer that holds them resident.
    pub fn from_weights(
        gpu: Arc<GpuDevice>,
        weights: TernaryTensor,
        weight_scale: f32,
    ) -> Self {
        let rows = weights.rows();
        let cols = weights.cols();

        // The shader reads weights as `array<u32>`, so the upload must be a
        // multiple of 4 bytes. Pad with zeros if the packed length isn't.
        let mut packed = weights.packed_data().to_vec();
        let remainder = packed.len() % 4;
        if remainder != 0 {
            packed.resize(packed.len() + (4 - remainder), 0);
        }

        // create_buffer + queue.write_buffer (persistent staging belt) over
        // create_buffer_init (per-call staging churn) — same fix as
        // GpuFloatLinear, see #16.
        let weight_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_bitlinear.weights"),
            size: packed.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&weight_buf, 0, &packed);

        Self { gpu, weight_buf, rows, cols, weight_scale }
    }

    /// Bytes occupied by resident weight data (excluding alignment padding overhead is small).
    pub fn resident_bytes(&self) -> u64 {
        self.weight_buf.size()
    }
}

impl LinearLayer for GpuBitLinear {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols, "input dimension mismatch");

        let (x_quant, act_scale) = quantize_absmax(input);

        // Pack i8 activations into a u32-aligned byte buffer.
        let mut act_bytes: Vec<u8> = x_quant.iter().map(|&v| v as u8).collect();
        let remainder = act_bytes.len() % 4;
        if remainder != 0 {
            act_bytes.resize(act_bytes.len() + (4 - remainder), 0);
        }

        let act_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_bitlinear.activations"),
            contents: &act_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (self.rows * std::mem::size_of::<i32>()) as u64;
        let output_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_bitlinear.output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.gpu.create_staging_buffer(output_size);

        let params = TernaryParams { rows: self.rows as u32, cols: self.cols as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.ternary_matvec;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[&self.weight_buf, &act_buf, &output_buf, &params_buf],
        );

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_bitlinear.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_bitlinear.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.rows as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("GPU readback failed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let combined_scale = act_scale * self.weight_scale;
        let out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 * combined_scale)
            .collect();
        drop(data);
        staging_buf.unmap();
        out
    }

    fn in_features(&self) -> usize { self.cols }
    fn out_features(&self) -> usize { self.rows }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl GpuBitLinear {
    /// Borrow the resident weight buffer (for fused-forward orchestrators
    /// that need to chain matvec dispatches against this layer's weights).
    pub fn weight_buffer(&self) -> &wgpu::Buffer { &self.weight_buf }

    /// Per-layer weight scale γ used when rescaling integer accumulators.
    pub fn weight_scale(&self) -> f32 { self.weight_scale }

    /// Shared GPU device handle.
    pub fn gpu(&self) -> &Arc<GpuDevice> { &self.gpu }
}

impl std::fmt::Debug for GpuBitLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuBitLinear({}→{}, γ={:.6}, resident={}B)",
            self.cols,
            self.rows,
            self.weight_scale,
            self.weight_buf.size(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::bitlinear::BitLinear;
    use crate::tensor::Ternary;

    fn weights_from_i8(values: &[i8], rows: usize, cols: usize) -> TernaryTensor {
        let ternary: Vec<Ternary> = values.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        TernaryTensor::pack(&ternary, rows, cols)
    }

    #[test]
    fn matches_cpu_bitlinear_identity() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let weights = weights_from_i8(&[1, 0, 0, 1], 2, 2);
        let cpu = BitLinear::new(weights.clone(), 1.0);
        let layer = GpuBitLinear::from_weights(gpu, weights, 1.0);

        let input = vec![1.0f32, 0.5];
        let cpu_out = cpu.forward(&input);
        let gpu_out = layer.forward(&input);

        assert_eq!(cpu_out.len(), gpu_out.len());
        for (c, g) in cpu_out.iter().zip(&gpu_out) {
            assert!((c - g).abs() < 0.02, "cpu={c} gpu={g}");
        }
    }

    #[test]
    fn matches_cpu_bitlinear_random() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 32;
        let cols = 64;
        let mut weights_i8 = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xCAFEBABE;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            weights_i8.push(((rng >> 33) % 3) as i8 - 1);
        }
        let weights = weights_from_i8(&weights_i8, rows, cols);

        let mut input = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            input.push(((rng >> 33) as i32 % 200 - 100) as f32 * 0.01);
        }

        let cpu = BitLinear::new(weights.clone(), 0.137);
        let layer = GpuBitLinear::from_weights(gpu, weights, 0.137);

        let cpu_out = cpu.forward(&input);
        let gpu_out = layer.forward(&input);

        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-4, "row {i}: cpu={c} gpu={g}");
        }
    }

    #[test]
    fn weights_remain_resident_across_calls() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let weights = weights_from_i8(&[1, -1, 0, 1, 0, 1, -1, 1], 2, 4);
        let layer = GpuBitLinear::from_weights(gpu, weights, 0.5);
        let resident_before = layer.resident_bytes();

        for _ in 0..10 {
            let _ = layer.forward(&[1.0, 0.0, -1.0, 0.5]);
        }

        // Buffer handle still valid and size unchanged — weights were not re-uploaded.
        assert_eq!(layer.resident_bytes(), resident_before);
        assert!(resident_before > 0);
    }
}
