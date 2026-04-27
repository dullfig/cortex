//! GpuFloatLinear — float linear layer with weights resident in GPU memory.
//!
//! Float-family counterpart to `GpuBitLinear`. Weights start as f32 (dequantized
//! from F16/BF16/Q4_K/etc. by the GGUF loader), then get packed into f16 pairs
//! and uploaded once to a `wgpu::Buffer` that stays resident for the model's
//! lifetime. Forward dispatches the existing `matvec` shader (f16-packed
//! weights × f32 activations → f32 output).
//!
//! Like `GpuBitLinear`, only the small activation vector moves per forward call
//! — the multi-MB weight tensor never re-uploads.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::compute::wgpu_backend::GpuDevice;
use crate::layers::linear::LinearLayer;
use crate::tensor::FloatTensor;

/// Params struct for the matvec shader (see `shaders/matvec.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams {
    rows: u32,
    cols: u32,
}

/// A float linear layer with weights resident on the GPU.
pub struct GpuFloatLinear {
    gpu: Arc<GpuDevice>,
    /// Resident f16-packed weights (out_features × in_features / 2 u32s).
    weight_buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

impl GpuFloatLinear {
    /// Upload an f32 weight tensor to the GPU as f16-packed, resident for the layer's life.
    pub fn from_float_tensor(gpu: Arc<GpuDevice>, tensor: FloatTensor) -> Self {
        assert_eq!(tensor.shape().len(), 2, "expected 2D tensor");
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(
            cols % 2 == 0,
            "GpuFloatLinear requires even in_features (f16 packing); got cols={cols}"
        );

        let packed = GpuDevice::pack_f16(tensor.data());

        let weight_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_floatlinear.weights"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self { gpu, weight_buf, rows, cols }
    }

    /// Bytes held in resident GPU memory by this layer's weights.
    pub fn resident_bytes(&self) -> u64 {
        self.weight_buf.size()
    }
}

impl LinearLayer for GpuFloatLinear {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols, "input dimension mismatch");

        let act_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_floatlinear.activations"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (self.rows * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_floatlinear.output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.gpu.create_staging_buffer(output_size);

        let params = MatvecParams { rows: self.rows as u32, cols: self.cols as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matvec;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[&self.weight_buf, &act_buf, &output_buf, &params_buf],
        );

        // The matvec shader computes row from `wid.x + wid.y * 65535`, so rows
        // above 65535 (vocab projections — Qwen2.5's vocab is 151k) need the
        // Y dimension to split the dispatch.
        let dispatch_x = self.rows.min(65535) as u32;
        let dispatch_y = ((self.rows + 65534) / 65535) as u32;

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_floatlinear.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_floatlinear.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
        let out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging_buf.unmap();
        out
    }

    fn in_features(&self) -> usize { self.cols }
    fn out_features(&self) -> usize { self.rows }
}

impl std::fmt::Debug for GpuFloatLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuFloatLinear({}→{}, resident={}B)",
            self.cols,
            self.rows,
            self.weight_buf.size(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::floatlinear::FloatLinear;

    #[test]
    fn matches_cpu_floatlinear_identity() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // 2x2 identity, cols even as required.
        let tensor = FloatTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let cpu = FloatLinear::from_float_tensor(tensor.clone());
        let layer = GpuFloatLinear::from_float_tensor(gpu, tensor);

        let input = vec![3.0f32, -2.5];
        let cpu_out = cpu.forward(&input);
        let gpu_out = layer.forward(&input);

        for (c, g) in cpu_out.iter().zip(&gpu_out) {
            assert!((c - g).abs() < 0.01, "cpu={c} gpu={g}");
        }
    }

    #[test]
    fn matches_cpu_floatlinear_random() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 48;
        let cols = 128;
        let mut weights = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xF00DCAFE;
        for _ in 0..(rows * cols) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            weights.push(((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001);
        }
        let tensor = FloatTensor::new(weights, vec![rows, cols]);

        let mut input = Vec::with_capacity(cols);
        for _ in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            input.push(((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001);
        }

        let cpu = FloatLinear::from_float_tensor(tensor.clone());
        let layer = GpuFloatLinear::from_float_tensor(gpu, tensor);

        let cpu_out = cpu.forward(&input);
        let gpu_out = layer.forward(&input);

        // f16 round-trip loses some precision; tolerance is generous.
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-2, "row {i}: cpu={c} gpu={g}");
        }
    }

    #[test]
    fn weights_remain_resident_across_calls() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let tensor = FloatTensor::new(vec![0.5, -0.25, 0.125, 0.0625], vec![2, 2]);
        let layer = GpuFloatLinear::from_float_tensor(gpu, tensor);
        let resident_before = layer.resident_bytes();

        for _ in 0..10 {
            let _ = layer.forward(&[1.0, 0.5]);
        }

        assert_eq!(layer.resident_bytes(), resident_before);
        assert!(resident_before > 0);
    }

    #[test]
    fn handles_rows_above_65k() {
        // Smoke test for the dispatch-split logic (Qwen vocab projections are
        // 151k rows). Use a tiny cols so the test doesn't take forever.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let rows = 70_000;
        let cols = 2;
        let weights = vec![1.0f32; rows * cols];
        let tensor = FloatTensor::new(weights, vec![rows, cols]);

        let layer = GpuFloatLinear::from_float_tensor(gpu, tensor);
        let out = layer.forward(&[1.0, 1.0]);

        assert_eq!(out.len(), rows);
        // Each row is [1.0, 1.0] · [1.0, 1.0] = 2.0 (within f16 tolerance).
        for (i, v) in out.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-2, "row {i}: got {v}");
        }
    }
}
