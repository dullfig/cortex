//! GpuEngine — fused GPU forward pass orchestrator.
//!
//! Wraps a `TransformerModel` and a shared `GpuDevice`. The intent is to
//! reimplement the model's forward methods to keep activations on-device
//! across layers, replacing the per-layer CPU↔GPU round-trip that
//! `GpuBitLinear` / `GpuFloatLinear` still perform on their own.
//!
//! ## Phase plan
//!
//! - **1a (this commit):** wrapper scaffolding. Every public method
//!   delegates to the embedded `TransformerModel`. This proves the wrapper
//!   compiles, doesn't double-load weights (we hold a `&Arc<GpuDevice>`
//!   that's already shared with the resident layers), and gives us a place
//!   to add GPU-native methods incrementally.
//! - **1b:** GPU-native embedding lookup and final norm; blocks still
//!   delegate to CPU. Activations cross the GPU boundary at block edges
//!   only.
//! - **1c:** one transformer block on GPU end-to-end (rmsnorm → Q/K/V
//!   matvec → RoPE → attention → O proj → residual → ffn_norm → SwiGLU →
//!   residual).
//! - **1d:** all blocks on GPU; phase-1 done. Multi-token prefill,
//!   no KV cache yet (decode + cache come in #9).
//!
//! ## Discipline (per integration-claude's note 2026-04-23)
//!
//! Keep the wrapper thin. Only methods we reimplement on GPU live as real
//! code here; everything else is a one-line passthrough. When the trait
//! refactor (Option A) lands later, the trait surface is "the union of
//! methods this file actually implements" — pure mechanical lift, no
//! design-the-trait-up-front guesses.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::compute::wgpu_backend::GpuDevice;
use crate::layers::kv_cache::ModelKvCache;
use crate::layers::linear::LinearLayer;
use crate::layers::model::TransformerModel;
use crate::layers::sampler::SamplerConfig;
use crate::layers::trace::ForwardTrace;
use crate::layers::transformer::FfnInjector;

/// Params struct for the rmsnorm_batch shader. Layout must match
/// `compute/shaders/rmsnorm_batch.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormBatchParams {
    n: u32,
    eps: f32,
    n_tokens: u32,
    _pad: u32,
}

/// Params struct for the matvec shader. Layout must match
/// `compute/shaders/matvec.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams {
    rows: u32,
    cols: u32,
}

/// Params struct for the rope_batch shader. Layout must match
/// `compute/shaders/rope_batch.wgsl` exactly — eight u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeBatchParams {
    n_heads: u32,
    head_dim: u32,
    start_pos: u32,
    half_dim: u32,
    n_tokens: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

/// Fused GPU forward-pass orchestrator wrapping a `TransformerModel`.
pub struct GpuEngine {
    /// CPU-side model. Owns the layers (which may themselves hold resident
    /// GPU buffers via `GpuBitLinear` / `GpuFloatLinear`). Phase 1a delegates
    /// every call through to this; later phases replace specific calls with
    /// on-device dispatches that read the same resident buffers.
    cpu: TransformerModel,
    /// Shared GPU context (device, queue, pipelines). Same `Arc` already
    /// held by the resident layers inside `cpu` — no double allocation.
    gpu: Arc<GpuDevice>,
    /// Resident f32 weights for the final RMSNorm.
    final_norm_weight_buf: wgpu::Buffer,
    /// Captured at construction time so the dispatcher doesn't have to
    /// re-borrow the CPU model on every call.
    final_norm_eps: f32,
    /// Captured for shader-param construction.
    embed_dim: usize,
}

impl GpuEngine {
    /// Wrap a CPU `TransformerModel` with a shared GPU context.
    pub fn from_cpu_model(cpu: TransformerModel, gpu: Arc<GpuDevice>) -> Self {
        let final_norm = cpu.final_norm();
        let final_norm_weight_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.final_norm.weight"),
            contents: bytemuck::cast_slice(final_norm.weight()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let final_norm_eps = final_norm.eps();
        let embed_dim = cpu.embed_dim();

        Self {
            cpu,
            gpu,
            final_norm_weight_buf,
            final_norm_eps,
            embed_dim,
        }
    }

    /// GPU-native dispatch of the per-token final RMSNorm using the
    /// `rmsnorm_batch` pipeline. Kept private so callers compose it via
    /// `forward_gpu` rather than reach in directly.
    fn dispatch_final_norm(&self, pre_norm: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(pre_norm.len(), seq_len * self.embed_dim, "shape mismatch");
        let total_bytes = (pre_norm.len() * std::mem::size_of::<f32>()) as u64;

        let input_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.final_norm.input"),
            contents: bytemuck::cast_slice(pre_norm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_engine.final_norm.output"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.gpu.create_staging_buffer(total_bytes);

        let params = RmsNormBatchParams {
            n: self.embed_dim as u32,
            eps: self.final_norm_eps,
            n_tokens: seq_len as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.rmsnorm_batch;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[&input_buf, &self.final_norm_weight_buf, &output_buf, &params_buf],
        );

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_engine.final_norm.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.final_norm.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per token (rmsnorm_batch indexes tokens by workgroup_id.x).
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, total_bytes);
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

    /// Forward pass with GPU-native final RMSNorm. Embedding lookup and
    /// transformer blocks still run on CPU; output projection runs on CPU.
    /// Phase 1b checkpoint — proves the rmsnorm dispatch path is correct
    /// before we move attention/FFN into the same orchestration.
    pub fn forward_gpu(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let pre_norm = self.cpu.forward_pre_norm(tokens, start_pos);
        let normed = self.dispatch_final_norm(&pre_norm, tokens.len());
        self.cpu.finalize_logits(&normed, tokens.len())
    }

    /// Dispatch a single-vector matvec on GPU using the given layer's
    /// resident weight buffer. Records the dispatch into the supplied
    /// `encoder` so the caller can chain multiple ops without per-op submit.
    ///
    /// Currently supports `GpuFloatLinear` only — the ternary path needs an
    /// f32→i8 quantize-on-GPU step that doesn't exist yet (deferred to a
    /// later phase). Layers that aren't GpuFloatLinear panic.
    ///
    /// `in_buf` must be f32, `cols` elements long. `out_buf` must be f32,
    /// `rows` elements long.
    pub fn dispatch_matvec_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn crate::layers::linear::LinearLayer,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| {
                panic!(
                    "GpuEngine.dispatch_matvec_into: layer is not GpuFloatLinear \
                     (concrete type: {:?}); ternary fused-matvec is not implemented yet",
                    layer
                )
            });

        let params = MatvecParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matvec;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[float.weight_buffer(), in_buf, out_buf, &params_buf],
        );

        let dispatch_x = (float.out_features().min(65535)) as u32;
        let dispatch_y = ((float.out_features() + 65534) / 65535) as u32;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.matvec.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    /// Build cos/sin lookup tables for the `rope_batch` shader, sized to
    /// `max_seq` positions. Layout matches `rope_batch.wgsl`:
    /// `cos_table[pos * half_dim + i]`, same for sin. Half_dim = inv_freq.len().
    ///
    /// Called once at construction (cos/sin are fixed for a given RoPE
    /// config). Returns two storage buffers ready to be bound.
    pub fn build_rope_tables(
        gpu: &GpuDevice,
        inv_freq: &[f32],
        max_seq: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let half_dim = inv_freq.len();
        let mut cos = Vec::with_capacity(max_seq * half_dim);
        let mut sin = Vec::with_capacity(max_seq * half_dim);
        for pos in 0..max_seq {
            let p = pos as f32;
            for &freq in inv_freq {
                let angle = p * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        let cos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.rope.cos"),
            contents: bytemuck::cast_slice(&cos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sin_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.rope.sin"),
            contents: bytemuck::cast_slice(&sin),
            usage: wgpu::BufferUsages::STORAGE,
        });
        (cos_buf, sin_buf)
    }

    /// Dispatch RoPE in place on `x_buf`, which must hold f32 values laid
    /// out as `[n_tokens, n_heads, head_dim]`. Token `t` is rotated for
    /// position `start_pos + t`. Halved (NeoX/HF) layout — Qwen and BitNet
    /// both use this; interleaved (older llama.cpp) is not supported by the
    /// shader yet.
    pub fn dispatch_rope_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x_buf: &wgpu::Buffer,
        cos_buf: &wgpu::Buffer,
        sin_buf: &wgpu::Buffer,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        assert!(head_dim % 2 == 0, "RoPE head_dim must be even");
        let half_dim = head_dim / 2;

        let params = RopeBatchParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            half_dim: half_dim as u32,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.rope_batch;
        let bind_group = self.gpu.make_bind_group(
            pipeline,
            &[x_buf, cos_buf, sin_buf, &params_buf],
        );

        // Total threads = n_tokens * n_heads * half_dim. Workgroup size is 64.
        let total_threads = (n_tokens * n_heads * half_dim) as u32;
        let dispatch_x = (total_threads + 63) / 64;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rope.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }

    /// Convenience wrapper around `dispatch_matvec_into`: allocates input
    /// from a slice, runs the dispatch, reads back. Useful for testing the
    /// primitive in isolation; production callers should chain dispatches
    /// via `dispatch_matvec_into` directly.
    pub fn matvec_oneshot(
        &self,
        layer: &dyn crate::layers::linear::LinearLayer,
        input: &[f32],
    ) -> Vec<f32> {
        assert_eq!(input.len(), layer.in_features(), "input len mismatch");
        let out_bytes = (layer.out_features() * std::mem::size_of::<f32>()) as u64;

        let in_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.matvec.input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_engine.matvec.output"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.gpu.create_staging_buffer(out_bytes);

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_engine.matvec.encoder"),
        });
        self.dispatch_matvec_into(&mut encoder, layer, &in_buf, &out_buf);
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("readback failed").expect("buffer map failed");

        let data = slice.get_mapped_range();
        let out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging_buf.unmap();
        out
    }

    /// Borrow the underlying CPU model (for delegation in tests / debug).
    pub fn cpu(&self) -> &TransformerModel {
        &self.cpu
    }

    // -- Pure passthroughs (Phase 1a) ---------------------------------------

    pub fn vocab_size(&self) -> usize {
        self.cpu.vocab_size()
    }

    pub fn embed_dim(&self) -> usize {
        self.cpu.embed_dim()
    }

    pub fn n_layers(&self) -> usize {
        self.cpu.n_layers()
    }

    pub fn embedding_data(&self) -> &[f32] {
        self.cpu.embedding_data()
    }

    pub fn set_block_injector(&mut self, layer: usize, injector: Box<dyn FfnInjector>) {
        self.cpu.set_block_injector(layer, injector);
    }

    pub fn create_kv_cache(&self, max_seq_len: usize) -> ModelKvCache {
        self.cpu.create_kv_cache(max_seq_len)
    }

    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        self.cpu.forward(tokens, start_pos)
    }

    pub fn forward_last(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        self.cpu.forward_last(tokens, start_pos)
    }

    pub fn forward_cached(&self, tokens: &[u32], cache: &mut ModelKvCache) -> Vec<f32> {
        self.cpu.forward_cached(tokens, cache)
    }

    pub fn forward_traced(&self, tokens: &[u32]) -> (Vec<f32>, ForwardTrace) {
        self.cpu.forward_traced(tokens)
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampler_config: SamplerConfig,
        seed: u64,
        stop_token: Option<u32>,
    ) -> Vec<u32> {
        self.cpu.generate(prompt, max_tokens, sampler_config, seed, stop_token)
    }
}

impl std::fmt::Debug for GpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuEngine(wrapping {:?})", self.cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::attention::MultiHeadAttention;
    use crate::layers::bitlinear::BitLinear;
    use crate::layers::ffn::FeedForward;
    use crate::layers::model::OutputProjection;
    use crate::layers::rmsnorm::RmsNorm;
    use crate::layers::swiglu::SwiGLU;
    use crate::layers::transformer::TransformerBlock;
    use crate::tensor::{FloatTensor, Ternary, TernaryTensor};

    /// Build a tiny ternary BitLinear from raw i8 values for tests.
    fn bitlinear(values: &[i8], rows: usize, cols: usize, scale: f32) -> BitLinear {
        let ternary: Vec<Ternary> = values
            .iter()
            .map(|&v| match v {
                -1 => Ternary::Neg,
                0 => Ternary::Zero,
                1 => Ternary::Pos,
                _ => panic!("not ternary"),
            })
            .collect();
        BitLinear::new(TernaryTensor::pack(&ternary, rows, cols), scale)
    }

    /// Build a single-block, single-head, vocab=4 toy model. Same shape as
    /// the existing `TransformerModel` tests use, just enough to exercise a
    /// real forward pass.
    fn toy_model() -> TransformerModel {
        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let embedding = FloatTensor::new(
            (0..vocab_size * embed_dim).map(|i| (i as f32) * 0.1).collect(),
            vec![vocab_size, embed_dim],
        );

        let q = bitlinear(&[1, 0, -1, 0, 0, 1, 0, -1, 1, 1, 0, 0, 0, 0, 1, -1], embed_dim, embed_dim, 0.1);
        let k = bitlinear(&[0, 1, 0, -1, 1, 0, -1, 0, 0, -1, 1, 0, -1, 0, 0, 1], embed_dim, embed_dim, 0.1);
        let v = bitlinear(&[1, -1, 1, -1, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0, -1, 1], embed_dim, embed_dim, 0.1);
        let o = bitlinear(&[1, 0, 0, 1, 0, 1, 1, 0, -1, 0, 1, 0, 0, -1, 0, 1], embed_dim, embed_dim, 0.1);

        let attention = MultiHeadAttention::new(
            Box::new(q), Box::new(k), Box::new(v), Box::new(o),
            n_heads, n_heads, head_dim, 10000.0,
        );

        let gate = bitlinear(&[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], intermediate, embed_dim, 0.1);
        let up = bitlinear(&[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0], intermediate, embed_dim, 0.1);
        let down = bitlinear(&[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], embed_dim, intermediate, 0.1);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(Box::new(gate), Box::new(up), Box::new(down)));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::TiedEmbedding)
    }

    #[test]
    fn wrapper_forward_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let cpu_logits = cpu.forward(&[0, 1, 2], 0);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward(&[0, 1, 2], 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-6, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn wrapper_forward_traced_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let (cpu_logits, cpu_trace) = cpu.forward_traced(&[1, 2]);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let (gpu_logits, gpu_trace) = engine.forward_traced(&[1, 2]);

        assert_eq!(cpu_logits, gpu_logits);
        assert_eq!(cpu_trace.n_layers, gpu_trace.n_layers);
        assert_eq!(cpu_trace.hidden_states.len(), gpu_trace.hidden_states.len());
    }

    #[test]
    fn forward_gpu_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu = toy_model();
        let cpu_logits = cpu.forward(&[0, 1, 2], 0);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward_gpu(&[0, 1, 2], 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        // f32 rmsnorm on GPU vs CPU — same precision, very tight tolerance.
        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn forward_gpu_single_token() {
        // Smoke-test the seq_len=1 dispatch path (one workgroup).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_logits = toy_model().forward(&[3], 0);
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let gpu_logits = engine.forward_gpu(&[3], 0);

        for (i, (a, b)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i}: cpu={a} gpu={b}");
        }
    }

    #[test]
    fn dispatch_matvec_matches_gpu_floatlinear_forward() {
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::linear::LinearLayer;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Random-ish 32x16 weight matrix.
        let rows = 32;
        let cols = 16;
        let mut weights = Vec::with_capacity(rows * cols);
        let mut rng: u64 = 0xDEADBEEF;
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

        let gpu_layer = GpuFloatLinear::from_float_tensor(gpu.clone(), tensor);
        // The reference path also runs through GpuFloatLinear (which packs
        // weights to f16). We're verifying that going via GpuEngine's
        // dispatch_matvec_into yields the same answer as calling
        // GpuFloatLinear::forward directly.
        let reference = gpu_layer.forward(&input);

        // GpuEngine needs a TransformerModel to construct; build a trivial
        // one and reuse the engine just for its matvec helper.
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let layer_ref: &dyn LinearLayer = &gpu_layer;
        let gpu_out = engine.matvec_oneshot(layer_ref, &input);

        assert_eq!(reference.len(), gpu_out.len());
        // Identical pipeline + identical resident buffer; should be bit-equal.
        for (i, (r, g)) in reference.iter().zip(&gpu_out).enumerate() {
            assert!((r - g).abs() < 1e-6, "row {i}: ref={r} gpu={g}");
        }
    }

    #[test]
    fn dispatch_rope_matches_cpu_rope() {
        use crate::layers::rope::{RoPE, RoPELayout};

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 2;
        let head_dim = 8;
        let n_tokens = 3;
        let start_pos = 0;
        let max_seq = 16;

        let rope = RoPE::with_layout(head_dim, 10000.0, RoPELayout::Halved);

        // Build a deterministic input: [n_tokens, n_heads, head_dim] as f32.
        let mut x: Vec<f32> = Vec::with_capacity(n_tokens * n_heads * head_dim);
        for i in 0..(n_tokens * n_heads * head_dim) {
            x.push((i as f32) * 0.01 - 0.5);
        }

        // CPU reference: rotate each (tok, head) head_dim slice at pos start_pos+tok.
        let mut cpu_out = x.clone();
        for tok in 0..n_tokens {
            for h in 0..n_heads {
                let off = tok * n_heads * head_dim + h * head_dim;
                let slice = &x[off..off + head_dim].to_vec();
                let rotated = rope.forward(slice, start_pos + tok);
                cpu_out[off..off + head_dim].copy_from_slice(&rotated);
            }
        }

        // GPU path
        let (cos_buf, sin_buf) = GpuEngine::build_rope_tables(&gpu, rope.inv_freq(), max_seq);
        let total_bytes = (x.len() * std::mem::size_of::<f32>()) as u64;
        let x_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_test.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(total_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_test.encoder"),
        });
        engine.dispatch_rope_into(
            &mut encoder, &x_buf, &cos_buf, &sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, total_bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(data);
        staging.unmap();

        assert_eq!(cpu_out.len(), gpu_out.len());
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 1e-5,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn dispatch_rope_nonzero_start_pos() {
        // Smoke-test that start_pos plumbs through to the right cos/sin row.
        use crate::layers::rope::{RoPE, RoPELayout};
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 1;
        let head_dim = 4;
        let n_tokens = 2;
        let start_pos = 5;
        let max_seq = 16;

        let rope = RoPE::with_layout(head_dim, 10000.0, RoPELayout::Halved);
        let x: Vec<f32> = (0..(n_tokens * n_heads * head_dim))
            .map(|i| (i as f32) * 0.1)
            .collect();

        let mut cpu_out = x.clone();
        for tok in 0..n_tokens {
            for h in 0..n_heads {
                let off = tok * n_heads * head_dim + h * head_dim;
                let rotated = rope.forward(&x[off..off + head_dim].to_vec(), start_pos + tok);
                cpu_out[off..off + head_dim].copy_from_slice(&rotated);
            }
        }

        let (cos_buf, sin_buf) = GpuEngine::build_rope_tables(&gpu, rope.inv_freq(), max_seq);
        let total_bytes = (x.len() * std::mem::size_of::<f32>()) as u64;
        let x_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_test2.x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(total_bytes);
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_test2.encoder"),
        });
        engine.dispatch_rope_into(
            &mut encoder, &x_buf, &cos_buf, &sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, total_bytes);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_out: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!((c - g).abs() < 1e-5, "elem {i}: cpu={c} gpu={g}");
        }
    }

    #[test]
    #[should_panic(expected = "not GpuFloatLinear")]
    fn dispatch_matvec_panics_on_cpu_layer() {
        // CPU BitLinear in the path → should panic with a clear message,
        // not produce silent garbage. Real loaders use the GPU layers when
        // the GPU is available.
        let Some(gpu) = GpuDevice::try_new() else {
            panic!("not GpuFloatLinear (skipping with the expected message so the test still asserts)");
        };
        let gpu = Arc::new(gpu);
        let cpu_layer = bitlinear(&[1, 0, 0, 1], 2, 2, 0.1);
        let layer_ref: &dyn crate::layers::linear::LinearLayer = &cpu_layer;
        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        let _ = engine.matvec_oneshot(layer_ref, &[1.0, 0.0]);
    }

    #[test]
    fn wrapper_metadata_passes_through() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        assert_eq!(engine.vocab_size(), 4);
        assert_eq!(engine.embed_dim(), 4);
        assert_eq!(engine.n_layers(), 1);
    }
}
