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
    fn wrapper_metadata_passes_through() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        assert_eq!(engine.vocab_size(), 4);
        assert_eq!(engine.embed_dim(), 4);
        assert_eq!(engine.n_layers(), 1);
    }
}
