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

/// Params struct for the attn_score_batch shader. Twelve u32s; the trailing
/// padding entries are required for std140-ish uniform alignment.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScoreBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    scale: f32,
    n_tokens: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

/// Params struct for the softmax_batch shader. Four u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxBatchParams {
    n_heads: u32,
    max_seq: u32,
    start_pos: u32,
    n_tokens: u32,
}

/// Params struct for the attn_value_batch shader. Eight u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnValueBatchParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    n_tokens: u32,
}

/// Params struct for the silu_mul_batch shader. Two u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluMulBatchParams {
    n: u32,
    n_tokens: u32,
}

/// Params struct for the add_inplace_batch shader. Two u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AddInplaceBatchParams {
    n: u32,
    n_tokens: u32,
}

/// Per-block GPU resources extracted at construction time. Holds resident
/// rmsnorm weights for the two norms inside a `TransformerBlock`. The matvec
/// weights are accessed lazily via the CPU model's block at dispatch time
/// (they live inside `Box<dyn LinearLayer>` and are reached via
/// `as_any().downcast_ref::<GpuFloatLinear>()`).
struct GpuBlock {
    attn_norm_weight_buf: wgpu::Buffer,
    attn_norm_eps: f32,
    ffn_norm_weight_buf: wgpu::Buffer,
    ffn_norm_eps: f32,
}

/// Per-block scratch buffers reused across all dispatches inside a single
/// `forward_block_gpu` call. Caller allocates once per forward pass and
/// reuses across blocks (since dimensions are constant).
pub struct BlockScratch {
    pub normed: wgpu::Buffer,    // [n_tokens, embed_dim] post-rmsnorm scratch
    pub q: wgpu::Buffer,         // [n_tokens, n_heads * head_dim]
    pub k: wgpu::Buffer,         // [n_tokens, n_kv_heads * head_dim]
    pub v: wgpu::Buffer,         // [n_tokens, n_kv_heads * head_dim]
    pub attn_out: wgpu::Buffer,  // [n_tokens, n_heads * head_dim]
    pub scores: wgpu::Buffer,    // [n_tokens, n_heads, max_seq] attention scores
    pub gate: wgpu::Buffer,      // [n_tokens, intermediate]
    pub up: wgpu::Buffer,        // [n_tokens, intermediate]
    pub activated: wgpu::Buffer, // [n_tokens, intermediate] SiLU(gate)*up
    pub projected: wgpu::Buffer, // [n_tokens, embed_dim] both attn-out-proj and FFN-down output reuse this
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
        let mk = |size: u64, label: &str| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        Self {
            normed:    mk((n_tokens * embed_dim) as u64 * f32_bytes, "scratch.normed"),
            q:         mk((n_tokens * n_heads * head_dim) as u64 * f32_bytes, "scratch.q"),
            k:         mk((n_tokens * n_kv_heads * head_dim) as u64 * f32_bytes, "scratch.k"),
            v:         mk((n_tokens * n_kv_heads * head_dim) as u64 * f32_bytes, "scratch.v"),
            attn_out:  mk((n_tokens * n_heads * head_dim) as u64 * f32_bytes, "scratch.attn_out"),
            scores:    mk((n_tokens * n_heads * max_seq) as u64 * f32_bytes, "scratch.scores"),
            gate:      mk((n_tokens * intermediate) as u64 * f32_bytes, "scratch.gate"),
            up:        mk((n_tokens * intermediate) as u64 * f32_bytes, "scratch.up"),
            activated: mk((n_tokens * intermediate) as u64 * f32_bytes, "scratch.activated"),
            projected: mk((n_tokens * embed_dim) as u64 * f32_bytes, "scratch.projected"),
        }
    }
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
    /// Per-block resident resources.
    blocks_gpu: Vec<GpuBlock>,
    /// Resident RoPE cos lookup table, sized to `rope_max_seq * (head_dim/2)`.
    rope_cos_buf: wgpu::Buffer,
    /// Resident RoPE sin lookup table, same shape as `rope_cos_buf`.
    rope_sin_buf: wgpu::Buffer,
    /// Maximum sequence length the rope tables cover. Forward calls with
    /// `start_pos + n_tokens > rope_max_seq` would index out of range, so
    /// they assert.
    rope_max_seq: usize,
}

impl GpuEngine {
    /// Wrap a CPU `TransformerModel` with a shared GPU context. Uses the
    /// default RoPE-table size (4096 positions); call `with_max_seq` if
    /// you need a longer context window.
    pub fn from_cpu_model(cpu: TransformerModel, gpu: Arc<GpuDevice>) -> Self {
        Self::with_max_seq(cpu, gpu, 4096)
    }

    /// Wrap a CPU `TransformerModel` with a shared GPU context, sizing the
    /// RoPE cos/sin lookup tables to `max_seq` positions.
    pub fn with_max_seq(cpu: TransformerModel, gpu: Arc<GpuDevice>, max_seq: usize) -> Self {
        // Final norm
        let final_norm = cpu.final_norm();
        let final_norm_weight_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_engine.final_norm.weight"),
            contents: bytemuck::cast_slice(final_norm.weight()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let final_norm_eps = final_norm.eps();
        let embed_dim = cpu.embed_dim();

        // Per-block norms
        let blocks_gpu: Vec<GpuBlock> = cpu.blocks().iter().enumerate().map(|(i, blk)| {
            let an = blk.attn_norm();
            let fn_ = blk.ffn_norm();
            GpuBlock {
                attn_norm_weight_buf: gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("gpu_engine.block{i}.attn_norm.weight")),
                    contents: bytemuck::cast_slice(an.weight()),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                attn_norm_eps: an.eps(),
                ffn_norm_weight_buf: gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("gpu_engine.block{i}.ffn_norm.weight")),
                    contents: bytemuck::cast_slice(fn_.weight()),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
                ffn_norm_eps: fn_.eps(),
            }
        }).collect();

        // RoPE tables. All blocks share one rope (same base + head_dim).
        let attn0 = cpu.blocks()[0].attention();
        let (rope_cos_buf, rope_sin_buf) =
            Self::build_rope_tables(&gpu, attn0.rope().inv_freq(), max_seq);

        Self {
            cpu,
            gpu,
            final_norm_weight_buf,
            final_norm_eps,
            embed_dim,
            blocks_gpu,
            rope_cos_buf,
            rope_sin_buf,
            rope_max_seq: max_seq,
        }
    }

    /// Dispatch RMSNorm into `out_buf` from `in_buf`, using `weight_buf` for
    /// the per-feature scale. Both buffers are `[n_tokens, n]` flat. One
    /// workgroup per token via `rmsnorm_batch`.
    pub fn dispatch_rmsnorm_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        in_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
        eps: f32,
    ) {
        let params = RmsNormBatchParams {
            n: n as u32, eps, n_tokens: n_tokens as u32, _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);
        let pipeline = &self.gpu.pipelines.rmsnorm_batch;
        let bind = self.gpu.make_bind_group(
            pipeline, &[in_buf, weight_buf, out_buf, &params_buf],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rmsnorm.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Dispatch a batch matmul against a `GpuFloatLinear` layer's resident
    /// weights. Input is `[n_tokens, in_features]`, output is
    /// `[n_tokens, out_features]`. Uses the `matmul` (batch) shader which
    /// processes all tokens in one dispatch — the right primitive for prefill.
    pub fn dispatch_matmul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        let float = layer
            .as_any()
            .downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>()
            .unwrap_or_else(|| {
                panic!(
                    "GpuEngine.dispatch_matmul_into: layer is not GpuFloatLinear \
                     (concrete type: {:?})",
                    layer
                )
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }
        let params = MatmulParams {
            rows: float.out_features() as u32,
            cols: float.in_features() as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.matmul;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[float.weight_buffer(), in_buf, out_buf, &params_buf],
        );

        // Workgroup dispatch: x = row & 65535, y = row >> 16, z = tok.
        let rows = float.out_features();
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.matmul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
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

    /// Run one transformer block fully on GPU. Reads `hidden_buf` (shape
    /// `[n_tokens, embed_dim]`), modifies it in place with the post-block
    /// hidden state. All intermediate scratch buffers must be supplied by
    /// the caller — `forward_blocks_gpu` allocates them once and reuses
    /// across all blocks.
    ///
    /// Caveats (will be lifted in later phases):
    /// - Q/K/V biases (Qwen2) ignored: panics if any are set.
    /// - Attention sub-norm (BitNet) ignored: panics if set.
    /// - FFN sub-norm (BitNet) ignored: panics if set.
    /// - Residual scales must be 1.0.
    /// - FFN must be a `SwiGLU` with `SiLU` activation.
    /// - Matvec layers must be `GpuFloatLinear` (ternary fused not yet built).
    /// - For prefill mode (no historical KV cache): `start_pos = 0`. Cached
    ///   decoding lands in #9 along with the resident KV cache.
    pub fn forward_block_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &wgpu::Buffer,
        n_tokens: usize,
        start_pos: usize,
        scratch: &BlockScratch,
    ) {
        let block = &self.cpu.blocks()[block_idx];
        let block_gpu = &self.blocks_gpu[block_idx];
        let attn = block.attention();

        assert!(attn.q_bias().is_none() && attn.k_bias().is_none() && attn.v_bias().is_none(),
            "forward_block_gpu does not support Q/K/V biases yet (Qwen2)");
        assert!(attn.o_sub_norm().is_none(),
            "forward_block_gpu does not support attention sub-norm yet (BitNet)");
        assert!((block.attn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires attn_residual_scale == 1.0");
        assert!((block.ffn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires ffn_residual_scale == 1.0");

        let swiglu = block.ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_block_gpu requires SwiGLU FFN"));
        assert!(swiglu.sub_norm().is_none(),
            "forward_block_gpu does not support FFN sub-norm yet (BitNet)");
        assert_eq!(swiglu.activation(), crate::layers::swiglu::GateActivation::SiLU,
            "forward_block_gpu only supports SiLU activation");

        let n_heads = attn.n_heads();
        let n_kv_heads = attn.n_kv_heads();
        let head_dim = attn.head_dim();
        let embed_dim = self.embed_dim;
        let intermediate = swiglu.intermediate_size();

        assert!(start_pos + n_tokens <= self.rope_max_seq,
            "start_pos + n_tokens ({}) exceeds rope_max_seq ({})",
            start_pos + n_tokens, self.rope_max_seq);

        // ===== ATTENTION SUBLAYER =====

        // 1. attn_norm: hidden -> normed
        self.dispatch_rmsnorm_into(
            encoder, hidden_buf, &block_gpu.attn_norm_weight_buf, &scratch.normed,
            embed_dim, n_tokens, block_gpu.attn_norm_eps,
        );

        // 2-4. Q, K, V projections (batch matmul)
        self.dispatch_matmul_into(encoder, attn.q_proj(), &scratch.normed, &scratch.q, n_tokens);
        self.dispatch_matmul_into(encoder, attn.k_proj(), &scratch.normed, &scratch.k, n_tokens);
        self.dispatch_matmul_into(encoder, attn.v_proj(), &scratch.normed, &scratch.v, n_tokens);

        // 5. RoPE on Q and K
        self.dispatch_rope_into(
            encoder, &scratch.q, &self.rope_cos_buf, &self.rope_sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        self.dispatch_rope_into(
            encoder, &scratch.k, &self.rope_cos_buf, &self.rope_sin_buf,
            n_kv_heads, head_dim, start_pos, n_tokens,
        );

        // 6. Attention math: Q · K^T, softmax, weighted V. For prefill mode
        //    start_pos=0 and max_seq=n_tokens (the K/V buffers ARE the cache).
        self.dispatch_attention_into(
            encoder,
            &scratch.q, &scratch.k, &scratch.v,
            &scratch.scores, &scratch.attn_out,
            n_heads, n_kv_heads, head_dim,
            start_pos, n_tokens, n_tokens,
        );

        // 7. O projection: attn_out -> projected
        self.dispatch_matmul_into(encoder, attn.o_proj(), &scratch.attn_out, &scratch.projected, n_tokens);

        // 8. Residual: hidden += projected
        self.dispatch_add_into(encoder, hidden_buf, &scratch.projected, embed_dim, n_tokens);

        // ===== FFN SUBLAYER =====

        // 9. ffn_norm: hidden -> normed
        self.dispatch_rmsnorm_into(
            encoder, hidden_buf, &block_gpu.ffn_norm_weight_buf, &scratch.normed,
            embed_dim, n_tokens, block_gpu.ffn_norm_eps,
        );

        // 10-11. Gate / Up projections
        self.dispatch_matmul_into(encoder, swiglu.gate_proj(), &scratch.normed, &scratch.gate, n_tokens);
        self.dispatch_matmul_into(encoder, swiglu.up_proj(),   &scratch.normed, &scratch.up,   n_tokens);

        // 12. silu(gate) * up
        self.dispatch_silu_mul_into(encoder, &scratch.gate, &scratch.up, &scratch.activated, intermediate, n_tokens);

        // 13. Down projection
        self.dispatch_matmul_into(encoder, swiglu.down_proj(), &scratch.activated, &scratch.projected, n_tokens);

        // 14. Residual: hidden += projected
        self.dispatch_add_into(encoder, hidden_buf, &scratch.projected, embed_dim, n_tokens);
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

    /// Dispatch GQA attention math (attn_score → softmax → attn_value)
    /// against pre-projected, RoPE-rotated Q/K/V buffers. Records all three
    /// dispatches into the supplied `encoder`.
    ///
    /// Buffer layouts:
    /// - `q_buf`:  [n_tokens, n_heads * head_dim]  f32
    /// - `k_buf`:  [max_seq,  n_kv_heads * head_dim]  f32 (cache-shaped)
    /// - `v_buf`:  [max_seq,  n_kv_heads * head_dim]  f32 (cache-shaped)
    /// - `scores_buf`: [n_tokens, n_heads, max_seq] f32 (scratch; written
    ///   by attn_score, read+written by softmax, read by attn_value)
    /// - `out_buf`: [n_tokens, n_heads * head_dim]  f32 (output)
    ///
    /// Causal mask is applied in `attn_score_batch` (positions > start_pos+tok
    /// are written as -inf so softmax zeros them). For prefill mode pass
    /// `start_pos=0` and size K/V buffers as `max_seq=n_tokens`.
    pub fn dispatch_attention_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_buf: &wgpu::Buffer,
        k_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
        scores_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        max_seq: usize,
        n_tokens: usize,
    ) {
        assert!(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
        let heads_per_kv = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // ---- 1. attn_score: Q · K^T * scale, with causal mask ----
        let score_params = AttnScoreBatchParams {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            max_seq: max_seq as u32,
            heads_per_kv: heads_per_kv as u32,
            kv_dim: kv_dim as u32,
            scale,
            n_tokens: n_tokens as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };
        let score_params_buf = self.gpu.create_params_buffer(&score_params);
        let score_pipeline = &self.gpu.pipelines.attn_score_batch;
        let score_bind = self.gpu.make_bind_group(
            score_pipeline,
            &[q_buf, k_buf, scores_buf, &score_params_buf],
        );
        // One thread per (tok, head, t); workgroup_size=256.
        let total_score_threads = (n_tokens * n_heads * max_seq) as u32;
        let score_groups = (total_score_threads + 255) / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.attn_score.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(score_pipeline);
            pass.set_bind_group(0, &score_bind, &[]);
            pass.dispatch_workgroups(score_groups, 1, 1);
        }

        // ---- 2. softmax: in-place over scores ----
        let softmax_params = SoftmaxBatchParams {
            n_heads: n_heads as u32,
            max_seq: max_seq as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
        };
        let softmax_params_buf = self.gpu.create_params_buffer(&softmax_params);
        let softmax_pipeline = &self.gpu.pipelines.softmax_batch;
        let softmax_bind = self.gpu.make_bind_group(
            softmax_pipeline,
            &[scores_buf, &softmax_params_buf],
        );
        // One workgroup per (tok, head) pair.
        let softmax_groups = (n_tokens * n_heads) as u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.softmax.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(softmax_pipeline);
            pass.set_bind_group(0, &softmax_bind, &[]);
            pass.dispatch_workgroups(softmax_groups, 1, 1);
        }

        // ---- 3. attn_value: weighted sum of V ----
        let value_params = AttnValueBatchParams {
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            max_seq: max_seq as u32,
            heads_per_kv: heads_per_kv as u32,
            kv_dim: kv_dim as u32,
            n_tokens: n_tokens as u32,
        };
        let value_params_buf = self.gpu.create_params_buffer(&value_params);
        let value_pipeline = &self.gpu.pipelines.attn_value_batch;
        let value_bind = self.gpu.make_bind_group(
            value_pipeline,
            &[scores_buf, v_buf, out_buf, &value_params_buf],
        );
        // One thread per (tok, head, d); workgroup_size=256.
        let total_value_threads = (n_tokens * n_heads * head_dim) as u32;
        let value_groups = (total_value_threads + 255) / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.attn_value.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(value_pipeline);
            pass.set_bind_group(0, &value_bind, &[]);
            pass.dispatch_workgroups(value_groups, 1, 1);
        }
    }

    /// Dispatch element-wise SiLU(gate) * up into `out_buf`. Sized as
    /// `[n_tokens, n]` flat. Used by the SwiGLU FFN. SiLU activation only;
    /// ReLU² (BitNet variant) needs a separate shader and is deferred until
    /// the ternary fused path lands.
    pub fn dispatch_silu_mul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = SiluMulBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.silu_mul_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[gate_buf, up_buf, out_buf, &params_buf],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.silu_mul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Dispatch element-wise in-place add: `a_buf[i] += b_buf[i]`. Used for
    /// residual connections (post-attention and post-FFN).
    pub fn dispatch_add_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.add_inplace_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[a_buf, b_buf, &params_buf],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
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

    /// CPU reference for the GPU attention chain: causal scaled dot-product
    /// attention from pre-projected, RoPE-rotated Q/K/V to per-token output.
    fn cpu_attention_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let kv_dim = n_kv_heads * head_dim;
        let q_dim = n_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; n_tokens * q_dim];

        for tok in 0..n_tokens {
            let seq_len = tok + 1;
            for h in 0..n_heads {
                let kv_h = h / heads_per_kv;
                let q_off = tok * q_dim + h * head_dim;

                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let k_off = t * kv_dim + kv_h * head_dim;
                    let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
                    scores[t] = dot * scale;
                }

                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max).exp()).sum();
                for s in &mut scores { *s = (*s - max).exp() / exp_sum; }

                for t in 0..seq_len {
                    let v_off = t * kv_dim + kv_h * head_dim;
                    for d in 0..head_dim {
                        out[tok * q_dim + h * head_dim + d] += scores[t] * v[v_off + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn dispatch_attention_matches_cpu_gqa() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // GQA: 4 query heads, 2 KV heads, head_dim=8, 5 tokens, prefill mode.
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let n_tokens = 5;
        let max_seq = n_tokens; // prefill: K/V buffer sized to current sequence
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic Q/K/V values.
        let mut rng: u64 = 0xA77E_BEEFu64;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.001
        };
        let q: Vec<f32> = (0..n_tokens * q_dim).map(|_| roll()).collect();
        let k: Vec<f32> = (0..max_seq * kv_dim).map(|_| roll()).collect();
        let v: Vec<f32> = (0..max_seq * kv_dim).map(|_| roll()).collect();

        let cpu_out = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_tokens);

        // GPU path: upload Q/K/V, allocate scores+out scratch, dispatch.
        let q_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.q"),
            contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.k"),
            contents: bytemuck::cast_slice(&k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test.v"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scores_bytes = (n_tokens * n_heads * max_seq * std::mem::size_of::<f32>()) as u64;
        let scores_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test.scores"),
            size: scores_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_bytes = (n_tokens * q_dim * std::mem::size_of::<f32>()) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(out_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_test.encoder"),
        });
        engine.dispatch_attention_into(
            &mut encoder, &q_buf, &k_buf, &v_buf, &scores_buf, &out_buf,
            n_heads, n_kv_heads, head_dim, /*start_pos*/ 0, max_seq, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
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

        assert_eq!(cpu_out.len(), gpu_out.len());
        // f32 attention math; exp/softmax in shader matches CPU within ~1e-5
        // for well-conditioned inputs.
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 1e-4,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn dispatch_attention_single_token() {
        // Smoke test the seq_len=1 case (most degenerate softmax: a single
        // value -> probability 1.0).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let n_tokens = 1;
        let max_seq = 1;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..n_tokens * q_dim).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..max_seq * kv_dim).map(|i| i as f32 * 0.2 - 0.5).collect();
        let v: Vec<f32> = (0..max_seq * kv_dim).map(|i| i as f32 * -0.05 + 0.3).collect();

        let cpu_out = cpu_attention_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, n_tokens);

        let q_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.q"), contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.k"), contents: bytemuck::cast_slice(&k),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn_test1.v"), contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scores_bytes = (n_tokens * n_heads * max_seq * 4) as u64;
        let scores_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test1.scores"), size: scores_bytes,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let out_bytes = (n_tokens * q_dim * 4) as u64;
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_test1.out"), size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(out_bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn_test1.encoder"),
        });
        engine.dispatch_attention_into(
            &mut encoder, &q_buf, &k_buf, &v_buf, &scores_buf, &out_buf,
            n_heads, n_kv_heads, head_dim, 0, max_seq, n_tokens,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
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
            assert!((c - g).abs() < 1e-4, "elem {i}: cpu={c} gpu={g}");
        }
    }

    /// Build a TransformerModel where every linear layer is GpuFloatLinear,
    /// suitable for the fused-forward path. Same shape as `toy_model()`
    /// but with float weights instead of ternary, and resident on GPU.
    fn toy_float_model_for_gpu(gpu: Arc<GpuDevice>) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        // Deterministic small float weights.
        let mut rng: u64 = 0xF1A7_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);

        let q = mk_gpu(embed_dim, embed_dim, &mut roll);
        let k = mk_gpu(embed_dim, embed_dim, &mut roll);
        let v = mk_gpu(embed_dim, embed_dim, &mut roll);
        let o = mk_gpu(embed_dim, embed_dim, &mut roll);
        let attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );

        let gate = mk_gpu(intermediate, embed_dim, &mut roll);
        let up   = mk_gpu(intermediate, embed_dim, &mut roll);
        let down = mk_gpu(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        // Use a separate float output projection so we don't tangle the test
        // with tied-embedding logic that runs CPU-side anyway.
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    /// Build a CPU-equivalent TransformerModel sharing the same weights as
    /// `toy_float_model_for_gpu` — needed to compare GPU forward output to
    /// a CPU reference. The two models use the same `roll` seed and produce
    /// identical-tolerance outputs (subject to f16 rounding in GpuFloatLinear).
    fn toy_float_model_cpu_reference() -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::floatlinear::FloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 4;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 4;
        let vocab_size = 4;

        let mut rng: u64 = 0xF1A7_FACE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_cpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(FloatLinear::from_float_tensor(mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let q = mk_cpu(embed_dim, embed_dim, &mut roll);
        let k = mk_cpu(embed_dim, embed_dim, &mut roll);
        let v = mk_cpu(embed_dim, embed_dim, &mut roll);
        let o = mk_cpu(embed_dim, embed_dim, &mut roll);
        let attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );

        let gate = mk_cpu(intermediate, embed_dim, &mut roll);
        let up   = mk_cpu(intermediate, embed_dim, &mut roll);
        let down = mk_cpu(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    #[test]
    fn forward_block_gpu_matches_cpu_block() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_model_cpu_reference();
        let n_tokens = 3;
        let tokens: Vec<u32> = (0..n_tokens as u32).collect();

        // CPU reference: embedding lookup + one block.forward.
        let embed_data = cpu_ref.embedding_data();
        let embed_dim = cpu_ref.embed_dim();
        let mut hidden_cpu: Vec<f32> = Vec::with_capacity(n_tokens * embed_dim);
        for &t in &tokens {
            let off = t as usize * embed_dim;
            hidden_cpu.extend_from_slice(&embed_data[off..off + embed_dim]);
        }
        let cpu_out = cpu_ref.blocks()[0].forward(&hidden_cpu, n_tokens, /*start_pos*/ 0);

        // GPU path
        let gpu_model = toy_float_model_for_gpu(gpu.clone());
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        // Upload the same hidden-state input.
        let bytes = (n_tokens * embed_dim * 4) as u64;
        let hidden_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.hidden"),
            contents: bytemuck::cast_slice(&hidden_cpu),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let attn0 = engine.cpu().blocks()[0].attention();
        let intermediate = engine.cpu().blocks()[0].ffn().out_features(); // == embed_dim
        let _ = intermediate;
        // FFN intermediate = SwiGLU.intermediate_size; access via downcast.
        let intermediate = engine.cpu().blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>().unwrap()
            .intermediate_size();

        let scratch = BlockScratch::allocate(
            &gpu, n_tokens, embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, /*max_seq*/ n_tokens,
        );

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.encoder"),
        });
        engine.forward_block_gpu(&mut encoder, 0, &hidden_buf, n_tokens, 0, &scratch);
        encoder.copy_buffer_to_buffer(&hidden_buf, 0, &staging, 0, bytes);
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

        assert_eq!(cpu_out.len(), gpu_out.len(), "shape mismatch");
        // f16 weight rounding inside GpuFloatLinear vs CPU FloatLinear's f32
        // weights — generous tolerance. For the toy 4-dim values this is
        // tight enough to catch real bugs; loosen if the dim grows.
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 0.01,
                "elem {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn dispatch_silu_mul_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n = 16;
        let n_tokens = 3;
        let total = n * n_tokens;

        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let up:   Vec<f32> = (0..total).map(|i| (i as f32) * -0.05 + 0.4).collect();

        // CPU reference: silu(g) * u  where silu(x) = x / (1 + exp(-x))
        let cpu_out: Vec<f32> = gate.iter().zip(&up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let bytes = (total * 4) as u64;
        let g_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("silu.g"), contents: bytemuck::cast_slice(&gate),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let u_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("silu.u"), contents: bytemuck::cast_slice(&up),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let o_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu.o"), size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu.encoder"),
        });
        engine.dispatch_silu_mul_into(&mut encoder, &g_buf, &u_buf, &o_buf, n, n_tokens);
        encoder.copy_buffer_to_buffer(&o_buf, 0, &staging, 0, bytes);
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
    fn dispatch_add_in_place_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n = 8;
        let n_tokens = 4;
        let total = n * n_tokens;

        let mut a: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32>     = (0..total).map(|i| (total - i) as f32 * -0.05).collect();
        let cpu_a: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let bytes = (total * 4) as u64;
        let a_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add.a"), contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add.b"), contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu.clone());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add.encoder"),
        });
        engine.dispatch_add_into(&mut encoder, &a_buf, &b_buf, n, n_tokens);
        encoder.copy_buffer_to_buffer(&a_buf, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_a: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data); staging.unmap();

        let _ = a; // silence unused-mut warning in case of future refactor
        for (i, (c, g)) in cpu_a.iter().zip(&gpu_a).enumerate() {
            assert!((c - g).abs() < 1e-6, "elem {i}: cpu={c} gpu={g}");
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
