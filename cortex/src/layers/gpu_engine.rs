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

/// Params struct for the kv_write_batch shader. Four u32s.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteBatchParams {
    kv_dim: u32,
    start_pos: u32,
    n_tokens: u32,
    _pad: u32,
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
    /// Optional Q/K/V biases (Qwen2 family). None for most LLaMA-style models.
    q_bias_buf: Option<wgpu::Buffer>,
    k_bias_buf: Option<wgpu::Buffer>,
    v_bias_buf: Option<wgpu::Buffer>,
    /// BitNet b1.58 attention sub-norm: RMSNorm applied to the attention
    /// output BEFORE `o_proj`. None for non-BitNet models. `attn.o_sub_norm()`
    /// supplies the source weight + eps when present.
    o_sub_norm_weight_buf: Option<wgpu::Buffer>,
    o_sub_norm_eps: f32,
    /// BitNet b1.58 FFN sub-norm: RMSNorm applied to `silu(gate) * up`
    /// BEFORE `down_proj`. None for non-BitNet SwiGLU.
    ffn_sub_norm_weight_buf: Option<wgpu::Buffer>,
    ffn_sub_norm_eps: f32,
}

/// Per-block scratch buffers reused across all dispatches inside a single
/// `forward_block_gpu` call. Caller allocates once per forward pass and
/// reuses across blocks (since dimensions are constant).
/// Output of `forward_full_gpu_with_hidden_capture`. Read-side hook
/// surface for cortex shims (`project_cortex_v1_shim_api.md`):
///
/// - `per_layer_hidden[i]` corresponds to `attachment.layer = "entrance:N+1"`
///   for the i-th element of `capture_layers` — the hidden state at the
///   END of block N (= start of block N+1). Layout: `[n_tokens, embed_dim]`
///   row-major f32.
/// - `final_post_norm_hidden` is what `attachment.layer = "final"` shims
///   read — the LM head's input. Same `[n_tokens, embed_dim]` shape.
///   Pool downstream per the manifest's `pooling` field.
pub struct HiddenCaptures {
    pub per_layer_hidden: Vec<Vec<f32>>,
    pub final_post_norm_hidden: Vec<f32>,
    pub n_tokens: usize,
    pub embed_dim: usize,
}

impl HiddenCaptures {
    /// Pull the last token's slice from the final post-norm hidden state —
    /// the most common pooling for gate / steer shims.
    pub fn final_last_token(&self) -> &[f32] {
        let off = (self.n_tokens - 1) * self.embed_dim;
        &self.final_post_norm_hidden[off..off + self.embed_dim]
    }

    /// Pull the last token's slice from the i-th captured layer's
    /// post-block hidden state.
    pub fn layer_last_token(&self, capture_index: usize) -> &[f32] {
        let off = (self.n_tokens - 1) * self.embed_dim;
        &self.per_layer_hidden[capture_index][off..off + self.embed_dim]
    }
}

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
    /// Scratch for the bitnet batch matmul path (#bn-5). Sized for the
    /// widest input dim across Q/K/V projections (embed_dim) and FFN
    /// (intermediate, for the gate/up projection in_features = embed_dim;
    /// for the down projection in_features = intermediate). One buffer
    /// pair fits all six call sites because the bottleneck is the
    /// largest in_features = max(embed_dim, intermediate).
    pub ternary: TernaryScratch,
}

/// Scratch buffers consumed by the bitnet batch matmul path.
/// `activations_i8` holds 4 i8 values per u32 (matches the layout
/// `ternary_matmul_batch.wgsl` expects). `scales` holds one f32 per
/// token. Both sized so that any of the six linear-call sites in a
/// block fits.
pub struct TernaryScratch {
    pub activations_i8: wgpu::Buffer,
    pub scales: wgpu::Buffer,
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
        let u32_bytes = std::mem::size_of::<u32>() as u64;
        // Ternary scratch sizing: largest in_features across all linear
        // call sites in a block is max(embed_dim, intermediate). One i8
        // per element packed 4-per-u32, plus one f32 scale per token.
        let max_in = embed_dim.max(intermediate);
        let act_q_u32_count = (n_tokens * ((max_in + 3) / 4)) as u64;
        let ternary = TernaryScratch {
            activations_i8: mk(act_q_u32_count * u32_bytes, "scratch.ternary.activations_i8"),
            scales:         mk((n_tokens as u64) * f32_bytes, "scratch.ternary.scales"),
        };
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
            ternary,
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

        // Per-block norms + optional Q/K/V biases (Qwen2)
        let upload_bias = |bias: &[f32], i: usize, name: &str| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("gpu_engine.block{i}.{name}_bias")),
                contents: bytemuck::cast_slice(bias),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let blocks_gpu: Vec<GpuBlock> = cpu.blocks().iter().enumerate().map(|(i, blk)| {
            let an = blk.attn_norm();
            let fn_ = blk.ffn_norm();
            let attn = blk.attention();
            let swiglu_opt = blk.ffn().as_any()
                .downcast_ref::<crate::layers::swiglu::SwiGLU>();
            let upload_norm = |w: &[f32], label: String| -> wgpu::Buffer {
                gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&label),
                    contents: bytemuck::cast_slice(w),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let (o_sub_norm_buf, o_sub_norm_eps) = match attn.o_sub_norm() {
                Some(n) => (Some(upload_norm(n.weight(), format!("gpu_engine.block{i}.o_sub_norm.weight"))), n.eps()),
                None => (None, 0.0),
            };
            let (ffn_sub_norm_buf, ffn_sub_norm_eps) = match swiglu_opt.and_then(|s| s.sub_norm()) {
                Some(n) => (Some(upload_norm(n.weight(), format!("gpu_engine.block{i}.ffn_sub_norm.weight"))), n.eps()),
                None => (None, 0.0),
            };
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
                q_bias_buf: attn.q_bias().map(|b| upload_bias(b, i, "q")),
                k_bias_buf: attn.k_bias().map(|b| upload_bias(b, i, "k")),
                v_bias_buf: attn.v_bias().map(|b| upload_bias(b, i, "v")),
                o_sub_norm_weight_buf: o_sub_norm_buf,
                o_sub_norm_eps,
                ffn_sub_norm_weight_buf: ffn_sub_norm_buf,
                ffn_sub_norm_eps,
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
    ///
    /// Opens its own compute pass. For callers that want to batch multiple
    /// dispatches into a single pass (saves ~0.8ms of Vulkan
    /// pipeline-barrier overhead per dispatch), use
    /// `dispatch_rmsnorm_in_pass`.
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rmsnorm.pass"),
            timestamp_writes: None,
        });
        self.dispatch_rmsnorm_in_pass(&mut pass, in_buf, weight_buf, out_buf, n, n_tokens, eps);
    }

    /// In-pass variant of `dispatch_rmsnorm_into`. Records the dispatch
    /// into the caller-supplied compute pass instead of opening a new one.
    /// See `dispatch_rmsnorm_into` for semantics.
    pub fn dispatch_rmsnorm_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.matmul.pass"),
            timestamp_writes: None,
        });
        self.dispatch_matmul_in_pass(&mut pass, layer, in_buf, out_buf, n_tokens);
    }

    /// In-pass variant. See `dispatch_matmul_into`. Routes to the
    /// tiled matmul shader when `n_tokens >= 8` (TILE_M), falling
    /// back to the per-output-element legacy shader for decode-sized
    /// batches where register blocking has nothing to amortize.
    pub fn dispatch_matmul_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
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
                    "GpuEngine.dispatch_matmul_in_pass: layer is not GpuFloatLinear \
                     (concrete type: {:?})",
                    layer
                )
            });

        self.dispatch_matmul_legacy_inner_in_pass(pass, float, in_buf, out_buf, n_tokens);
    }

    fn dispatch_matmul_legacy_inner_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        float: &crate::layers::gpu_floatlinear::GpuFloatLinear,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
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

        let rows = float.out_features();
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }

    /// Per-token absmax quantization of f32 hidden → packed i8 activations
    /// + per-token f32 scales. One workgroup per token. Used by the
    /// bitnet batch path (#6c) to feed `dispatch_ternary_matmul_batch_into`.
    ///
    /// `act_q_buf` must be sized at least `n_tokens * ceil(cols / 4)` u32s.
    /// `act_scales_buf` must be sized at least `n_tokens` f32s.
    pub fn dispatch_quantize_absmax_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_f32_buf: &wgpu::Buffer,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        cols: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.quantize_absmax_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_quantize_absmax_batch_in_pass(
            &mut pass, input_f32_buf, act_q_buf, act_scales_buf, cols, n_tokens,
        );
    }

    /// In-pass variant. See `dispatch_quantize_absmax_batch_into`.
    pub fn dispatch_quantize_absmax_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        input_f32_buf: &wgpu::Buffer,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        cols: usize,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct QuantParams { cols: u32, n_tokens: u32 }
        let params = QuantParams { cols: cols as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.quantize_absmax_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[input_f32_buf, act_q_buf, act_scales_buf, &params_buf],
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_tokens as u32, 1, 1);
    }

    /// Batched ternary matmul. Multi-token analog of the single-token
    /// `ternary_matvec`. Reads resident 2-bit packed weights from
    /// `GpuBitLinear`, i8 packed activations + per-token scales (output
    /// of `dispatch_quantize_absmax_batch_into`), writes f32 output with
    /// the per-token scale and the layer's weight_scale applied inline.
    pub fn dispatch_ternary_matmul_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &crate::layers::gpu_bitlinear::GpuBitLinear,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        out_f32_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.ternary_matmul_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_ternary_matmul_batch_in_pass(
            &mut pass, layer, act_q_buf, act_scales_buf, out_f32_buf, n_tokens,
        );
    }

    /// In-pass variant. See `dispatch_ternary_matmul_batch_into`.
    pub fn dispatch_ternary_matmul_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &crate::layers::gpu_bitlinear::GpuBitLinear,
        act_q_buf: &wgpu::Buffer,
        act_scales_buf: &wgpu::Buffer,
        out_f32_buf: &wgpu::Buffer,
        n_tokens: usize,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TmmParams { rows: u32, cols: u32, n_tokens: u32, weight_scale_bits: u32 }
        let params = TmmParams {
            rows: layer.out_features() as u32,
            cols: layer.in_features() as u32,
            n_tokens: n_tokens as u32,
            weight_scale_bits: layer.weight_scale().to_bits(),
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.ternary_matmul_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[layer.weight_buffer(), act_q_buf, act_scales_buf, out_f32_buf, &params_buf],
        );

        let rows = layer.out_features();
        let dx = (rows.min(65535)) as u32;
        let dy = ((rows + 65534) / 65535) as u32;
        let dz = n_tokens as u32;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }

    /// Unified linear-layer dispatcher used by `forward_block_gpu_inner`.
    /// Downcasts to `GpuFloatLinear` (existing float matmul path) or
    /// `GpuBitLinear` (new bitnet batch path: quantize then ternary
    /// matmul). Panics with the layer's concrete type if neither
    /// matches — same failure mode as `dispatch_matmul_into` had, just
    /// with both cases covered.
    pub fn dispatch_linear_batch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &dyn LinearLayer,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_tokens: usize,
        ternary_scratch: &TernaryScratch,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.linear_batch.pass"),
            timestamp_writes: None,
        });
        self.dispatch_linear_batch_in_pass(&mut pass, layer, in_buf, out_buf, n_tokens, ternary_scratch);
    }

    /// In-pass variant. See `dispatch_linear_batch_into`.
    pub fn dispatch_linear_batch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer: &dyn LinearLayer,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_tokens: usize,
        ternary_scratch: &TernaryScratch,
    ) {
        let any = layer.as_any();
        if any.downcast_ref::<crate::layers::gpu_floatlinear::GpuFloatLinear>().is_some() {
            self.dispatch_matmul_in_pass(pass, layer, in_buf, out_buf, n_tokens);
            return;
        }
        if let Some(bit) = any.downcast_ref::<crate::layers::gpu_bitlinear::GpuBitLinear>() {
            self.dispatch_quantize_absmax_batch_in_pass(
                pass, in_buf, &ternary_scratch.activations_i8, &ternary_scratch.scales,
                bit.in_features(), n_tokens,
            );
            self.dispatch_ternary_matmul_batch_in_pass(
                pass, bit, &ternary_scratch.activations_i8, &ternary_scratch.scales,
                out_buf, n_tokens,
            );
            return;
        }
        panic!(
            "GpuEngine.dispatch_linear_batch_in_pass: layer is neither GpuFloatLinear \
             nor GpuBitLinear (concrete type: {:?})",
            layer
        );
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

    /// Like `forward_full_gpu_traced` but skips the output projection
    /// entirely (no logits computed). Retrieval doesn't need logits; the
    /// per-token GpuFloatLinear vocab projection across 2000+ tokens is the
    /// expensive part and was hanging the server.
    pub fn forward_traced_scores_only(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let (_logits, scores) = self.forward_traced_inner(tokens, start_pos, capture_layers, false);
        scores
    }

    /// Forward pass that captures pre-softmax attention scores for the
    /// requested layers. Used by retrieval (memex) — the per-position
    /// attention weight aggregation in `cortex-cloud`'s `/v1/retrieve`
    /// handler reads these scores.
    ///
    /// `capture_layers`: indices of blocks whose pre-softmax attention
    /// scores should be captured. Each capture is sized
    /// `[n_tokens, n_heads, n_tokens]` f32 = O(n_tokens² × n_heads × 4)
    /// bytes per layer. For Qwen 3B at 2300 tokens × 16 heads × 4 bytes,
    /// that's ~340 MB per layer, so callers should keep the set small.
    /// memex architecture suggests "last few layers" carry the retrieval
    /// signal; default in cortex-cloud is the last 4.
    ///
    /// Returns `(logits, per_layer_scores)` where `per_layer_scores[i]`
    /// is the captured pre-softmax tensor for `capture_layers[i]`,
    /// flat as `[n_tokens, n_heads, n_tokens]`.
    pub fn forward_full_gpu_traced(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.forward_traced_inner(tokens, start_pos, capture_layers, true)
    }

    fn forward_traced_inner(
        &self,
        tokens: &[u32],
        start_pos: usize,
        capture_layers: &[usize],
        compute_logits: bool,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");
        let n_layers = self.cpu.n_layers();
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        // ---- 1. Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // ---- Allocate buffers ----
        let bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_full_traced.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_full_traced.normed"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed_staging = self.gpu.create_staging_buffer(bytes);

        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );

        // Per-captured-layer score storage buffers. Same shape as scratch.scores
        // but persistent across the whole forward.
        let scores_bytes = (n_tokens * n_heads * n_tokens * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_full_traced.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(scores_bytes))
            .collect();

        // Build a layer_idx -> capture_buf lookup for O(1) access in the loop.
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- 2-3. All blocks + final_norm in one encoder ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_full_traced.encoder"),
        });
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_inner(&mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch, capture, None, None, None);
        }
        self.dispatch_rmsnorm_into(
            &mut encoder, &hidden_buf, &self.final_norm_weight_buf, &normed_buf,
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &normed_staging, 0, bytes);
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, scores_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Issue all map_async calls together, then poll once. Sequential
        // poll(Wait) per buffer was hanging — possibly because the wgpu
        // device only fires callbacks inside poll, and re-polling after
        // a buffer's already mapped doesn't re-fire pending callbacks for
        // others in some cases. Single poll drives all of them at once.
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = Vec::with_capacity(1 + capture_stagings.len());
        let normed_slice = normed_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        normed_slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        receivers.push(rx);
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();

        self.gpu.device.poll(wgpu::Maintain::Wait);
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        // ---- Decode the readbacks ----
        let normed: Vec<f32> = {
            let data = normed_slice.get_mapped_range();
            let v: Vec<f32> = data[..bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            normed_staging.unmap();
            v
        };
        let per_layer_scores: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..scores_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        // ---- 4. Output projection (CPU; vocab matmul deferred). Skipped
        //         when caller doesn't need logits (retrieve path) — that
        //         saves 2000+ per-token GpuFloatLinear calls and the
        //         staging-buffer churn that comes with them. ----
        let logits = if compute_logits {
            self.cpu.finalize_logits(&normed, n_tokens)
        } else {
            Vec::new()
        };

        (logits, per_layer_scores)
    }

    /// Hidden-state capture forward pass — non-cached. Runs the model
    /// forward over `tokens` and returns:
    /// - `per_layer_hidden[i]` — the post-FFN-residual hidden state at
    ///   the END of block `capture_layers[i]`. Shape:
    ///   `n_tokens * embed_dim` f32. Same layout as `entrance:(N+1)`
    ///   inputs in shim manifests.
    /// - `final_post_norm_hidden` — the final RMSNorm output (input to
    ///   the LM head). Shape `n_tokens * embed_dim` f32. This is what
    ///   `attachment.layer = "final"` gate / steer shims read; pool
    ///   downstream (last_token / mean / etc.) per the manifest.
    ///
    /// Read-side hook only — no injection / steering. Those are #5/#6.
    pub fn forward_full_gpu_with_hidden_capture(
        &self,
        tokens: &[u32],
        capture_layers: &[usize],
    ) -> HiddenCaptures {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0);
        let n_layers = self.cpu.n_layers();
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        // Embedding lookup (CPU).
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let hidden_bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_hidden_capture.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_hidden_capture.normed"),
            size: hidden_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed_staging = self.gpu.create_staging_buffer(hidden_bytes);

        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_hidden_capture requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );

        // Per-captured-layer hidden buffers (post-FFN-residual). Same shape
        // as hidden_buf: [n_tokens, embed_dim] f32 flat.
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_hidden_capture.layer{l}")),
                size: hidden_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(hidden_bytes))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_hidden_capture.encoder"),
        });
        for i in 0..n_layers {
            let post_capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, /*start_pos*/ 0, &scratch,
                /*pre_softmax_capture*/ None, /*kv_cache_target*/ None, post_capture,
                /*pre_block_hidden_inject*/ None,
            );
        }
        // Final RMSNorm — gives the final post-norm hidden state shims read.
        self.dispatch_rmsnorm_into(
            &mut encoder, &hidden_buf, &self.final_norm_weight_buf, &normed_buf,
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &normed_staging, 0, hidden_bytes);
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, hidden_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Batched readback (single poll for all stagings).
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(1 + capture_stagings.len());
        let normed_slice = normed_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        normed_slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        receivers.push(rx);
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();

        self.gpu.device.poll(wgpu::Maintain::Wait);
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        let final_post_norm_hidden: Vec<f32> = {
            let data = normed_slice.get_mapped_range();
            let v: Vec<f32> = data[..hidden_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            normed_staging.unmap();
            v
        };
        let per_layer_hidden: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..hidden_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        HiddenCaptures {
            per_layer_hidden,
            final_post_norm_hidden,
            n_tokens,
            embed_dim: self.embed_dim,
        }
    }

    /// Cached + traced forward: process new `query_tokens` against a
    /// pre-populated `cache`, capturing pre-softmax attention scores for
    /// the requested layers. **Does not advance the cache cursor** — the
    /// cache stays at its original `seq_len`, so this is safe to call
    /// repeatedly with different queries against the same shard cache.
    ///
    /// The query tokens' K/V do get written into the cache buffers at
    /// offset `cache.seq_len()` during the dispatch, but those positions
    /// are logically unallocated (`cache.seq_len()` is the end of the
    /// real prefix), so the next call simply overwrites them. Subsequent
    /// `forward_full_gpu_with_cache` calls would also overwrite that
    /// region — the contract is "K/V at offsets >= seq_len are scratch."
    ///
    /// Returns per-layer pre-softmax score tensors flat as
    /// `[n_query_tokens, n_heads, max_seq]` where `max_seq = cache.seq_len()
    /// + n_query_tokens`. The score from query position q to corpus
    /// position k is `scores[q * n_heads * max_seq + h * max_seq + k]`,
    /// which is what cortex-cloud's retrieve handler aggregates over.
    pub fn forward_full_gpu_with_cache_traced(
        &self,
        query_tokens: &[u32],
        cache: &crate::layers::gpu_kv_cache::GpuKvCache,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let n_tokens = query_tokens.len();
        assert!(n_tokens > 0, "must have at least one query token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(
            attn_max_seq <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in query_tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_traced_with_cache.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );

        // Per-captured-layer score storage. Shape: [n_tokens, n_heads, max_seq]
        // where max_seq = start_pos + n_tokens (the full attention window).
        let scores_bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_traced_with_cache.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(scores_bytes))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- All blocks in one encoder, with cache target + optional capture ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_traced_with_cache.encoder"),
        });
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            let target = (cache.k_layer(i), cache.v_layer(i));
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                capture, Some(target), None, None,
            );
        }
        // Skip final_norm + output projection — retrieval doesn't need
        // logits, only the captured scores.
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, scores_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Batched readback (single poll for all stagings).
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = Vec::with_capacity(capture_stagings.len());
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();
        self.gpu.device.poll(wgpu::Maintain::Wait);
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        let per_layer_scores: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..scores_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        // NOTE: we deliberately do NOT call cache.advance() — see method docs.
        per_layer_scores
    }

    /// Polar variant of `forward_full_gpu_with_cache_traced`. Runs the
    /// transformer forward over `query_tokens` against a resident
    /// PolarQuant-compressed KV cache, capturing per-layer pre-softmax
    /// attention scores from `capture_layers`. The query's K/V are
    /// compressed and written into the polar cache at offset
    /// `polar_cache.seq_len()` for self-attention; the cache cursor is
    /// **not** advanced (mirrors the f32 traced forward — same cache can
    /// serve repeated queries).
    ///
    /// All blocks run; non-captured layers are still executed because the
    /// hidden state must propagate through every layer. `capture_layers`
    /// only controls which layers' pre-softmax attention scores are
    /// returned.
    pub fn forward_full_gpu_polar_traced(
        &self,
        query_tokens: &[u32],
        polar_cache: &crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        capture_layers: &[usize],
    ) -> Vec<Vec<f32>> {
        let n_tokens = query_tokens.len();
        assert!(n_tokens > 0, "must have at least one query token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, polar_cache.n_layers(), "polar cache layer count mismatch");
        for &l in capture_layers {
            assert!(l < n_layers, "capture layer {l} out of range (n_layers={n_layers})");
        }

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(polar_cache.n_kv_heads(), attn0.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(polar_cache.head_dim(), attn0.head_dim(),
            "polar cache head_dim mismatch");

        let start_pos = polar_cache.seq_len();
        let attn_max_seq = start_pos + n_tokens;
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: {} + {} > {}",
            start_pos, n_tokens, polar_cache.max_seq_len());

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in query_tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_polar_traced.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_polar_traced requires SwiGLU FFN"))
            .intermediate_size();
        let n_heads = attn0.n_heads();
        let head_dim = attn0.head_dim();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            n_heads, attn0.n_kv_heads(), head_dim,
            intermediate, attn_max_seq,
        );

        // Polar-only scratch: reused as both rq (post-rotate_q) and
        // weighted_rotated_V (pre-derotate) inside each block. One
        // allocation per trace call, not per block.
        let rotated_bytes = (n_tokens * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let rotated_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_polar_traced.rotated"),
            size: rotated_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Per-captured-layer scores. Same shape as the f32 path:
        // [n_tokens, n_heads, attn_max_seq].
        let scores_bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
        let capture_bufs: Vec<wgpu::Buffer> = capture_layers.iter().map(|&l| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("forward_polar_traced.scores.layer{l}")),
                size: scores_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();
        let capture_stagings: Vec<wgpu::Buffer> = (0..capture_layers.len())
            .map(|_| self.gpu.create_staging_buffer(scores_bytes))
            .collect();
        let capture_lookup: std::collections::HashMap<usize, &wgpu::Buffer> =
            capture_layers.iter().zip(capture_bufs.iter())
                .map(|(&l, buf)| (l, buf))
                .collect();

        // ---- All blocks in one encoder ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_polar_traced.encoder"),
        });
        for i in 0..n_layers {
            let capture = capture_lookup.get(&i).copied();
            self.forward_block_gpu_polar_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                &rotated_buf, polar_cache, capture, None, None,
            );
        }
        // Skip final_norm + output projection — retrieval doesn't need logits.
        for (cap_buf, stg_buf) in capture_bufs.iter().zip(capture_stagings.iter()) {
            encoder.copy_buffer_to_buffer(cap_buf, 0, stg_buf, 0, scores_bytes);
        }
        self.gpu.queue.submit(Some(encoder.finish()));

        // Batched readback (single poll for all stagings).
        use std::sync::mpsc;
        let mut receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = Vec::with_capacity(capture_stagings.len());
        let capture_slices: Vec<wgpu::BufferSlice> = capture_stagings.iter().map(|stg| {
            let slice = stg.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            receivers.push(rx);
            slice
        }).collect();
        self.gpu.device.poll(wgpu::Maintain::Wait);
        for rx in &receivers {
            rx.recv().expect("readback channel closed").expect("buffer map failed");
        }

        let per_layer_scores: Vec<Vec<f32>> = capture_slices.iter().zip(capture_stagings.iter()).map(|(slice, stg)| {
            let data = slice.get_mapped_range();
            let v: Vec<f32> = data[..scores_bytes as usize].chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            stg.unmap();
            v
        }).collect();

        per_layer_scores
    }

    /// Forward pass that writes new K/V into the supplied `cache` and reads
    /// the full prefix from the cache during attention. Both prefill (cache
    /// initially empty, `cache.seq_len() == 0`) and decode (cache populated,
    /// new tokens at positions `[cache.seq_len(), cache.seq_len() + n_tokens)`)
    /// are handled by the same code path — `cache.seq_len()` becomes the
    /// RoPE/attention `start_pos`, and the cache buffers are sized for the
    /// full prefix.
    ///
    /// On success the cache's write cursor is advanced by `n_tokens`.
    /// Returns logits over vocab for each new token (same shape as
    /// `forward_full_gpu`).
    pub fn forward_full_gpu_with_cache(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        let normed = self.forward_full_gpu_with_cache_inject_returning_hidden(tokens, cache, &[]);
        self.cpu.finalize_logits(&normed, n_tokens)
    }

    /// `forward_full_gpu_with_cache` plus injection-phase deltas.
    /// `inject_deltas` is either empty (no injection) or has length
    /// `n_layers()` with `Some(buf)` for each layer that carries a
    /// summed `[embed_dim]` f32 hidden_delta. Each present buffer is
    /// broadcast-added into hidden at that block's entrance.
    pub fn forward_full_gpu_with_cache_inject(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        let normed = self.forward_full_gpu_with_cache_inject_returning_hidden(
            tokens, cache, inject_deltas,
        );
        self.cpu.finalize_logits(&normed, n_tokens)
    }

    /// Same forward as `forward_full_gpu_with_cache` but returns the final
    /// post-norm hidden state ([n_tokens * embed_dim] row-major f32) WITHOUT
    /// running the LM-head projection. The hidden buffer is what
    /// `attachment.layer = "final"` shims read; steer-phase shims (#6b)
    /// mutate this slice and the caller re-projects via
    /// `cpu().finalize_logits(modified, n_tokens)` to get steered logits.
    ///
    /// Skipping the projection here matters when the caller is going to
    /// re-project anyway — for Qwen 3B the projection is the largest CPU
    /// cost in the per-token loop (~50–200ms), so doing it twice would
    /// halve decode throughput. On success the cache's write cursor is
    /// advanced by `n_tokens`.
    pub fn forward_full_gpu_with_cache_returning_hidden(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) -> Vec<f32> {
        self.forward_full_gpu_with_cache_inject_returning_hidden(tokens, cache, &[])
    }

    /// Fire-and-forget forward: runs all blocks, writes K/V into the
    /// cache, advances the cache cursor — but does NOT compute the final
    /// RMSNorm, does NOT copy to a staging buffer, and does NOT wait for
    /// the GPU. Callers that discard the hidden state entirely (e.g.
    /// `cache_append` building a shard) should use this to skip the
    /// `device.poll(Maintain::Wait)` round-trip, which empirical
    /// profiling shows costs ~300 ms per call on a 4080 Laptop /
    /// Vulkan independent of token count.
    ///
    /// **Ordering / safety**: wgpu's queue guarantees in-submission-order
    /// execution. The cache cursor is a CPU counter advanced after submit;
    /// any subsequent call that submits more work to the same cache will
    /// observe the previous K/V writes because the GPU executes them in
    /// order. If the GPU work fails (e.g. device lost), the failure
    /// surfaces on the next call that DOES wait — typical late-binding
    /// error reporting. For cache_append's HTTP handler this is
    /// acceptable: success means "queued and cursor advanced," not
    /// "GPU work completed."
    pub fn forward_full_gpu_with_cache_advance_only(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
    ) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        assert!(
            start_pos + n_tokens <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        let t_start = std::time::Instant::now();

        // Embedding lookup (CPU) — same as full forward.
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_advance_only.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache_advance_only requires SwiGLU FFN"))
            .intermediate_size();

        let attn_max_seq = start_pos + n_tokens;
        let t_alloc_start = std::time::Instant::now();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );
        let t_alloc = t_alloc_start.elapsed();

        let t_record_start = std::time::Instant::now();
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_advance_only.encoder"),
        });
        let skip_block_forward = std::env::var("CORTEX_SKIP_BLOCK_FORWARD").as_deref() == Ok("1");
        if !skip_block_forward {
            for i in 0..n_layers {
                let target = (cache.k_layer(i), cache.v_layer(i));
                self.forward_block_gpu_inner(
                    &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                    None, Some(target), None, None,
                );
            }
        }
        let t_record = t_record_start.elapsed();

        // No final RMSNorm, no copy_buffer_to_buffer, no readback.
        let t_submit_start = std::time::Instant::now();
        self.gpu.queue.submit(Some(encoder.finish()));
        let t_submit = t_submit_start.elapsed();

        // Force a sync so the per-call wall time actually reflects GPU work
        // (otherwise the next call's submit blocks behind ours). Gated:
        // CORTEX_SYNC_AFTER_ADVANCE=1 to enable. Default off (back-compat).
        let t_poll_start = std::time::Instant::now();
        if std::env::var("CORTEX_SYNC_AFTER_ADVANCE").as_deref() == Ok("1") {
            self.gpu.device.poll(wgpu::Maintain::Wait);
        }
        let t_poll = t_poll_start.elapsed();

        // Advance cache cursor (CPU). Subsequent submits will see ordered
        // K/V writes via the wgpu queue's in-order guarantee.
        cache.advance(n_tokens);
        let t_total = t_start.elapsed();
        tracing::info!(
            n_tokens, start_pos, attn_max_seq,
            alloc_us = t_alloc.as_micros() as u64,
            record_us = t_record.as_micros() as u64,
            submit_us = t_submit.as_micros() as u64,
            poll_us = t_poll.as_micros() as u64,
            total_us = t_total.as_micros() as u64,
            "fwd_advance_only stage timings",
        );
    }

    /// Inject-aware variant of `forward_full_gpu_with_cache_returning_hidden`.
    /// `inject_deltas` is either empty (no injection — same path as the
    /// no-inject variant) or has length `n_layers()` with `Some(buf)` for
    /// each layer that carries a [embed_dim] f32 hidden_delta. Each
    /// present buffer is broadcast-added into hidden BEFORE that block's
    /// attn_norm, every forward step (prefill + decode).
    ///
    /// The chat handler (cortex-cloud) computes injection deltas once
    /// per request (running each injection shim against the prompt's
    /// prefill hidden), uploads each summed per-layer delta via
    /// `upload_f32_to_storage`, then threads the same `inject_deltas`
    /// slice through every prefill / decode call for the request.
    pub fn forward_full_gpu_with_cache_inject_returning_hidden(
        &self,
        tokens: &[u32],
        cache: &mut crate::layers::gpu_kv_cache::GpuKvCache,
        inject_deltas: &[Option<wgpu::Buffer>],
    ) -> Vec<f32> {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        let n_layers = self.cpu.n_layers();
        assert_eq!(n_layers, cache.n_layers(), "cache layer count mismatch");

        let attn0 = self.cpu.blocks()[0].attention();
        assert_eq!(cache.n_kv_heads(), attn0.n_kv_heads(), "cache n_kv_heads mismatch");
        assert_eq!(cache.head_dim(), attn0.head_dim(), "cache head_dim mismatch");

        let start_pos = cache.seq_len();
        assert!(
            start_pos + n_tokens <= cache.max_seq_len(),
            "cache overflow: {} + {} > {}",
            start_pos, n_tokens, cache.max_seq_len(),
        );

        let t_start = std::time::Instant::now();

        // ---- Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }
        let t_embed = t_start.elapsed();

        let bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_with_cache.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_with_cache.normed"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(bytes);
        let t_io_alloc = t_start.elapsed() - t_embed;

        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu_with_cache requires SwiGLU FFN"))
            .intermediate_size();

        // Scratch sized for ATTENTION over the full prefix (max_seq =
        // start_pos + n_tokens). Scores buffer must hold the score grid.
        let attn_max_seq = start_pos + n_tokens;
        let t_pre_scratch = t_start.elapsed();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, attn_max_seq,
        );
        let t_scratch = t_start.elapsed() - t_pre_scratch;

        // ---- All blocks + final_norm in one encoder ----
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_with_cache.encoder"),
        });
        // Inject deltas: empty slice means no injection (same path as
        // before); non-empty must be exactly n_layers long with
        // Some(buf) for layers that carry a delta.
        if !inject_deltas.is_empty() {
            assert_eq!(
                inject_deltas.len(), n_layers,
                "inject_deltas must be empty or have length n_layers ({})", n_layers,
            );
        }
        let t_pre_record = t_start.elapsed();
        for i in 0..n_layers {
            let target = (cache.k_layer(i), cache.v_layer(i));
            let inject = inject_deltas.get(i).and_then(|opt| opt.as_ref());
            self.forward_block_gpu_inner(
                &mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch,
                None, Some(target), None, inject,
            );
        }
        self.dispatch_rmsnorm_into(
            &mut encoder, &hidden_buf, &self.final_norm_weight_buf, &normed_buf,
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &staging, 0, bytes);
        let t_record = t_start.elapsed() - t_pre_record;

        let t_pre_submit = t_start.elapsed();
        self.gpu.queue.submit(Some(encoder.finish()));
        let t_submit = t_start.elapsed() - t_pre_submit;

        let t_pre_readback = t_start.elapsed();
        let normed = read_back_buffer(&self.gpu, &staging, bytes as usize);
        let t_readback = t_start.elapsed() - t_pre_readback;

        // Successful forward — bump the cache's write cursor. The caller
        // either runs `cpu().finalize_logits(&normed, n_tokens)` for plain
        // decode, or applies steer hidden_delta to the last token's slice
        // first and then re-projects.
        cache.advance(n_tokens);
        let t_total = t_start.elapsed();
        tracing::info!(
            n_tokens, start_pos, attn_max_seq,
            embed_us = t_embed.as_micros() as u64,
            io_alloc_us = t_io_alloc.as_micros() as u64,
            scratch_us = t_scratch.as_micros() as u64,
            record_us = t_record.as_micros() as u64,
            submit_us = t_submit.as_micros() as u64,
            readback_us = t_readback.as_micros() as u64,
            total_us = t_total.as_micros() as u64,
            "fwd_cache stage timings",
        );
        normed
    }

    /// **Phase 1 close — full forward on GPU.** Embedding lookup runs CPU
    /// (cheap; saves an embedding-gather shader for now), then ALL N blocks
    /// chain into one command encoder against resident weights, then
    /// final_norm runs on GPU in the same encoder. One submit. Output
    /// projection still runs CPU (vocab-sized matmul; deferred to a later
    /// phase that wires GpuFloatLinear into the projection path).
    ///
    /// Same constraints as `forward_block_gpu`: every block must be SwiGLU
    /// + SiLU with no biases / sub-norms / non-1.0 residual scales, every
    /// matvec layer must be `GpuFloatLinear`. Asserts on violations.
    pub fn forward_full_gpu(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0, "must have at least one token");

        // ---- 1. Embedding lookup (CPU) ----
        let embed_data = self.cpu.embedding_data();
        let vocab_size = self.cpu.vocab_size();
        let mut hidden_init: Vec<f32> = Vec::with_capacity(n_tokens * self.embed_dim);
        for &tok in tokens {
            assert!((tok as usize) < vocab_size, "token {tok} out of vocab");
            let off = tok as usize * self.embed_dim;
            hidden_init.extend_from_slice(&embed_data[off..off + self.embed_dim]);
        }

        // ---- Allocate buffers ----
        let bytes = (hidden_init.len() * std::mem::size_of::<f32>()) as u64;
        let hidden_buf = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward_full.hidden"),
            contents: bytemuck::cast_slice(&hidden_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let normed_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_full.normed"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.gpu.create_staging_buffer(bytes);

        // Per-block sizing (consistent across blocks for non-MoE models).
        let attn0 = self.cpu.blocks()[0].attention();
        let intermediate = self.cpu.blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_full_gpu requires SwiGLU FFN"))
            .intermediate_size();
        let scratch = BlockScratch::allocate(
            &self.gpu, n_tokens, self.embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, n_tokens,
        );
        // Single encoder for all blocks + final norm = one submit for the
        // whole forward pass. Earlier per-block submit was a workaround for
        // what turned out to be a separate cross-device bug (#16).
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_full.encoder"),
        });
        for i in 0..self.cpu.n_layers() {
            self.forward_block_gpu(&mut encoder, i, &hidden_buf, n_tokens, start_pos, &scratch);
        }
        self.dispatch_rmsnorm_into(
            &mut encoder, &hidden_buf, &self.final_norm_weight_buf, &normed_buf,
            self.embed_dim, n_tokens, self.final_norm_eps,
        );
        encoder.copy_buffer_to_buffer(&normed_buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));

        // ---- Read back final-normed hidden state ----
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("readback failed").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let normed: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        staging.unmap();

        // ---- 4. Output projection (CPU; vocab matmul deferred) ----
        self.cpu.finalize_logits(&normed, n_tokens)
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
        self.forward_block_gpu_inner(encoder, block_idx, hidden_buf, n_tokens, start_pos, scratch, None, None, None, None);
    }

    /// Same as `forward_block_gpu` but with optional pre-softmax score
    /// capture for retrieval / traced forward use, plus optional KV cache
    /// targeting for cached forward (decode + cached prefill), plus
    /// optional post-block hidden state capture for shim hooks.
    ///
    /// When `kv_cache_target` is Some, this block's projected K/V get
    /// written into the supplied cache buffers at offset `start_pos`, and
    /// the attention dispatch reads K/V back from the cache (with
    /// max_seq = start_pos + n_tokens) so the new tokens attend over the
    /// full prefix. When None, K/V live only in scratch and attention reads
    /// scratch (the prefill-only path).
    ///
    /// When `post_block_hidden_capture` is Some, the hidden state at the
    /// END of this block (after the FFN residual add) is copied into the
    /// supplied buffer. This is the natural attachment point for
    /// "entrance:N+1" shim hooks (the input to block N+1) and, when
    /// captured at the last block, the input to the final RMSNorm — i.e.
    /// what gate / steer shims read. Buffer must be sized
    /// `n_tokens * embed_dim * 4` bytes.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_gpu_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &wgpu::Buffer,
        n_tokens: usize,
        start_pos: usize,
        scratch: &BlockScratch,
        pre_softmax_capture: Option<&wgpu::Buffer>,
        kv_cache_target: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
        post_block_hidden_capture: Option<&wgpu::Buffer>,
        pre_block_hidden_inject: Option<&wgpu::Buffer>,
    ) {
        let block = &self.cpu.blocks()[block_idx];
        let block_gpu = &self.blocks_gpu[block_idx];
        let attn = block.attention();

        // Per-block pass merging (#pass-2): collapse the ~17 sequential
        // `begin_compute_pass` calls per block into 3 grouped passes
        // separated only by encoder-level operations (copy_buffer_to_buffer
        // for capture hooks) and the attention triple (which has its own
        // internal pass splits at score↔softmax↔value hazards). Saves
        // ~14 × ~0.8ms = ~11ms per block × 30 blocks = ~330ms per forward
        // of Vulkan pipeline-barrier overhead.

        // BitNet b1.58 sub-norms (#bn-11): when present on the CPU layer,
        // their resident weight buffers are populated on the GpuBlock at
        // construction. The dispatch sites below insert dispatch_rmsnorm
        // calls before o_proj / down_proj respectively.
        assert!((block.attn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires attn_residual_scale == 1.0");
        assert!((block.ffn_residual_scale() - 1.0).abs() < f32::EPSILON,
            "forward_block_gpu requires ffn_residual_scale == 1.0");

        let swiglu = block.ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_block_gpu requires SwiGLU FFN"));
        // Activation routing: SiLU (LLaMA / Qwen) or ReLU² (BitNet b1.58).

        let n_heads = attn.n_heads();
        let n_kv_heads = attn.n_kv_heads();
        let head_dim = attn.head_dim();
        let embed_dim = self.embed_dim;
        let intermediate = swiglu.intermediate_size();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        assert!(start_pos + n_tokens <= self.rope_max_seq,
            "start_pos + n_tokens ({}) exceeds rope_max_seq ({})",
            start_pos + n_tokens, self.rope_max_seq);

        // ----- PASS 1: inject (optional) + attn_norm + Q/K/V projs (+biases)
        // + RoPE + kv_write. All dispatches read/write block-local scratch
        // buffers with the implicit ordering wgpu enforces between dispatches
        // in a single pass on most drivers.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.block.pass1"),
                timestamp_writes: None,
            });

            // Injection-phase hook (#6c). Broadcast-add a [embed_dim] delta
            // into every token's hidden BEFORE attn_norm.
            if let Some(delta_buf) = pre_block_hidden_inject {
                self.dispatch_add_broadcast_in_pass(
                    &mut pass, hidden_buf, delta_buf, embed_dim, n_tokens,
                );
            }

            // 1. attn_norm: hidden -> normed
            self.dispatch_rmsnorm_in_pass(
                &mut pass, hidden_buf, &block_gpu.attn_norm_weight_buf, &scratch.normed,
                embed_dim, n_tokens, block_gpu.attn_norm_eps,
            );

            // 2-4. Q, K, V projections (+ optional Qwen-style biases)
            self.dispatch_linear_batch_in_pass(&mut pass, attn.q_proj(), &scratch.normed, &scratch.q, n_tokens, &scratch.ternary);
            if let Some(buf) = block_gpu.q_bias_buf.as_ref() {
                self.dispatch_bias_add_in_pass(&mut pass, &scratch.q, buf, q_dim, n_tokens);
            }
            self.dispatch_linear_batch_in_pass(&mut pass, attn.k_proj(), &scratch.normed, &scratch.k, n_tokens, &scratch.ternary);
            if let Some(buf) = block_gpu.k_bias_buf.as_ref() {
                self.dispatch_bias_add_in_pass(&mut pass, &scratch.k, buf, kv_dim, n_tokens);
            }
            self.dispatch_linear_batch_in_pass(&mut pass, attn.v_proj(), &scratch.normed, &scratch.v, n_tokens, &scratch.ternary);
            if let Some(buf) = block_gpu.v_bias_buf.as_ref() {
                self.dispatch_bias_add_in_pass(&mut pass, &scratch.v, buf, kv_dim, n_tokens);
            }

            // 5. RoPE on Q and K (in-place on scratch.q / scratch.k)
            self.dispatch_rope_in_pass(
                &mut pass, &scratch.q, &self.rope_cos_buf, &self.rope_sin_buf,
                n_heads, head_dim, start_pos, n_tokens,
            );
            self.dispatch_rope_in_pass(
                &mut pass, &scratch.k, &self.rope_cos_buf, &self.rope_sin_buf,
                n_kv_heads, head_dim, start_pos, n_tokens,
            );

            // 5.5 (cached path) Write K/V into the layer's resident cache.
            if let Some((k_cache, v_cache)) = kv_cache_target {
                self.dispatch_kv_write_in_pass(
                    &mut pass, &scratch.k, &scratch.v, k_cache, v_cache,
                    kv_dim, start_pos, n_tokens,
                );
            }
            // pass1 ends here (drop)
        }

        // 6. Attention math: Q · K^T, softmax, weighted V.
        // Kept as its own dispatcher because score↔softmax↔value share
        // scratch.scores with conflicting access — needs explicit pass
        // splits inside `dispatch_attention_inner`.
        let (k_for_attn, v_for_attn, attn_max_seq) = match kv_cache_target {
            Some((kc, vc)) => (kc, vc, start_pos + n_tokens),
            None => (&scratch.k, &scratch.v, n_tokens),
        };
        // Perf-bisect: see CORTEX_SKIP_SCORE / SOFTMAX / VALUE inside
        // `dispatch_attention_inner` — per-stage skip flags that bypass
        // individual attention dispatches. (The previous CORTEX_SKIP_ATTENTION
        // gate that called `encoder.clear_buffer` here panicked at runtime
        // because `scratch.attn_out` lacks the COPY_DST usage; the per-stage
        // flags are the correct path.)
        self.dispatch_attention_inner(
            encoder,
            &scratch.q, k_for_attn, v_for_attn,
            &scratch.scores, &scratch.attn_out,
            n_heads, n_kv_heads, head_dim,
            start_pos, attn_max_seq, n_tokens,
            pre_softmax_capture,
        );

        // ----- PASS 2: o_sub_norm (BitNet) + o_proj + residual + ffn_norm
        // + gate/up + silu_mul + ffn_sub_norm (BitNet) + down + residual.
        // All purely intra-block; ends at the optional post-block capture
        // (encoder copy outside any pass).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.block.pass2"),
                timestamp_writes: None,
            });

            // 6.5 (BitNet) attention sub-norm before o_proj. Reuses
            // scratch.normed (free after Q/K/V consumed it in pass 1).
            let o_proj_in: &wgpu::Buffer = if let Some(sub_norm_w) = block_gpu.o_sub_norm_weight_buf.as_ref() {
                self.dispatch_rmsnorm_in_pass(
                    &mut pass, &scratch.attn_out, sub_norm_w, &scratch.normed,
                    embed_dim, n_tokens, block_gpu.o_sub_norm_eps,
                );
                &scratch.normed
            } else {
                &scratch.attn_out
            };

            // 7. O projection: (sub-normed) attn_out -> projected
            self.dispatch_linear_batch_in_pass(&mut pass, attn.o_proj(), o_proj_in, &scratch.projected, n_tokens, &scratch.ternary);

            // 8. Residual: hidden += projected
            self.dispatch_add_in_pass(&mut pass, hidden_buf, &scratch.projected, embed_dim, n_tokens);

            // 9. ffn_norm: hidden -> normed
            self.dispatch_rmsnorm_in_pass(
                &mut pass, hidden_buf, &block_gpu.ffn_norm_weight_buf, &scratch.normed,
                embed_dim, n_tokens, block_gpu.ffn_norm_eps,
            );

            // 10-11. Gate / Up projections
            self.dispatch_linear_batch_in_pass(&mut pass, swiglu.gate_proj(), &scratch.normed, &scratch.gate, n_tokens, &scratch.ternary);
            self.dispatch_linear_batch_in_pass(&mut pass, swiglu.up_proj(),   &scratch.normed, &scratch.up,   n_tokens, &scratch.ternary);

            // 12. silu(gate) * up
            self.dispatch_gate_mul_in_pass(&mut pass, &scratch.gate, &scratch.up, &scratch.activated, intermediate, n_tokens, swiglu.activation());

            // 12.5 (BitNet) FFN sub-norm before down_proj. Reuses scratch.up.
            let down_proj_in: &wgpu::Buffer = if let Some(sub_norm_w) = block_gpu.ffn_sub_norm_weight_buf.as_ref() {
                self.dispatch_rmsnorm_in_pass(
                    &mut pass, &scratch.activated, sub_norm_w, &scratch.up,
                    intermediate, n_tokens, block_gpu.ffn_sub_norm_eps,
                );
                &scratch.up
            } else {
                &scratch.activated
            };

            // 13. Down projection
            self.dispatch_linear_batch_in_pass(&mut pass, swiglu.down_proj(), down_proj_in, &scratch.projected, n_tokens, &scratch.ternary);

            // 14. Residual: hidden += projected
            self.dispatch_add_in_pass(&mut pass, hidden_buf, &scratch.projected, embed_dim, n_tokens);
            // pass2 ends here (drop)
        }

        // 15. (optional) capture post-block hidden state — shim hook point.
        if let Some(capture_buf) = post_block_hidden_capture {
            let bytes = (n_tokens * embed_dim * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(hidden_buf, 0, capture_buf, 0, bytes);
        }
    }

    /// Polar variant of `forward_block_gpu_inner`. Same RMSNorm + projections
    /// + RoPE + FFN as the f32 path, but the K/V cache write is replaced by
    /// the GPU compress shader (writing into a `GpuPolarKvCache` layer's
    /// resident buffers) and the score → softmax → value chain runs against
    /// the polar cache via the batch polar dispatchers + softmax_batch.
    ///
    /// Caller provides:
    /// - `polar_cache`: the resident compressed cache (must hold the full
    ///   prefix this attention attends to; the query's K/V are written here
    ///   at offset `start_pos`).
    /// - `rotated_buf`: scratch buffer of size `n_tokens * n_heads * head_dim`
    ///   f32, reused inside the block as both `rq` (post-rotate_q) and
    ///   `weighted_rotated_V` (pre-derotate). One allocation per trace call.
    /// - `pre_softmax_capture`: optional buffer to copy scores into before
    ///   softmax overwrites them — for retrieval-mode tracing.
    #[allow(clippy::too_many_arguments)]
    fn forward_block_gpu_polar_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        hidden_buf: &wgpu::Buffer,
        n_tokens: usize,
        start_pos: usize,
        scratch: &BlockScratch,
        rotated_buf: &wgpu::Buffer,
        polar_cache: &crate::layers::gpu_polar_kv_cache::GpuPolarKvCache,
        pre_softmax_capture: Option<&wgpu::Buffer>,
        post_block_hidden_capture: Option<&wgpu::Buffer>,
        pre_block_hidden_inject: Option<&wgpu::Buffer>,
    ) {
        let block = &self.cpu.blocks()[block_idx];
        let block_gpu = &self.blocks_gpu[block_idx];
        let attn = block.attention();

        // Injection-phase hook (#6c). See forward_block_gpu_inner for
        // full discussion — same broadcast-add semantics, just on the
        // polar attention path.
        if let Some(delta_buf) = pre_block_hidden_inject {
            self.dispatch_add_broadcast_into(
                encoder, hidden_buf, delta_buf, self.embed_dim, n_tokens,
            );
        }

        // BitNet b1.58 sub-norms are routed through the same GpuBlock-
        // resident buffers used by the f32 forward path (#bn-11).
        assert!((block.attn_residual_scale() - 1.0).abs() < f32::EPSILON);
        assert!((block.ffn_residual_scale() - 1.0).abs() < f32::EPSILON);

        let swiglu = block.ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>()
            .unwrap_or_else(|| panic!("forward_block_gpu_polar requires SwiGLU FFN"));
        // Activation routed at dispatch time (SiLU vs ReLU²).

        let n_heads = attn.n_heads();
        let n_kv_heads = attn.n_kv_heads();
        let head_dim = attn.head_dim();
        let embed_dim = self.embed_dim;
        let intermediate = swiglu.intermediate_size();
        let kv_dim = n_kv_heads * head_dim;
        let attn_max_seq = start_pos + n_tokens;

        assert_eq!(n_kv_heads, polar_cache.n_kv_heads(),
            "polar cache n_kv_heads mismatch");
        assert_eq!(head_dim, polar_cache.head_dim(),
            "polar cache head_dim mismatch");
        assert!(attn_max_seq <= polar_cache.max_seq_len(),
            "polar cache overflow: start_pos {} + n_tokens {} > max_seq {}",
            start_pos, n_tokens, polar_cache.max_seq_len());
        assert!(start_pos + n_tokens <= self.rope_max_seq);

        // ===== ATTENTION SUBLAYER =====

        // 1. attn_norm
        self.dispatch_rmsnorm_into(
            encoder, hidden_buf, &block_gpu.attn_norm_weight_buf, &scratch.normed,
            embed_dim, n_tokens, block_gpu.attn_norm_eps,
        );

        // 2-4. Q, K, V projections — dispatch_linear_batch_into routes by
        // layer type (float vs ternary). See forward_block_gpu_inner for
        // the full rationale.
        self.dispatch_linear_batch_into(encoder, attn.q_proj(), &scratch.normed, &scratch.q, n_tokens, &scratch.ternary);
        if let Some(buf) = block_gpu.q_bias_buf.as_ref() {
            self.dispatch_bias_add_into(encoder, &scratch.q, buf, n_heads * head_dim, n_tokens);
        }
        self.dispatch_linear_batch_into(encoder, attn.k_proj(), &scratch.normed, &scratch.k, n_tokens, &scratch.ternary);
        if let Some(buf) = block_gpu.k_bias_buf.as_ref() {
            self.dispatch_bias_add_into(encoder, &scratch.k, buf, kv_dim, n_tokens);
        }
        self.dispatch_linear_batch_into(encoder, attn.v_proj(), &scratch.normed, &scratch.v, n_tokens, &scratch.ternary);
        if let Some(buf) = block_gpu.v_bias_buf.as_ref() {
            self.dispatch_bias_add_into(encoder, &scratch.v, buf, kv_dim, n_tokens);
        }

        // 5. RoPE on Q and K
        self.dispatch_rope_into(
            encoder, &scratch.q, &self.rope_cos_buf, &self.rope_sin_buf,
            n_heads, head_dim, start_pos, n_tokens,
        );
        self.dispatch_rope_into(
            encoder, &scratch.k, &self.rope_cos_buf, &self.rope_sin_buf,
            n_kv_heads, head_dim, start_pos, n_tokens,
        );

        // 5.5 Compress K and V into the polar cache at [start_pos, start_pos+n_tokens).
        crate::layers::gpu_polar::compress_layer_into_polar(
            &self.gpu, encoder, polar_cache, block_idx,
            &scratch.k, &scratch.v, n_tokens, start_pos,
        );

        // 6a. rotate_q: scratch.q (RoPE-rotated, original space) → rotated_buf (compressed/rotated space).
        crate::layers::gpu_polar::dispatch_rotate_q(
            &self.gpu, encoder, &scratch.q, polar_cache.rotation_layer(block_idx), rotated_buf,
            n_tokens, n_heads, head_dim,
        );

        // 6b. attn_score_polar_batch: rotated_buf · K_polar → scratch.scores
        crate::layers::gpu_polar::dispatch_attn_score_polar_batch(
            &self.gpu, encoder, rotated_buf,
            polar_cache.k_angles_layer(block_idx),
            polar_cache.k_radius_layer(block_idx),
            &scratch.scores, polar_cache.lut_buffer(),
            n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
        );

        // 6c. (optional) capture pre-softmax scores
        if let Some(capture_buf) = pre_softmax_capture {
            let bytes = (n_tokens * n_heads * attn_max_seq * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(&scratch.scores, 0, capture_buf, 0, bytes);
        }

        // 6d. softmax in-place on scratch.scores. Reuses the f32 path's softmax_batch
        //     pipeline; same scores buffer layout.
        let sm_params = SoftmaxBatchParams {
            n_heads: n_heads as u32,
            max_seq: attn_max_seq as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
        };
        let sm_params_buf = self.gpu.create_params_buffer(&sm_params);
        let sm_pipeline = &self.gpu.pipelines.softmax_batch;
        let sm_bind = self.gpu.make_bind_group(
            sm_pipeline, &[&scratch.scores, &sm_params_buf],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.polar.softmax.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(sm_pipeline);
            pass.set_bind_group(0, &sm_bind, &[]);
            pass.dispatch_workgroups((n_tokens * n_heads) as u32, 1, 1);
        }

        // 6e. attn_value_polar_batch: scratch.scores * V_polar → rotated_buf
        //     (overwriting rq, no longer needed)
        crate::layers::gpu_polar::dispatch_attn_value_polar_batch(
            &self.gpu, encoder, &scratch.scores,
            polar_cache.v_angles_layer(block_idx),
            polar_cache.v_radius_layer(block_idx),
            rotated_buf, polar_cache.lut_buffer(),
            n_heads, n_kv_heads, head_dim, start_pos, n_tokens, attn_max_seq,
        );

        // 6f. derotate: rotated_buf → scratch.attn_out (apply R^T per (tok, head))
        //     Treat (n_tokens * n_heads) as effective head count — R is per-layer.
        crate::layers::gpu_polar::dispatch_derotate(
            &self.gpu, encoder, rotated_buf,
            polar_cache.rotation_layer(block_idx), &scratch.attn_out,
            n_tokens * n_heads, head_dim,
        );

        // 6.5 (BitNet b1.58) attention sub-norm before o_proj
        let o_proj_in: &wgpu::Buffer = if let Some(sub_norm_w) = block_gpu.o_sub_norm_weight_buf.as_ref() {
            self.dispatch_rmsnorm_into(
                encoder, &scratch.attn_out, sub_norm_w, &scratch.normed,
                embed_dim, n_tokens, block_gpu.o_sub_norm_eps,
            );
            &scratch.normed
        } else {
            &scratch.attn_out
        };

        // 7. O projection
        self.dispatch_linear_batch_into(encoder, attn.o_proj(), o_proj_in, &scratch.projected, n_tokens, &scratch.ternary);

        // 8. Residual
        self.dispatch_add_into(encoder, hidden_buf, &scratch.projected, embed_dim, n_tokens);

        // ===== FFN SUBLAYER (unchanged from f32 path apart from sub-norm) =====

        self.dispatch_rmsnorm_into(
            encoder, hidden_buf, &block_gpu.ffn_norm_weight_buf, &scratch.normed,
            embed_dim, n_tokens, block_gpu.ffn_norm_eps,
        );
        self.dispatch_linear_batch_into(encoder, swiglu.gate_proj(), &scratch.normed, &scratch.gate, n_tokens, &scratch.ternary);
        self.dispatch_linear_batch_into(encoder, swiglu.up_proj(),   &scratch.normed, &scratch.up,   n_tokens, &scratch.ternary);
        self.dispatch_gate_mul_into(encoder, &scratch.gate, &scratch.up, &scratch.activated, intermediate, n_tokens, swiglu.activation());

        // 12.5 (BitNet b1.58) FFN sub-norm before down_proj
        let down_proj_in: &wgpu::Buffer = if let Some(sub_norm_w) = block_gpu.ffn_sub_norm_weight_buf.as_ref() {
            self.dispatch_rmsnorm_into(
                encoder, &scratch.activated, sub_norm_w, &scratch.up,
                intermediate, n_tokens, block_gpu.ffn_sub_norm_eps,
            );
            &scratch.up
        } else {
            &scratch.activated
        };

        self.dispatch_linear_batch_into(encoder, swiglu.down_proj(), down_proj_in, &scratch.projected, n_tokens, &scratch.ternary);
        self.dispatch_add_into(encoder, hidden_buf, &scratch.projected, embed_dim, n_tokens);

        // (optional) post-block hidden state capture — shim hook point.
        if let Some(capture_buf) = post_block_hidden_capture {
            let bytes = (n_tokens * embed_dim * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(hidden_buf, 0, capture_buf, 0, bytes);
        }
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.rope.pass"),
            timestamp_writes: None,
        });
        self.dispatch_rope_in_pass(&mut pass, x_buf, cos_buf, sin_buf, n_heads, head_dim, start_pos, n_tokens);
    }

    /// In-pass variant. See `dispatch_rope_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_rope_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
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

        let total_threads = (n_tokens * n_heads * half_dim) as u32;
        let dispatch_x = (total_threads + 63) / 64;

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
        self.dispatch_attention_inner(
            encoder, q_buf, k_buf, v_buf, scores_buf, out_buf,
            n_heads, n_kv_heads, head_dim, start_pos, max_seq, n_tokens,
            None,
        );
    }

    /// Same as `dispatch_attention_into` but if `pre_softmax_capture` is
    /// `Some`, the pre-softmax `scores_buf` contents are copied into it
    /// after the attn_score dispatch and before softmax overwrites them.
    /// Used by the retrieval / traced forward path to extract per-layer
    /// raw attention scores.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_attention_inner(
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
        pre_softmax_capture: Option<&wgpu::Buffer>,
    ) {
        assert!(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
        let heads_per_kv = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Bisect within attention (attn-3 followup): env-gated skip of
        // individual stages, to identify which of score/softmax/value
        // actually dominates wall time. Stage that, when skipped, gives
        // back the most time is the next optimization target.
        // Values: CORTEX_SKIP_SCORE / SOFTMAX / VALUE = 1 to skip.
        let skip_score = std::env::var("CORTEX_SKIP_SCORE").as_deref() == Ok("1");
        let skip_softmax = std::env::var("CORTEX_SKIP_SOFTMAX").as_deref() == Ok("1");
        let skip_value = std::env::var("CORTEX_SKIP_VALUE").as_deref() == Ok("1");

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
        // 256-thread workgroups over (head*max_seq, tok); gid.x covers
        // (head, t), gid.y covers tok.
        let inner_threads = (n_heads * max_seq) as u32;
        let score_groups_x = (inner_threads + 255) / 256;
        let score_groups_y = n_tokens as u32;
        if !skip_score {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_engine.attn_score.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(score_pipeline);
            pass.set_bind_group(0, &score_bind, &[]);
            pass.dispatch_workgroups(score_groups_x, score_groups_y, 1);
        }

        // ---- 1.5. (optional) capture pre-softmax scores ----
        if let Some(capture_buf) = pre_softmax_capture {
            let bytes = (n_tokens * n_heads * max_seq * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(scores_buf, 0, capture_buf, 0, bytes);
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
        if !skip_softmax {
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
        if !skip_value {
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
        self.dispatch_gate_mul_into(
            encoder, gate_buf, up_buf, out_buf, n, n_tokens,
            crate::layers::swiglu::GateActivation::SiLU,
        );
    }

    /// Activation-aware gate*up dispatcher: routes to `silu_mul_batch`
    /// or `relu2_mul_batch` based on the SwiGLU's activation kind.
    /// Both pipelines share the same binding layout, so the buffer
    /// arguments are interchangeable.
    pub fn dispatch_gate_mul_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.gate_mul.pass"),
            timestamp_writes: None,
        });
        self.dispatch_gate_mul_in_pass(&mut pass, gate_buf, up_buf, out_buf, n, n_tokens, activation);
    }

    /// In-pass variant. See `dispatch_gate_mul_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gate_mul_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        gate_buf: &wgpu::Buffer,
        up_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
        activation: crate::layers::swiglu::GateActivation,
    ) {
        let params = SiluMulBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = match activation {
            crate::layers::swiglu::GateActivation::SiLU => &self.gpu.pipelines.silu_mul_batch,
            crate::layers::swiglu::GateActivation::ReLU2 => &self.gpu.pipelines.relu2_mul_batch,
        };
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[gate_buf, up_buf, out_buf, &params_buf],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Dispatch kv_write_batch: copy K and V vectors for `n_tokens` new
    /// positions from per-block scratch buffers into the layer's resident
    /// cache buffers, starting at offset `start_pos` (in tokens). Used by
    /// the cached forward path.
    pub fn dispatch_kv_write_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k_src: &wgpu::Buffer,
        v_src: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        kv_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.kv_write.pass"),
            timestamp_writes: None,
        });
        self.dispatch_kv_write_in_pass(&mut pass, k_src, v_src, k_cache, v_cache, kv_dim, start_pos, n_tokens);
    }

    /// In-pass variant. See `dispatch_kv_write_into`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_kv_write_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        k_src: &wgpu::Buffer,
        v_src: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        kv_dim: usize,
        start_pos: usize,
        n_tokens: usize,
    ) {
        let params = KvWriteBatchParams {
            kv_dim: kv_dim as u32,
            start_pos: start_pos as u32,
            n_tokens: n_tokens as u32,
            _pad: 0,
        };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.kv_write_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[k_src, v_src, k_cache, v_cache, &params_buf],
        );

        let total = (kv_dim * n_tokens) as u32;
        let groups = (total + 127) / 128;
        let dx = groups.min(65535);
        let dy = (groups + 65534) / 65535;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Dispatch broadcast bias add: `a[tok, i] += bias[i]` for all tokens.
    /// Used for Q/K/V projection biases in Qwen-family models.
    pub fn dispatch_bias_add_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        bias_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.bias_add.pass"),
            timestamp_writes: None,
        });
        self.dispatch_bias_add_in_pass(&mut pass, a_buf, bias_buf, n, n_tokens);
    }

    /// In-pass variant. See `dispatch_bias_add_into`.
    pub fn dispatch_bias_add_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: &wgpu::Buffer,
        bias_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.bias_add_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[a_buf, bias_buf, &params_buf],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;

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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add.pass"),
            timestamp_writes: None,
        });
        self.dispatch_add_in_pass(&mut pass, a_buf, b_buf, n, n_tokens);
    }

    /// In-pass variant. See `dispatch_add_into`.
    pub fn dispatch_add_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Broadcast in-place add: every row of `a` ([n_tokens, n]) gets the
    /// same `delta` ([n]) added. Used by injection-phase shims (#6c) —
    /// one [embed_dim] delta is computed per request and applied at the
    /// chosen layer's entrance during every forward step (n_tokens=1
    /// during decode, prompt_len during prefill).
    pub fn dispatch_add_broadcast_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a_buf: &wgpu::Buffer,
        delta_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_engine.add_broadcast.pass"),
            timestamp_writes: None,
        });
        self.dispatch_add_broadcast_in_pass(&mut pass, a_buf, delta_buf, n, n_tokens);
    }

    /// In-pass variant. See `dispatch_add_broadcast_into`.
    pub fn dispatch_add_broadcast_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        a_buf: &wgpu::Buffer,
        delta_buf: &wgpu::Buffer,
        n: usize,
        n_tokens: usize,
    ) {
        let params = AddInplaceBatchParams { n: n as u32, n_tokens: n_tokens as u32 };
        let params_buf = self.gpu.create_params_buffer(&params);

        let pipeline = &self.gpu.pipelines.add_broadcast_batch;
        let bind = self.gpu.make_bind_group(
            pipeline,
            &[a_buf, delta_buf, &params_buf],
        );

        let total = (n * n_tokens) as u32;
        let groups = (total + 255) / 256;

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Upload an f32 slice to a GPU storage buffer. Used by callers
    /// (cortex-cloud) to materialize injection-phase hidden_delta
    /// vectors as resident buffers before threading them through
    /// `forward_full_gpu_with_cache_inject_returning_hidden`.
    pub fn upload_f32_to_storage(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
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

    /// Allocate a GPU-resident KV cache sized for this model.
    pub fn create_gpu_kv_cache(&self, max_seq_len: usize) -> crate::layers::gpu_kv_cache::GpuKvCache {
        let attn0 = self.cpu.blocks()[0].attention();
        crate::layers::gpu_kv_cache::GpuKvCache::new(
            self.gpu.clone(),
            self.cpu.n_layers(),
            attn0.n_kv_heads(),
            attn0.head_dim(),
            max_seq_len,
        )
    }

    /// Allocate a fresh `GpuPolarKvCache` shaped to this engine's model.
    /// `rotation_seed_base` selects the per-layer rotation matrices —
    /// must be consistent across caches that participate in the same
    /// composition / shared retrieval session.
    pub fn create_gpu_polar_kv_cache(
        &self,
        max_seq_len: usize,
        rotation_seed_base: u64,
    ) -> crate::layers::gpu_polar_kv_cache::GpuPolarKvCache {
        let attn0 = self.cpu.blocks()[0].attention();
        crate::layers::gpu_polar_kv_cache::GpuPolarKvCache::new(
            self.gpu.clone(),
            self.cpu.n_layers(),
            attn0.n_kv_heads(),
            attn0.head_dim(),
            max_seq_len,
            rotation_seed_base,
        )
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

/// Read back a buffer's contents to a `Vec<f32>`. The buffer must have
/// `MAP_READ` usage (typically created via `create_staging_buffer`) and
/// must already be the destination of a copy_buffer_to_buffer that was
/// included in the most recent submit.
fn read_back_buffer(gpu: &GpuDevice, staging: &wgpu::Buffer, bytes: usize) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("readback failed").expect("buffer map failed");
    let data = slice.get_mapped_range();
    let out: Vec<f32> = data[..bytes].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(data);
    staging.unmap();
    out
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

    /// Build a multi-block GPU model. `n_blocks` independent transformer
    /// blocks, each with its own freshly-rolled weights. Used to validate
    /// the block-stacking loop in `forward_full_gpu`.
    fn toy_float_multi_block_for_gpu(gpu: Arc<GpuDevice>, n_blocks: usize) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
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

        let mut rng: u64 = 0xABCD_1234;
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

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
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
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Polar-compatible toy: same shape as `toy_float_multi_block_for_gpu`
    /// but with `head_dim = 32` so PolarQuant has enough pairs (16) for
    /// the radius averaging to be stable. head_dim=8 lands in PolarQuant's
    /// pathologically-lossy regime (only 4 pairs → 22.5° per-pair angle
    /// noise, single-vector radius averaging) and produces directional
    /// agreement near 0.3, which doesn't usefully test correctness vs
    /// noise.
    fn toy_float_multi_block_for_gpu_polar(gpu: Arc<GpuDevice>, n_blocks: usize) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let embed_dim = 32;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 32;
        let vocab_size = 4;

        let mut rng: u64 = 0xABCD_1234;
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

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
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
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// CPU-equivalent multi-block model with the same seeded weights.
    fn toy_float_multi_block_cpu(n_blocks: usize) -> TransformerModel {
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

        let mut rng: u64 = 0xABCD_1234;
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

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
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
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Build CPU + GPU one-block models with Q/K/V biases set, sharing the
    /// same seeded weights (deterministic RNG state). Used to validate the
    /// bias dispatch path lights up correctly.
    fn toy_with_biases_pair(gpu: Arc<GpuDevice>) -> (TransformerModel, TransformerModel) {
        let cpu = build_toy_with_biases(None);
        let gpu_model = build_toy_with_biases(Some(gpu));
        (cpu, gpu_model)
    }

    /// One-block model with biases. If `gpu` is Some, layers are GpuFloatLinear;
    /// otherwise CPU FloatLinear. Same RNG seed → identical weights.
    fn build_toy_with_biases(gpu: Option<Arc<GpuDevice>>) -> TransformerModel {
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
        let q_dim = n_heads * head_dim;
        let kv_dim = n_heads * head_dim;

        let mut rng: u64 = 0xB1A5_C0DE;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        };

        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_layer = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32|
            -> Box<dyn crate::layers::linear::LinearLayer> {
            let t = mk_tensor(rows, cols, roll);
            match &gpu {
                Some(g) => Box::new(GpuFloatLinear::from_float_tensor(g.clone(), t)),
                None    => Box::new(FloatLinear::from_float_tensor(t)),
            }
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let q = mk_layer(embed_dim, embed_dim, &mut roll);
        let k = mk_layer(embed_dim, embed_dim, &mut roll);
        let v = mk_layer(embed_dim, embed_dim, &mut roll);
        let o = mk_layer(embed_dim, embed_dim, &mut roll);
        let mut attention = MultiHeadAttention::with_rope_layout(
            q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
            crate::layers::rope::RoPELayout::Halved,
        );
        let q_bias: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
        let k_bias: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
        let v_bias: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
        attention.set_biases(q_bias, k_bias, v_bias);

        let gate = mk_layer(intermediate, embed_dim, &mut roll);
        let up   = mk_layer(intermediate, embed_dim, &mut roll);
        let down = mk_layer(embed_dim, intermediate, &mut roll);
        let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, vec![block], final_norm, OutputProjection::Float(out_proj))
    }

    #[test]
    fn forward_full_gpu_matches_cpu_with_biases() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let (cpu_ref, gpu_model) = toy_with_biases_pair(gpu.clone());
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![0u32, 1, 2];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.05,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn forward_full_gpu_traced_matches_cpu_traced() {
        // Validates that the captured pre-softmax scores match what CPU
        // forward_traced would produce. Same toy model in both paths
        // (CPU FloatLinear vs GPU GpuFloatLinear, identical seeded weights).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(2);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let n_tokens = tokens.len();
        let n_layers = 2;
        let n_heads = 1;

        let (cpu_logits, cpu_trace) = cpu_ref.forward_traced(&tokens);
        let (gpu_logits, gpu_per_layer) =
            engine.forward_full_gpu_traced(&tokens, 0, &(0..n_layers).collect::<Vec<_>>());

        // Logits should match within tolerance (same as forward_full_gpu test).
        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!((c - g).abs() < 0.1, "logit {i}: cpu={c} gpu={g}");
        }

        // Pre-softmax scores: per-layer comparison.
        // CPU layout: [n_heads, seq_len, seq_len] flat; row(layer, h, q) returns &[seq_len].
        // GPU layout: [n_tokens, n_heads, n_tokens] flat (matches attn_score_batch shader).
        // These are different orderings — we have to index carefully.
        for layer in 0..n_layers {
            let gpu_scores = &gpu_per_layer[layer];
            assert_eq!(gpu_scores.len(), n_tokens * n_heads * n_tokens);
            for h in 0..n_heads {
                for q in 0..n_tokens {
                    let cpu_row = cpu_trace.pre_score_row(layer, h, q);
                    // Causal: only positions 0..=q are real; rest -inf -> filtered by callers.
                    for k in 0..=q {
                        let gpu_idx = q * n_heads * n_tokens + h * n_tokens + k;
                        let gpu_v = gpu_scores[gpu_idx];
                        let cpu_v = cpu_row[k];
                        assert!(
                            (cpu_v - gpu_v).abs() < 0.05,
                            "layer={layer} h={h} q={q} k={k}: cpu={cpu_v} gpu={gpu_v}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn forward_with_cache_prefill_matches_no_cache() {
        // Cached forward starting from an EMPTY cache should be identical
        // to plain forward_full_gpu — both run the same dispatches over
        // start_pos=0, n_tokens=N. Validates the kv_write + cache-read path.
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let no_cache_logits = engine.forward_full_gpu(&tokens, 0);

        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let cached_logits = engine.forward_full_gpu_with_cache(&tokens, &mut cache);

        assert_eq!(cache.seq_len(), tokens.len());
        assert_eq!(no_cache_logits.len(), cached_logits.len());
        for (i, (a, b)) in no_cache_logits.iter().zip(&cached_logits).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "logit {i}: no_cache={a} cached={b} (diff={})",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn forward_traced_with_cache_matches_fresh_traced() {
        // Compare:
        //   A) prefill shard tokens into cache, then forward_full_gpu_with_cache_traced
        //      on the query tokens with that cache
        //   B) forward_full_gpu_traced over [shard_tokens; query_tokens] from
        //      scratch (the path retrieve uses today)
        //
        // The pre-softmax scores at the QUERY positions should match.
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_heads = attn0.n_heads();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let shard_tokens = vec![1u32, 2, 3];
        let query_tokens = vec![0u32, 1];
        let all: Vec<u32> = shard_tokens.iter().chain(query_tokens.iter()).copied().collect();

        let capture_layers: Vec<usize> = (0..n_layers).collect();

        // Path A: cached
        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&shard_tokens, &mut cache);
        assert_eq!(cache.seq_len(), shard_tokens.len());
        let cached_scores = engine.forward_full_gpu_with_cache_traced(
            &query_tokens, &cache, &capture_layers,
        );

        // Path B: fresh
        let (_, fresh_scores) = engine.forward_full_gpu_traced(&all, 0, &capture_layers);

        // Compare scores at query positions. Cached layout is
        // [n_query=2, n_heads, max_seq=5]; fresh layout is
        // [n_total=5, n_heads, max_seq=5]. For query token q in [0, 2),
        // global position is shard.len() + q. Compare row-by-row.
        let n_query = query_tokens.len();
        let max_seq = shard_tokens.len() + query_tokens.len();
        let n_total = all.len();
        for layer in 0..n_layers {
            for q in 0..n_query {
                for h in 0..n_heads {
                    let global_pos = shard_tokens.len() + q;
                    for k in 0..=global_pos { // causal
                        let cached_idx = q * n_heads * max_seq + h * max_seq + k;
                        let fresh_idx = global_pos * n_heads * n_total + h * n_total + k;
                        let c = cached_scores[layer][cached_idx];
                        let f = fresh_scores[layer][fresh_idx];
                        assert!(
                            (c - f).abs() < 0.05,
                            "layer={layer} q={q} (global {global_pos}) h={h} k={k}: cached={c} fresh={f}",
                        );
                    }
                }
            }
        }

        // Cache must NOT have advanced.
        assert_eq!(cache.seq_len(), shard_tokens.len(),
            "forward_full_gpu_with_cache_traced should not advance the cache");
    }

    /// Load-bearing phase 2c.5b test: end-to-end polar trace forward.
    ///
    /// Compare:
    ///   A) prefill shard tokens into f32 cache, then
    ///      forward_full_gpu_with_cache_traced (existing f32 path)
    ///   B) prefill shard tokens into f32 cache, convert to polar via
    ///      populate_from_f32_cache_gpu, then forward_full_gpu_polar_traced
    ///      (new polar path)
    ///
    /// Both paths run the same model on the same query against the same
    /// underlying K/V data; the only difference is the polar quantization
    /// in path B. Per-layer pre-softmax scores at query positions should
    /// agree by cosine similarity (PolarQuant is lossy, so we check
    /// directional agreement, not numeric equality). This is the
    /// load-bearing test that the polar attention chain in each block is
    /// computing what a polar-attention forward should compute.
    /// Hidden-state extraction hook: capture per-layer post-block hidden
    /// states + final post-norm hidden via `forward_full_gpu_with_hidden_capture`.
    /// Sanity-check shapes are right, all values are finite, and the
    /// last-token slicer helpers return what they should.
    #[test]
    fn hidden_capture_returns_finite_states() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let n_layers = gpu_model.n_layers();
        let embed_dim = 4; // toy model embed_dim
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![1u32, 2, 3];
        let n_tokens = tokens.len();
        let capture_layers: Vec<usize> = (0..n_layers).collect();

        let hc = engine.forward_full_gpu_with_hidden_capture(&tokens, &capture_layers);

        assert_eq!(hc.n_tokens, n_tokens);
        assert_eq!(hc.embed_dim, embed_dim);
        assert_eq!(hc.per_layer_hidden.len(), n_layers);
        for (i, layer) in hc.per_layer_hidden.iter().enumerate() {
            assert_eq!(layer.len(), n_tokens * embed_dim,
                "layer {i} hidden size");
            assert!(layer.iter().all(|v| v.is_finite()),
                "layer {i} hidden contains non-finite values");
        }
        assert_eq!(hc.final_post_norm_hidden.len(), n_tokens * embed_dim);
        assert!(hc.final_post_norm_hidden.iter().all(|v| v.is_finite()));

        // The post-FFN hidden of the LAST block is the input to the final
        // RMSNorm. After RMSNorm, magnitudes change but direction is
        // preserved. Sanity: both have nonzero norm at the last token.
        let last_layer_last_tok = hc.layer_last_token(n_layers - 1);
        let final_last_tok = hc.final_last_token();
        let mag_last_layer: f32 = last_layer_last_tok.iter().map(|x| x*x).sum::<f32>().sqrt();
        let mag_final: f32 = final_last_tok.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!(mag_last_layer > 0.0, "last-block last-token hidden should be nonzero");
        assert!(mag_final > 0.0, "final post-norm last-token hidden should be nonzero");
    }

    /// Hooks-are-optional invariant: forward without capture (capture_layers
    /// empty) should still produce a valid final post-norm hidden, and not
    /// allocate any per-layer capture buffers.
    #[test]
    fn hidden_capture_with_empty_layers_works() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let hc = engine.forward_full_gpu_with_hidden_capture(&[1u32, 2], &[]);
        assert!(hc.per_layer_hidden.is_empty());
        assert!(hc.final_post_norm_hidden.iter().all(|v| v.is_finite()));
        assert_eq!(hc.final_post_norm_hidden.len(), 2 * 4); // n_tokens * embed_dim
    }

    #[test]
    fn polar_traced_matches_f32_traced_directionally() {
        use crate::layers::gpu_kv_cache::GpuKvCache;
        use crate::layers::gpu_polar_kv_cache::GpuPolarKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // head_dim=8 polar-compat fixture (toy_float_multi_block_for_gpu's
        // head_dim=4 doesn't satisfy the polar shader's head_dim%8==0).
        let gpu_model = toy_float_multi_block_for_gpu_polar(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_heads = attn0.n_heads();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let shard_tokens = vec![1u32, 2, 3];
        let query_tokens = vec![0u32, 1];
        let capture_layers: Vec<usize> = (0..n_layers).collect();

        // Path A: f32 prefill + f32 traced forward.
        let mut f32_cache = GpuKvCache::new(gpu.clone(), n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&shard_tokens, &mut f32_cache);
        assert_eq!(f32_cache.seq_len(), shard_tokens.len());
        let f32_scores = engine.forward_full_gpu_with_cache_traced(
            &query_tokens, &f32_cache, &capture_layers,
        );

        // Path B: convert f32→polar on GPU, then polar traced forward.
        let mut polar_cache = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, /*max_seq*/ 16, /*seed_base*/ 42,
        );
        polar_cache.populate_from_f32_cache_gpu(&f32_cache);
        assert_eq!(polar_cache.seq_len(), shard_tokens.len());
        let polar_scores = engine.forward_full_gpu_polar_traced(
            &query_tokens, &polar_cache, &capture_layers,
        );

        // Cache must NOT have advanced (mirrors f32 trace semantics).
        assert_eq!(polar_cache.seq_len(), shard_tokens.len(),
            "forward_full_gpu_polar_traced should not advance the cache");

        // Per-layer cosine similarity of valid (causally-unmasked) score
        // rows. Each layer captures [n_query, n_heads, max_seq] scores;
        // for query token q, valid positions are [0, shard_len + q + 1).
        let n_query = query_tokens.len();
        let max_seq = shard_tokens.len() + n_query;
        for layer in 0..n_layers {
            for q in 0..n_query {
                for h in 0..n_heads {
                    let row_off = q * n_heads * max_seq + h * max_seq;
                    let valid_len = shard_tokens.len() + q + 1;
                    let f_row = &f32_scores[layer][row_off..row_off + valid_len];
                    let p_row = &polar_scores[layer][row_off..row_off + valid_len];

                    let dot: f32 = f_row.iter().zip(p_row).map(|(&a, &b)| a * b).sum();
                    let na: f32 = f_row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = p_row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let cos = dot / (na * nb).max(1e-12);

                    // PolarQuant is lossy; agreement is directional, not
                    // exact. Worst rows in this fixture land around 0.5 —
                    // pin a 0.4 floor to catch sign-flips / dimension
                    // shuffles, the failure modes that actually matter.
                    assert!(
                        cos > 0.4,
                        "layer {layer} q {q} h {h}: cos={cos:.4} below 0.4 floor",
                    );
                }
            }
        }
    }

    #[test]
    fn forward_with_cache_decode_matches_full_forward() {
        // Validates the decode path: prefill [1,2,3] then decode [4,5]
        // through the cache should produce logits matching positions 3
        // and 4 of forward_full_gpu([1,2,3,4,5]).
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let attn0 = gpu_model.blocks()[0].attention();
        let n_layers = gpu_model.n_layers();
        let n_kv_heads = attn0.n_kv_heads();
        let head_dim = attn0.head_dim();
        let vocab_size = gpu_model.vocab_size();

        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let prefill = vec![1u32, 2, 3];
        let decode = vec![0u32, 1];
        let all: Vec<u32> = prefill.iter().chain(decode.iter()).copied().collect();

        // Reference: single big forward over [1,2,3,0,1].
        let ref_logits = engine.forward_full_gpu(&all, 0);

        // Cached path: prefill then decode.
        let mut cache = GpuKvCache::new(gpu, n_layers, n_kv_heads, head_dim, 16);
        let _ = engine.forward_full_gpu_with_cache(&prefill, &mut cache);
        assert_eq!(cache.seq_len(), 3);
        let decode_logits = engine.forward_full_gpu_with_cache(&decode, &mut cache);
        assert_eq!(cache.seq_len(), 5);

        // decode_logits is logits for the 2 decode tokens (positions 3 and 4
        // of the full sequence). They should match the corresponding rows
        // of ref_logits.
        for tok_idx in 0..decode.len() {
            let global_pos = prefill.len() + tok_idx;
            let ref_row = &ref_logits[global_pos * vocab_size..(global_pos + 1) * vocab_size];
            let decode_row = &decode_logits[tok_idx * vocab_size..(tok_idx + 1) * vocab_size];
            for (i, (r, d)) in ref_row.iter().zip(decode_row).enumerate() {
                assert!(
                    (r - d).abs() < 1e-2,
                    "tok {tok_idx} (global pos {global_pos}) logit {i}: ref={r} decoded={d}",
                );
            }
        }
    }

    #[test]
    fn forward_full_gpu_traced_capture_subset() {
        // Capturing only a subset of layers: check we get back exactly
        // those layers' worth of buffers.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 4);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu, 16);

        let tokens = vec![0u32, 1, 2];
        // Capture only layers 1 and 3.
        let (_logits, per_layer) = engine.forward_full_gpu_traced(&tokens, 0, &[1, 3]);
        assert_eq!(per_layer.len(), 2);
        // Each capture is [n_tokens=3, n_heads=1, n_tokens=3] = 9 floats.
        assert_eq!(per_layer[0].len(), 9);
        assert_eq!(per_layer[1].len(), 9);
    }

    #[test]
    fn forward_full_gpu_matches_cpu_single_block() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(1);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 1);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2, 3];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.05,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
    }

    #[test]
    fn forward_full_gpu_thirty_blocks_no_crash() {
        // Reproducer for the validation error that hits the bench at ~block
        // 26 with Qwen 3B. Just runs forward without comparing to CPU
        // (32-block deep f16 chains diverge precision-wise; this test only
        // cares about "does it complete without a wgpu validation panic").
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 30);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0); // must not panic
    }

    #[test]
    fn forward_full_gpu_fifty_blocks_no_crash() {
        // Bisection probe for the Qwen failure: if dispatch count is the
        // culprit (~494 dispatches at the Qwen failure point), 50 toy
        // blocks × 16 dispatches = 800 dispatches should reproduce.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 50);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    /// Multi-block toy model WITH Q/K/V biases set.
    fn toy_float_multi_block_with_biases_for_gpu(
        gpu: Arc<GpuDevice>, n_blocks: usize,
    ) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
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
        let q_dim = n_heads * head_dim;

        let mut rng: u64 = 0xBEAD_BEAD;
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
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(embed_dim, embed_dim, &mut roll);
            let k = mk_gpu(embed_dim, embed_dim, &mut roll);
            let v = mk_gpu(embed_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, embed_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let kb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let vb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            attention.set_biases(qb, kb, vb);

            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    /// Larger-tensor multi-block model. n_heads != n_kv_heads (GQA).
    fn toy_gqa_multi_block_for_gpu(
        gpu: Arc<GpuDevice>, n_blocks: usize,
        embed_dim: usize, intermediate: usize,
        n_heads: usize, n_kv_heads: usize,
    ) -> TransformerModel {
        toy_gqa_multi_block_for_gpu_inner(gpu, n_blocks, embed_dim, intermediate, n_heads, n_kv_heads, /*with_biases*/ false)
    }

    fn toy_gqa_multi_block_for_gpu_inner(
        gpu: Arc<GpuDevice>, n_blocks: usize,
        embed_dim: usize, intermediate: usize,
        n_heads: usize, n_kv_heads: usize,
        with_biases: bool,
    ) -> TransformerModel {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let head_dim = embed_dim / n_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab_size = 32;

        let mut rng: u64 = 0xCAFE_F00D;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.001
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(q_dim, embed_dim, &mut roll);
            let k = mk_gpu(kv_dim, embed_dim, &mut roll);
            let v = mk_gpu(kv_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, q_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_kv_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            if with_biases {
                let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
                let kb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
                let vb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
                attention.set_biases(qb, kb, vb);
            }
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = mk_tensor(vocab_size, embed_dim, &mut roll);
        TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Float(out_proj))
    }

    #[test]
    fn forward_full_gpu_qwen_shaped_with_biases_no_crash() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_gqa_multi_block_for_gpu_inner(
            gpu.clone(), 36,
            /*embed*/ 2048, /*intermediate*/ 11008,
            /*n_heads*/ 16, /*n_kv_heads*/ 2,
            /*with_biases*/ true,
        );
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);
        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    #[ignore = "loads Qwen 3B from disk; run with --ignored on workstation"]
    fn forward_full_gpu_real_qwen3b_no_crash() {
        // Loads the actual Qwen 2.5-3B Q4_K_M from disk (same path the bench
        // uses). If this crashes inside the test framework, we have a
        // standalone reproducer of the bench failure.
        let path = "C:\\Users\\danu\\AppData\\Roaming\\memory-rlm\\models\\model-qwen2.5-3b-q4km.gguf";
        if !std::path::Path::new(path).exists() { return; }

        let loaded = crate::load_model(path).expect("load_model");
        let gpu = loaded.gpu.clone().expect("model loaded without GPU");
        let engine = GpuEngine::with_max_seq(loaded.model, gpu, 4096);
        let tokens: Vec<u32> = (0..16u32).collect();
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_qwen_shape_with_gpu_output_proj_no_crash() {
        // Final bisection: same shape as the bench, biases on, output proj
        // routed through GpuFloatLinear (matches what loader.rs does for
        // float models with GPU available). If THIS panics, we've isolated
        // the trigger to "GpuFloatLinear on the output projection".
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_floatlinear::GpuFloatLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Smaller vocab to keep memory in check; the bug should still
        // trigger if output-proj-via-GpuFloatLinear is the cause.
        let embed_dim = 2048;
        let n_heads = 16;
        let n_kv_heads = 2;
        let head_dim = embed_dim / n_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = 11008;
        let vocab_size = 151_936; // Qwen 2.5 vocab
        let n_blocks = 36;

        let mut rng: u64 = 0xDEAD_F00D;
        let mut roll = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.001
        };
        let mk_tensor = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> FloatTensor {
            FloatTensor::new((0..rows * cols).map(|_| roll()).collect(), vec![rows, cols])
        };
        let mk_gpu = |rows: usize, cols: usize, roll: &mut dyn FnMut() -> f32| -> Box<dyn crate::layers::linear::LinearLayer> {
            Box::new(GpuFloatLinear::from_float_tensor(gpu.clone(), mk_tensor(rows, cols, roll)))
        };

        let embedding = mk_tensor(vocab_size, embed_dim, &mut roll);
        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let q = mk_gpu(q_dim, embed_dim, &mut roll);
            let k = mk_gpu(kv_dim, embed_dim, &mut roll);
            let v = mk_gpu(kv_dim, embed_dim, &mut roll);
            let o = mk_gpu(embed_dim, q_dim, &mut roll);
            let mut attention = MultiHeadAttention::with_rope_layout(
                q, k, v, o, n_heads, n_kv_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let qb: Vec<f32> = (0..q_dim).map(|_| roll()).collect();
            let kb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
            let vb: Vec<f32> = (0..kv_dim).map(|_| roll()).collect();
            attention.set_biases(qb, kb, vb);
            let gate = mk_gpu(intermediate, embed_dim, &mut roll);
            let up   = mk_gpu(intermediate, embed_dim, &mut roll);
            let down = mk_gpu(embed_dim, intermediate, &mut roll);
            let ffn: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate, up, down));
            let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            blocks.push(TransformerBlock::new(attn_norm, attention, ffn_norm, ffn));
        }
        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-6);

        // The candidate trigger: output proj through GpuFloatLinear.
        let out_proj_gpu = mk_gpu(vocab_size, embed_dim, &mut roll);

        let model = TransformerModel::new(embedding, blocks, final_norm, OutputProjection::Linear(out_proj_gpu));
        let engine = GpuEngine::with_max_seq(model, gpu.clone(), 16);
        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_qwen_shaped_no_crash() {
        // Qwen 2.5-3B shape (embed=2048, GQA 16/2, intermediate=11008,
        // 36 blocks) — the actual case that crashes via cortex-bench-fwd.
        // If THIS panics in tests, the bug is reproducible without the
        // bench harness; if it doesn't, something about the bench's
        // execution path matters (warmup ordering, finalize_logits, etc.).
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_gqa_multi_block_for_gpu(
            gpu.clone(), 36,
            /*embed*/ 2048, /*intermediate*/ 11008,
            /*n_heads*/ 16, /*n_kv_heads*/ 2,
        );
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_fifty_blocks_with_biases_no_crash() {
        // Probe: does adding biases (extra 3 dispatches per block) trigger
        // the Qwen failure with toy-size tensors?
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let gpu_model = toy_float_multi_block_with_biases_for_gpu(gpu.clone(), 50);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![0u32, 1, 2];
        let _ = engine.forward_full_gpu(&tokens, 0);
    }

    #[test]
    fn forward_full_gpu_matches_cpu_two_blocks() {
        // Validates the block-stacking loop: that hidden state correctly
        // flows from block 0 -> block 1 inside one encoder.
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cpu_ref = toy_float_multi_block_cpu(2);
        let gpu_model = toy_float_multi_block_for_gpu(gpu.clone(), 2);
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let tokens = vec![1u32, 2, 3];
        let cpu_logits = cpu_ref.forward(&tokens, 0);
        let gpu_logits = engine.forward_full_gpu(&tokens, 0);

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        // 2 blocks ~24 dispatches; error compounds slightly.
        for (i, (c, g)) in cpu_logits.iter().zip(&gpu_logits).enumerate() {
            assert!(
                (c - g).abs() < 0.1,
                "logit {i}: cpu={c} gpu={g} (diff={})",
                (c - g).abs()
            );
        }
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

    /// Build a matched (CPU, GPU) toy block pair backed by ternary
    /// `BitLinear` / `GpuBitLinear` layers with shared weights. Used by
    /// `forward_block_gpu_matches_cpu_bitnet_block` to validate the new
    /// batched ternary matmul path against the CPU reference.
    fn toy_ternary_block_pair(gpu: Arc<GpuDevice>)
        -> (TransformerModel, TransformerModel)
    {
        use crate::layers::attention::MultiHeadAttention;
        use crate::layers::bitlinear::BitLinear;
        use crate::layers::ffn::FeedForward;
        use crate::layers::gpu_bitlinear::GpuBitLinear;
        use crate::layers::model::OutputProjection;
        use crate::layers::rmsnorm::RmsNorm;
        use crate::layers::swiglu::SwiGLU;
        use crate::layers::transformer::TransformerBlock;
        use crate::tensor::TernaryTensor;

        let embed_dim = 8;
        let n_heads = 1;
        let head_dim = embed_dim;
        let intermediate = 16;
        let vocab_size = 4;

        let mut rng: u64 = 0xBADD_F00D_FACE_BEEF;
        fn step(rng: &mut u64) -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *rng
        }
        let roll_ternary = |rng: &mut u64| -> crate::tensor::Ternary {
            let r = step(rng);
            match ((r >> 33) % 3) as i8 {
                0 => crate::tensor::Ternary::Neg,
                1 => crate::tensor::Ternary::Zero,
                _ => crate::tensor::Ternary::Pos,
            }
        };
        let roll_f32 = |rng: &mut u64| -> f32 {
            let r = step(rng);
            ((r >> 33) as i32 % 200 - 100) as f32 * 0.01
        };

        let mk_pair = |rng: &mut u64, rows: usize, cols: usize, scale: f32|
            -> (Box<dyn crate::layers::linear::LinearLayer>, Box<dyn crate::layers::linear::LinearLayer>) {
            let values: Vec<crate::tensor::Ternary> = (0..rows * cols).map(|_| roll_ternary(rng)).collect();
            let tensor = TernaryTensor::pack(&values, rows, cols);
            let cpu_layer: Box<dyn crate::layers::linear::LinearLayer> =
                Box::new(BitLinear::new(tensor.clone(), scale));
            let gpu_layer: Box<dyn crate::layers::linear::LinearLayer> =
                Box::new(GpuBitLinear::from_weights(gpu.clone(), tensor, scale));
            (cpu_layer, gpu_layer)
        };

        let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| roll_f32(&mut rng)).collect();
        let embedding = FloatTensor::new(embed_data, vec![vocab_size, embed_dim]);

        let mut cpu_blocks = Vec::new();
        let mut gpu_blocks = Vec::new();
        let scale = 0.02f32;
        // One block is enough to exercise all six linear call sites.
        for _ in 0..1 {
            let (q_c, q_g) = mk_pair(&mut rng, embed_dim, embed_dim, scale);
            let (k_c, k_g) = mk_pair(&mut rng, embed_dim, embed_dim, scale);
            let (v_c, v_g) = mk_pair(&mut rng, embed_dim, embed_dim, scale);
            let (o_c, o_g) = mk_pair(&mut rng, embed_dim, embed_dim, scale);
            let attn_c = MultiHeadAttention::with_rope_layout(
                q_c, k_c, v_c, o_c, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let attn_g = MultiHeadAttention::with_rope_layout(
                q_g, k_g, v_g, o_g, n_heads, n_heads, head_dim, 10000.0,
                crate::layers::rope::RoPELayout::Halved,
            );
            let (gate_c, gate_g) = mk_pair(&mut rng, intermediate, embed_dim, scale);
            let (up_c, up_g)     = mk_pair(&mut rng, intermediate, embed_dim, scale);
            let (down_c, down_g) = mk_pair(&mut rng, embed_dim, intermediate, scale);
            let ffn_c: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate_c, up_c, down_c));
            let ffn_g: Box<dyn FeedForward> = Box::new(SwiGLU::new(gate_g, up_g, down_g));
            let an_c = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let an_g = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let fn_c = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            let fn_g = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
            cpu_blocks.push(TransformerBlock::new(an_c, attn_c, fn_c, ffn_c));
            gpu_blocks.push(TransformerBlock::new(an_g, attn_g, fn_g, ffn_g));
        }

        let final_norm_c = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let final_norm_g = RmsNorm::new(vec![1.0; embed_dim], 1e-6);
        let out_proj = FloatTensor::new(
            (0..vocab_size * embed_dim).map(|_| roll_f32(&mut rng)).collect(),
            vec![vocab_size, embed_dim],
        );
        let cpu_model = TransformerModel::new(
            embedding.clone(), cpu_blocks, final_norm_c, OutputProjection::Float(out_proj.clone()),
        );
        let gpu_model = TransformerModel::new(
            embedding, gpu_blocks, final_norm_g, OutputProjection::Float(out_proj),
        );
        (cpu_model, gpu_model)
    }

    #[test]
    fn forward_block_gpu_matches_cpu_bitnet_block() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let (cpu_model, gpu_model) = toy_ternary_block_pair(gpu.clone());

        let n_tokens = 4;
        let embed_dim = cpu_model.embed_dim();

        // Random f32 hidden state input (not embedded — direct injection).
        let mut rng: u64 = 0xAA11_BB22_CC33_DD44;
        let hidden_in: Vec<f32> = (0..n_tokens * embed_dim).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 200 - 100) as f32 * 0.01
        }).collect();

        // CPU reference.
        let cpu_out = cpu_model.blocks()[0].forward(&hidden_in, n_tokens, /*start_pos*/ 0);

        // GPU path.
        let engine = GpuEngine::with_max_seq(gpu_model, gpu.clone(), 16);

        let bytes = (n_tokens * embed_dim * 4) as u64;
        let hidden_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test.bitnet.hidden"),
            contents: bytemuck::cast_slice(&hidden_in),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.create_staging_buffer(bytes);

        let attn0 = engine.cpu().blocks()[0].attention();
        let intermediate = engine.cpu().blocks()[0].ffn().as_any()
            .downcast_ref::<crate::layers::swiglu::SwiGLU>().unwrap()
            .intermediate_size();
        let scratch = BlockScratch::allocate(
            &gpu, n_tokens, embed_dim,
            attn0.n_heads(), attn0.n_kv_heads(), attn0.head_dim(),
            intermediate, /*max_seq*/ n_tokens,
        );

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test.bitnet.encoder"),
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
        drop(data);
        staging.unmap();

        // Tolerance: per-token absmax quantization rounding accumulates
        // through Q/K/V/attention/o_proj/gate/up/down. ~1e-2 is realistic
        // for a single block with embed_dim=8 (small enough that one
        // unit of quantization noise scales meaningfully). The CPU
        // BitLinear uses the same absmax-then-quantize approach, so
        // discrepancy comes from FMA reordering + per-token-vs-per-
        // matvec accumulation order, not algorithmic differences.
        assert_eq!(cpu_out.len(), gpu_out.len(), "shape mismatch");
        for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
            assert!(
                (c - g).abs() < 1e-2,
                "bitnet block elem {i}: cpu={c} gpu={g} (diff={})",
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
    fn wrapper_metadata_passes_through() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let engine = GpuEngine::from_cpu_model(toy_model(), gpu);
        assert_eq!(engine.vocab_size(), 4);
        assert_eq!(engine.embed_dim(), 4);
        assert_eq!(engine.n_layers(), 1);
    }
}
