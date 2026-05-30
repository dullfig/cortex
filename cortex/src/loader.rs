//! Model loader — wires GGUF tensor data into a live TransformerModel.
//!
//! Float-only loader (Qwen-class GGUF). Ternary loading moved to the
//! ternary-rs crate on 2026-05-29 (see `bitnet-archive-2026-05-29`
//! cortex tag for the last loader.rs with the ternary path).
//!
//! LLaMA/Qwen GGUF tensor naming convention:
//!
//! - `token_embd.weight` — embedding table (float)
//! - `blk.{i}.attn_q.weight` — Q projection (Q4_K / F16 / F32)
//! - `blk.{i}.attn_k.weight` — K projection
//! - `blk.{i}.attn_v.weight` — V projection
//! - `blk.{i}.attn_output.weight` — O projection
//! - `blk.{i}.ffn_gate.weight` — SwiGLU gate
//! - `blk.{i}.ffn_up.weight` — SwiGLU up
//! - `blk.{i}.ffn_down.weight` — SwiGLU down
//! - `blk.{i}.attn_norm.weight` — attention RMSNorm
//! - `blk.{i}.ffn_norm.weight` — FFN RMSNorm
//! - `output_norm.weight` — final RMSNorm
//! - `output.weight` — output projection (or absent if tied)

use std::sync::Arc;

use tracing::info;

use crate::compute;
use crate::gguf::{GgufFile, GgufError, ModelConfig};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::floatlinear::FloatLinear;
use crate::layers::linear::LinearLayer;
use crate::layers::model::{OutputProjection, TransformerModel};
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::RoPELayout;
use crate::layers::swiglu::SwiGLU;
use crate::layers::transformer::TransformerBlock;
use crate::tokenizer::Tokenizer;

/// Bundles the compute resources passed to every layer constructor.
struct LoadCtx<'a> {
    #[cfg(feature = "gpu")]
    gpu: Option<&'a Arc<crate::compute::wgpu_backend::GpuDevice>>,
    #[cfg(not(feature = "gpu"))]
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Load a weight tensor as a LinearLayer. Float-only after the
/// 2026-05-29 BitNet un-merge: all weights are dequantized to f32 at
/// load time (Q4_K / Q5_K / Q6_K / F16 / BF16 / F32). The GPU-resident
/// path keeps weights on the device for the model's lifetime.
fn load_linear_layer(
    gguf: &GgufFile,
    name: &str,
    ctx: &LoadCtx,
) -> Result<Box<dyn LinearLayer>, GgufError> {
    let _info = gguf
        .tensor_info(name)
        .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

    // Quantized or float → dequantize to f32
    let tensor = gguf.load_float(name)?;
    #[cfg(feature = "gpu")]
    if let Some(gpu) = ctx.gpu {
        // f16 packing requires even in_features; fall back to CPU if odd.
        if tensor.shape().len() == 2 && tensor.shape()[1] % 2 == 0 {
            return Ok(Box::new(crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
                gpu.clone(), tensor,
            )));
        }
    }
    Ok(Box::new(FloatLinear::from_float_tensor(tensor)))
}

/// A fully loaded model ready for inference.
pub struct LoadedModel {
    /// The transformer model.
    pub model: TransformerModel,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
    /// Model hyperparameters.
    pub config: ModelConfig,
    /// Shared GPU context the model's layers are tied to (when GPU available).
    /// Callers building a `GpuEngine` MUST reuse this `Arc` rather than calling
    /// `compute::detect_gpu_device()` again — a second device produces
    /// cross-device buffer-binding errors that surface as confusing wgpu
    /// validation panics (see #16).
    #[cfg(feature = "gpu")]
    pub gpu: Option<std::sync::Arc<crate::compute::wgpu_backend::GpuDevice>>,
}

/// Load a transformer model and tokenizer from a GGUF file.
pub fn load_model(path: &str) -> Result<LoadedModel, GgufError> {
    // Hardware detection and boot banner
    let hw = crate::compute::device::HardwareInfo::detect();
    hw.print_boot_banner();

    let gguf = GgufFile::open(path)?;
    let config = gguf.model_config()?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    info!(
        vocab_size = config.vocab_size,
        embed_dim = config.embedding_dim,
        n_layers = config.n_layers,
        n_heads = config.n_heads,
        n_kv_heads = config.n_kv_heads,
        intermediate = config.intermediate_size,
        rope_theta = config.rope_theta,
        "loading model"
    );

    let embed_dim = config.embedding_dim as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = embed_dim / n_heads;
    let intermediate = config.intermediate_size as usize;

    let arch = gguf
        .get_metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    // RoPE layout: HuggingFace models (Qwen2 etc.) use halved (NeoX).
    // Only pure llama.cpp-converted LLaMA models use interleaved.
    let rope_layout = match config.rope_type {
        2 => {
            info!("using halved (NeoX/HF) RoPE layout (from rope_type=2)");
            RoPELayout::Halved
        }
        _ if arch.contains("qwen") => {
            info!(arch, "using halved (NeoX/HF) RoPE layout (architecture default)");
            RoPELayout::Halved
        }
        _ => {
            info!("using interleaved (llama.cpp) RoPE layout");
            RoPELayout::Interleaved
        }
    };

    // If a discrete GPU is present, stand up a shared GpuDevice so
    // float weights can be uploaded once and stay resident.
    #[cfg(feature = "gpu")]
    let gpu = compute::detect_gpu_device();
    #[cfg(feature = "gpu")]
    if gpu.is_some() {
        eprintln!("  [boot] Resident-weights runtime: enabled (GpuFloatLinear)");
        info!("GPU device available; float layers will be GPU-resident");
    }

    let ctx = LoadCtx {
        #[cfg(feature = "gpu")]
        gpu: gpu.as_ref(),
        #[cfg(not(feature = "gpu"))]
        _marker: std::marker::PhantomData,
    };

    // Embedding table
    let embedding = gguf.load_float("token_embd.weight")?;
    info!("loaded embedding: {:?}", embedding.shape());

    // Transformer blocks
    let mut blocks = Vec::with_capacity(config.n_layers as usize);

    for i in 0..config.n_layers as usize {
        let q_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_q.weight"), &ctx)?;
        let k_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_k.weight"), &ctx)?;
        let v_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_v.weight"), &ctx)?;
        let o_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_output.weight"), &ctx)?;

        let mut attention = MultiHeadAttention::with_rope_layout(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, config.rope_theta, rope_layout,
        );

        // Optional attention biases (Qwen2 has Q/K/V biases)
        if gguf.tensor_info(&format!("blk.{i}.attn_q.bias")).is_some() {
            let q_bias = gguf.load_float(&format!("blk.{i}.attn_q.bias"))?.data().to_vec();
            let k_bias = gguf.load_float(&format!("blk.{i}.attn_k.bias"))?.data().to_vec();
            let v_bias = gguf.load_float(&format!("blk.{i}.attn_v.bias"))?.data().to_vec();
            if i == 0 {
                info!("loading attention biases (Q/K/V)");
            }
            attention.set_biases(q_bias, k_bias, v_bias);
        }

        // Build FFN: MoE or dense SwiGLU.
        let ffn: Box<dyn crate::layers::ffn::FeedForward> = if let Some(n_experts) = config.expert_count {
            let n_experts = n_experts as usize;
            let top_k = config.expert_used_count.unwrap_or(2) as usize;
            let mut experts = Vec::with_capacity(n_experts);

            for e in 0..n_experts {
                let e_gate = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.{e}.weight"), &ctx)?;
                let e_up = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.{e}.weight"), &ctx)?;
                let e_down = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.{e}.weight"), &ctx)?;
                experts.push(SwiGLU::new(e_gate, e_up, e_down));
            }

            let router = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate_inp.weight"), &ctx)?;

            if i == 0 {
                info!(n_experts, top_k, "loading MoE experts");
            }

            Box::new(crate::layers::moe::MoELayer::new(experts, router, top_k))
        } else {
            let gate_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.weight"), &ctx)?;
            let up_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.weight"), &ctx)?;
            let down_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.weight"), &ctx)?;
            Box::new(SwiGLU::new(gate_proj, up_proj, down_proj))
        };

        // Norms (float)
        let attn_norm_w = gguf.load_float(&format!("blk.{i}.attn_norm.weight"))?;
        let ffn_norm_w = gguf.load_float(&format!("blk.{i}.ffn_norm.weight"))?;

        let attn_norm = RmsNorm::new(attn_norm_w.data().to_vec(), config.rms_norm_eps);
        let ffn_norm = RmsNorm::new(ffn_norm_w.data().to_vec(), config.rms_norm_eps);

        let block = TransformerBlock::new(attn_norm, attention, ffn_norm, ffn);
        blocks.push(block);

        // wgpu 29 mitigation: queue.write_buffer enqueues into wgpu's
        // staging belt; the staging chunks are NOT recycled until the
        // queue receives a submit that drains them. During model load
        // we never submit, so the belt accumulates unbounded — fix by
        // submitting an empty command buffer between block loads.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = ctx.gpu.as_ref() {
            gpu.queue.submit(std::iter::empty());
            let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        }

        info!(layer = i, "loaded transformer block {}/{}", i + 1, config.n_layers);
    }

    // Final norm
    let final_norm_w = gguf.load_float("output_norm.weight")?;
    let final_norm = RmsNorm::new(final_norm_w.data().to_vec(), config.rms_norm_eps);

    // Output projection
    let output_proj = if let Some(_out_info) = gguf.tensor_info("output.weight") {
        let out_tensor = gguf.load_float("output.weight")?;
        info!(shape = ?out_tensor.shape(), "loaded output projection (float)");
        #[cfg(feature = "gpu")]
        if let Some(gpu) = ctx.gpu {
            if out_tensor.shape().len() == 2 && out_tensor.shape()[1] % 2 == 0 {
                let gpu_layer = crate::layers::gpu_floatlinear::GpuFloatLinear::from_float_tensor(
                    gpu.clone(), out_tensor,
                );
                OutputProjection::Linear(Box::new(gpu_layer))
            } else {
                OutputProjection::Float(out_tensor)
            }
        } else {
            OutputProjection::Float(out_tensor)
        }
        #[cfg(not(feature = "gpu"))]
        OutputProjection::Float(out_tensor)
    } else {
        info!("using tied embedding for output projection");
        OutputProjection::TiedEmbedding
    };

    let model = TransformerModel::new(embedding, blocks, final_norm, output_proj);
    info!(
        vocab = model.vocab_size(),
        embed = model.embed_dim(),
        layers = model.n_layers(),
        "model loaded successfully"
    );

    assert_eq!(
        model.vocab_size(),
        tokenizer.vocab_size(),
        "model vocab size ({}) != tokenizer vocab size ({})",
        model.vocab_size(),
        tokenizer.vocab_size()
    );

    let _ = intermediate;

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
        #[cfg(feature = "gpu")]
        gpu,
    })
}
