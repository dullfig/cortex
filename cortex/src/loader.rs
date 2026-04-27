//! Model loader — wires GGUF tensor data into a live TransformerModel.
//!
//! LLaMA/BitNet GGUF tensor naming convention:
//!
//! - `token_embd.weight` — embedding table (float)
//! - `blk.{i}.attn_q.weight` — Q projection (ternary)
//! - `blk.{i}.attn_k.weight` — K projection (ternary)
//! - `blk.{i}.attn_v.weight` — V projection (ternary)
//! - `blk.{i}.attn_output.weight` — O projection (ternary)
//! - `blk.{i}.ffn_gate.weight` — SwiGLU gate (ternary)
//! - `blk.{i}.ffn_up.weight` — SwiGLU up (ternary)
//! - `blk.{i}.ffn_down.weight` — SwiGLU down (ternary)
//! - `blk.{i}.attn_norm.weight` — attention RMSNorm (float)
//! - `blk.{i}.ffn_norm.weight` — FFN RMSNorm (float)
//! - `blk.{i}.attn_sub_norm.weight` — post-attention sub-norm (BitNet b1.58)
//! - `blk.{i}.ffn_sub_norm.weight` — pre-down-projection sub-norm (BitNet b1.58)
//! - `output_norm.weight` — final RMSNorm (float)
//! - `output.weight` — output projection (ternary, or absent if tied)

use std::sync::Arc;

use tracing::info;

use crate::compute::{self, ComputeBackend};
use crate::gguf::{GgufFile, GgufError, ModelConfig};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::bitlinear::BitLinear;
use crate::layers::floatlinear::FloatLinear;
use crate::layers::linear::LinearLayer;
use crate::layers::model::{OutputProjection, TransformerModel};
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::RoPELayout;
use crate::layers::swiglu::{GateActivation, SwiGLU};
use crate::layers::transformer::TransformerBlock;
use crate::tokenizer::Tokenizer;

/// Bundles the compute resources passed to every layer constructor:
/// the always-present CPU `ComputeBackend` and (with `gpu` feature) an
/// optional shared `GpuDevice` for resident weights.
struct LoadCtx<'a> {
    backend: &'a Arc<dyn ComputeBackend>,
    #[cfg(feature = "gpu")]
    gpu: Option<&'a Arc<crate::compute::wgpu_backend::GpuDevice>>,
}

/// Load a weight tensor as a LinearLayer, auto-detecting its type.
///
/// - Ternary (TQ1_0, TQ2_0, I2S) → GpuBitLinear if a GpuDevice is in ctx, else BitLinear
/// - Quantized (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, etc.) → FloatLinear (dequantized at load time)
/// - Float (F32, F16, BF16) → FloatLinear
///
/// The GPU-resident path keeps weights on the device for the model's lifetime.
/// FloatLinear stays CPU-side for now; a `GpuFloatLinear` lands when the
/// resident runtime is extended to f16-packed weights.
fn load_linear_layer(
    gguf: &GgufFile,
    name: &str,
    ctx: &LoadCtx,
) -> Result<Box<dyn LinearLayer>, GgufError> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

    if info.ggml_type.is_ternary() {
        let (w, s) = gguf.load_ternary(name)?;
        #[cfg(feature = "gpu")]
        if let Some(gpu) = ctx.gpu {
            return Ok(Box::new(crate::layers::gpu_bitlinear::GpuBitLinear::from_weights(
                gpu.clone(), w, s,
            )));
        }
        Ok(Box::new(BitLinear::with_backend(w, s, ctx.backend.clone())))
    } else {
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
}

/// A fully loaded model ready for inference.
pub struct LoadedModel {
    /// The transformer model.
    pub model: TransformerModel,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
    /// Model hyperparameters.
    pub config: ModelConfig,
}

/// Load a transformer model and tokenizer from a GGUF file.
///
/// This reads all tensor data into memory — for a 2B model at ternary
/// precision, that's roughly ~400MB of weights.
pub fn load_model(path: &str) -> Result<LoadedModel, GgufError> {
    // Hardware detection and boot banner
    let hw = crate::compute::device::HardwareInfo::detect();
    hw.print_boot_banner();

    let gguf = GgufFile::open(path)?;
    let config = gguf.model_config()?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    eprintln!(
        "  [boot] Model: {} layers, {} embed, {} vocab, {:.0}MB weights",
        config.n_layers,
        config.embedding_dim,
        config.vocab_size,
        // Ternary weights: 2 bits per param, plus float norms/embeddings
        (config.vocab_size * config.embedding_dim) as f64 * 4.0 / (1024.0 * 1024.0)
            + (config.n_layers * config.embedding_dim * config.embedding_dim * 7) as f64 * 0.25
                / (1024.0 * 1024.0),
    );

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

    // Detect architecture string (used for RoPE and activation inference)
    let arch = gguf
        .get_metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    // Determine RoPE layout.
    // BitNet models from HuggingFace use halved (NeoX) RoPE convention.
    // BitNet.cpp's converter does NOT apply the Q/K weight permutation,
    // so we need halved RoPE for bitnet architectures.
    // Determine RoPE layout.
    // Most HuggingFace models (Qwen2, BitNet, etc.) use halved (NeoX) RoPE.
    // Only pure llama.cpp-converted LLaMA models use interleaved.
    let rope_layout = match config.rope_type {
        2 => {
            info!("using halved (NeoX/HF) RoPE layout (from rope_type=2)");
            RoPELayout::Halved
        }
        _ if arch.contains("bitnet") || arch.contains("qwen") => {
            info!(arch, "using halved (NeoX/HF) RoPE layout (architecture default)");
            RoPELayout::Halved
        }
        _ => {
            info!("using interleaved (llama.cpp) RoPE layout");
            RoPELayout::Interleaved
        }
    };

    let activation = match config.hidden_act.as_str() {
        "relu2" | "relu_squared" | "squared_relu" => {
            info!("using squared ReLU (relu²) activation (from metadata)");
            GateActivation::ReLU2
        }
        "silu" if arch.contains("bitnet") => {
            // Metadata defaulted to "silu" but architecture is bitnet → use relu²
            info!("detected bitnet architecture, using squared ReLU (relu²) activation");
            GateActivation::ReLU2
        }
        other => {
            info!(act = %other, arch, "using SiLU activation");
            GateActivation::SiLU
        }
    };

    // Auto-detect compute backend (scalar / AVX2 / wgpu)
    let backend = compute::detect();
    eprintln!("  [boot] Compute: {} ternary kernel", backend.name());
    info!(backend = backend.name(), "compute backend selected");

    // If a discrete GPU is present, also stand up a shared GpuDevice so
    // ternary weights can be uploaded once and stay resident.
    #[cfg(feature = "gpu")]
    let gpu = compute::detect_gpu_device();
    #[cfg(feature = "gpu")]
    if gpu.is_some() {
        eprintln!("  [boot] Resident-weights runtime: enabled (GpuBitLinear + GpuFloatLinear)");
        info!("GPU device available; ternary and float layers will be GPU-resident");
    }

    let ctx = LoadCtx {
        backend: &backend,
        #[cfg(feature = "gpu")]
        gpu: gpu.as_ref(),
    };

    // Embedding table
    let embedding = gguf.load_float("token_embd.weight")?;
    info!("loaded embedding: {:?}", embedding.shape());

    // Transformer blocks
    let mut blocks = Vec::with_capacity(config.n_layers as usize);

    for i in 0..config.n_layers as usize {
        // Attention projections — auto-detect ternary vs quantized vs float
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

        // Build FFN: MoE or dense SwiGLU
        let ffn: Box<dyn crate::layers::ffn::FeedForward> = if let Some(n_experts) = config.expert_count {
            // MoE: load per-expert weights and router
            let n_experts = n_experts as usize;
            let top_k = config.expert_used_count.unwrap_or(2) as usize;
            let mut experts = Vec::with_capacity(n_experts);

            for e in 0..n_experts {
                let e_gate = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.{e}.weight"), &ctx)?;
                let e_up = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.{e}.weight"), &ctx)?;
                let e_down = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.{e}.weight"), &ctx)?;
                experts.push(SwiGLU::with_activation(e_gate, e_up, e_down, activation));
            }

            let router = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate_inp.weight"), &ctx)?;

            if i == 0 {
                info!(n_experts, top_k, "loading MoE experts");
            }

            Box::new(crate::layers::moe::MoELayer::new(experts, router, top_k))
        } else {
            // Dense FFN projections — auto-detect type
            let gate_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.weight"), &ctx)?;
            let up_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.weight"), &ctx)?;
            let down_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.weight"), &ctx)?;

            // Optional ffn_sub_norm [intermediate_dim]: applied between activation(gate)⊙up and down
            if gguf.tensor_info(&format!("blk.{i}.ffn_sub_norm.weight")).is_some() {
                let w = gguf.load_float(&format!("blk.{i}.ffn_sub_norm.weight"))?;
                let sub_norm = RmsNorm::new(w.data().to_vec(), config.rms_norm_eps);
                Box::new(SwiGLU::with_sub_norm_and_activation(gate_proj, up_proj, down_proj, sub_norm, activation))
            } else {
                Box::new(SwiGLU::with_activation(gate_proj, up_proj, down_proj, activation))
            }
        };

        // Norms (float)
        let attn_norm_w = gguf.load_float(&format!("blk.{i}.attn_norm.weight"))?;
        let ffn_norm_w = gguf.load_float(&format!("blk.{i}.ffn_norm.weight"))?;

        let attn_norm = RmsNorm::new(attn_norm_w.data().to_vec(), config.rms_norm_eps);
        let ffn_norm = RmsNorm::new(ffn_norm_w.data().to_vec(), config.rms_norm_eps);

        // Optional attn_sub_norm [embed_dim]: applied to attention output before residual
        let attn_sub_norm = if gguf.tensor_info(&format!("blk.{i}.attn_sub_norm.weight")).is_some() {
            let w = gguf.load_float(&format!("blk.{i}.attn_sub_norm.weight"))?;
            Some(RmsNorm::new(w.data().to_vec(), config.rms_norm_eps))
        } else {
            None
        };

        let block = TransformerBlock::with_sub_norms(
            attn_norm, attention, attn_sub_norm, ffn_norm, ffn,
        );
        blocks.push(block);

        info!(layer = i, "loaded transformer block {}/{}", i + 1, config.n_layers);
    }

    // Final norm
    let final_norm_w = gguf.load_float("output_norm.weight")?;
    let final_norm = RmsNorm::new(final_norm_w.data().to_vec(), config.rms_norm_eps);

    // Output projection: check if "output.weight" exists and its type
    let output_proj = if let Some(out_info) = gguf.tensor_info("output.weight") {
        if out_info.ggml_type.is_ternary() {
            // Ternary output projection → GpuBitLinear if available, else BitLinear
            let (out_w, out_s) = gguf.load_ternary("output.weight")?;
            info!(
                rows = out_w.rows(),
                cols = out_w.cols(),
                "loaded output projection (ternary)"
            );
            #[cfg(feature = "gpu")]
            let layer: Box<dyn LinearLayer> = if let Some(gpu) = ctx.gpu {
                Box::new(crate::layers::gpu_bitlinear::GpuBitLinear::from_weights(
                    gpu.clone(), out_w, out_s,
                ))
            } else {
                Box::new(BitLinear::with_backend(out_w, out_s, backend.clone()))
            };
            #[cfg(not(feature = "gpu"))]
            let layer: Box<dyn LinearLayer> =
                Box::new(BitLinear::with_backend(out_w, out_s, backend.clone()));
            OutputProjection::Linear(layer)
        } else {
            // Float or quantized → dequantize to f32 tensor
            let out_tensor = gguf.load_float("output.weight")?;
            info!(
                shape = ?out_tensor.shape(),
                dtype = ?out_info.ggml_type,
                "loaded output projection (float)"
            );
            // Route through GpuFloatLinear when GPU available (resident weights,
            // GPU matvec for the vocab-sized projection — Qwen vocab=151k rows).
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
        }
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

    // Sanity check
    assert_eq!(
        model.vocab_size(),
        tokenizer.vocab_size(),
        "model vocab size ({}) != tokenizer vocab size ({})",
        model.vocab_size(),
        tokenizer.vocab_size()
    );

    let _ = intermediate; // used implicitly through GGUF tensor shapes

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
    })
}
