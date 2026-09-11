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

/// Review #17/#19: a tensor whose shape disagrees with the model config is
/// an `Err` naming the tensor, checked BEFORE any layer is built from it —
/// the layer constructors' `assert!`s then only guard programming errors.
fn expect_shape(t: &crate::tensor::FloatTensor, name: &str, expected: &[usize]) -> Result<(), GgufError> {
    if t.shape() != expected {
        return Err(GgufError::DimensionMismatch {
            tensor: name.to_string(),
            expected: expected.to_vec(),
            actual: t.shape().to_vec(),
        });
    }
    Ok(())
}

/// A 1-D float vector of exactly `len` elements (norm weights, biases).
fn load_vector(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, GgufError> {
    let t = gguf.load_float(name)?;
    expect_shape(&t, name, &[len])?;
    Ok(t.data().to_vec())
}

/// Review #19: an RMSNorm whose weight length is checked against the
/// model dimension at load. A short weight used to be accepted, panic on
/// the CPU path's first forward, and on the GPU be read past its end under
/// robust-buffer clamping — silently wrong output.
fn load_norm(gguf: &GgufFile, name: &str, dim: usize, eps: f32) -> Result<RmsNorm, GgufError> {
    Ok(RmsNorm::new(load_vector(gguf, name, dim)?, eps))
}

/// Load a weight tensor as a LinearLayer. Float-only after the
/// 2026-05-29 BitNet un-merge: all weights are dequantized to f32 at
/// load time (Q4_K / Q5_K / Q6_K / F16 / BF16 / F32). The GPU-resident
/// path keeps weights on the device for the model's lifetime.
/// `expected` is `[out_features, in_features]` — GGUF stores dims
/// innermost-first, so the parsed shape is already `[out, in]`.
fn load_linear_layer(
    gguf: &GgufFile,
    name: &str,
    ctx: &LoadCtx,
    expected: [usize; 2],
) -> Result<Box<dyn LinearLayer>, GgufError> {
    let _info = gguf
        .tensor_info(name)
        .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

    // Quantized or float → dequantize to f32
    let tensor = gguf.load_float(name)?;
    expect_shape(&tensor, name, &expected)?;
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

/// Review #18: the RoPE layout is a property of the architecture — what
/// llama.cpp calls `LLAMA_ROPE_TYPE_NORM` (adjacent pairs, "interleaved")
/// versus `LLAMA_ROPE_TYPE_NEOX` (first half / second half, "halved") —
/// not of `rope.scaling.type`, which the old code read (as the wrong type)
/// and treated as this selector. An architecture this table does not
/// know is refused with a clear message rather than loaded with a
/// guessed layout, which produces fluent garbage with no error;
/// `CORTEX_ROPE_LAYOUT=halved|interleaved` overrides for experiments.
pub fn rope_layout_for(arch: &str) -> Result<RoPELayout, GgufError> {
    if let Ok(v) = std::env::var("CORTEX_ROPE_LAYOUT") {
        return match v.as_str() {
            "halved" | "neox" => Ok(RoPELayout::Halved),
            "interleaved" | "norm" => Ok(RoPELayout::Interleaved),
            other => Err(GgufError::InvalidConfig {
                key: "CORTEX_ROPE_LAYOUT".to_string(),
                value: other.to_string(),
                reason: "expected halved|neox or interleaved|norm",
            }),
        };
    }
    // llama.cpp `llama_model_rope_type` (NEOX list).
    const HALVED: &[&str] = &[
        "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe",
        "phi2", "phi3", "phimoe", "gemma", "gemma2", "gemma3",
        "stablelm", "starcoder2", "gptneox", "falcon", "olmo2", "olmoe",
        "exaone", "nemotron", "orion", "codeshell", "dbrx", "grok", "plamo",
        "minicpm3", "openelm", "bitnet",
    ];
    // llama.cpp `LLAMA_ROPE_TYPE_NORM` list.
    const INTERLEAVED: &[&str] = &[
        "llama", "llama4", "mistral", "mixtral", "deci", "baichuan",
        "starcoder", "internlm2", "minicpm", "xverse", "command-r", "cohere2",
        "olmo", "arctic", "deepseek", "deepseek2", "chatglm", "granite",
        "granitemoe", "chameleon", "bailingmoe",
    ];
    if HALVED.contains(&arch) {
        Ok(RoPELayout::Halved)
    } else if INTERLEAVED.contains(&arch) {
        Ok(RoPELayout::Interleaved)
    } else {
        Err(GgufError::InvalidConfig {
            key: "general.architecture".to_string(),
            value: arch.to_string(),
            reason: "unknown architecture: its RoPE layout (NORM vs NEOX) is not in loader::rope_layout_for; \
                     add it there, or set CORTEX_ROPE_LAYOUT=halved|interleaved to experiment",
        })
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
    /// Shared GPU context the model's layers are tied to (when GPU available).
    /// Callers building a `GpuEngine` MUST reuse this `Arc` rather than calling
    /// `compute::detect_gpu_device()` again — a second device produces
    /// cross-device buffer-binding errors that surface as confusing wgpu
    /// validation panics (see #16).
    #[cfg(feature = "gpu")]
    pub gpu: Option<std::sync::Arc<crate::compute::wgpu_backend::GpuDevice>>,
}

/// Load a transformer model and tokenizer from a GGUF file.
///
/// The model file is **untrusted input**: any malformed field is an
/// `Err` naming it (review #13–#19); this never aborts or panics on file
/// contents.
pub fn load_model(path: &str) -> Result<LoadedModel, GgufError> {
    // Hardware detection and boot banner
    let hw = crate::compute::device::HardwareInfo::detect();
    hw.print_boot_banner();

    let gguf = GgufFile::open(path)?;

    // If a discrete GPU is present, stand up a shared GpuDevice so
    // float weights can be uploaded once and stay resident.
    #[cfg(feature = "gpu")]
    let gpu = compute::detect_gpu_device();
    #[cfg(feature = "gpu")]
    if gpu.is_some() {
        eprintln!("  [boot] Resident-weights runtime: enabled (GpuFloatLinear)");
        info!("GPU device available; float layers will be GPU-resident");
    }

    load_from_gguf(
        gguf,
        #[cfg(feature = "gpu")]
        gpu,
    )
}

/// Build the model and tokenizer from an already-parsed GGUF (any source —
/// file or `GgufFile::open_bytes`). `gpu = None` keeps every layer on the
/// CPU, which is how the loader's own tests run.
pub fn load_from_gguf(
    gguf: GgufFile,
    #[cfg(feature = "gpu")] gpu: Option<Arc<crate::compute::wgpu_backend::GpuDevice>>,
) -> Result<LoadedModel, GgufError> {
    let config = gguf.model_config()?;
    // Review #17: reject at load, before any allocation, anything the
    // loader or the forward pass would divide by or assert on.
    config.validate()?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    info!(
        vocab_size = config.vocab_size,
        embed_dim = config.embedding_dim,
        n_layers = config.n_layers,
        n_heads = config.n_heads,
        n_kv_heads = config.n_kv_heads,
        intermediate = config.intermediate_size,
        rope_theta = config.rope_theta,
        context_length = config.context_length,
        "loading model (context_length is the trained window; the KV window is --max-seq-len)"
    );

    let vocab = config.vocab_size as usize;
    let embed_dim = config.embedding_dim as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = config.head_dim() as usize;
    let intermediate = config.intermediate_size as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // The tokenizer must describe exactly the rows the embedding has.
    // Checked here, before any tensor is loaded (it used to be an
    // `assert_eq!` after the whole model was resident).
    if tokenizer.vocab_size() != vocab {
        return Err(GgufError::DimensionMismatch {
            tensor: "tokenizer.ggml.tokens".to_string(),
            expected: vec![vocab],
            actual: vec![tokenizer.vocab_size()],
        });
    }

    let arch = gguf
        .get_metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    // RoPE layout from the architecture (review #18).
    let rope_layout = rope_layout_for(arch)?;
    info!(arch, ?rope_layout, "RoPE layout");
    if let Some(scaling) = &config.rope_scaling {
        tracing::warn!(
            arch,
            scaling = %scaling,
            context_length = config.context_length,
            "the model declares RoPE scaling that cortex does not implement; \
             positions beyond the original (unscaled) context will degrade"
        );
    }
    // The FFN is SiLU-gated on every path; a model that needs another
    // activation must not load silently wrong (review #18).
    match config.hidden_act.as_str() {
        "silu" | "swiglu" => {}
        other => {
            return Err(GgufError::InvalidConfig {
                key: "hidden_act".to_string(),
                value: other.to_string(),
                reason: "only SiLU-gated FFNs (silu / swiglu) are supported",
            });
        }
    }

    let ctx = LoadCtx {
        #[cfg(feature = "gpu")]
        gpu: gpu.as_ref(),
        #[cfg(not(feature = "gpu"))]
        _marker: std::marker::PhantomData,
    };

    // Embedding table: exactly [vocab, embed].
    let embedding = gguf.load_float("token_embd.weight")?;
    expect_shape(&embedding, "token_embd.weight", &[vocab, embed_dim])?;
    info!("loaded embedding: {:?}", embedding.shape());

    // Transformer blocks (n_layers validated <= MAX_LAYERS)
    let mut blocks = Vec::with_capacity(config.n_layers as usize);

    for i in 0..config.n_layers as usize {
        let q_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_q.weight"), &ctx, [q_dim, embed_dim])?;
        let k_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_k.weight"), &ctx, [kv_dim, embed_dim])?;
        let v_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_v.weight"), &ctx, [kv_dim, embed_dim])?;
        let o_proj = load_linear_layer(&gguf, &format!("blk.{i}.attn_output.weight"), &ctx, [embed_dim, q_dim])?;

        let mut attention = MultiHeadAttention::with_rope_layout(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, config.rope_theta, rope_layout,
        );

        // Optional attention biases (Qwen2 has Q/K/V biases)
        if gguf.tensor_info(&format!("blk.{i}.attn_q.bias")).is_some() {
            let q_bias = load_vector(&gguf, &format!("blk.{i}.attn_q.bias"), q_dim)?;
            let k_bias = load_vector(&gguf, &format!("blk.{i}.attn_k.bias"), kv_dim)?;
            let v_bias = load_vector(&gguf, &format!("blk.{i}.attn_v.bias"), kv_dim)?;
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
                let e_gate = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.{e}.weight"), &ctx, [intermediate, embed_dim])?;
                let e_up = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.{e}.weight"), &ctx, [intermediate, embed_dim])?;
                let e_down = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.{e}.weight"), &ctx, [embed_dim, intermediate])?;
                experts.push(SwiGLU::new(e_gate, e_up, e_down));
            }

            let router = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate_inp.weight"), &ctx, [n_experts, embed_dim])?;

            if i == 0 {
                info!(n_experts, top_k, "loading MoE experts");
            }

            Box::new(crate::layers::moe::MoELayer::new(experts, router, top_k))
        } else {
            let gate_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_gate.weight"), &ctx, [intermediate, embed_dim])?;
            let up_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_up.weight"), &ctx, [intermediate, embed_dim])?;
            let down_proj = load_linear_layer(&gguf, &format!("blk.{i}.ffn_down.weight"), &ctx, [embed_dim, intermediate])?;
            Box::new(SwiGLU::new(gate_proj, up_proj, down_proj))
        };

        // Norms (float), length-checked against the model dimension (#19)
        let attn_norm = load_norm(&gguf, &format!("blk.{i}.attn_norm.weight"), embed_dim, config.rms_norm_eps)?;
        let ffn_norm = load_norm(&gguf, &format!("blk.{i}.ffn_norm.weight"), embed_dim, config.rms_norm_eps)?;

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

    // Final norm (#19: length-checked; it was compared to nothing at all)
    let final_norm = load_norm(&gguf, "output_norm.weight", embed_dim, config.rms_norm_eps)?;

    // Output projection
    let output_proj = if let Some(_out_info) = gguf.tensor_info("output.weight") {
        let out_tensor = gguf.load_float("output.weight")?;
        expect_shape(&out_tensor, "output.weight", &[vocab, embed_dim])?;
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

    debug_assert_eq!(model.vocab_size(), tokenizer.vocab_size(), "checked before loading");

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
        #[cfg(feature = "gpu")]
        gpu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::GgufBuilder;

    const F32: u32 = 0;

    /// Knobs for corrupting one field of the tiny model.
    #[derive(Default)]
    struct Corrupt {
        attn_norm_len: Option<usize>,
        final_norm_len: Option<usize>,
        q_shape: Option<[u64; 2]>,
        n_tokens: Option<usize>,
        head_count: Option<u32>,
        q_bias_len: Option<usize>,
        arch: Option<&'static str>,
        hidden_act: Option<&'static str>,
        rope_scaling: Option<&'static str>,
    }

    /// A complete, loadable toy model: vocab 8, embed 8, 2 layers, 2 heads,
    /// 1 kv head (head_dim 4), intermediate 16, tied output, SentencePiece
    /// tokenizer with default ids. Every tensor is F32 zeros.
    fn tiny_model(c: &Corrupt) -> Vec<u8> {
        let (vocab, embed, inter, n_layers) = (8usize, 8usize, 16usize, 2u32);
        let zeros = |n: usize| vec![0u8; n * 4];
        let arch = c.arch.unwrap_or("llama");
        let mut b = GgufBuilder::new();
        b.add_metadata_string("general.architecture", arch);
        b.add_metadata_u32(&format!("{arch}.embedding_length"), embed as u32);
        b.add_metadata_u32(&format!("{arch}.block_count"), n_layers);
        b.add_metadata_u32(&format!("{arch}.attention.head_count"), c.head_count.unwrap_or(2));
        b.add_metadata_u32(&format!("{arch}.attention.head_count_kv"), 1);
        b.add_metadata_u32(&format!("{arch}.context_length"), 16);
        b.add_metadata_u32(&format!("{arch}.feed_forward_length"), inter as u32);
        b.add_metadata_f32(&format!("{arch}.rope.freq_base"), 10000.0);
        b.add_metadata_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);
        if let Some(act) = c.hidden_act {
            b.add_metadata_string("general.hidden_act", act);
        }
        if let Some(s) = c.rope_scaling {
            b.add_metadata_string(&format!("{arch}.rope.scaling.type"), s);
        }
        let names: Vec<String> = (0..c.n_tokens.unwrap_or(vocab)).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        b.add_metadata_array_string("tokenizer.ggml.tokens", &refs);
        b.add_metadata_string("tokenizer.ggml.model", "llama");

        b.add_tensor("token_embd.weight", &[vocab as u64, embed as u64], F32, zeros(vocab * embed));
        for i in 0..n_layers {
            let q = c.q_shape.unwrap_or([embed as u64, embed as u64]);
            b.add_tensor(&format!("blk.{i}.attn_q.weight"), &q, F32, zeros((q[0] * q[1]) as usize));
            b.add_tensor(&format!("blk.{i}.attn_k.weight"), &[4, embed as u64], F32, zeros(4 * embed));
            b.add_tensor(&format!("blk.{i}.attn_v.weight"), &[4, embed as u64], F32, zeros(4 * embed));
            b.add_tensor(&format!("blk.{i}.attn_output.weight"), &[embed as u64, embed as u64], F32, zeros(embed * embed));
            if let Some(n) = c.q_bias_len {
                b.add_tensor(&format!("blk.{i}.attn_q.bias"), &[n as u64], F32, zeros(n));
                b.add_tensor(&format!("blk.{i}.attn_k.bias"), &[4], F32, zeros(4));
                b.add_tensor(&format!("blk.{i}.attn_v.bias"), &[4], F32, zeros(4));
            }
            b.add_tensor(&format!("blk.{i}.ffn_gate.weight"), &[inter as u64, embed as u64], F32, zeros(inter * embed));
            b.add_tensor(&format!("blk.{i}.ffn_up.weight"), &[inter as u64, embed as u64], F32, zeros(inter * embed));
            b.add_tensor(&format!("blk.{i}.ffn_down.weight"), &[embed as u64, inter as u64], F32, zeros(embed * inter));
            let an = if i == 0 { c.attn_norm_len.unwrap_or(embed) } else { embed };
            b.add_tensor(&format!("blk.{i}.attn_norm.weight"), &[an as u64], F32, zeros(an));
            b.add_tensor(&format!("blk.{i}.ffn_norm.weight"), &[embed as u64], F32, zeros(embed));
        }
        let fnl = c.final_norm_len.unwrap_or(embed);
        b.add_tensor("output_norm.weight", &[fnl as u64], F32, zeros(fnl));
        b.build()
    }

    fn load(bytes: Vec<u8>) -> Result<LoadedModel, GgufError> {
        let gguf = GgufFile::open_bytes(bytes)?;
        load_from_gguf(
            gguf,
            #[cfg(feature = "gpu")]
            None,
        )
    }

    #[test]
    fn tiny_model_loads_on_the_cpu_path() {
        let m = load(tiny_model(&Corrupt::default())).expect("tiny model must load");
        assert_eq!(m.model.n_layers(), 2);
        assert_eq!(m.model.vocab_size(), 8);
        assert_eq!(m.model.embed_dim(), 8);
        assert_eq!(m.tokenizer.vocab_size(), 8);
        // With biases too.
        let m = load(tiny_model(&Corrupt { q_bias_len: Some(8), ..Default::default() })).unwrap();
        assert_eq!(m.model.n_layers(), 2);
    }

    fn mismatch_tensor(r: Result<LoadedModel, GgufError>) -> String {
        match r {
            Err(GgufError::DimensionMismatch { tensor, .. }) => tensor,
            Err(other) => panic!("expected DimensionMismatch, got {other}"),
            Ok(_) => panic!("expected DimensionMismatch, got Ok"),
        }
    }

    #[test]
    fn short_attn_norm_is_rejected_at_load() {
        // Review #19: this used to load and panic on the first forward.
        let t = mismatch_tensor(load(tiny_model(&Corrupt { attn_norm_len: Some(7), ..Default::default() })));
        assert_eq!(t, "blk.0.attn_norm.weight");
    }

    #[test]
    fn short_final_norm_is_rejected_at_load() {
        // Review #19: final_norm was compared to nothing at all.
        let t = mismatch_tensor(load(tiny_model(&Corrupt { final_norm_len: Some(7), ..Default::default() })));
        assert_eq!(t, "output_norm.weight");
    }

    #[test]
    fn wrong_projection_shape_is_rejected_at_load() {
        let t = mismatch_tensor(load(tiny_model(&Corrupt { q_shape: Some([8, 7]), ..Default::default() })));
        assert_eq!(t, "blk.0.attn_q.weight");
    }

    #[test]
    fn wrong_bias_length_is_rejected_at_load() {
        let t = mismatch_tensor(load(tiny_model(&Corrupt { q_bias_len: Some(6), ..Default::default() })));
        assert_eq!(t, "blk.0.attn_q.bias");
    }

    #[test]
    fn tokenizer_vocab_mismatch_is_rejected_before_any_tensor() {
        // vocab_size derives from the token list (9) but the embedding has
        // 8 rows: used to be an assert_eq! after the whole model was built.
        let t = mismatch_tensor(load(tiny_model(&Corrupt { n_tokens: Some(9), ..Default::default() })));
        assert_eq!(t, "token_embd.weight");
    }

    #[test]
    fn rope_layout_comes_from_the_architecture_table() {
        // Review #18: rope.scaling.type was read as a u32 (always 0) and only
        // an `arch.contains("qwen")` heuristic kept Qwen on NEOX.
        for a in ["qwen2", "qwen3", "phi3", "gemma2", "stablelm", "starcoder2", "gptneox", "olmo2"] {
            assert!(matches!(rope_layout_for(a), Ok(RoPELayout::Halved)), "{a}");
        }
        for a in ["llama", "mistral", "deepseek2", "granite", "internlm2", "olmo", "command-r"] {
            assert!(matches!(rope_layout_for(a), Ok(RoPELayout::Interleaved)), "{a}");
        }
        match rope_layout_for("made-up-arch") {
            Err(GgufError::InvalidConfig { key, .. }) => assert_eq!(key, "general.architecture"),
            other => panic!("expected InvalidConfig, got {:?}", other.map(|_| ())),
        }
        // An unknown architecture refuses to load (a guessed layout is
        // fluent garbage with no error).
        let mut arch_model = tiny_model(&Corrupt { arch: Some("made-up-arch"), ..Default::default() });
        match load(std::mem::take(&mut arch_model)) {
            Err(GgufError::InvalidConfig { key, .. }) => assert_eq!(key, "general.architecture"),
            other => panic!("expected InvalidConfig, got {:?}", other.map(|_| ())),
        }
        // Known NEOX architecture loads.
        assert!(load(tiny_model(&Corrupt { arch: Some("qwen2"), ..Default::default() })).is_ok());
    }

    #[test]
    fn rope_scaling_is_read_as_a_string_and_only_warns() {
        let m = load(tiny_model(&Corrupt { rope_scaling: Some("yarn"), ..Default::default() })).unwrap();
        assert_eq!(m.config.rope_scaling.as_deref(), Some("yarn"));
        let m = load(tiny_model(&Corrupt { rope_scaling: Some("none"), ..Default::default() })).unwrap();
        assert_eq!(m.config.rope_scaling, None);
    }

    #[test]
    fn unsupported_hidden_act_is_rejected_at_load() {
        // A relu2 model used to load with a SiLU FFN and run silently wrong.
        match load(tiny_model(&Corrupt { hidden_act: Some("relu2"), ..Default::default() })) {
            Err(GgufError::InvalidConfig { key, value, .. }) => {
                assert_eq!(key, "hidden_act");
                assert_eq!(value, "relu2");
            }
            other => panic!("expected InvalidConfig, got {:?}", other.map(|_| ())),
        }
        assert!(load(tiny_model(&Corrupt { hidden_act: Some("silu"), ..Default::default() })).is_ok());
    }

    #[test]
    fn invalid_config_is_rejected_before_any_tensor() {
        // Review #17: head_count = 0 used to divide by zero at loader.rs:114.
        match load(tiny_model(&Corrupt { head_count: Some(0), ..Default::default() })) {
            Err(GgufError::InvalidConfig { key, .. }) => assert_eq!(key, "attention.head_count"),
            other => panic!("expected InvalidConfig, got {:?}", other.map(|_| ())),
        }
    }
}
