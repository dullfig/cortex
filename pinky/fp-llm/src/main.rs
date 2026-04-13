//! fp-llm — Field-Programmable LLM experiment
//!
//! Tests whether injecting per-layer hidden-state deltas (captured from
//! an instruct model) into a base model via FfnInjector can nudge the
//! base model toward instruction-following behavior without fine-tuning.
//!
//! The experiment:
//! 1. Load the instruct model, run a prompt, capture hidden states
//! 2. Unload the instruct model (free memory)
//! 3. Load the base model, run the same prompt, capture hidden states
//! 4. Compute per-layer deltas: instruct_hidden - base_hidden
//! 5. Run the base model again with a DeltaInjector that adds the
//!    deltas at late-layer FFN outputs
//! 6. Compare: does the base model's output shift from "text completion"
//!    toward "instruction following"?
//!
//! Usage:
//!   fp-llm \
//!     --instruct-model /path/to/qwen2.5-0.5b-instruct-q8_0.gguf \
//!     --base-model /path/to/qwen2.5-0.5b-base-q8_0.gguf \
//!     [--inject-from-layer 16] \
//!     [--scale 1.0]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;
use cortex::layers::model::TransformerModel;
use cortex::layers::sampler::SamplerConfig;
use cortex::layers::transformer::FfnInjector;
use cortex::Tokenizer;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "fp-llm")]
struct Cli {
    /// Path to the instruct model GGUF (source of instruction-following behavior).
    #[arg(long)]
    instruct_model: PathBuf,

    /// Path to the base model GGUF (the model we want to nudge).
    #[arg(long)]
    base_model: PathBuf,

    /// First layer to inject deltas at (0-indexed). Layers before this
    /// are untouched. Gemini's analysis recommends the upper third
    /// (layers 16-23 for a 24-layer model).
    #[arg(long, default_value = "16")]
    inject_from_layer: usize,

    /// Scale factor for the injected deltas. 1.0 = full delta.
    /// Lower values give a gentler nudge; higher values amplify.
    #[arg(long, default_value = "1.0")]
    scale: f32,

    /// Maximum tokens to generate.
    #[arg(long, default_value = "100")]
    max_tokens: usize,
}

// ---------------------------------------------------------------------------
// DeltaInjector — the FfnInjector that nudges the base model
// ---------------------------------------------------------------------------

/// Injects precomputed per-position deltas into the FFN output.
///
/// The deltas are the difference between the instruct model's hidden
/// states and the base model's hidden states at the same layer and
/// position. Adding them to the base model's FFN output "nudges" the
/// residual stream toward the instruct model's representation.
struct DeltaInjector {
    /// Per-position delta vectors, shape [seq_len * embed_dim].
    /// These are (instruct_hidden[layer] - base_hidden[layer]) for
    /// the specific layer this injector is attached to.
    deltas: Vec<f32>,
    embed_dim: usize,
    scale: f32,
}

impl FfnInjector for DeltaInjector {
    fn inject(&self, _hidden_normed: &[f32], ffn_output: &mut [f32]) {
        // ffn_output is [embed_dim] for a single token position.
        // We need to know which position we're at, but the FfnInjector
        // interface doesn't tell us. However, in the cached forward pass,
        // inject() is called once per token in the sequence, in order.
        // We use an atomic counter to track which position we're injecting.
        //
        // For the v0 experiment, we inject the MEAN delta across all
        // positions rather than per-position deltas. This is a simplification
        // that tests whether the overall "direction" of instruction-following
        // is capturable as a single vector per layer, independent of position.
        // If this works, per-position injection is a refinement.
        let n_positions = self.deltas.len() / self.embed_dim;
        if n_positions == 0 {
            return;
        }

        // Compute mean delta across positions for this layer
        for d in 0..self.embed_dim {
            let mut sum = 0.0f32;
            for pos in 0..n_positions {
                sum += self.deltas[pos * self.embed_dim + d];
            }
            let mean_delta = sum / n_positions as f32;
            ffn_output[d] += mean_delta * self.scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is barbershop music? Explain briefly.<|im_end|>\n<|im_start|>assistant\n";

    println!("================================================================");
    println!("FP-LLM EXPERIMENT: Can residual injection create instruction-following?");
    println!("================================================================");
    println!();
    println!("prompt: {:?}", &prompt[..80.min(prompt.len())]);
    println!("inject from layer: {}", cli.inject_from_layer);
    println!("delta scale: {}", cli.scale);
    println!();

    // ---- Phase 1: Load instruct model, capture hidden states ----
    println!("=== Phase 1: Instruct model ===");
    println!("loading: {}", cli.instruct_model.display());
    let instruct_loaded = cortex::load_model(cli.instruct_model.to_str().unwrap())
        .map_err(|e| anyhow!("load instruct failed: {e}"))?;

    let tokens = instruct_loaded.tokenizer.encode(prompt, instruct_loaded.tokenizer.add_bos_default());
    println!("prompt tokens: {}", tokens.len());

    // Generate with instruct model (to see what "good" output looks like)
    let instruct_output = instruct_loaded.model.generate(
        &tokens,
        cli.max_tokens,
        SamplerConfig { temperature: 0.7, top_k: 40, top_p: 0.95, ..Default::default() },
        42,
        Some(instruct_loaded.tokenizer.eos_token_id()),
    );
    let instruct_text = instruct_loaded.tokenizer.decode(&instruct_output[tokens.len()..]);
    println!("instruct output: {:?}", instruct_text);
    println!();

    // Capture hidden states via forward_traced
    println!("capturing instruct hidden states...");
    let (_logits, instruct_trace) = instruct_loaded.model.forward_traced(&tokens);
    let n_layers = instruct_trace.n_layers;
    let embed_dim = instruct_trace.embed_dim;
    println!("  {} layers, embed_dim={}, seq_len={}", n_layers, embed_dim, instruct_trace.seq_len);

    // Save the hidden states we need (layers inject_from..n_layers)
    let mut instruct_hiddens: Vec<Vec<f32>> = Vec::new();
    for layer_idx in 0..=n_layers {
        instruct_hiddens.push(instruct_trace.hidden_states[layer_idx].clone());
    }

    // Drop instruct model to free memory
    drop(instruct_loaded);
    println!("instruct model unloaded");
    println!();

    // ---- Phase 2: Load base model, capture hidden states, generate baseline ----
    println!("=== Phase 2: Base model (no injection) ===");
    println!("loading: {}", cli.base_model.display());
    let mut base_loaded = cortex::load_model(cli.base_model.to_str().unwrap())
        .map_err(|e| anyhow!("load base failed: {e}"))?;

    // Generate with base model (to see what "bad" output looks like)
    let base_tokens = base_loaded.tokenizer.encode(prompt, base_loaded.tokenizer.add_bos_default());
    let base_output = base_loaded.model.generate(
        &base_tokens,
        cli.max_tokens,
        SamplerConfig { temperature: 0.7, top_k: 40, top_p: 0.95, ..Default::default() },
        42,
        Some(base_loaded.tokenizer.eos_token_id()),
    );
    let base_text = base_loaded.tokenizer.decode(&base_output[base_tokens.len()..]);
    println!("base output (no injection): {:?}", base_text);
    println!();

    // Capture base hidden states
    println!("capturing base hidden states...");
    let (_logits, base_trace) = base_loaded.model.forward_traced(&base_tokens);

    // ---- Phase 3: Compute deltas and inject ----
    println!("=== Phase 3: Base model + delta injection ===");
    println!();

    // Compute per-layer deltas for the injection range
    println!("computing deltas for layers {}..{}", cli.inject_from_layer, n_layers);
    let mut layer_deltas: Vec<Vec<f32>> = Vec::new();
    for layer_idx in 0..=n_layers {
        if layer_idx < cli.inject_from_layer || layer_idx > n_layers {
            layer_deltas.push(Vec::new());
            continue;
        }
        let instruct_h = &instruct_hiddens[layer_idx];
        let base_h = &base_trace.hidden_states[layer_idx];

        // Delta = instruct - base
        let delta: Vec<f32> = instruct_h
            .iter()
            .zip(base_h.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Report magnitude
        let l2: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  layer {:>2}: delta L2 = {:.4}", layer_idx, l2);
        layer_deltas.push(delta);
    }
    println!();

    // ---- Phase 3b: Attach DeltaInjectors and generate with injection ----
    println!("attaching DeltaInjectors at layers {}..{}", cli.inject_from_layer, n_layers);

    // We need mutable access to the model to attach injectors.
    // base_loaded.model is owned, so we can get &mut.
    let model = &mut base_loaded.model;

    for layer in cli.inject_from_layer..n_layers {
        let delta = layer_deltas[layer].clone();
        if delta.is_empty() {
            continue;
        }
        model.set_block_injector(
            layer,
            Box::new(DeltaInjector {
                deltas: delta,
                embed_dim,
                scale: cli.scale,
            }),
        );
    }

    println!("generating with injection active...");
    let injected_output = model.generate(
        &base_tokens,
        cli.max_tokens,
        SamplerConfig { temperature: 0.7, top_k: 40, top_p: 0.95, ..Default::default() },
        42,
        Some(base_loaded.tokenizer.eos_token_id()),
    );
    let injected_text = base_loaded.tokenizer.decode(&injected_output[base_tokens.len()..]);

    println!();
    println!("================================================================");
    println!("RESULTS");
    println!("================================================================");
    println!();
    println!("INSTRUCT model output:");
    println!("  {:?}", instruct_text);
    println!();
    println!("BASE model output (no injection):");
    println!("  {:?}", base_text);
    println!();
    println!("BASE model output (WITH delta injection at layers {}-{}):",
        cli.inject_from_layer, n_layers - 1);
    println!("  {:?}", injected_text);
    println!();

    // Report delta magnitudes by layer range
    println!("Delta magnitudes by layer range:");
    let early_l2: f32 = (0..8.min(n_layers))
        .filter_map(|l| {
            let d = &layer_deltas[l];
            if d.is_empty() { None } else { Some(d.iter().map(|x| x * x).sum::<f32>()) }
        })
        .sum::<f32>()
        .sqrt();
    let mid_l2: f32 = (8..16.min(n_layers))
        .filter_map(|l| {
            let d = &layer_deltas[l];
            if d.is_empty() { None } else { Some(d.iter().map(|x| x * x).sum::<f32>()) }
        })
        .sum::<f32>()
        .sqrt();
    let late_l2: f32 = (16..n_layers)
        .filter_map(|l| {
            let d = &layer_deltas[l];
            if d.is_empty() { None } else { Some(d.iter().map(|x| x * x).sum::<f32>()) }
        })
        .sum::<f32>()
        .sqrt();
    println!("  early (0-7):   L2 = {:.4}", early_l2);
    println!("  middle (8-15): L2 = {:.4}", mid_l2);
    println!("  late (16-23):  L2 = {:.4}", late_l2);
    println!();

    println!("INTERPRETATION:");
    println!("  If late-layer deltas are largest → instruction-following");
    println!("  behavior is concentrated in upper layers (confirms Gemini's");
    println!("  recommendation to inject at layers 16+).");
    println!();
    println!("  If deltas are uniform across layers → instruction-following");
    println!("  is a distributed property, not localizable to specific layers.");
    println!("  This would mean FFN injection at specific layers may not be");
    println!("  sufficient — the whole model needs to change.");
    println!();

    println!("NOTE: v0 does not yet inject mid-forward-pass because");
    println!("TransformerModel doesn't expose mutable block access.");
    println!("To test actual injection, cortex needs:");
    println!("  1. TransformerModel::set_block_injector(layer, Box<dyn FfnInjector>)");
    println!("  2. Or: a forward_with_injectors() method that accepts a map");
    println!("     of layer → injector");
    println!("This is a small cortex change (~20 lines) that unblocks the");
    println!("full FP-LLM experiment.");

    Ok(())
}
