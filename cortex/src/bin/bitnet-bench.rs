//! Benchmark: scalar vs AVX2 vs wgpu ternary matvec on real model weights.
//!
//! Usage: cargo run --release --bin bitnet-bench -- <model.gguf>

use std::time::Instant;

use cortex::compute::ComputeBackend;
use cortex::compute::scalar::ScalarBackend;
use cortex::gguf::GgufFile;
use cortex::ops::quantize::quantize_absmax;
use cortex::tensor::TernaryTensor;

/// Benchmark a single backend on a weight matrix.
/// Returns average microseconds per call.
fn bench_matvec(
    backend: &dyn ComputeBackend,
    weights: &TernaryTensor,
    input: &[i8],
    warmup: usize,
    iterations: usize,
) -> f64 {
    for _ in 0..warmup {
        let _ = backend.ternary_matvec(weights, input);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.ternary_matvec(weights, input);
    }
    start.elapsed().as_micros() as f64 / iterations as f64
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bitnet-bench <model.gguf>");
        std::process::exit(1);
    });

    println!("=== Ternary Compute Backend Benchmark ===\n");

    // Collect all available backends
    let mut backends: Vec<Box<dyn ComputeBackend>> = Vec::new();
    backends.push(Box::new(ScalarBackend));

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        backends.push(Box::new(cortex::compute::avx2::Avx2Backend));
    }

    #[cfg(feature = "gpu")]
    if let Some(gpu) = cortex::compute::wgpu_backend::WgpuBackend::try_new() {
        backends.push(Box::new(gpu));
    }

    println!("Backends: {}\n", backends.iter().map(|b| b.name()).collect::<Vec<_>>().join(", "));

    // Load model
    let gguf = GgufFile::open(&model_path).expect("failed to open GGUF");
    let config = gguf.model_config().expect("failed to read config");

    println!("Model: {} layers, embed_dim={}, intermediate={}\n",
        config.n_layers, config.embedding_dim, config.intermediate_size);

    // Load representative tensors
    let (q_weights, _) = gguf.load_ternary("blk.0.attn_q.weight")
        .expect("failed to load Q weights");
    println!("Q projection: {}×{} ({} packed bytes)",
        q_weights.rows(), q_weights.cols(), q_weights.packed_len());

    let (gate_weights, _) = gguf.load_ternary("blk.0.ffn_gate.weight")
        .expect("failed to load gate weights");
    println!("FFN gate:     {}×{} ({} packed bytes)",
        gate_weights.rows(), gate_weights.cols(), gate_weights.packed_len());

    // Generate input
    let embed_dim = config.embedding_dim as usize;
    let fake_input: Vec<f32> = (0..embed_dim)
        .map(|i| ((i * 17 + 5) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let (input_q, _) = quantize_absmax(&fake_input);

    let warmup = 5;
    let iterations = 30;

    // Reference result for correctness checks
    let ref_q = ScalarBackend.ternary_matvec(&q_weights, &input_q);
    let ref_gate = ScalarBackend.ternary_matvec(&gate_weights, &input_q);

    // --- Benchmark Q projection ---
    println!("\n--- Q projection ({}×{}) ---", q_weights.rows(), q_weights.cols());
    let mut q_times: Vec<(&str, f64)> = Vec::new();
    for backend in &backends {
        let us = bench_matvec(backend.as_ref(), &q_weights, &input_q, warmup, iterations);
        let result = backend.ternary_matvec(&q_weights, &input_q);
        let correct = result == ref_q;
        println!("  {:8}: {:7.0} µs  {}", backend.name(), us,
            if correct { "VERIFIED" } else { "MISMATCH!" });
        if !correct {
            eprintln!("  ERROR: {} result does not match scalar!", backend.name());
        }
        q_times.push((backend.name(), us));
    }

    // --- Benchmark FFN gate ---
    println!("\n--- FFN gate ({}×{}) ---", gate_weights.rows(), gate_weights.cols());
    let mut gate_times: Vec<(&str, f64)> = Vec::new();
    for backend in &backends {
        let us = bench_matvec(backend.as_ref(), &gate_weights, &input_q, warmup, iterations);
        let result = backend.ternary_matvec(&gate_weights, &input_q);
        let correct = result == ref_gate;
        println!("  {:8}: {:7.0} µs  {}", backend.name(), us,
            if correct { "VERIFIED" } else { "MISMATCH!" });
        gate_times.push((backend.name(), us));
    }

    // --- Full model estimates ---
    // Per layer: 4 attn projections (Q,K,V,O) + 3 FFN projections (gate, up, down)
    let layers = config.n_layers as f64;

    println!("\n--- Full model estimate ({} layers) ---", config.n_layers);
    println!("  Per layer: 4 × Q-size + 3 × gate-size matmuls\n");

    let baseline_ms = {
        let q = q_times[0].1;
        let g = gate_times[0].1;
        (4.0 * q + 3.0 * g) / 1000.0 * layers
    };

    for (i, backend) in backends.iter().enumerate() {
        let q_us = q_times[i].1;
        let gate_us = gate_times[i].1;
        let total_ms = (4.0 * q_us + 3.0 * gate_us) / 1000.0 * layers;
        let tps = 1000.0 / total_ms;
        let speedup = baseline_ms / total_ms;
        println!("  {:8}: {:6.0} ms/token  {:5.1} tok/s  ({:.1}× vs scalar)",
            backend.name(), total_ms, tps, speedup);
    }
}
