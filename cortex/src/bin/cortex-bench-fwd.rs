//! Benchmark: CPU `TransformerModel::forward` vs GPU `GpuEngine::forward_full_gpu`
//! on a real GGUF model.
//!
//! Usage:
//!   cargo run --release --bin cortex-bench-fwd -- <model.gguf> [n_tokens] [iters]
//!
//! Defaults: n_tokens=128, iters=5. The benchmark warms up once for each
//! path before timing. Both paths produce logits over the vocabulary; we
//! ignore the values and just measure wall-clock per forward pass.

use std::time::Instant;

#[cfg(feature = "gpu")]
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: cortex-bench-fwd <model.gguf> [n_tokens] [iters]");
    let n_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    eprintln!("[bench] loading {}", path);
    let loaded = cortex::load_model(&path).expect("load_model failed");
    let vocab = loaded.model.vocab_size();

    // Build a deterministic prompt: tokens 0..n_tokens, clamped to vocab size.
    let tokens: Vec<u32> = (0..n_tokens).map(|i| (i as u32) % (vocab as u32)).collect();

    let gpu = cortex::compute::detect_gpu_device()
        .expect("no discrete GPU — bench requires one");
    eprintln!("[bench] discrete GPU detected, building GpuEngine");

    // Construct GpuEngine. with_max_seq sized to fit our prompt.
    let max_seq = (n_tokens + 16).max(1024);
    let engine = cortex::layers::gpu_engine::GpuEngine::with_max_seq(
        loaded.model, gpu, max_seq,
    );

    eprintln!("[bench] vocab={} n_tokens={} iters={}", vocab, n_tokens, iters);

    // ---- CPU bench ----
    eprintln!("[bench] CPU warmup...");
    let _ = engine.cpu().forward(&tokens, 0);
    eprintln!("[bench] CPU timing {} iters...", iters);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.cpu().forward(&tokens, 0);
    }
    let cpu_total = t0.elapsed();
    let cpu_per = cpu_total.as_secs_f64() / iters as f64;

    // ---- GPU bench ----
    eprintln!("[bench] GPU warmup...");
    let _ = engine.forward_full_gpu(&tokens, 0);
    eprintln!("[bench] GPU timing {} iters...", iters);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.forward_full_gpu(&tokens, 0);
    }
    let gpu_total = t0.elapsed();
    let gpu_per = gpu_total.as_secs_f64() / iters as f64;

    println!();
    println!("=== Forward-pass benchmark ===");
    println!("model:       {}", path);
    println!("n_tokens:    {}", n_tokens);
    println!("iters:       {}", iters);
    println!();
    println!("CPU forward: {:>8.3} s/forward   ({:>8.1} tokens/sec)",
             cpu_per, n_tokens as f64 / cpu_per);
    println!("GPU forward: {:>8.3} s/forward   ({:>8.1} tokens/sec)",
             gpu_per, n_tokens as f64 / gpu_per);
    println!();
    if gpu_per > 0.0 {
        println!("speedup:     {:>8.2}x", cpu_per / gpu_per);
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("cortex-bench-fwd requires the 'gpu' feature");
    std::process::exit(1);
}
