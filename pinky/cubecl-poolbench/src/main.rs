//! CubeCL memory-pool cliff bench.
//!
//! Loops 100x: allocate ~500MB, write data into it, drop. Records per-iter wall
//! time. Run twice: once with Auto mode (default — wgpu returns memory to driver
//! on drop, which on NVIDIA Vulkan hits the ~17s vkFreeMemory cliff after the
//! first request), once with Persistent mode (CubeCL holds pages in its pool,
//! never returns to driver).
//!
//! Expected if Persistent works: total time ≈ 1× alloc cost. If not: 100× cost.
//!
//! Args: `auto` (default) | `persistent`. Optionally a second arg with iter count.

use cubecl::wgpu::WgpuRuntime;
use cubecl::Runtime;
use cubecl_runtime::memory_management::MemoryAllocationMode;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode_str = args.get(1).map(|s| s.as_str()).unwrap_or("auto");
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let mode = match mode_str {
        "persistent" => MemoryAllocationMode::Persistent,
        _ => MemoryAllocationMode::Auto,
    };

    let size_bytes: usize = 500 * 1024 * 1024;
    let data: Vec<u8> = vec![0u8; size_bytes];

    let device = Default::default();
    println!("[init] creating WgpuRuntime client (this may take a few seconds)...");
    let client = WgpuRuntime::client(&device);

    // SAFETY: docs say this is unsafe because it changes a global runtime
    // setting; for a single-threaded bench this is fine.
    unsafe { client.allocation_mode(mode); }

    println!(
        "[run] mode={mode:?} size_mb=500 iters={iters}",
    );

    let mut times = Vec::with_capacity(iters);
    let total_t0 = Instant::now();

    for i in 0..iters {
        let t0 = Instant::now();
        // create_from_slice forces actual allocation + upload (vs empty which
        // may be lazy). Closest analogue to cortex's per-request pattern.
        let handle = client.create_from_slice(&data);
        drop(handle);
        let dt = t0.elapsed();
        times.push(dt);

        if i % 10 == 0 || i == iters - 1 {
            let usage = client.memory_usage().unwrap();
            println!(
                "  iter {i:3}: dt={:>9.2}ms  in_use={} MB  reserved={} MB  allocs={}",
                dt.as_secs_f64() * 1000.0,
                usage.bytes_in_use / (1024 * 1024),
                usage.bytes_reserved / (1024 * 1024),
                usage.number_allocs,
            );
        }
    }

    let total = total_t0.elapsed();
    let avg = total / iters as u32;
    let min = times.iter().min().copied().unwrap();
    let max = times.iter().max().copied().unwrap();
    let p50 = {
        let mut s = times.clone();
        s.sort();
        s[s.len() / 2]
    };

    println!();
    println!("[result] mode={mode:?}");
    println!(
        "  total={:>9.2}s  avg={:>7.2}ms  min={:>7.2}ms  p50={:>7.2}ms  max={:>9.2}ms",
        total.as_secs_f64(),
        avg.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        p50.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
    let final_usage = client.memory_usage().unwrap();
    println!(
        "  final memory_usage: in_use={} MB reserved={} MB allocs={}",
        final_usage.bytes_in_use / (1024 * 1024),
        final_usage.bytes_reserved / (1024 * 1024),
        final_usage.number_allocs,
    );
}
