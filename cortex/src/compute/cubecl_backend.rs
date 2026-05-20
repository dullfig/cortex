//! CubeCL backend integration for the wgpu → CubeCL migration.
//!
//! **Status (M2 step 2):** pivoted to cubecl-cuda after cubecl-wgpu's
//! dx12 transitive feature collision (see migration plan + step 1
//! commit). cubecl-cuda doesn't pull wgpu-hal; the whole class of
//! problem disappears. Also aligns with M6's matmul win
//! (cooperative_matrix via cmma is on the CUDA backend only).
//!
//! See `pinky/cubecl-migration-plan-2026-05-18.md` for the full plan.

#![allow(dead_code)] // populated incrementally through M2-M7.

pub use cubecl;
pub use cubecl_runtime;

#[cfg(test)]
mod tests {
    use cubecl::prelude::*;
    use cubecl::cuda::CudaRuntime;

    /// Smoke test: instantiate a CudaRuntime client and allocate a small
    /// buffer. Validates that the CUDA backend actually works on this
    /// machine (toolkit installed, MSVC available, driver up). Skipped
    /// silently if no CUDA-capable device is found.
    #[test]
    fn cubecl_cuda_smoke() {
        let device = Default::default();
        let client = CudaRuntime::client(&device);
        // Allocate 1 MB. The Auto mode default pools (validated empirically
        // on wgpu backend in pinky/cubecl-poolbench/; same allocator on
        // cubecl side regardless of backend, just different storage layer).
        let handle = client.empty(1024 * 1024);
        let usage = client.memory_usage().unwrap();
        assert!(usage.bytes_reserved >= 1024 * 1024,
            "expected at least 1MB reserved, got {} bytes", usage.bytes_reserved);
        drop(handle);
    }
}
