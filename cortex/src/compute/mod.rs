//! Compute backends for cortex inference.
//!
//! The legacy CPU `ComputeBackend` trait (scalar / AVX2 ternary matmul)
//! was removed 2026-05-29 alongside the BitNet path. Cortex now runs
//! entirely on GPU via wgpu (or falls back to CPU dequantization paths
//! inside `FloatLinear` for non-GPU builds).

pub mod device;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

/// Try to build a shared `GpuDevice` (resident-weights runtime).
///
/// Returns `Some(Arc<GpuDevice>)` only if a discrete GPU adapter is available.
/// One `Arc<GpuDevice>` is shared across all GPU-resident layers in a model.
#[cfg(feature = "gpu")]
pub fn detect_gpu_device() -> Option<std::sync::Arc<wgpu_backend::GpuDevice>> {
    let has_discrete = device::HardwareInfo::detect()
        .gpus
        .iter()
        .any(|g| g.device_type == device::GpuDeviceType::Discrete);
    if !has_discrete {
        tracing::info!("no discrete GPU; cortex requires a discrete GPU adapter");
        return None;
    }
    wgpu_backend::GpuDevice::try_new().map(std::sync::Arc::new)
}

#[cfg(not(feature = "gpu"))]
pub fn detect_gpu_device() -> Option<std::sync::Arc<()>> {
    None
}
