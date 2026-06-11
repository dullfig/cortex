//! Resident KV cache on the GPU.
//!
//! Per-layer K and V storage buffers, sized to `max_seq * kv_dim` each,
//! allocated once at session/shard start and reused across forward calls.
//! `kv_write_batch` writes new K/V entries during prefill; the attention
//! shaders read from them during both prefill and decode.
//!
//! Counterpart to the CPU `KvCache` / `ModelKvCache`. Where the CPU version
//! holds `Vec<f32>` and is read by index, this version holds `wgpu::Buffer`
//! references that get bound directly into compute pipelines — no readback
//! per token, no upload per token. Generation throughput on GPU is bound
//! by GPU compute and not by CPU↔GPU transfer.
//!
//! Memory (Phase A f16): K and V stored as packed f16 (2 per u32).
//!   2 * n_layers * max_seq * n_kv_heads * head_dim * 2 bytes
//!
//! For Qwen 2.5-3B at max_seq=4096: 36 * 2 * 4096 * 256 * 2 = 144 MB
//! (down from 288 MB f32). For Harmonizer-scale memex (13M tokens):
//! 36 * 2 * 13M * 256 * 2 = 72 GB. Still needs TurboQuant compression
//! for memex-scale, but half-precision buys 2x headroom.

use std::sync::Arc;

use crate::compute::wgpu_backend::GpuDevice;

/// Resident KV cache on the GPU. One pair of (K, V) buffers per transformer
/// layer, sized to `max_seq * n_kv_heads * head_dim` f16 elements each
/// (Phase A: packed 2 f16 per u32; storage buffer holds u32s).
///
/// Phase F (vram-heap): the per-layer K/V buffers are sub-allocations of
/// a single internal `VramHeap` owned by the cache. No same-dispatch R+RW
/// conflict exists between K and V (kv_write writes both, attn_score reads
/// only K, attn_value reads only V — never paired R+RW on the same
/// backing), so a single heap is sufficient. Cache drop → heap drop →
/// backing `wgpu::Buffer` released to wgpu.
pub struct GpuKvCache {
    gpu: Arc<GpuDevice>,
    kv_heap: Arc<::vram_heap::VramHeap>,
    /// Per-layer K storage. `k_buffers[i]` holds `[max_seq, kv_dim/2]` u32
    /// flat — each u32 packs 2 f16 K-values (Phase A). Shader unpacks via
    /// `unpack2x16float` on read; `kv_write_batch` packs via
    /// `pack2x16float` on write. kv_dim must be even (asserted on alloc).
    k_buffers: Vec<::vram_heap::VramAllocation>,
    /// Per-layer V storage. Same shape + packing as `k_buffers`.
    v_buffers: Vec<::vram_heap::VramAllocation>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Number of tokens currently written to the cache (next write position).
    len: usize,
}

impl GpuKvCache {
    /// Allocate cache buffers for a model with the given shape.
    pub fn new(
        gpu: Arc<GpuDevice>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        assert!(n_layers > 0 && n_kv_heads > 0 && head_dim > 0 && max_seq_len > 0);
        let kv_dim = n_kv_heads * head_dim;
        // Phase A: pack 2 f16 per u32 → 2 bytes per element instead of 4.
        // kv_dim must be even (head_dim is always even for our supported
        // models; assertion makes this explicit for any future model).
        assert!(kv_dim % 2 == 0, "kv_dim ({kv_dim}) must be even for f16 packing");
        let bytes_per_buffer = (max_seq_len * kv_dim * 2) as u64;

        let align = ::vram_heap::STORAGE_BUFFER_OFFSET_ALIGNMENT_NVIDIA;
        let round_up = |bytes: u64| ((bytes + align - 1) / align) * align;
        // Heap capacity = 2 buffers per layer (K + V), each padded to alignment.
        // Add per-allocation slack so free-list bookkeeping headroom is sane.
        let pad_per_alloc = align;
        let heap_size = round_up(bytes_per_buffer) * 2 * n_layers as u64
            + pad_per_alloc * (n_layers as u64 * 2);

        let kv_heap = ::vram_heap::VramHeap::new(
            &gpu.device,
            ::vram_heap::MemoryTier::DeviceLocal,
            heap_size,
            "gpu_kv.heap",
        ).expect("gpu_kv_cache heap construction failed");

        let mut k_buffers = Vec::with_capacity(n_layers);
        let mut v_buffers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            k_buffers.push(kv_heap.allocate(
                bytes_per_buffer, align, &format!("gpu_kv_cache.k.layer{i}"),
            ).expect("gpu_kv_cache heap capacity for K"));
            v_buffers.push(kv_heap.allocate(
                bytes_per_buffer, align, &format!("gpu_kv_cache.v.layer{i}"),
            ).expect("gpu_kv_cache heap capacity for V"));
        }

        Self {
            gpu,
            kv_heap,
            k_buffers,
            v_buffers,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Borrow the K buffer for a specific layer.
    pub fn k_layer(&self, idx: usize) -> &::vram_heap::VramAllocation {
        &self.k_buffers[idx]
    }

    /// Borrow the V buffer for a specific layer.
    pub fn v_layer(&self, idx: usize) -> &::vram_heap::VramAllocation {
        &self.v_buffers[idx]
    }

    /// Number of tokens currently cached (next write position).
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Pre-allocated capacity in tokens.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of transformer layers this cache covers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Number of KV heads (GQA).
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Combined KV dimension (`n_kv_heads * head_dim`). Number of f32 values
    /// per cached position per layer per K (or V).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// True if no positions are cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reset the write cursor. Buffers stay allocated and resident; the next
    /// `append` writes from offset 0.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Advance the write cursor by `n` tokens. Used by the orchestrator after
    /// a prefill-or-decode forward records writes into this cache.
    ///
    /// Panics if the new length would exceed `max_seq_len`.
    pub fn advance(&mut self, n: usize) {
        assert!(
            self.len + n <= self.max_seq_len,
            "GpuKvCache overflow: {} + {} > {}",
            self.len, n, self.max_seq_len,
        );
        self.len += n;
    }

    /// Total VRAM bytes used by this cache (2 buffers per layer × layers ×
    /// max_seq × kv_dim × 2 bytes; f16 packed since Phase A).
    pub fn memory_bytes(&self) -> u64 {
        2 * (self.n_layers as u64) * (self.max_seq_len as u64) * (self.kv_dim() as u64) * 2
    }

    /// Borrow the GPU device this cache is bound to (for orchestrators that
    /// need to dispatch shaders against the cache buffers).
    pub fn gpu(&self) -> &Arc<GpuDevice> {
        &self.gpu
    }
}

impl std::fmt::Debug for GpuKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuKvCache(layers={}, kv_heads={}, head_dim={}, len={}/{}, {:.1}MB)",
            self.n_layers,
            self.n_kv_heads,
            self.head_dim,
            self.len,
            self.max_seq_len,
            self.memory_bytes() as f64 / (1024.0 * 1024.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_inspect() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cache = GpuKvCache::new(gpu, /*n_layers*/ 4, /*n_kv_heads*/ 2, /*head_dim*/ 8, /*max_seq*/ 16);
        assert_eq!(cache.n_layers(), 4);
        assert_eq!(cache.n_kv_heads(), 2);
        assert_eq!(cache.head_dim(), 8);
        assert_eq!(cache.kv_dim(), 16);
        assert_eq!(cache.max_seq_len(), 16);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
        // Phase A: f16 packed, 2 bytes per element.
        // 2 * 4 layers * 16 max_seq * 16 kv_dim * 2 bytes = 4096 bytes
        assert_eq!(cache.memory_bytes(), 2 * 4 * 16 * 16 * 2);
    }

    #[test]
    fn advance_and_clear() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let mut cache = GpuKvCache::new(gpu, 2, 1, 4, 32);
        assert_eq!(cache.seq_len(), 0);

        cache.advance(5);
        assert_eq!(cache.seq_len(), 5);
        assert!(!cache.is_empty());

        cache.advance(10);
        assert_eq!(cache.seq_len(), 15);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn advance_past_max_panics() {
        let Some(gpu) = GpuDevice::try_new() else {
            // Test must still panic when no GPU is available, so the
            // #[should_panic] expectation is satisfied.
            panic!("overflow (no GPU; faking)");
        };
        // Reuse the single device. A second `try_new()` would coexist with
        // this one (the binding stays live to function end), doubling the
        // ~7 GB weights-heap reservation and OOMing on a 16 GB GPU.
        let gpu = Arc::new(gpu);
        let mut cache = GpuKvCache::new(gpu, 1, 1, 4, 8);
        cache.advance(10); // 10 > 8
    }

    #[test]
    fn per_layer_buffer_handles() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let cache = GpuKvCache::new(gpu, 3, 1, 4, 16);
        // Each layer has a distinct K and V buffer (different handles).
        for i in 0..3 {
            let _k = cache.k_layer(i);
            let _v = cache.v_layer(i);
        }
        // Different layers have different buffer handles.
        assert!(!std::ptr::eq(
            cache.k_layer(0) as *const _,
            cache.k_layer(1) as *const _,
        ));
    }
}
