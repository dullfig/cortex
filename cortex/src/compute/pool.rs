//! ScratchPool — freelist-of-whole-buffers for cortex's BlockScratch.
//!
//! **Why a separate concept from `GpuArena`.** GpuArena does
//! sub-allocation (offset+size within a shared slab); ScratchPool
//! does whole-buffer recycling (each handle is a complete
//! `wgpu::Buffer`). The trade-off:
//!   - Arena: more general (supports PagedAttention block sharing),
//!     requires bind-group offset binding refactor (~78 call sites
//!     in `gpu_engine.rs`).
//!   - ScratchPool: cortex's bind groups stay unchanged (each handle
//!     IS a real wgpu::Buffer), at the cost of small over-allocation
//!     per size class.
//!
//! ScratchPool is the right shape for cortex's existing scratch
//! pattern (variable-size, short-lived per-forward, bound as whole
//! buffers in bind groups). The Arena will be used later for the
//! KvBlockPool (fixed-size blocks, cross-sequence sharing).
//!
//! **What it solves.** The TTFT cliff (`vkFreeMemory` 1-2 sec per
//! call on NVIDIA after the first request) dies because we never
//! actually drop the wgpu::Buffer instances. On `release`, the buffer
//! goes back to the size-class freelist; on `acquire`, the smallest
//! free buffer >= requested size is returned, or a new one created.
//!
//! **Size classes.** Power-of-2 buckets. A request for 17 MB gets
//! bucketed into the 32 MB class. Over-allocation per bucket is at
//! most 2× the requested size — the cost we pay for not doing
//! sub-allocation. For cortex's BlockScratch (max ~500 MB scores
//! buffer), the worst-case total VRAM overhead is bounded.
//!
//! **Usage flags.** Each pool is bound to one set of `BufferUsages`
//! (cortex's scratch needs `STORAGE | COPY_SRC`). Multiple pools can
//! coexist for different usage classes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::compute::wgpu_backend::GpuDevice;

/// Round size up to the next power of 2. The bucket for a size is the
/// power-of-2 it falls into. A 17 MB request → 32 MB bucket.
fn size_class(bytes: u64) -> u64 {
    if bytes == 0 {
        return 0;
    }
    let pow = 64 - (bytes - 1).leading_zeros();
    1u64 << pow
}

/// Freelist of wgpu::Buffer instances keyed by (size_class, usage).
/// On drop of a PooledBuffer, the buffer returns here instead of
/// being released to the driver.
pub struct ScratchPool {
    gpu: Arc<GpuDevice>,
    state: Mutex<PoolState>,
    usage: wgpu::BufferUsages,
}

struct PoolState {
    /// Free buffers keyed by size class (power-of-2 bucket).
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    /// Total bytes ever reserved by this pool (high water mark).
    bytes_reserved: u64,
    /// Bytes currently held by live PooledBuffers (not on the freelist).
    bytes_in_use: u64,
    /// Count of wgpu::Buffer create_buffer calls. Should stay flat after
    /// the warm-up phase if the pool is doing its job.
    create_calls: u64,
}

impl ScratchPool {
    pub fn new(gpu: Arc<GpuDevice>, usage: wgpu::BufferUsages) -> Arc<Self> {
        Arc::new(Self {
            gpu,
            state: Mutex::new(PoolState {
                free: HashMap::new(),
                bytes_reserved: 0,
                bytes_in_use: 0,
                create_calls: 0,
            }),
            usage,
        })
    }

    /// Acquire a buffer of at least `min_size` bytes. Returns a
    /// `PooledBuffer` which, on drop, returns to the pool. Allocates
    /// a new wgpu::Buffer only if no free buffer of the matching size
    /// class is available.
    pub fn acquire(self: &Arc<Self>, min_size: u64) -> PooledBuffer {
        let class = size_class(min_size);
        let mut state = self.state.lock().unwrap();

        // Reuse a free buffer of this class if available.
        let buffer = if let Some(free_list) = state.free.get_mut(&class) {
            free_list.pop()
        } else {
            None
        };

        let buffer = match buffer {
            Some(b) => b,
            None => {
                state.create_calls += 1;
                state.bytes_reserved += class;
                self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("scratch.pool.{class}")),
                    size: class,
                    usage: self.usage,
                    mapped_at_creation: false,
                })
            }
        };
        state.bytes_in_use += class;

        PooledBuffer {
            pool: Arc::clone(self),
            buffer: Some(buffer),
            size_class: class,
        }
    }

    pub fn stats(&self) -> PoolStats {
        let state = self.state.lock().unwrap();
        let free_count: usize = state.free.values().map(|v| v.len()).sum();
        PoolStats {
            bytes_reserved: state.bytes_reserved,
            bytes_in_use: state.bytes_in_use,
            create_calls: state.create_calls,
            free_buffer_count: free_count,
        }
    }

    fn release(&self, size_class: u64, buffer: wgpu::Buffer) {
        let mut state = self.state.lock().unwrap();
        state.bytes_in_use = state.bytes_in_use.saturating_sub(size_class);
        state.free.entry(size_class).or_default().push(buffer);
    }
}

/// Snapshot of pool state. Useful for tests + telemetry.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub bytes_reserved: u64,
    pub bytes_in_use: u64,
    pub create_calls: u64,
    pub free_buffer_count: usize,
}

/// A buffer acquired from a `ScratchPool`. Drop returns it to the
/// pool's freelist — no wgpu::Buffer drop, no driver cleanup, no cliff.
pub struct PooledBuffer {
    pool: Arc<ScratchPool>,
    buffer: Option<wgpu::Buffer>,
    size_class: u64,
}

impl PooledBuffer {
    /// Direct access to the underlying buffer. Most call sites can
    /// just use `&pooled_buffer` directly — `PooledBuffer` implements
    /// `Deref<Target = wgpu::Buffer>`, so deref coercion handles the
    /// `&PooledBuffer` → `&wgpu::Buffer` conversion in function args.
    /// This explicit accessor is for places where deref doesn't fire
    /// automatically (e.g. some generic contexts).
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("PooledBuffer accessed after drop")
    }

    /// Size of the actual underlying buffer (size class, may be larger
    /// than the requested min_size). Useful when callers care about
    /// the real buffer extent (e.g. for staging buffer mapping).
    pub fn allocated_size(&self) -> u64 {
        self.size_class
    }
}

/// Deref to the underlying buffer. This is the key ergonomic move
/// that lets BlockScratch field swaps from wgpu::Buffer to PooledBuffer
/// without touching cortex's ~78 BlockScratch call sites — Rust's
/// deref coercion converts `&PooledBuffer` to `&wgpu::Buffer` in
/// function argument positions automatically.
impl std::ops::Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.buffer()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.pool.release(self.size_class, buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<Arc<GpuDevice>> {
        GpuDevice::try_new().map(Arc::new)
    }

    #[test]
    fn size_class_buckets() {
        assert_eq!(size_class(1), 1);
        assert_eq!(size_class(2), 2);
        assert_eq!(size_class(3), 4);
        assert_eq!(size_class(8), 8);
        assert_eq!(size_class(9), 16);
        assert_eq!(size_class(17 * 1024 * 1024), 32 * 1024 * 1024);
        assert_eq!(size_class(500 * 1024 * 1024), 512 * 1024 * 1024);
    }

    #[test]
    fn pool_acquire_release_reuses() {
        let Some(gpu) = try_gpu() else { return };
        let pool = ScratchPool::new(gpu, wgpu::BufferUsages::STORAGE);

        let b1 = pool.acquire(1024);
        let stats1 = pool.stats();
        drop(b1);

        let b2 = pool.acquire(1024);
        let stats2 = pool.stats();

        assert_eq!(stats1.create_calls, 1);
        assert_eq!(stats2.create_calls, 1,
            "second acquire should reuse, not create");
        drop(b2);
    }

    #[test]
    fn pool_kills_per_drop_cost() {
        // The actual cliff-fix proof. Loop allocating + releasing a
        // large buffer many times. Total time should be O(N * pool_math)
        // not O(N * vkFreeMemory).
        let Some(gpu) = try_gpu() else { return };
        let pool = ScratchPool::new(gpu, wgpu::BufferUsages::STORAGE);

        let size: u64 = 16 * 1024 * 1024;
        let t0 = std::time::Instant::now();
        for _ in 0..30 {
            let b = pool.acquire(size);
            drop(b);
        }
        let elapsed = t0.elapsed();
        let stats = pool.stats();

        // Should be sub-second total — NOT 30 * 1.4s if vkFreeMemory fired.
        assert!(elapsed.as_secs() < 5,
            "pool should reuse, got {:?} for 30 cycles", elapsed);
        assert_eq!(stats.create_calls, 1,
            "should have allocated just once");
        eprintln!("pool_kills_per_drop_cost: 30 cycles in {:?}, stats={stats:?}", elapsed);
    }

    #[test]
    fn pool_different_size_classes_separate_freelists() {
        let Some(gpu) = try_gpu() else { return };
        let pool = ScratchPool::new(gpu, wgpu::BufferUsages::STORAGE);

        // Acquire two different size classes.
        let b_small = pool.acquire(1024);
        let b_big = pool.acquire(1024 * 1024);
        drop(b_small);
        drop(b_big);

        // Acquire same sizes again - both should reuse.
        let b_small2 = pool.acquire(1024);
        let b_big2 = pool.acquire(1024 * 1024);
        let stats = pool.stats();
        drop(b_small2);
        drop(b_big2);

        assert_eq!(stats.create_calls, 2,
            "different size classes should each create once; reused on second acquire");
    }
}
