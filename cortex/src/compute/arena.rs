//! GPU memory arena — the substrate for cortex's pool-based memory.
//!
//! **Problem this exists to solve.** The wgpu/NVIDIA driver's
//! `vkFreeMemory` accumulates internal state across allocations and
//! eventually charges 1-2 seconds per call. cortex's chat path
//! allocates ~12 scratch buffers per request; the first request's
//! drops are fast, but request 2+ pays 17 seconds total (the famous
//! TTFT cliff diagnosed 2026-05-17). The fix is "don't actually free
//! memory" — keep one big wgpu::Buffer slab and sub-allocate from
//! within it.
//!
//! **What this provides.** A `GpuArena` owns one or more wgpu::Buffer
//! slabs and hands out `ArenaSlice` handles via `acquire(size)`. On
//! drop, the slice returns to the arena's freelist (no wgpu::Buffer
//! drop, no driver-side cleanup). Subsequent acquires reuse free
//! ranges. Slabs grow on demand (doubling) when no existing slab
//! has a large-enough free range.
//!
//! **Why arena (sub-allocation) vs freelist-of-buffers.** The arena
//! pattern generalizes to PagedAttention's KV-block pool (fixed-size
//! blocks, ref-counting, copy-on-write across sequences) — a future
//! `KvBlockPool` can sit on the same arena substrate alongside today's
//! `ScratchPool`. A freelist-of-buffers couldn't.
//!
//! **API shape.** Each arena holds one set of `wgpu::BufferUsages`
//! (cortex's scratch is `STORAGE | COPY_SRC`). Multiple arenas can
//! coexist for different usage classes. `ArenaSlice::binding()` yields
//! a `wgpu::BufferBinding { buffer, offset, size }` ready to slot
//! into a bind group entry.
//!
//! **Allocator algorithm.** Best-fit search across all slabs;
//! coalesce-on-free with adjacent free ranges. Adequate for cortex's
//! limited allocation pattern (~12 buffers per forward, all freed at
//! end of forward). Production hardening (anti-fragmentation buddy
//! allocator etc.) is Stage 1's "production hardening" line item.

use std::sync::{Arc, Mutex};

use crate::compute::wgpu_backend::GpuDevice;

/// Minimum offset alignment for storage buffer bindings on most GPUs.
/// wgpu's `min_storage_buffer_offset_alignment` limit is typically 256
/// on NVIDIA, 32 on AMD/Intel. We use 256 unconditionally to keep the
/// arena hardware-portable.
const ALIGN: u64 = 256;

/// A free region within a slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

impl FreeRange {
    fn end(&self) -> u64 {
        self.offset + self.size
    }
}

/// One physical wgpu::Buffer slab and its freelist.
struct Slab {
    buffer: wgpu::Buffer,
    size: u64,
    /// Free ranges, kept sorted by `offset`. Adjacent ranges are
    /// coalesced on free so the list never has two touching entries.
    free_ranges: Vec<FreeRange>,
}

impl Slab {
    fn new(gpu: &GpuDevice, size: u64, usage: wgpu::BufferUsages, idx: usize) -> Self {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("arena.slab.{idx}")),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            free_ranges: vec![FreeRange { offset: 0, size }],
        }
    }

    /// Best-fit allocation: smallest free range that holds `aligned_size`.
    /// Returns the offset of the allocated region, or None if no fit.
    fn try_acquire(&mut self, aligned_size: u64) -> Option<u64> {
        let mut best_idx: Option<usize> = None;
        let mut best_size: u64 = u64::MAX;
        for (i, r) in self.free_ranges.iter().enumerate() {
            if r.size >= aligned_size && r.size < best_size {
                best_size = r.size;
                best_idx = Some(i);
            }
        }
        let idx = best_idx?;
        let range = self.free_ranges[idx];
        let offset = range.offset;
        if range.size == aligned_size {
            self.free_ranges.remove(idx);
        } else {
            // Shrink from the front. Remainder stays as a smaller free range.
            self.free_ranges[idx] = FreeRange {
                offset: range.offset + aligned_size,
                size: range.size - aligned_size,
            };
        }
        Some(offset)
    }

    /// Return [offset, offset+size) to the free list. Coalesces with
    /// adjacent free ranges so the list never has two touching entries.
    fn release(&mut self, offset: u64, size: u64) {
        let new = FreeRange { offset, size };
        // Find insertion point (keep list sorted by offset).
        let insert_pos = self.free_ranges
            .iter()
            .position(|r| r.offset > offset)
            .unwrap_or(self.free_ranges.len());
        self.free_ranges.insert(insert_pos, new);

        // Coalesce with right neighbor first (so left-coalesce can absorb both).
        if insert_pos + 1 < self.free_ranges.len() {
            let right = self.free_ranges[insert_pos + 1];
            if self.free_ranges[insert_pos].end() == right.offset {
                self.free_ranges[insert_pos].size += right.size;
                self.free_ranges.remove(insert_pos + 1);
            }
        }
        // Coalesce with left neighbor.
        if insert_pos > 0 {
            let left = self.free_ranges[insert_pos - 1];
            if left.end() == self.free_ranges[insert_pos].offset {
                self.free_ranges[insert_pos - 1].size += self.free_ranges[insert_pos].size;
                self.free_ranges.remove(insert_pos);
            }
        }
    }

    /// Sum of free space across all free ranges.
    fn free_bytes(&self) -> u64 {
        self.free_ranges.iter().map(|r| r.size).sum()
    }
}

/// Owns one or more wgpu::Buffer slabs; sub-allocates `ArenaSlice`s
/// against them. The arena never returns memory to the wgpu driver
/// during normal operation — slabs grow as needed.
pub struct GpuArena {
    gpu: Arc<GpuDevice>,
    state: Mutex<ArenaState>,
    usage: wgpu::BufferUsages,
    /// Size of the next slab to create. Doubles each time so we don't
    /// pay alloc cost for every individual allocation that exceeds the
    /// current slabs' free ranges.
    next_slab_size: Mutex<u64>,
}

struct ArenaState {
    slabs: Vec<Slab>,
}

impl GpuArena {
    /// Create an empty arena. The first allocation will trigger the
    /// first slab create. `initial_slab_size` is a hint — slabs grow
    /// from here.
    pub fn new(gpu: Arc<GpuDevice>, usage: wgpu::BufferUsages, initial_slab_size: u64) -> Arc<Self> {
        Arc::new(Self {
            gpu,
            state: Mutex::new(ArenaState { slabs: Vec::new() }),
            usage,
            next_slab_size: Mutex::new(initial_slab_size),
        })
    }

    /// Acquire a slice of at least `size` bytes. Rounds up to ALIGN.
    /// Never fails — grows a new slab if no existing slab fits.
    pub fn acquire(self: &Arc<Self>, size: u64) -> ArenaSlice {
        let aligned = (size + ALIGN - 1) / ALIGN * ALIGN;
        let mut state = self.state.lock().unwrap();

        // Best-fit across existing slabs.
        for (slab_idx, slab) in state.slabs.iter_mut().enumerate() {
            if let Some(offset) = slab.try_acquire(aligned) {
                return ArenaSlice {
                    arena: Arc::clone(self),
                    slab_idx,
                    buffer: slab.buffer.clone(),
                    offset,
                    size: aligned,
                };
            }
        }

        // No fit — grow a new slab. Size = max(next_slab_size, 2*aligned).
        let new_slab_size = {
            let mut next = self.next_slab_size.lock().unwrap();
            let s = (*next).max(aligned * 2);
            *next = s * 2;
            s
        };
        let slab_idx = state.slabs.len();
        let mut slab = Slab::new(&self.gpu, new_slab_size, self.usage, slab_idx);
        let offset = slab.try_acquire(aligned).expect("fresh slab must fit aligned allocation");
        let buffer = slab.buffer.clone();
        state.slabs.push(slab);

        ArenaSlice {
            arena: Arc::clone(self),
            slab_idx,
            buffer,
            offset,
            size: aligned,
        }
    }

    /// Stats snapshot. Useful for telemetry + bench assertions.
    pub fn stats(&self) -> ArenaStats {
        let state = self.state.lock().unwrap();
        let total_bytes: u64 = state.slabs.iter().map(|s| s.size).sum();
        let free_bytes: u64 = state.slabs.iter().map(|s| s.free_bytes()).sum();
        ArenaStats {
            slab_count: state.slabs.len(),
            total_bytes,
            free_bytes,
            in_use_bytes: total_bytes - free_bytes,
        }
    }

    fn release(&self, slab_idx: usize, offset: u64, size: u64) {
        let mut state = self.state.lock().unwrap();
        state.slabs[slab_idx].release(offset, size);
    }
}

/// Snapshot of an arena's memory state. Used by tests + telemetry.
#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub slab_count: usize,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub in_use_bytes: u64,
}

/// A sub-allocated range within an arena slab. Drop returns the range
/// to the arena's freelist — does NOT drop the underlying wgpu::Buffer.
/// This is the whole point of the arena: avoid the per-drop driver
/// cleanup that causes the TTFT cliff.
pub struct ArenaSlice {
    arena: Arc<GpuArena>,
    slab_idx: usize,
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl ArenaSlice {
    /// The underlying wgpu::Buffer. For bind groups, prefer `binding()`
    /// which carries the offset + size info.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Convenience: `wgpu::BufferBinding` ready to use in a bind group
    /// entry. Includes the (offset, size) sub-range info so the GPU
    /// only sees this slice, not the whole slab.
    pub fn binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: Some(std::num::NonZeroU64::new(self.size).expect("ArenaSlice size > 0")),
        }
    }
}

impl Drop for ArenaSlice {
    fn drop(&mut self) {
        self.arena.release(self.slab_idx, self.offset, self.size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests below need a real GPU device for the wgpu::Buffer
    // allocation. Skip silently if no GPU is available so CI runners
    // without GPU don't fail.

    fn try_gpu() -> Option<Arc<GpuDevice>> {
        GpuDevice::try_new().map(Arc::new)
    }

    #[test]
    fn freerange_coalesce_left_right_both() {
        let mut slab = Slab {
            buffer: dummy_buffer(),  // we won't actually use it
            size: 1024,
            free_ranges: vec![],
        };
        // Manually build a layout: [free 0..100][used 100..200][free 200..300]
        slab.free_ranges = vec![
            FreeRange { offset: 0, size: 100 },
            FreeRange { offset: 200, size: 100 },
        ];
        // Release 100..200 → should coalesce into one [0..300].
        slab.release(100, 100);
        assert_eq!(slab.free_ranges.len(), 1);
        assert_eq!(slab.free_ranges[0], FreeRange { offset: 0, size: 300 });
    }

    #[test]
    fn best_fit_picks_smallest_adequate() {
        let mut slab = Slab {
            buffer: dummy_buffer(),
            size: 1024,
            free_ranges: vec![
                FreeRange { offset: 0, size: 100 },
                FreeRange { offset: 256, size: 50 },   // smallest >= 30
                FreeRange { offset: 512, size: 500 },
            ],
        };
        let off = slab.try_acquire(30).unwrap();
        assert_eq!(off, 256, "best-fit should pick the 50-byte range");
    }

    #[test]
    fn arena_acquire_release_reuses() {
        let Some(gpu) = try_gpu() else { return };
        let arena = GpuArena::new(gpu, wgpu::BufferUsages::STORAGE, 4096);

        let s1 = arena.acquire(1024);
        let off1 = s1.offset();
        let stats1 = arena.stats();
        drop(s1);

        let s2 = arena.acquire(1024);
        let off2 = s2.offset();
        let stats2 = arena.stats();

        assert_eq!(off1, off2, "released slice should be reused");
        assert_eq!(stats1.slab_count, 1);
        assert_eq!(stats2.slab_count, 1, "no new slab should be allocated");
    }

    #[test]
    fn arena_grows_slab_when_no_fit() {
        let Some(gpu) = try_gpu() else { return };
        // initial_slab_size hint = 4096; first 3 KB allocation fits.
        // Second 3 KB allocation doesn't fit in the remaining ~1KB of the
        // first slab, so a new slab gets created. (note: first slab is
        // grown to max(hint, 2*aligned) = max(4096, 2*3072) = 6144;
        // second 3 KB fits a third time but not in remaining 3072 either
        // after the first goes in. Let me size more carefully.)
        //
        // Pin sizes: hint 4096 → first slab = max(4096, 2*3328) = 6656,
        // aligned ALIGN=256. First 3 KB acquire → aligned to 3328,
        // remaining 3328 free. Second 3 KB acquire → also 3328 aligned,
        // fits in same slab. Third 3 KB acquire → no fit → new slab.
        let arena = GpuArena::new(gpu, wgpu::BufferUsages::STORAGE, 4096);
        let _a = arena.acquire(3 * 1024);
        let _b = arena.acquire(3 * 1024);
        let _c = arena.acquire(3 * 1024);
        let stats = arena.stats();
        assert!(stats.slab_count >= 2, "expected slab growth, got {} slabs", stats.slab_count);
    }

    #[test]
    fn arena_pool_kills_per_drop_cost() {
        // The actual test for "the cliff dies": loop allocating + releasing
        // a large slice many times. Total time should be O(N * GPU compute)
        // not O(N * vkFreeMemory).
        let Some(gpu) = try_gpu() else { return };
        let arena = GpuArena::new(gpu, wgpu::BufferUsages::STORAGE, 64 * 1024 * 1024);

        let size = 16 * 1024 * 1024;  // 16 MB per cycle
        let t0 = std::time::Instant::now();
        for _ in 0..30 {
            let s = arena.acquire(size);
            drop(s);
        }
        let elapsed = t0.elapsed();
        let stats = arena.stats();

        // The first acquire creates a slab, all subsequent acquires reuse
        // the same range. Should be sub-second total — NOT 30 * 1.4s.
        assert!(elapsed.as_secs() < 5,
            "arena pool should reuse, got {:?} for 30 cycles", elapsed);
        assert_eq!(stats.slab_count, 1, "should have created just one slab");
        eprintln!("arena_pool_kills_per_drop_cost: {} cycles in {:?}, stats={stats:?}", 30, elapsed);
    }

    /// Standalone wgpu::Buffer for tests that don't need real GPU.
    /// Won't be used by the slab tests above that only check free-list math.
    fn dummy_buffer() -> wgpu::Buffer {
        // Use a real GPU if available — these tests should still compile
        // and skip cleanly when no GPU is present.
        let gpu = GpuDevice::try_new();
        let gpu = gpu.expect("freerange tests need a wgpu device for dummy_buffer; skip on no-GPU CI");
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
}
