//! ParamsArena — frame-allocator for per-dispatch uniform buffers.
//!
//! **Why this exists.** Every shader dispatch needs a tiny uniform buffer
//! holding its scalar args (rows, cols, n_tokens, seq_len, etc. — usually
//! 8-32 bytes). A 36-block forward issues ~17 dispatches per block × 2
//! (prefill + decode) = ~600+ such buffers per request. Creating them via
//! `device.create_buffer` is one `vkAllocateMemory` call each; dropping
//! them is one `vkFreeMemory` each. NVIDIA's `vkFreeMemory` is
//! ~25-30 ms per call regardless of size, so 600 drops adds up to
//! ~15-18 seconds of cliff — the bulk of cortex's TTFT problem.
//!
//! **What this does.** Allocates one big `wgpu::Buffer` slab and
//! sub-allocates regions out of it via a bump cursor. Each `acquire()`
//! advances the cursor, writes the params via `queue.write_buffer`, and
//! returns a `ParamsHandle` containing the slab buffer + offset + size.
//! The handle is used to build a `BufferBinding` for the bind group.
//!
//! No per-acquire allocation cost. No per-drop free cost. At the end of
//! a forward, the caller invokes `reset()` to rewind the cursor; the
//! next forward reuses the same slab from offset 0. If a forward's
//! cumulative params exceed the slab size, the arena grows (allocates a
//! larger slab; old slab drops when no live handles reference it).
//!
//! **Why a separate concept from `GpuArena`.** `GpuArena` is a
//! general-purpose freelist allocator (acquire returns a handle, drop
//! releases the region for reuse). For the per-dispatch params pattern,
//! recycling within a forward would be UNSAFE — if dispatch N+1 writes
//! into a slot still being read by in-flight dispatch N on the GPU, we
//! race. The frame-allocator pattern sidesteps this: no region is reused
//! until the entire forward completes (reset is called after the submit
//! that consumes all dispatches finishes — typically when the wrapper's
//! readback_buffer returns from poll(Wait)).
//!
//! **Alignment.** wgpu (and Vulkan) require uniform buffer bindings to
//! be aligned to `Limits::min_uniform_buffer_offset_alignment` —
//! typically 64 or 256 bytes on desktop GPUs. We align to 256 to cover
//! every desktop GPU we care about.

use std::sync::{Arc, Mutex};

use crate::compute::wgpu_backend::GpuDevice;

const ALIGN: u64 = 256;

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

/// Frame-allocator for tiny per-dispatch uniform buffers. Hold one of
/// these per `GpuEngine`. Acquire returns a handle pointing into the
/// shared slab; call `reset()` after each forward completes to rewind.
pub struct ParamsArena {
    gpu: Arc<GpuDevice>,
    state: Mutex<ArenaState>,
}

struct ArenaState {
    slab: wgpu::Buffer,
    slab_size: u64,
    cursor: u64,
    /// Count of slab allocations over the arena's lifetime. Should stay
    /// at 1 after warm-up; jumps mean we resized.
    slab_allocations: u64,
    /// Count of `acquire` calls over the arena's lifetime. Diagnostic.
    acquire_count: u64,
}

impl ParamsArena {
    /// Create an arena with an initial slab of `initial_slab_size` bytes.
    /// 64 KiB is plenty for a typical forward (~20 KB of params) and the
    /// arena grows if needed.
    pub fn new(gpu: Arc<GpuDevice>, initial_slab_size: u64) -> Arc<Self> {
        let slab = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params_arena.slab"),
            size: initial_slab_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Arc::new(Self {
            gpu,
            state: Mutex::new(ArenaState {
                slab,
                slab_size: initial_slab_size,
                cursor: 0,
                slab_allocations: 1,
                acquire_count: 0,
            }),
        })
    }

    /// Acquire a slot, write the params into it, return a handle for
    /// binding. Grows the slab (allocating a new one) if the current
    /// slab can't fit the request.
    pub fn acquire<T: bytemuck::Pod>(&self, params: &T) -> ParamsHandle {
        let size = std::mem::size_of::<T>() as u64;
        let aligned = align_up(size, ALIGN);

        let mut state = self.state.lock().unwrap();
        state.acquire_count += 1;

        // Grow the slab if the new acquire wouldn't fit. We double size
        // each time so growth amortizes. Old slab is dropped when its
        // last cloned buffer handle goes out of scope (Arc-counted by
        // wgpu) — handles from earlier acquires keep it alive until the
        // command buffer that bound them is consumed.
        if state.cursor + aligned > state.slab_size {
            let new_size = (state.slab_size.max(aligned) * 2).max(state.cursor + aligned);
            state.slab = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params_arena.slab.grown"),
                size: new_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            state.slab_size = new_size;
            state.cursor = 0;
            state.slab_allocations += 1;
        }

        let offset = state.cursor;
        state.cursor += aligned;
        self.gpu.queue.write_buffer(&state.slab, offset, bytemuck::bytes_of(params));

        ParamsHandle {
            buffer: state.slab.clone(),
            offset,
            size,
        }
    }

    /// Reset the bump cursor to 0. Call after the submit that consumed
    /// all outstanding params (e.g. at the end of each forward
    /// function, AFTER readback returns). Does NOT free the slab.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.cursor = 0;
    }

    pub fn stats(&self) -> ArenaStats {
        let state = self.state.lock().unwrap();
        ArenaStats {
            slab_size: state.slab_size,
            current_cursor: state.cursor,
            slab_allocations: state.slab_allocations,
            acquire_count: state.acquire_count,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub slab_size: u64,
    pub current_cursor: u64,
    pub slab_allocations: u64,
    pub acquire_count: u64,
}

/// Handle to a region of the params arena's slab. Holds a Clone of the
/// slab's wgpu::Buffer (Arc-counted internally by wgpu) so the binding
/// stays valid even if the arena resizes to a new slab afterwards.
pub struct ParamsHandle {
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl ParamsHandle {
    /// Build a `BufferBinding` suitable for use in a bind group entry.
    pub fn binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<Arc<GpuDevice>> {
        GpuDevice::try_new().map(Arc::new)
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestParams { rows: u32, cols: u32 }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn acquire_and_reset_does_not_grow() {
        let Some(gpu) = try_gpu() else { return };
        let arena = ParamsArena::new(gpu, 4096);

        for _ in 0..10 {
            let _ = arena.acquire(&TestParams { rows: 1, cols: 2 });
        }
        let before = arena.stats();
        arena.reset();
        for _ in 0..10 {
            let _ = arena.acquire(&TestParams { rows: 3, cols: 4 });
        }
        let after = arena.stats();

        assert_eq!(before.slab_allocations, 1);
        assert_eq!(after.slab_allocations, 1, "no growth across reset");
        assert!(after.current_cursor <= 10 * ALIGN, "cursor sane");
    }

    #[test]
    fn arena_grows_when_slab_full() {
        let Some(gpu) = try_gpu() else { return };
        // Start with 512 bytes — fits 2 aligned slots.
        let arena = ParamsArena::new(gpu, 512);

        // Acquire 10 slots, forcing growth. Each grow doubles the
        // slab and resets the cursor (handles cloned the old buffer
        // and stay valid). The 10 slots end up split across multiple
        // slab generations, so we only assert that growth happened —
        // the final slab does NOT need to hold all 10.
        for _ in 0..10 {
            let _ = arena.acquire(&TestParams { rows: 0, cols: 0 });
        }
        let stats = arena.stats();
        assert!(stats.slab_allocations > 1, "should have grown at least once");
        assert!(stats.slab_size > 512, "final slab should be larger than initial");
        assert_eq!(stats.acquire_count, 10);
    }

    #[test]
    fn many_acquires_no_cliff() {
        // Empirical: 1000 acquires + resets should be sub-second.
        // If we accidentally call vkAllocateMemory per acquire this
        // would take 30+ seconds.
        let Some(gpu) = try_gpu() else { return };
        let arena = ParamsArena::new(gpu, 64 * 1024);

        let t0 = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = arena.acquire(&TestParams { rows: 1, cols: 2 });
            arena.reset();
        }
        let elapsed = t0.elapsed();
        let stats = arena.stats();

        assert!(elapsed.as_secs() < 5,
            "1000 acquires took {elapsed:?}, expected sub-second");
        assert_eq!(stats.slab_allocations, 1, "no growth, no churn");
        eprintln!("many_acquires_no_cliff: 1000 in {elapsed:?}, stats={stats:?}");
    }
}
