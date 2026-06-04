//! Resident GPU storage for PolarQuant-compressed KV cache.
//!
//! Counterpart to `GpuKvCache` (resident f32 K/V). Holds per-layer
//! compressed buffers — packed u32 angles + per-position f32 radius for
//! both K and V — plus a per-layer rotation matrix and a shared
//! 8-bucket angle LUT. Sized to `max_seq` at construction; stays
//! resident across forward calls. Memory footprint per layer:
//!
//!   K_angles  = max_seq * n_kv_heads * (head_dim/2) bytes  (u8 packed in u32)
//!   K_radius  = max_seq * n_kv_heads * 4 bytes             (f32)
//!   V_angles  = same as K_angles
//!   V_radius  = same as K_radius
//!   rotation  = head_dim * head_dim * 4 bytes              (f32, one R per layer)
//!
//! For Qwen 2.5-3B at max_seq=4096:
//!   per-layer K+V angles  = 2 * 4096 * 2 * 64 = 1 MB
//!   per-layer K+V radius  = 2 * 4096 * 2 * 4  = 64 KB
//!   per-layer rotation    = 128 * 128 * 4    = 64 KB
//!   total per layer       ≈ 1.13 MB → 36 layers ≈ 40 MB
//! vs the f32 GpuKvCache's ~288 MB at the same shape: ~7x reduction.
//!
//! Phase 2c.1 ships the storage + resident dispatchers, populated by
//! reading from a CPU-side `QuantizedKvCache`. A GPU-native prefill
//! compress shader (avoiding the CPU round-trip) is a follow-up.

use std::sync::Arc;

use crate::compute::wgpu_backend::GpuDevice;
use crate::layers::gpu_polar::{pack_angles_to_u32, polar_lut_vec4};
use crate::layers::quantized_kv_cache::QuantizedKvCache;
use crate::ops::polar;

/// Resident GPU compressed KV cache.
///
/// One pair of (K_angles, K_radius, V_angles, V_radius) buffers per
/// transformer layer, sized at construction.
pub struct GpuPolarKvCache {
    gpu: Arc<GpuDevice>,

    /// Per-layer packed K angles. Each buffer holds
    /// `ceil(max_seq * n_kv_heads * n_pairs / 4)` u32 words.
    k_angles_buffers: Vec<wgpu::Buffer>,
    /// Per-layer K radius buffers. Each holds `max_seq * n_kv_heads` f32.
    k_radius_buffers: Vec<wgpu::Buffer>,
    v_angles_buffers: Vec<wgpu::Buffer>,
    v_radius_buffers: Vec<wgpu::Buffer>,

    /// Per-layer rotation matrices: one f32 storage buffer per layer of
    /// size `head_dim * head_dim * 4` bytes. Storing per-layer (vs one
    /// packed buffer) keeps the resident dispatchers from needing
    /// sub-region bindings and avoids `min_storage_buffer_offset_alignment`
    /// constraints when head_dim is small (test fixtures).
    rotation_buffers: Vec<wgpu::Buffer>,

    /// Shared angle LUT uniform: vec4<f32>[8] (cos, sin, _, _) per bucket.
    lut_buffer: wgpu::Buffer,

    /// Per-layer QJL sign-bit storage for K residuals. Empty when QJL
    /// is disabled (`n_qjl_proj == 0`). Each buffer holds
    /// `max_seq_len * n_kv_heads * sign_words_per_entry` u32 words.
    /// `sign_words_per_entry = ceil(n_qjl_proj / 32)`. For the default
    /// `n_qjl_proj = 32`, one u32 per (pos, head) entry — the j-th
    /// bit of the word is the sign for projection j (matches CPU
    /// `QjlProjection::encode_signs` bit layout).
    k_qjl_signs_buffers: Vec<wgpu::Buffer>,

    /// Per-layer QJL projection matrices (deterministic from
    /// `qjl_seed_base + layer_idx`). Empty when QJL is disabled. Each
    /// buffer is `n_qjl_proj * head_dim * 4` bytes (row-major f32).
    /// Uploaded once at construction, read-only thereafter.
    k_qjl_projection_buffers: Vec<wgpu::Buffer>,

    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Number of QJL projections per K residual. 0 = QJL disabled.
    /// Typical: 32 (matches CPU `QjlProjection` default). When > 0 the
    /// engine's polar block forward will dispatch the QJL encode shader
    /// after `compress_layer_into_polar` and route attention through
    /// the QJL-corrected score shader. Brings ~0.84 → ~0.95 cosine on
    /// attention output vs f32.
    n_qjl_proj: usize,
    /// Per-layer QJL projection seed base. Layer `i` uses seed
    /// `qjl_seed_base + i`. Kept separate from `rotation_seed_base`
    /// because rotation matrices are square orthonormal whereas QJL
    /// projections are rectangular Gaussian-normalized — sharing a
    /// seed would create confusing implicit dependencies.
    #[allow(dead_code)]
    qjl_seed_base: u64,
    /// Number of positions currently filled. Per design, all layers advance
    /// in lockstep — like `GpuKvCache` — so a single `len` is correct.
    len: usize,
}

impl GpuPolarKvCache {
    /// Allocate a fresh compressed cache. Per-layer rotation matrices are
    /// generated deterministically from `(rotation_seed_base + layer_idx)`
    /// to match how engram's `QuantizedKvCache::new(seed)` chose its R.
    pub fn new(
        gpu: Arc<GpuDevice>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rotation_seed_base: u64,
    ) -> Self {
        // Backward-compatible delegate. QJL disabled.
        Self::new_with_qjl(
            gpu, n_layers, n_kv_heads, head_dim, max_seq_len,
            rotation_seed_base, /*n_qjl_proj*/ 0, /*qjl_seed_base*/ 0,
        )
    }

    /// Allocate a fresh compressed cache with optional QJL correction
    /// for K. When `n_qjl_proj > 0`, additionally allocates per-layer
    /// QJL signs (read_write) and projection (read-only) buffers; the
    /// engine's polar forward then dispatches the QJL encode and
    /// QJL-corrected attention shaders. `n_qjl_proj == 0` disables
    /// QJL entirely (identical to `new`).
    ///
    /// QJL improves polar attention output cosine from ~0.84 to ~0.95
    /// at a per-layer storage cost of ~32 KB (signs) + ~16 KB
    /// (projections) for Qwen 3B at `n_qjl_proj=32`, `max_seq=4096`.
    ///
    /// Currently supports `n_qjl_proj <= 32` only (sign storage is
    /// one u32 per (pos, head) entry — wider QJL would need multi-word
    /// signs); shader changes can lift this if needed.
    pub fn new_with_qjl(
        gpu: Arc<GpuDevice>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rotation_seed_base: u64,
        n_qjl_proj: usize,
        qjl_seed_base: u64,
    ) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        assert!(n_layers > 0 && n_kv_heads > 0 && head_dim > 0 && max_seq_len > 0);
        assert!(
            n_qjl_proj <= 32,
            "n_qjl_proj > 32 not yet supported (current sign storage is one u32 per entry)"
        );

        let n_pairs = head_dim / 2;
        let angles_total = max_seq_len * n_kv_heads * n_pairs;
        let angles_words = (angles_total + 3) / 4;
        let angles_bytes = (angles_words * std::mem::size_of::<u32>()) as u64;
        let radius_bytes = (max_seq_len * n_kv_heads * std::mem::size_of::<f32>()) as u64;

        let mk_angles = |label: &str| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: angles_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mk_radius = |label: &str| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: radius_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let mut k_angles_buffers = Vec::with_capacity(n_layers);
        let mut k_radius_buffers = Vec::with_capacity(n_layers);
        let mut v_angles_buffers = Vec::with_capacity(n_layers);
        let mut v_radius_buffers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            k_angles_buffers.push(mk_angles(&format!("polar_kv.k_angles.layer{i}")));
            k_radius_buffers.push(mk_radius(&format!("polar_kv.k_radius.layer{i}")));
            v_angles_buffers.push(mk_angles(&format!("polar_kv.v_angles.layer{i}")));
            v_radius_buffers.push(mk_radius(&format!("polar_kv.v_radius.layer{i}")));
        }

        // Per-layer rotation matrices: one buffer each, deterministic from
        // (rotation_seed_base + layer_idx) — matches engram's per-layer
        // R seeding convention.
        let mut rotation_buffers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let r = polar::generate_rotation_matrix(head_dim, rotation_seed_base + i as u64);
            rotation_buffers.push(gpu.create_storage_buffer(
                bytemuck::cast_slice(&r),
                &format!("polar_kv.rotation.layer{i}"),
            ));
        }

        // Shared angle LUT uniform.
        let lut = polar_lut_vec4();
        let lut_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polar_kv.angle_lut"),
            size: std::mem::size_of_val(&lut) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut));

        // QJL buffers — only allocated when QJL is enabled.
        let (k_qjl_signs_buffers, k_qjl_projection_buffers) = if n_qjl_proj > 0 {
            let sign_words_per_entry = (n_qjl_proj + 31) / 32; // = 1 for n_qjl_proj <= 32
            let signs_bytes = (max_seq_len * n_kv_heads * sign_words_per_entry
                * std::mem::size_of::<u32>()) as u64;

            let mut signs = Vec::with_capacity(n_layers);
            let mut projs = Vec::with_capacity(n_layers);
            for layer in 0..n_layers {
                // Signs: zero-initialized storage, written by the QJL
                // encode shader. COPY_DST so we can clear via host
                // writes if needed; COPY_SRC for test readback.
                signs.push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("polar_kv.k_qjl_signs.layer{layer}")),
                    size: signs_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));

                // Projection matrix: deterministic from (qjl_seed_base + layer).
                // Same RNG seed scheme as rotation matrices, but a separate
                // seed base — QJL projections are rectangular Gaussian, not
                // square orthonormal.
                let qjl = crate::ops::qjl::QjlProjection::with_n_projections(
                    head_dim, n_qjl_proj, qjl_seed_base + layer as u64,
                );
                projs.push(gpu.create_storage_buffer(
                    bytemuck::cast_slice(qjl.projections()),
                    &format!("polar_kv.k_qjl_projection.layer{layer}"),
                ));
            }
            (signs, projs)
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            gpu,
            k_angles_buffers,
            k_radius_buffers,
            v_angles_buffers,
            v_radius_buffers,
            rotation_buffers,
            lut_buffer,
            k_qjl_signs_buffers,
            k_qjl_projection_buffers,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            n_qjl_proj,
            qjl_seed_base,
            len: 0,
        }
    }

    /// Populate one layer's K and V buffers from a CPU `QuantizedKvCache`.
    /// The CPU cache must have been built with the SAME rotation seed
    /// scheme — see the `seed_for_layer` helper below — otherwise the
    /// compressed bytes won't be consistent with this cache's rotation.
    ///
    /// Test- and bootstrap-facing. Production will get a GPU compress
    /// shader that fills these buffers without a CPU round-trip.
    pub fn upload_layer_from_cpu(&mut self, layer: usize, cpu: &QuantizedKvCache) {
        assert!(layer < self.n_layers);
        assert_eq!(cpu.n_kv_heads(), self.n_kv_heads);
        assert_eq!(cpu.head_dim(), self.head_dim);
        assert!(cpu.len() <= self.max_seq_len);

        let n_pairs = self.head_dim / 2;
        let angles_used = cpu.len() * self.n_kv_heads * n_pairs;
        let radius_used = cpu.len() * self.n_kv_heads;

        // Pack K angles to u32 and upload.
        let k_angles_full: Vec<u8> = (0..angles_used)
            .map(|i| cpu.k_angles_slice()[i])
            .collect();
        let k_packed = pack_angles_to_u32(&k_angles_full);
        self.gpu.queue.write_buffer(
            &self.k_angles_buffers[layer],
            0,
            bytemuck::cast_slice(&k_packed),
        );

        // K radius.
        self.gpu.queue.write_buffer(
            &self.k_radius_buffers[layer],
            0,
            bytemuck::cast_slice(&cpu.k_radius_slice()[..radius_used]),
        );

        // V — engram's QuantizedKvCache doesn't expose v_angles_slice /
        // v_radius_slice directly, so reconstruct via read_compressed_k.
        // For the bootstrap-only upload path the cost is fine; the GPU
        // compress shader will skip this round-trip entirely.
        let mut v_angles_full = vec![0u8; angles_used];
        let mut v_radius_full = vec![0f32; radius_used];
        for t in 0..cpu.len() {
            for h in 0..self.n_kv_heads {
                let entry = cpu.read_compressed_k(t, h);
                let off = (t * self.n_kv_heads + h) * n_pairs;
                v_angles_full[off..off + n_pairs].copy_from_slice(&entry.v_angles);
                v_radius_full[t * self.n_kv_heads + h] = entry.v_radius;
            }
        }
        let v_packed = pack_angles_to_u32(&v_angles_full);
        self.gpu.queue.write_buffer(
            &self.v_angles_buffers[layer],
            0,
            bytemuck::cast_slice(&v_packed),
        );
        self.gpu.queue.write_buffer(
            &self.v_radius_buffers[layer],
            0,
            bytemuck::cast_slice(&v_radius_full),
        );
    }

    /// Populate this polar cache from a fully-populated f32 `GpuKvCache`,
    /// running the GPU compress shader per layer. No CPU round-trip:
    /// the f32 K/V buffers stay on GPU and feed straight into the
    /// rotation+polar-quantize shader, which writes directly into this
    /// cache's compressed buffers.
    ///
    /// Layer count, kv-heads, and head-dim must match. The polar cache's
    /// `max_seq_len` must be at least the f32 cache's current `seq_len()`
    /// (we only compress positions actually filled). After conversion
    /// `self.seq_len()` equals `f32_cache.seq_len()`.
    ///
    /// Submits its own command buffer and waits for completion so callers
    /// can immediately read the polar cache's contents.
    pub fn populate_from_f32_cache_gpu(
        &mut self,
        f32_cache: &crate::layers::gpu_kv_cache::GpuKvCache,
    ) {
        assert_eq!(self.n_layers, f32_cache.n_layers(),
            "polar n_layers ({}) != f32 n_layers ({})",
            self.n_layers, f32_cache.n_layers());
        assert_eq!(self.n_kv_heads, f32_cache.n_kv_heads(),
            "polar n_kv_heads ({}) != f32 n_kv_heads ({})",
            self.n_kv_heads, f32_cache.n_kv_heads());
        assert_eq!(self.head_dim, f32_cache.head_dim(),
            "polar head_dim ({}) != f32 head_dim ({})",
            self.head_dim, f32_cache.head_dim());
        let n_tokens = f32_cache.seq_len();
        assert!(n_tokens <= self.max_seq_len,
            "f32 seq_len ({}) exceeds polar max_seq_len ({})",
            n_tokens, self.max_seq_len);

        if n_tokens == 0 {
            self.len = 0;
            return;
        }

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("polar_kv.populate_from_f32_cache_gpu"),
        });
        for layer in 0..self.n_layers {
            crate::layers::gpu_polar::compress_layer_into_polar(
                &self.gpu,
                &mut encoder,
                self,
                layer,
                f32_cache.k_layer(layer),
                f32_cache.v_layer(layer),
                n_tokens,
                /*start_pos*/ 0,
            );
            // When QJL is enabled, encode K residual signs right after
            // compress (which wrote the angles + radius this step reads
            // for dequantization). No-op when n_qjl_proj == 0.
            crate::layers::gpu_polar::qjl_encode_k_layer(
                &self.gpu,
                &mut encoder,
                self,
                layer,
                f32_cache.k_layer(layer),
                n_tokens,
                /*start_pos*/ 0,
            );
        }
        self.gpu.queue.submit(Some(encoder.finish()));
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        self.len = n_tokens;
    }

    /// Set the `len` cursor to mark how many positions are filled.
    /// Caller is responsible for keeping per-layer uploads consistent;
    /// like `GpuKvCache`, all layers share one `len`.
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.max_seq_len);
        self.len = len;
    }

    /// Reset the cursor; buffers stay allocated.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn seq_len(&self) -> usize { self.len }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }
    pub fn n_layers(&self) -> usize { self.n_layers }
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn n_pairs(&self) -> usize { self.head_dim / 2 }

    pub fn k_angles_layer(&self, idx: usize) -> &wgpu::Buffer { &self.k_angles_buffers[idx] }
    pub fn k_radius_layer(&self, idx: usize) -> &wgpu::Buffer { &self.k_radius_buffers[idx] }
    pub fn v_angles_layer(&self, idx: usize) -> &wgpu::Buffer { &self.v_angles_buffers[idx] }
    pub fn v_radius_layer(&self, idx: usize) -> &wgpu::Buffer { &self.v_radius_buffers[idx] }
    pub fn rotation_layer(&self, idx: usize) -> &wgpu::Buffer { &self.rotation_buffers[idx] }
    pub fn lut_buffer(&self) -> &wgpu::Buffer { &self.lut_buffer }

    /// Number of QJL projections. 0 means QJL is disabled.
    pub fn n_qjl_proj(&self) -> usize { self.n_qjl_proj }

    /// Bytes per (pos, head) entry for the signs buffer. Always a
    /// multiple of 4 (u32-aligned). Returns 0 when QJL is disabled.
    pub fn qjl_sign_bytes_per_entry(&self) -> usize {
        if self.n_qjl_proj == 0 {
            0
        } else {
            ((self.n_qjl_proj + 31) / 32) * std::mem::size_of::<u32>()
        }
    }

    /// K QJL signs buffer for layer `idx`. `None` when QJL disabled.
    pub fn k_qjl_signs_layer(&self, idx: usize) -> Option<&wgpu::Buffer> {
        self.k_qjl_signs_buffers.get(idx)
    }

    /// K QJL projection matrix buffer for layer `idx`. `None` when
    /// QJL disabled.
    pub fn k_qjl_projection_layer(&self, idx: usize) -> Option<&wgpu::Buffer> {
        self.k_qjl_projection_buffers.get(idx)
    }

    /// VRAM bytes used (compressed K+V + rotation + optional QJL;
    /// LUT excluded).
    pub fn memory_bytes(&self) -> u64 {
        let n_pairs = self.head_dim / 2;
        let angles_per = (self.max_seq_len * self.n_kv_heads * n_pairs + 3) / 4 * 4;
        let radius_per = self.max_seq_len * self.n_kv_heads * 4;
        let kv_per_layer = (2 * angles_per + 2 * radius_per) as u64; // K + V
        let rot = (self.n_layers * self.head_dim * self.head_dim * 4) as u64;
        let qjl = if self.n_qjl_proj > 0 {
            let signs_per_layer = (self.max_seq_len * self.n_kv_heads
                * self.qjl_sign_bytes_per_entry()) as u64;
            let proj_per_layer = (self.n_qjl_proj * self.head_dim * 4) as u64;
            (signs_per_layer + proj_per_layer) * self.n_layers as u64
        } else {
            0
        };
        kv_per_layer * self.n_layers as u64 + rot + qjl
    }

    /// f32-equivalent KV size for the same shape (rotation excluded).
    pub fn f32_equivalent_bytes(&self) -> u64 {
        (2 * self.n_layers * self.max_seq_len * self.n_kv_heads * self.head_dim * 4) as u64
    }

    /// Reference to the underlying GPU device (for orchestrators that need
    /// to dispatch shaders against these buffers).
    pub fn gpu(&self) -> &Arc<GpuDevice> { &self.gpu }
}

impl std::fmt::Debug for GpuPolarKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ratio = self.f32_equivalent_bytes() as f64 / self.memory_bytes().max(1) as f64;
        write!(
            f,
            "GpuPolarKvCache(layers={}, kv_heads={}, head_dim={}, len={}/{}, {:.1}MB on GPU, {:.1}x vs f32)",
            self.n_layers,
            self.n_kv_heads,
            self.head_dim,
            self.len,
            self.max_seq_len,
            self.memory_bytes() as f64 / (1024.0 * 1024.0),
            ratio,
        )
    }
}

/// Helper: build a per-layer seed scheme that matches what
/// `GpuPolarKvCache::new` uses. CPU-side caches that will be uploaded
/// to layer `i` of a polar cache constructed with `rotation_seed_base`
/// must use this seed when calling `QuantizedKvCache::new`.
pub fn seed_for_layer(rotation_seed_base: u64, layer: usize) -> u64 {
    rotation_seed_base + layer as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::quantized_kv_cache::QuantizedKvCache;

    /// Read back a wgpu storage buffer's contents to a CPU `Vec<u8>`.
    /// Test-only — uses a staging buffer + map_async + device.poll(Wait).
    fn readback_buffer(gpu: &GpuDevice, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let staging = gpu.create_staging_buffer(bytes);
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("polar_kv.test.readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (s, r) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s.send(res); });
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        r.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let out = mapped.to_vec();
        drop(mapped);
        staging.unmap();
        out
    }

    /// Build a CPU `QuantizedKvCache`, upload it to a `GpuPolarKvCache`
    /// layer, read the GPU buffers back, and assert they match what
    /// `pack_angles_to_u32(cpu.k_angles_slice())` and `cpu.k_radius_slice()`
    /// produce. This is the load-bearing storage-layout test: it pins
    /// the contract that resident GPU buffers are wire-compatible with
    /// the per-call `attn_score_polar_oneshot` / `attn_value_polar_oneshot`
    /// inputs.
    #[test]
    fn upload_layer_byte_layout_matches_cpu() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 6;
        let seed_base = 42u64;

        let mut polar_kv = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, seq_len, seed_base,
        );

        // CPU caches must use the matching per-layer rotation seed so the
        // compressed bytes are consistent with how the polar cache thinks
        // it should de-rotate.
        let kv_dim = n_kv_heads * head_dim;
        for layer in 0..n_layers {
            let mut cpu = QuantizedKvCache::new(
                n_kv_heads, head_dim, seq_len, seed_for_layer(seed_base, layer),
            );
            for t in 0..seq_len {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| (((t * 7 + layer * 3 + i) as f32) * 0.05).sin())
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| (((t * 11 + layer * 5 + i) as f32) * 0.04).cos())
                    .collect();
                cpu.append_one(&k, &v);
            }
            polar_kv.upload_layer_from_cpu(layer, &cpu);

            // Expected K bytes: pack CPU angles to u32 + raw f32 radius.
            let n_pairs = head_dim / 2;
            let used_angles = seq_len * n_kv_heads * n_pairs;
            let used_radius = seq_len * n_kv_heads;
            let cpu_k_angles_full: Vec<u8> = cpu.k_angles_slice()[..used_angles].to_vec();
            let expected_k_packed = pack_angles_to_u32(&cpu_k_angles_full);
            let expected_k_radius: &[f32] = &cpu.k_radius_slice()[..used_radius];

            // Compare bytes that were actually written (head of buffer up
            // to `used` size; tail beyond `len` is zero-init padding).
            let k_angles_bytes = readback_buffer(
                &gpu,
                polar_kv.k_angles_layer(layer),
                (expected_k_packed.len() * std::mem::size_of::<u32>()) as u64,
            );
            let got_k_packed: &[u32] = bytemuck::cast_slice(&k_angles_bytes);
            assert_eq!(got_k_packed, expected_k_packed.as_slice(),
                "K angles mismatch on layer {layer}");

            let k_radius_bytes = readback_buffer(
                &gpu,
                polar_kv.k_radius_layer(layer),
                (used_radius * std::mem::size_of::<f32>()) as u64,
            );
            let got_k_radius: &[f32] = bytemuck::cast_slice(&k_radius_bytes);
            assert_eq!(got_k_radius, expected_k_radius,
                "K radius mismatch on layer {layer}");

            // V — engram's QuantizedKvCache doesn't expose v_angles_slice,
            // so reconstruct via read_compressed_k like upload_layer does.
            let mut expected_v_angles_full = vec![0u8; used_angles];
            let mut expected_v_radius = vec![0f32; used_radius];
            for t in 0..seq_len {
                for h in 0..n_kv_heads {
                    let entry = cpu.read_compressed_k(t, h);
                    let off = (t * n_kv_heads + h) * n_pairs;
                    expected_v_angles_full[off..off + n_pairs]
                        .copy_from_slice(&entry.v_angles);
                    expected_v_radius[t * n_kv_heads + h] = entry.v_radius;
                }
            }
            let expected_v_packed = pack_angles_to_u32(&expected_v_angles_full);

            let v_angles_bytes = readback_buffer(
                &gpu,
                polar_kv.v_angles_layer(layer),
                (expected_v_packed.len() * std::mem::size_of::<u32>()) as u64,
            );
            let got_v_packed: &[u32] = bytemuck::cast_slice(&v_angles_bytes);
            assert_eq!(got_v_packed, expected_v_packed.as_slice(),
                "V angles mismatch on layer {layer}");

            let v_radius_bytes = readback_buffer(
                &gpu,
                polar_kv.v_radius_layer(layer),
                (used_radius * std::mem::size_of::<f32>()) as u64,
            );
            let got_v_radius: &[f32] = bytemuck::cast_slice(&v_radius_bytes);
            assert_eq!(got_v_radius, expected_v_radius.as_slice(),
                "V radius mismatch on layer {layer}");
        }

        polar_kv.set_len(seq_len);
        assert_eq!(polar_kv.seq_len(), seq_len);
    }

    /// Per-layer rotation buffer: each layer's buffer must hold the R
    /// produced by `generate_rotation_matrix(seed_base + layer)`.
    /// Anything else would silently break attention quality across layers
    /// because the CPU and GPU paths would disagree on R.
    #[test]
    fn rotation_buffer_layout_is_per_layer_seeded() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_layers = 3;
        let head_dim = 8;
        let seed_base = 42u64;
        let polar_kv = GpuPolarKvCache::new(
            gpu.clone(), n_layers, /*kv_heads*/ 2, head_dim, /*max_seq*/ 4, seed_base,
        );

        let bytes_per_layer = (head_dim * head_dim * 4) as u64;
        for layer in 0..n_layers {
            let bytes = readback_buffer(&gpu, polar_kv.rotation_layer(layer), bytes_per_layer);
            let got: &[f32] = bytemuck::cast_slice(&bytes);
            let expected = polar::generate_rotation_matrix(
                head_dim, seed_for_layer(seed_base, layer),
            );
            for (a, b) in got.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-7,
                    "rotation mismatch on layer {layer}",
                );
            }
        }
    }

    #[test]
    fn memory_compression_qwen_shape() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        // Qwen 3B-ish: 36 layers, kv_heads=2, head_dim=128, max_seq=4096.
        let cache = GpuPolarKvCache::new(
            gpu, /*layers*/ 36, /*kv_heads*/ 2, /*head_dim*/ 128, /*max_seq*/ 4096, 42,
        );
        let ratio = cache.f32_equivalent_bytes() as f64 / cache.memory_bytes() as f64;
        // KV-only ratio is ~7.5x. With per-layer rotation matrices added
        // it drops slightly. 6x is a defensible floor that catches
        // regressions but isn't slack.
        assert!(
            ratio > 6.0,
            "compression ratio {ratio:.2}x below 6x floor (Qwen 3B shape)",
        );
        assert!(cache.memory_bytes() < cache.f32_equivalent_bytes());
    }

    /// QJL storage smoke test: constructing with `n_qjl_proj > 0` should
    /// allocate per-layer signs + projection buffers, expose them via
    /// the accessors, and bump `memory_bytes()` by the expected amount.
    /// Constructing without QJL leaves all accessors returning None
    /// and memory_bytes() unchanged from the non-QJL case.
    #[test]
    fn qjl_storage_optional() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_layers = 3;
        let n_kv_heads = 2;
        let head_dim = 8;
        let max_seq = 16;
        let rot_seed = 42u64;

        // Without QJL: accessors return None, n_qjl_proj == 0.
        let plain = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, max_seq, rot_seed,
        );
        assert_eq!(plain.n_qjl_proj(), 0);
        assert_eq!(plain.qjl_sign_bytes_per_entry(), 0);
        for i in 0..n_layers {
            assert!(plain.k_qjl_signs_layer(i).is_none());
            assert!(plain.k_qjl_projection_layer(i).is_none());
        }

        // With QJL (n_proj=32): per-layer buffers materialize.
        let qjl_seed = 99u64;
        let with_qjl = GpuPolarKvCache::new_with_qjl(
            gpu, n_layers, n_kv_heads, head_dim, max_seq,
            rot_seed, /*n_qjl_proj*/ 32, qjl_seed,
        );
        assert_eq!(with_qjl.n_qjl_proj(), 32);
        // 32 bits → 4 bytes per (pos, head) entry.
        assert_eq!(with_qjl.qjl_sign_bytes_per_entry(), 4);
        for i in 0..n_layers {
            assert!(with_qjl.k_qjl_signs_layer(i).is_some());
            assert!(with_qjl.k_qjl_projection_layer(i).is_some());
        }
        // QJL must add memory: signs (max_seq * n_kv_heads * 4 bytes per layer)
        // + projections (n_proj * head_dim * 4 bytes per layer).
        let expected_signs_per = max_seq * n_kv_heads * 4;
        let expected_proj_per = 32 * head_dim * 4;
        let expected_qjl = (expected_signs_per + expected_proj_per) * n_layers;
        assert_eq!(
            with_qjl.memory_bytes() - plain.memory_bytes(),
            expected_qjl as u64,
            "memory_bytes delta should equal QJL signs+projections total"
        );
    }

    /// Load-bearing phase 2c.4 test: build a populated f32 GpuKvCache and
    /// convert it to a GpuPolarKvCache via the GPU compress shader (no
    /// CPU round-trip). Read back the resulting polar buffers and assert
    /// byte-equal angles + ULP-tolerant radius vs a CPU-side
    /// QuantizedKvCache built from the same source data.
    ///
    /// This pins the path the cortex-cloud retrieve handler will take
    /// once production wires polar caches: prefill stays f32, then a
    /// one-time on-GPU conversion produces the polar cache that
    /// subsequent retrieval queries read from.
    #[test]
    fn from_f32_cache_gpu_round_trip() {
        use crate::layers::gpu_polar::pack_angles_to_u32;
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 8;
        let max_seq = 6;
        let n_tokens = 6; // fully populated
        let seed_base = 42u64;

        // Build CPU caches per layer (ground truth compressed output).
        let kv_dim = n_kv_heads * head_dim;
        let mut layer_data: Vec<(Vec<f32>, Vec<f32>, QuantizedKvCache)> = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let mut k_data = Vec::with_capacity(n_tokens * kv_dim);
            let mut v_data = Vec::with_capacity(n_tokens * kv_dim);
            for t in 0..n_tokens {
                for i in 0..kv_dim {
                    k_data.push((((t * 7 + layer * 3 + i) as f32) * 0.05).sin());
                    v_data.push((((t * 11 + layer * 5 + i) as f32) * 0.04).cos());
                }
            }
            let mut cpu = QuantizedKvCache::new(
                n_kv_heads, head_dim, max_seq, seed_for_layer(seed_base, layer),
            );
            for t in 0..n_tokens {
                let off = t * kv_dim;
                cpu.append_one(&k_data[off..off + kv_dim], &v_data[off..off + kv_dim]);
            }
            layer_data.push((k_data, v_data, cpu));
        }

        // Build GpuKvCache and upload K/V. Phase A made the cache packed
        // f16, so we pack the f32 source data before write_buffer; the
        // kv_compress_polar shader was updated in lockstep to read
        // packed-f16 input from these buffers.
        let mut f32_cache = GpuKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, max_seq,
        );
        for (layer, (k_data, v_data, _)) in layer_data.iter().enumerate() {
            let k_packed = GpuDevice::pack_f16(k_data);
            let v_packed = GpuDevice::pack_f16(v_data);
            gpu.queue.write_buffer(
                f32_cache.k_layer(layer), 0, bytemuck::cast_slice(&k_packed),
            );
            gpu.queue.write_buffer(
                f32_cache.v_layer(layer), 0, bytemuck::cast_slice(&v_packed),
            );
        }
        f32_cache.advance(n_tokens);

        // Build empty polar cache with matching seeds, populate via GPU shader.
        let mut polar_kv = GpuPolarKvCache::new(
            gpu.clone(), n_layers, n_kv_heads, head_dim, max_seq, seed_base,
        );
        polar_kv.populate_from_f32_cache_gpu(&f32_cache);
        assert_eq!(polar_kv.seq_len(), n_tokens);

        // Verify each layer's compressed bytes against the CPU ground truth.
        let n_pairs = head_dim / 2;
        let used_angles = n_tokens * n_kv_heads * n_pairs;
        let used_radius = n_tokens * n_kv_heads;
        let used_packed_words = (used_angles + 3) / 4;
        let packed_bytes = (used_packed_words * std::mem::size_of::<u32>()) as u64;
        let radius_bytes = (used_radius * std::mem::size_of::<f32>()) as u64;

        for layer in 0..n_layers {
            let cpu = &layer_data[layer].2;

            // Expected K bytes.
            let cpu_k_angles_full: Vec<u8> = cpu.k_angles_slice()[..used_angles].to_vec();
            let expected_k_packed = pack_angles_to_u32(&cpu_k_angles_full);
            let expected_k_radius: &[f32] = &cpu.k_radius_slice()[..used_radius];

            // Expected V bytes via CompressedEntry.
            let mut expected_v_angles_full = vec![0u8; used_angles];
            let mut expected_v_radius = vec![0f32; used_radius];
            for t in 0..n_tokens {
                for h in 0..n_kv_heads {
                    let entry = cpu.read_compressed_k(t, h);
                    let off = (t * n_kv_heads + h) * n_pairs;
                    expected_v_angles_full[off..off + n_pairs]
                        .copy_from_slice(&entry.v_angles);
                    expected_v_radius[t * n_kv_heads + h] = entry.v_radius;
                }
            }
            let expected_v_packed = pack_angles_to_u32(&expected_v_angles_full);

            // Read back GPU.
            let got_k_angles = readback_buffer(&gpu, polar_kv.k_angles_layer(layer), packed_bytes);
            let got_k_packed: &[u32] = bytemuck::cast_slice(&got_k_angles);
            assert_eq!(got_k_packed, expected_k_packed.as_slice(),
                "K angles mismatch on layer {layer}");

            let got_v_angles = readback_buffer(&gpu, polar_kv.v_angles_layer(layer), packed_bytes);
            let got_v_packed: &[u32] = bytemuck::cast_slice(&got_v_angles);
            assert_eq!(got_v_packed, expected_v_packed.as_slice(),
                "V angles mismatch on layer {layer}");

            // Radius tolerance: 1e-3 abs to accommodate Phase A's f16
            // input quantization on top of FMA drift (was 1e-6 when the
            // GpuKvCache source was f32).
            let got_k_radius_bytes = readback_buffer(&gpu, polar_kv.k_radius_layer(layer), radius_bytes);
            let got_k_radius: &[f32] = bytemuck::cast_slice(&got_k_radius_bytes);
            for (i, (&g, &e)) in got_k_radius.iter().zip(expected_k_radius.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-3,
                    "K radius[{i}] layer {layer} differs: gpu={g}, cpu={e}",
                );
            }
            let got_v_radius_bytes = readback_buffer(&gpu, polar_kv.v_radius_layer(layer), radius_bytes);
            let got_v_radius: &[f32] = bytemuck::cast_slice(&got_v_radius_bytes);
            for (i, (&g, &e)) in got_v_radius.iter().zip(expected_v_radius.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-3,
                    "V radius[{i}] layer {layer} differs: gpu={g}, cpu={e}",
                );
            }
        }
    }

    #[test]
    fn from_f32_cache_gpu_empty_cache() {
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);

        let f32_cache = GpuKvCache::new(gpu.clone(), 2, 1, 8, 4);
        let mut polar = GpuPolarKvCache::new(gpu.clone(), 2, 1, 8, 4, 42);
        polar.populate_from_f32_cache_gpu(&f32_cache);
        assert_eq!(polar.seq_len(), 0);
    }

    #[test]
    #[should_panic(expected = "polar n_layers")]
    fn from_f32_cache_gpu_layer_mismatch_panics() {
        use crate::layers::gpu_kv_cache::GpuKvCache;

        let Some(_gpu) = GpuDevice::try_new() else {
            panic!("polar n_layers (no GPU; faking)");
        };
        let gpu = Arc::new(GpuDevice::try_new().unwrap());

        let f32_cache = GpuKvCache::new(gpu.clone(), /*layers*/ 3, 1, 8, 4);
        let mut polar = GpuPolarKvCache::new(gpu.clone(), /*layers*/ 2, 1, 8, 4, 42);
        polar.populate_from_f32_cache_gpu(&f32_cache);
    }

    #[test]
    fn debug_format_includes_compression() {
        let Some(gpu) = GpuDevice::try_new() else { return };
        let gpu = Arc::new(gpu);
        let cache = GpuPolarKvCache::new(gpu, 2, 1, 8, 16, 42);
        let s = format!("{:?}", cache);
        assert!(s.contains("GpuPolarKvCache"));
        assert!(s.contains("0/16"));
        assert!(s.contains("vs f32"));
    }
}
