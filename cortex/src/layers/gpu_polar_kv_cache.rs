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

    /// Per-layer rotation matrices, packed into one buffer with row stride
    /// `head_dim * head_dim`. Layer i lives at offset `i * head_dim^2`.
    /// Resident as f32 so the de-rotation shader can index directly.
    rotation_buffer: wgpu::Buffer,

    /// Shared angle LUT uniform: vec4<f32>[8] (cos, sin, _, _) per bucket.
    lut_buffer: wgpu::Buffer,

    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
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
        assert!(head_dim % 2 == 0, "head_dim must be even");
        assert!(n_layers > 0 && n_kv_heads > 0 && head_dim > 0 && max_seq_len > 0);

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

        // Per-layer rotation matrices, packed into one f32 buffer of size
        // n_layers * head_dim^2. Build on CPU (deterministic from seed),
        // upload once.
        let mut rotation_data = Vec::with_capacity(n_layers * head_dim * head_dim);
        for i in 0..n_layers {
            let r = polar::generate_rotation_matrix(head_dim, rotation_seed_base + i as u64);
            rotation_data.extend_from_slice(&r);
        }
        let rotation_buffer = gpu.create_storage_buffer(
            bytemuck::cast_slice(&rotation_data),
            "polar_kv.rotation_per_layer",
        );

        // Shared angle LUT uniform.
        let lut = polar_lut_vec4();
        let lut_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polar_kv.angle_lut"),
            size: std::mem::size_of_val(&lut) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut));

        Self {
            gpu,
            k_angles_buffers,
            k_radius_buffers,
            v_angles_buffers,
            v_radius_buffers,
            rotation_buffer,
            lut_buffer,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
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
    pub fn rotation_buffer(&self) -> &wgpu::Buffer { &self.rotation_buffer }
    pub fn lut_buffer(&self) -> &wgpu::Buffer { &self.lut_buffer }

    /// VRAM bytes used (compressed K+V + rotation; LUT excluded).
    pub fn memory_bytes(&self) -> u64 {
        let n_pairs = self.head_dim / 2;
        let angles_per = (self.max_seq_len * self.n_kv_heads * n_pairs + 3) / 4 * 4;
        let radius_per = self.max_seq_len * self.n_kv_heads * 4;
        let kv_per_layer = (2 * angles_per + 2 * radius_per) as u64; // K + V
        let rot = (self.n_layers * self.head_dim * self.head_dim * 4) as u64;
        kv_per_layer * self.n_layers as u64 + rot
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
        gpu.device.poll(wgpu::Maintain::Wait);
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

    /// Per-layer rotation buffer: the layout must be one R per layer at
    /// stride `head_dim^2`, deterministic from `(seed_base + layer)`.
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

        let total_bytes = (n_layers * head_dim * head_dim * 4) as u64;
        let bytes = readback_buffer(&gpu, polar_kv.rotation_buffer(), total_bytes);
        let got: &[f32] = bytemuck::cast_slice(&bytes);

        for layer in 0..n_layers {
            let expected = polar::generate_rotation_matrix(
                head_dim, seed_for_layer(seed_base, layer),
            );
            let off = layer * head_dim * head_dim;
            let slice = &got[off..off + head_dim * head_dim];
            for (a, b) in slice.iter().zip(expected.iter()) {
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
