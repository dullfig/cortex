//! QuantizedKvCache — TurboQuant-compressed KV storage (CPU side).
//!
//! Replaces the f32 KV cache with 3-bit PolarQuant angles + per-head radius,
//! optionally with QJL 1-bit residual correction. Achieves ~12x memory
//! reduction over f32, turning the KV cache from the dominant memory consumer
//! into a fraction of the model weights.
//!
//! Counterpart to `KvCache` (uncompressed CPU) and `GpuKvCache` (resident
//! GPU). The GPU compressed variant is in a separate module and shares the
//! same on-disk byte layout for zero-copy upload.

use crate::ops::polar::{self, AngleLUT};
use crate::ops::qjl::QjlProjection;

/// 3-bit quantized KV cache for one transformer layer.
pub struct QuantizedKvCache {
    // -- PolarQuant compressed storage --
    /// Quantized angle indices for K cache.
    /// Shape: [max_seq_len, n_kv_heads, head_dim/2] stored flat.
    /// Each byte holds one angle bucket (0..7). Future: bit-pack to 3 bits.
    k_angles: Vec<u8>,

    /// Quantized angle indices for V cache (same layout).
    v_angles: Vec<u8>,

    /// Per-position, per-head radius scale for K.
    /// Shape: [max_seq_len, n_kv_heads] stored flat.
    k_radius: Vec<f32>,

    /// Per-position, per-head radius scale for V.
    v_radius: Vec<f32>,

    // -- QJL correction (optional) --
    /// Packed sign bits for K residual correction.
    /// Shape: [max_seq_len, n_kv_heads, sign_bytes] stored flat.
    k_qjl_signs: Option<Vec<u8>>,

    /// Packed sign bits for V residual correction. Phase O: sized by
    /// the V projection set (`DEFAULT_V_N_PROJECTIONS` = 256 → 32
    /// bytes/entry), NOT by the K set — V reconstructs a full residual
    /// vector, which needs far more bits than K's scalar correction.
    v_qjl_signs: Option<Vec<u8>>,

    /// Per-entry V residual norms (Phase O). Shape:
    /// [max_seq_len, n_kv_heads]. The Γ-scaled estimator multiplies
    /// the sign-mean direction by this stored norm — see
    /// `QjlProjection::reconstruct_residual`.
    v_residual_norm: Option<Vec<f32>>,

    // -- Fixed per-layer state (regenerated from seed, not stored) --
    /// Orthogonal rotation matrix [head_dim, head_dim].
    rotation_matrix: Vec<f32>,

    /// Angle lookup table (shared across all positions/heads).
    lut: AngleLUT,

    /// K-side QJL projections (None if QJL disabled). n_proj = 32.
    qjl: Option<QjlProjection>,

    /// V-side QJL projections (None if QJL disabled). n_proj = 256
    /// (`DEFAULT_V_N_PROJECTIONS`), seed derived from the K seed so
    /// the two sets are decorrelated but deterministic.
    v_qjl: Option<QjlProjection>,

    // -- Dimensions --
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,
}

/// Raw compressed data for a single (position, head) entry.
///
/// Used for zero-loss tier migration — copy bytes between caches
/// without dequant/requant error accumulation.
#[derive(Debug, Clone)]
pub struct CompressedEntry {
    /// Quantized angle indices for K (length: head_dim/2).
    pub k_angles: Vec<u8>,
    /// Quantized angle indices for V (length: head_dim/2).
    pub v_angles: Vec<u8>,
    /// K radius scale.
    pub k_radius: f32,
    /// V radius scale.
    pub v_radius: f32,
    /// Optional QJL sign bits (k_signs, v_signs). NOTE: since Phase O
    /// the two have different lengths (K: n_proj=32 → 4 bytes; V:
    /// n_proj=256 → 32 bytes).
    pub qjl_signs: Option<(Vec<u8>, Vec<u8>)>,
    /// V residual norm (Phase O); present iff `qjl_signs` is.
    pub v_residual_norm: Option<f32>,
}

impl QuantizedKvCache {
    /// Create a new compressed cache for one layer (PolarQuant only, no QJL).
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize, seed: u64) -> Self {
        assert_eq!(head_dim % 2, 0, "head_dim must be even for polar pairs");

        let n_pairs = head_dim / 2;
        let angle_capacity = max_seq_len * n_kv_heads * n_pairs;
        let radius_capacity = max_seq_len * n_kv_heads;

        Self {
            k_angles: vec![0u8; angle_capacity],
            v_angles: vec![0u8; angle_capacity],
            k_radius: vec![0.0f32; radius_capacity],
            v_radius: vec![0.0f32; radius_capacity],
            k_qjl_signs: None,
            v_qjl_signs: None,
            v_residual_norm: None,
            rotation_matrix: polar::generate_rotation_matrix(head_dim, seed),
            lut: AngleLUT::new(),
            qjl: None,
            v_qjl: None,
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Create with QJL correction enabled (K: 32 projections; V: 256
    /// projections + stored residual norm — see Phase O notes on the
    /// `v_qjl_signs` field).
    pub fn with_qjl(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rotation_seed: u64,
        qjl_seed: u64,
    ) -> Self {
        let mut cache = Self::new(n_kv_heads, head_dim, max_seq_len, rotation_seed);

        let qjl = QjlProjection::new(head_dim, qjl_seed);
        let k_sign_capacity = max_seq_len * n_kv_heads * qjl.sign_bytes();

        // V projections: distinct deterministic seed (golden-ratio
        // offset) so K and V sets are decorrelated.
        let v_qjl = QjlProjection::with_n_projections(
            head_dim,
            crate::ops::qjl::DEFAULT_V_N_PROJECTIONS,
            qjl_seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        );
        let v_sign_capacity = max_seq_len * n_kv_heads * v_qjl.sign_bytes();

        cache.k_qjl_signs = Some(vec![0u8; k_sign_capacity]);
        cache.v_qjl_signs = Some(vec![0u8; v_sign_capacity]);
        cache.v_residual_norm = Some(vec![0.0f32; max_seq_len * n_kv_heads]);
        cache.qjl = Some(qjl);
        cache.v_qjl = Some(v_qjl);
        cache
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append K and V vectors for one new position.
    ///
    /// `key`: f32 slice of length `n_kv_heads * head_dim`.
    /// `value`: f32 slice of same length.
    /// These should already have RoPE applied (for keys).
    pub fn append_one(&mut self, key: &[f32], value: &[f32]) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        assert_eq!(key.len(), kv_dim);
        assert_eq!(value.len(), kv_dim);
        assert!(self.len < self.max_seq_len, "cache overflow");

        let pos = self.len;
        let n_pairs = self.head_dim / 2;
        let mut rotated = vec![0.0f32; self.head_dim];

        for head in 0..self.n_kv_heads {
            let head_off = head * self.head_dim;

            // --- Compress K ---
            let k_vec = &key[head_off..head_off + self.head_dim];
            polar::rotate(&self.rotation_matrix, k_vec, &mut rotated);

            let (angles, radius) = polar::to_polar_quantized(&rotated);
            let angle_off = (pos * self.n_kv_heads + head) * n_pairs;
            self.k_angles[angle_off..angle_off + n_pairs].copy_from_slice(&angles);
            self.k_radius[pos * self.n_kv_heads + head] = radius;

            // QJL correction for K.
            if let (Some(qjl), Some(signs_buf)) = (&self.qjl, &mut self.k_qjl_signs) {
                let reconstructed = polar::from_polar_quantized(&angles, radius, &self.lut);
                let residual: Vec<f32> = rotated
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(&r, &q)| r - q)
                    .collect();
                let signs = qjl.encode_signs(&residual);
                let sign_bytes = qjl.sign_bytes();
                let sign_off = (pos * self.n_kv_heads + head) * sign_bytes;
                signs_buf[sign_off..sign_off + sign_bytes].copy_from_slice(&signs);
            }

            // --- Compress V ---
            let v_vec = &value[head_off..head_off + self.head_dim];
            polar::rotate(&self.rotation_matrix, v_vec, &mut rotated);

            let (angles, radius) = polar::to_polar_quantized(&rotated);
            self.v_angles[angle_off..angle_off + n_pairs].copy_from_slice(&angles);
            self.v_radius[pos * self.n_kv_heads + head] = radius;

            // QJL correction for V (Phase O): 256-projection signs +
            // stored residual norm for the Γ-scaled estimator.
            if let (Some(v_qjl), Some(signs_buf), Some(rnorm_buf)) =
                (&self.v_qjl, &mut self.v_qjl_signs, &mut self.v_residual_norm)
            {
                let reconstructed = polar::from_polar_quantized(&angles, radius, &self.lut);
                let residual: Vec<f32> = rotated
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(&r, &q)| r - q)
                    .collect();
                let signs = v_qjl.encode_signs(&residual);
                let sign_bytes = v_qjl.sign_bytes();
                let sign_off = (pos * self.n_kv_heads + head) * sign_bytes;
                signs_buf[sign_off..sign_off + sign_bytes].copy_from_slice(&signs);
                rnorm_buf[pos * self.n_kv_heads + head] =
                    residual.iter().map(|r| r * r).sum::<f32>().sqrt();
            }
        }

        self.len += 1;
    }

    /// Compute dot(query, cached_key[pos, head]) in compressed domain.
    ///
    /// The query should be in original (non-rotated) space. This method
    /// rotates it internally and dots against the compressed K directly.
    pub fn dot_key(&self, pos: usize, kv_head: usize, query: &[f32]) -> f32 {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);
        debug_assert_eq!(query.len(), self.head_dim);

        let n_pairs = self.head_dim / 2;

        // Rotate query into compressed domain.
        let mut rq = vec![0.0f32; self.head_dim];
        polar::rotate(&self.rotation_matrix, query, &mut rq);

        // Dot against compressed K using angle LUT.
        let angle_off = (pos * self.n_kv_heads + kv_head) * n_pairs;
        let radius = self.k_radius[pos * self.n_kv_heads + kv_head];

        let mut sum = 0.0f32;
        for i in 0..n_pairs {
            let bucket = self.k_angles[angle_off + i] as usize;
            sum += rq[2 * i] * self.lut.cos[bucket] + rq[2 * i + 1] * self.lut.sin[bucket];
        }
        sum *= radius;

        // QJL correction.
        if let (Some(qjl), Some(signs_buf)) = (&self.qjl, &self.k_qjl_signs) {
            let sign_bytes = qjl.sign_bytes();
            let sign_off = (pos * self.n_kv_heads + kv_head) * sign_bytes;
            let signs = &signs_buf[sign_off..sign_off + sign_bytes];
            sum += qjl.correction_dot(signs, &rq);
        }

        sum
    }

    /// Dequantize the V vector at (pos, head) back to f32.
    ///
    /// Returns a Vec<f32> of length head_dim in original (non-rotated) space.
    ///
    /// Phase O: when QJL is enabled the stored V residual signs (written
    /// by `append_one` since the original QJL landing, but unread until
    /// now) refine the LUT dequant: `v̂ += reconstruct_residual(signs)`
    /// in rotated space, before the unrotate. This is the V-side
    /// counterpart of `dot_key`'s K correction and closes most of the
    /// attention-output cosine gap PolarQuant alone leaves (~0.84).
    pub fn value_at_dequant(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);

        let n_pairs = self.head_dim / 2;
        let angle_off = (pos * self.n_kv_heads + kv_head) * n_pairs;
        let radius = self.v_radius[pos * self.n_kv_heads + kv_head];
        let angles = &self.v_angles[angle_off..angle_off + n_pairs];

        let mut rotated = polar::from_polar_quantized(angles, radius, &self.lut);

        if let (Some(v_qjl), Some(signs_buf), Some(rnorm_buf)) =
            (&self.v_qjl, &self.v_qjl_signs, &self.v_residual_norm)
        {
            let sign_bytes = v_qjl.sign_bytes();
            let sign_off = (pos * self.n_kv_heads + kv_head) * sign_bytes;
            let signs = &signs_buf[sign_off..sign_off + sign_bytes];
            let rnorm = rnorm_buf[pos * self.n_kv_heads + kv_head];
            let residual = v_qjl.reconstruct_residual(signs, rnorm);
            for (r, d) in rotated.iter_mut().zip(&residual) {
                *r += d;
            }
        }

        let mut out = vec![0.0f32; self.head_dim];
        polar::rotate_transpose(&self.rotation_matrix, &rotated, &mut out);
        out
    }

    /// Dequantize the K vector at (pos, head) back to f32.
    ///
    /// Returns a Vec<f32> of length head_dim in original (non-rotated) space.
    pub fn key_at_dequant(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);

        let n_pairs = self.head_dim / 2;
        let angle_off = (pos * self.n_kv_heads + kv_head) * n_pairs;
        let radius = self.k_radius[pos * self.n_kv_heads + kv_head];
        let angles = &self.k_angles[angle_off..angle_off + n_pairs];

        let rotated = polar::from_polar_quantized(angles, radius, &self.lut);

        let mut out = vec![0.0f32; self.head_dim];
        polar::rotate_transpose(&self.rotation_matrix, &rotated, &mut out);
        out
    }

    /// Remaining capacity (positions that can still be appended).
    pub fn remaining(&self) -> usize {
        self.max_seq_len - self.len
    }

    /// Read compressed data for a single (pos, head) entry — zero-loss tier migration.
    pub fn read_compressed_k(&self, pos: usize, head: usize) -> CompressedEntry {
        debug_assert!(pos < self.len);
        debug_assert!(head < self.n_kv_heads);

        let n_pairs = self.head_dim / 2;
        let angle_off = (pos * self.n_kv_heads + head) * n_pairs;
        let radius_off = pos * self.n_kv_heads + head;

        let k_angles = self.k_angles[angle_off..angle_off + n_pairs].to_vec();
        let v_angles = self.v_angles[angle_off..angle_off + n_pairs].to_vec();
        let k_radius = self.k_radius[radius_off];
        let v_radius = self.v_radius[radius_off];

        let qjl_signs = if let (Some(qjl), Some(v_qjl), Some(k_signs), Some(v_signs)) =
            (&self.qjl, &self.v_qjl, &self.k_qjl_signs, &self.v_qjl_signs)
        {
            let ksb = qjl.sign_bytes();
            let vsb = v_qjl.sign_bytes();
            let k_off = (pos * self.n_kv_heads + head) * ksb;
            let v_off = (pos * self.n_kv_heads + head) * vsb;
            Some((
                k_signs[k_off..k_off + ksb].to_vec(),
                v_signs[v_off..v_off + vsb].to_vec(),
            ))
        } else {
            None
        };
        let v_residual_norm = self
            .v_residual_norm
            .as_ref()
            .map(|buf| buf[pos * self.n_kv_heads + head]);

        CompressedEntry {
            k_angles,
            v_angles,
            k_radius,
            v_radius,
            qjl_signs,
            v_residual_norm,
        }
    }

    /// Append a compressed entry directly — zero-loss tier migration (no dequant/requant).
    pub fn append_compressed(&mut self, entry: &CompressedEntry, head: usize) {
        debug_assert!(head < self.n_kv_heads);
        assert!(self.len < self.max_seq_len, "cache overflow");

        let n_pairs = self.head_dim / 2;
        debug_assert_eq!(entry.k_angles.len(), n_pairs);

        let pos = self.len;
        let angle_off = (pos * self.n_kv_heads + head) * n_pairs;
        let radius_off = pos * self.n_kv_heads + head;

        self.k_angles[angle_off..angle_off + n_pairs].copy_from_slice(&entry.k_angles);
        self.v_angles[angle_off..angle_off + n_pairs].copy_from_slice(&entry.v_angles);
        self.k_radius[radius_off] = entry.k_radius;
        self.v_radius[radius_off] = entry.v_radius;

        if let (Some(qjl), Some(v_qjl), Some(k_signs), Some(v_signs), Some((ek, ev))) = (
            &self.qjl,
            &self.v_qjl,
            &mut self.k_qjl_signs,
            &mut self.v_qjl_signs,
            &entry.qjl_signs,
        ) {
            let ksb = qjl.sign_bytes();
            let vsb = v_qjl.sign_bytes();
            let k_off = (pos * self.n_kv_heads + head) * ksb;
            let v_off = (pos * self.n_kv_heads + head) * vsb;
            k_signs[k_off..k_off + ksb].copy_from_slice(ek);
            v_signs[v_off..v_off + vsb].copy_from_slice(ev);
        }
        if let (Some(rnorm_buf), Some(rn)) = (&mut self.v_residual_norm, entry.v_residual_norm) {
            rnorm_buf[pos * self.n_kv_heads + head] = rn;
        }
    }

    /// Advance len by 1 — call after appending compressed entries for ALL heads at a position.
    pub fn advance_len(&mut self) {
        assert!(self.len < self.max_seq_len, "cache overflow");
        self.len += 1;
    }

    /// Reset the cache (reuse allocations).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Raw K angle bytes slice — for GPU upload.
    pub fn k_angles_slice(&self) -> &[u8] {
        let n_pairs = self.head_dim / 2;
        let used = self.len * self.n_kv_heads * n_pairs;
        &self.k_angles[..used]
    }

    /// Raw K radius slice — for GPU upload.
    pub fn k_radius_slice(&self) -> &[f32] {
        let used = self.len * self.n_kv_heads;
        &self.k_radius[..used]
    }

    /// Memory usage in bytes (compressed storage only, excludes rotation matrix).
    pub fn memory_bytes(&self) -> usize {
        let n_pairs = self.head_dim / 2;
        let angle_bytes = 2 * self.max_seq_len * self.n_kv_heads * n_pairs; // k + v
        let radius_bytes = 2 * self.max_seq_len * self.n_kv_heads * 4; // k + v, f32
        let qjl_bytes = self
            .qjl
            .as_ref()
            .map(|q| 2 * self.max_seq_len * self.n_kv_heads * q.sign_bytes())
            .unwrap_or(0);
        angle_bytes + radius_bytes + qjl_bytes
    }

    /// Equivalent f32 cache size for comparison.
    pub fn f32_equivalent_bytes(&self) -> usize {
        2 * self.max_seq_len * self.n_kv_heads * self.head_dim * 4
    }
}

impl std::fmt::Debug for QuantizedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ratio = self.f32_equivalent_bytes() as f64 / self.memory_bytes().max(1) as f64;
        write!(
            f,
            "QuantizedKvCache(kv_heads={}, head_dim={}, len={}/{}, {:.1}KB, {:.1}x compression{})",
            self.n_kv_heads,
            self.head_dim,
            self.len,
            self.max_seq_len,
            self.memory_bytes() as f64 / 1024.0,
            ratio,
            if self.qjl.is_some() { " +QJL" } else { "" },
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_empty() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_seq_len(), 2048);
    }

    #[test]
    fn append_and_len() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];

        cache.append_one(&key, &value);
        assert_eq!(cache.len(), 1);

        cache.append_one(&key, &value);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn dot_key_is_finite() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];
        cache.append_one(&key, &value);

        let query = vec![1.0f32; 8];
        let dot = cache.dot_key(0, 0, &query);
        assert!(dot.is_finite(), "dot should be finite, got {dot}");
    }

    #[test]
    fn value_dequant_is_finite() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];
        cache.append_one(&key, &value);

        let v = cache.value_at_dequant(0, 0);
        assert_eq!(v.len(), 8);
        for &val in &v {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn compression_ratio() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        let compressed = cache.memory_bytes();
        let f32_equiv = cache.f32_equivalent_bytes();
        let ratio = f32_equiv as f64 / compressed as f64;
        // Should be roughly 10-16x compression.
        assert!(ratio > 5.0, "expected significant compression, got {ratio:.1}x");
    }

    #[test]
    fn with_qjl_has_higher_memory() {
        let without = QuantizedKvCache::new(4, 64, 2048, 42);
        let with = QuantizedKvCache::with_qjl(4, 64, 2048, 42, 99);
        assert!(with.memory_bytes() > without.memory_bytes());
    }

    /// Phase O: the stored V residual signs must measurably improve
    /// `value_at_dequant` fidelity vs PolarQuant alone. Random vectors
    /// at Qwen head_dim (128), n_proj = 32 default.
    #[test]
    fn v_qjl_correction_improves_dequant_cosine() {
        let n_kv_heads = 2;
        let head_dim = 128;
        let kv_dim = n_kv_heads * head_dim;
        let n_vecs = 16;

        let mut plain = QuantizedKvCache::new(n_kv_heads, head_dim, 64, 42);
        let mut qjl = QuantizedKvCache::with_qjl(n_kv_heads, head_dim, 64, 42, 99);

        // Deterministic pseudo-random vectors (LCG, same recipe as the
        // gpu_floatlinear tests).
        let mut rng: u64 = 0xC0FFEE42;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as i32 % 1000 - 500) as f32 * 0.002
        };
        let mut originals: Vec<Vec<f32>> = Vec::new();
        for _ in 0..n_vecs {
            let key: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            let value: Vec<f32> = (0..kv_dim).map(|_| next()).collect();
            plain.append_one(&key, &value);
            qjl.append_one(&key, &value);
            originals.push(value);
        }

        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }

        let mut sum_plain = 0.0f32;
        let mut sum_qjl = 0.0f32;
        for (pos, original) in originals.iter().enumerate() {
            for h in 0..n_kv_heads {
                let orig = &original[h * head_dim..(h + 1) * head_dim];
                sum_plain += cosine(&plain.value_at_dequant(pos, h), orig);
                sum_qjl += cosine(&qjl.value_at_dequant(pos, h), orig);
            }
        }
        let n = (n_vecs * n_kv_heads) as f32;
        let mean_plain = sum_plain / n;
        let mean_qjl = sum_qjl / n;

        println!("V dequant cosine: plain={mean_plain:.4} qjl256={mean_qjl:.4}");
        // Strict improvement, by a real margin — not float noise.
        assert!(
            mean_qjl > mean_plain + 0.05,
            "V QJL-256 correction should improve dequant cosine substantially: \
             plain={mean_plain:.4} qjl={mean_qjl:.4}",
        );
        // Absolute floor: theory says ~0.91 at n_proj=256/head_dim=128
        // from a 0.797 polar baseline. Pin conservatively below.
        assert!(
            mean_qjl > 0.88,
            "V QJL-256 dequant cosine below expected floor: {mean_qjl:.4}",
        );
    }

    #[test]
    fn clear_resets_len() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn overflow_panics() {
        let mut cache = QuantizedKvCache::new(1, 4, 2, 42);
        let kv_dim = 4;
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]); // 3 > 2
    }

    #[test]
    fn key_at_dequant_roundtrip() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        // Use varied values so the key isn't degenerate.
        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let value = vec![0.3f32; kv_dim];
        cache.append_one(&key, &value);

        let k = cache.key_at_dequant(0, 0);
        assert_eq!(k.len(), 8);
        for &val in &k {
            assert!(val.is_finite());
        }
        // Quantization loses precision, but the direction should be roughly preserved.
        let norm: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "dequantized key should be nonzero");
    }

    #[test]
    fn compressed_entry_roundtrip() {
        let mut src = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let value = vec![0.3f32; kv_dim];
        src.append_one(&key, &value);

        // Read compressed, write to a fresh cache.
        let mut dst = QuantizedKvCache::new(2, 8, 100, 42);
        for head in 0..2 {
            let entry = src.read_compressed_k(0, head);
            dst.append_compressed(&entry, head);
        }
        dst.advance_len();

        assert_eq!(dst.len(), 1);

        // Verify the K dot product matches between src and dst.
        let query = vec![1.0f32; 8];
        let dot_src = src.dot_key(0, 0, &query);
        let dot_dst = dst.dot_key(0, 0, &query);
        assert!(
            (dot_src - dot_dst).abs() < 1e-6,
            "compressed roundtrip should be lossless: {dot_src} vs {dot_dst}"
        );
    }

    #[test]
    fn remaining_capacity() {
        let mut cache = QuantizedKvCache::new(1, 4, 5, 42);
        assert_eq!(cache.remaining(), 5);
        cache.append_one(&vec![1.0; 4], &vec![2.0; 4]);
        assert_eq!(cache.remaining(), 4);
    }

    #[test]
    fn debug_format() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("QuantizedKvCache"));
        assert!(debug.contains("0/2048"));
        assert!(debug.contains("compression"));
    }

    /// Sanity check on real-world Qwen 3B shape (per layer):
    /// 2 (KV) × 4096 max_seq × 2 (kv_heads) × 128 (head_dim) × 4 = 8 MB f32.
    /// Compressed (one byte per angle bucket, no bit-pack):
    ///   angles  = 2 × 4096 × 2 × 64 (n_pairs) = 1 MB
    ///   radius  = 2 × 4096 × 2 × 4 = 64 KB
    /// ≈ 1.06 MB total → ~7.5x compression at the cache level.
    /// Future: bit-pack 3 bits per angle would push this to ~12x.
    #[test]
    fn qwen_shape_compression_target() {
        let cache = QuantizedKvCache::new(/*kv_heads*/ 2, /*head_dim*/ 128, /*max_seq*/ 4096, 42);
        let ratio = cache.f32_equivalent_bytes() as f64 / cache.memory_bytes() as f64;
        // Current u8-per-angle representation lands ~7.5x. Tighter bound is
        // a future bit-packing follow-up.
        assert!(ratio >= 7.0, "PolarQuant should hit ~7.5x compression on Qwen shape, got {ratio:.2}x");
    }
}
