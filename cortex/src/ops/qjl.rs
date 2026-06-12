//! QJL — Quantized Johnson-Lindenstrauss residual correction.
//!
//! Stage 2 of TurboQuant. Stores 1-bit signs of random projections of the
//! PolarQuant quantization residual. Provides unbiased correction to dot
//! product estimates, improving attention score accuracy without significant
//! storage overhead.
//!
//! Each projection adds 1 bit per (position, head). 32 projections = 4 bytes
//! per cache entry — negligible compared to the angle storage.

/// Number of random projections for QJL correction.
/// 32 gives a good accuracy/storage tradeoff.
const DEFAULT_N_PROJECTIONS: usize = 32;

/// Number of projections for the V-side residual correction (Phase O).
///
/// V needs far more bits than K: the K correction is effectively a
/// 1-dimensional quantity (a score adjustment along the query), while V
/// must reconstruct the full residual VECTOR. At head_dim = 128,
/// direction quality scales as sqrt(nκ²/(1+nκ²)) with κ² = 2/(π·d):
/// n=32 → dir-cos 0.38, n=128 → 0.62, n=256 → 0.75 (≈0.91 dequant
/// cosine after correction vs 0.797 for PolarQuant alone). 256 signs =
/// 32 B/entry; with angles + radius + rnorm the V entry is 104 B vs
/// 256 B f16 — still 2.5x compression on V.
pub const DEFAULT_V_N_PROJECTIONS: usize = 256;

/// Scale constant for the V residual estimate (Phase O):
///
/// ```text
/// r̂ = Γ(n,d) · ||r|| · (1/n) Σ_j sign_j · p_j
/// Γ(n,d) = sqrt(nκ² / (1+nκ²)) / sqrt(κ² + (1−κ²)/n),  κ² = 2/(π·d)
/// ```
///
/// The numerator is the expected direction-cosine of the sign-mean
/// against the true residual; the denominator is the expected norm of
/// the sign-mean. Using the analytic constant instead of per-vector
/// normalization keeps the estimate LINEAR in the sign bits, which is
/// what lets the GPU value shader sum-swap the correction
/// (`C_j = Σ_t w_t·rnorm_t·s_tj` then one 256-MAC pass per output
/// element). Numerically validated: at n=256, d=128, Γ = 7.951 achieves
/// residual-energy ratio 0.436 vs the theoretical optimum 0.44.
pub fn v_correction_gamma(n_proj: usize, dim: usize) -> f32 {
    let k2 = 2.0 / (std::f32::consts::PI * dim as f32);
    let nk2 = n_proj as f32 * k2;
    let dir_cos2 = nk2 / (1.0 + nk2);
    let mv_norm2 = k2 + (1.0 - k2) / n_proj as f32;
    (dir_cos2 / mv_norm2).sqrt()
}

/// Random projection matrix + encoding/decoding for QJL correction.
pub struct QjlProjection {
    /// Random projection vectors: `[n_proj, dim]`, row-major.
    projections: Vec<f32>,
    /// Number of projections.
    n_proj: usize,
    /// Vector dimension.
    dim: usize,
}

impl QjlProjection {
    /// Generate deterministic random projections from a seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self::with_n_projections(dim, DEFAULT_N_PROJECTIONS, seed)
    }

    /// Generate with a specific number of projections.
    pub fn with_n_projections(dim: usize, n_proj: usize, seed: u64) -> Self {
        // Reuse the same RNG strategy as polar.rs.
        let mut rng = super::polar::SimpleRng::seed(seed);
        let mut projections = vec![0.0f32; n_proj * dim];

        // Random Gaussian projections, normalized to unit length.
        for row in 0..n_proj {
            let row_off = row * dim;
            let mut norm = 0.0f32;
            for col in 0..dim {
                let v = rng.next_normal();
                projections[row_off + col] = v;
                norm += v * v;
            }
            let inv_norm = 1.0 / norm.sqrt();
            for col in 0..dim {
                projections[row_off + col] *= inv_norm;
            }
        }

        Self {
            projections,
            n_proj,
            dim,
        }
    }

    /// Number of projections.
    pub fn n_proj(&self) -> usize {
        self.n_proj
    }

    /// Bytes needed to store sign bits for one vector.
    pub fn sign_bytes(&self) -> usize {
        (self.n_proj + 7) / 8
    }

    /// Raw projection matrix — `[n_proj, dim]` row-major f32. Exposed
    /// so callers (like `GpuPolarKvCache`) can upload it to a GPU
    /// storage buffer for shader use.
    pub fn projections(&self) -> &[f32] {
        &self.projections
    }

    /// Encode the residual as packed sign bits.
    ///
    /// `residual`: the quantization error vector (rotated - dequantized) of
    /// length `dim`.
    /// Returns packed sign bits (1 = non-negative, 0 = negative).
    pub fn encode_signs(&self, residual: &[f32]) -> Vec<u8> {
        debug_assert_eq!(residual.len(), self.dim);

        let n_bytes = self.sign_bytes();
        let mut signs = vec![0u8; n_bytes];

        for j in 0..self.n_proj {
            let row_off = j * self.dim;
            let dot: f32 = residual
                .iter()
                .zip(&self.projections[row_off..row_off + self.dim])
                .map(|(&r, &p)| r * p)
                .sum();

            if dot >= 0.0 {
                signs[j / 8] |= 1 << (j % 8);
            }
        }

        signs
    }

    /// Compute the QJL correction term for a dot product.
    ///
    /// `signs`: packed sign bits from `encode_signs`.
    /// `query`: the query vector (in rotated space) of length `dim`.
    /// Returns the correction to add to the PolarQuant dot product estimate.
    pub fn correction_dot(&self, signs: &[u8], query: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), self.dim);

        // Precompute dot(query, projection[j]) for each projection.
        let mut correction = 0.0f32;

        for j in 0..self.n_proj {
            let row_off = j * self.dim;
            let qp_dot: f32 = query
                .iter()
                .zip(&self.projections[row_off..row_off + self.dim])
                .map(|(&q, &p)| q * p)
                .sum();

            let sign_bit = (signs[j / 8] >> (j % 8)) & 1;
            let sign = if sign_bit == 1 { 1.0f32 } else { -1.0f32 };

            correction += sign * qp_dot;
        }

        correction / self.n_proj as f32
    }

    /// Reconstruct a residual-vector estimate from packed sign bits and
    /// the stored residual norm (Phase O, V-side):
    ///
    /// `r̂ = Γ(n,d) · residual_norm · (1/n) Σ_j sign_j · projection_j`
    ///
    /// The vector counterpart of `correction_dot` (which computes
    /// `q · r̂`-like scalars without materializing the vector). Used by
    /// the V-side correction: attention output adds `Σ_t w_t · r̂_t`,
    /// so the residual estimate itself is needed. See
    /// [`v_correction_gamma`] for why the analytic Γ constant replaces
    /// per-vector normalization.
    pub fn reconstruct_residual(&self, signs: &[u8], residual_norm: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for j in 0..self.n_proj {
            let row_off = j * self.dim;
            let sign_bit = (signs[j / 8] >> (j % 8)) & 1;
            let sign = if sign_bit == 1 { 1.0f32 } else { -1.0f32 };
            for (o, &p) in out.iter_mut().zip(&self.projections[row_off..row_off + self.dim]) {
                *o += sign * p;
            }
        }
        let scale = v_correction_gamma(self.n_proj, self.dim) * residual_norm / self.n_proj as f32;
        for o in &mut out {
            *o *= scale;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_deterministic() {
        let a = QjlProjection::new(64, 42);
        let b = QjlProjection::new(64, 42);
        assert_eq!(a.projections, b.projections);
    }

    #[test]
    fn sign_bytes_calculation() {
        assert_eq!(QjlProjection::new(64, 0).sign_bytes(), 4); // 32 proj / 8
        let qjl = QjlProjection::with_n_projections(64, 16, 0);
        assert_eq!(qjl.sign_bytes(), 2); // 16 / 8
    }

    #[test]
    fn encode_decode_roundtrip_sign_count() {
        let qjl = QjlProjection::new(8, 42);
        let residual = vec![0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.2, -0.3];
        let signs = qjl.encode_signs(&residual);
        assert_eq!(signs.len(), qjl.sign_bytes());
    }

    #[test]
    fn correction_is_finite() {
        let qjl = QjlProjection::new(8, 42);
        let residual = vec![0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.2, -0.3];
        let query = vec![1.0, 0.5, -0.3, 0.8, 0.0, 1.0, -0.7, -0.2];

        let signs = qjl.encode_signs(&residual);
        let correction = qjl.correction_dot(&signs, &query);
        assert!(correction.is_finite());
    }

    #[test]
    fn zero_residual_small_correction() {
        let qjl = QjlProjection::new(8, 42);
        let residual = vec![0.0f32; 8];
        let query = vec![1.0, 0.5, -0.3, 0.8, 0.0, 1.0, -0.7, -0.2];

        let signs = qjl.encode_signs(&residual);
        let correction = qjl.correction_dot(&signs, &query);

        // With zero residual, all projections dot to 0, signs are all positive
        // (>= 0.0), but the correction should still be bounded.
        assert!(correction.abs() < 1.0, "correction should be small for zero residual");
    }
}
