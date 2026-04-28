//! FloatLinear — dense linear layer with f32 weights.
//!
//! Used for dequantized GGUF models (Q4_K_M, Q8_0, etc.) where weights are
//! dequantized to f32 at load time. The forward pass is a standard float
//! matrix-vector product.
//!
//! ## Memory note
//!
//! Stores weights as f32, so a 7B model would use ~28GB. Fine for small
//! models (0.5B-3B). For larger models, future work will add on-the-fly
//! dequantization that keeps weights in their quantized format.

use crate::layers::linear::LinearLayer;
use crate::tensor::FloatTensor;

/// A dense float linear layer: y = W · x (no bias).
///
/// Weights are stored as a flat f32 array in row-major order
/// `[out_features × in_features]`.
pub struct FloatLinear {
    /// Row-major weights [out_features × in_features].
    weights: Vec<f32>,
    /// Number of output features (rows).
    rows: usize,
    /// Number of input features (columns).
    cols: usize,
}

impl FloatLinear {
    /// Create from a 2D FloatTensor (as returned by GGUF dequantization).
    pub fn from_float_tensor(tensor: FloatTensor) -> Self {
        assert_eq!(tensor.shape().len(), 2, "expected 2D tensor");
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        // FloatTensor owns its data, extract it
        let weights = tensor.data().to_vec();
        Self { weights, rows, cols }
    }

    /// Create from raw data.
    pub fn new(weights: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols, "weight size mismatch");
        Self { weights, rows, cols }
    }
}

impl LinearLayer for FloatLinear {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols, "input dimension mismatch");

        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let row_start = row * self.cols;
            let row_weights = &self.weights[row_start..row_start + self.cols];
            output.push(dot_product(row_weights, input));
        }
        output
    }

    fn in_features(&self) -> usize { self.cols }
    fn out_features(&self) -> usize { self.rows }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl std::fmt::Debug for FloatLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FloatLinear({}→{})", self.cols, self.rows)
    }
}

/// SIMD-friendly dot product with 4-wide manual unrolling.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for i in 0..chunks {
        let j = i * 4;
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    let mut sum = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_forward() {
        // 2×2 identity matrix
        let layer = FloatLinear::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let out = layer.forward(&[3.0, 5.0]);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn scale_forward() {
        // [2, 0; 0, 3] — scales x by 2, y by 3
        let layer = FloatLinear::new(vec![2.0, 0.0, 0.0, 3.0], 2, 2);
        let out = layer.forward(&[1.0, 1.0]);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn non_square() {
        // 3×2 matrix: project from 2D to 3D
        let layer = FloatLinear::new(vec![
            1.0, 0.0,  // row 0: select x
            0.0, 1.0,  // row 1: select y
            1.0, 1.0,  // row 2: sum
        ], 3, 2);
        let out = layer.forward(&[2.0, 3.0]);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        assert!((out[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn from_float_tensor() {
        let tensor = FloatTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let layer = FloatLinear::from_float_tensor(tensor);
        assert_eq!(layer.in_features(), 2);
        assert_eq!(layer.out_features(), 2);
        let out = layer.forward(&[1.0, 0.0]);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn debug_format() {
        let layer = FloatLinear::new(vec![0.0; 6], 2, 3);
        let s = format!("{:?}", layer);
        assert!(s.contains("FloatLinear"));
        assert!(s.contains("3→2"));
    }
}
