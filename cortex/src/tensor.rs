//! Tensor types for cortex inference. Float-only after the
//! 2026-05-29 BitNet un-merge (ternary tensors moved to ternary-rs).

use std::fmt;

/// A simple f32 tensor for non-quantized operations (embeddings, RMSNorm, logits)
/// and dequantized weight matrices (Q4_K → f32, F16 → f32, etc.).
#[derive(Clone)]
pub struct FloatTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl FloatTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "data length must match shape");
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    #[inline]
    pub fn data(&self) -> &[f32] { &self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] { &mut self.data }

    #[inline]
    pub fn shape(&self) -> &[usize] { &self.shape }

    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

impl fmt::Debug for FloatTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FloatTensor({:?})", self.shape)
    }
}
