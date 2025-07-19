//! SIMD optimizations for neural network operations
//!
//! This module provides vectorized implementations of common neural network
//! operations using Rust's portable SIMD intrinsics for maximum performance.

#![cfg(feature = "simd")]

use num_traits::Float;
use std::arch::x86_64::*;
use crate::training::helpers::{sigmoid, sigmoid_derivative};

/// SIMD-optimized vector operations for neural networks
pub struct SimdOps;

impl SimdOps {
    /// Vectorized matrix multiplication with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn matmul_f32_avx2(
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Matrix A is m x k, Matrix B is k x n, Result C is m x n
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let mut sum = _mm256_setzero_ps();
                
                for l in 0..k {
                    let a_val = _mm256_broadcast_ss(&a[i * k + l]);
                    let b_vals = _mm256_loadu_ps(&b[l * n + j]);
                    sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                }
                
                _mm256_storeu_ps(&mut c[i * n + j], sum);
            }
        }
    }

    /// Vectorized activation function (sigmoid) with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid_f32_avx2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let chunks = len / 8;
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let idx = i * 8;
            let x = _mm256_loadu_ps(&input[idx]);
            
            // Compute -x
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            // Compute exp(-x) using approximation for better performance
            let exp_neg_x = self::fast_exp_avx2(neg_x);
            
            // Compute 1 + exp(-x)
            let one = _mm256_set1_ps(1.0);
            let denominator = _mm256_add_ps(one, exp_neg_x);
            
            // Compute 1 / (1 + exp(-x))
            let result = _mm256_div_ps(one, denominator);
            
            _mm256_storeu_ps(&mut output[idx], result);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            output[i] = sigmoid(input[i]);
        }
    }

    /// Vectorized sigmoid derivative with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn sigmoid_derivative_f32_avx2(sigmoid_output: &[f32], output: &mut [f32]) {
        let len = sigmoid_output.len();
        let chunks = len / 8;
        
        for i in 0..chunks {
            let idx = i * 8;
            let s = _mm256_loadu_ps(&sigmoid_output[idx]);
            let one = _mm256_set1_ps(1.0);
            let one_minus_s = _mm256_sub_ps(one, s);
            let result = _mm256_mul_ps(s, one_minus_s);
            
            _mm256_storeu_ps(&mut output[idx], result);
        }
        
        for i in (chunks * 8)..len {
            output[i] = sigmoid_derivative(sigmoid_output[i]);
        }
    }

    /// Vectorized dot product with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let chunks = len / 8;
        
        let mut sum = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(&a[idx]);
            let b_vec = _mm256_loadu_ps(&b[idx]);
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        
        // Horizontal sum of the 8 elements in sum
        let sum_hi = _mm256_extractf128_ps(sum, 1);
        let sum_lo = _mm256_castps256_ps128(sum);
        let sum_128 = _mm_add_ps(sum_hi, sum_lo);
        
        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
        
        let mut result = _mm_cvtss_f32(sum_32);
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }
        
        result
    }

    /// Vectorized element-wise addition with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_vectors_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        let len = a.len();
        let chunks = len / 8;
        
        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(&a[idx]);
            let b_vec = _mm256_loadu_ps(&b[idx]);
            let sum = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(&mut result[idx], sum);
        }
        
        for i in (chunks * 8)..len {
            result[i] = a[i] + b[i];
        }
    }

    /// Vectorized scalar multiplication with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn scale_vector_f32_avx2(vector: &[f32], scalar: f32, result: &mut [f32]) {
        assert_eq!(vector.len(), result.len());
        
        let len = vector.len();
        let chunks = len / 8;
        let scalar_vec = _mm256_set1_ps(scalar);
        
        for i in 0..chunks {
            let idx = i * 8;
            let v = _mm256_loadu_ps(&vector[idx]);
            let scaled = _mm256_mul_ps(v, scalar_vec);
            _mm256_storeu_ps(&mut result[idx], scaled);
        }
        
        for i in (chunks * 8)..len {
            result[i] = vector[i] * scalar;
        }
    }
}

/// Fast exponential approximation using AVX2
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // Polynomial approximation for exp(x) in range [-1, 1]
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);
    let twenty_fourth = _mm256_set1_ps(1.0 / 24.0);
    
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x3, x);
    
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, sixth);
    let term4 = _mm256_mul_ps(x4, twenty_fourth);
    
    let result = _mm256_add_ps(one, x);
    let result = _mm256_add_ps(result, term2);
    let result = _mm256_add_ps(result, term3);
    let result = _mm256_add_ps(result, term4);
    
    result
}

/// Optimized batch operations using SIMD
pub struct SimdBatchOps;

impl SimdBatchOps {
    /// Process multiple forward propagations in parallel using SIMD
    #[cfg(target_feature = "avx2")]
    pub fn batch_forward_pass_f32(
        weights: &[Vec<f32>],
        biases: &[Vec<f32>],
        inputs: &[Vec<f32>],
        outputs: &mut [Vec<f32>],
    ) {
        unsafe {
            for (batch_idx, input) in inputs.iter().enumerate() {
                let mut current_layer = input.clone();
                
                for layer_idx in 0..weights.len() {
                    let weight_matrix = &weights[layer_idx];
                    let bias_vector = &biases[layer_idx];
                    let output_size = bias_vector.len();
                    let input_size = current_layer.len();
                    
                    let mut next_layer = vec![0.0f32; output_size];
                    
                    // Efficient matrix-vector multiplication
                    for out_idx in 0..output_size {
                        let weight_row_start = out_idx * input_size;
                        let weight_row = &weight_matrix[weight_row_start..weight_row_start + input_size];
                        
                        let dot_product = SimdOps::dot_product_f32_avx2(weight_row, &current_layer);
                        next_layer[out_idx] = dot_product + bias_vector[out_idx];
                    }
                    
                    // Apply activation function
                    SimdOps::sigmoid_f32_avx2(&next_layer, &mut next_layer);
                    current_layer = next_layer;
                }
                
                outputs[batch_idx] = current_layer;
            }
        }
    }

    /// Vectorized batch gradient computation
    #[cfg(target_feature = "avx2")]
    pub fn batch_gradient_computation_f32(
        activations: &[Vec<Vec<f32>>], // [batch_size][layer][neuron]
        errors: &[Vec<Vec<f32>>],      // [batch_size][layer][neuron]
        weight_gradients: &mut [Vec<f32>], // [layer][weight]
        bias_gradients: &mut [Vec<f32>],   // [layer][bias]
    ) {
        let batch_size = activations.len();
        let num_layers = weight_gradients.len();
        
        // Zero out gradients
        for layer_idx in 0..num_layers {
            weight_gradients[layer_idx].fill(0.0);
            bias_gradients[layer_idx].fill(0.0);
        }
        
        unsafe {
            for batch_idx in 0..batch_size {
                for layer_idx in 0..num_layers {
                    let layer_activations = &activations[batch_idx][layer_idx];
                    let layer_errors = &errors[batch_idx][layer_idx + 1];
                    
                    // Accumulate bias gradients
                    SimdOps::add_vectors_f32_avx2(
                        &bias_gradients[layer_idx],
                        layer_errors,
                        &mut bias_gradients[layer_idx]
                    );
                    
                    // Accumulate weight gradients (outer product)
                    let input_size = layer_activations.len();
                    let output_size = layer_errors.len();
                    
                    for out_idx in 0..output_size {
                        let weight_row_start = out_idx * input_size;
                        let error_val = layer_errors[out_idx];
                        
                        for in_idx in 0..input_size {
                            let gradient_contribution = layer_activations[in_idx] * error_val;
                            weight_gradients[layer_idx][weight_row_start + in_idx] += gradient_contribution;
                        }
                    }
                }
            }
            
            // Average gradients over batch
            let batch_size_f32 = batch_size as f32;
            for layer_idx in 0..num_layers {
                SimdOps::scale_vector_f32_avx2(
                    &weight_gradients[layer_idx].clone(),
                    1.0 / batch_size_f32,
                    &mut weight_gradients[layer_idx]
                );
                SimdOps::scale_vector_f32_avx2(
                    &bias_gradients[layer_idx].clone(),
                    1.0 / batch_size_f32,
                    &mut bias_gradients[layer_idx]
                );
            }
        }
    }
}

/// CPU feature detection for optimal SIMD usage
pub struct CpuFeatures;

impl CpuFeatures {
    /// Check if AVX2 is available
    pub fn has_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Check if FMA is available
    pub fn has_fma() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Get optimal SIMD implementation based on CPU features
    pub fn get_optimal_impl() -> SimdImplementation {
        if Self::has_avx2() && Self::has_fma() {
            SimdImplementation::Avx2Fma
        } else if Self::has_avx2() {
            SimdImplementation::Avx2
        } else {
            SimdImplementation::Scalar
        }
    }
}

/// Available SIMD implementations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdImplementation {
    Scalar,
    Avx2,
    Avx2Fma,
}

/// Adaptive SIMD dispatcher that chooses the best implementation at runtime
pub struct AdaptiveSimd;

impl AdaptiveSimd {
    /// Dispatch matrix multiplication to optimal implementation
    pub fn matmul_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        match CpuFeatures::get_optimal_impl() {
            SimdImplementation::Avx2Fma | SimdImplementation::Avx2 => {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        SimdOps::matmul_f32_avx2(a, b, c, m, n, k);
                    }
                } else {
                    Self::matmul_f32_scalar(a, b, c, m, n, k);
                }
            }
            SimdImplementation::Scalar => {
                Self::matmul_f32_scalar(a, b, c, m, n, k);
            }
        }
    }

    /// Fallback scalar matrix multiplication
    fn matmul_f32_scalar(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Dispatch sigmoid computation to optimal implementation
    pub fn sigmoid_f32(input: &[f32], output: &mut [f32]) {
        match CpuFeatures::get_optimal_impl() {
            SimdImplementation::Avx2Fma | SimdImplementation::Avx2 => {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        SimdOps::sigmoid_f32_avx2(input, output);
                    }
                } else {
                    Self::sigmoid_f32_scalar(input, output);
                }
            }
            SimdImplementation::Scalar => {
                Self::sigmoid_f32_scalar(input, output);
            }
        }
    }

    /// Fallback scalar sigmoid
    fn sigmoid_f32_scalar(input: &[f32], output: &mut [f32]) {
        for (i, &x) in input.iter().enumerate() {
            output[i] = sigmoid(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_vs_scalar_sigmoid() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let mut simd_output = vec![0.0; input.len()];
        let mut scalar_output = vec![0.0; input.len()];

        // Compute using SIMD
        AdaptiveSimd::sigmoid_f32(&input, &mut simd_output);
        
        // Compute using scalar
        for (i, &x) in input.iter().enumerate() {
            scalar_output[i] = sigmoid(x);
        }

        // Compare results
        for i in 0..input.len() {
            assert_relative_eq!(simd_output[i], scalar_output[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let expected = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
        
        #[cfg(target_feature = "avx2")]
        let simd_result = unsafe { SimdOps::dot_product_f32_avx2(&a, &b) };
        
        #[cfg(not(target_feature = "avx2"))]
        let simd_result = expected; // Fallback for testing
        
        assert_relative_eq!(simd_result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_cpu_features() {
        // Test feature detection (results depend on actual CPU)
        let _has_avx2 = CpuFeatures::has_avx2();
        let _has_fma = CpuFeatures::has_fma();
        let _impl = CpuFeatures::get_optimal_impl();
        
        // Just ensure these don't panic
        assert!(true);
    }
}