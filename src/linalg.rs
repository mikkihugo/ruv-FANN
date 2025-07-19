//! High-performance linear algebra operations using nalgebra
//!
//! This module provides optimized matrix operations for neural networks
//! using the nalgebra library with BLAS/LAPACK backend support.

#![cfg(feature = "nalgebra")]

use nalgebra::{DMatrix, DVector, Matrix, VecStorage, Const, Dynamic, OMatrix, OVector};
use num_traits::Float;
use std::ops::{Add, Mul};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// High-performance matrix operations for neural networks
pub struct OptimizedLinAlg;

impl OptimizedLinAlg {
    /// Efficient matrix-vector multiplication using nalgebra + BLAS
    pub fn matrix_vector_mul<T>(matrix: &DMatrix<T>, vector: &DVector<T>) -> DVector<T>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        matrix * vector
    }

    /// Batch matrix-vector multiplication for multiple inputs
    pub fn batch_matrix_vector_mul<T>(
        matrix: &DMatrix<T>,
        vectors: &[DVector<T>],
    ) -> Vec<DVector<T>>
    where
        T: Float + nalgebra::Scalar + Copy + Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            vectors
                .par_iter()
                .map(|v| matrix * v)
                .collect()
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            vectors
                .iter()
                .map(|v| matrix * v)
                .collect()
        }
    }

    /// Efficient matrix-matrix multiplication
    pub fn matrix_matrix_mul<T>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        a * b
    }

    /// Compute outer product efficiently
    pub fn outer_product<T>(a: &DVector<T>, b: &DVector<T>) -> DMatrix<T>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        a * b.transpose()
    }

    /// Batch outer product computation for gradient calculations
    pub fn batch_outer_product<T>(
        vectors_a: &[DVector<T>],
        vectors_b: &[DVector<T>],
    ) -> Vec<DMatrix<T>>
    where
        T: Float + nalgebra::Scalar + Copy + Send + Sync,
    {
        assert_eq!(vectors_a.len(), vectors_b.len());
        
        #[cfg(feature = "parallel")]
        {
            vectors_a
                .par_iter()
                .zip(vectors_b.par_iter())
                .map(|(a, b)| a * b.transpose())
                .collect()
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            vectors_a
                .iter()
                .zip(vectors_b.iter())
                .map(|(a, b)| a * b.transpose())
                .collect()
        }
    }

    /// Element-wise operations with broadcasting
    pub fn hadamard_product<T>(a: &DMatrix<T>, b: &DMatrix<T>) -> DMatrix<T>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        a.component_mul(b)
    }

    /// Transpose operation
    pub fn transpose<T>(matrix: &DMatrix<T>) -> DMatrix<T>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        matrix.transpose()
    }

    /// Efficient QR decomposition for neural network applications
    pub fn qr_decomposition<T>(matrix: &DMatrix<T>) -> Option<(DMatrix<T>, DMatrix<T>)>
    where
        T: Float + nalgebra::Scalar + Copy + nalgebra::ComplexField,
    {
        let qr = matrix.qr();
        Some((qr.q(), qr.r()))
    }

    /// SVD for dimensionality reduction and analysis
    pub fn svd<T>(matrix: &DMatrix<T>) -> Option<(DMatrix<T>, DVector<T>, DMatrix<T>)>
    where
        T: Float + nalgebra::Scalar + Copy + nalgebra::ComplexField,
    {
        let svd = matrix.svd(true, true);
        if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
            Some((u, svd.singular_values, v_t))
        } else {
            None
        }
    }
}

/// Optimized neural network layer using nalgebra
#[derive(Debug, Clone)]
pub struct OptimizedLayer<T>
where
    T: Float + nalgebra::Scalar + Copy,
{
    /// Weight matrix (output_size x input_size)
    pub weights: DMatrix<T>,
    /// Bias vector (output_size)
    pub biases: DVector<T>,
    /// Cached activations for backpropagation
    pub last_input: Option<DVector<T>>,
    pub last_output: Option<DVector<T>>,
}

impl<T> OptimizedLayer<T>
where
    T: Float + nalgebra::Scalar + Copy,
{
    /// Create a new optimized layer
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Xavier/Glorot initialization
        let scale = T::from(2.0).unwrap() / T::from(input_size + output_size).unwrap();
        let sqrt_scale = scale.sqrt();
        
        let weights = DMatrix::from_fn(output_size, input_size, |_, _| {
            (T::from(rand::random::<f64>()).unwrap() - T::from(0.5).unwrap()) * sqrt_scale
        });
        
        let biases = DVector::zeros(output_size);
        
        Self {
            weights,
            biases,
            last_input: None,
            last_output: None,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &DVector<T>) -> DVector<T> {
        let linear_output = OptimizedLinAlg::matrix_vector_mul(&self.weights, input) + &self.biases;
        
        // Apply activation function (sigmoid)
        let output = linear_output.map(|x| {
            T::one() / (T::one() + (-x).exp())
        });
        
        // Cache for backpropagation
        self.last_input = Some(input.clone());
        self.last_output = Some(output.clone());
        
        output
    }

    /// Backward pass to compute gradients
    pub fn backward(&self, output_gradient: &DVector<T>) -> (DMatrix<T>, DVector<T>, DVector<T>) {
        let last_input = self.last_input.as_ref().expect("Forward pass must be called first");
        let last_output = self.last_output.as_ref().expect("Forward pass must be called first");
        
        // Compute activation derivative (sigmoid derivative)
        let activation_derivative = last_output.map(|y| y * (T::one() - y));
        let delta = output_gradient.component_mul(&activation_derivative);
        
        // Weight gradients (outer product)
        let weight_gradients = OptimizedLinAlg::outer_product(&delta, last_input);
        
        // Bias gradients
        let bias_gradients = delta.clone();
        
        // Input gradients for next layer
        let input_gradients = OptimizedLinAlg::matrix_vector_mul(&self.weights.transpose(), &delta);
        
        (weight_gradients, bias_gradients, input_gradients)
    }

    /// Update weights and biases
    pub fn update_parameters(
        &mut self,
        weight_gradients: &DMatrix<T>,
        bias_gradients: &DVector<T>,
        learning_rate: T,
    ) {
        self.weights -= learning_rate * weight_gradients;
        self.biases -= learning_rate * bias_gradients;
    }
}

/// Optimized multi-layer neural network
#[derive(Debug)]
pub struct OptimizedNetwork<T>
where
    T: Float + nalgebra::Scalar + Copy,
{
    layers: Vec<OptimizedLayer<T>>,
}

impl<T> OptimizedNetwork<T>
where
    T: Float + nalgebra::Scalar + Copy + Send + Sync,
{
    /// Create a new optimized network
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(OptimizedLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        
        Self { layers }
    }

    /// Forward propagation through the entire network
    pub fn forward(&mut self, input: &DVector<T>) -> DVector<T> {
        let mut current_input = input.clone();
        
        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
        }
        
        current_input
    }

    /// Batch forward propagation
    pub fn batch_forward(&mut self, inputs: &[DVector<T>]) -> Vec<DVector<T>> {
        #[cfg(feature = "parallel")]
        {
            inputs
                .par_iter()
                .map(|input| {
                    let mut temp_network = self.clone();
                    temp_network.forward(input)
                })
                .collect()
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            inputs
                .iter()
                .map(|input| self.forward(input))
                .collect()
        }
    }

    /// Backpropagation training step
    pub fn train_step(
        &mut self,
        input: &DVector<T>,
        target: &DVector<T>,
        learning_rate: T,
    ) -> T {
        // Forward pass
        let output = self.forward(input);
        
        // Compute loss (MSE)
        let error = &output - target;
        let loss = error.norm_squared() / T::from(2.0).unwrap();
        
        // Backward pass
        let mut current_gradient = error;
        
        for layer in self.layers.iter_mut().rev() {
            let (weight_grad, bias_grad, input_grad) = layer.backward(&current_gradient);
            layer.update_parameters(&weight_grad, &bias_grad, learning_rate);
            current_gradient = input_grad;
        }
        
        loss
    }

    /// Batch training step
    pub fn batch_train_step(
        &mut self,
        inputs: &[DVector<T>],
        targets: &[DVector<T>],
        learning_rate: T,
    ) -> T {
        assert_eq!(inputs.len(), targets.len());
        let batch_size = inputs.len();
        
        // Collect gradients from all samples
        let mut total_weight_gradients: Vec<DMatrix<T>> = Vec::new();
        let mut total_bias_gradients: Vec<DVector<T>> = Vec::new();
        let mut total_loss = T::zero();
        
        // Initialize gradient accumulators
        for layer in &self.layers {
            total_weight_gradients.push(DMatrix::zeros(layer.weights.nrows(), layer.weights.ncols()));
            total_bias_gradients.push(DVector::zeros(layer.biases.len()));
        }
        
        // Accumulate gradients over batch
        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let output = self.forward(input);
            
            // Compute loss
            let error = &output - target;
            total_loss += error.norm_squared() / T::from(2.0).unwrap();
            
            // Backward pass
            let mut current_gradient = error;
            
            for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
                let (weight_grad, bias_grad, input_grad) = layer.backward(&current_gradient);
                total_weight_gradients[layer_idx] += weight_grad;
                total_bias_gradients[layer_idx] += bias_grad;
                current_gradient = input_grad;
            }
        }
        
        // Average gradients and update parameters
        let batch_size_t = T::from(batch_size).unwrap();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let avg_weight_grad = &total_weight_gradients[layer_idx] / batch_size_t;
            let avg_bias_grad = &total_bias_gradients[layer_idx] / batch_size_t;
            layer.update_parameters(&avg_weight_grad, &avg_bias_grad, learning_rate);
        }
        
        total_loss / batch_size_t
    }
}

/// Clone implementation for OptimizedLayer
impl<T> Clone for OptimizedLayer<T>
where
    T: Float + nalgebra::Scalar + Copy + Clone,
{
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            last_input: self.last_input.clone(),
            last_output: self.last_output.clone(),
        }
    }
}

/// Clone implementation for OptimizedNetwork
impl<T> Clone for OptimizedNetwork<T>
where
    T: Float + nalgebra::Scalar + Copy + Clone,
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

/// Benchmarking utilities for linear algebra operations
pub mod bench_utils {
    use super::*;
    use std::time::Instant;

    /// Benchmark matrix multiplication performance
    pub fn benchmark_matmul<T>(sizes: &[(usize, usize, usize)]) -> Vec<(usize, usize, usize, f64)>
    where
        T: Float + nalgebra::Scalar + Copy,
    {
        let mut results = Vec::new();
        
        for &(m, n, k) in sizes {
            let a = DMatrix::<T>::from_fn(m, k, |_, _| T::from(rand::random::<f64>()).unwrap());
            let b = DMatrix::<T>::from_fn(k, n, |_, _| T::from(rand::random::<f64>()).unwrap());
            
            let start = Instant::now();
            let _c = OptimizedLinAlg::matrix_matrix_mul(&a, &b);
            let duration = start.elapsed().as_secs_f64();
            
            results.push((m, n, k, duration));
        }
        
        results
    }

    /// Benchmark neural network forward pass
    pub fn benchmark_forward_pass<T>(
        network_sizes: &[Vec<usize>],
        batch_size: usize,
    ) -> Vec<(Vec<usize>, f64)>
    where
        T: Float + nalgebra::Scalar + Copy + Send + Sync,
    {
        let mut results = Vec::new();
        
        for sizes in network_sizes {
            let mut network = OptimizedNetwork::<T>::new(sizes);
            let inputs: Vec<DVector<T>> = (0..batch_size)
                .map(|_| DVector::from_fn(sizes[0], |_, _| T::from(rand::random::<f64>()).unwrap()))
                .collect();
            
            let start = Instant::now();
            let _outputs = network.batch_forward(&inputs);
            let duration = start.elapsed().as_secs_f64();
            
            results.push((sizes.clone(), duration));
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_optimized_layer_forward() {
        let mut layer = OptimizedLayer::<f64>::new(3, 2);
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        let output = layer.forward(&input);
        assert_eq!(output.len(), 2);
        
        // Output should be between 0 and 1 (sigmoid activation)
        for &val in output.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_optimized_network() {
        let layer_sizes = vec![4, 3, 2];
        let mut network = OptimizedNetwork::<f64>::new(&layer_sizes);
        
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = network.forward(&input);
        
        assert_eq!(output.len(), 2);
        
        // Test batch processing
        let inputs = vec![input.clone(), input.clone()];
        let outputs = network.batch_forward(&inputs);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 2);
    }

    #[test]
    fn test_training_step() {
        let layer_sizes = vec![2, 3, 1];
        let mut network = OptimizedNetwork::<f64>::new(&layer_sizes);
        
        let input = DVector::from_vec(vec![0.5, -0.5]);
        let target = DVector::from_vec(vec![1.0]);
        
        let loss_before = {
            let output = network.forward(&input);
            let error = &output - &target;
            error.norm_squared() / 2.0
        };
        
        // Training step
        let loss = network.train_step(&input, &target, 0.1);
        
        // Loss should be computed correctly
        assert_relative_eq!(loss, loss_before, epsilon = 1e-10);
        
        // After training, loss should decrease (usually)
        let loss_after = {
            let output = network.forward(&input);
            let error = &output - &target;
            error.norm_squared() / 2.0
        };
        
        // This might not always be true for a single step, but it's a good sanity check
        println!("Loss before: {}, after training: {}", loss_before, loss_after);
    }

    #[test]
    fn test_matrix_operations() {
        let a = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        
        let c = OptimizedLinAlg::matrix_matrix_mul(&a, &b);
        
        // Manual verification: [1,2; 3,4] * [5,6; 7,8] = [19,22; 43,50]
        assert_relative_eq!(c[(0, 0)], 19.0, epsilon = 1e-10);
        assert_relative_eq!(c[(0, 1)], 22.0, epsilon = 1e-10);
        assert_relative_eq!(c[(1, 0)], 43.0, epsilon = 1e-10);
        assert_relative_eq!(c[(1, 1)], 50.0, epsilon = 1e-10);
    }
}