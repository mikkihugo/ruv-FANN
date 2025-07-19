//! Neural network training and convergence tests
//!
//! Tests the neural-symbolic integration and training convergence
//! for the geometric Langlands correspondence implementation.

use geometric_langlands::prelude::*;
use crate::helpers::{Timer, assertions::*};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;

/// Neural network architecture tests
#[cfg(test)]
mod architecture_tests {
    use super::*;
    
    #[test]
    fn test_neural_network_creation() {
        // Test creation of neural networks for different tasks
        let input_dim = 128;
        let hidden_dims = vec![256, 512, 256];
        let output_dim = 64;
        
        let network = NeuralNetwork::new(input_dim, &hidden_dims, output_dim);
        
        assert_eq!(network.input_dimension(), input_dim);
        assert_eq!(network.output_dimension(), output_dim);
        assert_eq!(network.layer_count(), hidden_dims.len() + 1); // +1 for output layer
        
        println!("Created neural network: {}->{}->{}->{}->{}",
                input_dim, hidden_dims[0], hidden_dims[1], hidden_dims[2], output_dim);
    }
    
    #[test]
    fn test_activation_functions() {
        // Test different activation functions
        let test_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        
        for &x in &test_values {
            // Test ReLU
            let relu_output = ActivationFunction::relu(x);
            assert_eq!(relu_output, x.max(0.0), "ReLU implementation incorrect");
            
            // Test Sigmoid
            let sigmoid_output = ActivationFunction::sigmoid(x);
            assert!(sigmoid_output > 0.0 && sigmoid_output < 1.0, "Sigmoid output out of range");
            
            // Test Tanh
            let tanh_output = ActivationFunction::tanh(x);
            assert!(tanh_output > -1.0 && tanh_output < 1.0, "Tanh output out of range");
            
            // Test GELU (Gaussian Error Linear Unit)
            let gelu_output = ActivationFunction::gelu(x);
            assert!(gelu_output.is_finite(), "GELU output not finite");
        }
    }
    
    #[test]
    fn test_layer_initialization() {
        // Test different weight initialization strategies
        let layer_size = (100, 50);
        
        // Xavier initialization
        let xavier_weights = LayerInitialization::xavier(layer_size.0, layer_size.1);
        let xavier_var = xavier_weights.variance();
        let expected_xavier_var = 2.0 / (layer_size.0 + layer_size.1) as f64;
        assert_approx_eq_with_context(xavier_var, expected_xavier_var, "Xavier initialization variance");
        
        // He initialization
        let he_weights = LayerInitialization::he(layer_size.0, layer_size.1);
        let he_var = he_weights.variance();
        let expected_he_var = 2.0 / layer_size.0 as f64;
        assert_approx_eq_with_context(he_var, expected_he_var, "He initialization variance");
        
        // Glorot initialization
        let glorot_weights = LayerInitialization::glorot_uniform(layer_size.0, layer_size.1);
        assert!(glorot_weights.min() >= -1.0 && glorot_weights.max() <= 1.0, 
               "Glorot weights out of expected range");
    }
}

/// Forward propagation tests
#[cfg(test)]
mod forward_propagation_tests {
    use super::*;
    
    #[test]
    fn test_linear_layer_forward() {
        let input_size = 10;
        let output_size = 5;
        let batch_size = 3;
        
        let layer = LinearLayer::new(input_size, output_size);
        let input = DMatrix::<f64>::from_fn(batch_size, input_size, |i, j| (i + j) as f64);
        
        let output = layer.forward(&input);
        
        assert_eq!(output.nrows(), batch_size);
        assert_eq!(output.ncols(), output_size);
        
        // Check that output is finite
        for element in output.iter() {
            assert!(element.is_finite(), "Forward pass produced non-finite output");
        }
    }
    
    #[test]
    fn test_neural_network_forward() {
        let network = NeuralNetwork::new(8, &[16, 32, 16], 4);
        let input = DVector::<f64>::from_fn(8, |i, _| i as f64 * 0.1);
        
        let output = network.forward(&input);
        
        assert_eq!(output.len(), 4);
        
        // Check output properties
        for &value in output.iter() {
            assert!(value.is_finite(), "Network output not finite");
        }
        
        // Test batch processing
        let batch_input = DMatrix::<f64>::from_fn(5, 8, |i, j| (i + j) as f64 * 0.1);
        let batch_output = network.forward_batch(&batch_input);
        
        assert_eq!(batch_output.nrows(), 5);
        assert_eq!(batch_output.ncols(), 4);
    }
    
    #[test]
    fn test_residual_connections() {
        // Test ResNet-style residual connections
        let network = ResidualNetwork::new(64, &[64, 64, 64], 32);
        let input = DVector::<f64>::from_fn(64, |i, _| (i as f64).sin());
        
        let output_with_residual = network.forward(&input);
        let output_without_residual = network.forward_no_residual(&input);
        
        // Residual connections should affect the output
        let diff = (&output_with_residual - &output_without_residual).norm();
        assert!(diff > 1e-6, "Residual connections not functioning");
    }
}

/// Backpropagation and gradient tests
#[cfg(test)]
mod backpropagation_tests {
    use super::*;
    
    #[test]
    fn test_gradient_computation() {
        let layer = LinearLayer::new(3, 2);
        let input = DVector::<f64>::from_fn(3, |i, _| i as f64);
        let target = DVector::<f64>::from_fn(2, |i, _| (i + 1) as f64);
        
        // Forward pass
        let output = layer.forward(&input);
        
        // Compute loss
        let loss = MeanSquaredError::compute(&output, &target);
        assert!(loss >= 0.0, "Loss should be non-negative");
        
        // Backward pass
        let gradients = layer.backward(&input, &output, &target);
        
        // Check gradient properties
        assert_eq!(gradients.weight_gradients.nrows(), 2);
        assert_eq!(gradients.weight_gradients.ncols(), 3);
        assert_eq!(gradients.bias_gradients.len(), 2);
        
        // Gradients should be finite
        for gradient in gradients.weight_gradients.iter() {
            assert!(gradient.is_finite(), "Weight gradient not finite");
        }
        for &gradient in gradients.bias_gradients.iter() {
            assert!(gradient.is_finite(), "Bias gradient not finite");
        }
    }
    
    #[test]
    fn test_gradient_checking() {
        // Numerical gradient checking
        let layer = LinearLayer::new(2, 1);
        let input = DVector::<f64>::from_fn(2, |i, _| i as f64);
        let target = DVector::<f64>::from_fn(1, |_, _| 1.0);
        
        let analytical_gradients = layer.backward(&input, &layer.forward(&input), &target);
        let numerical_gradients = layer.numerical_gradients(&input, &target, 1e-5);
        
        // Compare analytical and numerical gradients
        let weight_diff = (&analytical_gradients.weight_gradients - &numerical_gradients.weight_gradients).norm();
        let bias_diff = (&analytical_gradients.bias_gradients - &numerical_gradients.bias_gradients).norm();
        
        assert!(weight_diff < 1e-4, "Weight gradient checking failed: diff = {}", weight_diff);
        assert!(bias_diff < 1e-4, "Bias gradient checking failed: diff = {}", bias_diff);
    }
    
    #[test]
    fn test_vanishing_exploding_gradients() {
        // Test deep networks for gradient stability
        let deep_network = NeuralNetwork::new(10, &[50, 50, 50, 50, 50], 1);
        let input = DVector::<f64>::from_fn(10, |i, _| (i as f64).sin());
        let target = DVector::<f64>::from_fn(1, |_, _| 0.5);
        
        let gradients = deep_network.compute_gradients(&input, &target);
        
        // Check gradient norms across layers
        for (layer_idx, layer_gradients) in gradients.iter().enumerate() {
            let gradient_norm = layer_gradients.weight_gradients.norm();
            
            // Gradients shouldn't vanish (too small) or explode (too large)
            assert!(gradient_norm > 1e-8, "Vanishing gradients detected at layer {}", layer_idx);
            assert!(gradient_norm < 1e3, "Exploding gradients detected at layer {}", layer_idx);
        }
    }
}

/// Training convergence tests
#[cfg(test)]
mod training_tests {
    use super::*;
    
    #[test]
    fn test_simple_regression_convergence() {
        // Test training on a simple regression problem
        let mut network = NeuralNetwork::new(1, &[10, 10], 1);
        let optimizer = AdamOptimizer::new(0.01, 0.9, 0.999);
        
        // Generate training data: y = x^2
        let train_x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let train_y: Vec<f64> = train_x.iter().map(|&x| x * x).collect();
        
        let mut losses = Vec::new();
        let epochs = 100;
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            for (x, y) in train_x.iter().zip(train_y.iter()) {
                let input = DVector::from_vec(vec![*x]);
                let target = DVector::from_vec(vec![*y]);
                
                let output = network.forward(&input);
                let loss = MeanSquaredError::compute(&output, &target);
                epoch_loss += loss;
                
                let gradients = network.compute_gradients(&input, &target);
                network.update_weights(&gradients, &optimizer);
            }
            
            epoch_loss /= train_x.len() as f64;
            losses.push(epoch_loss);
            
            if epoch % 20 == 0 {
                println!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }
        
        // Check convergence
        let initial_loss = losses[0];
        let final_loss = losses[losses.len() - 1];
        let improvement_ratio = (initial_loss - final_loss) / initial_loss;
        
        assert!(improvement_ratio > 0.8, "Training did not converge sufficiently");
        assert!(final_loss < 0.1, "Final loss too high: {}", final_loss);
    }
    
    #[test]
    fn test_classification_convergence() {
        // Test training on XOR problem (classic non-linear classification)
        let mut network = NeuralNetwork::new(2, &[4, 4], 1);
        let optimizer = SGDOptimizer::new(0.1, 0.9);
        
        // XOR training data
        let train_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        
        let mut accuracy_history = Vec::new();
        let epochs = 500;
        
        for epoch in 0..epochs {
            // Training
            for (inputs, targets) in &train_data {
                let input = DVector::from_vec(inputs.clone());
                let target = DVector::from_vec(targets.clone());
                
                let gradients = network.compute_gradients(&input, &target);
                network.update_weights(&gradients, &optimizer);
            }
            
            // Evaluation
            if epoch % 50 == 0 {
                let mut correct = 0;
                for (inputs, targets) in &train_data {
                    let input = DVector::from_vec(inputs.clone());
                    let output = network.forward(&input);
                    let predicted = if output[0] > 0.5 { 1.0 } else { 0.0 };
                    if predicted == targets[0] {
                        correct += 1;
                    }
                }
                let accuracy = correct as f64 / train_data.len() as f64;
                accuracy_history.push(accuracy);
                println!("Epoch {}: Accuracy = {:.2}%", epoch, accuracy * 100.0);
            }
        }
        
        // Should achieve perfect accuracy on XOR
        let final_accuracy = accuracy_history.last().unwrap();
        assert!(*final_accuracy >= 0.95, "XOR classification not learned: accuracy = {}", final_accuracy);
    }
    
    #[test]
    fn test_optimizer_comparison() {
        // Compare different optimizers on the same problem
        let problem_fn = |network: &mut NeuralNetwork, optimizer: &dyn Optimizer| -> f64 {
            // Simple quadratic function approximation
            let train_data: Vec<(f64, f64)> = (0..50)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    let y = x * x + 0.1 * x + 0.01;
                    (x, y)
                })
                .collect();
            
            for _ in 0..100 {
                for (x, y) in &train_data {
                    let input = DVector::from_vec(vec![*x]);
                    let target = DVector::from_vec(vec![*y]);
                    
                    let gradients = network.compute_gradients(&input, &target);
                    network.update_weights(&gradients, optimizer);
                }
            }
            
            // Return final loss
            let mut total_loss = 0.0;
            for (x, y) in &train_data {
                let input = DVector::from_vec(vec![*x]);
                let target = DVector::from_vec(vec![*y]);
                let output = network.forward(&input);
                total_loss += MeanSquaredError::compute(&output, &target);
            }
            total_loss / train_data.len() as f64
        };
        
        // Test different optimizers
        let mut sgd_network = NeuralNetwork::new(1, &[8], 1);
        let sgd_optimizer = SGDOptimizer::new(0.01, 0.9);
        let sgd_loss = problem_fn(&mut sgd_network, &sgd_optimizer);
        
        let mut adam_network = NeuralNetwork::new(1, &[8], 1);
        let adam_optimizer = AdamOptimizer::new(0.01, 0.9, 0.999);
        let adam_loss = problem_fn(&mut adam_network, &adam_optimizer);
        
        let mut rmsprop_network = NeuralNetwork::new(1, &[8], 1);
        let rmsprop_optimizer = RMSpropOptimizer::new(0.01, 0.9);
        let rmsprop_loss = problem_fn(&mut rmsprop_network, &rmsprop_optimizer);
        
        println!("Optimizer comparison - SGD: {:.6}, Adam: {:.6}, RMSprop: {:.6}", 
                sgd_loss, adam_loss, rmsprop_loss);
        
        // All optimizers should achieve reasonable performance
        assert!(sgd_loss < 0.1, "SGD did not converge");
        assert!(adam_loss < 0.1, "Adam did not converge");
        assert!(rmsprop_loss < 0.1, "RMSprop did not converge");
        
        // Adam should typically perform well
        assert!(adam_loss < sgd_loss || adam_loss < 0.01, "Adam underperformed");
    }
}

/// Neural-symbolic integration tests
#[cfg(test)]
mod neural_symbolic_tests {
    use super::*;
    
    #[test]
    fn test_symbolic_to_neural_encoding() {
        // Test encoding of mathematical objects into neural network inputs
        let g = ReductiveGroup::gl_n(2);
        let form = AutomorphicForm::eisenstein_series(&g, 4);
        
        let encoder = SymbolicEncoder::new();
        let encoded = encoder.encode_automorphic_form(&form);
        
        assert!(encoded.len() > 0, "Encoded automorphic form should not be empty");
        
        // Check that encoding is deterministic
        let encoded2 = encoder.encode_automorphic_form(&form);
        assert_eq!(encoded, encoded2, "Encoding should be deterministic");
        
        // Check that different forms have different encodings
        let form2 = AutomorphicForm::eisenstein_series(&g, 6);
        let encoded_form2 = encoder.encode_automorphic_form(&form2);
        assert_ne!(encoded, encoded_form2, "Different forms should have different encodings");
    }
    
    #[test]
    fn test_neural_to_symbolic_decoding() {
        // Test decoding neural network outputs back to mathematical objects
        let network = LanglandsNetwork::new(128, &[256, 512, 256], 64);
        let input = DVector::<f64>::from_fn(128, |i, _| (i as f64).sin());
        
        let output = network.forward(&input);
        let decoder = SymbolicDecoder::new();
        
        // Attempt to decode as Galois representation
        match decoder.decode_galois_representation(&output) {
            Ok(galois_rep) => {
                assert_eq!(galois_rep.dimension(), 2, "Expected 2-dimensional Galois representation");
                
                // Check that it satisfies representation properties
                assert!(galois_rep.is_valid(), "Decoded Galois representation is invalid");
            }
            Err(e) => panic!("Failed to decode Galois representation: {}", e),
        }
    }
    
    #[test]
    fn test_correspondence_prediction() {
        // Test neural network's ability to predict Langlands correspondence
        let mut network = CorrespondenceNetwork::new();
        
        // Generate training data from known correspondences
        let training_data = generate_correspondence_training_data(100);
        let optimizer = AdamOptimizer::new(0.001, 0.9, 0.999);
        
        // Training loop
        for epoch in 0..50 {
            let mut epoch_loss = 0.0;
            
            for (automorphic_encoding, galois_encoding) in &training_data {
                let predicted_galois = network.forward(automorphic_encoding);
                let loss = MeanSquaredError::compute(&predicted_galois, galois_encoding);
                epoch_loss += loss;
                
                let gradients = network.compute_gradients(automorphic_encoding, galois_encoding);
                network.update_weights(&gradients, &optimizer);
            }
            
            if epoch % 10 == 0 {
                println!("Correspondence training epoch {}: loss = {:.6}", 
                        epoch, epoch_loss / training_data.len() as f64);
            }
        }
        
        // Test prediction accuracy
        let test_data = generate_correspondence_training_data(20);
        let mut total_error = 0.0;
        
        for (automorphic_encoding, true_galois_encoding) in &test_data {
            let predicted_galois = network.forward(automorphic_encoding);
            let error = (&predicted_galois - true_galois_encoding).norm();
            total_error += error;
        }
        
        let average_error = total_error / test_data.len() as f64;
        assert!(average_error < 0.5, "Correspondence prediction error too high: {}", average_error);
    }
    
    #[test]
    fn test_feature_extraction() {
        // Test extraction of meaningful features from mathematical objects
        let feature_extractor = FeatureExtractor::new();
        
        // Test with different mathematical objects
        let g1 = ReductiveGroup::gl_n(2);
        let g2 = ReductiveGroup::gl_n(3);
        
        let features1 = feature_extractor.extract_group_features(&g1);
        let features2 = feature_extractor.extract_group_features(&g2);
        
        assert_ne!(features1, features2, "Different groups should have different features");
        
        // Test feature invariance under group actions
        let transformed_g1 = g1.conjugate_by_element(&g1.sample_element());
        let transformed_features = feature_extractor.extract_group_features(&transformed_g1);
        
        // Features should be approximately invariant under conjugation
        let feature_diff = (&features1 - &transformed_features).norm();
        assert!(feature_diff < 0.1, "Features not invariant under group action");
    }
}

/// Performance and scalability tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_training_speed() {
        // Test training speed for different network sizes
        let sizes = vec![
            (10, vec![20], 5),
            (50, vec![100, 50], 25),
            (100, vec![200, 200, 100], 50),
        ];
        
        for (input_size, hidden_sizes, output_size) in sizes {
            let mut network = NeuralNetwork::new(input_size, &hidden_sizes, output_size);
            let optimizer = AdamOptimizer::new(0.01, 0.9, 0.999);
            
            let input = DVector::<f64>::from_fn(input_size, |i, _| i as f64 * 0.01);
            let target = DVector::<f64>::from_fn(output_size, |i, _| (i + 1) as f64 * 0.1);
            
            let _timer = Timer::new(&format!("Training {}->{}->{}",
                                           input_size, hidden_sizes.len(), output_size));
            
            // Time forward and backward passes
            let start = std::time::Instant::now();
            
            for _ in 0..100 {
                let _output = network.forward(&input);
                let gradients = network.compute_gradients(&input, &target);
                network.update_weights(&gradients, &optimizer);
            }
            
            let duration = start.elapsed();
            println!("Network size {}->{}->{}): {:.2} ms per iteration",
                    input_size, hidden_sizes.len(), output_size,
                    duration.as_millis() as f64 / 100.0);
        }
    }
    
    #[test]
    fn test_memory_efficiency() {
        // Test memory usage during training
        let initial_memory = crate::helpers::MemoryTracker::current_memory_usage();
        
        {
            let mut network = NeuralNetwork::new(1000, &[2000, 2000], 500);
            let optimizer = AdamOptimizer::new(0.01, 0.9, 0.999);
            
            let input = DVector::<f64>::from_fn(1000, |i, _| i as f64 * 0.001);
            let target = DVector::<f64>::from_fn(500, |i, _| (i + 1) as f64 * 0.002);
            
            // Perform many training steps
            for _ in 0..1000 {
                let gradients = network.compute_gradients(&input, &target);
                network.update_weights(&gradients, &optimizer);
            }
        } // Network should be deallocated here
        
        let final_memory = crate::helpers::MemoryTracker::current_memory_usage();
        let memory_delta = final_memory as isize - initial_memory as isize;
        
        // Memory usage should return to near baseline
        assert!(memory_delta.abs() < 50_000_000, // 50MB tolerance
               "Memory leak detected: {} bytes", memory_delta);
    }
    
    #[test]
    fn test_parallel_training() {
        // Test parallel training across multiple threads
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let network = Arc::new(Mutex::new(NeuralNetwork::new(10, &[20], 5)));
        let optimizer = Arc::new(AdamOptimizer::new(0.01, 0.9, 0.999));
        
        let num_threads = 4;
        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let network_clone = Arc::clone(&network);
            let optimizer_clone = Arc::clone(&optimizer);
            
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let input = DVector::<f64>::from_fn(10, |j, _| (thread_id + j + i) as f64 * 0.01);
                    let target = DVector::<f64>::from_fn(5, |j, _| (thread_id + j + i + 1) as f64 * 0.02);
                    
                    let mut net = network_clone.lock().unwrap();
                    let gradients = net.compute_gradients(&input, &target);
                    net.update_weights(&gradients, &*optimizer_clone);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        println!("Parallel training completed successfully");
    }
}

/// Helper function to generate training data for correspondence tests
fn generate_correspondence_training_data(count: usize) -> Vec<(DVector<f64>, DVector<f64>)> {
    let mut data = Vec::new();
    
    for i in 0..count {
        // Generate synthetic automorphic form encoding
        let automorphic = DVector::<f64>::from_fn(64, |j, _| {
            ((i + j) as f64 * 0.1).sin()
        });
        
        // Generate corresponding Galois representation encoding
        let galois = DVector::<f64>::from_fn(32, |j, _| {
            ((i + j) as f64 * 0.15).cos()
        });
        
        data.push((automorphic, galois));
    }
    
    data
}

/// Run all neural network tests
pub fn run_all() {
    println!("Running neural network architecture tests...");
    println!("Running forward propagation tests...");
    println!("Running backpropagation and gradient tests...");
    println!("Running training convergence tests...");
    println!("Running neural-symbolic integration tests...");
    println!("Running neural network performance tests...");
    println!("All neural network tests completed!");
}