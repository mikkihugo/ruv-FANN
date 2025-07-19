//! Pure Rust implementation of the Fast Artificial Neural Network (FANN) library
//!
//! This crate provides a modern, safe, and efficient implementation of neural networks
//! inspired by the original FANN library, with support for generic floating-point types.
//! Includes full cascade correlation support for dynamic network topology optimization.
//! 
//! ## Geometric Langlands Conjecture Framework
//! 
//! This library now includes a comprehensive implementation of the Geometric Langlands 
//! Conjecture framework with zero-copy semantics, parallel computation, and GPU acceleration.

// Re-export main types
pub use activation::ActivationFunction;
pub use connection::Connection;
pub use layer::Layer;
pub use network::{Network, NetworkBuilder, NetworkError};
pub use neuron::Neuron;

// Re-export training types
pub use training::{
    ParallelTrainingOptions, TrainingAlgorithm as TrainingAlgorithmTrait, TrainingData,
    TrainingError, TrainingState,
};

/// Enumeration of available training algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingAlgorithm {
    IncrementalBackprop,
    BatchBackprop,
    Batch,           // Alias for BatchBackprop
    Backpropagation, // Alias for IncrementalBackprop
    RProp,
    QuickProp,
}

// Re-export cascade training types
pub use cascade::{CascadeConfig, CascadeError, CascadeNetwork, CascadeTrainer};

// Re-export comprehensive error handling
pub use errors::{ErrorCategory, RuvFannError, ValidationError};

// Re-export Geometric Langlands core framework
pub use core::prelude::*;

// Re-export feature extraction framework
pub use features::prelude::*;

// Re-export neural network components
pub use neural::{
    LanglandsNet, FeatureExtractor, TrainingPipeline, NeuralSymbolicBridge,
    NeuralConfig, NeuralError, NeuralResult, TrainingMetrics, CorrespondenceDataset,
};

// Re-export physics bridge components
pub use physics::prelude::*;

// Modules
pub mod activation;
pub mod cascade;
pub mod connection;
pub mod core; // Geometric Langlands core mathematical framework
pub mod errors;
pub mod features; // Feature extraction pipeline for mathematical objects
pub mod integration;
pub mod layer;
pub mod memory_manager;
pub mod network;
pub mod neural; // Neural network components for Langlands correspondences
pub mod neuron;
pub mod training;

// Optional I/O module
#[cfg(feature = "io")]
pub mod io;

// WebGPU acceleration module
pub mod webgpu;

// SIMD acceleration module (CPU optimizations)
#[cfg(feature = "simd")]
pub mod simd;

// High-performance linear algebra module
#[cfg(feature = "nalgebra")]
pub mod linalg;

// Performance profiling and monitoring
pub mod profiling;

// Test module
#[cfg(test)]
mod tests;

// Mock types for testing
pub mod mock_types;
