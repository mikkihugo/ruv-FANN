//! Neural network components for Geometric Langlands Conjecture
//!
//! This module provides specialized neural architectures for learning patterns
//! in mathematical objects and correspondences related to the Langlands program.

pub mod feature_extractor;
pub mod langlands_net;
pub mod training_pipeline;
pub mod neural_symbolic_bridge;

pub use feature_extractor::*;
pub use langlands_net::*;
pub use training_pipeline::*;
pub use neural_symbolic_bridge::*;

use crate::core::{MathObject, MathError};
use num_traits::Float;
use std::collections::HashMap;

/// Configuration for neural network components
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Feature vector dimension
    pub feature_dim: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![512, 256, 128],
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            use_gpu: true,
            feature_dim: 256,
        }
    }
}

/// Result type for neural operations
pub type NeuralResult<T> = Result<T, NeuralError>;

/// Errors that can occur in neural computations
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralError {
    /// Mathematical computation error
    MathError(MathError),
    /// Feature extraction failed
    FeatureExtractionFailed(String),
    /// Network architecture error
    ArchitectureError(String),
    /// Training error
    TrainingError(String),
    /// Validation error
    ValidationError(String),
    /// GPU operation failed
    GpuError(String),
}

impl From<MathError> for NeuralError {
    fn from(err: MathError) -> Self {
        NeuralError::MathError(err)
    }
}

impl std::fmt::Display for NeuralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeuralError::MathError(e) => write!(f, "Math error: {}", e),
            NeuralError::FeatureExtractionFailed(msg) => write!(f, "Feature extraction failed: {}", msg),
            NeuralError::ArchitectureError(msg) => write!(f, "Architecture error: {}", msg),
            NeuralError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            NeuralError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            NeuralError::GpuError(msg) => write!(f, "GPU error: {}", msg),
        }
    }
}

impl std::error::Error for NeuralError {}

/// Trait for objects that can be used as neural network features
pub trait NeuralFeature: Send + Sync {
    type T: Float + Send + Sync;
    
    /// Convert to feature vector
    fn to_feature_vector(&self) -> Vec<Self::T>;
    
    /// Get feature dimension
    fn feature_dim(&self) -> usize;
    
    /// Normalize features
    fn normalize(&mut self);
}

/// Training metrics for monitoring neural network performance
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f64>,
    /// Validation loss history
    pub val_loss: Vec<f64>,
    /// Accuracy history
    pub accuracy: Vec<f64>,
    /// Current epoch
    pub epoch: usize,
    /// Training time in seconds
    pub training_time: f64,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            accuracy: Vec::new(),
            epoch: 0,
            training_time: 0.0,
        }
    }
    
    pub fn add_epoch(&mut self, train_loss: f64, val_loss: f64, accuracy: f64) {
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);
        self.accuracy.push(accuracy);
        self.epoch += 1;
    }
    
    pub fn current_accuracy(&self) -> Option<f64> {
        self.accuracy.last().copied()
    }
    
    pub fn is_improving(&self) -> bool {
        if self.val_loss.len() < 2 {
            return true;
        }
        let recent = &self.val_loss[self.val_loss.len() - 5..];
        recent.is_sorted_by(|a, b| a >= b)
    }
}

/// Cache for storing computed features
pub struct FeatureCache<T: Float> {
    cache: HashMap<String, Vec<T>>,
    max_size: usize,
}

impl<T: Float> FeatureCache<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }
    
    pub fn get(&self, key: &str) -> Option<&Vec<T>> {
        self.cache.get(key)
    }
    
    pub fn insert(&mut self, key: String, features: Vec<T>) {
        if self.cache.len() >= self.max_size {
            // Simple LRU-like eviction (remove first entry)
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, features);
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
    }
    
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}