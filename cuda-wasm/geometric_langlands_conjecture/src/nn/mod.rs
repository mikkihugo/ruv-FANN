//! Neural Network Module for Geometric Langlands Conjecture
//! 
//! This module implements the neural network architecture using ruv-FANN
//! for learning the correspondence between automorphic and spectral sides
//! of the geometric Langlands conjecture.

pub mod langlands_net;
pub mod feature_encoder;
pub mod loss_functions;
pub mod network_builder;

pub use langlands_net::{LanglandsNetwork, NetworkConfig};
pub use feature_encoder::{FeatureEncoder, EncodingStrategy};
pub use loss_functions::{LanglandsLoss, CorrespondenceLoss};
pub use network_builder::NetworkBuilder;

/// Result type for neural network operations
pub type NNResult<T> = Result<T, NNError>;

/// Neural network error types
#[derive(Debug, thiserror::Error)]
pub enum NNError {
    #[error("Network initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Feature encoding error: {0}")]
    EncodingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("ruv-FANN error: {0}")]
    FANNError(#[from] ruv_fann::FannError),
}