//! Training Module for Geometric Langlands Neural Networks
//! 
//! This module provides comprehensive training utilities including
//! data generation, batch processing, validation, and performance monitoring.

pub mod data_generator;
pub mod batch_processor;
pub mod validator;
pub mod performance_monitor;
pub mod training_scheduler;

pub use data_generator::{DataGenerator, SyntheticDataConfig};
pub use batch_processor::{BatchProcessor, BatchConfig};
pub use validator::{Validator, ValidationResult};
pub use performance_monitor::{PerformanceMonitor, TrainingMetrics};
pub use training_scheduler::{TrainingScheduler, ScheduleConfig};

/// Result type for training operations
pub type TrainingResult<T> = Result<T, TrainingError>;

/// Training error types
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("Data generation error: {0}")]
    DataGeneration(String),
    
    #[error("Batch processing error: {0}")]
    BatchProcessing(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Training pipeline configuration
#[derive(Debug, Clone)]
pub struct TrainingPipelineConfig {
    /// Data generation configuration
    pub data_config: SyntheticDataConfig,
    /// Batch processing configuration
    pub batch_config: BatchConfig,
    /// Validation configuration
    pub validation_split: f32,
    /// Number of training epochs
    pub epochs: u32,
    /// Whether to use CUDA acceleration
    pub use_cuda: bool,
    /// Whether to save checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: String,
}

impl Default for TrainingPipelineConfig {
    fn default() -> Self {
        Self {
            data_config: SyntheticDataConfig::default(),
            batch_config: BatchConfig::default(),
            validation_split: 0.2,
            epochs: 1000,
            use_cuda: true,
            save_checkpoints: true,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}