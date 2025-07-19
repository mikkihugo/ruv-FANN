//! # Geometric Langlands Conjecture - Neural Network Implementation
//! 
//! This crate implements a neural network approach to learning the geometric
//! Langlands correspondence using the ruv-FANN library. The geometric Langlands
//! conjecture establishes a profound duality between:
//! 
//! - **Automorphic side**: D-modules on moduli stacks of G-bundles
//! - **Spectral side**: Local systems (representations of fundamental groups)
//! 
//! ## Architecture
//! 
//! The implementation follows a hybrid symbolic-neural approach:
//! 
//! 1. **Feature Extraction**: Convert mathematical objects to numerical features
//! 2. **Neural Network**: Learn the correspondence using ruv-FANN
//! 3. **Verification**: Validate predictions using mathematical constraints
//! 
//! ## Modules
//! 
//! - [`geometry`]: Geometric objects (bundles, curves, moduli spaces)
//! - [`topology`]: Topological structures (local systems, fundamental groups)
//! - [`algebra`]: Algebraic structures (D-modules, representations)
//! - [`nn`]: Neural network architecture and training
//! - [`training`]: Training utilities and data generation
//! - [`validation`]: Mathematical verification and testing
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use geometric_langlands_conjecture::prelude::*;
//! 
//! // Create a neural network for GL(2) correspondence
//! let network = NetworkBuilder::with_preset(NetworkPreset::GL2Specialized)
//!     .dimensions(256, 256)
//!     .build()?;
//! 
//! // Generate training data
//! let mut generator = DataGenerator::new(SyntheticDataConfig {
//!     rank: 2,
//!     genus: 1,
//!     num_samples: 1000,
//!     ..Default::default()
//! });
//! 
//! let dataset = generator.generate_training_data()?;
//! 
//! // Train the network
//! // ... (training code)
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod geometry;
pub mod topology;
pub mod algebra;
pub mod nn;
pub mod training;
pub mod validation;

/// Common imports for ease of use
pub mod prelude {
    pub use crate::geometry::{Bundle, Curve, ModuliSpace};
    pub use crate::topology::{LocalSystem, FundamentalGroup};
    pub use crate::algebra::{DModule, Representation};
    pub use crate::nn::{
        LanglandsNetwork, NetworkBuilder, NetworkConfig, NetworkPreset,
        FeatureEncoder, EncodingStrategy,
    };
    pub use crate::training::{
        DataGenerator, SyntheticDataConfig, TrainingDataset,
        BatchProcessor, Validator,
    };
    pub use crate::validation::{
        CorrespondenceValidator, MathematicalConstraint,
    };
}

/// Result type for the entire crate
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the crate
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Neural network related errors
    #[error("Neural network error: {0}")]
    NeuralNetwork(#[from] nn::NNError),
    
    /// Training related errors
    #[error("Training error: {0}")]
    Training(#[from] training::TrainingError),
    
    /// Validation related errors
    #[error("Validation error: {0}")]
    Validation(#[from] validation::ValidationError),
    
    /// Geometry related errors
    #[error("Geometry error: {0}")]
    Geometry(String),
    
    /// Topology related errors
    #[error("Topology error: {0}")]
    Topology(String),
    
    /// Algebra related errors
    #[error("Algebra error: {0}")]
    Algebra(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Mathematical computation errors
    #[error("Mathematical error: {0}")]
    Mathematical(String),
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Initialize logging for the library
pub fn init_logging() {
    env_logger::init();
}

/// Configuration for the entire library
#[derive(Debug, Clone)]
pub struct Config {
    /// Enable CUDA acceleration if available
    pub use_cuda: bool,
    /// Number of threads for parallel processing
    pub num_threads: usize,
    /// Logging level
    pub log_level: log::LevelFilter,
    /// Working directory for temporary files
    pub work_dir: std::path::PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            use_cuda: true,
            num_threads: rayon::current_num_threads(),
            log_level: log::LevelFilter::Info,
            work_dir: std::env::temp_dir().join("geometric_langlands"),
        }
    }
}

/// Global configuration
static mut GLOBAL_CONFIG: Option<Config> = None;

/// Initialize the library with configuration
pub fn init(config: Config) -> Result<()> {
    // Set up global configuration
    unsafe {
        GLOBAL_CONFIG = Some(config.clone());
    }
    
    // Initialize Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build_global()
        .map_err(|e| Error::Configuration(format!("Failed to initialize thread pool: {}", e)))?;
    
    // Create work directory
    std::fs::create_dir_all(&config.work_dir)
        .map_err(|e| Error::Configuration(format!("Failed to create work directory: {}", e)))?;
    
    log::info!("Geometric Langlands library initialized with {} threads", config.num_threads);
    
    Ok(())
}

/// Get the global configuration
pub fn get_config() -> Config {
    unsafe {
        GLOBAL_CONFIG.clone().unwrap_or_default()
    }
}