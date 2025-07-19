//! Geometric Langlands Conjecture: Neural-Symbolic Implementation
//!
//! This crate provides a comprehensive framework for computational exploration
//! of the geometric Langlands conjecture using neural-symbolic methods.
//!
//! # Overview
//!
//! The geometric Langlands conjecture establishes a deep correspondence between:
//! - Sheaves on the moduli stack of G-bundles on a curve
//! - D-modules on the moduli stack of local systems
//!
//! This implementation combines:
//! - **Symbolic computation** for mathematical rigor
//! - **Neural networks** for pattern recognition and learning
//! - **High-performance computing** for large-scale exploration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │           Neural Layer              │
//! │  (Pattern Learning & Prediction)    │
//! ├─────────────────────────────────────┤
//! │          Symbolic Layer             │
//! │   (Mathematical Verification)      │
//! ├─────────────────────────────────────┤
//! │          Data Layer                 │
//! │  (Feature Extraction & Caching)    │
//! └─────────────────────────────────────┘
//! ```
//!
//! # Modules
//!
//! - [`data`] - Data engineering and feature extraction
//! - [`neural`] - Neural network architectures
//! - [`symbolic`] - Symbolic computation engines
//! - [`algorithms`] - Core mathematical algorithms
//! - [`utils`] - Utility functions and helpers

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

// pub mod types;
// pub mod data;
// pub mod neural;
// pub mod symbolic;
// pub mod algorithms;
// pub mod physics;
// pub mod utils;
pub mod geometry;

use thiserror::Error;

/// Main error type for the geometric Langlands framework
#[derive(Error, Debug)]
pub enum Error {
    /// Data processing error
    #[error("Data error: {0}")]
    Data(#[from] data::DataError),
    
    /// Neural network error
    #[error("Neural error: {0}")]
    Neural(String),
    
    /// Symbolic computation error
    #[error("Symbolic error: {0}")]
    Symbolic(String),
    
    /// Algorithm error
    #[error("Algorithm error: {0}")]
    Algorithm(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for the framework
pub type Result<T> = std::result::Result<T, Error>;

/// Main configuration for the framework
#[derive(Debug, Clone)]
pub struct Config {
    /// Data processing configuration
    pub data: data::DataConfig,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Number of worker threads
    pub num_threads: usize,
    
    /// Verbosity level
    pub verbose: bool,
    
    /// Output directory for results
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: data::DataConfig::default(),
            use_gpu: false,
            num_threads: num_cpus::get(),
            verbose: false,
            output_dir: "./output".to_string(),
        }
    }
}

/// Main framework interface
pub struct GeometricLanglands {
    config: Config,
    data_manager: Option<data::DataManager>,
}

impl GeometricLanglands {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            data_manager: None,
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            data_manager: None,
        }
    }
    
    /// Initialize the framework
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize data manager
        self.data_manager = Some(
            data::DataManager::new(self.config.data.clone()).await?
        );
        
        // Create output directory
        tokio::fs::create_dir_all(&self.config.output_dir).await?;
        
        if self.config.verbose {
            println!("Geometric Langlands framework initialized");
            println!("  - Threads: {}", self.config.num_threads);
            println!("  - GPU: {}", self.config.use_gpu);
            println!("  - Output: {}", self.config.output_dir);
        }
        
        Ok(())
    }
    
    /// Get reference to data manager
    pub fn data_manager(&self) -> Option<&data::DataManager> {
        self.data_manager.as_ref()
    }
    
    /// Get mutable reference to data manager
    pub fn data_manager_mut(&mut self) -> Option<&mut data::DataManager> {
        self.data_manager.as_mut()
    }
    
    /// Get framework configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl Default for GeometricLanglands {
    fn default() -> Self {
        Self::new()
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get framework information
pub fn info() -> String {
    format!(
        "Geometric Langlands Conjecture Framework v{}\n\
         Features:\n\
         - Neural-symbolic computation\n\
         - High-performance data processing\n\
         - Mathematical verification\n\
         - WASM/CUDA support",
        VERSION
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_framework_initialization() {
        let mut framework = GeometricLanglands::new();
        framework.initialize().await.unwrap();
        
        assert!(framework.data_manager().is_some());
    }
    
    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(!config.use_gpu);
        assert!(config.num_threads > 0);
    }
    
    #[test]
    fn test_info() {
        let info = info();
        assert!(info.contains("Geometric Langlands"));
        assert!(info.contains(VERSION));
    }
}