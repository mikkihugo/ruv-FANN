//! Feature extraction pipeline for geometric Langlands mathematical objects
//!
//! This module provides a comprehensive feature extraction system that converts
//! abstract mathematical objects into numerical feature vectors suitable for
//! neural network processing. The pipeline includes:
//!
//! - Mathematical object → feature vector conversion
//! - Efficient serialization with serde
//! - Data validation and integrity checks
//! - Feature normalization and scaling
//! - Data pipeline optimization
//!
//! ## Architecture
//!
//! ```
//! Mathematical Object → Encoder → Feature Vector → Normalizer → Neural Network
//!                                      ↓
//!                              Serialization/Storage
//! ```

pub mod extractors;
pub mod encoders;
pub mod normalizers;
pub mod validators;
pub mod serialization;
pub mod pipeline;
pub mod storage;
pub mod optimization;

pub use extractors::*;
pub use encoders::*;
pub use normalizers::*;
pub use validators::*;
pub use serialization::*;
pub use pipeline::*;
pub use storage::*;
pub use optimization::*;

/// Re-export commonly used types
pub mod prelude {
    pub use super::extractors::{FeatureExtractor, ExtractorConfig};
    pub use super::encoders::{MathObjectEncoder, EncodingStrategy};
    pub use super::normalizers::{FeatureNormalizer, NormalizationType};
    pub use super::validators::{DataValidator, ValidationResult};
    pub use super::pipeline::{FeaturePipeline, PipelineConfig};
    pub use super::storage::{FeatureStorage, StorageBackend};
}

/// Core error types for the feature extraction system
#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    #[error("Extraction failed: {message}")]
    ExtractionFailed { message: String },
    
    #[error("Encoding failed: {message}")]
    EncodingFailed { message: String },
    
    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },
    
    #[error("Normalization failed: {message}")]
    NormalizationFailed { message: String },
    
    #[error("Storage operation failed: {message}")]
    StorageFailed { message: String },
    
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
    
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
    
    #[error("Pipeline configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

pub type FeatureResult<T> = Result<T, FeatureError>;

/// Feature vector representation with metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureVector {
    /// The actual feature values
    pub values: Vec<f64>,
    /// Dimension of the feature space
    pub dimension: usize,
    /// Feature names/labels for interpretability
    pub labels: Option<Vec<String>>,
    /// Metadata about the source mathematical object
    pub metadata: FeatureMetadata,
}

/// Metadata associated with a feature vector
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureMetadata {
    /// Type of the source mathematical object
    pub object_type: String,
    /// Encoding strategy used
    pub encoding_strategy: String,
    /// Normalization applied
    pub normalization: Option<String>,
    /// Timestamp of extraction
    pub timestamp: Option<u64>,
    /// Additional key-value metadata
    pub extra: std::collections::HashMap<String, String>,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(
        values: Vec<f64>,
        object_type: String,
        encoding_strategy: String,
    ) -> Self {
        let dimension = values.len();
        let metadata = FeatureMetadata {
            object_type,
            encoding_strategy,
            normalization: None,
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()),
            extra: std::collections::HashMap::new(),
        };
        
        Self {
            values,
            dimension,
            labels: None,
            metadata,
        }
    }
    
    /// Set feature labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.extra.insert(key, value);
        self
    }
    
    /// Get the L2 norm of the feature vector
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    
    /// Dot product with another feature vector
    pub fn dot(&self, other: &FeatureVector) -> FeatureResult<f64> {
        if self.dimension != other.dimension {
            return Err(FeatureError::ValidationFailed {
                message: format!(
                    "Dimension mismatch: {} vs {}", 
                    self.dimension, 
                    other.dimension
                ),
            });
        }
        
        Ok(self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum())
    }
    
    /// Cosine similarity with another feature vector
    pub fn cosine_similarity(&self, other: &FeatureVector) -> FeatureResult<f64> {
        let dot = self.dot(other)?;
        let norm_product = self.norm() * other.norm();
        
        if norm_product == 0.0 {
            return Err(FeatureError::ValidationFailed {
                message: "Cannot compute cosine similarity with zero vectors".to_string(),
            });
        }
        
        Ok(dot / norm_product)
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureConfig {
    /// Target dimension for feature vectors
    pub target_dimension: usize,
    /// Enable data validation
    pub validate_data: bool,
    /// Apply normalization
    pub normalize: bool,
    /// Normalization type
    pub normalization_type: NormalizationType,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable parallel processing
    pub parallel: bool,
    /// Compression level for storage (0-9)
    pub compression_level: u32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            target_dimension: 512,
            validate_data: true,
            normalize: true,
            normalization_type: NormalizationType::L2,
            batch_size: 32,
            parallel: true,
            compression_level: 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_vector_creation() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let fv = FeatureVector::new(
            values.clone(),
            "test_object".to_string(),
            "test_encoding".to_string(),
        );
        
        assert_eq!(fv.values, values);
        assert_eq!(fv.dimension, 4);
        assert_eq!(fv.metadata.object_type, "test_object");
        assert_eq!(fv.metadata.encoding_strategy, "test_encoding");
    }
    
    #[test]
    fn test_feature_vector_operations() {
        let fv1 = FeatureVector::new(
            vec![1.0, 0.0, 0.0],
            "test".to_string(),
            "test".to_string(),
        );
        let fv2 = FeatureVector::new(
            vec![0.0, 1.0, 0.0],
            "test".to_string(),
            "test".to_string(),
        );
        
        assert_eq!(fv1.norm(), 1.0);
        assert_eq!(fv1.dot(&fv2).unwrap(), 0.0);
        assert_eq!(fv1.cosine_similarity(&fv2).unwrap(), 0.0);
    }
}