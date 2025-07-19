//! Neural Network Builder for Langlands Networks
//! 
//! This module provides a builder pattern for constructing specialized
//! neural networks for the geometric Langlands correspondence.

use ruv_fann::{ActivationFunction, TrainingAlgorithm};
use std::collections::HashMap;

use super::{LanglandsNetwork, NetworkConfig, NNResult, NNError};
use super::feature_encoder::{FeatureEncoder, EncodingStrategy};
use super::loss_functions::{LanglandsLoss, CorrespondenceLoss, HeckeLoss, ModularFormLoss, CombinedLoss};

/// Network architecture presets for different mathematical contexts
#[derive(Debug, Clone, Copy)]
pub enum NetworkPreset {
    /// Basic correspondence learning
    Basic,
    /// Deep architecture for complex bundles
    DeepCorrespondence,
    /// Cascade correlation for adaptive topology
    Adaptive,
    /// Specialized for GL(2) correspondence
    GL2Specialized,
    /// For arithmetic Langlands (modular forms)
    Arithmetic,
    /// Custom architecture
    Custom,
}

/// Builder for Langlands neural networks
pub struct NetworkBuilder {
    config: NetworkConfig,
    input_encoding: Option<EncodingStrategy>,
    output_encoding: Option<EncodingStrategy>,
    loss_function: Option<Box<dyn LanglandsLoss>>,
    preprocessing: Vec<PreprocessingStep>,
    validation_config: ValidationConfig,
}

/// Preprocessing steps for input data
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// Normalize features to unit variance
    Normalize,
    /// Apply PCA dimensionality reduction
    PCA { components: usize },
    /// Remove outliers using IQR method
    RemoveOutliers { threshold: f64 },
    /// Apply polynomial features
    PolynomialFeatures { degree: usize },
    /// Custom transformation function
    Custom { name: String },
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Fraction of data to use for validation
    pub validation_split: f32,
    /// Enable cross-validation
    pub cross_validation: bool,
    /// Number of folds for cross-validation
    pub cv_folds: usize,
    /// Early stopping patience
    pub early_stopping_patience: u32,
    /// Metrics to track during validation
    pub metrics: Vec<ValidationMetric>,
}

/// Validation metrics
#[derive(Debug, Clone, Copy)]
pub enum ValidationMetric {
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// Cosine similarity
    CosineSimilarity,
    /// Correspondence validity rate
    CorrespondenceRate,
    /// Mathematical invariant preservation
    InvariantPreservation,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validation_split: 0.2,
            cross_validation: false,
            cv_folds: 5,
            early_stopping_patience: 500,
            metrics: vec![
                ValidationMetric::MSE,
                ValidationMetric::CosineSimilarity,
                ValidationMetric::CorrespondenceRate,
            ],
        }
    }
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            input_encoding: None,
            output_encoding: None,
            loss_function: None,
            preprocessing: Vec::new(),
            validation_config: ValidationConfig::default(),
        }
    }
    
    /// Start with a preset configuration
    pub fn with_preset(preset: NetworkPreset) -> Self {
        let mut builder = Self::new();
        builder.apply_preset(preset);
        builder
    }
    
    /// Set input and output dimensions
    pub fn dimensions(mut self, input_dim: usize, output_dim: usize) -> Self {
        self.config.input_dim = input_dim;
        self.config.output_dim = output_dim;
        self
    }
    
    /// Set hidden layer architecture
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.hidden_layers = layers;
        self
    }
    
    /// Enable cascade correlation
    pub fn with_cascade(mut self, enable: bool) -> Self {
        self.config.use_cascade = enable;
        self
    }
    
    /// Set activation functions
    pub fn activations(
        mut self,
        hidden: ActivationFunction,
        output: ActivationFunction,
    ) -> Self {
        self.config.hidden_activation = hidden;
        self.config.output_activation = output;
        self
    }
    
    /// Set training parameters
    pub fn training_params(
        mut self,
        learning_rate: f32,
        momentum: f32,
        max_epochs: u32,
        target_error: f32,
    ) -> Self {
        self.config.learning_rate = learning_rate;
        self.config.momentum = momentum;
        self.config.max_epochs = max_epochs;
        self.config.target_error = target_error;
        self
    }
    
    /// Set encoding strategies
    pub fn encoding(
        mut self,
        input_strategy: EncodingStrategy,
        output_strategy: EncodingStrategy,
    ) -> Self {
        self.input_encoding = Some(input_strategy);
        self.output_encoding = Some(output_strategy);
        self
    }
    
    /// Set loss function
    pub fn with_loss(mut self, loss_fn: Box<dyn LanglandsLoss>) -> Self {
        self.loss_function = Some(loss_fn);
        self
    }
    
    /// Add preprocessing step
    pub fn add_preprocessing(mut self, step: PreprocessingStep) -> Self {
        self.preprocessing.push(step);
        self
    }
    
    /// Set validation configuration
    pub fn validation(mut self, config: ValidationConfig) -> Self {
        self.validation_config = config;
        self
    }
    
    /// Enable CUDA acceleration
    pub fn with_cuda(mut self, enable: bool) -> Self {
        self.config.use_cuda = enable;
        self
    }
    
    /// Build the network
    pub fn build(self) -> NNResult<LanglandsNetwork> {
        // Validate configuration
        self.validate_config()?;
        
        // Create the network
        let mut network = LanglandsNetwork::new(self.config)?;
        
        // Set loss function if provided
        if let Some(loss_fn) = self.loss_function {
            network.set_loss_function(loss_fn);
        }
        
        Ok(network)
    }
    
    /// Apply a preset configuration
    fn apply_preset(&mut self, preset: NetworkPreset) {
        match preset {
            NetworkPreset::Basic => {
                self.config.hidden_layers = vec![256, 128];
                self.config.hidden_activation = ActivationFunction::Sigmoid;
                self.config.output_activation = ActivationFunction::Linear;
                self.config.learning_rate = 0.01;
                self.input_encoding = Some(EncodingStrategy::InvariantBased);
                self.output_encoding = Some(EncodingStrategy::InvariantBased);
                self.loss_function = Some(Box::new(CorrespondenceLoss::default()));
            },
            
            NetworkPreset::DeepCorrespondence => {
                self.config.hidden_layers = vec![512, 512, 256, 256, 128];
                self.config.hidden_activation = ActivationFunction::Sigmoid;
                self.config.output_activation = ActivationFunction::Linear;
                self.config.learning_rate = 0.005;
                self.config.momentum = 0.95;
                self.input_encoding = Some(EncodingStrategy::Hybrid);
                self.output_encoding = Some(EncodingStrategy::Hybrid);
                self.loss_function = Some(Box::new(CombinedLoss::new_default()));
                self.preprocessing.push(PreprocessingStep::Normalize);
            },
            
            NetworkPreset::Adaptive => {
                self.config.hidden_layers = vec![]; // Cascade will build dynamically
                self.config.use_cascade = true;
                self.config.hidden_activation = ActivationFunction::Sigmoid;
                self.config.output_activation = ActivationFunction::Linear;
                self.config.learning_rate = 0.01;
                self.input_encoding = Some(EncodingStrategy::MultiScale);
                self.output_encoding = Some(EncodingStrategy::MultiScale);
                self.loss_function = Some(Box::new(CorrespondenceLoss::default()));
            },
            
            NetworkPreset::GL2Specialized => {
                self.config.hidden_layers = vec![128, 64, 32];
                self.config.hidden_activation = ActivationFunction::Sigmoid;
                self.config.output_activation = ActivationFunction::Linear;
                self.config.learning_rate = 0.02;
                self.input_encoding = Some(EncodingStrategy::Spectral);
                self.output_encoding = Some(EncodingStrategy::Spectral);
                
                // Specialized loss for GL(2) case
                let mut combined = CombinedLoss::new_default();
                combined = combined.add_component(Box::new(HeckeLoss::default()), 1.5);
                self.loss_function = Some(Box::new(combined));
            },
            
            NetworkPreset::Arithmetic => {
                self.config.hidden_layers = vec![256, 128, 64];
                self.config.hidden_activation = ActivationFunction::Sigmoid;
                self.config.output_activation = ActivationFunction::Linear;
                self.config.learning_rate = 0.01;
                self.input_encoding = Some(EncodingStrategy::InvariantBased);
                self.output_encoding = Some(EncodingStrategy::InvariantBased);
                
                // Focus on modular form matching
                let mut combined = CombinedLoss::new_default();
                combined = combined.add_component(Box::new(ModularFormLoss::default()), 2.0);
                self.loss_function = Some(Box::new(combined));
                
                self.preprocessing.push(PreprocessingStep::PolynomialFeatures { degree: 2 });
            },
            
            NetworkPreset::Custom => {
                // Leave defaults for custom configuration
            },
        }
    }
    
    /// Validate the configuration
    fn validate_config(&self) -> NNResult<()> {
        if self.config.input_dim == 0 {
            return Err(NNError::InitializationError("Input dimension must be > 0".to_string()));
        }
        
        if self.config.output_dim == 0 {
            return Err(NNError::InitializationError("Output dimension must be > 0".to_string()));
        }
        
        if self.config.learning_rate <= 0.0 {
            return Err(NNError::InitializationError("Learning rate must be > 0".to_string()));
        }
        
        if self.config.max_epochs == 0 {
            return Err(NNError::InitializationError("Max epochs must be > 0".to_string()));
        }
        
        if self.config.use_cascade && !self.config.hidden_layers.is_empty() {
            return Err(NNError::InitializationError(
                "Cannot use both cascade correlation and fixed hidden layers".to_string()
            ));
        }
        
        if self.validation_config.validation_split < 0.0 || self.validation_config.validation_split >= 1.0 {
            return Err(NNError::InitializationError(
                "Validation split must be in [0, 1)".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common configurations
impl NetworkBuilder {
    /// Quick configuration for bundle-to-representation correspondence
    pub fn bundle_to_representation(input_dim: usize, output_dim: usize) -> Self {
        Self::with_preset(NetworkPreset::Basic)
            .dimensions(input_dim, output_dim)
            .encoding(EncodingStrategy::Hybrid, EncodingStrategy::Spectral)
            .with_loss(Box::new(CorrespondenceLoss {
                mse_weight: 1.0,
                invariant_weight: 3.0, // Emphasize invariant preservation
                functorial_weight: 2.0,
                spectral_weight: 2.0,
                regularization: 0.01,
            }))
    }
    
    /// Quick configuration for D-module correspondence
    pub fn d_module_correspondence(input_dim: usize, output_dim: usize) -> Self {
        Self::with_preset(NetworkPreset::DeepCorrespondence)
            .dimensions(input_dim, output_dim)
            .encoding(EncodingStrategy::Spectral, EncodingStrategy::Spectral)
            .with_loss(Box::new(CombinedLoss::new_default()
                .add_component(Box::new(HeckeLoss::default()), 2.0)))
    }
    
    /// Quick configuration for small rank cases (experimental)
    pub fn small_rank_experimental() -> Self {
        Self::with_preset(NetworkPreset::Adaptive)
            .dimensions(64, 64)
            .training_params(0.02, 0.9, 5000, 0.01)
            .add_preprocessing(PreprocessingStep::Normalize)
            .add_preprocessing(PreprocessingStep::RemoveOutliers { threshold: 2.0 })
    }
    
    /// Configuration for large-scale correspondence
    pub fn large_scale_correspondence(input_dim: usize, output_dim: usize) -> Self {
        Self::with_preset(NetworkPreset::DeepCorrespondence)
            .dimensions(input_dim, output_dim)
            .hidden_layers(vec![1024, 1024, 512, 512, 256, 256])
            .training_params(0.001, 0.99, 20000, 0.0001)
            .with_cuda(true)
            .add_preprocessing(PreprocessingStep::Normalize)
            .add_preprocessing(PreprocessingStep::PCA { components: input_dim.min(512) })
    }
}

/// Helper functions for creating specialized networks
pub fn create_gl2_network(genus: usize) -> NNResult<LanglandsNetwork> {
    let input_dim = 32 + genus * 8; // Dimension based on genus
    let output_dim = 32 + genus * 8;
    
    NetworkBuilder::with_preset(NetworkPreset::GL2Specialized)
        .dimensions(input_dim, output_dim)
        .build()
}

pub fn create_arithmetic_langlands_network(level: usize, weight: usize) -> NNResult<LanglandsNetwork> {
    let input_dim = 64 + level * 4 + weight * 2;
    let output_dim = 64 + level * 4 + weight * 2;
    
    NetworkBuilder::with_preset(NetworkPreset::Arithmetic)
        .dimensions(input_dim, output_dim)
        .build()
}

pub fn create_generic_correspondence_network(
    input_dim: usize,
    output_dim: usize,
    complexity: f64, // 0.0 = simple, 1.0 = very complex
) -> NNResult<LanglandsNetwork> {
    let preset = if complexity < 0.3 {
        NetworkPreset::Basic
    } else if complexity < 0.7 {
        NetworkPreset::DeepCorrespondence
    } else {
        NetworkPreset::Adaptive
    };
    
    NetworkBuilder::with_preset(preset)
        .dimensions(input_dim, output_dim)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_builder() {
        let builder = NetworkBuilder::new()
            .dimensions(128, 128)
            .hidden_layers(vec![64, 32]);
        
        let network = builder.build();
        assert!(network.is_ok());
    }
    
    #[test]
    fn test_preset_configurations() {
        let presets = [
            NetworkPreset::Basic,
            NetworkPreset::DeepCorrespondence,
            NetworkPreset::Adaptive,
            NetworkPreset::GL2Specialized,
            NetworkPreset::Arithmetic,
        ];
        
        for preset in presets.iter() {
            let builder = NetworkBuilder::with_preset(*preset)
                .dimensions(256, 256);
            let network = builder.build();
            assert!(network.is_ok(), "Failed to build network with preset {:?}", preset);
        }
    }
    
    #[test]
    fn test_validation_errors() {
        // Test invalid input dimension
        let result = NetworkBuilder::new()
            .dimensions(0, 128)
            .build();
        assert!(result.is_err());
        
        // Test invalid learning rate
        let result = NetworkBuilder::new()
            .dimensions(128, 128)
            .training_params(-0.1, 0.9, 1000, 0.01)
            .build();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_convenience_functions() {
        let network = create_gl2_network(2);
        assert!(network.is_ok());
        
        let network = create_arithmetic_langlands_network(11, 2);
        assert!(network.is_ok());
        
        let network = create_generic_correspondence_network(256, 256, 0.5);
        assert!(network.is_ok());
    }
}