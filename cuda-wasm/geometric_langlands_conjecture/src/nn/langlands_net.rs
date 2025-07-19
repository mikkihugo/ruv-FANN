//! Langlands Correspondence Neural Network
//! 
//! This module implements the core neural network architecture using ruv-FANN
//! for learning the geometric Langlands correspondence between automorphic
//! and spectral objects.

use ruv_fann::{
    Network, NetworkBuilder, ActivationFunction, TrainingAlgorithm, 
    TrainingData, CascadeParams, CallbackResult
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::{NNResult, NNError};
use super::feature_encoder::{FeatureEncoder, EncodingStrategy};
use super::loss_functions::{LanglandsLoss, CorrespondenceLoss};

/// Configuration for the Langlands network
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Output feature dimension
    pub output_dim: usize,
    /// Hidden layer sizes (empty for cascade correlation)
    pub hidden_layers: Vec<usize>,
    /// Use cascade correlation for dynamic topology
    pub use_cascade: bool,
    /// Activation function for hidden layers
    pub hidden_activation: ActivationFunction,
    /// Activation function for output layer
    pub output_activation: ActivationFunction,
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum for training
    pub momentum: f32,
    /// Maximum epochs for training
    pub max_epochs: u32,
    /// Target error for early stopping
    pub target_error: f32,
    /// Enable CUDA acceleration if available
    pub use_cuda: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_dim: 256,
            output_dim: 256,
            hidden_layers: vec![512, 512, 256],
            use_cascade: false,
            hidden_activation: ActivationFunction::Sigmoid,
            output_activation: ActivationFunction::Linear,
            learning_rate: 0.01,
            momentum: 0.9,
            max_epochs: 10000,
            target_error: 0.001,
            use_cuda: true,
        }
    }
}

/// Main neural network for Langlands correspondence
pub struct LanglandsNetwork {
    /// The underlying FANN network
    network: Network,
    /// Network configuration
    config: NetworkConfig,
    /// Feature encoder for input
    input_encoder: FeatureEncoder,
    /// Feature encoder for output
    output_encoder: FeatureEncoder,
    /// Loss function
    loss_function: Box<dyn LanglandsLoss>,
    /// Training history
    training_history: Arc<Mutex<TrainingHistory>>,
}

/// Training history tracking
#[derive(Debug, Default)]
struct TrainingHistory {
    /// Loss values over epochs
    loss_history: Vec<f32>,
    /// Validation loss values
    validation_loss: Vec<f32>,
    /// Best validation loss
    best_validation_loss: f32,
    /// Epoch of best validation
    best_epoch: u32,
    /// Total training time
    training_time: f64,
}

impl LanglandsNetwork {
    /// Create a new Langlands network with given configuration
    pub fn new(config: NetworkConfig) -> NNResult<Self> {
        // Build the network
        let network = if config.use_cascade {
            Self::build_cascade_network(&config)?
        } else {
            Self::build_feedforward_network(&config)?
        };
        
        // Create encoders
        let input_encoder = FeatureEncoder::new(
            EncodingStrategy::Hybrid,
            config.input_dim,
        );
        
        let output_encoder = FeatureEncoder::new(
            EncodingStrategy::Hybrid,
            config.output_dim,
        );
        
        // Default loss function
        let loss_function = Box::new(CorrespondenceLoss::default());
        
        Ok(Self {
            network,
            config,
            input_encoder,
            output_encoder,
            loss_function,
            training_history: Arc::new(Mutex::new(TrainingHistory::default())),
        })
    }
    
    /// Build a feedforward network
    fn build_feedforward_network(config: &NetworkConfig) -> NNResult<Network> {
        let mut builder = NetworkBuilder::new();
        builder.input(config.input_dim);
        
        // Add hidden layers
        for &size in config.hidden_layers.iter() {
            builder.hidden_layer(size, config.hidden_activation);
        }
        
        // Add output layer
        builder.output(config.output_dim, config.output_activation);
        
        // Build the network
        builder.build()
            .map_err(|e| NNError::InitializationError(format!("Failed to build network: {}", e)))
    }
    
    /// Build a cascade correlation network
    fn build_cascade_network(config: &NetworkConfig) -> NNResult<Network> {
        // For cascade correlation, we start with a minimal network
        let network = NetworkBuilder::new()
            .input(config.input_dim)
            .output(config.output_dim, config.output_activation)
            .build()
            .map_err(|e| NNError::InitializationError(format!("Failed to build cascade network: {}", e)))?;
        
        // Cascade parameters will be set during training
        Ok(network)
    }
    
    /// Train the network on paired data
    pub fn train(
        &mut self,
        automorphic_data: &Array2<f64>,
        spectral_data: &Array2<f64>,
        validation_split: f32,
    ) -> NNResult<TrainingReport> {
        if automorphic_data.nrows() != spectral_data.nrows() {
            return Err(NNError::TrainingError(
                "Automorphic and spectral data must have same number of samples".to_string()
            ));
        }
        
        let n_samples = automorphic_data.nrows();
        let n_train = ((1.0 - validation_split) * n_samples as f32) as usize;
        
        // Split data
        let train_auto = automorphic_data.slice(s![..n_train, ..]);
        let train_spec = spectral_data.slice(s![..n_train, ..]);
        let val_auto = automorphic_data.slice(s![n_train.., ..]);
        let val_spec = spectral_data.slice(s![n_train.., ..]);
        
        // Prepare training data
        let training_data = self.prepare_training_data(&train_auto, &train_spec)?;
        
        // Setup training callback
        let history = Arc::clone(&self.training_history);
        let val_auto_clone = val_auto.to_owned();
        let val_spec_clone = val_spec.to_owned();
        let config = self.config.clone();
        
        let callback = move |epoch: u32, network: &Network| -> CallbackResult {
            // Compute validation loss
            if epoch % 100 == 0 {
                let val_loss = Self::compute_validation_loss(
                    network,
                    &val_auto_clone,
                    &val_spec_clone,
                ).unwrap_or(f32::INFINITY);
                
                let mut hist = history.lock().unwrap();
                hist.validation_loss.push(val_loss);
                
                if val_loss < hist.best_validation_loss || hist.best_validation_loss == 0.0 {
                    hist.best_validation_loss = val_loss;
                    hist.best_epoch = epoch;
                }
                
                // Early stopping
                if epoch > hist.best_epoch + 1000 {
                    return CallbackResult::Stop;
                }
            }
            
            // Check target error
            if network.get_mse() < config.target_error {
                return CallbackResult::Stop;
            }
            
            CallbackResult::Continue
        };
        
        // Train the network
        let start_time = std::time::Instant::now();
        
        if self.config.use_cascade {
            // Cascade correlation training
            let cascade_params = CascadeParams {
                max_neurons: 100,
                neurons_between_reports: 10,
                desired_error: self.config.target_error,
                ..Default::default()
            };
            
            self.network.cascade_train(
                &training_data,
                cascade_params,
                self.config.max_epochs,
                Some(Box::new(callback)),
            ).map_err(|e| NNError::TrainingError(format!("Cascade training failed: {}", e)))?;
        } else {
            // Standard training
            self.network.train_callback(
                &training_data,
                TrainingAlgorithm::Rprop,
                self.config.max_epochs,
                self.config.target_error,
                Some(Box::new(callback)),
            ).map_err(|e| NNError::TrainingError(format!("Training failed: {}", e)))?;
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        // Update history
        {
            let mut hist = self.training_history.lock().unwrap();
            hist.training_time = training_time;
        }
        
        // Generate training report
        let report = self.generate_training_report();
        
        Ok(report)
    }
    
    /// Predict spectral features from automorphic features
    pub fn predict_spectral(&self, automorphic_features: &Array1<f64>) -> NNResult<Array1<f64>> {
        // Ensure correct input dimension
        if automorphic_features.len() != self.config.input_dim {
            return Err(NNError::PredictionError(
                format!("Expected input dimension {}, got {}", 
                    self.config.input_dim, automorphic_features.len())
            ));
        }
        
        // Run through network
        let input_slice = automorphic_features.as_slice().unwrap();
        let output = self.network.run(input_slice)
            .map_err(|e| NNError::PredictionError(format!("Prediction failed: {}", e)))?;
        
        // Convert to Array1
        Ok(Array1::from_vec(output))
    }
    
    /// Predict automorphic features from spectral features (inverse direction)
    pub fn predict_automorphic(&self, spectral_features: &Array1<f64>) -> NNResult<Array1<f64>> {
        // This would require a separate inverse network or bidirectional training
        // For now, return an error
        Err(NNError::PredictionError(
            "Inverse prediction not yet implemented. Train a separate network for spectral->automorphic".to_string()
        ))
    }
    
    /// Batch prediction
    pub fn predict_batch(&self, inputs: &Array2<f64>) -> NNResult<Array2<f64>> {
        let n_samples = inputs.nrows();
        let mut outputs = Array2::zeros((n_samples, self.config.output_dim));
        
        for i in 0..n_samples {
            let input = inputs.row(i);
            let output = self.predict_spectral(&input.to_owned())?;
            outputs.row_mut(i).assign(&output);
        }
        
        Ok(outputs)
    }
    
    /// Verify a proposed correspondence
    pub fn verify_correspondence(
        &self,
        automorphic_features: &Array1<f64>,
        spectral_features: &Array1<f64>,
    ) -> NNResult<CorrespondenceVerification> {
        // Predict what the spectral features should be
        let predicted_spectral = self.predict_spectral(automorphic_features)?;
        
        // Compute various metrics
        let mse = (&predicted_spectral - spectral_features)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(f32::INFINITY as f64);
        
        let cosine_similarity = Self::cosine_similarity(&predicted_spectral, spectral_features);
        
        // Use the loss function for a more sophisticated check
        let loss = self.loss_function.compute_loss(
            &predicted_spectral,
            spectral_features,
            automorphic_features,
        );
        
        // Determine if correspondence is valid based on thresholds
        let is_valid = mse < 0.1 && cosine_similarity > 0.9 && loss < 0.1;
        
        Ok(CorrespondenceVerification {
            is_valid,
            mse,
            cosine_similarity,
            loss,
            predicted_spectral,
        })
    }
    
    /// Save the trained network
    pub fn save(&self, path: &Path) -> NNResult<()> {
        self.network.save(path)
            .map_err(|e| NNError::TrainingError(format!("Failed to save network: {}", e)))
    }
    
    /// Load a trained network
    pub fn load(path: &Path, config: NetworkConfig) -> NNResult<Self> {
        let network = Network::load(path)
            .map_err(|e| NNError::InitializationError(format!("Failed to load network: {}", e)))?;
        
        let input_encoder = FeatureEncoder::new(EncodingStrategy::Hybrid, config.input_dim);
        let output_encoder = FeatureEncoder::new(EncodingStrategy::Hybrid, config.output_dim);
        let loss_function = Box::new(CorrespondenceLoss::default());
        
        Ok(Self {
            network,
            config,
            input_encoder,
            output_encoder,
            loss_function,
            training_history: Arc::new(Mutex::new(TrainingHistory::default())),
        })
    }
    
    /// Set a custom loss function
    pub fn set_loss_function(&mut self, loss_fn: Box<dyn LanglandsLoss>) {
        self.loss_function = loss_fn;
    }
    
    /// Get network statistics
    pub fn get_statistics(&self) -> NetworkStatistics {
        NetworkStatistics {
            num_inputs: self.network.num_inputs(),
            num_outputs: self.network.num_outputs(),
            num_layers: self.network.num_layers(),
            total_neurons: self.network.total_neurons(),
            total_connections: self.network.total_connections(),
            mse: self.network.get_mse(),
        }
    }
    
    // Helper methods
    
    fn prepare_training_data(
        &self,
        automorphic: &ArrayView2<f64>,
        spectral: &ArrayView2<f64>,
    ) -> NNResult<TrainingData> {
        let n_samples = automorphic.nrows();
        
        // Flatten the data
        let inputs: Vec<f64> = automorphic.iter().cloned().collect();
        let outputs: Vec<f64> = spectral.iter().cloned().collect();
        
        TrainingData::from_arrays(
            n_samples,
            self.config.input_dim,
            &inputs,
            self.config.output_dim,
            &outputs,
        ).map_err(|e| NNError::TrainingError(format!("Failed to prepare training data: {}", e)))
    }
    
    fn compute_validation_loss(
        network: &Network,
        val_auto: &Array2<f64>,
        val_spec: &Array2<f64>,
    ) -> NNResult<f32> {
        let n_samples = val_auto.nrows();
        let mut total_loss = 0.0;
        
        for i in 0..n_samples {
            let input = val_auto.row(i);
            let target = val_spec.row(i);
            
            let output = network.run(input.as_slice().unwrap())
                .map_err(|e| NNError::PredictionError(format!("Validation prediction failed: {}", e)))?;
            
            // Compute MSE
            let mse: f32 = target.iter()
                .zip(output.iter())
                .map(|(&t, &o)| (t - o as f64).powi(2))
                .sum::<f64>() as f32 / target.len() as f32;
            
            total_loss += mse;
        }
        
        Ok(total_loss / n_samples as f32)
    }
    
    fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a * norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    fn generate_training_report(&self) -> TrainingReport {
        let hist = self.training_history.lock().unwrap();
        let stats = self.get_statistics();
        
        TrainingReport {
            final_loss: stats.mse,
            best_validation_loss: hist.best_validation_loss,
            best_epoch: hist.best_epoch,
            total_epochs: hist.loss_history.len() as u32,
            training_time: hist.training_time,
            network_stats: stats,
        }
    }
}

/// Result of correspondence verification
#[derive(Debug)]
pub struct CorrespondenceVerification {
    /// Whether the correspondence is valid
    pub is_valid: bool,
    /// Mean squared error
    pub mse: f64,
    /// Cosine similarity between predicted and actual
    pub cosine_similarity: f64,
    /// Loss function value
    pub loss: f64,
    /// Predicted spectral features
    pub predicted_spectral: Array1<f64>,
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_layers: usize,
    pub total_neurons: usize,
    pub total_connections: usize,
    pub mse: f32,
}

/// Training report
#[derive(Debug)]
pub struct TrainingReport {
    pub final_loss: f32,
    pub best_validation_loss: f32,
    pub best_epoch: u32,
    pub total_epochs: u32,
    pub training_time: f64,
    pub network_stats: NetworkStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::default();
        let network = LanglandsNetwork::new(config);
        assert!(network.is_ok());
    }
    
    #[test]
    fn test_cascade_network_creation() {
        let mut config = NetworkConfig::default();
        config.use_cascade = true;
        let network = LanglandsNetwork::new(config);
        assert!(network.is_ok());
    }
    
    // Additional tests would require mock data
}