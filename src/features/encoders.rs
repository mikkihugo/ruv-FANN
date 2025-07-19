//! Mathematical object encoders for the geometric Langlands framework
//!
//! This module provides specialized encoders that convert mathematical objects
//! into efficient numerical representations suitable for neural network processing.

use crate::core::prelude::*;
use crate::features::{FeatureVector, FeatureResult, FeatureError};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Encoding strategies for mathematical objects
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EncodingStrategy {
    /// Dense numerical encoding
    Dense,
    /// Sparse encoding for high-dimensional objects
    Sparse,
    /// Hash-based encoding for discrete structures
    Hash,
    /// Spectral encoding using eigenvalues/eigenvectors
    Spectral,
    /// Topological encoding focusing on topological invariants
    Topological,
    /// Algebraic encoding for algebraic structures
    Algebraic,
    /// Geometric encoding for geometric properties
    Geometric,
    /// Hybrid encoding combining multiple strategies
    Hybrid,
}

impl Default for EncodingStrategy {
    fn default() -> Self {
        EncodingStrategy::Dense
    }
}

/// Configuration for mathematical object encoding
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodingConfig {
    /// Encoding strategy to use
    pub strategy: EncodingStrategy,
    /// Target dimension for encoded features
    pub target_dimension: usize,
    /// Precision for numerical computations
    pub precision: f64,
    /// Enable invariant preservation
    pub preserve_invariants: bool,
    /// Include metadata in encoding
    pub include_metadata: bool,
    /// Hash seed for reproducible hash encodings
    pub hash_seed: u64,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            strategy: EncodingStrategy::Dense,
            target_dimension: 512,
            precision: 1e-12,
            preserve_invariants: true,
            include_metadata: true,
            hash_seed: 42,
        }
    }
}

/// Main trait for mathematical object encoders
pub trait MathObjectEncoder<T> {
    /// Encode a mathematical object into a feature vector
    fn encode(&self, object: &T, config: &EncodingConfig) -> FeatureResult<FeatureVector>;
    
    /// Decode a feature vector back to a mathematical object (if possible)
    fn decode(&self, features: &FeatureVector, config: &EncodingConfig) -> FeatureResult<T>;
    
    /// Check if encoding is reversible
    fn is_reversible(&self) -> bool {
        false
    }
    
    /// Get encoding quality metrics
    fn encoding_quality(&self, original: &T, reconstructed: &T) -> FeatureResult<f64>;
}

/// Encoder for sheaf objects
#[derive(Debug, Clone)]
pub struct SheafEncoder {
    /// Encoding strategy
    pub strategy: EncodingStrategy,
    /// Cache for computed encodings
    pub cache: HashMap<u64, FeatureVector>,
}

impl Default for SheafEncoder {
    fn default() -> Self {
        Self {
            strategy: EncodingStrategy::Topological,
            cache: HashMap::new(),
        }
    }
}

impl MathObjectEncoder<Sheaf> for SheafEncoder {
    fn encode(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        match config.strategy {
            EncodingStrategy::Dense => self.encode_dense(sheaf, config),
            EncodingStrategy::Sparse => self.encode_sparse(sheaf, config),
            EncodingStrategy::Hash => self.encode_hash(sheaf, config),
            EncodingStrategy::Spectral => self.encode_spectral(sheaf, config),
            EncodingStrategy::Topological => self.encode_topological(sheaf, config),
            EncodingStrategy::Hybrid => self.encode_hybrid(sheaf, config),
            _ => Err(FeatureError::EncodingFailed {
                message: format!("Unsupported encoding strategy: {:?}", config.strategy),
            }),
        }
    }
    
    fn decode(&self, features: &FeatureVector, config: &EncodingConfig) -> FeatureResult<Sheaf> {
        match config.strategy {
            EncodingStrategy::Hash => self.decode_hash(features, config),
            _ => Err(FeatureError::EncodingFailed {
                message: "Decoding not supported for this strategy".to_string(),
            }),
        }
    }
    
    fn is_reversible(&self) -> bool {
        matches!(self.strategy, EncodingStrategy::Hash)
    }
    
    fn encoding_quality(&self, original: &Sheaf, reconstructed: &Sheaf) -> FeatureResult<f64> {
        // Compute similarity between original and reconstructed sheaf
        let rank_diff = (original.rank() as f64 - reconstructed.rank() as f64).abs();
        let degree_diff = original.degree().unwrap_or(0.0) - reconstructed.degree().unwrap_or(0.0);
        
        // Simple quality metric (higher is better)
        let quality = 1.0 / (1.0 + rank_diff + degree_diff.abs());
        Ok(quality)
    }
}

impl SheafEncoder {
    /// Dense encoding: extract all available numerical features
    fn encode_dense(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        
        // Basic properties
        features.push(sheaf.rank() as f64);
        features.push(sheaf.degree().unwrap_or(0.0));
        
        // Cohomology dimensions
        for i in 0..20 {
            let dim = sheaf.cohomology_dimension(i).unwrap_or(0);
            features.push(dim as f64);
        }
        
        // Euler characteristic
        let chi = sheaf.euler_characteristic().unwrap_or(0);
        features.push(chi as f64);
        
        // Chern classes
        for i in 1..=4 {
            let chern = sheaf.chern_class(i).unwrap_or(0.0);
            features.push(chern);
        }
        
        // Stalk dimensions at key points
        let key_points = sheaf.key_points();
        for point in key_points.iter().take(50) {
            let stalk_dim = sheaf.stalk_dimension(point).unwrap_or(0);
            features.push(stalk_dim as f64);
        }
        
        // Pad or truncate to target dimension
        self.adjust_dimension(&mut features, config.target_dimension);
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ))
    }
    
    /// Sparse encoding: focus on non-zero/significant features
    fn encode_sparse(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut features = vec![0.0; config.target_dimension];
        let mut feature_index = 0;
        
        // Store only significant features
        if sheaf.rank() > 0 {
            features[feature_index] = sheaf.rank() as f64;
            feature_index += 1;
        }
        
        if let Some(degree) = sheaf.degree() {
            if degree.abs() > config.precision {
                features[feature_index] = degree;
                feature_index += 1;
            }
        }
        
        // Non-zero cohomology dimensions
        for i in 0..20 {
            if let Ok(dim) = sheaf.cohomology_dimension(i) {
                if dim > 0 {
                    features[feature_index] = dim as f64;
                    feature_index += 1;
                    if feature_index >= config.target_dimension {
                        break;
                    }
                }
            }
        }
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ))
    }
    
    /// Hash encoding: create hash-based fingerprint
    fn encode_hash(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash key properties
        sheaf.rank().hash(&mut hasher);
        if let Some(degree) = sheaf.degree() {
            (degree * 1000000.0) as i64.hash(&mut hasher);
        }
        
        // Hash cohomology signature
        for i in 0..10 {
            if let Ok(dim) = sheaf.cohomology_dimension(i) {
                dim.hash(&mut hasher);
            }
        }
        
        let hash_value = hasher.finish();
        
        // Convert hash to feature vector using pseudo-random distribution
        let mut features = Vec::with_capacity(config.target_dimension);
        let mut rng = hash_value;
        
        for _ in 0..config.target_dimension {
            // Linear congruential generator for deterministic pseudo-random numbers
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (rng % 10000) as f64 / 10000.0;
            features.push(normalized);
        }
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ).with_metadata("hash_value".to_string(), hash_value.to_string()))
    }
    
    /// Spectral encoding: use eigenvalue decomposition
    fn encode_spectral(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        // Get spectral data from sheaf (e.g., Laplacian eigenvalues)
        let eigenvalues = match sheaf.spectrum() {
            Ok(spec) => spec.eigenvalues(),
            Err(_) => {
                // Fallback: construct approximate spectrum from available data
                let mut vals = Vec::new();
                vals.push(sheaf.rank() as f64);
                if let Some(degree) = sheaf.degree() {
                    vals.push(degree);
                }
                vals
            }
        };
        
        let mut features = Vec::new();
        
        // First few eigenvalues
        for eigenval in eigenvalues.iter().take(config.target_dimension / 2) {
            features.push(*eigenval);
        }
        
        // Spectral invariants
        if !eigenvalues.is_empty() {
            let trace = eigenvalues.iter().sum::<f64>();
            let spectral_norm = eigenvalues.iter().map(|x| x * x).sum::<f64>().sqrt();
            let spectral_gap = eigenvalues.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
                             - eigenvalues.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            
            features.push(trace);
            features.push(spectral_norm);
            features.push(spectral_gap);
        }
        
        // Pad or truncate to target dimension
        self.adjust_dimension(&mut features, config.target_dimension);
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ))
    }
    
    /// Topological encoding: focus on topological invariants
    fn encode_topological(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        
        // Euler characteristic
        let chi = sheaf.euler_characteristic().unwrap_or(0);
        features.push(chi as f64);
        
        // Betti numbers (cohomology dimensions)
        for i in 0..10 {
            let betti = sheaf.cohomology_dimension(i).unwrap_or(0);
            features.push(betti as f64);
        }
        
        // Chern numbers
        for i in 1..=sheaf.rank().min(4) {
            let chern = sheaf.chern_class(i).unwrap_or(0.0);
            features.push(chern);
        }
        
        // Todd class
        if let Ok(todd) = sheaf.todd_class() {
            features.push(todd);
        } else {
            features.push(0.0);
        }
        
        // Hirzebruch signature (if applicable)
        if let Ok(signature) = sheaf.hirzebruch_signature() {
            features.push(signature as f64);
        } else {
            features.push(0.0);
        }
        
        // Pad or truncate to target dimension
        self.adjust_dimension(&mut features, config.target_dimension);
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ))
    }
    
    /// Hybrid encoding: combine multiple strategies
    fn encode_hybrid(&self, sheaf: &Sheaf, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let dim_per_strategy = config.target_dimension / 3;
        
        // Topological features
        let mut topo_config = config.clone();
        topo_config.strategy = EncodingStrategy::Topological;
        topo_config.target_dimension = dim_per_strategy;
        let topo_features = self.encode_topological(sheaf, &topo_config)?;
        
        // Dense features
        let mut dense_config = config.clone();
        dense_config.strategy = EncodingStrategy::Dense;
        dense_config.target_dimension = dim_per_strategy;
        let dense_features = self.encode_dense(sheaf, &dense_config)?;
        
        // Spectral features
        let mut spectral_config = config.clone();
        spectral_config.strategy = EncodingStrategy::Spectral;
        spectral_config.target_dimension = dim_per_strategy;
        let spectral_features = self.encode_spectral(sheaf, &spectral_config)?;
        
        // Combine features
        let mut combined_features = Vec::new();
        combined_features.extend(topo_features.values);
        combined_features.extend(dense_features.values);
        combined_features.extend(spectral_features.values);
        
        // Pad or truncate to target dimension
        self.adjust_dimension(&mut combined_features, config.target_dimension);
        
        Ok(FeatureVector::new(
            combined_features,
            "Sheaf".to_string(),
            format!("SheafEncoder::{:?}", config.strategy),
        ))
    }
    
    /// Decode from hash encoding (limited reconstruction)
    fn decode_hash(&self, features: &FeatureVector, _config: &EncodingConfig) -> FeatureResult<Sheaf> {
        // Extract hash value from metadata
        let hash_str = features.metadata.extra.get("hash_value")
            .ok_or_else(|| FeatureError::DeserializationFailed(
                "Hash value not found in metadata".to_string()
            ))?;
        
        let hash_value = hash_str.parse::<u64>()
            .map_err(|e| FeatureError::DeserializationFailed(e.to_string()))?;
        
        // This is a simplified reconstruction - in practice, we'd need
        // to store more information for accurate reconstruction
        let rank = ((hash_value % 10) + 1) as usize;
        let degree = ((hash_value >> 8) % 100) as f64 - 50.0;
        
        Sheaf::new_with_properties(rank, Some(degree))
            .map_err(|e| FeatureError::DeserializationFailed(format!("Sheaf creation failed: {:?}", e)))
    }
    
    fn adjust_dimension(&self, features: &mut Vec<f64>, target_dim: usize) {
        if features.len() < target_dim {
            features.resize(target_dim, 0.0);
        } else if features.len() > target_dim {
            features.truncate(target_dim);
        }
    }
}

/// Encoder for bundle objects
#[derive(Debug, Clone)]
pub struct BundleEncoder {
    pub strategy: EncodingStrategy,
}

impl Default for BundleEncoder {
    fn default() -> Self {
        Self {
            strategy: EncodingStrategy::Geometric,
        }
    }
}

impl MathObjectEncoder<Bundle> for BundleEncoder {
    fn encode(&self, bundle: &Bundle, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        
        // Basic bundle properties
        features.push(bundle.rank() as f64);
        features.push(bundle.degree().unwrap_or(0.0));
        
        // Connection features
        if let Some(connection) = bundle.connection() {
            features.push(connection.trace().unwrap_or(0.0));
            features.push(connection.determinant().unwrap_or(0.0));
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Curvature features
        if let Some(curvature) = bundle.curvature() {
            features.push(curvature.scalar().unwrap_or(0.0));
            features.push(curvature.frobenius_norm().unwrap_or(0.0));
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Characteristic classes
        for i in 1..=bundle.rank().min(4) {
            features.push(bundle.chern_class(i).unwrap_or(0.0));
        }
        
        // Pad to target dimension
        if features.len() < config.target_dimension {
            features.resize(config.target_dimension, 0.0);
        } else if features.len() > config.target_dimension {
            features.truncate(config.target_dimension);
        }
        
        Ok(FeatureVector::new(
            features,
            "Bundle".to_string(),
            format!("BundleEncoder::{:?}", config.strategy),
        ))
    }
    
    fn decode(&self, _features: &FeatureVector, _config: &EncodingConfig) -> FeatureResult<Bundle> {
        Err(FeatureError::EncodingFailed {
            message: "Bundle decoding not implemented".to_string(),
        })
    }
    
    fn encoding_quality(&self, original: &Bundle, reconstructed: &Bundle) -> FeatureResult<f64> {
        let rank_diff = (original.rank() as f64 - reconstructed.rank() as f64).abs();
        let degree_diff = original.degree().unwrap_or(0.0) - reconstructed.degree().unwrap_or(0.0);
        
        let quality = 1.0 / (1.0 + rank_diff + degree_diff.abs());
        Ok(quality)
    }
}

/// Encoder for representation objects
#[derive(Debug, Clone)]
pub struct RepresentationEncoder {
    pub strategy: EncodingStrategy,
}

impl Default for RepresentationEncoder {
    fn default() -> Self {
        Self {
            strategy: EncodingStrategy::Algebraic,
        }
    }
}

impl MathObjectEncoder<Representation> for RepresentationEncoder {
    fn encode(&self, rep: &Representation, config: &EncodingConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        
        // Basic representation properties
        features.push(rep.dimension() as f64);
        if let Some(degree) = rep.degree() {
            features.push(degree);
        } else {
            features.push(0.0);
        }
        
        // Character features
        if let Some(character) = rep.character() {
            features.push(character.at_identity());
            features.push(character.norm().unwrap_or(0.0));
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Weight features
        let weights = rep.weights();
        if let Some(highest_weight) = weights.highest_weight() {
            features.push(highest_weight.norm());
            // Add first few coordinates
            for coord in highest_weight.coordinates().iter().take(10) {
                features.push(*coord);
            }
        }
        
        // Weight multiplicities
        for mult in weights.multiplicities().iter().take(20) {
            features.push(*mult as f64);
        }
        
        // Pad to target dimension
        if features.len() < config.target_dimension {
            features.resize(config.target_dimension, 0.0);
        } else if features.len() > config.target_dimension {
            features.truncate(config.target_dimension);
        }
        
        Ok(FeatureVector::new(
            features,
            "Representation".to_string(),
            format!("RepresentationEncoder::{:?}", config.strategy),
        ))
    }
    
    fn decode(&self, _features: &FeatureVector, _config: &EncodingConfig) -> FeatureResult<Representation> {
        Err(FeatureError::EncodingFailed {
            message: "Representation decoding not implemented".to_string(),
        })
    }
    
    fn encoding_quality(&self, original: &Representation, reconstructed: &Representation) -> FeatureResult<f64> {
        let dim_diff = (original.dimension() as f64 - reconstructed.dimension() as f64).abs();
        let quality = 1.0 / (1.0 + dim_diff);
        Ok(quality)
    }
}

/// Universal encoder that can handle multiple object types
#[derive(Debug, Clone)]
pub struct UniversalEncoder {
    sheaf_encoder: SheafEncoder,
    bundle_encoder: BundleEncoder,
    representation_encoder: RepresentationEncoder,
}

impl Default for UniversalEncoder {
    fn default() -> Self {
        Self {
            sheaf_encoder: SheafEncoder::default(),
            bundle_encoder: BundleEncoder::default(),
            representation_encoder: RepresentationEncoder::default(),
        }
    }
}

impl UniversalEncoder {
    /// Encode any supported mathematical object
    pub fn encode_object(
        &self,
        object: &dyn std::any::Any,
        config: &EncodingConfig,
    ) -> FeatureResult<FeatureVector> {
        if let Some(sheaf) = object.downcast_ref::<Sheaf>() {
            self.sheaf_encoder.encode(sheaf, config)
        } else if let Some(bundle) = object.downcast_ref::<Bundle>() {
            self.bundle_encoder.encode(bundle, config)
        } else if let Some(rep) = object.downcast_ref::<Representation>() {
            self.representation_encoder.encode(rep, config)
        } else {
            Err(FeatureError::EncodingFailed {
                message: "Unsupported object type for encoding".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encoding_strategy_default() {
        assert_eq!(EncodingStrategy::default(), EncodingStrategy::Dense);
    }
    
    #[test]
    fn test_encoding_config_default() {
        let config = EncodingConfig::default();
        assert_eq!(config.strategy, EncodingStrategy::Dense);
        assert_eq!(config.target_dimension, 512);
        assert_eq!(config.precision, 1e-12);
        assert!(config.preserve_invariants);
    }
    
    #[test]
    fn test_sheaf_encoder_creation() {
        let encoder = SheafEncoder::default();
        assert_eq!(encoder.strategy, EncodingStrategy::Topological);
    }
    
    #[test]
    fn test_bundle_encoder_creation() {
        let encoder = BundleEncoder::default();
        assert_eq!(encoder.strategy, EncodingStrategy::Geometric);
    }
    
    #[test]
    fn test_representation_encoder_creation() {
        let encoder = RepresentationEncoder::default();
        assert_eq!(encoder.strategy, EncodingStrategy::Algebraic);
    }
}