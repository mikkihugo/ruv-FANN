//! Feature encoders for converting mathematical objects to neural network inputs

use std::collections::HashMap;
use std::sync::Arc;
use ndarray::{Array1, Array2, Axis};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rayon::prelude::*;

use super::{DataError, Result, features::FeatureVector};

/// Encoding strategies for different data types
#[derive(Debug, Clone, Copy)]
pub enum EncodingStrategy {
    /// Direct encoding (raw features)
    Direct,
    /// L2 normalized encoding
    Normalized,
    /// Standardized encoding (zero mean, unit variance)
    Standardized,
    /// Min-max scaled encoding
    MinMaxScaled,
    /// Logarithmic encoding
    Logarithmic,
    /// Spherical encoding
    Spherical,
}

/// Statistical properties for standardization
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub min: Array1<f64>,
    pub max: Array1<f64>,
    pub count: usize,
}

impl FeatureStatistics {
    /// Compute statistics from a batch of features
    pub fn from_features(features: &[FeatureVector]) -> Result<Self> {
        if features.is_empty() {
            return Err(DataError::FeatureExtractionError(
                "Cannot compute statistics from empty feature set".to_string()
            ));
        }
        
        let first = &features[0];
        let total_dim = first.dimension();
        
        let mut mean = Array1::zeros(total_dim);
        let mut min = Array1::from_elem(total_dim, f64::INFINITY);
        let mut max = Array1::from_elem(total_dim, f64::NEG_INFINITY);
        
        // Compute mean, min, max
        for feature in features {
            let flattened = feature.flatten();
            mean = &mean + &flattened;
            
            for (i, &val) in flattened.iter().enumerate() {
                if val < min[i] { min[i] = val; }
                if val > max[i] { max[i] = val; }
            }
        }
        
        mean /= features.len() as f64;
        
        // Compute standard deviation
        let mut var = Array1::zeros(total_dim);
        for feature in features {
            let flattened = feature.flatten();
            let diff = &flattened - &mean;
            var = &var + &(&diff * &diff);
        }
        
        var /= features.len() as f64;
        let std = var.mapv(|x| x.sqrt());
        
        Ok(Self {
            mean,
            std,
            min,
            max,
            count: features.len(),
        })
    }
    
    /// Update statistics with new batch
    pub fn update(&mut self, features: &[FeatureVector]) -> Result<()> {
        if features.is_empty() {
            return Ok(());
        }
        
        let new_stats = Self::from_features(features)?;
        
        // Online update formulas
        let old_count = self.count as f64;
        let new_count = features.len() as f64;
        let total_count = old_count + new_count;
        
        // Update mean
        self.mean = (&self.mean * old_count + &new_stats.mean * new_count) / total_count;
        
        // Update min/max
        for i in 0..self.min.len() {
            self.min[i] = self.min[i].min(new_stats.min[i]);
            self.max[i] = self.max[i].max(new_stats.max[i]);
        }
        
        // Update standard deviation (simplified)
        let combined_var = (&self.std * &self.std * old_count + 
                           &new_stats.std * &new_stats.std * new_count) / total_count;
        self.std = combined_var.mapv(|x| x.sqrt());
        
        self.count += features.len();
        
        Ok(())
    }
}

/// Feature encoder for neural network input
pub struct FeatureEncoder {
    strategy: EncodingStrategy,
    statistics: Option<FeatureStatistics>,
    epsilon: f64,
}

impl FeatureEncoder {
    /// Create a new encoder with the given strategy
    pub fn new(strategy: EncodingStrategy) -> Self {
        Self {
            strategy,
            statistics: None,
            epsilon: 1e-8,
        }
    }
    
    /// Fit the encoder to a dataset (compute statistics)
    pub fn fit(&mut self, features: &[FeatureVector]) -> Result<()> {
        match self.strategy {
            EncodingStrategy::Standardized | 
            EncodingStrategy::MinMaxScaled => {
                self.statistics = Some(FeatureStatistics::from_features(features)?);
            }
            _ => {} // Other strategies don't need fitting
        }
        Ok(())
    }
    
    /// Encode a single feature vector
    pub fn encode(&self, feature: &FeatureVector) -> Result<Array1<f64>> {
        let mut encoded = feature.flatten();
        
        match self.strategy {
            EncodingStrategy::Direct => {
                // No transformation
            }
            EncodingStrategy::Normalized => {
                let norm = encoded.dot(&encoded).sqrt();
                if norm > self.epsilon {
                    encoded /= norm;
                }
            }
            EncodingStrategy::Standardized => {
                if let Some(stats) = &self.statistics {
                    encoded = (&encoded - &stats.mean) / &stats.std.mapv(|x| x.max(self.epsilon));
                } else {
                    return Err(DataError::FeatureExtractionError(
                        "Encoder not fitted for standardization".to_string()
                    ));
                }
            }
            EncodingStrategy::MinMaxScaled => {
                if let Some(stats) = &self.statistics {
                    let range = &stats.max - &stats.min;
                    encoded = (&encoded - &stats.min) / &range.mapv(|x| x.max(self.epsilon));
                } else {
                    return Err(DataError::FeatureExtractionError(
                        "Encoder not fitted for min-max scaling".to_string()
                    ));
                }
            }
            EncodingStrategy::Logarithmic => {
                encoded = encoded.mapv(|x| (x.abs() + self.epsilon).ln() * x.signum());
            }
            EncodingStrategy::Spherical => {
                // Map to unit sphere
                let norm = encoded.dot(&encoded).sqrt();
                if norm > self.epsilon {
                    encoded /= norm;
                }
                // Apply inverse stereographic projection for higher dimensions
                let r_sq = encoded.dot(&encoded);
                let scale = 2.0 / (1.0 + r_sq);
                encoded *= scale;
            }
        }
        
        Ok(encoded)
    }
    
    /// Encode a batch of features in parallel
    pub fn encode_batch(&self, features: &[FeatureVector]) -> Result<Array2<f64>> {
        let encoded: Result<Vec<Array1<f64>>> = features.par_iter()
            .map(|f| self.encode(f))
            .collect();
        
        let encoded = encoded?;
        
        if encoded.is_empty() {
            return Err(DataError::FeatureExtractionError(
                "Empty feature batch".to_string()
            ));
        }
        
        let num_samples = encoded.len();
        let feature_dim = encoded[0].len();
        
        let mut batch = Array2::zeros((num_samples, feature_dim));
        for (i, enc) in encoded.into_iter().enumerate() {
            batch.row_mut(i).assign(&enc);
        }
        
        Ok(batch)
    }
    
    /// Decode from encoded representation (if possible)
    pub fn decode(&self, encoded: &Array1<f64>) -> Result<Array1<f64>> {
        let mut decoded = encoded.clone();
        
        match self.strategy {
            EncodingStrategy::Direct | 
            EncodingStrategy::Normalized => {
                // These are invertible or close enough
            }
            EncodingStrategy::Standardized => {
                if let Some(stats) = &self.statistics {
                    decoded = &decoded * &stats.std + &stats.mean;
                } else {
                    return Err(DataError::FeatureExtractionError(
                        "Cannot decode without fitted statistics".to_string()
                    ));
                }
            }
            EncodingStrategy::MinMaxScaled => {
                if let Some(stats) = &self.statistics {
                    let range = &stats.max - &stats.min;
                    decoded = &decoded * &range + &stats.min;
                } else {
                    return Err(DataError::FeatureExtractionError(
                        "Cannot decode without fitted statistics".to_string()
                    ));
                }
            }
            EncodingStrategy::Logarithmic => {
                decoded = decoded.mapv(|x| x.exp());
            }
            EncodingStrategy::Spherical => {
                // Approximate inverse (not exact for stereographic projection)
                let norm = decoded.dot(&decoded).sqrt();
                if norm > self.epsilon {
                    decoded /= norm;
                }
            }
        }
        
        Ok(decoded)
    }
}

/// Multi-modal encoder for different feature types
pub struct MultiModalEncoder {
    geometric_encoder: FeatureEncoder,
    algebraic_encoder: FeatureEncoder,
    spectral_encoder: Option<SpectralEncoder>,
    topological_encoder: Option<TopologicalEncoder>,
}

impl MultiModalEncoder {
    /// Create a new multi-modal encoder
    pub fn new(
        geometric_strategy: EncodingStrategy,
        algebraic_strategy: EncodingStrategy,
        enable_spectral: bool,
        enable_topological: bool,
    ) -> Self {
        Self {
            geometric_encoder: FeatureEncoder::new(geometric_strategy),
            algebraic_encoder: FeatureEncoder::new(algebraic_strategy),
            spectral_encoder: if enable_spectral { 
                Some(SpectralEncoder::new()) 
            } else { 
                None 
            },
            topological_encoder: if enable_topological { 
                Some(TopologicalEncoder::new()) 
            } else { 
                None 
            },
        }
    }
    
    /// Fit all encoders to the dataset
    pub fn fit(&mut self, features: &[FeatureVector]) -> Result<()> {
        // Extract individual feature types
        let geometric_features: Vec<_> = features.iter()
            .map(|f| {
                let mut fv = FeatureVector::new(f.id.clone(), f.geometric.len(), 0);
                fv.geometric = f.geometric.clone();
                fv
            })
            .collect();
        
        let algebraic_features: Vec<_> = features.iter()
            .map(|f| {
                let mut fv = FeatureVector::new(f.id.clone(), 0, f.algebraic.len());
                fv.algebraic = f.algebraic.clone();
                fv
            })
            .collect();
        
        self.geometric_encoder.fit(&geometric_features)?;
        self.algebraic_encoder.fit(&algebraic_features)?;
        
        if let Some(spectral) = &mut self.spectral_encoder {
            spectral.fit(features)?;
        }
        
        if let Some(topo) = &mut self.topological_encoder {
            topo.fit(features)?;
        }
        
        Ok(())
    }
    
    /// Encode a feature vector using all modalities
    pub fn encode(&self, feature: &FeatureVector) -> Result<EncodedFeature> {
        // Create temporary feature vectors for each modality
        let mut geo_fv = FeatureVector::new(feature.id.clone(), feature.geometric.len(), 0);
        geo_fv.geometric = feature.geometric.clone();
        
        let mut alg_fv = FeatureVector::new(feature.id.clone(), 0, feature.algebraic.len());
        alg_fv.algebraic = feature.algebraic.clone();
        
        let geometric = self.geometric_encoder.encode(&geo_fv)?;
        let algebraic = self.algebraic_encoder.encode(&alg_fv)?;
        
        let spectral = if let Some(encoder) = &self.spectral_encoder {
            Some(encoder.encode(feature)?)
        } else {
            None
        };
        
        let topological = if let Some(encoder) = &self.topological_encoder {
            Some(encoder.encode(feature)?)
        } else {
            None
        };
        
        Ok(EncodedFeature {
            geometric,
            algebraic,
            spectral,
            topological,
        })
    }
    
    /// Encode a batch and concatenate all modalities
    pub fn encode_batch_concatenated(&self, features: &[FeatureVector]) -> Result<Array2<f64>> {
        let encoded: Result<Vec<EncodedFeature>> = features.par_iter()
            .map(|f| self.encode(f))
            .collect();
        
        let encoded = encoded?;
        
        if encoded.is_empty() {
            return Err(DataError::FeatureExtractionError(
                "Empty feature batch".to_string()
            ));
        }
        
        // Calculate total dimension
        let first = &encoded[0];
        let mut total_dim = first.geometric.len() + first.algebraic.len();
        
        if let Some(ref spectral) = first.spectral {
            total_dim += spectral.len();
        }
        
        if let Some(ref topo) = first.topological {
            total_dim += topo.len();
        }
        
        let num_samples = encoded.len();
        let mut batch = Array2::zeros((num_samples, total_dim));
        
        for (i, enc) in encoded.iter().enumerate() {
            let mut row = batch.row_mut(i);
            let mut offset = 0;
            
            // Copy geometric features
            let geo_len = enc.geometric.len();
            row.slice_mut(ndarray::s![offset..offset + geo_len])
                .assign(&enc.geometric);
            offset += geo_len;
            
            // Copy algebraic features
            let alg_len = enc.algebraic.len();
            row.slice_mut(ndarray::s![offset..offset + alg_len])
                .assign(&enc.algebraic);
            offset += alg_len;
            
            // Copy spectral features
            if let Some(ref spectral) = enc.spectral {
                let spec_len = spectral.len();
                row.slice_mut(ndarray::s![offset..offset + spec_len])
                    .assign(spectral);
                offset += spec_len;
            }
            
            // Copy topological features
            if let Some(ref topo) = enc.topological {
                let topo_len = topo.len();
                row.slice_mut(ndarray::s![offset..offset + topo_len])
                    .assign(topo);
            }
        }
        
        Ok(batch)
    }
}

/// Encoded feature with separate modalities
#[derive(Debug, Clone)]
pub struct EncodedFeature {
    pub geometric: Array1<f64>,
    pub algebraic: Array1<f64>,
    pub spectral: Option<Array1<f64>>,
    pub topological: Option<Array1<f64>>,
}

/// Specialized encoder for spectral features (complex numbers)
pub struct SpectralEncoder {
    strategy: SpectralEncodingStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum SpectralEncodingStrategy {
    /// Real and imaginary parts separately
    RealImaginary,
    /// Magnitude and phase
    MagnitudePhase,
    /// Only magnitude
    MagnitudeOnly,
}

impl SpectralEncoder {
    pub fn new() -> Self {
        Self {
            strategy: SpectralEncodingStrategy::RealImaginary,
        }
    }
    
    pub fn fit(&mut self, _features: &[FeatureVector]) -> Result<()> {
        // No fitting needed for spectral features currently
        Ok(())
    }
    
    pub fn encode(&self, feature: &FeatureVector) -> Result<Array1<f64>> {
        if let Some(spectral) = &feature.spectral {
            match self.strategy {
                SpectralEncodingStrategy::RealImaginary => {
                    let mut encoded = Vec::with_capacity(spectral.len() * 2);
                    for &c in spectral.iter() {
                        encoded.push(c.re);
                        encoded.push(c.im);
                    }
                    Ok(Array1::from_vec(encoded))
                }
                SpectralEncodingStrategy::MagnitudePhase => {
                    let mut encoded = Vec::with_capacity(spectral.len() * 2);
                    for &c in spectral.iter() {
                        encoded.push(c.norm());
                        encoded.push(c.arg());
                    }
                    Ok(Array1::from_vec(encoded))
                }
                SpectralEncodingStrategy::MagnitudeOnly => {
                    let encoded: Vec<f64> = spectral.iter()
                        .map(|c| c.norm())
                        .collect();
                    Ok(Array1::from_vec(encoded))
                }
            }
        } else {
            Ok(Array1::zeros(0))
        }
    }
}

/// Specialized encoder for topological features (integers)
pub struct TopologicalEncoder {
    max_values: HashMap<usize, i32>,
}

impl TopologicalEncoder {
    pub fn new() -> Self {
        Self {
            max_values: HashMap::new(),
        }
    }
    
    pub fn fit(&mut self, features: &[FeatureVector]) -> Result<()> {
        // Find maximum values for each topological feature
        for feature in features {
            if let Some(topo) = &feature.topological {
                for (i, &val) in topo.iter().enumerate() {
                    let current_max = self.max_values.get(&i).copied().unwrap_or(val);
                    self.max_values.insert(i, current_max.max(val));
                }
            }
        }
        Ok(())
    }
    
    pub fn encode(&self, feature: &FeatureVector) -> Result<Array1<f64>> {
        if let Some(topo) = &feature.topological {
            let encoded: Vec<f64> = topo.iter()
                .enumerate()
                .map(|(i, &val)| {
                    // Normalize by maximum value
                    let max_val = self.max_values.get(&i).copied().unwrap_or(1);
                    if max_val > 0 {
                        val as f64 / max_val as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            Ok(Array1::from_vec(encoded))
        } else {
            Ok(Array1::zeros(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    
    #[test]
    fn test_feature_statistics() {
        let mut features = vec![
            FeatureVector::new("test1".to_string(), 3, 2),
            FeatureVector::new("test2".to_string(), 3, 2),
        ];
        
        features[0].geometric = arr1(&[1.0, 2.0, 3.0]);
        features[0].algebraic = arr1(&[0.5, 1.5]);
        
        features[1].geometric = arr1(&[2.0, 3.0, 4.0]);
        features[1].algebraic = arr1(&[1.0, 2.0]);
        
        let stats = FeatureStatistics::from_features(&features).unwrap();
        
        assert_eq!(stats.count, 2);
        assert!((stats.mean[0] - 1.5).abs() < 1e-10);
        assert!((stats.mean[1] - 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_encoder_strategies() {
        let mut feature = FeatureVector::new("test".to_string(), 2, 2);
        feature.geometric = arr1(&[3.0, 4.0]);
        feature.algebraic = arr1(&[1.0, 2.0]);
        
        // Test normalization
        let encoder = FeatureEncoder::new(EncodingStrategy::Normalized);
        let encoded = encoder.encode(&feature).unwrap();
        let norm = encoded.dot(&encoded).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_multi_modal_encoder() {
        let features = vec![
            FeatureVector::new("test1".to_string(), 2, 2),
        ];
        
        let mut encoder = MultiModalEncoder::new(
            EncodingStrategy::Normalized,
            EncodingStrategy::Standardized,
            false,
            false,
        );
        
        encoder.fit(&features).unwrap();
        let encoded = encoder.encode(&features[0]).unwrap();
        
        assert_eq!(encoded.geometric.len(), 2);
        assert_eq!(encoded.algebraic.len(), 2);
    }
}