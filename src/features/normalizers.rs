//! Feature normalization and scaling for neural network compatibility
//!
//! This module provides various normalization strategies to ensure feature vectors
//! are properly scaled for neural network training and inference.

use crate::features::{FeatureVector, FeatureResult, FeatureError};
use std::collections::HashMap;

/// Types of normalization available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NormalizationType {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to [-1, 1]
    MinMaxSymmetric,
    /// Z-score normalization (zero mean, unit variance)
    ZScore,
    /// L1 normalization (sum of absolute values = 1)
    L1,
    /// L2 normalization (Euclidean norm = 1)
    L2,
    /// Max normalization (max absolute value = 1)
    Max,
    /// Robust normalization using median and IQR
    Robust,
    /// Quantile normalization
    Quantile,
    /// Unit variance normalization
    UnitVariance,
}

impl Default for NormalizationType {
    fn default() -> Self {
        NormalizationType::L2
    }
}

/// Statistics for feature normalization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalizationStats {
    /// Minimum values per feature
    pub min_values: Vec<f64>,
    /// Maximum values per feature
    pub max_values: Vec<f64>,
    /// Mean values per feature
    pub mean_values: Vec<f64>,
    /// Standard deviation per feature
    pub std_values: Vec<f64>,
    /// Median values per feature
    pub median_values: Vec<f64>,
    /// Interquartile range per feature
    pub iqr_values: Vec<f64>,
    /// Number of samples used to compute statistics
    pub sample_count: usize,
    /// Normalization type used
    pub normalization_type: NormalizationType,
}

impl NormalizationStats {
    /// Create new normalization statistics
    pub fn new(dimension: usize, normalization_type: NormalizationType) -> Self {
        Self {
            min_values: vec![f64::INFINITY; dimension],
            max_values: vec![f64::NEG_INFINITY; dimension],
            mean_values: vec![0.0; dimension],
            std_values: vec![1.0; dimension],
            median_values: vec![0.0; dimension],
            iqr_values: vec![1.0; dimension],
            sample_count: 0,
            normalization_type,
        }
    }
    
    /// Update statistics with a new feature vector
    pub fn update(&mut self, features: &FeatureVector) -> FeatureResult<()> {
        if features.dimension != self.min_values.len() {
            return Err(FeatureError::ValidationFailed {
                message: format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    self.min_values.len(),
                    features.dimension
                ),
            });
        }
        
        // Update min/max
        for (i, value) in features.values.iter().enumerate() {
            self.min_values[i] = self.min_values[i].min(*value);
            self.max_values[i] = self.max_values[i].max(*value);
        }
        
        // Update running mean (Welford's online algorithm for numerical stability)
        for (i, value) in features.values.iter().enumerate() {
            let delta = value - self.mean_values[i];
            self.mean_values[i] += delta / (self.sample_count + 1) as f64;
        }
        
        self.sample_count += 1;
        Ok(())
    }
    
    /// Finalize statistics computation (compute std, median, etc.)
    pub fn finalize(&mut self, all_features: &[FeatureVector]) -> FeatureResult<()> {
        if all_features.is_empty() {
            return Ok(());
        }
        
        let dimension = all_features[0].dimension;
        
        // Compute standard deviation
        for i in 0..dimension {
            let variance = all_features
                .iter()
                .map(|fv| {
                    let diff = fv.values[i] - self.mean_values[i];
                    diff * diff
                })
                .sum::<f64>() / all_features.len() as f64;
            self.std_values[i] = variance.sqrt();
        }
        
        // Compute median and IQR
        for i in 0..dimension {
            let mut sorted_values: Vec<f64> = all_features
                .iter()
                .map(|fv| fv.values[i])
                .collect();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let len = sorted_values.len();
            self.median_values[i] = if len % 2 == 0 {
                (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
            } else {
                sorted_values[len / 2]
            };
            
            // Compute IQR
            let q1_idx = len / 4;
            let q3_idx = (3 * len) / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx.min(len - 1)];
            self.iqr_values[i] = (q3 - q1).max(1e-8); // Avoid division by zero
        }
        
        Ok(())
    }
}

/// Feature normalizer
#[derive(Debug, Clone)]
pub struct FeatureNormalizer {
    /// Normalization statistics
    pub stats: Option<NormalizationStats>,
    /// Whether the normalizer is fitted
    pub fitted: bool,
    /// Epsilon for numerical stability
    pub epsilon: f64,
}

impl Default for FeatureNormalizer {
    fn default() -> Self {
        Self {
            stats: None,
            fitted: false,
            epsilon: 1e-8,
        }
    }
}

impl FeatureNormalizer {
    /// Create a new normalizer
    pub fn new(epsilon: f64) -> Self {
        Self {
            stats: None,
            fitted: false,
            epsilon,
        }
    }
    
    /// Fit the normalizer to a collection of feature vectors
    pub fn fit(
        &mut self,
        features: &[FeatureVector],
        normalization_type: NormalizationType,
    ) -> FeatureResult<()> {
        if features.is_empty() {
            return Err(FeatureError::ValidationFailed {
                message: "Cannot fit normalizer on empty feature set".to_string(),
            });
        }
        
        let dimension = features[0].dimension;
        
        // Check all features have the same dimension
        for (i, fv) in features.iter().enumerate() {
            if fv.dimension != dimension {
                return Err(FeatureError::ValidationFailed {
                    message: format!(
                        "Inconsistent feature dimensions at index {}: expected {}, got {}",
                        i, dimension, fv.dimension
                    ),
                });
            }
        }
        
        let mut stats = NormalizationStats::new(dimension, normalization_type);
        
        // Update statistics with all features
        for fv in features {
            stats.update(fv)?;
        }
        
        // Finalize statistics
        stats.finalize(features)?;
        
        self.stats = Some(stats);
        self.fitted = true;
        
        Ok(())
    }
    
    /// Transform a feature vector using the fitted normalizer
    pub fn transform(&self, features: &FeatureVector) -> FeatureResult<FeatureVector> {
        if !self.fitted {
            return Err(FeatureError::NormalizationFailed {
                message: "Normalizer not fitted. Call fit() first.".to_string(),
            });
        }
        
        let stats = self.stats.as_ref().unwrap();
        
        if features.dimension != stats.min_values.len() {
            return Err(FeatureError::ValidationFailed {
                message: format!(
                    "Feature dimension mismatch: expected {}, got {}",
                    stats.min_values.len(),
                    features.dimension
                ),
            });
        }
        
        let normalized_values = match stats.normalization_type {
            NormalizationType::None => features.values.clone(),
            NormalizationType::MinMax => self.normalize_min_max(&features.values, stats, false)?,
            NormalizationType::MinMaxSymmetric => self.normalize_min_max(&features.values, stats, true)?,
            NormalizationType::ZScore => self.normalize_z_score(&features.values, stats)?,
            NormalizationType::L1 => self.normalize_l1(&features.values)?,
            NormalizationType::L2 => self.normalize_l2(&features.values)?,
            NormalizationType::Max => self.normalize_max(&features.values)?,
            NormalizationType::Robust => self.normalize_robust(&features.values, stats)?,
            NormalizationType::Quantile => self.normalize_quantile(&features.values, stats)?,
            NormalizationType::UnitVariance => self.normalize_unit_variance(&features.values, stats)?,
        };
        
        let mut result = FeatureVector::new(
            normalized_values,
            features.metadata.object_type.clone(),
            features.metadata.encoding_strategy.clone(),
        );
        
        result.metadata.normalization = Some(format!("{:?}", stats.normalization_type));
        result.labels = features.labels.clone();
        
        Ok(result)
    }
    
    /// Inverse transform (denormalize) a feature vector
    pub fn inverse_transform(&self, features: &FeatureVector) -> FeatureResult<FeatureVector> {
        if !self.fitted {
            return Err(FeatureError::NormalizationFailed {
                message: "Normalizer not fitted. Call fit() first.".to_string(),
            });
        }
        
        let stats = self.stats.as_ref().unwrap();
        
        let denormalized_values = match stats.normalization_type {
            NormalizationType::None => features.values.clone(),
            NormalizationType::MinMax => self.denormalize_min_max(&features.values, stats, false)?,
            NormalizationType::MinMaxSymmetric => self.denormalize_min_max(&features.values, stats, true)?,
            NormalizationType::ZScore => self.denormalize_z_score(&features.values, stats)?,
            NormalizationType::Robust => self.denormalize_robust(&features.values, stats)?,
            NormalizationType::UnitVariance => self.denormalize_unit_variance(&features.values, stats)?,
            _ => {
                return Err(FeatureError::NormalizationFailed {
                    message: format!(
                        "Inverse transform not supported for {:?}",
                        stats.normalization_type
                    ),
                });
            }
        };
        
        let mut result = FeatureVector::new(
            denormalized_values,
            features.metadata.object_type.clone(),
            features.metadata.encoding_strategy.clone(),
        );
        
        result.labels = features.labels.clone();
        
        Ok(result)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        features: &[FeatureVector],
        normalization_type: NormalizationType,
    ) -> FeatureResult<Vec<FeatureVector>> {
        self.fit(features, normalization_type)?;
        
        let mut results = Vec::with_capacity(features.len());
        for fv in features {
            results.push(self.transform(fv)?);
        }
        
        Ok(results)
    }
    
    // Normalization implementations
    
    fn normalize_min_max(
        &self,
        values: &[f64],
        stats: &NormalizationStats,
        symmetric: bool,
    ) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let min_val = stats.min_values[i];
            let max_val = stats.max_values[i];
            let range = max_val - min_val;
            
            let normalized = if range.abs() < self.epsilon {
                if symmetric { 0.0 } else { 0.5 }
            } else {
                let unit_normalized = (value - min_val) / range;
                if symmetric {
                    2.0 * unit_normalized - 1.0 // Map to [-1, 1]
                } else {
                    unit_normalized // Map to [0, 1]
                }
            };
            
            result.push(normalized);
        }
        
        Ok(result)
    }
    
    fn normalize_z_score(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let mean = stats.mean_values[i];
            let std = stats.std_values[i];
            
            let normalized = if std < self.epsilon {
                0.0
            } else {
                (value - mean) / std
            };
            
            result.push(normalized);
        }
        
        Ok(result)
    }
    
    fn normalize_l1(&self, values: &[f64]) -> FeatureResult<Vec<f64>> {
        let l1_norm = values.iter().map(|x| x.abs()).sum::<f64>();
        
        if l1_norm < self.epsilon {
            return Ok(vec![0.0; values.len()]);
        }
        
        Ok(values.iter().map(|x| x / l1_norm).collect())
    }
    
    fn normalize_l2(&self, values: &[f64]) -> FeatureResult<Vec<f64>> {
        let l2_norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if l2_norm < self.epsilon {
            return Ok(vec![0.0; values.len()]);
        }
        
        Ok(values.iter().map(|x| x / l2_norm).collect())
    }
    
    fn normalize_max(&self, values: &[f64]) -> FeatureResult<Vec<f64>> {
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        
        if max_abs < self.epsilon {
            return Ok(vec![0.0; values.len()]);
        }
        
        Ok(values.iter().map(|x| x / max_abs).collect())
    }
    
    fn normalize_robust(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let median = stats.median_values[i];
            let iqr = stats.iqr_values[i];
            
            let normalized = if iqr < self.epsilon {
                0.0
            } else {
                (value - median) / iqr
            };
            
            result.push(normalized);
        }
        
        Ok(result)
    }
    
    fn normalize_quantile(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        // This is a simplified quantile normalization
        // In practice, you'd want to use the full empirical CDF
        self.normalize_robust(values, stats)
    }
    
    fn normalize_unit_variance(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let std = stats.std_values[i];
            
            let normalized = if std < self.epsilon {
                value
            } else {
                value / std
            };
            
            result.push(normalized);
        }
        
        Ok(result)
    }
    
    // Denormalization implementations
    
    fn denormalize_min_max(
        &self,
        values: &[f64],
        stats: &NormalizationStats,
        symmetric: bool,
    ) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let min_val = stats.min_values[i];
            let max_val = stats.max_values[i];
            let range = max_val - min_val;
            
            let denormalized = if range.abs() < self.epsilon {
                min_val
            } else {
                let unit_value = if symmetric {
                    (value + 1.0) / 2.0 // Map from [-1, 1] to [0, 1]
                } else {
                    value
                };
                min_val + unit_value * range
            };
            
            result.push(denormalized);
        }
        
        Ok(result)
    }
    
    fn denormalize_z_score(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let mean = stats.mean_values[i];
            let std = stats.std_values[i];
            
            let denormalized = mean + value * std;
            result.push(denormalized);
        }
        
        Ok(result)
    }
    
    fn denormalize_robust(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let median = stats.median_values[i];
            let iqr = stats.iqr_values[i];
            
            let denormalized = median + value * iqr;
            result.push(denormalized);
        }
        
        Ok(result)
    }
    
    fn denormalize_unit_variance(&self, values: &[f64], stats: &NormalizationStats) -> FeatureResult<Vec<f64>> {
        let mut result = Vec::with_capacity(values.len());
        
        for (i, &value) in values.iter().enumerate() {
            let std = stats.std_values[i];
            let denormalized = value * std;
            result.push(denormalized);
        }
        
        Ok(result)
    }
}

/// Batch normalizer for processing multiple feature vectors efficiently
#[derive(Debug, Clone)]
pub struct BatchNormalizer {
    base_normalizer: FeatureNormalizer,
    batch_size: usize,
}

impl BatchNormalizer {
    /// Create a new batch normalizer
    pub fn new(batch_size: usize) -> Self {
        Self {
            base_normalizer: FeatureNormalizer::default(),
            batch_size,
        }
    }
    
    /// Process features in batches
    pub fn fit_transform_batched(
        &mut self,
        features: &[FeatureVector],
        normalization_type: NormalizationType,
    ) -> FeatureResult<Vec<FeatureVector>> {
        // First, fit on all data
        self.base_normalizer.fit(features, normalization_type)?;
        
        let mut results = Vec::with_capacity(features.len());
        
        // Process in batches
        for batch in features.chunks(self.batch_size) {
            for fv in batch {
                results.push(self.base_normalizer.transform(fv)?);
            }
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_features() -> Vec<FeatureVector> {
        vec![
            FeatureVector::new(vec![1.0, 2.0, 3.0], "test".to_string(), "test".to_string()),
            FeatureVector::new(vec![4.0, 5.0, 6.0], "test".to_string(), "test".to_string()),
            FeatureVector::new(vec![7.0, 8.0, 9.0], "test".to_string(), "test".to_string()),
        ]
    }
    
    #[test]
    fn test_normalization_type_default() {
        assert_eq!(NormalizationType::default(), NormalizationType::L2);
    }
    
    #[test]
    fn test_feature_normalizer_creation() {
        let normalizer = FeatureNormalizer::new(1e-6);
        assert!(!normalizer.fitted);
        assert_eq!(normalizer.epsilon, 1e-6);
    }
    
    #[test]
    fn test_l2_normalization() {
        let mut normalizer = FeatureNormalizer::default();
        let features = create_test_features();
        
        let normalized = normalizer.fit_transform(&features, NormalizationType::L2).unwrap();
        
        // Check that each vector has unit L2 norm
        for fv in &normalized {
            let norm = fv.norm();
            assert!((norm - 1.0).abs() < 1e-10, "L2 norm should be 1, got {}", norm);
        }
    }
    
    #[test]
    fn test_min_max_normalization() {
        let mut normalizer = FeatureNormalizer::default();
        let features = create_test_features();
        
        let normalized = normalizer.fit_transform(&features, NormalizationType::MinMax).unwrap();
        
        // Check that values are in [0, 1] range
        for fv in &normalized {
            for &value in &fv.values {
                assert!(value >= 0.0 && value <= 1.0, "Value {} not in [0,1] range", value);
            }
        }
    }
    
    #[test]
    fn test_z_score_normalization() {
        let mut normalizer = FeatureNormalizer::default();
        let features = create_test_features();
        
        let normalized = normalizer.fit_transform(&features, NormalizationType::ZScore).unwrap();
        
        // Check that each feature dimension has approximately zero mean
        let n_features = normalized.len();
        let n_dims = normalized[0].dimension;
        
        for dim in 0..n_dims {
            let mean: f64 = normalized.iter().map(|fv| fv.values[dim]).sum::<f64>() / n_features as f64;
            assert!(mean.abs() < 1e-10, "Mean for dimension {} should be ~0, got {}", dim, mean);
        }
    }
    
    #[test]
    fn test_inverse_transform() {
        let mut normalizer = FeatureNormalizer::default();
        let features = create_test_features();
        
        let normalized = normalizer.fit_transform(&features, NormalizationType::ZScore).unwrap();
        
        // Test inverse transform
        let denormalized = normalizer.inverse_transform(&normalized[0]).unwrap();
        
        // Should be close to original
        for (orig, denorm) in features[0].values.iter().zip(denormalized.values.iter()) {
            assert!((orig - denorm).abs() < 1e-10, "Original {} != denormalized {}", orig, denorm);
        }
    }
}