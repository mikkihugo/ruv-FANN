//! Data validation and integrity checks for mathematical objects and feature vectors
//!
//! This module provides comprehensive validation systems to ensure data integrity
//! throughout the feature extraction pipeline.

use crate::core::prelude::*;
use crate::features::{FeatureVector, FeatureResult, FeatureError};
use std::collections::HashMap;

/// Validation result containing details about validation checks
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings
    pub warnings: Vec<String>,
    /// Validation score (0.0 = invalid, 1.0 = perfect)
    pub score: f64,
    /// Detailed validation metrics
    pub metrics: HashMap<String, f64>,
}

impl ValidationResult {
    /// Create a new successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: 1.0,
            metrics: HashMap::new(),
        }
    }
    
    /// Create a new failed validation result
    pub fn failure(error: String) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
            score: 0.0,
            metrics: HashMap::new(),
        }
    }
    
    /// Add an error to the validation result
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
        self.score = 0.0;
    }
    
    /// Add a warning to the validation result
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// Add a metric to the validation result
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }
    
    /// Update the validation score
    pub fn set_score(&mut self, score: f64) {
        self.score = score.max(0.0).min(1.0);
        if score < 0.5 {
            self.is_valid = false;
        }
    }
}

/// Configuration for data validation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationConfig {
    /// Check for NaN values
    pub check_nan: bool,
    /// Check for infinite values
    pub check_infinite: bool,
    /// Check value ranges
    pub check_ranges: bool,
    /// Minimum allowed value
    pub min_value: Option<f64>,
    /// Maximum allowed value
    pub max_value: Option<f64>,
    /// Check for zero vectors
    pub check_zero_vectors: bool,
    /// Check mathematical properties
    pub check_math_properties: bool,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Check feature consistency
    pub check_consistency: bool,
    /// Check metadata integrity
    pub check_metadata: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_infinite: true,
            check_ranges: true,
            min_value: Some(-1e6),
            max_value: Some(1e6),
            check_zero_vectors: true,
            check_math_properties: true,
            tolerance: 1e-12,
            check_consistency: true,
            check_metadata: true,
        }
    }
}

/// Main trait for data validators
pub trait DataValidator<T> {
    /// Validate an object and return validation result
    fn validate(&self, object: &T, config: &ValidationConfig) -> ValidationResult;
    
    /// Quick validity check (true/false)
    fn is_valid(&self, object: &T, config: &ValidationConfig) -> bool {
        self.validate(object, config).is_valid
    }
    
    /// Get validator name
    fn name(&self) -> &'static str;
}

/// Validator for feature vectors
#[derive(Debug, Clone)]
pub struct FeatureVectorValidator;

impl DataValidator<FeatureVector> for FeatureVectorValidator {
    fn validate(&self, features: &FeatureVector, config: &ValidationConfig) -> ValidationResult {
        let mut result = ValidationResult::success();
        
        // Check basic properties
        self.validate_basic_properties(features, &mut result, config);
        
        // Check numerical properties
        self.validate_numerical_properties(features, &mut result, config);
        
        // Check metadata
        if config.check_metadata {
            self.validate_metadata(features, &mut result, config);
        }
        
        // Check mathematical consistency
        if config.check_math_properties {
            self.validate_mathematical_properties(features, &mut result, config);
        }
        
        // Compute overall score
        self.compute_validation_score(&mut result);
        
        result
    }
    
    fn name(&self) -> &'static str {
        "FeatureVectorValidator"
    }
}

impl FeatureVectorValidator {
    /// Validate basic properties of feature vector
    fn validate_basic_properties(
        &self,
        features: &FeatureVector,
        result: &mut ValidationResult,
        _config: &ValidationConfig,
    ) {
        // Check dimension consistency
        if features.values.len() != features.dimension {
            result.add_error(format!(
                "Dimension mismatch: declared {}, actual {}",
                features.dimension,
                features.values.len()
            ));
        }
        
        // Check for empty features
        if features.values.is_empty() {
            result.add_error("Feature vector is empty".to_string());
        }
        
        // Check labels consistency
        if let Some(ref labels) = features.labels {
            if labels.len() != features.dimension {
                result.add_warning(format!(
                    "Label count ({}) doesn't match feature dimension ({})",
                    labels.len(),
                    features.dimension
                ));
            }
        }
    }
    
    /// Validate numerical properties
    fn validate_numerical_properties(
        &self,
        features: &FeatureVector,
        result: &mut ValidationResult,
        config: &ValidationConfig,
    ) {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut out_of_range_count = 0;
        let mut zero_count = 0;
        
        for (i, &value) in features.values.iter().enumerate() {
            // Check for NaN
            if config.check_nan && value.is_nan() {
                nan_count += 1;
                result.add_error(format!("NaN value at index {}", i));
            }
            
            // Check for infinity
            if config.check_infinite && value.is_infinite() {
                inf_count += 1;
                result.add_error(format!("Infinite value at index {}: {}", i, value));
            }
            
            // Check ranges
            if config.check_ranges {
                if let Some(min_val) = config.min_value {
                    if value < min_val {
                        out_of_range_count += 1;
                        result.add_warning(format!(
                            "Value {} at index {} below minimum {}",
                            value, i, min_val
                        ));
                    }
                }
                
                if let Some(max_val) = config.max_value {
                    if value > max_val {
                        out_of_range_count += 1;
                        result.add_warning(format!(
                            "Value {} at index {} above maximum {}",
                            value, i, max_val
                        ));
                    }
                }
            }
            
            // Check for zero values
            if value.abs() < config.tolerance {
                zero_count += 1;
            }
        }
        
        // Check for zero vector
        if config.check_zero_vectors && zero_count == features.dimension {
            result.add_warning("Feature vector is all zeros".to_string());
        }
        
        // Add metrics
        result.add_metric("nan_count".to_string(), nan_count as f64);
        result.add_metric("inf_count".to_string(), inf_count as f64);
        result.add_metric("out_of_range_count".to_string(), out_of_range_count as f64);
        result.add_metric("zero_ratio".to_string(), zero_count as f64 / features.dimension as f64);
        
        // Statistical metrics
        let mean = features.values.iter().sum::<f64>() / features.dimension as f64;
        let variance = features.values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / features.dimension as f64;
        let std_dev = variance.sqrt();
        
        result.add_metric("mean".to_string(), mean);
        result.add_metric("std_dev".to_string(), std_dev);
        result.add_metric("l2_norm".to_string(), features.norm());
    }
    
    /// Validate metadata
    fn validate_metadata(
        &self,
        features: &FeatureVector,
        result: &mut ValidationResult,
        _config: &ValidationConfig,
    ) {
        // Check required metadata fields
        if features.metadata.object_type.is_empty() {
            result.add_warning("Object type not specified in metadata".to_string());
        }
        
        if features.metadata.encoding_strategy.is_empty() {
            result.add_warning("Encoding strategy not specified in metadata".to_string());
        }
        
        // Check timestamp validity
        if let Some(timestamp) = features.metadata.timestamp {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            if timestamp > current_time {
                result.add_warning("Feature timestamp is in the future".to_string());
            }
            
            // Check if timestamp is too old (more than 1 year)
            if current_time - timestamp > 365 * 24 * 3600 {
                result.add_warning("Feature timestamp is more than 1 year old".to_string());
            }
        }
    }
    
    /// Validate mathematical properties
    fn validate_mathematical_properties(
        &self,
        features: &FeatureVector,
        result: &mut ValidationResult,
        config: &ValidationConfig,
    ) {
        // Check for numerical stability
        let max_abs = features.values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let min_abs = features.values.iter()
            .filter(|&&x| x.abs() > config.tolerance)
            .map(|x| x.abs())
            .fold(f64::INFINITY, f64::min);
        
        if max_abs.is_finite() && min_abs.is_finite() && min_abs > 0.0 {
            let condition_number = max_abs / min_abs;
            result.add_metric("condition_number".to_string(), condition_number);
            
            if condition_number > 1e12 {
                result.add_warning(format!(
                    "High condition number {:.2e} indicates potential numerical instability",
                    condition_number
                ));
            }
        }
        
        // Check for distribution properties
        let sorted_values: Vec<f64> = {
            let mut vals = features.values.clone();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals
        };
        
        // Check for outliers using IQR method
        let q1_idx = features.dimension / 4;
        let q3_idx = (3 * features.dimension) / 4;
        
        if q3_idx < sorted_values.len() && q1_idx < sorted_values.len() {
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx];
            let iqr = q3 - q1;
            
            if iqr > config.tolerance {
                let lower_bound = q1 - 1.5 * iqr;
                let upper_bound = q3 + 1.5 * iqr;
                
                let outlier_count = features.values.iter()
                    .filter(|&&x| x < lower_bound || x > upper_bound)
                    .count();
                
                result.add_metric("outlier_count".to_string(), outlier_count as f64);
                result.add_metric("outlier_ratio".to_string(), 
                    outlier_count as f64 / features.dimension as f64);
                
                if outlier_count as f64 / features.dimension as f64 > 0.1 {
                    result.add_warning(format!(
                        "High outlier ratio: {:.2%}",
                        outlier_count as f64 / features.dimension as f64
                    ));
                }
            }
        }
    }
    
    /// Compute overall validation score
    fn compute_validation_score(&self, result: &mut ValidationResult) {
        if !result.errors.is_empty() {
            result.set_score(0.0);
            return;
        }
        
        let mut score = 1.0;
        
        // Deduct points for warnings
        score -= result.warnings.len() as f64 * 0.1;
        
        // Deduct points for high outlier ratio
        if let Some(&outlier_ratio) = result.metrics.get("outlier_ratio") {
            score -= outlier_ratio * 0.3;
        }
        
        // Deduct points for high zero ratio
        if let Some(&zero_ratio) = result.metrics.get("zero_ratio") {
            if zero_ratio > 0.9 {
                score -= 0.2;
            }
        }
        
        // Deduct points for numerical instability
        if let Some(&condition_number) = result.metrics.get("condition_number") {
            if condition_number > 1e6 {
                score -= 0.1;
            }
        }
        
        result.set_score(score.max(0.0));
    }
}

/// Validator for sheaf objects
#[derive(Debug, Clone)]
pub struct SheafValidator;

impl DataValidator<Sheaf> for SheafValidator {
    fn validate(&self, sheaf: &Sheaf, config: &ValidationConfig) -> ValidationResult {
        let mut result = ValidationResult::success();
        
        // Check basic sheaf properties
        self.validate_basic_properties(sheaf, &mut result, config);
        
        // Check mathematical consistency
        if config.check_math_properties {
            self.validate_mathematical_consistency(sheaf, &mut result, config);
        }
        
        result
    }
    
    fn name(&self) -> &'static str {
        "SheafValidator"
    }
}

impl SheafValidator {
    fn validate_basic_properties(
        &self,
        sheaf: &Sheaf,
        result: &mut ValidationResult,
        _config: &ValidationConfig,
    ) {
        // Check rank
        if sheaf.rank() == 0 {
            result.add_warning("Sheaf has rank 0".to_string());
        }
        
        result.add_metric("rank".to_string(), sheaf.rank() as f64);
        
        // Check degree
        if let Some(degree) = sheaf.degree() {
            result.add_metric("degree".to_string(), degree);
        }
        
        // Check Euler characteristic
        match sheaf.euler_characteristic() {
            Ok(chi) => {
                result.add_metric("euler_characteristic".to_string(), chi as f64);
            }
            Err(e) => {
                result.add_warning(format!("Could not compute Euler characteristic: {:?}", e));
            }
        }
    }
    
    fn validate_mathematical_consistency(
        &self,
        sheaf: &Sheaf,
        result: &mut ValidationResult,
        config: &ValidationConfig,
    ) {
        // Check cohomology dimensions
        let mut total_cohomology = 0;
        for i in 0..10 {
            match sheaf.cohomology_dimension(i) {
                Ok(dim) => {
                    total_cohomology += dim;
                    result.add_metric(format!("cohomology_h{}", i), dim as f64);
                }
                Err(_) => break,
            }
        }
        
        result.add_metric("total_cohomology".to_string(), total_cohomology as f64);
        
        // Check Riemann-Roch relation (if applicable)
        if let (Some(degree), Ok(chi)) = (sheaf.degree(), sheaf.euler_characteristic()) {
            let expected_chi = degree * sheaf.rank() as f64; // Simplified
            let chi_diff = (chi as f64 - expected_chi).abs();
            
            if chi_diff > config.tolerance * 100.0 {
                result.add_warning(format!(
                    "Euler characteristic {} deviates from expected {} by {}",
                    chi, expected_chi, chi_diff
                ));
            }
        }
    }
}

/// Validator for bundle objects
#[derive(Debug, Clone)]
pub struct BundleValidator;

impl DataValidator<Bundle> for BundleValidator {
    fn validate(&self, bundle: &Bundle, config: &ValidationConfig) -> ValidationResult {
        let mut result = ValidationResult::success();
        
        // Check basic properties
        if bundle.rank() == 0 {
            result.add_warning("Bundle has rank 0".to_string());
        }
        
        result.add_metric("rank".to_string(), bundle.rank() as f64);
        
        // Check connection properties
        if let Some(connection) = bundle.connection() {
            match connection.trace() {
                Ok(trace) => {
                    result.add_metric("connection_trace".to_string(), trace);
                    if config.check_infinite && trace.is_infinite() {
                        result.add_error("Connection trace is infinite".to_string());
                    }
                }
                Err(_) => {
                    result.add_warning("Could not compute connection trace".to_string());
                }
            }
        }
        
        // Check curvature properties
        if let Some(curvature) = bundle.curvature() {
            match curvature.frobenius_norm() {
                Ok(norm) => {
                    result.add_metric("curvature_norm".to_string(), norm);
                    if norm < config.tolerance {
                        result.add_warning("Bundle appears to be flat (zero curvature)".to_string());
                    }
                }
                Err(_) => {
                    result.add_warning("Could not compute curvature norm".to_string());
                }
            }
        }
        
        result
    }
    
    fn name(&self) -> &'static str {
        "BundleValidator"
    }
}

/// Composite validator that can validate multiple types
#[derive(Debug, Clone)]
pub struct CompositeValidator {
    feature_validator: FeatureVectorValidator,
    sheaf_validator: SheafValidator,
    bundle_validator: BundleValidator,
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self {
            feature_validator: FeatureVectorValidator,
            sheaf_validator: SheafValidator,
            bundle_validator: BundleValidator,
        }
    }
}

impl CompositeValidator {
    /// Validate any supported object type
    pub fn validate_object(
        &self,
        object: &dyn std::any::Any,
        config: &ValidationConfig,
    ) -> ValidationResult {
        if let Some(features) = object.downcast_ref::<FeatureVector>() {
            self.feature_validator.validate(features, config)
        } else if let Some(sheaf) = object.downcast_ref::<Sheaf>() {
            self.sheaf_validator.validate(sheaf, config)
        } else if let Some(bundle) = object.downcast_ref::<Bundle>() {
            self.bundle_validator.validate(bundle, config)
        } else {
            ValidationResult::failure("Unsupported object type for validation".to_string())
        }
    }
    
    /// Batch validate multiple feature vectors
    pub fn batch_validate_features(
        &self,
        features: &[FeatureVector],
        config: &ValidationConfig,
    ) -> Vec<ValidationResult> {
        features.iter()
            .map(|fv| self.feature_validator.validate(fv, config))
            .collect()
    }
    
    /// Get validation summary for a batch of feature vectors
    pub fn validation_summary(
        &self,
        features: &[FeatureVector],
        config: &ValidationConfig,
    ) -> ValidationSummary {
        let results = self.batch_validate_features(features, config);
        
        let valid_count = results.iter().filter(|r| r.is_valid).count();
        let total_errors = results.iter().map(|r| r.errors.len()).sum();
        let total_warnings = results.iter().map(|r| r.warnings.len()).sum();
        let avg_score = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
        
        ValidationSummary {
            total_objects: results.len(),
            valid_objects: valid_count,
            total_errors,
            total_warnings,
            average_score: avg_score,
            validation_rate: valid_count as f64 / results.len() as f64,
        }
    }
}

/// Summary of validation results for a batch of objects
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationSummary {
    pub total_objects: usize,
    pub valid_objects: usize,
    pub total_errors: usize,
    pub total_warnings: usize,
    pub average_score: f64,
    pub validation_rate: f64,
}

impl ValidationSummary {
    /// Check if the overall validation is acceptable
    pub fn is_acceptable(&self, min_validation_rate: f64) -> bool {
        self.validation_rate >= min_validation_rate && self.total_errors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_feature_vector() -> FeatureVector {
        FeatureVector::new(
            vec![1.0, 2.0, 3.0, 4.0],
            "test".to_string(),
            "test".to_string(),
        )
    }
    
    #[test]
    fn test_validation_result_creation() {
        let success = ValidationResult::success();
        assert!(success.is_valid);
        assert_eq!(success.score, 1.0);
        assert!(success.errors.is_empty());
        
        let failure = ValidationResult::failure("test error".to_string());
        assert!(!failure.is_valid);
        assert_eq!(failure.score, 0.0);
        assert_eq!(failure.errors.len(), 1);
    }
    
    #[test]
    fn test_feature_vector_validation() {
        let validator = FeatureVectorValidator;
        let config = ValidationConfig::default();
        let features = create_test_feature_vector();
        
        let result = validator.validate(&features, &config);
        assert!(result.is_valid);
        assert!(result.score > 0.8);
    }
    
    #[test]
    fn test_invalid_feature_vector() {
        let validator = FeatureVectorValidator;
        let config = ValidationConfig::default();
        
        // Create feature vector with NaN
        let mut features = create_test_feature_vector();
        features.values[0] = f64::NAN;
        
        let result = validator.validate(&features, &config);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("NaN")));
    }
    
    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.check_nan);
        assert!(config.check_infinite);
        assert!(config.check_ranges);
        assert_eq!(config.tolerance, 1e-12);
    }
    
    #[test]
    fn test_composite_validator() {
        let validator = CompositeValidator::default();
        let config = ValidationConfig::default();
        let features = vec![create_test_feature_vector()]; 
        
        let summary = validator.validation_summary(&features, &config);
        assert_eq!(summary.total_objects, 1);
        assert_eq!(summary.valid_objects, 1);
        assert_eq!(summary.validation_rate, 1.0);
    }
}