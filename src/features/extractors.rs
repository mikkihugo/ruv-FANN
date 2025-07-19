//! Feature extractors for mathematical objects in the geometric Langlands framework
//!
//! This module provides specialized extractors for converting various mathematical
//! objects into numerical feature representations.

use crate::core::prelude::*;
use crate::features::{FeatureVector, FeatureResult, FeatureError};
use std::collections::HashMap;

/// Configuration for feature extraction
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractorConfig {
    /// Target feature dimension
    pub dimension: usize,
    /// Include topological features
    pub include_topology: bool,
    /// Include geometric features
    pub include_geometry: bool,
    /// Include algebraic features
    pub include_algebra: bool,
    /// Precision for numerical computations
    pub precision: f64,
    /// Maximum complexity for extracted features
    pub max_complexity: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            include_topology: true,
            include_geometry: true,
            include_algebra: true,
            precision: 1e-12,
            max_complexity: 1000,
        }
    }
}

/// Main feature extractor trait
pub trait FeatureExtractor<T> {
    /// Extract features from a mathematical object
    fn extract(&self, object: &T, config: &ExtractorConfig) -> FeatureResult<FeatureVector>;
    
    /// Get the expected feature dimension for this extractor
    fn feature_dimension(&self, config: &ExtractorConfig) -> usize;
    
    /// Get feature labels for interpretability
    fn feature_labels(&self) -> Vec<String>;
}

/// Extractor for sheaf objects
#[derive(Debug, Clone)]
pub struct SheafExtractor {
    /// Include cohomology features
    pub include_cohomology: bool,
    /// Include stalk features
    pub include_stalks: bool,
    /// Maximum cohomology degree
    pub max_cohomology_degree: usize,
}

impl Default for SheafExtractor {
    fn default() -> Self {
        Self {
            include_cohomology: true,
            include_stalks: true,
            max_cohomology_degree: 10,
        }
    }
}

impl FeatureExtractor<Sheaf> for SheafExtractor {
    fn extract(&self, sheaf: &Sheaf, config: &ExtractorConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        // Extract basic sheaf properties
        self.extract_basic_properties(sheaf, &mut features, &mut labels)?;
        
        // Extract cohomology features if enabled
        if self.include_cohomology && config.include_topology {
            self.extract_cohomology_features(sheaf, &mut features, &mut labels, config)?;
        }
        
        // Extract stalk features if enabled
        if self.include_stalks && config.include_geometry {
            self.extract_stalk_features(sheaf, &mut features, &mut labels, config)?;
        }
        
        // Pad or truncate to target dimension
        self.adjust_dimension(&mut features, config.dimension);
        
        Ok(FeatureVector::new(
            features,
            "Sheaf".to_string(),
            "SheafExtractor".to_string(),
        ).with_labels(labels))
    }
    
    fn feature_dimension(&self, config: &ExtractorConfig) -> usize {
        config.dimension
    }
    
    fn feature_labels(&self) -> Vec<String> {
        vec![
            "rank".to_string(),
            "degree".to_string(),
            "euler_characteristic".to_string(),
            "chern_class_1".to_string(),
            "chern_class_2".to_string(),
        ]
    }
}

impl SheafExtractor {
    fn extract_basic_properties(
        &self,
        sheaf: &Sheaf,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
    ) -> FeatureResult<()> {
        // Extract rank
        features.push(sheaf.rank() as f64);
        labels.push("rank".to_string());
        
        // Extract degree (if available)
        if let Some(degree) = sheaf.degree() {
            features.push(degree);
            labels.push("degree".to_string());
        } else {
            features.push(0.0);
            labels.push("degree_default".to_string());
        }
        
        // Extract Euler characteristic
        match sheaf.euler_characteristic() {
            Ok(chi) => {
                features.push(chi as f64);
                labels.push("euler_characteristic".to_string());
            }
            Err(_) => {
                features.push(0.0);
                labels.push("euler_characteristic_default".to_string());
            }
        }
        
        Ok(())
    }
    
    fn extract_cohomology_features(
        &self,
        sheaf: &Sheaf,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        for degree in 0..=self.max_cohomology_degree.min(config.max_complexity) {
            match sheaf.cohomology_dimension(degree) {
                Ok(dim) => {
                    features.push(dim as f64);
                    labels.push(format!("cohomology_h{}", degree));
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push(format!("cohomology_h{}_default", degree));
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_stalk_features(
        &self,
        sheaf: &Sheaf,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        // Extract stalk dimensions at key points
        let key_points = sheaf.key_points();
        
        for (i, point) in key_points.iter().enumerate().take(10) { // Limit to 10 key points
            match sheaf.stalk_dimension(point) {
                Ok(dim) => {
                    features.push(dim as f64);
                    labels.push(format!("stalk_dim_{}", i));
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push(format!("stalk_dim_{}_default", i));
                }
            }
        }
        
        Ok(())
    }
    
    fn adjust_dimension(&self, features: &mut Vec<f64>, target_dim: usize) {
        if features.len() < target_dim {
            // Pad with zeros
            features.resize(target_dim, 0.0);
        } else if features.len() > target_dim {
            // Truncate
            features.truncate(target_dim);
        }
    }
}

/// Extractor for bundle objects
#[derive(Debug, Clone)]
pub struct BundleExtractor {
    /// Include connection features
    pub include_connection: bool,
    /// Include curvature features
    pub include_curvature: bool,
    /// Include characteristic classes
    pub include_characteristic_classes: bool,
}

impl Default for BundleExtractor {
    fn default() -> Self {
        Self {
            include_connection: true,
            include_curvature: true,
            include_characteristic_classes: true,
        }
    }
}

impl FeatureExtractor<Bundle> for BundleExtractor {
    fn extract(&self, bundle: &Bundle, config: &ExtractorConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        // Extract basic bundle properties
        self.extract_basic_properties(bundle, &mut features, &mut labels)?;
        
        // Extract connection features if enabled
        if self.include_connection && config.include_geometry {
            self.extract_connection_features(bundle, &mut features, &mut labels, config)?;
        }
        
        // Extract curvature features if enabled
        if self.include_curvature && config.include_geometry {
            self.extract_curvature_features(bundle, &mut features, &mut labels, config)?;
        }
        
        // Extract characteristic classes if enabled
        if self.include_characteristic_classes && config.include_topology {
            self.extract_characteristic_classes(bundle, &mut features, &mut labels, config)?;
        }
        
        // Adjust to target dimension
        self.adjust_dimension(&mut features, config.dimension);
        
        Ok(FeatureVector::new(
            features,
            "Bundle".to_string(),
            "BundleExtractor".to_string(),
        ).with_labels(labels))
    }
    
    fn feature_dimension(&self, config: &ExtractorConfig) -> usize {
        config.dimension
    }
    
    fn feature_labels(&self) -> Vec<String> {
        vec![
            "rank".to_string(),
            "degree".to_string(),
            "first_chern_class".to_string(),
            "second_chern_class".to_string(),
        ]
    }
}

impl BundleExtractor {
    fn extract_basic_properties(
        &self,
        bundle: &Bundle,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
    ) -> FeatureResult<()> {
        // Extract rank
        features.push(bundle.rank() as f64);
        labels.push("rank".to_string());
        
        // Extract degree
        if let Some(degree) = bundle.degree() {
            features.push(degree);
            labels.push("degree".to_string());
        } else {
            features.push(0.0);
            labels.push("degree_default".to_string());
        }
        
        Ok(())
    }
    
    fn extract_connection_features(
        &self,
        bundle: &Bundle,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        if let Some(connection) = bundle.connection() {
            // Extract connection matrix trace
            match connection.trace() {
                Ok(trace) => {
                    features.push(trace);
                    labels.push("connection_trace".to_string());
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push("connection_trace_default".to_string());
                }
            }
            
            // Extract connection determinant
            match connection.determinant() {
                Ok(det) => {
                    features.push(det);
                    labels.push("connection_det".to_string());
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push("connection_det_default".to_string());
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_curvature_features(
        &self,
        bundle: &Bundle,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        if let Some(curvature) = bundle.curvature() {
            // Extract curvature scalar
            match curvature.scalar() {
                Ok(scalar) => {
                    features.push(scalar);
                    labels.push("curvature_scalar".to_string());
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push("curvature_scalar_default".to_string());
                }
            }
            
            // Extract curvature norm
            match curvature.frobenius_norm() {
                Ok(norm) => {
                    features.push(norm);
                    labels.push("curvature_norm".to_string());
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push("curvature_norm_default".to_string());
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_characteristic_classes(
        &self,
        bundle: &Bundle,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        // Extract Chern classes
        for i in 1..=bundle.rank().min(4) {
            match bundle.chern_class(i) {
                Ok(chern) => {
                    features.push(chern);
                    labels.push(format!("chern_class_{}", i));
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push(format!("chern_class_{}_default", i));
                }
            }
        }
        
        // Extract Pontryagin classes for real bundles
        if bundle.is_real() {
            for i in 1..=(bundle.rank() / 2).min(2) {
                match bundle.pontryagin_class(i) {
                    Ok(pont) => {
                        features.push(pont);
                        labels.push(format!("pontryagin_class_{}", i));
                    }
                    Err(_) => {
                        features.push(0.0);
                        labels.push(format!("pontryagin_class_{}_default", i));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn adjust_dimension(&self, features: &mut Vec<f64>, target_dim: usize) {
        if features.len() < target_dim {
            features.resize(target_dim, 0.0);
        } else if features.len() > target_dim {
            features.truncate(target_dim);
        }
    }
}

/// Extractor for representation objects
#[derive(Debug, Clone)]
pub struct RepresentationExtractor {
    /// Include character features
    pub include_characters: bool,
    /// Include weight features
    pub include_weights: bool,
    /// Maximum weight multiplicity to consider
    pub max_weight_multiplicity: usize,
}

impl Default for RepresentationExtractor {
    fn default() -> Self {
        Self {
            include_characters: true,
            include_weights: true,
            max_weight_multiplicity: 100,
        }
    }
}

impl FeatureExtractor<Representation> for RepresentationExtractor {
    fn extract(&self, rep: &Representation, config: &ExtractorConfig) -> FeatureResult<FeatureVector> {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        // Extract basic representation properties
        self.extract_basic_properties(rep, &mut features, &mut labels)?;
        
        // Extract character features if enabled
        if self.include_characters && config.include_algebra {
            self.extract_character_features(rep, &mut features, &mut labels, config)?;
        }
        
        // Extract weight features if enabled
        if self.include_weights && config.include_algebra {
            self.extract_weight_features(rep, &mut features, &mut labels, config)?;
        }
        
        // Adjust to target dimension
        self.adjust_dimension(&mut features, config.dimension);
        
        Ok(FeatureVector::new(
            features,
            "Representation".to_string(),
            "RepresentationExtractor".to_string(),
        ).with_labels(labels))
    }
    
    fn feature_dimension(&self, config: &ExtractorConfig) -> usize {
        config.dimension
    }
    
    fn feature_labels(&self) -> Vec<String> {
        vec![
            "dimension".to_string(),
            "character_norm".to_string(),
            "highest_weight_norm".to_string(),
        ]
    }
}

impl RepresentationExtractor {
    fn extract_basic_properties(
        &self,
        rep: &Representation,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
    ) -> FeatureResult<()> {
        // Extract dimension
        features.push(rep.dimension() as f64);
        labels.push("dimension".to_string());
        
        // Extract degree (if applicable)
        if let Some(degree) = rep.degree() {
            features.push(degree);
            labels.push("degree".to_string());
        }
        
        Ok(())
    }
    
    fn extract_character_features(
        &self,
        rep: &Representation,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        if let Some(character) = rep.character() {
            // Extract character at identity
            features.push(character.at_identity());
            labels.push("character_identity".to_string());
            
            // Extract character norm
            match character.norm() {
                Ok(norm) => {
                    features.push(norm);
                    labels.push("character_norm".to_string());
                }
                Err(_) => {
                    features.push(0.0);
                    labels.push("character_norm_default".to_string());
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_weight_features(
        &self,
        rep: &Representation,
        features: &mut Vec<f64>,
        labels: &mut Vec<String>,
        _config: &ExtractorConfig,
    ) -> FeatureResult<()> {
        let weights = rep.weights();
        
        // Extract highest weight
        if let Some(highest_weight) = weights.highest_weight() {
            features.push(highest_weight.norm());
            labels.push("highest_weight_norm".to_string());
            
            // Extract weight coordinates (first few components)
            for (i, coord) in highest_weight.coordinates().iter().enumerate().take(5) {
                features.push(*coord);
                labels.push(format!("highest_weight_coord_{}", i));
            }
        }
        
        // Extract weight multiplicities
        let weight_multiplicities = weights.multiplicities();
        for (i, mult) in weight_multiplicities.iter().enumerate().take(10) {
            features.push(*mult as f64);
            labels.push(format!("weight_multiplicity_{}", i));
        }
        
        Ok(())
    }
    
    fn adjust_dimension(&self, features: &mut Vec<f64>, target_dim: usize) {
        if features.len() < target_dim {
            features.resize(target_dim, 0.0);
        } else if features.len() > target_dim {
            features.truncate(target_dim);
        }
    }
}

/// Multi-object feature extractor that can handle different mathematical objects
#[derive(Debug, Clone)]
pub struct UniversalExtractor {
    sheaf_extractor: SheafExtractor,
    bundle_extractor: BundleExtractor,
    representation_extractor: RepresentationExtractor,
}

impl Default for UniversalExtractor {
    fn default() -> Self {
        Self {
            sheaf_extractor: SheafExtractor::default(),
            bundle_extractor: BundleExtractor::default(),
            representation_extractor: RepresentationExtractor::default(),
        }
    }
}

impl UniversalExtractor {
    /// Extract features from any supported mathematical object
    pub fn extract_from_object(
        &self,
        object: &dyn std::any::Any,
        config: &ExtractorConfig,
    ) -> FeatureResult<FeatureVector> {
        if let Some(sheaf) = object.downcast_ref::<Sheaf>() {
            self.sheaf_extractor.extract(sheaf, config)
        } else if let Some(bundle) = object.downcast_ref::<Bundle>() {
            self.bundle_extractor.extract(bundle, config)
        } else if let Some(rep) = object.downcast_ref::<Representation>() {
            self.representation_extractor.extract(rep, config)
        } else {
            Err(FeatureError::ExtractionFailed {
                message: "Unsupported object type for feature extraction".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extractor_config_default() {
        let config = ExtractorConfig::default();
        assert_eq!(config.dimension, 512);
        assert!(config.include_topology);
        assert!(config.include_geometry);
        assert!(config.include_algebra);
    }
    
    #[test]
    fn test_sheaf_extractor_creation() {
        let extractor = SheafExtractor::default();
        assert!(extractor.include_cohomology);
        assert!(extractor.include_stalks);
        assert_eq!(extractor.max_cohomology_degree, 10);
    }
    
    #[test]
    fn test_bundle_extractor_creation() {
        let extractor = BundleExtractor::default();
        assert!(extractor.include_connection);
        assert!(extractor.include_curvature);
        assert!(extractor.include_characteristic_classes);
    }
    
    #[test]
    fn test_representation_extractor_creation() {
        let extractor = RepresentationExtractor::default();
        assert!(extractor.include_characters);
        assert!(extractor.include_weights);
        assert_eq!(extractor.max_weight_multiplicity, 100);
    }
}