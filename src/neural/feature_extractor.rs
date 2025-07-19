//! Feature extraction for mathematical objects in the Langlands program
//!
//! This module provides sophisticated feature extraction techniques for converting
//! abstract mathematical objects into dense vector representations suitable for
//! neural network processing.

use crate::core::*;
use crate::neural::{NeuralError, NeuralFeature, NeuralResult, FeatureCache};
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Trait for extracting features from mathematical objects
pub trait FeatureExtractor<T: Float, M: MathObject>: Send + Sync {
    /// Extract features from a mathematical object
    fn extract(&self, obj: &M) -> NeuralResult<Vec<T>>;
    
    /// Get the dimension of extracted features
    fn feature_dimension(&self) -> usize;
    
    /// Extract features from multiple objects in parallel
    fn extract_batch(&self, objects: &[M]) -> NeuralResult<Vec<Vec<T>>> {
        use rayon::prelude::*;
        
        objects
            .par_iter()
            .map(|obj| self.extract(obj))
            .collect()
    }
    
    /// Extract and cache features
    fn extract_cached(&self, obj: &M, cache: &Arc<Mutex<FeatureCache<T>>>) -> NeuralResult<Vec<T>> {
        let cache_key = format!("{:?}", obj.id());
        
        // Try to get from cache first
        {
            let cache_guard = cache.lock().unwrap();
            if let Some(features) = cache_guard.get(&cache_key) {
                return Ok(features.clone());
            }
        }
        
        // Extract features if not cached
        let features = self.extract(obj)?;
        
        // Store in cache
        {
            let mut cache_guard = cache.lock().unwrap();
            cache_guard.insert(cache_key, features.clone());
        }
        
        Ok(features)
    }
}

/// Feature extractor for sheaves and coherent sheaves
pub struct SheafFeatureExtractor<T: Float> {
    feature_dim: usize,
    include_cohomology: bool,
    include_chern_classes: bool,
}

impl<T: Float> SheafFeatureExtractor<T> {
    pub fn new(feature_dim: usize) -> Self {
        Self {
            feature_dim,
            include_cohomology: true,
            include_chern_classes: true,
        }
    }
    
    pub fn with_cohomology(mut self, include: bool) -> Self {
        self.include_cohomology = include;
        self
    }
    
    pub fn with_chern_classes(mut self, include: bool) -> Self {
        self.include_chern_classes = include;
        self
    }
}

impl<T: Float> FeatureExtractor<T, crate::core::sheaf::SimpleSheaf<T>> for SheafFeatureExtractor<T> {
    fn extract(&self, sheaf: &crate::core::sheaf::SimpleSheaf<T>) -> NeuralResult<Vec<T>> {
        let mut features = Vec::with_capacity(self.feature_dim);
        
        // Basic sheaf properties
        features.push(T::from(sheaf.rank()).unwrap());
        features.push(T::from(sheaf.dimension()).unwrap());
        
        // Cohomology features
        if self.include_cohomology {
            let cohomology_dims = sheaf.cohomology_dimensions()
                .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Cohomology extraction failed: {}", e)))?;
            
            for dim in cohomology_dims {
                features.push(T::from(dim).unwrap());
            }
        }
        
        // Chern classes (if applicable)
        if self.include_chern_classes && sheaf.is_vector_bundle() {
            let chern_classes = sheaf.chern_classes()
                .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Chern class extraction failed: {}", e)))?;
            
            for class in chern_classes {
                features.push(class);
            }
        }
        
        // Stability information
        if let Ok(stability) = sheaf.compute_stability() {
            features.push(stability.slope);
            features.push(if stability.is_stable { T::one() } else { T::zero() });
            features.push(if stability.is_semistable { T::one() } else { T::zero() });
        }
        
        // Local system features (if applicable)
        // Note: simplified for now - in full implementation would handle local systems
        // if let Some(local_system) = sheaf.as_local_system() {
        //     let monodromy_features = self.extract_monodromy_features(local_system)?;
        //     features.extend(monodromy_features);
        // }
        
        // Pad or truncate to desired dimension
        features.resize(self.feature_dim, T::zero());
        
        Ok(features)
    }
    
    fn feature_dimension(&self) -> usize {
        self.feature_dim
    }
}

impl<T: Float> SheafFeatureExtractor<T> {
    fn extract_monodromy_features(&self, local_system: &LocalSystem<T>) -> NeuralResult<Vec<T>> {
        let mut features = Vec::new();
        
        // Monodromy group properties
        let monodromy_group = local_system.monodromy_group()
            .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Monodromy group extraction failed: {}", e)))?;
        
        features.push(T::from(monodromy_group.dimension()).unwrap());
        features.push(if monodromy_group.is_reductive() { T::one() } else { T::zero() });
        features.push(if monodromy_group.is_unipotent() { T::one() } else { T::zero() });
        
        // Character variety features
        if let Ok(char_variety) = local_system.character_variety() {
            features.push(T::from(char_variety.dimension()).unwrap());
            features.push(T::from(char_variety.num_components()).unwrap());
        }
        
        Ok(features)
    }
}

/// Feature extractor for vector bundles and principal bundles
pub struct BundleFeatureExtractor<T: Float> {
    feature_dim: usize,
    include_moduli_coords: bool,
}

impl<T: Float> BundleFeatureExtractor<T> {
    pub fn new(feature_dim: usize) -> Self {
        Self {
            feature_dim,
            include_moduli_coords: true,
        }
    }
}

impl<T: Float> FeatureExtractor<T, crate::core::bundle::SimpleVectorBundle<T>> for BundleFeatureExtractor<T> {
    fn extract(&self, bundle: &crate::core::bundle::SimpleVectorBundle<T>) -> NeuralResult<Vec<T>> {
        let mut features = Vec::with_capacity(self.feature_dim);
        
        // Basic bundle properties
        features.push(T::from(bundle.rank()).unwrap());
        features.push(T::from(bundle.degree()).unwrap());
        features.push(bundle.slope());
        
        // Stability properties
        let stability = bundle.compute_stability()
            .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Stability computation failed: {}", e)))?;
        
        features.push(if stability.is_stable { T::one() } else { T::zero() });
        features.push(if stability.is_semistable { T::one() } else { T::zero() });
        features.push(stability.harder_narasimhan_filtration.len() as f64)
            .map(|x| T::from(x).unwrap())
            .unwrap();
        
        // Moduli space coordinates (if requested)
        if self.include_moduli_coords {
            let moduli_coords = bundle.moduli_coordinates()
                .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Moduli coordinates failed: {}", e)))?;
            features.extend(moduli_coords);
        }
        
        // Higgs bundle features (if applicable)
        if let Some(higgs_field) = bundle.higgs_field() {
            let higgs_features = self.extract_higgs_features(higgs_field)?;
            features.extend(higgs_features);
        }
        
        // Pad or truncate to desired dimension
        features.resize(self.feature_dim, T::zero());
        
        Ok(features)
    }
    
    fn feature_dimension(&self) -> usize {
        self.feature_dim
    }
}

impl<T: Float> BundleFeatureExtractor<T> {
    fn extract_higgs_features(&self, higgs_field: &HiggsField<T>) -> NeuralResult<Vec<T>> {
        let mut features = Vec::new();
        
        // Higgs field properties
        features.push(higgs_field.spectral_norm());
        features.push(if higgs_field.is_stable() { T::one() } else { T::zero() });
        
        // Spectral curve features
        if let Ok(spectral_curve) = higgs_field.spectral_curve() {
            features.push(T::from(spectral_curve.genus()).unwrap());
            features.push(T::from(spectral_curve.degree()).unwrap());
        }
        
        Ok(features)
    }
}

/// Feature extractor for representations
pub struct RepresentationFeatureExtractor<T: Float> {
    feature_dim: usize,
    group_type: String,
}

impl<T: Float> RepresentationFeatureExtractor<T> {
    pub fn new(feature_dim: usize, group_type: String) -> Self {
        Self {
            feature_dim,
            group_type,
        }
    }
}

impl<T: Float> FeatureExtractor<T, crate::core::representation::SimpleRepresentation<T>> for RepresentationFeatureExtractor<T> {
    fn extract(&self, rep: &crate::core::representation::SimpleRepresentation<T>) -> NeuralResult<Vec<T>> {
        let mut features = Vec::with_capacity(self.feature_dim);
        
        // Basic representation properties
        features.push(T::from(rep.dimension()).unwrap());
        features.push(if rep.is_irreducible() { T::one() } else { T::zero() });
        features.push(if rep.is_unitary() { T::one() } else { T::zero() });
        
        // Character features
        let character = rep.character()
            .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Character computation failed: {}", e)))?;
        
        // Sample character values at standard elements
        let sample_elements = rep.group().sample_elements(16)?;
        for element in sample_elements {
            let char_value = character.evaluate(&element)?;
            features.push(char_value.re);
            features.push(char_value.im);
        }
        
        // Highest weight features (for reductive groups)
        if rep.group().is_reductive() {
            let highest_weight = rep.highest_weight()
                .map_err(|e| NeuralError::FeatureExtractionFailed(format!("Highest weight failed: {}", e)))?;
            features.extend(highest_weight);
        }
        
        // L-function coefficients (if available)
        if let Ok(l_function) = rep.l_function() {
            let coefficients = l_function.fourier_coefficients(32)?;
            for coeff in coefficients {
                features.push(coeff.re);
                features.push(coeff.im);
            }
        }
        
        // Pad or truncate to desired dimension
        features.resize(self.feature_dim, T::zero());
        
        Ok(features)
    }
    
    fn feature_dimension(&self) -> usize {
        self.feature_dim
    }
}

/// Composite feature extractor that combines multiple extractors
pub struct CompositeFeatureExtractor<T: Float> {
    extractors: Vec<Box<dyn FeatureExtractor<T, dyn MathObject> + Send + Sync>>,
    total_dim: usize,
}

impl<T: Float> CompositeFeatureExtractor<T> {
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
            total_dim: 0,
        }
    }
    
    pub fn add_extractor<E, M>(mut self, extractor: E) -> Self 
    where
        E: FeatureExtractor<T, M> + Send + Sync + 'static,
        M: MathObject + 'static,
    {
        self.total_dim += extractor.feature_dimension();
        // Note: This is a simplified version. In practice, we'd need more sophisticated
        // type erasure to handle different mathematical object types
        self
    }
}

/// Feature normalization utilities
pub struct FeatureNormalizer<T: Float> {
    means: Vec<T>,
    std_devs: Vec<T>,
}

impl<T: Float> FeatureNormalizer<T> {
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            std_devs: Vec::new(),
        }
    }
    
    /// Fit the normalizer to a dataset
    pub fn fit(&mut self, features: &[Vec<T>]) -> NeuralResult<()> {
        if features.is_empty() {
            return Err(NeuralError::FeatureExtractionFailed("Empty feature dataset".to_string()));
        }
        
        let dim = features[0].len();
        self.means = vec![T::zero(); dim];
        self.std_devs = vec![T::one(); dim];
        
        // Compute means
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                self.means[i] = self.means[i] + val;
            }
        }
        
        let n = T::from(features.len()).unwrap();
        for mean in &mut self.means {
            *mean = *mean / n;
        }
        
        // Compute standard deviations
        for feature_vec in features {
            for (i, &val) in feature_vec.iter().enumerate() {
                let diff = val - self.means[i];
                self.std_devs[i] = self.std_devs[i] + diff * diff;
            }
        }
        
        for std_dev in &mut self.std_devs {
            *std_dev = (*std_dev / n).sqrt();
            // Avoid division by zero
            if *std_dev < T::from(1e-8).unwrap() {
                *std_dev = T::one();
            }
        }
        
        Ok(())
    }
    
    /// Normalize a feature vector
    pub fn normalize(&self, features: &mut Vec<T>) -> NeuralResult<()> {
        if features.len() != self.means.len() {
            return Err(NeuralError::FeatureExtractionFailed(
                format!("Feature dimension mismatch: expected {}, got {}", 
                       self.means.len(), features.len())
            ));
        }
        
        for (i, feature) in features.iter_mut().enumerate() {
            *feature = (*feature - self.means[i]) / self.std_devs[i];
        }
        
        Ok(())
    }
    
    /// Denormalize a feature vector
    pub fn denormalize(&self, features: &mut Vec<T>) -> NeuralResult<()> {
        if features.len() != self.means.len() {
            return Err(NeuralError::FeatureExtractionFailed(
                format!("Feature dimension mismatch: expected {}, got {}", 
                       self.means.len(), features.len())
            ));
        }
        
        for (i, feature) in features.iter_mut().enumerate() {
            *feature = *feature * self.std_devs[i] + self.means[i];
        }
        
        Ok(())
    }
}