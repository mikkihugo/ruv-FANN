//! Feature Encoding for Mathematical Objects
//! 
//! This module provides feature extraction and encoding strategies for
//! converting mathematical objects (G-bundles, D-modules, local systems)
//! into numerical feature vectors suitable for neural network processing.

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use std::collections::HashMap;

use crate::geometry::{Bundle, Sheaf, ModuliPoint};
use crate::topology::LocalSystem;
use crate::algebra::DModule;
use super::{NNResult, NNError};

/// Feature encoding strategies
#[derive(Debug, Clone, Copy)]
pub enum EncodingStrategy {
    /// Direct invariant encoding
    InvariantBased,
    /// Multi-scale feature hierarchy
    MultiScale,
    /// Spectral decomposition encoding
    Spectral,
    /// Topological signature encoding
    Topological,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Feature encoder for converting mathematical objects to vectors
pub struct FeatureEncoder {
    strategy: EncodingStrategy,
    feature_dim: usize,
    normalization: bool,
    cache: HashMap<u64, Array1<f64>>,
}

impl FeatureEncoder {
    /// Create a new feature encoder with specified strategy
    pub fn new(strategy: EncodingStrategy, feature_dim: usize) -> Self {
        Self {
            strategy,
            feature_dim,
            normalization: true,
            cache: HashMap::new(),
        }
    }
    
    /// Enable or disable feature normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalization = normalize;
        self
    }
    
    /// Encode a G-bundle into a feature vector
    pub fn encode_bundle(&mut self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        match self.strategy {
            EncodingStrategy::InvariantBased => self.encode_bundle_invariants(bundle),
            EncodingStrategy::MultiScale => self.encode_bundle_multiscale(bundle),
            EncodingStrategy::Spectral => self.encode_bundle_spectral(bundle),
            EncodingStrategy::Topological => self.encode_bundle_topological(bundle),
            EncodingStrategy::Hybrid => self.encode_bundle_hybrid(bundle),
        }
    }
    
    /// Encode a local system into a feature vector
    pub fn encode_local_system(&mut self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        match self.strategy {
            EncodingStrategy::InvariantBased => self.encode_system_invariants(system),
            EncodingStrategy::MultiScale => self.encode_system_multiscale(system),
            EncodingStrategy::Spectral => self.encode_system_spectral(system),
            EncodingStrategy::Topological => self.encode_system_topological(system),
            EncodingStrategy::Hybrid => self.encode_system_hybrid(system),
        }
    }
    
    /// Encode a D-module into a feature vector
    pub fn encode_d_module(&mut self, d_module: &DModule) -> NNResult<Array1<f64>> {
        match self.strategy {
            EncodingStrategy::InvariantBased => self.encode_dmodule_invariants(d_module),
            EncodingStrategy::MultiScale => self.encode_dmodule_multiscale(d_module),
            EncodingStrategy::Spectral => self.encode_dmodule_spectral(d_module),
            EncodingStrategy::Topological => self.encode_dmodule_topological(d_module),
            EncodingStrategy::Hybrid => self.encode_dmodule_hybrid(d_module),
        }
    }
    
    // Invariant-based encoding implementations
    
    fn encode_bundle_invariants(&self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let mut idx = 0;
        
        // Basic invariants
        features[idx] = bundle.rank() as f64;
        idx += 1;
        features[idx] = bundle.degree() as f64;
        idx += 1;
        
        // Chern classes
        for (i, chern) in bundle.chern_classes().iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = *chern;
        }
        idx += bundle.chern_classes().len();
        
        // Stability parameter
        if idx < self.feature_dim {
            features[idx] = bundle.slope();
            idx += 1;
        }
        
        // Automorphism group dimension
        if idx < self.feature_dim {
            features[idx] = bundle.automorphism_dim() as f64;
            idx += 1;
        }
        
        // Moduli coordinates (if available)
        if let Some(coords) = bundle.moduli_coordinates() {
            for (i, coord) in coords.iter().enumerate() {
                if idx + i >= self.feature_dim { break; }
                features[idx + i] = *coord;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_system_invariants(&self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let mut idx = 0;
        
        // Dimension of representation
        features[idx] = system.dimension() as f64;
        idx += 1;
        
        // Monodromy eigenvalues (sorted by magnitude)
        let eigenvalues = system.monodromy_eigenvalues();
        let mut sorted_eigenvalues: Vec<Complex64> = eigenvalues.to_vec();
        sorted_eigenvalues.sort_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap());
        
        // Store real and imaginary parts separately
        for eigenval in sorted_eigenvalues.iter() {
            if idx + 1 >= self.feature_dim { break; }
            features[idx] = eigenval.re;
            features[idx + 1] = eigenval.im;
            idx += 2;
        }
        
        // Traces of monodromy matrices
        for (i, trace) in system.monodromy_traces().iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = *trace;
        }
        idx += system.monodromy_traces().len();
        
        // Irreducibility indicator
        if idx < self.feature_dim {
            features[idx] = if system.is_irreducible() { 1.0 } else { 0.0 };
            idx += 1;
        }
        
        // Character variety dimension
        if idx < self.feature_dim {
            features[idx] = system.character_variety_dim() as f64;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_dmodule_invariants(&self, d_module: &DModule) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let mut idx = 0;
        
        // Rank
        features[idx] = d_module.rank() as f64;
        idx += 1;
        
        // Regular singularities count
        features[idx] = d_module.regular_singularities().len() as f64;
        idx += 1;
        
        // Hecke eigenvalues
        for (i, eigenval) in d_module.hecke_eigenvalues().iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = *eigenval;
        }
        idx += d_module.hecke_eigenvalues().len();
        
        // Characteristic variety dimension
        if idx < self.feature_dim {
            features[idx] = d_module.characteristic_variety_dim() as f64;
            idx += 1;
        }
        
        // Holonomic indicator
        if idx < self.feature_dim {
            features[idx] = if d_module.is_holonomic() { 1.0 } else { 0.0 };
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    // Multi-scale encoding implementations
    
    fn encode_bundle_multiscale(&self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let scales = [1, 2, 4, 8, 16]; // Different resolution scales
        let features_per_scale = self.feature_dim / scales.len();
        
        for (scale_idx, &scale) in scales.iter().enumerate() {
            let start_idx = scale_idx * features_per_scale;
            let end_idx = ((scale_idx + 1) * features_per_scale).min(self.feature_dim);
            
            // Extract features at this scale
            let scale_features = self.extract_bundle_features_at_scale(bundle, scale)?;
            
            // Copy to main feature vector
            for (i, &feat) in scale_features.iter().enumerate() {
                if start_idx + i >= end_idx { break; }
                features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_system_multiscale(&self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let scales = [1, 2, 4, 8, 16];
        let features_per_scale = self.feature_dim / scales.len();
        
        for (scale_idx, &scale) in scales.iter().enumerate() {
            let start_idx = scale_idx * features_per_scale;
            let end_idx = ((scale_idx + 1) * features_per_scale).min(self.feature_dim);
            
            let scale_features = self.extract_system_features_at_scale(system, scale)?;
            
            for (i, &feat) in scale_features.iter().enumerate() {
                if start_idx + i >= end_idx { break; }
                features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_dmodule_multiscale(&self, d_module: &DModule) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        let scales = [1, 2, 4, 8, 16];
        let features_per_scale = self.feature_dim / scales.len();
        
        for (scale_idx, &scale) in scales.iter().enumerate() {
            let start_idx = scale_idx * features_per_scale;
            let end_idx = ((scale_idx + 1) * features_per_scale).min(self.feature_dim);
            
            let scale_features = self.extract_dmodule_features_at_scale(d_module, scale)?;
            
            for (i, &feat) in scale_features.iter().enumerate() {
                if start_idx + i >= end_idx { break; }
                features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    // Spectral encoding implementations
    
    fn encode_bundle_spectral(&self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Compute spectral decomposition of transition matrices
        let spectral_data = bundle.spectral_decomposition()?;
        
        // Encode eigenvalues (sorted by magnitude)
        let mut idx = 0;
        for (i, eigenval) in spectral_data.eigenvalues.iter().enumerate() {
            if idx + 1 >= self.feature_dim { break; }
            features[idx] = eigenval.re;
            features[idx + 1] = eigenval.im;
            idx += 2;
        }
        
        // Encode spectral gaps
        for i in 1..spectral_data.eigenvalues.len() {
            if idx >= self.feature_dim { break; }
            let gap = (spectral_data.eigenvalues[i].norm() - 
                      spectral_data.eigenvalues[i-1].norm()).abs();
            features[idx] = gap;
            idx += 1;
        }
        
        // Encode principal eigenvector components
        if let Some(principal) = spectral_data.principal_eigenvector {
            for (i, &comp) in principal.iter().enumerate() {
                if idx + i >= self.feature_dim { break; }
                features[idx + i] = comp;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_system_spectral(&self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Get spectral data from monodromy representations
        let spectral_data = system.spectral_analysis()?;
        
        let mut idx = 0;
        
        // Encode eigenvalue spectrum
        for eigenval in spectral_data.spectrum.iter() {
            if idx + 1 >= self.feature_dim { break; }
            features[idx] = eigenval.re;
            features[idx + 1] = eigenval.im;
            idx += 2;
        }
        
        // Encode spectral density
        if idx < self.feature_dim {
            features[idx] = spectral_data.spectral_density;
            idx += 1;
        }
        
        // Encode spectral rigidity
        if idx < self.feature_dim {
            features[idx] = spectral_data.rigidity;
            idx += 1;
        }
        
        // Encode Fourier coefficients of characteristic polynomial
        for (i, &coeff) in spectral_data.fourier_coeffs.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = coeff;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_dmodule_spectral(&self, d_module: &DModule) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Spectral analysis of differential operators
        let spectral_data = d_module.operator_spectrum()?;
        
        let mut idx = 0;
        
        // Encode operator eigenvalues
        for eigenval in spectral_data.eigenvalues.iter() {
            if idx >= self.feature_dim { break; }
            features[idx] = *eigenval;
            idx += 1;
        }
        
        // Encode spectral zeta function values
        for (i, &zeta_val) in spectral_data.zeta_values.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = zeta_val;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    // Topological encoding implementations
    
    fn encode_bundle_topological(&self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Topological invariants
        let topo_data = bundle.topological_invariants()?;
        
        let mut idx = 0;
        
        // Betti numbers
        for (i, &betti) in topo_data.betti_numbers.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = betti as f64;
        }
        idx += topo_data.betti_numbers.len();
        
        // Persistent homology features
        for (dim, persistence) in topo_data.persistence_diagrams.iter() {
            if idx + 2 >= self.feature_dim { break; }
            features[idx] = *dim as f64;
            features[idx + 1] = persistence.total_persistence();
            idx += 2;
        }
        
        // Euler characteristic
        if idx < self.feature_dim {
            features[idx] = topo_data.euler_characteristic as f64;
            idx += 1;
        }
        
        // Topological entropy
        if idx < self.feature_dim {
            features[idx] = topo_data.topological_entropy;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_system_topological(&self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Topological properties of the representation
        let topo_data = system.topological_analysis()?;
        
        let mut idx = 0;
        
        // Representation variety dimension
        features[idx] = topo_data.variety_dimension as f64;
        idx += 1;
        
        // Number of connected components
        features[idx] = topo_data.connected_components as f64;
        idx += 1;
        
        // Fundamental group invariants
        for (i, &inv) in topo_data.pi1_invariants.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = inv;
        }
        idx += topo_data.pi1_invariants.len();
        
        // Cohomological dimensions
        for (i, &cohom_dim) in topo_data.cohomology_dims.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = cohom_dim as f64;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    fn encode_dmodule_topological(&self, d_module: &DModule) -> NNResult<Array1<f64>> {
        let mut features = Array1::zeros(self.feature_dim);
        
        // Topological properties of characteristic variety
        let topo_data = d_module.characteristic_topology()?;
        
        let mut idx = 0;
        
        // Singular support dimension
        features[idx] = topo_data.singular_support_dim as f64;
        idx += 1;
        
        // Number of irreducible components
        features[idx] = topo_data.irreducible_components as f64;
        idx += 1;
        
        // Milnor numbers at singularities
        for (i, &milnor) in topo_data.milnor_numbers.iter().enumerate() {
            if idx + i >= self.feature_dim { break; }
            features[idx + i] = milnor as f64;
        }
        
        if self.normalization {
            self.normalize_features(&mut features);
        }
        
        Ok(features)
    }
    
    // Hybrid encoding implementations
    
    fn encode_bundle_hybrid(&mut self, bundle: &Bundle) -> NNResult<Array1<f64>> {
        // Combine multiple encoding strategies
        let strategies = [
            EncodingStrategy::InvariantBased,
            EncodingStrategy::Spectral,
            EncodingStrategy::Topological,
        ];
        
        let features_per_strategy = self.feature_dim / strategies.len();
        let mut combined_features = Array1::zeros(self.feature_dim);
        
        for (idx, &strategy) in strategies.iter().enumerate() {
            let temp_encoder = FeatureEncoder::new(strategy, features_per_strategy);
            let partial_features = match strategy {
                EncodingStrategy::InvariantBased => temp_encoder.encode_bundle_invariants(bundle)?,
                EncodingStrategy::Spectral => temp_encoder.encode_bundle_spectral(bundle)?,
                EncodingStrategy::Topological => temp_encoder.encode_bundle_topological(bundle)?,
                _ => unreachable!(),
            };
            
            let start_idx = idx * features_per_strategy;
            for (i, &feat) in partial_features.iter().enumerate() {
                if start_idx + i >= self.feature_dim { break; }
                combined_features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut combined_features);
        }
        
        Ok(combined_features)
    }
    
    fn encode_system_hybrid(&mut self, system: &LocalSystem) -> NNResult<Array1<f64>> {
        let strategies = [
            EncodingStrategy::InvariantBased,
            EncodingStrategy::Spectral,
            EncodingStrategy::Topological,
        ];
        
        let features_per_strategy = self.feature_dim / strategies.len();
        let mut combined_features = Array1::zeros(self.feature_dim);
        
        for (idx, &strategy) in strategies.iter().enumerate() {
            let temp_encoder = FeatureEncoder::new(strategy, features_per_strategy);
            let partial_features = match strategy {
                EncodingStrategy::InvariantBased => temp_encoder.encode_system_invariants(system)?,
                EncodingStrategy::Spectral => temp_encoder.encode_system_spectral(system)?,
                EncodingStrategy::Topological => temp_encoder.encode_system_topological(system)?,
                _ => unreachable!(),
            };
            
            let start_idx = idx * features_per_strategy;
            for (i, &feat) in partial_features.iter().enumerate() {
                if start_idx + i >= self.feature_dim { break; }
                combined_features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut combined_features);
        }
        
        Ok(combined_features)
    }
    
    fn encode_dmodule_hybrid(&mut self, d_module: &DModule) -> NNResult<Array1<f64>> {
        let strategies = [
            EncodingStrategy::InvariantBased,
            EncodingStrategy::Spectral,
            EncodingStrategy::Topological,
        ];
        
        let features_per_strategy = self.feature_dim / strategies.len();
        let mut combined_features = Array1::zeros(self.feature_dim);
        
        for (idx, &strategy) in strategies.iter().enumerate() {
            let temp_encoder = FeatureEncoder::new(strategy, features_per_strategy);
            let partial_features = match strategy {
                EncodingStrategy::InvariantBased => temp_encoder.encode_dmodule_invariants(d_module)?,
                EncodingStrategy::Spectral => temp_encoder.encode_dmodule_spectral(d_module)?,
                EncodingStrategy::Topological => temp_encoder.encode_dmodule_topological(d_module)?,
                _ => unreachable!(),
            };
            
            let start_idx = idx * features_per_strategy;
            for (i, &feat) in partial_features.iter().enumerate() {
                if start_idx + i >= self.feature_dim { break; }
                combined_features[start_idx + i] = feat;
            }
        }
        
        if self.normalization {
            self.normalize_features(&mut combined_features);
        }
        
        Ok(combined_features)
    }
    
    // Helper methods
    
    fn extract_bundle_features_at_scale(&self, bundle: &Bundle, scale: usize) -> NNResult<Vec<f64>> {
        // Extract features at a specific resolution scale
        let mut scale_features = Vec::new();
        
        // Sample transition functions at this scale
        let samples = bundle.sample_transitions(scale)?;
        for sample in samples.iter() {
            scale_features.push(sample.norm());
            scale_features.push(sample.trace().re);
        }
        
        // Local moduli at this scale
        if let Some(local_coords) = bundle.local_moduli_at_scale(scale) {
            scale_features.extend_from_slice(&local_coords);
        }
        
        Ok(scale_features)
    }
    
    fn extract_system_features_at_scale(&self, system: &LocalSystem, scale: usize) -> NNResult<Vec<f64>> {
        let mut scale_features = Vec::new();
        
        // Sample monodromy at different loops at this scale
        let loop_samples = system.sample_monodromy_loops(scale)?;
        for matrix in loop_samples.iter() {
            scale_features.push(matrix.trace());
            scale_features.push(matrix.determinant());
        }
        
        Ok(scale_features)
    }
    
    fn extract_dmodule_features_at_scale(&self, d_module: &DModule, scale: usize) -> NNResult<Vec<f64>> {
        let mut scale_features = Vec::new();
        
        // Sample differential operators at this scale
        let op_samples = d_module.sample_operators(scale)?;
        for op in op_samples.iter() {
            scale_features.push(op.order() as f64);
            scale_features.push(op.leading_coefficient());
        }
        
        Ok(scale_features)
    }
    
    fn normalize_features(&self, features: &mut Array1<f64>) {
        // L2 normalization
        let norm = features.dot(features).sqrt();
        if norm > 1e-10 {
            features.mapv_inplace(|x| x / norm);
        }
        
        // Optional: apply tanh to bound features in [-1, 1]
        features.mapv_inplace(|x| x.tanh());
    }
    
    /// Batch encode multiple bundles
    pub fn batch_encode_bundles(&mut self, bundles: &[Bundle]) -> NNResult<Array2<f64>> {
        let n_samples = bundles.len();
        let mut batch_features = Array2::zeros((n_samples, self.feature_dim));
        
        for (i, bundle) in bundles.iter().enumerate() {
            let features = self.encode_bundle(bundle)?;
            batch_features.row_mut(i).assign(&features);
        }
        
        Ok(batch_features)
    }
    
    /// Batch encode multiple local systems
    pub fn batch_encode_systems(&mut self, systems: &[LocalSystem]) -> NNResult<Array2<f64>> {
        let n_samples = systems.len();
        let mut batch_features = Array2::zeros((n_samples, self.feature_dim));
        
        for (i, system) in systems.iter().enumerate() {
            let features = self.encode_local_system(system)?;
            batch_features.row_mut(i).assign(&features);
        }
        
        Ok(batch_features)
    }
    
    /// Clear the feature cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Get the feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
    
    /// Get the current encoding strategy
    pub fn strategy(&self) -> EncodingStrategy {
        self.strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_encoder_creation() {
        let encoder = FeatureEncoder::new(EncodingStrategy::InvariantBased, 128);
        assert_eq!(encoder.feature_dim(), 128);
        assert!(matches!(encoder.strategy(), EncodingStrategy::InvariantBased));
    }
    
    // Additional tests would require mock implementations of Bundle, LocalSystem, etc.
}