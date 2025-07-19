//! Feature extraction for mathematical objects in the Geometric Langlands framework

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, ArrayView1};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rayon::prelude::*;

use super::{DataError, Result, FeatureConfig};

/// Trait for objects that can be converted to feature vectors
pub trait Featurizable: Send + Sync {
    /// Extract features from the object
    fn extract_features(&self, config: &FeatureConfig) -> Result<FeatureVector>;
    
    /// Get unique identifier for caching
    fn cache_key(&self) -> String;
}

/// High-dimensional feature vector representing mathematical objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Unique identifier
    pub id: String,
    
    /// Geometric features (positions, curvatures, etc.)
    pub geometric: Array1<f64>,
    
    /// Algebraic features (invariants, representations, etc.)
    pub algebraic: Array1<f64>,
    
    /// Spectral features (eigenvalues, traces, etc.)
    pub spectral: Option<Array1<Complex64>>,
    
    /// Topological features (Betti numbers, characteristic classes, etc.)
    pub topological: Option<Array1<i32>>,
    
    /// Metadata for tracking
    pub metadata: HashMap<String, String>,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(id: String, geometric_dim: usize, algebraic_dim: usize) -> Self {
        Self {
            id,
            geometric: Array1::zeros(geometric_dim),
            algebraic: Array1::zeros(algebraic_dim),
            spectral: None,
            topological: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Concatenate all features into a single vector
    pub fn flatten(&self) -> Array1<f64> {
        let mut result = Vec::new();
        
        // Add geometric features
        result.extend_from_slice(self.geometric.as_slice().unwrap());
        
        // Add algebraic features
        result.extend_from_slice(self.algebraic.as_slice().unwrap());
        
        // Add spectral features (real and imaginary parts)
        if let Some(spectral) = &self.spectral {
            for c in spectral.iter() {
                result.push(c.re);
                result.push(c.im);
            }
        }
        
        // Add topological features as floats
        if let Some(topo) = &self.topological {
            result.extend(topo.iter().map(|&x| x as f64));
        }
        
        Array1::from_vec(result)
    }
    
    /// Get total dimension
    pub fn dimension(&self) -> usize {
        let mut dim = self.geometric.len() + self.algebraic.len();
        
        if let Some(spectral) = &self.spectral {
            dim += spectral.len() * 2; // Complex numbers have 2 components
        }
        
        if let Some(topo) = &self.topological {
            dim += topo.len();
        }
        
        dim
    }
    
    /// Apply normalization
    pub fn normalize(&mut self) {
        // L2 normalization for geometric features
        let geo_norm = self.geometric.dot(&self.geometric).sqrt();
        if geo_norm > 1e-10 {
            self.geometric /= geo_norm;
        }
        
        // L2 normalization for algebraic features
        let alg_norm = self.algebraic.dot(&self.algebraic).sqrt();
        if alg_norm > 1e-10 {
            self.algebraic /= alg_norm;
        }
        
        // Spectral features: normalize by spectral radius
        if let Some(spectral) = &mut self.spectral {
            let max_abs = spectral.iter()
                .map(|c| c.norm())
                .fold(0.0, f64::max);
            
            if max_abs > 1e-10 {
                for c in spectral.iter_mut() {
                    *c /= max_abs;
                }
            }
        }
    }
}

/// Extractor for geometric features
pub struct GeometricFeatureExtractor {
    config: Arc<FeatureConfig>,
}

impl GeometricFeatureExtractor {
    pub fn new(config: Arc<FeatureConfig>) -> Self {
        Self { config }
    }
    
    /// Extract curvature features from a Riemann surface
    pub fn extract_curvature_features(&self, points: &Array2<f64>) -> Result<Array1<f64>> {
        if points.nrows() < 3 {
            return Err(DataError::FeatureExtractionError(
                "Not enough points for curvature calculation".to_string()
            ));
        }
        
        let mut features = Vec::with_capacity(self.config.geometric_dim);
        
        // Calculate discrete curvatures
        for i in 1..points.nrows() - 1 {
            let p1 = points.row(i - 1);
            let p2 = points.row(i);
            let p3 = points.row(i + 1);
            
            let curvature = self.discrete_curvature(p1, p2, p3)?;
            features.push(curvature);
        }
        
        // Pad or truncate to match dimension
        features.resize(self.config.geometric_dim, 0.0);
        
        Ok(Array1::from_vec(features))
    }
    
    /// Calculate discrete curvature at a point
    fn discrete_curvature(&self, p1: ArrayView1<f64>, p2: ArrayView1<f64>, p3: ArrayView1<f64>) 
        -> Result<f64> 
    {
        let v1 = &p2 - &p1;
        let v2 = &p3 - &p2;
        
        let cross_norm = (v1[0] * v2[1] - v1[1] * v2[0]).abs();
        let v1_norm = v1.dot(&v1).sqrt();
        let v2_norm = v2.dot(&v2).sqrt();
        
        if v1_norm < self.config.precision || v2_norm < self.config.precision {
            return Ok(0.0);
        }
        
        Ok(2.0 * cross_norm / (v1_norm * v2_norm * (v1_norm + v2_norm)))
    }
    
    /// Extract metric tensor features
    pub fn extract_metric_features(&self, metric: &DMatrix<f64>) -> Result<Array1<f64>> {
        let mut features = Vec::new();
        
        // Eigenvalues of the metric
        if let Ok(eigen) = metric.symmetric_eigen() {
            for eval in eigen.eigenvalues.iter().take(self.config.geometric_dim / 4) {
                features.push(*eval);
            }
        }
        
        // Trace and determinant
        features.push(metric.trace());
        features.push(metric.determinant());
        
        // Frobenius norm
        features.push(metric.norm_squared().sqrt());
        
        // Pad to required dimension
        features.resize(self.config.geometric_dim, 0.0);
        
        Ok(Array1::from_vec(features))
    }
}

/// Extractor for algebraic features
pub struct AlgebraicFeatureExtractor {
    config: Arc<FeatureConfig>,
}

impl AlgebraicFeatureExtractor {
    pub fn new(config: Arc<FeatureConfig>) -> Self {
        Self { config }
    }
    
    /// Extract polynomial invariants
    pub fn extract_polynomial_invariants(&self, coeffs: &[f64]) -> Result<Array1<f64>> {
        let mut features = Vec::with_capacity(self.config.algebraic_dim);
        
        // Compute various polynomial invariants
        for degree in 1..=self.config.max_degree.min(coeffs.len()) {
            let invariant = self.compute_invariant(coeffs, degree);
            features.push(invariant);
        }
        
        // Add discriminant-like features
        if coeffs.len() >= 3 {
            let disc = self.compute_discriminant(&coeffs[..3]);
            features.push(disc);
        }
        
        // Pad to required dimension
        features.resize(self.config.algebraic_dim, 0.0);
        
        Ok(Array1::from_vec(features))
    }
    
    /// Compute polynomial invariant of given degree
    fn compute_invariant(&self, coeffs: &[f64], degree: usize) -> f64 {
        coeffs.iter()
            .take(degree)
            .enumerate()
            .map(|(i, &c)| c.powi(i as i32 + 1))
            .sum()
    }
    
    /// Compute discriminant for quadratic-like terms
    fn compute_discriminant(&self, coeffs: &[f64]) -> f64 {
        if coeffs.len() < 3 {
            return 0.0;
        }
        let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);
        b * b - 4.0 * a * c
    }
    
    /// Extract representation-theoretic features
    pub fn extract_representation_features(&self, rep_matrix: &DMatrix<f64>) -> Result<Array1<f64>> {
        let mut features = Vec::new();
        
        // Character (trace)
        features.push(rep_matrix.trace());
        
        // Determinant
        features.push(rep_matrix.determinant());
        
        // Eigenvalue features
        if let Ok(eigen) = rep_matrix.eigenvalues() {
            for (i, eval) in eigen.iter().enumerate() {
                if i >= self.config.algebraic_dim / 4 {
                    break;
                }
                features.push(eval.re);
                features.push(eval.im);
            }
        }
        
        // Casimir invariants (simplified)
        let casimir = rep_matrix.transpose() * rep_matrix;
        features.push(casimir.trace());
        
        // Pad to required dimension
        features.resize(self.config.algebraic_dim, 0.0);
        
        Ok(Array1::from_vec(features))
    }
}

/// Extractor for spectral features
pub struct SpectralFeatureExtractor {
    config: Arc<FeatureConfig>,
}

impl SpectralFeatureExtractor {
    pub fn new(config: Arc<FeatureConfig>) -> Self {
        Self { config }
    }
    
    /// Extract spectral features from an operator
    pub fn extract_spectral_features(&self, operator: &DMatrix<f64>) -> Result<Array1<Complex64>> {
        let eigenvalues = operator.eigenvalues()
            .ok_or_else(|| DataError::FeatureExtractionError(
                "Failed to compute eigenvalues".to_string()
            ))?;
        
        let mut features: Vec<Complex64> = eigenvalues.iter()
            .take(self.config.geometric_dim / 2)
            .cloned()
            .collect();
        
        // Add spectral traces
        let traces = self.compute_spectral_traces(operator, 5)?;
        features.extend(traces);
        
        // Ensure consistent dimension
        features.resize(self.config.geometric_dim / 2, Complex64::new(0.0, 0.0));
        
        Ok(Array1::from_vec(features))
    }
    
    /// Compute spectral traces Tr(A^k) for k = 1, ..., max_power
    fn compute_spectral_traces(&self, operator: &DMatrix<f64>, max_power: usize) 
        -> Result<Vec<Complex64>> 
    {
        let mut traces = Vec::new();
        let mut power = operator.clone();
        
        for _ in 1..=max_power {
            traces.push(Complex64::new(power.trace(), 0.0));
            power = &power * operator;
        }
        
        Ok(traces)
    }
}

/// Extractor for topological features
pub struct TopologicalFeatureExtractor {
    config: Arc<FeatureConfig>,
}

impl TopologicalFeatureExtractor {
    pub fn new(config: Arc<FeatureConfig>) -> Self {
        Self { config }
    }
    
    /// Extract Betti numbers and other topological invariants
    pub fn extract_topological_features(&self, complex: &SimplicialComplex) -> Result<Array1<i32>> {
        let mut features = Vec::new();
        
        // Betti numbers
        let betti = complex.compute_betti_numbers()?;
        features.extend_from_slice(&betti);
        
        // Euler characteristic
        let euler = complex.euler_characteristic();
        features.push(euler);
        
        // Number of connected components
        features.push(complex.connected_components() as i32);
        
        // Pad to consistent size
        features.resize(32, 0); // Fixed size for topological features
        
        Ok(Array1::from_vec(features))
    }
}

/// Simplified simplicial complex for topological computations
pub struct SimplicialComplex {
    vertices: Vec<usize>,
    edges: Vec<(usize, usize)>,
    faces: Vec<(usize, usize, usize)>,
}

impl SimplicialComplex {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
        }
    }
    
    /// Compute Betti numbers (simplified)
    pub fn compute_betti_numbers(&self) -> Result<Vec<i32>> {
        // Simplified computation - in practice would use proper homology
        let b0 = self.connected_components() as i32;
        let b1 = (self.edges.len() as i32) - (self.vertices.len() as i32) + b0;
        let b2 = self.faces.len() as i32;
        
        Ok(vec![b0, b1, b2])
    }
    
    /// Compute Euler characteristic
    pub fn euler_characteristic(&self) -> i32 {
        self.vertices.len() as i32 - self.edges.len() as i32 + self.faces.len() as i32
    }
    
    /// Count connected components (simplified using DFS)
    pub fn connected_components(&self) -> usize {
        if self.vertices.is_empty() {
            return 0;
        }
        
        let mut visited = vec![false; self.vertices.len()];
        let mut components = 0;
        
        for i in 0..self.vertices.len() {
            if !visited[i] {
                self.dfs(i, &mut visited);
                components += 1;
            }
        }
        
        components
    }
    
    fn dfs(&self, vertex: usize, visited: &mut [bool]) {
        visited[vertex] = true;
        
        for &(u, v) in &self.edges {
            if u == vertex && !visited[v] {
                self.dfs(v, visited);
            } else if v == vertex && !visited[u] {
                self.dfs(u, visited);
            }
        }
    }
}

/// Combined feature extractor
pub struct FeatureExtractor {
    geometric: GeometricFeatureExtractor,
    algebraic: AlgebraicFeatureExtractor,
    spectral: SpectralFeatureExtractor,
    topological: TopologicalFeatureExtractor,
    config: Arc<FeatureConfig>,
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        let config = Arc::new(config);
        
        Self {
            geometric: GeometricFeatureExtractor::new(config.clone()),
            algebraic: AlgebraicFeatureExtractor::new(config.clone()),
            spectral: SpectralFeatureExtractor::new(config.clone()),
            topological: TopologicalFeatureExtractor::new(config.clone()),
            config,
        }
    }
    
    /// Extract features in parallel from multiple objects
    pub fn extract_batch<T>(&self, objects: &[T]) -> Result<Vec<FeatureVector>>
    where
        T: Featurizable + Send + Sync,
    {
        objects.par_iter()
            .map(|obj| obj.extract_features(&self.config))
            .collect::<Result<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    
    #[test]
    fn test_feature_vector_creation() {
        let fv = FeatureVector::new("test".to_string(), 10, 5);
        assert_eq!(fv.id, "test");
        assert_eq!(fv.geometric.len(), 10);
        assert_eq!(fv.algebraic.len(), 5);
        assert_eq!(fv.dimension(), 15);
    }
    
    #[test]
    fn test_feature_normalization() {
        let mut fv = FeatureVector::new("test".to_string(), 3, 3);
        fv.geometric = arr1(&[3.0, 4.0, 0.0]);
        fv.algebraic = arr1(&[1.0, 1.0, 1.0]);
        
        fv.normalize();
        
        let geo_norm = fv.geometric.dot(&fv.geometric).sqrt();
        let alg_norm = fv.algebraic.dot(&fv.algebraic).sqrt();
        
        assert!((geo_norm - 1.0).abs() < 1e-10);
        assert!((alg_norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_simplicial_complex() {
        let mut complex = SimplicialComplex::new();
        complex.vertices = vec![0, 1, 2, 3];
        complex.edges = vec![(0, 1), (1, 2), (2, 0)];
        complex.faces = vec![(0, 1, 2)];
        
        assert_eq!(complex.euler_characteristic(), 2);
        assert_eq!(complex.connected_components(), 2); // One triangle, one isolated vertex
    }
}