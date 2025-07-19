//! Data generation utilities for creating synthetic mathematical objects

use std::f64::consts::PI;
use rand::{Rng, SeedableRng, distributions::{Distribution, Uniform}};
use rand_chacha::ChaCha8Rng;
use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rayon::prelude::*;

use super::{Result, features::{FeatureVector, Featurizable, SimplicialComplex}, FeatureConfig};

/// Generator configuration
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Random seed for reproducibility
    pub seed: u64,
    
    /// Number of samples to generate
    pub num_samples: usize,
    
    /// Complexity level (affects object complexity)
    pub complexity: ComplexityLevel,
    
    /// Enable noise in generated data
    pub add_noise: bool,
    
    /// Noise level (if enabled)
    pub noise_level: f64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            num_samples: 1000,
            complexity: ComplexityLevel::Medium,
            add_noise: true,
            noise_level: 0.1,
        }
    }
}

/// Complexity levels for generated objects
#[derive(Debug, Clone, Copy)]
pub enum ComplexityLevel {
    Simple,
    Medium,
    Complex,
}

impl ComplexityLevel {
    /// Get dimension ranges for this complexity
    pub fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Simple => (2, 4),
            Self::Medium => (4, 8),
            Self::Complex => (8, 16),
        }
    }
    
    /// Get degree range for polynomial objects
    pub fn degree_range(&self) -> (usize, usize) {
        match self {
            Self::Simple => (2, 4),
            Self::Medium => (4, 8),
            Self::Complex => (8, 12),
        }
    }
}

/// Base trait for data generators
pub trait DataGenerator: Send + Sync {
    type Output: Featurizable;
    
    /// Generate a single sample
    fn generate_one(&self, rng: &mut ChaCha8Rng) -> Self::Output;
    
    /// Generate a batch of samples
    fn generate_batch(&self, config: &GeneratorConfig) -> Vec<Self::Output> {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        
        (0..config.num_samples)
            .map(|_| self.generate_one(&mut rng))
            .collect()
    }
    
    /// Generate samples in parallel
    fn generate_parallel(&self, config: &GeneratorConfig) -> Vec<Self::Output> {
        (0..config.num_samples)
            .into_par_iter()
            .map(|i| {
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed + i as u64);
                self.generate_one(&mut rng)
            })
            .collect()
    }
}

/// Synthetic Riemann surface representation
#[derive(Clone, Debug)]
pub struct SyntheticRiemannSurface {
    pub id: String,
    pub genus: usize,
    pub points: Array2<f64>,
    pub metric: DMatrix<f64>,
    pub holomorphic_forms: Vec<Complex64>,
}

impl Featurizable for SyntheticRiemannSurface {
    fn extract_features(&self, config: &FeatureConfig) -> Result<FeatureVector> {
        let mut features = FeatureVector::new(
            self.id.clone(),
            config.geometric_dim,
            config.algebraic_dim,
        );
        
        // Extract geometric features from metric
        let eigen = self.metric.symmetric_eigen();
        for (i, eval) in eigen.eigenvalues.iter().enumerate() {
            if i < config.geometric_dim / 4 {
                features.geometric[i] = *eval;
            }
        }
        
        // Add topological features
        features.geometric[config.geometric_dim / 4] = self.genus as f64;
        
        // Extract algebraic features from holomorphic forms
        for (i, form) in self.holomorphic_forms.iter().enumerate() {
            if i * 2 < config.algebraic_dim {
                features.algebraic[i * 2] = form.re;
                features.algebraic[i * 2 + 1] = form.im;
            }
        }
        
        // Add spectral features if enabled
        if config.enable_spectral {
            features.spectral = Some(Array1::from_vec(self.holomorphic_forms.clone()));
        }
        
        Ok(features)
    }
    
    fn cache_key(&self) -> String {
        format!("riemann_surface:{}", self.id)
    }
}

/// Generator for Riemann surfaces
pub struct RiemannSurfaceGenerator {
    complexity: ComplexityLevel,
}

impl RiemannSurfaceGenerator {
    pub fn new(complexity: ComplexityLevel) -> Self {
        Self { complexity }
    }
    
    fn generate_metric(&self, dim: usize, rng: &mut ChaCha8Rng) -> DMatrix<f64> {
        let mut metric = DMatrix::<f64>::zeros(dim, dim);
        
        // Generate positive definite metric
        for i in 0..dim {
            for j in i..dim {
                let value = if i == j {
                    rng.gen_range(0.5..2.0)
                } else {
                    rng.gen_range(-0.3..0.3)
                };
                metric[(i, j)] = value;
                metric[(j, i)] = value;
            }
        }
        
        // Ensure positive definiteness
        let eigen = metric.symmetric_eigen();
        let min_eval = eigen.eigenvalues.min();
        if min_eval < 0.1 {
            for i in 0..dim {
                metric[(i, i)] += 0.1 - min_eval;
            }
        }
        
        metric
    }
}

impl DataGenerator for RiemannSurfaceGenerator {
    type Output = SyntheticRiemannSurface;
    
    fn generate_one(&self, rng: &mut ChaCha8Rng) -> Self::Output {
        let (min_dim, max_dim) = self.complexity.dimensions();
        let dim = rng.gen_range(min_dim..=max_dim);
        let genus = rng.gen_range(0..=3);
        
        // Generate points on surface
        let num_points = rng.gen_range(50..200);
        let mut points = Array2::zeros((num_points, 2));
        
        for i in 0..num_points {
            let theta = 2.0 * PI * (i as f64) / (num_points as f64);
            let r = 1.0 + 0.3 * (genus as f64 * theta).sin();
            points[[i, 0]] = r * theta.cos();
            points[[i, 1]] = r * theta.sin();
        }
        
        // Generate metric
        let metric = self.generate_metric(dim, rng);
        
        // Generate holomorphic forms
        let num_forms = genus + 1;
        let holomorphic_forms: Vec<Complex64> = (0..num_forms)
            .map(|k| {
                let phase = 2.0 * PI * (k as f64) / (num_forms as f64);
                Complex64::new(
                    rng.gen_range(-1.0..1.0) * phase.cos(),
                    rng.gen_range(-1.0..1.0) * phase.sin(),
                )
            })
            .collect();
        
        SyntheticRiemannSurface {
            id: format!("riemann_{}", rng.gen::<u32>()),
            genus,
            points,
            metric,
            holomorphic_forms,
        }
    }
}

/// Synthetic vector bundle representation
#[derive(Clone, Debug)]
pub struct SyntheticVectorBundle {
    pub id: String,
    pub rank: usize,
    pub base_dim: usize,
    pub connection: DMatrix<f64>,
    pub curvature: DMatrix<f64>,
    pub chern_classes: Vec<f64>,
}

impl Featurizable for SyntheticVectorBundle {
    fn extract_features(&self, config: &FeatureConfig) -> Result<FeatureVector> {
        let mut features = FeatureVector::new(
            self.id.clone(),
            config.geometric_dim,
            config.algebraic_dim,
        );
        
        // Geometric features from curvature
        let curvature_trace = self.curvature.trace();
        let curvature_norm = self.curvature.norm_squared().sqrt();
        
        features.geometric[0] = curvature_trace;
        features.geometric[1] = curvature_norm;
        features.geometric[2] = self.rank as f64;
        features.geometric[3] = self.base_dim as f64;
        
        // Connection eigenvalues
        if let Ok(eigen) = self.connection.eigenvalues() {
            for (i, eval) in eigen.iter().enumerate() {
                if i + 4 < config.geometric_dim / 2 {
                    features.geometric[i * 2 + 4] = eval.re;
                    features.geometric[i * 2 + 5] = eval.im;
                }
            }
        }
        
        // Algebraic features from Chern classes
        for (i, &chern) in self.chern_classes.iter().enumerate() {
            if i < config.algebraic_dim / 2 {
                features.algebraic[i] = chern;
            }
        }
        
        Ok(features)
    }
    
    fn cache_key(&self) -> String {
        format!("vector_bundle:{}", self.id)
    }
}

/// Generator for vector bundles
pub struct VectorBundleGenerator {
    complexity: ComplexityLevel,
}

impl VectorBundleGenerator {
    pub fn new(complexity: ComplexityLevel) -> Self {
        Self { complexity }
    }
}

impl DataGenerator for VectorBundleGenerator {
    type Output = SyntheticVectorBundle;
    
    fn generate_one(&self, rng: &mut ChaCha8Rng) -> Self::Output {
        let (min_dim, max_dim) = self.complexity.dimensions();
        let base_dim = rng.gen_range(min_dim..=max_dim);
        let rank = rng.gen_range(1..=4);
        
        let total_dim = base_dim * rank;
        
        // Generate connection matrix
        let mut connection = DMatrix::<f64>::zeros(total_dim, total_dim);
        for i in 0..total_dim {
            for j in 0..total_dim {
                connection[(i, j)] = rng.gen_range(-1.0..1.0);
            }
        }
        
        // Make it antisymmetric (connection 1-form property)
        let connection = &connection - &connection.transpose();
        
        // Generate curvature (simplified as F = dA + A∧A)
        let curvature = &connection * &connection;
        
        // Generate Chern classes (simplified)
        let chern_classes: Vec<f64> = (1..=rank)
            .map(|k| {
                let trace_power = curvature.pow(k).trace();
                trace_power / (2.0 * PI).powi(k as i32)
            })
            .collect();
        
        SyntheticVectorBundle {
            id: format!("bundle_{}", rng.gen::<u32>()),
            rank,
            base_dim,
            connection,
            curvature,
            chern_classes,
        }
    }
}

/// Synthetic D-module representation
#[derive(Clone, Debug)]
pub struct SyntheticDModule {
    pub id: String,
    pub dimension: usize,
    pub differential_operators: Vec<DMatrix<f64>>,
    pub holonomic_rank: usize,
    pub characteristic_variety: Vec<Vec<f64>>,
}

impl Featurizable for SyntheticDModule {
    fn extract_features(&self, config: &FeatureConfig) -> Result<FeatureVector> {
        let mut features = FeatureVector::new(
            self.id.clone(),
            config.geometric_dim,
            config.algebraic_dim,
        );
        
        // Geometric features from differential operators
        for (i, op) in self.differential_operators.iter().enumerate() {
            if i * 3 < config.geometric_dim {
                features.geometric[i * 3] = op.trace();
                features.geometric[i * 3 + 1] = op.determinant();
                features.geometric[i * 3 + 2] = op.norm_squared().sqrt();
            }
        }
        
        // Algebraic features from characteristic variety
        for (i, point) in self.characteristic_variety.iter().enumerate() {
            for (j, &coord) in point.iter().enumerate() {
                let idx = i * self.dimension + j;
                if idx < config.algebraic_dim {
                    features.algebraic[idx] = coord;
                }
            }
        }
        
        // Add holonomic rank
        if config.algebraic_dim > 0 {
            features.algebraic[0] = self.holonomic_rank as f64;
        }
        
        Ok(features)
    }
    
    fn cache_key(&self) -> String {
        format!("d_module:{}", self.id)
    }
}

/// Generator for D-modules
pub struct DModuleGenerator {
    complexity: ComplexityLevel,
}

impl DModuleGenerator {
    pub fn new(complexity: ComplexityLevel) -> Self {
        Self { complexity }
    }
}

impl DataGenerator for DModuleGenerator {
    type Output = SyntheticDModule;
    
    fn generate_one(&self, rng: &mut ChaCha8Rng) -> Self::Output {
        let (min_dim, max_dim) = self.complexity.dimensions();
        let dimension = rng.gen_range(min_dim..=max_dim);
        
        // Generate differential operators
        let num_operators = rng.gen_range(2..5);
        let differential_operators: Vec<DMatrix<f64>> = (0..num_operators)
            .map(|_| {
                let mut op = DMatrix::<f64>::zeros(dimension, dimension);
                for i in 0..dimension {
                    for j in 0..dimension {
                        op[(i, j)] = rng.gen_range(-2.0..2.0);
                    }
                }
                op
            })
            .collect();
        
        // Generate characteristic variety (simplified)
        let num_points = rng.gen_range(5..20);
        let characteristic_variety: Vec<Vec<f64>> = (0..num_points)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        
        let holonomic_rank = rng.gen_range(1..=dimension);
        
        SyntheticDModule {
            id: format!("dmodule_{}", rng.gen::<u32>()),
            dimension,
            differential_operators,
            holonomic_rank,
            characteristic_variety,
        }
    }
}

/// Combined data generator for all object types
pub struct GeometricLanglandsDataGenerator {
    riemann_gen: RiemannSurfaceGenerator,
    bundle_gen: VectorBundleGenerator,
    dmodule_gen: DModuleGenerator,
    config: GeneratorConfig,
}

impl GeometricLanglandsDataGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let complexity = config.complexity;
        
        Self {
            riemann_gen: RiemannSurfaceGenerator::new(complexity),
            bundle_gen: VectorBundleGenerator::new(complexity),
            dmodule_gen: DModuleGenerator::new(complexity),
            config,
        }
    }
    
    /// Generate a mixed dataset with all object types
    pub fn generate_mixed_dataset(&self) -> Vec<Box<dyn Featurizable>> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut dataset: Vec<Box<dyn Featurizable>> = Vec::new();
        
        let samples_per_type = self.config.num_samples / 3;
        
        // Generate Riemann surfaces
        for _ in 0..samples_per_type {
            let surface = self.riemann_gen.generate_one(&mut rng);
            dataset.push(Box::new(surface));
        }
        
        // Generate vector bundles
        for _ in 0..samples_per_type {
            let bundle = self.bundle_gen.generate_one(&mut rng);
            dataset.push(Box::new(bundle));
        }
        
        // Generate D-modules
        for _ in 0..samples_per_type {
            let dmodule = self.dmodule_gen.generate_one(&mut rng);
            dataset.push(Box::new(dmodule));
        }
        
        // Shuffle if needed
        use rand::seq::SliceRandom;
        dataset.shuffle(&mut rng);
        
        dataset
    }
    
    /// Generate paired data for correspondence learning
    pub fn generate_correspondence_pairs(&self) -> Vec<(Box<dyn Featurizable>, Box<dyn Featurizable>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut pairs = Vec::new();
        
        for i in 0..self.config.num_samples {
            // Generate related objects (simplified correspondence)
            let surface = self.riemann_gen.generate_one(&mut rng);
            
            // Generate corresponding D-module with related parameters
            let mut dmodule = self.dmodule_gen.generate_one(&mut rng);
            dmodule.dimension = surface.genus + 1; // Simple correspondence rule
            
            pairs.push((
                Box::new(surface) as Box<dyn Featurizable>,
                Box::new(dmodule) as Box<dyn Featurizable>,
            ));
        }
        
        pairs
    }
}

/// Utility functions for data augmentation
pub mod augmentation {
    use super::*;
    
    /// Add Gaussian noise to features
    pub fn add_gaussian_noise(
        features: &mut FeatureVector,
        noise_level: f64,
        rng: &mut ChaCha8Rng,
    ) {
        let normal = rand_distr::Normal::new(0.0, noise_level).unwrap();
        
        // Add noise to geometric features
        for val in features.geometric.iter_mut() {
            *val += normal.sample(rng);
        }
        
        // Add noise to algebraic features
        for val in features.algebraic.iter_mut() {
            *val += normal.sample(rng);
        }
    }
    
    /// Apply random rotation to geometric features
    pub fn random_rotation(
        features: &mut FeatureVector,
        rng: &mut ChaCha8Rng,
    ) {
        let angle = rng.gen_range(0.0..2.0 * PI);
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        // Apply 2D rotation to pairs of features
        for i in (0..features.geometric.len()).step_by(2) {
            if i + 1 < features.geometric.len() {
                let x = features.geometric[i];
                let y = features.geometric[i + 1];
                
                features.geometric[i] = x * cos_a - y * sin_a;
                features.geometric[i + 1] = x * sin_a + y * cos_a;
            }
        }
    }
    
    /// Scale features by random factor
    pub fn random_scaling(
        features: &mut FeatureVector,
        scale_range: (f64, f64),
        rng: &mut ChaCha8Rng,
    ) {
        let scale = rng.gen_range(scale_range.0..scale_range.1);
        
        features.geometric *= scale;
        features.algebraic *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_riemann_surface_generator() {
        let gen = RiemannSurfaceGenerator::new(ComplexityLevel::Simple);
        let config = GeneratorConfig::default();
        
        let surfaces = gen.generate_batch(&config);
        assert_eq!(surfaces.len(), config.num_samples);
        
        // Check generated properties
        for surface in surfaces {
            assert!(surface.genus <= 3);
            assert!(surface.points.nrows() >= 50);
            assert!(surface.metric.nrows() >= 2);
        }
    }
    
    #[test]
    fn test_mixed_dataset_generation() {
        let config = GeneratorConfig {
            num_samples: 30,
            ..Default::default()
        };
        
        let gen = GeometricLanglandsDataGenerator::new(config);
        let dataset = gen.generate_mixed_dataset();
        
        assert_eq!(dataset.len(), 30);
    }
    
    #[test]
    fn test_augmentation() {
        let mut features = FeatureVector::new("test".to_string(), 10, 10);
        features.geometric.fill(1.0);
        
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        augmentation::add_gaussian_noise(&mut features, 0.1, &mut rng);
        
        // Check that noise was added
        let all_ones = features.geometric.iter().all(|&x| x == 1.0);
        assert!(!all_ones);
    }
}