//! Data Generation for Langlands Correspondence Training
//! 
//! This module generates synthetic training data for known cases of the
//! geometric Langlands correspondence, including GL(1), GL(2), and other
//! special cases where the correspondence is well-understood.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::geometry::{Bundle, Curve};
use crate::topology::LocalSystem;
use crate::algebra::DModule;
use super::{TrainingResult, TrainingError};

/// Configuration for synthetic data generation
#[derive(Debug, Clone)]
pub struct SyntheticDataConfig {
    /// Number of samples to generate
    pub num_samples: usize,
    /// Curve genus for samples
    pub genus: usize,
    /// Group rank (1 for GL(1), 2 for GL(2), etc.)
    pub rank: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Include known correspondence cases
    pub include_known_cases: bool,
    /// Noise level for synthetic data (0.0 = no noise, 1.0 = high noise)
    pub noise_level: f64,
    /// Data complexity level (affects parameter ranges)
    pub complexity_level: DataComplexity,
}

/// Data complexity levels
#[derive(Debug, Clone, Copy)]
pub enum DataComplexity {
    /// Simple cases (small parameters, well-conditioned)
    Simple,
    /// Moderate complexity
    Moderate,
    /// High complexity (challenging cases)
    High,
    /// Mixed complexity levels
    Mixed,
}

impl Default for SyntheticDataConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            genus: 1,
            rank: 2,
            seed: 42,
            include_known_cases: true,
            noise_level: 0.1,
            complexity_level: DataComplexity::Moderate,
        }
    }
}

/// Data generator for training samples
pub struct DataGenerator {
    config: SyntheticDataConfig,
    rng: ChaCha8Rng,
    known_correspondences: HashMap<String, Vec<(Bundle, LocalSystem)>>,
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new(config: SyntheticDataConfig) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut generator = Self {
            config,
            rng,
            known_correspondences: HashMap::new(),
        };
        
        generator.initialize_known_cases();
        generator
    }
    
    /// Generate training data
    pub fn generate_training_data(&mut self) -> TrainingResult<TrainingDataset> {
        let mut bundles = Vec::new();
        let mut local_systems = Vec::new();
        let mut labels = Vec::new();
        
        // Generate known correspondence pairs
        if self.config.include_known_cases {
            let known_count = (self.config.num_samples as f64 * 0.3) as usize;
            for i in 0..known_count {
                let (bundle, system) = self.generate_known_correspondence_pair()?;
                bundles.push(bundle);
                local_systems.push(system);
                labels.push(format!("known_{}", i));
            }
        }
        
        // Generate synthetic pairs
        let remaining_count = self.config.num_samples - bundles.len();
        for i in 0..remaining_count {
            let (bundle, system) = self.generate_synthetic_pair()?;
            bundles.push(bundle);
            local_systems.push(system);
            labels.push(format!("synthetic_{}", i));
        }
        
        // Add noise if requested
        if self.config.noise_level > 0.0 {
            self.add_noise_to_data(&mut bundles, &mut local_systems)?;
        }
        
        Ok(TrainingDataset {
            bundles,
            local_systems,
            labels,
            metadata: self.generate_metadata(),
        })
    }
    
    /// Generate a known correspondence pair
    fn generate_known_correspondence_pair(&mut self) -> TrainingResult<(Bundle, LocalSystem)> {
        match self.config.rank {
            1 => self.generate_gl1_pair(),
            2 => self.generate_gl2_pair(),
            _ => self.generate_higher_rank_pair(),
        }
    }
    
    /// Generate GL(1) correspondence (abelian case)
    fn generate_gl1_pair(&mut self) -> TrainingResult<(Bundle, LocalSystem)> {
        // For GL(1), the correspondence is via Fourier transform
        let curve = Curve::new_genus(self.config.genus);
        
        // Generate a line bundle (rank 1 bundle)
        let degree = self.rng.gen_range(-5..=5);
        let bundle = Bundle::line_bundle(&curve, degree);
        
        // Corresponding character of π₁
        let character_eigenvalue = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * self.rng.gen::<f64>());
        let local_system = LocalSystem::character(&curve, character_eigenvalue);
        
        Ok((bundle, local_system))
    }
    
    /// Generate GL(2) correspondence
    fn generate_gl2_pair(&mut self) -> TrainingResult<(Bundle, LocalSystem)> {
        let curve = Curve::new_genus(self.config.genus);
        
        // Generate a rank-2 bundle
        let degree = self.rng.gen_range(-3..=3);
        let bundle = self.generate_rank2_bundle(&curve, degree)?;
        
        // Generate corresponding representation
        let local_system = self.generate_corresponding_gl2_representation(&bundle)?;
        
        Ok((bundle, local_system))
    }
    
    /// Generate higher rank correspondence
    fn generate_higher_rank_pair(&mut self) -> TrainingResult<(Bundle, LocalSystem)> {
        let curve = Curve::new_genus(self.config.genus);
        
        // For higher rank, use simplified models
        let degree = self.rng.gen_range(-2..=2);
        let bundle = Bundle::stable_bundle(&curve, self.config.rank, degree);
        
        // Generate a semi-simple representation
        let eigenvalues = self.generate_semisimple_eigenvalues(self.config.rank);
        let local_system = LocalSystem::semisimple(&curve, eigenvalues);
        
        Ok((bundle, local_system))
    }
    
    /// Generate synthetic (possibly non-corresponding) pair
    fn generate_synthetic_pair(&mut self) -> TrainingResult<(Bundle, LocalSystem)> {
        let curve = Curve::new_genus(self.config.genus);
        
        // Generate random bundle
        let bundle = self.generate_random_bundle(&curve)?;
        
        // Generate random local system (may or may not correspond)
        let local_system = if self.rng.gen_bool(0.7) {
            // 70% chance of generating a potentially corresponding system
            self.generate_plausible_system(&bundle)?
        } else {
            // 30% chance of generating a random system
            self.generate_random_system(&curve)?
        };
        
        Ok((bundle, local_system))
    }
    
    /// Generate a random bundle
    fn generate_random_bundle(&mut self, curve: &Curve) -> TrainingResult<Bundle> {
        let rank = self.config.rank;
        let degree = match self.config.complexity_level {
            DataComplexity::Simple => self.rng.gen_range(-2..=2),
            DataComplexity::Moderate => self.rng.gen_range(-5..=5),
            DataComplexity::High => self.rng.gen_range(-10..=10),
            DataComplexity::Mixed => self.rng.gen_range(-10..=10),
        };
        
        // Generate transition functions
        let num_patches = 2 + self.config.genus; // Basic cover
        let mut transition_matrices = Vec::new();
        
        for _ in 0..num_patches * (num_patches - 1) / 2 {
            let matrix = self.generate_random_gl_matrix(rank)?;
            transition_matrices.push(matrix);
        }
        
        Bundle::from_transitions(curve.clone(), transition_matrices)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create bundle: {}", e)))
    }
    
    /// Generate a random GL(n) matrix
    fn generate_random_gl_matrix(&mut self, n: usize) -> TrainingResult<Array2<Complex64>> {
        let mut matrix = Array2::zeros((n, n));
        
        // Generate a random matrix and ensure it's invertible
        for i in 0..n {
            for j in 0..n {
                let real_part = self.rng.gen_range(-2.0..2.0);
                let imag_part = self.rng.gen_range(-2.0..2.0);
                matrix[[i, j]] = Complex64::new(real_part, imag_part);
            }
        }
        
        // Ensure diagonal dominance for stability
        for i in 0..n {
            matrix[[i, i]] += Complex64::new(3.0, 0.0);
        }
        
        Ok(matrix)
    }
    
    /// Generate a plausible local system that might correspond to a bundle
    fn generate_plausible_system(&mut self, bundle: &Bundle) -> TrainingResult<LocalSystem> {
        let curve = bundle.base_curve();
        let rank = bundle.rank();
        
        // Use bundle invariants to guide system generation
        let monodromy_matrices = self.generate_monodromy_from_bundle(bundle)?;
        
        LocalSystem::from_monodromy(curve.clone(), monodromy_matrices)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create local system: {}", e)))
    }
    
    /// Generate monodromy matrices based on bundle properties
    fn generate_monodromy_from_bundle(&mut self, bundle: &Bundle) -> TrainingResult<Vec<Array2<Complex64>>> {
        let genus = bundle.base_curve().genus();
        let rank = bundle.rank();
        let num_generators = 2 * genus; // Standard presentation of π₁
        
        let mut matrices = Vec::new();
        
        // Generate matrices that satisfy the fundamental relation
        for i in 0..num_generators {
            let matrix = if i % 2 == 0 {
                // a_i generators
                self.generate_matrix_with_eigenvalue_constraint(rank, bundle.chern_classes())?
            } else {
                // b_i generators
                self.generate_matrix_with_trace_constraint(rank, bundle.degree() as f64)?
            };
            matrices.push(matrix);
        }
        
        // Adjust to satisfy the relation ∏[a_i, b_i] = I
        self.adjust_matrices_for_relation(&mut matrices)?;
        
        Ok(matrices)
    }
    
    /// Generate matrix with eigenvalue constraints
    fn generate_matrix_with_eigenvalue_constraint(
        &mut self,
        rank: usize,
        chern_classes: &[f64],
    ) -> TrainingResult<Array2<Complex64>> {
        let mut matrix = Array2::zeros((rank, rank));
        
        // Generate eigenvalues based on Chern classes
        let mut eigenvalues = Vec::new();
        for i in 0..rank {
            let constraint = if i < chern_classes.len() { chern_classes[i] } else { 0.0 };
            let angle = 2.0 * std::f64::consts::PI * (constraint + self.rng.gen::<f64>()) / rank as f64;
            eigenvalues.push(Complex64::from_polar(1.0, angle));
        }
        
        // Create diagonal matrix with these eigenvalues
        for i in 0..rank {
            matrix[[i, i]] = eigenvalues[i];
        }
        
        // Conjugate by a random matrix to make it non-diagonal
        let conjugator = self.generate_random_gl_matrix(rank)?;
        let conjugator_inv = self.matrix_inverse(&conjugator)?;
        
        Ok(conjugator.dot(&matrix).dot(&conjugator_inv))
    }
    
    /// Generate matrix with trace constraints
    fn generate_matrix_with_trace_constraint(
        &mut self,
        rank: usize,
        target_trace: f64,
    ) -> TrainingResult<Array2<Complex64>> {
        let mut matrix = self.generate_random_gl_matrix(rank)?;
        
        // Adjust diagonal to achieve target trace
        let current_trace: Complex64 = (0..rank).map(|i| matrix[[i, i]]).sum();
        let adjustment = (Complex64::new(target_trace, 0.0) - current_trace) / rank as f64;
        
        for i in 0..rank {
            matrix[[i, i]] += adjustment;
        }
        
        Ok(matrix)
    }
    
    /// Adjust matrices to satisfy fundamental group relations
    fn adjust_matrices_for_relation(&mut self, matrices: &mut Vec<Array2<Complex64>>) -> TrainingResult<()> {
        let genus = matrices.len() / 2;
        
        if genus == 0 {
            return Ok(());
        }
        
        // Compute the product ∏[a_i, b_i]
        let rank = matrices[0].nrows();
        let mut product = Array2::eye(rank);
        
        for g in 0..genus {
            let a = &matrices[2 * g];
            let b = &matrices[2 * g + 1];
            
            // Compute commutator [a, b] = aba⁻¹b⁻¹
            let a_inv = self.matrix_inverse(a)?;
            let b_inv = self.matrix_inverse(b)?;
            let commutator = a.dot(b).dot(&a_inv).dot(&b_inv);
            
            product = product.dot(&commutator);
        }
        
        // Adjust the last matrix to make product = identity
        if genus > 0 {
            let correction = self.matrix_inverse(&product)?;
            let last_idx = matrices.len() - 1;
            matrices[last_idx] = matrices[last_idx].dot(&correction);
        }
        
        Ok(())
    }
    
    /// Generate a random local system
    fn generate_random_system(&mut self, curve: &Curve) -> TrainingResult<LocalSystem> {
        let rank = self.config.rank;
        let genus = curve.genus();
        let num_generators = 2 * genus;
        
        let mut matrices = Vec::new();
        for _ in 0..num_generators {
            matrices.push(self.generate_random_gl_matrix(rank)?);
        }
        
        // Ensure the fundamental relation is satisfied
        self.adjust_matrices_for_relation(&mut matrices)?;
        
        LocalSystem::from_monodromy(curve.clone(), matrices)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create random system: {}", e)))
    }
    
    /// Generate semi-simple eigenvalues
    fn generate_semisimple_eigenvalues(&mut self, rank: usize) -> Vec<Complex64> {
        let mut eigenvalues = Vec::new();
        
        for i in 0..rank {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / rank as f64;
            let radius = 1.0 + 0.1 * self.rng.gen::<f64>(); // Slight perturbation
            eigenvalues.push(Complex64::from_polar(radius, angle));
        }
        
        eigenvalues
    }
    
    /// Add noise to the generated data
    fn add_noise_to_data(
        &mut self,
        bundles: &mut Vec<Bundle>,
        systems: &mut Vec<LocalSystem>,
    ) -> TrainingResult<()> {
        for bundle in bundles.iter_mut() {
            bundle.add_noise(self.config.noise_level, &mut self.rng)?;
        }
        
        for system in systems.iter_mut() {
            system.add_noise(self.config.noise_level, &mut self.rng)?;
        }
        
        Ok(())
    }
    
    /// Initialize known correspondence cases
    fn initialize_known_cases(&mut self) {
        // This would be populated with known mathematical examples
        // For now, we'll use the generation methods above
    }
    
    /// Generate metadata for the dataset
    fn generate_metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            num_samples: self.config.num_samples,
            genus: self.config.genus,
            rank: self.config.rank,
            complexity_level: self.config.complexity_level,
            noise_level: self.config.noise_level,
            generation_timestamp: chrono::Utc::now(),
        }
    }
    
    /// Generate rank-2 bundle for GL(2) case
    fn generate_rank2_bundle(&mut self, curve: &Curve, degree: i32) -> TrainingResult<Bundle> {
        // Create a stable rank-2 bundle
        Bundle::stable_bundle(curve, 2, degree)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create rank-2 bundle: {}", e)))
    }
    
    /// Generate corresponding GL(2) representation
    fn generate_corresponding_gl2_representation(&mut self, bundle: &Bundle) -> TrainingResult<LocalSystem> {
        // Use Narasimhan-Seshadri correspondence for genus > 1
        // or Drinfeld's results for function fields
        
        let curve = bundle.base_curve();
        let genus = curve.genus();
        
        if genus == 0 {
            // For genus 0, use simpler correspondence
            self.generate_genus0_gl2_system(bundle)
        } else {
            // For higher genus, use stable bundle correspondence
            self.generate_stable_bundle_system(bundle)
        }
    }
    
    /// Generate GL(2) system for genus 0
    fn generate_genus0_gl2_system(&mut self, bundle: &Bundle) -> TrainingResult<LocalSystem> {
        let curve = bundle.base_curve();
        
        // For genus 0, fewer constraints
        let matrix = self.generate_random_gl_matrix(2)?;
        
        LocalSystem::from_single_monodromy(curve.clone(), matrix)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create genus-0 system: {}", e)))
    }
    
    /// Generate system corresponding to stable bundle
    fn generate_stable_bundle_system(&mut self, bundle: &Bundle) -> TrainingResult<LocalSystem> {
        let curve = bundle.base_curve();
        let genus = curve.genus();
        
        // Generate monodromy with traces related to bundle invariants
        let mut matrices = Vec::new();
        
        for g in 0..genus {
            // a_g generator
            let trace_a = bundle.chern_classes()[0] + self.rng.gen_range(-0.5..0.5);
            let matrix_a = self.generate_sl2_matrix_with_trace(trace_a)?;
            matrices.push(matrix_a);
            
            // b_g generator  
            let trace_b = bundle.degree() as f64 / bundle.rank() as f64 + self.rng.gen_range(-0.5..0.5);
            let matrix_b = self.generate_sl2_matrix_with_trace(trace_b)?;
            matrices.push(matrix_b);
        }
        
        // Ensure relation is satisfied
        self.adjust_matrices_for_relation(&mut matrices)?;
        
        LocalSystem::from_monodromy(curve.clone(), matrices)
            .map_err(|e| TrainingError::DataGeneration(format!("Failed to create stable system: {}", e)))
    }
    
    /// Generate SL(2) matrix with given trace
    fn generate_sl2_matrix_with_trace(&mut self, trace: f64) -> TrainingResult<Array2<Complex64>> {
        // For SL(2), if trace = t, eigenvalues are (t ± √(t²-4))/2
        let discriminant = trace * trace - 4.0;
        
        let (lambda1, lambda2) = if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            ((trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0)
        } else {
            let sqrt_disc = (-discriminant).sqrt();
            (trace / 2.0, sqrt_disc / 2.0) // Will create complex eigenvalues
        };
        
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(lambda1, 0.0);
        matrix[[1, 1]] = Complex64::new(lambda2, 0.0);
        
        // Add off-diagonal terms
        matrix[[0, 1]] = Complex64::new(self.rng.gen_range(-1.0..1.0), 0.0);
        matrix[[1, 0]] = Complex64::new(1.0 - lambda1 * lambda2, 0.0) / matrix[[0, 1]];
        
        Ok(matrix)
    }
    
    /// Compute matrix inverse
    fn matrix_inverse(&self, matrix: &Array2<Complex64>) -> TrainingResult<Array2<Complex64>> {
        // For small matrices, use explicit formula
        let n = matrix.nrows();
        
        if n == 1 {
            let mut result = Array2::zeros((1, 1));
            result[[0, 0]] = Complex64::new(1.0, 0.0) / matrix[[0, 0]];
            return Ok(result);
        }
        
        if n == 2 {
            let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
            let mut result = Array2::zeros((2, 2));
            result[[0, 0]] = matrix[[1, 1]] / det;
            result[[0, 1]] = -matrix[[0, 1]] / det;
            result[[1, 0]] = -matrix[[1, 0]] / det;
            result[[1, 1]] = matrix[[0, 0]] / det;
            return Ok(result);
        }
        
        // For larger matrices, would need proper linear algebra library
        Err(TrainingError::DataGeneration(
            "Matrix inversion not implemented for n > 2".to_string()
        ))
    }
}

/// Training dataset
#[derive(Debug)]
pub struct TrainingDataset {
    /// Vector bundles (automorphic side)
    pub bundles: Vec<Bundle>,
    /// Local systems (spectral side)
    pub local_systems: Vec<LocalSystem>,
    /// Sample labels for tracking
    pub labels: Vec<String>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Dataset metadata
#[derive(Debug)]
pub struct DatasetMetadata {
    pub num_samples: usize,
    pub genus: usize,
    pub rank: usize,
    pub complexity_level: DataComplexity,
    pub noise_level: f64,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

impl TrainingDataset {
    /// Split dataset into training and validation sets
    pub fn split(&self, validation_fraction: f32) -> (TrainingDataset, TrainingDataset) {
        let n_total = self.bundles.len();
        let n_train = ((1.0 - validation_fraction) * n_total as f32) as usize;
        
        let train_dataset = TrainingDataset {
            bundles: self.bundles[..n_train].to_vec(),
            local_systems: self.local_systems[..n_train].to_vec(),
            labels: self.labels[..n_train].to_vec(),
            metadata: DatasetMetadata {
                num_samples: n_train,
                ..self.metadata
            },
        };
        
        let val_dataset = TrainingDataset {
            bundles: self.bundles[n_train..].to_vec(),
            local_systems: self.local_systems[n_train..].to_vec(),
            labels: self.labels[n_train..].to_vec(),
            metadata: DatasetMetadata {
                num_samples: n_total - n_train,
                ..self.metadata
            },
        };
        
        (train_dataset, val_dataset)
    }
    
    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.bundles.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_generator_creation() {
        let config = SyntheticDataConfig::default();
        let generator = DataGenerator::new(config);
        assert_eq!(generator.config.num_samples, 1000);
    }
    
    #[test]
    fn test_gl1_data_generation() {
        let mut config = SyntheticDataConfig::default();
        config.rank = 1;
        config.num_samples = 10;
        
        let mut generator = DataGenerator::new(config);
        let result = generator.generate_training_data();
        assert!(result.is_ok());
        
        let dataset = result.unwrap();
        assert_eq!(dataset.len(), 10);
    }
    
    #[test]
    fn test_dataset_split() {
        let mut config = SyntheticDataConfig::default();
        config.num_samples = 100;
        
        let mut generator = DataGenerator::new(config);
        let dataset = generator.generate_training_data().unwrap();
        
        let (train, val) = dataset.split(0.2);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }
}