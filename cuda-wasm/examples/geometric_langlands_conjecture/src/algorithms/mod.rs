//! Core mathematical algorithms for the geometric Langlands conjecture

pub mod hecke;
pub mod hitchin;
pub mod fourier_mukai;
pub mod correspondences;

// Placeholder - will be implemented by various specialist agents
pub struct AlgorithmConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}