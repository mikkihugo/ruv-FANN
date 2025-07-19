//! Common types and definitions for the Geometric Langlands Conjecture implementation
//!
//! This module provides the fundamental types used across all modules

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main result type for the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Generic element type for mathematical objects
pub type Element = Complex64;

/// Matrix type for linear algebra operations
pub type Matrix = DMatrix<Element>;

/// Vector type for linear algebra operations
pub type Vector = DVector<Element>;

/// Precision configuration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    /// Single precision (f32)
    Single,
    /// Double precision (f64)
    Double,
    /// Arbitrary precision
    Arbitrary,
}

impl Default for Precision {
    fn default() -> Self {
        Self::Double
    }
}

/// Error types for the library
#[derive(Debug, Clone)]
pub enum GeometricError {
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid parameter
    InvalidParameter { param: String, reason: String },
    /// Computation failed
    ComputationFailed { operation: String, reason: String },
    /// Not implemented
    NotImplemented { feature: String },
    /// Numerical instability
    NumericalInstability { tolerance: f64 },
}

impl std::fmt::Display for GeometricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::InvalidParameter { param, reason } => {
                write!(f, "Invalid parameter '{}': {}", param, reason)
            }
            Self::ComputationFailed { operation, reason } => {
                write!(f, "Computation '{}' failed: {}", operation, reason)
            }
            Self::NotImplemented { feature } => {
                write!(f, "Feature '{}' not implemented", feature)
            }
            Self::NumericalInstability { tolerance } => {
                write!(f, "Numerical instability detected (tolerance: {})", tolerance)
            }
        }
    }
}

impl std::error::Error for GeometricError {}

/// Configuration for geometric computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationConfig {
    /// Numerical precision
    pub precision: Precision,
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use parallel computation when available
    pub use_parallel: bool,
    /// Cache intermediate results
    pub enable_cache: bool,
    /// Use GPU acceleration when available
    pub use_gpu: bool,
}

impl Default for ComputationConfig {
    fn default() -> Self {
        Self {
            precision: Precision::Double,
            max_iterations: 1000,
            tolerance: 1e-12,
            use_parallel: true,
            enable_cache: true,
            use_gpu: false,
        }
    }
}

/// Point in a complex manifold
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexPoint {
    /// Coordinates
    pub coordinates: Vector,
    /// Chart information (optional)
    pub chart: Option<String>,
}

impl ComplexPoint {
    /// Create a new complex point
    pub fn new(coordinates: Vector) -> Self {
        Self {
            coordinates,
            chart: None,
        }
    }

    /// Create with chart information
    pub fn with_chart(coordinates: Vector, chart: String) -> Self {
        Self {
            coordinates,
            chart: Some(chart),
        }
    }

    /// Dimension of the point
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Distance to another point
    pub fn distance(&self, other: &Self) -> Result<f64> {
        if self.dimension() != other.dimension() {
            return Err(GeometricError::DimensionMismatch {
                expected: self.dimension(),
                got: other.dimension(),
            }.into());
        }
        
        Ok((&self.coordinates - &other.coordinates).norm())
    }
}

/// Generic algebraic structure
pub trait AlgebraicStructure {
    /// Identity element
    fn identity() -> Self;
    
    /// Check if element is identity
    fn is_identity(&self) -> bool;
    
    /// Inverse element (if exists)
    fn inverse(&self) -> Option<Self> where Self: Sized;
}

/// Metric structure on a manifold
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric tensor at each point
    pub tensor: Box<dyn Fn(&ComplexPoint) -> Matrix + Send + Sync>,
    /// Signature (p, q) for real metrics
    pub signature: Option<(usize, usize)>,
}

/// Bundle structure
#[derive(Debug, Clone)]
pub struct Bundle {
    /// Base manifold dimension
    pub base_dim: usize,
    /// Fiber dimension
    pub fiber_dim: usize,
    /// Total space dimension
    pub total_dim: usize,
    /// Structure group
    pub structure_group: String,
    /// Connection (if any)
    pub connection: Option<Connection>,
}

/// Connection on a bundle
#[derive(Debug, Clone)]
pub struct Connection {
    /// Connection 1-form
    pub form: Matrix,
    /// Curvature 2-form
    pub curvature: Option<Matrix>,
    /// Holonomy group
    pub holonomy: Option<String>,
}

impl Connection {
    /// Compute curvature from connection
    pub fn compute_curvature(&mut self) -> Result<()> {
        // F = dA + A ∧ A (simplified)
        let d_a = &self.form; // Placeholder for exterior derivative
        let wedge = &self.form * &self.form;
        self.curvature = Some(d_a + wedge);
        Ok(())
    }
    
    /// Check if connection is flat
    pub fn is_flat(&self) -> bool {
        if let Some(ref curvature) = self.curvature {
            curvature.norm() < 1e-12
        } else {
            false
        }
    }
}

/// Sheaf on a complex manifold
#[derive(Debug, Clone)]
pub struct Sheaf {
    /// Sections over open sets
    pub sections: HashMap<String, Vector>,
    /// Restriction maps
    pub restrictions: HashMap<(String, String), Matrix>,
    /// Base space
    pub base: String,
}

impl Sheaf {
    /// Create a new sheaf
    pub fn new(base: String) -> Self {
        Self {
            sections: HashMap::new(),
            restrictions: HashMap::new(),
            base,
        }
    }

    /// Add sections over an open set
    pub fn add_sections(&mut self, open_set: String, sections: Vector) {
        self.sections.insert(open_set, sections);
    }

    /// Add restriction map
    pub fn add_restriction(&mut self, from: String, to: String, map: Matrix) {
        self.restrictions.insert((from, to), map);
    }

    /// Compute sheaf cohomology (simplified)
    pub fn cohomology(&self, degree: usize) -> Result<Vector> {
        let n = self.sections.len();
        if degree > n {
            return Ok(Vector::zeros(0));
        }
        
        // Simplified: return first non-trivial cohomology group
        if let Some(sections) = self.sections.values().next() {
            Ok(sections.clone())
        } else {
            Ok(Vector::zeros(1))
        }
    }
}

/// D-module structure
#[derive(Debug, Clone)]
pub struct DModule {
    /// Underlying vector space
    pub vector_space: Vector,
    /// Action of differential operators
    pub operators: HashMap<String, Matrix>,
    /// Characteristic variety
    pub char_variety: Option<String>,
}

impl DModule {
    /// Create a new D-module
    pub fn new(dim: usize) -> Self {
        Self {
            vector_space: Vector::zeros(dim),
            operators: HashMap::new(),
            char_variety: None,
        }
    }

    /// Add differential operator
    pub fn add_operator(&mut self, name: String, operator: Matrix) {
        self.operators.insert(name, operator);
    }

    /// Check if module is holonomic
    pub fn is_holonomic(&self) -> bool {
        // Simplified: check if dimension is minimal
        self.char_variety.is_some()
    }
}

/// Performance metrics for tracking computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Computation time in milliseconds
    pub computation_time: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Error estimate
    pub error_estimate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            computation_time: 0.0,
            memory_usage: 0,
            iterations: 0,
            converged: false,
            error_estimate: f64::INFINITY,
        }
    }
}

/// Cache for storing intermediate results
#[derive(Debug, Clone)]
pub struct ComputationCache {
    /// Cached matrices
    pub matrices: HashMap<String, Matrix>,
    /// Cached vectors
    pub vectors: HashMap<String, Vector>,
    /// Cached scalars
    pub scalars: HashMap<String, Element>,
    /// Cache hit count
    pub hit_count: usize,
    /// Cache miss count
    pub miss_count: usize,
}

impl ComputationCache {
    /// Create a new cache
    pub fn new() -> Self {
        Self {
            matrices: HashMap::new(),
            vectors: HashMap::new(),
            scalars: HashMap::new(),
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Store a matrix in cache
    pub fn store_matrix(&mut self, key: String, matrix: Matrix) {
        self.matrices.insert(key, matrix);
    }

    /// Retrieve a matrix from cache
    pub fn get_matrix(&mut self, key: &str) -> Option<&Matrix> {
        if let Some(matrix) = self.matrices.get(key) {
            self.hit_count += 1;
            Some(matrix)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.matrices.clear();
        self.vectors.clear();
        self.scalars.clear();
        self.hit_count = 0;
        self.miss_count = 0;
    }
}

impl Default for ComputationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate matrix dimensions for multiplication
    pub fn validate_matrix_mult(a: &Matrix, b: &Matrix) -> Result<()> {
        if a.ncols() != b.nrows() {
            return Err(GeometricError::DimensionMismatch {
                expected: a.ncols(),
                got: b.nrows(),
            }.into());
        }
        Ok(())
    }

    /// Validate vector dimensions for operations
    pub fn validate_vector_op(a: &Vector, b: &Vector) -> Result<()> {
        if a.len() != b.len() {
            return Err(GeometricError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            }.into());
        }
        Ok(())
    }

    /// Check numerical stability
    pub fn check_numerical_stability(value: f64, tolerance: f64) -> Result<()> {
        if value.is_nan() || value.is_infinite() {
            return Err(GeometricError::NumericalInstability { tolerance }.into());
        }
        Ok(())
    }

    /// Validate positive parameter
    pub fn validate_positive(value: f64, param_name: &str) -> Result<()> {
        if value <= 0.0 {
            return Err(GeometricError::InvalidParameter {
                param: param_name.to_string(),
                reason: "must be positive".to_string(),
            }.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_point() {
        let coords = Vector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]);
        let point = ComplexPoint::new(coords);
        
        assert_eq!(point.dimension(), 2);
        assert!(point.chart.is_none());
    }

    #[test]
    fn test_distance() {
        let p1 = ComplexPoint::new(Vector::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let p2 = ComplexPoint::new(Vector::from_vec(vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]));
        
        let dist = p1.distance(&p2).unwrap();
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sheaf() {
        let mut sheaf = Sheaf::new("P1".to_string());
        let sections = Vector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]);
        sheaf.add_sections("U".to_string(), sections);
        
        let cohomology = sheaf.cohomology(0).unwrap();
        assert_eq!(cohomology.len(), 2);
    }

    #[test]
    fn test_cache() {
        let mut cache = ComputationCache::new();
        let matrix = Matrix::identity(3, 3);
        
        cache.store_matrix("test".to_string(), matrix.clone());
        let retrieved = cache.get_matrix("test").unwrap();
        
        assert_eq!(retrieved, &matrix);
        assert_eq!(cache.hit_count, 1);
        assert_eq!(cache.miss_count, 0);
    }

    #[test]
    fn test_validation() {
        use validation::*;
        
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(3, 2);
        assert!(validate_matrix_mult(&a, &b).is_ok());
        
        let c = Matrix::zeros(2, 2);
        assert!(validate_matrix_mult(&a, &c).is_err());
        
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(-1.0, "test").is_err());
    }
}