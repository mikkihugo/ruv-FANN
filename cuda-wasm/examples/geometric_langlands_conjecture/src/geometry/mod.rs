// Geometry module for Geometric Langlands Conjecture
// Implements sheaves, bundles, and moduli spaces

use serde::{Serialize, Deserialize};

/// Trait for geometric objects in the Langlands program
pub trait GeometricObject: std::fmt::Debug + Clone {
    /// Dimension of the geometric object
    fn dimension(&self) -> usize;
    
    /// Check if the object is smooth
    fn is_smooth(&self) -> bool;
    
    /// Compute geometric invariants
    fn invariants(&self) -> Vec<f64>;
}

/// Base trait for algebraic varieties
pub trait AlgebraicVariety: GeometricObject {
    /// Genus of the variety (for curves)
    fn genus(&self) -> Option<usize>;
    
    /// Picard group rank
    fn picard_rank(&self) -> usize;
}

/// Configuration for geometric computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryConfig {
    /// Precision for numerical computations
    pub precision: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Enable parallel computation
    pub use_parallel: bool,
    /// Cache intermediate results
    pub enable_cache: bool,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            precision: 1e-10,
            max_iterations: 1000,
            use_parallel: true,
            enable_cache: true,
        }
    }
}

/// Simple geometric object implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleGeometry {
    pub dimension: usize,
    pub smooth: bool,
    pub invariants: Vec<f64>,
}

impl GeometricObject for SimpleGeometry {
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn is_smooth(&self) -> bool {
        self.smooth
    }
    
    fn invariants(&self) -> Vec<f64> {
        self.invariants.clone()
    }
}

impl AlgebraicVariety for SimpleGeometry {
    fn genus(&self) -> Option<usize> {
        if self.dimension == 1 {
            Some(self.invariants.get(0).map(|x| *x as usize).unwrap_or(0))
        } else {
            None
        }
    }
    
    fn picard_rank(&self) -> usize {
        self.invariants.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_config() {
        let config = GeometryConfig::default();
        assert!(config.precision < 1e-9);
        assert!(config.use_parallel);
    }
    
    #[test]
    fn test_simple_geometry() {
        let geom = SimpleGeometry {
            dimension: 2,
            smooth: true,
            invariants: vec![1.0, 2.0],
        };
        
        assert_eq!(geom.dimension(), 2);
        assert!(geom.is_smooth());
        assert_eq!(geom.picard_rank(), 2);
    }
}