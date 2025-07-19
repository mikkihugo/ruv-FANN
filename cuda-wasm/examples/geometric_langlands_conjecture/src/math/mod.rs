//! Core mathematical structures for the Geometric Langlands Conjecture
//! 
//! This module provides the fundamental mathematical abstractions including:
//! - Category theory framework
//! - Sheaf cohomology
//! - Bundle theory
//! - Morphism types with composition laws

use std::marker::PhantomData;
use std::hash::Hash;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};

pub mod category;
pub mod sheaf;
pub mod bundle;
pub mod morphism;
pub mod validation;

// Re-export core types
pub use category::{Category, Object, CategoryError};
pub use morphism::{Morphism, MorphismComposition, MorphismError};
pub use sheaf::{Sheaf, Presheaf, SheafCohomology};
pub use bundle::{VectorBundle, PrincipalBundle, Connection};
pub use validation::{MathValidator, ValidationResult};

/// Core mathematical trait that all objects must implement
pub trait MathObject: Debug + Clone + Serialize + for<'de> Deserialize<'de> {
    /// Unique identifier for the mathematical object
    type Id: Eq + Hash + Clone + Debug;
    
    /// Get the unique identifier
    fn id(&self) -> &Self::Id;
    
    /// Validate the mathematical properties of this object
    fn validate(&self) -> ValidationResult;
    
    /// Get a human-readable description
    fn description(&self) -> String;
}

/// Trait for objects that can be compared for mathematical equivalence
pub trait MathEquivalence {
    /// Check if two objects are mathematically equivalent
    /// This may be weaker than equality (e.g., isomorphism vs equality)
    fn is_equivalent(&self, other: &Self) -> bool;
}

/// Trait for objects that support a group action
pub trait GroupAction<G> {
    /// Apply a group element to transform this object
    fn act(&self, g: &G) -> Self;
    
    /// Check if the action preserves the structure (default implementation)
    fn is_equivariant(&self, _g: &G, _h: &G) -> bool {
        // This would need to be implemented for specific group types
        // For now, we assume the action is equivariant
        true
    }
}

/// Trait for objects with a notion of dimension
pub trait Dimensional {
    /// Get the dimension of this object
    fn dimension(&self) -> usize;
    
    /// Check if this is finite-dimensional
    fn is_finite_dimensional(&self) -> bool {
        true // Default to finite
    }
}

/// Result type for mathematical operations
pub type MathResult<T> = Result<T, MathError>;

/// Errors that can occur in mathematical operations
#[derive(Debug, thiserror::Error)]
pub enum MathError {
    #[error("Category error: {0}")]
    Category(#[from] CategoryError),
    
    #[error("Morphism error: {0}")]
    Morphism(#[from] MorphismError),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Mathematical constraint violated: {0}")]
    ConstraintViolation(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Configuration for mathematical computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathConfig {
    /// Tolerance for numerical comparisons
    pub epsilon: f64,
    
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    
    /// Enable caching of intermediate results
    pub enable_caching: bool,
    
    /// Validation level (0 = none, 1 = basic, 2 = full)
    pub validation_level: u8,
}

impl Default for MathConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            max_iterations: 1000,
            enable_caching: true,
            validation_level: 1,
        }
    }
}

/// Initialize the mathematical framework
pub fn init_math_framework(config: MathConfig) -> MathResult<()> {
    // Initialize global configuration
    log::info!("Initializing mathematical framework with config: {:?}", config);
    
    // Validate configuration
    if config.epsilon <= 0.0 {
        return Err(MathError::ValidationFailed(
            "Epsilon must be positive".to_string()
        ));
    }
    
    if config.validation_level > 2 {
        return Err(MathError::ValidationFailed(
            "Validation level must be 0, 1, or 2".to_string()
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_math_config_default() {
        let config = MathConfig::default();
        assert_eq!(config.epsilon, 1e-10);
        assert_eq!(config.max_iterations, 1000);
        assert!(config.enable_caching);
        assert_eq!(config.validation_level, 1);
    }
    
    #[test]
    fn test_init_math_framework() {
        let config = MathConfig::default();
        assert!(init_math_framework(config).is_ok());
        
        let bad_config = MathConfig {
            epsilon: -1.0,
            ..Default::default()
        };
        assert!(init_math_framework(bad_config).is_err());
    }
}