//! # Geometric Langlands Conjecture Implementation
//!
//! This crate provides a comprehensive implementation of the Geometric Langlands
//! correspondence, connecting automorphic forms with Galois representations
//! through advanced mathematical computations.
//!
//! ## Features
//!
//! - High-performance GPU acceleration via CUDA
//! - WebAssembly support for browser deployment
//! - Comprehensive mathematical type system
//! - Parallel algorithms for large-scale computations
//!
//! ## Example
//!
//! ```rust,no_run
//! use geometric_langlands::prelude::*;
//! 
//! // Create a reductive group
//! let g = ReductiveGroup::gl_n(3);
//! 
//! // Construct an automorphic form
//! let form = AutomorphicForm::eisenstein_series(&g, 2);
//! 
//! // Apply Hecke operator
//! let hecke = HeckeOperator::new(&g, 5);
//! let eigenform = hecke.apply(&form);
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core mathematical structures
pub mod core;

// Automorphic forms and representations
pub mod automorphic;

// Galois representations and l-adic sheaves
pub mod galois;

// Category theory implementations
pub mod category;

// Sheaf theory and cohomology
pub mod sheaf;

// Representation theory
pub mod representation;

// Harmonic analysis
pub mod harmonic;

// Spectral theory
pub mod spectral;

// Trace formulas
pub mod trace;

// Main Langlands correspondence
pub mod langlands;

// Performance modules
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod cuda;

#[cfg(feature = "wasm")]
#[cfg_attr(docsrs, doc(cfg(feature = "wasm")))]
pub mod wasm;

// Utilities
pub mod utils;

// Benchmarking utilities
#[cfg(any(test, feature = "bench"))]
pub mod benchmarks;

// Error types
mod error;
pub use error::{Error, Result};

// Re-export commonly used items
pub mod prelude {
    //! Common imports for users of this crate
    
    pub use crate::core::{
        Field, Group, Ring,
        AlgebraicVariety, Scheme, ModuliSpace,
        ReductiveGroup, LieAlgebra, MatrixRepresentation,
    };
    
    pub use crate::automorphic::{
        AutomorphicForm, AutomorphicRepresentation,
        HeckeOperator, EisensteinSeries,
    };
    
    pub use crate::galois::{
        GaloisRepresentation, LocalSystem,
        LAdic, PerverseSheaf,
    };
    
    pub use crate::error::{Error, Result};
}

// Version information
/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
        assert_eq!(NAME, "geometric-langlands");
    }
}