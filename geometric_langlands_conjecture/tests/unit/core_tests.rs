//! Unit tests for core mathematical structures
//!
//! Tests the fundamental mathematical types and operations that form
//! the foundation of the Geometric Langlands implementation.

use geometric_langlands::prelude::*;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Test basic field operations
#[cfg(test)]
mod field_tests {
    use super::*;
    
    #[test]
    fn test_field_creation() {
        let field = Field::rationals();
        assert_eq!(field.characteristic, 0);
        assert_eq!(field.degree, 1);
        
        let finite_field = Field::finite_field(7);
        assert_eq!(finite_field.characteristic, 7);
        assert_eq!(finite_field.degree, 1);
    }
    
    #[test]
    fn test_field_extension() {
        let base = Field::rationals();
        let extension = Field::extension(&base, 2);
        assert_eq!(extension.characteristic, 0);
        assert_eq!(extension.degree, 2);
    }
}

/// Test group theory implementations
#[cfg(test)]
mod group_tests {
    use super::*;
    
    #[test]
    fn test_group_creation() {
        let group = Group::new(3, true, false);
        assert_eq!(group.dimension, 3);
        assert!(group.is_connected);
        assert!(!group.is_reductive);
    }
    
    #[test]
    fn test_reductive_group_gl_n() {
        let gl2 = ReductiveGroup::gl_n(2);
        assert_eq!(gl2.rank, 2);
        assert_eq!(gl2.dimension, 4);
        assert_eq!(gl2.root_system, "A1");
        
        let gl3 = ReductiveGroup::gl_n(3);
        assert_eq!(gl3.rank, 3);
        assert_eq!(gl3.dimension, 9);
        assert_eq!(gl3.root_system, "A2");
    }
    
    #[test]
    fn test_special_linear_group() {
        let sl3 = ReductiveGroup::sl_n(3);
        assert_eq!(sl3.rank, 2);
        assert_eq!(sl3.dimension, 8);
        assert_eq!(sl3.root_system, "A2");
    }
    
    #[test]
    fn test_orthogonal_group() {
        let so3 = ReductiveGroup::so_n(3);
        assert_eq!(so3.rank, 1);
        assert_eq!(so3.dimension, 3);
        assert_eq!(so3.root_system, "B1");
        
        let so4 = ReductiveGroup::so_n(4);
        assert_eq!(so4.rank, 2);
        assert_eq!(so4.dimension, 6);
        assert_eq!(so4.root_system, "D2");
    }
    
    #[test]
    fn test_symplectic_group() {
        let sp4 = ReductiveGroup::sp_2n(2);
        assert_eq!(sp4.rank, 2);
        assert_eq!(sp4.dimension, 10);
        assert_eq!(sp4.root_system, "C2");
    }
}

/// Test matrix representations
#[cfg(test)]
mod matrix_tests {
    use super::*;
    
    #[test]
    fn test_matrix_representation() {
        let group = ReductiveGroup::gl_n(2);
        let identity = MatrixRepresentation::identity(group.clone(), 2);
        
        assert_eq!(identity.matrix.nrows(), 2);
        assert_eq!(identity.matrix.ncols(), 2);
        assert_eq!(identity.group, group);
    }
    
    #[test]
    fn test_matrix_composition() {
        let group = ReductiveGroup::gl_n(2);
        let rep1 = MatrixRepresentation::identity(group.clone(), 2);
        let rep2 = MatrixRepresentation::identity(group, 2);
        
        let composition = rep1.compose(&rep2);
        assert!(composition.is_ok());
        
        let result = composition.unwrap();
        assert_eq!(result.matrix, DMatrix::identity(2, 2));
    }
}

/// Test Lie algebra functionality
#[cfg(test)]
mod lie_algebra_tests {
    use super::*;
    
    #[test]
    fn test_lie_algebra_from_group() {
        let group = ReductiveGroup::gl_n(3);
        let lie_algebra = group.lie_algebra();
        
        assert_eq!(lie_algebra.dimension, group.dimension);
        assert_eq!(lie_algebra.root_system, group.root_system);
        assert_eq!(lie_algebra.base_field, group.base_field);
    }
}

/// Test ring structures
#[cfg(test)]
mod ring_tests {
    use super::*;
    
    #[test]
    fn test_polynomial_ring() {
        let field = Field::rationals();
        let ring = Ring::polynomial_ring(field.clone(), 3);
        
        assert!(ring.is_commutative);
        assert!(ring.has_unity);
        assert_eq!(ring.base_field, Some(field));
    }
}

/// Run all core tests
pub fn run_all() {
    println!("Running core mathematical structure tests...");
    // Tests will be automatically run by the test framework
}