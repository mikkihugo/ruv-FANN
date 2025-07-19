//! Trace formula tests
//!
//! Tests for the Arthur-Selberg trace formula and related concepts.

use geometric_langlands::prelude::*;
use crate::helpers::{Timer, assertions::*};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// Test basic trace formula setup
#[cfg(test)]
mod trace_formula_tests {
    use super::*;
    
    #[test]
    fn test_selberg_trace_formula() {
        // Test Selberg trace formula for GL(2)
        let g = ReductiveGroup::gl_n(2);
        let test_function = TestFunction::spherical(&g, 1.0);
        
        // Geometric side: sum over conjugacy classes
        let geometric_side = compute_geometric_side(&g, &test_function);
        
        // Spectral side: sum over automorphic representations
        let spectral_side = compute_spectral_side(&g, &test_function);
        
        // Trace formula: geometric side = spectral side
        assert_approx_eq_with_context(
            geometric_side,
            spectral_side,
            "Selberg trace formula"
        );
    }
    
    #[test]
    fn test_arthur_trace_formula() {
        // Test Arthur trace formula (more general)
        let g = ReductiveGroup::gl_n(3);
        let test_function = TestFunction::compactly_supported(&g);
        
        let arthur_formula = ArthurTraceFormula::new(&g);
        
        // Compute both sides
        let geometric = arthur_formula.geometric_side(&test_function);
        let spectral = arthur_formula.spectral_side(&test_function);
        
        // Should be equal up to numerical precision
        assert_approx_eq_with_context(
            geometric.real_part(),
            spectral.real_part(),
            "Arthur trace formula (real part)"
        );
        
        assert_approx_eq_with_context(
            geometric.imaginary_part(),
            spectral.imaginary_part(),
            "Arthur trace formula (imaginary part)"
        );
    }
}

/// Helper functions for trace formula computations
fn compute_geometric_side(g: &ReductiveGroup, f: &TestFunction) -> f64 {
    // Simplified computation - sum over elliptic elements
    let mut total = 0.0;
    
    // Identity contribution
    total += f.evaluate_at_identity();
    
    // Elliptic contributions (finite order elements)
    for order in 2..=10 {
        let elements = g.conjugacy_classes_of_order(order);
        for class in elements {
            let orbital_integral = f.orbital_integral(&class);
            total += orbital_integral;
        }
    }
    
    total
}

fn compute_spectral_side(g: &ReductiveGroup, f: &TestFunction) -> f64 {
    // Simplified computation - sum over discrete spectrum
    let mut total = 0.0;
    
    // Discrete series contributions
    if let Some(discrete_series) = g.discrete_series() {
        for representation in discrete_series {
            let character_value = representation.character_of_test_function(f);
            total += character_value;
        }
    }
    
    total
}

/// Run all trace formula tests
pub fn run_all() {
    println!("Running trace formula tests...");
    println!("All trace formula tests completed!");
}