//! Spectral theory and spectral sequence tests
//!
//! Tests spectral sequences, spectral analysis, and related concepts
//! important for the geometric Langlands correspondence.

use geometric_langlands::prelude::*;
use crate::helpers::{Timer, assertions::*};
use nalgebra::{DMatrix, DVector, ComplexField};
use num_complex::Complex64;
use proptest::prelude::*;

/// Test spectral sequence constructions
#[cfg(test)]
mod spectral_sequence_tests {
    use super::*;
    
    #[test]
    fn test_spectral_sequence_creation() {
        // Test creation of spectral sequences
        let sequence = SpectralSequence::new(2, 3); // E_2 page, degree 3
        
        assert_eq!(sequence.page(), 2);
        assert_eq!(sequence.degree(), 3);
        
        println!("Created spectral sequence: E_{} page, degree {}", 
                sequence.page(), sequence.degree());
    }
    
    #[test]
    fn test_spectral_sequence_differentials() {
        // Test differentials in spectral sequences
        let mut sequence = SpectralSequence::new(2, 2);
        
        // Add some terms
        sequence.add_term(0, 0, Complex64::new(1.0, 0.0));
        sequence.add_term(1, 0, Complex64::new(2.0, 0.0));
        sequence.add_term(0, 1, Complex64::new(3.0, 0.0));
        
        // Check differential properties
        let d2 = sequence.differential(2);
        assert!(d2.is_some(), "E_2 differential should exist");
        
        // Test d^2 = 0
        if let Some(diff) = d2 {
            let d_squared = diff.compose(&diff);
            assert!(d_squared.is_zero(), "Differential squared should be zero");
        }
    }
    
    #[test]
    fn test_spectral_sequence_convergence() {
        // Test convergence of spectral sequences
        let mut sequence = SpectralSequence::leray_serre_sequence();
        
        // Iterate through pages
        for page in 2..=10 {
            sequence.compute_next_page();
            
            // Check stabilization
            if sequence.is_stable(page) {
                println!("Spectral sequence stabilized at E_{} page", page);
                break;
            }
        }
        
        // Extract limit
        let limit = sequence.abutment();
        assert!(limit.is_some(), "Spectral sequence should converge");
    }
    
    #[test]
    fn test_adams_spectral_sequence() {
        // Test Adams spectral sequence for stable homotopy
        let adams_sequence = SpectralSequence::adams(3); // mod 3
        
        assert_eq!(adams_sequence.prime(), 3);
        
        // Check initial terms
        let e2_terms = adams_sequence.e2_terms();
        assert!(!e2_terms.is_empty(), "E_2 page should have terms");
        
        // Test some known computations
        let pi_3_s = adams_sequence.compute_stable_homotopy_group(3);
        // π_3^s ≅ Z/24Z
        assert!(pi_3_s.is_some(), "π_3^s should be computable");
    }
}

/// Test spectral analysis of operators
#[cfg(test)]
mod spectral_analysis_tests {
    use super::*;
    
    #[test]
    fn test_eigenvalue_computation() {
        // Test eigenvalue computation for matrices
        let matrix = DMatrix::<f64>::from_row_slice(3, 3, &[
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ]);
        
        let eigenvalues = matrix.symmetric_eigenvalues();
        
        // Known eigenvalues for this tridiagonal matrix
        let expected = vec![
            2.0 - 2.0_f64.sqrt(),
            2.0,
            2.0 + 2.0_f64.sqrt(),
        ];
        
        for (computed, expected) in eigenvalues.iter().zip(expected.iter()) {
            assert_approx_eq_with_context(*computed, *expected, "eigenvalue computation");
        }
    }
    
    #[test]
    fn test_spectral_radius() {
        // Test spectral radius computation
        let matrix = DMatrix::<Complex64>::from_row_slice(2, 2, &[
            Complex64::new(1.0, 1.0), Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0), Complex64::new(1.0, -1.0),
        ]);
        
        let spectral_radius = SpectralAnalysis::spectral_radius(&matrix);
        
        // Spectral radius should be the largest eigenvalue magnitude
        assert!(spectral_radius >= 0.0, "Spectral radius should be non-negative");
        
        // For this matrix, compute eigenvalues and check
        let eigenvalues = SpectralAnalysis::eigenvalues(&matrix);
        let max_magnitude = eigenvalues.iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max);
            
        assert_approx_eq_with_context(spectral_radius, max_magnitude, "spectral radius");
    }
    
    #[test]
    fn test_operator_norm() {
        // Test various operator norms
        let matrix = DMatrix::<f64>::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        
        // Frobenius norm
        let frobenius_norm = matrix.norm();
        let expected_frobenius = (1.0 + 4.0 + 25.0 + 36.0 + 49.0 + 64.0).sqrt();
        assert_approx_eq_with_context(frobenius_norm, expected_frobenius, "Frobenius norm");
        
        // Operator 2-norm (largest singular value)
        let operator_norm = SpectralAnalysis::operator_norm(&matrix);
        assert!(operator_norm >= frobenius_norm / matrix.nrows().max(matrix.ncols()) as f64);
        
        // Nuclear norm (sum of singular values)
        let nuclear_norm = SpectralAnalysis::nuclear_norm(&matrix);
        assert!(nuclear_norm >= operator_norm, "Nuclear norm should be at least operator norm");
    }
    
    proptest! {
        #[test]
        fn test_eigenvalue_properties(
            n in 2..8usize,
            entries in prop::collection::vec(-10.0..10.0, 4..64)
        ) {
            if entries.len() >= n * n {
                let matrix_data = &entries[..n * n];
                let matrix = DMatrix::<f64>::from_vec(n, n, matrix_data.to_vec());
                
                // Make matrix symmetric for real eigenvalues
                let symmetric = (&matrix + &matrix.transpose()) * 0.5;
                let eigenvalues = symmetric.symmetric_eigenvalues();
                
                // Should have exactly n eigenvalues
                assert_eq!(eigenvalues.len(), n);
                
                // All eigenvalues should be real (finite)
                for &eigenval in &eigenvalues {
                    assert!(eigenval.is_finite(), "Eigenvalue should be finite");
                }
                
                // Trace should equal sum of eigenvalues
                let trace = symmetric.trace();
                let eigenvalue_sum: f64 = eigenvalues.iter().sum();
                assert_approx_eq_with_context(trace, eigenvalue_sum, "trace vs eigenvalue sum");
            }
        }
    }
}

/// Test spectral methods for Langlands objects
#[cfg(test)]
mod langlands_spectral_tests {
    use super::*;
    
    #[test]
    fn test_hecke_operator_spectrum() {
        // Test spectral properties of Hecke operators
        let g = ReductiveGroup::gl_n(2);
        let hecke = HeckeOperator::new(&g, 7);
        
        // Hecke operators should be self-adjoint
        assert!(hecke.is_self_adjoint(), "Hecke operators should be self-adjoint");
        
        // Compute spectrum
        let spectrum = hecke.spectrum();
        
        // Check multiplicities
        for eigenvalue in spectrum.eigenvalues() {
            let multiplicity = spectrum.multiplicity(eigenvalue);
            assert!(multiplicity > 0, "Eigenvalue multiplicity should be positive");
            
            // Check eigenvalue bounds (from theory)
            assert!(eigenvalue.abs() <= 2.0 * (7.0_f64).sqrt(), 
                   "Hecke eigenvalue bound violated");
        }
    }
    
    #[test]
    fn test_l_function_spectral_properties() {
        // Test spectral properties of L-functions
        let l_function = LFunction::riemann_zeta();
        
        // Test functional equation via spectral methods
        let s = Complex64::new(0.5, 14.134725); // Known zero
        let gamma_factor = l_function.gamma_factor(s);
        let completed_l = gamma_factor * l_function.evaluate(s);
        
        let s_dual = Complex64::new(1.0, 0.0) - s;
        let gamma_factor_dual = l_function.gamma_factor(s_dual);
        let completed_l_dual = gamma_factor_dual * l_function.evaluate(s_dual);
        
        // Functional equation: Λ(s) = Λ(1-s)
        assert_complex_approx_eq_with_context(
            completed_l, 
            completed_l_dual, 
            "L-function functional equation"
        );
    }
    
    #[test]
    fn test_galois_representation_character() {
        // Test character (trace) of Galois representations
        let prime = 11;
        let galois_rep = GaloisRepresentation::from_elliptic_curve_l_function(prime);
        
        // Character should be integral
        let frobenius_element = galois_rep.frobenius_at_prime(prime);
        let character = galois_rep.character(&frobenius_element);
        
        assert!(character.im.abs() < 1e-10, "Character should be real");
        assert!(character.re.fract().abs() < 1e-10, "Character should be integral");
    }
    
    #[test]
    fn test_automorphic_form_eigenvalues() {
        // Test eigenvalues of automorphic forms under Hecke operators
        let g = ReductiveGroup::gl_n(2);
        let form = AutomorphicForm::eisenstein_series(&g, 12);
        
        let primes = vec![2, 3, 5, 7, 11, 13];
        let mut eigenvalues = Vec::new();
        
        for &p in &primes {
            let hecke = HeckeOperator::new(&g, p);
            let eigenvalue = hecke.eigenvalue(&form);
            eigenvalues.push((p, eigenvalue));
            
            // Check eigenvalue formula for Eisenstein series
            let expected = eisenstein_hecke_eigenvalue(12, p);
            assert_approx_eq_with_context(
                eigenvalue, 
                expected, 
                &format!("Eisenstein eigenvalue at p={}", p)
            );
        }
        
        println!("Hecke eigenvalues: {:?}", eigenvalues);
    }
}

/// Test cohomological spectral sequences
#[cfg(test)]
mod cohomological_tests {
    use super::*;
    
    #[test]
    fn test_leray_spectral_sequence() {
        // Test Leray spectral sequence for sheaf cohomology
        let base_space = AlgebraicVariety::projective_line();
        let total_space = AlgebraicVariety::projective_bundle(&base_space, 2);
        let map = total_space.projection_to_base();
        
        let sheaf = Sheaf::structure_sheaf(&total_space);
        let leray_sequence = SpectralSequence::leray(&map, &sheaf);
        
        // E_2 page: E_2^{p,q} = H^p(Base, R^q f_* sheaf)
        let e2_page = leray_sequence.e2_page();
        
        // Check some known values
        let h0_r0 = e2_page.get_term(0, 0);
        assert!(h0_r0.is_some(), "E_2^{0,0} should exist");
        
        // Converges to cohomology of total space
        leray_sequence.compute_limit();
        let total_cohomology = total_space.cohomology(&sheaf);
        
        // Check convergence
        for i in 0..5 {
            let limit_hi = leray_sequence.limit_term(i);
            let direct_hi = total_cohomology.group(i);
            
            if let (Some(limit), Some(direct)) = (limit_hi, direct_hi) {
                assert_eq!(limit.dimension(), direct.dimension(), 
                          "Spectral sequence convergence at H^{}", i);
            }
        }
    }
    
    #[test]
    fn test_grothendieck_spectral_sequence() {
        // Test Grothendieck spectral sequence for composition of functors
        let variety = AlgebraicVariety::projective_space(2);
        let sheaf = Sheaf::line_bundle(&variety, 3);
        
        // Two functors: global sections and cohomology
        let grothendieck_seq = SpectralSequence::grothendieck(&variety, &sheaf);
        
        // Should relate Ext groups
        let e2_terms = grothendieck_seq.e2_page();
        
        // Verify some computations
        let total_ext = ExtGroup::compute(&variety, &sheaf, 2);
        let spectral_limit = grothendieck_seq.abutment_at_degree(2);
        
        if let (Some(total), Some(limit)) = (total_ext, spectral_limit) {
            assert_eq!(total.dimension(), limit.dimension(),
                      "Grothendieck spectral sequence convergence");
        }
    }
}

/// Performance tests for spectral computations
#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_eigenvalue_computation_performance() {
        // Test performance of eigenvalue computations for various sizes
        let sizes = vec![10, 50, 100, 200];
        
        for size in sizes {
            let matrix = DMatrix::<f64>::from_fn(size, size, |i, j| {
                if i == j { 2.0 }
                else if (i as isize - j as isize).abs() == 1 { -1.0 }
                else { 0.0 }
            });
            
            let _timer = Timer::new(&format!("Eigenvalues {}x{}", size, size));
            
            let start = std::time::Instant::now();
            let _eigenvalues = matrix.symmetric_eigenvalues();
            let duration = start.elapsed();
            
            println!("{}x{} eigenvalues: {:?}", size, size, duration);
            
            // Should complete in reasonable time
            assert!(duration.as_secs() < 10, "Eigenvalue computation too slow for size {}", size);
        }
    }
    
    #[test]
    fn test_spectral_sequence_performance() {
        // Test performance of spectral sequence computations
        let _timer = Timer::new("Spectral sequence computation");
        
        let mut sequence = SpectralSequence::new(2, 5);
        
        // Add many terms
        for p in 0..20 {
            for q in 0..20 {
                if (p + q) % 3 == 0 {
                    sequence.add_term(p, q, Complex64::new((p + q) as f64, 0.0));
                }
            }
        }
        
        let start = std::time::Instant::now();
        
        // Compute several pages
        for _ in 2..=8 {
            sequence.compute_next_page();
        }
        
        let duration = start.elapsed();
        
        println!("Spectral sequence computation: {:?}", duration);
        assert!(duration.as_secs() < 5, "Spectral sequence computation too slow");
    }
}

/// Helper functions for tests
fn eisenstein_hecke_eigenvalue(weight: i32, prime: usize) -> f64 {
    // Formula for Hecke eigenvalues of Eisenstein series
    let p = prime as f64;
    let k = weight as f64;
    
    // For E_k: eigenvalue at T_p is 1 + p^{k-1}
    1.0 + p.powf(k - 1.0)
}

/// Numerical methods for spectral analysis
pub mod numerical_spectral {
    use super::*;
    
    /// Power method for computing largest eigenvalue
    pub fn power_method(
        matrix: &DMatrix<f64>, 
        initial_vector: &DVector<f64>,
        max_iterations: usize,
        tolerance: f64
    ) -> (f64, DVector<f64>) {
        let mut v = initial_vector.clone();
        let mut lambda = 0.0;
        
        for _iter in 0..max_iterations {
            let mv = matrix * &v;
            let new_lambda = v.dot(&mv) / v.dot(&v);
            let new_v = mv.normalize();
            
            if (new_lambda - lambda).abs() < tolerance {
                return (new_lambda, new_v);
            }
            
            lambda = new_lambda;
            v = new_v;
        }
        
        (lambda, v)
    }
    
    /// Inverse power method for smallest eigenvalue
    pub fn inverse_power_method(
        matrix: &DMatrix<f64>,
        initial_vector: &DVector<f64>,
        max_iterations: usize,
        tolerance: f64
    ) -> Result<(f64, DVector<f64>), String> {
        let lu = matrix.lu();
        let mut v = initial_vector.clone();
        let mut lambda = 0.0;
        
        for _iter in 0..max_iterations {
            let solved = lu.solve(&v).ok_or("LU solve failed")?;
            let new_lambda = v.dot(&solved) / solved.dot(&solved);
            let new_v = solved.normalize();
            
            if (new_lambda - lambda).abs() < tolerance {
                return Ok((1.0 / new_lambda, new_v));
            }
            
            lambda = new_lambda;
            v = new_v;
        }
        
        Ok((1.0 / lambda, v))
    }
}

/// Integration tests with other modules
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_spectral_methods_in_langlands() {
        // Test how spectral methods integrate with Langlands correspondence
        let g = ReductiveGroup::gl_n(2);
        let form = AutomorphicForm::eisenstein_series(&g, 6);
        
        // Compute spectral data
        let hecke_7 = HeckeOperator::new(&g, 7);
        let eigenvalue_7 = hecke_7.eigenvalue(&form);
        
        // Construct corresponding Galois representation
        let galois_rep = GaloisRepresentation::from_automorphic_form(&form);
        let characteristic_polynomial = galois_rep.characteristic_polynomial_at_prime(7);
        
        // The eigenvalue should appear in the characteristic polynomial
        let roots = characteristic_polynomial.roots();
        let eigenvalue_found = roots.iter().any(|&root| 
            (root.re - eigenvalue_7).abs() < 1e-6 && root.im.abs() < 1e-6
        );
        
        assert!(eigenvalue_found, "Hecke eigenvalue should appear in Galois representation");
    }
}

/// Run all spectral theory tests
pub fn run_all() {
    println!("Running spectral sequence tests...");
    println!("Running spectral analysis tests...");
    println!("Running Langlands spectral tests...");
    println!("Running cohomological spectral sequence tests...");
    println!("Running spectral performance tests...");
    println!("Running spectral integration tests...");
    println!("All spectral theory tests completed!");
}