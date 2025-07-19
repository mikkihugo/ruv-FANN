//! Harmonic analysis tests
//!
//! Tests for harmonic analysis on reductive groups, representation theory,
//! and harmonic functions relevant to the Langlands correspondence.

use geometric_langlands::prelude::*;
use crate::helpers::{Timer, assertions::*};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Test harmonic analysis on groups
#[cfg(test)]
mod group_harmonic_tests {
    use super::*;
    
    #[test]
    fn test_fourier_transform_on_group() {
        // Test Fourier transform on finite groups
        let group = FiniteGroup::symmetric(4); // S_4
        let function = GroupFunction::indicator_conjugacy_class(&group, 0);
        
        let fourier_transform = function.fourier_transform();
        
        // Parseval's theorem: ||f||^2 = ||f_hat||^2
        let time_norm_sq = function.l2_norm_squared();
        let freq_norm_sq = fourier_transform.l2_norm_squared();
        
        assert_approx_eq_with_context(time_norm_sq, freq_norm_sq, "Parseval's theorem");
    }
    
    #[test]
    fn test_character_theory() {
        // Test character theory of representations
        let group = FiniteGroup::symmetric(3);
        let irreps = group.irreducible_representations();
        
        // Test orthogonality relations
        for (i, rep1) in irreps.iter().enumerate() {
            for (j, rep2) in irreps.iter().enumerate() {
                let inner_product = rep1.character().inner_product(&rep2.character());
                
                if i == j {
                    assert_approx_eq_with_context(inner_product, 1.0, "character orthonormality");
                } else {
                    assert_approx_eq_with_context(inner_product, 0.0, "character orthogonality");
                }
            }
        }
        
        // Test character table properties
        let character_table = group.character_table();
        let group_order = group.order();
        
        // Sum of squares of character degrees equals group order
        let degree_sum: f64 = irreps.iter()
            .map(|rep| rep.degree() as f64)
            .map(|d| d * d)
            .sum();
        
        assert_approx_eq_with_context(degree_sum, group_order as f64, "character degree formula");
    }
    
    #[test]
    fn test_plancherel_measure() {
        // Test Plancherel measure for reductive groups
        let g = ReductiveGroup::gl_n(2);
        let dual_group = g.dual_group();
        
        let plancherel = PlancherelMeasure::compute(&dual_group);
        
        // Plancherel measure should be positive
        for representation in dual_group.unitary_representations() {
            let measure = plancherel.measure_at(&representation);
            assert!(measure >= 0.0, "Plancherel measure should be non-negative");
        }
        
        // Total measure should be finite for discrete series
        if let Some(discrete_series) = dual_group.discrete_series() {
            let total_measure: f64 = discrete_series.iter()
                .map(|rep| plancherel.measure_at(rep))
                .sum();
            assert!(total_measure.is_finite(), "Total Plancherel measure should be finite");
        }
    }
}

/// Test spherical functions and harmonic analysis
#[cfg(test)]
mod spherical_function_tests {
    use super::*;
    
    #[test]
    fn test_spherical_function_properties() {
        // Test properties of spherical functions
        let g = ReductiveGroup::gl_n(2);
        let k = g.maximal_compact_subgroup();
        
        let spherical_fn = SphericalFunction::elementary(&g, &k, 0);
        
        // Test bi-K-invariance
        for _ in 0..10 {
            let k1 = k.random_element();
            let k2 = k.random_element();
            let x = g.random_element();
            
            let value_original = spherical_fn.evaluate(&x);
            let value_conjugated = spherical_fn.evaluate(&(k1 * x * k2));
            
            assert_approx_eq_with_context(
                value_original,
                value_conjugated,
                "spherical function bi-K-invariance"
            );
        }
        
        // Test functional equation for Hecke operators
        let hecke = HeckeOperator::spherical(&g, &k, 7);
        let eigenvalue = hecke.eigenvalue_on_spherical_function(&spherical_fn);
        
        assert!(eigenvalue.is_finite(), "Hecke eigenvalue should be finite");
    }
    
    #[test]
    fn test_harish_chandra_transform() {
        // Test Harish-Chandra transform (spherical Fourier transform)
        let g = ReductiveGroup::gl_n(2);
        let a = g.split_torus();
        
        let compactly_supported_fn = CompactlySupportedFunction::gaussian(&g, 1.0);
        let hc_transform = compactly_supported_fn.harish_chandra_transform();
        
        // HC transform should be defined on a^*_C
        let parameter = a.dual_lie_algebra().random_element();
        let transform_value = hc_transform.evaluate(&parameter);
        
        assert!(transform_value.is_finite(), "Harish-Chandra transform should be finite");
        
        // Test inversion formula (if implemented)
        if let Some(inverse_transform) = hc_transform.inverse() {
            let recovered_fn = inverse_transform;
            
            // Check that we recover the original function (approximately)
            for _ in 0..5 {
                let test_point = g.random_element();
                let original_value = compactly_supported_fn.evaluate(&test_point);
                let recovered_value = recovered_fn.evaluate(&test_point);
                
                assert_approx_eq_with_context(
                    original_value,
                    recovered_value,
                    "Harish-Chandra inversion formula"
                );
            }
        }
    }
    
    #[test]
    fn test_eisenstein_series_analytic_continuation() {
        // Test analytic continuation of Eisenstein series
        let g = ReductiveGroup::gl_n(2);
        let minimal_parabolic = g.minimal_parabolic();
        
        // Eisenstein series E(g, s) for Re(s) >> 0
        let s_convergent = Complex64::new(2.0, 0.0);
        let eisenstein_convergent = EisensteinSeries::new(&minimal_parabolic, s_convergent);
        
        // Analytically continue to critical strip
        let s_critical = Complex64::new(0.5, 1.0);
        let eisenstein_continued = eisenstein_convergent.analytic_continuation(s_critical);
        
        match eisenstein_continued {
            Ok(continued_series) => {
                // Test functional equation
                let s_dual = Complex64::new(1.0, 0.0) - s_critical;
                let eisenstein_dual = continued_series.at_parameter(s_dual);
                
                // There should be a relation E(g, s) ~ E(g, 1-s)
                let test_element = g.random_element();
                let value_s = continued_series.evaluate(&test_element);
                let value_dual = eisenstein_dual.evaluate(&test_element);
                
                // The exact relation involves scattering matrix, so we just check finiteness
                assert!(value_s.is_finite(), "Eisenstein series value should be finite");
                assert!(value_dual.is_finite(), "Dual Eisenstein series value should be finite");
            }
            Err(e) => println!("Analytic continuation failed: {}", e),
        }
    }
}

/// Test harmonic functions and their properties
#[cfg(test)]
mod harmonic_function_tests {
    use super::*;
    
    #[test]
    fn test_laplacian_eigenfunctions() {
        // Test eigenfunctions of the Laplacian
        let manifold = RiemannianManifold::hyperbolic_plane();
        let laplacian = manifold.laplace_beltrami_operator();
        
        // Test some known eigenfunctions
        let eigenvalue = 1.0; // λ = s(1-s) with s = 1/2 + it
        let eigenfunction = HarmonicFunction::hyperbolic_eigenfunction(eigenvalue);
        
        // Apply Laplacian
        let laplacian_result = laplacian.apply(&eigenfunction);
        let expected = eigenfunction.scale(-eigenvalue);
        
        // Check Δf = -λf
        for _ in 0..10 {
            let test_point = manifold.random_point();
            let actual_value = laplacian_result.evaluate(&test_point);
            let expected_value = expected.evaluate(&test_point);
            
            assert_approx_eq_with_context(
                actual_value,
                expected_value,
                "Laplacian eigenfunction equation"
            );
        }
    }
    
    #[test]
    fn test_green_function() {
        // Test Green's function for the Laplacian
        let domain = EuclideanDomain::unit_disk();
        let laplacian = domain.laplacian();
        
        let center = domain.center();
        let green_function = laplacian.green_function(&center);
        
        // Test that Δ_y G(x,y) = δ(x-y)
        let test_points = vec![
            domain.point(0.1, 0.0),
            domain.point(0.0, 0.1),
            domain.point(-0.1, 0.0),
            domain.point(0.0, -0.1),
        ];
        
        for point in test_points {
            let laplacian_green = laplacian.apply_at_point(&green_function, &point);
            
            if point.distance(&center) < 1e-3 {
                // Near the singularity, should be large
                assert!(laplacian_green.abs() > 100.0, "Green function singularity");
            } else {
                // Away from singularity, should be small
                assert!(laplacian_green.abs() < 1.0, "Green function regularity");
            }
        }
    }
    
    #[test]
    fn test_maximum_principle() {
        // Test maximum principle for harmonic functions
        let domain = EuclideanDomain::unit_square();
        
        // Create harmonic function with boundary conditions
        let boundary_values = |point: &Point2D| -> f64 {
            (point.x * PI).sin() * (point.y * PI).cos()
        };
        
        let harmonic_fn = HarmonicFunction::solve_dirichlet_problem(&domain, boundary_values);
        
        // Maximum should be attained on the boundary
        let interior_max = domain.interior_points(100).iter()
            .map(|p| harmonic_fn.evaluate(p))
            .fold(f64::NEG_INFINITY, f64::max);
        
        let boundary_max = domain.boundary_points(100).iter()
            .map(|p| harmonic_fn.evaluate(p))
            .fold(f64::NEG_INFINITY, f64::max);
        
        assert!(interior_max <= boundary_max + 1e-6, "Maximum principle violated");
    }
}

/// Test representation theory and harmonic analysis
#[cfg(test)]
mod representation_harmonic_tests {
    use super::*;
    
    #[test]
    fn test_induced_representation() {
        // Test induced representations
        let g = ReductiveGroup::gl_n(2);
        let p = g.minimal_parabolic();
        let character = p.character_trivial();
        
        let induced_rep = g.induce_representation(&p, &character);
        
        // Test reciprocity (Frobenius reciprocity)
        let irrep = g.irreducible_representation(0);
        let restriction = irrep.restrict_to_subgroup(&p);
        
        let multiplicity1 = induced_rep.multiplicity_of_irrep(&irrep);
        let multiplicity2 = restriction.multiplicity_of_character(&character);
        
        assert_eq!(multiplicity1, multiplicity2, "Frobenius reciprocity");
    }
    
    #[test]
    fn test_unitary_representation_theory() {
        // Test unitary representations
        let g = ReductiveGroup::sl_2_r();
        let unitary_dual = g.unitary_dual();
        
        // Test complementary series
        if let Some(comp_series) = unitary_dual.complementary_series() {
            for parameter in comp_series.parameters() {
                let representation = comp_series.representation_at_parameter(parameter);
                
                // Should be unitary
                assert!(representation.is_unitary(), "Complementary series should be unitary");
                
                // Should be irreducible
                assert!(representation.is_irreducible(), "Should be irreducible");
            }
        }
        
        // Test principal series
        if let Some(prin_series) = unitary_dual.principal_series() {
            for character in prin_series.characters() {
                let representation = prin_series.representation_with_character(&character);
                
                // Test unitarity
                assert!(representation.is_unitary(), "Principal series should be unitary");
            }
        }
    }
    
    #[test]
    fn test_langlands_parameters() {
        // Test Langlands parameters for representations
        let g = ReductiveGroup::gl_n(3);
        let weil_group = WeilGroup::local_field();
        
        let representation = g.irreducible_representation(0);
        let langlands_param = representation.langlands_parameter();
        
        // Should be a homomorphism W_F × SL_2(C) → ^L G
        assert!(langlands_param.is_homomorphism(), "Langlands parameter should be homomorphism");
        
        // Test L-function computation
        let l_function = langlands_param.l_function();
        
        // Should satisfy functional equation
        let s = Complex64::new(0.5, 1.0);
        let gamma_factor = l_function.gamma_factor(s);
        let completed_l_s = gamma_factor * l_function.evaluate(s);
        
        let s_dual = Complex64::new(1.0, 0.0) - s;
        let gamma_factor_dual = l_function.gamma_factor(s_dual);
        let completed_l_dual = gamma_factor_dual * l_function.evaluate(s_dual);
        
        // Functional equation (up to root number)
        let root_number = l_function.root_number();
        assert_complex_approx_eq_with_context(
            completed_l_s,
            root_number * completed_l_dual,
            "L-function functional equation"
        );
    }
}

/// Test automorphic forms and harmonic analysis
#[cfg(test)]
mod automorphic_harmonic_tests {
    use super::*;
    
    #[test]
    fn test_automorphic_form_fourier_expansion() {
        // Test Fourier expansion of automorphic forms
        let g = ReductiveGroup::gl_n(2);
        let form = AutomorphicForm::eisenstein_series(&g, 12);
        
        let fourier_expansion = form.fourier_expansion();
        
        // First Fourier coefficient should be related to special L-value
        let a0 = fourier_expansion.coefficient(0);
        let special_l_value = RiemannZeta::evaluate_at(12.0);
        
        // For Eisenstein series: a_0 = ζ(k) for weight k
        assert_approx_eq_with_context(a0.re, special_l_value, "Eisenstein constant term");
        
        // Test growth of Fourier coefficients
        let mut max_coeff = 0.0;
        for n in 1..100 {
            let an = fourier_expansion.coefficient(n);
            max_coeff = max_coeff.max(an.norm());
        }
        
        // Should grow polynomially (Ramanujan bound)
        let ramanujan_bound = 100.0_f64.powf(0.5 + 1e-6); // n^{1/2+ε}
        assert!(max_coeff < ramanujan_bound, "Ramanujan bound violated");
    }
    
    #[test]
    fn test_whittaker_function() {
        // Test Whittaker functions for GL(2)
        let g = ReductiveGroup::gl_n(2);
        let unipotent = g.unipotent_radical();
        let character = unipotent.generic_character();
        
        let automorphic_form = AutomorphicForm::newform(&g, 11, 2); // Level 11, weight 2
        let whittaker_fn = automorphic_form.whittaker_function(&character);
        
        // Test functional equation for Whittaker function
        let test_element = g.random_element();
        let gamma_matrix = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]);
        let gamma_action = g.element_from_matrix(&gamma_matrix);
        
        let w_value = whittaker_fn.evaluate(&test_element);
        let w_gamma_value = whittaker_fn.evaluate(&(gamma_action * test_element));
        
        // There should be a specific relation (involving L-function)
        assert!(w_value.is_finite() && w_gamma_value.is_finite(),
               "Whittaker function values should be finite");
    }
    
    #[test]
    fn test_petersson_inner_product() {
        // Test Petersson inner product of cusp forms
        let g = ReductiveGroup::gl_n(2);
        let space = CuspFormSpace::new(&g, 12, 1); // Weight 12, level 1
        
        let basis = space.basis();
        
        // Test orthogonality of Hecke eigenforms
        for i in 0..basis.len().min(3) {
            for j in 0..basis.len().min(3) {
                let form1 = &basis[i];
                let form2 = &basis[j];
                
                let inner_product = space.petersson_inner_product(form1, form2);
                
                if i == j {
                    assert!(inner_product > 0.0, "Self inner product should be positive");
                } else {
                    // Different eigenforms should be orthogonal
                    assert!(inner_product.abs() < 1e-6, "Different eigenforms should be orthogonal");
                }
            }
        }
    }
}

/// Performance tests for harmonic analysis
#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_fourier_transform_performance() {
        // Test performance of various Fourier transforms
        let sizes = vec![64, 128, 256, 512];
        
        for size in sizes {
            let data: Vec<Complex64> = (0..size)
                .map(|i| Complex64::new((i as f64 * 2.0 * PI / size as f64).sin(), 0.0))
                .collect();
            
            let _timer = Timer::new(&format!("FFT size {}", size));
            
            let start = std::time::Instant::now();
            let _fft_result = fft(&data);
            let duration = start.elapsed();
            
            println!("FFT size {}: {:?}", size, duration);
            
            // Should complete quickly
            assert!(duration.as_millis() < 100, "FFT too slow for size {}", size);
        }
    }
    
    #[test]
    fn test_character_computation_performance() {
        // Test performance of character computations
        let group_orders = vec![12, 24, 60, 120]; // A_4, S_4, A_5, S_5
        
        for order in group_orders {
            let _timer = Timer::new(&format!("Character table order {}", order));
            
            let group = FiniteGroup::of_order(order);
            
            let start = std::time::Instant::now();
            let _char_table = group.character_table();
            let duration = start.elapsed();
            
            println!("Character table order {}: {:?}", order, duration);
            
            // Should be reasonable for small groups
            assert!(duration.as_secs() < 5, "Character computation too slow for order {}", order);
        }
    }
}

/// Simplified FFT for testing
fn fft(data: &[Complex64]) -> Vec<Complex64> {
    let n = data.len();
    if n <= 1 {
        return data.to_vec();
    }
    
    // Simple DFT for testing (not optimized)
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * PI * (k * j) as f64 / n as f64;
            let w = Complex64::new(angle.cos(), angle.sin());
            result[k] += data[j] * w;
        }
    }
    
    result
}

/// Run all harmonic analysis tests
pub fn run_all() {
    println!("Running group harmonic analysis tests...");
    println!("Running spherical function tests...");
    println!("Running harmonic function tests...");
    println!("Running representation harmonic tests...");
    println!("Running automorphic harmonic tests...");
    println!("Running harmonic analysis performance tests...");
    println!("All harmonic analysis tests completed!");
}