//! Test fixtures and known mathematical examples
//!
//! This module provides well-known mathematical objects, test data,
//! and reference implementations for validation testing.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;
use geometric_langlands::prelude::*;

/// Mathematical constants used in testing
pub mod constants {
    pub const PI: f64 = std::f64::consts::PI;
    pub const E: f64 = std::f64::consts::E;
    pub const EULER_GAMMA: f64 = 0.5772156649015329;
    pub const GOLDEN_RATIO: f64 = 1.618033988749894;
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    pub const LN_2: f64 = std::f64::consts::LN_2;
    
    /// Riemann zeta values
    pub const ZETA_2: f64 = 1.6449340668482264; // π²/6
    pub const ZETA_4: f64 = 1.0823232337111381; // π⁴/90
    pub const ZETA_6: f64 = 1.0173430619844491; // π⁶/945
    
    /// Special values for testing
    pub const TEST_EPSILON: f64 = 1e-12;
    pub const LARGE_EPSILON: f64 = 1e-8;
}

/// Well-known matrices for testing
pub mod matrices {
    use super::*;
    
    /// Pauli matrices (fundamental 2x2 matrices)
    pub struct PauliMatrices;
    
    impl PauliMatrices {
        pub fn sigma_0() -> DMatrix<Complex64> {
            DMatrix::identity(2, 2)
        }
        
        pub fn sigma_x() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        pub fn sigma_y() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        pub fn sigma_z() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ])
        }
    }
    
    /// Special unitary matrices
    pub struct SpecialMatrices;
    
    impl SpecialMatrices {
        /// 2x2 rotation matrix
        pub fn rotation_2d(theta: f64) -> DMatrix<f64> {
            DMatrix::from_row_slice(2, 2, &[
                theta.cos(), -theta.sin(),
                theta.sin(), theta.cos(),
            ])
        }
        
        /// Hadamard matrix (2x2)
        pub fn hadamard_2() -> DMatrix<f64> {
            let sqrt_2_inv = 1.0 / constants::SQRT_2;
            DMatrix::from_row_slice(2, 2, &[
                sqrt_2_inv, sqrt_2_inv,
                sqrt_2_inv, -sqrt_2_inv,
            ])
        }
        
        /// Discrete Fourier Transform matrix
        pub fn dft_matrix(n: usize) -> DMatrix<Complex64> {
            let mut matrix = DMatrix::zeros(n, n);
            let omega = Complex64::new(0.0, -2.0 * constants::PI / n as f64).exp();
            
            for i in 0..n {
                for j in 0..n {
                    matrix[(i, j)] = omega.powf((i * j) as f64) / (n as f64).sqrt();
                }
            }
            
            matrix
        }
        
        /// Vandermonde matrix
        pub fn vandermonde(points: &[f64]) -> DMatrix<f64> {
            let n = points.len();
            let mut matrix = DMatrix::zeros(n, n);
            
            for i in 0..n {
                for j in 0..n {
                    matrix[(i, j)] = points[i].powi(j as i32);
                }
            }
            
            matrix
        }
    }
    
    /// Test matrices with known properties
    pub struct TestMatrices;
    
    impl TestMatrices {
        /// Symmetric tridiagonal matrix with known eigenvalues
        pub fn symmetric_tridiagonal(n: usize) -> DMatrix<f64> {
            let mut matrix = DMatrix::zeros(n, n);
            
            // Main diagonal: 2
            for i in 0..n {
                matrix[(i, i)] = 2.0;
            }
            
            // Super/sub diagonals: -1
            for i in 0..n-1 {
                matrix[(i, i+1)] = -1.0;
                matrix[(i+1, i)] = -1.0;
            }
            
            matrix
        }
        
        /// Hilbert matrix (known to be ill-conditioned)
        pub fn hilbert(n: usize) -> DMatrix<f64> {
            DMatrix::from_fn(n, n, |i, j| 1.0 / ((i + j + 1) as f64))
        }
        
        /// Pascal matrix
        pub fn pascal(n: usize) -> DMatrix<f64> {
            let mut matrix = DMatrix::zeros(n, n);
            
            for i in 0..n {
                for j in 0..n {
                    if j <= i {
                        matrix[(i, j)] = binomial_coefficient(i, j) as f64;
                    }
                }
            }
            
            matrix
        }
    }
    
    /// Helper function for binomial coefficients
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

/// Known modular forms and their properties
pub mod modular_forms {
    use super::*;
    
    /// Ramanujan tau function values
    pub struct RamanujanTau;
    
    impl RamanujanTau {
        /// First few values of τ(n)
        pub fn known_values() -> HashMap<usize, i64> {
            let mut values = HashMap::new();
            values.insert(1, 1);
            values.insert(2, -24);
            values.insert(3, 252);
            values.insert(4, -1472);
            values.insert(5, 4830);
            values.insert(6, -6048);
            values.insert(7, -16744);
            values.insert(8, 84480);
            values.insert(9, -113643);
            values.insert(10, -115920);
            values.insert(11, 534612);
            values.insert(12, -370944);
            values
        }
        
        /// Test Ramanujan's congruences
        pub fn test_congruences() -> Vec<(usize, usize, i64)> {
            vec![
                // τ(n) ≡ σ₁₁(n) (mod 691) for all n
                // τ(n) ≡ 0 (mod 23) when n ≡ 0 (mod 23)
                (23, 23, 0),
                (46, 23, 0),
                (69, 23, 0),
            ]
        }
    }
    
    /// Eisenstein series coefficients
    pub struct EisensteinSeries;
    
    impl EisensteinSeries {
        /// E₄ Fourier coefficients (first few)
        pub fn e4_coefficients() -> HashMap<usize, f64> {
            let mut coeffs = HashMap::new();
            coeffs.insert(0, 1.0);
            coeffs.insert(1, 240.0);
            coeffs.insert(2, 2160.0);
            coeffs.insert(3, 6720.0);
            coeffs.insert(4, 17520.0);
            coeffs.insert(5, 30240.0);
            coeffs
        }
        
        /// E₆ Fourier coefficients (first few)
        pub fn e6_coefficients() -> HashMap<usize, f64> {
            let mut coeffs = HashMap::new();
            coeffs.insert(0, 1.0);
            coeffs.insert(1, -504.0);
            coeffs.insert(2, 16632.0);
            coeffs.insert(3, -122976.0);
            coeffs.insert(4, 532728.0);
            coeffs.insert(5, -1575504.0);
            coeffs
        }
        
        /// Known special values
        pub fn special_values() -> HashMap<String, f64> {
            let mut values = HashMap::new();
            // E₄(i) where i is the imaginary unit
            values.insert("E4_at_i".to_string(), 1.0); // Normalized value
            // E₆(i)
            values.insert("E6_at_i".to_string(), 1.0); // Normalized value
            values
        }
    }
    
    /// j-invariant values
    pub struct JInvariant;
    
    impl JInvariant {
        /// Known special values of j(τ)
        pub fn special_values() -> HashMap<String, f64> {
            let mut values = HashMap::new();
            values.insert("j(i)".to_string(), 1728.0);
            values.insert("j(ρ)".to_string(), 0.0); // ρ = e^(2πi/3)
            values.insert("j(2i)".to_string(), 66.0); // Approximate
            values
        }
        
        /// j-invariant Fourier series coefficients (first few)
        pub fn fourier_coefficients() -> HashMap<i32, f64> {
            let mut coeffs = HashMap::new();
            coeffs.insert(-1, 1.0); // q^(-1) term
            coeffs.insert(0, 744.0);
            coeffs.insert(1, 196884.0);
            coeffs.insert(2, 21493760.0);
            coeffs.insert(3, 864299970.0);
            coeffs
        }
    }
}

/// Test elliptic curves and their properties
pub mod elliptic_curves {
    use super::*;
    
    /// Well-known elliptic curves for testing
    pub struct TestCurves;
    
    impl TestCurves {
        /// Curve y² = x³ - x (conductor 32)
        pub fn curve_32() -> EllipticCurve {
            EllipticCurve {
                a4: -1.0,
                a6: 0.0,
                conductor: 32,
                discriminant: 64.0,
            }
        }
        
        /// Curve y² = x³ - 2x (conductor 37) 
        pub fn curve_37() -> EllipticCurve {
            EllipticCurve {
                a4: -2.0,
                a6: 0.0,
                conductor: 37,
                discriminant: 4096.0,
            }
        }
        
        /// Curve y² = x³ + x + 1 (conductor 43)
        pub fn curve_43() -> EllipticCurve {
            EllipticCurve {
                a4: 1.0,
                a6: 1.0,
                conductor: 43,
                discriminant: -11.0,
            }
        }
    }
    
    /// Elliptic curve structure for testing
    #[derive(Debug, Clone)]
    pub struct EllipticCurve {
        pub a4: f64,
        pub a6: f64,
        pub conductor: usize,
        pub discriminant: f64,
    }
    
    impl EllipticCurve {
        /// Compute discriminant Δ = -16(4a₄³ + 27a₆²)
        pub fn compute_discriminant(&self) -> f64 {
            -16.0 * (4.0 * self.a4.powi(3) + 27.0 * self.a6.powi(2))
        }
        
        /// Check if curve is non-singular (Δ ≠ 0)
        pub fn is_non_singular(&self) -> bool {
            self.compute_discriminant().abs() > constants::TEST_EPSILON
        }
        
        /// Known L-function coefficients for test curves
        pub fn l_function_coefficients(&self, prime: usize) -> Option<i32> {
            // Placeholder - would compute a_p coefficients
            match (self.conductor, prime) {
                (32, 2) => Some(0), // Additive reduction
                (32, 3) => Some(-2),
                (37, 2) => Some(0),
                (37, 3) => Some(0),
                _ => None,
            }
        }
    }
}

/// Number theory test data
pub mod number_theory {
    use super::*;
    
    /// Prime numbers for testing
    pub struct Primes;
    
    impl Primes {
        pub fn small() -> Vec<usize> {
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        }
        
        pub fn medium() -> Vec<usize> {
            vec![53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        }
        
        pub fn large() -> Vec<usize> {
            vec![127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197]
        }
        
        /// Primes with special properties
        pub fn safe_primes() -> Vec<usize> {
            // Primes p where (p-1)/2 is also prime
            vec![5, 7, 11, 23, 47, 59, 83, 107, 167, 179, 191]
        }
        
        pub fn twin_primes() -> Vec<(usize, usize)> {
            vec![(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]
        }
    }
    
    /// Quadratic residues and symbols
    pub struct QuadraticResidues;
    
    impl QuadraticResidues {
        /// Legendre symbol (a/p) for small values
        pub fn legendre_symbol_table() -> HashMap<(usize, usize), i32> {
            let mut table = HashMap::new();
            
            // Some known values for testing
            table.insert((2, 3), -1);
            table.insert((2, 5), -1);
            table.insert((2, 7), 1);
            table.insert((3, 5), -1);
            table.insert((3, 7), -1);
            table.insert((5, 7), -1);
            
            table
        }
        
        /// Test quadratic reciprocity law
        pub fn reciprocity_test_cases() -> Vec<(usize, usize, bool)> {
            vec![
                (3, 5, true),  // Both ≡ 3 (mod 4), so product is positive
                (3, 7, false), // Different residue classes
                (5, 7, false),
                (7, 11, true),
            ]
        }
    }
    
    /// Cyclotomic polynomials and roots of unity
    pub struct Cyclotomic;
    
    impl Cyclotomic {
        /// Primitive roots of unity for small n
        pub fn primitive_roots(n: usize) -> Vec<Complex64> {
            let mut roots = Vec::new();
            let omega = Complex64::new(0.0, 2.0 * constants::PI / n as f64).exp();
            
            for k in 1..n {
                if gcd(k, n) == 1 {
                    roots.push(omega.powf(k as f64));
                }
            }
            
            roots
        }
        
        /// Cyclotomic polynomial coefficients
        pub fn polynomial_coefficients(n: usize) -> Vec<i32> {
            match n {
                1 => vec![-1, 1], // Φ₁(x) = x - 1
                2 => vec![1, 1],  // Φ₂(x) = x + 1
                3 => vec![1, 1, 1], // Φ₃(x) = x² + x + 1
                4 => vec![1, 0, 1], // Φ₄(x) = x² + 1
                5 => vec![1, 1, 1, 1, 1], // Φ₅(x) = x⁴ + x³ + x² + x + 1
                6 => vec![1, -1, 1], // Φ₆(x) = x² - x + 1
                _ => vec![1], // Placeholder
            }
        }
    }
    
    /// Greatest common divisor
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
}

/// L-functions test data
pub mod l_functions {
    use super::*;
    
    /// Riemann zeta function values
    pub struct RiemannZeta;
    
    impl RiemannZeta {
        /// Known values at positive even integers
        pub fn even_values() -> HashMap<usize, f64> {
            let mut values = HashMap::new();
            values.insert(2, constants::ZETA_2);
            values.insert(4, constants::ZETA_4);
            values.insert(6, constants::ZETA_6);
            values
        }
        
        /// Trivial zeros (negative even integers)
        pub fn trivial_zeros() -> Vec<i32> {
            vec![-2, -4, -6, -8, -10, -12, -14, -16, -18, -20]
        }
        
        /// First few non-trivial zeros (imaginary parts)
        pub fn nontrivial_zeros() -> Vec<f64> {
            vec![
                14.134725142, 21.022039639, 25.010857580, 30.424876126,
                32.935061588, 37.586178159, 40.918719012, 43.327073281,
            ]
        }
    }
    
    /// Dirichlet L-functions
    pub struct DirichletL;
    
    impl DirichletL {
        /// Characters mod small primes
        pub fn characters_mod_3() -> Vec<Vec<i32>> {
            vec![
                vec![1, 1],    // Principal character
                vec![1, -1],   // Non-principal character
            ]
        }
        
        pub fn characters_mod_4() -> Vec<Vec<i32>> {
            vec![
                vec![1, 1],     // Principal character  
                vec![1, -1],    // Character mod 4
            ]
        }
        
        /// Known L-function values
        pub fn known_values() -> HashMap<String, f64> {
            let mut values = HashMap::new();
            // L(1, χ) where χ is character mod 3
            values.insert("L(1,chi_3)".to_string(), constants::PI / (3.0 * 3_f64.sqrt()));
            // L(1, χ) where χ is character mod 4  
            values.insert("L(1,chi_4)".to_string(), constants::PI / 4.0);
            values
        }
    }
}

/// Galois theory test data
pub mod galois_theory {
    use super::*;
    
    /// Small finite fields
    pub struct FiniteFields;
    
    impl FiniteFields {
        /// Multiplicative group of F_p for small p
        pub fn multiplicative_group(p: usize) -> Vec<usize> {
            (1..p).collect()
        }
        
        /// Primitive elements of F_p
        pub fn primitive_elements(p: usize) -> Vec<usize> {
            let mut primitives = Vec::new();
            
            for g in 2..p {
                if is_primitive_root(g, p) {
                    primitives.push(g);
                }
            }
            
            primitives
        }
        
        /// Irreducible polynomials over F_p
        pub fn irreducible_polynomials() -> HashMap<usize, Vec<Vec<usize>>> {
            let mut polys = HashMap::new();
            
            // Over F_2
            polys.insert(2, vec![
                vec![1, 1, 1], // x² + x + 1
            ]);
            
            // Over F_3  
            polys.insert(3, vec![
                vec![1, 0, 1], // x² + 1
                vec![2, 1, 1], // x² + x + 2
            ]);
            
            polys
        }
    }
    
    /// Check if g is a primitive root mod p
    fn is_primitive_root(g: usize, p: usize) -> bool {
        if gcd(g, p) != 1 {
            return false;
        }
        
        let phi_p = p - 1;
        
        // Check if g^d ≢ 1 (mod p) for all proper divisors d of φ(p)
        for d in proper_divisors(phi_p) {
            if mod_pow(g, d, p) == 1 {
                return false;
            }
        }
        
        true
    }
    
    /// Compute proper divisors of n
    fn proper_divisors(n: usize) -> Vec<usize> {
        let mut divisors = Vec::new();
        for i in 1..n {
            if n % i == 0 {
                divisors.push(i);
            }
        }
        divisors
    }
    
    /// Modular exponentiation
    fn mod_pow(base: usize, exp: usize, modulus: usize) -> usize {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1;
        let mut base = base % modulus;
        let mut exp = exp;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        
        result
    }
    
    /// Greatest common divisor
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
}

/// Regression test data for known results
pub mod regression_data {
    use super::*;
    
    /// Expected test results for regression testing
    pub struct ExpectedResults;
    
    impl ExpectedResults {
        /// Expected eigenvalues for test matrices
        pub fn matrix_eigenvalues() -> HashMap<String, Vec<f64>> {
            let mut eigenvals = HashMap::new();
            
            // 2x2 identity matrix
            eigenvals.insert("identity_2x2".to_string(), vec![1.0, 1.0]);
            
            // Pauli-Z matrix eigenvalues
            eigenvals.insert("pauli_z".to_string(), vec![1.0, -1.0]);
            
            eigenvals
        }
        
        /// Expected L-function critical values
        pub fn l_function_critical_values() -> HashMap<String, f64> {
            let mut values = HashMap::new();
            
            // ζ(2) = π²/6
            values.insert("zeta(2)".to_string(), constants::ZETA_2);
            
            // L(1, χ₄) = π/4 where χ₄ is character mod 4
            values.insert("L(1,chi_4)".to_string(), constants::PI / 4.0);
            
            values
        }
        
        /// Expected Fourier coefficients
        pub fn fourier_coefficients() -> HashMap<String, HashMap<usize, f64>> {
            let mut coeffs = HashMap::new();
            
            // Eisenstein E₄ series
            let mut e4 = HashMap::new();
            e4.insert(0, 1.0);
            e4.insert(1, 240.0);
            e4.insert(2, 2160.0);
            coeffs.insert("E4".to_string(), e4);
            
            coeffs
        }
    }
}