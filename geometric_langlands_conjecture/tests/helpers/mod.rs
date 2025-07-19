//! Test helper utilities and common functionality

use std::time::Instant;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use geometric_langlands::prelude::*;

/// Test data generators for mathematical structures
pub mod generators {
    use super::*;
    use proptest::prelude::*;
    
    /// Generate random field elements (using f64 as base field)
    pub fn field_element() -> impl Strategy<Value = f64> {
        (-1000.0..1000.0).prop_filter("non-zero", |&x| x.abs() > 1e-10)
    }
    
    /// Generate random complex numbers
    pub fn complex_number() -> impl Strategy<Value = Complex64> {
        (field_element(), field_element()).prop_map(|(re, im)| Complex64::new(re, im))
    }
    
    /// Generate random matrices of given dimensions
    pub fn random_matrix(rows: usize, cols: usize) -> impl Strategy<Value = DMatrix<f64>> {
        prop::collection::vec(field_element(), rows * cols)
            .prop_map(move |data| DMatrix::from_vec(rows, cols, data))
    }
    
    /// Generate random vectors of given dimension
    pub fn random_vector(dim: usize) -> impl Strategy<Value = DVector<f64>> {
        prop::collection::vec(field_element(), dim)
            .prop_map(|data| DVector::from_vec(data))
    }
    
    /// Generate polynomial coefficients
    pub fn polynomial_coeffs(degree: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(field_element(), degree + 1)
    }
    
    /// Generate prime numbers for testing
    pub fn small_prime() -> impl Strategy<Value = usize> {
        prop::sample::select(vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    }
    
    /// Generate matrix dimensions for testing
    pub fn matrix_dimension() -> impl Strategy<Value = usize> {
        1..10usize
    }
}

/// Timing utilities for performance testing
pub struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    pub fn new(label: &str) -> Self {
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }
    
    pub fn report(&self) {
        println!("{}: {}ms", self.label, self.elapsed_ms());
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.report();
    }
}

/// Macro for timing test execution
#[macro_export]
macro_rules! time_test {
    ($label:expr, $test:expr) => {{
        let _timer = $crate::helpers::Timer::new($label);
        $test
    }};
}

/// Memory usage tracking utilities
pub struct MemoryTracker {
    initial_usage: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: Self::current_memory_usage(),
        }
    }
    
    pub fn current_memory_usage() -> usize {
        // Simplified memory tracking - in practice would use more sophisticated methods
        std::mem::size_of::<usize>() * 1000 // Placeholder
    }
    
    pub fn memory_delta(&self) -> isize {
        Self::current_memory_usage() as isize - self.initial_usage as isize
    }
}

/// Test assertion helpers with detailed error messages
pub mod assertions {
    use super::*;
    use crate::math_utils::{approx_eq, complex_approx_eq, matrix_approx_eq, vector_approx_eq};
    
    /// Assert two floating point numbers are approximately equal with context
    pub fn assert_approx_eq_with_context(
        a: f64, 
        b: f64, 
        context: &str
    ) {
        assert!(
            approx_eq(a, b),
            "Floating point assertion failed in {}: {} ≠ {} (diff: {})",
            context, a, b, (a - b).abs()
        );
    }
    
    /// Assert two complex numbers are approximately equal with context
    pub fn assert_complex_approx_eq_with_context(
        a: Complex64, 
        b: Complex64, 
        context: &str
    ) {
        assert!(
            complex_approx_eq(a, b),
            "Complex number assertion failed in {}: {} ≠ {} (diff: {})",
            context, a, b, (a - b).norm()
        );
    }
    
    /// Assert two matrices are approximately equal with context
    pub fn assert_matrix_approx_eq_with_context(
        a: &DMatrix<f64>, 
        b: &DMatrix<f64>, 
        context: &str
    ) {
        assert!(
            matrix_approx_eq(a, b),
            "Matrix assertion failed in {}: shapes ({:?} vs {:?})",
            context, a.shape(), b.shape()
        );
    }
    
    /// Assert mathematical property holds
    pub fn assert_mathematical_property<F>(
        property: F,
        description: &str,
        context: &str
    ) where F: FnOnce() -> bool {
        assert!(
            property(),
            "Mathematical property '{}' failed in context: {}",
            description, context
        );
    }
}

/// Test fixtures and known test cases
pub mod fixtures {
    use super::*;
    
    /// Well-known mathematical constants for testing
    pub struct Constants;
    
    impl Constants {
        pub const PI: f64 = std::f64::consts::PI;
        pub const E: f64 = std::f64::consts::E;
        pub const GOLDEN_RATIO: f64 = 1.618033988749894;
        pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    }
    
    /// Known test matrices for validation
    pub struct TestMatrices;
    
    impl TestMatrices {
        /// 2x2 identity matrix
        pub fn identity_2x2() -> DMatrix<f64> {
            DMatrix::identity(2, 2)
        }
        
        /// 3x3 identity matrix
        pub fn identity_3x3() -> DMatrix<f64> {
            DMatrix::identity(3, 3)
        }
        
        /// Pauli matrices for testing
        pub fn pauli_x() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        pub fn pauli_y() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        pub fn pauli_z() -> DMatrix<Complex64> {
            DMatrix::from_row_slice(2, 2, &[
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ])
        }
        
        /// Test matrix with known eigenvalues
        pub fn test_symmetric_3x3() -> DMatrix<f64> {
            DMatrix::from_row_slice(3, 3, &[
                2.0, -1.0, 0.0,
                -1.0, 2.0, -1.0,
                0.0, -1.0, 2.0,
            ])
        }
    }
}

/// Utilities for testing parallel algorithms
pub mod parallel {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;
    
    /// Test that parallel operations produce same results as sequential
    pub fn test_parallel_consistency<T, F1, F2>(
        data: T,
        sequential_fn: F1,
        parallel_fn: F2,
    ) -> bool
    where
        T: Clone + Send + Sync,
        F1: Fn(T) -> String + Send + Sync,
        F2: Fn(T) -> String + Send + Sync,
    {
        let sequential_result = sequential_fn(data.clone());
        let parallel_result = parallel_fn(data);
        
        sequential_result == parallel_result
    }
    
    /// Test thread safety of operations
    pub fn test_thread_safety<T, F>(
        data: Arc<T>,
        operation: F,
        num_threads: usize,
    ) -> bool
    where
        T: Send + Sync + 'static,
        F: Fn(Arc<T>) -> bool + Send + Sync + 'static + Clone,
    {
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        
        for _ in 0..num_threads {
            let data_clone = Arc::clone(&data);
            let operation_clone = operation.clone();
            let results_clone = Arc::clone(&results);
            
            let handle = thread::spawn(move || {
                let result = operation_clone(data_clone);
                results_clone.lock().unwrap().push(result);
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Check all operations succeeded
        let final_results = results.lock().unwrap();
        final_results.iter().all(|&x| x)
    }
}

/// Debugging utilities for tests
pub mod debug {
    use super::*;
    
    /// Print matrix in readable format for debugging
    pub fn print_matrix(m: &DMatrix<f64>, label: &str) {
        println!("\n{} ({}x{}):", label, m.nrows(), m.ncols());
        for i in 0..m.nrows() {
            print!("  [");
            for j in 0..m.ncols() {
                print!("{:8.4}", m[(i, j)]);
                if j < m.ncols() - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }
        println!();
    }
    
    /// Print complex matrix in readable format
    pub fn print_complex_matrix(m: &DMatrix<Complex64>, label: &str) {
        println!("\n{} ({}x{}):", label, m.nrows(), m.ncols());
        for i in 0..m.nrows() {
            print!("  [");
            for j in 0..m.ncols() {
                let c = m[(i, j)];
                print!("{:6.2}+{:6.2}i", c.re, c.im);
                if j < m.ncols() - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }
        println!();
    }
    
    /// Create test report
    pub fn create_test_report(
        test_name: &str,
        passed: bool,
        execution_time: Duration,
        details: &str,
    ) {
        let status = if passed { "PASS" } else { "FAIL" };
        println!(
            "\n=== TEST REPORT ===\nTest: {}\nStatus: {}\nTime: {:?}\nDetails: {}\n==================",
            test_name, status, execution_time, details
        );
    }
}