//! Unit tests for individual modules and components
//!
//! This module contains focused unit tests for each component of the
//! Geometric Langlands implementation.

pub mod core_tests;
pub mod automorphic_tests;
pub mod galois_tests;
pub mod category_tests;
pub mod sheaf_tests;
pub mod representation_tests;
pub mod harmonic_tests;
pub mod spectral_tests;
pub mod trace_tests;
pub mod langlands_tests;

#[cfg(feature = "cuda")]
pub mod cuda_tests;

#[cfg(feature = "wasm")]
pub mod wasm_tests;

pub mod utils_tests;
pub mod error_tests;

use crate::helpers::Timer;

/// Run all unit tests with timing
pub fn run_all_unit_tests() {
    let _timer = Timer::new("All Unit Tests");
    
    println!("Running core mathematical structure tests...");
    core_tests::run_all();
    
    println!("Running automorphic forms tests...");
    automorphic_tests::run_all();
    
    println!("Running Galois representation tests...");
    galois_tests::run_all();
    
    println!("Running category theory tests...");
    category_tests::run_all();
    
    println!("Running sheaf theory tests...");
    sheaf_tests::run_all();
    
    println!("Running representation theory tests...");
    representation_tests::run_all();
    
    println!("Running harmonic analysis tests...");
    harmonic_tests::run_all();
    
    println!("Running spectral theory tests...");
    spectral_tests::run_all();
    
    println!("Running trace formula tests...");
    trace_tests::run_all();
    
    println!("Running main Langlands correspondence tests...");
    langlands_tests::run_all();
    
    #[cfg(feature = "cuda")]
    {
        println!("Running CUDA acceleration tests...");
        cuda_tests::run_all();
    }
    
    #[cfg(feature = "wasm")]
    {
        println!("Running WASM compilation tests...");
        wasm_tests::run_all();
    }
    
    println!("Running utility function tests...");
    utils_tests::run_all();
    
    println!("Running error handling tests...");
    error_tests::run_all();
    
    println!("All unit tests completed!");
}