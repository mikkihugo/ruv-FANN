//! Benchmarking utilities
//!
//! This module provides utilities for performance benchmarking.

// TODO: Duke Performance Engineer - Add benchmark utilities here

/// Benchmark configuration
pub struct BenchmarkConfig {
    /// Number of iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup: usize,
}