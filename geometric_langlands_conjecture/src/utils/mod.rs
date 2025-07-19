//! Utility functions and helpers
//!
//! This module provides various utility functions used throughout the library.

// TODO: Various agents - Add utilities as needed

/// Numerical precision constant
pub const EPSILON: f64 = 1e-10;

/// Check if two floating point numbers are approximately equal
pub fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}