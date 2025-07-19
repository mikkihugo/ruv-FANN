//! Tests for error handling

use geometric_langlands::Error;

/// Run all error handling tests
pub fn run_all() {
    println!("Running error handling tests...");
    test_error_types();
}

#[test]
fn test_error_types() {
    let err = Error::MathError("test error".to_string());
    assert_eq!(err.to_string(), "Mathematical error: test error");
    
    let err = Error::DimensionMismatch { expected: 3, actual: 2 };
    assert_eq!(err.to_string(), "Dimension mismatch: expected 3, got 2");
}