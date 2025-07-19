//! Error types for the Geometric Langlands implementation

use thiserror::Error;

/// Main error type for the library
#[derive(Error, Debug)]
pub enum Error {
    /// Mathematical computation error
    #[error("Mathematical error: {0}")]
    MathError(String),
    
    /// Dimension mismatch in operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { 
        /// Expected dimension
        expected: usize, 
        /// Actual dimension found
        actual: usize 
    },
    
    /// Invalid parameter provided
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Convergence failure in iterative algorithms
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailure { 
        /// Number of iterations attempted
        iterations: usize 
    },
    
    /// Mismatch between group structures
    #[error("Group structures are incompatible")]
    GroupMismatch,
    
    /// CUDA-related errors
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    /// WASM-related errors
    #[cfg(feature = "wasm")]
    #[error("WASM error: {0}")]
    WasmError(String),
    
    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Generic errors
    #[error("Error: {0}")]
    Other(String),
}

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

// Implement conversions from other error types
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Other(err.to_string())
    }
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for Error {
    fn from(err: cust::error::CudaError) -> Self {
        Error::CudaError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::MathError("Division by zero".to_string());
        assert_eq!(err.to_string(), "Mathematical error: Division by zero");
        
        let err = Error::DimensionMismatch { expected: 3, actual: 2 };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 3, got 2");
    }
}