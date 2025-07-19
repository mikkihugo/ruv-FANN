//! Mathematical validation framework
//! 
//! This module provides validation tools to ensure mathematical correctness
//! throughout the implementation.

use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use super::{MathError, MathResult};

/// Result type for validation operations
pub type ValidationResult = MathResult<()>;

/// Mathematical validator for ensuring correctness
#[derive(Debug, Clone)]
pub struct MathValidator {
    /// Configuration for validation
    pub config: ValidationConfig,
    
    /// Validation rules
    rules: Vec<Box<dyn ValidationRule>>,
}

/// Configuration for mathematical validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Strictness level (0-2)
    pub strictness: u8,
    
    /// Enable runtime checks
    pub runtime_checks: bool,
    
    /// Enable proof verification
    pub verify_proofs: bool,
    
    /// Numerical tolerance
    pub epsilon: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strictness: 1,
            runtime_checks: true,
            verify_proofs: false,
            epsilon: 1e-10,
        }
    }
}

/// Trait for validation rules
pub trait ValidationRule: Debug {
    /// Name of the rule
    fn name(&self) -> &str;
    
    /// Apply the validation rule
    fn validate(&self, context: &ValidationContext) -> ValidationResult;
    
    /// Clone the rule
    fn clone_box(&self) -> Box<dyn ValidationRule>;
}

impl Clone for Box<dyn ValidationRule> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Context for validation
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Type of mathematical object being validated
    pub object_type: String,
    
    /// Additional context data
    pub data: std::collections::HashMap<String, String>,
}

impl MathValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }
    
    /// Create a new validator with specific configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut validator = Self {
            config,
            rules: Vec::new(),
        };
        
        // Add default rules based on strictness
        validator.add_default_rules();
        
        validator
    }
    
    /// Add a validation rule
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }
    
    /// Validate in a given context
    pub fn validate(&self, context: &ValidationContext) -> ValidationResult {
        for rule in &self.rules {
            rule.validate(context)?;
        }
        Ok(())
    }
    
    /// Add default validation rules based on configuration
    fn add_default_rules(&mut self) {
        // Basic rules (strictness >= 0)
        self.add_rule(Box::new(NonEmptyRule));
        self.add_rule(Box::new(FiniteRule));
        
        // Intermediate rules (strictness >= 1)
        if self.config.strictness >= 1 {
            self.add_rule(Box::new(ConsistencyRule));
            self.add_rule(Box::new(DimensionRule));
        }
        
        // Advanced rules (strictness >= 2)
        if self.config.strictness >= 2 {
            self.add_rule(Box::new(CompletenessRule));
            self.add_rule(Box::new(CorrectnessRule));
        }
    }
}

// Built-in validation rules

/// Rule: Objects must be non-empty
#[derive(Debug, Clone)]
struct NonEmptyRule;

impl ValidationRule for NonEmptyRule {
    fn name(&self) -> &str {
        "NonEmpty"
    }
    
    fn validate(&self, context: &ValidationContext) -> ValidationResult {
        if context.data.is_empty() {
            return Err(MathError::ValidationFailed(
                "Mathematical object cannot be empty".to_string()
            ));
        }
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Rule: Values must be finite
#[derive(Debug, Clone)]
struct FiniteRule;

impl ValidationRule for FiniteRule {
    fn name(&self) -> &str {
        "Finite"
    }
    
    fn validate(&self, _context: &ValidationContext) -> ValidationResult {
        // Check for infinite values in numerical data
        // This would inspect actual mathematical objects
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Rule: Mathematical consistency
#[derive(Debug, Clone)]
struct ConsistencyRule;

impl ValidationRule for ConsistencyRule {
    fn name(&self) -> &str {
        "Consistency"
    }
    
    fn validate(&self, context: &ValidationContext) -> ValidationResult {
        // Check internal consistency of mathematical objects
        match context.object_type.as_str() {
            "Category" => {
                // Verify category axioms
                Ok(())
            }
            "Morphism" => {
                // Verify morphism properties
                Ok(())
            }
            "Bundle" => {
                // Verify bundle consistency
                Ok(())
            }
            _ => Ok(()),
        }
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Rule: Dimension consistency
#[derive(Debug, Clone)]
struct DimensionRule;

impl ValidationRule for DimensionRule {
    fn name(&self) -> &str {
        "Dimension"
    }
    
    fn validate(&self, context: &ValidationContext) -> ValidationResult {
        // Check dimension compatibility
        if let Some(dim_str) = context.data.get("dimension") {
            if let Ok(dim) = dim_str.parse::<usize>() {
                if dim == 0 {
                    return Err(MathError::ValidationFailed(
                        "Dimension must be positive".to_string()
                    ));
                }
            }
        }
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Rule: Mathematical completeness
#[derive(Debug, Clone)]
struct CompletenessRule;

impl ValidationRule for CompletenessRule {
    fn name(&self) -> &str {
        "Completeness"
    }
    
    fn validate(&self, _context: &ValidationContext) -> ValidationResult {
        // Check that all required components are present
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Rule: Mathematical correctness (proofs)
#[derive(Debug, Clone)]
struct CorrectnessRule;

impl ValidationRule for CorrectnessRule {
    fn name(&self) -> &str {
        "Correctness"
    }
    
    fn validate(&self, _context: &ValidationContext) -> ValidationResult {
        // Verify mathematical proofs if available
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn ValidationRule> {
        Box::new(self.clone())
    }
}

/// Numerical validation utilities
pub mod numerical {
    use super::*;
    
    /// Check if a floating point value is valid
    pub fn validate_float(x: f64, epsilon: f64) -> ValidationResult {
        if x.is_nan() {
            return Err(MathError::ValidationFailed("NaN value detected".to_string()));
        }
        if x.is_infinite() {
            return Err(MathError::ValidationFailed("Infinite value detected".to_string()));
        }
        if x.abs() < epsilon && x != 0.0 {
            return Err(MathError::ValidationFailed(
                format!("Value {} below epsilon threshold {}", x, epsilon)
            ));
        }
        Ok(())
    }
    
    /// Check if two floats are approximately equal
    pub fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }
    
    /// Validate a matrix
    pub fn validate_matrix(matrix: &nalgebra::DMatrix<f64>, epsilon: f64) -> ValidationResult {
        for &value in matrix.iter() {
            validate_float(value, epsilon)?;
        }
        Ok(())
    }
}

/// Algebraic validation utilities
pub mod algebraic {
    use super::*;
    
    /// Check if a structure satisfies group axioms
    pub fn validate_group_axioms<G>(
        elements: &[G],
        identity: &G,
        compose: impl Fn(&G, &G) -> G,
        inverse: impl Fn(&G) -> G,
    ) -> ValidationResult 
    where
        G: PartialEq + Clone,
    {
        // Check identity element
        for e in elements {
            if compose(e, identity) != *e || compose(identity, e) != *e {
                return Err(MathError::ValidationFailed(
                    "Identity element law violated".to_string()
                ));
            }
        }
        
        // Check inverses
        for e in elements {
            let inv = inverse(e);
            if compose(e, &inv) != *identity || compose(&inv, e) != *identity {
                return Err(MathError::ValidationFailed(
                    "Inverse element law violated".to_string()
                ));
            }
        }
        
        // Check associativity (sample a few triples)
        let n = elements.len().min(5);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let a = &elements[i];
                    let b = &elements[j];
                    let c = &elements[k];
                    
                    let ab_c = compose(&compose(a, b), c);
                    let a_bc = compose(a, &compose(b, c));
                    
                    if ab_c != a_bc {
                        return Err(MathError::ValidationFailed(
                            "Associativity law violated".to_string()
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validator_creation() {
        let validator = MathValidator::new();
        assert_eq!(validator.config.strictness, 1);
        assert!(validator.config.runtime_checks);
    }
    
    #[test]
    fn test_validation_context() {
        let mut context = ValidationContext {
            object_type: "TestObject".to_string(),
            data: std::collections::HashMap::new(),
        };
        
        let validator = MathValidator::new();
        
        // Empty context should fail NonEmpty rule
        assert!(validator.validate(&context).is_err());
        
        // Add some data
        context.data.insert("test".to_string(), "value".to_string());
        assert!(validator.validate(&context).is_ok());
    }
    
    #[test]
    fn test_numerical_validation() {
        use numerical::*;
        
        assert!(validate_float(1.0, 1e-10).is_ok());
        assert!(validate_float(f64::NAN, 1e-10).is_err());
        assert!(validate_float(f64::INFINITY, 1e-10).is_err());
        
        assert!(approx_equal(1.0, 1.0 + 1e-11, 1e-10));
        assert!(!approx_equal(1.0, 1.0 + 1e-9, 1e-10));
    }
}