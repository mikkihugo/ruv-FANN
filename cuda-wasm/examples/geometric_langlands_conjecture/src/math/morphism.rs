//! Morphism types and composition laws for category theory
//! 
//! This module implements various types of morphisms with proper
//! composition rules and validation.

use std::marker::PhantomData;
use serde::{Serialize, Deserialize};
use super::{MathObject, MathEquivalence, ValidationResult};
use super::category::{Object, Morphism as MorphismTrait, MorphismError};

/// Generic morphism implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Morphism<O: Object> {
    /// Unique identifier
    pub id: String,
    
    /// Source object ID
    pub source_id: O::Id,
    
    /// Target object ID
    pub target_id: O::Id,
    
    /// Morphism data (could be a matrix, function, etc.)
    pub data: MorphismData,
    
    /// Properties of the morphism
    pub properties: MorphismProperties,
    
    #[serde(skip)]
    _phantom: PhantomData<O>,
}

/// Data representation of a morphism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MorphismData {
    /// Identity morphism
    Identity,
    
    /// Linear map represented as a matrix
    LinearMap(nalgebra::DMatrix<f64>),
    
    /// Polynomial map
    PolynomialMap(Vec<Polynomial>),
    
    /// Abstract morphism (for theoretical work)
    Abstract(String),
    
    /// Composite of two morphisms
    Composite(Box<MorphismData>, Box<MorphismData>),
}

/// Properties that a morphism might have
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MorphismProperties {
    pub is_identity: bool,
    pub is_isomorphism: bool,
    pub is_monomorphism: bool,
    pub is_epimorphism: bool,
    pub is_endomorphism: bool,
    pub is_automorphism: bool,
}

/// Polynomial representation for polynomial maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients of the polynomial
    pub coefficients: Vec<f64>,
    
    /// Variables involved
    pub variables: Vec<String>,
}

impl<O: Object> MathObject for Morphism<O> {
    type Id = String;
    
    fn id(&self) -> &Self::Id {
        &self.id
    }
    
    fn validate(&self) -> ValidationResult {
        // Validate morphism properties consistency
        if self.properties.is_automorphism {
            if !self.properties.is_isomorphism || !self.properties.is_endomorphism {
                return Err(super::MathError::ValidationFailed(
                    "Automorphism must be both isomorphism and endomorphism".to_string()
                ));
            }
        }
        
        if self.properties.is_identity {
            if self.source_id != self.target_id {
                return Err(super::MathError::ValidationFailed(
                    "Identity morphism must have same source and target".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn description(&self) -> String {
        format!("Morphism {} from {:?} to {:?}", self.id, self.source_id, self.target_id)
    }
}

impl<O: Object> MathEquivalence for Morphism<O> {
    fn is_equivalent(&self, other: &Self) -> bool {
        // Two morphisms are equivalent if they have the same effect
        self.source_id == other.source_id && 
        self.target_id == other.target_id &&
        self.data.is_equivalent(&other.data)
    }
}

impl<O: Object> MorphismTrait<O> for Morphism<O> {
    fn source(&self) -> &O::Id {
        &self.source_id
    }
    
    fn target(&self) -> &O::Id {
        &self.target_id
    }
    
    fn compose(&self, other: &Self) -> Result<Self, MorphismError> {
        // Check if morphisms are composable
        if self.target_id != other.source_id {
            return Err(MorphismError::NotComposable(
                format!("{:?}", self.target_id),
                format!("{:?}", other.source_id),
            ));
        }
        
        // Compose the morphism data
        let composed_data = match (&self.data, &other.data) {
            (MorphismData::Identity, _) => other.data.clone(),
            (_, MorphismData::Identity) => self.data.clone(),
            (MorphismData::LinearMap(m1), MorphismData::LinearMap(m2)) => {
                // Matrix multiplication for linear maps
                if m1.ncols() != m2.nrows() {
                    return Err(MorphismError::Invalid(
                        "Matrix dimensions incompatible for composition".to_string()
                    ));
                }
                MorphismData::LinearMap(m2 * m1)
            }
            _ => MorphismData::Composite(
                Box::new(self.data.clone()),
                Box::new(other.data.clone()),
            ),
        };
        
        // Determine properties of the composition
        let mut properties = MorphismProperties::default();
        
        if self.properties.is_identity && other.properties.is_identity {
            properties.is_identity = true;
        }
        
        if self.properties.is_isomorphism && other.properties.is_isomorphism {
            properties.is_isomorphism = true;
        }
        
        if self.properties.is_monomorphism && other.properties.is_monomorphism {
            properties.is_monomorphism = true;
        }
        
        if self.properties.is_epimorphism && other.properties.is_epimorphism {
            properties.is_epimorphism = true;
        }
        
        if self.source_id == other.target_id {
            properties.is_endomorphism = true;
            if properties.is_isomorphism {
                properties.is_automorphism = true;
            }
        }
        
        Ok(Morphism {
            id: format!("{} ∘ {}", other.id, self.id),
            source_id: self.source_id.clone(),
            target_id: other.target_id.clone(),
            data: composed_data,
            properties,
            _phantom: PhantomData,
        })
    }
    
    fn is_identity(&self) -> bool {
        self.properties.is_identity
    }
    
    fn is_isomorphism(&self) -> bool {
        self.properties.is_isomorphism
    }
    
    fn inverse(&self) -> Option<Self> {
        if !self.is_isomorphism() {
            return None;
        }
        
        match &self.data {
            MorphismData::Identity => Some(self.clone()),
            MorphismData::LinearMap(matrix) => {
                // Try to compute matrix inverse
                matrix.try_inverse().map(|inv| Morphism {
                    id: format!("{}⁻¹", self.id),
                    source_id: self.target_id.clone(),
                    target_id: self.source_id.clone(),
                    data: MorphismData::LinearMap(inv),
                    properties: self.properties.clone(),
                    _phantom: PhantomData,
                })
            }
            _ => None, // Other types need specific inverse implementations
        }
    }
}

impl MathEquivalence for MorphismData {
    fn is_equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (MorphismData::Identity, MorphismData::Identity) => true,
            (MorphismData::LinearMap(m1), MorphismData::LinearMap(m2)) => {
                m1.nrows() == m2.nrows() && m1.ncols() == m2.ncols() &&
                (m1 - m2).norm() < 1e-10
            }
            (MorphismData::Abstract(s1), MorphismData::Abstract(s2)) => s1 == s2,
            _ => false,
        }
    }
}

/// Trait for morphism composition with associativity laws
pub trait MorphismComposition: Sized {
    /// Compose two morphisms (other ∘ self)
    fn compose(&self, other: &Self) -> Result<Self, MorphismError>;
    
    /// Verify associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    fn verify_associativity(f: &Self, g: &Self, h: &Self) -> Result<bool, MorphismError> {
        let gf = f.compose(g)?;
        let hgf = gf.compose(h)?;
        
        let hg = g.compose(h)?;
        let h_gf = f.compose(&hg)?;
        
        Ok(hgf.is_equivalent(&h_gf))
    }
}

/// Special morphism types

/// Zero morphism (in categories with zero object)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMorphism<O: Object> {
    pub source_id: O::Id,
    pub target_id: O::Id,
}

/// Projection morphism (for product objects)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionMorphism<O: Object> {
    pub source_id: O::Id,
    pub target_id: O::Id,
    pub index: usize,
}

/// Inclusion morphism (for coproduct objects)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InclusionMorphism<O: Object> {
    pub source_id: O::Id,
    pub target_id: O::Id,
    pub index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_morphism_validation() {
        // Test identity morphism validation
        let id_morphism = Morphism::<TestObject> {
            id: "id_A".to_string(),
            source_id: TestObjectId("A".to_string()),
            target_id: TestObjectId("A".to_string()),
            data: MorphismData::Identity,
            properties: MorphismProperties {
                is_identity: true,
                is_isomorphism: true,
                is_endomorphism: true,
                is_automorphism: true,
                ..Default::default()
            },
            _phantom: PhantomData,
        };
        
        assert!(id_morphism.validate().is_ok());
        
        // Test invalid identity morphism
        let bad_id = Morphism::<TestObject> {
            id: "bad_id".to_string(),
            source_id: TestObjectId("A".to_string()),
            target_id: TestObjectId("B".to_string()),
            data: MorphismData::Identity,
            properties: MorphismProperties {
                is_identity: true,
                ..Default::default()
            },
            _phantom: PhantomData,
        };
        
        assert!(bad_id.validate().is_err());
    }
    
    #[test]
    fn test_linear_map_composition() {
        use nalgebra::DMatrix;
        
        // Create two 2x2 matrices
        let m1 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let m2 = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        
        let morph1 = Morphism::<TestObject> {
            id: "f".to_string(),
            source_id: TestObjectId("A".to_string()),
            target_id: TestObjectId("B".to_string()),
            data: MorphismData::LinearMap(m1),
            properties: Default::default(),
            _phantom: PhantomData,
        };
        
        let morph2 = Morphism::<TestObject> {
            id: "g".to_string(),
            source_id: TestObjectId("B".to_string()),
            target_id: TestObjectId("C".to_string()),
            data: MorphismData::LinearMap(m2),
            properties: Default::default(),
            _phantom: PhantomData,
        };
        
        let composed = morph1.compose(&morph2).unwrap();
        assert_eq!(composed.source_id, TestObjectId("A".to_string()));
        assert_eq!(composed.target_id, TestObjectId("C".to_string()));
    }
    
    // Mock types for testing
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
    struct TestObjectId(String);
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestObject {
        id: TestObjectId,
    }
    
    impl MathObject for TestObject {
        type Id = TestObjectId;
        fn id(&self) -> &Self::Id { &self.id }
        fn validate(&self) -> ValidationResult { Ok(()) }
        fn description(&self) -> String { "Test object".to_string() }
    }
    
    impl Object for TestObject {
        type Morphism = Morphism<Self>;
    }
}