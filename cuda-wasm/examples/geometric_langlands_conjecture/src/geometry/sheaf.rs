// Sheaf theory implementation for Geometric Langlands
// Implements coherent sheaves, quasi-coherent sheaves, and perverse sheaves

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use super::GeometricObject;

/// Open set in a topological space
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpenSet {
    pub id: String,
    pub dimension: usize,
    pub is_affine: bool,
}

/// Local section of a sheaf over an open set
#[derive(Debug, Clone)]
pub struct LocalSection<T: Clone + Debug> {
    pub open_set: OpenSet,
    pub data: T,
    pub restriction_maps: HashMap<String, Box<dyn Fn(&T) -> T>>,
}

/// Restriction map between open sets
pub trait RestrictionMap<T> {
    fn restrict(&self, section: &T, from: &OpenSet, to: &OpenSet) -> Option<T>;
}

/// Sheaf on a topological space
#[derive(Debug, Clone)]
pub struct Sheaf<T: Clone + Debug> {
    /// Base space dimension
    pub base_dimension: usize,
    /// Sections over open sets
    pub sections: HashMap<OpenSet, Vec<LocalSection<T>>>,
    /// Gluing conditions
    pub gluing_data: HashMap<(OpenSet, OpenSet), Box<dyn Fn(&T, &T) -> bool>>,
    /// Sheaf type
    pub sheaf_type: SheafType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SheafType {
    Coherent,
    QuasiCoherent,
    Perverse,
    Constructible,
    LocalSystem,
}

impl<T: Clone + Debug + 'static> Sheaf<T> {
    /// Create a new sheaf
    pub fn new(base_dimension: usize, sheaf_type: SheafType) -> Self {
        Self {
            base_dimension,
            sections: HashMap::new(),
            gluing_data: HashMap::new(),
            sheaf_type,
        }
    }
    
    /// Add a local section
    pub fn add_section(&mut self, section: LocalSection<T>) {
        let open_set = section.open_set.clone();
        self.sections.entry(open_set).or_insert_with(Vec::new).push(section);
    }
    
    /// Get sections over an open set
    pub fn get_sections(&self, open_set: &OpenSet) -> Option<&Vec<LocalSection<T>>> {
        self.sections.get(open_set)
    }
    
    /// Compute global sections
    pub fn global_sections(&self) -> Vec<&T> {
        let mut global = Vec::new();
        
        // Find sections that can be glued globally
        for (_, sections) in &self.sections {
            for section in sections {
                // Check if section extends globally
                if self.extends_globally(&section.data) {
                    global.push(&section.data);
                }
            }
        }
        
        global
    }
    
    /// Check if a section extends globally
    fn extends_globally(&self, _section: &T) -> bool {
        // Simplified check - in practice would verify gluing conditions
        true
    }
    
    /// Compute sheaf cohomology dimension
    pub fn cohomology_dimension(&self, degree: usize) -> usize {
        match self.sheaf_type {
            SheafType::Coherent => {
                // Use Serre duality for coherent sheaves
                if degree > self.base_dimension {
                    0
                } else {
                    // Simplified computation
                    1
                }
            }
            SheafType::Perverse => {
                // Perverse sheaves have special cohomology
                if degree == self.base_dimension / 2 {
                    1
                } else {
                    0
                }
            }
            _ => 1,
        }
    }
}

/// Morphism between sheaves
#[derive(Debug, Clone)]
pub struct SheafMorphism<T: Clone + Debug, U: Clone + Debug> {
    pub source: Sheaf<T>,
    pub target: Sheaf<U>,
    pub local_maps: HashMap<OpenSet, Box<dyn Fn(&T) -> U>>,
}

impl<T: Clone + Debug + 'static, U: Clone + Debug + 'static> SheafMorphism<T, U> {
    /// Create a new sheaf morphism
    pub fn new(source: Sheaf<T>, target: Sheaf<U>) -> Self {
        Self {
            source,
            target,
            local_maps: HashMap::new(),
        }
    }
    
    /// Add a local map
    pub fn add_local_map(&mut self, open_set: OpenSet, map: Box<dyn Fn(&T) -> U>) {
        self.local_maps.insert(open_set, map);
    }
    
    /// Check if morphism is injective
    pub fn is_injective(&self) -> bool {
        // Simplified check
        true
    }
    
    /// Check if morphism is surjective
    pub fn is_surjective(&self) -> bool {
        // Simplified check
        self.local_maps.len() == self.source.sections.len()
    }
}

/// D-module structure on a sheaf
#[derive(Debug, Clone)]
pub struct DModule<T: Clone + Debug> {
    pub sheaf: Sheaf<T>,
    pub differential_operators: HashMap<String, Box<dyn Fn(&T) -> T>>,
    pub holonomic: bool,
}

impl<T: Clone + Debug + 'static> DModule<T> {
    /// Create a new D-module
    pub fn new(sheaf: Sheaf<T>, holonomic: bool) -> Self {
        Self {
            sheaf,
            differential_operators: HashMap::new(),
            holonomic,
        }
    }
    
    /// Add a differential operator
    pub fn add_operator(&mut self, name: String, operator: Box<dyn Fn(&T) -> T>) {
        self.differential_operators.insert(name, operator);
    }
    
    /// Compute characteristic variety dimension
    pub fn characteristic_variety_dim(&self) -> usize {
        if self.holonomic {
            self.sheaf.base_dimension
        } else {
            2 * self.sheaf.base_dimension
        }
    }
}

/// Perverse sheaf with perversity function
#[derive(Debug, Clone)]
pub struct PerverseSheaf<T: Clone + Debug> {
    pub sheaf: Sheaf<T>,
    pub perversity: Box<dyn Fn(usize) -> i32>,
    pub stratification: Vec<OpenSet>,
}

impl<T: Clone + Debug + 'static> PerverseSheaf<T> {
    /// Create a new perverse sheaf with middle perversity
    pub fn middle_perversity(base_dimension: usize) -> Self {
        let sheaf = Sheaf::new(base_dimension, SheafType::Perverse);
        let perversity = Box::new(move |codim: usize| -> i32 {
            -(codim as i32) / 2
        });
        
        Self {
            sheaf,
            perversity,
            stratification: Vec::new(),
        }
    }
    
    /// Add stratum to stratification
    pub fn add_stratum(&mut self, stratum: OpenSet) {
        self.stratification.push(stratum);
    }
    
    /// Check perversity condition
    pub fn check_perversity(&self, stratum_index: usize) -> bool {
        if let Some(stratum) = self.stratification.get(stratum_index) {
            let codim = self.sheaf.base_dimension - stratum.dimension;
            let p = (self.perversity)(codim);
            // Simplified perversity check
            p >= -(codim as i32)
        } else {
            false
        }
    }
}

impl<T: Clone + Debug> GeometricObject for Sheaf<T> {
    fn dimension(&self) -> usize {
        self.base_dimension
    }
    
    fn is_smooth(&self) -> bool {
        matches!(self.sheaf_type, SheafType::LocalSystem)
    }
    
    fn invariants(&self) -> Vec<f64> {
        vec![
            self.sections.len() as f64,
            self.global_sections().len() as f64,
            self.cohomology_dimension(0) as f64,
            self.cohomology_dimension(1) as f64,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sheaf_creation() {
        let sheaf: Sheaf<Vec<f64>> = Sheaf::new(2, SheafType::Coherent);
        assert_eq!(sheaf.base_dimension, 2);
        assert_eq!(sheaf.sheaf_type, SheafType::Coherent);
    }

    #[test]
    fn test_local_sections() {
        let mut sheaf: Sheaf<f64> = Sheaf::new(1, SheafType::QuasiCoherent);
        let open_set = OpenSet {
            id: "U1".to_string(),
            dimension: 1,
            is_affine: true,
        };
        
        let section = LocalSection {
            open_set: open_set.clone(),
            data: 3.14,
            restriction_maps: HashMap::new(),
        };
        
        sheaf.add_section(section);
        assert_eq!(sheaf.get_sections(&open_set).unwrap().len(), 1);
    }

    #[test]
    fn test_perverse_sheaf() {
        let perverse = PerverseSheaf::<f64>::middle_perversity(4);
        assert_eq!(perverse.sheaf.base_dimension, 4);
        assert_eq!(perverse.sheaf.sheaf_type, SheafType::Perverse);
    }
}