//! Core mathematical abstractions for the Geometric Langlands Conjecture
//! 
//! This module provides zero-copy, high-performance implementations of fundamental
//! mathematical objects used in the geometric Langlands program.

pub mod traits;
pub mod category;
pub mod sheaf;
pub mod bundle;
pub mod moduli;
pub mod cohomology;
pub mod representation;
pub mod correspondence;
pub mod algorithms;
pub mod dmodule;

#[cfg(test)]
pub mod geometric_tests;

pub use traits::*;
pub use category::*;
pub use sheaf::*;
pub use bundle::*;
pub use moduli::*;
pub use cohomology::*;
pub use representation::*;
pub use correspondence::*;
pub use dmodule::*;

/// Re-export commonly used types
pub mod prelude {
    pub use super::traits::*;
    pub use super::category::{Category, Functor};
    pub use super::sheaf::{Sheaf, LocalSystem};
    pub use super::bundle::{Bundle, VectorBundle, PrincipalBundle};
    pub use super::moduli::{ModuliSpace, ModuliPoint};
    pub use super::cohomology::{CohomologyGroup, CohomologyComplex};
    pub use super::representation::{Representation, GroupRep};
    pub use super::correspondence::{Correspondence, LanglandsCorrespondence};
}