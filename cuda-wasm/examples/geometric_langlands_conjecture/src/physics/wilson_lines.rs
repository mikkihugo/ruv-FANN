//! Wilson and 't Hooft line operators
//!
//! Implements non-local observables that play a central role
//! in the gauge theory / Langlands correspondence

use crate::physics::{GaugeParameters, gauge_theory::GaugeField};
use crate::types::{Element, Result};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Wilson line operator
#[derive(Debug, Clone)]
pub struct WilsonLine {
    /// Path in spacetime
    pub path: Vec<DVector<f64>>,
    /// Representation of gauge group
    pub representation: Representation,
    /// Gauge field configuration
    pub gauge_field: Option<GaugeField>,
}

/// 't Hooft line operator (magnetic dual)
#[derive(Debug, Clone)]
pub struct THooftLine {
    /// Path in spacetime
    pub path: Vec<DVector<f64>>,
    /// Magnetic weight
    pub magnetic_weight: DVector<i32>,
    /// Singular gauge field
    pub gauge_field: Option<GaugeField>,
}

/// Representation of gauge group
#[derive(Debug, Clone)]
pub enum Representation {
    /// Fundamental representation
    Fundamental,
    /// Anti-fundamental representation
    Antifundamental,
    /// Adjoint representation
    Adjoint,
    /// General highest weight representation
    HighestWeight(DVector<i32>),
}

impl Representation {
    /// Dimension of representation for SU(N)
    pub fn dimension(&self, rank: usize) -> usize {
        match self {
            Self::Fundamental => rank,
            Self::Antifundamental => rank,
            Self::Adjoint => rank * rank - 1,
            Self::HighestWeight(weight) => {
                // Use Weyl dimension formula (simplified)
                weight.iter().map(|&w| (w + 1) as usize).product()
            }
        }
    }

    /// Casimir eigenvalue
    pub fn casimir(&self, rank: usize) -> f64 {
        match self {
            Self::Fundamental => (rank as f64 * rank as f64 - 1.0) / (2.0 * rank as f64),
            Self::Antifundamental => (rank as f64 * rank as f64 - 1.0) / (2.0 * rank as f64),
            Self::Adjoint => rank as f64,
            Self::HighestWeight(weight) => {
                // Quadratic Casimir
                let rho: DVector<f64> = DVector::from_fn(weight.len(), |i| (weight.len() - i) as f64);
                let lambda = weight.map(|x| x as f64);
                (&lambda + &rho).dot(&(&lambda + &rho)) - rho.dot(&rho)
            }
        }
    }
}

/// Compute Wilson line expectation value
pub fn compute_wilson_line(
    params: &GaugeParameters,
    path: &[DVector<f64>],
    representation: &str,
) -> Result<Complex64> {
    let rep = match representation {
        "fundamental" => Representation::Fundamental,
        "antifundamental" => Representation::Antifundamental,
        "adjoint" => Representation::Adjoint,
        _ => return Err("Unknown representation".into()),
    };
    
    // Path-ordered exponential W = P exp(i∮ A)
    let mut wilson = DMatrix::<Complex64>::identity(params.rank, params.rank);
    
    // Discretized path integral
    for i in 0..path.len() - 1 {
        let segment = &path[i + 1] - &path[i];
        let midpoint = (&path[i] + &path[i + 1]).scale(0.5);
        
        // Simplified: assume constant field along segment
        let a_segment = evaluate_gauge_field(&midpoint, params)?;
        
        // Evolution operator for segment
        let evolution = (&a_segment * segment.norm()).exp();
        wilson = evolution * wilson;
    }
    
    // Take trace in given representation
    let trace = match rep {
        Representation::Fundamental => wilson.trace(),
        Representation::Adjoint => {
            // Trace in adjoint = |Tr|²
            let tr = wilson.trace();
            tr * tr.conj() - Complex64::new(params.rank as f64, 0.0)
        }
        _ => wilson.trace(),
    };
    
    // Include perimeter law factor
    let perimeter = compute_perimeter(path);
    let vev = trace * (-params.g * params.g * perimeter / 4.0).exp();
    
    Ok(vev)
}

/// Compute 't Hooft line expectation value
pub fn compute_t_hooft_line(
    params: &GaugeParameters,
    path: &[DVector<f64>],
    magnetic_charge: i32,
) -> Result<Complex64> {
    // 't Hooft line creates magnetic flux
    // Under S-duality, this maps to Wilson line in dual theory
    
    let dual_params = params.s_dual();
    
    // Magnetic charge m → electric charge m in dual theory
    let dual_rep = if magnetic_charge == 1 {
        "fundamental"
    } else {
        "adjoint"
    };
    
    // Compute as Wilson line in S-dual theory
    let dual_wilson = compute_wilson_line(&dual_params, path, dual_rep)?;
    
    // Include S-duality phase factor
    let phase = (PI * params.tau().im / 4.0).exp();
    
    Ok(dual_wilson * phase)
}

/// Evaluate gauge field at a point (simplified)
fn evaluate_gauge_field(
    x: &DVector<f64>,
    params: &GaugeParameters,
) -> Result<DMatrix<Complex64>> {
    // Placeholder: would use actual gauge field configuration
    // For now, return a simple ansatz
    
    let r = x.norm();
    let scale = (-r / params.g).exp();
    
    // Random hermitian matrix (simplified)
    let mut a = DMatrix::<Complex64>::zeros(params.rank, params.rank);
    for i in 0..params.rank {
        for j in i..params.rank {
            if i == j {
                a[(i, j)] = Complex64::new(scale * (i as f64).sin(), 0.0);
            } else {
                let real = scale * ((i + j) as f64).cos() / params.rank as f64;
                let imag = scale * ((i - j) as f64).sin() / params.rank as f64;
                a[(i, j)] = Complex64::new(real, imag);
                a[(j, i)] = Complex64::new(real, -imag);
            }
        }
    }
    
    Ok(a)
}

/// Compute perimeter of path
fn compute_perimeter(path: &[DVector<f64>]) -> f64 {
    let mut perimeter = 0.0;
    for i in 0..path.len() - 1 {
        perimeter += (&path[i + 1] - &path[i]).norm();
    }
    perimeter
}

/// Wilson-'t Hooft loop (dyonic operator)
#[derive(Debug)]
pub struct WilsonTHooftLoop {
    /// Electric charge
    pub electric: i32,
    /// Magnetic charge
    pub magnetic: i32,
    /// Path
    pub path: Vec<DVector<f64>>,
}

impl WilsonTHooftLoop {
    /// Compute expectation value
    pub fn expectation_value(&self, params: &GaugeParameters) -> Result<Complex64> {
        // Dyonic operator with charges (e,m)
        // ⟨L_{e,m}⟩ = exp(2πi(eθ/2π + m*4πi/g²))
        
        let tau = params.tau();
        let charge = Complex64::new(self.electric as f64, 0.0)
            + Complex64::new(self.magnetic as f64, 0.0) * tau;
        
        // BPS bound gives mass/action
        let action = charge.norm() * compute_perimeter(&self.path);
        
        Ok((-action).exp())
    }

    /// S-duality transformation
    pub fn s_dual(&self) -> Self {
        // S: (e,m) → (-m,e)
        Self {
            electric: -self.magnetic,
            magnetic: self.electric,
            path: self.path.clone(),
        }
    }

    /// Check if operator is mutually local with another
    pub fn is_mutually_local(&self, other: &Self) -> bool {
        // Dirac pairing ⟨(e₁,m₁), (e₂,m₂)⟩ = e₁m₂ - e₂m₁
        let pairing = self.electric * other.magnetic - other.electric * self.magnetic;
        pairing == 0
    }
}

/// Surface operator (codimension-2 defect)
#[derive(Debug)]
pub struct SurfaceOperator {
    /// Surface in 4D spacetime
    pub surface: Vec<Vec<DVector<f64>>>,
    /// Type of operator
    pub operator_type: SurfaceType,
}

#[derive(Debug, Clone)]
pub enum SurfaceType {
    /// Gukov-Witten surface operator
    GukovWitten { label: DVector<i32> },
    /// Vortex string
    Vortex { flux: i32 },
    /// Monodromy defect
    Monodromy { eigenvalues: DVector<Complex64> },
}

impl SurfaceOperator {
    /// Compute VEV in presence of Wilson line
    pub fn vev_with_wilson(
        &self,
        wilson: &WilsonLine,
        params: &GaugeParameters,
    ) -> Result<Complex64> {
        // Surface operators modify Wilson line VEV
        match &self.operator_type {
            SurfaceType::GukovWitten { label } => {
                // Modifies by character of representation
                let character = label.iter()
                    .enumerate()
                    .map(|(i, &l)| (l as f64 * PI * (i + 1) as f64 / params.rank as f64).cos())
                    .product::<f64>();
                
                let wilson_vev = compute_wilson_line(
                    params,
                    &wilson.path,
                    "fundamental",
                )?;
                
                Ok(wilson_vev * character)
            }
            SurfaceType::Vortex { flux } => {
                // Adds magnetic flux through surface
                let phase = Complex64::new(0.0, 2.0 * PI * *flux as f64 / params.rank as f64).exp();
                let wilson_vev = compute_wilson_line(
                    params,
                    &wilson.path,
                    "fundamental",
                )?;
                
                Ok(wilson_vev * phase)
            }
            SurfaceType::Monodromy { eigenvalues } => {
                // Creates monodromy around surface
                let monodromy_factor = eigenvalues.iter()
                    .map(|&z| z)
                    .fold(Complex64::new(1.0, 0.0), |acc, z| acc * z);
                
                let wilson_vev = compute_wilson_line(
                    params,
                    &wilson.path,
                    "fundamental",
                )?;
                
                Ok(wilson_vev * monodromy_factor)
            }
        }
    }
}

/// Line operator algebra
#[derive(Debug)]
pub struct LineOperatorAlgebra {
    /// Basis of line operators
    pub basis: Vec<WilsonTHooftLoop>,
    /// Fusion rules
    pub fusion_rules: Vec<Vec<Vec<(usize, Complex64)>>>,
    /// Braiding matrices
    pub braiding: Vec<Vec<DMatrix<Complex64>>>,
}

impl LineOperatorAlgebra {
    /// Create algebra for given gauge group rank
    pub fn new(rank: usize) -> Self {
        let mut basis = Vec::new();
        
        // Generate basis of mutually local operators
        for e in 0..rank as i32 {
            for m in 0..rank as i32 {
                if gcd(e, m) == 1 {  // Coprime charges
                    basis.push(WilsonTHooftLoop {
                        electric: e,
                        magnetic: m,
                        path: vec![DVector::zeros(4), DVector::from_element(4, 1.0)],
                    });
                }
            }
        }
        
        let n = basis.len();
        let fusion_rules = vec![vec![vec![]; n]; n];
        let braiding = vec![vec![DMatrix::zeros(1, 1); n]; n];
        
        Self { basis, fusion_rules, braiding }
    }

    /// Compute operator product expansion
    pub fn ope(&self, i: usize, j: usize) -> Result<Vec<(usize, Complex64)>> {
        if i >= self.basis.len() || j >= self.basis.len() {
            return Err("Invalid operator indices".into());
        }
        
        // Use fusion rules if computed
        if !self.fusion_rules[i][j].is_empty() {
            return Ok(self.fusion_rules[i][j].clone());
        }
        
        // Otherwise compute from charges
        let op1 = &self.basis[i];
        let op2 = &self.basis[j];
        
        let e_total = op1.electric + op2.electric;
        let m_total = op1.magnetic + op2.magnetic;
        
        // Find result in basis
        for (k, op) in self.basis.iter().enumerate() {
            if op.electric == e_total && op.magnetic == m_total {
                return Ok(vec![(k, Complex64::new(1.0, 0.0))]);
            }
        }
        
        Ok(vec![])
    }

    /// Compute braiding phase
    pub fn braiding_phase(&self, i: usize, j: usize) -> Result<Complex64> {
        let op1 = &self.basis[i];
        let op2 = &self.basis[j];
        
        // Braiding phase from Dirac pairing
        let pairing = op1.electric * op2.magnetic - op2.electric * op1.magnetic;
        let phase = Complex64::new(0.0, 2.0 * PI * pairing as f64).exp();
        
        Ok(phase)
    }
}

/// Helper: greatest common divisor
fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_representation() {
        let fund = Representation::Fundamental;
        assert_eq!(fund.dimension(3), 3);
        
        let adj = Representation::Adjoint;
        assert_eq!(adj.dimension(3), 8);
        
        // Check Casimir values
        let c_fund = fund.casimir(3);
        let c_adj = adj.casimir(3);
        assert!(c_fund < c_adj); // Adjoint has larger Casimir
    }

    #[test]
    fn test_wilson_t_hooft_duality() {
        let wt = WilsonTHooftLoop {
            electric: 1,
            magnetic: 0,
            path: vec![DVector::zeros(4), DVector::from_element(4, 1.0)],
        };
        
        let dual = wt.s_dual();
        assert_eq!(dual.electric, 0);
        assert_eq!(dual.magnetic, 1);
        
        // Check locality
        assert!(wt.is_mutually_local(&wt));
        assert!(!wt.is_mutually_local(&dual));
    }

    #[test]
    fn test_line_algebra() {
        let algebra = LineOperatorAlgebra::new(2);
        assert!(!algebra.basis.is_empty());
        
        // Check braiding is consistent
        if algebra.basis.len() >= 2 {
            let phase12 = algebra.braiding_phase(0, 1).unwrap();
            let phase21 = algebra.braiding_phase(1, 0).unwrap();
            assert!((phase12 * phase21 - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        }
    }
}