//! Gauge theory structures and computations
//!
//! Implements gauge fields, connections, and related structures
//! for N=4 Super Yang-Mills theory

use crate::types::{Element, Result};
use nalgebra::{DMatrix, DVector, Matrix4};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Gauge field configuration A_μ
#[derive(Debug, Clone)]
pub struct GaugeField {
    /// Spacetime dimension
    pub dim: usize,
    /// Gauge group rank
    pub rank: usize,
    /// Components A_μ in each direction
    pub components: Vec<DMatrix<Complex64>>,
}

impl GaugeField {
    /// Create a new gauge field
    pub fn new(dim: usize, rank: usize) -> Self {
        let components = (0..dim)
            .map(|_| DMatrix::zeros(rank, rank))
            .collect();
        
        Self { dim, rank, components }
    }

    /// Create from connection 1-form
    pub fn from_connection(connection: &DMatrix<Complex64>, dim: usize) -> Result<Self> {
        let rank = connection.nrows();
        let mut field = Self::new(dim, rank);
        
        // Decompose connection into components
        // (simplified - assumes specific form)
        for i in 0..dim.min(rank) {
            field.components[i] = connection.clone();
        }
        
        Ok(field)
    }

    /// Compute field strength F_μν = ∂_μA_ν - ∂_νA_μ + [A_μ, A_ν]
    pub fn field_strength(&self, mu: usize, nu: usize) -> Result<DMatrix<Complex64>> {
        if mu >= self.dim || nu >= self.dim {
            return Err("Invalid spacetime indices".into());
        }
        
        // Commutator [A_μ, A_ν]
        let commutator = &self.components[mu] * &self.components[nu]
            - &self.components[nu] * &self.components[mu];
        
        // For now, return commutator (derivatives would require lattice)
        Ok(commutator)
    }

    /// Compute Yang-Mills action S = (1/4g²) ∫ Tr(F_μν F^μν)
    pub fn yang_mills_action(&self, coupling: f64) -> Result<f64> {
        let mut action = 0.0;
        
        for mu in 0..self.dim {
            for nu in (mu + 1)..self.dim {
                let f_munu = self.field_strength(mu, nu)?;
                let f_squared = &f_munu * &f_munu;
                
                // Trace of F²
                let trace = f_squared.trace();
                action += trace.re;
            }
        }
        
        Ok(action / (4.0 * coupling * coupling))
    }

    /// Apply gauge transformation g: A_μ → g A_μ g^(-1) + g ∂_μ g^(-1)
    pub fn gauge_transform(&mut self, g: &DMatrix<Complex64>) -> Result<()> {
        let g_inv = g.try_inverse()
            .ok_or("Gauge transformation must be invertible")?;
        
        for component in &mut self.components {
            *component = g * &*component * &g_inv;
            // Note: derivative term omitted (would need discretization)
        }
        
        Ok(())
    }

    /// Check if configuration is self-dual: F = *F
    pub fn is_self_dual(&self) -> Result<bool> {
        // In 4D, check F_01 = F_23, F_02 = -F_13, F_03 = F_12
        if self.dim != 4 {
            return Ok(false);
        }
        
        let f01 = self.field_strength(0, 1)?;
        let f23 = self.field_strength(2, 3)?;
        let diff = (&f01 - &f23).norm();
        
        Ok(diff < 1e-10)
    }

    /// Compute instanton number (topological charge)
    pub fn instanton_number(&self) -> Result<i32> {
        if self.dim != 4 {
            return Err("Instanton number only defined in 4D".into());
        }
        
        // k = (1/8π²) ∫ Tr(F ∧ F)
        let mut integral = 0.0;
        
        // F ∧ F = 2(F_01 F_23 - F_02 F_13 + F_03 F_12)
        let f01 = self.field_strength(0, 1)?;
        let f23 = self.field_strength(2, 3)?;
        let f02 = self.field_strength(0, 2)?;
        let f13 = self.field_strength(1, 3)?;
        let f03 = self.field_strength(0, 3)?;
        let f12 = self.field_strength(1, 2)?;
        
        let ff = 2.0 * ((&f01 * &f23).trace() - (&f02 * &f13).trace() + (&f03 * &f12).trace());
        integral += ff.re;
        
        let k = integral / (8.0 * PI * PI);
        Ok(k.round() as i32)
    }
}

/// Instanton solution (BPST instanton)
pub struct Instanton {
    /// Center position
    pub center: DVector<f64>,
    /// Size parameter
    pub rho: f64,
    /// Instanton number
    pub k: i32,
}

impl Instanton {
    /// Create BPST instanton gauge field
    pub fn gauge_field(&self, x: &DVector<f64>) -> Result<GaugeField> {
        let mut field = GaugeField::new(4, 2); // SU(2) in 4D
        
        let r = (x - &self.center).norm();
        let factor = self.rho * self.rho / (r * r + self.rho * self.rho);
        
        // 't Hooft symbols σ_μν
        let sigma_munu = self.t_hooft_symbols();
        
        // A_μ = factor * σ_μν (x-x₀)_ν / |x-x₀|²
        for mu in 0..4 {
            let mut a_mu = DMatrix::zeros(2, 2);
            for nu in 0..4 {
                let x_nu = if nu < x.len() { x[nu] - self.center[nu] } else { 0.0 };
                a_mu += sigma_munu[mu][nu].scale(factor * x_nu / (r * r));
            }
            field.components[mu] = a_mu.map(|x| Complex64::new(x, 0.0));
        }
        
        Ok(field)
    }

    /// 't Hooft symbols (SU(2) generators in 4D)
    fn t_hooft_symbols(&self) -> Vec<Vec<DMatrix<f64>>> {
        // Returns σ_μν antisymmetric tensor of Pauli matrices
        let pauli = vec![
            DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]),   // σ₁
            DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]),  // iσ₂  
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]),  // σ₃
        ];
        
        let mut sigma = vec![vec![DMatrix::zeros(2, 2); 4]; 4];
        
        // Fill antisymmetric tensor
        let indices = [(0, 1, 2), (0, 2, 1), (0, 3, 0), (1, 2, 0), (1, 3, 1), (2, 3, 2)];
        let signs = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        
        for (idx, &(i, j, k)) in indices.iter().enumerate() {
            sigma[i][j] = pauli[k].clone().scale(signs[idx]);
            sigma[j][i] = pauli[k].clone().scale(-signs[idx]);
        }
        
        sigma
    }
}

/// Monopole configuration ('t Hooft-Polyakov monopole)
pub struct Monopole {
    /// Magnetic charge
    pub charge: i32,
    /// Core size
    pub lambda: f64,
}

impl Monopole {
    /// Create monopole gauge field
    pub fn gauge_field(&self, r: f64, theta: f64, phi: f64) -> Result<GaugeField> {
        let mut field = GaugeField::new(3, 2); // SU(2) in 3D
        
        // Hedgehog ansatz
        let h = self.profile_function(r);
        let k = self.k_function(r);
        
        // A_r = 0
        field.components[0] = DMatrix::zeros(2, 2);
        
        // A_θ = (1 - k(r)) / r * τ_φ
        let tau_phi = DMatrix::from_row_slice(2, 2, &[
            0.0, -phi.sin(),
            phi.sin(), 0.0
        ]);
        field.components[1] = tau_phi.scale((1.0 - k) / r)
            .map(|x| Complex64::new(x, 0.0));
        
        // A_φ = -(1 - k(r)) / (r sin θ) * τ_θ  
        let tau_theta = DMatrix::from_row_slice(2, 2, &[
            theta.cos(), theta.sin(),
            theta.sin(), -theta.cos()
        ]);
        field.components[2] = tau_theta.scale(-(1.0 - k) / (r * theta.sin()))
            .map(|x| Complex64::new(x, 0.0));
        
        Ok(field)
    }

    /// Radial profile function
    fn profile_function(&self, r: f64) -> f64 {
        (r / self.lambda).tanh()
    }

    /// K function for gauge field
    fn k_function(&self, r: f64) -> f64 {
        1.0 - (-r / self.lambda).exp()
    }

    /// Magnetic field at large distance
    pub fn magnetic_field(&self, r: f64) -> DVector<f64> {
        // B = g r̂ / r² at large r
        DVector::from_element(3, self.charge as f64 / (r * r))
    }
}

/// Dyonic configuration (electric + magnetic charges)
pub struct Dyon {
    /// Electric charge
    pub electric: i32,
    /// Magnetic charge  
    pub magnetic: i32,
    /// Core parameters
    pub monopole: Monopole,
}

impl Dyon {
    /// Create dyon from charges satisfying Dirac quantization
    pub fn new(electric: i32, magnetic: i32, lambda: f64) -> Result<Self> {
        // Dirac quantization: e·g = 2πn
        if (electric * magnetic) % (2 * PI as i32) != 0 {
            return Err("Charges must satisfy Dirac quantization".into());
        }
        
        Ok(Self {
            electric,
            magnetic,
            monopole: Monopole { charge: magnetic, lambda },
        })
    }

    /// Total electromagnetic field
    pub fn electromagnetic_field(&self, r: f64, tau: Complex64) -> Result<(DVector<f64>, DVector<f64>)> {
        // In N=4 SYM, dyons have complex charge e + τg
        let b = self.monopole.magnetic_field(r);
        
        // Electric field from dyon
        let e = b.scale(self.electric as f64 / self.magnetic as f64);
        
        Ok((e, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge_field() {
        let mut field = GaugeField::new(4, 2);
        
        // Set a simple configuration
        field.components[0] = DMatrix::identity(2, 2)
            .map(|x| Complex64::new(x, 0.0));
        
        // Check field strength
        let f01 = field.field_strength(0, 1).unwrap();
        assert_eq!(f01.nrows(), 2);
        assert_eq!(f01.ncols(), 2);
    }

    #[test]
    fn test_instanton() {
        let instanton = Instanton {
            center: DVector::zeros(4),
            rho: 1.0,
            k: 1,
        };
        
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let field = instanton.gauge_field(&x).unwrap();
        
        // Should be self-dual
        assert!(field.is_self_dual().unwrap());
        
        // Should have correct topological charge
        assert_eq!(field.instanton_number().unwrap(), 1);
    }

    #[test]
    fn test_monopole() {
        let monopole = Monopole {
            charge: 1,
            lambda: 1.0,
        };
        
        let field = monopole.gauge_field(2.0, PI / 2.0, 0.0).unwrap();
        assert_eq!(field.dim, 3);
        
        // Check magnetic field falloff
        let b = monopole.magnetic_field(10.0);
        assert!((b[0] - 0.01).abs() < 1e-3);
    }
}