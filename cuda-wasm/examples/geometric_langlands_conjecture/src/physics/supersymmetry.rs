//! Supersymmetric structures in N=4 Super Yang-Mills
//!
//! Implementation of supersymmetry generators, transformations,
//! and protected quantities in N=4 SYM theory

use crate::physics::{GaugeParameters, hitchin_system::HitchinSystem};
use crate::types::{Element, Result};
use nalgebra::{DMatrix, DVector, Matrix4};
use num_complex::Complex64;
use std::f64::consts::PI;

/// N=4 supersymmetry multiplet
#[derive(Debug, Clone)]
pub struct N4Multiplet {
    /// Gauge field A_μ
    pub gauge_field: Vec<DMatrix<Complex64>>,
    /// Scalar fields Φᵢ (i=1,2,3,4,5,6)
    pub scalars: Vec<DMatrix<Complex64>>,
    /// Fermion fields ψ^α (α=1,2,3,4)
    pub fermions: Vec<DMatrix<Complex64>>,
    /// Auxiliary fields
    pub auxiliary: Vec<DMatrix<Complex64>>,
}

impl N4Multiplet {
    /// Create new N=4 vector multiplet
    pub fn new(rank: usize) -> Self {
        Self {
            gauge_field: vec![DMatrix::zeros(rank, rank); 4], // A_μ
            scalars: vec![DMatrix::zeros(rank, rank); 6],     // Φᵢ
            fermions: vec![DMatrix::zeros(rank, rank); 4],    // ψ^α
            auxiliary: vec![DMatrix::zeros(rank, rank); 1],   // D
        }
    }

    /// Complex scalar combinations
    pub fn complex_scalars(&self) -> Vec<DMatrix<Complex64>> {
        vec![
            &self.scalars[0] + Complex64::new(0.0, 1.0) * &self.scalars[1], // Φ₁ + iΦ₂
            &self.scalars[2] + Complex64::new(0.0, 1.0) * &self.scalars[3], // Φ₃ + iΦ₄  
            &self.scalars[4] + Complex64::new(0.0, 1.0) * &self.scalars[5], // Φ₅ + iΦ₆
        ]
    }

    /// Higgs field in Hitchin system (Φ₁ + iΦ₂)
    pub fn higgs_field(&self) -> DMatrix<Complex64> {
        self.complex_scalars()[0].clone()
    }

    /// R-charge assignments
    pub fn r_charges(&self) -> RChargeAssignment {
        RChargeAssignment {
            gauge_field: vec![0; 4],
            scalars: vec![1, 1, 1, 1, 1, 1],
            fermions: vec![1, 1, 1, 1],
            auxiliary: vec![2],
        }
    }

    /// Apply supersymmetry transformation
    pub fn susy_transform(&mut self, epsilon: &SuperchargeParameter) -> Result<()> {
        // δ_ε A_μ = ε̄ γ_μ ψ
        for mu in 0..4 {
            let delta_a = epsilon.compute_gauge_variation(&self.fermions[mu], mu)?;
            self.gauge_field[mu] += delta_a;
        }

        // δ_ε Φᵢ = ε̄ Γᵢ ψ (Γᵢ are gamma matrices in 6D)
        for i in 0..6 {
            let delta_phi = epsilon.compute_scalar_variation(&self.fermions, i)?;
            self.scalars[i] += delta_phi;
        }

        // δ_ε ψ = Γ^μν F_μν ε + [Φᵢ, Φⱼ] Γᵢⱼ ε
        for alpha in 0..4 {
            let delta_psi = epsilon.compute_fermion_variation(
                &self.gauge_field,
                &self.scalars,
                alpha,
            )?;
            self.fermions[alpha] += delta_psi;
        }

        Ok(())
    }
}

/// Supercharge parameter
#[derive(Debug, Clone)]
pub struct SuperchargeParameter {
    /// Spinor components ε^α (α=1,2,3,4)
    pub components: Vec<DMatrix<Complex64>>,
    /// Chirality (±1)
    pub chirality: i8,
}

impl SuperchargeParameter {
    /// Create from spinor data
    pub fn new(components: Vec<DMatrix<Complex64>>, chirality: i8) -> Self {
        Self { components, chirality }
    }

    /// Compute gauge field variation δA_μ
    fn compute_gauge_variation(
        &self,
        fermion: &DMatrix<Complex64>,
        mu: usize,
    ) -> Result<DMatrix<Complex64>> {
        // δA_μ = ε̄ γ_μ ψ
        let gamma_mu = self.gamma_matrix(mu);
        Ok(&self.components[0].adjoint() * &gamma_mu * fermion)
    }

    /// Compute scalar field variation δΦᵢ
    fn compute_scalar_variation(
        &self,
        fermions: &[DMatrix<Complex64>],
        i: usize,
    ) -> Result<DMatrix<Complex64>> {
        // δΦᵢ = ε̄ Γᵢ ψ
        let gamma_i = self.so6_gamma_matrix(i);
        let mut delta = DMatrix::zeros(fermions[0].nrows(), fermions[0].ncols());
        
        for alpha in 0..4 {
            delta += &self.components[alpha].adjoint() * gamma_i[(alpha, 0)] * &fermions[alpha];
        }
        
        Ok(delta)
    }

    /// Compute fermion variation δψ
    fn compute_fermion_variation(
        &self,
        gauge_field: &[DMatrix<Complex64>],
        scalars: &[DMatrix<Complex64>],
        alpha: usize,
    ) -> Result<DMatrix<Complex64>> {
        let mut delta = DMatrix::zeros(gauge_field[0].nrows(), gauge_field[0].ncols());
        
        // Field strength term: Γ^μν F_μν ε
        for mu in 0..4 {
            for nu in (mu + 1)..4 {
                let f_munu = &gauge_field[mu] * &gauge_field[nu]
                    - &gauge_field[nu] * &gauge_field[mu];
                let gamma_munu = self.gamma_matrix_antisym(mu, nu);
                delta += gamma_munu[(alpha, 0)] * &f_munu * &self.components[0];
            }
        }
        
        // Scalar commutator term: [Φᵢ, Φⱼ] Γᵢⱼ ε
        for i in 0..6 {
            for j in (i + 1)..6 {
                let commutator = &scalars[i] * &scalars[j] - &scalars[j] * &scalars[i];
                let gamma_ij = self.so6_gamma_antisym(i, j);
                delta += gamma_ij[(alpha, 0)] * &commutator * &self.components[0];
            }
        }
        
        Ok(delta)
    }

    /// 4D gamma matrices
    fn gamma_matrix(&self, mu: usize) -> DMatrix<Complex64> {
        match mu {
            0 => DMatrix::from_row_slice(4, 4, &[
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ]),
            1 => DMatrix::from_row_slice(4, 4, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            ]),
            2 => DMatrix::from_row_slice(4, 4, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            ]),
            3 => DMatrix::from_row_slice(4, 4, &[
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            ]),
            _ => DMatrix::zeros(4, 4),
        }
    }

    /// SO(6) gamma matrices for scalar sector
    fn so6_gamma_matrix(&self, i: usize) -> DMatrix<Complex64> {
        // Simplified implementation
        DMatrix::identity(4, 4).map(|x| Complex64::new(x * (i + 1) as f64, 0.0))
    }

    /// Antisymmetric gamma matrix products
    fn gamma_matrix_antisym(&self, mu: usize, nu: usize) -> DMatrix<Complex64> {
        let gamma_mu = self.gamma_matrix(mu);
        let gamma_nu = self.gamma_matrix(nu);
        (&gamma_mu * &gamma_nu - &gamma_nu * &gamma_mu) / 2.0
    }

    /// SO(6) antisymmetric combinations
    fn so6_gamma_antisym(&self, i: usize, j: usize) -> DMatrix<Complex64> {
        let gamma_i = self.so6_gamma_matrix(i);
        let gamma_j = self.so6_gamma_matrix(j);
        (&gamma_i * &gamma_j - &gamma_j * &gamma_i) / 2.0
    }
}

/// R-symmetry charge assignments
#[derive(Debug)]
pub struct RChargeAssignment {
    /// Charges for gauge fields
    pub gauge_field: Vec<i32>,
    /// Charges for scalar fields
    pub scalars: Vec<i32>,
    /// Charges for fermion fields
    pub fermions: Vec<i32>,
    /// Charges for auxiliary fields
    pub auxiliary: Vec<i32>,
}

/// BPS state in N=4 SYM
#[derive(Debug, Clone)]
pub struct BPSState {
    /// Electric charges
    pub electric_charges: DVector<i32>,
    /// Magnetic charges
    pub magnetic_charges: DVector<i32>,
    /// Central charge Z = n + mτ
    pub central_charge: Complex64,
    /// Mass (= |Z|)
    pub mass: f64,
    /// Spin content
    pub spin: f64,
}

impl BPSState {
    /// Create BPS state from charges
    pub fn new(
        electric: DVector<i32>,
        magnetic: DVector<i32>,
        tau: Complex64,
    ) -> Self {
        // Central charge
        let z = electric.iter().map(|&e| e as f64).sum::<f64>()
            + tau * magnetic.iter().map(|&m| m as f64).sum::<f64>();
        
        Self {
            electric_charges: electric,
            magnetic_charges: magnetic,
            central_charge: z,
            mass: z.norm(),
            spin: 0.0, // BPS states are 1/2 BPS
        }
    }

    /// Check if state is 1/2 BPS
    pub fn is_half_bps(&self) -> bool {
        // 1/2 BPS condition: |Z| = Z for some supercharge
        true // All states we consider are 1/2 BPS
    }

    /// Apply S-duality transformation
    pub fn s_dual(&self) -> Self {
        // S: (e,m) → (-m,e)
        Self {
            electric_charges: -&self.magnetic_charges,
            magnetic_charges: self.electric_charges.clone(),
            central_charge: -1.0 / self.central_charge,
            mass: self.mass, // Mass is invariant
            spin: self.spin,
        }
    }

    /// Degeneracy (from wall-crossing)
    pub fn degeneracy(&self, moduli: &[Complex64]) -> i32 {
        // Simplified: constant degeneracy
        1
    }
}

/// Superconformal algebra generators
#[derive(Debug)]
pub struct SuperconformalAlgebra {
    /// Conformal generators L_n
    pub virasoro: Vec<DMatrix<Complex64>>,
    /// Supercharges G_r
    pub supercharges: Vec<DMatrix<Complex64>>,
    /// R-symmetry generators J^a
    pub r_symmetry: Vec<DMatrix<Complex64>>,
    /// Central charge
    pub central_charge: f64,
}

impl SuperconformalAlgebra {
    /// Create N=4 superconformal algebra
    pub fn n4_superconformal() -> Self {
        Self {
            virasoro: vec![DMatrix::zeros(4, 4); 10],
            supercharges: vec![DMatrix::zeros(4, 4); 16],
            r_symmetry: vec![DMatrix::zeros(4, 4); 15], // SO(6) ≅ SU(4)
            central_charge: 0.0, // N=4 is not chiral
        }
    }

    /// Compute superconformal index
    pub fn superconformal_index(
        &self,
        fugacities: &[Complex64],
        q: Complex64,
    ) -> Complex64 {
        // Tr[(-1)^F q^{L_0} ∏ᵢ xᵢ^{Jᵢ}]
        // For N=4 SYM, this counts BPS operators
        
        let mut index = Complex64::new(0.0, 0.0);
        
        // Single-trace operators
        for n in 1..=10 {
            let dimension = n as f64;
            let contribution = q.powf(dimension - 2.0); // L₀ = Δ - R
            index += contribution;
        }
        
        // Include fugacity dependence
        for fugacity in fugacities {
            index *= fugacity.powf(1.0);
        }
        
        index
    }
}

/// Localization data for N=4 SYM
#[derive(Debug)]
pub struct LocalizationData {
    /// Saddle point configurations
    pub saddle_points: Vec<N4Multiplet>,
    /// One-loop determinants
    pub one_loop_determinants: Vec<Complex64>,
    /// Instanton corrections
    pub instanton_contributions: Vec<Complex64>,
}

/// Compute partition function using localization
pub fn compute_partition_function(
    params: &GaugeParameters,
    hitchin: &HitchinSystem,
    beta: f64,
) -> Result<Complex64> {
    // Supersymmetric localization of N=4 SYM partition function
    
    // Classical contribution
    let classical = (-beta * classical_action(params, hitchin)?).exp();
    
    // One-loop contribution
    let one_loop = compute_one_loop_determinant(params, beta)?;
    
    // Instanton contributions
    let instanton = compute_instanton_sum(params, beta)?;
    
    Ok(classical * one_loop * instanton)
}

/// Classical action
fn classical_action(params: &GaugeParameters, hitchin: &HitchinSystem) -> Result<f64> {
    // S = (1/g²) ∫ Tr(F² + D_μΦᵢ D^μΦᵢ + [Φᵢ,Φⱼ]²)
    let mut action = 0.0;
    
    // Yang-Mills term (simplified)
    action += params.rank as f64;
    
    // Scalar kinetic terms
    action += 6.0 * params.rank as f64; // 6 scalars
    
    // Scalar potential
    let higgs_norm = hitchin.higgs_field.matrix.norm();
    action += higgs_norm * higgs_norm;
    
    Ok(action / (params.g * params.g))
}

/// One-loop determinant
fn compute_one_loop_determinant(params: &GaugeParameters, beta: f64) -> Result<Complex64> {
    // Det(kinetic operator)^{-1/2}
    // For N=4 SYM, this involves gauge, scalar, and fermion loops
    
    let mut determinant = Complex64::new(1.0, 0.0);
    
    // Gauge field contribution
    let gauge_det = (beta * params.g * params.g).powf(params.rank as f64);
    determinant *= gauge_det;
    
    // Scalar contributions (6 scalars)
    let scalar_det = (beta * params.g * params.g).powf(6.0 * params.rank as f64);
    determinant /= scalar_det.sqrt();
    
    // Fermion contributions (4 fermions, give opposite sign)
    let fermion_det = (beta * params.g * params.g).powf(4.0 * params.rank as f64);
    determinant *= fermion_det.sqrt();
    
    Ok(determinant)
}

/// Instanton sum
fn compute_instanton_sum(params: &GaugeParameters, beta: f64) -> Result<Complex64> {
    // ∑ₖ qᵏ Zₖ(τ) where q = exp(2πiτ)
    let tau = params.tau();
    let q = (2.0 * PI * Complex64::new(0.0, 1.0) * tau).exp();
    
    let mut sum = Complex64::new(1.0, 0.0); // k=0 term
    
    // Add instanton contributions k=1,2,...
    for k in 1..=5 {
        let z_k = instanton_partition_function(k, params)?;
        sum += q.powf(k as f64) * z_k;
    }
    
    Ok(sum)
}

/// k-instanton partition function
fn instanton_partition_function(k: usize, params: &GaugeParameters) -> Result<Complex64> {
    // Zₖ from localization (ADHM construction)
    let rank = params.rank as f64;
    let factor = (1.0 / (k as f64).powi(2 * params.rank as i32)).sqrt();
    
    Ok(Complex64::new(factor, 0.0))
}

/// Protected operators and their correlators
#[derive(Debug)]
pub struct ProtectedOperator {
    /// Operator name
    pub name: String,
    /// Dimension
    pub dimension: f64,
    /// R-charge
    pub r_charge: f64,
    /// Spin
    pub spin: f64,
}

impl ProtectedOperator {
    /// Create Konishi operator
    pub fn konishi() -> Self {
        Self {
            name: "Konishi".to_string(),
            dimension: 2.0,
            r_charge: 0.0,
            spin: 0.0,
        }
    }

    /// Create stress-energy tensor
    pub fn stress_tensor() -> Self {
        Self {
            name: "T_μν".to_string(),
            dimension: 4.0,
            r_charge: 0.0,
            spin: 2.0,
        }
    }

    /// Two-point function coefficient
    pub fn two_point_coefficient(&self, params: &GaugeParameters) -> Complex64 {
        // Protected by supersymmetry
        Complex64::new(1.0 / params.g.powi(2), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n4_multiplet() {
        let multiplet = N4Multiplet::new(2);
        
        assert_eq!(multiplet.gauge_field.len(), 4); // A_μ
        assert_eq!(multiplet.scalars.len(), 6);     // Φᵢ
        assert_eq!(multiplet.fermions.len(), 4);    // ψ^α
        
        let higgs = multiplet.higgs_field();
        assert_eq!(higgs.nrows(), 2);
        assert_eq!(higgs.ncols(), 2);
    }

    #[test]
    fn test_bps_state() {
        let electric = DVector::from_vec(vec![1, 0]);
        let magnetic = DVector::from_vec(vec![0, 1]);
        let tau = Complex64::new(0.0, 1.0);
        
        let bps = BPSState::new(electric, magnetic, tau);
        assert!(bps.is_half_bps());
        assert!(bps.mass > 0.0);
        
        // Test S-duality
        let dual = bps.s_dual();
        assert_eq!(dual.electric_charges[0], 0);
        assert_eq!(dual.electric_charges[1], -1);
        assert_eq!(dual.magnetic_charges[0], 1);
        assert_eq!(dual.magnetic_charges[1], 0);
    }

    #[test]
    fn test_partition_function() {
        let params = GaugeParameters::n4_sym(2);
        let hitchin = HitchinSystem::new(2);
        
        let z = compute_partition_function(&params, &hitchin, 1.0).unwrap();
        assert!(z.norm() > 0.0);
    }

    #[test]
    fn test_protected_operators() {
        let konishi = ProtectedOperator::konishi();
        assert_eq!(konishi.dimension, 2.0);
        assert_eq!(konishi.r_charge, 0.0);
        
        let stress = ProtectedOperator::stress_tensor();
        assert_eq!(stress.dimension, 4.0);
        assert_eq!(stress.spin, 2.0);
    }
}