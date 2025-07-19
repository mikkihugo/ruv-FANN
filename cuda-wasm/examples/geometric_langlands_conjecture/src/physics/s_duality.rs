//! S-duality transformations and validation
//!
//! Implements electromagnetic duality in N=4 Super Yang-Mills theory
//! and its connection to geometric Langlands duality

use crate::physics::GaugeParameters;
use crate::types::{Element, Result};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// S-duality transformation group SL(2,Z)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SL2Z {
    /// Matrix entries [[a,b],[c,d]] with ad-bc = 1
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
}

impl SL2Z {
    /// Identity transformation
    pub fn identity() -> Self {
        Self { a: 1, b: 0, c: 0, d: 1 }
    }

    /// S transformation: τ → -1/τ
    pub fn s_transform() -> Self {
        Self { a: 0, b: -1, c: 1, d: 0 }
    }

    /// T transformation: τ → τ + 1
    pub fn t_transform() -> Self {
        Self { a: 1, b: 1, c: 0, d: 1 }
    }

    /// Verify SL(2,Z) condition
    pub fn is_valid(&self) -> bool {
        self.a * self.d - self.b * self.c == 1
    }

    /// Compose two transformations
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }

    /// Apply to complexified coupling τ
    pub fn transform_tau(&self, tau: Complex64) -> Complex64 {
        let num = Complex64::new(self.a as f64, 0.0) * tau + Complex64::new(self.b as f64, 0.0);
        let den = Complex64::new(self.c as f64, 0.0) * tau + Complex64::new(self.d as f64, 0.0);
        num / den
    }

    /// Apply to (electric, magnetic) charges
    pub fn transform_charges(&self, electric: i32, magnetic: i32) -> (i32, i32) {
        let e_new = self.a * electric + self.b * magnetic;
        let m_new = self.c * electric + self.d * magnetic;
        (e_new, m_new)
    }
}

/// Duality transformation data
#[derive(Debug, Clone)]
pub struct DualityTransform {
    /// SL(2,Z) element
    pub sl2z: SL2Z,
    /// Gauge group transformation
    pub gauge_map: HashMap<String, String>,
    /// Representation mapping
    pub rep_map: HashMap<String, String>,
}

impl DualityTransform {
    /// Create S-duality transformation
    pub fn s_duality() -> Self {
        let mut gauge_map = HashMap::new();
        let mut rep_map = HashMap::new();
        
        // S-duality exchanges representations
        gauge_map.insert("electric".to_string(), "magnetic".to_string());
        gauge_map.insert("magnetic".to_string(), "electric".to_string());
        
        // Fundamental ↔ Antifundamental under S
        rep_map.insert("fundamental".to_string(), "antifundamental".to_string());
        rep_map.insert("antifundamental".to_string(), "fundamental".to_string());
        rep_map.insert("adjoint".to_string(), "adjoint".to_string());
        
        Self {
            sl2z: SL2Z::s_transform(),
            gauge_map,
            rep_map,
        }
    }

    /// Apply to gauge parameters
    pub fn transform_parameters(&self, params: &GaugeParameters) -> GaugeParameters {
        let tau = params.tau();
        let tau_new = self.sl2z.transform_tau(tau);
        
        // Extract new g and θ from transformed τ
        let theta_new = tau_new.re * 2.0 * PI;
        let g_new = (4.0 * PI / tau_new.im).sqrt();
        
        GaugeParameters {
            g: g_new,
            theta: theta_new,
            rank: params.rank,
            n_susy: params.n_susy,
        }
    }
}

/// Transform operator under S-duality
pub fn transform_operator(
    op: &DMatrix<Complex64>,
    params: &GaugeParameters,
) -> Result<DMatrix<Complex64>> {
    // Under S-duality, operators transform by conjugation
    // This is a simplified implementation
    let tau = params.tau();
    let phase = (-PI * tau.im / 2.0).exp();
    
    Ok(op.scale(Complex64::new(phase, 0.0)))
}

/// Montonen-Olive duality data
#[derive(Debug)]
pub struct MontonenOliveDuality {
    /// Original theory coupling
    pub tau_original: Complex64,
    /// Dual theory coupling
    pub tau_dual: Complex64,
    /// Charge lattice transformation
    pub charge_transform: SL2Z,
    /// BPS spectrum mapping
    pub bps_map: HashMap<(i32, i32), (i32, i32)>,
}

impl MontonenOliveDuality {
    /// Create from gauge parameters
    pub fn from_parameters(params: &GaugeParameters) -> Self {
        let tau_original = params.tau();
        let s = SL2Z::s_transform();
        let tau_dual = s.transform_tau(tau_original);
        
        // Build BPS state mapping
        let mut bps_map = HashMap::new();
        
        // Map fundamental states
        for n in -5..=5 {
            for m in -5..=5 {
                let (n_dual, m_dual) = s.transform_charges(n, m);
                bps_map.insert((n, m), (n_dual, m_dual));
            }
        }
        
        Self {
            tau_original,
            tau_dual,
            charge_transform: s,
            bps_map,
        }
    }

    /// Verify BPS mass formula is preserved
    pub fn verify_bps_masses(&self) -> bool {
        for ((n, m), (n_dual, m_dual)) in &self.bps_map {
            let mass_orig = self.bps_mass(*n, *m, self.tau_original);
            let mass_dual = self.bps_mass(*n_dual, *m_dual, self.tau_dual);
            
            if (mass_orig - mass_dual).abs() > 1e-10 {
                return false;
            }
        }
        true
    }

    /// BPS mass formula M = |n + m*τ|
    fn bps_mass(&self, n: i32, m: i32, tau: Complex64) -> f64 {
        let charge = Complex64::new(n as f64, 0.0) + Complex64::new(m as f64, 0.0) * tau;
        charge.norm()
    }
}

/// Duality group orbit
#[derive(Debug)]
pub struct DualityOrbit {
    /// Starting point
    pub tau_0: Complex64,
    /// Orbit under duality group
    pub orbit: Vec<(SL2Z, Complex64)>,
    /// Fundamental domain representative
    pub fundamental_rep: Complex64,
}

impl DualityOrbit {
    /// Compute duality orbit of τ
    pub fn compute(tau_0: Complex64, max_iterations: usize) -> Self {
        let mut orbit = vec![(SL2Z::identity(), tau_0)];
        let mut tau = tau_0;
        
        // Apply modular transformations to find fundamental domain
        for _ in 0..max_iterations {
            if tau.re.abs() > 0.5 {
                // Apply T or T^(-1) to reduce |Re(τ)|
                let t = if tau.re > 0.0 {
                    SL2Z { a: 1, b: -1, c: 0, d: 1 }
                } else {
                    SL2Z::t_transform()
                };
                tau = t.transform_tau(tau);
                orbit.push((t, tau));
            } else if tau.norm() < 1.0 - 1e-10 {
                // Apply S to increase |τ|
                let s = SL2Z::s_transform();
                tau = s.transform_tau(tau);
                orbit.push((s, tau));
            } else {
                // In fundamental domain
                break;
            }
        }
        
        Self {
            tau_0,
            orbit,
            fundamental_rep: tau,
        }
    }

    /// Check if two couplings are duality-related
    pub fn are_dual(&self, tau_other: Complex64, tolerance: f64) -> bool {
        for (transform, tau_orbit) in &self.orbit {
            if (tau_orbit - tau_other).norm() < tolerance {
                return true;
            }
        }
        false
    }
}

/// S-duality validator for physical quantities
pub struct SDualityValidator {
    /// Tolerance for comparisons
    pub tolerance: f64,
    /// Transformations to check
    pub transforms: Vec<DualityTransform>,
}

impl SDualityValidator {
    /// Create standard validator
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            transforms: vec![
                DualityTransform::s_duality(),
            ],
        }
    }

    /// Validate partition function transforms correctly
    pub fn validate_partition_function(
        &self,
        z_original: impl Fn(Complex64) -> Complex64,
        z_dual: impl Fn(Complex64) -> Complex64,
        tau: Complex64,
    ) -> Result<bool> {
        for transform in &self.transforms {
            let tau_dual = transform.sl2z.transform_tau(tau);
            
            // Partition functions should be equal (up to phase)
            let z1 = z_original(tau);
            let z2 = z_dual(tau_dual);
            
            // Check modular weight
            let ratio = z2 / z1;
            if (ratio.norm() - 1.0).abs() > self.tolerance {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Validate operator expectation values
    pub fn validate_expectation_value(
        &self,
        op_original: &DMatrix<Complex64>,
        op_dual: &DMatrix<Complex64>,
        params: &GaugeParameters,
    ) -> Result<bool> {
        let transformed = transform_operator(op_original, params)?;
        
        // Check operators are related by similarity transformation
        let diff = op_dual - transformed;
        Ok(diff.norm() < self.tolerance)
    }

    /// Validate Wilson-'t Hooft duality
    pub fn validate_line_operators(
        &self,
        wilson_vev: Complex64,
        t_hooft_vev: Complex64,
        params: &GaugeParameters,
    ) -> Result<bool> {
        // Under S-duality: W ↔ T
        let dual_params = params.s_dual();
        
        // Phase factors may appear
        let ratio = t_hooft_vev / wilson_vev;
        let expected_phase = (-PI * params.tau().im / 2.0).exp();
        
        Ok((ratio / expected_phase - 1.0).norm() < self.tolerance)
    }
}

/// Extended duality including T-duality
#[derive(Debug)]
pub struct ExtendedDuality {
    /// S-duality component
    pub s_duality: DualityTransform,
    /// T-duality data (for string theory embedding)
    pub t_duality_radius: Option<f64>,
    /// Full duality group element
    pub group_element: String,
}

impl ExtendedDuality {
    /// Create S-T-S duality
    pub fn sts_duality() -> Self {
        let s = SL2Z::s_transform();
        let t = SL2Z::t_transform();
        let sts = s.compose(&t).compose(&s);
        
        Self {
            s_duality: DualityTransform::s_duality(),
            t_duality_radius: Some(1.0),
            group_element: "STS".to_string(),
        }
    }

    /// Apply extended duality to parameters
    pub fn transform(&self, params: &GaugeParameters) -> Result<GaugeParameters> {
        let mut result = self.s_duality.transform_parameters(params);
        
        // Apply T-duality if present
        if let Some(radius) = self.t_duality_radius {
            // T-duality: R → 1/R in string theory
            // This affects the coupling through string/M-theory
            result.g *= radius;
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sl2z() {
        let s = SL2Z::s_transform();
        let t = SL2Z::t_transform();
        
        assert!(s.is_valid());
        assert!(t.is_valid());
        
        // S² = -1 in PSL(2,Z)
        let s2 = s.compose(&s);
        assert_eq!(s2, SL2Z { a: -1, b: 0, c: 0, d: -1 });
        
        // (ST)³ = -1
        let st = s.compose(&t);
        let st3 = st.compose(&st).compose(&st);
        assert_eq!(st3, SL2Z { a: -1, b: 0, c: 0, d: -1 });
    }

    #[test]
    fn test_charge_transformation() {
        let s = SL2Z::s_transform();
        
        // S exchanges electric and magnetic
        let (e_new, m_new) = s.transform_charges(1, 0);
        assert_eq!((e_new, m_new), (0, 1));
        
        let (e_new, m_new) = s.transform_charges(0, 1);
        assert_eq!((e_new, m_new), (-1, 0));
    }

    #[test]
    fn test_montonen_olive() {
        let params = GaugeParameters::n4_sym(2);
        let mo = MontonenOliveDuality::from_parameters(&params);
        
        // Check BPS masses are preserved
        assert!(mo.verify_bps_masses());
        
        // Check τ_dual = -1/τ
        let expected = -1.0 / mo.tau_original;
        assert!((mo.tau_dual - expected).norm() < 1e-10);
    }

    #[test]
    fn test_duality_orbit() {
        let tau = Complex64::new(0.1, 2.5);
        let orbit = DualityOrbit::compute(tau, 100);
        
        // Should end in fundamental domain
        assert!(orbit.fundamental_rep.re.abs() <= 0.5 + 1e-10);
        assert!(orbit.fundamental_rep.norm() >= 1.0 - 1e-10);
    }
}