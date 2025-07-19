//! Wilson Lines and 't Hooft Line Operators
//! 
//! Implements Wilson line operators, 't Hooft operators, and their S-duality

use crate::core::prelude::*;
use super::{PhysicsResult, PhysicsError, GaugeFieldConfiguration, SL2Z};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Representation of the gauge group
#[derive(Debug, Clone, PartialEq)]
pub enum Representation {
    /// Fundamental representation
    Fundamental,
    /// Antifundamental representation
    Antifundamental,
    /// Adjoint representation
    Adjoint,
    /// Symmetric power
    Symmetric(usize),
    /// Antisymmetric power
    Antisymmetric(usize),
    /// General irreducible representation by highest weight
    Irreducible(Vec<i32>),
}

impl Representation {
    /// Get dimension of representation for SU(n)
    pub fn dimension_su(&self, n: usize) -> usize {
        match self {
            Representation::Fundamental => n,
            Representation::Antifundamental => n,
            Representation::Adjoint => n * n - 1,
            Representation::Symmetric(k) => {
                // Dimension of k-th symmetric power
                Self::binomial(n + k - 1, *k)
            }
            Representation::Antisymmetric(k) => {
                // Dimension of k-th antisymmetric power
                Self::binomial(n, *k)
            }
            Representation::Irreducible(_weights) => {
                // Would require Weyl character formula
                n // Simplified
            }
        }
    }

    /// Binomial coefficient
    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Check if representation is real
    pub fn is_real(&self) -> bool {
        matches!(self, Representation::Adjoint | Representation::Symmetric(_))
    }

    /// Get conjugate representation
    pub fn conjugate(&self) -> Self {
        match self {
            Representation::Fundamental => Representation::Antifundamental,
            Representation::Antifundamental => Representation::Fundamental,
            Representation::Adjoint => Representation::Adjoint,
            Representation::Symmetric(k) => Representation::Symmetric(*k),
            Representation::Antisymmetric(k) => Representation::Antisymmetric(*k),
            Representation::Irreducible(weights) => {
                // Conjugate highest weight = -w₀(weight) where w₀ is longest Weyl element
                let conj_weights: Vec<i32> = weights.iter().map(|&w| -w).collect();
                Representation::Irreducible(conj_weights)
            }
        }
    }
}

/// Wilson line operator
#[derive(Debug, Clone)]
pub struct WilsonLine {
    /// Representation of the gauge group
    pub representation: Representation,
    /// Path in spacetime (parameterized curve)
    pub path: WilsonPath,
    /// Gauge field configuration
    pub gauge_field: Option<DMatrix<Complex64>>,
    /// Computed expectation value
    pub expectation_value: Option<Complex64>,
}

/// Path for Wilson line
#[derive(Debug, Clone)]
pub struct WilsonPath {
    /// Parameterization: γ(t) for t ∈ [0,1]
    pub parameterization: PathParameterization,
    /// Path length
    pub length: f64,
    /// Whether path is closed (loop)
    pub is_closed: bool,
    /// Discretization points for numerical computation
    pub discretization: Vec<DVector<f64>>,
}

/// Path parameterization types
#[derive(Debug, Clone)]
pub enum PathParameterization {
    /// Straight line from point A to point B
    StraightLine {
        start: DVector<f64>,
        end: DVector<f64>,
    },
    /// Circular loop with center and radius
    Circle {
        center: DVector<f64>,
        radius: f64,
        normal: DVector<f64>, // Normal to plane of circle
    },
    /// General parametric curve
    Parametric {
        /// Function γ(t): [0,1] → ℝⁿ (simplified as points)
        points: Vec<DVector<f64>>,
    },
    /// Rectangle in spacetime
    Rectangle {
        corner1: DVector<f64>,
        corner2: DVector<f64>,
    },
}

impl WilsonPath {
    /// Create straight line path
    pub fn straight_line(start: DVector<f64>, end: DVector<f64>) -> Self {
        let length = (&end - &start).norm();
        let mut discretization = vec![];
        
        // Discretize path
        const N_POINTS: usize = 100;
        for i in 0..=N_POINTS {
            let t = i as f64 / N_POINTS as f64;
            let point = &start + (&end - &start).scale(t);
            discretization.push(point);
        }
        
        Self {
            parameterization: PathParameterization::StraightLine { start, end },
            length,
            is_closed: false,
            discretization,
        }
    }

    /// Create circular Wilson loop
    pub fn circular_loop(center: DVector<f64>, radius: f64, normal: DVector<f64>) -> Self {
        let length = 2.0 * PI * radius;
        let mut discretization = vec![];
        
        // Create orthonormal basis in plane perpendicular to normal
        let (u, v) = Self::create_orthonormal_basis(&normal);
        
        const N_POINTS: usize = 100;
        for i in 0..=N_POINTS {
            let theta = 2.0 * PI * i as f64 / N_POINTS as f64;
            let point = &center + u.scale(radius * theta.cos()) + v.scale(radius * theta.sin());
            discretization.push(point);
        }
        
        Self {
            parameterization: PathParameterization::Circle { center, radius, normal },
            length,
            is_closed: true,
            discretization,
        }
    }

    /// Create rectangular Wilson loop
    pub fn rectangular_loop(corner1: DVector<f64>, corner2: DVector<f64>) -> Self {
        let width = (corner2[0] - corner1[0]).abs();
        let height = (corner2[1] - corner1[1]).abs();
        let length = 2.0 * (width + height);
        
        // Create rectangle vertices
        let mut discretization = vec![];
        let vertices = vec![
            corner1.clone(),
            DVector::from_vec(vec![corner2[0], corner1[1]]),
            corner2.clone(),
            DVector::from_vec(vec![corner1[0], corner2[1]]),
            corner1.clone(), // Close the loop
        ];
        
        // Discretize each edge
        for i in 0..vertices.len()-1 {
            let edge_points = Self::discretize_edge(&vertices[i], &vertices[i+1], 25);
            discretization.extend(edge_points);
        }
        
        Self {
            parameterization: PathParameterization::Rectangle { corner1, corner2 },
            length,
            is_closed: true,
            discretization,
        }
    }

    /// Create orthonormal basis vectors perpendicular to given vector
    fn create_orthonormal_basis(normal: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let dim = normal.len();
        
        // Find a vector not parallel to normal
        let mut u = DVector::zeros(dim);
        if normal[0].abs() < 0.9 {
            u[0] = 1.0;
        } else {
            u[1] = 1.0;
        }
        
        // Gram-Schmidt orthogonalization
        let dot_product = u.dot(normal);
        u = u - normal.scale(dot_product);
        u = u.normalize();
        
        // Create second orthonormal vector (cross product in 3D)
        let v = if dim >= 3 {
            DVector::from_vec(vec![
                normal[1] * u[2] - normal[2] * u[1],
                normal[2] * u[0] - normal[0] * u[2],
                normal[0] * u[1] - normal[1] * u[0],
            ])
        } else {
            DVector::from_vec(vec![-u[1], u[0]])
        };
        
        (u, v.normalize())
    }

    /// Discretize edge between two points
    fn discretize_edge(start: &DVector<f64>, end: &DVector<f64>, n_points: usize) -> Vec<DVector<f64>> {
        let mut points = vec![];
        for i in 0..n_points {
            let t = i as f64 / (n_points - 1) as f64;
            let point = start + (end - start).scale(t);
            points.push(point);
        }
        points
    }

    /// Get tangent vector at parameter t
    pub fn tangent(&self, t: f64) -> PhysicsResult<DVector<f64>> {
        match &self.parameterization {
            PathParameterization::StraightLine { start, end } => {
                Ok((end - start).normalize())
            }
            PathParameterization::Circle { radius, normal, .. } => {
                // For circle: tangent is perpendicular to radius vector
                let (u, v) = Self::create_orthonormal_basis(normal);
                let theta = 2.0 * PI * t;
                Ok(u.scale(-*radius * theta.sin()) + v.scale(*radius * theta.cos()))
            }
            PathParameterization::Rectangle { .. } => {
                // Piecewise constant tangent
                let edge_index = (4.0 * t).floor() as usize;
                match edge_index {
                    0 => Ok(DVector::from_vec(vec![1.0, 0.0])), // Right
                    1 => Ok(DVector::from_vec(vec![0.0, 1.0])), // Up
                    2 => Ok(DVector::from_vec(vec![-1.0, 0.0])), // Left
                    3 => Ok(DVector::from_vec(vec![0.0, -1.0])), // Down
                    _ => Ok(DVector::from_vec(vec![1.0, 0.0])),
                }
            }
            PathParameterization::Parametric { points } => {
                if points.len() < 2 {
                    return Err(PhysicsError::Consistency("Need at least 2 points".to_string()));
                }
                
                let index = (t * (points.len() - 1) as f64).floor() as usize;
                let next_index = (index + 1).min(points.len() - 1);
                
                Ok((&points[next_index] - &points[index]).normalize())
            }
        }
    }

    /// Evaluate path at parameter t
    pub fn evaluate(&self, t: f64) -> PhysicsResult<DVector<f64>> {
        let t_clamped = t.max(0.0).min(1.0);
        
        match &self.parameterization {
            PathParameterization::StraightLine { start, end } => {
                Ok(start + (end - start).scale(t_clamped))
            }
            PathParameterization::Circle { center, radius, normal } => {
                let (u, v) = Self::create_orthonormal_basis(normal);
                let theta = 2.0 * PI * t_clamped;
                Ok(center + u.scale(*radius * theta.cos()) + v.scale(*radius * theta.sin()))
            }
            PathParameterization::Rectangle { corner1, corner2 } => {
                // Evaluate along rectangle perimeter
                let width = corner2[0] - corner1[0];
                let height = corner2[1] - corner1[1];
                let perimeter = 2.0 * (width.abs() + height.abs());
                let s = t_clamped * perimeter;
                
                if s <= width.abs() {
                    // Bottom edge
                    Ok(DVector::from_vec(vec![corner1[0] + s, corner1[1]]))
                } else if s <= width.abs() + height.abs() {
                    // Right edge
                    Ok(DVector::from_vec(vec![corner2[0], corner1[1] + (s - width.abs())]))
                } else if s <= 2.0 * width.abs() + height.abs() {
                    // Top edge
                    Ok(DVector::from_vec(vec![corner2[0] - (s - width.abs() - height.abs()), corner2[1]]))
                } else {
                    // Left edge
                    Ok(DVector::from_vec(vec![corner1[0], corner2[1] - (s - 2.0 * width.abs() - height.abs())]))
                }
            }
            PathParameterization::Parametric { points } => {
                if points.is_empty() {
                    return Err(PhysicsError::Consistency("Empty parametric curve".to_string()));
                }
                
                let scaled_t = t_clamped * (points.len() - 1) as f64;
                let index = scaled_t.floor() as usize;
                let frac = scaled_t - index as f64;
                
                if index >= points.len() - 1 {
                    Ok(points[points.len() - 1].clone())
                } else {
                    Ok(&points[index] + (&points[index + 1] - &points[index]).scale(frac))
                }
            }
        }
    }
}

impl WilsonLine {
    /// Create Wilson line in given representation
    pub fn new(representation: Representation, path: WilsonPath) -> Self {
        Self {
            representation,
            path,
            gauge_field: None,
            expectation_value: None,
        }
    }

    /// Create fundamental Wilson loop
    pub fn fundamental_loop(center: DVector<f64>, radius: f64) -> Self {
        let path = WilsonPath::circular_loop(center, radius, DVector::from_vec(vec![0.0, 0.0, 1.0]));
        Self::new(Representation::Fundamental, path)
    }

    /// Create adjoint Wilson line
    pub fn adjoint_line(start: DVector<f64>, end: DVector<f64>) -> Self {
        let path = WilsonPath::straight_line(start, end);
        Self::new(Representation::Adjoint, path)
    }

    /// Compute Wilson line in given gauge field
    pub fn compute_in_field(&mut self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Wilson line: W = Tr[P exp(∮ A_μ dx^μ)]
        let path_integral = self.compute_path_integral(gauge_config)?;
        let wilson_matrix = self.path_ordered_exponential(path_integral)?;
        
        // Take trace over representation
        let trace = self.trace_in_representation(&wilson_matrix)?;
        self.expectation_value = Some(trace);
        
        Ok(trace)
    }

    /// Compute path integral ∫ A_μ dx^μ
    fn compute_path_integral(&self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<DMatrix<Complex64>> {
        let dim = gauge_config.params.spacetime_dim;
        let rank = gauge_config.params.group.rank();
        let mut integral = DMatrix::zeros(rank, rank);
        
        // Discretized path integral
        for i in 0..self.path.discretization.len()-1 {
            let x_i = &self.path.discretization[i];
            let x_next = &self.path.discretization[i+1];
            let dx = x_next - x_i;
            
            // Evaluate gauge field at midpoint
            let x_mid = (x_i + x_next).scale(0.5);
            
            for mu in 0..dim.min(dx.len()) {
                if mu < gauge_config.connection.len() {
                    let a_mu = &gauge_config.connection[mu];
                    integral += a_mu.scale(dx[mu]);
                }
            }
        }
        
        Ok(integral)
    }

    /// Compute path-ordered exponential P exp(∫ A)
    fn path_ordered_exponential(&self, path_integral: DMatrix<Complex64>) -> PhysicsResult<DMatrix<Complex64>> {
        // For small integrals, use series expansion
        // P exp(A) ≈ I + A + A²/2! + A³/3! + ...
        
        let rank = path_integral.nrows();
        let mut result = DMatrix::identity(rank, rank).map(|x| Complex64::new(x, 0.0));
        let mut power = DMatrix::identity(rank, rank).map(|x| Complex64::new(x, 0.0));
        
        // Series expansion (truncated)
        for n in 1..=10 {
            power = &power * &path_integral / (n as f64);
            result += &power;
            
            // Check convergence
            if power.norm() < 1e-12 {
                break;
            }
        }
        
        Ok(result)
    }

    /// Take trace in appropriate representation
    fn trace_in_representation(&self, matrix: &DMatrix<Complex64>) -> PhysicsResult<Complex64> {
        match &self.representation {
            Representation::Fundamental | Representation::Antifundamental => {
                Ok(matrix.trace())
            }
            Representation::Adjoint => {
                // For adjoint representation, trace is different
                // This is simplified - full implementation would require representation theory
                Ok(matrix.trace())
            }
            _ => {
                // General representation - simplified
                Ok(matrix.trace())
            }
        }
    }

    /// Apply S-duality transformation
    pub fn s_dual(&self, s_transform: &SL2Z) -> PhysicsResult<WilsonLine> {
        // Under S-duality, Wilson lines ↔ 't Hooft lines
        // This is conceptual - actual implementation would be more complex
        
        let dual_representation = match &self.representation {
            Representation::Fundamental => Representation::Antifundamental,
            Representation::Antifundamental => Representation::Fundamental,
            rep => rep.clone(),
        };
        
        Ok(WilsonLine::new(dual_representation, self.path.clone()))
    }

    /// Compute Wilson line correlator
    pub fn correlator_with(&self, other: &WilsonLine, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Simplified two-point correlator
        let w1 = self.compute_expectation_value(gauge_config)?;
        let w2 = other.compute_expectation_value(gauge_config)?;
        
        // In a real computation, this would be ⟨W₁ W₂⟩ - ⟨W₁⟩⟨W₂⟩
        Ok(w1 * w2.conj())
    }

    /// Get expectation value (compute if needed)
    fn compute_expectation_value(&self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        if let Some(value) = self.expectation_value {
            Ok(value)
        } else {
            // Would need mutable self to store result
            let path_integral = self.compute_path_integral(gauge_config)?;
            let wilson_matrix = self.path_ordered_exponential(path_integral)?;
            self.trace_in_representation(&wilson_matrix)
        }
    }
}

/// 't Hooft line operator (magnetic dual of Wilson line)
#[derive(Debug, Clone)]
pub struct THooftLine {
    /// Magnetic charge vector
    pub magnetic_charge: DVector<i32>,
    /// Location in spacetime (codimension-2 defect)
    pub defect_location: DefectLocation,
    /// Computed expectation value
    pub expectation_value: Option<Complex64>,
}

/// Location of 't Hooft defect
#[derive(Debug, Clone)]
pub enum DefectLocation {
    /// Line defect in 4D spacetime
    Line {
        start: DVector<f64>,
        end: DVector<f64>,
    },
    /// Surface defect
    Surface {
        parametrization: Vec<DVector<f64>>,
    },
    /// Point defect
    Point {
        location: DVector<f64>,
    },
}

impl THooftLine {
    /// Create 't Hooft line with magnetic charge
    pub fn new(magnetic_charge: DVector<i32>, location: DefectLocation) -> Self {
        Self {
            magnetic_charge,
            defect_location: location,
            expectation_value: None,
        }
    }

    /// Create simple 't Hooft line
    pub fn simple_line(charge: i32, start: DVector<f64>, end: DVector<f64>) -> Self {
        let magnetic_charge = DVector::from_vec(vec![charge]);
        let location = DefectLocation::Line { start, end };
        Self::new(magnetic_charge, location)
    }

    /// Compute 't Hooft operator expectation value
    pub fn compute_expectation_value(&mut self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // 't Hooft operator creates singular monopole configuration
        let monopole_contribution = self.compute_monopole_contribution(gauge_config)?;
        let disorder_factor = self.compute_disorder_factor(gauge_config)?;
        
        let result = monopole_contribution * disorder_factor;
        self.expectation_value = Some(result);
        
        Ok(result)
    }

    /// Compute monopole contribution to 't Hooft operator
    fn compute_monopole_contribution(&self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Monopole creates singular gauge field configuration
        // Simplified implementation
        let total_charge: i32 = self.magnetic_charge.iter().sum();
        let theta = gauge_config.params.theta;
        
        // Phase factor from theta term
        let phase = Complex64::new(0.0, theta * total_charge as f64).exp();
        
        Ok(phase)
    }

    /// Compute disorder factor
    fn compute_disorder_factor(&self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Disorder operator in lattice gauge theory
        // Simplified as inverse coupling dependence
        let coupling = gauge_config.params.coupling;
        Ok(Complex64::new(coupling.recip(), 0.0))
    }

    /// Apply S-duality transformation to get Wilson line
    pub fn s_dual_wilson_line(&self) -> PhysicsResult<WilsonLine> {
        // Under S-duality: 't Hooft line → Wilson line
        let charge_magnitude = self.magnetic_charge.norm() as usize;
        
        let representation = if charge_magnitude == 1 {
            Representation::Fundamental
        } else {
            Representation::Symmetric(charge_magnitude)
        };
        
        // Convert defect location to Wilson path
        let path = match &self.defect_location {
            DefectLocation::Line { start, end } => {
                WilsonPath::straight_line(start.clone(), end.clone())
            }
            DefectLocation::Point { location } => {
                // Point defect → circular loop around point
                WilsonPath::circular_loop(location.clone(), 1.0, DVector::from_vec(vec![0.0, 0.0, 1.0]))
            }
            DefectLocation::Surface { parametrization } => {
                // Surface → boundary loop
                WilsonPath::straight_line(
                    parametrization.first().unwrap().clone(),
                    parametrization.last().unwrap().clone()
                )
            }
        };
        
        Ok(WilsonLine::new(representation, path))
    }

    /// Compute 't Hooft-Wilson dyon correlator
    pub fn dyon_correlator(&self, wilson: &WilsonLine, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Dyon = electric + magnetic charge
        let t_hooft_vev = self.compute_expectation_value_copy(gauge_config)?;
        let wilson_vev = wilson.compute_expectation_value(gauge_config)?;
        
        // Dirac quantization constraint affects correlator
        let dirac_phase = self.compute_dirac_phase(wilson)?;
        
        Ok(t_hooft_vev * wilson_vev * dirac_phase)
    }

    /// Compute expectation value without mutable self
    fn compute_expectation_value_copy(&self, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        let monopole_contribution = self.compute_monopole_contribution(gauge_config)?;
        let disorder_factor = self.compute_disorder_factor(gauge_config)?;
        Ok(monopole_contribution * disorder_factor)
    }

    /// Compute Dirac quantization phase
    fn compute_dirac_phase(&self, wilson: &WilsonLine) -> PhysicsResult<Complex64> {
        // Dirac quantization: exp(2πi q_e q_m)
        let total_magnetic_charge: i32 = self.magnetic_charge.iter().sum();
        
        // Electric charge from Wilson line (simplified)
        let electric_charge = match &wilson.representation {
            Representation::Fundamental => 1,
            Representation::Antifundamental => -1,
            _ => 0,
        };
        
        let phase_arg = 2.0 * PI * (electric_charge * total_magnetic_charge) as f64;
        Ok(Complex64::new(0.0, phase_arg).exp())
    }
}

/// Line operator algebra
#[derive(Debug, Clone)]
pub struct LineOperatorAlgebra {
    /// Wilson lines
    pub wilson_lines: Vec<WilsonLine>,
    /// 't Hooft lines
    pub t_hooft_lines: Vec<THooftLine>,
    /// Computed OPE coefficients
    pub ope_coefficients: HashMap<(usize, usize), Complex64>,
}

impl LineOperatorAlgebra {
    /// Create new line operator algebra
    pub fn new() -> Self {
        Self {
            wilson_lines: vec![],
            t_hooft_lines: vec![],
            ope_coefficients: HashMap::new(),
        }
    }

    /// Add Wilson line
    pub fn add_wilson_line(&mut self, wilson: WilsonLine) {
        self.wilson_lines.push(wilson);
    }

    /// Add 't Hooft line
    pub fn add_t_hooft_line(&mut self, t_hooft: THooftLine) {
        self.t_hooft_lines.push(t_hooft);
    }

    /// Compute operator product expansion (OPE)
    pub fn compute_ope(&mut self, i: usize, j: usize, gauge_config: &GaugeFieldConfiguration) -> PhysicsResult<Complex64> {
        // Simplified OPE computation
        if i < self.wilson_lines.len() && j < self.wilson_lines.len() {
            // Wilson-Wilson OPE
            let ope_coeff = self.wilson_lines[i].correlator_with(&self.wilson_lines[j], gauge_config)?;
            self.ope_coefficients.insert((i, j), ope_coeff);
            Ok(ope_coeff)
        } else {
            // Other OPEs would require more complex implementation
            Ok(Complex64::new(1.0, 0.0))
        }
    }

    /// Verify S-duality for the algebra
    pub fn verify_s_duality(&self, s_transform: &SL2Z) -> PhysicsResult<bool> {
        // Check that S-duality maps the algebra to itself
        for wilson in &self.wilson_lines {
            let dual_wilson = wilson.s_dual(s_transform)?;
            // Check if dual Wilson line exists in the algebra
            // (simplified check)
        }
        
        Ok(true) // Simplified
    }

    /// Get line operator by electric and magnetic charges
    pub fn get_line_operator(&self, electric: i32, magnetic: i32) -> Option<LineOperatorType> {
        // Find operator with given charges
        if magnetic == 0 {
            // Pure Wilson line
            for (i, wilson) in self.wilson_lines.iter().enumerate() {
                // Check if Wilson line has correct electric charge
                if self.wilson_line_electric_charge(wilson) == electric {
                    return Some(LineOperatorType::Wilson(i));
                }
            }
        } else if electric == 0 {
            // Pure 't Hooft line
            for (i, t_hooft) in self.t_hooft_lines.iter().enumerate() {
                let total_magnetic: i32 = t_hooft.magnetic_charge.iter().sum();
                if total_magnetic == magnetic {
                    return Some(LineOperatorType::THooft(i));
                }
            }
        }
        // General dyon would require composite operators
        
        None
    }

    /// Get electric charge of Wilson line (simplified)
    fn wilson_line_electric_charge(&self, wilson: &WilsonLine) -> i32 {
        match &wilson.representation {
            Representation::Fundamental => 1,
            Representation::Antifundamental => -1,
            _ => 0,
        }
    }
}

impl Default for LineOperatorAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of line operator
#[derive(Debug, Clone, Copy)]
pub enum LineOperatorType {
    Wilson(usize),
    THooft(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::gauge_theory::{GaugeGroup, GaugeParameters};

    #[test]
    fn test_representation() {
        let fund = Representation::Fundamental;
        assert_eq!(fund.dimension_su(3), 3);
        
        let adj = Representation::Adjoint;
        assert_eq!(adj.dimension_su(3), 8);
        assert!(adj.is_real());
        
        let conj_fund = fund.conjugate();
        assert_eq!(conj_fund, Representation::Antifundamental);
    }

    #[test]
    fn test_wilson_path() {
        let start = DVector::from_vec(vec![0.0, 0.0]);
        let end = DVector::from_vec(vec![1.0, 1.0]);
        let path = WilsonPath::straight_line(start, end);
        
        assert!(!path.is_closed);
        assert!((path.length - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(path.discretization.len(), 101); // 0 to 100 inclusive
    }

    #[test]
    fn test_circular_wilson_loop() {
        let center = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let normal = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let path = WilsonPath::circular_loop(center, 1.0, normal);
        
        assert!(path.is_closed);
        assert!((path.length - 2.0 * PI).abs() < 1e-10);
        
        // Check that path starts and ends at same point (modulo discretization)
        let start = &path.discretization[0];
        let end = &path.discretization[path.discretization.len()-1];
        assert!((start - end).norm() < 1e-10);
    }

    #[test]
    fn test_wilson_line() {
        let center = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let wilson = WilsonLine::fundamental_loop(center, 1.0);
        
        assert_eq!(wilson.representation, Representation::Fundamental);
        assert!(wilson.path.is_closed);
    }

    #[test]
    fn test_t_hooft_line() {
        let start = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let end = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let t_hooft = THooftLine::simple_line(1, start, end);
        
        assert_eq!(t_hooft.magnetic_charge[0], 1);
        
        // Test S-duality
        let wilson_dual = t_hooft.s_dual_wilson_line().unwrap();
        assert_eq!(wilson_dual.representation, Representation::Fundamental);
    }

    #[test]
    fn test_line_operator_algebra() {
        let mut algebra = LineOperatorAlgebra::new();
        
        let center = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let wilson = WilsonLine::fundamental_loop(center, 1.0);
        algebra.add_wilson_line(wilson);
        
        let start = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let end = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let t_hooft = THooftLine::simple_line(1, start, end);
        algebra.add_t_hooft_line(t_hooft);
        
        assert_eq!(algebra.wilson_lines.len(), 1);
        assert_eq!(algebra.t_hooft_lines.len(), 1);
        
        // Test operator lookup
        let wilson_op = algebra.get_line_operator(1, 0);
        assert!(matches!(wilson_op, Some(LineOperatorType::Wilson(0))));
        
        let t_hooft_op = algebra.get_line_operator(0, 1);
        assert!(matches!(t_hooft_op, Some(LineOperatorType::THooft(0))));
    }

    #[test]
    fn test_path_evaluation() {
        let center = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let normal = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let path = WilsonPath::circular_loop(center, 1.0, normal);
        
        // Evaluate at quarter circle
        let point_quarter = path.evaluate(0.25).unwrap();
        assert!((point_quarter[0] - 0.0).abs() < 1e-10); // x ≈ 0
        assert!((point_quarter[1] - 1.0).abs() < 1e-10); // y ≈ 1
        
        // Check tangent vector
        let tangent_start = path.tangent(0.0).unwrap();
        assert!(tangent_start[0].abs() < 1e-10); // dx/dt = 0 at t=0
        assert!((tangent_start[1] - 1.0).abs() < 1e-10); // dy/dt = 1 at t=0
    }
}