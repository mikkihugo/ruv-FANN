//! Physics connections for the Geometric Langlands Conjecture
//! 
//! Simplified physics module for WASM compatibility

use serde::{Serialize, Deserialize};

/// Gauge theory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeTheoryConfig {
    pub group: String,
    pub coupling: f64,
    pub theta_angle: f64,
}

impl Default for GaugeTheoryConfig {
    fn default() -> Self {
        Self {
            group: "SU(2)".to_string(),
            coupling: 1.0,
            theta_angle: 0.0,
        }
    }
}

/// Physics parameters for gauge theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeParameters {
    /// Gauge coupling constant
    pub g: f64,
    /// Theta angle
    pub theta: f64,
    /// Rank of gauge group
    pub rank: usize,
    /// Supersymmetry parameter (N=4)
    pub n_susy: u8,
}

impl GaugeParameters {
    /// Create parameters for N=4 SYM theory
    pub fn n4_sym(rank: usize) -> Self {
        Self {
            g: 1.0,
            theta: 0.0,
            rank,
            n_susy: 4,
        }
    }

    /// Compute complexified coupling τ = θ/2π + 4πi/g²
    pub fn tau_real(&self) -> f64 {
        self.theta / (2.0 * std::f64::consts::PI)
    }
    
    pub fn tau_imag(&self) -> f64 {
        4.0 * std::f64::consts::PI / (self.g * self.g)
    }
}