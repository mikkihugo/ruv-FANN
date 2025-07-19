//! Loss Functions for Langlands Correspondence
//! 
//! This module implements specialized loss functions that capture
//! the mathematical properties of the geometric Langlands correspondence.

use ndarray::{Array1, ArrayView1};

/// Trait for Langlands-specific loss functions
pub trait LanglandsLoss: Send + Sync {
    /// Compute the loss between predicted and target features
    fn compute_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> f64;
    
    /// Compute the gradient of the loss with respect to predictions
    fn compute_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> Array1<f64>;
    
    /// Get a descriptive name for the loss function
    fn name(&self) -> &str;
}

/// Standard correspondence loss combining multiple components
#[derive(Debug, Clone)]
pub struct CorrespondenceLoss {
    /// Weight for MSE component
    pub mse_weight: f64,
    /// Weight for invariant matching component
    pub invariant_weight: f64,
    /// Weight for functorial consistency component
    pub functorial_weight: f64,
    /// Weight for spectral matching component
    pub spectral_weight: f64,
    /// Regularization parameter
    pub regularization: f64,
}

impl Default for CorrespondenceLoss {
    fn default() -> Self {
        Self {
            mse_weight: 1.0,
            invariant_weight: 2.0,
            functorial_weight: 1.5,
            spectral_weight: 1.5,
            regularization: 0.01,
        }
    }
}

impl LanglandsLoss for CorrespondenceLoss {
    fn compute_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> f64 {
        let mut total_loss = 0.0;
        
        // 1. Mean Squared Error component
        let mse = (predicted - target).mapv(|x| x * x).mean().unwrap_or(0.0);
        total_loss += self.mse_weight * mse;
        
        // 2. Invariant matching loss
        let invariant_loss = self.compute_invariant_loss(predicted, target);
        total_loss += self.invariant_weight * invariant_loss;
        
        // 3. Functorial consistency loss
        let functorial_loss = self.compute_functorial_loss(predicted, target, input);
        total_loss += self.functorial_weight * functorial_loss;
        
        // 4. Spectral matching loss
        let spectral_loss = self.compute_spectral_loss(predicted, target);
        total_loss += self.spectral_weight * spectral_loss;
        
        // 5. Regularization
        let reg_loss = predicted.mapv(|x| x * x).sum() + input.mapv(|x| x * x).sum();
        total_loss += self.regularization * reg_loss;
        
        total_loss
    }
    
    fn compute_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        
        // MSE gradient
        gradient += &(self.mse_weight * 2.0 * (predicted - target));
        
        // Invariant gradient
        gradient += &(self.invariant_weight * self.compute_invariant_gradient(predicted, target));
        
        // Functorial gradient
        gradient += &(self.functorial_weight * self.compute_functorial_gradient(predicted, target, input));
        
        // Spectral gradient
        gradient += &(self.spectral_weight * self.compute_spectral_gradient(predicted, target));
        
        // Regularization gradient
        gradient += &(self.regularization * 2.0 * predicted);
        
        gradient
    }
    
    fn name(&self) -> &str {
        "CorrespondenceLoss"
    }
}

impl CorrespondenceLoss {
    /// Compute loss based on mathematical invariants
    fn compute_invariant_loss(&self, predicted: &Array1<f64>, target: &Array1<f64>) -> f64 {
        // Extract invariant features (first few components typically)
        let n_invariants = predicted.len().min(10);
        
        let pred_invariants = predicted.slice(s![..n_invariants]);
        let target_invariants = target.slice(s![..n_invariants]);
        
        // Weighted difference (invariants are more important)
        let weights = Array1::linspace(2.0, 1.0, n_invariants);
        let diff = &pred_invariants - &target_invariants;
        
        (diff * diff * weights).sum()
    }
    
    /// Compute loss based on functorial properties
    fn compute_functorial_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> f64 {
        // Check if the mapping preserves certain algebraic relations
        
        // Example: Trace preservation
        let input_trace = input.slice(s![..4]).sum();
        let pred_trace = predicted.slice(s![..4]).sum();
        let target_trace = target.slice(s![..4]).sum();
        
        let trace_error = ((pred_trace - target_trace) / (target_trace.abs() + 1e-6)).abs();
        
        // Example: Determinant preservation (encoded in specific features)
        let det_idx = 4;
        if predicted.len() > det_idx && target.len() > det_idx {
            let det_error = ((predicted[det_idx] - target[det_idx]) / (target[det_idx].abs() + 1e-6)).abs();
            trace_error + det_error
        } else {
            trace_error
        }
    }
    
    /// Compute loss based on spectral properties
    fn compute_spectral_loss(&self, predicted: &Array1<f64>, target: &Array1<f64>) -> f64 {
        // Focus on spectral features (eigenvalues, gaps, etc.)
        let spectral_start = 10;
        let spectral_end = 30.min(predicted.len());
        
        if spectral_start < spectral_end {
            let pred_spectral = predicted.slice(s![spectral_start..spectral_end]);
            let target_spectral = target.slice(s![spectral_start..spectral_end]);
            
            // Wasserstein-like distance for spectral distributions
            let mut pred_sorted = pred_spectral.to_vec();
            let mut target_sorted = target_spectral.to_vec();
            pred_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            target_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            pred_sorted.iter()
                .zip(target_sorted.iter())
                .map(|(p, t)| (p - t).abs())
                .sum()
        } else {
            0.0
        }
    }
    
    fn compute_invariant_gradient(&self, predicted: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        let n_invariants = predicted.len().min(10);
        
        let weights = Array1::linspace(2.0, 1.0, n_invariants);
        for i in 0..n_invariants {
            gradient[i] = 2.0 * weights[i] * (predicted[i] - target[i]);
        }
        
        gradient
    }
    
    fn compute_functorial_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        
        // Gradient for trace preservation
        let trace_weight = 1.0 / (target.slice(s![..4]).sum().abs() + 1e-6);
        for i in 0..4.min(predicted.len()) {
            gradient[i] += trace_weight;
        }
        
        // Gradient for determinant preservation
        let det_idx = 4;
        if predicted.len() > det_idx {
            gradient[det_idx] += 1.0 / (target[det_idx].abs() + 1e-6);
        }
        
        gradient
    }
    
    fn compute_spectral_gradient(&self, predicted: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        
        let spectral_start = 10;
        let spectral_end = 30.min(predicted.len());
        
        if spectral_start < spectral_end {
            for i in spectral_start..spectral_end {
                gradient[i] = (predicted[i] - target[i]).signum();
            }
        }
        
        gradient
    }
}

/// Hecke eigenvalue matching loss
#[derive(Debug, Clone)]
pub struct HeckeLoss {
    /// Number of Hecke operators to consider
    pub num_hecke_ops: usize,
    /// Weight for eigenvalue matching
    pub eigenvalue_weight: f64,
    /// Weight for eigenvector alignment
    pub eigenvector_weight: f64,
}

impl Default for HeckeLoss {
    fn default() -> Self {
        Self {
            num_hecke_ops: 5,
            eigenvalue_weight: 1.0,
            eigenvector_weight: 0.5,
        }
    }
}

impl LanglandsLoss for HeckeLoss {
    fn compute_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> f64 {
        let mut loss = 0.0;
        
        // Assume Hecke eigenvalues are stored in specific positions
        let eigenval_start = 20;
        let eigenval_end = (eigenval_start + self.num_hecke_ops).min(predicted.len());
        
        if eigenval_start < eigenval_end {
            // Eigenvalue matching loss
            for i in eigenval_start..eigenval_end {
                let diff = predicted[i] - target[i];
                loss += self.eigenvalue_weight * diff * diff;
            }
            
            // Eigenvector alignment (simplified as correlation)
            let eigenvec_start = eigenval_end;
            let eigenvec_end = (eigenvec_start + 10).min(predicted.len());
            
            if eigenvec_start < eigenvec_end {
                let pred_vec = predicted.slice(s![eigenvec_start..eigenvec_end]);
                let target_vec = target.slice(s![eigenvec_start..eigenvec_end]);
                
                let correlation = Self::compute_correlation(&pred_vec, &target_vec);
                loss += self.eigenvector_weight * (1.0 - correlation).max(0.0);
            }
        }
        
        loss
    }
    
    fn compute_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        
        let eigenval_start = 20;
        let eigenval_end = (eigenval_start + self.num_hecke_ops).min(predicted.len());
        
        if eigenval_start < eigenval_end {
            // Eigenvalue gradient
            for i in eigenval_start..eigenval_end {
                gradient[i] = 2.0 * self.eigenvalue_weight * (predicted[i] - target[i]);
            }
            
            // Eigenvector gradient
            let eigenvec_start = eigenval_end;
            let eigenvec_end = (eigenvec_start + 10).min(predicted.len());
            
            if eigenvec_start < eigenvec_end {
                let pred_vec = predicted.slice(s![eigenvec_start..eigenvec_end]);
                let target_vec = target.slice(s![eigenvec_start..eigenvec_end]);
                
                let grad_slice = Self::correlation_gradient(&pred_vec, &target_vec);
                for i in 0..grad_slice.len() {
                    gradient[eigenvec_start + i] = -self.eigenvector_weight * grad_slice[i];
                }
            }
        }
        
        gradient
    }
    
    fn name(&self) -> &str {
        "HeckeLoss"
    }
}

impl HeckeLoss {
    fn compute_correlation(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let mean_a = a.mean().unwrap_or(0.0);
        let mean_b = b.mean().unwrap_or(0.0);
        
        let cov = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - mean_a) * (y - mean_b))
            .sum::<f64>();
        
        let var_a = a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>();
        let var_b = b.iter().map(|&y| (y - mean_b).powi(2)).sum::<f64>();
        
        if var_a > 0.0 && var_b > 0.0 {
            cov / (var_a.sqrt() * var_b.sqrt())
        } else {
            0.0
        }
    }
    
    fn correlation_gradient(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let n = a.len() as f64;
        let mean_a = a.mean().unwrap_or(0.0);
        let mean_b = b.mean().unwrap_or(0.0);
        
        let mut gradient = Array1::zeros(a.len());
        
        // Simplified gradient computation
        for i in 0..a.len() {
            gradient[i] = (b[i] - mean_b) / (n * (a.var() * b.var()).sqrt() + 1e-6);
        }
        
        gradient
    }
}

/// Modular form matching loss (for arithmetic cases)
#[derive(Debug, Clone)]
pub struct ModularFormLoss {
    /// Number of Fourier coefficients to match
    pub num_coefficients: usize,
    /// Weight decay for higher coefficients
    pub decay_rate: f64,
}

impl Default for ModularFormLoss {
    fn default() -> Self {
        Self {
            num_coefficients: 20,
            decay_rate: 0.9,
        }
    }
}

impl LanglandsLoss for ModularFormLoss {
    fn compute_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> f64 {
        let mut loss = 0.0;
        let coeff_start = 30;
        let coeff_end = (coeff_start + self.num_coefficients).min(predicted.len());
        
        if coeff_start < coeff_end {
            for i in coeff_start..coeff_end {
                let idx = i - coeff_start;
                let weight = self.decay_rate.powi(idx as i32);
                let diff = predicted[i] - target[i];
                loss += weight * diff * diff;
            }
        }
        
        loss
    }
    
    fn compute_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut gradient = Array1::zeros(predicted.len());
        let coeff_start = 30;
        let coeff_end = (coeff_start + self.num_coefficients).min(predicted.len());
        
        if coeff_start < coeff_end {
            for i in coeff_start..coeff_end {
                let idx = i - coeff_start;
                let weight = self.decay_rate.powi(idx as i32);
                gradient[i] = 2.0 * weight * (predicted[i] - target[i]);
            }
        }
        
        gradient
    }
    
    fn name(&self) -> &str {
        "ModularFormLoss"
    }
}

/// Combined loss function that uses multiple criteria
#[derive(Debug)]
pub struct CombinedLoss {
    /// Component loss functions with weights
    pub components: Vec<(Box<dyn LanglandsLoss>, f64)>,
}

impl CombinedLoss {
    /// Create a new combined loss with default components
    pub fn new_default() -> Self {
        Self {
            components: vec![
                (Box::new(CorrespondenceLoss::default()), 1.0),
                (Box::new(HeckeLoss::default()), 0.5),
                (Box::new(ModularFormLoss::default()), 0.3),
            ],
        }
    }
    
    /// Add a component loss function
    pub fn add_component(mut self, loss: Box<dyn LanglandsLoss>, weight: f64) -> Self {
        self.components.push((loss, weight));
        self
    }
}

impl LanglandsLoss for CombinedLoss {
    fn compute_loss(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> f64 {
        self.components.iter()
            .map(|(loss_fn, weight)| weight * loss_fn.compute_loss(predicted, target, input))
            .sum()
    }
    
    fn compute_gradient(
        &self,
        predicted: &Array1<f64>,
        target: &Array1<f64>,
        input: &Array1<f64>,
    ) -> Array1<f64> {
        let mut total_gradient = Array1::zeros(predicted.len());
        
        for (loss_fn, weight) in &self.components {
            let grad = loss_fn.compute_gradient(predicted, target, input);
            total_gradient += &(grad * *weight);
        }
        
        total_gradient
    }
    
    fn name(&self) -> &str {
        "CombinedLoss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_correspondence_loss() {
        let loss = CorrespondenceLoss::default();
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let target = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1, 4.9]);
        let input = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
        
        let loss_val = loss.compute_loss(&predicted, &target, &input);
        assert!(loss_val > 0.0);
        
        let gradient = loss.compute_gradient(&predicted, &target, &input);
        assert_eq!(gradient.len(), predicted.len());
    }
    
    #[test]
    fn test_hecke_loss() {
        let loss = HeckeLoss::default();
        let predicted = Array1::zeros(50);
        let target = Array1::ones(50);
        let input = Array1::zeros(50);
        
        let loss_val = loss.compute_loss(&predicted, &target, &input);
        assert!(loss_val > 0.0);
    }
    
    #[test]
    fn test_combined_loss() {
        let loss = CombinedLoss::new_default();
        let predicted = Array1::linspace(0.0, 1.0, 100);
        let target = Array1::linspace(0.1, 1.1, 100);
        let input = Array1::zeros(100);
        
        let loss_val = loss.compute_loss(&predicted, &target, &input);
        assert!(loss_val > 0.0);
        
        let gradient = loss.compute_gradient(&predicted, &target, &input);
        assert_eq!(gradient.len(), 100);
    }
}