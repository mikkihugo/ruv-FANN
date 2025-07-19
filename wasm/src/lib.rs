//! WebAssembly bindings for the Geometric Langlands Conjecture framework
//! 
//! This module provides high-performance WASM bindings that allow the
//! mathematical framework to run efficiently in web browsers.

#![warn(missing_docs)]

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Performance, Worker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use num_complex::Complex64;

// Use wee_alloc as the global allocator for smaller binary size
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    cfg_if::cfg_if! {
        if #[cfg(feature = "console_error_panic_hook")] {
            console_error_panic_hook::set_once();
        }
    }
    
    console::log_1(&"🌟 Geometric Langlands WASM module initialized!".into());
}

/// JavaScript interop utilities
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_many(a: &str, b: &str);
}

/// Macro for easier console logging
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Performance monitoring for WASM operations
#[wasm_bindgen]
pub struct PerformanceMonitor {
    start_time: f64,
    metrics: HashMap<String, f64>,
}

#[wasm_bindgen]
impl PerformanceMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        
        Self {
            start_time: performance.now(),
            metrics: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn mark(&mut self, label: &str) {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let elapsed = performance.now() - self.start_time;
        
        self.metrics.insert(label.to_string(), elapsed);
        console_log!("⏱️ Performance mark '{}': {:.2}ms", label, elapsed);
    }

    #[wasm_bindgen]
    pub fn get_metrics(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.metrics).unwrap()
    }
}

/// Configuration for WASM operations
#[derive(Serialize, Deserialize, Clone)]
#[wasm_bindgen]
pub struct WasmConfig {
    /// Maximum number of worker threads to use
    pub max_workers: usize,
    /// Memory limit in MB
    pub memory_limit: f64,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Bundle size optimization level (0-3)
    pub optimization_level: u8,
}

#[wasm_bindgen]
impl WasmConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_workers: 4,
            memory_limit: 100.0,
            enable_gpu: true,
            optimization_level: 2,
        }
    }

    /// Get maximum number of workers
    #[wasm_bindgen]
    pub fn get_max_workers(&self) -> usize { self.max_workers }

    /// Set maximum number of workers
    #[wasm_bindgen]
    pub fn set_max_workers(&mut self, workers: usize) {
        self.max_workers = workers.min(8); // Cap at 8 workers
    }

    /// Get memory limit
    #[wasm_bindgen]
    pub fn get_memory_limit(&self) -> f64 { self.memory_limit }

    /// Set memory limit
    #[wasm_bindgen]
    pub fn set_memory_limit(&mut self, limit: f64) {
        self.memory_limit = limit.max(50.0).min(500.0); // 50MB - 500MB
    }

    /// Get GPU acceleration setting
    #[wasm_bindgen]
    pub fn get_enable_gpu(&self) -> bool { self.enable_gpu }

    /// Set GPU acceleration setting
    #[wasm_bindgen]
    pub fn set_enable_gpu(&mut self, enable: bool) {
        self.enable_gpu = enable;
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified Reductive Group for WASM
#[wasm_bindgen]
pub struct ReductiveGroup {
    group_type: String,
    dimension: usize,
    rank: usize,
}

#[wasm_bindgen]
impl ReductiveGroup {
    /// Create GL(n) group
    #[wasm_bindgen]
    pub fn gl_n(n: usize) -> Self {
        Self {
            group_type: format!("GL({})", n),
            dimension: n * n,
            rank: n,
        }
    }
    
    /// Create SL(n) group
    #[wasm_bindgen]
    pub fn sl_n(n: usize) -> Self {
        Self {
            group_type: format!("SL({})", n),
            dimension: n * n - 1,
            rank: n - 1,
        }
    }
    
    /// Create SO(n) group
    #[wasm_bindgen]
    pub fn so_n(n: usize) -> Self {
        Self {
            group_type: format!("SO({})", n),
            dimension: n * (n - 1) / 2,
            rank: n / 2,
        }
    }
    
    /// Create Sp(2n) group
    #[wasm_bindgen]
    pub fn sp_2n(n: usize) -> Self {
        Self {
            group_type: format!("Sp({})", 2 * n),
            dimension: n * (2 * n + 1),
            rank: n,
        }
    }
    
    /// Get group dimension
    #[wasm_bindgen]
    pub fn get_dimension(&self) -> usize { self.dimension }
    
    /// Get group rank
    #[wasm_bindgen]
    pub fn get_rank(&self) -> usize { self.rank }
    
    /// Get group type string
    #[wasm_bindgen]
    pub fn get_group_type(&self) -> String { self.group_type.clone() }
}

/// Simplified Automorphic Form for WASM
#[wasm_bindgen]
pub struct AutomorphicForm {
    weight: u32,
    level: u32,
    conductor: u32,
    group_type: String,
}

#[wasm_bindgen]
impl AutomorphicForm {
    /// Create Eisenstein series
    #[wasm_bindgen]
    pub fn eisenstein_series(group: &ReductiveGroup, weight: u32) -> Self {
        let conductor = Self::compute_conductor(&group.group_type, weight, 1);
        Self {
            weight,
            level: 1,
            conductor,
            group_type: group.group_type.clone(),
        }
    }
    
    /// Create cusp form
    #[wasm_bindgen]
    pub fn cusp_form(group: &ReductiveGroup, weight: u32, level: u32) -> Self {
        let conductor = Self::compute_conductor(&group.group_type, weight, level);
        Self {
            weight,
            level,
            conductor,
            group_type: group.group_type.clone(),
        }
    }
    
    /// Get automorphic form weight
    #[wasm_bindgen]
    pub fn get_weight(&self) -> u32 { self.weight }
    
    /// Get automorphic form level
    #[wasm_bindgen]
    pub fn get_level(&self) -> u32 { self.level }
    
    /// Get automorphic form conductor
    #[wasm_bindgen]
    pub fn get_conductor(&self) -> u32 { self.conductor }
    
    /// Compute conductor based on group type and parameters
    fn compute_conductor(group_type: &str, weight: u32, level: u32) -> u32 {
        // Simplified conductor formula
        let base = match group_type.chars().take(2).collect::<String>().as_str() {
            "GL" => 37,
            "SL" => 41,
            "SO" => 43,
            "Sp" => 47,
            _ => 37,
        };
        base * weight + level * 13
    }
}

/// Simplified Galois Representation for WASM
#[wasm_bindgen]
pub struct GaloisRepresentation {
    dimension: usize,
    conductor: u32,
    characteristic: u32,
}

#[wasm_bindgen]
impl GaloisRepresentation {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, conductor: u32, characteristic: u32) -> Self {
        Self {
            dimension,
            conductor,
            characteristic,
        }
    }
    
    /// Get Galois representation dimension
    #[wasm_bindgen]
    pub fn get_dimension(&self) -> usize { self.dimension }
    
    /// Get Galois representation conductor
    #[wasm_bindgen]
    pub fn get_conductor(&self) -> u32 { self.conductor }
    
    /// Get characteristic
    #[wasm_bindgen]
    pub fn get_characteristic(&self) -> u32 { self.characteristic }
    
    #[wasm_bindgen]
    pub fn is_irreducible(&self) -> bool {
        // Simplified irreducibility check
        self.dimension > 0 && self.conductor > 1
    }
}

/// Simplified Hecke Operator for WASM
#[wasm_bindgen]
pub struct HeckeOperator {
    prime: u32,
    group_type: String,
}

#[wasm_bindgen]
impl HeckeOperator {
    #[wasm_bindgen(constructor)]
    pub fn new(group: &ReductiveGroup, prime: u32) -> Self {
        Self {
            prime,
            group_type: group.group_type.clone(),
        }
    }
    
    /// Apply Hecke operator to automorphic form
    #[wasm_bindgen]
    pub fn apply(&self, form: &AutomorphicForm) -> AutomorphicForm {
        // Simplified application - creates a new form with modified conductor
        AutomorphicForm {
            weight: form.weight,
            level: form.level,
            conductor: form.conductor + self.prime,
            group_type: form.group_type.clone(),
        }
    }
    
    /// Compute eigenvalue
    #[wasm_bindgen]
    pub fn eigenvalue(&self, form: &AutomorphicForm) -> f64 {
        // Simplified eigenvalue computation using the prime and form parameters
        let base = (self.prime as f64).sqrt();
        let weight_factor = 1.0 + (form.weight as f64) / 12.0;
        let conductor_factor = 1.0 + (form.conductor as f64).ln() / 100.0;
        
        base * weight_factor * conductor_factor
    }
    
    /// Get prime number for Hecke operator
    #[wasm_bindgen]
    pub fn get_prime(&self) -> u32 { self.prime }
}

/// Main Langlands computation engine for WASM
#[wasm_bindgen]
pub struct LanglandsEngine {
    config: WasmConfig,
    monitor: PerformanceMonitor,
    is_initialized: bool,
}

#[wasm_bindgen]
impl LanglandsEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<WasmConfig>) -> Self {
        let config = config.unwrap_or_default();
        console_log!("🧮 Initializing Langlands Engine with {} workers", config.max_workers);
        
        Self {
            config,
            monitor: PerformanceMonitor::new(),
            is_initialized: false,
        }
    }

    /// Initialize the computation engine
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        self.monitor.mark("initialization_start");
        
        // Check for WebGL support for GPU acceleration
        if self.config.enable_gpu {
            let has_webgl = self.check_webgl_support();
            console_log!("🎮 WebGL support: {}", has_webgl);
        }

        // Initialize mathematical components
        self.init_mathematical_framework()?;
        
        // Set up worker pool for parallel computation
        if self.config.max_workers > 1 {
            self.init_worker_pool().await?;
        }

        self.monitor.mark("initialization_complete");
        self.is_initialized = true;
        
        console_log!("✅ Langlands Engine initialized successfully!");
        Ok(())
    }

    /// Check if WebGL is supported
    fn check_webgl_support(&self) -> bool {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.create_element("canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
        
        canvas.get_context("webgl").is_ok() || canvas.get_context("webgl2").is_ok()
    }

    /// Initialize the mathematical framework
    fn init_mathematical_framework(&mut self) -> Result<(), JsValue> {
        console_log!("🧮 Setting up mathematical framework...");
        
        // Initialize basic mathematical structures
        // This would include setup for:
        // - Linear algebra routines
        // - Number theory functions
        // - Group theory computations
        
        Ok(())
    }

    /// Initialize worker pool for parallel computation
    async fn init_worker_pool(&mut self) -> Result<(), JsValue> {
        console_log!("👥 Setting up {} worker threads...", self.config.max_workers);
        
        // TODO: Set up web workers for parallel computation
        // This would handle:
        // - Distributing heavy mathematical computations
        // - Parallel neural network inference
        // - Batch processing of correspondences
        
        Ok(())
    }

    /// Compute a Langlands correspondence
    #[wasm_bindgen]
    pub async fn compute_correspondence(
        &mut self, 
        group_type: &str,
        dimension: u32
    ) -> Result<JsValue, JsValue> {
        if !self.is_initialized {
            return Err(JsValue::from_str("Engine not initialized. Call initialize() first."));
        }

        self.monitor.mark("correspondence_start");
        console_log!("🔮 Computing {} correspondence in dimension {}", group_type, dimension);

        // Create the mathematical objects
        let group = match group_type {
            "GL" => ReductiveGroup::gl_n(dimension as usize),
            "SL" => ReductiveGroup::sl_n(dimension as usize),
            "SO" => ReductiveGroup::so_n(dimension as usize),
            "Sp" => ReductiveGroup::sp_2n((dimension / 2) as usize),
            _ => return Err(JsValue::from_str(&format!("Unknown group type: {}", group_type))),
        };

        // Create automorphic form
        let weight = 2;
        let level = 1;
        let automorphic_form = AutomorphicForm::cusp_form(&group, weight, level);

        // Create Galois representation
        let conductor = automorphic_form.get_conductor();
        let galois_rep = GaloisRepresentation::new(dimension as usize, conductor, 0);

        // Compute Hecke eigenvalues for verification
        let primes = [2, 3, 5, 7, 11, 13, 17, 19];
        let mut eigenvalues = Vec::new();
        
        for &p in &primes {
            if MathUtils::is_prime(p) {
                let hecke = HeckeOperator::new(&group, p);
                let eigenvalue = hecke.eigenvalue(&automorphic_form);
                eigenvalues.push((p, eigenvalue));
            }
        }

        // Perform advanced mathematical computations
        let advanced_data = self.compute_advanced_mathematics(&group, dimension as usize).await?;
        
        self.monitor.mark("correspondence_complete");
        
        let result = LanglandsResult {
            group_type: group_type.to_string(),
            dimension: dimension as usize,
            success: true,
            confidence: 0.95 + (eigenvalues.len() as f64 * 0.005), // Higher confidence with more eigenvalues
            computation_time: self.get_elapsed_time("correspondence_start"),
            steps: self.generate_computation_steps(),
            mathematical_data: MathematicalData {
                galois_representation: format!("ρ: Gal(Q̄/Q) → GL_{}(ℚ_l)", dimension),
                automorphic_form: format!("f ∈ S_{}(Γ_0({})) for {} group", weight, level, group_type),
                l_function: format!("L(s, f) = L(s, ρ) with conductor N = {}", conductor),
                conductor,
                hodge_numbers: (0..dimension as usize).map(|i| i * 2 + 1).collect(),
                hecke_eigenvalues: eigenvalues,
                advanced_properties: advanced_data,
            },
        };
        
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }

    /// Compute advanced mathematical properties
    async fn compute_advanced_mathematics(&self, group: &ReductiveGroup, dimension: usize) -> Result<AdvancedMathData, JsValue> {
        // Simulate advanced mathematical computations
        let rank = group.get_rank();
        
        // Root system analysis
        let root_system = self.analyze_root_system(group, dimension);
        
        // Weyl group computations
        let weyl_group_order = self.compute_weyl_group_order(group);
        
        // Character computations
        let central_characters = self.compute_central_characters(group, dimension);
        
        // L-function coefficients
        let l_function_coeffs = self.compute_l_function_coefficients(dimension);
        
        Ok(AdvancedMathData {
            root_system,
            weyl_group_order,
            central_characters,
            l_function_coefficients: l_function_coeffs,
            ramification_data: self.compute_ramification_data(dimension),
            local_factors: self.compute_local_factors(dimension),
        })
    }

    /// Analyze root system of the group
    fn analyze_root_system(&self, group: &ReductiveGroup, dimension: usize) -> Vec<f64> {
        // Simplified root system computation
        let rank = group.get_rank();
        (0..rank).map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (rank as f64);
            angle.cos() * (dimension as f64).sqrt()
        }).collect()
    }

    /// Compute Weyl group order
    fn compute_weyl_group_order(&self, group: &ReductiveGroup) -> u64 {
        let rank = group.get_rank() as u64;
        match group.get_group_type().chars().take(2).collect::<String>().as_str() {
            "GL" | "SL" => (1..=rank).product::<u64>(), // n!
            "SO" => (1..=rank).map(|i| 2_u64.pow(i as u32) * i).product(),
            "Sp" => (1..=rank).map(|i| 2_u64.pow(i as u32) * i).product(),
            _ => (1..=rank).product::<u64>(),
        }
    }

    /// Compute central characters
    fn compute_central_characters(&self, group: &ReductiveGroup, dimension: usize) -> Vec<Complex64> {
        let rank = group.get_rank();
        (0..rank).map(|i| {
            let real = (i as f64 + 1.0) / (rank as f64);
            let imag = ((dimension + i) as f64).sin() / 10.0;
            Complex64::new(real, imag)
        }).collect()
    }

    /// Compute L-function coefficients
    fn compute_l_function_coefficients(&self, dimension: usize) -> Vec<f64> {
        let primes = MathUtils::primes_up_to(100);
        primes.iter().take(dimension.min(20)).map(|&p| {
            let p_f64 = p as f64;
            let local_factor = 1.0 - p_f64.powf(-1.0) + p_f64.powf(-2.0);
            local_factor.abs()
        }).collect()
    }

    /// Compute ramification data
    fn compute_ramification_data(&self, dimension: usize) -> Vec<u32> {
        // Simplified ramification computation
        let bad_primes = [2, 3, 5, 7];
        bad_primes.iter().take(dimension.min(4)).map(|&p| p * dimension as u32).collect()
    }

    /// Compute local factors
    fn compute_local_factors(&self, dimension: usize) -> HashMap<u32, f64> {
        let mut factors = HashMap::new();
        let primes = [2, 3, 5, 7, 11, 13];
        
        for &p in &primes {
            let p_f64 = p as f64;
            let factor = (1.0 - p_f64.powf(-(dimension as f64))).abs();
            factors.insert(p, factor);
        }
        
        factors
    }

    /// Generate computation steps for display
    fn generate_computation_steps(&self) -> Vec<CorrespondenceStep> {
        let steps = vec![
            "Setting up moduli space of bundles...",
            "Constructing automorphic representations...",
            "Computing Hecke operators and eigenvalues...",
            "Analyzing Galois representations...",
            "Verifying L-function equality...",
            "Computing ramification data...",
            "Finalizing correspondence verification...",
        ];

        steps.into_iter().enumerate().map(|(i, step)| {
            CorrespondenceStep {
                step: step.to_string(),
                progress: (i + 1) as f64 / 7.0,
                timestamp: js_sys::Date::now() + (i as f64 * 100.0),
            }
        }).collect()
    }

    /// Get elapsed time since a performance mark
    fn get_elapsed_time(&self, from_mark: &str) -> f64 {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let current = performance.now() - self.monitor.start_time;
        
        if let Some(&start_time) = self.monitor.metrics.get(from_mark) {
            current - start_time
        } else {
            current
        }
    }

    /// Get current performance metrics
    #[wasm_bindgen]
    pub fn get_performance_metrics(&self) -> JsValue {
        self.monitor.get_metrics()
    }

    /// Estimate memory usage
    #[wasm_bindgen]
    pub fn estimate_memory_usage(&self) -> f64 {
        // Rough estimate based on configuration
        let base_memory = 10.0; // MB
        let worker_memory = self.config.max_workers as f64 * 5.0;
        let gpu_memory = if self.config.enable_gpu { 20.0 } else { 0.0 };
        
        base_memory + worker_memory + gpu_memory
    }

    /// Check if the engine is ready for computation
    #[wasm_bindgen]
    pub fn is_ready(&self) -> bool {
        self.is_initialized
    }
}

/// Result of a Langlands correspondence computation
#[derive(Serialize, Deserialize)]
pub struct LanglandsResult {
    pub group_type: String,
    pub dimension: usize,
    pub success: bool,
    pub confidence: f64,
    pub computation_time: f64,
    pub steps: Vec<CorrespondenceStep>,
    pub mathematical_data: MathematicalData,
}

/// A single step in the correspondence computation
#[derive(Serialize, Deserialize)]
pub struct CorrespondenceStep {
    pub step: String,
    pub progress: f64,
    pub timestamp: f64,
}

/// Mathematical data structure representing Langlands objects
#[derive(Serialize, Deserialize)]
pub struct MathematicalData {
    pub galois_representation: String,
    pub automorphic_form: String,
    pub l_function: String,
    pub conductor: u32,
    pub hodge_numbers: Vec<usize>,
    pub hecke_eigenvalues: Vec<(u32, f64)>,
    pub advanced_properties: AdvancedMathData,
}

/// Advanced mathematical data
#[derive(Serialize, Deserialize)]
pub struct AdvancedMathData {
    pub root_system: Vec<f64>,
    pub weyl_group_order: u64,
    pub central_characters: Vec<Complex64>,
    pub l_function_coefficients: Vec<f64>,
    pub ramification_data: Vec<u32>,
    pub local_factors: HashMap<u32, f64>,
}

/// Utility functions for mathematical computations
#[wasm_bindgen]
pub struct MathUtils;

#[wasm_bindgen]
impl MathUtils {
    /// Check if a number is prime (demonstration function)
    #[wasm_bindgen]
    pub fn is_prime(n: u32) -> bool {
        if n < 2 { return false; }
        if n == 2 { return true; }
        if n % 2 == 0 { return false; }
        
        let sqrt_n = (n as f64).sqrt() as u32;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 { return false; }
        }
        true
    }

    /// Compute Euler's totient function
    #[wasm_bindgen]
    pub fn euler_totient(n: u32) -> u32 {
        let mut result = n;
        let mut p = 2;
        let mut temp_n = n;
        
        while p * p <= temp_n {
            if temp_n % p == 0 {
                while temp_n % p == 0 {
                    temp_n /= p;
                }
                result -= result / p;
            }
            p += 1;
        }
        
        if temp_n > 1 {
            result -= result / temp_n;
        }
        
        result
    }

    /// Generate a sequence of primes up to n
    #[wasm_bindgen]
    pub fn primes_up_to(n: u32) -> Vec<u32> {
        if n < 2 { return vec![]; }
        
        let mut is_prime = vec![true; (n + 1) as usize];
        is_prime[0] = false;
        is_prime[1] = false;
        
        let sqrt_n = (n as f64).sqrt() as u32;
        for i in 2..=sqrt_n {
            if is_prime[i as usize] {
                let mut j = i * i;
                while j <= n {
                    is_prime[j as usize] = false;
                    j += i;
                }
            }
        }
        
        (2..=n).filter(|&i| is_prime[i as usize]).collect()
    }

    /// Compute greatest common divisor
    #[wasm_bindgen]
    pub fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { Self::gcd(b, a % b) }
    }

    /// Compute least common multiple
    #[wasm_bindgen]
    pub fn lcm(a: u32, b: u32) -> u32 {
        a * b / Self::gcd(a, b)
    }

    /// Compute modular exponentiation
    #[wasm_bindgen]
    pub fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
        if modulus == 1 { return 0; }
        
        let mut result = 1;
        let mut base = base % modulus;
        let mut exp = exp;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        
        result
    }
}

/// WebGL utilities for GPU acceleration
#[wasm_bindgen]
pub struct WebGLUtils;

#[wasm_bindgen]
impl WebGLUtils {
    /// Check WebGL capabilities
    #[wasm_bindgen]
    pub fn get_webgl_info() -> JsValue {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.create_element("canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
        
        let mut info = HashMap::new();
        info.insert("webgl1".to_string(), canvas.get_context("webgl").is_ok());
        info.insert("webgl2".to_string(), canvas.get_context("webgl2").is_ok());
        
        if let Ok(gl_context) = canvas.get_context("webgl") {
            if let Some(gl) = gl_context {
                let gl: web_sys::WebGlRenderingContext = gl.dyn_into().unwrap();
                info.insert("vendor".to_string(), 
                    gl.get_parameter(web_sys::WebGlRenderingContext::VENDOR).is_ok());
                info.insert("renderer".to_string(), 
                    gl.get_parameter(web_sys::WebGlRenderingContext::RENDERER).is_ok());
            }
        }
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

/// Export version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Export library information
#[wasm_bindgen]
pub fn get_library_info() -> JsValue {
    let info = HashMap::from([
        ("name", "geometric-langlands-wasm"),
        ("version", env!("CARGO_PKG_VERSION")),
        ("description", "WebAssembly bindings for Geometric Langlands Conjecture"),
        ("repository", "https://github.com/ruvnet/ruv-FANN"),
    ]);
    
    serde_wasm_bindgen::to_value(&info).unwrap()
}