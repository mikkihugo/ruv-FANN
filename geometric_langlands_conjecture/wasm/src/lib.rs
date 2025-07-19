//! WebAssembly Interface for Geometric Langlands Conjecture
//!
//! This module provides an optimized WASM interface for running Langlands
//! correspondence computations in web browsers with high performance.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Performance, Window};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import the main library with WASM features
use geometric_langlands::prelude::*;
use geometric_langlands::wasm::*;

// Global allocator optimized for WASM
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module with performance optimizations
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better debugging
    console_error_panic_hook::set_once();
    
    // Initialize logging
    console::log_1(&"🌟 Geometric Langlands WASM v0.1.0 initialized".into());
    console::log_1(&"🚀 Ready for mathematical computations!".into());
}

/// Advanced WASM-optimized Langlands computation engine
#[wasm_bindgen]
pub struct LanglandsEngine {
    inner: WasmLanglandsEngine,
    computation_cache: HashMap<String, String>,
    memory_pool: Vec<f64>,
    performance_start: f64,
}

#[wasm_bindgen]
impl LanglandsEngine {
    /// Create a new Langlands computation engine
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        
        Self {
            inner: WasmLanglandsEngine::new(),
            computation_cache: HashMap::new(),
            memory_pool: Vec::with_capacity(1024),
            performance_start: performance.now(),
        }
    }
    
    /// Compute correspondence with caching and optimization
    #[wasm_bindgen]
    pub async fn compute_correspondence_advanced(
        &mut self,
        group_type: &str,
        dimension: usize,
        characteristic: u32,
        use_cache: bool,
    ) -> Result<JsValue, JsValue> {
        let cache_key = format!("{}:{}:{}", group_type, dimension, characteristic);
        
        // Check cache first
        if use_cache && self.computation_cache.contains_key(&cache_key) {
            console::log_1(&"📋 Using cached result".into());
            let cached = self.computation_cache.get(&cache_key).unwrap();
            return Ok(JsValue::from_str(cached));
        }
        
        // Mark computation start
        self.mark_performance("advanced_computation_start");
        
        // Run the computation
        let result = self.inner.compute_correspondence(group_type, dimension, characteristic)?;
        
        // Cache the result
        if use_cache {
            let result_str = js_sys::JSON::stringify(&result)
                .map_err(|e| JsValue::from_str("Failed to stringify result"))?
                .as_string()
                .unwrap();
            self.computation_cache.insert(cache_key, result_str);
        }
        
        self.mark_performance("advanced_computation_complete");
        
        Ok(result)
    }
    
    /// Batch process multiple correspondences efficiently
    #[wasm_bindgen]
    pub async fn batch_compute(
        &mut self,
        requests: JsValue,
    ) -> Result<JsValue, JsValue> {
        #[derive(Deserialize)]
        struct BatchRequest {
            group_type: String,
            dimension: usize,
            characteristic: u32,
        }
        
        let requests: Vec<BatchRequest> = serde_wasm_bindgen::from_value(requests)
            .map_err(|e| JsValue::from_str(&format!("Invalid batch request: {}", e)))?;
        
        console::log_1(&format!("🔄 Processing {} batch requests", requests.len()).into());
        
        let mut results = Vec::new();
        
        for (i, req) in requests.iter().enumerate() {
            console::log_1(&format!("📊 Processing request {}/{}", i + 1, requests.len()).into());
            
            let result = self.inner.compute_correspondence(
                &req.group_type,
                req.dimension,
                req.characteristic,
            )?;
            
            results.push(result);
        }
        
        let batch_result = BatchResult {
            results,
            processed_count: requests.len(),
            success_rate: 1.0,
        };
        
        serde_wasm_bindgen::to_value(&batch_result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize batch result: {}", e)))
    }
    
    /// Get detailed performance metrics
    #[wasm_bindgen]
    pub fn get_advanced_metrics(&self) -> JsValue {
        let basic_metrics = self.inner.get_performance_metrics();
        
        let advanced = AdvancedMetrics {
            basic_metrics: serde_wasm_bindgen::from_value(basic_metrics).unwrap_or_default(),
            cache_size: self.computation_cache.len(),
            memory_pool_size: self.memory_pool.len(),
            total_runtime: self.get_total_runtime(),
        };
        
        serde_wasm_bindgen::to_value(&advanced).unwrap()
    }
    
    /// Clear computation cache to free memory
    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.computation_cache.clear();
        console::log_1(&"🧹 Cache cleared".into());
    }
    
    /// Optimize memory usage
    #[wasm_bindgen]
    pub fn optimize_memory(&mut self) {
        self.memory_pool.clear();
        self.memory_pool.shrink_to_fit();
        console::log_1(&"💾 Memory optimized".into());
    }
    
    /// Mark performance checkpoint
    fn mark_performance(&self, label: &str) {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        let elapsed = performance.now() - self.performance_start;
        
        console::log_1(&format!("⏱️ {}: {:.2}ms", label, elapsed).into());
    }
    
    /// Get total runtime
    fn get_total_runtime(&self) -> f64 {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        performance.now() - self.performance_start
    }
}

/// Specialized mathematical utilities for WASM
#[wasm_bindgen]
pub struct MathUtils;

#[wasm_bindgen]
impl MathUtils {
    /// Generate eigenvalue spectrum for visualization
    #[wasm_bindgen]
    pub fn generate_eigenvalue_spectrum(
        group_type: &str,
        dimension: usize,
        count: usize,
    ) -> Result<JsValue, JsValue> {
        let mut eigenvalues = Vec::new();
        
        for i in 0..count {
            let prime = Self::nth_prime(i + 1);
            let eigenvalue = Self::compute_eigenvalue(group_type, dimension, prime)?;
            
            eigenvalues.push(EigenvaluePair {
                prime,
                eigenvalue,
                multiplicity: 1,
            });
        }
        
        let spectrum = EigenvalueSpectrum {
            group_type: group_type.to_string(),
            dimension,
            eigenvalues,
            spectral_gap: Self::compute_spectral_gap(&eigenvalues),
        };
        
        serde_wasm_bindgen::to_value(&spectrum)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize spectrum: {}", e)))
    }
    
    /// Compute matrix representation for visualization
    #[wasm_bindgen]
    pub fn compute_matrix_representation(
        group_type: &str,
        dimension: usize,
    ) -> Result<JsValue, JsValue> {
        let size = dimension * dimension;
        let mut matrix = Vec::with_capacity(size);
        
        // Generate a meaningful matrix representation
        for i in 0..dimension {
            for j in 0..dimension {
                let value = match group_type {
                    "GL" => if i == j { 1.0 } else { 0.0 },
                    "SL" => if i == j { 1.0 } else { 0.1 * (i as f64 * j as f64).sin() },
                    "SO" => if i == j { 1.0 } else if i + j == dimension - 1 { -1.0 } else { 0.0 },
                    "Sp" => {
                        let mid = dimension / 2;
                        if i < mid && j >= mid && i + mid == j { 1.0 }
                        else if i >= mid && j < mid && i == j + mid { -1.0 }
                        else { 0.0 }
                    },
                    _ => 0.0,
                };
                matrix.push(value);
            }
        }
        
        let representation = MatrixRepresentation {
            group_type: group_type.to_string(),
            dimension,
            matrix,
            determinant: Self::compute_determinant(&matrix, dimension),
            trace: Self::compute_trace(&matrix, dimension),
        };
        
        serde_wasm_bindgen::to_value(&representation)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize matrix: {}", e)))
    }
    
    /// Generate Hecke algebra data for interactive exploration
    #[wasm_bindgen]
    pub fn generate_hecke_algebra(
        group_type: &str,
        dimension: usize,
        prime_count: usize,
    ) -> Result<JsValue, JsValue> {
        let mut operators = Vec::new();
        
        for i in 0..prime_count {
            let prime = Self::nth_prime(i + 1);
            let eigenvalue = Self::compute_eigenvalue(group_type, dimension, prime)?;
            
            operators.push(HeckeOperatorData {
                prime,
                eigenvalue,
                polynomial_degree: dimension,
                ramification: prime % 4 == 1,
            });
        }
        
        let algebra = HeckeAlgebraData {
            group_type: group_type.to_string(),
            dimension,
            operators,
            rank: dimension,
            central_character: Self::compute_central_character(group_type, dimension),
        };
        
        serde_wasm_bindgen::to_value(&algebra)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize Hecke algebra: {}", e)))
    }
    
    // Helper methods
    fn nth_prime(n: usize) -> u32 {
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];
        if n <= primes.len() {
            primes[n - 1]
        } else {
            primes[n % primes.len()]
        }
    }
    
    fn compute_eigenvalue(group_type: &str, dimension: usize, prime: u32) -> Result<f64, JsValue> {
        let base = match group_type {
            "GL" => prime as f64,
            "SL" => (prime as f64) - 1.0,
            "SO" => (prime as f64).sqrt(),
            "Sp" => 2.0 * (prime as f64).sqrt(),
            _ => return Err(JsValue::from_str("Unknown group type")),
        };
        
        Ok(base * (dimension as f64).ln())
    }
    
    fn compute_spectral_gap(eigenvalues: &[EigenvaluePair]) -> f64 {
        if eigenvalues.len() < 2 { return 0.0; }
        
        let mut values: Vec<f64> = eigenvalues.iter().map(|e| e.eigenvalue).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        values[1] - values[0]
    }
    
    fn compute_determinant(matrix: &[f64], dimension: usize) -> f64 {
        if dimension == 1 { return matrix[0]; }
        if dimension == 2 {
            return matrix[0] * matrix[3] - matrix[1] * matrix[2];
        }
        
        // Simplified determinant for larger matrices
        matrix.iter().step_by(dimension + 1).product()
    }
    
    fn compute_trace(matrix: &[f64], dimension: usize) -> f64 {
        (0..dimension).map(|i| matrix[i * dimension + i]).sum()
    }
    
    fn compute_central_character(group_type: &str, dimension: usize) -> f64 {
        match group_type {
            "GL" => dimension as f64,
            "SL" => 1.0,
            "SO" => if dimension % 2 == 0 { 1.0 } else { -1.0 },
            "Sp" => 1.0,
            _ => 0.0,
        }
    }
}

/// Visualization data generator
#[wasm_bindgen]
pub struct VisualizationEngine;

#[wasm_bindgen]
impl VisualizationEngine {
    /// Generate data for correspondence network visualization
    #[wasm_bindgen]
    pub fn generate_correspondence_network(
        max_dimension: usize,
        group_types: JsValue,
    ) -> Result<JsValue, JsValue> {
        let group_types: Vec<String> = serde_wasm_bindgen::from_value(group_types)
            .map_err(|e| JsValue::from_str(&format!("Invalid group types: {}", e)))?;
        
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 0;
        
        // Create nodes for each group and dimension
        for group_type in &group_types {
            for dim in 1..=max_dimension {
                nodes.push(NetworkNode {
                    id: node_id,
                    label: format!("{}({})", group_type, dim),
                    group_type: group_type.clone(),
                    dimension: dim,
                    x: (node_id as f64 * 50.0) % 500.0,
                    y: (dim as f64 * 80.0) % 400.0,
                    size: 10.0 + (dim as f64 * 2.0),
                });
                node_id += 1;
            }
        }
        
        // Create edges representing correspondences
        for (i, node1) in nodes.iter().enumerate() {
            for (j, node2) in nodes.iter().enumerate() {
                if i < j && Self::has_correspondence(&node1, &node2) {
                    edges.push(NetworkEdge {
                        source: node1.id,
                        target: node2.id,
                        weight: Self::correspondence_strength(&node1, &node2),
                        correspondence_type: Self::get_correspondence_type(&node1, &node2),
                    });
                }
            }
        }
        
        let network = CorrespondenceNetwork {
            nodes,
            edges,
            metadata: NetworkMetadata {
                max_dimension,
                group_count: group_types.len(),
                total_correspondences: edges.len(),
            },
        };
        
        serde_wasm_bindgen::to_value(&network)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize network: {}", e)))
    }
    
    /// Generate 3D visualization data for moduli spaces
    #[wasm_bindgen]
    pub fn generate_moduli_space_3d(
        group_type: &str,
        dimension: usize,
        resolution: usize,
    ) -> Result<JsValue, JsValue> {
        let mut points = Vec::new();
        let step = 2.0 / resolution as f64;
        
        for i in 0..resolution {
            for j in 0..resolution {
                let x = -1.0 + i as f64 * step;
                let y = -1.0 + j as f64 * step;
                let z = Self::compute_moduli_height(group_type, dimension, x, y);
                
                points.push(Point3D { x, y, z });
            }
        }
        
        let space = ModuliSpace3D {
            group_type: group_type.to_string(),
            dimension,
            resolution,
            points,
            min_z: points.iter().map(|p| p.z).fold(f64::INFINITY, f64::min),
            max_z: points.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max),
        };
        
        serde_wasm_bindgen::to_value(&space)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize 3D space: {}", e)))
    }
    
    // Helper methods
    fn has_correspondence(node1: &NetworkNode, node2: &NetworkNode) -> bool {
        // Simplified correspondence detection
        node1.dimension == node2.dimension || 
        (node1.group_type == "GL" && node2.group_type == "SL") ||
        (node1.group_type == "SO" && node2.group_type == "Sp")
    }
    
    fn correspondence_strength(node1: &NetworkNode, node2: &NetworkNode) -> f64 {
        if node1.dimension == node2.dimension {
            0.9
        } else {
            0.3 * (-((node1.dimension as f64 - node2.dimension as f64).abs())).exp()
        }
    }
    
    fn get_correspondence_type(node1: &NetworkNode, node2: &NetworkNode) -> String {
        if node1.dimension == node2.dimension {
            "dimensional".to_string()
        } else if node1.group_type == node2.group_type {
            "rank".to_string()
        } else {
            "duality".to_string()
        }
    }
    
    fn compute_moduli_height(group_type: &str, dimension: usize, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        let base_height = match group_type {
            "GL" => (1.0 - r * r).max(0.0),
            "SL" => (1.0 - r * r).max(0.0).sqrt(),
            "SO" => (r * std::f64::consts::PI).sin() * (1.0 - r).max(0.0),
            "Sp" => (r * 2.0 * std::f64::consts::PI).cos() * (1.0 - r * r).max(0.0),
            _ => 0.0,
        };
        
        base_height * (dimension as f64).ln()
    }
}

// Data structures for serialization

#[derive(Serialize, Deserialize)]
struct BatchResult {
    results: Vec<JsValue>,
    processed_count: usize,
    success_rate: f64,
}

#[derive(Serialize, Deserialize, Default)]
struct AdvancedMetrics {
    basic_metrics: HashMap<String, f64>,
    cache_size: usize,
    memory_pool_size: usize,
    total_runtime: f64,
}

#[derive(Serialize, Deserialize)]
struct EigenvaluePair {
    prime: u32,
    eigenvalue: f64,
    multiplicity: usize,
}

#[derive(Serialize, Deserialize)]
struct EigenvalueSpectrum {
    group_type: String,
    dimension: usize,
    eigenvalues: Vec<EigenvaluePair>,
    spectral_gap: f64,
}

#[derive(Serialize, Deserialize)]
struct MatrixRepresentation {
    group_type: String,
    dimension: usize,
    matrix: Vec<f64>,
    determinant: f64,
    trace: f64,
}

#[derive(Serialize, Deserialize)]
struct HeckeOperatorData {
    prime: u32,
    eigenvalue: f64,
    polynomial_degree: usize,
    ramification: bool,
}

#[derive(Serialize, Deserialize)]
struct HeckeAlgebraData {
    group_type: String,
    dimension: usize,
    operators: Vec<HeckeOperatorData>,
    rank: usize,
    central_character: f64,
}

#[derive(Serialize, Deserialize)]
struct NetworkNode {
    id: usize,
    label: String,
    group_type: String,
    dimension: usize,
    x: f64,
    y: f64,
    size: f64,
}

#[derive(Serialize, Deserialize)]
struct NetworkEdge {
    source: usize,
    target: usize,
    weight: f64,
    correspondence_type: String,
}

#[derive(Serialize, Deserialize)]
struct NetworkMetadata {
    max_dimension: usize,
    group_count: usize,
    total_correspondences: usize,
}

#[derive(Serialize, Deserialize)]
struct CorrespondenceNetwork {
    nodes: Vec<NetworkNode>,
    edges: Vec<NetworkEdge>,
    metadata: NetworkMetadata,
}

#[derive(Serialize, Deserialize)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Serialize, Deserialize)]
struct ModuliSpace3D {
    group_type: String,
    dimension: usize,
    resolution: usize,
    points: Vec<Point3D>,
    min_z: f64,
    max_z: f64,
}

/// Export version and build information
#[wasm_bindgen]
pub fn get_wasm_info() -> JsValue {
    let info = serde_json::json!({
        "name": "geometric-langlands-wasm",
        "version": env!("CARGO_PKG_VERSION"),
        "build_time": env!("BUILD_TIMESTAMP"),
        "features": ["advanced-computation", "visualization", "caching"],
        "bundle_size": "optimized",
        "target": "wasm32-unknown-unknown"
    });
    
    serde_wasm_bindgen::to_value(&info).unwrap()
}