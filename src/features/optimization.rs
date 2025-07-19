//! Performance optimization for feature extraction pipeline
//!
//! This module provides various optimization strategies to improve
//! the performance of feature extraction operations.

use crate::features::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Optimization strategies available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizationStrategy {
    /// No optimization
    None,
    /// Memory optimization (reduce memory usage)
    Memory,
    /// Speed optimization (faster processing)
    Speed,
    /// Balanced optimization
    Balanced,
    /// Cache-heavy optimization
    Cache,
    /// Parallel processing optimization
    Parallel,
    /// Custom optimization
    Custom,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        OptimizationStrategy::Balanced
    }
}

/// Optimization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizationConfig {
    /// Primary optimization strategy
    pub strategy: OptimizationStrategy,
    /// Enable result caching
    pub enable_caching: bool,
    /// Maximum cache size (number of items)
    pub max_cache_size: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Number of worker threads (0 = auto)
    pub num_threads: usize,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Enable lazy evaluation
    pub enable_lazy_evaluation: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit_bytes: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Balanced,
            enable_caching: true,
            max_cache_size: 1000,
            enable_parallel: true,
            num_threads: 0, // Auto-detect
            enable_memory_pooling: true,
            enable_lazy_evaluation: false,
            batch_size: 32,
            enable_simd: true,
            memory_limit_bytes: 0, // Unlimited
        }
    }
}

/// Performance metrics for optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerformanceMetrics {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Throughput (objects per second)
    pub throughput_ops: f64,
    /// Average latency per operation in milliseconds
    pub avg_latency_ms: f64,
    /// Number of operations performed
    pub operations_count: u64,
    /// SIMD utilization (0.0 to 1.0)
    pub simd_utilization: f64,
    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            processing_time_ms: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
            throughput_ops: 0.0,
            avg_latency_ms: 0.0,
            operations_count: 0,
            simd_utilization: 0.0,
            parallel_efficiency: 0.0,
        }
    }
    
    pub fn calculate_throughput(&mut self) {
        if self.processing_time_ms > 0 {
            self.throughput_ops = (self.operations_count as f64 * 1000.0) / self.processing_time_ms as f64;
            self.avg_latency_ms = self.processing_time_ms as f64 / self.operations_count as f64;
        }
    }
}

/// Cache for storing computed results
#[derive(Debug)]
pub struct ResultCache<K, V> {
    cache: Arc<Mutex<HashMap<K, V>>>,
    max_size: usize,
    hits: Arc<Mutex<u64>>,
    misses: Arc<Mutex<u64>>,
}

impl<K, V> ResultCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        if let Ok(cache) = self.cache.lock() {
            if let Some(value) = cache.get(key) {
                if let Ok(mut hits) = self.hits.lock() {
                    *hits += 1;
                }
                return Some(value.clone());
            }
        }
        
        if let Ok(mut misses) = self.misses.lock() {
            *misses += 1;
        }
        None
    }
    
    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut cache) = self.cache.lock() {
            // Simple eviction strategy: remove random entry if at capacity
            if cache.len() >= self.max_size {
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }
            cache.insert(key, value);
        }
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.lock().map(|h| *h).unwrap_or(0);
        let misses = self.misses.lock().map(|m| *m).unwrap_or(0);
        
        if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        }
    }
    
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut hits) = self.hits.lock() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.misses.lock() {
            *misses = 0;
        }
    }
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<f64>>>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
        }
    }
    
    /// Get a vector of the specified size from the pool
    pub fn get_vector(&mut self, size: usize) -> Vec<f64> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(mut vec) = pool.pop() {
                vec.clear();
                vec.resize(size, 0.0);
                return vec;
            }
        }
        
        // Create new vector if pool is empty
        vec![0.0; size]
    }
    
    /// Return a vector to the pool
    pub fn return_vector(&mut self, vec: Vec<f64>) {
        let size = vec.capacity();
        
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        
        if pool.len() < self.max_pool_size {
            pool.push(vec);
        }
        // Otherwise, let it be dropped
    }
    
    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
    }
}

/// Optimized feature extractor with caching and parallel processing
#[derive(Debug)]
pub struct OptimizedExtractor {
    config: OptimizationConfig,
    base_extractor: UniversalExtractor,
    cache: ResultCache<String, FeatureVector>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl OptimizedExtractor {
    pub fn new(config: OptimizationConfig) -> Self {
        let cache = ResultCache::new(config.max_cache_size);
        let memory_pool = Arc::new(Mutex::new(MemoryPool::new(100)));
        
        Self {
            config,
            base_extractor: UniversalExtractor::default(),
            cache,
            memory_pool,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        }
    }
    
    /// Extract features with optimization
    pub fn extract_optimized(
        &self,
        object: &dyn std::any::Any,
        config: &ExtractorConfig,
        cache_key: Option<String>,
    ) -> FeatureResult<FeatureVector> {
        let start_time = std::time::Instant::now();
        
        // Check cache if enabled and key provided
        if self.config.enable_caching {
            if let Some(key) = &cache_key {
                if let Some(cached_result) = self.cache.get(key) {
                    self.update_metrics(start_time, true);
                    return Ok(cached_result);
                }
            }
        }
        
        // Perform extraction
        let result = self.base_extractor.extract_from_object(object, config)?;
        
        // Cache result if enabled
        if self.config.enable_caching {
            if let Some(key) = cache_key {
                self.cache.insert(key, result.clone());
            }
        }
        
        self.update_metrics(start_time, false);
        Ok(result)
    }
    
    /// Batch extract with parallel processing
    #[cfg(feature = "parallel")]
    pub fn extract_batch_parallel(
        &self,
        objects: &[Box<dyn std::any::Any + Send + Sync>],
        config: &ExtractorConfig,
    ) -> FeatureResult<Vec<FeatureVector>> {
        use rayon::prelude::*;
        
        if !self.config.enable_parallel {
            return self.extract_batch_sequential(objects, config);
        }
        
        let start_time = std::time::Instant::now();
        
        let results: Vec<FeatureResult<FeatureVector>> = objects
            .par_iter()
            .map(|obj| self.base_extractor.extract_from_object(obj.as_ref(), config))
            .collect();
        
        let mut features = Vec::with_capacity(objects.len());
        for result in results {
            features.push(result?);
        }
        
        self.update_batch_metrics(start_time, objects.len());
        Ok(features)
    }
    
    /// Sequential batch extraction
    pub fn extract_batch_sequential(
        &self,
        objects: &[Box<dyn std::any::Any>],
        config: &ExtractorConfig,
    ) -> FeatureResult<Vec<FeatureVector>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(objects.len());
        
        for obj in objects {
            let features = self.base_extractor.extract_from_object(obj.as_ref(), config)?;
            results.push(features);
        }
        
        self.update_batch_metrics(start_time, objects.len());
        Ok(results)
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.cache_hit_rate = self.cache.hit_rate();
            metrics.calculate_throughput();
            metrics.clone()
        } else {
            PerformanceMetrics::new()
        }
    }
    
    /// Reset metrics
    pub fn reset_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = PerformanceMetrics::new();
        }
        self.cache.clear();
    }
    
    fn update_metrics(&self, start_time: std::time::Instant, was_cached: bool) {
        let elapsed = start_time.elapsed();
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.processing_time_ms += elapsed.as_millis() as u64;
            metrics.operations_count += 1;
            
            if !was_cached {
                // Estimate memory usage (rough)
                metrics.memory_usage_bytes += 1024; // Rough estimate
            }
        }
    }
    
    fn update_batch_metrics(&self, start_time: std::time::Instant, batch_size: usize) {
        let elapsed = start_time.elapsed();
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.processing_time_ms += elapsed.as_millis() as u64;
            metrics.operations_count += batch_size as u64;
            metrics.memory_usage_bytes += (batch_size * 1024) as u64; // Rough estimate
        }
    }
}

/// SIMD-optimized operations for feature vectors
#[cfg(feature = "simd")]
pub struct SimdOptimizer;

#[cfg(feature = "simd")]
impl SimdOptimizer {
    /// SIMD-optimized L2 normalization
    pub fn normalize_l2_simd(values: &mut [f64]) {
        // This is a placeholder - actual SIMD implementation would use
        // platform-specific intrinsics or libraries like wide/simdeez
        let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for value in values.iter_mut() {
                *value /= norm;
            }
        }
    }
    
    /// SIMD-optimized dot product
    pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        
        // Placeholder implementation
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    /// SIMD-optimized vector addition
    pub fn add_vectors_simd(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *result_val = a_val + b_val;
        }
    }
}

/// Optimization analyzer that suggests best optimization strategies
#[derive(Debug)]
pub struct OptimizationAnalyzer {
    performance_history: Vec<(OptimizationStrategy, PerformanceMetrics)>,
}

impl OptimizationAnalyzer {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
        }
    }
    
    /// Record performance metrics for a given strategy
    pub fn record_performance(
        &mut self,
        strategy: OptimizationStrategy,
        metrics: PerformanceMetrics,
    ) {
        self.performance_history.push((strategy, metrics));
        
        // Keep only recent history (last 100 entries)
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }
    
    /// Suggest the best optimization strategy based on historical performance
    pub fn suggest_strategy(&self) -> OptimizationStrategy {
        if self.performance_history.is_empty() {
            return OptimizationStrategy::Balanced;
        }
        
        // Calculate average throughput for each strategy
        let mut strategy_performance: HashMap<OptimizationStrategy, (f64, usize)> = HashMap::new();
        
        for (strategy, metrics) in &self.performance_history {
            let entry = strategy_performance.entry(*strategy).or_insert((0.0, 0));
            entry.0 += metrics.throughput_ops;
            entry.1 += 1;
        }
        
        // Find strategy with highest average throughput
        let mut best_strategy = OptimizationStrategy::Balanced;
        let mut best_throughput = 0.0;
        
        for (strategy, (total_throughput, count)) in strategy_performance {
            let avg_throughput = total_throughput / count as f64;
            if avg_throughput > best_throughput {
                best_throughput = avg_throughput;
                best_strategy = strategy;
            }
        }
        
        best_strategy
    }
    
    /// Get performance summary for all strategies
    pub fn performance_summary(&self) -> HashMap<OptimizationStrategy, PerformanceMetrics> {
        let mut summary = HashMap::new();
        let mut strategy_data: HashMap<OptimizationStrategy, Vec<&PerformanceMetrics>> = HashMap::new();
        
        // Group metrics by strategy
        for (strategy, metrics) in &self.performance_history {
            strategy_data.entry(*strategy).or_insert_with(Vec::new).push(metrics);
        }
        
        // Calculate averages
        for (strategy, metrics_list) in strategy_data {
            if metrics_list.is_empty() {
                continue;
            }
            
            let mut avg_metrics = PerformanceMetrics::new();
            let count = metrics_list.len() as f64;
            
            for metrics in &metrics_list {
                avg_metrics.processing_time_ms += metrics.processing_time_ms;
                avg_metrics.memory_usage_bytes += metrics.memory_usage_bytes;
                avg_metrics.cache_hit_rate += metrics.cache_hit_rate;
                avg_metrics.throughput_ops += metrics.throughput_ops;
                avg_metrics.avg_latency_ms += metrics.avg_latency_ms;
                avg_metrics.operations_count += metrics.operations_count;
                avg_metrics.simd_utilization += metrics.simd_utilization;
                avg_metrics.parallel_efficiency += metrics.parallel_efficiency;
            }
            
            // Calculate averages
            avg_metrics.processing_time_ms = (avg_metrics.processing_time_ms as f64 / count) as u64;
            avg_metrics.memory_usage_bytes = (avg_metrics.memory_usage_bytes as f64 / count) as u64;
            avg_metrics.cache_hit_rate /= count;
            avg_metrics.throughput_ops /= count;
            avg_metrics.avg_latency_ms /= count;
            avg_metrics.operations_count = (avg_metrics.operations_count as f64 / count) as u64;
            avg_metrics.simd_utilization /= count;
            avg_metrics.parallel_efficiency /= count;
            
            summary.insert(strategy, avg_metrics);
        }
        
        summary
    }
}

impl Default for OptimizationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-tuning optimizer that adjusts parameters based on performance
#[derive(Debug)]
pub struct AutoTuner {
    analyzer: OptimizationAnalyzer,
    current_config: OptimizationConfig,
    tuning_iterations: usize,
}

impl AutoTuner {
    pub fn new(initial_config: OptimizationConfig) -> Self {
        Self {
            analyzer: OptimizationAnalyzer::new(),
            current_config: initial_config,
            tuning_iterations: 0,
        }
    }
    
    /// Run auto-tuning for the given workload
    pub fn auto_tune<F>(
        &mut self,
        workload: F,
        max_iterations: usize,
    ) -> FeatureResult<OptimizationConfig>
    where
        F: Fn(&OptimizationConfig) -> FeatureResult<PerformanceMetrics>,
    {
        let strategies = [
            OptimizationStrategy::Speed,
            OptimizationStrategy::Memory,
            OptimizationStrategy::Balanced,
            OptimizationStrategy::Cache,
            OptimizationStrategy::Parallel,
        ];
        
        for _ in 0..max_iterations {
            for &strategy in &strategies {
                let mut test_config = self.current_config.clone();
                test_config.strategy = strategy;
                
                // Adjust parameters based on strategy
                self.adjust_config_for_strategy(&mut test_config);
                
                match workload(&test_config) {
                    Ok(metrics) => {
                        self.analyzer.record_performance(strategy, metrics);
                    }
                    Err(_) => {
                        // Skip this configuration if it fails
                        continue;
                    }
                }
            }
            
            self.tuning_iterations += 1;
        }
        
        // Select best strategy and update config
        let best_strategy = self.analyzer.suggest_strategy();
        self.current_config.strategy = best_strategy;
        self.adjust_config_for_strategy(&mut self.current_config);
        
        Ok(self.current_config.clone())
    }
    
    fn adjust_config_for_strategy(&self, config: &mut OptimizationConfig) {
        match config.strategy {
            OptimizationStrategy::Speed => {
                config.enable_parallel = true;
                config.enable_simd = true;
                config.batch_size = 64;
                config.enable_caching = false; // Reduce overhead
            }
            OptimizationStrategy::Memory => {
                config.enable_memory_pooling = true;
                config.max_cache_size = 100; // Smaller cache
                config.batch_size = 16; // Smaller batches
                config.enable_lazy_evaluation = true;
            }
            OptimizationStrategy::Cache => {
                config.enable_caching = true;
                config.max_cache_size = 2000; // Larger cache
                config.enable_memory_pooling = true;
            }
            OptimizationStrategy::Parallel => {
                config.enable_parallel = true;
                config.num_threads = num_cpus::get(); // Use all cores
                config.batch_size = 128; // Larger batches for parallel
            }
            OptimizationStrategy::Balanced => {
                // Keep defaults
            }
            _ => {}
        }
    }
    
    /// Get current optimization configuration
    pub fn current_config(&self) -> &OptimizationConfig {
        &self.current_config
    }
    
    /// Get performance analysis
    pub fn performance_analysis(&self) -> HashMap<OptimizationStrategy, PerformanceMetrics> {
        self.analyzer.performance_summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert_eq!(config.strategy, OptimizationStrategy::Balanced);
        assert!(config.enable_caching);
        assert!(config.enable_parallel);
        assert_eq!(config.batch_size, 32);
    }
    
    #[test]
    fn test_result_cache() {
        let cache: ResultCache<String, i32> = ResultCache::new(2);
        
        // Test miss
        assert!(cache.get(&"key1".to_string()).is_none());
        
        // Test insert and hit
        cache.insert("key1".to_string(), 42);
        assert_eq!(cache.get(&"key1".to_string()), Some(42));
        
        // Test eviction
        cache.insert("key2".to_string(), 24);
        cache.insert("key3".to_string(), 12); // Should evict key1
        
        assert!(cache.get(&"key1".to_string()).is_none());
        assert_eq!(cache.get(&"key2".to_string()), Some(24));
        assert_eq!(cache.get(&"key3".to_string()), Some(12));
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(2);
        
        // Get new vector
        let vec1 = pool.get_vector(10);
        assert_eq!(vec1.len(), 10);
        
        // Return to pool
        pool.return_vector(vec1);
        
        // Get from pool (should reuse)
        let vec2 = pool.get_vector(10);
        assert_eq!(vec2.len(), 10);
    }
    
    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.operations_count = 100;
        metrics.processing_time_ms = 1000;
        metrics.calculate_throughput();
        
        assert_eq!(metrics.throughput_ops, 100.0);
        assert_eq!(metrics.avg_latency_ms, 10.0);
    }
    
    #[test]
    fn test_optimization_analyzer() {
        let mut analyzer = OptimizationAnalyzer::new();
        
        let mut metrics = PerformanceMetrics::new();
        metrics.throughput_ops = 50.0;
        analyzer.record_performance(OptimizationStrategy::Speed, metrics.clone());
        
        metrics.throughput_ops = 100.0;
        analyzer.record_performance(OptimizationStrategy::Parallel, metrics);
        
        let suggested = analyzer.suggest_strategy();
        assert_eq!(suggested, OptimizationStrategy::Parallel);
    }
}