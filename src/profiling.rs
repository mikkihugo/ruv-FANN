//! Performance profiling and monitoring for neural network operations
//!
//! This module provides comprehensive profiling tools to measure and optimize
//! neural network performance across different operations and configurations.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use num_traits::Float;

/// Performance profiler for neural network operations
#[derive(Debug, Clone)]
pub struct Profiler {
    /// Timing data for different operations
    timing_data: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    /// Memory usage tracking
    memory_usage: Arc<Mutex<HashMap<String, usize>>>,
    /// Operation counters
    operation_counts: Arc<Mutex<HashMap<String, u64>>>,
    /// Current profiling session start time
    session_start: Instant,
    /// Whether profiling is enabled
    enabled: bool,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timing_data: Arc::new(Mutex::new(HashMap::new())),
            memory_usage: Arc::new(Mutex::new(HashMap::new())),
            operation_counts: Arc::new(Mutex::new(HashMap::new())),
            session_start: Instant::now(),
            enabled: true,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
        self.session_start = Instant::now();
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Start timing an operation
    pub fn start_timer(&self, operation: &str) -> Timer {
        if self.enabled {
            Timer::new(operation.to_string(), self.timing_data.clone())
        } else {
            Timer::disabled()
        }
    }

    /// Record memory usage for an operation
    pub fn record_memory_usage(&self, operation: &str, bytes: usize) {
        if !self.enabled {
            return;
        }

        let mut memory = self.memory_usage.lock().unwrap();
        memory.insert(operation.to_string(), bytes);
    }

    /// Increment operation counter
    pub fn increment_counter(&self, operation: &str) {
        if !self.enabled {
            return;
        }

        let mut counts = self.operation_counts.lock().unwrap();
        *counts.entry(operation.to_string()).or_insert(0) += 1;
    }

    /// Get timing statistics for an operation
    pub fn get_timing_stats(&self, operation: &str) -> Option<TimingStats> {
        let timing_data = self.timing_data.lock().unwrap();
        let times = timing_data.get(operation)?;

        if times.is_empty() {
            return None;
        }

        let mut sorted_times = times.clone();
        sorted_times.sort();

        let total: Duration = times.iter().sum();
        let count = times.len();
        let mean = total / count as u32;

        let median = sorted_times[count / 2];
        let min = sorted_times[0];
        let max = sorted_times[count - 1];

        // Calculate percentiles
        let p95_idx = (count as f64 * 0.95) as usize;
        let p99_idx = (count as f64 * 0.99) as usize;
        let p95 = sorted_times[p95_idx.min(count - 1)];
        let p99 = sorted_times[p99_idx.min(count - 1)];

        Some(TimingStats {
            operation: operation.to_string(),
            count,
            total,
            mean,
            median,
            min,
            max,
            p95,
            p99,
        })
    }

    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        let timing_data = self.timing_data.lock().unwrap();
        let memory_usage = self.memory_usage.lock().unwrap();
        let operation_counts = self.operation_counts.lock().unwrap();

        let mut timing_stats = HashMap::new();
        for operation in timing_data.keys() {
            if let Some(stats) = self.get_timing_stats(operation) {
                timing_stats.insert(operation.clone(), stats);
            }
        }

        PerformanceReport {
            session_duration: self.session_start.elapsed(),
            timing_stats,
            memory_usage: memory_usage.clone(),
            operation_counts: operation_counts.clone(),
            total_operations: operation_counts.values().sum(),
        }
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        self.timing_data.lock().unwrap().clear();
        self.memory_usage.lock().unwrap().clear();
        self.operation_counts.lock().unwrap().clear();
        self.session_start = Instant::now();
    }

    /// Export profiling data to JSON
    pub fn export_json(&self) -> String {
        let report = self.get_performance_report();
        serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation duration
pub struct Timer {
    operation: String,
    start_time: Instant,
    timing_data: Option<Arc<Mutex<HashMap<String, Vec<Duration>>>>>,
}

impl Timer {
    fn new(operation: String, timing_data: Arc<Mutex<HashMap<String, Vec<Duration>>>>) -> Self {
        Self {
            operation,
            start_time: Instant::now(),
            timing_data: Some(timing_data),
        }
    }

    fn disabled() -> Self {
        Self {
            operation: String::new(),
            start_time: Instant::now(),
            timing_data: None,
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(ref timing_data) = self.timing_data {
            let duration = self.start_time.elapsed();
            let mut data = timing_data.lock().unwrap();
            data.entry(self.operation.clone()).or_insert_with(Vec::new).push(duration);
        }
    }
}

/// Timing statistics for an operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimingStats {
    pub operation: String,
    pub count: usize,
    pub total: Duration,
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerformanceReport {
    pub session_duration: Duration,
    pub timing_stats: HashMap<String, TimingStats>,
    pub memory_usage: HashMap<String, usize>,
    pub operation_counts: HashMap<String, u64>,
    pub total_operations: u64,
}

/// Macro for easy profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $operation:expr, $code:block) => {{
        let _timer = $profiler.start_timer($operation);
        $profiler.increment_counter($operation);
        $code
    }};
}

/// Global profiler instance
lazy_static::lazy_static! {
    static ref GLOBAL_PROFILER: Profiler = Profiler::new();
}

/// Get the global profiler instance
pub fn global_profiler() -> &'static Profiler {
    &GLOBAL_PROFILER
}

/// Profile a neural network training session
pub struct TrainingProfiler<T: Float> {
    profiler: Profiler,
    epoch_times: Vec<Duration>,
    loss_history: Vec<T>,
    current_epoch: usize,
}

impl<T: Float> TrainingProfiler<T> {
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
            epoch_times: Vec::new(),
            loss_history: Vec::new(),
            current_epoch: 0,
        }
    }

    pub fn start_epoch(&mut self) -> EpochTimer<T> {
        self.current_epoch += 1;
        EpochTimer::new(self.current_epoch, &mut self.epoch_times)
    }

    pub fn record_loss(&mut self, loss: T) {
        self.loss_history.push(loss);
    }

    pub fn get_training_report(&self) -> TrainingReport<T> {
        let total_time: Duration = self.epoch_times.iter().sum();
        let avg_epoch_time = if !self.epoch_times.is_empty() {
            total_time / self.epoch_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        TrainingReport {
            total_epochs: self.current_epoch,
            total_time,
            avg_epoch_time,
            epoch_times: self.epoch_times.clone(),
            loss_history: self.loss_history.clone(),
            final_loss: self.loss_history.last().copied(),
            performance_report: self.profiler.get_performance_report(),
        }
    }
}

impl<T: Float> Default for TrainingProfiler<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for individual training epochs
pub struct EpochTimer<T: Float> {
    epoch: usize,
    start_time: Instant,
    epoch_times: *mut Vec<Duration>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> EpochTimer<T> {
    fn new(epoch: usize, epoch_times: &mut Vec<Duration>) -> Self {
        Self {
            epoch,
            start_time: Instant::now(),
            epoch_times: epoch_times as *mut Vec<Duration>,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Drop for EpochTimer<T> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        unsafe {
            (*self.epoch_times).push(duration);
        }
    }
}

/// Training session performance report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingReport<T: Float> {
    pub total_epochs: usize,
    pub total_time: Duration,
    pub avg_epoch_time: Duration,
    pub epoch_times: Vec<Duration>,
    pub loss_history: Vec<T>,
    pub final_loss: Option<T>,
    pub performance_report: PerformanceReport,
}

/// Hardware performance monitor
pub struct HardwareMonitor {
    profiler: Profiler,
}

impl HardwareMonitor {
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
        }
    }

    /// Get CPU usage percentage (simplified implementation)
    pub fn get_cpu_usage(&self) -> f64 {
        // This is a simplified implementation
        // In a real implementation, you'd query system APIs
        50.0 // Placeholder
    }

    /// Get memory usage in bytes
    pub fn get_memory_usage(&self) -> usize {
        // This would query actual system memory usage
        // For now, return a placeholder
        1024 * 1024 * 512 // 512 MB placeholder
    }

    /// Check if SIMD instructions are available
    pub fn get_simd_capabilities(&self) -> SimdCapabilities {
        SimdCapabilities {
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            sse4_1: is_x86_feature_detected!("sse4.1"),
            sse4_2: is_x86_feature_detected!("sse4.2"),
        }
    }

    /// Get number of CPU cores
    pub fn get_cpu_cores(&self) -> usize {
        num_cpus::get()
    }

    /// Generate hardware report
    pub fn generate_report(&self) -> HardwareReport {
        HardwareReport {
            cpu_cores: self.get_cpu_cores(),
            cpu_usage: self.get_cpu_usage(),
            memory_usage: self.get_memory_usage(),
            simd_capabilities: self.get_simd_capabilities(),
        }
    }
}

impl Default for HardwareMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD instruction capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimdCapabilities {
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
}

/// Hardware performance report
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HardwareReport {
    pub cpu_cores: usize,
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub simd_capabilities: SimdCapabilities,
}

/// Benchmark runner for comparing different implementations
pub struct BenchmarkRunner {
    profiler: Profiler,
    results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
            results: HashMap::new(),
        }
    }

    /// Run a benchmark and record results
    pub fn benchmark<F, R>(&mut self, name: &str, iterations: usize, mut func: F) -> BenchmarkResult
    where
        F: FnMut() -> R,
    {
        let mut times = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            let _result = func();
            times.push(start.elapsed());
        }

        times.sort();
        let total: Duration = times.iter().sum();
        let mean = total / iterations as u32;
        let median = times[iterations / 2];
        let min = times[0];
        let max = times[iterations - 1];

        let result = BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_time: total,
            mean_time: mean,
            median_time: median,
            min_time: min,
            max_time: max,
            throughput: iterations as f64 / total.as_secs_f64(),
        };

        self.results.insert(name.to_string(), result.clone());
        result
    }

    /// Compare two benchmark results
    pub fn compare(&self, name1: &str, name2: &str) -> Option<BenchmarkComparison> {
        let result1 = self.results.get(name1)?;
        let result2 = self.results.get(name2)?;

        let speedup = result1.mean_time.as_secs_f64() / result2.mean_time.as_secs_f64();
        let throughput_ratio = result2.throughput / result1.throughput;

        Some(BenchmarkComparison {
            baseline: result1.clone(),
            optimized: result2.clone(),
            speedup,
            throughput_ratio,
        })
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &HashMap<String, BenchmarkResult> {
        &self.results
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput: f64, // operations per second
}

/// Comparison between two benchmark results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BenchmarkComparison {
    pub baseline: BenchmarkResult,
    pub optimized: BenchmarkResult,
    pub speedup: f64,
    pub throughput_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_timing() {
        let profiler = Profiler::new();
        
        {
            let _timer = profiler.start_timer("test_operation");
            thread::sleep(Duration::from_millis(10));
        }

        let stats = profiler.get_timing_stats("test_operation").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_counters() {
        let profiler = Profiler::new();
        
        profiler.increment_counter("test_op");
        profiler.increment_counter("test_op");
        profiler.increment_counter("test_op");

        let report = profiler.get_performance_report();
        assert_eq!(report.operation_counts.get("test_op"), Some(&3));
    }

    #[test]
    fn test_benchmark_runner() {
        let mut runner = BenchmarkRunner::new();
        
        let result = runner.benchmark("simple_add", 1000, || {
            let a = 1 + 1;
            a
        });

        assert_eq!(result.iterations, 1000);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_profile_macro() {
        let profiler = Profiler::new();
        
        profile!(profiler, "macro_test", {
            thread::sleep(Duration::from_millis(1));
        });

        let stats = profiler.get_timing_stats("macro_test").unwrap();
        assert_eq!(stats.count, 1);

        let report = profiler.get_performance_report();
        assert_eq!(report.operation_counts.get("macro_test"), Some(&1));
    }
}