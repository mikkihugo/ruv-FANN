//! Comprehensive optimization comparison benchmarks
//!
//! This benchmark suite compares different optimization levels:
//! 1. Baseline implementation (no optimizations)
//! 2. SIMD-optimized operations
//! 3. Parallel processing
//! 4. Memory pooling
//! 5. Combined optimizations
//! 6. nalgebra-based operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruv_fann::*;
use ruv_fann::training::*;
use ruv_fann::profiling::*;

#[cfg(feature = "simd")]
use ruv_fann::simd::*;

#[cfg(feature = "nalgebra")]
use ruv_fann::linalg::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::time::Instant;

/// Benchmark configuration for optimization comparison
struct BenchConfig {
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
    batch_size: usize,
    samples: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            input_size: 784,
            hidden_sizes: vec![256, 128, 64],
            output_size: 10,
            batch_size: 32,
            samples: 1000,
        }
    }
}

/// Benchmark forward propagation with different optimization levels
fn bench_forward_propagation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_propagation_comparison");
    let config = BenchConfig::default();
    
    // Create test data
    let network = create_standard_network(&config);
    let inputs = create_test_inputs(&config);
    
    // Baseline: Standard implementation
    group.bench_function("baseline", |b| {
        b.iter(|| {
            for input in &inputs {
                let _output = network.run(black_box(input)).unwrap();
            }
        })
    });
    
    // SIMD optimized
    #[cfg(feature = "simd")]
    group.bench_function("simd_optimized", |b| {
        b.iter(|| {
            for input in &inputs {
                let mut output = vec![0.0f32; config.output_size];
                // Convert input to SIMD-friendly format and process
                AdaptiveSimd::sigmoid_f32(black_box(input), black_box(&mut output));
            }
        })
    });
    
    // Parallel processing
    #[cfg(feature = "parallel")]
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let _outputs: Vec<_> = inputs
                .par_iter()
                .map(|input| network.run(black_box(input)).unwrap())
                .collect();
        })
    });
    
    // nalgebra optimized
    #[cfg(feature = "nalgebra")]
    group.bench_function("nalgebra_optimized", |b| {
        let mut optimized_network = create_optimized_network(&config);
        let nalgebra_inputs = convert_to_nalgebra_inputs(&inputs);
        
        b.iter(|| {
            for input in &nalgebra_inputs {
                let _output = optimized_network.forward(black_box(input));
            }
        })
    });
    
    // Combined optimizations
    #[cfg(all(feature = "simd", feature = "parallel", feature = "nalgebra"))]
    group.bench_function("combined_optimizations", |b| {
        let mut optimized_network = create_optimized_network(&config);
        let nalgebra_inputs = convert_to_nalgebra_inputs(&inputs);
        
        b.iter(|| {
            let _outputs = optimized_network.batch_forward(black_box(&nalgebra_inputs));
        })
    });
    
    group.finish();
}

/// Benchmark matrix operations with different implementations
fn bench_matrix_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    
    let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];
    
    for (m, n) in sizes {
        // Create test matrices
        let a: Vec<f32> = (0..m*n).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..m*n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut c = vec![0.0f32; m*n];
        
        // Baseline scalar implementation
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", m, n)),
            &(m, n),
            |bench, &(rows, cols)| {
                bench.iter(|| {
                    for i in 0..rows {
                        for j in 0..cols {
                            let mut sum = 0.0f32;
                            for k in 0..cols {
                                sum += a[i * cols + k] * b[k * cols + j];
                            }
                            c[i * cols + j] = sum;
                        }
                    }
                    black_box(&c)
                })
            },
        );
        
        // SIMD implementation
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", m, n)),
            &(m, n),
            |bench, &(rows, cols)| {
                bench.iter(|| {
                    AdaptiveSimd::matmul_f32(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        rows,
                        cols,
                        cols,
                    );
                    black_box(&c)
                })
            },
        );
        
        // nalgebra implementation
        #[cfg(feature = "nalgebra")]
        group.bench_with_input(
            BenchmarkId::new("nalgebra", format!("{}x{}", m, n)),
            &(m, n),
            |bench, &(rows, cols)| {
                use nalgebra::DMatrix;
                let matrix_a = DMatrix::from_vec(rows, cols, a.clone());
                let matrix_b = DMatrix::from_vec(cols, rows, b.clone());
                
                bench.iter(|| {
                    let result = OptimizedLinAlg::matrix_matrix_mul(
                        black_box(&matrix_a),
                        black_box(&matrix_b)
                    );
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark training algorithms with optimizations
fn bench_training_algorithms_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_comparison");
    let config = BenchConfig {
        input_size: 100,
        hidden_sizes: vec![50, 25],
        output_size: 10,
        batch_size: 16,
        samples: 200,
    };
    
    let training_data = create_training_data(&config);
    
    // Standard training
    group.bench_function("standard_training", |b| {
        b.iter(|| {
            let mut network = create_standard_network(&config);
            let mut trainer = IncrementalBackprop::new(0.01);
            
            for _ in 0..5 { // 5 epochs for comparison
                let _error = trainer.train_epoch(black_box(&mut network), black_box(&training_data)).unwrap();
            }
            black_box(network)
        })
    });
    
    // Optimized training pipeline
    group.bench_function("optimized_training", |b| {
        b.iter(|| {
            let mut network = create_standard_network(&config);
            let training_config = optimized_pipeline::OptimizedTrainingConfigBuilder::new()
                .learning_rate(0.01)
                .batch_size(config.batch_size)
                .max_epochs(5)
                .use_simd(true)
                .use_parallel(true)
                .enable_profiling(false) // Disable for benchmarking
                .build();
            
            let mut trainer = optimized_pipeline::OptimizedTrainer::new(training_config);
            let _result = trainer.train(black_box(&mut network), black_box(&training_data)).unwrap();
            black_box(network)
        })
    });
    
    group.finish();
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for size in sizes {
        // Standard allocation
        group.bench_with_input(
            BenchmarkId::new("standard_allocation", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let buffer: Vec<f32> = vec![0.0; s];
                    black_box(buffer)
                })
            },
        );
        
        // Memory pool allocation
        group.bench_with_input(
            BenchmarkId::new("pool_allocation", size),
            &size,
            |b, &s| {
                let mut manager = memory_manager::MemoryManager::new();
                manager.create_pool("bench", s);
                
                b.iter(|| {
                    let buffer = manager.allocate("bench", s).unwrap();
                    manager.deallocate("bench", buffer).unwrap();
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark profiling overhead
fn bench_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_overhead");
    
    let iterations = 10000;
    
    // No profiling
    group.bench_function("no_profiling", |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let x = black_box(1.0f32 + 1.0f32);
                black_box(x);
            }
        })
    });
    
    // With profiling
    group.bench_function("with_profiling", |b| {
        let profiler = Profiler::new();
        
        b.iter(|| {
            for _ in 0..iterations {
                let _timer = profiler.start_timer("simple_operation");
                let x = black_box(1.0f32 + 1.0f32);
                black_box(x);
            }
        })
    });
    
    group.finish();
}

/// Performance scaling benchmark
fn bench_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_scaling");
    
    let network_sizes = vec![
        (10, vec![5], 2),
        (50, vec![25], 5),
        (100, vec![50, 25], 10),
        (500, vec![250, 125], 20),
        (1000, vec![500, 250], 50),
    ];
    
    for (input_size, hidden_sizes, output_size) in network_sizes {
        let config = BenchConfig {
            input_size,
            hidden_sizes,
            output_size,
            batch_size: 32,
            samples: 100,
        };
        
        let training_data = create_training_data(&config);
        let size_label = format!("{}_{:?}_{}", input_size, config.hidden_sizes, output_size);
        
        // Standard implementation
        group.bench_with_input(
            BenchmarkId::new("standard", &size_label),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let mut network = create_standard_network(cfg);
                    let mut trainer = IncrementalBackprop::new(0.01);
                    let _error = trainer.train_epoch(black_box(&mut network), black_box(&training_data)).unwrap();
                    black_box(network)
                })
            },
        );
        
        // Optimized implementation
        #[cfg(all(feature = "simd", feature = "parallel"))]
        group.bench_with_input(
            BenchmarkId::new("optimized", &size_label),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let mut network = create_standard_network(cfg);
                    let training_config = optimized_pipeline::OptimizedTrainingConfigBuilder::new()
                        .learning_rate(0.01)
                        .batch_size(cfg.batch_size)
                        .max_epochs(1)
                        .use_simd(true)
                        .use_parallel(true)
                        .enable_profiling(false)
                        .build();
                    
                    let mut trainer = optimized_pipeline::OptimizedTrainer::new(training_config);
                    let _result = trainer.train(black_box(&mut network), black_box(&training_data)).unwrap();
                    black_box(network)
                })
            },
        );
    }
    
    group.finish();
}

/// Utility functions

fn create_standard_network(config: &BenchConfig) -> Network<f32> {
    let mut builder = NetworkBuilder::new();
    builder.add_layer(config.input_size);
    
    for &size in &config.hidden_sizes {
        builder.add_layer(size);
    }
    
    builder.add_layer(config.output_size);
    builder.build().unwrap()
}

#[cfg(feature = "nalgebra")]
fn create_optimized_network(config: &BenchConfig) -> OptimizedNetwork<f32> {
    let mut sizes = vec![config.input_size];
    sizes.extend(&config.hidden_sizes);
    sizes.push(config.output_size);
    
    OptimizedNetwork::new(&sizes)
}

fn create_test_inputs(config: &BenchConfig) -> Vec<Vec<f32>> {
    (0..config.samples)
        .map(|_| {
            (0..config.input_size)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect()
}

#[cfg(feature = "nalgebra")]
fn convert_to_nalgebra_inputs(inputs: &[Vec<f32>]) -> Vec<nalgebra::DVector<f32>> {
    inputs
        .iter()
        .map(|input| nalgebra::DVector::from_vec(input.clone()))
        .collect()
}

fn create_training_data(config: &BenchConfig) -> TrainingData<f32> {
    let inputs = create_test_inputs(config);
    let outputs: Vec<Vec<f32>> = (0..config.samples)
        .map(|_| {
            (0..config.output_size)
                .map(|_| rand::random::<f32>())
                .collect()
        })
        .collect();
    
    TrainingData { inputs, outputs }
}

criterion_group!(
    optimization_benches,
    bench_forward_propagation_comparison,
    bench_matrix_operations_comparison,
    bench_training_algorithms_comparison,
    bench_memory_operations,
    bench_profiling_overhead,
    bench_performance_scaling,
);

criterion_main!(optimization_benches);