//! Comprehensive performance benchmarks for ruv-fann
//!
//! This benchmark suite measures performance across all major operations:
//! - Matrix operations (with and without SIMD)
//! - Training algorithms
//! - Memory management
//! - Parallel processing
//! - Geometric Langlands operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruv_fann::*;
use ruv_fann::training::*;
use ruv_fann::core::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use ruv_fann::simd::*;

/// Benchmark neural network forward propagation
fn bench_forward_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_propagation");
    
    // Test different network sizes
    let network_configs = vec![
        (784, vec![128, 64, 10]),      // Small MNIST-like
        (1024, vec![512, 256, 128, 10]), // Medium network
        (2048, vec![1024, 512, 256, 50]), // Large network
    ];
    
    for (input_size, hidden_layers) in network_configs {
        let network = create_test_network(input_size, &hidden_layers);
        let inputs: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01).collect();
        
        group.throughput(Throughput::Elements(input_size as u64));
        group.bench_with_input(
            BenchmarkId::new("forward_pass", format!("{}_{:?}", input_size, hidden_layers)),
            &(network, inputs),
            |b, (net, inp)| {
                b.iter(|| {
                    black_box(net.run(black_box(inp)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark training algorithms
fn bench_training_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_algorithms");
    
    let network_size = (100, vec![50, 25, 10]);
    let training_data = create_training_data(1000, network_size.0, network_size.2.last().unwrap().clone());
    
    // Benchmark different training algorithms
    let algorithms = vec![
        ("incremental_backprop", TrainingAlgorithm::IncrementalBackprop),
        ("batch_backprop", TrainingAlgorithm::BatchBackprop),
        ("rprop", TrainingAlgorithm::RProp),
        ("quickprop", TrainingAlgorithm::QuickProp),
    ];
    
    for (name, algorithm) in algorithms {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut network = create_test_network(network_size.0, &network_size.1);
                let mut trainer = create_trainer(algorithm);
                
                black_box(trainer.train_epoch(black_box(&mut network), black_box(&training_data)).unwrap())
            })
        });
    }
    
    group.finish();
}

/// Benchmark SIMD operations vs scalar
#[cfg(feature = "simd")]
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    let sizes = vec![128, 512, 1024, 4096];
    
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; size];
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark sigmoid computation
        group.bench_with_input(
            BenchmarkId::new("sigmoid_scalar", size),
            &data,
            |b, input| {
                b.iter(|| {
                    for (i, &x) in input.iter().enumerate() {
                        output[i] = black_box(1.0 / (1.0 + (-x).exp()));
                    }
                    black_box(&output)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid_simd", size),
            &data,
            |b, input| {
                b.iter(|| {
                    AdaptiveSimd::sigmoid_f32(black_box(input), black_box(&mut output));
                    black_box(&output)
                })
            },
        );
        
        // Benchmark matrix multiplication
        let matrix_size = (size as f64).sqrt() as usize;
        if matrix_size * matrix_size == size {
            let a = vec![1.0f32; matrix_size * matrix_size];
            let b = vec![1.0f32; matrix_size * matrix_size];
            let mut c = vec![0.0f32; matrix_size * matrix_size];
            
            group.bench_with_input(
                BenchmarkId::new("matmul_simd", size),
                &(a, b),
                |bench, (matrix_a, matrix_b)| {
                    bench.iter(|| {
                        AdaptiveSimd::matmul_f32(
                            black_box(matrix_a),
                            black_box(matrix_b),
                            black_box(&mut c),
                            matrix_size,
                            matrix_size,
                            matrix_size,
                        );
                        black_box(&c)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential training
#[cfg(feature = "parallel")]
fn bench_parallel_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_training");
    
    let network_size = (500, vec![200, 100, 50]);
    let batch_sizes = vec![32, 64, 128, 256];
    
    for batch_size in batch_sizes {
        let training_data = create_training_data(batch_size, network_size.0, network_size.2.last().unwrap().clone());
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // Sequential processing
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &training_data,
            |b, data| {
                b.iter(|| {
                    let mut network = create_test_network(network_size.0, &network_size.1);
                    for i in 0..data.inputs.len() {
                        let _ = network.run(black_box(&data.inputs[i]));
                    }
                    black_box(&network)
                })
            },
        );
        
        // Parallel processing
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &training_data,
            |b, data| {
                b.iter(|| {
                    let network = create_test_network(network_size.0, &network_size.1);
                    let results: Vec<_> = data.inputs
                        .par_iter()
                        .map(|input| network.run(black_box(input)).unwrap())
                        .collect();
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory management operations
fn bench_memory_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_management");
    
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Standard allocation
        group.bench_with_input(
            BenchmarkId::new("std_allocation", size),
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
                b.iter(|| {
                    let mut manager = memory_manager::MemoryManager::new();
                    manager.create_pool("test", s);
                    let buffer = manager.allocate("test", s).unwrap();
                    black_box(buffer)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Geometric Langlands operations
fn bench_geometric_langlands(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_langlands");
    
    // Create test mathematical objects
    let base_curve = create_test_curve();
    let vector_space = create_test_vector_space(100);
    
    // Benchmark bundle operations
    group.bench_function("vector_bundle_creation", |b| {
        b.iter(|| {
            let bundle = VectorBundle::new(
                black_box("test_bundle".to_string()),
                black_box(Arc::new(base_curve.clone())),
                black_box(5), // rank
            );
            black_box(bundle)
        })
    });
    
    // Benchmark Hecke operator computation
    #[cfg(feature = "parallel")]
    group.bench_function("parallel_hecke_operator", |b| {
        let bundle = VectorBundle::new(
            "test".to_string(),
            Arc::new(base_curve.clone()),
            3,
        );
        let correspondence_points = vec![(0.0, 0.0); 100];
        let hecke_op = core::algorithms::ParallelHeckeOperator::new(
            Arc::new(base_curve.clone()),
            correspondence_points,
        );
        
        b.iter(|| {
            let result = hecke_op.apply_parallel(black_box(&bundle));
            black_box(result)
        })
    });
    
    // Benchmark cohomology computation
    #[cfg(feature = "parallel")]
    group.bench_function("parallel_cohomology", |b| {
        let sheaf = create_test_sheaf();
        let computer = core::algorithms::ParallelCohomologyComputer::new();
        
        b.iter(|| {
            let result = computer.compute_sheaf_cohomology(black_box(&sheaf), black_box(5));
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark cascade correlation
fn bench_cascade_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cascade_correlation");
    
    let input_size = 10;
    let output_size = 2;
    let training_data = create_training_data(500, input_size, output_size);
    
    group.bench_function("cascade_training", |b| {
        b.iter(|| {
            let config = cascade::CascadeConfig {
                max_hidden_neurons: 20,
                candidate_neurons: 8,
                max_epochs: 50,
                error_threshold: 0.01,
                activation_function: ActivationFunction::Sigmoid,
                candidate_activation_functions: vec![
                    ActivationFunction::Sigmoid,
                    ActivationFunction::Tanh,
                ],
                correlation_threshold: 0.4,
                quickprop_mu: 1.75,
                weight_decay: 0.0001,
            };
            
            let mut trainer = cascade::CascadeTrainer::new(config);
            let mut network = cascade::CascadeNetwork::new(input_size, output_size);
            
            let result = trainer.train(black_box(&mut network), black_box(&training_data), 10);
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark GPU operations (if available)
#[cfg(feature = "gpu")]
fn bench_gpu_operations(c: &mut Criterion) {
    if !training::is_gpu_available() {
        return; // Skip if GPU not available
    }
    
    let mut group = c.benchmark_group("gpu_operations");
    
    let network_size = (784, vec![256, 128, 10]);
    let batch_size = 128;
    let training_data = create_training_data(batch_size, network_size.0, network_size.2.last().unwrap().clone());
    
    // GPU vs CPU training comparison
    group.bench_function("cpu_batch_training", |b| {
        b.iter(|| {
            let mut network = create_test_network(network_size.0, &network_size.1);
            let mut trainer = BatchBackprop::new(0.01);
            
            let result = trainer.train_epoch(black_box(&mut network), black_box(&training_data));
            black_box(result)
        })
    });
    
    group.bench_function("gpu_batch_training", |b| {
        b.iter(|| {
            let mut network = create_test_network(network_size.0, &network_size.1);
            let mut trainer = training::GpuBatchBackprop::new(0.01).unwrap();
            
            let result = trainer.train_epoch(black_box(&mut network), black_box(&training_data));
            black_box(result)
        })
    });
    
    group.finish();
}

/// Utility functions for creating test data

fn create_test_network(input_size: usize, hidden_layers: &[usize]) -> Network<f32> {
    let mut builder = NetworkBuilder::new();
    builder.add_layer(input_size);
    
    for &size in hidden_layers {
        builder.add_layer(size);
    }
    
    builder.build().unwrap()
}

fn create_training_data(samples: usize, input_size: usize, output_size: usize) -> TrainingData<f32> {
    let inputs: Vec<Vec<f32>> = (0..samples)
        .map(|_| {
            (0..input_size)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    
    let outputs: Vec<Vec<f32>> = (0..samples)
        .map(|_| {
            (0..output_size)
                .map(|_| rand::random::<f32>())
                .collect()
        })
        .collect();
    
    TrainingData { inputs, outputs }
}

fn create_trainer(algorithm: TrainingAlgorithm) -> Box<dyn TrainingAlgorithmTrait<f32>> {
    match algorithm {
        TrainingAlgorithm::IncrementalBackprop => Box::new(IncrementalBackprop::new(0.01)),
        TrainingAlgorithm::BatchBackprop => Box::new(BatchBackprop::new(0.01)),
        TrainingAlgorithm::RProp => Box::new(Rprop::new()),
        TrainingAlgorithm::QuickProp => Box::new(Quickprop::new(0.01)),
        _ => Box::new(IncrementalBackprop::new(0.01)),
    }
}

fn create_test_curve() -> impl AlgebraicCurve {
    // Simplified test curve implementation
    TestCurve { genus: 1 }
}

fn create_test_vector_space(dimension: usize) -> impl VectorSpace {
    TestVectorSpace { dimension }
}

fn create_test_sheaf() -> Sheaf<TestSpace, TestSection, TestVectorSpace> {
    Sheaf::new(
        "test_sheaf".to_string(),
        TestSpace { dimension: 2 },
        TestSection { data: vec![0.0; 100] },
    )
}

// Simple test implementations for geometric objects
#[derive(Debug, Clone)]
struct TestCurve { genus: usize }

impl AlgebraicCurve for TestCurve {
    type Coordinate = (f64, f64);
    
    fn genus(&self) -> usize { self.genus }
    fn is_smooth(&self) -> bool { true }
    fn canonical_divisor(&self) -> Vec<Self::Coordinate> { vec![] }
}

impl GeometricObject for TestCurve {
    fn dimension(&self) -> usize { 1 }
    fn is_compact(&self) -> bool { true }
}

#[derive(Debug, Clone)]
struct TestVectorSpace { dimension: usize }

impl VectorSpace for TestVectorSpace {
    fn dimension(&self) -> usize { self.dimension }
    fn zero() -> Self { TestVectorSpace { dimension: 0 } }
    fn add(&self, _other: &Self) -> Self { self.clone() }
    fn scale(&self, _scalar: f64) -> Self { self.clone() }
}

impl MathObject for TestVectorSpace {
    fn id(&self) -> String { format!("test_vector_space_{}", self.dimension) }
    fn validate(&self) -> Result<(), MathError> { Ok(()) }
}

#[derive(Debug, Clone)]
struct TestSpace { dimension: usize }

impl TopologicalSpace for TestSpace {
    fn dimension(&self) -> usize { self.dimension }
    fn is_compact(&self) -> bool { true }
    fn euler_characteristic(&self) -> i64 { 0 }
}

impl MathObject for TestSpace {
    fn id(&self) -> String { format!("test_space_{}", self.dimension) }
    fn validate(&self) -> Result<(), MathError> { Ok(()) }
}

#[derive(Debug, Clone)]
struct TestSection { data: Vec<f64> }

impl MathObject for TestSection {
    fn id(&self) -> String { "test_section".to_string() }
    fn validate(&self) -> Result<(), MathError> { Ok(()) }
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_forward_propagation,
    bench_training_algorithms,
    bench_memory_management,
    bench_geometric_langlands,
    bench_cascade_correlation,
);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_simd_operations);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_training);

#[cfg(feature = "gpu")]
criterion_group!(gpu_benches, bench_gpu_operations);

// Main benchmark runner
criterion_main!(
    benches,
    #[cfg(feature = "simd")] simd_benches,
    #[cfg(feature = "parallel")] parallel_benches,
    #[cfg(feature = "gpu")] gpu_benches,
);