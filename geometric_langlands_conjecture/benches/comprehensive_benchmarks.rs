//! Comprehensive benchmarks for Geometric Langlands implementation
//!
//! This file contains extensive benchmarks covering all aspects of the
//! mathematical computations, from basic operations to full correspondence
//! pipelines. These benchmarks help identify performance bottlenecks and
//! track regression over time.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use geometric_langlands::prelude::*;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::time::Duration;

/// Benchmark basic mathematical operations
fn benchmark_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");
    
    // Matrix operations
    for size in [16, 32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*size * *size));
        
        let matrix_a = DMatrix::<f64>::identity(*size, *size);
        let matrix_b = DMatrix::<f64>::identity(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("matrix_multiplication", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _result = black_box(&matrix_a * &matrix_b);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_eigenvalues", size),
            size,
            |b, _| {
                b.iter(|| {
                    let symmetric = (&matrix_a + &matrix_a.transpose()) * 0.5;
                    let _eigenvals = black_box(symmetric.symmetric_eigenvalues());
                });
            },
        );
    }
    
    // Complex number operations
    for count in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*count));
        
        let complex_nums: Vec<Complex64> = (0..*count)
            .map(|i| Complex64::new(i as f64, (i + 1) as f64))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("complex_multiplication", count),
            count,
            |b, _| {
                b.iter(|| {
                    let _sum: Complex64 = black_box(
                        complex_nums.iter()
                            .zip(complex_nums.iter().skip(1))
                            .map(|(a, b)| a * b)
                            .sum()
                    );
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark group theory operations
fn benchmark_group_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Reductive group constructions
    for n in [2, 3, 4, 5].iter() {
        group.bench_with_input(
            BenchmarkId::new("gl_n_construction", n),
            n,
            |b, &n| {
                b.iter(|| {
                    let _g = black_box(ReductiveGroup::gl_n(n));
                });
            },
        );
    }
    
    // Group operations
    let g = ReductiveGroup::gl_n(3);
    group.bench_function("group_element_multiplication", |b| {
        b.iter(|| {
            let elem1 = g.random_element();
            let elem2 = g.random_element();
            let _product = black_box(elem1.multiply(&elem2));
        });
    });
    
    group.bench_function("group_element_inverse", |b| {
        b.iter(|| {
            let elem = g.random_element();
            let _inverse = black_box(elem.inverse());
        });
    });
    
    group.finish();
}

/// Benchmark automorphic forms
fn benchmark_automorphic_forms(c: &mut Criterion) {
    let mut group = c.benchmark_group("automorphic_forms");
    group.measurement_time(Duration::from_secs(15));
    
    let g = ReductiveGroup::gl_n(2);
    
    // Eisenstein series construction
    for weight in [2, 4, 6, 8, 10, 12].iter() {
        group.bench_with_input(
            BenchmarkId::new("eisenstein_series", weight),
            weight,
            |b, &weight| {
                b.iter(|| {
                    let _form = black_box(AutomorphicForm::eisenstein_series(&g, weight));
                });
            },
        );
    }
    
    // Hecke operator applications
    let form = AutomorphicForm::eisenstein_series(&g, 4);
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23].iter() {
        group.bench_with_input(
            BenchmarkId::new("hecke_operator", prime),
            prime,
            |b, &prime| {
                b.iter(|| {
                    let hecke = HeckeOperator::new(&g, prime);
                    let _result = black_box(hecke.apply(&form));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Galois representations
fn benchmark_galois_representations(c: &mut Criterion) {
    let mut group = c.benchmark_group("galois_representations");
    group.measurement_time(Duration::from_secs(10));
    
    // Galois representation construction
    for prime in [7, 11, 13, 17, 19].iter() {
        group.bench_with_input(
            BenchmarkId::new("galois_rep_construction", prime),
            prime,
            |b, &prime| {
                b.iter(|| {
                    let _rep = black_box(GaloisRepresentation::from_prime(prime));
                });
            },
        );
    }
    
    // Character computations
    let rep = GaloisRepresentation::from_prime(11);
    group.bench_function("character_computation", |b| {
        b.iter(|| {
            let frobenius = rep.frobenius_element();
            let _character = black_box(rep.character(&frobenius));
        });
    });
    
    group.finish();
}

/// Benchmark L-functions
fn benchmark_l_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("l_functions");
    group.measurement_time(Duration::from_secs(20));
    
    let l_function = LFunction::riemann_zeta();
    
    // L-function evaluation at multiple points
    let evaluation_points: Vec<Complex64> = (1..100)
        .map(|i| Complex64::new(1.5, i as f64 * 0.1))
        .collect();
    
    group.throughput(Throughput::Elements(evaluation_points.len() as u64));
    group.bench_function("l_function_evaluation_batch", |b| {
        b.iter(|| {
            let _values: Vec<Complex64> = black_box(
                evaluation_points.iter()
                    .map(|&s| l_function.evaluate(s))
                    .collect()
            );
        });
    });
    
    // Single point evaluation with different precisions
    let s = Complex64::new(0.5, 14.134725); // Known zero
    for precision in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("l_function_precision", precision),
            precision,
            |b, &precision| {
                b.iter(|| {
                    let _value = black_box(l_function.evaluate_with_precision(s, precision));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark spectral analysis
fn benchmark_spectral_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_analysis");
    group.measurement_time(Duration::from_secs(15));
    
    // Eigenvalue computations for various matrix sizes
    for size in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*size * *size));
        
        let matrix = DMatrix::<f64>::from_fn(*size, *size, |i, j| {
            if i == j { 2.0 }
            else if (i as isize - j as isize).abs() == 1 { -1.0 }
            else { 0.0 }
        });
        
        group.bench_with_input(
            BenchmarkId::new("eigenvalues_symmetric", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _eigenvals = black_box(matrix.symmetric_eigenvalues());
                });
            },
        );
    }
    
    // Spectral sequence computations
    for page_limit in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("spectral_sequence", page_limit),
            page_limit,
            |b, &page_limit| {
                b.iter(|| {
                    let mut sequence = SpectralSequence::new(2, 3);
                    for p in 0..10 {
                        for q in 0..10 {
                            sequence.add_term(p, q, Complex64::new((p + q) as f64, 0.0));
                        }
                    }
                    
                    for _ in 2..=page_limit {
                        sequence.compute_next_page();
                    }
                    
                    black_box(sequence);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark harmonic analysis
fn benchmark_harmonic_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("harmonic_analysis");
    group.measurement_time(Duration::from_secs(10));
    
    // Fourier transforms
    for size in [64, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*size));
        
        let data: Vec<Complex64> = (0..*size)
            .map(|i| Complex64::new((i as f64 * 2.0 * std::f64::consts::PI / *size as f64).sin(), 0.0))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("fft", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _fft_result = black_box(naive_fft(&data));
                });
            },
        );
    }
    
    // Spherical function computations
    let g = ReductiveGroup::gl_n(2);
    let k = g.maximal_compact_subgroup();
    
    group.bench_function("spherical_function_evaluation", |b| {
        b.iter(|| {
            let spherical_fn = SphericalFunction::elementary(&g, &k, 0);
            let test_elements: Vec<_> = (0..100).map(|_| g.random_element()).collect();
            
            let _values: Vec<f64> = black_box(
                test_elements.iter()
                    .map(|elem| spherical_fn.evaluate(elem))
                    .collect()
            );
        });
    });
    
    group.finish();
}

/// Benchmark category theory operations
fn benchmark_category_theory(c: &mut Criterion) {
    let mut group = c.benchmark_group("category_theory");
    group.measurement_time(Duration::from_secs(10));
    
    // Sheaf operations
    let variety = AlgebraicVariety::projective_space(2);
    
    group.bench_function("sheaf_cohomology", |b| {
        b.iter(|| {
            let sheaf = Sheaf::structure_sheaf(&variety);
            let _cohomology = black_box(variety.cohomology(&sheaf));
        });
    });
    
    // Derived category operations
    group.bench_function("derived_functor", |b| {
        b.iter(|| {
            let functor = Functor::global_sections();
            let sheaf = Sheaf::line_bundle(&variety, 2);
            let _derived = black_box(functor.derived_functor(&sheaf));
        });
    });
    
    group.finish();
}

/// Benchmark neural network operations
fn benchmark_neural_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_networks");
    group.measurement_time(Duration::from_secs(15));
    
    // Forward propagation for different network sizes
    let network_configs = vec![
        (64, vec![128], 32),
        (128, vec![256, 128], 64),
        (256, vec![512, 512, 256], 128),
    ];
    
    for (input_size, hidden_sizes, output_size) in network_configs {
        let network = NeuralNetwork::new(input_size, &hidden_sizes, output_size);
        let input = DVector::<f64>::from_fn(input_size, |i, _| i as f64 * 0.01);
        
        group.bench_function(&format!("forward_{}_{}_{}",
                                     input_size, hidden_sizes.len(), output_size), |b| {
            b.iter(|| {
                let _output = black_box(network.forward(&input));
            });
        });
    }
    
    // Batch processing
    let network = NeuralNetwork::new(128, &[256, 256], 64);
    for batch_size in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size));
        
        let batch_input = DMatrix::<f64>::from_fn(*batch_size, 128, |i, j| (i + j) as f64 * 0.01);
        
        group.bench_with_input(
            BenchmarkId::new("batch_forward", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let _output = black_box(network.forward_batch(&batch_input));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark full Langlands correspondence pipeline
fn benchmark_langlands_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("langlands_pipeline");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10); // Fewer samples for expensive operations
    
    // Complete correspondence for different parameters
    for weight in [4, 6, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("full_correspondence", weight),
            weight,
            |b, &weight| {
                b.iter(|| {
                    // Step 1: Create automorphic form
                    let g = ReductiveGroup::gl_n(2);
                    let form = AutomorphicForm::eisenstein_series(&g, weight);
                    
                    // Step 2: Compute Hecke eigenvalues
                    let primes = vec![2, 3, 5, 7, 11];
                    let mut eigenvalues = Vec::new();
                    
                    for &p in &primes {
                        let hecke = HeckeOperator::new(&g, p);
                        let eigenvalue = hecke.eigenvalue(&form);
                        eigenvalues.push((p, eigenvalue));
                    }
                    
                    // Step 3: Construct Galois representation
                    let galois_rep = GaloisRepresentation::from_automorphic_form(&form);
                    
                    // Step 4: Verify correspondence
                    for (p, expected_eigenvalue) in eigenvalues {
                        let actual_trace = galois_rep.trace_of_frobenius_at_prime(p);
                        // In a real implementation, we'd verify these match
                        black_box((expected_eigenvalue, actual_trace));
                    }
                    
                    black_box((form, galois_rep));
                });
            },
        );
    }
    
    // Multi-prime consistency
    group.bench_function("multi_prime_consistency", |b| {
        b.iter(|| {
            let g = ReductiveGroup::gl_n(2);
            let form = AutomorphicForm::eisenstein_series(&g, 6);
            let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
            
            let mut l_function_data = Vec::new();
            
            for &p in &primes {
                let hecke = HeckeOperator::new(&g, p);
                let eigenvalue = hecke.eigenvalue(&form);
                l_function_data.push((p, eigenvalue));
            }
            
            // Construct L-function from eigenvalues
            let l_function = LFunction::from_eigenvalues(&l_function_data);
            
            // Test functional equation
            let s = Complex64::new(0.5, 1.0);
            let l_s = l_function.evaluate(s);
            let l_dual = l_function.evaluate(Complex64::new(1.0, 0.0) - s);
            
            black_box((l_s, l_dual));
        });
    });
    
    group.finish();
}

/// Benchmark memory usage and scalability
fn benchmark_memory_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_performance");
    group.measurement_time(Duration::from_secs(10));
    
    // Large matrix operations
    for size in [512, 1024].iter() {
        group.throughput(Throughput::Bytes((*size * *size * 8) as u64)); // 8 bytes per f64
        
        group.bench_with_input(
            BenchmarkId::new("large_matrix_ops", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let matrix = DMatrix::<f64>::identity(size, size);
                    let _product = black_box(&matrix * &matrix);
                });
            },
        );
    }
    
    // Memory allocation patterns
    group.bench_function("repeated_allocations", |b| {
        b.iter(|| {
            for i in 1..=100 {
                let matrix = DMatrix::<f64>::zeros(i, i);
                black_box(matrix);
            }
        });
    });
    
    group.finish();
}

/// Benchmark parallel operations
fn benchmark_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");
    group.measurement_time(Duration::from_secs(15));
    
    // Parallel matrix operations
    group.bench_function("parallel_hecke_computation", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            
            let g = ReductiveGroup::gl_n(2);
            let form = AutomorphicForm::eisenstein_series(&g, 4);
            let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
            
            let eigenvalues: Vec<f64> = black_box(
                primes.par_iter()
                    .map(|&p| {
                        let hecke = HeckeOperator::new(&g, p);
                        hecke.eigenvalue(&form)
                    })
                    .collect()
            );
            
            black_box(eigenvalues);
        });
    });
    
    group.finish();
}

/// Naive FFT implementation for benchmarking
fn naive_fft(data: &[Complex64]) -> Vec<Complex64> {
    let n = data.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let w = Complex64::new(angle.cos(), angle.sin());
            result[k] += data[j] * w;
        }
    }
    
    result
}

/// Benchmark configuration and grouping
criterion_group!(
    name = comprehensive_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets = 
        benchmark_basic_operations,
        benchmark_group_operations,
        benchmark_automorphic_forms,
        benchmark_galois_representations,
        benchmark_l_functions,
        benchmark_spectral_analysis,
        benchmark_harmonic_analysis,
        benchmark_category_theory,
        benchmark_neural_networks,
        benchmark_langlands_pipeline,
        benchmark_memory_performance,
        benchmark_parallel_operations
);

criterion_main!(comprehensive_benches);