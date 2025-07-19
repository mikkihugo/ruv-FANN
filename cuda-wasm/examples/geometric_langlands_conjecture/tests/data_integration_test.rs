//! Integration tests for the data processing pipeline

use std::path::PathBuf;
use tempfile::TempDir;
use tokio;

use geometric_langlands_conjecture::data::*;
use geometric_langlands_conjecture::data::generators::*;
use geometric_langlands_conjecture::data::features::*;
use geometric_langlands_conjecture::data::encoders::*;

#[tokio::test]
async fn test_full_data_pipeline() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    
    // Configure the data pipeline
    let config = DataConfig {
        cache_size_mb: 100,
        parallel_processing: true,
        num_workers: 2,
        batch_size: 32,
        enable_compression: true,
        cache_dir: temp_dir.path().join("cache").to_string_lossy().to_string(),
        feature_config: FeatureConfig {
            geometric_dim: 64,
            algebraic_dim: 32,
            enable_spectral: true,
            enable_topological: true,
            precision: 1e-10,
            max_degree: 8,
        },
    };
    
    // Initialize data manager
    let mut data_manager = DataManager::new(config.clone()).await
        .expect("Failed to create data manager");
    
    // Generate synthetic data
    let generator_config = GeneratorConfig {
        seed: 42,
        num_samples: 100,
        complexity: ComplexityLevel::Medium,
        add_noise: true,
        noise_level: 0.05,
    };
    
    let data_generator = GeometricLanglandsDataGenerator::new(generator_config);
    
    // Generate Riemann surfaces
    let riemann_gen = RiemannSurfaceGenerator::new(ComplexityLevel::Medium);
    let riemann_surfaces = riemann_gen.generate_batch(&GeneratorConfig {
        num_samples: 50,
        ..Default::default()
    });
    
    // Process the surfaces through the pipeline
    let processed_features = data_manager.process_batch(riemann_surfaces).await
        .expect("Failed to process batch");
    
    assert_eq!(processed_features.len(), 50);
    
    // Verify feature dimensions
    for feature in &processed_features {
        assert_eq!(feature.geometric.len(), 64);
        assert_eq!(feature.algebraic.len(), 32);
        assert!(feature.spectral.is_some());
        assert!(feature.topological.is_some());
    }
    
    // Test serialization
    let output_path = temp_dir.path().join("test_features.bin");
    serialization::save_dataset(&processed_features, &output_path).await
        .expect("Failed to save dataset");
    
    let loaded_features = serialization::load_dataset(&output_path).await
        .expect("Failed to load dataset");
    
    assert_eq!(loaded_features.len(), processed_features.len());
    
    // Test encoding
    let mut encoder = MultiModalEncoder::new(
        EncodingStrategy::Normalized,
        EncodingStrategy::Standardized,
        true,
        true,
    );
    
    encoder.fit(&processed_features).expect("Failed to fit encoder");
    let encoded_batch = encoder.encode_batch_concatenated(&processed_features)
        .expect("Failed to encode batch");
    
    assert_eq!(encoded_batch.nrows(), processed_features.len());
    println!("Encoded feature dimension: {}", encoded_batch.ncols());
    
    // Test cache functionality
    let cache_stats = data_manager.cache_stats();
    if let Some(stats) = cache_stats {
        println!("Cache statistics: hit_rate={:.2}%, total_items={}", 
                 stats.hit_rate() * 100.0, stats.total_items);
    }
    
    println!("Full data pipeline test completed successfully!");
}

#[tokio::test]
async fn test_correspondence_pairs_generation() {
    let generator_config = GeneratorConfig {
        seed: 123,
        num_samples: 20,
        complexity: ComplexityLevel::Simple,
        add_noise: false,
        noise_level: 0.0,
    };
    
    let data_generator = GeometricLanglandsDataGenerator::new(generator_config);
    let correspondence_pairs = data_generator.generate_correspondence_pairs();
    
    assert_eq!(correspondence_pairs.len(), 20);
    
    // Verify that pairs are properly generated
    for (_left, _right) in correspondence_pairs {
        // Each pair should contain one object from each side of the correspondence
        // In a real implementation, we would verify the mathematical relationship
    }
    
    println!("Correspondence pairs generation test completed!");
}

#[tokio::test]
async fn test_streaming_pipeline() {
    use futures::stream;
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config = DataConfig {
        cache_size_mb: 50,
        batch_size: 10,
        cache_dir: temp_dir.path().join("cache").to_string_lossy().to_string(),
        ..Default::default()
    };
    
    let pipeline = pipeline::DataPipeline::new(&config)
        .expect("Failed to create pipeline");
    
    let streaming_pipeline = pipeline::StreamingPipeline::new(pipeline, 50);
    
    // Generate objects to stream
    let riemann_gen = RiemannSurfaceGenerator::new(ComplexityLevel::Simple);
    let surfaces: Vec<_> = (0..25)
        .map(|i| {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(i);
            riemann_gen.generate_one(&mut rng)
        })
        .collect();
    
    // Create a stream
    let object_stream = stream::iter(surfaces);
    
    // Process the stream
    let mut result_receiver = streaming_pipeline.process_stream(object_stream).await
        .expect("Failed to process stream");
    
    let mut results = Vec::new();
    while let Some(feature) = result_receiver.recv().await {
        results.push(feature);
    }
    
    assert!(!results.is_empty());
    println!("Streaming pipeline processed {} features", results.len());
}

#[tokio::test]
async fn test_feature_extraction_performance() {
    use std::time::Instant;
    
    let config = FeatureConfig {
        geometric_dim: 128,
        algebraic_dim: 64,
        enable_spectral: true,
        enable_topological: true,
        precision: 1e-12,
        max_degree: 10,
    };
    
    let extractor = FeatureExtractor::new(config);
    
    // Generate test objects
    let bundle_gen = VectorBundleGenerator::new(ComplexityLevel::Complex);
    let bundles: Vec<_> = (0..100)
        .map(|i| {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(i);
            bundle_gen.generate_one(&mut rng)
        })
        .collect();
    
    // Measure extraction time
    let start = Instant::now();
    let features = extractor.extract_batch(&bundles)
        .expect("Failed to extract features");
    let duration = start.elapsed();
    
    assert_eq!(features.len(), 100);
    
    let avg_time_per_object = duration.as_millis() as f64 / 100.0;
    println!("Feature extraction: {:.2}ms per object", avg_time_per_object);
    
    // Verify feature quality
    for feature in &features {
        assert_eq!(feature.geometric.len(), 128);
        assert_eq!(feature.algebraic.len(), 64);
        
        // Check for reasonable feature values (not all zeros)
        let geo_sum: f64 = feature.geometric.iter().sum();
        let alg_sum: f64 = feature.algebraic.iter().sum();
        
        assert!(geo_sum.abs() > 1e-6, "Geometric features are too small");
        assert!(alg_sum.abs() > 1e-6, "Algebraic features are too small");
    }
    
    println!("Feature extraction performance test completed!");
}

#[tokio::test]
async fn test_data_augmentation() {
    let mut feature = FeatureVector::new("test".to_string(), 8, 4);
    feature.geometric = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    feature.algebraic = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
    
    let original_geo = feature.geometric.clone();
    let original_alg = feature.algebraic.clone();
    
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    
    // Test noise addition
    generators::augmentation::add_gaussian_noise(&mut feature, 0.1, &mut rng);
    
    // Verify that features have changed
    let geo_changed = feature.geometric.iter()
        .zip(original_geo.iter())
        .any(|(new, old)| (new - old).abs() > 1e-6);
    
    assert!(geo_changed, "Gaussian noise should have changed geometric features");
    
    // Test rotation
    let mut feature2 = FeatureVector::new("test2".to_string(), 4, 2);
    feature2.geometric = ndarray::arr1(&[1.0, 0.0, 0.0, 1.0]);
    
    generators::augmentation::random_rotation(&mut feature2, &mut rng);
    
    // After rotation, the norm should be preserved (approximately)
    let original_norm = 1.0_f64.hypot(1.0);
    let new_norm = feature2.geometric[0].hypot(feature2.geometric[1]);
    
    assert!((original_norm - new_norm).abs() < 1e-10, 
            "Rotation should preserve norm");
    
    println!("Data augmentation test completed!");
}

#[test]
fn test_complexity_levels() {
    // Test dimension ranges for different complexity levels
    let simple = ComplexityLevel::Simple;
    let medium = ComplexityLevel::Medium;
    let complex = ComplexityLevel::Complex;
    
    let (simple_min, simple_max) = simple.dimensions();
    let (medium_min, medium_max) = medium.dimensions();
    let (complex_min, complex_max) = complex.dimensions();
    
    assert!(simple_max <= medium_min);
    assert!(medium_max <= complex_min);
    
    // Test degree ranges
    let (simple_deg_min, simple_deg_max) = simple.degree_range();
    let (complex_deg_min, complex_deg_max) = complex.degree_range();
    
    assert!(simple_deg_max <= complex_deg_min);
    
    println!("Complexity levels test completed!");
}