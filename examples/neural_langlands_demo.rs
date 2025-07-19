//! Demonstration of the neural-symbolic Langlands correspondence framework
//!
//! This example shows how to use the neural network components for learning
//! patterns in mathematical objects related to the Langlands program.

use ruv_fann::{
    core::{
        sheaf::SimpleSheaf,
        bundle::SimpleVectorBundle,
        representation::SimpleRepresentation,
    },
    neural::{
        LanglandsNet, NeuralConfig, FeatureExtractor,
        SheafFeatureExtractor, BundleFeatureExtractor, RepresentationFeatureExtractor,
        TrainingPipeline, CorrespondenceDataset,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural-Symbolic Langlands Correspondence Demo");
    println!("================================================");
    
    // Create neural network configuration
    let config = NeuralConfig {
        hidden_dims: vec![512, 256, 128],
        learning_rate: 0.001,
        batch_size: 16,
        epochs: 50,
        use_gpu: false, // CPU only for demo
        feature_dim: 256,
    };
    
    println!("📊 Configuration:");
    println!("  - Hidden layers: {:?}", config.hidden_dims);
    println!("  - Learning rate: {}", config.learning_rate);
    println!("  - Batch size: {}", config.batch_size);
    println!("  - Feature dimension: {}", config.feature_dim);
    
    // Create neural network
    let mut langlands_net = LanglandsNet::<f64>::new(config.clone())?;
    println!("\n✅ Created LanglandsNet with specialized architecture");
    
    // Create feature extractors
    let sheaf_extractor = SheafFeatureExtractor::new(config.feature_dim);
    let bundle_extractor = BundleFeatureExtractor::new(config.feature_dim);
    let rep_extractor = RepresentationFeatureExtractor::new(config.feature_dim, "GL(2)".to_string());
    
    println!("🔧 Initialized feature extractors for:");
    println!("  - Sheaves and local systems");
    println!("  - Vector bundles and Higgs bundles");
    println!("  - Group representations");
    
    // Create sample mathematical objects
    println!("\n📐 Creating sample mathematical objects...");
    
    // Sample sheaves
    let sheaf1 = SimpleSheaf::new("E1".to_string(), 2, 1); // rank 2 on curve
    let sheaf2 = SimpleSheaf::new("E2".to_string(), 3, 1); // rank 3 on curve
    
    // Sample vector bundles
    let bundle1 = SimpleVectorBundle::new("V1".to_string(), 2, 0, 1); // stable bundle
    let bundle2 = SimpleVectorBundle::new("V2".to_string(), 2, 2, 1); // positive degree
    
    // Sample representations
    let rep1 = SimpleRepresentation::new("ρ1".to_string(), 2, "GL(2)".to_string());
    let rep2 = SimpleRepresentation::new("ρ2".to_string(), 3, "GL(3)".to_string());
    
    println!("✅ Created {} sheaves, {} bundles, {} representations", 2, 2, 2);
    
    // Extract features
    println!("\n🧮 Extracting neural features...");
    
    let sheaf1_features = sheaf_extractor.extract(&sheaf1)?;
    let sheaf2_features = sheaf_extractor.extract(&sheaf2)?;
    
    let bundle1_features = bundle_extractor.extract(&bundle1)?;
    let bundle2_features = bundle_extractor.extract(&bundle2)?;
    
    let rep1_features = rep_extractor.extract(&rep1)?;
    let rep2_features = rep_extractor.extract(&rep2)?;
    
    println!("✅ Extracted feature vectors (dimension: {})", config.feature_dim);
    println!("  - Sheaf features: non-zero elements = {}", 
             sheaf1_features.iter().filter(|&&x| x != 0.0).count());
    println!("  - Bundle features: non-zero elements = {}", 
             bundle1_features.iter().filter(|&&x| x != 0.0).count());
    println!("  - Representation features: non-zero elements = {}", 
             rep1_features.iter().filter(|&&x| x != 0.0).count());
    
    // Test correspondence predictions
    println!("\n🔮 Testing correspondence predictions...");
    
    let prediction1 = langlands_net.predict_correspondence(&sheaf1_features, &rep1_features)?;
    let prediction2 = langlands_net.predict_correspondence(&bundle1_features, &rep2_features)?;
    
    println!("📈 Prediction 1 (Sheaf E1 ↔ Representation ρ1):");
    println!("  - Correspondence score: {:.4}", prediction1.correspondence_score);
    println!("  - Confidence: {:.4}", prediction1.confidence);
    println!("  - Verification scores: {:?}", 
             prediction1.verification_scores.iter().map(|&x| format!("{:.3}", x)).collect::<Vec<_>>());
    
    println!("📈 Prediction 2 (Bundle V1 ↔ Representation ρ2):");
    println!("  - Correspondence score: {:.4}", prediction2.correspondence_score);
    println!("  - Confidence: {:.4}", prediction2.confidence);
    
    // Create training dataset
    println!("\n📚 Creating training dataset...");
    
    let mut dataset = CorrespondenceDataset::new();
    
    // Add known correspondences (simplified for demo)
    dataset.add_sample(sheaf1_features.clone(), rep1_features.clone(), 1.0); // positive example
    dataset.add_sample(bundle1_features.clone(), rep1_features.clone(), 1.0); // positive example
    dataset.add_sample(sheaf2_features.clone(), rep2_features.clone(), 0.0); // negative example
    dataset.add_sample(bundle2_features.clone(), rep2_features.clone(), 0.0); // negative example
    
    println!("✅ Created training dataset with {} samples", dataset.len());
    
    // Split into train/validation
    let (train_size, val_size) = (3, 1);
    println!("📊 Dataset split: {} training, {} validation", train_size, val_size);
    
    // Demonstrate feature analysis
    println!("\n🔍 Feature Analysis:");
    analyze_feature_distribution(&sheaf1_features, "Sheaf E1");
    analyze_feature_distribution(&rep1_features, "Representation ρ1");
    
    // Display mathematical properties
    println!("\n📐 Mathematical Properties:");
    println!("Sheaf E1:");
    println!("  - Rank: {}", sheaf1.rank());
    println!("  - Dimension: {}", sheaf1.dimension());
    println!("  - Is valid: {}", sheaf1.is_valid());
    
    println!("Bundle V1:");
    println!("  - Rank: {}", bundle1.rank());
    println!("  - Degree: {}", bundle1.degree());
    println!("  - Slope: {:.3}", bundle1.slope());
    
    println!("Representation ρ1:");
    println!("  - Dimension: {}", rep1.dimension());
    println!("  - Is irreducible: {}", rep1.is_irreducible());
    println!("  - Is unitary: {}", rep1.is_unitary());
    
    println!("\n🎯 Demo completed successfully!");
    println!("The neural-symbolic bridge is ready for:");
    println!("  ✓ Feature extraction from mathematical objects");
    println!("  ✓ Correspondence prediction with confidence scores");
    println!("  ✓ Training on known mathematical relationships");
    println!("  ✓ Symbolic verification of neural predictions");
    
    Ok(())
}

/// Analyze the distribution of features in a vector
fn analyze_feature_distribution(features: &[f64], name: &str) {
    let non_zero_count = features.iter().filter(|&&x| x != 0.0).count();
    let mean = features.iter().sum::<f64>() / features.len() as f64;
    let max_val = features.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = features.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("  {}: {} non-zero, mean={:.3}, range=[{:.3}, {:.3}]", 
             name, non_zero_count, mean, min_val, max_val);
}