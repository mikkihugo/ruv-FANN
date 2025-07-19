//! Feature extraction pipeline orchestration
//!
//! This module provides a complete pipeline that orchestrates feature extraction,
//! encoding, normalization, validation, and storage operations.

use crate::core::prelude::*;
use crate::features::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Pipeline configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineConfig {
    /// Extraction configuration
    pub extraction: ExtractorConfig,
    /// Encoding configuration
    pub encoding: EncodingConfig,
    /// Normalization type
    pub normalization: NormalizationType,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Enable parallel processing
    pub parallel: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable progress reporting
    pub progress_reporting: bool,
    /// Cache intermediate results
    pub cache_intermediates: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            extraction: ExtractorConfig::default(),
            encoding: EncodingConfig::default(),
            normalization: NormalizationType::L2,
            validation: ValidationConfig::default(),
            storage: StorageConfig::default(),
            parallel: true,
            batch_size: 32,
            progress_reporting: true,
            cache_intermediates: true,
        }
    }
}

/// Pipeline processing statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineStats {
    /// Total objects processed
    pub total_processed: usize,
    /// Successfully processed
    pub successful: usize,
    /// Failed processing
    pub failed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Average processing time per object
    pub avg_time_per_object_ms: f64,
    /// Validation success rate
    pub validation_success_rate: f64,
    /// Storage success rate
    pub storage_success_rate: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self {
            total_processed: 0,
            successful: 0,
            failed: 0,
            processing_time_ms: 0,
            avg_time_per_object_ms: 0.0,
            validation_success_rate: 0.0,
            storage_success_rate: 0.0,
            memory_usage_bytes: 0,
        }
    }
    
    pub fn finalize(&mut self) {
        if self.total_processed > 0 {
            self.avg_time_per_object_ms = self.processing_time_ms as f64 / self.total_processed as f64;
            self.validation_success_rate = self.successful as f64 / self.total_processed as f64;
            self.storage_success_rate = self.successful as f64 / self.total_processed as f64;
        }
    }
}

/// Progress callback for pipeline processing
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Main feature extraction pipeline
#[derive(Debug)]
pub struct FeaturePipeline {
    config: PipelineConfig,
    extractor: UniversalExtractor,
    encoder: UniversalEncoder,
    normalizer: FeatureNormalizer,
    validator: CompositeValidator,
    storage: Option<UniversalStorage>,
    stats: Arc<Mutex<PipelineStats>>,
    progress_callback: Option<Arc<ProgressCallback>>,
}

impl FeaturePipeline {
    /// Create a new pipeline with default configuration
    pub fn new() -> FeatureResult<Self> {
        Self::with_config(PipelineConfig::default())
    }
    
    /// Create a new pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> FeatureResult<Self> {
        let storage = if config.storage.backend != StorageBackend::Memory || config.cache_intermediates {
            Some(UniversalStorage::new(config.storage.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            extractor: UniversalExtractor::default(),
            encoder: UniversalEncoder::default(),
            normalizer: FeatureNormalizer::default(),
            validator: CompositeValidator::default(),
            storage,
            stats: Arc::new(Mutex::new(PipelineStats::new())),
            progress_callback: None,
        })
    }
    
    /// Set a progress callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, usize, &str) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(Box::new(callback)));
        self
    }
    
    /// Process a single mathematical object through the complete pipeline
    pub fn process_object(
        &mut self,
        object: &dyn std::any::Any,
        object_id: Option<String>,
    ) -> FeatureResult<FeatureVector> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Extract features
        let raw_features = self.extractor.extract_from_object(object, &self.config.extraction)
            .map_err(|e| {
                self.update_stats_failure();
                e
            })?;
        
        // Step 2: Encode features
        let encoded_features = self.encoder.encode_object(object, &self.config.encoding)
            .map_err(|e| {
                self.update_stats_failure();
                e
            })?;
        
        // Step 3: Normalize features
        let normalized_features = if self.normalizer.fitted {
            self.normalizer.transform(&encoded_features)?
        } else {
            // For single object processing, we can't normalize without a fitted normalizer
            encoded_features
        };
        
        // Step 4: Validate features
        let validation_result = self.validator.feature_validator.validate(
            &normalized_features,
            &self.config.validation,
        );
        
        if !validation_result.is_valid {
            self.update_stats_failure();
            return Err(FeatureError::ValidationFailed {
                message: format!("Validation failed: {:?}", validation_result.errors),
            });
        }
        
        // Step 5: Store if requested
        if let (Some(storage), Some(id)) = (&mut self.storage, object_id) {
            storage.store(&id, &normalized_features)
                .map_err(|e| {
                    self.update_stats_failure();
                    e
                })?;
        }
        
        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_stats_success(elapsed.as_millis() as u64);
        
        Ok(normalized_features)
    }
    
    /// Process multiple objects in batch
    pub fn process_batch(
        &mut self,
        objects: &[(Box<dyn std::any::Any>, Option<String>)],
    ) -> FeatureResult<Vec<FeatureVector>> {
        if objects.is_empty() {
            return Ok(Vec::new());
        }
        
        let total_objects = objects.len();
        let mut results = Vec::with_capacity(total_objects);
        let mut extracted_features = Vec::new();
        let mut encoded_features = Vec::new();
        
        self.report_progress(0, total_objects, "Starting batch processing");
        
        // Step 1: Extract all features
        self.report_progress(0, total_objects, "Extracting features");
        for (i, (object, _)) in objects.iter().enumerate() {
            match self.extractor.extract_from_object(object.as_ref(), &self.config.extraction) {
                Ok(features) => extracted_features.push(features),
                Err(e) => {
                    self.update_stats_failure();
                    return Err(e);
                }
            }
            
            if i % 10 == 0 {
                self.report_progress(i, total_objects, "Extracting features");
            }
        }
        
        // Step 2: Encode all features
        self.report_progress(0, total_objects, "Encoding features");
        for (i, (object, _)) in objects.iter().enumerate() {
            match self.encoder.encode_object(object.as_ref(), &self.config.encoding) {
                Ok(features) => encoded_features.push(features),
                Err(e) => {
                    self.update_stats_failure();
                    return Err(e);
                }
            }
            
            if i % 10 == 0 {
                self.report_progress(i, total_objects, "Encoding features");
            }
        }
        
        // Step 3: Normalize all features
        self.report_progress(0, total_objects, "Normalizing features");
        let normalized_features = if !encoded_features.is_empty() {
            self.normalizer.fit_transform(&encoded_features, self.config.normalization)?
        } else {
            encoded_features
        };
        
        // Step 4: Validate all features
        self.report_progress(0, total_objects, "Validating features");
        let validation_results = self.validator.batch_validate_features(
            &normalized_features,
            &self.config.validation,
        );
        
        // Step 5: Store and collect results
        self.report_progress(0, total_objects, "Storing results");
        for (i, (features, validation_result)) in normalized_features.iter()
            .zip(validation_results.iter())
            .enumerate() {
            
            if !validation_result.is_valid {
                self.update_stats_failure();
                continue;
            }
            
            // Store if requested
            if let (Some(storage), Some(id)) = (&mut self.storage, &objects[i].1) {
                if let Err(_) = storage.store(id, features) {
                    self.update_stats_failure();
                    continue;
                }
            }
            
            results.push(features.clone());
            self.update_stats_success(0); // Time will be set later
            
            if i % 10 == 0 {
                self.report_progress(i, total_objects, "Storing results");
            }
        }
        
        self.report_progress(total_objects, total_objects, "Batch processing complete");
        
        Ok(results)
    }
    
    /// Process objects from a vector of any type that implements the required traits
    pub fn process_sheaves(
        &mut self,
        sheaves: &[Sheaf],
        ids: Option<&[String]>,
    ) -> FeatureResult<Vec<FeatureVector>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        for (i, sheaf) in sheaves.iter().enumerate() {
            let id = ids.and_then(|ids| ids.get(i)).map(|s| s.clone());
            
            match self.process_sheaf(sheaf, id) {
                Ok(features) => results.push(features),
                Err(e) => {
                    log::warn!("Failed to process sheaf {}: {:?}", i, e);
                    self.update_stats_failure();
                    continue;
                }
            }
            
            if self.config.progress_reporting && i % 10 == 0 {
                self.report_progress(i, sheaves.len(), "Processing sheaves");
            }
        }
        
        // Update total processing time
        let elapsed = start_time.elapsed();
        if let Ok(mut stats) = self.stats.lock() {
            stats.processing_time_ms += elapsed.as_millis() as u64;
        }
        
        Ok(results)
    }
    
    /// Process a single sheaf
    pub fn process_sheaf(
        &mut self,
        sheaf: &Sheaf,
        id: Option<String>,
    ) -> FeatureResult<FeatureVector> {
        let start_time = std::time::Instant::now();
        
        // Extract features using sheaf-specific extractor
        let extractor = SheafExtractor::default();
        let raw_features = extractor.extract(sheaf, &self.config.extraction)?;
        
        // Encode using sheaf-specific encoder
        let encoder = SheafEncoder::default();
        let encoded_features = encoder.encode(sheaf, &self.config.encoding)?;
        
        // Normalize if normalizer is fitted
        let normalized_features = if self.normalizer.fitted {
            self.normalizer.transform(&encoded_features)?
        } else {
            encoded_features
        };
        
        // Validate
        let validation_result = self.validator.feature_validator.validate(
            &normalized_features,
            &self.config.validation,
        );
        
        if !validation_result.is_valid {
            return Err(FeatureError::ValidationFailed {
                message: format!("Sheaf validation failed: {:?}", validation_result.errors),
            });
        }
        
        // Store if requested
        if let (Some(storage), Some(id)) = (&mut self.storage, id) {
            storage.store(&id, &normalized_features)?;
        }
        
        let elapsed = start_time.elapsed();
        self.update_stats_success(elapsed.as_millis() as u64);
        
        Ok(normalized_features)
    }
    
    /// Process bundles
    pub fn process_bundles(
        &mut self,
        bundles: &[Bundle],
        ids: Option<&[String]>,
    ) -> FeatureResult<Vec<FeatureVector>> {
        let mut results = Vec::new();
        
        for (i, bundle) in bundles.iter().enumerate() {
            let id = ids.and_then(|ids| ids.get(i)).map(|s| s.clone());
            
            match self.process_bundle(bundle, id) {
                Ok(features) => results.push(features),
                Err(e) => {
                    log::warn!("Failed to process bundle {}: {:?}", i, e);
                    self.update_stats_failure();
                    continue;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Process a single bundle
    pub fn process_bundle(
        &mut self,
        bundle: &Bundle,
        id: Option<String>,
    ) -> FeatureResult<FeatureVector> {
        let extractor = BundleExtractor::default();
        let encoder = BundleEncoder::default();
        
        let extracted = extractor.extract(bundle, &self.config.extraction)?;
        let encoded = encoder.encode(bundle, &self.config.encoding)?;
        
        let normalized = if self.normalizer.fitted {
            self.normalizer.transform(&encoded)?
        } else {
            encoded
        };
        
        let validation_result = self.validator.feature_validator.validate(
            &normalized,
            &self.config.validation,
        );
        
        if !validation_result.is_valid {
            return Err(FeatureError::ValidationFailed {
                message: format!("Bundle validation failed: {:?}", validation_result.errors),
            });
        }
        
        if let (Some(storage), Some(id)) = (&mut self.storage, id) {
            storage.store(&id, &normalized)?;
        }
        
        self.update_stats_success(0);
        Ok(normalized)
    }
    
    /// Process representations
    pub fn process_representations(
        &mut self,
        representations: &[Representation],
        ids: Option<&[String]>,
    ) -> FeatureResult<Vec<FeatureVector>> {
        let mut results = Vec::new();
        
        for (i, rep) in representations.iter().enumerate() {
            let id = ids.and_then(|ids| ids.get(i)).map(|s| s.clone());
            
            match self.process_representation(rep, id) {
                Ok(features) => results.push(features),
                Err(e) => {
                    log::warn!("Failed to process representation {}: {:?}", i, e);
                    self.update_stats_failure();
                    continue;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Process a single representation
    pub fn process_representation(
        &mut self,
        representation: &Representation,
        id: Option<String>,
    ) -> FeatureResult<FeatureVector> {
        let extractor = RepresentationExtractor::default();
        let encoder = RepresentationEncoder::default();
        
        let extracted = extractor.extract(representation, &self.config.extraction)?;
        let encoded = encoder.encode(representation, &self.config.encoding)?;
        
        let normalized = if self.normalizer.fitted {
            self.normalizer.transform(&encoded)?
        } else {
            encoded
        };
        
        let validation_result = self.validator.feature_validator.validate(
            &normalized,
            &self.config.validation,
        );
        
        if !validation_result.is_valid {
            return Err(FeatureError::ValidationFailed {
                message: format!("Representation validation failed: {:?}", validation_result.errors),
            });
        }
        
        if let (Some(storage), Some(id)) = (&mut self.storage, id) {
            storage.store(&id, &normalized)?;
        }
        
        self.update_stats_success(0);
        Ok(normalized)
    }
    
    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        if let Ok(mut stats) = self.stats.lock() {
            stats.finalize();
            stats.clone()
        } else {
            PipelineStats::new()
        }
    }
    
    /// Reset pipeline statistics
    pub fn reset_stats(&mut self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = PipelineStats::new();
        }
    }
    
    /// Get storage statistics if available
    pub fn storage_stats(&self) -> Option<StorageStats> {
        self.storage.as_ref().map(|s| s.stats())
    }
    
    /// Compact storage
    pub fn compact_storage(&mut self) -> FeatureResult<()> {
        if let Some(storage) = &mut self.storage {
            storage.compact()
        } else {
            Ok(())
        }
    }
    
    // Private helper methods
    
    fn update_stats_success(&self, processing_time_ms: u64) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_processed += 1;
            stats.successful += 1;
            stats.processing_time_ms += processing_time_ms;
        }
    }
    
    fn update_stats_failure(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_processed += 1;
            stats.failed += 1;
        }
    }
    
    fn report_progress(&self, current: usize, total: usize, message: &str) {
        if self.config.progress_reporting {
            if let Some(callback) = &self.progress_callback {
                callback(current, total, message);
            }
        }
    }
}

impl Default for FeaturePipeline {
    fn default() -> Self {
        Self::new().expect("Failed to create default pipeline")
    }
}

/// Builder for creating feature pipelines with custom configurations
#[derive(Debug, Clone)]
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }
    
    /// Set extraction configuration
    pub fn with_extraction(mut self, config: ExtractorConfig) -> Self {
        self.config.extraction = config;
        self
    }
    
    /// Set encoding configuration
    pub fn with_encoding(mut self, config: EncodingConfig) -> Self {
        self.config.encoding = config;
        self
    }
    
    /// Set normalization type
    pub fn with_normalization(mut self, normalization: NormalizationType) -> Self {
        self.config.normalization = normalization;
        self
    }
    
    /// Set validation configuration
    pub fn with_validation(mut self, config: ValidationConfig) -> Self {
        self.config.validation = config;
        self
    }
    
    /// Set storage configuration
    pub fn with_storage(mut self, config: StorageConfig) -> Self {
        self.config.storage = config;
        self
    }
    
    /// Enable/disable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }
    
    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }
    
    /// Enable/disable progress reporting
    pub fn with_progress_reporting(mut self, enabled: bool) -> Self {
        self.config.progress_reporting = enabled;
        self
    }
    
    /// Build the pipeline
    pub fn build(self) -> FeatureResult<FeaturePipeline> {
        FeaturePipeline::with_config(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.parallel);
        assert_eq!(config.batch_size, 32);
        assert!(config.progress_reporting);
        assert!(config.cache_intermediates);
    }
    
    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .with_parallel(false)
            .with_batch_size(16)
            .with_progress_reporting(false)
            .build()
            .unwrap();
        
        assert!(!pipeline.config.parallel);
        assert_eq!(pipeline.config.batch_size, 16);
        assert!(!pipeline.config.progress_reporting);
    }
    
    #[test]
    fn test_pipeline_stats() {
        let mut stats = PipelineStats::new();
        stats.total_processed = 100;
        stats.successful = 90;
        stats.processing_time_ms = 1000;
        stats.finalize();
        
        assert_eq!(stats.avg_time_per_object_ms, 10.0);
        assert_eq!(stats.validation_success_rate, 0.9);
    }
    
    #[test]
    fn test_pipeline_creation() {
        let pipeline = FeaturePipeline::new().unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.successful, 0);
    }
}