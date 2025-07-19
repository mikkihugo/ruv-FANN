//! Data processing pipeline for efficient batch processing

use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use futures::stream::{self, StreamExt};
use rayon::prelude::*;
use dashmap::DashMap;
use tracing::{info, debug, warn};

use super::{DataConfig, DataError, Result, features::{Featurizable, FeatureVector, FeatureExtractor}};
use super::cache::CacheManager;

/// Data processing pipeline with parallel execution support
pub struct DataPipeline {
    config: Arc<DataConfig>,
    extractor: Arc<FeatureExtractor>,
    semaphore: Arc<Semaphore>,
    metrics: Arc<PipelineMetrics>,
}

/// Pipeline performance metrics
#[derive(Default)]
pub struct PipelineMetrics {
    pub total_processed: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
    pub extraction_errors: std::sync::atomic::AtomicU64,
    pub avg_extraction_time_ms: std::sync::atomic::AtomicU64,
}

impl PipelineMetrics {
    pub fn record_processed(&self, count: u64) {
        self.total_processed.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn record_error(&self) {
        self.extraction_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn update_avg_time(&self, time_ms: u64) {
        // Simple moving average
        let current = self.avg_extraction_time_ms.load(std::sync::atomic::Ordering::Relaxed);
        let new_avg = (current * 9 + time_ms) / 10;
        self.avg_extraction_time_ms.store(new_avg, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn report(&self) -> String {
        let total = self.total_processed.load(std::sync::atomic::Ordering::Relaxed);
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let errors = self.extraction_errors.load(std::sync::atomic::Ordering::Relaxed);
        let avg_time = self.avg_extraction_time_ms.load(std::sync::atomic::Ordering::Relaxed);
        
        let hit_rate = if hits + misses > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };
        
        format!(
            "Pipeline Metrics:\n\
             - Total Processed: {}\n\
             - Cache Hit Rate: {:.2}%\n\
             - Extraction Errors: {}\n\
             - Avg Extraction Time: {}ms",
            total, hit_rate, errors, avg_time
        )
    }
}

impl DataPipeline {
    pub fn new(config: &DataConfig) -> Result<Self> {
        let num_workers = if config.num_workers == 0 {
            num_cpus::get()
        } else {
            config.num_workers
        };
        
        let extractor = Arc::new(FeatureExtractor::new(config.feature_config.clone()));
        let semaphore = Arc::new(Semaphore::new(num_workers));
        let metrics = Arc::new(PipelineMetrics::default());
        
        info!("Initialized data pipeline with {} workers", num_workers);
        
        Ok(Self {
            config: Arc::new(config.clone()),
            extractor,
            semaphore,
            metrics,
        })
    }
    
    /// Process a batch of objects with caching and parallel execution
    pub async fn process_batch<T>(&self, objects: Vec<T>) -> Result<Vec<FeatureVector>>
    where
        T: Featurizable + Send + Sync + 'static,
    {
        let start_time = std::time::Instant::now();
        let batch_size = objects.len();
        
        info!("Processing batch of {} objects", batch_size);
        
        // Check if we should use parallel processing
        if self.config.parallel_processing && batch_size > 10 {
            self.process_parallel(objects).await
        } else {
            self.process_sequential(objects).await
        }
        .map(|results| {
            let elapsed = start_time.elapsed().as_millis() as u64;
            self.metrics.record_processed(results.len() as u64);
            self.metrics.update_avg_time(elapsed / results.len().max(1) as u64);
            
            info!(
                "Processed {} objects in {}ms ({:.2} objects/sec)",
                results.len(),
                elapsed,
                (results.len() as f64 / elapsed as f64) * 1000.0
            );
            
            results
        })
    }
    
    /// Sequential processing for small batches
    async fn process_sequential<T>(&self, objects: Vec<T>) -> Result<Vec<FeatureVector>>
    where
        T: Featurizable + Send + Sync,
    {
        let mut results = Vec::with_capacity(objects.len());
        
        for obj in objects {
            match self.process_single(obj).await {
                Ok(features) => results.push(features),
                Err(e) => {
                    self.metrics.record_error();
                    warn!("Failed to extract features: {}", e);
                    // Continue processing other objects
                }
            }
        }
        
        Ok(results)
    }
    
    /// Parallel processing for large batches
    async fn process_parallel<T>(&self, objects: Vec<T>) -> Result<Vec<FeatureVector>>
    where
        T: Featurizable + Send + Sync + 'static,
    {
        let chunk_size = self.config.batch_size;
        let chunks: Vec<Vec<T>> = objects
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        debug!("Processing {} chunks of size {}", chunks.len(), chunk_size);
        
        // Process chunks concurrently
        let mut handles = Vec::new();
        
        for chunk in chunks {
            let pipeline = self.clone();
            let handle = tokio::spawn(async move {
                pipeline.process_chunk(chunk).await
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut all_results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(results)) => all_results.extend(results),
                Ok(Err(e)) => {
                    warn!("Chunk processing failed: {}", e);
                    // Continue with other chunks
                }
                Err(e) => {
                    warn!("Task panicked: {}", e);
                }
            }
        }
        
        Ok(all_results)
    }
    
    /// Process a chunk of objects using thread pool
    async fn process_chunk<T>(&self, chunk: Vec<T>) -> Result<Vec<FeatureVector>>
    where
        T: Featurizable + Send + Sync,
    {
        let _permit = self.semaphore.acquire().await?;
        
        let extractor = self.extractor.clone();
        let config = self.config.feature_config.clone();
        
        // Use Rayon for CPU-bound feature extraction
        let results = tokio::task::spawn_blocking(move || {
            chunk.par_iter()
                .filter_map(|obj| {
                    match obj.extract_features(&config) {
                        Ok(features) => Some(features),
                        Err(e) => {
                            warn!("Feature extraction failed: {}", e);
                            None
                        }
                    }
                })
                .collect::<Vec<_>>()
        }).await?;
        
        Ok(results)
    }
    
    /// Process a single object with caching
    async fn process_single<T>(&self, obj: T) -> Result<FeatureVector>
    where
        T: Featurizable,
    {
        let cache_key = obj.cache_key();
        
        // Check cache first
        if let Some(cached) = self.check_cache(&cache_key).await {
            self.metrics.record_cache_hit();
            return Ok(cached);
        }
        
        self.metrics.record_cache_miss();
        
        // Extract features
        let features = obj.extract_features(&self.config.feature_config)?;
        
        // Store in cache for future use
        self.store_cache(&cache_key, &features).await?;
        
        Ok(features)
    }
    
    /// Check cache for existing features
    async fn check_cache(&self, key: &str) -> Option<FeatureVector> {
        // TODO: Implement actual cache lookup
        None
    }
    
    /// Store features in cache
    async fn store_cache(&self, key: &str, features: &FeatureVector) -> Result<()> {
        // TODO: Implement actual cache storage
        Ok(())
    }
    
    /// Get pipeline metrics
    pub fn metrics(&self) -> String {
        self.metrics.report()
    }
}

impl Clone for DataPipeline {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            extractor: self.extractor.clone(),
            semaphore: self.semaphore.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

/// Streaming pipeline for processing large datasets
pub struct StreamingPipeline {
    pipeline: DataPipeline,
    buffer_size: usize,
}

impl StreamingPipeline {
    pub fn new(pipeline: DataPipeline, buffer_size: usize) -> Self {
        Self { pipeline, buffer_size }
    }
    
    /// Process a stream of objects
    pub async fn process_stream<T, S>(&self, stream: S) -> Result<mpsc::Receiver<FeatureVector>>
    where
        T: Featurizable + Send + Sync + 'static,
        S: futures::Stream<Item = T> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(self.buffer_size);
        let pipeline = self.pipeline.clone();
        
        tokio::spawn(async move {
            let mut stream = Box::pin(stream);
            let mut buffer = Vec::with_capacity(pipeline.config.batch_size);
            
            while let Some(item) = stream.next().await {
                buffer.push(item);
                
                if buffer.len() >= pipeline.config.batch_size {
                    let batch = std::mem::replace(
                        &mut buffer, 
                        Vec::with_capacity(pipeline.config.batch_size)
                    );
                    
                    if let Ok(results) = pipeline.process_batch(batch).await {
                        for feature in results {
                            if tx.send(feature).await.is_err() {
                                return; // Receiver dropped
                            }
                        }
                    }
                }
            }
            
            // Process remaining items
            if !buffer.is_empty() {
                if let Ok(results) = pipeline.process_batch(buffer).await {
                    for feature in results {
                        let _ = tx.send(feature).await;
                    }
                }
            }
        });
        
        Ok(rx)
    }
}

/// Pipeline builder for fluent configuration
pub struct PipelineBuilder {
    config: DataConfig,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: DataConfig::default(),
        }
    }
    
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }
    
    pub fn with_workers(mut self, num: usize) -> Self {
        self.config.num_workers = num;
        self
    }
    
    pub fn with_cache_size(mut self, mb: usize) -> Self {
        self.config.cache_size_mb = mb;
        self
    }
    
    pub fn enable_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel_processing = enabled;
        self
    }
    
    pub fn build(self) -> Result<DataPipeline> {
        DataPipeline::new(&self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestObject {
        id: String,
        value: f64,
    }
    
    impl Featurizable for TestObject {
        fn extract_features(&self, config: &super::super::FeatureConfig) -> Result<FeatureVector> {
            let mut fv = FeatureVector::new(
                self.id.clone(),
                config.geometric_dim,
                config.algebraic_dim
            );
            fv.geometric[0] = self.value;
            Ok(fv)
        }
        
        fn cache_key(&self) -> String {
            self.id.clone()
        }
    }
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = PipelineBuilder::new()
            .with_batch_size(64)
            .with_workers(4)
            .build()
            .unwrap();
        
        assert_eq!(pipeline.config.batch_size, 64);
    }
    
    #[tokio::test]
    async fn test_sequential_processing() {
        let pipeline = PipelineBuilder::new()
            .enable_parallel(false)
            .build()
            .unwrap();
        
        let objects = vec![
            TestObject { id: "1".to_string(), value: 1.0 },
            TestObject { id: "2".to_string(), value: 2.0 },
        ];
        
        let results = pipeline.process_batch(objects).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert_eq!(results[1].id, "2");
    }
    
    #[tokio::test]
    async fn test_metrics_reporting() {
        let pipeline = PipelineBuilder::new().build().unwrap();
        
        pipeline.metrics.record_processed(100);
        pipeline.metrics.record_cache_hit();
        pipeline.metrics.record_cache_hit();
        pipeline.metrics.record_cache_miss();
        
        let report = pipeline.metrics();
        assert!(report.contains("Total Processed: 100"));
        assert!(report.contains("Cache Hit Rate: 66.67%"));
    }
}