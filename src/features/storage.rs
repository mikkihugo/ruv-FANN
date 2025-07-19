//! Persistent storage system for feature vectors
//!
//! This module provides efficient storage backends with indexing,
//! caching, and batch operations for feature vectors.

use crate::features::{FeatureVector, FeatureResult, FeatureError, SerializationConfig, FeatureSerializer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter};

/// Storage backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageBackend {
    /// In-memory storage (fast, not persistent)
    Memory,
    /// File-based storage
    FileSystem,
    /// Directory with indexed files
    IndexedFiles,
    /// SQLite database (requires sqlite feature)
    SQLite,
    /// Custom storage backend
    Custom,
}

impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::FileSystem
    }
}

/// Storage configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StorageConfig {
    /// Storage backend type
    pub backend: StorageBackend,
    /// Base directory for file-based storage
    pub base_path: PathBuf,
    /// Serialization configuration
    pub serialization: SerializationConfig,
    /// Enable caching
    pub enable_cache: bool,
    /// Maximum cache size (number of feature vectors)
    pub max_cache_size: usize,
    /// Enable indexing for fast lookups
    pub enable_indexing: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Auto-create directories
    pub auto_create_dirs: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::FileSystem,
            base_path: PathBuf::from("./feature_storage"),
            serialization: SerializationConfig::default(),
            enable_cache: true,
            max_cache_size: 1000,
            enable_indexing: true,
            batch_size: 32,
            enable_compression: true,
            auto_create_dirs: true,
        }
    }
}

/// Feature storage index entry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexEntry {
    /// Unique identifier
    pub id: String,
    /// Object type
    pub object_type: String,
    /// Encoding strategy
    pub encoding_strategy: String,
    /// Feature dimension
    pub dimension: usize,
    /// File path (for file-based storage)
    pub file_path: Option<PathBuf>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last accessed timestamp
    pub accessed_at: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl IndexEntry {
    pub fn new(id: String, features: &FeatureVector) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            id,
            object_type: features.metadata.object_type.clone(),
            encoding_strategy: features.metadata.encoding_strategy.clone(),
            dimension: features.dimension,
            file_path: None,
            created_at: now,
            accessed_at: now,
            size_bytes: 0,
            metadata: features.metadata.extra.clone(),
        }
    }
}

/// Main trait for feature storage
pub trait FeatureStorage {
    /// Store a feature vector with a given ID
    fn store(&mut self, id: &str, features: &FeatureVector) -> FeatureResult<()>;
    
    /// Retrieve a feature vector by ID
    fn retrieve(&mut self, id: &str) -> FeatureResult<FeatureVector>;
    
    /// Check if a feature vector exists
    fn exists(&self, id: &str) -> bool;
    
    /// Delete a feature vector
    fn delete(&mut self, id: &str) -> FeatureResult<()>;
    
    /// List all stored feature vector IDs
    fn list_ids(&self) -> Vec<String>;
    
    /// Store multiple feature vectors
    fn store_batch(&mut self, items: &[(String, FeatureVector)]) -> FeatureResult<()> {
        for (id, features) in items {
            self.store(id, features)?;
        }
        Ok(())
    }
    
    /// Retrieve multiple feature vectors
    fn retrieve_batch(&mut self, ids: &[String]) -> FeatureResult<Vec<FeatureVector>> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.retrieve(id)?);
        }
        Ok(results)
    }
    
    /// Get storage statistics
    fn stats(&self) -> StorageStats;
    
    /// Compact storage (remove deleted entries, optimize)
    fn compact(&mut self) -> FeatureResult<()> {
        Ok(()) // Default: no-op
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StorageStats {
    /// Total number of stored feature vectors
    pub total_count: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Average feature dimension
    pub avg_dimension: f64,
    /// Storage backend type
    pub backend_type: String,
}

/// In-memory feature storage
#[derive(Debug, Clone)]
pub struct MemoryStorage {
    storage: HashMap<String, FeatureVector>,
    index: HashMap<String, IndexEntry>,
    cache_hits: u64,
    cache_misses: u64,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
            index: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureStorage for MemoryStorage {
    fn store(&mut self, id: &str, features: &FeatureVector) -> FeatureResult<()> {
        let mut entry = IndexEntry::new(id.to_string(), features);
        
        // Estimate size
        entry.size_bytes = (features.dimension * 8 + 256) as u64; // Rough estimate
        
        self.storage.insert(id.to_string(), features.clone());
        self.index.insert(id.to_string(), entry);
        
        Ok(())
    }
    
    fn retrieve(&mut self, id: &str) -> FeatureResult<FeatureVector> {
        match self.storage.get(id) {
            Some(features) => {
                self.cache_hits += 1;
                
                // Update access time
                if let Some(entry) = self.index.get_mut(id) {
                    entry.accessed_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                }
                
                Ok(features.clone())
            }
            None => {
                self.cache_misses += 1;
                Err(FeatureError::StorageFailed {
                    message: format!("Feature vector with ID '{}' not found", id),
                })
            }
        }
    }
    
    fn exists(&self, id: &str) -> bool {
        self.storage.contains_key(id)
    }
    
    fn delete(&mut self, id: &str) -> FeatureResult<()> {
        self.storage.remove(id);
        self.index.remove(id);
        Ok(())
    }
    
    fn list_ids(&self) -> Vec<String> {
        self.storage.keys().cloned().collect()
    }
    
    fn stats(&self) -> StorageStats {
        let total_count = self.storage.len();
        let total_size = self.index.values().map(|e| e.size_bytes).sum();
        let avg_dimension = if total_count > 0 {
            self.index.values().map(|e| e.dimension as f64).sum::<f64>() / total_count as f64
        } else {
            0.0
        };
        
        let cache_hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };
        
        StorageStats {
            total_count,
            total_size_bytes: total_size,
            cache_hit_rate,
            avg_dimension,
            backend_type: "Memory".to_string(),
        }
    }
}

/// File-based feature storage
#[derive(Debug)]
pub struct FileStorage {
    config: StorageConfig,
    serializer: FeatureSerializer,
    index: HashMap<String, IndexEntry>,
    cache: HashMap<String, FeatureVector>,
    cache_hits: u64,
    cache_misses: u64,
}

impl FileStorage {
    pub fn new(config: StorageConfig) -> FeatureResult<Self> {
        if config.auto_create_dirs {
            create_dir_all(&config.base_path)
                .map_err(|e| FeatureError::StorageFailed {
                    message: format!("Failed to create storage directory: {}", e),
                })?;
        }
        
        let serializer = FeatureSerializer::with_config(config.serialization.clone());
        
        let mut storage = Self {
            config,
            serializer,
            index: HashMap::new(),
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        };
        
        // Load existing index
        storage.load_index()?;
        
        Ok(storage)
    }
    
    fn get_file_path(&self, id: &str) -> PathBuf {
        self.config.base_path.join(format!("{}.fv", id)) // .fv = feature vector
    }
    
    fn get_index_path(&self) -> PathBuf {
        self.config.base_path.join("index.json")
    }
    
    fn load_index(&mut self) -> FeatureResult<()> {
        let index_path = self.get_index_path();
        
        if index_path.exists() {
            let file = File::open(&index_path)
                .map_err(|e| FeatureError::StorageFailed {
                    message: format!("Failed to open index file: {}", e),
                })?;
            
            let reader = BufReader::new(file);
            
            #[cfg(feature = "serde")]
            {
                self.index = serde_json::from_reader(reader)
                    .map_err(|e| FeatureError::StorageFailed {
                        message: format!("Failed to parse index file: {}", e),
                    })?;
            }
            
            #[cfg(not(feature = "serde"))]
            {
                // Fallback: empty index
                self.index = HashMap::new();
            }
        }
        
        Ok(())
    }
    
    fn save_index(&self) -> FeatureResult<()> {
        let index_path = self.get_index_path();
        
        let file = File::create(&index_path)
            .map_err(|e| FeatureError::StorageFailed {
                message: format!("Failed to create index file: {}", e),
            })?;
        
        let writer = BufWriter::new(file);
        
        #[cfg(feature = "serde")]
        {
            serde_json::to_writer_pretty(writer, &self.index)
                .map_err(|e| FeatureError::StorageFailed {
                    message: format!("Failed to write index file: {}", e),
                })?;
        }
        
        #[cfg(not(feature = "serde"))]
        {
            // Fallback: write simple format
            use std::io::Write;
            let mut writer = writer;
            writeln!(writer, "{{")
                .map_err(|e| FeatureError::StorageFailed { message: e.to_string() })?;
            for (id, entry) in &self.index {
                writeln!(writer, "  \"{}\": {{\"dimension\": {}}},", id, entry.dimension)
                    .map_err(|e| FeatureError::StorageFailed { message: e.to_string() })?;
            }
            writeln!(writer, "}}")
                .map_err(|e| FeatureError::StorageFailed { message: e.to_string() })?;
        }
        
        Ok(())
    }
    
    fn manage_cache(&mut self, id: &str, features: &FeatureVector) {
        if !self.config.enable_cache {
            return;
        }
        
        // Remove oldest entries if cache is full
        if self.cache.len() >= self.config.max_cache_size {
            // Simple LRU: remove first entry (in practice, you'd want a proper LRU implementation)
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        
        self.cache.insert(id.to_string(), features.clone());
    }
}

impl FeatureStorage for FileStorage {
    fn store(&mut self, id: &str, features: &FeatureVector) -> FeatureResult<()> {
        let file_path = self.get_file_path(id);
        
        // Serialize feature vector
        let serialized = self.serializer.serialize(features)?;
        
        // Write to file
        std::fs::write(&file_path, &serialized)
            .map_err(|e| FeatureError::StorageFailed {
                message: format!("Failed to write feature vector file: {}", e),
            })?;
        
        // Update index
        let mut entry = IndexEntry::new(id.to_string(), features);
        entry.file_path = Some(file_path);
        entry.size_bytes = serialized.len() as u64;
        
        self.index.insert(id.to_string(), entry);
        
        // Save index
        self.save_index()?;
        
        // Update cache
        self.manage_cache(id, features);
        
        Ok(())
    }
    
    fn retrieve(&mut self, id: &str) -> FeatureResult<FeatureVector> {
        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.cache.get(id) {
                self.cache_hits += 1;
                
                // Update access time in index
                if let Some(entry) = self.index.get_mut(id) {
                    entry.accessed_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                }
                
                return Ok(cached.clone());
            }
        }
        
        self.cache_misses += 1;
        
        // Check if entry exists in index
        let entry = self.index.get(id)
            .ok_or_else(|| FeatureError::StorageFailed {
                message: format!("Feature vector with ID '{}' not found in index", id),
            })?;
        
        // Get file path
        let file_path = entry.file_path.as_ref()
            .unwrap_or(&self.get_file_path(id));
        
        // Read from file
        let serialized = std::fs::read(file_path)
            .map_err(|e| FeatureError::StorageFailed {
                message: format!("Failed to read feature vector file: {}", e),
            })?;
        
        // Deserialize
        let features = self.serializer.deserialize(&serialized)?;
        
        // Update cache
        self.manage_cache(id, &features);
        
        // Update access time
        if let Some(entry) = self.index.get_mut(id) {
            entry.accessed_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
        
        Ok(features)
    }
    
    fn exists(&self, id: &str) -> bool {
        self.index.contains_key(id)
    }
    
    fn delete(&mut self, id: &str) -> FeatureResult<()> {
        // Remove from cache
        self.cache.remove(id);
        
        // Remove file
        if let Some(entry) = self.index.get(id) {
            let file_path = entry.file_path.as_ref()
                .unwrap_or(&self.get_file_path(id));
            
            if file_path.exists() {
                std::fs::remove_file(file_path)
                    .map_err(|e| FeatureError::StorageFailed {
                        message: format!("Failed to delete feature vector file: {}", e),
                    })?;
            }
        }
        
        // Remove from index
        self.index.remove(id);
        
        // Save updated index
        self.save_index()?;
        
        Ok(())
    }
    
    fn list_ids(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }
    
    fn stats(&self) -> StorageStats {
        let total_count = self.index.len();
        let total_size = self.index.values().map(|e| e.size_bytes).sum();
        let avg_dimension = if total_count > 0 {
            self.index.values().map(|e| e.dimension as f64).sum::<f64>() / total_count as f64
        } else {
            0.0
        };
        
        let cache_hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };
        
        StorageStats {
            total_count,
            total_size_bytes: total_size,
            cache_hit_rate,
            avg_dimension,
            backend_type: "FileSystem".to_string(),
        }
    }
    
    fn compact(&mut self) -> FeatureResult<()> {
        // Remove orphaned files
        let entries: Vec<_> = std::fs::read_dir(&self.config.base_path)
            .map_err(|e| FeatureError::StorageFailed {
                message: format!("Failed to read storage directory: {}", e),
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().and_then(|s| s.to_str()) == Some("fv")
            })
            .collect();
        
        for entry in entries {
            let file_path = entry.path();
            let file_name = file_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            
            if !self.index.contains_key(file_name) {
                // Orphaned file - remove it
                std::fs::remove_file(&file_path)
                    .map_err(|e| FeatureError::StorageFailed {
                        message: format!("Failed to remove orphaned file: {}", e),
                    })?;
            }
        }
        
        // Save compacted index
        self.save_index()?;
        
        Ok(())
    }
}

/// Universal storage that can use different backends
#[derive(Debug)]
pub struct UniversalStorage {
    backend: Box<dyn FeatureStorage>,
}

impl UniversalStorage {
    /// Create storage with the specified backend
    pub fn new(config: StorageConfig) -> FeatureResult<Self> {
        let backend: Box<dyn FeatureStorage> = match config.backend {
            StorageBackend::Memory => Box::new(MemoryStorage::new()),
            StorageBackend::FileSystem | StorageBackend::IndexedFiles => {
                Box::new(FileStorage::new(config)?)
            }
            _ => {
                return Err(FeatureError::StorageFailed {
                    message: format!("Storage backend {:?} not implemented", config.backend),
                });
            }
        };
        
        Ok(Self { backend })
    }
    
    /// Search for feature vectors by metadata
    pub fn search_by_metadata(
        &mut self,
        key: &str,
        value: &str,
    ) -> FeatureResult<Vec<(String, FeatureVector)>> {
        let mut results = Vec::new();
        
        for id in self.backend.list_ids() {
            if let Ok(features) = self.backend.retrieve(&id) {
                if let Some(metadata_value) = features.metadata.extra.get(key) {
                    if metadata_value == value {
                        results.push((id, features));
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search for feature vectors by object type
    pub fn search_by_type(&mut self, object_type: &str) -> FeatureResult<Vec<(String, FeatureVector)>> {
        let mut results = Vec::new();
        
        for id in self.backend.list_ids() {
            if let Ok(features) = self.backend.retrieve(&id) {
                if features.metadata.object_type == object_type {
                    results.push((id, features));
                }
            }
        }
        
        Ok(results)
    }
    
    /// Find similar feature vectors using cosine similarity
    pub fn find_similar(
        &mut self,
        target: &FeatureVector,
        threshold: f64,
        max_results: usize,
    ) -> FeatureResult<Vec<(String, FeatureVector, f64)>> {
        let mut similarities = Vec::new();
        
        for id in self.backend.list_ids() {
            if let Ok(features) = self.backend.retrieve(&id) {
                if let Ok(similarity) = target.cosine_similarity(&features) {
                    if similarity >= threshold {
                        similarities.push((id, features, similarity));
                    }
                }
            }
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        similarities.truncate(max_results);
        
        Ok(similarities)
    }
}

impl FeatureStorage for UniversalStorage {
    fn store(&mut self, id: &str, features: &FeatureVector) -> FeatureResult<()> {
        self.backend.store(id, features)
    }
    
    fn retrieve(&mut self, id: &str) -> FeatureResult<FeatureVector> {
        self.backend.retrieve(id)
    }
    
    fn exists(&self, id: &str) -> bool {
        self.backend.exists(id)
    }
    
    fn delete(&mut self, id: &str) -> FeatureResult<()> {
        self.backend.delete(id)
    }
    
    fn list_ids(&self) -> Vec<String> {
        self.backend.list_ids()
    }
    
    fn stats(&self) -> StorageStats {
        self.backend.stats()
    }
    
    fn compact(&mut self) -> FeatureResult<()> {
        self.backend.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_dir_all;
    
    fn create_test_feature_vector() -> FeatureVector {
        FeatureVector::new(
            vec![1.0, 2.0, 3.0, 4.0],
            "test".to_string(),
            "test_encoder".to_string(),
        )
    }
    
    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::new();
        let features = create_test_feature_vector();
        
        // Test store and retrieve
        storage.store("test_id", &features).unwrap();
        assert!(storage.exists("test_id"));
        
        let retrieved = storage.retrieve("test_id").unwrap();
        assert_eq!(features.values, retrieved.values);
        
        // Test delete
        storage.delete("test_id").unwrap();
        assert!(!storage.exists("test_id"));
    }
    
    #[test]
    fn test_file_storage() {
        let temp_dir = std::env::temp_dir().join("test_storage");
        let config = StorageConfig {
            backend: StorageBackend::FileSystem,
            base_path: temp_dir.clone(),
            ..Default::default()
        };
        
        let mut storage = FileStorage::new(config).unwrap();
        let features = create_test_feature_vector();
        
        // Test store and retrieve
        storage.store("test_id", &features).unwrap();
        assert!(storage.exists("test_id"));
        
        let retrieved = storage.retrieve("test_id").unwrap();
        assert_eq!(features.values, retrieved.values);
        
        // Test stats
        let stats = storage.stats();
        assert_eq!(stats.total_count, 1);
        
        // Cleanup
        let _ = remove_dir_all(temp_dir);
    }
    
    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.backend, StorageBackend::FileSystem);
        assert!(config.enable_cache);
        assert!(config.enable_indexing);
        assert_eq!(config.max_cache_size, 1000);
    }
    
    #[test]
    fn test_universal_storage() {
        let config = StorageConfig {
            backend: StorageBackend::Memory,
            ..Default::default()
        };
        
        let mut storage = UniversalStorage::new(config).unwrap();
        let features = create_test_feature_vector();
        
        storage.store("test_id", &features).unwrap();
        let retrieved = storage.retrieve("test_id").unwrap();
        assert_eq!(features.values, retrieved.values);
    }
}