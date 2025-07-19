//! Caching strategies for expensive mathematical computations

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use bincode;

#[cfg(feature = "cache")]
use rocksdb::{DB, Options, WriteBatch};

use super::{DataConfig, DataError, Result, features::FeatureVector};

/// Cache manager for feature vectors and intermediate computations
pub struct CacheManager {
    memory_cache: Arc<DashMap<String, CachedItem>>,
    #[cfg(feature = "cache")]
    disk_cache: Option<Arc<DB>>,
    config: Arc<DataConfig>,
    stats: Arc<RwLock<CacheStats>>,
}

/// Cached item with metadata
#[derive(Clone, Serialize, Deserialize)]
struct CachedItem {
    data: Vec<u8>,
    created_at: u64,
    access_count: u32,
    size_bytes: usize,
}

/// Cache statistics
#[derive(Default, Clone)]
pub struct CacheStats {
    pub total_items: usize,
    pub memory_size_bytes: usize,
    pub disk_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
}

impl CacheManager {
    pub async fn new(config: &DataConfig) -> Result<Self> {
        let memory_cache = Arc::new(DashMap::new());
        
        #[cfg(feature = "cache")]
        let disk_cache = if config.cache_size_mb > 0 {
            let path = PathBuf::from(&config.cache_dir);
            tokio::fs::create_dir_all(&path).await?;
            
            let mut opts = Options::default();
            opts.create_if_missing(true);
            opts.set_max_total_wal_size(config.cache_size_mb as u64 * 1024 * 1024);
            
            let db = DB::open(&opts, path)
                .map_err(|e| DataError::CacheError(e.to_string()))?;
            
            Some(Arc::new(db))
        } else {
            None
        };
        
        #[cfg(not(feature = "cache"))]
        let disk_cache = None;
        
        Ok(Self {
            memory_cache,
            #[cfg(feature = "cache")]
            disk_cache,
            config: Arc::new(config.clone()),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }
    
    /// Get a feature vector from cache
    pub async fn get(&self, key: &str) -> Option<FeatureVector> {
        // Check memory cache first
        if let Some(mut item) = self.memory_cache.get_mut(key) {
            item.access_count += 1;
            let data = item.data.clone();
            drop(item);
            
            self.stats.write().await.hit_count += 1;
            
            return bincode::deserialize(&data).ok();
        }
        
        // Check disk cache if available
        #[cfg(feature = "cache")]
        if let Some(db) = &self.disk_cache {
            if let Ok(Some(data)) = db.get(key.as_bytes()) {
                self.stats.write().await.hit_count += 1;
                
                // Deserialize
                if let Ok(features) = bincode::deserialize::<FeatureVector>(&data) {
                    // Promote to memory cache
                    let item = CachedItem {
                        data: data.to_vec(),
                        created_at: current_timestamp(),
                        access_count: 1,
                        size_bytes: data.len(),
                    };
                    
                    self.memory_cache.insert(key.to_string(), item);
                    return Some(features);
                }
            }
        }
        
        self.stats.write().await.miss_count += 1;
        None
    }
    
    /// Store a feature vector in cache
    pub async fn put(&self, key: &str, features: &FeatureVector) -> Result<()> {
        let data = bincode::serialize(features)?;
        let size = data.len();
        
        // Check if we need to evict items
        if self.should_evict(size).await {
            self.evict_lru().await?;
        }
        
        let item = CachedItem {
            data: data.clone(),
            created_at: current_timestamp(),
            access_count: 0,
            size_bytes: size,
        };
        
        // Store in memory cache
        self.memory_cache.insert(key.to_string(), item);
        
        // Store in disk cache if available
        #[cfg(feature = "cache")]
        if let Some(db) = &self.disk_cache {
            db.put(key.as_bytes(), &data)
                .map_err(|e| DataError::CacheError(e.to_string()))?;
        }
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_items += 1;
        stats.memory_size_bytes += size;
        
        Ok(())
    }
    
    /// Batch get operation
    pub async fn get_batch(&self, keys: &[String]) -> Vec<Option<FeatureVector>> {
        let mut results = Vec::with_capacity(keys.len());
        
        for key in keys {
            results.push(self.get(key).await);
        }
        
        results
    }
    
    /// Batch put operation
    pub async fn put_batch(&self, items: Vec<(String, FeatureVector)>) -> Result<()> {
        #[cfg(feature = "cache")]
        if let Some(db) = &self.disk_cache {
            let mut batch = WriteBatch::default();
            
            for (key, features) in &items {
                let data = bincode::serialize(features)?;
                batch.put(key.as_bytes(), &data);
            }
            
            db.write(batch)
                .map_err(|e| DataError::CacheError(e.to_string()))?;
        }
        
        // Store in memory cache
        for (key, features) in items {
            self.put(&key, &features).await?;
        }
        
        Ok(())
    }
    
    /// Check if eviction is needed
    async fn should_evict(&self, new_size: usize) -> bool {
        let stats = self.stats.read().await;
        let max_memory = self.config.cache_size_mb * 1024 * 1024 / 2; // Use half for memory
        stats.memory_size_bytes + new_size > max_memory
    }
    
    /// Evict least recently used items
    async fn evict_lru(&self) -> Result<()> {
        let target_size = self.config.cache_size_mb * 1024 * 1024 / 4;
        let mut eviction_candidates = Vec::new();
        
        // Collect candidates
        for entry in self.memory_cache.iter() {
            eviction_candidates.push((
                entry.key().clone(),
                entry.value().access_count,
                entry.value().created_at,
                entry.value().size_bytes,
            ));
        }
        
        // Sort by access count and age
        eviction_candidates.sort_by_key(|(_, access_count, created_at, _)| {
            (*access_count, *created_at)
        });
        
        // Evict until we reach target size
        let mut evicted_size = 0;
        let mut evicted_count = 0;
        
        for (key, _, _, size) in eviction_candidates {
            if evicted_size >= target_size {
                break;
            }
            
            self.memory_cache.remove(&key);
            evicted_size += size;
            evicted_count += 1;
        }
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.memory_size_bytes = stats.memory_size_bytes.saturating_sub(evicted_size);
        stats.eviction_count += evicted_count;
        
        Ok(())
    }
    
    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        self.memory_cache.clear();
        
        #[cfg(feature = "cache")]
        if let Some(db) = &self.disk_cache {
            // Clear disk cache by iterating and deleting
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            let mut batch = WriteBatch::default();
            
            for (key, _) in iter {
                batch.delete(&key);
            }
            
            db.write(batch)
                .map_err(|e| DataError::CacheError(e.to_string()))?;
        }
        
        *self.stats.write().await = CacheStats::default();
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.blocking_read().clone()
    }
    
    /// Compute cache key for mathematical objects
    pub fn compute_key(object_type: &str, params: &[f64]) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        object_type.hash(&mut hasher);
        
        for &param in params {
            param.to_bits().hash(&mut hasher);
        }
        
        format!("{}:{:x}", object_type, hasher.finish())
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Cache-aware computation wrapper
pub struct CachedComputation<T> {
    cache: Arc<CacheManager>,
    compute_fn: Box<dyn Fn() -> Result<T> + Send + Sync>,
}

impl<T> CachedComputation<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone,
{
    pub fn new<F>(cache: Arc<CacheManager>, compute_fn: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            cache,
            compute_fn: Box::new(compute_fn),
        }
    }
    
    /// Execute computation with caching
    pub async fn execute(&self, key: &str) -> Result<T> {
        // Check cache first
        if let Some(mut item) = self.cache.memory_cache.get_mut(key) {
            item.access_count += 1;
            let data = item.data.clone();
            drop(item);
            
            if let Ok(result) = bincode::deserialize(&data) {
                return Ok(result);
            }
        }
        
        // Compute if not cached
        let result = (self.compute_fn)()?;
        
        // Cache the result
        let data = bincode::serialize(&result)?;
        let item = CachedItem {
            data,
            created_at: current_timestamp(),
            access_count: 1,
            size_bytes: std::mem::size_of_val(&result),
        };
        
        self.cache.memory_cache.insert(key.to_string(), item);
        
        Ok(result)
    }
}

/// LRU cache implementation for hot data
pub struct LRUCache<K, V> {
    capacity: usize,
    map: DashMap<K, V>,
    access_order: Arc<RwLock<Vec<K>>>,
}

impl<K, V> LRUCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: DashMap::new(),
            access_order: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn get(&self, key: &K) -> Option<V> {
        if let Some(value) = self.map.get(key) {
            // Update access order
            let mut order = self.access_order.write().await;
            order.retain(|k| k != key);
            order.push(key.clone());
            
            Some(value.clone())
        } else {
            None
        }
    }
    
    pub async fn put(&self, key: K, value: V) {
        // Check capacity
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            let mut order = self.access_order.write().await;
            if let Some(lru_key) = order.first().cloned() {
                self.map.remove(&lru_key);
                order.remove(0);
            }
        }
        
        self.map.insert(key.clone(), value);
        
        // Update access order
        let mut order = self.access_order.write().await;
        order.retain(|k| k != &key);
        order.push(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_key_generation() {
        let key1 = CacheManager::compute_key("sheaf", &[1.0, 2.0, 3.0]);
        let key2 = CacheManager::compute_key("sheaf", &[1.0, 2.0, 3.0]);
        let key3 = CacheManager::compute_key("sheaf", &[1.0, 2.0, 3.1]);
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
    
    #[tokio::test]
    async fn test_lru_cache() {
        let cache = LRUCache::new(2);
        
        cache.put("a", 1).await;
        cache.put("b", 2).await;
        
        assert_eq!(cache.get(&"a").await, Some(1));
        
        cache.put("c", 3).await; // Should evict "b"
        
        assert_eq!(cache.get(&"a").await, Some(1));
        assert_eq!(cache.get(&"b").await, None);
        assert_eq!(cache.get(&"c").await, Some(3));
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let config = DataConfig {
            cache_size_mb: 10,
            ..Default::default()
        };
        
        let cache = CacheManager::new(&config).await.unwrap();
        let stats = cache.stats();
        
        assert_eq!(stats.total_items, 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }
}