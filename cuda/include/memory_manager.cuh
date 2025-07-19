// Advanced CUDA Memory Management for Geometric Langlands
// Optimized memory allocation and management strategies

#ifndef MEMORY_MANAGER_CUH
#define MEMORY_MANAGER_CUH

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <functional>

namespace langlands {
namespace memory {

// Memory allocation strategies
enum class AllocationStrategy {
    POOL_BASED,        // Pre-allocated memory pools
    UNIFIED_MEMORY,    // CUDA Unified Memory
    STREAM_ORDERED,    // Stream-ordered allocations
    ADAPTIVE,          // Dynamic strategy selection
    HIERARCHICAL       // Multi-level memory hierarchy
};

// Memory access patterns
enum class AccessPattern {
    SEQUENTIAL,        // Sequential access
    RANDOM,           // Random access
    TEMPORAL_LOCALITY, // High temporal locality
    SPATIAL_LOCALITY,  // High spatial locality
    STREAMING         // One-time streaming access
};

// Memory pressure levels
enum class MemoryPressure {
    LOW,    // < 50% memory used
    MEDIUM, // 50-80% memory used
    HIGH,   // 80-95% memory used
    CRITICAL // > 95% memory used
};

// Memory block descriptor
struct MemoryBlock {
    void* ptr;
    size_t size;
    size_t alignment;
    bool in_use;
    int device_id;
    cudaStream_t stream;
    AccessPattern access_pattern;
    uint64_t allocation_time;
    uint64_t last_access_time;
    uint32_t access_count;
    
    MemoryBlock() : ptr(nullptr), size(0), alignment(0), in_use(false), 
                   device_id(0), stream(0), access_pattern(AccessPattern::RANDOM),
                   allocation_time(0), last_access_time(0), access_count(0) {}
};

// Memory statistics
struct MemoryStats {
    size_t total_allocated;
    size_t currently_used;
    size_t peak_usage;
    size_t fragmentation_bytes;
    uint32_t allocation_count;
    uint32_t deallocation_count;
    uint32_t cache_hits;
    uint32_t cache_misses;
    double average_allocation_time_us;
    double average_access_time_us;
    
    void reset() {
        total_allocated = 0;
        currently_used = 0;
        peak_usage = 0;
        fragmentation_bytes = 0;
        allocation_count = 0;
        deallocation_count = 0;
        cache_hits = 0;
        cache_misses = 0;
        average_allocation_time_us = 0.0;
        average_access_time_us = 0.0;
    }
};

// Memory pool for specific allocation sizes
class MemoryPool {
private:
    size_t block_size;
    size_t max_blocks;
    std::queue<void*> available_blocks;
    std::vector<void*> all_blocks;
    int device_id;
    cudaStream_t stream;
    mutable std::mutex pool_mutex;
    
public:
    MemoryPool(size_t block_sz, size_t max_blks, int dev_id, cudaStream_t strm = 0);
    ~MemoryPool();
    
    void* allocate();
    void deallocate(void* ptr);
    bool canAllocate(size_t size) const;
    size_t getBlockSize() const { return block_size; }
    size_t getAvailableBlocks() const;
    void expand(size_t additional_blocks);
    void shrink(size_t blocks_to_remove);
};

// Unified memory manager with adaptive strategies
class UnifiedMemoryManager {
private:
    struct UnifiedBlock {
        void* ptr;
        size_t size;
        bool in_use;
        int preferred_location; // GPU device or CPU (-1)
        uint64_t last_cpu_access;
        uint64_t last_gpu_access;
        AccessPattern pattern;
    };
    
    std::vector<UnifiedBlock> blocks;
    std::mutex blocks_mutex;
    
    // Memory advise optimization
    void optimizeMemoryAdvise(void* ptr, size_t size, AccessPattern pattern);
    void migrateToOptimalLocation(UnifiedBlock& block);
    
public:
    UnifiedMemoryManager();
    ~UnifiedMemoryManager();
    
    void* allocate(size_t size, AccessPattern pattern = AccessPattern::RANDOM);
    void deallocate(void* ptr);
    void prefetchToDevice(void* ptr, int device_id, cudaStream_t stream = 0);
    void prefetchToHost(void* ptr, cudaStream_t stream = 0);
    void setAccessPattern(void* ptr, AccessPattern pattern);
    
    // Performance optimization
    void optimizeLayout();
    void setMemoryAdvise(void* ptr, size_t size, cudaMemoryAdvise advice, int device);
};

// Stream-ordered memory allocator
class StreamOrderedAllocator {
private:
    struct StreamPool {
        cudaStream_t stream;
        std::queue<void*> available_blocks;
        size_t total_allocated;
        size_t currently_used;
    };
    
    std::unordered_map<cudaStream_t, StreamPool> stream_pools;
    std::mutex allocator_mutex;
    
public:
    StreamOrderedAllocator();
    ~StreamOrderedAllocator();
    
    void* allocateAsync(size_t size, cudaStream_t stream);
    void deallocateAsync(void* ptr, cudaStream_t stream);
    void synchronizeAndCleanup(cudaStream_t stream);
    void cleanupAllStreams();
};

// Hierarchical memory manager with multi-level caching
class HierarchicalMemoryManager {
private:
    // Memory hierarchy levels
    enum class MemoryLevel {
        SHARED_MEMORY,  // On-chip shared memory
        L1_CACHE,       // L1 data cache
        L2_CACHE,       // L2 cache
        GLOBAL_MEMORY,  // Global GPU memory
        UNIFIED_MEMORY, // Unified CPU/GPU memory
        HOST_MEMORY     // Host memory
    };
    
    struct CacheEntry {
        void* gpu_ptr;
        void* cache_ptr;
        size_t size;
        MemoryLevel level;
        uint64_t last_access;
        uint32_t access_count;
        bool dirty;
    };
    
    std::unordered_map<void*, CacheEntry> cache_map;
    std::vector<std::unique_ptr<MemoryPool>> level_pools;
    mutable std::mutex cache_mutex;
    
    MemoryLevel selectOptimalLevel(size_t size, AccessPattern pattern);
    void evictLRU(MemoryLevel level);
    void prefetchToLevel(void* ptr, MemoryLevel target_level);
    
public:
    HierarchicalMemoryManager();
    ~HierarchicalMemoryManager();
    
    void* allocate(size_t size, AccessPattern pattern = AccessPattern::RANDOM);
    void deallocate(void* ptr);
    void* getCachedData(void* original_ptr);
    void updateAccessPattern(void* ptr, AccessPattern pattern);
    void flush(MemoryLevel level);
    void flushAll();
    
    // Cache management
    void setCachePolicy(MemoryLevel level, size_t max_size);
    double getCacheHitRatio(MemoryLevel level) const;
    void warmupCache(const std::vector<void*>& ptrs);
};

// Main adaptive memory manager
class AdaptiveMemoryManager {
private:
    AllocationStrategy current_strategy;
    MemoryPressure current_pressure;
    
    // Strategy implementations
    std::unique_ptr<MemoryPool> pool_allocator;
    std::unique_ptr<UnifiedMemoryManager> unified_manager;
    std::unique_ptr<StreamOrderedAllocator> stream_allocator;
    std::unique_ptr<HierarchicalMemoryManager> hierarchical_manager;
    
    // Statistics and monitoring
    MemoryStats stats;
    std::vector<MemoryBlock> all_blocks;
    mutable std::mutex manager_mutex;
    
    // Performance monitoring
    cudaEvent_t alloc_start, alloc_end;
    std::vector<double> allocation_times;
    
    // Strategy selection
    AllocationStrategy selectOptimalStrategy(size_t size, AccessPattern pattern);
    void updateMemoryPressure();
    void adaptStrategy();
    
    // Defragmentation
    void defragmentMemory();
    bool shouldDefragment() const;
    
    // Memory pressure management
    void handleMemoryPressure();
    void releaseUnusedMemory();
    
public:
    AdaptiveMemoryManager();
    ~AdaptiveMemoryManager();
    
    // Primary allocation interface
    void* allocate(size_t size, AccessPattern pattern = AccessPattern::RANDOM);
    void* allocate(size_t size, size_t alignment, AccessPattern pattern = AccessPattern::RANDOM);
    void* allocateAsync(size_t size, cudaStream_t stream, AccessPattern pattern = AccessPattern::RANDOM);
    
    void deallocate(void* ptr);
    void deallocateAsync(void* ptr, cudaStream_t stream);
    
    // Memory management
    void* reallocate(void* ptr, size_t new_size);
    void memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind, 
               cudaStream_t stream = 0);
    void memset(void* ptr, int value, size_t size, cudaStream_t stream = 0);
    
    // Strategy control
    void setStrategy(AllocationStrategy strategy);
    AllocationStrategy getStrategy() const { return current_strategy; }
    void enableAdaptiveStrategy(bool enable = true);
    
    // Performance and monitoring
    const MemoryStats& getStats() const { return stats; }
    void resetStats();
    void printStats() const;
    MemoryPressure getCurrentPressure() const { return current_pressure; }
    
    // Memory optimization
    void optimizeMemoryLayout();
    void prefetchOptimal(void* ptr, size_t size, int device_id = -1);
    void setAccessHint(void* ptr, AccessPattern pattern);
    
    // Device management
    void setDevice(int device_id);
    void enablePeerAccess(int peer_device);
    void synchronizeAllDevices();
    
    // Advanced features
    void registerMemoryCallback(std::function<void(const MemoryStats&)> callback);
    void enableMemoryProfiling(bool enable = true);
    void exportMemoryTrace(const std::string& filename) const;
    
    // Memory guards for debugging
    void enableMemoryGuards(bool enable = true);
    bool checkMemoryIntegrity() const;
    void reportMemoryLeaks() const;
};

// Global memory manager instance
AdaptiveMemoryManager& getGlobalMemoryManager();

// Convenience macros for memory operations
#define CUDA_MALLOC(ptr, size) \
    (ptr = langlands::memory::getGlobalMemoryManager().allocate(size))

#define CUDA_MALLOC_ALIGNED(ptr, size, alignment) \
    (ptr = langlands::memory::getGlobalMemoryManager().allocate(size, alignment))

#define CUDA_MALLOC_ASYNC(ptr, size, stream) \
    (ptr = langlands::memory::getGlobalMemoryManager().allocateAsync(size, stream))

#define CUDA_FREE(ptr) \
    langlands::memory::getGlobalMemoryManager().deallocate(ptr)

#define CUDA_FREE_ASYNC(ptr, stream) \
    langlands::memory::getGlobalMemoryManager().deallocateAsync(ptr, stream)

#define CUDA_MEMCPY(dst, src, size, kind) \
    langlands::memory::getGlobalMemoryManager().memcpy(dst, src, size, kind)

#define CUDA_MEMCPY_ASYNC(dst, src, size, kind, stream) \
    langlands::memory::getGlobalMemoryManager().memcpy(dst, src, size, kind, stream)

#define CUDA_MEMSET(ptr, value, size) \
    langlands::memory::getGlobalMemoryManager().memset(ptr, value, size)

#define CUDA_MEMSET_ASYNC(ptr, value, size, stream) \
    langlands::memory::getGlobalMemoryManager().memset(ptr, value, size, stream)

// Memory pattern analysis utilities
namespace patterns {
    
    // Analyze memory access patterns
    AccessPattern analyzeAccessPattern(const std::vector<void*>& accesses,
                                     const std::vector<uint64_t>& timestamps);
    
    // Predict optimal allocation strategy
    AllocationStrategy predictOptimalStrategy(size_t size, AccessPattern pattern,
                                            const MemoryStats& current_stats);
    
    // Memory layout optimization
    void optimizeDataLayout(void** ptrs, size_t* sizes, size_t count,
                           AccessPattern pattern);
    
    // Prefetching strategies
    void setupOptimalPrefetching(void* ptr, size_t size, AccessPattern pattern,
                               int target_device);
}

// Memory debugging and validation
namespace debug {
    
    // Memory leak detection
    class MemoryTracker {
    private:
        struct AllocationInfo {
            size_t size;
            std::string file;
            int line;
            uint64_t timestamp;
        };
        
        std::unordered_map<void*, AllocationInfo> allocations;
        mutable std::mutex tracker_mutex;
        
    public:
        void recordAllocation(void* ptr, size_t size, const char* file, int line);
        void recordDeallocation(void* ptr);
        void reportLeaks() const;
        size_t getTotalAllocated() const;
        size_t getActiveAllocations() const;
    };
    
    extern MemoryTracker global_tracker;
    
    // Memory corruption detection
    class MemoryGuard {
    private:
        static constexpr uint32_t GUARD_PATTERN = 0xDEADBEEF;
        static constexpr size_t GUARD_SIZE = 32;
        
    public:
        static void* guardedAllocate(size_t size);
        static void guardedDeallocate(void* ptr);
        static bool checkGuards(void* ptr);
        static void checkAllGuards();
    };
    
    // Memory access validation
    void validateMemoryAccess(void* ptr, size_t size);
    void validateDevicePointer(void* ptr);
    void validateHostPointer(void* ptr);
}

} // namespace memory
} // namespace langlands

#endif // MEMORY_MANAGER_CUH