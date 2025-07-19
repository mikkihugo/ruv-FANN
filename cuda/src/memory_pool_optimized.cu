// Optimized CUDA Memory Pool Implementation
// High-performance memory management for 10x speedup target

#include "../include/memory_manager.cuh"
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cassert>

namespace langlands {
namespace memory {

// Memory pool block alignment (64 bytes for optimal coalescing)
constexpr size_t MEMORY_ALIGNMENT = 64;

// Pool size configuration optimized for geometric computations
constexpr size_t SMALL_BLOCK_SIZE = 1024;           // 1KB blocks
constexpr size_t MEDIUM_BLOCK_SIZE = 1024 * 1024;   // 1MB blocks  
constexpr size_t LARGE_BLOCK_SIZE = 16 * 1024 * 1024; // 16MB blocks

// Pool limits for different block sizes
constexpr size_t MAX_SMALL_BLOCKS = 1024;
constexpr size_t MAX_MEDIUM_BLOCKS = 256;
constexpr size_t MAX_LARGE_BLOCKS = 64;

// Timing utilities for performance monitoring
using steady_clock = std::chrono::steady_clock;
using microseconds = std::chrono::microseconds;

inline uint64_t getCurrentTimestamp() {
    return steady_clock::now().time_since_epoch().count();
}

// High-performance memory pool implementation
class OptimizedMemoryPool::Impl {
private:
    // Memory block metadata
    struct BlockInfo {
        void* ptr;
        size_t size;
        size_t actual_size; // Including alignment padding
        bool in_use;
        uint64_t allocation_time;
        uint64_t last_access_time;
        AccessPattern access_pattern;
        int device_id;
        cudaStream_t stream;
        
        BlockInfo() : ptr(nullptr), size(0), actual_size(0), in_use(false),
                     allocation_time(0), last_access_time(0), 
                     access_pattern(AccessPattern::RANDOM),
                     device_id(0), stream(0) {}
    };
    
    // Thread-safe pool for specific block sizes
    template<size_t BlockSize>
    class TypedPool {
    private:
        std::vector<BlockInfo> blocks;
        std::stack<size_t> available_indices;
        std::mutex pool_mutex;
        std::atomic<size_t> peak_usage{0};
        std::atomic<size_t> current_usage{0};
        std::atomic<size_t> allocation_count{0};
        size_t max_blocks;
        int device_id;
        
    public:
        TypedPool(size_t max_blks, int dev_id) 
            : max_blocks(max_blks), device_id(dev_id) {
            blocks.reserve(max_blocks);
            
            // Pre-allocate blocks for better performance
            preallocateBlocks();
        }
        
        ~TypedPool() {
            cleanup();
        }
        
        void* allocate(AccessPattern pattern = AccessPattern::RANDOM) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            
            if (available_indices.empty()) {
                if (blocks.size() >= max_blocks) {
                    // Try to evict unused blocks
                    evictOldBlocks();
                    if (available_indices.empty()) {
                        return nullptr; // Pool exhausted
                    }
                } else {
                    // Allocate new block
                    if (!allocateNewBlock()) {
                        return nullptr;
                    }
                }
            }
            
            size_t index = available_indices.top();
            available_indices.pop();
            
            BlockInfo& block = blocks[index];
            block.in_use = true;
            block.allocation_time = getCurrentTimestamp();
            block.last_access_time = block.allocation_time;
            block.access_pattern = pattern;
            
            // Apply memory advice based on access pattern
            applyMemoryAdvice(block.ptr, BlockSize, pattern);
            
            current_usage.fetch_add(1);
            allocation_count.fetch_add(1);
            
            size_t current = current_usage.load();
            size_t peak = peak_usage.load();
            while (current > peak && !peak_usage.compare_exchange_weak(peak, current)) {
                // Spin until we successfully update peak usage
            }
            
            return block.ptr;
        }
        
        bool deallocate(void* ptr) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            
            auto it = std::find_if(blocks.begin(), blocks.end(),
                [ptr](const BlockInfo& block) { return block.ptr == ptr; });
            
            if (it != blocks.end() && it->in_use) {
                it->in_use = false;
                it->last_access_time = getCurrentTimestamp();
                
                size_t index = std::distance(blocks.begin(), it);
                available_indices.push(index);
                
                current_usage.fetch_sub(1);
                return true;
            }
            
            return false;
        }
        
        size_t getCurrentUsage() const { return current_usage.load(); }
        size_t getPeakUsage() const { return peak_usage.load(); }
        size_t getAllocationCount() const { return allocation_count.load(); }
        
        void printStats() const {
            std::lock_guard<std::mutex> lock(pool_mutex);
            printf("Pool Stats [%zuB blocks]:\n", BlockSize);
            printf("  Current usage: %zu/%zu\n", current_usage.load(), max_blocks);
            printf("  Peak usage: %zu\n", peak_usage.load());
            printf("  Total allocations: %zu\n", allocation_count.load());
            printf("  Available blocks: %zu\n", available_indices.size());
        }
        
    private:
        void preallocateBlocks() {
            // Pre-allocate a portion of blocks for better performance
            size_t prealloc_count = std::min(max_blocks / 4, static_cast<size_t>(32));
            
            for (size_t i = 0; i < prealloc_count; ++i) {
                if (!allocateNewBlock()) {
                    break;
                }
            }
        }
        
        bool allocateNewBlock() {
            void* ptr;
            size_t aligned_size = alignSize(BlockSize);
            
            cudaError_t error = cudaMalloc(&ptr, aligned_size);
            if (error != cudaSuccess) {
                return false;
            }
            
            // Initialize block
            BlockInfo block;
            block.ptr = ptr;
            block.size = BlockSize;
            block.actual_size = aligned_size;
            block.device_id = device_id;
            
            size_t index = blocks.size();
            blocks.push_back(block);
            available_indices.push(index);
            
            return true;
        }
        
        void evictOldBlocks() {
            // Find and evict blocks that haven't been used recently
            uint64_t current_time = getCurrentTimestamp();
            constexpr uint64_t eviction_threshold = 1000000000; // 1 second in nanoseconds
            
            for (size_t i = 0; i < blocks.size(); ++i) {
                BlockInfo& block = blocks[i];
                if (!block.in_use && 
                    (current_time - block.last_access_time) > eviction_threshold) {
                    
                    // Free the block and make it available
                    cudaFree(block.ptr);
                    block.ptr = nullptr;
                    available_indices.push(i);
                    break; // Only evict one block at a time
                }
            }
        }
        
        void applyMemoryAdvice(void* ptr, size_t size, AccessPattern pattern) {
#ifdef CUDA_VERSION_MAJOR
#if CUDA_VERSION_MAJOR >= 8
            // Apply memory advice based on access pattern
            switch (pattern) {
                case AccessPattern::SEQUENTIAL:
                    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
                    break;
                case AccessPattern::RANDOM:
                    cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device_id);
                    break;
                case AccessPattern::STREAMING:
                    cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device_id);
                    break;
                default:
                    break;
            }
#endif
#endif
        }
        
        size_t alignSize(size_t size) {
            return (size + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
        }
        
        void cleanup() {
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (auto& block : blocks) {
                if (block.ptr) {
                    cudaFree(block.ptr);
                }
            }
            blocks.clear();
            while (!available_indices.empty()) {
                available_indices.pop();
            }
        }
    };
    
    // Specialized pools for different block sizes
    std::unique_ptr<TypedPool<SMALL_BLOCK_SIZE>> small_pool;
    std::unique_ptr<TypedPool<MEDIUM_BLOCK_SIZE>> medium_pool;
    std::unique_ptr<TypedPool<LARGE_BLOCK_SIZE>> large_pool;
    
    // Fallback allocator for sizes that don't fit in pools
    std::unordered_map<void*, size_t> large_allocations;
    std::mutex large_alloc_mutex;
    
    // Statistics
    mutable std::mutex stats_mutex;
    MemoryStats stats;
    
    // Device management
    int device_id;
    bool initialized;
    
public:
    Impl(int dev_id = 0) : device_id(dev_id), initialized(false) {
        initialize();
    }
    
    ~Impl() {
        cleanup();
    }
    
    bool initialize() {
        if (initialized) return true;
        
        // Set device
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            return false;
        }
        
        // Create pools
        small_pool = std::make_unique<TypedPool<SMALL_BLOCK_SIZE>>(
            MAX_SMALL_BLOCKS, device_id);
        medium_pool = std::make_unique<TypedPool<MEDIUM_BLOCK_SIZE>>(
            MAX_MEDIUM_BLOCKS, device_id);
        large_pool = std::make_unique<TypedPool<LARGE_BLOCK_SIZE>>(
            MAX_LARGE_BLOCKS, device_id);
        
        initialized = true;
        return true;
    }
    
    void* allocate(size_t size, AccessPattern pattern = AccessPattern::RANDOM) {
        if (!initialized && !initialize()) {
            return nullptr;
        }
        
        auto start_time = steady_clock::now();
        void* ptr = nullptr;
        
        // Route to appropriate pool based on size
        if (size <= SMALL_BLOCK_SIZE) {
            ptr = small_pool->allocate(pattern);
        } else if (size <= MEDIUM_BLOCK_SIZE) {
            ptr = medium_pool->allocate(pattern);
        } else if (size <= LARGE_BLOCK_SIZE) {
            ptr = large_pool->allocate(pattern);
        }
        
        // Fallback to direct allocation for very large sizes
        if (!ptr && size > LARGE_BLOCK_SIZE) {
            ptr = directAllocate(size, pattern);
        }
        
        // Update statistics
        if (ptr) {
            auto end_time = steady_clock::now();
            auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
            
            updateStats(size, duration.count(), true);
        }
        
        return ptr;
    }
    
    bool deallocate(void* ptr) {
        if (!ptr || !initialized) return false;
        
        auto start_time = steady_clock::now();
        bool success = false;
        
        // Try pools first
        if (small_pool->deallocate(ptr) ||
            medium_pool->deallocate(ptr) ||
            large_pool->deallocate(ptr)) {
            success = true;
        } else {
            // Try large allocations
            success = directDeallocate(ptr);
        }
        
        if (success) {
            auto end_time = steady_clock::now();
            auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
            
            updateStats(0, duration.count(), false);
        }
        
        return success;
    }
    
    const MemoryStats& getStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // Update pool statistics
        stats.currently_used = small_pool->getCurrentUsage() * SMALL_BLOCK_SIZE +
                              medium_pool->getCurrentUsage() * MEDIUM_BLOCK_SIZE +
                              large_pool->getCurrentUsage() * LARGE_BLOCK_SIZE;
        
        stats.peak_usage = std::max({
            small_pool->getPeakUsage() * SMALL_BLOCK_SIZE,
            medium_pool->getPeakUsage() * MEDIUM_BLOCK_SIZE,
            large_pool->getPeakUsage() * LARGE_BLOCK_SIZE
        });
        
        return stats;
    }
    
    void printDetailedStats() const {
        printf("=== Optimized Memory Pool Statistics ===\n");
        small_pool->printStats();
        medium_pool->printStats();
        large_pool->printStats();
        
        std::lock_guard<std::mutex> lock(large_alloc_mutex);
        printf("Large allocations: %zu\n", large_allocations.size());
        
        const auto& s = getStats();
        printf("Total statistics:\n");
        printf("  Currently used: %.2f MB\n", s.currently_used / (1024.0 * 1024.0));
        printf("  Peak usage: %.2f MB\n", s.peak_usage / (1024.0 * 1024.0));
        printf("  Cache hit ratio: %.2f%%\n", 
               s.allocation_count > 0 ? (s.cache_hits * 100.0 / s.allocation_count) : 0.0);
        printf("  Avg allocation time: %.2f μs\n", s.average_allocation_time_us);
    }
    
    void resetStats() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.reset();
    }
    
private:
    void* directAllocate(size_t size, AccessPattern pattern) {
        void* ptr;
        cudaError_t error = cudaMalloc(&ptr, size);
        if (error != cudaSuccess) {
            return nullptr;
        }
        
        {
            std::lock_guard<std::mutex> lock(large_alloc_mutex);
            large_allocations[ptr] = size;
        }
        
        // Apply memory advice for large allocations
        switch (pattern) {
            case AccessPattern::STREAMING:
                cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device_id);
                break;
            case AccessPattern::SEQUENTIAL:
                cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
                break;
            default:
                break;
        }
        
        return ptr;
    }
    
    bool directDeallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(large_alloc_mutex);
        
        auto it = large_allocations.find(ptr);
        if (it != large_allocations.end()) {
            cudaFree(ptr);
            large_allocations.erase(it);
            return true;
        }
        
        return false;
    }
    
    void updateStats(size_t size, long duration_us, bool is_allocation) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        if (is_allocation) {
            stats.allocation_count++;
            stats.total_allocated += size;
            
            // Update average allocation time
            double total_time = stats.average_allocation_time_us * (stats.allocation_count - 1);
            stats.average_allocation_time_us = (total_time + duration_us) / stats.allocation_count;
        } else {
            stats.deallocation_count++;
        }
    }
    
    void cleanup() {
        small_pool.reset();
        medium_pool.reset();
        large_pool.reset();
        
        // Clean up large allocations
        std::lock_guard<std::mutex> lock(large_alloc_mutex);
        for (auto& pair : large_allocations) {
            cudaFree(pair.first);
        }
        large_allocations.clear();
        
        initialized = false;
    }
};

// OptimizedMemoryPool implementation
OptimizedMemoryPool::OptimizedMemoryPool(int device_id) 
    : pImpl(std::make_unique<Impl>(device_id)) {}

OptimizedMemoryPool::~OptimizedMemoryPool() = default;

void* OptimizedMemoryPool::allocate(size_t size, AccessPattern pattern) {
    return pImpl->allocate(size, pattern);
}

bool OptimizedMemoryPool::deallocate(void* ptr) {
    return pImpl->deallocate(ptr);
}

const MemoryStats& OptimizedMemoryPool::getStats() const {
    return pImpl->getStats();
}

void OptimizedMemoryPool::printStats() const {
    pImpl->printDetailedStats();
}

void OptimizedMemoryPool::resetStats() {
    pImpl->resetStats();
}

// Global optimized memory manager instance
static std::unique_ptr<OptimizedMemoryPool> g_optimized_pool;
static std::once_flag g_pool_init_flag;

OptimizedMemoryPool& getOptimizedMemoryPool() {
    std::call_once(g_pool_init_flag, []() {
        g_optimized_pool = std::make_unique<OptimizedMemoryPool>();
    });
    return *g_optimized_pool;
}

} // namespace memory
} // namespace langlands