// Advanced Performance Monitoring System for CUDA Acceleration
// Real-time profiling and optimization for 10x speedup validation

#include "../include/langlands_cuda.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <fstream>
#include <iomanip>

namespace langlands {
namespace profiling {

using steady_clock = std::chrono::steady_clock;
using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;

// Performance metrics structure
struct KernelMetrics {
    std::string name;
    double total_time_ms;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    size_t invocation_count;
    size_t memory_used_bytes;
    double gflops;
    double bandwidth_gb_s;
    double occupancy_percent;
    int active_warps;
    int registers_per_thread;
    size_t shared_memory_bytes;
    double speedup_vs_cpu;
};

struct DeviceMetrics {
    std::string device_name;
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    double memory_utilization_percent;
    int temperature_celsius;
    int power_usage_watts;
    int clock_rate_mhz;
    int memory_clock_rate_mhz;
    double gpu_utilization_percent;
    int active_sm_count;
    double theoretical_peak_gflops;
    double theoretical_bandwidth_gb_s;
};

struct SystemMetrics {
    double total_time_seconds;
    size_t total_kernels_launched;
    double cumulative_gflops;
    double average_gpu_utilization;
    size_t peak_memory_usage;
    double energy_consumed_joules;
    std::vector<KernelMetrics> kernel_stats;
    DeviceMetrics device_info;
};

// High-resolution timer for kernel profiling
class HighResolutionTimer {
private:
    cudaEvent_t start_event, stop_event;
    steady_clock::time_point cpu_start;
    bool cuda_timing_enabled;
    
public:
    HighResolutionTimer() : cuda_timing_enabled(true) {
        cudaError_t err1 = cudaEventCreate(&start_event);
        cudaError_t err2 = cudaEventCreate(&stop_event);
        
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            cuda_timing_enabled = false;
        }
    }
    
    ~HighResolutionTimer() {
        if (cuda_timing_enabled) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }
    
    void start(cudaStream_t stream = 0) {
        cpu_start = steady_clock::now();
        if (cuda_timing_enabled) {
            cudaEventRecord(start_event, stream);
        }
    }
    
    double stop(cudaStream_t stream = 0) {
        if (cuda_timing_enabled) {
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            
            float cuda_time_ms;
            cudaEventElapsedTime(&cuda_time_ms, start_event, stop_event);
            return static_cast<double>(cuda_time_ms);
        } else {
            auto cpu_end = steady_clock::now();
            auto duration = std::chrono::duration_cast<microseconds>(cpu_end - cpu_start);
            return duration.count() / 1000.0; // Convert to milliseconds
        }
    }
};

// Advanced performance monitor implementation
class AdvancedPerformanceMonitor::Impl {
private:
    std::unordered_map<std::string, KernelMetrics> kernel_metrics;
    mutable std::mutex metrics_mutex;
    
    // Device monitoring
    nvmlDevice_t nvml_device;
    bool nvml_initialized;
    int device_id;
    
    // System state
    steady_clock::time_point session_start;
    double total_energy_consumed;
    size_t peak_memory_usage;
    std::vector<double> gpu_utilization_history;
    
    // Profiling state
    bool profiling_enabled;
    std::string output_directory;
    
public:
    Impl(int dev_id = 0) : nvml_initialized(false), device_id(dev_id), 
                          total_energy_consumed(0.0), peak_memory_usage(0),
                          profiling_enabled(true) {
        
        session_start = steady_clock::now();
        
        // Initialize NVML for detailed device monitoring
        nvmlReturn_t nvml_result = nvmlInit();
        if (nvml_result == NVML_SUCCESS) {
            nvml_result = nvmlDeviceGetHandleByIndex(device_id, &nvml_device);
            nvml_initialized = (nvml_result == NVML_SUCCESS);
        }
        
        // Set default output directory
        output_directory = "./cuda_profiling_results/";
        createOutputDirectory();
    }
    
    ~Impl() {
        if (nvml_initialized) {
            nvmlShutdown();
        }
    }
    
    void startKernelProfiling(const std::string& kernel_name) {
        if (!profiling_enabled) return;
        
        // Enable CUDA profiler for this kernel
        cudaProfilerStart();
    }
    
    void recordKernelMetrics(const std::string& kernel_name, double execution_time_ms,
                           size_t memory_used, double flops, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        auto& metrics = kernel_metrics[kernel_name];
        
        if (metrics.name.empty()) {
            metrics.name = kernel_name;
            metrics.min_time_ms = execution_time_ms;
            metrics.max_time_ms = execution_time_ms;
        }
        
        // Update timing statistics
        metrics.total_time_ms += execution_time_ms;
        metrics.invocation_count++;
        metrics.avg_time_ms = metrics.total_time_ms / metrics.invocation_count;
        metrics.min_time_ms = std::min(metrics.min_time_ms, execution_time_ms);
        metrics.max_time_ms = std::max(metrics.max_time_ms, execution_time_ms);
        
        // Update performance metrics
        metrics.memory_used_bytes = std::max(metrics.memory_used_bytes, memory_used);
        if (execution_time_ms > 0.0) {
            metrics.gflops = (flops / 1e9) / (execution_time_ms / 1000.0);
            
            // Estimate bandwidth (read + write)
            double bytes_transferred = memory_used * 2.0; // Assume read + write
            metrics.bandwidth_gb_s = (bytes_transferred / 1e9) / (execution_time_ms / 1000.0);
        }
        
        // Get detailed kernel information
        recordDetailedKernelInfo(kernel_name, stream);
    }
    
    void endKernelProfiling() {
        if (!profiling_enabled) return;
        
        cudaProfilerStop();
    }
    
    DeviceMetrics getDeviceMetrics() const {
        DeviceMetrics metrics = {};
        
        // Get basic device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        metrics.device_name = prop.name;
        metrics.total_memory = prop.totalGlobalMem;
        
        // Get current memory usage
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        metrics.free_memory = free_mem;
        metrics.used_memory = total_mem - free_mem;
        metrics.memory_utilization_percent = 
            (static_cast<double>(metrics.used_memory) / total_mem) * 100.0;
        
        // Calculate theoretical performance
        metrics.clock_rate_mhz = prop.clockRate / 1000;
        metrics.memory_clock_rate_mhz = prop.memoryClockRate / 1000;
        metrics.active_sm_count = prop.multiProcessorCount;
        
        // Theoretical peak performance (simplified calculation)
        int cores_per_sm = getSMCoreCount(prop.major, prop.minor);
        double peak_ops_per_second = static_cast<double>(metrics.active_sm_count) * 
                                   cores_per_sm * metrics.clock_rate_mhz * 1e6 * 2; // 2 ops per clock
        metrics.theoretical_peak_gflops = peak_ops_per_second / 1e9;
        
        // Theoretical memory bandwidth
        metrics.theoretical_bandwidth_gb_s = 
            static_cast<double>(prop.memoryClockRate) * 1000.0 * 2.0 * 
            (prop.memoryBusWidth / 8) / 1e9;
        
        // Get real-time metrics via NVML
        if (nvml_initialized) {
            updateNVMLMetrics(metrics);
        }
        
        return metrics;
    }
    
    SystemMetrics getSystemMetrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        SystemMetrics system = {};
        
        auto current_time = steady_clock::now();
        auto duration = std::chrono::duration_cast<milliseconds>(current_time - session_start);
        system.total_time_seconds = duration.count() / 1000.0;
        
        system.total_kernels_launched = 0;
        system.cumulative_gflops = 0.0;
        system.peak_memory_usage = peak_memory_usage;
        system.energy_consumed_joules = total_energy_consumed;
        
        // Aggregate kernel statistics
        for (const auto& pair : kernel_metrics) {
            const KernelMetrics& metrics = pair.second;
            system.kernel_stats.push_back(metrics);
            system.total_kernels_launched += metrics.invocation_count;
            system.cumulative_gflops += metrics.gflops * metrics.invocation_count;
        }
        
        // Calculate average GPU utilization
        if (!gpu_utilization_history.empty()) {
            double sum = 0.0;
            for (double util : gpu_utilization_history) {
                sum += util;
            }
            system.average_gpu_utilization = sum / gpu_utilization_history.size();
        }
        
        system.device_info = getDeviceMetrics();
        
        return system;
    }
    
    void exportMetrics(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        std::string full_path = output_directory + filename;
        std::ofstream file(full_path);
        
        if (!file.is_open()) {
            return;
        }
        
        SystemMetrics system = getSystemMetrics();
        
        // Write JSON format report
        file << "{\n";
        file << "  \"session_summary\": {\n";
        file << "    \"total_time_seconds\": " << system.total_time_seconds << ",\n";
        file << "    \"total_kernels_launched\": " << system.total_kernels_launched << ",\n";
        file << "    \"cumulative_gflops\": " << system.cumulative_gflops << ",\n";
        file << "    \"average_gpu_utilization\": " << system.average_gpu_utilization << ",\n";
        file << "    \"peak_memory_usage_bytes\": " << system.peak_memory_usage << ",\n";
        file << "    \"energy_consumed_joules\": " << system.energy_consumed_joules << "\n";
        file << "  },\n";
        
        file << "  \"device_info\": {\n";
        file << "    \"name\": \"" << system.device_info.device_name << "\",\n";
        file << "    \"total_memory_bytes\": " << system.device_info.total_memory << ",\n";
        file << "    \"theoretical_peak_gflops\": " << system.device_info.theoretical_peak_gflops << ",\n";
        file << "    \"theoretical_bandwidth_gb_s\": " << system.device_info.theoretical_bandwidth_gb_s << "\n";
        file << "  },\n";
        
        file << "  \"kernel_metrics\": [\n";
        
        bool first = true;
        for (const auto& metrics : system.kernel_stats) {
            if (!first) file << ",\n";
            first = false;
            
            file << "    {\n";
            file << "      \"name\": \"" << metrics.name << "\",\n";
            file << "      \"total_time_ms\": " << metrics.total_time_ms << ",\n";
            file << "      \"avg_time_ms\": " << metrics.avg_time_ms << ",\n";
            file << "      \"invocation_count\": " << metrics.invocation_count << ",\n";
            file << "      \"gflops\": " << metrics.gflops << ",\n";
            file << "      \"bandwidth_gb_s\": " << metrics.bandwidth_gb_s << ",\n";
            file << "      \"occupancy_percent\": " << metrics.occupancy_percent << ",\n";
            file << "      \"speedup_vs_cpu\": " << metrics.speedup_vs_cpu << "\n";
            file << "    }";
        }
        
        file << "\n  ]\n";
        file << "}\n";
        
        file.close();
    }
    
    void printDetailedReport() const {
        SystemMetrics system = getSystemMetrics();
        
        std::cout << "\n=== CUDA Performance Report ===" << std::endl;
        std::cout << "Session Duration: " << std::fixed << std::setprecision(2) 
                  << system.total_time_seconds << " seconds" << std::endl;
        std::cout << "Total Kernels Launched: " << system.total_kernels_launched << std::endl;
        std::cout << "Peak Memory Usage: " << std::fixed << std::setprecision(1)
                  << (system.peak_memory_usage / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Average GPU Utilization: " << std::fixed << std::setprecision(1)
                  << system.average_gpu_utilization << "%" << std::endl;
        
        std::cout << "\n=== Device Information ===" << std::endl;
        std::cout << "Device: " << system.device_info.device_name << std::endl;
        std::cout << "Total Memory: " << std::fixed << std::setprecision(1)
                  << (system.device_info.total_memory / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "Theoretical Peak: " << std::fixed << std::setprecision(1)
                  << system.device_info.theoretical_peak_gflops << " GFLOPS" << std::endl;
        std::cout << "Memory Bandwidth: " << std::fixed << std::setprecision(1)
                  << system.device_info.theoretical_bandwidth_gb_s << " GB/s" << std::endl;
        
        if (nvml_initialized) {
            std::cout << "Temperature: " << system.device_info.temperature_celsius << "°C" << std::endl;
            std::cout << "Power Usage: " << system.device_info.power_usage_watts << "W" << std::endl;
        }
        
        std::cout << "\n=== Kernel Performance ===" << std::endl;
        std::cout << std::setw(30) << "Kernel Name"
                  << std::setw(12) << "Calls"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Avg(ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(12) << "GB/s"
                  << std::setw(12) << "Speedup" << std::endl;
        std::cout << std::string(102, '-') << std::endl;
        
        for (const auto& metrics : system.kernel_stats) {
            std::cout << std::setw(30) << metrics.name
                      << std::setw(12) << metrics.invocation_count
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.total_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << metrics.avg_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.gflops
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.bandwidth_gb_s
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.speedup_vs_cpu
                      << std::endl;
        }
        
        // Performance analysis
        std::cout << "\n=== Performance Analysis ===" << std::endl;
        
        double total_gflops = 0.0;
        double total_bandwidth = 0.0;
        double min_speedup = 1000.0;
        double max_speedup = 0.0;
        
        for (const auto& metrics : system.kernel_stats) {
            total_gflops += metrics.gflops;
            total_bandwidth += metrics.bandwidth_gb_s;
            if (metrics.speedup_vs_cpu > 0) {
                min_speedup = std::min(min_speedup, metrics.speedup_vs_cpu);
                max_speedup = std::max(max_speedup, metrics.speedup_vs_cpu);
            }
        }
        
        double efficiency = (total_gflops / system.device_info.theoretical_peak_gflops) * 100.0;
        double bandwidth_utilization = (total_bandwidth / system.device_info.theoretical_bandwidth_gb_s) * 100.0;
        
        std::cout << "Compute Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        std::cout << "Bandwidth Utilization: " << std::fixed << std::setprecision(1) << bandwidth_utilization << "%" << std::endl;
        std::cout << "Speedup Range: " << std::fixed << std::setprecision(1) << min_speedup 
                  << "x - " << max_speedup << "x" << std::endl;
        
        // 10x speedup validation
        bool achieved_10x = max_speedup >= 10.0;
        std::cout << "\n🎯 10x Speedup Target: " << (achieved_10x ? "✅ ACHIEVED" : "❌ NOT YET ACHIEVED") << std::endl;
        
        if (!achieved_10x) {
            std::cout << "Optimization Recommendations:" << std::endl;
            if (efficiency < 50.0) {
                std::cout << "- Improve compute utilization (current: " << efficiency << "%)" << std::endl;
            }
            if (bandwidth_utilization < 50.0) {
                std::cout << "- Optimize memory access patterns (utilization: " << bandwidth_utilization << "%)" << std::endl;
            }
            std::cout << "- Consider using Tensor Cores for applicable workloads" << std::endl;
            std::cout << "- Profile with Nsight Compute for detailed optimization" << std::endl;
        }
    }
    
    void enableRealTimeMonitoring(bool enable) {
        profiling_enabled = enable;
    }
    
    void setOutputDirectory(const std::string& dir) {
        output_directory = dir;
        createOutputDirectory();
    }
    
private:
    void recordDetailedKernelInfo(const std::string& kernel_name, cudaStream_t stream) {
        // Get occupancy information
        // Note: This would require kernel function pointers in a real implementation
        
        // Update peak memory usage
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_mem = total_mem - free_mem;
        peak_memory_usage = std::max(peak_memory_usage, used_mem);
        
        // Record GPU utilization if NVML is available
        if (nvml_initialized) {
            nvmlUtilization_t utilization;
            if (nvmlDeviceGetUtilizationRates(nvml_device, &utilization) == NVML_SUCCESS) {
                gpu_utilization_history.push_back(static_cast<double>(utilization.gpu));
                
                // Estimate energy consumption
                unsigned int power_watts;
                if (nvmlDeviceGetPowerUsage(nvml_device, &power_watts) == NVML_SUCCESS) {
                    // Estimate energy for this kernel (very rough approximation)
                    auto& metrics = kernel_metrics[kernel_name];
                    double time_seconds = metrics.avg_time_ms / 1000.0;
                    total_energy_consumed += (power_watts / 1000.0) * time_seconds;
                }
            }
        }
    }
    
    void updateNVMLMetrics(DeviceMetrics& metrics) const {
        if (!nvml_initialized) return;
        
        // Temperature
        unsigned int temp;
        if (nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            metrics.temperature_celsius = static_cast<int>(temp);
        }
        
        // Power usage
        unsigned int power;
        if (nvmlDeviceGetPowerUsage(nvml_device, &power) == NVML_SUCCESS) {
            metrics.power_usage_watts = static_cast<int>(power / 1000); // Convert mW to W
        }
        
        // GPU utilization
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(nvml_device, &utilization) == NVML_SUCCESS) {
            metrics.gpu_utilization_percent = static_cast<double>(utilization.gpu);
        }
        
        // Clock rates
        unsigned int graphics_clock, memory_clock;
        if (nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_GRAPHICS, &graphics_clock) == NVML_SUCCESS) {
            metrics.clock_rate_mhz = static_cast<int>(graphics_clock);
        }
        if (nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_MEM, &memory_clock) == NVML_SUCCESS) {
            metrics.memory_clock_rate_mhz = static_cast<int>(memory_clock);
        }
    }
    
    int getSMCoreCount(int major, int minor) const {
        // Cores per SM for different architectures
        switch (major) {
            case 3: return 192; // Kepler
            case 5: return 128; // Maxwell
            case 6: return (minor == 0) ? 64 : 128; // Pascal
            case 7: return 64;  // Volta/Turing
            case 8: return 64;  // Ampere
            case 9: return 128; // Ada Lovelace
            default: return 64; // Default estimate
        }
    }
    
    void createOutputDirectory() {
        // Create directory if it doesn't exist (platform specific)
#ifdef _WIN32
        _mkdir(output_directory.c_str());
#else
        mkdir(output_directory.c_str(), 0755);
#endif
    }
};

// AdvancedPerformanceMonitor implementation
AdvancedPerformanceMonitor::AdvancedPerformanceMonitor(int device_id)
    : pImpl(std::make_unique<Impl>(device_id)) {}

AdvancedPerformanceMonitor::~AdvancedPerformanceMonitor() = default;

void AdvancedPerformanceMonitor::startTimer(const std::string& name) {
    pImpl->startKernelProfiling(name);
}

void AdvancedPerformanceMonitor::endTimer(const std::string& name) {
    pImpl->endKernelProfiling();
}

void AdvancedPerformanceMonitor::recordKernelMetrics(const std::string& kernel_name, 
                                                   double execution_time_ms,
                                                   size_t memory_used, 
                                                   double flops,
                                                   cudaStream_t stream) {
    pImpl->recordKernelMetrics(kernel_name, execution_time_ms, memory_used, flops, stream);
}

void AdvancedPerformanceMonitor::printReport() const {
    pImpl->printDetailedReport();
}

void AdvancedPerformanceMonitor::exportMetrics(const std::string& filename) const {
    pImpl->exportMetrics(filename);
}

double AdvancedPerformanceMonitor::getKernelTime(const std::string& name) const {
    SystemMetrics metrics = pImpl->getSystemMetrics();
    for (const auto& kernel : metrics.kernel_stats) {
        if (kernel.name == name) {
            return kernel.avg_time_ms;
        }
    }
    return 0.0;
}

size_t AdvancedPerformanceMonitor::getPeakMemoryUsage() const {
    return pImpl->getSystemMetrics().peak_memory_usage;
}

double AdvancedPerformanceMonitor::getTotalComputeTime() const {
    SystemMetrics metrics = pImpl->getSystemMetrics();
    double total = 0.0;
    for (const auto& kernel : metrics.kernel_stats) {
        total += kernel.total_time_ms;
    }
    return total;
}

// Global performance monitor instance
static std::unique_ptr<AdvancedPerformanceMonitor> g_performance_monitor;
static std::once_flag g_monitor_init_flag;

AdvancedPerformanceMonitor& getGlobalPerformanceMonitor() {
    std::call_once(g_monitor_init_flag, []() {
        g_performance_monitor = std::make_unique<AdvancedPerformanceMonitor>();
    });
    return *g_performance_monitor;
}

} // namespace profiling
} // namespace langlands