// CUDA Benchmarking Suite for Geometric Langlands
// Performance testing and optimization validation

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "../include/langlands_cuda.h"

namespace langlands {
namespace benchmarks {

using namespace std::chrono;

// Benchmark configuration
struct BenchmarkConfig {
    int warmup_iterations = 10;
    int benchmark_iterations = 100;
    bool use_tensor_cores = true;
    bool use_fp16 = true;
    int device_id = 0;
};

// Benchmark results
struct BenchmarkResult {
    std::string name;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double throughput_gops;
    double bandwidth_gb_s;
    size_t memory_used_bytes;
    int compute_intensity;
};

class CudaBenchmarker {
private:
    BenchmarkConfig config;
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;
    
    // Helper to measure kernel execution time
    template<typename Func>
    double measureKernelTime(Func kernel_func, int iterations) {
        // Warmup
        for (int i = 0; i < config.warmup_iterations; ++i) {
            kernel_func();
        }
        
        // Synchronize before measurement
        cudaStreamSynchronize(stream);
        
        // Time measurement
        cudaEventRecord(start_event, stream);
        
        for (int i = 0; i < iterations; ++i) {
            kernel_func();
        }
        
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        
        return elapsed_ms / iterations;
    }
    
public:
    CudaBenchmarker(const BenchmarkConfig& cfg = BenchmarkConfig()) 
        : config(cfg) {
        cudaSetDevice(config.device_id);
        cublasCreate(&cublas_handle);
        cudaStreamCreate(&stream);
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        // Set cublas to use our stream
        cublasSetStream(cublas_handle, stream);
    }
    
    ~CudaBenchmarker() {
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    // Benchmark matrix multiplication
    BenchmarkResult benchmarkMatrixMultiply(int M, int N, int K) {
        BenchmarkResult result;
        result.name = "Matrix Multiply " + std::to_string(M) + "x" + 
                     std::to_string(N) + "x" + std::to_string(K);
        
        // Allocate memory
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_A, M * K);
        curandGenerateUniform(gen, d_B, K * N);
        
        // Benchmark kernel
        auto kernel_func = [&]() {
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        };
        
        // Run benchmarks
        std::vector<double> times;
        for (int i = 0; i < 10; ++i) {
            double time = measureKernelTime(kernel_func, config.benchmark_iterations);
            times.push_back(time);
        }
        
        // Calculate statistics
        result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        result.min_time_ms = *std::min_element(times.begin(), times.end());
        result.max_time_ms = *std::max_element(times.begin(), times.end());
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double t : times) {
            variance += (t - result.avg_time_ms) * (t - result.avg_time_ms);
        }
        result.std_dev_ms = sqrt(variance / times.size());
        
        // Calculate throughput
        double flops = 2.0 * M * N * K;
        result.throughput_gops = (flops / 1e9) / (result.avg_time_ms / 1000.0);
        
        // Calculate bandwidth (read A + read B + write C)
        double bytes = size_A + size_B + size_C;
        result.bandwidth_gb_s = (bytes / 1e9) / (result.avg_time_ms / 1000.0);
        
        result.memory_used_bytes = size_A + size_B + size_C;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Benchmark tensor contraction
    BenchmarkResult benchmarkTensorContraction(int d1, int d2, int d3, int d4) {
        BenchmarkResult result;
        result.name = "Tensor Contraction " + std::to_string(d1) + "x" + 
                     std::to_string(d2) + "x" + std::to_string(d3) + "x" + 
                     std::to_string(d4);
        
        size_t size_T1 = d1 * d2 * d3 * sizeof(float);
        size_t size_T2 = d2 * d3 * d4 * sizeof(float);
        size_t size_out = d1 * d4 * sizeof(float);
        
        float *d_T1, *d_T2, *d_out;
        cudaMalloc(&d_T1, size_T1);
        cudaMalloc(&d_T2, size_T2);
        cudaMalloc(&d_out, size_out);
        
        // Initialize
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_T1, d1 * d2 * d3);
        curandGenerateUniform(gen, d_T2, d2 * d3 * d4);
        
        // Configure kernel
        dim3 block(256);
        dim3 grid((d1 * d4 + block.x - 1) / block.x);
        
        auto kernel_func = [&]() {
            cuda::tensorContraction<<<grid, block, 0, stream>>>(
                d_T1, d_T2, d_out, d1, d2, d3, d4);
        };
        
        // Benchmark
        double time = measureKernelTime(kernel_func, config.benchmark_iterations);
        
        result.avg_time_ms = time;
        result.min_time_ms = time;
        result.max_time_ms = time;
        result.std_dev_ms = 0.0;
        
        // Performance metrics
        double flops = 2.0 * d1 * d2 * d3 * d4;
        result.throughput_gops = (flops / 1e9) / (time / 1000.0);
        
        double bytes = size_T1 + size_T2 + size_out;
        result.bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0);
        
        result.memory_used_bytes = bytes;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(d_T1);
        cudaFree(d_T2);
        cudaFree(d_out);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Benchmark neural network forward propagation
    BenchmarkResult benchmarkNeuralForward(int batch_size, int input_dim, int output_dim) {
        BenchmarkResult result;
        result.name = "Neural Forward Pass B=" + std::to_string(batch_size) + 
                     " I=" + std::to_string(input_dim) + 
                     " O=" + std::to_string(output_dim);
        
        // Use FP16 if available and requested
        if (config.use_fp16 && config.use_tensor_cores) {
            return benchmarkNeuralForwardFP16(batch_size, input_dim, output_dim);
        }
        
        size_t input_size = batch_size * input_dim * sizeof(float);
        size_t weight_size = input_dim * output_dim * sizeof(float);
        size_t output_size = batch_size * output_dim * sizeof(float);
        size_t bias_size = output_dim * sizeof(float);
        
        float *d_input, *d_weights, *d_bias, *d_output;
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_weights, weight_size);
        cudaMalloc(&d_bias, bias_size);
        cudaMalloc(&d_output, output_size);
        
        // Initialize
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_input, batch_size * input_dim);
        curandGenerateUniform(gen, d_weights, input_dim * output_dim);
        curandGenerateUniform(gen, d_bias, output_dim);
        
        auto kernel_func = [&]() {
            cuda::NeuralOperations::forwardPropagation(
                d_input, d_weights, d_bias, d_output,
                batch_size, input_dim, output_dim,
                cuda::NeuralOperations::GELU, stream);
        };
        
        double time = measureKernelTime(kernel_func, config.benchmark_iterations);
        
        result.avg_time_ms = time;
        result.min_time_ms = time;
        result.max_time_ms = time;
        result.std_dev_ms = 0.0;
        
        // Performance metrics
        double flops = 2.0 * batch_size * input_dim * output_dim + 
                      batch_size * output_dim; // GEMM + bias
        result.throughput_gops = (flops / 1e9) / (time / 1000.0);
        
        double bytes = input_size + weight_size + bias_size + output_size;
        result.bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0);
        
        result.memory_used_bytes = bytes;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Benchmark with FP16 and Tensor Cores
    BenchmarkResult benchmarkNeuralForwardFP16(int batch_size, int input_dim, int output_dim) {
        BenchmarkResult result;
        result.name = "Neural Forward Pass FP16/TC B=" + std::to_string(batch_size) + 
                     " I=" + std::to_string(input_dim) + 
                     " O=" + std::to_string(output_dim);
        
        // Ensure dimensions are aligned for Tensor Cores
        int aligned_batch = ((batch_size + 15) / 16) * 16;
        int aligned_input = ((input_dim + 15) / 16) * 16;
        int aligned_output = ((output_dim + 15) / 16) * 16;
        
        size_t input_size = aligned_batch * aligned_input * sizeof(half);
        size_t weight_size = aligned_input * aligned_output * sizeof(half);
        size_t output_size = aligned_batch * aligned_output * sizeof(half);
        size_t bias_size = aligned_output * sizeof(half);
        
        half *d_input, *d_weights, *d_bias, *d_output;
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_weights, weight_size);
        cudaMalloc(&d_bias, bias_size);
        cudaMalloc(&d_output, output_size);
        
        // Initialize with random data (using float then converting)
        float *temp_float;
        cudaMalloc(&temp_float, std::max({input_size, weight_size, output_size}));
        
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        
        // Convert to FP16
        auto float_to_half = [](float* src, half* dst, int n) {
            for (int i = 0; i < n; ++i) {
                dst[i] = __float2half(src[i]);
            }
        };
        
        // Time only the kernel execution
        auto kernel_func = [&]() {
            cuda::NeuralOperations::forwardPropagationTC(
                d_input, d_weights, d_bias, d_output,
                aligned_batch, aligned_input, aligned_output,
                cuda::NeuralOperations::GELU, stream);
        };
        
        double time = measureKernelTime(kernel_func, config.benchmark_iterations);
        
        result.avg_time_ms = time;
        
        // Performance metrics (Tensor Core theoretical peak is much higher)
        double flops = 2.0 * batch_size * input_dim * output_dim;
        result.throughput_gops = (flops / 1e9) / (time / 1000.0);
        
        double bytes = input_size + weight_size + bias_size + output_size;
        result.bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0);
        
        result.memory_used_bytes = bytes;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(temp_float);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Benchmark attention mechanism
    BenchmarkResult benchmarkAttention(int batch_size, int seq_len, int num_heads, int head_dim) {
        BenchmarkResult result;
        result.name = "Multi-Head Attention B=" + std::to_string(batch_size) + 
                     " L=" + std::to_string(seq_len) + 
                     " H=" + std::to_string(num_heads) + 
                     " D=" + std::to_string(head_dim);
        
        int hidden_dim = num_heads * head_dim;
        size_t qkv_size = batch_size * seq_len * hidden_dim * sizeof(float);
        size_t attention_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
        
        float *d_Q, *d_K, *d_V, *d_output, *d_attention;
        cudaMalloc(&d_Q, qkv_size);
        cudaMalloc(&d_K, qkv_size);
        cudaMalloc(&d_V, qkv_size);
        cudaMalloc(&d_output, qkv_size);
        cudaMalloc(&d_attention, attention_size);
        
        // Initialize
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_Q, batch_size * seq_len * hidden_dim);
        curandGenerateUniform(gen, d_K, batch_size * seq_len * hidden_dim);
        curandGenerateUniform(gen, d_V, batch_size * seq_len * hidden_dim);
        
        float scale = 1.0f / sqrtf((float)head_dim);
        
        auto kernel_func = [&]() {
            dim3 block(256);
            dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);
            size_t shared_mem = seq_len * sizeof(float);
            
            neural::multiHeadAttention<<<grid, block, shared_mem, stream>>>(
                d_Q, d_K, d_V, d_output, d_attention,
                batch_size, seq_len, num_heads, head_dim, scale);
        };
        
        double time = measureKernelTime(kernel_func, config.benchmark_iterations);
        
        result.avg_time_ms = time;
        
        // Attention computation: Q*K^T (2*L^2*D ops per head) + softmax + attention*V
        double flops = batch_size * num_heads * 
                      (2.0 * seq_len * seq_len * head_dim + // Q*K^T
                       seq_len * seq_len +                   // softmax
                       2.0 * seq_len * seq_len * head_dim);  // attention*V
        
        result.throughput_gops = (flops / 1e9) / (time / 1000.0);
        
        double bytes = 3 * qkv_size + qkv_size + attention_size; // read Q,K,V + write output,attention
        result.bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0);
        
        result.memory_used_bytes = 4 * qkv_size + attention_size;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_output);
        cudaFree(d_attention);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Benchmark geometric computations
    BenchmarkResult benchmarkSheafCohomology(int num_patches, int sheaf_dim, int degree) {
        BenchmarkResult result;
        result.name = "Sheaf Cohomology P=" + std::to_string(num_patches) + 
                     " D=" + std::to_string(sheaf_dim) + 
                     " deg=" + std::to_string(degree);
        
        // Allocate sheaf data
        size_t section_size = num_patches * sheaf_dim * sizeof(float);
        size_t support_size = num_patches * num_patches * sizeof(int);
        size_t cohom_size = (degree + 1) * num_patches * sheaf_dim * sizeof(float);
        size_t betti_size = (degree + 1) * sizeof(int);
        
        geometric::Sheaf sheaf;
        cudaMalloc(&sheaf.sections, section_size);
        cudaMalloc(&sheaf.support, support_size);
        sheaf.dim = sheaf_dim;
        sheaf.rank = sheaf_dim;
        
        float *d_cohomology;
        int *d_betti;
        cudaMalloc(&d_cohomology, cohom_size);
        cudaMalloc(&d_betti, betti_size);
        cudaMemset(d_betti, 0, betti_size);
        
        // Initialize
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, sheaf.sections, num_patches * sheaf_dim);
        
        // Generate support (adjacency matrix)
        std::vector<int> h_support(num_patches * num_patches);
        for (int i = 0; i < num_patches; ++i) {
            for (int j = 0; j < num_patches; ++j) {
                h_support[i * num_patches + j] = (abs(i - j) <= 1) ? 1 : 0;
            }
        }
        cudaMemcpy(sheaf.support, h_support.data(), support_size, cudaMemcpyHostToDevice);
        
        auto kernel_func = [&]() {
            dim3 block(256);
            dim3 grid(num_patches);
            size_t shared_mem = COHOMOLOGY_DIM * sizeof(float);
            
            geometric::cechCohomology<<<grid, block, shared_mem, stream>>>(
                &sheaf, d_cohomology, d_betti, degree, num_patches);
        };
        
        double time = measureKernelTime(kernel_func, config.benchmark_iterations);
        
        result.avg_time_ms = time;
        
        // Cohomology computation complexity
        double flops = num_patches * sheaf_dim * num_patches * 2.0; // differential maps
        result.throughput_gops = (flops / 1e9) / (time / 1000.0);
        
        double bytes = section_size + support_size + cohom_size + betti_size;
        result.bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0);
        
        result.memory_used_bytes = bytes;
        result.compute_intensity = flops / bytes;
        
        // Cleanup
        cudaFree(sheaf.sections);
        cudaFree(sheaf.support);
        cudaFree(d_cohomology);
        cudaFree(d_betti);
        curandDestroyGenerator(gen);
        
        return result;
    }
    
    // Print benchmark results
    void printResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== CUDA Benchmark Results ===" << std::endl;
        std::cout << std::setw(40) << "Operation" 
                  << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(12) << "GB/s"
                  << std::setw(15) << "Memory (MB)"
                  << std::setw(12) << "Intensity" << std::endl;
        std::cout << std::string(103, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(40) << result.name
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.throughput_gops
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.bandwidth_gb_s
                      << std::setw(15) << std::fixed << std::setprecision(1) 
                      << (result.memory_used_bytes / (1024.0 * 1024.0))
                      << std::setw(12) << std::fixed << std::setprecision(1) 
                      << result.compute_intensity << std::endl;
        }
        
        // Print device capabilities
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, config.device_id);
        
        std::cout << "\nDevice: " << prop.name << std::endl;
        std::cout << "Peak FP32 Performance: " 
                  << (prop.multiProcessorCount * prop.clockRate * 2 * 
                      prop.maxThreadsPerMultiProcessor / 1e6) << " GFLOPS" << std::endl;
        std::cout << "Memory Bandwidth: " 
                  << (prop.memoryClockRate * prop.memoryBusWidth * 2 / 8e6) 
                  << " GB/s" << std::endl;
        
        if (prop.major >= 7) {
            std::cout << "Tensor Core Support: Yes" << std::endl;
            std::cout << "Peak FP16 TC Performance: " 
                      << (prop.multiProcessorCount * prop.clockRate * 8 * 
                          prop.maxThreadsPerMultiProcessor / 1e6) << " GFLOPS" << std::endl;
        }
    }
};

// Main benchmark runner
void runComprehensiveBenchmarks() {
    BenchmarkConfig config;
    config.warmup_iterations = 10;
    config.benchmark_iterations = 100;
    config.use_tensor_cores = true;
    config.use_fp16 = true;
    
    CudaBenchmarker benchmarker(config);
    std::vector<BenchmarkResult> results;
    
    std::cout << "Running CUDA benchmarks for Geometric Langlands..." << std::endl;
    
    // Matrix operations
    results.push_back(benchmarker.benchmarkMatrixMultiply(1024, 1024, 1024));
    results.push_back(benchmarker.benchmarkMatrixMultiply(2048, 2048, 2048));
    results.push_back(benchmarker.benchmarkMatrixMultiply(4096, 4096, 4096));
    
    // Tensor operations
    results.push_back(benchmarker.benchmarkTensorContraction(64, 64, 64, 64));
    results.push_back(benchmarker.benchmarkTensorContraction(128, 128, 128, 128));
    
    // Neural network operations
    results.push_back(benchmarker.benchmarkNeuralForward(32, 1024, 1024));
    results.push_back(benchmarker.benchmarkNeuralForward(64, 2048, 2048));
    results.push_back(benchmarker.benchmarkNeuralForward(128, 4096, 4096));
    
    // Attention mechanism
    results.push_back(benchmarker.benchmarkAttention(8, 512, 8, 64));
    results.push_back(benchmarker.benchmarkAttention(16, 1024, 16, 64));
    
    // Geometric operations
    results.push_back(benchmarker.benchmarkSheafCohomology(100, 64, 3));
    results.push_back(benchmarker.benchmarkSheafCohomology(200, 128, 4));
    
    // Print results
    benchmarker.printResults(results);
}

} // namespace benchmarks
} // namespace langlands

// Entry point
int main(int argc, char** argv) {
    try {
        langlands::benchmarks::runComprehensiveBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}