// CUDA Header for Geometric Langlands Implementation
// Public interfaces for GPU-accelerated computations

#ifndef LANGLANDS_CUDA_H
#define LANGLANDS_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cusparse.h>
#include <cufft.h>
#include <vector>
#include <memory>

namespace langlands {
namespace cuda {

// Forward declarations
struct Complex;
struct Sheaf;
struct DModule;
class CudaMemoryPool;
class NeuralMemoryManager;
class GeometricMultiGPU;

// Error handling
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", 
                file, line, error, cudaGetErrorString(error));
        exit(1);
    }
}

// Device capabilities structure
struct DeviceCapabilities {
    int device_count;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    size_t shared_memory_per_block;
    int max_threads_per_block;
    int max_grid_size[3];
    bool tensor_core_available;
    bool fp16_available;
    int multiprocessor_count;
    
    static DeviceCapabilities query();
    void print() const;
};

// Matrix operations interface
class MatrixOperations {
public:
    // Basic operations
    static void multiply(const float* A, const float* B, float* C, 
                        int M, int N, int K, cudaStream_t stream = 0);
    
    static void multiplyShared(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream = 0);
    
    // Tensor operations
    static void tensorContraction(const float* T1, const float* T2, float* result,
                                 int dim1, int dim2, int dim3, int dim4,
                                 cudaStream_t stream = 0);
    
    // Eigenvalue computation
    static void powerIteration(const float* A, float* eigenvalue, float* eigenvector,
                              int n, int max_iter = 100, cudaStream_t stream = 0);
    
    // Sparse operations
    static void sparseMatVecCSR(const float* values, const int* rowPtr, 
                               const int* colIdx, const float* x, float* y,
                               int numRows, cudaStream_t stream = 0);
    
    // FFT operations
    static void fft(Complex* data, int n, bool inverse = false, 
                   cudaStream_t stream = 0);
    
    // Matrix functions
    static void matrixExponential(const float* A, float* expA, int n, 
                                 int terms = 20, cudaStream_t stream = 0);
};

// Neural network operations interface
class NeuralOperations {
public:
    enum ActivationType {
        RELU = 0,
        GELU = 1,
        SWISH = 2
    };
    
    // Forward propagation
    static void forwardPropagation(const float* input, const float* weights,
                                   const float* bias, float* output,
                                   int batch_size, int input_dim, int output_dim,
                                   ActivationType activation, cudaStream_t stream = 0);
    
    // Tensor Core operations (FP16)
    static void forwardPropagationTC(const half* input, const half* weights,
                                    const half* bias, half* output,
                                    int batch_size, int input_dim, int output_dim,
                                    ActivationType activation, cudaStream_t stream = 0);
    
    // Attention mechanism
    static void multiHeadAttention(const float* Q, const float* K, const float* V,
                                  float* output, float* attention_weights,
                                  int batch_size, int seq_length, int num_heads,
                                  int head_dim, cudaStream_t stream = 0);
    
    // Batch normalization
    static void batchNorm(float* x, const float* gamma, const float* beta,
                         float* running_mean, float* running_var,
                         int batch_size, int channels, int spatial_dim,
                         float momentum = 0.9f, float epsilon = 1e-5f,
                         bool training = true, cudaStream_t stream = 0);
    
    // Backpropagation
    static void backwardPropagation(const float* input, const float* weights,
                                   const float* grad_output, float* grad_input,
                                   float* grad_weights, float* grad_bias,
                                   int batch_size, int input_dim, int output_dim,
                                   cudaStream_t stream = 0);
    
    // Optimizers
    static void adamOptimizer(float* params, const float* gradients,
                             float* m, float* v, int size,
                             float learning_rate, float beta1 = 0.9f,
                             float beta2 = 0.999f, float epsilon = 1e-8f,
                             int timestep = 1, cudaStream_t stream = 0);
    
    // Dropout
    static void dropout(float* x, float* mask, int size, float dropout_rate,
                       unsigned long long seed, cudaStream_t stream = 0);
    
    // Convolution
    static void convolution2D(const float* input, const float* kernel,
                             const float* bias, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width, int kernel_size,
                             int stride = 1, int padding = 0,
                             cudaStream_t stream = 0);
};

// Geometric computations interface
class GeometricOperations {
public:
    // Sheaf cohomology
    static void computeCechCohomology(const Sheaf* sheaf, float* cohomology_groups,
                                     int* betti_numbers, int degree, int num_patches,
                                     cudaStream_t stream = 0);
    
    // Hitchin system
    static void computeHitchinFibration(const Complex* higgs_field,
                                       float* spectral_curve, float* cameral_cover,
                                       int genus, int rank, cudaStream_t stream = 0);
    
    // Flat connections
    static void computeFlatConnection(const float* connection_form,
                                     const float* tangent_vectors,
                                     float* parallel_transport, float* holonomy,
                                     int dim, int num_paths, cudaStream_t stream = 0);
    
    // Hecke operators
    static void computeHeckeEigensheaf(const float* sheaf_data,
                                      const float* hecke_correspondence,
                                      float* eigenvalues, float* eigensheaves,
                                      int sheaf_rank, int correspondence_dim,
                                      int prime, cudaStream_t stream = 0);
    
    // Perverse sheaves
    static void computePerverseSheaf(const float* stratification,
                                    const float* local_systems,
                                    float* perverse_cohomology, int* perversity,
                                    int num_strata, int dim, cudaStream_t stream = 0);
    
    // Ramification
    static void computeRamifiedLanglands(const float* ramification_data,
                                        const Complex* local_monodromy,
                                        float* wild_character, Complex* stokes_data,
                                        int num_points, int rank,
                                        cudaStream_t stream = 0);
    
    // Derived categories
    static void computeDerivedFunctor(const float* complex_in, float* complex_out,
                                     const float* functor_data,
                                     int* homological_degree,
                                     int complex_length, int obj_dim,
                                     cudaStream_t stream = 0);
    
    // Quantum deformation
    static void computeQuantumLanglands(const Complex* quantum_group,
                                       const float* q_parameter,
                                       Complex* braiding_matrix,
                                       float* knot_invariants,
                                       int dim, float hbar = 1.0f,
                                       cudaStream_t stream = 0);
    
    // Spectral data
    static void computeSpectralCorrespondence(const float* opers,
                                             const Complex* flat_connections,
                                             float* spectral_data,
                                             Complex* eigen_functions,
                                             int num_points, int rank,
                                             cudaStream_t stream = 0);
};

// Memory management interface
class CudaMemoryManager {
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    CudaMemoryManager(size_t pool_size = 1024 * 1024 * 1024); // 1GB default
    ~CudaMemoryManager();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();
    
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    void printStats() const;
};

// Multi-GPU support
class MultiGPUManager {
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    MultiGPUManager();
    ~MultiGPUManager();
    
    int getDeviceCount() const;
    void setDevice(int device);
    
    // Distribute data across GPUs
    void distributeData(const void* host_data, void** device_data, 
                       size_t size, size_t* chunk_sizes);
    
    // Gather results from all GPUs
    void gatherResults(void* host_data, void** device_data,
                      size_t size, const size_t* chunk_sizes);
    
    // Execute kernel on all GPUs
    template<typename KernelFunc, typename... Args>
    void executeOnAllGPUs(KernelFunc kernel, dim3 grid, dim3 block,
                         size_t shared_mem, Args... args);
    
    // Synchronize all GPUs
    void synchronizeAll();
};

// Performance monitoring
class PerformanceMonitor {
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    void startTimer(const std::string& name);
    void endTimer(const std::string& name);
    
    void recordMemoryUsage(const std::string& tag);
    void recordKernelMetrics(const std::string& kernel_name);
    
    void printReport() const;
    void exportMetrics(const std::string& filename) const;
    
    // Get specific metrics
    double getKernelTime(const std::string& name) const;
    size_t getPeakMemoryUsage() const;
    double getTotalComputeTime() const;
};

// Benchmarking utilities
class Benchmarker {
public:
    struct Result {
        double time_ms;
        double gflops;
        double bandwidth_gb_s;
        size_t memory_used;
    };
    
    // Matrix operations benchmarks
    static Result benchmarkMatrixMultiply(int M, int N, int K, int iterations = 100);
    static Result benchmarkTensorContraction(int dim1, int dim2, int dim3, int dim4,
                                           int iterations = 100);
    
    // Neural network benchmarks
    static Result benchmarkForwardProp(int batch_size, int input_dim, int output_dim,
                                      int iterations = 100);
    static Result benchmarkAttention(int batch_size, int seq_length, int num_heads,
                                    int head_dim, int iterations = 100);
    
    // Geometric benchmarks
    static Result benchmarkSheafCohomology(int num_patches, int sheaf_dim,
                                          int iterations = 100);
    static Result benchmarkHitchinSystem(int genus, int rank, int iterations = 100);
    
    // Compare with CPU implementation
    static void compareWithCPU(const std::string& operation_name,
                              std::function<void()> gpu_func,
                              std::function<void()> cpu_func);
};

// Utility functions
namespace utils {
    // Initialize CUDA environment
    void initializeCuda(int device = 0);
    
    // Query and print device information
    void printDeviceInfo(int device = 0);
    
    // Check available memory
    size_t getAvailableMemory();
    
    // Optimal grid/block configuration
    void getOptimalLaunchConfig(int total_threads, dim3& grid, dim3& block,
                               int device = 0);
    
    // Data transfer utilities
    template<typename T>
    void copyToDevice(T* device_ptr, const T* host_ptr, size_t count,
                     cudaStream_t stream = 0);
    
    template<typename T>
    void copyToHost(T* host_ptr, const T* device_ptr, size_t count,
                   cudaStream_t stream = 0);
    
    // Synchronization
    void synchronizeDevice();
    void synchronizeStream(cudaStream_t stream);
}

} // namespace cuda
} // namespace langlands

#endif // LANGLANDS_CUDA_H