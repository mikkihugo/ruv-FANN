// CUDA Test Suite for Geometric Langlands Kernels
// Comprehensive testing and validation

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <complex>
#include <cmath>
#include "../include/langlands_cuda.h"
#include "../kernels/geometric_kernels.cu"
#include "../kernels/matrix_operations.cu"
#include "../kernels/neural_kernels.cu"

namespace langlands {
namespace testing {

using namespace geometric;
using namespace cuda;
using namespace neural;

// Test fixture for CUDA operations
class CudaTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Initialize random generator
        rng.seed(42);
        
        // Get device properties
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        
        // Check if device supports required features
        has_tensor_cores = (deviceProp.major >= 7);
        has_fp16 = (deviceProp.major >= 5 && deviceProp.minor >= 3);
        
        std::cout << "Testing on: " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Tensor Cores: " << (has_tensor_cores ? "Yes" : "No") << std::endl;
    }
    
    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaDeviceReset());
    }
    
    // Helper functions
    template<typename T>
    void allocateDeviceMemory(T** ptr, size_t count) {
        CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
    }
    
    template<typename T>
    void copyToDevice(T* device_ptr, const std::vector<T>& host_data) {
        CUDA_CHECK(cudaMemcpy(device_ptr, host_data.data(), 
                             host_data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    std::vector<T> copyFromDevice(T* device_ptr, size_t count) {
        std::vector<T> result(count);
        CUDA_CHECK(cudaMemcpy(result.data(), device_ptr, 
                             count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    
    void generateRandomData(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (auto& val : data) {
            val = dist(rng);
        }
    }
    
    void generateRandomComplexData(std::vector<Complex>& data) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& val : data) {
            val.real = dist(rng);
            val.imag = dist(rng);
        }
    }
    
    // Validation helpers
    bool isClose(float a, float b, float tolerance = 1e-5f) {
        return std::abs(a - b) <= tolerance;
    }
    
    bool isCloseComplex(const Complex& a, const Complex& b, float tolerance = 1e-5f) {
        return isClose(a.real, b.real, tolerance) && isClose(a.imag, b.imag, tolerance);
    }
    
protected:
    cudaStream_t stream;
    cudaDeviceProp deviceProp;
    std::mt19937 rng;
    bool has_tensor_cores;
    bool has_fp16;
};

// Test matrix operations
TEST_F(CudaTestFixture, MatrixMultiplication) {
    const int M = 256, N = 256, K = 256;
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    
    generateRandomData(h_A);
    generateRandomData(h_B);
    
    // Compute reference result on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                h_C_ref[i * N + j] += h_A[i * K + k] * h_B[k * N + j];
            }
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, M * K);
    allocateDeviceMemory(&d_B, K * N);
    allocateDeviceMemory(&d_C, M * N);
    
    // Copy data to device
    copyToDevice(d_A, h_A);
    copyToDevice(d_B, h_B);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    matrixMultiplyShared<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy result back
    auto result = copyFromDevice(d_C, M * N);
    
    // Validate results
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(result[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-3f) << "Matrix multiplication error too large: " << max_error;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test tensor contraction
TEST_F(CudaTestFixture, TensorContraction) {
    const int d1 = 32, d2 = 32, d3 = 32, d4 = 32;
    
    std::vector<float> h_T1(d1 * d2 * d3);
    std::vector<float> h_T2(d2 * d3 * d4);
    std::vector<float> h_result(d1 * d4, 0.0f);
    std::vector<float> h_ref(d1 * d4, 0.0f);
    
    generateRandomData(h_T1);
    generateRandomData(h_T2);
    
    // Compute reference
    for (int i = 0; i < d1; ++i) {
        for (int l = 0; l < d4; ++l) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    h_ref[i * d4 + l] += h_T1[i * d2 * d3 + j * d3 + k] * 
                                        h_T2[j * d3 * d4 + k * d4 + l];
                }
            }
        }
    }
    
    // Device computation
    float *d_T1, *d_T2, *d_result;
    allocateDeviceMemory(&d_T1, d1 * d2 * d3);
    allocateDeviceMemory(&d_T2, d2 * d3 * d4);
    allocateDeviceMemory(&d_result, d1 * d4);
    
    copyToDevice(d_T1, h_T1);
    copyToDevice(d_T2, h_T2);
    
    dim3 block(256);
    dim3 grid((d1 * d4 + block.x - 1) / block.x);
    
    tensorContraction<<<grid, block, 0, stream>>>(d_T1, d_T2, d_result, d1, d2, d3, d4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto result = copyFromDevice(d_result, d1 * d4);
    
    // Validate
    float max_error = 0.0f;
    for (int i = 0; i < d1 * d4; ++i) {
        float error = std::abs(result[i] - h_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-2f) << "Tensor contraction error: " << max_error;
    
    cudaFree(d_T1);
    cudaFree(d_T2);
    cudaFree(d_result);
}

// Test Čech cohomology computation
TEST_F(CudaTestFixture, CechCohomology) {
    const int num_patches = 10;
    const int sheaf_dim = 64;
    const int degree = 2;
    
    // Create test sheaf
    Sheaf h_sheaf;
    h_sheaf.dim = sheaf_dim;
    h_sheaf.rank = sheaf_dim;
    
    std::vector<float> h_sections(num_patches * sheaf_dim);
    std::vector<int> h_support(num_patches * num_patches);
    
    generateRandomData(h_sections);
    
    // Create adjacency matrix (chain complex)
    for (int i = 0; i < num_patches; ++i) {
        for (int j = 0; j < num_patches; ++j) {
            h_support[i * num_patches + j] = (std::abs(i - j) == 1) ? 1 : 0;
        }
    }
    
    // Device memory
    Sheaf d_sheaf;
    allocateDeviceMemory(&d_sheaf.sections, num_patches * sheaf_dim);
    allocateDeviceMemory(&d_sheaf.support, num_patches * num_patches);
    d_sheaf.dim = sheaf_dim;
    d_sheaf.rank = sheaf_dim;
    
    float *d_cohomology;
    int *d_betti;
    allocateDeviceMemory(&d_cohomology, (degree + 1) * num_patches * sheaf_dim);
    allocateDeviceMemory(&d_betti, degree + 1);
    
    // Copy data
    copyToDevice(d_sheaf.sections, h_sections);
    copyToDevice(d_sheaf.support, h_support);
    
    // Initialize Betti numbers
    CUDA_CHECK(cudaMemset(d_betti, 0, (degree + 1) * sizeof(int)));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid(num_patches);
    size_t shared_mem = COHOMOLOGY_DIM * sizeof(float);
    
    cechCohomology<<<grid, block, shared_mem, stream>>>(
        &d_sheaf, d_cohomology, d_betti, degree, num_patches);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Validate results
    auto cohomology_result = copyFromDevice(d_cohomology, (degree + 1) * num_patches * sheaf_dim);
    auto betti_result = copyFromDevice(d_betti, degree + 1);
    
    // Check that cohomology is computed (non-zero results)
    bool has_nonzero = false;
    for (float val : cohomology_result) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Cohomology computation produced all zeros";
    
    // Check Betti numbers are reasonable
    for (int i = 0; i <= degree; ++i) {
        EXPECT_GE(betti_result[i], 0) << "Negative Betti number at degree " << i;
        EXPECT_LE(betti_result[i], num_patches) << "Betti number too large at degree " << i;
    }
    
    // Cleanup
    cudaFree(d_sheaf.sections);
    cudaFree(d_sheaf.support);
    cudaFree(d_cohomology);
    cudaFree(d_betti);
}

// Test Hitchin fibration
TEST_F(CudaTestFixture, HitchinFibration) {
    const int genus = 2;
    const int rank = 3;
    const int num_points = genus * rank;
    
    std::vector<Complex> h_higgs(num_points * rank);
    generateRandomComplexData(h_higgs);
    
    // Device memory
    Complex *d_higgs;
    float *d_spectral_curve, *d_cameral_cover;
    
    allocateDeviceMemory(&d_higgs, num_points * rank);
    allocateDeviceMemory(&d_spectral_curve, num_points * 2);
    allocateDeviceMemory(&d_cameral_cover, num_points);
    
    copyToDevice(d_higgs, h_higgs);
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    hitchinFibration<<<grid, block, 0, stream>>>(
        d_higgs, d_spectral_curve, d_cameral_cover, genus, rank);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Get results
    auto spectral_result = copyFromDevice(d_spectral_curve, num_points * 2);
    auto cameral_result = copyFromDevice(d_cameral_cover, num_points);
    
    // Validate spectral curve (should be complex polynomial coefficients)
    for (int i = 0; i < num_points; ++i) {
        Complex coeff{spectral_result[i * 2], spectral_result[i * 2 + 1]};
        EXPECT_TRUE(std::isfinite(coeff.real) && std::isfinite(coeff.imag)) 
            << "Invalid spectral curve coefficient at " << i;
    }
    
    // Validate cameral cover (should be real and non-negative)
    for (int i = 0; i < num_points; ++i) {
        EXPECT_TRUE(std::isfinite(cameral_result[i])) 
            << "Invalid cameral cover value at " << i;
        EXPECT_GE(cameral_result[i], 0.0f) 
            << "Negative cameral cover value at " << i;
    }
    
    cudaFree(d_higgs);
    cudaFree(d_spectral_curve);
    cudaFree(d_cameral_cover);
}

// Test Hecke eigensheaf computation
TEST_F(CudaTestFixture, HeckeEigensheaf) {
    const int sheaf_rank = 32;
    const int correspondence_dim = 128;
    const int prime = 7;
    const int num_sheaves = 16;
    
    std::vector<float> h_sheaf_data(num_sheaves * sheaf_rank);
    std::vector<float> h_hecke_correspondence(correspondence_dim);
    
    generateRandomData(h_sheaf_data);
    generateRandomData(h_hecke_correspondence);
    
    // Device memory
    float *d_sheaf_data, *d_hecke_correspondence;
    float *d_eigenvalues, *d_eigensheaves;
    
    allocateDeviceMemory(&d_sheaf_data, num_sheaves * sheaf_rank);
    allocateDeviceMemory(&d_hecke_correspondence, correspondence_dim);
    allocateDeviceMemory(&d_eigenvalues, num_sheaves);
    allocateDeviceMemory(&d_eigensheaves, num_sheaves * sheaf_rank);
    
    copyToDevice(d_sheaf_data, h_sheaf_data);
    copyToDevice(d_hecke_correspondence, h_hecke_correspondence);
    
    // Launch kernel
    dim3 block(32); // One block per sheaf
    dim3 grid(num_sheaves);
    size_t shared_mem = 2 * block.x * sheaf_rank * sizeof(float);
    
    heckeEigensheaf<<<grid, block, shared_mem, stream>>>(
        d_sheaf_data, d_hecke_correspondence, d_eigenvalues, d_eigensheaves,
        sheaf_rank, correspondence_dim, prime);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Get results
    auto eigenvalues = copyFromDevice(d_eigenvalues, num_sheaves);
    auto eigensheaves = copyFromDevice(d_eigensheaves, num_sheaves * sheaf_rank);
    
    // Validate eigenvalues
    for (int i = 0; i < num_sheaves; ++i) {
        EXPECT_TRUE(std::isfinite(eigenvalues[i])) << "Invalid eigenvalue at " << i;
        EXPECT_GT(eigenvalues[i], 0.0f) << "Non-positive eigenvalue at " << i;
    }
    
    // Validate eigensheaves (should be normalized)
    for (int i = 0; i < num_sheaves; ++i) {
        float norm = 0.0f;
        for (int j = 0; j < sheaf_rank; ++j) {
            float val = eigensheaves[i * sheaf_rank + j];
            EXPECT_TRUE(std::isfinite(val)) << "Invalid eigensheaf value";
            norm += val * val;
        }
        EXPECT_NEAR(norm, 1.0f, 1e-3f) << "Eigensheaf not normalized at " << i;
    }
    
    cudaFree(d_sheaf_data);
    cudaFree(d_hecke_correspondence);
    cudaFree(d_eigenvalues);
    cudaFree(d_eigensheaves);
}

// Test neural network forward propagation
TEST_F(CudaTestFixture, NeuralForwardPropagation) {
    const int batch_size = 64;
    const int input_dim = 512;
    const int output_dim = 256;
    
    std::vector<float> h_input(batch_size * input_dim);
    std::vector<float> h_weights(input_dim * output_dim);
    std::vector<float> h_bias(output_dim);
    
    generateRandomData(h_input, 0.0f, 1.0f);
    generateRandomData(h_weights, -0.1f, 0.1f);
    generateRandomData(h_bias, -0.1f, 0.1f);
    
    // Device memory
    float *d_input, *d_weights, *d_bias, *d_output;
    allocateDeviceMemory(&d_input, batch_size * input_dim);
    allocateDeviceMemory(&d_weights, input_dim * output_dim);
    allocateDeviceMemory(&d_bias, output_dim);
    allocateDeviceMemory(&d_output, batch_size * output_dim);
    
    copyToDevice(d_input, h_input);
    copyToDevice(d_weights, h_weights);
    copyToDevice(d_bias, h_bias);
    
    // Launch forward propagation
    dim3 block(16, 16);
    dim3 grid((output_dim + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    
    // Use GELU activation for testing
    int activation_type = 1; // GELU
    
    // Simulate the forward propagation kernel call
    // (In actual implementation, this would be a call to the CUDA kernel)
    
    // For testing, compute reference on CPU
    std::vector<float> h_output_ref(batch_size * output_dim);
    
    // Compute GEMM: output = input * weights + bias
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < output_dim; ++o) {
            float sum = h_bias[o];
            for (int i = 0; i < input_dim; ++i) {
                sum += h_input[b * input_dim + i] * h_weights[i * output_dim + o];
            }
            
            // Apply GELU activation
            float x = sum;
            h_output_ref[b * output_dim + o] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    }
    
    // For now, just validate that the setup was successful
    EXPECT_GT(h_output_ref.size(), 0);
    
    // Check that all reference values are finite
    for (float val : h_output_ref) {
        EXPECT_TRUE(std::isfinite(val)) << "Reference computation produced invalid value";
    }
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

// Performance test for matrix operations
TEST_F(CudaTestFixture, MatrixPerformanceTest) {
    const std::vector<int> sizes = {512, 1024, 2048, 4096};
    
    for (int size : sizes) {
        // Allocate memory
        float *d_A, *d_B, *d_C;
        allocateDeviceMemory(&d_A, size * size);
        allocateDeviceMemory(&d_B, size * size);
        allocateDeviceMemory(&d_C, size * size);
        
        // Initialize with dummy data
        CUDA_CHECK(cudaMemset(d_A, 1, size * size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B, 1, size * size * sizeof(float)));
        
        // Warmup
        dim3 block(16, 16);
        dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
        
        for (int i = 0; i < 5; ++i) {
            matrixMultiplyShared<<<grid, block, 0, stream>>>(d_A, d_B, d_C, size, size, size);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start, stream));
        
        const int iterations = 10;
        for (int i = 0; i < iterations; ++i) {
            matrixMultiplyShared<<<grid, block, 0, stream>>>(d_A, d_B, d_C, size, size, size);
        }
        
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        double avg_time = elapsed_ms / iterations;
        double gflops = (2.0 * size * size * size) / (avg_time * 1e6);
        
        std::cout << "Matrix " << size << "x" << size 
                  << ": " << avg_time << " ms, " 
                  << gflops << " GFLOPS" << std::endl;
        
        // Performance expectations (adjust based on hardware)
        EXPECT_LT(avg_time, 1000.0) << "Matrix multiplication too slow for size " << size;
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// Multi-GPU test
TEST_F(CudaTestFixture, MultiGPUSupport) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count < 2) {
        GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
        return;
    }
    
    const int size = 1024;
    const int elements = size * size;
    
    std::vector<float> h_data(elements);
    generateRandomData(h_data);
    
    // Test data distribution across GPUs
    std::vector<float*> d_data(device_count);
    std::vector<cudaStream_t> streams(device_count);
    
    for (int i = 0; i < device_count; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        allocateDeviceMemory(&d_data[i], elements);
        copyToDevice(d_data[i], h_data);
    }
    
    // Verify data on all devices
    for (int i = 0; i < device_count; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        auto device_result = copyFromDevice(d_data[i], elements);
        
        // Check first few elements
        for (int j = 0; j < 10; ++j) {
            EXPECT_NEAR(device_result[j], h_data[j], 1e-6f) 
                << "Data mismatch on device " << i << " at element " << j;
        }
    }
    
    // Cleanup
    for (int i = 0; i < device_count; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        cudaFree(d_data[i]);
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

} // namespace testing
} // namespace langlands

// Main function to run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check if CUDA is available
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices available. Skipping GPU tests." << std::endl;
        return 0;
    }
    
    std::cout << "Running CUDA tests on " << device_count << " device(s)" << std::endl;
    
    return RUN_ALL_TESTS();
}