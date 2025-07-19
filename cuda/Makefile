# CUDA Acceleration Makefile for Geometric Langlands Conjecture
# Optimized for maximum performance and 10x speedup target

# CUDA Configuration
CUDA_PATH ?= /usr/local/cuda
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64

# Compiler Configuration
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Architecture Configuration (auto-detect and optimize)
GPU_ARCH = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.' | xargs printf "sm_%s")
GPU_ARCH_COMPAT = compute_$(shell echo $(GPU_ARCH) | sed 's/sm_//')

# Compiler Flags
NVCC_FLAGS = -std=c++17 \
             -arch=$(GPU_ARCH_COMPAT) \
             -code=$(GPU_ARCH) \
             -O3 \
             -use_fast_math \
             -maxrregcount=64 \
             --ptxas-options=-v \
             -Xcompiler -fPIC \
             -Xcompiler -Wall \
             -Xcompiler -Wextra \
             -lineinfo

CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC

# Advanced optimization flags for Tensor Core and modern GPUs
ifeq ($(shell echo $(GPU_ARCH) | grep -E "(sm_70|sm_75|sm_80|sm_86|sm_87|sm_89|sm_90)"),)
    # No Tensor Cores available
    TENSOR_CORE_FLAGS = 
else
    # Enable Tensor Core optimizations
    TENSOR_CORE_FLAGS = -DUSE_TENSOR_CORES \
                       -DCUDA_ENABLE_FP16_FALLBACK \
                       -gencode arch=compute_70,code=sm_70 \
                       -gencode arch=compute_75,code=sm_75 \
                       -gencode arch=compute_80,code=sm_80
endif

# Include paths
INCLUDES = -I$(CUDA_INC_PATH) \
           -I./include \
           -I/usr/include/eigen3 \
           -I./third_party

# Library paths and libraries
LIBS = -L$(CUDA_LIB_PATH) \
       -lcuda \
       -lcudart \
       -lcublas \
       -lcusparse \
       -lcufft \
       -lcurand \
       -lcudnn \
       -lpthread \
       -lm

# Debug flags (disabled by default for performance)
ifdef DEBUG
    NVCC_FLAGS += -g -G -DDEBUG
    CXX_FLAGS += -g -DDEBUG
endif

# Performance optimization flags
PERFORMANCE_FLAGS = -DUSE_COOPERATIVE_GROUPS \
                   -DENABLE_MEMORY_POOL \
                   -DOPTIMIZE_MEMORY_ACCESS \
                   -DPARALLEL_REDUCTION \
                   $(TENSOR_CORE_FLAGS)

# Source files
KERNEL_SOURCES = kernels/matrix_operations.cu \
                kernels/geometric_kernels.cu \
                kernels/neural_kernels.cu

TEST_SOURCES = tests/test_geometric_kernels.cu
BENCHMARK_SOURCES = benchmarks/benchmark_suite.cu

# Object files
KERNEL_OBJECTS = $(KERNEL_SOURCES:.cu=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cu=.o)
BENCHMARK_OBJECTS = $(BENCHMARK_SOURCES:.cu=.o)

# Targets
KERNEL_LIB = lib/liblanglands_cuda.a
KERNEL_SHARED = lib/liblanglands_cuda.so
TEST_BINARY = bin/test_cuda_kernels
BENCHMARK_BINARY = bin/benchmark_cuda

# Default target
all: directories $(KERNEL_LIB) $(KERNEL_SHARED) $(TEST_BINARY) $(BENCHMARK_BINARY)

# Create necessary directories
directories:
	@mkdir -p lib bin obj

# Static library
$(KERNEL_LIB): $(KERNEL_OBJECTS)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Shared library
$(KERNEL_SHARED): $(KERNEL_OBJECTS)
	@echo "Creating shared library: $@"
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ $(LIBS)

# Test binary
$(TEST_BINARY): $(TEST_OBJECTS) $(KERNEL_OBJECTS)
	@echo "Building test binary: $@"
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS) -lgtest -lgtest_main

# Benchmark binary
$(BENCHMARK_BINARY): $(BENCHMARK_OBJECTS) $(KERNEL_OBJECTS)
	@echo "Building benchmark binary: $@"
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compile CUDA source files
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(PERFORMANCE_FLAGS) $(INCLUDES) -c $< -o $@

# Performance optimized build
performance: NVCC_FLAGS += -DPERFORMANCE_MODE -DNDEBUG
performance: clean all

# Memory debugging build
debug-memory: NVCC_FLAGS += -DMEMORY_DEBUG -DCUDA_MEMCHECK
debug-memory: DEBUG=1
debug-memory: clean all

# Profile-guided optimization
pgo-generate: NVCC_FLAGS += -fprofile-generate
pgo-generate: clean all

pgo-use: NVCC_FLAGS += -fprofile-use
pgo-use: clean all

# Run benchmarks
benchmark: $(BENCHMARK_BINARY)
	@echo "Running CUDA benchmarks..."
	./$(BENCHMARK_BINARY)

# Run tests
test: $(TEST_BINARY)
	@echo "Running CUDA tests..."
	./$(TEST_BINARY)

# Memory check with CUDA-MEMCHECK
memcheck: $(TEST_BINARY)
	@echo "Running memory check..."
	cuda-memcheck --tool memcheck ./$(TEST_BINARY)

# Performance profiling with nvprof
profile: $(BENCHMARK_BINARY)
	@echo "Profiling with nvprof..."
	nvprof --analysis-metrics -o profile.nvprof ./$(BENCHMARK_BINARY)

# Performance profiling with nsight compute
nsight-profile: $(BENCHMARK_BINARY)
	@echo "Profiling with Nsight Compute..."
	ncu --set full -o nsight_profile ./$(BENCHMARK_BINARY)

# Clean build artifacts
clean:
	rm -rf $(KERNEL_OBJECTS) $(TEST_OBJECTS) $(BENCHMARK_OBJECTS)
	rm -rf lib/* bin/* obj/*
	rm -f *.nvprof *.ncu-rep

# Install libraries
install: $(KERNEL_LIB) $(KERNEL_SHARED)
	@echo "Installing CUDA libraries..."
	cp $(KERNEL_LIB) /usr/local/lib/
	cp $(KERNEL_SHARED) /usr/local/lib/
	cp -r include/* /usr/local/include/
	ldconfig

# Uninstall libraries
uninstall:
	@echo "Uninstalling CUDA libraries..."
	rm -f /usr/local/lib/liblanglands_cuda.*
	rm -rf /usr/local/include/langlands_cuda.h
	rm -rf /usr/local/include/memory_manager.cuh

# Check GPU capabilities
gpu-info:
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo "Target Architecture: $(GPU_ARCH)"
	@echo "Compute Capability: $(GPU_ARCH_COMPAT)"

# Performance validation
validate-performance: $(BENCHMARK_BINARY)
	@echo "=== Performance Validation ==="
	@echo "Target: 10x speedup over CPU baseline"
	./$(BENCHMARK_BINARY) --validate-speedup

# Code coverage analysis
coverage: NVCC_FLAGS += --coverage
coverage: clean $(TEST_BINARY)
	./$(TEST_BINARY)
	gcov $(KERNEL_SOURCES)

# Documentation generation
docs:
	@echo "Generating CUDA documentation..."
	doxygen docs/Doxyfile

# Static analysis
analyze:
	@echo "Running static analysis..."
	cppcheck --enable=all --std=c++17 kernels/ include/

# Format code
format:
	@echo "Formatting CUDA code..."
	clang-format -i kernels/*.cu include/*.h include/*.cuh tests/*.cu benchmarks/*.cu

# Help target
help:
	@echo "CUDA Acceleration Makefile for Geometric Langlands"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all targets (default)"
	@echo "  performance      - Performance optimized build"
	@echo "  debug-memory     - Memory debugging build"
	@echo "  test             - Run unit tests"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  memcheck         - Run CUDA memory checker"
	@echo "  profile          - Profile with nvprof"
	@echo "  nsight-profile   - Profile with Nsight Compute"
	@echo "  gpu-info         - Display GPU information"
	@echo "  validate-performance - Validate 10x speedup target"
	@echo "  install          - Install libraries system-wide"
	@echo "  clean            - Clean build artifacts"
	@echo "  format           - Format source code"
	@echo "  help             - Show this help"

.PHONY: all clean test benchmark install uninstall gpu-info help format analyze docs coverage validate-performance
.SUFFIXES: .cu .o