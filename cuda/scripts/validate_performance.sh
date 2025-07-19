#!/bin/bash

# CUDA Performance Validation Script
# Validates 10x speedup target for Geometric Langlands GPU acceleration

set -e

# Configuration
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
BUILD_DIR="./build"
RESULTS_DIR="./performance_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${RESULTS_DIR}/performance_report_${TIMESTAMP}.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check CUDA installation
    if ! command -v nvcc &> /dev/null; then
        log_error "NVCC not found. Please install CUDA toolkit."
        exit 1
    fi
    
    # Check for compatible GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please check NVIDIA driver installation."
        exit 1
    fi
    
    # Check GPU compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
    log_info "GPU Compute Capability: ${COMPUTE_CAP}"
    
    if (( $(echo "${COMPUTE_CAP} < 3.5" | bc -l) )); then
        log_warning "GPU compute capability < 3.5. Performance may be limited."
    fi
    
    # Check available memory
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log_info "Total GPU Memory: ${TOTAL_MEM} MB"
    
    if (( TOTAL_MEM < 4000 )); then
        log_warning "Less than 4GB GPU memory available. Large tests may fail."
    fi
    
    log_success "Prerequisites check passed"
}

# Build optimized CUDA kernels
build_kernels() {
    log_info "Building optimized CUDA kernels..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    # Clean previous build
    make clean > /dev/null 2>&1 || true
    
    # Build with performance optimizations
    log_info "Compiling with performance optimizations..."
    make performance -j$(nproc) 2>&1 | tee build.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Build failed. Check build.log for details."
        exit 1
    fi
    
    log_success "Build completed successfully"
    cd ..
}

# Run CPU baseline benchmarks
run_cpu_baseline() {
    log_info "Running CPU baseline benchmarks..."
    
    # Matrix multiplication baseline
    python3 << 'EOF'
import numpy as np
import time
import json

results = {}

# Matrix multiplication benchmark
sizes = [512, 1024, 2048]
for size in sizes:
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        C = np.dot(A, B)
    
    # Timing
    times = []
    for _ in range(10):
        start = time.time()
        C = np.dot(A, B)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    flops = 2.0 * size**3 / (avg_time / 1000.0) / 1e9
    
    results[f"cpu_matrix_multiply_{size}"] = {
        "avg_time_ms": avg_time,
        "gflops": flops
    }

# Eigenvalue computation baseline (using simple power iteration)
size = 512
A = np.random.rand(size, size).astype(np.float32)
A = (A + A.T) / 2  # Make symmetric

start = time.time()
eigenvals, eigenvecs = np.linalg.eigh(A)
end = time.time()

cpu_eigen_time = (end - start) * 1000
results["cpu_eigenvalue_512"] = {
    "avg_time_ms": cpu_eigen_time,
    "gflops": 0.0  # Placeholder
}

# Save results
with open("cpu_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"CPU baseline completed. Results saved to cpu_baseline.json")
EOF

    if [ $? -ne 0 ]; then
        log_error "CPU baseline benchmarks failed"
        exit 1
    fi
    
    log_success "CPU baseline benchmarks completed"
}

# Run GPU benchmarks
run_gpu_benchmarks() {
    log_info "Running GPU benchmarks..."
    
    cd "${BUILD_DIR}"
    
    # Run comprehensive benchmarks
    log_info "Executing CUDA benchmark suite..."
    
    # Set optimal GPU clocks if possible
    nvidia-smi -pm 1 > /dev/null 2>&1 || log_warning "Could not enable persistence mode"
    nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.graphics --format=csv,noheader,nounits | tr ',' ' ') > /dev/null 2>&1 || log_warning "Could not set maximum clocks"
    
    # Run benchmarks with detailed profiling
    ./bin/benchmark_cuda --detailed --export-json=gpu_results.json 2>&1 | tee benchmark.log
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "GPU benchmarks failed. Check benchmark.log for details."
        exit 1
    fi
    
    log_success "GPU benchmarks completed"
    cd ..
}

# Analyze performance and calculate speedups
analyze_performance() {
    log_info "Analyzing performance results..."
    
    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    
    # Copy results files
    cp "${BUILD_DIR}/cpu_baseline.json" "${RESULTS_DIR}/"
    cp "${BUILD_DIR}/gpu_results.json" "${RESULTS_DIR}/"
    
    # Analyze results using Python
    python3 << 'EOF'
import json
import sys
from pathlib import Path

def load_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return {}

# Load results
cpu_results = load_json("./performance_results/cpu_baseline.json")
gpu_results = load_json("./performance_results/gpu_results.json")

if not cpu_results or not gpu_results:
    print("Error: Could not load benchmark results")
    sys.exit(1)

print("\n=== PERFORMANCE ANALYSIS ===")
print(f"{'Operation':<30} {'CPU Time (ms)':<15} {'GPU Time (ms)':<15} {'Speedup':<10} {'Target Met':<12}")
print("-" * 85)

total_speedups = []
target_met_count = 0
total_tests = 0

# Matrix multiplication analysis
for size in [512, 1024, 2048]:
    cpu_key = f"cpu_matrix_multiply_{size}"
    gpu_key = f"Matrix Multiply {size}x{size}x{size}"
    
    if cpu_key in cpu_results and any(k["name"] == gpu_key for k in gpu_results.get("kernel_metrics", [])):
        cpu_time = cpu_results[cpu_key]["avg_time_ms"]
        gpu_metric = next(k for k in gpu_results["kernel_metrics"] if k["name"] == gpu_key)
        gpu_time = gpu_metric["avg_time_ms"]
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        target_met = speedup >= 10.0
        
        total_speedups.append(speedup)
        if target_met:
            target_met_count += 1
        total_tests += 1
        
        print(f"{'Matrix Multiply ' + str(size):<30} {cpu_time:<15.3f} {gpu_time:<15.3f} {speedup:<10.2f}x {'✅' if target_met else '❌':<12}")

# Eigenvalue computation analysis
cpu_eigen = cpu_results.get("cpu_eigenvalue_512", {})
gpu_eigen = next((k for k in gpu_results.get("kernel_metrics", []) if "eigenvalue" in k["name"].lower()), None)

if cpu_eigen and gpu_eigen:
    cpu_time = cpu_eigen["avg_time_ms"]
    gpu_time = gpu_eigen["avg_time_ms"]
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    target_met = speedup >= 10.0
    
    total_speedups.append(speedup)
    if target_met:
        target_met_count += 1
    total_tests += 1
    
    print(f"{'Eigenvalue Computation':<30} {cpu_time:<15.3f} {gpu_time:<15.3f} {speedup:<10.2f}x {'✅' if target_met else '❌':<12}")

# Summary
print("\n=== SUMMARY ===")
if total_speedups:
    avg_speedup = sum(total_speedups) / len(total_speedups)
    max_speedup = max(total_speedups)
    min_speedup = min(total_speedups)
    
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")
    print(f"Minimum Speedup: {min_speedup:.2f}x")
    print(f"Tests Meeting 10x Target: {target_met_count}/{total_tests} ({target_met_count/total_tests*100:.1f}%)")
    
    # Overall assessment
    overall_success = avg_speedup >= 10.0 and target_met_count / total_tests >= 0.8
    
    print(f"\n🎯 OVERALL TARGET ASSESSMENT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    
    if not overall_success:
        print("\n📋 OPTIMIZATION RECOMMENDATIONS:")
        if avg_speedup < 5.0:
            print("- Critical: Fundamental algorithmic improvements needed")
            print("- Consider using cuBLAS for matrix operations")
            print("- Implement Tensor Core optimizations")
        elif avg_speedup < 8.0:
            print("- Memory access pattern optimization")
            print("- Increase occupancy through better resource usage")
            print("- Implement cooperative groups for better synchronization")
        else:
            print("- Fine-tune memory coalescing")
            print("- Optimize shared memory usage")
            print("- Consider multi-GPU scaling")
    
    # Generate report file
    report = {
        "timestamp": "TIMESTAMP_PLACEHOLDER",
        "summary": {
            "average_speedup": avg_speedup,
            "maximum_speedup": max_speedup,
            "minimum_speedup": min_speedup,
            "tests_meeting_target": target_met_count,
            "total_tests": total_tests,
            "target_met_percentage": target_met_count/total_tests*100,
            "overall_success": overall_success
        },
        "cpu_results": cpu_results,
        "gpu_results": gpu_results,
        "detailed_analysis": total_speedups
    }
    
    with open("REPORT_FILE_PLACEHOLDER", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDetailed report saved to: REPORT_FILE_PLACEHOLDER")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)
else:
    print("No valid speedup measurements found")
    sys.exit(1)
EOF

    # Replace placeholders in the Python script
    python3 -c "
import json
import sys
from datetime import datetime

# Read the generated report template and update placeholders
try:
    with open('${RESULTS_DIR}/report_template.json', 'r') as f:
        content = f.read()
    
    content = content.replace('TIMESTAMP_PLACEHOLDER', '${TIMESTAMP}')
    content = content.replace('REPORT_FILE_PLACEHOLDER', '${REPORT_FILE}')
    
    # Parse and rewrite the final report
    report = json.loads(content)
    
    with open('${REPORT_FILE}', 'w') as f:
        json.dump(report, f, indent=2)
        
except FileNotFoundError:
    pass  # Template might not exist yet
"

    return $?
}

# Run memory profiling
run_memory_profiling() {
    log_info "Running memory profiling..."
    
    cd "${BUILD_DIR}"
    
    # Run with cuda-memcheck if available
    if command -v cuda-memcheck &> /dev/null; then
        log_info "Running memory error checking..."
        cuda-memcheck --tool memcheck ./bin/test_cuda_kernels 2>&1 | tee memcheck.log
        
        if grep -q "ERROR" memcheck.log; then
            log_warning "Memory errors detected. Check memcheck.log for details."
        else
            log_success "No memory errors detected"
        fi
    fi
    
    # Profile with nvprof if available
    if command -v nvprof &> /dev/null; then
        log_info "Running nvprof analysis..."
        nvprof --analysis-metrics -o profile.nvprof ./bin/benchmark_cuda --quick 2>&1 | tee nvprof.log
        log_success "Profiling data saved to profile.nvprof"
    fi
    
    cd ..
}

# Generate final report
generate_final_report() {
    log_info "Generating final performance report..."
    
    cat << EOF > "${RESULTS_DIR}/README.md"
# CUDA Performance Validation Report

**Generated:** $(date)
**Timestamp:** ${TIMESTAMP}

## System Information

- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader)
- **Driver:** $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
- **CUDA Version:** $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
- **Compute Capability:** $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
- **Memory:** $(nvidia-smi --query-gpu=memory.total --format=csv,noheader) 

## Performance Results

The detailed performance analysis can be found in:
- \`performance_report_${TIMESTAMP}.json\` - Complete numerical results
- \`cpu_baseline.json\` - CPU baseline measurements  
- \`gpu_results.json\` - GPU benchmark results

## Build Artifacts

- \`build.log\` - Compilation output
- \`benchmark.log\` - Benchmark execution log
- \`memcheck.log\` - Memory error checking results (if available)
- \`nvprof.log\` - Profiling output (if available)

## Optimization Notes

This validation targets a **10x speedup** over CPU baselines for mathematical
operations in the Geometric Langlands correspondence implementation.

Key optimization areas:
1. Matrix operations (GEMM, tensor contractions)
2. Eigenvalue computations (power iteration, Lanczos)
3. Sheaf cohomology calculations
4. Memory bandwidth utilization
5. Compute occupancy

## Files

EOF

    # List all generated files
    find "${RESULTS_DIR}" -type f -exec basename {} \; | sort >> "${RESULTS_DIR}/README.md"
    
    log_success "Final report generated in ${RESULTS_DIR}/"
}

# Main execution
main() {
    log_info "Starting CUDA Performance Validation"
    log_info "Target: 10x speedup over CPU baseline"
    log_info "Timestamp: ${TIMESTAMP}"
    
    check_prerequisites
    build_kernels
    run_cpu_baseline
    run_gpu_benchmarks
    
    # Analyze performance and capture exit code
    if analyze_performance; then
        PERFORMANCE_RESULT="SUCCESS"
        PERFORMANCE_EXIT_CODE=0
    else
        PERFORMANCE_RESULT="NEEDS_IMPROVEMENT"
        PERFORMANCE_EXIT_CODE=1
    fi
    
    run_memory_profiling
    generate_final_report
    
    # Final summary
    echo
    log_info "=== VALIDATION COMPLETE ==="
    log_info "Results directory: ${RESULTS_DIR}/"
    log_info "Performance target: ${PERFORMANCE_RESULT}"
    
    if [ $PERFORMANCE_EXIT_CODE -eq 0 ]; then
        log_success "🎯 10x speedup target ACHIEVED!"
    else
        log_warning "🎯 10x speedup target not yet achieved. See optimization recommendations."
    fi
    
    exit $PERFORMANCE_EXIT_CODE
}

# Execute main function
main "$@"