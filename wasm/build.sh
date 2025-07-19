#!/bin/bash

# Geometric Langlands WASM Build Script
# Optimized for minimal bundle size and maximum performance

set -e

echo "🌟 Building Geometric Langlands WASM module..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_requirements() {
    print_status "Checking build requirements..."
    
    if ! command -v wasm-pack &> /dev/null; then
        print_error "wasm-pack is required but not installed."
        echo "Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        exit 1
    fi
    
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo is required but not installed."
        echo "Install from: https://rustup.rs/"
        exit 1
    fi
    
    print_success "All requirements met!"
}

# Clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    rm -rf pkg pkg-node pkg-bundler target/wasm32-unknown-unknown
    print_success "Build artifacts cleaned!"
}

# Build for web (ES modules)
build_web() {
    print_status "Building for web (ES modules)..."
    wasm-pack build \
        --target web \
        --out-dir pkg \
        --release \
        --no-typescript \
        -- --features wasm
    
    if [ $? -eq 0 ]; then
        print_success "Web build completed!"
    else
        print_error "Web build failed!"
        exit 1
    fi
}

# Build for Node.js
build_node() {
    print_status "Building for Node.js..."
    wasm-pack build \
        --target nodejs \
        --out-dir pkg-node \
        --release \
        --no-typescript \
        -- --features wasm
    
    if [ $? -eq 0 ]; then
        print_success "Node.js build completed!"
    else
        print_error "Node.js build failed!"
        exit 1
    fi
}

# Build for bundlers (webpack, rollup, etc.)
build_bundler() {
    print_status "Building for bundlers..."
    wasm-pack build \
        --target bundler \
        --out-dir pkg-bundler \
        --release \
        --no-typescript \
        -- --features wasm
    
    if [ $? -eq 0 ]; then
        print_success "Bundler build completed!"
    else
        print_error "Bundler build failed!"
        exit 1
    fi
}

# Optimize WASM binary
optimize_wasm() {
    print_status "Optimizing WASM binaries..."
    
    # Check if wasm-opt is available
    if command -v wasm-opt &> /dev/null; then
        for dir in pkg pkg-node pkg-bundler; do
            if [ -d "$dir" ]; then
                for wasm_file in "$dir"/*.wasm; do
                    if [ -f "$wasm_file" ]; then
                        print_status "Optimizing $wasm_file..."
                        wasm-opt -Oz --enable-bulk-memory "$wasm_file" -o "$wasm_file.opt"
                        mv "$wasm_file.opt" "$wasm_file"
                    fi
                done
            fi
        done
        print_success "WASM optimization completed!"
    else
        print_warning "wasm-opt not found. Install binaryen for size optimization."
        print_warning "Ubuntu/Debian: sudo apt install binaryen"
        print_warning "macOS: brew install binaryen"
    fi
}

# Generate TypeScript definitions
generate_types() {
    print_status "Generating TypeScript definitions..."
    
    for dir in pkg pkg-node pkg-bundler; do
        if [ -d "$dir" ]; then
            cat > "$dir/index.d.ts" << 'EOF'
/* tslint:disable */
/* eslint-disable */
/**
 * WebAssembly bindings for the Geometric Langlands Conjecture framework
 */

export function init(): Promise<void>;
export function get_version(): string;
export function get_library_info(): any;

export class WasmConfig {
  constructor();
  max_workers: number;
  memory_limit: number;
  enable_gpu: boolean;
  optimization_level: number;
}

export class LanglandsEngine {
  constructor(config?: WasmConfig);
  initialize(): Promise<void>;
  compute_correspondence(group_type: string, dimension: number): Promise<any>;
  get_performance_metrics(): any;
  estimate_memory_usage(): number;
  is_ready(): boolean;
}

export class PerformanceMonitor {
  constructor();
  mark(label: string): void;
  get_metrics(): any;
}

export class MathUtils {
  static is_prime(n: number): boolean;
  static euler_totient(n: number): number;
  static primes_up_to(n: number): number[];
}

export class WebGLUtils {
  static get_webgl_info(): any;
}

export interface LanglandsResult {
  group_type: string;
  dimension: number;
  success: boolean;
  confidence: number;
  computation_time: number;
  steps: CorrespondenceStep[];
  mathematical_data: MathematicalData;
}

export interface CorrespondenceStep {
  step: string;
  progress: number;
  timestamp: number;
}

export interface MathematicalData {
  galois_representation: string;
  automorphic_form: string;
  l_function: string;
  conductor: number;
  hodge_numbers: number[];
}
EOF
        fi
    done
    
    print_success "TypeScript definitions generated!"
}

# Show build statistics
show_stats() {
    print_status "Build Statistics:"
    echo ""
    
    for dir in pkg pkg-node pkg-bundler; do
        if [ -d "$dir" ]; then
            echo "📦 $dir:"
            for wasm_file in "$dir"/*.wasm; do
                if [ -f "$wasm_file" ]; then
                    size=$(ls -lah "$wasm_file" | awk '{print $5}')
                    echo "   WASM: $size"
                fi
            done
            
            for js_file in "$dir"/*.js; do
                if [ -f "$js_file" ]; then
                    size=$(ls -lah "$js_file" | awk '{print $5}')
                    echo "   JS: $size"
                fi
            done
            echo ""
        fi
    done
}

# Test the build
test_build() {
    print_status "Testing builds..."
    
    # Test Node.js build if available
    if [ -d "pkg-node" ] && command -v node &> /dev/null; then
        print_status "Testing Node.js build..."
        node -e "
            const wasm = require('./pkg-node/geometric_langlands_wasm.js');
            console.log('✅ Node.js build working, version:', wasm.get_version());
        " 2>/dev/null && print_success "Node.js build test passed!" || print_warning "Node.js build test failed"
    fi
    
    print_success "Build testing completed!"
}

# Create demo files
setup_demo() {
    print_status "Setting up demo files..."
    
    # Create a simple package.json for the demo
    cat > demo/package.json << 'EOF'
{
  "name": "geometric-langlands-demo",
  "version": "0.1.0",
  "description": "Interactive demo for Geometric Langlands WASM",
  "scripts": {
    "serve": "python3 -m http.server 8000",
    "serve-alt": "npx serve ."
  },
  "devDependencies": {
    "serve": "^14.0.0"
  }
}
EOF

    # Create a simple README for the demo
    cat > demo/README.md << 'EOF'
# Geometric Langlands Demo

This directory contains an interactive web demo of the Geometric Langlands Conjecture framework.

## Running the Demo

1. Make sure the WASM module is built:
   ```bash
   cd .. && ./build.sh
   ```

2. Start a local server:
   ```bash
   npm run serve
   # or
   python3 -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser

## Features

- Interactive Langlands correspondence computation
- Real-time performance monitoring
- WebAssembly-powered mathematical functions
- GPU acceleration support (WebGL)
- Responsive design for mobile devices

## Browser Requirements

- Modern browser with WebAssembly support
- WebGL support for GPU acceleration (optional)
- ES6 modules support
EOF

    print_success "Demo files created!"
}

# Main build process
main() {
    echo "🌟 Geometric Langlands WASM Build System"
    echo "========================================"
    echo ""
    
    check_requirements
    
    # Parse command line arguments
    BUILD_TARGET="all"
    SKIP_OPTIMIZE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            --skip-optimize)
                SKIP_OPTIMIZE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --target TARGET    Build target: web, node, bundler, or all (default: all)"
                echo "  --skip-optimize    Skip WASM optimization step"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    clean_build
    
    case $BUILD_TARGET in
        web)
            build_web
            ;;
        node)
            build_node
            ;;
        bundler)
            build_bundler
            ;;
        all)
            build_web
            build_node
            build_bundler
            ;;
        *)
            print_error "Invalid target: $BUILD_TARGET"
            exit 1
            ;;
    esac
    
    if [ "$SKIP_OPTIMIZE" = false ]; then
        optimize_wasm
    fi
    
    generate_types
    setup_demo
    show_stats
    test_build
    
    echo ""
    print_success "🎉 Build completed successfully!"
    echo ""
    echo "📁 Output directories:"
    echo "   pkg/         - Web/ES modules build"
    echo "   pkg-node/    - Node.js build"  
    echo "   pkg-bundler/ - Bundler build"
    echo "   demo/        - Interactive demo"
    echo ""
    echo "🚀 To run the demo:"
    echo "   cd demo && python3 -m http.server 8000"
    echo "   then open http://localhost:8000"
    echo ""
}

# Run the main function
main "$@"