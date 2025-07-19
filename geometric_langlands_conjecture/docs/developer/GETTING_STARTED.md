# Getting Started with Geometric Langlands Framework

## 🚀 Quick Start Guide

This guide will get you up and running with the Geometric Langlands Conjecture computational framework in under 15 minutes.

## 📋 Prerequisites

### System Requirements
- **Rust 1.70+** (for latest async/await features)
- **Git** for version control
- **CUDA Toolkit 12.0+** (optional, for GPU acceleration)
- **Node.js 18+** (optional, for WASM development)
- **8GB+ RAM** (recommended for large computations)

### Mathematical Background
- **Linear Algebra**: Matrix operations, eigenvalues
- **Abstract Algebra**: Groups, rings, fields (helpful but not required)
- **Category Theory**: Basic concepts (helpful for advanced usage)

## 🔧 Installation

### 1. Clone the Repository

```bash
# Clone the main repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/geometric_langlands_conjecture

# Or clone directly (if available)
git clone https://github.com/ruvnet/geometric-langlands.git
cd geometric-langlands
```

### 2. Install Rust

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### 3. Build the Project

```bash
# Basic build (CPU only)
cargo build --release

# Build with all features
cargo build --release --all-features

# Run tests to verify installation
cargo test
```

### 4. Optional: CUDA Setup

```bash
# Install CUDA Toolkit (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Set environment variables
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build with CUDA support
cargo build --release --features cuda
```

### 5. Optional: WASM Setup

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM module
wasm-pack build --target web --features wasm

# Verify WASM output
ls pkg/
```

## 🧪 First Examples

### Example 1: Basic Group Operations

Create a new file `examples/my_first_example.rs`:

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    // Create a reductive group GL(2)
    let g = ReductiveGroup::gl_n(2);
    println!("Created group: {:?}", g);
    
    // Get the Lie algebra
    let lie_alg = g.lie_algebra();
    println!("Lie algebra dimension: {}", lie_alg.dimension());
    
    // Create a matrix representation
    let rep = MatrixRepresentation::standard(&g);
    println!("Standard representation dimension: {}", rep.dimension());
    
    Ok(())
}
```

Run the example:
```bash
cargo run --example my_first_example
```

### Example 2: Automorphic Forms

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    // Create GL(2) over a function field
    let g = ReductiveGroup::gl_n(2);
    
    // Create an Eisenstein series
    let eisenstein = AutomorphicForm::eisenstein_series(&g, 2)?;
    println!("Created Eisenstein series of weight 2");
    
    // Apply a Hecke operator
    let hecke = HeckeOperator::new(&g, 5)?;
    let eigenform = hecke.apply(&eisenstein)?;
    
    println!("Applied Hecke operator T_5");
    println!("Eigenvalue: {}", eigenform.eigenvalue());
    
    Ok(())
}
```

### Example 3: Galois Representations

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    // Create an elliptic curve
    let curve = AlgebraicVariety::elliptic_curve([1, 0, 1, -1, 0])?;
    println!("Created elliptic curve: y² + xy = x³ - x");
    
    // Construct the associated Galois representation
    let galois_rep = GaloisRepresentation::from_curve(&curve)?;
    println!("Galois representation rank: {}", galois_rep.rank());
    
    // Convert to local system
    let local_system = LocalSystem::from_galois_rep(&galois_rep)?;
    println!("Local system created on curve");
    
    Ok(())
}
```

### Example 4: GPU-Accelerated Computation

```rust
#[cfg(feature = "cuda")]
use geometric_langlands::{prelude::*, cuda::CudaContext};

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    // Initialize CUDA context
    let ctx = CudaContext::new()?;
    println!("CUDA context initialized");
    
    // Create a large Hecke operator matrix
    let g = ReductiveGroup::gl_n(3);
    let matrix = generate_large_hecke_matrix(&g, 1000)?;
    
    // Compute spectral decomposition on GPU
    let start = std::time::Instant::now();
    let decomp = matrix.spectral_decomposition_cuda(&ctx)?;
    let duration = start.elapsed();
    
    println!("GPU computation completed in {:?}", duration);
    println!("Found {} eigenvalues", decomp.eigenvalues().len());
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires CUDA support. Build with --features cuda");
}
```

## 📚 Core Concepts

### 1. Mathematical Objects

The framework provides strongly-typed mathematical objects:

```rust
// Algebraic structures
let field = Field::finite(101);  // F_101
let ring = Ring::polynomial(&field, 2);  // F_101[x,y]

// Geometric objects  
let curve = Curve::genus_g(2, &field);
let bundle = Bundle::vector_bundle(&curve, 3);

// Representations
let group = ReductiveGroup::gl_n(3);
let rep = group.standard_representation();
```

### 2. The Langlands Correspondence

The main correspondence is implemented as:

```rust
use geometric_langlands::langlands::Correspondence;

let correspondence = Correspondence::new(group, curve)?;

// Automorphic side
let automorphic_rep = correspondence.automorphic_representation(params)?;

// Galois side  
let galois_rep = correspondence.galois_representation(automorphic_rep)?;

// Verify the correspondence
assert!(correspondence.verify(&automorphic_rep, &galois_rep)?);
```

### 3. Performance Features

#### Parallel Computing
```rust
use geometric_langlands::utils::ParallelIterator;

// Parallel computation over collections
let results: Vec<_> = large_dataset
    .par_iter()
    .map(|item| expensive_computation(item))
    .collect();
```

#### GPU Acceleration
```rust
#[cfg(feature = "cuda")]
use geometric_langlands::cuda::{CudaMatrix, CudaContext};

let ctx = CudaContext::new()?;
let gpu_matrix = CudaMatrix::from_host(&cpu_matrix, &ctx)?;
let result = gpu_matrix.eigendecomposition()?;
```

#### WASM Deployment
```rust
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn compute_langlands_correspondence(data: &str) -> String {
    // Computation available in browser
    let result = perform_computation(data);
    serde_json::to_string(&result).unwrap()
}
```

## 🧪 Testing Your Installation

### Run the Test Suite

```bash
# Basic tests
cargo test

# Test with all features
cargo test --all-features

# Test specific modules
cargo test core::tests
cargo test automorphic::tests

# Integration tests
cargo test --test integration_test
```

### Benchmark Performance

```bash
# Run benchmarks
cargo bench

# Specific benchmarks
cargo bench langlands_correspondence
cargo bench hecke_operators

# Generate benchmark reports
cargo bench -- --output-format html
```

### Example Output

A successful installation should produce output like:

```
$ cargo test
    Finished test [unoptimized + debuginfo] target(s) in 12.34s
    
    Running unittests (src/lib.rs)
test core::tests::test_reductive_group ... ok
test automorphic::tests::test_eisenstein_series ... ok
test galois::tests::test_local_system ... ok
test langlands::tests::test_correspondence ... ok

test result: ok. 47 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

    Running tests/integration_test.rs
test langlands_correspondence_gl2 ... ok
test gpu_acceleration ... ok
test wasm_compatibility ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## 🚧 Troubleshooting

### Common Issues

#### 1. CUDA Compilation Errors
```bash
# Error: CUDA toolkit not found
export CUDA_PATH=/usr/local/cuda-12.0  # Adjust version
export PATH=$CUDA_PATH/bin:$PATH

# Error: nvcc not found  
sudo apt-get install nvidia-cuda-toolkit
```

#### 2. WASM Build Issues
```bash
# Error: wasm32 target not installed
rustup target add wasm32-unknown-unknown

# Error: wasm-pack not found
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### 3. Memory Issues
```bash
# Increase stack size for large computations
export RUST_MIN_STACK=8388608  # 8MB

# Or use with cargo
RUST_MIN_STACK=8388608 cargo run --example large_computation
```

#### 4. Dependency Conflicts
```bash
# Clean and rebuild
cargo clean
cargo build --release

# Update dependencies
cargo update
```

### Getting Help

1. **Check the Issues**: [GitHub Issues](https://github.com/ruvnet/ruv-FANN/issues)
2. **Read the Docs**: [Documentation](https://docs.rs/geometric-langlands)
3. **Mathematical Questions**: See [Math Background](../math/MATHEMATICAL_BACKGROUND.md)
4. **Performance**: See [Performance Guide](PERFORMANCE.md)

## 📖 Next Steps

### Learning Path

1. **Read the Mathematical Background**: [Mathematical Guide](../math/MATHEMATICAL_BACKGROUND.md)
2. **Explore Examples**: Browse the `examples/` directory
3. **Try Advanced Features**: GPU acceleration, WASM deployment
4. **Contribute**: See [Contributing Guide](CONTRIBUTING.md)

### Advanced Topics

- **[Architecture Overview](ARCHITECTURE.md)**: System design and patterns
- **[Performance Optimization](PERFORMANCE.md)**: CUDA and parallel computing
- **[Testing Strategies](TESTING.md)**: Property-based testing
- **[API Reference](../api/)**: Complete API documentation

### Community

- **Discussions**: Join mathematical and technical discussions
- **Contributions**: Help improve the framework
- **Research**: Collaborate on new mathematical insights

---

**Congratulations!** You're now ready to explore the fascinating world of computational geometric Langlands correspondence. Start with the examples above and gradually work your way up to more complex mathematical computations.

*Happy computing! 🧮✨*