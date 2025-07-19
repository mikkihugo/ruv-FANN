# Examples and Tutorials

This directory contains a comprehensive collection of examples and tutorials for the Geometric Langlands Conjecture framework, organized by complexity and topic.

## 📚 Organization

### By Complexity Level

#### 🟢 Basic Examples (`basic/`)
Perfect for getting started - no advanced mathematical background required.

- **[Hello Langlands](basic/hello_langlands.rs)** - Your first computation
- **[Group Operations](basic/group_operations.rs)** - Working with reductive groups  
- **[Simple Curves](basic/simple_curves.rs)** - Creating and manipulating curves
- **[Matrix Representations](basic/matrix_representations.rs)** - Basic representation theory

#### 🟡 Intermediate Examples (`intermediate/`)
Assumes familiarity with basic algebraic concepts.

- **[Automorphic Forms](intermediate/automorphic_forms.rs)** - Eisenstein series and Hecke operators
- **[Local Systems](intermediate/local_systems.rs)** - Galois representations and local systems
- **[Bundle Moduli](intermediate/bundle_moduli.rs)** - Working with moduli stacks
- **[Correspondence Basics](intermediate/correspondence_basics.rs)** - Simple Langlands correspondences

#### 🔴 Advanced Examples (`advanced/`)
For users comfortable with advanced mathematical concepts.

- **[Derived Categories](advanced/derived_categories.rs)** - D-modules and perverse sheaves
- **[Hecke Correspondences](advanced/hecke_correspondences.rs)** - Geometric Hecke operators
- **[S-Duality](advanced/s_duality.rs)** - Physics connections and gauge theory
- **[Neural Enhancement](advanced/neural_enhancement.rs)** - AI-assisted computations

### By Mathematical Topic

#### 🧮 Algebra (`algebra/`)
- Group theory and representations
- Ring and field operations
- Lie algebras and root systems

#### 📐 Geometry (`geometry/`)
- Algebraic varieties and schemes
- Curves and their properties
- Moduli spaces and stacks

#### 🔄 Correspondence (`correspondence/`)
- Automorphic-Galois connections
- Hecke operators and eigenforms
- L-functions and reciprocity

#### ⚡ Performance (`performance/`)
- GPU acceleration examples
- Parallel computing patterns
- Memory optimization techniques

#### 🌐 Applications (`applications/`)
- Cryptographic applications
- Quantum computing connections
- Physics interpretations

## 🚀 Quick Start Examples

### Example 1: Hello Langlands

```rust
// examples/basic/hello_langlands.rs
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    println!("🌟 Welcome to Geometric Langlands!");
    
    // Create a simple group
    let group = ReductiveGroup::gl_n(2);
    println!("Created GL(2): rank = {}", group.rank());
    
    // Create a curve
    let curve = Curve::rational_curve(FiniteField::new(5)?);
    println!("Created rational curve over F_5");
    
    // The Langlands correspondence connects these!
    println!("Group ↔ Automorphic representations");
    println!("Curve ↔ Galois representations");
    println!("Correspondence: Deep mathematical duality! ✨");
    
    Ok(())
}
```

### Example 2: Basic Computation

```rust
// examples/basic/basic_computation.rs
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    // Setup: GL(2) over a finite field
    let field = FiniteField::new(101)?;
    let group = ReductiveGroup::gl_n_over_field(2, Box::new(field));
    
    // Create an automorphic form (Eisenstein series)
    let eisenstein = AutomorphicForm::eisenstein_series(&group, 2)?;
    println!("Created Eisenstein series E_2");
    
    // Apply Hecke operator
    let hecke_5 = HeckeOperator::new(&group, 5)?;
    let eigenform = hecke_5.apply(&eisenstein)?;
    
    println!("Applied T_5 Hecke operator");
    println!("Eigenvalue: {}", eigenform.eigenvalue());
    
    // This eigenvalue encodes deep arithmetic information!
    
    Ok(())
}
```

## 📖 Tutorial Progression

### Tutorial Path for Mathematicians

1. **[Mathematical Foundations](tutorials/math_foundations.md)**
   - Review of algebraic geometry
   - Representation theory essentials
   - Category theory basics

2. **[The Classical Program](tutorials/classical_langlands.md)**
   - Automorphic forms over number fields
   - Galois representations
   - L-functions and reciprocity

3. **[Geometric Version](tutorials/geometric_langlands.md)**
   - Function fields and curves
   - D-modules and derived categories
   - The correspondence statement

4. **[Computational Implementation](tutorials/computational_methods.md)**
   - Discretization strategies
   - Algorithmic approaches
   - Verification methods

### Tutorial Path for Developers

1. **[Rust Basics for Math](tutorials/rust_for_math.md)**
   - Type-safe mathematical programming
   - Working with traits and generics
   - Error handling in mathematical contexts

2. **[Framework Architecture](tutorials/architecture_overview.md)**
   - Module organization
   - Core abstractions
   - Extension patterns

3. **[Performance Optimization](tutorials/performance_tuning.md)**
   - Parallel algorithms
   - GPU acceleration
   - Memory management

4. **[Contributing Code](tutorials/contributing.md)**
   - Development workflow
   - Testing strategies
   - Documentation standards

### Tutorial Path for Physicists

1. **[Mathematical Physics Background](tutorials/physics_background.md)**
   - Gauge theory and S-duality
   - Topological field theories
   - Quantum geometry

2. **[Langlands and Physics](tutorials/langlands_physics.md)**
   - Geometric Langlands as S-duality
   - Wilson and 't Hooft operators
   - Mirror symmetry connections

3. **[Computational Physics](tutorials/computational_physics.md)**
   - Implementing gauge theories
   - Quantum corrections
   - Numerical methods

## 🧪 Interactive Examples

### Jupyter Notebooks (Coming Soon)

The `notebooks/` directory will contain interactive Jupyter notebooks:

- **Introduction to Langlands.ipynb** - Interactive mathematical exploration
- **Visualization Examples.ipynb** - Plotting and visualization
- **Performance Analysis.ipynb** - Benchmarking and profiling
- **Research Playground.ipynb** - Experimental computations

### WASM Examples

Browser-based examples for easy experimentation:

```html
<!-- examples/wasm/langlands_web.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Langlands in the Browser</title>
    <script type="module">
        import init, { compute_correspondence } from './pkg/geometric_langlands.js';
        
        async function run() {
            await init();
            
            const group_data = { type: "GL", rank: 2 };
            const curve_data = { genus: 1, field_size: 101 };
            
            const result = compute_correspondence(group_data, curve_data);
            console.log("Correspondence computed:", result);
        }
        
        run();
    </script>
</head>
<body>
    <h1>Geometric Langlands in Your Browser! 🌐</h1>
    <p>Check the console for results.</p>
</body>
</html>
```

## 📊 Benchmarking Examples

### Performance Comparison

```rust
// examples/performance/benchmark_comparison.rs
use criterion::{criterion_group, criterion_main, Criterion};
use geometric_langlands::prelude::*;

fn benchmark_hecke_operators(c: &mut Criterion) {
    let group = ReductiveGroup::gl_n(3);
    let form = AutomorphicForm::eisenstein_series(&group, 2).unwrap();
    
    let mut group = c.benchmark_group("hecke_operators");
    
    // CPU benchmark
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let hecke = HeckeOperator::new(&group, 7).unwrap();
            hecke.apply(&form)
        })
    });
    
    // GPU benchmark (if available)
    #[cfg(feature = "cuda")]
    group.bench_function("gpu", |b| {
        let ctx = CudaContext::new().unwrap();
        b.iter(|| {
            let hecke = HeckeOperator::new_cuda(&group, 7, &ctx).unwrap();
            hecke.apply_cuda(&form)
        })
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_hecke_operators);
criterion_main!(benches);
```

## 🔬 Research Examples

### Experimental Computations

```rust
// examples/research/correspondence_search.rs
use geometric_langlands::prelude::*;

/// Search for new correspondences using neural enhancement
fn search_correspondences() -> Result<()> {
    let neural_engine = NeuralEngine::new()?;
    
    // Generate test cases
    for rank in 2..=5 {
        for genus in 0..=3 {
            let group = ReductiveGroup::gl_n(rank);
            let curve = Curve::genus_g(genus, FiniteField::new(101)?)?;
            
            // Use neural network to predict correspondence
            let prediction = neural_engine.predict_correspondence(&group, &curve)?;
            
            // Verify prediction mathematically
            let verification = verify_correspondence(&group, &curve, &prediction)?;
            
            if verification.confidence > 0.95 {
                println!("🎯 Strong correspondence found: GL({}) ↔ genus {} curve", rank, genus);
                println!("   Neural confidence: {:.3}", prediction.confidence);
                println!("   Mathematical verification: {:.3}", verification.confidence);
            }
        }
    }
    
    Ok(())
}
```

## 🎓 Educational Examples

### Step-by-Step Learning

```rust
// examples/educational/step_by_step_langlands.rs

/// Educational example that builds up the correspondence step by step
fn educational_langlands() -> Result<()> {
    println!("📚 Step-by-Step Langlands Correspondence");
    println!("=========================================");
    
    // Step 1: Classical setup
    println!("\n🔢 Step 1: Classical Number Theory");
    let number_field = NumberField::rationals();
    println!("Working over: ℚ (rational numbers)");
    
    // Step 2: Move to function fields  
    println!("\n📐 Step 2: Geometric Setup");
    let curve = Curve::elliptic_curve_over_fq(101)?;
    let function_field = curve.function_field();
    println!("Function field: F_101(elliptic curve)");
    
    // Step 3: Automorphic side
    println!("\n🎵 Step 3: Automorphic Forms");
    let group = ReductiveGroup::gl_n(2);
    let automorphic_rep = AutomorphicRepresentation::new(&group, &function_field)?;
    println!("Created automorphic representation of GL(2)");
    
    // Step 4: Galois side
    println!("\n🔄 Step 4: Galois Representations");
    let galois_rep = GaloisRepresentation::from_curve(&curve)?;
    let local_system = LocalSystem::from_galois_rep(&galois_rep)?;
    println!("Created corresponding local system");
    
    // Step 5: The correspondence!
    println!("\n✨ Step 5: The Correspondence");
    let correspondence = Correspondence::new(&group, &curve)?;
    let verified = correspondence.verify(&automorphic_rep, &local_system)?;
    
    if verified {
        println!("🎉 Correspondence verified! Automorphic ↔ Galois");
    } else {
        println!("❌ Correspondence failed verification");
    }
    
    println!("\n🌟 This is the magic of Langlands duality!");
    
    Ok(())
}
```

## 🛠️ Development Examples

### Testing Patterns

```rust
// examples/development/testing_patterns.rs
use geometric_langlands::prelude::*;
use proptest::prelude::*;

// Property-based testing example
proptest! {
    #[test]
    fn test_correspondence_functoriality(
        rank in 2usize..6,
        genus in 0usize..4,
        prime in prop::num::usize::range(5, 100).prop_filter("prime", |&p| is_prime(p))
    ) {
        let group = ReductiveGroup::gl_n(rank);
        let curve = Curve::genus_g(genus, FiniteField::new(prime)?)?;
        
        let correspondence = Correspondence::new(&group, &curve)?;
        
        // Test functoriality property
        let auto_rep1 = generate_automorphic_rep(&group)?;
        let auto_rep2 = generate_automorphic_rep(&group)?;
        
        let galois_rep1 = correspondence.to_galois(&auto_rep1)?;
        let galois_rep2 = correspondence.to_galois(&auto_rep2)?;
        
        // Correspondence should preserve tensor products
        let tensor_auto = auto_rep1.tensor(&auto_rep2)?;
        let tensor_galois = galois_rep1.tensor(&galois_rep2)?;
        
        let correspondence_of_tensor = correspondence.to_galois(&tensor_auto)?;
        
        assert_eq!(correspondence_of_tensor, tensor_galois);
    }
}
```

## 📈 Usage Analytics

Track which examples are most helpful:

```rust
// examples/utils/analytics.rs

pub fn track_example_usage(example_name: &str) {
    // Optional analytics for improving documentation
    println!("📊 Example '{}' completed successfully", example_name);
}

pub fn suggest_next_examples(current: &str) -> Vec<&'static str> {
    match current {
        "hello_langlands" => vec!["group_operations", "simple_curves"],
        "group_operations" => vec!["matrix_representations", "automorphic_forms"],
        "automorphic_forms" => vec!["local_systems", "correspondence_basics"],
        _ => vec!["See examples/README.md for full progression"]
    }
}
```

## 🚀 Running Examples

### Basic Usage

```bash
# Run a specific example
cargo run --example hello_langlands

# Run with features
cargo run --example gpu_acceleration --features cuda

# Run all basic examples
for example in examples/basic/*.rs; do
    cargo run --example $(basename $example .rs)
done
```

### Interactive Mode

```bash
# Start interactive exploration
cargo run --example interactive_explorer

# REPL-style interface
>> let group = ReductiveGroup::gl_n(2)
>> let curve = Curve::genus_g(1, FiniteField::new(7)?)
>> let correspondence = Correspondence::new(&group, &curve)?
>> correspondence.verify_known_cases()
```

### Benchmarking

```bash
# Run performance benchmarks
cargo bench --features bench

# Compare different implementations
cargo bench correspondence_algorithms

# Profile memory usage
cargo run --example memory_profiling --features profiling
```

## 🤝 Contributing Examples

We welcome example contributions! See [Contributing Guide](../developer/CONTRIBUTING.md) for:

- Example quality standards
- Mathematical accuracy requirements
- Code style guidelines
- Documentation expectations

### Example Template

```rust
// examples/template.rs

//! # Example Title
//! 
//! Brief description of what this example demonstrates.
//! 
//! ## Mathematical Background
//! 
//! Explain the mathematical concepts used.
//! 
//! ## Learning Objectives
//! 
//! - Objective 1
//! - Objective 2
//! 
//! ## Prerequisites
//! 
//! - Required background knowledge
//! - Other examples to try first

use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    println!("🎯 Example: {}", env!("CARGO_PKG_NAME"));
    
    // Your example code here
    
    println!("✅ Example completed successfully!");
    
    // Suggest next steps
    println!("\n📚 Try these next:");
    for suggestion in suggest_next_examples("your_example") {
        println!("  - {}", suggestion);
    }
    
    Ok(())
}
```

---

*These examples are designed to make the geometric Langlands correspondence accessible to everyone, from curious beginners to expert mathematicians. Start with the basics and work your way up to cutting-edge research applications! 🌟*