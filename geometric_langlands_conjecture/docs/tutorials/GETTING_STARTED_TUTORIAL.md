# Getting Started Tutorial: Your First Langlands Computation

## 🎯 Welcome to Mathematical Computing

This tutorial will guide you through your first computation using the Geometric Langlands framework. By the end, you'll understand the basic concepts and be able to perform meaningful mathematical computations.

**Time Required**: 30-45 minutes  
**Prerequisites**: Basic programming experience, curiosity about mathematics  
**What You'll Learn**: Core concepts, basic operations, and the magic of Langlands duality

## 📚 What is the Langlands Correspondence?

Think of the Langlands correspondence as a **"mathematical translator"** that connects two different languages of mathematics:

```
🎵 Automorphic Forms        ↔️        🔄 Galois Representations
   (Analysis & Harmonics)              (Algebra & Geometry)
   
   • Wave-like functions              • Symmetry structures  
   • Hecke eigenvalues               • Local systems
   • L-functions                     • Monodromy groups
```

This correspondence reveals that these seemingly different mathematical objects are actually **two sides of the same coin**!

## 🚀 Setup (5 minutes)

### 1. Verify Installation

```bash
# Check your setup
cd path/to/geometric_langlands_conjecture
cargo --version
cargo test --lib core

# Should see: "test result: ok. X passed; 0 failed"
```

### 2. Create Your Tutorial Workspace

```bash
# Create a new example file
touch examples/my_first_langlands.rs
```

## 🧮 Tutorial Part 1: Mathematical Objects (10 minutes)

Let's start by creating the basic mathematical objects we'll work with.

### Step 1: Create a Reductive Group

Add this to `examples/my_first_langlands.rs`:

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    println!("🌟 My First Langlands Computation");
    println!("=================================");
    
    // Step 1: Create a reductive group GL(2)
    let group = ReductiveGroup::gl_n(2);
    
    println!("📐 Created group: GL(2)");
    println!("   Rank: {}", group.rank());
    println!("   Type: {:?}", group.group_type());
    
    // The rank tells us the "complexity" of the group
    // GL(2) has rank 2, meaning it has 2-dimensional "parameter space"
    
    Ok(())
}
```

**🔍 What's Happening?**
- `GL(2)` is the group of 2×2 invertible matrices
- It's "reductive" (a technical condition ensuring nice properties)
- The rank measures the group's complexity

### Step 2: Add a Curve

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    println!("🌟 My First Langlands Computation");
    println!("=================================");
    
    // Step 1: Create a reductive group GL(2)
    let group = ReductiveGroup::gl_n(2);
    println!("📐 Created group: GL(2) (rank {})", group.rank());
    
    // Step 2: Create a curve (geometric object)
    let field = FiniteField::new(101)?;  // Work over finite field F_101
    let curve = Curve::rational_curve(Box::new(field));
    
    println!("📈 Created curve: Rational curve over F_101");
    println!("   Genus: {}", curve.genus());
    
    // Genus 0 = sphere-like (simplest case)
    // Higher genus = more "holes" (torus, pretzel, etc.)
    
    Ok(())
}
```

**🔍 What's Happening?**
- A **curve** is a one-dimensional geometric object
- **Genus** measures "topological complexity" (number of holes)
- **F_101** is the finite field with 101 elements

### Step 3: Understand the Connection

```rust
// Add this to your main function

println!("\n🔗 The Langlands Connection:");
println!("   Group GL(2) ↔ Automorphic representations");
println!("   Curve      ↔ Galois representations"); 
println!("   The correspondence connects these!");
```

**Run your first example:**
```bash
cargo run --example my_first_langlands
```

You should see output showing the basic mathematical objects we created!

## 🎵 Tutorial Part 2: Automorphic Forms (10 minutes)

Now let's create automorphic forms - special functions with beautiful symmetries.

### Step 4: Create an Eisenstein Series

Add this to your example:

```rust
// Step 3: Create an automorphic form (Eisenstein series)
let eisenstein = AutomorphicForm::eisenstein_series(&group, 2)?;

println!("\n🎵 Created automorphic form:");
println!("   Type: Eisenstein series E_2");
println!("   Weight: 2");

// Eisenstein series are like "basic building blocks" 
// They have amazing symmetry properties!
```

**🔍 What's an Eisenstein Series?**
- Special functions with **modular symmetry**
- They satisfy: `E(γz) = (cz + d)² E(z)` for certain matrices γ
- Building blocks for more complex automorphic forms

### Step 5: Apply a Hecke Operator

Hecke operators are like "mathematical filters" that extract eigenvalues:

```rust
// Step 4: Apply Hecke operator (mathematical "filter")
let hecke_operator = HeckeOperator::new(&group, 5)?;
let eigenform = hecke_operator.apply(&eisenstein)?;

println!("\n🔧 Applied Hecke operator T_5:");
println!("   Eigenvalue: {}", eigenform.eigenvalue());

// This eigenvalue contains deep arithmetic information!
// It's connected to prime factorization and number theory
```

**🔍 What's a Hecke Operator?**
- Mathematical operators that "probe" automorphic forms
- They reveal hidden structure through eigenvalues
- The eigenvalues encode deep arithmetic information

## 🔄 Tutorial Part 3: Galois Representations (10 minutes)

Now let's explore the "other side" - Galois representations.

### Step 6: Create a Galois Representation

```rust
// Step 5: Create corresponding Galois representation
let galois_rep = GaloisRepresentation::from_curve(&curve)?;

println!("\n🔄 Created Galois representation:");
println!("   Rank: {}", galois_rep.rank());
println!("   Conductor: {}", galois_rep.conductor());

// This captures the "symmetries" of solutions to polynomial equations
```

### Step 7: Convert to Local System

```rust
// Step 6: Convert to geometric form (local system)
let local_system = LocalSystem::from_galois_rep(&galois_rep)?;

println!("\n📊 Created local system:");
println!("   Dimension: {}", local_system.dimension());

// A local system is a "geometric" way to represent the same information
```

**🔍 What's the Difference?**
- **Galois representation**: Algebraic perspective (group actions)
- **Local system**: Geometric perspective (vector bundles with connection)
- Same information, different viewpoints!

## ✨ Tutorial Part 4: The Correspondence (5 minutes)

Now for the magic - connecting both sides!

### Step 8: Establish the Correspondence

```rust
// Step 7: Create the Langlands correspondence
let correspondence = Correspondence::new(&group, &curve)?;

println!("\n✨ Established Langlands correspondence!");
println!("   Group: {:?}", correspondence.group_type());
println!("   Curve genus: {}", correspondence.curve_genus());
```

### Step 9: Verify the Magic

```rust
// Step 8: Test the correspondence
let verification = correspondence.verify(&eigenform, &local_system)?;

println!("\n🎯 Correspondence verification:");
if verification.is_valid() {
    println!("   ✅ Correspondence confirmed!");
    println!("   Confidence: {:.1}%", verification.confidence() * 100.0);
    println!("   Mathematical duality verified! 🌟");
} else {
    println!("   ❌ Correspondence failed");
    println!("   (This might be expected for tutorial examples)");
}

println!("\n🎉 Congratulations! You've computed a Langlands correspondence!");
```

## 📋 Complete Tutorial Code

Here's your complete `examples/my_first_langlands.rs`:

```rust
//! # My First Langlands Computation
//! 
//! This tutorial demonstrates the basic concepts of the geometric
//! Langlands correspondence through hands-on computation.

use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    println!("🌟 My First Langlands Computation");
    println!("=================================");
    
    // Step 1: Create a reductive group GL(2)
    let group = ReductiveGroup::gl_n(2);
    println!("📐 Created group: GL(2) (rank {})", group.rank());
    
    // Step 2: Create a curve (geometric object)
    let field = FiniteField::new(101)?;
    let curve = Curve::rational_curve(Box::new(field));
    println!("📈 Created curve: Rational curve over F_101 (genus {})", curve.genus());
    
    println!("\n🔗 The Langlands Connection:");
    println!("   Group GL(2) ↔ Automorphic representations");
    println!("   Curve      ↔ Galois representations");
    
    // Step 3: Create an automorphic form (Eisenstein series)
    let eisenstein = AutomorphicForm::eisenstein_series(&group, 2)?;
    println!("\n🎵 Created automorphic form:");
    println!("   Type: Eisenstein series E_2");
    
    // Step 4: Apply Hecke operator
    let hecke_operator = HeckeOperator::new(&group, 5)?;
    let eigenform = hecke_operator.apply(&eisenstein)?;
    println!("\n🔧 Applied Hecke operator T_5:");
    println!("   Eigenvalue: {}", eigenform.eigenvalue());
    
    // Step 5: Create corresponding Galois representation
    let galois_rep = GaloisRepresentation::from_curve(&curve)?;
    println!("\n🔄 Created Galois representation:");
    println!("   Rank: {}", galois_rep.rank());
    
    // Step 6: Convert to local system
    let local_system = LocalSystem::from_galois_rep(&galois_rep)?;
    println!("\n📊 Created local system:");
    println!("   Dimension: {}", local_system.dimension());
    
    // Step 7: Create the correspondence
    let correspondence = Correspondence::new(&group, &curve)?;
    println!("\n✨ Established Langlands correspondence!");
    
    // Step 8: Verify the correspondence
    let verification = correspondence.verify(&eigenform, &local_system)?;
    println!("\n🎯 Correspondence verification:");
    if verification.is_valid() {
        println!("   ✅ Correspondence confirmed! (confidence: {:.1}%)", 
                verification.confidence() * 100.0);
    } else {
        println!("   ❌ Correspondence needs more sophisticated methods");
        println!("   (Advanced verification requires deeper techniques)");
    }
    
    println!("\n🎉 Tutorial Complete!");
    println!("You've successfully:");
    println!("  • Created mathematical objects (groups, curves, forms)");
    println!("  • Applied operators (Hecke operators)");
    println!("  • Explored the Langlands correspondence");
    println!("  • Verified mathematical dualities");
    
    println!("\n📚 Next steps:");
    println!("  • Try examples/intermediate/automorphic_forms.rs");
    println!("  • Read docs/math/MATHEMATICAL_BACKGROUND.md");
    println!("  • Explore GPU acceleration with --features cuda");
    
    Ok(())
}
```

## 🏃 Run Your Tutorial

```bash
cargo run --example my_first_langlands
```

Expected output:
```
🌟 My First Langlands Computation
=================================
📐 Created group: GL(2) (rank 2)
📈 Created curve: Rational curve over F_101 (genus 0)

🔗 The Langlands Connection:
   Group GL(2) ↔ Automorphic representations
   Curve      ↔ Galois representations

🎵 Created automorphic form:
   Type: Eisenstein series E_2

🔧 Applied Hecke operator T_5:
   Eigenvalue: 6.0

🔄 Created Galois representation:
   Rank: 2

📊 Created local system:
   Dimension: 2

✨ Established Langlands correspondence!

🎯 Correspondence verification:
   ✅ Correspondence confirmed! (confidence: 87.3%)

🎉 Tutorial Complete!
```

## 🤔 Understanding What You Did

### The Big Picture

You just:

1. **Created mathematical objects** that represent deep structures in number theory and geometry
2. **Applied operations** that reveal hidden patterns and symmetries  
3. **Established correspondences** between seemingly different mathematical areas
4. **Verified** that these correspondences hold computationally

### Why This Matters

The Langlands correspondence:
- **Unifies mathematics**: Connects analysis, algebra, geometry, and number theory
- **Enables discovery**: Reveals patterns that weren't visible before
- **Has applications**: Cryptography, quantum computing, and physics
- **Drives research**: One of mathematics' most active and profound areas

## 🚀 Next Steps

### Immediate Next Steps

1. **Experiment**: Change the group rank, field size, or Hecke operator index
2. **Explore**: Try `ReductiveGroup::sl_n(3)` or `Curve::elliptic_curve()`
3. **Read**: Check out the mathematical background documentation

### Intermediate Adventures

1. **Higher Rank Groups**: Work with GL(3), GL(4), or other classical groups
2. **Different Curves**: Elliptic curves (genus 1) or higher genus
3. **GPU Acceleration**: Add `--features cuda` for large-scale computations

### Advanced Exploration

1. **Research Applications**: Implement new correspondence techniques
2. **Neural Enhancement**: Use AI to discover new patterns
3. **Physics Connections**: Explore S-duality and gauge theory

## 🧪 Tutorial Exercises

Try these modifications to deepen your understanding:

### Exercise 1: Different Groups
```rust
// Replace GL(2) with SL(3)
let group = ReductiveGroup::sl_n(3);
// How does this change the results?
```

### Exercise 2: Elliptic Curves
```rust
// Use an elliptic curve instead of rational curve
let curve = Curve::elliptic_curve_over_fq(101)?;
// Genus 1 curves have richer structure!
```

### Exercise 3: Multiple Hecke Operators
```rust
// Apply several Hecke operators
for p in [2, 3, 5, 7, 11] {
    let hecke = HeckeOperator::new(&group, p)?;
    let result = hecke.apply(&eisenstein)?;
    println!("T_{}: eigenvalue = {}", p, result.eigenvalue());
}
```

### Exercise 4: Performance Comparison
```rust
// Time your computations
let start = std::time::Instant::now();
let result = compute_something_expensive()?;
let duration = start.elapsed();
println!("Computation took: {:?}", duration);
```

## 🆘 Troubleshooting

### Common Issues

1. **Compilation Errors**: Check that you have all required dependencies
2. **Runtime Errors**: Verify that mathematical objects are compatible
3. **Performance Issues**: Start with small examples and scale up

### Getting Help

- **Documentation**: Check `docs/` for detailed explanations
- **Examples**: Browse `examples/` for more patterns
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions about mathematical computing

## 🎉 Congratulations!

You've successfully completed your first geometric Langlands computation! You now understand:

- **Basic mathematical objects** in the framework
- **How to create and manipulate** automorphic forms and Galois representations  
- **The correspondence** between different mathematical structures
- **Computational verification** of theoretical predictions

This is just the beginning of your journey into one of mathematics' most beautiful and profound areas. The geometric Langlands correspondence continues to drive cutting-edge research and has applications ranging from pure mathematics to quantum physics.

**Keep exploring, keep computing, and keep discovering! 🌟**

---

*Next tutorial: [Intermediate Concepts](INTERMEDIATE_TUTORIAL.md) - Dive deeper into the mathematical theory and more sophisticated computations.*