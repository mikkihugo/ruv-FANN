//! Demonstration of the core type system
//!
//! This example shows how to create and work with the fundamental
//! mathematical structures in the Geometric Langlands implementation.

use geometric_langlands::prelude::*;

fn main() -> geometric_langlands::Result<()> {
    println!("🧮 Geometric Langlands Core Type System Demo");
    println!("============================================");
    
    // Create fundamental fields
    println!("\n📐 Creating Fields:");
    let rationals = Field::rationals();
    println!("  ℚ (rationals): {:?}", rationals);
    
    let finite_field = Field::finite_field(7);
    println!("  𝔽₇ (finite field): {:?}", finite_field);
    
    let extension = Field::extension(&rationals, 2);
    println!("  ℚ[√2] (quadratic extension): {:?}", extension);
    
    // Create reductive groups
    println!("\n🔗 Creating Reductive Groups:");
    let gl3 = ReductiveGroup::gl_n(3);
    println!("  GL(3): {:?}", gl3);
    println!("    Rank: {}, Dimension: {}", gl3.rank, gl3.dimension);
    println!("    Root system: {}", gl3.root_system);
    
    let sl2 = ReductiveGroup::sl_n(2);
    println!("  SL(2): {:?}", sl2);
    println!("    Rank: {}, Dimension: {}", sl2.rank, sl2.dimension);
    
    let so4 = ReductiveGroup::so_n(4);
    println!("  SO(4): {:?}", so4);
    println!("    Root system: {}", so4.root_system);
    
    let sp4 = ReductiveGroup::sp_2n(2);
    println!("  Sp(4): {:?}", sp4);
    println!("    Root system: {}", sp4.root_system);
    
    // Work with Lie algebras
    println!("\n🌀 Lie Algebras:");
    let gl3_lie = gl3.lie_algebra();
    println!("  𝔤𝔩₃ Lie algebra: {:?}", gl3_lie);
    
    // Create polynomial rings
    println!("\n💍 Creating Rings:");
    let poly_ring = Ring::polynomial_ring(rationals.clone(), 3);
    println!("  ℚ[x,y,z]: {:?}", poly_ring);
    
    // Matrix representations
    println!("\n🔢 Matrix Representations:");
    let identity_rep = MatrixRepresentation::identity(gl3.clone(), 3);
    println!("  Identity representation: {}×{} matrix", 
             identity_rep.matrix.nrows(), 
             identity_rep.matrix.ncols());
    
    // Group operations
    println!("\n⚡ Group Operations:");
    let abstract_group = gl3.to_group();
    println!("  GL(3) as abstract group: {:?}", abstract_group);
    println!("  Connected: {}, Reductive: {}", 
             abstract_group.is_connected, 
             abstract_group.is_reductive);
    
    println!("\n✅ Core type system demonstration complete!");
    println!("   Ready for higher-level mathematical structures.");
    
    Ok(())
}