# Core API Documentation

## Overview

The `core` module provides fundamental mathematical structures that form the foundation of the geometric Langlands implementation. This module implements algebraic and geometric objects with strong type safety and computational efficiency.

## 📚 Module Structure

```rust
pub mod core {
    // Algebraic structures
    pub mod algebra;
    pub mod field;
    pub mod ring;
    pub mod group;
    
    // Geometric objects
    pub mod variety;
    pub mod scheme;
    pub mod curve;
    pub mod moduli;
    
    // Representations
    pub mod representation;
    pub mod lie_algebra;
    pub mod matrix_rep;
}
```

## 🔢 Algebraic Structures

### Field

The `Field` trait represents mathematical fields with operations.

```rust
pub trait Field: Ring {
    /// Multiplicative inverse of a non-zero element
    fn inverse(&self, element: &Self::Element) -> Result<Self::Element>;
    
    /// Check if field is finite
    fn is_finite(&self) -> bool;
    
    /// Characteristic of the field
    fn characteristic(&self) -> BigInt;
}

// Implementations
impl Field for FiniteField;
impl Field for RationalField;
impl Field for ComplexField;
```

#### Examples

```rust
use geometric_langlands::core::{Field, FiniteField};

// Create finite field F_101
let f = FiniteField::new(101)?;
assert_eq!(f.characteristic(), BigInt::from(101));

// Field operations
let a = f.element(5);
let b = f.element(7);
let sum = f.add(&a, &b)?;  // 5 + 7 = 12 in F_101
let inv = f.inverse(&a)?;  // 5^(-1) mod 101

// Verify field axioms
assert_eq!(f.multiply(&a, &inv)?, f.one());
```

### Ring

The `Ring` trait provides ring operations with computational efficiency.

```rust
pub trait Ring: Clone + Debug {
    type Element: Clone + Debug + PartialEq;
    
    // Ring operations
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element>;
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element>;
    fn negate(&self, a: &Self::Element) -> Self::Element;
    
    // Ring elements
    fn zero(&self) -> Self::Element;
    fn one(&self) -> Self::Element;
    
    // Properties
    fn is_commutative(&self) -> bool;
    fn is_integral_domain(&self) -> bool;
}
```

#### Examples

```rust
use geometric_langlands::core::{Ring, PolynomialRing, FiniteField};

// Create polynomial ring F_5[x, y]
let base_field = FiniteField::new(5)?;
let poly_ring = PolynomialRing::new(base_field, 2)?;  // 2 variables

// Create polynomials
let x = poly_ring.variable(0);  // x
let y = poly_ring.variable(1);  // y
let f = poly_ring.add(&poly_ring.multiply(&x, &x)?, &y)?;  // x² + y

// Ring properties
assert!(poly_ring.is_commutative());
assert!(poly_ring.is_integral_domain());
```

### Group

The `Group` trait represents mathematical groups with efficient operations.

```rust
pub trait Group: Clone + Debug {
    type Element: Clone + Debug + PartialEq;
    
    // Group operations
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element>;
    fn inverse(&self, a: &Self::Element) -> Result<Self::Element>;
    fn identity(&self) -> Self::Element;
    
    // Group properties
    fn order(&self) -> Option<BigInt>;
    fn is_abelian(&self) -> bool;
    fn is_finite(&self) -> bool;
}
```

#### Reductive Groups

The most important implementation is `ReductiveGroup`:

```rust
pub struct ReductiveGroup {
    group_type: GroupType,
    rank: usize,
    base_field: Box<dyn Field>,
}

impl ReductiveGroup {
    /// Create GL(n) group
    pub fn gl_n(n: usize) -> Self;
    
    /// Create SL(n) group  
    pub fn sl_n(n: usize) -> Self;
    
    /// Create SO(n) group
    pub fn so_n(n: usize) -> Self;
    
    /// Create Sp(2n) group
    pub fn sp_2n(n: usize) -> Self;
    
    /// Get the Lie algebra
    pub fn lie_algebra(&self) -> LieAlgebra;
    
    /// Get the dual group
    pub fn dual_group(&self) -> Self;
    
    /// Standard representation
    pub fn standard_representation(&self) -> MatrixRepresentation;
}
```

#### Examples

```rust
use geometric_langlands::core::ReductiveGroup;

// Create various reductive groups
let gl2 = ReductiveGroup::gl_n(2);
let sl3 = ReductiveGroup::sl_n(3);
let so4 = ReductiveGroup::so_n(4);

// Group properties
assert_eq!(gl2.rank(), 2);
assert_eq!(sl3.rank(), 2);  // rank = n-1 for SL(n)
assert!(!gl2.is_finite());

// Lie algebra
let lie_alg = gl2.lie_algebra();
assert_eq!(lie_alg.dimension(), 4);  // gl(2) has dimension 4

// Dual group (important for Langlands)
let gl2_dual = gl2.dual_group();
assert_eq!(gl2_dual.group_type(), gl2.group_type());  // GL is self-dual
```

## 📐 Geometric Objects

### Algebraic Variety

Represents algebraic varieties as zero sets of polynomials.

```rust
pub struct AlgebraicVariety {
    defining_polynomials: Vec<Polynomial>,
    ambient_space: ProjectiveSpace,
    dimension: usize,
}

impl AlgebraicVariety {
    /// Create variety from defining polynomials
    pub fn new(polynomials: Vec<Polynomial>) -> Result<Self>;
    
    /// Create elliptic curve from Weierstrass equation
    pub fn elliptic_curve(coefficients: [i64; 5]) -> Result<Self>;
    
    /// Create projective space
    pub fn projective_space(dimension: usize, field: Box<dyn Field>) -> Self;
    
    /// Compute dimension
    pub fn dimension(&self) -> usize;
    
    /// Check if point is on variety
    pub fn contains_point(&self, point: &Point) -> bool;
    
    /// Tangent space at a point
    pub fn tangent_space(&self, point: &Point) -> Result<VectorSpace>;
}
```

#### Examples

```rust
use geometric_langlands::core::{AlgebraicVariety, FiniteField};

// Create elliptic curve y² = x³ + ax + b
let curve = AlgebraicVariety::elliptic_curve([0, 0, 0, 1, 1])?;
assert_eq!(curve.dimension(), 1);

// Create projective plane
let field = FiniteField::new(7)?;
let proj_plane = AlgebraicVariety::projective_space(2, Box::new(field));
assert_eq!(proj_plane.dimension(), 2);

// Check points
let point = Point::new(vec![1, 2, 3]);  // [1:2:3] in projective space
assert!(proj_plane.contains_point(&point));
```

### Scheme

More general than varieties, schemes allow nilpotent elements.

```rust
pub struct Scheme {
    structure_sheaf: StructureSheaf,
    underlying_space: TopologicalSpace,
}

impl Scheme {
    /// Create affine scheme from ring
    pub fn affine_scheme(ring: Box<dyn Ring>) -> Self;
    
    /// Create projective scheme
    pub fn projective_scheme(graded_ring: GradedRing) -> Self;
    
    /// Fiber product of schemes
    pub fn fiber_product(x: &Self, y: &Self, base: &Self) -> Result<Self>;
    
    /// Morphism to another scheme
    pub fn morphism_to(&self, target: &Self) -> Option<SchemeMorphism>;
}
```

### Curve

Specialized one-dimensional schemes with additional structure.

```rust
pub struct Curve {
    genus: usize,
    base_field: Box<dyn Field>,
    function_field: FunctionField,
}

impl Curve {
    /// Create curve of given genus
    pub fn genus_g(g: usize, field: Box<dyn Field>) -> Result<Self>;
    
    /// Create elliptic curve (genus 1)
    pub fn elliptic_curve(j_invariant: FieldElement) -> Result<Self>;
    
    /// Rational curve (genus 0)
    pub fn rational_curve(field: Box<dyn Field>) -> Self;
    
    /// Function field of the curve
    pub fn function_field(&self) -> &FunctionField;
    
    /// Jacobian variety
    pub fn jacobian(&self) -> AlgebraicVariety;
    
    /// Moduli of bundles on this curve
    pub fn bundle_moduli<G: ReductiveGroup>(&self, group: &G) -> ModuliStack;
}
```

### Moduli Stack

Represents moduli problems as algebraic stacks.

```rust
pub struct ModuliStack {
    moduli_problem: ModuliProblem,
    atlas: Vec<Scheme>,
    transition_maps: TransitionData,
}

impl ModuliStack {
    /// Bundle moduli stack
    pub fn bundle_moduli<G: ReductiveGroup>(
        curve: &Curve, 
        group: &G
    ) -> Self;
    
    /// Local system moduli stack
    pub fn local_system_moduli<G: ReductiveGroup>(
        curve: &Curve,
        group: &G
    ) -> Self;
    
    /// Dimension of the stack
    pub fn dimension(&self) -> isize;  // Can be negative (virtual dimension)
    
    /// Tangent complex
    pub fn tangent_complex(&self) -> Complex;
}
```

## 🔄 Representations

### Matrix Representation

Concrete realizations of abstract groups as matrices.

```rust
pub struct MatrixRepresentation<G: Group> {
    group: G,
    dimension: usize,
    representation_map: RepresentationMap,
}

impl<G: ReductiveGroup> MatrixRepresentation<G> {
    /// Standard representation
    pub fn standard(group: &G) -> Self;
    
    /// Adjoint representation
    pub fn adjoint(group: &G) -> Self;
    
    /// Irreducible representation with highest weight
    pub fn irreducible(group: &G, highest_weight: Weight) -> Result<Self>;
    
    /// Represent group element as matrix
    pub fn represent(&self, element: &G::Element) -> Matrix;
    
    /// Character of the representation
    pub fn character(&self, element: &G::Element) -> FieldElement;
    
    /// Decompose into irreducibles
    pub fn decompose(&self) -> Vec<IrreducibleRep>;
}
```

### Lie Algebra

Infinitesimal version of Lie groups.

```rust
pub struct LieAlgebra {
    group: ReductiveGroup,
    basis: Vec<LieAlgebraElement>,
    structure_constants: StructureConstants,
}

impl LieAlgebra {
    /// Dimension of the Lie algebra
    pub fn dimension(&self) -> usize;
    
    /// Root system
    pub fn root_system(&self) -> RootSystem;
    
    /// Cartan subalgebra
    pub fn cartan_subalgebra(&self) -> LieAlgebra;
    
    /// Killing form
    pub fn killing_form(&self, x: &LieAlgebraElement, y: &LieAlgebraElement) -> FieldElement;
    
    /// Adjoint representation
    pub fn adjoint_representation(&self) -> MatrixRepresentation<Self>;
}
```

## ⚡ Performance Features

### Parallel Operations

Many operations are automatically parallelized:

```rust
// Parallel matrix operations
let matrices: Vec<Matrix> = generate_large_matrix_set();
let eigenvalues: Vec<_> = matrices
    .par_iter()  // Parallel iterator
    .map(|m| m.eigenvalues())
    .collect();

// Parallel group operations
let elements: Vec<GroupElement> = generate_group_elements();
let products: Vec<_> = elements
    .par_chunks(2)
    .map(|chunk| group.multiply(&chunk[0], &chunk[1]))
    .collect();
```

### CUDA Acceleration

GPU acceleration for large computations:

```rust
#[cfg(feature = "cuda")]
use geometric_langlands::core::cuda::*;

// GPU matrix operations
let ctx = CudaContext::new()?;
let gpu_matrix = CudaMatrix::from_host(&cpu_matrix, &ctx)?;
let eigenvalues = gpu_matrix.eigenvalues_cuda()?;

// GPU group computations
let gpu_group = CudaReductiveGroup::new(&group, &ctx)?;
let gpu_result = gpu_group.parallel_multiply(&elements)?;
```

### Memory Optimization

Efficient memory usage for large mathematical objects:

```rust
// Lazy evaluation for large computations
let lazy_matrix = Matrix::lazy_construction(|| expensive_computation());

// Streaming operations for large datasets
let stream = PolynomialStream::new(degree_bound);
let results: Vec<_> = stream
    .take(1000)
    .map(|poly| poly.evaluate(&point))
    .collect();

// Memory-mapped storage for persistent objects
let persistent_group = PersisentGroup::memory_mapped("large_group.dat")?;
```

## 🧪 Testing and Validation

### Property-Based Testing

All core operations are validated with property-based tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_group_axioms(
            a in group_element_strategy(),
            b in group_element_strategy(),
            c in group_element_strategy()
        ) {
            let group = ReductiveGroup::gl_n(2);
            
            // Associativity: (a * b) * c = a * (b * c)
            let left = group.multiply(&group.multiply(&a, &b)?, &c)?;
            let right = group.multiply(&a, &group.multiply(&b, &c)?)?;
            assert_eq!(left, right);
            
            // Identity: a * e = e * a = a
            let identity = group.identity();
            assert_eq!(group.multiply(&a, &identity)?, a);
            assert_eq!(group.multiply(&identity, &a)?, a);
            
            // Inverse: a * a^(-1) = a^(-1) * a = e
            let inv_a = group.inverse(&a)?;
            assert_eq!(group.multiply(&a, &inv_a)?, identity);
            assert_eq!(group.multiply(&inv_a, &a)?, identity);
        }
    }
}
```

## 📖 Usage Examples

### Complete Example: Computing with GL(2)

```rust
use geometric_langlands::prelude::*;

fn main() -> Result<()> {
    // Create GL(2) over finite field
    let field = FiniteField::new(101)?;
    let group = ReductiveGroup::gl_n_over_field(2, Box::new(field));
    
    // Get representations
    let std_rep = group.standard_representation();
    let adj_rep = group.adjoint_representation();
    
    println!("Standard representation dimension: {}", std_rep.dimension());
    println!("Adjoint representation dimension: {}", adj_rep.dimension());
    
    // Create group elements
    let matrix1 = Matrix::new([[1, 2], [0, 1]])?;
    let matrix2 = Matrix::new([[1, 0], [3, 1]])?;
    
    let elem1 = group.element_from_matrix(matrix1)?;
    let elem2 = group.element_from_matrix(matrix2)?;
    
    // Group operations
    let product = group.multiply(&elem1, &elem2)?;
    let inverse = group.inverse(&elem1)?;
    
    // Representation theory
    let char1 = std_rep.character(&elem1);
    let char2 = adj_rep.character(&elem1);
    
    println!("Standard character: {}", char1);
    println!("Adjoint character: {}", char2);
    
    // Lie algebra
    let lie_alg = group.lie_algebra();
    let root_system = lie_alg.root_system();
    
    println!("Root system type: {:?}", root_system.cartan_type());
    println!("Number of positive roots: {}", root_system.positive_roots().len());
    
    Ok(())
}
```

## 🚀 Advanced Usage

### Custom Group Implementations

You can extend the framework with custom groups:

```rust
use geometric_langlands::core::{Group, ReductiveGroup};

#[derive(Clone, Debug)]
pub struct CustomGroup {
    // Custom implementation
}

impl Group for CustomGroup {
    type Element = CustomElement;
    
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element> {
        // Custom multiplication
    }
    
    // Implement other required methods...
}

// Register with the framework
let custom_group = CustomGroup::new();
let moduli = ModuliStack::bundle_moduli(&curve, &custom_group);
```

### Integration with External Libraries

The framework integrates with other mathematical libraries:

```rust
// Integration with nalgebra
use nalgebra::DMatrix;

let core_matrix: Matrix = /* ... */;
let nalgebra_matrix: DMatrix<f64> = core_matrix.to_nalgebra();

// Integration with ndarray
use ndarray::Array2;

let ndarray_matrix: Array2<f64> = core_matrix.to_ndarray();

// Custom conversions
impl From<Matrix> for YourMatrixType {
    fn from(matrix: Matrix) -> Self {
        // Your conversion logic
    }
}
```

---

*This completes the Core API documentation. The core module provides the mathematical foundation that all other modules build upon, ensuring type safety, computational efficiency, and mathematical rigor throughout the framework.*