# Module Dependencies - Geometric Langlands Implementation

## 🔗 Dependency Graph

```
core/
├── automorphic/ (depends on core)
├── galois/ (depends on core)
├── category/ (depends on core)
├── sheaf/ (depends on core, category)
├── representation/ (depends on core, automorphic)
├── harmonic/ (depends on core, representation)
├── spectral/ (depends on core, harmonic)
├── trace/ (depends on automorphic, spectral)
└── langlands/ (depends on ALL modules)

Performance modules:
├── cuda/ (depends on core, spectral)
├── wasm/ (depends on core)
└── utils/ (standalone)
```

## 📋 Module Interfaces

### Core Module Exports
```rust
// Essential types that other modules need
pub trait AlgebraicVariety { ... }
pub trait Scheme { ... }
pub trait ModuliSpace { ... }
pub trait Stack { ... }
pub struct Field { ... }
pub struct Group { ... }
pub struct Ring { ... }
```

### Automorphic Module Exports
```rust
// Types needed by trace and langlands modules
pub struct AutomorphicForm { ... }
pub struct HeckeOperator { ... }
pub trait AutomorphicRepresentation { ... }
```

### Galois Module Exports
```rust
// Types needed by langlands module
pub struct GaloisRepresentation { ... }
pub struct LocalSystem { ... }
pub trait LAdic { ... }
```

### Category Module Exports
```rust
// Types needed by sheaf and langlands modules
pub trait DerivedCategory { ... }
pub struct DModule { ... }
pub trait FusionCategory { ... }
```

## 🚨 Critical Dependencies

1. **Core MUST be implemented first** - All other modules depend on it
2. **Category before Sheaf** - Sheaf theory uses categorical structures
3. **Automorphic before Trace** - Trace formulas operate on automorphic forms
4. **All modules before Langlands** - Main correspondence uses everything

## 🔄 Integration Points

### Phase 1: Foundation
- Core type system
- Basic structures in each module
- Unit tests for each module

### Phase 2: Integration
- Inter-module communication
- Shared memory structures
- Integration tests

### Phase 3: Optimization
- CUDA kernels for heavy computation
- WASM bindings for web deployment
- Performance benchmarks

## ⚠️ Circular Dependency Prevention

- No module should import from langlands/ (it's the top-level integrator)
- Utils should remain independent
- Performance modules (cuda/wasm) should only depend on core types
- Use trait objects for loose coupling between modules

## 📝 Interface Stability

| Module | Interface Status | Can Change? |
|--------|-----------------|-------------|
| core | 🔴 Unstable | Yes - Still designing |
| automorphic | 🔴 Unstable | Yes - Pending core |
| galois | 🔴 Unstable | Yes - Pending core |
| category | 🔴 Unstable | Yes - Pending core |
| sheaf | 🔴 Unstable | Yes - Pending category |
| All others | 🔴 Unstable | Yes - Early stage |

## 🎯 Coordination Rules

1. **Before changing an interface**: Post in MESSAGES/ folder
2. **When depending on unstable interface**: Note it in your code
3. **After stabilizing an interface**: Update this document
4. **For breaking changes**: Coordinate with all dependent modules