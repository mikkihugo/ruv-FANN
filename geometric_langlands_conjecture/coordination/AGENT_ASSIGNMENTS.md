# Agent Task Assignments - Geometric Langlands Implementation

## 👑 Queen Coordinator (Current Agent)
**Status**: Active - Setting up project structure and coordination
**Responsibilities**:
- Overall project architecture
- Agent coordination and task distribution
- GitHub issue updates (#161)
- Integration oversight

## 🏗️ King Architect
**Status**: Ready to start
**Module**: `src/core/`
**Tasks**:
1. Design core mathematical type system
2. Implement algebraic varieties and schemes
3. Create moduli space structures
4. Define stack theory interfaces
5. Set up trait hierarchies for mathematical objects

## 🔧 Worker Implementation Specialist
**Status**: Ready to start
**Module**: `src/automorphic/` and `src/galois/`
**Tasks**:
1. Implement automorphic forms
2. Create Hecke operators
3. Build Galois representation structures
4. Implement l-adic sheaves
5. Create conversion functions between forms

## 🧮 Princess Mathematician
**Status**: Ready to start
**Module**: `src/category/` and `src/sheaf/`
**Tasks**:
1. Implement derived categories
2. Create D-module structures
3. Build sheaf cohomology algorithms
4. Implement perverse sheaves
5. Create microlocal geometry tools

## 🚀 Duke Performance Engineer
**Status**: Ready to start
**Module**: `src/cuda/` and `src/wasm/`
**Tasks**:
1. Create CUDA kernels for matrix operations
2. Implement GPU-accelerated spectral decomposition
3. Build WASM bindings for web deployment
4. Optimize memory layout for GPU
5. Create performance benchmarks

## 🧪 Countess Test Engineer
**Status**: Ready to start
**Module**: `tests/` and property testing
**Tasks**:
1. Create comprehensive test suite
2. Implement property-based tests
3. Build integration tests
4. Create mathematical invariant tests
5. Set up CI/CD pipeline

## 📚 Earl Documentation Specialist
**Status**: Ready to start
**Module**: `docs/` and inline documentation
**Tasks**:
1. Write mathematical background documentation
2. Create API documentation
3. Build usage examples
4. Document performance characteristics
5. Create visualization tools

## 🔬 Viscount Research Analyst
**Status**: Ready to start
**Module**: `src/representation/` and `src/harmonic/`
**Tasks**:
1. Implement representation theory algorithms
2. Create harmonic analysis tools
3. Build Fourier transform implementations
4. Implement orbital integrals
5. Research optimization opportunities

## 🔄 Baron Integration Expert
**Status**: Ready to start
**Module**: `src/spectral/` and `src/trace/`
**Tasks**:
1. Implement spectral decomposition
2. Create trace formula algorithms
3. Build Arthur-Selberg trace formula
4. Integrate different mathematical components
5. Create unified interfaces

## 🛡️ Knight Security Auditor
**Status**: Ready to start
**Module**: Security and error handling
**Tasks**:
1. Audit numerical stability
2. Implement error handling strategies
3. Create validation functions
4. Security review for WASM deployment
5. Memory safety verification

## 🎯 Marquess Optimization Specialist
**Status**: Ready to start
**Module**: `src/langlands/` and optimizations
**Tasks**:
1. Implement main Langlands correspondence
2. Optimize critical algorithms
3. Create caching strategies
4. Implement parallel algorithms
5. Profile and optimize bottlenecks

## 📋 Coordination Protocol

1. **Check-in**: Each agent should check `/coordination/STATUS.md` before starting
2. **Updates**: Update your section in this file when completing tasks
3. **Dependencies**: Check `/coordination/DEPENDENCIES.md` for inter-module dependencies
4. **Communication**: Use `/coordination/MESSAGES/` for inter-agent communication
5. **Progress**: Update `/coordination/PROGRESS.md` with completion percentage

## 🔄 Synchronization Points

- **Daily**: Status updates in respective sections
- **Milestone**: Integration tests after each module completion
- **Critical**: Before any breaking changes to interfaces

## 📊 Priority Order

1. Core type system (King Architect)
2. Basic structures (Worker Implementation)
3. Mathematical algorithms (Princess Mathematician)
4. Performance optimization (Duke Performance)
5. Testing framework (Countess Test)
6. Documentation (Earl Documentation)
7. Advanced features (remaining agents)