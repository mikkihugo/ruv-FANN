#!/bin/bash
# Build script for Geometric Langlands WASM module

set -e

echo "Building Geometric Langlands WASM module..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}Error: wasm-pack is not installed!${NC}"
    echo "Install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build profiles
BUILD_PROFILE=${1:-release}

case $BUILD_PROFILE in
    "dev")
        echo -e "${BLUE}Building in development mode...${NC}"
        wasm-pack build --dev --target web --out-dir pkg
        ;;
    "release")
        echo -e "${BLUE}Building in release mode (optimized for speed)...${NC}"
        wasm-pack build --release --target web --out-dir pkg -- --profile wasm-speed
        ;;
    "size")
        echo -e "${BLUE}Building in size-optimized mode...${NC}"
        wasm-pack build --release --target web --out-dir pkg -- --profile wasm-size
        ;;
    "profiling")
        echo -e "${BLUE}Building with profiling enabled...${NC}"
        wasm-pack build --profiling --target web --out-dir pkg -- --features profiling
        ;;
    *)
        echo -e "${RED}Unknown build profile: $BUILD_PROFILE${NC}"
        echo "Usage: ./build.sh [dev|release|size|profiling]"
        exit 1
        ;;
esac

# Post-processing: optimize WASM with wasm-opt if available
if command -v wasm-opt &> /dev/null; then
    echo -e "${YELLOW}Optimizing WASM with wasm-opt...${NC}"
    wasm-opt -O4 --enable-simd \
        pkg/geometric_langlands_conjecture_bg.wasm \
        -o pkg/geometric_langlands_conjecture_bg_optimized.wasm
    
    # Check size reduction
    ORIGINAL_SIZE=$(stat -f%z pkg/geometric_langlands_conjecture_bg.wasm 2>/dev/null || stat -c%s pkg/geometric_langlands_conjecture_bg.wasm)
    OPTIMIZED_SIZE=$(stat -f%z pkg/geometric_langlands_conjecture_bg_optimized.wasm 2>/dev/null || stat -c%s pkg/geometric_langlands_conjecture_bg_optimized.wasm)
    
    echo -e "${GREEN}Original size: $((ORIGINAL_SIZE / 1024))KB${NC}"
    echo -e "${GREEN}Optimized size: $((OPTIMIZED_SIZE / 1024))KB${NC}"
    echo -e "${GREEN}Size reduction: $(( (ORIGINAL_SIZE - OPTIMIZED_SIZE) * 100 / ORIGINAL_SIZE ))%${NC}"
    
    # Replace original with optimized
    mv pkg/geometric_langlands_conjecture_bg_optimized.wasm pkg/geometric_langlands_conjecture_bg.wasm
fi

# Generate TypeScript definitions if tsc is available
if command -v tsc &> /dev/null; then
    echo -e "${YELLOW}Generating TypeScript definitions...${NC}"
    cd pkg
    tsc geometric_langlands_conjecture.d.ts --declaration --emitDeclarationOnly --allowJs
    cd ..
fi

echo -e "${GREEN}✓ Build complete!${NC}"
echo -e "${GREEN}Output directory: ./pkg${NC}"

# Show generated files
echo -e "\n${BLUE}Generated files:${NC}"
ls -la pkg/

# Create example usage snippet
cat > pkg/example.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Geometric Langlands WASM Example</title>
</head>
<body>
    <h1>Geometric Langlands Computations</h1>
    <div id="output"></div>
    
    <script type="module">
        import init, { 
            GeometricLanglandsComputer,
            compute_moduli_dimension,
            compute_sheaf_cohomology,
            compute_hitchin_map
        } from './geometric_langlands_conjecture.js';
        
        async function run() {
            // Initialize the WASM module
            await init();
            
            // Create computer instance
            const computer = new GeometricLanglandsComputer();
            
            // Initialize GPU if available
            try {
                await computer.init_gpu();
                console.log('GPU initialized successfully');
            } catch (e) {
                console.log('GPU not available, using CPU fallback');
            }
            
            // Example computations
            const moduli_dim = compute_moduli_dimension(2, 3, 1);
            console.log('Moduli space dimension:', moduli_dim);
            
            const cohomology = compute_sheaf_cohomology(3, 2, false);
            console.log('Sheaf cohomology:', cohomology);
            
            // Get performance metrics
            const metrics = computer.get_metrics();
            console.log('Performance metrics:', JSON.parse(metrics));
            
            document.getElementById('output').innerHTML = `
                <p>Moduli space dimension: ${moduli_dim}</p>
                <p>Performance metrics: ${metrics}</p>
            `;
        }
        
        run();
    </script>
</body>
</html>
EOF

echo -e "\n${GREEN}Example usage file created at: pkg/example.html${NC}"
echo -e "${BLUE}To test, run: python3 -m http.server 8000 --directory pkg${NC}"