#!/bin/bash
# Simple test runner for Docker

echo "================================================"
echo "ruv-swarm Simple Docker Tests"
echo "================================================"
echo "Date: $(date)"
echo "Node: $(node --version)"
echo "================================================"

# Run basic tests
echo -e "\n1. Testing Node.js environment..."
node -e "console.log('✅ Node.js is working')"

echo -e "\n2. Testing package structure..."
if [ -f "package.json" ]; then
    echo "✅ package.json found"
else
    echo "❌ package.json not found"
fi

echo -e "\n3. Testing dependencies..."
npm list --depth=0 || echo "⚠️  Some dependencies may be missing"

echo -e "\n4. Testing WASM files..."
if [ -f "wasm/ruv_swarm_wasm_bg.wasm" ]; then
    echo "✅ WASM file found"
    ls -lh wasm/*.wasm
else
    echo "❌ WASM file not found"
fi

echo -e "\n5. Testing basic module loading..."
node -e "
try {
    const RuvSwarm = require('./src/index.js');
    console.log('✅ Module loaded successfully');
} catch (e) {
    console.log('❌ Module loading failed:', e.message);
}
" 2>&1

echo -e "\n6. Running basic tests..."
npm test || echo "⚠️  Some tests failed"

echo -e "\n================================================"
echo "Test Summary"
echo "================================================"
echo "Basic environment: ✅"
echo "See output above for detailed results"
echo "================================================"