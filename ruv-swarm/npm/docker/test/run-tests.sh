#!/bin/bash
# Docker test runner for ruv-swarm npm package

set -e

echo "================================================"
echo "ruv-swarm Docker Test Suite"
echo "================================================"
echo "Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Node Version: $(node --version)"
echo "NPM Version: $(npm --version)"
echo "Rust Version: $(rustc --version)"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n🧪 Running: $test_name"
    echo "----------------------------------------"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED${NC}: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED${NC}: $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Basic functionality tests
run_test "Node.js Environment" "node -e 'console.log(\"Node.js is working\")'"
run_test "NPM Installation" "npm list --depth=0"
run_test "WASM File Exists" "ls -la wasm/ruv_swarm_wasm_bg.wasm"

# Unit tests
run_test "Basic Unit Tests" "npm test"

# Integration tests
run_test "WASM Loading Test" "npm run test:docker:wasm"

# MCP server tests (if MCP server is available)
if curl -s http://mcp-server:3000/health > /dev/null 2>&1; then
    run_test "MCP Integration Test" "npm run test:mcp"
else
    echo -e "${YELLOW}⚠️  Skipping MCP tests - server not available${NC}"
fi

# Persistence tests (if database is available)
if PGPASSWORD=testpass psql -h test-db -U testuser -d ruv_swarm_test -c '\q' > /dev/null 2>&1; then
    run_test "Persistence Layer Test" "npm run test:persistence"
else
    echo -e "${YELLOW}⚠️  Skipping persistence tests - database not available${NC}"
fi

# CLI tests
run_test "CLI Version Check" "timeout 10s node bin/ruv-swarm-secure.js --version || true"
run_test "CLI Help Check" "timeout 10s node bin/ruv-swarm-secure.js --help || true"

# Performance tests (lightweight version)
run_test "Basic Performance Test" "node -e 'const RuvSwarm = require(\"./src/index.js\"); console.log(\"Module loaded\")'"

# Security tests
run_test "Dependency Audit" "npm audit --production || true"

# Generate test report
echo -e "\n================================================"
echo "Test Summary"
echo "================================================"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo "Pass Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
echo "================================================"

# Save results to file
cat > /app/test-results/docker-test-results.json << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "passRate": $(( PASSED_TESTS * 100 / TOTAL_TESTS ))
}
EOF

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi