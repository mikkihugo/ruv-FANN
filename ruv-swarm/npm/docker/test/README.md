# Docker Testing for ruv-swarm npm package

This directory contains Docker configurations for comprehensive testing of the ruv-swarm npm package in isolated environments.

## 📁 Structure

- `Dockerfile` - Main test container with Node.js and Rust
- `Dockerfile.mcp` - MCP server container for integration tests
- `docker-compose.yml` - Orchestration for all test services
- `run-tests.sh` - Test runner script
- `Makefile` - Convenient commands for testing

## 🚀 Quick Start

### Run basic tests:
```bash
make test
```

### Run full test suite with all services:
```bash
make test-full
```

## 📋 Available Commands

```bash
make help        # Show available commands
make build       # Build Docker images
make test        # Run basic tests
make test-full   # Run full test suite with all services
make clean       # Clean up containers and volumes
make logs        # Show container logs
make shell       # Open shell in test container
```

## 🧪 Test Coverage

The Docker test suite includes:

1. **Environment Tests**
   - Node.js functionality
   - NPM dependencies
   - WASM file availability

2. **Unit Tests**
   - Basic functionality tests
   - Module loading tests

3. **Integration Tests**
   - WASM loading and execution
   - MCP server communication
   - Database persistence
   - Redis caching

4. **CLI Tests**
   - Version checking
   - Help documentation
   - Command execution

5. **Security Tests**
   - Dependency auditing
   - Vulnerability scanning

## 🔧 Services

### Main Test Container (`ruv-swarm-test`)
- Node.js 22
- Rust toolchain
- wasm-pack
- All test dependencies

### MCP Server (`mcp-server`)
- Rust-based MCP server
- Health check endpoint
- Port 3000

### PostgreSQL Database (`test-db`)
- PostgreSQL 16
- Test database for persistence
- Port 5432

### Redis Cache (`test-cache`)
- Redis 7
- In-memory caching tests
- Port 6379

## 📊 Test Results

Test results are saved to the `test-results` volume and can be accessed:

```bash
make test-results
```

Results include:
- Total tests run
- Pass/fail counts
- Pass rate percentage
- Timestamp

## 🔍 Debugging

### View logs:
```bash
make logs
```

### Open shell in test container:
```bash
make shell
```

### Run specific test:
```bash
docker-compose run --rm ruv-swarm-test npm run test:specific
```

## 🛠️ Customization

### Environment Variables

You can customize test behavior with environment variables:

```yaml
environment:
  - NODE_ENV=test
  - RUST_LOG=debug
  - NO_COLOR=1
  - CI=true
  - CUSTOM_VAR=value
```

### Adding New Tests

1. Add test script to `run-tests.sh`
2. Use the `run_test` function:
   ```bash
   run_test "Test Name" "test command"
   ```

## 🐛 Known Issues

1. **MCP server startup** - May take 10-15 seconds to be ready
2. **Database connections** - Ensure services are healthy before running tests
3. **WASM compilation** - First build may take several minutes

## 📝 Notes

- Tests run in isolated containers for consistency
- All services are on the same Docker network
- Volumes persist test results between runs
- Use `make clean` to remove all test artifacts