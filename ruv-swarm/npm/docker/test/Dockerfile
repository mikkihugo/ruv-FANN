# Multi-stage Dockerfile for ruv-swarm npm testing
# Stage 1: Base image with Node.js and Rust
FROM node:22-bookworm AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    python3 \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Stage 2: Build environment
FROM base AS builder

# Set working directory
WORKDIR /app

# Copy the entire ruv-FANN project
COPY ../../../.. .

# Navigate to ruv-swarm npm directory
WORKDIR /app/ruv-swarm/npm

# Install npm dependencies
RUN npm ci --prefer-offline --no-audit

# Build WASM if needed
RUN if [ -f "scripts/build-wasm.sh" ]; then \
        chmod +x scripts/build-wasm.sh && \
        ./scripts/build-wasm.sh; \
    fi

# Stage 3: Test environment
FROM base AS tester

# Set working directory
WORKDIR /app/ruv-swarm/npm

# Copy built application from builder
COPY --from=builder /app/ruv-swarm /app/ruv-swarm

# Copy test-specific files
COPY docker/test/run-tests.sh /app/ruv-swarm/npm/
RUN chmod +x /app/ruv-swarm/npm/run-tests.sh

# Set environment variables for testing
ENV NODE_ENV=test
ENV RUST_LOG=debug
ENV NO_COLOR=1

# Create test results directory
RUN mkdir -p /app/test-results

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "console.log('OK')" || exit 1

# Default command runs all tests
CMD ["/app/ruv-swarm/npm/run-tests.sh"]