#!/bin/bash

# Geometric Langlands Conjecture Framework Release Script
# Usage: ./scripts/release.sh [patch|minor|major]

set -euo pipefail

# Configuration
PACKAGE_NAME="geometric-langlands"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_DIR="$CURRENT_DIR/wasm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help function
show_help() {
    cat << EOF
Geometric Langlands Release Script

Usage: $0 [OPTIONS] [VERSION_TYPE]

VERSION_TYPE:
    patch    Increment patch version (0.1.0 -> 0.1.1)
    minor    Increment minor version (0.1.0 -> 0.2.0)
    major    Increment major version (0.1.0 -> 1.0.0)

OPTIONS:
    --dry-run         Show what would be done without executing
    --skip-tests      Skip running tests (not recommended)
    --skip-wasm       Skip WASM package preparation
    --skip-publish    Prepare release but don't publish
    --help, -h        Show this help message

Examples:
    $0 patch                 # Release patch version
    $0 minor --dry-run       # Show what minor release would do
    $0 major --skip-tests    # Release major version without tests

EOF
}

# Parse command line arguments
VERSION_TYPE=""
DRY_RUN=false
SKIP_TESTS=false
SKIP_WASM=false
SKIP_PUBLISH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        patch|minor|major)
            VERSION_TYPE="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-wasm)
            SKIP_WASM=true
            shift
            ;;
        --skip-publish)
            SKIP_PUBLISH=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$VERSION_TYPE" ]]; then
    log_error "Version type is required"
    show_help
    exit 1
fi

# Utility functions
run_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: $cmd"
        return 0
    fi
    
    log_info "$description"
    if ! eval "$cmd"; then
        log_error "Failed to $description"
        exit 1
    fi
}

check_tool() {
    local tool="$1"
    if ! command -v "$tool" &> /dev/null; then
        log_error "$tool is required but not installed"
        exit 1
    fi
}

get_current_version() {
    grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'
}

calculate_new_version() {
    local current="$1"
    local type="$2"
    
    IFS='.' read -ra VERSION_PARTS <<< "$current"
    local major="${VERSION_PARTS[0]}"
    local minor="${VERSION_PARTS[1]}"
    local patch="${VERSION_PARTS[2]}"
    
    case "$type" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "$major.$((minor + 1)).0"
            ;;
        patch)
            echo "$major.$minor.$((patch + 1))"
            ;;
    esac
}

# Preflight checks
log_info "Starting release process for $PACKAGE_NAME"

# Check required tools
check_tool "cargo"
check_tool "git"
check_tool "jq"

if [[ "$SKIP_WASM" == "false" ]]; then
    check_tool "wasm-pack"
    check_tool "npm"
fi

# Check git status
if [[ -n "$(git status --porcelain)" ]]; then
    log_error "Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Check we're on main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "main" && "$current_branch" != "geometric-langlands-conjecture" ]]; then
    log_warning "Not on main branch (currently on $current_branch)"
    if [[ "$DRY_RUN" == "false" ]]; then
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Get version information
CURRENT_VERSION=$(get_current_version)
NEW_VERSION=$(calculate_new_version "$CURRENT_VERSION" "$VERSION_TYPE")

log_info "Current version: $CURRENT_VERSION"
log_info "New version: $NEW_VERSION"

# Update Cargo.toml version
log_info "Updating version in Cargo.toml"
if [[ "$DRY_RUN" == "false" ]]; then
    sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
fi

# Update WASM Cargo.toml if it exists
if [[ -f "$WASM_DIR/Cargo.toml" && "$SKIP_WASM" == "false" ]]; then
    log_info "Updating version in WASM Cargo.toml"
    if [[ "$DRY_RUN" == "false" ]]; then
        sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$WASM_DIR/Cargo.toml"
    fi
fi

# Run tests
if [[ "$SKIP_TESTS" == "false" ]]; then
    run_command "cargo test --all-features" "Running all tests"
    run_command "cargo test --no-default-features" "Running tests without default features"
    run_command "cargo clippy --all-targets --all-features -- -D warnings" "Running clippy"
    run_command "cargo fmt -- --check" "Checking code formatting"
fi

# Run benchmarks
log_info "Running benchmarks"
if [[ "$DRY_RUN" == "false" ]]; then
    cargo bench --bench langlands_benchmarks 2>/dev/null || log_warning "Benchmarks failed or not available"
fi

# Validate package
run_command "cargo package --allow-dirty" "Validating package"

# Build WASM package
if [[ "$SKIP_WASM" == "false" && -d "$WASM_DIR" ]]; then
    log_info "Building WASM package"
    if [[ "$DRY_RUN" == "false" ]]; then
        cd "$WASM_DIR"
        wasm-pack build --target web --out-dir pkg
        cd "$CURRENT_DIR"
    fi
fi

# Check documentation
run_command "cargo doc --all-features --no-deps" "Building documentation"

# Update CHANGELOG.md
log_info "Updating CHANGELOG.md"
if [[ "$DRY_RUN" == "false" ]]; then
    # Create a backup
    cp CHANGELOG.md CHANGELOG.md.backup
    
    # Add new version entry
    sed -i "/## \[Unreleased\]/a\\
\\
## [$NEW_VERSION] - $(date +%Y-%m-%d)\\
\\
### Added\\
- Release $NEW_VERSION\\
\\
### Changed\\
- Version bump to $NEW_VERSION\\
" CHANGELOG.md
fi

# Commit changes
if [[ "$DRY_RUN" == "false" ]]; then
    git add Cargo.toml CHANGELOG.md
    if [[ -f "$WASM_DIR/Cargo.toml" ]]; then
        git add "$WASM_DIR/Cargo.toml"
    fi
    git commit -m "chore: Release v$NEW_VERSION

- Bump version to $NEW_VERSION
- Update CHANGELOG.md
- Prepare for publication"
fi

# Create git tag
run_command "git tag -a v$NEW_VERSION -m 'Release v$NEW_VERSION'" "Creating git tag"

# Publish to crates.io
if [[ "$SKIP_PUBLISH" == "false" ]]; then
    log_info "Publishing to crates.io"
    if [[ "$DRY_RUN" == "false" ]]; then
        cargo publish
    fi
fi

# Push changes and tags
if [[ "$DRY_RUN" == "false" ]]; then
    git push origin HEAD
    git push origin "v$NEW_VERSION"
fi

log_success "Release v$NEW_VERSION completed successfully!"
log_info "Next steps:"
log_info "1. Verify the release on crates.io: https://crates.io/crates/$PACKAGE_NAME"
log_info "2. Create a GitHub release with release notes"
log_info "3. Update documentation if needed"
if [[ "$SKIP_WASM" == "false" ]]; then
    log_info "4. Publish WASM package to npm if desired"
fi

exit 0