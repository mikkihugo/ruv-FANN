# Makefile for Docker testing

.PHONY: help build test test-basic test-full clean logs shell

help:
	@echo "Available commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make test        - Run basic tests"
	@echo "  make test-full   - Run full test suite with all services"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make logs        - Show container logs"
	@echo "  make shell       - Open shell in test container"

build:
	docker-compose build

test: build
	docker-compose run --rm ruv-swarm-test

test-full: build
	docker-compose up -d mcp-server test-db test-cache
	sleep 15  # Wait for services to be ready
	docker-compose run --rm ruv-swarm-test
	docker-compose down

clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

logs:
	docker-compose logs -f

shell:
	docker-compose run --rm --entrypoint /bin/bash ruv-swarm-test

# Individual service commands
mcp-server:
	docker-compose up -d mcp-server
	docker-compose logs -f mcp-server

test-results:
	@echo "Test results:"
	@cat test-results/docker-test-results.json 2>/dev/null || echo "No results found"