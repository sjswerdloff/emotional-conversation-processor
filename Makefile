# Makefile for Emotional Conversation Processor
# Uses uv for fast package management and ruff for linting/formatting

.PHONY: help install install-dev test lint format check clean setup-db health

# Default target
help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install production dependencies with uv
	uv pip install -r requirements.txt

install-dev:  ## Install development dependencies with uv
	uv pip install -r requirements-dev.txt
	uv pip install -e .

# Environment setup
venv:  ## Create virtual environment with uv
	uv venv
	@echo "Activate with: source .venv/bin/activate"

sync:  ## Sync dependencies with uv
	uv pip sync requirements-dev.txt

# Code quality
lint:  ## Run ruff linter
	ruff check emotional_processor/ tests/ scripts/

format:  ## Format code with ruff
	ruff format emotional_processor/ tests/ scripts/

format-check:  ## Check if code is formatted
	ruff format --check emotional_processor/ tests/ scripts/

type-check:  ## Run mypy type checking
	uv run mypy emotional_processor/

check: lint format-check type-check  ## Run all code quality checks

fix:  ## Fix linting issues automatically
	ruff check --fix emotional_processor/ tests/ scripts/
	ruff format emotional_processor/ tests/ scripts/

# Testing
test:  ## Run tests with pytest
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=emotional_processor --cov-report=html --cov-report=term

test-fast:  ## Run tests excluding slow ones
	uv run pytest -m "not slow"

test-unit:  ## Run only unit tests
	uv run pytest tests/unit/

test-integration:  ## Run only integration tests
	uv run pytest tests/integration/

# Database
setup-db:  ## Set up Qdrant database
	uv run python scripts/setup_database.py

health:  ## Check system health
	uv run python scripts/health_check.py

# Docker
docker-up:  ## Start services with docker-compose
	docker-compose -f docker/docker-compose.yml up -d

docker-down:  ## Stop services
	docker-compose -f docker/docker-compose.yml down

# Processing
process:  ## Process a conversation file (requires CONV_FILE variable)
	@if [ -z "$(CONV_FILE)" ]; then \
		echo "Usage: make process CONV_FILE=path/to/conversation.json"; \
		exit 1; \
	fi
	python scripts/process_conversation_enhanced.py $(CONV_FILE)

# Cleaning
clean:  ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data:  ## Clean processed data (be careful!)
	@echo "This will delete processed embeddings and exports. Continue? [y/N]"
	@read ans && [ $${ans:-N} = y ]
	rm -rf data/processed/embeddings/*
	rm -rf data/exports/replayed_conversations/*

# Development workflow
dev-setup: venv install-dev setup-db  ## Complete development setup
	@echo "Development environment ready!"
	@echo "Don't forget to activate: source .venv/bin/activate"

pre-commit: check test-fast  ## Run pre-commit checks

ci: check test  ## Run full CI pipeline

# Documentation
docs:  ## Build documentation
	sphinx-build -b html docs/ docs/_build/html/

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

# Release
build:  ## Build package
	uv build

release-test:  ## Build and test release
	uv build
	uv pip install dist/*.whl
	python -c "import emotional_processor; print('✅ Package imports successfully')"

# Utilities
deps-update:  ## Update dependencies (generates new requirements)
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

deps-outdated:  ## Check for outdated dependencies
	uv pip list --outdated

profile:  ## Profile the main processing script
	@if [ -z "$(CONV_FILE)" ]; then \
		echo "Usage: make profile CONV_FILE=path/to/conversation.json"; \
		exit 1; \
	fi
	py-spy record -o profile.svg -- python scripts/process_conversation_enhanced.py $(CONV_FILE)

# Quick start
quickstart: dev-setup docker-up  ## Complete quickstart setup
	@echo ""
	@echo "🎉 Emotional Conversation Processor is ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source .venv/bin/activate"
	@echo "  2. Process a conversation: make process CONV_FILE=your_file.json"
	@echo "  3. Check health: make health"
	@echo "  4. Run tests: make test"
