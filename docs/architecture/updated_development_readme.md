# Development Guide - Emotional Conversation Processor

This project uses modern Python tooling for fast development and consistent code quality.

## 🚀 Quick Setup

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --dev

# Install in development mode
uv pip install -e .

# Download required models
uv run python -m spacy download en_core_web_sm
```

### Alternative Installation with pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
python -m spacy download en_core_web_sm
```

## 🛠️ Development Tools

### Code Quality with Ruff

We use [Ruff](https://docs.astral.sh/ruff/) - a fast Python linter and formatter that replaces Black, Flake8, and isort:

```bash
# Lint code
uv run ruff check src/ tests/ scripts/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/ scripts/

# Format code
uv run ruff format src/ tests/ scripts/

# Check formatting without making changes
uv run ruff format --check src/ tests/ scripts/
```

### Type Checking with MyPy

```bash
# Type check the source code
uv run mypy src/
```

### Using Make Commands

We provide a Makefile for common development tasks:

```bash
# See all available commands
make help

# Complete development setup
make dev-setup

# Code quality checks
make lint          # Run linting
make format        # Format code
make type-check    # Type checking
make check         # All quality checks

# Fix code issues
make fix           # Auto-fix linting and format

# Testing
make test          # Run all tests
make test-unit     # Unit tests only
make test-fast     # Skip slow tests
make test-cov      # With coverage report

# Database operations
make setup-db      # Initialize Qdrant
make health        # Check system health

# Docker
make docker-up     # Start services
make docker-down   # Stop services

# Process conversations
make process CONV_FILE=path/to/file.json

# Clean up
make clean         # Remove generated files
```

## 📋 Pre-commit Hooks

Set up pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run on all files manually
uv run pre-commit run --all-files
```

The pre-commit configuration includes:

- Ruff linting and formatting
- MyPy type checking
- Bandit security scanning
- YAML/JSON validation
- Trailing whitespace removal

## 🧪 Testing

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src/emotional_processor --cov-report=html

# Specific test categories
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not slow"

# Specific test file
uv run pytest tests/unit/test_emotion_classifier.py
```

### Test Markers

- `unit`: Fast unit tests
- `integration`: Tests requiring external services
- `performance`: Performance benchmarks
- `slow`: Time-consuming tests
- `requires_qdrant`: Needs Qdrant database

### Writing Tests

```python
import pytest
from emotional_processor.core.models import ConversationSegment

@pytest.mark.unit
def test_conversation_segment_creation():
    segment = ConversationSegment(
        content="I'm excited about this project!",
        speaker="User"
    )
    assert segment.content == "I'm excited about this project!"
    assert segment.speaker == "User"

@pytest.mark.integration
@pytest.mark.requires_qdrant
def test_vector_storage(vector_store):
    # Integration test requiring Qdrant
    pass
```

## 🔧 Configuration

### Ruff Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py38"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.format]
quote-style = "double"
```

### MyPy Configuration

```toml
[tool.mypy]
python_version = "3.8"
disallow_untyped_defs = true
warn_return_any = true
```

## 📦 Dependency Management

### With uv (Recommended)

```bash
# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Update dependencies
uv sync

# Generate requirements files
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

### With pip

```bash
# Add to pyproject.toml manually, then:
pip install -e .

# Or add to requirements-dev.txt and:
pip install -r requirements-dev.txt
```

## 🐳 Docker Development

### Using Docker Compose

```bash
# Start all services (Qdrant, monitoring, etc.)
docker-compose -f docker/docker-compose.yml up -d

# Stop services
docker-compose -f docker/docker-compose.yml down

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

### Development Container

```bash
# Build development image
docker build -f docker/Dockerfile --target development -t ecp:dev .

# Run interactive development container
docker run -it --rm -v $(pwd):/app ecp:dev bash
```

## 🔍 Debugging and Profiling

### Debugging

```python
# Use loguru for logging
from loguru import logger

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")

# Use pdb for debugging
import pdb; pdb.set_trace()
```

### Profiling

```bash
# Profile script execution
make profile CONV_FILE=path/to/conversation.json

# Memory profiling
uv run python -m memory_profiler scripts/process_conversation_enhanced.py

# Line profiling
uv run kernprof -l -v scripts/process_conversation_enhanced.py
```

## 📊 Performance Monitoring

### System Health

```bash
# Check overall system health
uv run python scripts/health_check.py --verbose

# Monitor resource usage
uv run python scripts/health_check.py --component system
```

### Processing Performance

```bash
# Benchmark processing performance
uv run pytest tests/performance/ -v

# Process with timing
time uv run python scripts/process_conversation_enhanced.py conversation.json
```

## 🎯 Best Practices

### Code Style

- Use type hints for all function parameters and returns
- Follow PEP 8 naming conventions (enforced by Ruff)
- Write docstrings for public functions and classes
- Keep functions focused and testable

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and run checks
make pre-commit

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/new-feature
```

### Testing Strategy

- Write unit tests for individual components
- Write integration tests for component interactions
- Add performance tests for critical paths
- Use fixtures for complex test data
- Mock external dependencies in unit tests

## 🚨 Troubleshooting

### Common Issues

**uv not found:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your shell or source your profile
```

**Import errors:**

```bash
# Ensure package is installed in development mode
uv pip install -e .
```

**Test failures:**

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if needed
make docker-up
```

**Slow performance:**

```bash
# Check system resources
make health

# Use GPU acceleration if available
export CUDA_VISIBLE_DEVICES=0
```

For more help, check the [troubleshooting guide](docs/troubleshooting.md) or open an issue.
