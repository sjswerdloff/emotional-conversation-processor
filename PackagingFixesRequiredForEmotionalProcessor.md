# Packaging Fixes Required for emotional-processor

## 🚨 Critical Issues Identified

The emotional-processor package has multiple packaging configuration problems that prevent proper editable installations and cause import failures. These issues must be resolved for production medical software reliability.

## 📋 Specific Problems Found

### 1. **Conflicting Build Systems**
**Problem**: The project has both `pyproject.toml` and `setup.py` with different backends:
- `pyproject.toml` uses `hatchling` backend
- `setup.py` uses `setuptools` backend

**Evidence**:
```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python
# setup.py
from setuptools import find_packages, setup
```

**Impact**: Installation conflicts, empty .pth files, unpredictable behavior

### 2. **Wrong Package Directory Structure**
**Problem**: `setup.py` expects a `src/` layout but package is at root level:

**Current Structure**:
```
emotional-conversation-processor/
├── emotional_processor/    # ← Actual location
│   ├── __init__.py
│   ├── processors/
│   ├── core/
│   └── ...
└── setup.py
```

**setup.py Configuration**:
```python
packages=find_packages(where="src"),
package_dir={"": "src"},
```

**Expected by setup.py**:
```
emotional-conversation-processor/
├── src/
│   └── emotional_processor/  # ← Expected location
└── setup.py
```

**Impact**: Package discovery failure, installation shows success but imports fail

### 3. **Package Name Inconsistency**
**Problem**: Different names in configuration files:
- `setup.py`: `name="emotional-conversation-processor"`
- `pyproject.toml`: `name = "emotional-processor"`

**Impact**: Installation confusion, dependency resolution issues

### 4. **Missing Package Discovery in pyproject.toml**
**Problem**: No explicit package discovery configuration in `pyproject.toml`

**Current**:
```toml
[project]
name = "emotional-processor"
# ... no package discovery configuration
```

**Impact**: Hatchling can't find the packages to install

## 🛠️ Recommended Fixes

### Option A: Full pyproject.toml Migration (Recommended)

**Step 1: Remove setup.py**
```bash
rm setup.py
```

**Step 2: Update pyproject.toml**
Add package discovery configuration:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "emotional-processor"  # Keep consistent with current installs
version = "0.1.0"
description = "Process and replay emotional context from LLM conversations using vector databases"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Stuart Swerdloff", email = "stuart@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
]
requires-python = ">=3.10"
dependencies = [
    "qdrant-client>=1.7.0",
    "sentence-transformers>=2.2.2",
    "transformers>=4.35.0",
    "torch>=2.1.0",
    "spacy>=3.7.0",
    "numpy>=1.24.0",
    "pandas>=2.1.0",
    "pydantic>=2.5.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "click>=8.1.0",
    "tqdm>=4.66.0",
    "loguru>=0.7.0",
    "tiktoken>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "black>=23.0.0",
    "isort>=5.12.0",
]

[project.scripts]
ecp-process = "emotional_processor.cli:process_command"
ecp-setup = "emotional_processor.cli:setup_command"
ecp-health = "emotional_processor.cli:health_command"

# CRITICAL: Add package discovery for root-level package
[tool.hatch.build.targets.wheel]
packages = ["emotional_processor"]

[tool.hatch.build.targets.sdist]
include = [
    "/emotional_processor",
    "/tests",
    "/README.md",
    "/pyproject.toml",
]
```

### Option B: Fix setup.py (Alternative)

If you prefer to keep `setup.py`, fix the package discovery:

**Step 1: Remove pyproject.toml build-system section**
```toml
# Remove this section from pyproject.toml:
# [build-system]
# requires = ["hatchling"]
# build-backend = "hatchling.build"
```

**Step 2: Fix setup.py**
```python
"""Setup script for Emotional Conversation Processor."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="emotional-processor",  # Match pyproject.toml
    version="0.1.0",
    author="Stuart Swerdloff",
    author_email="stuart@example.com",
    description="Process and replay emotional context from LLM conversations using vector databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjswerdloff/emotional-conversation-processor",
    packages=find_packages(),  # Remove where="src"
    # Remove package_dir line
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=[
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "spacy>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "tiktoken>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecp-process=emotional_processor.cli:process_command",
            "ecp-setup=emotional_processor.cli:setup_command",
            "ecp-health=emotional_processor.cli:health_command",
        ],
    },
)
```

## 🧪 Testing the Fix

After implementing either option, test the packaging:

### 1. **Clean Installation Test**
```bash
# Remove any existing installation
pip uninstall emotional-processor emotional-conversation-processor

# Test fresh install
cd /path/to/emotional-conversation-processor
pip install -e .

# Test imports
python -c "from emotional_processor.processors.emotion_classifier import EmotionClassifier; print('Success!')"
```

### 2. **Build Test**
```bash
# Test that the package builds correctly
python -m build

# Check that wheel contains the right files
unzip -l dist/*.whl | grep emotional_processor
```

### 3. **Clean Environment Test**
```bash
# Test in fresh virtual environment
python -m venv test_env
source test_env/bin/activate
pip install -e .
python -c "import emotional_processor.processors.emotion_classifier"
deactivate
rm -rf test_env
```

## 🎯 Priority Order

1. **Immediate**: Choose Option A (pyproject.toml only) - it's the modern standard
2. **Critical**: Add the `[tool.hatch.build.targets.wheel]` section for package discovery
3. **Important**: Remove `setup.py` to eliminate conflicts
4. **Verification**: Test editable install works without manual .pth file fixes

## 📝 Verification Checklist

- [ ] Only one build system (either pyproject.toml OR setup.py, not both)
- [ ] Package discovery correctly configured for actual directory structure
- [ ] Consistent package name across all configuration files
- [ ] `pip install -e .` creates working imports without manual intervention
- [ ] No empty .pth files after editable install
- [ ] All CLI entry points work correctly
- [ ] Package builds successfully with `python -m build`

## 🔗 Related Issues

These packaging fixes will resolve:
- Empty .pth files during editable installs
- Import errors despite "successful" installation
- Need for manual path manipulation in development environments
- VS Code import resolution issues
- Dependency management reliability problems

## 💡 Future Recommendations

1. **Add packaging tests** to CI/CD pipeline
2. **Use `check-manifest`** to verify package contents
3. **Test installation** in multiple Python versions
4. **Document installation process** clearly in README
5. **Consider using `hatch`** CLI for development workflow consistency