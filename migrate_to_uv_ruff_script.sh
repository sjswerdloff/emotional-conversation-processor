#!/bin/bash

# Migration script: Convert from pip/black/flake8/isort to uv/ruff

echo "🔄 Migrating to uv and ruff..."

# Remove old configuration files
echo "🗑️  Removing old configuration files..."
if [ -f ".flake8" ]; then
    rm .flake8
    echo "  ✅ Removed .flake8"
fi

# Remove old tool sections from setup.cfg if it exists
if [ -f "setup.cfg" ]; then
    echo "⚠️  Found setup.cfg - please manually remove [flake8], [isort], [tool:pytest] sections"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  ✅ uv installed"
    echo "  ⚠️  Please restart your shell or run: source ~/.bashrc"
else
    echo "✅ uv already installed"
fi

# Create new virtual environment with uv
echo "🐍 Creating new virtual environment with uv..."
if [ -d "venv" ]; then
    echo "  ⚠️  Removing old venv directory..."
    rm -rf venv
fi

if [ -d ".venv" ]; then
    echo "  ⚠️  Removing old .venv directory..."
    rm -rf .venv
fi

uv venv
echo "  ✅ New virtual environment created in .venv/"

echo ""
echo "🎉 Migration setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the new environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Install dependencies:"
echo "     uv sync --dev"
echo "     uv pip install -e ."
echo ""
echo "  3. Test the new tools:"
echo "     uv run ruff check src/"
echo "     uv run ruff format src/"
echo "     uv run mypy src/"
echo ""
echo "  4. Install pre-commit hooks:"
echo "     uv run pre-commit install"
echo ""
echo "  5. Run tests:"
echo "     uv run pytest"
echo ""
echo "🚀 Or use the Makefile for convenience:"
echo "     make dev-setup    # Complete setup"
echo "     make check        # Run all quality checks"
echo "     make test         # Run tests"
echo "     make help         # See all available commands"
echo ""
echo "📚 See DEVELOPMENT.md for detailed usage instructions"

# Update IDE configurations if they exist
echo ""
echo "🔧 IDE Configuration Updates:"

if [ -f ".vscode/settings.json" ]; then
    echo "  ⚠️  VS Code: Update .vscode/settings.json to use ruff instead of flake8/black/isort"
fi

if [ -f ".idea/workspace.xml" ]; then
    echo "  ⚠️  PyCharm: Update external tools configuration to use ruff"
fi

echo ""
echo "✨ Migration complete! Your project now uses uv + ruff for faster development."
