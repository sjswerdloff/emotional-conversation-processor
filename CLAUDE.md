# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands

- Install deps: `make install-dev` or `uv pip install -r requirements-dev.txt`
- Run all tests: `make test` or `pytest`
- Run unit tests: `make test-unit` or `pytest tests/unit/`
- Run single test: `pytest tests/path/to/test_file.py::TestClass::test_function -v`
- Format code: `make format` or `ruff format src/ tests/ scripts/`
- Lint code: `make lint` or `ruff check src/ tests/ scripts/`
- Type check: `make type-check` or `mypy src/`
- Fix lint issues: `make fix`

## Code Style

- **Imports**: Use `ruff` format (stdlib → third-party → first-party → local)
- **Formatting**: Double quotes, 127 max line length, 4-space indentation
- **Types**: Use type annotations (typed Python). Union types with `|` (Python 3.10+)
- **Docstrings**: Google style with full args/returns documentation
- **Error handling**: Use specific exceptions, log with `loguru.logger`
- **Classes**: Use dataclasses for data containers
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Testing**: Use pytest fixtures, mark tests with appropriate markers

read the following documents that are in /Users/stuartswerdloff/ai/AICollaboratorGuidelines/
AI_Human_Collaboration_Methodology.md
CodingToAvoidLintingErrors.md
InsightsIntoFixingLintingErrors.md
TestWritingPhilosophy.md
Testing_and_Debugging_Methodology.md

after completing a round of editing, reflect on the work done, and if you learned anything that wasn't in one of the documents listed above,
leave an Insights.md document in the project root that you are working on.
