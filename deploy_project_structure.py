#!/usr/bin/env python3
"""
Deployment script for Emotional Conversation Processor project structure.
Run this script from within the emotional-conversation-processor directory.
"""

from pathlib import Path

from loguru import logger


class ProjectDeployer:
    """Handles deployment of project structure and files."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.artifacts_dir = self.project_root / "artifacts"

    def create_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories = [
            "src/emotional_processor/config",
            "src/emotional_processor/core",
            "src/emotional_processor/processors",
            "src/emotional_processor/embeddings",
            "src/emotional_processor/storage",
            "src/emotional_processor/retrieval",
            "src/emotional_processor/utils",
            "src/emotional_processor/api/endpoints",
            "src/emotional_processor/api/middleware",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "tests/e2e",
            "tests/fixtures/test_conversations",
            "data/raw/conversations",
            "data/processed/embeddings",
            "data/models/custom_models",
            "data/exports/replayed_conversations",
            "config",
            "docs/examples",
            "docs/architecture",
            "scripts",
            "notebooks",
            "docker/qdrant",
            ".github/workflows",
            "logs/application",
            "logs/tests",
            "monitoring/prometheus",
            "monitoring/grafana/dashboards",
            "monitoring/alerts",
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Create .gitkeep files for empty directories
        gitkeep_dirs = [
            "data/raw/conversations",
            "data/processed/embeddings",
            "data/models/custom_models",
            "data/exports/replayed_conversations",
            "logs/application",
            "logs/tests",
        ]

        for directory in gitkeep_dirs:
            gitkeep_path = self.project_root / directory / ".gitkeep"
            gitkeep_path.touch()

    def deploy_file(self, content: str, target_path: str) -> None:
        """Deploy a single file to its target location."""
        target_file = self.project_root / target_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Deployed: {target_path}")

    def deploy_artifacts_manually(self) -> None:
        """Deploy artifacts by creating files with their content."""
        logger.info("Deploying artifacts manually...")

        # Since we don't have an artifacts directory, we'll create the essential files
        # Users should copy the artifact contents from Claude's responses

        essential_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "setup.py",
            "pyproject.toml",
            "pytest.ini",
            ".flake8",
            "mypy.ini",
            ".gitignore",
            "README.md",
            "src/emotional_processor/core/models.py",
            "src/emotional_processor/processors/emotion_classifier.py",
            "src/emotional_processor/processors/technical_detector.py",
            "src/emotional_processor/embeddings/emotion_aware_embedder.py",
            "src/emotional_processor/storage/vector_store.py",
            "scripts/process_conversation.py",
            "scripts/setup_database.py",
            "scripts/health_check.py",
            "tests/conftest.py",
            "tests/unit/test_emotion_classifier.py",
            "config/development.yaml",
            "config/testing.yaml",
            "docker/docker-compose.yml",
            "docker/Dockerfile",
            ".github/workflows/ci.yml",
        ]

        logger.info("Essential files that need to be created:")
        for file_path in essential_files:
            logger.info(f"  - {file_path}")

        logger.info("Please copy the artifact contents from Claude's responses into these files.")
        logger.info("You can use the file paths above to create the correct directory structure.")

    def create_init_files(self) -> None:
        """Create __init__.py files for Python packages."""
        init_locations = [
            "src/emotional_processor/__init__.py",
            "src/emotional_processor/config/__init__.py",
            "src/emotional_processor/core/__init__.py",
            "src/emotional_processor/processors/__init__.py",
            "src/emotional_processor/embeddings/__init__.py",
            "src/emotional_processor/storage/__init__.py",
            "src/emotional_processor/retrieval/__init__.py",
            "src/emotional_processor/utils/__init__.py",
            "src/emotional_processor/api/__init__.py",
            "src/emotional_processor/api/endpoints/__init__.py",
            "src/emotional_processor/api/middleware/__init__.py",
            "tests/__init__.py",
            "tests/unit/__init__.py",
            "tests/integration/__init__.py",
            "tests/performance/__init__.py",
            "tests/e2e/__init__.py",
            "tests/fixtures/__init__.py",
            "scripts/__init__.py",
        ]

        for init_file in init_locations:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.touch()
                logger.info(f"Created: {init_file}")

    def set_permissions(self) -> None:
        """Set appropriate permissions for scripts."""
        script_files = [
            "scripts/process_conversation.py",
            "scripts/setup_database.py",
            "scripts/benchmark_performance.py",
            "scripts/export_conversations.py",
            "scripts/health_check.py",
        ]

        for script_file in script_files:
            script_path = self.project_root / script_file
            if script_path.exists():
                script_path.chmod(0o755)
                logger.info(f"Set executable permissions: {script_file}")

    def deploy_all(self) -> None:
        """Deploy the complete project structure."""
        logger.info(f"Deploying to: {self.project_root}")
        logger.info("=" * 50)

        self.create_directory_structure()
        self.create_init_files()
        self.deploy_artifacts_manually()
        self.set_permissions()

        logger.info("=" * 50)
        logger.info("Project structure deployment complete!")
        logger.info("Next steps:")
        logger.info("1. Copy artifact contents from Claude's responses into the listed files")
        logger.info("2. Review the generated files")
        logger.info("3. Install dependencies: pip install -r requirements-dev.txt")
        logger.info("4. Set up Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        logger.info("5. Initialize database: python scripts/setup_database.py")
        logger.info("6. Run tests: pytest")
        logger.info("7. Start development!")


if __name__ == "__main__":
    deployer = ProjectDeployer()
    deployer.deploy_all()
