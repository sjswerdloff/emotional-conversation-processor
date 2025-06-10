"""Test configuration and fixtures for the Emotional Conversation Processor."""

import contextlib
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import docker
import numpy as np
import pytest

from emotional_processor.core.models import ConversationSegment, ProcessingStats
from emotional_processor.embeddings.emotion_aware_embedder import EmotionAwareEmbedder
from emotional_processor.processors.emotion_classifier import EmotionClassifier
from emotional_processor.processors.technical_detector import TechnicalContentDetector
from emotional_processor.storage.vector_store import ConversationVectorStore


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Global test configuration."""
    return {
        "qdrant_host": "localhost",
        "qdrant_port": 6334,  # Different port for testing
        "collection_name": "test_conversation_history",
        "embedding_dimension": 384,
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


@pytest.fixture(scope="session", autouse=True)
def _setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment including Qdrant container."""
    container = None
    try:
        # Try to start Qdrant container for integration tests
        client = docker.from_env()

        # Check if container already exists
        existing_containers = client.containers.list(all=True, filters={"name": "qdrant-test"})
        if existing_containers:
            container = existing_containers[0]
            if container.status != "running":
                container.start()
        else:
            # Start new Qdrant container
            container = client.containers.run(
                "qdrant/qdrant",
                ports={"6333/tcp": 6334},  # Map to different port for testing
                detach=True,
                remove=True,
                name="qdrant-test",
            )

        # Wait for Qdrant to be ready
        time.sleep(5)

        yield

    except Exception:
        # If Docker is not available, skip container setup
        yield

    finally:
        # Cleanup
        if container:
            with contextlib.suppress(Exception):
                container.stop()


@pytest.fixture
def sample_conversation_segments() -> list[ConversationSegment]:
    """Generate sample conversation segments for testing."""
    return [
        ConversationSegment(
            content="I'm really excited about working on this project with you!",
            speaker="User",
            timestamp="2024-01-01T10:00:00Z",
            emotional_score=0.8,
            emotional_labels=["joy", "excitement"],
            technical_score=0.1,
            importance_weight=0.7,
            segment_id="seg_001",
            conversation_id="conv_test",
        ),
        ConversationSegment(
            content="Let me implement the database connection function for you.",
            speaker="Assistant",
            timestamp="2024-01-01T10:05:00Z",
            emotional_score=0.2,
            technical_score=0.9,
            emotional_labels=[],
            importance_weight=0.2,
            segment_id="seg_002",
            conversation_id="conv_test",
        ),
        ConversationSegment(
            content="Thank you so much! Your help means the world to me.",
            speaker="User",
            timestamp="2024-01-01T10:10:00Z",
            emotional_score=0.9,
            emotional_labels=["gratitude", "joy"],
            technical_score=0.0,
            importance_weight=0.8,
            segment_id="seg_003",
            conversation_id="conv_test",
        ),
        ConversationSegment(
            content="Here's the Python function implementation:\n```python\ndef process_data():\n    return True\n```",
            speaker="Assistant",
            timestamp="2024-01-01T10:15:00Z",
            emotional_score=0.1,
            emotional_labels=[],
            technical_score=0.95,
            importance_weight=0.1,
            segment_id="seg_004",
            conversation_id="conv_test",
        ),
        ConversationSegment(
            content="I'm feeling overwhelmed by all these technical details, but I appreciate your patience.",
            speaker="User",
            timestamp="2024-01-01T10:20:00Z",
            emotional_score=0.7,
            emotional_labels=["anxiety", "gratitude"],
            technical_score=0.3,
            importance_weight=0.6,
            segment_id="seg_005",
            conversation_id="conv_test",
        ),
    ]


@pytest.fixture
def sample_emotional_texts() -> dict[str, dict[str, Any]]:
    """Sample texts with expected emotional classifications."""
    return {
        "high_joy": {
            "text": "I'm absolutely thrilled that we solved this together! This collaboration has been wonderful.",
            "expected_score_min": 0.6,
            "expected_labels": ["joy"],
        },
        "frustration": {
            "text": "This is incredibly frustrating. Nothing seems to work correctly.",
            "expected_score_min": 0.5,
            "expected_labels": ["anger", "frustration"],
        },
        "neutral_technical": {
            "text": "The function returns a dictionary with the processed data.",
            "expected_score_max": 0.4,
            "expected_labels": [],
        },
        "complex_mixed": {
            "text": "I'm grateful for your help, but I'm also worried about the timeline.",
            "expected_score_min": 0.4,
            "expected_labels": ["gratitude", "fear"],
        },
        "pure_gratitude": {
            "text": "Thank you so much for your patience and understanding. This means everything to me.",
            "expected_score_min": 0.7,
            "expected_labels": ["gratitude", "joy"],
        },
    }


@pytest.fixture
def technical_content_samples() -> dict[str, dict[str, Any]]:
    """Sample texts with expected technical scores."""
    return {
        "high_technical": {
            "text": """
            Here's the Python function implementation:
            ```python
            def process_data(data):
                try:
                    return json.loads(data)
                except Exception as e:
                    print(f"Error: {e}")
            ```
            This handles the API response parsing.
            """,
            "expected_score_min": 0.7,
        },
        "low_technical": {
            "text": "I really appreciate your patience and understanding. This conversation means a lot to me.",
            "expected_score_max": 0.1,
        },
        "mixed_content": {
            "text": "I'm excited about this project! We need to implement the authentication API, but I'm confident we can do it.",
            "expected_score_range": (0.2, 0.6),
        },
        "pure_code": {
            "text": "```javascript\nfunction getData() {\n  return fetch('/api/data').then(res => res.json());\n}\n```",
            "expected_score_min": 0.8,
        },
    }


@pytest.fixture
def sample_conversation_file(tmp_path) -> str:
    """Create a temporary conversation file for testing."""
    conversation_content = """
User [2024-01-01T10:00:00Z]: I'm really struggling with this problem and feeling overwhelmed.

Assistant [2024-01-01T10:01:00Z]: I understand that you're feeling overwhelmed. Let's break this down step by step.

User [2024-01-01T10:02:00Z]: Thank you, that helps a lot. Can you show me the implementation?

Assistant [2024-01-01T10:03:00Z]: Here's the code:
```python
def solve_problem():
    return "solution"
```

User [2024-01-01T10:05:00Z]: Perfect! I'm so grateful for your patience and help.
    """

    conversation_file = tmp_path / "test_conversation.txt"
    conversation_file.write_text(conversation_content)
    return str(conversation_file)


@pytest.fixture
def mock_emotion_classifier() -> Mock:
    """Mock emotion classifier for testing."""
    mock_classifier = Mock(spec=EmotionClassifier)

    # Setup default return values
    mock_classifier.classify_single.return_value = (0.7, ["joy", "gratitude"])
    mock_classifier.classify_batch.return_value = [(0.7, ["joy"]), (0.3, [])]
    mock_classifier.is_highly_emotional.return_value = True
    mock_classifier.get_primary_emotion.return_value = "joy"

    return mock_classifier


@pytest.fixture
def mock_technical_detector() -> Mock:
    """Mock technical content detector for testing."""
    mock_detector = Mock(spec=TechnicalContentDetector)

    # Setup default return values
    mock_detector.calculate_technical_score.return_value = 0.3
    mock_detector.is_highly_technical.return_value = False
    mock_detector.should_deprioritize.return_value = False

    return mock_detector


@pytest.fixture
def mock_embedder() -> Mock:
    """Mock emotion-aware embedder for testing."""
    mock_embedder = Mock(spec=EmotionAwareEmbedder)

    # Setup default return values
    mock_embedding = [0.1] * 384  # 384-dimensional vector
    mock_embedder.create_embedding.return_value = mock_embedding
    mock_embedder.create_contextual_embedding.return_value = mock_embedding
    mock_embedder.create_batch_embeddings.return_value = [mock_embedding, mock_embedding]
    mock_embedder.create_query_embedding.return_value = mock_embedding
    mock_embedder.similarity.return_value = 0.85
    mock_embedder.dimension = 384

    return mock_embedder


@pytest.fixture
def mock_vector_store(test_config: dict[str, Any]) -> Mock:
    """Mock vector store for testing."""
    mock_store = Mock(spec=ConversationVectorStore)

    # Setup default return values
    mock_store.store_segment.return_value = "point_123"
    mock_store.store_batch_segments.return_value = ["point_1", "point_2", "point_3"]
    mock_store.search_similar.return_value = []
    mock_store.search_emotional_context.return_value = []
    mock_store.get_conversation_segments.return_value = []
    mock_store.delete_segment.return_value = True
    mock_store.collection_name = test_config["collection_name"]

    return mock_store


@pytest.fixture
def real_vector_store(test_config: dict[str, Any]) -> Generator[ConversationVectorStore, None, None]:
    """Real vector store for integration tests (requires Qdrant)."""
    try:
        store = ConversationVectorStore(
            collection_name=f"test_{int(time.time())}",
            host=test_config["qdrant_host"],
            port=test_config["qdrant_port"],
            embedding_dimension=test_config["embedding_dimension"],
        )

        yield store

        # Cleanup
        with contextlib.suppress(Exception):
            store.clear_collection()

    except Exception as e:
        pytest.skip(f"Qdrant not available for integration tests: {e}")


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Generate sample embedding vectors for testing."""
    np.random.seed(42)  # For reproducible tests

    embeddings = []
    for _i in range(5):
        # Create somewhat realistic embeddings
        embedding = np.random.normal(0, 0.1, 384).tolist()
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        embeddings.append(embedding)

    return embeddings


@pytest.fixture
def processing_stats() -> ProcessingStats:
    """Sample processing statistics for testing."""
    return ProcessingStats(
        total_segments=100,
        emotional_segments=30,
        technical_segments=25,
        processing_time=45.5,
        embedding_time=12.3,
        classification_time=8.7,
        errors=2,
        warnings=5,
    )


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests that don't require external dependencies")
    config.addinivalue_line("markers", "integration: Integration tests that require external services")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_qdrant: Tests that require Qdrant server")
    config.addinivalue_line("markers", "requires_gpu: Tests that require GPU acceleration")


# Custom assertions for testing
def assert_emotional_score_reasonable(score: float, min_score: float = 0.0, max_score: float = 1.0) -> None:
    """Assert that an emotional score is within reasonable bounds."""
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert min_score <= score <= max_score, f"Score {score} not in range [{min_score}, {max_score}]"


def assert_embedding_valid(embedding: list[float], expected_dim: int = 384) -> None:
    """Assert that an embedding vector is valid."""
    assert isinstance(embedding, list), f"Expected list, got {type(embedding)}"
    assert len(embedding) == expected_dim, f"Expected {expected_dim} dimensions, got {len(embedding)}"
    assert all(isinstance(x, float) for x in embedding), "All embedding values must be floats"

    # Check for reasonable values (not all zeros, not too extreme)
    embedding_array = np.array(embedding)
    assert not np.all(embedding_array == 0), "Embedding should not be all zeros"
    assert np.all(np.abs(embedding_array) < 10), "Embedding values seem too extreme"


def assert_conversation_segment_valid(segment: ConversationSegment) -> None:
    """Assert that a conversation segment is valid."""
    assert isinstance(segment, ConversationSegment)
    assert segment.content, "Segment content should not be empty"
    assert segment.speaker, "Segment should have a speaker"
    assert 0.0 <= segment.emotional_score <= 1.0, "Emotional score should be between 0 and 1"
    assert 0.0 <= segment.technical_score <= 1.0, "Technical score should be between 0 and 1"
    assert 0.0 <= segment.importance_weight <= 1.0, "Importance weight should be between 0 and 1"
    assert segment.segment_id, "Segment should have an ID"


# Utility functions for tests
def create_test_segment(
    content: str = "Test content",
    emotional_score: float = 0.5,
    technical_score: float = 0.3,
    emotional_labels: list[str] = None,  # type: ignore
) -> ConversationSegment:
    """Create a test conversation segment with specified parameters."""
    if emotional_labels is None:
        emotional_labels = []
    if not emotional_labels:
        emotional_labels = ["neutral"]

    return ConversationSegment(
        content=content,
        speaker="User",
        emotional_score=emotional_score,
        emotional_labels=emotional_labels,
        technical_score=technical_score,
        importance_weight=max(0.0, emotional_score - technical_score * 0.5),
    )


def load_test_conversation(filename: str) -> str:
    """Load a test conversation file from fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "test_conversations"
    file_path = fixtures_dir / filename

    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    else:
        pytest.skip(f"Test conversation file not found: {filename}")
