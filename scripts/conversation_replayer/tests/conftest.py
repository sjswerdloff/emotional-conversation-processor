"""
Test configuration and fixtures for conversation replayer tests

This module provides common test fixtures and utilities for testing
the emotional conversation replayer system.

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Sample conversation data for testing
VALID_CONVERSATION_DATA = {
    "export_info": {
        "timestamp": "2025-06-04T12:00:00.000Z",
        "source": "Test Conversation Exporter",
        "selectorConfig": {"version": "1.0.0", "description": "Test selectors"},
        "total_chats": 1,
        "total_messages": 3,
    },
    "chats": [
        {
            "id": "test-chat-123",
            "title": "Test Conversation",
            "timestamp": "2025-06-04T12:00:00.000Z",
            "url": "https://example.com/chat/test-123",
            "messageCount": 3,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, I'm testing the conversation system.",
                    "timestamp": "2025-06-04T12:00:00.000Z",
                    "element_id": "user-message-0",
                    "element_classes": "user-message",
                    "word_count": 8,
                },
                {
                    "role": "assistant",
                    "content": "Hello! I'm happy to help you test the conversation system. How can I assist you today?",
                    "timestamp": "2025-06-04T12:00:01.000Z",
                    "element_id": "assistant-message-1",
                    "element_classes": "assistant-message",
                    "word_count": 16,
                },
                {
                    "role": "user",
                    "content": "This looks like it's working well.",
                    "timestamp": "2025-06-04T12:00:02.000Z",
                    "element_id": "user-message-2",
                    "element_classes": "user-message",
                    "word_count": 8,
                },
            ],
        }
    ],
}

# Conversation with emotional processor extensions
CONVERSATION_WITH_EXTENSIONS = {
    "export_info": {
        "timestamp": "2025-06-04T12:00:00.000Z",
        "source": "Emotional Conversation Processor",
        "total_chats": 1,
        "total_messages": 2,
    },
    "chats": [
        {
            "id": "enhanced-chat-456",
            "title": "Enhanced Conversation",
            "timestamp": "2025-06-04T12:00:00.000Z",
            "messageCount": 2,
            "messages": [
                {
                    "role": "user",
                    "content": "I need help with a complex technical problem.",
                    "timestamp": "2025-06-04T12:00:00.000Z",
                    "word_count": 9,
                    "emotional_metadata": {
                        "emotion_score": 0.2,
                        "technical_score": 0.8,
                        "importance_weight": 0.7,
                        "processing_status": "human_curated",
                    },
                },
                {
                    "role": "assistant",
                    "content": "I'd be happy to help with your technical problem.",
                    "timestamp": "2025-06-04T12:00:01.000Z",
                    "word_count": 10,
                    "emotional_metadata": {
                        "emotion_score": 0.6,
                        "technical_score": 0.4,
                        "importance_weight": 0.8,
                        "processing_status": "human_curated",
                    },
                    "vector_references": [
                        {
                            "type": "technical_summary",
                            "vector_id": "vec-123-456",
                            "validation_hash": "abc123def456",
                            "fallback_summary": "Technical assistance response",
                        }
                    ],
                    "document_references": [
                        {
                            "type": "original_technical_data",
                            "document_id": "doc-789-012",
                            "size_bytes": 15000,
                            "validation_hash": "def456ghi789",
                        }
                    ],
                },
            ],
        }
    ],
}

# Invalid conversation data for testing validation failures
INVALID_CONVERSATION_MISSING_FIELDS = {
    "export_info": {
        "timestamp": "2025-06-04T12:00:00.000Z"
        # Missing required fields
    },
    "chats": [],
}

INVALID_CONVERSATION_BAD_TIMESTAMPS = {
    "export_info": {"timestamp": "not-a-timestamp", "source": "Test", "total_chats": 1, "total_messages": 1},
    "chats": [
        {
            "id": "bad-timestamp-chat",
            "title": "Bad Timestamps",
            "timestamp": "also-not-a-timestamp",
            "messageCount": 1,
            "messages": [{"role": "user", "content": "Test message", "timestamp": "still-not-a-timestamp"}],
        }
    ],
}

INVALID_CONVERSATION_COUNT_MISMATCH = {
    "export_info": {
        "timestamp": "2025-06-04T12:00:00.000Z",
        "source": "Test",
        "total_chats": 1,
        "total_messages": 5,  # Wrong count
    },
    "chats": [
        {
            "id": "count-mismatch-chat",
            "title": "Count Mismatch",
            "timestamp": "2025-06-04T12:00:00.000Z",
            "messageCount": 3,  # Wrong count
            "messages": [{"role": "user", "content": "Only message", "timestamp": "2025-06-04T12:00:00.000Z"}],
        }
    ],
}


@pytest.fixture
def valid_conversation_data() -> dict[str, Any]:
    """Fixture providing valid conversation data"""
    return VALID_CONVERSATION_DATA.copy()


@pytest.fixture
def conversation_with_extensions() -> dict[str, Any]:
    """Fixture providing conversation with emotional processor extensions"""
    return CONVERSATION_WITH_EXTENSIONS.copy()


@pytest.fixture
def invalid_conversation_missing_fields() -> dict[str, Any]:
    """Fixture providing invalid conversation with missing fields"""
    return INVALID_CONVERSATION_MISSING_FIELDS.copy()


@pytest.fixture
def invalid_conversation_bad_timestamps() -> dict[str, Any]:
    """Fixture providing invalid conversation with bad timestamps"""
    return INVALID_CONVERSATION_BAD_TIMESTAMPS.copy()


@pytest.fixture
def invalid_conversation_count_mismatch() -> dict[str, Any]:
    """Fixture providing invalid conversation with count mismatches"""
    return INVALID_CONVERSATION_COUNT_MISMATCH.copy()


@pytest.fixture
def temp_conversation_file(valid_conversation_data: dict[str, Any]) -> Path:
    """Fixture providing a temporary conversation file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_conversation_data, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_invalid_conversation_file(invalid_conversation_missing_fields: dict[str, Any]) -> Path:
    """Fixture providing a temporary invalid conversation file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_conversation_missing_fields, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_malformed_json_file() -> Path:
    """Fixture providing a temporary malformed JSON file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ invalid json content")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def anthropic_api_key() -> str:
    """Fixture providing a mock Anthropic API key"""
    return "test-anthropic-api-key-1234567890"


@pytest.fixture
def mock_anthropic_response() -> Any:
    """Fixture providing a mock Anthropic API response"""

    class MockContent:
        def __init__(self, text: str):
            self.text = text

    class MockResponse:
        def __init__(self, text: str):
            self.content = [MockContent(text)]

    return MockResponse("This is a test response from the mock Anthropic API.")


# Constants for testing
TEST_MODELS = {
    "claude-4-sonnet": "claude-4-sonnet-20250514",
    "claude-4-opus": "claude-4-opus-20250514",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
}

NON_ANTHROPIC_MODELS = ["gpt-4", "gpt-3.5-turbo", "gemini-pro", "llama-2-70b", "mistral-large"]
