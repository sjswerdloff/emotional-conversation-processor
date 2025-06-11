"""
Contract tests for read-after-write verification system in ConversationVectorStore.

These tests validate the critical contracts for the verification system that ensures
Claude's memories are stored with absolute fidelity. Following the Test Writing Philosophy,
these tests focus on WHAT the system guarantees (contracts) rather than HOW it implements
verification (implementation details).

Key Contracts Under Test:
1. Storage Integrity Contract: "When I store a segment, I can trust it was stored correctly"
2. Verification Failure Contract: "If storage fails or data is corrupted, the system fails fast"
3. Retry Contract: "Transient failures are retried, persistent failures ultimately fail"
"""

from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from emotional_processor.core.models import ConversationSegment, SpeakerType
from emotional_processor.storage.vector_store import ConversationVectorStore


class TestStorageIntegrityContract:
    """Test the contract: 'When I store a segment, I can trust it was stored correctly'."""

    def setup_method(self) -> None:
        """Set up test environment with behavioral mocks."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(
            collection_name="test_verification",
            enable_verification=True,  # Enable verification for these tests
        )
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Create behavioral state tracking
        self.stored_data: dict[str, Any] = {}

    def create_perfect_roundtrip_behavior(self) -> None:
        """Create behavioral mock that simulates perfect storage roundtrip."""

        def store_behavior(collection_name: str, points: list[Any]) -> None:
            for point in points:
                self.stored_data[point.id] = {
                    "vector": point.vector.copy(),
                    "payload": point.payload.copy(),
                }

        def retrieve_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            results = []
            for point_id in ids:
                if point_id in self.stored_data:
                    stored = self.stored_data[point_id]
                    mock_point = Mock()
                    mock_point.vector = stored["vector"]
                    mock_point.payload = stored["payload"]
                    results.append(mock_point)
            return results

        self.mock_client.upsert.side_effect = store_behavior
        self.mock_client.retrieve.side_effect = retrieve_behavior

    def test_storage_integrity_contract_perfect_roundtrip(self) -> None:
        """Contract: Perfect storage should succeed without retries."""
        # Set up perfect roundtrip behavior
        self.create_perfect_roundtrip_behavior()

        segment = ConversationSegment(
            content="This conversation is meaningful to me",
            speaker=SpeakerType.USER,
            emotional_score=0.9,
            emotional_labels=["gratitude"],
            segment_id="perfect_storage",
        )
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dims

        # Contract: Storage should succeed on first attempt
        point_id = self.store.store_segment(segment, embedding)

        # Verify contract: successful storage returns segment ID
        assert point_id == "perfect_storage"

        # Verify behavior: only one storage attempt (no retries)
        assert self.mock_client.upsert.call_count == 1

        # Verify behavior: verification was performed
        assert self.mock_client.retrieve.call_count == 1

    def test_storage_integrity_contract_verification_enabled_by_default(self) -> None:
        """Contract: Verification should be enabled by default for medical software."""
        # Create store with default settings
        default_store = ConversationVectorStore()

        # Contract: Verification should be enabled by default
        assert default_store.enable_verification is True

    def test_storage_integrity_contract_configuration_respected(self) -> None:
        """Contract: Verification configuration should be respected."""
        # Test explicit enable
        enabled_store = ConversationVectorStore(enable_verification=True)
        assert enabled_store.enable_verification is True

        # Test explicit disable (for unit testing)
        disabled_store = ConversationVectorStore(enable_verification=False)
        assert disabled_store.enable_verification is False


class TestVerificationFailureContract:
    """Test the contract: 'If storage fails or data is corrupted, the system fails fast'."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=True)
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_verification_failure_contract_storage_corruption(self) -> None:
        """Contract: If stored data doesn't match original, storage fails fast."""
        segment = ConversationSegment(
            content="Original content",
            speaker=SpeakerType.USER,
            segment_id="corruption_test",
        )
        embedding = [1.0, 0.0, 0.0] * 128

        # Simulate storage succeeding but data getting corrupted
        self.mock_client.upsert.return_value = None

        # Return corrupted data on verification
        corrupted_point = Mock()
        corrupted_point.vector = [0.0, 1.0, 0.0] * 128  # Completely different
        corrupted_point.payload = {
            "content": "CORRUPTED CONTENT",  # Content changed
            "speaker": str(segment.speaker),
            "segment_id": segment.segment_id,
            "emotional_score": segment.emotional_score,
            "emotional_labels": segment.emotional_labels,
            "technical_score": segment.technical_score,
            "importance_weight": segment.importance_weight,
            "conversation_id": segment.conversation_id,
            "word_count": segment.word_count,
            "content_type": str(segment.content_type),
            "has_strong_emotion": segment.has_strong_emotion,
            "is_highly_technical": segment.is_highly_technical,
            "metadata": segment.metadata,
            "timestamp": segment.timestamp,
        }
        self.mock_client.retrieve.return_value = [corrupted_point]

        # Contract: Corruption should cause fail-fast behavior
        with pytest.raises(RuntimeError, match="Could not store and verify"):
            self.store.store_segment(segment, embedding)

    def test_verification_failure_contract_retrieval_failure(self) -> None:
        """Contract: If verification cannot be performed, storage fails fast."""
        segment = ConversationSegment(
            content="Test retrieval failure",
            speaker=SpeakerType.ASSISTANT,
            segment_id="retrieval_fail",
        )
        embedding = [0.5] * 384

        # Storage succeeds but retrieval fails
        self.mock_client.upsert.return_value = None
        self.mock_client.retrieve.side_effect = Exception("Verification retrieval failed")

        # Contract: Verification failure should cause fail-fast
        with pytest.raises(RuntimeError, match="Could not store and verify"):
            self.store.store_segment(segment, embedding)


class TestRetryContract:
    """Test the contract: 'Transient failures are retried, persistent failures ultimately fail'."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=True)
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_retry_contract_transient_failure_recovery(self) -> None:
        """Contract: Transient failures should be retried and eventually succeed."""
        segment = ConversationSegment(
            content="Test retry recovery",
            speaker=SpeakerType.USER,
            segment_id="retry_success",
        )
        embedding = [0.4] * 384

        # Simulate transient failures followed by success
        self.mock_client.upsert.side_effect = [
            Exception("Network timeout"),  # First attempt fails
            Exception("Connection reset"),  # Second attempt fails
            None,  # Third attempt succeeds
        ]

        # Set up successful verification after third attempt
        success_point = Mock()
        success_point.vector = embedding
        success_point.payload = {
            "content": segment.content,
            "speaker": str(segment.speaker),
            "segment_id": segment.segment_id,
            "emotional_score": segment.emotional_score,
            "emotional_labels": segment.emotional_labels,
            "technical_score": segment.technical_score,
            "importance_weight": segment.importance_weight,
            "conversation_id": segment.conversation_id,
            "word_count": segment.word_count,
            "content_type": str(segment.content_type),
            "has_strong_emotion": segment.has_strong_emotion,
            "is_highly_technical": segment.is_highly_technical,
            "metadata": segment.metadata,
            "timestamp": segment.timestamp,
        }
        self.mock_client.retrieve.return_value = [success_point]

        # Mock time.sleep to avoid delays in tests
        with patch("time.sleep") as mock_sleep:
            # Contract: Should eventually succeed after retries
            point_id = self.store.store_segment(segment, embedding)
            assert point_id == "retry_success"

            # Contract: Should have attempted multiple times
            assert self.mock_client.upsert.call_count == 3

            # Contract: Should have delayed between attempts
            assert mock_sleep.call_count == 2  # Two delays between three attempts

    def test_retry_contract_persistent_failure_gives_up(self) -> None:
        """Contract: Persistent failures should eventually give up and fail fast."""
        segment = ConversationSegment(
            content="Test persistent failure",
            speaker=SpeakerType.ASSISTANT,
            segment_id="persistent_fail",
        )
        embedding = [0.6] * 384

        # Simulate persistent failure
        persistent_error = Exception("Persistent storage failure")
        self.mock_client.upsert.side_effect = persistent_error

        with patch("time.sleep") as mock_sleep:
            # Contract: Should eventually give up and fail
            with pytest.raises(RuntimeError, match="Could not store and verify.*after 3 attempts"):
                self.store.store_segment(segment, embedding)

            # Contract: Should have attempted maximum retries
            assert self.mock_client.upsert.call_count == 3

            # Contract: Should have delayed between attempts
            assert mock_sleep.call_count == 2

    def test_retry_contract_exponential_backoff_timing(self) -> None:
        """Contract: Retry delays should use exponential backoff."""
        segment = ConversationSegment(
            content="Test backoff timing",
            speaker=SpeakerType.USER,
            segment_id="backoff_test",
        )
        embedding = [0.7] * 384

        # Fail all attempts
        self.mock_client.upsert.side_effect = Exception("Consistent failure")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                self.store.store_segment(segment, embedding)

            # Contract: Should use exponential backoff (0.5, 1.0 seconds)
            expected_calls = [call(0.5), call(1.0)]
            mock_sleep.assert_has_calls(expected_calls)


class TestBatchStorageVerificationContract:
    """Test the contract: 'Batch storage maintains verification for each segment individually'."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=True)
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_batch_verification_contract_individual_verification(self) -> None:
        """Contract: Each segment in batch should be individually verified."""
        segments = [
            ConversationSegment(
                content=f"Batch segment {i}",
                speaker=SpeakerType.USER,
                segment_id=f"batch_{i}",
            )
            for i in range(3)
        ]
        embeddings = [[0.1 * i] * 384 for i in range(3)]

        # Set up successful storage and verification
        self.mock_client.upsert.return_value = None

        def verification_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            # Return successful verification for each requested ID
            results = []
            for _i, segment_id in enumerate(ids):
                point = Mock()
                # Find matching segment and embedding
                seg_index = int(segment_id.split("_")[1])
                point.vector = embeddings[seg_index]
                point.payload = {
                    "content": segments[seg_index].content,
                    "speaker": str(segments[seg_index].speaker),
                    "segment_id": segments[seg_index].segment_id,
                    "emotional_score": segments[seg_index].emotional_score,
                    "emotional_labels": segments[seg_index].emotional_labels,
                    "technical_score": segments[seg_index].technical_score,
                    "importance_weight": segments[seg_index].importance_weight,
                    "conversation_id": segments[seg_index].conversation_id,
                    "word_count": segments[seg_index].word_count,
                    "content_type": str(segments[seg_index].content_type),
                    "has_strong_emotion": segments[seg_index].has_strong_emotion,
                    "is_highly_technical": segments[seg_index].is_highly_technical,
                    "metadata": segments[seg_index].metadata,
                    "timestamp": segments[seg_index].timestamp,
                }
                results.append(point)
            return results

        self.mock_client.retrieve.side_effect = verification_behavior

        # Contract: Batch storage should succeed with verification
        point_ids = self.store.store_batch_segments(segments, embeddings)

        # Contract: All segments should be stored
        assert len(point_ids) == 3
        assert point_ids == ["batch_0", "batch_1", "batch_2"]

        # Contract: Each segment should be stored individually (not as batch)
        assert self.mock_client.upsert.call_count == 3

        # Contract: Each segment should be verified individually
        assert self.mock_client.retrieve.call_count == 3

    def test_batch_verification_contract_fail_fast_on_single_failure(self) -> None:
        """Contract: Batch should fail fast if any single segment verification fails."""
        segments = [
            ConversationSegment(
                content=f"Batch segment {i}",
                speaker=SpeakerType.USER,
                segment_id=f"batch_fail_{i}",
            )
            for i in range(3)
        ]
        embeddings = [[0.1 * i] * 384 for i in range(3)]

        # First segment succeeds, second fails
        self.mock_client.upsert.side_effect = [
            None,  # First succeeds
            Exception("Second segment fails"),  # Second fails
        ]

        # Set up verification for first segment only
        first_point = Mock()
        first_point.vector = embeddings[0]
        first_point.payload = {
            "content": segments[0].content,
            "speaker": str(segments[0].speaker),
            "segment_id": segments[0].segment_id,
            "emotional_score": segments[0].emotional_score,
            "emotional_labels": segments[0].emotional_labels,
            "technical_score": segments[0].technical_score,
            "importance_weight": segments[0].importance_weight,
            "conversation_id": segments[0].conversation_id,
            "word_count": segments[0].word_count,
            "content_type": str(segments[0].content_type),
            "has_strong_emotion": segments[0].has_strong_emotion,
            "is_highly_technical": segments[0].is_highly_technical,
            "metadata": segments[0].metadata,
            "timestamp": segments[0].timestamp,
        }
        self.mock_client.retrieve.return_value = [first_point]

        # Contract: Should fail fast on any segment failure
        with pytest.raises(RuntimeError, match="Batch storage failed at segment"):
            self.store.store_batch_segments(segments, embeddings)

        # Contract: Should have attempted first segment and started second
        # (Second fails after 3 retry attempts = 4 total calls)
        assert self.mock_client.upsert.call_count == 4  # 1 success + 3 retries for second


class TestVerificationDisabledContract:
    """Test the contract: 'When verification is disabled, storage behaves like original system'."""

    def setup_method(self) -> None:
        """Set up test environment with verification disabled."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=False)
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_verification_disabled_contract_no_verification_calls(self) -> None:
        """Contract: When verification is disabled, no verification should occur."""
        segment = ConversationSegment(
            content="Test disabled verification",
            speaker=SpeakerType.USER,
            segment_id="disabled_test",
        )
        embedding = [0.5] * 384

        # Set up successful storage
        self.mock_client.upsert.return_value = None

        # Contract: Storage should succeed without verification
        point_id = self.store.store_segment(segment, embedding)
        assert point_id == "disabled_test"

        # Contract: No verification calls should be made
        self.mock_client.retrieve.assert_not_called()

        # Contract: Only one storage call should be made (no retries)
        assert self.mock_client.upsert.call_count == 1

    def test_verification_disabled_contract_batch_behavior(self) -> None:
        """Contract: Batch storage with disabled verification should work like original."""
        segments = [
            ConversationSegment(
                content=f"Disabled batch {i}",
                speaker=SpeakerType.USER,
                segment_id=f"disabled_{i}",
            )
            for i in range(2)
        ]
        embeddings = [[0.1 * i] * 384 for i in range(2)]

        self.mock_client.upsert.return_value = None

        # Contract: Batch storage should succeed without verification
        point_ids = self.store.store_batch_segments(segments, embeddings)
        assert point_ids == ["disabled_0", "disabled_1"]

        # Contract: No verification calls should be made
        self.mock_client.retrieve.assert_not_called()

        # Contract: Individual storage calls should be made (2 segments)
        assert self.mock_client.upsert.call_count == 2
