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

import hashlib
import json
import time
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from emotional_processor.core.models import ConversationSegment, SpeakerType
from emotional_processor.storage.vector_store import AIMemoryStorageIntegrityError, ConversationVectorStore


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
        """Contract: Batch storage with verification should verify each segment's integrity."""
        segments = [
            ConversationSegment(
                content=f"Batch segment {i}",
                speaker=SpeakerType.USER,
                segment_id=f"batch_{i}",
            )
            for i in range(3)
        ]
        embeddings = [[0.1 * i] * 384 for i in range(3)]

        # Set up successful batch storage and verification
        self.mock_client.upsert.return_value = None

        def batch_verification_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            # Simulate successful batch verification - return all requested points
            results = []
            for segment_id in ids:
                point = Mock()
                point.id = segment_id
                # Find matching segment and embedding
                seg_index = int(segment_id.split("_")[1])
                point.vector = embeddings[seg_index]

                # Create payload with verification metadata
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
                    "_verification": {
                        "vector_hash": f"hash_{segment_id}_vector",
                        "payload_hash": f"hash_{segment_id}_payload",
                        "write_timestamp": 1234567890.0,
                        "write_attempt": 1,
                        "verification_enabled": True,
                    },
                }
                results.append(point)
            return results

        self.mock_client.retrieve.side_effect = batch_verification_behavior

        # Contract: Batch storage should succeed with verification
        point_ids = self.store.store_batch_segments(segments, embeddings)

        # Contract: All segments should be stored
        assert len(point_ids) == 3
        assert point_ids == ["batch_0", "batch_1", "batch_2"]

        # Contract: Batch approach should be attempted first (fewer calls than individual)
        # In successful batch case, we expect 1 batch upsert + 1 batch verification
        assert self.mock_client.upsert.call_count >= 1
        assert self.mock_client.retrieve.call_count >= 1

    def test_batch_verification_contract_fail_fast_on_single_failure(self) -> None:
        """Contract: Batch should fail fast if any segment verification fails after all retries."""
        segments = [
            ConversationSegment(
                content=f"Batch segment {i}",
                speaker=SpeakerType.USER,
                segment_id=f"batch_fail_{i}",
            )
            for i in range(3)
        ]
        embeddings = [[0.0] * 384 for i in range(3)]

        # Configure consistent failure to exhaust all retries (batch + individual fallback)
        failure_error = Exception("Persistent storage failure")
        self.mock_client.upsert.side_effect = [failure_error] * 20  # Enough for all retry attempts

        # Contract: Should fail fast with AIMemoryStorageIntegrityError after all retries
        from emotional_processor.storage.vector_store import AIMemoryStorageIntegrityError

        with pytest.raises(AIMemoryStorageIntegrityError) as exc_info:
            self.store.store_batch_segments(segments, embeddings)

        # Contract: Error should contain context about the failure
        error = exc_info.value
        assert error.point_id is not None  # Should identify which segment failed
        assert error.error_type == "individual_verification_failed"

        # Contract: Should have attempted multiple retries before failing
        assert self.mock_client.upsert.call_count >= 3  # At least 3 attempts per retry logic


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


class TestBatchPartialFailureContracts:
    """Contract tests for batch verification partial failure scenarios."""

    def setup_method(self) -> None:
        """Set up test environment with mocked Qdrant client."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=True, max_batch_retries=3)
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_batch_partial_failure_retry_contract(self) -> None:
        """Contract: When some segments fail verification, system retries batch then falls back to individual."""
        segments = [
            ConversationSegment(
                content=f"Partial failure segment {i}",
                speaker=SpeakerType.USER,
                segment_id=f"partial_{i}",
            )
            for i in range(5)
        ]
        embeddings = [[0.1 * i] * 384 for i in range(5)]

        # Behavioral mock: simulate partial verification failures
        def partial_failure_verification_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            """Simulate partial verification failure - some segments verify, others don't."""
            results = []
            for i, segment_id in enumerate(ids):
                point = Mock()
                point.id = segment_id

                # First 3 attempts: return partial failures (segments 2,3 fail verification)
                if self.mock_client.retrieve.call_count <= 3 and i in [2, 3]:
                    # Simulate verification failure: wrong vector or missing point
                    if i == 2:
                        # Wrong vector (will fail cosine similarity)
                        point.vector = [0.9] * 384  # Different from expected [0.2] * 384
                    else:
                        # Missing point (simulate retrieval failure)
                        continue  # Don't add to results - simulates missing point
                else:
                    # Correct vector for successful verification
                    point.vector = embeddings[i]

                # Add verification metadata for successful cases
                point.payload = {
                    "content": segments[i].content,
                    "speaker": str(segments[i].speaker),
                    "segment_id": segments[i].segment_id,
                    "_verification": {
                        "vector_hash": "correct_hash",
                        "payload_hash": "correct_payload_hash",
                        "verification_attempt": 1,
                    },
                }
                results.append(point)
            return results

        self.mock_client.retrieve.side_effect = partial_failure_verification_behavior
        self.mock_client.upsert.return_value = None

        # Mock individual storage success for fallback
        def individual_storage_behavior(collection_name: str, points: list[Any], **kwargs: Any) -> None:
            """Simulate successful individual storage."""

        # Mock time.sleep to avoid delays in tests
        with (
            patch("time.sleep") as mock_sleep,
            patch.object(
                self.store, "_store_segments_individually_with_verification", return_value=["partial_2", "partial_3"]
            ) as mock_individual,
        ):
            # Contract: Should eventually succeed with partial batch failure + individual fallback
            # This should trigger batch retries then individual fallback
            point_ids = self.store.store_batch_segments(segments, embeddings)

            # Contract: Should succeed with individual fallback
            assert point_ids == ["partial_2", "partial_3"], f"Expected fallback point IDs, got: {point_ids}"

            # Contract: Should have attempted batch storage multiple times (up to max_batch_retries)
            assert self.mock_client.upsert.call_count >= 3, "Should have retried batch storage 3 times"
            assert self.mock_client.retrieve.call_count >= 3, "Should have verified batch storage 3 times"

            # Contract: Should have used exponential backoff between batch retries
            assert mock_sleep.call_count >= 2, "Should have used exponential backoff delays"

            # Contract: Should have fallen back to individual storage
            mock_individual.assert_called_once(), "Should have called individual fallback"

    def test_batch_retry_exponential_backoff_contract(self) -> None:
        """Contract: Retry delays follow exponential backoff (0.5s, 1s, 2s) before fallback."""
        segments = [
            ConversationSegment(
                content="Backoff test segment",
                speaker=SpeakerType.USER,
                segment_id="backoff_test",
            )
        ]
        embeddings = [[0.5] * 384]

        # Behavioral mock: simulate consistent batch verification failure
        def failing_verification_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            """Simulate consistent verification failure to trigger all retries."""
            # Return empty list to simulate verification failure (missing points)
            return []

        self.mock_client.retrieve.side_effect = failing_verification_behavior
        self.mock_client.upsert.return_value = None

        # Mock individual fallback to also fail (to test full retry sequence)
        with (
            patch("time.sleep") as mock_sleep,
            patch.object(
                self.store,
                "_store_segments_individually_with_verification",
                side_effect=AIMemoryStorageIntegrityError(
                    "Individual verification failed", point_id="backoff_test", error_type="individual_verification_failed"
                ),
            ),
        ):
            # Contract: Should fail after all retries and fallback attempts
            with pytest.raises(AIMemoryStorageIntegrityError):
                self.store.store_batch_segments(segments, embeddings)

            # Contract: Should have used exponential backoff pattern
            actual_delays = [call.args[0] for call in mock_sleep.call_args_list]

            # Should have multiple backoff delays (batch retries + individual retries)
            assert len(actual_delays) >= 4, f"Expected multiple backoff delays, got: {actual_delays}"

            # Verify batch exponential backoff pattern exists in the delays
            # Expected batch delays: 0.5s, 1.0s (first two delays should be for batch retries)
            assert 0.5 in actual_delays, f"Expected 0.5s batch retry delay, got delays: {actual_delays}"
            assert 1.0 in actual_delays, f"Expected 1.0s batch retry delay, got delays: {actual_delays}"

            # Verify individual retry delays also exist (0.1s is typical for individual)
            assert any(delay <= 0.1 for delay in actual_delays), f"Expected individual retry delays, got: {actual_delays}"

    def test_batch_verification_threshold_contract(self) -> None:
        """Contract: System detects and recovers from vectors stored with degraded similarity.

        This test validates the medical-grade verification contract:
        - When retrieved vectors have cosine similarity < 0.999, verification fails
        - The system retries and eventually recovers
        - This protects against data corruption during storage/retrieval

        Following the Test Writing Philosophy:
        - We test the CONTRACT (detection and recovery), not implementation
        - We use BEHAVIORAL mocking that simulates real system behavior
        - We focus on the WHAT: system protects data integrity
        """
        # Arrange: Create test data
        segments = [
            ConversationSegment(
                content="Threshold test segment",
                speaker=SpeakerType.USER,
                segment_id="threshold_test",
            )
        ]
        embeddings = [[0.7] * 384]

        # Create realistic vector corruption scenario
        # In production, corruption might occur due to:
        # - Precision loss during storage/retrieval
        # - Network transmission errors
        # - Hardware failures
        original_vector = np.array(embeddings[0])

        # Simulate realistic corruption: ~10% of dimensions significantly degraded
        # This models scenarios like:
        # - Bit flips in storage
        # - Precision loss in network transmission
        # - Memory corruption
        corrupted_vector = original_vector.copy()
        corruption_indices = np.random.RandomState(42).choice(384, 38, replace=False)  # 10% of 384
        corrupted_vector[corruption_indices] *= 0.5  # Significant degradation to ensure < 0.999

        # Verify this creates similarity just below threshold
        corrupted_similarity = np.dot(original_vector, corrupted_vector) / (
            np.linalg.norm(original_vector) * np.linalg.norm(corrupted_vector)
        )
        assert corrupted_similarity < 0.999, (
            f"Test vector corruption should create similarity < 0.999, got {corrupted_similarity:.6f}"
        )

        # Create behavioral mock that simulates real verification system
        verification_state = {"attempt_count": 0, "corruption_active": True, "verification_history": []}

        def realistic_verification_behavior(collection_name: str, ids: list[str], **kwargs: Any) -> list[Mock]:
            """Behavioral mock: Simulates real system with transient corruption.

            This mock behaves like the real Qdrant system:
            1. First attempt returns corrupted data (simulating real-world corruption)
            2. Subsequent attempts return correct data (simulating retry success)
            3. Maintains verification metadata like the real system
            """
            verification_state["attempt_count"] += 1
            attempt = verification_state["attempt_count"]

            # Track verification attempts for diagnostic purposes
            verification_state["verification_history"].append(
                {"attempt": attempt, "corruption_active": verification_state["corruption_active"], "timestamp": time.time()}
            )

            results = []
            for segment_id in ids:
                point = Mock()
                point.id = segment_id

                # Behavioral decision: First attempt returns corrupted data
                if attempt == 1 and verification_state["corruption_active"]:
                    # Return corrupted vector (simulating real corruption)
                    point.vector = corrupted_vector.tolist()
                    vector_to_hash = corrupted_vector
                else:
                    # Return correct vector (simulating successful retry)
                    point.vector = original_vector.tolist()
                    vector_to_hash = original_vector
                    verification_state["corruption_active"] = False

                # Simulate real system's hash computation
                vector_bytes = np.array(vector_to_hash, dtype=np.float32).tobytes()
                actual_vector_hash = hashlib.sha256(vector_bytes).hexdigest()

                # Build complete payload as real system would
                segment = segments[0]  # We know there's only one in this test
                payload_data = {
                    "content": segment.content,
                    "speaker": str(segment.speaker),
                    "segment_id": segment.segment_id,
                    "timestamp": segment.timestamp,
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
                }

                # Compute payload hash as real system would
                payload_str = json.dumps(payload_data, sort_keys=True, default=str)
                actual_payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()

                # Complete payload with verification metadata
                point.payload = {
                    **payload_data,
                    "_verification": {
                        "vector_hash": actual_vector_hash,
                        "payload_hash": actual_payload_hash,
                        "write_timestamp": time.time() - 0.5,  # Written 0.5s ago
                        "write_attempt": 1,
                        "verification_enabled": True,
                        "verification_attempt": attempt,
                    },
                }
                results.append(point)

            return results

        # Configure mock with behavioral implementation
        self.mock_client.retrieve.side_effect = realistic_verification_behavior
        self.mock_client.upsert.return_value = None

        # Act: Execute the operation under test
        with patch("time.sleep") as mock_sleep:
            point_ids = self.store.store_batch_segments(segments, embeddings)

        # Assert: Verify the CONTRACT was fulfilled
        # Contract 1: Operation eventually succeeds despite corruption
        assert point_ids == ["threshold_test"], "Contract violation: System should eventually store segment successfully"

        # Contract 2: System detected corruption and retried
        assert verification_state["attempt_count"] >= 2, "Contract violation: System should retry when similarity < 0.999"

        # Contract 3: Corruption was resolved through retry
        assert not verification_state["corruption_active"], (
            "Contract violation: System should recover from transient corruption"
        )

        # Contract 4: System used appropriate retry delays
        assert mock_sleep.call_count >= 1, "Contract violation: System should delay between retry attempts"

        # Diagnostic information (following philosophy: tests are documentation)
        # print(f"\nDiagnostic: Verification attempts: {verification_state['attempt_count']}")
        # print(f"Diagnostic: Corruption detected with similarity: {corrupted_similarity:.6f}")
        # print(f"Diagnostic: 10% of dimensions corrupted (38/384) by factor 0.5")
        # print(f"Diagnostic: Recovery successful after retry")
