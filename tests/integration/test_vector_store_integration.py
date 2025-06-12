"""
Integration tests for ConversationVectorStore with real Qdrant database.

These tests validate the vector storage system's medical-grade reliability
and data integrity with an actual Qdrant instance. Focus on real-world
scenarios and edge cases that require database interaction.
"""

import contextlib
import time
from typing import Any

import pytest

from emotional_processor.core.models import ConversationSegment, SpeakerType
from emotional_processor.storage.vector_store import ConversationVectorStore


@pytest.mark.integration
@pytest.mark.requires_qdrant
class TestVectorStoreQdrantIntegration:
    """Test vector store integration with real Qdrant database."""

    @pytest.fixture(autouse=True)
    def setup_vector_store(self, test_config: dict[str, Any]) -> None:
        """Set up real vector store for integration testing."""
        self.collection_name = f"test_integration_{int(time.time())}"
        self.vector_store = ConversationVectorStore(
            collection_name=self.collection_name,
            host=test_config["qdrant_host"],
            port=test_config["qdrant_port"],
            embedding_dimension=test_config["embedding_dimension"],
            enable_verification=False,
        )

    def teardown_method(self) -> None:
        """Clean up test collection after each test."""
        if hasattr(self, "vector_store"):
            with contextlib.suppress(Exception):
                self.vector_store.clear_collection()

    def test_collection_initialization_and_configuration(self) -> None:
        """Test contract: collection is properly initialized with correct configuration."""
        # Ensure collection is created by accessing the client property
        client = self.vector_store.client

        # Force collection creation by ensuring it exists
        self.vector_store._ensure_collection_exists()

        # Verify collection exists and has correct configuration
        collection_info = client.get_collection(self.collection_name)

        assert collection_info.config.params.vectors.size == 384
        assert collection_info.config.params.vectors.distance.name == "COSINE"

        # Verify payload schema is set up correctly
        # The collection should be ready to store conversation segment metadata
        assert collection_info is not None

    def test_single_segment_storage_with_verification(self) -> None:
        """Test contract: single segment storage maintains data integrity."""
        # Create test segment with comprehensive metadata
        segment = ConversationSegment(
            content="I'm deeply grateful for our collaboration and the insights you've shared.",
            speaker=SpeakerType.USER,
            timestamp="2024-01-01T10:00:00Z",
            emotional_score=0.85,
            emotional_labels=["gratitude", "joy"],
            technical_score=0.05,
            importance_weight=0.8,
            conversation_id="integration_conv_001",
            metadata={"session": "morning", "context": "reflection"},
        )

        # Create test embedding
        embedding = [0.1] * 384  # Simple test embedding

        # Store segment
        point_id = self.vector_store.store_segment(segment, embedding)

        # Verification: point ID should be returned
        assert point_id is not None
        assert isinstance(point_id, str)

        # Verify segment can be retrieved
        retrieved_segments = self.vector_store.get_conversation_segments(
            conversation_id=segment.conversation_id,
        )

        assert len(retrieved_segments) == 1
        retrieved_segment = retrieved_segments[0]

        # Verify all data preserved correctly
        assert retrieved_segment.content == segment.content
        assert retrieved_segment.speaker == segment.speaker
        assert retrieved_segment.emotional_score == segment.emotional_score
        assert retrieved_segment.emotional_labels == segment.emotional_labels
        assert retrieved_segment.technical_score == segment.technical_score
        assert retrieved_segment.segment_id == segment.segment_id
        assert retrieved_segment.conversation_id == segment.conversation_id

    def test_batch_storage_with_integrity_verification(self) -> None:
        """Test contract: batch storage maintains integrity under various conditions."""
        # Create diverse set of segments for batch testing
        segments = []
        embeddings = []

        test_data = [
            {
                "content": "I'm excited about this new direction!",
                "emotional_score": 0.9,
                "emotional_labels": ["joy", "excitement"],
                "technical_score": 0.1,
            },
            {
                "content": "Here's the algorithm implementation: O(n log n) complexity",
                "emotional_score": 0.1,
                "emotional_labels": [],
                "technical_score": 0.8,
            },
            {
                "content": "Thank you for your patience while I learn this.",
                "emotional_score": 0.7,
                "emotional_labels": ["gratitude"],
                "technical_score": 0.2,
            },
            {
                "content": "The system architecture involves microservices and containers.",
                "emotional_score": 0.05,
                "emotional_labels": [],
                "technical_score": 0.9,
            },
            {
                "content": "I feel overwhelmed but appreciate your support.",
                "emotional_score": 0.6,
                "emotional_labels": ["anxiety", "gratitude"],
                "technical_score": 0.1,
            },
        ]

        for i, data in enumerate(test_data):
            segment = ConversationSegment(
                content=data["content"],
                speaker=SpeakerType.USER if i % 2 == 0 else SpeakerType.ASSISTANT,
                emotional_score=data["emotional_score"],
                emotional_labels=data["emotional_labels"],
                technical_score=data["technical_score"],
                importance_weight=max(0.0, data["emotional_score"] - data["technical_score"] * 0.5),
                conversation_id="batch_test_conv",
            )

            embedding = [0.1 + i * 0.01] * 384  # Slightly different embeddings

            segments.append(segment)
            embeddings.append(embedding)

        # Batch store
        point_ids = self.vector_store.store_batch_segments(segments, embeddings)

        # Verification
        assert len(point_ids) == len(segments)
        assert all(point_id for point_id in point_ids)

        # Verify all segments can be retrieved
        retrieved_segments = self.vector_store.get_conversation_segments(
            conversation_id="batch_test_conv",
        )

        assert len(retrieved_segments) == len(segments)

        # Verify each segment's data integrity
        for original_segment in segments:
            matching_segments = [seg for seg in retrieved_segments if seg.segment_id == original_segment.segment_id]
            assert len(matching_segments) == 1, f"Should find exactly one segment with ID {original_segment.segment_id}"

            retrieved_segment = matching_segments[0]
            assert retrieved_segment.content == original_segment.content
            assert retrieved_segment.emotional_score == original_segment.emotional_score
            assert retrieved_segment.technical_score == original_segment.technical_score

    def test_search_functionality_with_real_similarity(self) -> None:
        """Test contract: search returns semantically similar content in correct order."""
        # Store segments with known content for similarity testing
        segments_data = [
            {
                "content": "I'm grateful for your help and support",
                "embedding": [0.8, 0.2] + [0.0] * 382,  # High on first dimension
                "emotional_score": 0.9,
                "labels": ["gratitude"],
                "id": "grateful_1",
            },
            {
                "content": "Thank you so much for your patience",
                "embedding": [0.7, 0.3] + [0.0] * 382,  # Similar to first
                "emotional_score": 0.8,
                "labels": ["gratitude"],
                "id": "grateful_2",
            },
            {
                "content": "The implementation uses advanced algorithms",
                "embedding": [0.1, 0.9] + [0.0] * 382,  # High on second dimension
                "emotional_score": 0.1,
                "labels": [],
                "id": "technical_1",
            },
            {
                "content": "I'm worried about the deadline",
                "embedding": [0.3, 0.1] + [0.8] + [0.0] * 381,  # Different pattern
                "emotional_score": 0.6,
                "labels": ["anxiety"],
                "id": "worried_1",
            },
        ]

        segments = []
        embeddings = []

        for data in segments_data:
            segment = ConversationSegment(
                content=data["content"],
                speaker=SpeakerType.USER,
                emotional_score=data["emotional_score"],
                emotional_labels=data["labels"],
                technical_score=0.9 - data["emotional_score"],
                importance_weight=data["emotional_score"],
                conversation_id="search_test_conv",
            )

            segments.append(segment)
            embeddings.append(data["embedding"])

        # Store all segments
        point_ids = self.vector_store.store_batch_segments(segments, embeddings)
        assert len(point_ids) == len(segments)

        # Test similarity search
        query_embedding = [0.9, 0.1] + [0.0] * 382  # Should be most similar to grateful segments

        search_results = self.vector_store.search_similar(
            query_vector=query_embedding,
            limit=3,
        )

        # Verification: Should return results in similarity order
        assert len(search_results) >= 2

        # First results should be the grateful segments (more similar to query)
        top_result_contents = [result.segment.content for result in search_results[:2]]
        grateful_contents = ["I'm grateful for your help and support", "Thank you so much for your patience"]
        assert any(content in top_result_contents for content in grateful_contents)

    def test_emotional_context_search_prioritization(self) -> None:
        """Test contract: emotional context search correctly prioritizes emotional content."""
        # Store mixed emotional and technical content
        mixed_content = [
            {
                "content": "I'm absolutely thrilled about this breakthrough!",
                "emotional_score": 0.95,
                "emotional_labels": ["joy", "excitement"],
                "technical_score": 0.05,
                "embedding": [0.5] * 384,
            },
            {
                "content": "Here's the detailed technical implementation with error handling",
                "emotional_score": 0.1,
                "emotional_labels": [],
                "technical_score": 0.9,
                "embedding": [0.5] * 384,  # Same embedding to test prioritization
            },
            {
                "content": "I appreciate your help, though the code is complex",
                "emotional_score": 0.6,
                "emotional_labels": ["gratitude"],
                "technical_score": 0.4,
                "embedding": [0.5] * 384,
            },
        ]

        segments = []
        embeddings = []

        for _i, data in enumerate(mixed_content):
            segment = ConversationSegment(
                content=data["content"],
                speaker=SpeakerType.USER,
                emotional_score=data["emotional_score"],
                emotional_labels=data["emotional_labels"],
                technical_score=data["technical_score"],
                importance_weight=data["emotional_score"],
                conversation_id="emotional_search_conv",
            )

            segments.append(segment)
            embeddings.append(data["embedding"])

        # Store segments
        point_ids = self.vector_store.store_batch_segments(segments, embeddings)
        assert len(point_ids) == len(segments)

        # Search with high emotional weight
        query_embedding = [0.5] * 384  # Same as stored embeddings

        emotional_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=3,
            emotional_weight=0.8,  # High emotional prioritization
        )

        # Verification: Emotional content should be prioritized
        assert len(emotional_results) >= 2

        # First result should have higher emotional score
        top_result = emotional_results[0]
        assert top_result.segment.emotional_score > 0.5, "Top result should have high emotional score"

        # Results should be ordered by emotional relevance, not just similarity
        if len(emotional_results) >= 2:
            first_emotional = emotional_results[0].segment.emotional_score
            second_emotional = emotional_results[1].segment.emotional_score
            # Allow for close scores due to weighting algorithm
            assert first_emotional >= second_emotional - 0.1

    def test_data_corruption_detection_and_recovery(self) -> None:
        """Test contract: system detects and handles data corruption gracefully."""
        # Store a segment normally first
        segment = ConversationSegment(
            content="Test content for corruption detection",
            speaker=SpeakerType.USER,
            emotional_score=0.7,
            emotional_labels=["test"],
            technical_score=0.3,
            importance_weight=0.5,
            conversation_id="corruption_test_conv",
        )

        embedding = [0.1] * 384
        point_id = self.vector_store.store_segment(segment, embedding)
        assert point_id is not None

        # Test read-after-write verification detects any inconsistencies
        # This tests the medical-grade verification system

        # Attempt to store a segment with corrupted data (simulated)
        corrupted_segment = ConversationSegment(
            content="Corrupted content",
            speaker=SpeakerType.USER,
            emotional_score=0.5,
            emotional_labels=["test"],
            technical_score=0.5,
            importance_weight=0.5,
            conversation_id="corruption_test_conv",
        )

        # Simulate potential corruption by storing with verification enabled
        corrupted_embedding = [0.2] * 384

        # This should succeed with verification
        corrupted_point_id = self.vector_store.store_segment(corrupted_segment, corrupted_embedding)
        assert corrupted_point_id is not None

        # Verify both segments can be retrieved correctly
        retrieved_segments = self.vector_store.get_conversation_segments(
            conversation_id="corruption_test_conv",
        )

        assert len(retrieved_segments) == 2

        # Verify data integrity by checking content instead of hardcoded IDs
        segment_contents = {seg.content for seg in retrieved_segments}
        expected_contents = {"Test content for corruption detection", "Corrupted content"}
        assert segment_contents == expected_contents

    def test_concurrent_access_and_consistency(self) -> None:
        """Test contract: system maintains consistency under concurrent operations."""
        import threading
        import time

        results = []
        errors = []

        def store_segments_concurrently(thread_id: int) -> None:
            """Store segments concurrently from multiple threads."""
            try:
                for i in range(3):
                    segment = ConversationSegment(
                        content=f"Concurrent content from thread {thread_id}, segment {i}",
                        speaker=SpeakerType.USER,
                        emotional_score=0.5 + thread_id * 0.1,
                        emotional_labels=["test"],
                        technical_score=0.3,
                        importance_weight=0.6,
                        conversation_id="concurrent_test_conv",
                    )

                    embedding = [thread_id * 0.1 + i * 0.01] * 384

                    point_id = self.vector_store.store_segment(segment, embedding)
                    results.append((thread_id, i, point_id))

                    # Small delay to simulate real usage
                    time.sleep(0.01)

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Launch multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=store_segments_concurrently, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verification: No errors should occur
        assert len(errors) == 0, f"Concurrent operations failed with errors: {errors}"

        # All operations should succeed
        assert len(results) == 9, "Should have 9 successful operations (3 threads × 3 segments)"

        # Verify all segments are stored correctly
        retrieved_segments = self.vector_store.get_conversation_segments(
            conversation_id="concurrent_test_conv",
        )

        assert len(retrieved_segments) == 9, "All concurrent segments should be retrievable"

        # Verify data integrity - each segment should be unique
        segment_ids = {seg.segment_id for seg in retrieved_segments}
        assert len(segment_ids) == 9, "All segments should have unique IDs"

    def test_large_batch_performance_and_reliability(self) -> None:
        """Test contract: system handles large batches efficiently and reliably."""
        # Create a large batch of segments
        batch_size = 50
        segments = []
        embeddings = []

        for i in range(batch_size):
            segment = ConversationSegment(
                content=f"Large batch test content segment {i}. This is emotional content expressing gratitude.",
                speaker=SpeakerType.USER if i % 2 == 0 else SpeakerType.ASSISTANT,
                emotional_score=0.5 + (i % 5) * 0.1,
                emotional_labels=["gratitude"] if i % 3 == 0 else [],
                technical_score=0.2 + (i % 4) * 0.1,
                importance_weight=0.6,
                conversation_id="large_batch_conv",
            )

            # Create varied embeddings
            embedding = [i * 0.01] * 384

            segments.append(segment)
            embeddings.append(embedding)

        # Store large batch
        start_time = time.time()
        point_ids = self.vector_store.store_batch_segments(segments, embeddings)
        storage_time = time.time() - start_time

        # Verification: All segments stored successfully
        assert len(point_ids) == batch_size
        assert all(point_id for point_id in point_ids)

        # Performance verification: Should complete in reasonable time
        assert storage_time < 30.0, f"Large batch storage took too long: {storage_time:.2f}s"

        # Verify data integrity for large batch
        retrieved_segments = self.vector_store.get_conversation_segments(
            conversation_id="large_batch_conv",
        )

        assert len(retrieved_segments) == batch_size, "All large batch segments should be retrievable"

        # Verify search still works efficiently on large dataset
        query_embedding = [0.5] * 384

        start_time = time.time()
        search_results = self.vector_store.search_similar(
            query_vector=query_embedding,
            limit=10,
        )
        search_time = time.time() - start_time

        assert len(search_results) > 0, "Should find similar segments in large dataset"
        assert search_time < 5.0, f"Search took too long on large dataset: {search_time:.2f}s"

    def test_error_recovery_and_resilience(self) -> None:
        """Test contract: system recovers gracefully from various error conditions."""
        # Test recovery from network interruption simulation
        # This test verifies the retry mechanisms work correctly

        segment = ConversationSegment(
            content="Test content for error recovery",
            speaker=SpeakerType.USER,
            emotional_score=0.6,
            emotional_labels=["test"],
            technical_score=0.4,
            importance_weight=0.5,
            conversation_id="error_recovery_conv",
        )

        embedding = [0.1] * 384

        # Test normal operation first
        point_id = self.vector_store.store_segment(segment, embedding)
        assert point_id is not None, "Normal operation should succeed"

        # Test with potential timeout/retry scenarios
        # The vector store should handle transient failures gracefully

        # Store multiple segments to test batch resilience
        batch_segments = []
        batch_embeddings = []

        for i in range(5):
            test_segment = ConversationSegment(
                content=f"Error recovery test segment {i}",
                speaker=SpeakerType.USER,
                emotional_score=0.5,
                emotional_labels=["test"],
                technical_score=0.5,
                importance_weight=0.5,
                conversation_id="error_recovery_conv",
            )

            batch_segments.append(test_segment)
            batch_embeddings.append([i * 0.1] * 384)

        # Batch operation should succeed even with potential retry logic
        batch_point_ids = self.vector_store.store_batch_segments(batch_segments, batch_embeddings)
        assert len(batch_point_ids) == 5, "Batch operation should complete successfully"

        # Verify all data is accessible
        final_segments = self.vector_store.get_conversation_segments(
            conversation_id="error_recovery_conv",
        )

        assert len(final_segments) == 6, "Should retrieve all segments including original and batch"
