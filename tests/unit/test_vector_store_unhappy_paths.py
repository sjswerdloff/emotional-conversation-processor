"""
Additional unhappy path tests for ConversationVectorStore.

These tests cover edge cases and error conditions that weren't covered in the main
contract tests, focusing on data validation, boundary conditions, and graceful
degradation under adverse conditions.
"""

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from qdrant_client.http.models import Distance

from emotional_processor.core.models import ConversationSegment, SpeakerType
from emotional_processor.storage.vector_store import ConversationVectorStore


class TestVectorStoreDataValidationUnhappyPaths:
    """Test unhappy paths related to data validation and edge cases."""

    def setup_method(self) -> None:
        """Set up test environment with mocked client."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=False)  # Disable for unit tests
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_store_segment_empty_content_edge_case(self) -> None:
        """Edge case: storing segment with empty content should work."""
        segment = ConversationSegment(
            content="",  # Empty content
            speaker=SpeakerType.USER,
            segment_id="empty_content",
        )
        embedding = [0.0] * 384

        # Should not raise an exception
        point_id = self.store.store_segment(segment, embedding)
        assert point_id == "empty_content"
        self.mock_client.upsert.assert_called_once()

    def test_store_segment_none_values_edge_case(self) -> None:
        """Edge case: storing segment with None values in optional fields."""
        segment = ConversationSegment(
            content="Valid content",
            speaker=SpeakerType.ASSISTANT,
            timestamp=None,  # None timestamp
            conversation_id=None,  # None conversation ID
            metadata={},  # Empty metadata
            segment_id="none_values",
        )
        embedding = [0.1] * 384

        point_id = self.store.store_segment(segment, embedding)
        assert point_id == "none_values"

        # Verify None values are handled properly in payload
        call_args = self.mock_client.upsert.call_args
        stored_point = call_args[1]["points"][0]
        payload = stored_point.payload

        assert payload["timestamp"] is None
        assert payload["conversation_id"] is None
        assert payload["metadata"] == {}

    def test_store_segment_extreme_scores_edge_case(self) -> None:
        """Edge case: storing segment with extreme score values."""
        segment = ConversationSegment(
            content="Test with extreme scores",
            speaker=SpeakerType.USER,
            emotional_score=2.5,  # Will be clamped to 1.0
            technical_score=-0.5,  # Will be clamped to 0.0
            importance_weight=10.0,  # Will be clamped to 1.0
            segment_id="extreme_scores",
        )
        embedding = [0.5] * 384

        point_id = self.store.store_segment(segment, embedding)
        assert point_id == "extreme_scores"

        # Verify scores were normalized by the model
        call_args = self.mock_client.upsert.call_args
        stored_point = call_args[1]["points"][0]
        payload = stored_point.payload

        assert payload["emotional_score"] == 1.0  # Clamped from 2.5
        assert payload["technical_score"] == 0.0  # Clamped from -0.5
        assert payload["importance_weight"] == 1.0  # Clamped from 10.0

    def test_store_segment_wrong_embedding_dimension(self) -> None:
        """Unhappy path: storing segment with incorrect embedding dimension."""
        segment = ConversationSegment(
            content="Test content",
            speaker=SpeakerType.USER,
            segment_id="wrong_dim",
        )

        # Wrong dimension - expect 384, provide 256
        wrong_embedding = [0.1] * 256

        # The vector store doesn't validate dimension - Qdrant would catch this
        # But the store should pass it through without crashing
        point_id = self.store.store_segment(segment, wrong_embedding)
        assert point_id == "wrong_dim"

        # Verify the wrong embedding was passed through
        call_args = self.mock_client.upsert.call_args
        stored_point = call_args[1]["points"][0]
        assert len(stored_point.vector) == 256

    def test_store_batch_segments_empty_list_edge_case(self) -> None:
        """Edge case: storing empty lists should work gracefully."""
        empty_segments: list[ConversationSegment] = []
        empty_embeddings: list[list[float]] = []

        # Should not crash with empty lists
        point_ids = self.store.store_batch_segments(empty_segments, empty_embeddings)

        assert point_ids == []
        self.mock_client.upsert.assert_not_called()

    def test_store_batch_segments_large_batch_performance(self) -> None:
        """Edge case: very large batch should be handled reliably with individual storage."""
        # Create a large number of segments
        large_segments = [
            ConversationSegment(
                content=f"Large batch content {i}",
                speaker=SpeakerType.USER if i % 2 else SpeakerType.ASSISTANT,
                segment_id=f"large_{i}",
            )
            for i in range(1000)
        ]
        large_embeddings = [[0.001 * i] * 384 for i in range(1000)]

        # Track batch calls
        batch_calls = []

        def track_batches(collection_name: str, points: list[Any]) -> None:
            batch_calls.append(len(points))

        self.mock_client.upsert.side_effect = track_batches

        # Store with small batch size to test batching
        point_ids = self.store.store_batch_segments(large_segments, large_embeddings, batch_size=100)

        # Verify individual storage worked correctly (with verification disabled)
        assert len(point_ids) == 1000
        assert len(batch_calls) == 1000  # Each segment stored individually for reliability
        assert all(call_size == 1 for call_size in batch_calls)  # Each call stores 1 segment

    def test_search_similar_invalid_query_vector_dimensions(self) -> None:
        """Unhappy path: search with wrong query vector dimensions."""
        # Wrong dimension query vector
        wrong_query_vector = [0.1] * 256  # Should be 384

        # Configure mock to simulate Qdrant dimension error
        self.mock_client.search.side_effect = Exception("Vector dimension mismatch")

        # Should raise RuntimeError with proper error wrapping
        with pytest.raises(RuntimeError, match="Could not perform similarity search"):
            self.store.search_similar(wrong_query_vector)

    def test_search_similar_empty_query_vector(self) -> None:
        """Edge case: search with empty query vector."""
        empty_query_vector: list[float] = []

        # Configure mock to simulate empty vector error
        self.mock_client.search.side_effect = Exception("Empty query vector")

        with pytest.raises(RuntimeError, match="Could not perform similarity search"):
            self.store.search_similar(empty_query_vector)

    def test_search_similar_zero_limit(self) -> None:
        """Edge case: search with zero limit should return empty results."""
        query_vector = [0.5] * 384

        # Configure mock to return empty results for zero limit
        self.mock_client.search.return_value = []

        results = self.store.search_similar(query_vector, limit=0)
        assert results == []

    def test_search_similar_negative_score_threshold(self) -> None:
        """Edge case: search with negative score threshold should work."""
        query_vector = [0.5] * 384

        # Configure mock to return results
        mock_results = [
            Mock(
                id="seg_1",
                score=-0.1,  # Negative score (unusual but possible)
                payload={
                    "content": "Negative similarity content",
                    "speaker": "User",
                    "emotional_score": 0.5,
                    "emotional_labels": [],
                    "technical_score": 0.5,
                    "importance_weight": 0.5,
                    "segment_id": "seg_1",
                    "conversation_id": "conv_1",
                    "metadata": {},
                },
            )
        ]
        self.mock_client.search.return_value = mock_results

        # Should handle negative scores gracefully
        results = self.store.search_similar(query_vector, score_threshold=-1.0)
        assert len(results) == 1
        assert results[0].similarity_score == -0.1

    def test_get_conversation_segments_nonexistent_conversation(self) -> None:
        """Edge case: retrieving segments for non-existent conversation returns empty list."""
        nonexistent_id = "does_not_exist"

        # Configure mock to return empty results (legitimate empty state, not error)
        self.mock_client.scroll.return_value = ([], None)

        segments = self.store.get_conversation_segments(nonexistent_id)
        assert segments == []

    def test_get_conversation_segments_database_error_fail_fast(self) -> None:
        """Unhappy path: database errors must fail fast, not return empty list."""
        conversation_id = "database_error"

        # Configure mock to fail immediately (database error)
        self.mock_client.scroll.side_effect = Exception("Database connection lost")

        # Should raise RuntimeError with proper error wrapping (fail fast)
        with pytest.raises(RuntimeError, match="Could not retrieve conversation segments for database_error"):
            self.store.get_conversation_segments(conversation_id)

    def test_get_conversation_segments_scroll_interruption(self) -> None:
        """Unhappy path: scroll operation gets interrupted."""
        conversation_id = "interrupted_scroll"

        # Configure mock to fail on second scroll call
        self.mock_client.scroll.side_effect = [
            ([Mock(id="seg_1", payload={"content": "First batch"})], "offset_1"),
            Exception("Network timeout during scroll"),
        ]

        # Should raise RuntimeError on scroll failure (fail fast)
        with pytest.raises(RuntimeError, match="Could not retrieve conversation segments for interrupted_scroll"):
            self.store.get_conversation_segments(conversation_id)

    def test_search_emotional_context_malformed_filter(self) -> None:
        """Unhappy path: emotional context search with filter construction issues."""
        query_vector = [0.5] * 384

        # Simulate filter construction causing Qdrant error
        self.mock_client.search.side_effect = Exception("Invalid filter format")

        with pytest.raises(RuntimeError, match="Could not perform similarity search"):
            self.store.search_emotional_context(
                query_vector=query_vector,
                min_emotional_score=1.5,  # Invalid: > 1.0
                max_technical_score=-0.5,  # Invalid: < 0.0
            )


class TestVectorStoreCorruptedDataUnhappyPaths:
    """Test unhappy paths related to corrupted or inconsistent data."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=False)  # Disable for unit tests
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_search_results_missing_payload(self) -> None:
        """Unhappy path: search results with missing payload data."""
        query_vector = [0.5] * 384

        # Configure mock to return results without payload
        corrupted_results = [
            Mock(id="seg_1", score=0.9, payload=None),  # Missing payload
            Mock(id="seg_2", score=0.8, payload={}),  # Empty payload
        ]
        self.mock_client.search.return_value = corrupted_results

        # Should handle missing payload gracefully
        results = self.store.search_similar(query_vector)

        # Should skip results without payload
        assert len(results) == 0

    def test_search_results_corrupted_payload_data(self) -> None:
        """Unhappy path: search results with corrupted payload data."""
        query_vector = [0.5] * 384

        # Configure mock with corrupted payload data
        corrupted_results = [
            Mock(
                id="seg_corrupted",
                score=0.9,
                payload={
                    "content": "",  # Missing content
                    # Missing speaker
                    "emotional_score": "not_a_number",  # Wrong type
                    "emotional_labels": "not_a_list",  # Wrong type
                    "segment_id": "seg_corrupted",
                },
            )
        ]
        self.mock_client.search.return_value = corrupted_results

        # Should handle corrupted data gracefully with defaults
        results = self.store.search_similar(query_vector)
        assert len(results) == 1

        segment = results[0].segment
        assert segment.content == ""
        assert segment.speaker == SpeakerType.UNKNOWN  # Default for missing speaker
        assert segment.emotional_score == 0.0  # Normalized from invalid type
        assert segment.emotional_labels == []  # Normalized from invalid type

        # Verify corruption tracking for Claude's memory integrity
        assert segment.has_memory_corruption
        assert segment.is_field_corrupted("emotional_score")
        assert segment.is_field_corrupted("emotional_labels")
        assert not segment.emotional_state_known()  # Can't remember emotional state
        assert "emotional_score" in segment.corrupted_fields
        assert "emotional_labels" in segment.corrupted_fields

    def test_memory_corruption_tracking_detailed(self) -> None:
        """Test detailed corruption tracking for Claude's memory integrity."""
        from emotional_processor.core.models import ConversationSegment

        # Test segment with mixed corruption
        segment = ConversationSegment(
            content="Test content",
            speaker="User",
            emotional_score="invalid_score",  # Will be corrupted
            emotional_labels="invalid_labels",  # Will be corrupted
            technical_score=0.8,  # Valid
            segment_id="test_corruption",
        )

        # Verify corruption detection
        assert segment.has_memory_corruption
        assert segment.is_field_corrupted("emotional_score")
        assert segment.is_field_corrupted("emotional_labels")
        assert not segment.is_field_corrupted("technical_score")

        # Verify memory state assessment
        assert not segment.emotional_state_known()  # Both score and labels corrupted
        assert segment.technical_state_known()  # Technical score is valid

        # Verify corruption details
        emotional_reason = segment.get_corruption_reason("emotional_score")
        assert emotional_reason is not None
        assert "conversion failed" in emotional_reason

        labels_reason = segment.get_corruption_reason("emotional_labels")
        assert labels_reason is not None
        assert "invalid type" in labels_reason

        # Verify no corruption reason for valid field
        tech_reason = segment.get_corruption_reason("technical_score")
        assert tech_reason is None

    def test_collection_info_union_types_edge_case(self) -> None:
        """Edge case: collection info with union vector configuration types."""
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Configure mock with dict-based vectors config (multi-vector case)
        mock_info = Mock()
        mock_info.vectors_count = 500
        mock_info.points_count = 500
        mock_info.segments_count = 2

        # Multi-vector configuration (dict instead of single VectorParams)
        mock_vectors_config = {
            "dense": Mock(size=384, distance=Distance.COSINE),
            "sparse": Mock(size=1024, distance=Distance.DOT),
        }
        mock_info.config.params.vectors = mock_vectors_config

        self.mock_client.get_collection.return_value = mock_info

        # Should handle multi-vector configuration
        info = self.store.get_collection_info()

        assert info["name"] == self.store.collection_name
        assert info["vectors_count"] == 500
        assert info["config"]["vector_size"] == 384  # Should use first vector config
        assert "Cosine" in info["config"]["distance"]

    def test_collection_info_missing_attributes(self) -> None:
        """Edge case: collection info with missing optional attributes."""
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Configure mock with missing optional attributes
        mock_info = Mock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.segments_count = 1

        # Missing disk_data_size and ram_data_size attributes
        del mock_info.disk_data_size
        del mock_info.ram_data_size

        mock_vectors_config = Mock()
        mock_vectors_config.size = 384
        mock_vectors_config.distance = Distance.COSINE
        mock_info.config.params.vectors = mock_vectors_config

        self.mock_client.get_collection.return_value = mock_info

        # Should handle missing attributes with defaults
        info = self.store.get_collection_info()

        assert info["disk_data_size"] == 0  # Default value
        assert info["ram_data_size"] == 0  # Default value
        assert info["vectors_count"] == 100
