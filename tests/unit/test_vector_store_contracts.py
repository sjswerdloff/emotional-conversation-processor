"""
Contract tests for ConversationVectorStore - the emotional context preservation system.

These tests validate the critical contracts for storing and retrieving conversation
segments with their emotional metadata. The vector store is responsible for ensuring
that emotional context and relational dynamics are preserved and retrievable.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.models import Distance, FieldCondition, Filter, Range

from emotional_processor.core.models import ConversationSegment, RetrievalResult, SpeakerType
from emotional_processor.storage.vector_store import ConversationVectorStore


class TestVectorStoreStorageContracts:
    """Test storage contracts for preserving conversation segments with emotional context."""

    def setup_method(self) -> None:
        """Set up test environment with mocked Qdrant client."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(
            collection_name="test_collection",
            host="localhost",
            port=6333,
            embedding_dimension=384,
            enable_verification=False,  # Disable for unit tests with behavioral mocks
        )
        # Inject mocked client
        self.store._client = self.mock_client
        self.store._collection_initialized = True  # Skip initialization

    def test_store_segment_contract_metadata_preservation(self) -> None:
        """Contract: store_segment must preserve all emotional and technical metadata."""
        # Create segment with rich emotional context
        segment = ConversationSegment(
            content="I'm deeply grateful for our collaboration",
            speaker=SpeakerType.ASSISTANT,
            timestamp="2024-01-01T10:00:00Z",
            emotional_score=0.9,
            emotional_labels=["gratitude", "joy"],
            technical_score=0.1,
            importance_weight=0.85,
            segment_id="seg_123",
            conversation_id="conv_456",
            metadata={"session": "morning", "context": "reflection"},
        )

        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim embedding

        # Configure mock to track what was stored
        stored_points = []

        def capture_upsert(collection_name: str, points: list[Any]) -> None:
            stored_points.extend(points)

        self.mock_client.upsert.side_effect = capture_upsert

        # Store segment
        point_id = self.store.store_segment(segment, embedding)

        # Contract verification: all metadata preserved
        assert point_id == "seg_123"
        self.mock_client.upsert.assert_called_once()

        # Verify stored data preserves emotional context
        assert len(stored_points) == 1
        stored_point = stored_points[0]

        assert stored_point.id == "seg_123"
        assert stored_point.vector == embedding

        # Verify complete metadata preservation
        payload = stored_point.payload
        assert payload["content"] == segment.content
        assert payload["speaker"] == "SpeakerType.ASSISTANT"  # Speaker enum is converted to string representation
        assert payload["timestamp"] == segment.timestamp
        assert payload["emotional_score"] == 0.9
        assert payload["emotional_labels"] == ["gratitude", "joy"]
        assert payload["technical_score"] == 0.1
        assert payload["importance_weight"] == 0.85
        assert payload["segment_id"] == "seg_123"
        assert payload["conversation_id"] == "conv_456"
        assert payload["metadata"] == {"session": "morning", "context": "reflection"}
        assert payload["has_strong_emotion"] is True
        assert payload["is_highly_technical"] is False

    def test_store_segment_contract_upsert_behavior(self) -> None:
        """Contract: store_segment with upsert=True must update existing segments."""
        segment = ConversationSegment(
            content="Updated content",
            speaker=SpeakerType.USER,
            emotional_score=0.5,
            emotional_labels=["neutral"],
            technical_score=0.5,
            segment_id="existing_seg",
        )

        embedding = [0.5] * 384

        # Test upsert=True (default)
        self.store.store_segment(segment, embedding, upsert=True)
        self.mock_client.upsert.assert_called_once()
        self.mock_client.upload_points.assert_not_called()

        # Reset mocks
        self.mock_client.reset_mock()

        # Test upsert=False
        self.store.store_segment(segment, embedding, upsert=False)
        self.mock_client.upload_points.assert_called_once()
        self.mock_client.upsert.assert_not_called()

    def test_store_batch_segments_contract_reliability(self) -> None:
        """Contract: batch storage must reliably store all segments."""
        # Create diverse segments with varying emotional content
        segments = [
            ConversationSegment(
                content=f"Segment {i}",
                speaker=SpeakerType.USER if i % 2 else SpeakerType.ASSISTANT,
                emotional_score=0.1 * i,
                emotional_labels=["joy"] if i > 5 else [],
                technical_score=0.9 - 0.1 * i,
                importance_weight=0.1 * i,
                segment_id=f"seg_{i}",
                conversation_id="conv_batch",
            )
            for i in range(10)
        ]

        embeddings = [[i * 0.1] * 384 for i in range(10)]

        # Configure mock to track storage calls
        upserted_calls = []

        def track_upsert(collection_name: str, points: list[Any]) -> None:
            upserted_calls.append(len(points))

        self.mock_client.upsert.side_effect = track_upsert

        # Store batch with custom batch size
        point_ids = self.store.store_batch_segments(segments, embeddings, batch_size=3)

        # Contract verification: all segments stored successfully
        assert len(point_ids) == 10
        assert point_ids == [f"seg_{i}" for i in range(10)]

        # Contract: With verification disabled, individual storage calls are made for reliability
        # Each segment is stored individually to ensure data integrity
        assert len(upserted_calls) == 10
        assert all(call_size == 1 for call_size in upserted_calls)  # Each call stores 1 segment

    def test_store_batch_segments_contract_data_integrity(self) -> None:
        """Contract: batch storage must maintain segment-embedding alignment."""
        segments = [
            ConversationSegment(
                content="Important emotional content",
                speaker=SpeakerType.ASSISTANT,
                emotional_score=0.9,
                segment_id="seg_1",
            ),
            ConversationSegment(
                content="Technical implementation details",
                speaker=SpeakerType.USER,
                technical_score=0.9,
                segment_id="seg_2",
            ),
        ]

        embeddings = [[1.0] * 384, [2.0] * 384]

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Number of segments must match number of embeddings"):
            self.store.store_batch_segments(segments, embeddings[:-1])

        # Configure mock to verify correct pairing
        stored_points = []

        def capture_points(collection_name: str, points: list[Any]) -> None:
            stored_points.extend(points)

        self.mock_client.upsert.side_effect = capture_points

        # Store with matching lengths
        self.store.store_batch_segments(segments, embeddings)

        # Verify segment-embedding pairing preserved
        assert len(stored_points) == 2
        assert stored_points[0].id == "seg_1"
        assert stored_points[0].vector == [1.0] * 384
        assert stored_points[0].payload["emotional_score"] == 0.9

        assert stored_points[1].id == "seg_2"
        assert stored_points[1].vector == [2.0] * 384
        assert stored_points[1].payload["technical_score"] == 0.9

    def test_store_segment_contract_error_handling(self) -> None:
        """Contract: storage errors must be properly wrapped and logged."""
        segment = ConversationSegment(
            content="Test content",
            speaker=SpeakerType.USER,
            segment_id="seg_error",
        )
        embedding = [0.1] * 384

        # Configure mock to fail consistently for retry attempts
        # Need enough failures to exhaust all retry attempts (3 attempts + individual fallback)
        connection_error = Exception("Qdrant connection failed")
        self.mock_client.upsert.side_effect = [connection_error] * 10  # Enough for all retries

        # Verify error handling - expect AIMemoryStorageIntegrityError for AI memory failures
        from emotional_processor.storage.vector_store import AIMemoryStorageIntegrityError

        with pytest.raises(AIMemoryStorageIntegrityError) as exc_info:
            self.store.store_segment(segment, embedding)

        # Verify exception contains proper context
        error = exc_info.value
        assert error.point_id == "seg_error"
        assert error.error_type == "storage_operation_failed"

        # Verify client was called multiple times due to retry logic
        assert self.mock_client.upsert.call_count >= 1


class TestVectorStoreRetrievalContracts:
    """Test retrieval contracts for finding emotionally relevant conversation context."""

    def setup_method(self) -> None:
        """Set up test environment with behavioral mocks."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(
            collection_name="test_collection",
            embedding_dimension=384,
            enable_verification=False,  # Disable for unit tests with behavioral mocks
        )
        self.store._client = self.mock_client
        self.store._collection_initialized = True

    def test_search_similar_contract_result_transformation(self) -> None:
        """Contract: search results must be transformed into RetrievalResult objects."""
        query_vector = [0.5] * 384

        # Configure mock search results
        mock_search_results = [
            Mock(
                id="seg_1",
                score=0.95,
                payload={
                    "content": "I appreciate your thoughtful approach",
                    "speaker": "Assistant",
                    "timestamp": "2024-01-01T10:00:00Z",
                    "emotional_score": 0.8,
                    "emotional_labels": ["gratitude", "respect"],
                    "technical_score": 0.1,
                    "importance_weight": 0.75,
                    "segment_id": "seg_1",
                    "conversation_id": "conv_123",
                    "metadata": {},
                },
            ),
            Mock(
                id="seg_2",
                score=0.87,
                payload={
                    "content": "This collaboration has been meaningful",
                    "speaker": "User",
                    "emotional_score": 0.7,
                    "emotional_labels": ["joy"],
                    "technical_score": 0.2,
                    "importance_weight": 0.6,
                    "segment_id": "seg_2",
                    "conversation_id": "conv_123",
                    "metadata": {"topic": "reflection"},
                },
            ),
        ]

        self.mock_client.search.return_value = mock_search_results

        # Perform search
        results = self.store.search_similar(
            query_vector=query_vector,
            limit=10,
            score_threshold=0.5,
        )

        # Contract verification: proper transformation
        assert len(results) == 2

        # First result
        assert isinstance(results[0], RetrievalResult)
        assert results[0].segment.content == "I appreciate your thoughtful approach"
        assert results[0].segment.emotional_score == 0.8
        assert results[0].segment.emotional_labels == ["gratitude", "respect"]
        assert results[0].similarity_score == 0.95
        assert results[0].rank == 0
        assert results[0].retrieval_reason == "vector_similarity"

        # Second result
        assert results[1].segment.content == "This collaboration has been meaningful"
        assert results[1].similarity_score == 0.87
        assert results[1].rank == 1

    def test_search_similar_contract_filtering(self) -> None:
        """Contract: search must respect filter conditions."""
        query_vector = [0.3] * 384

        # Create filter for high emotional content
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="emotional_score",
                    range=Range(gte=0.7),
                )
            ]
        )

        # Perform filtered search
        self.store.search_similar(
            query_vector=query_vector,
            limit=5,
            score_threshold=0.3,
            filter_conditions=filter_conditions,
            with_payload=True,
            with_vectors=False,
        )

        # Verify filter was passed correctly
        self.mock_client.search.assert_called_once()
        call_args = self.mock_client.search.call_args

        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["query_vector"] == query_vector
        assert call_args.kwargs["query_filter"] == filter_conditions
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["score_threshold"] == 0.3
        assert call_args.kwargs["with_payload"] is True
        assert call_args.kwargs["with_vectors"] is False

    def test_search_emotional_context_contract_reranking(self) -> None:
        """Contract: emotional context search must rerank by emotional relevance."""
        query_vector = [0.7] * 384

        # Configure mock results with varying emotional/technical scores
        mock_results = [
            Mock(
                id="seg_technical",
                score=0.9,  # High similarity but technical
                payload={
                    "content": "Here's the implementation",
                    "emotional_score": 0.2,
                    "technical_score": 0.8,
                    "emotional_labels": [],
                    "segment_id": "seg_technical",
                    "speaker": "AI",
                    "importance_weight": 0.2,
                },
            ),
            Mock(
                id="seg_emotional",
                score=0.7,  # Lower similarity but emotional
                payload={
                    "content": "I'm grateful for your patience",
                    "emotional_score": 0.9,
                    "technical_score": 0.1,
                    "emotional_labels": ["gratitude"],
                    "segment_id": "seg_emotional",
                    "speaker": "AI",
                    "importance_weight": 0.85,
                },
            ),
        ]

        self.mock_client.search.return_value = mock_results

        # Search with emotional bias
        results = self.store.search_emotional_context(
            query_vector=query_vector,
            emotional_weight=0.7,
            limit=2,
            min_emotional_score=0.1,
            max_technical_score=0.9,
        )

        # Contract verification: emotional content ranked higher
        assert len(results) == 2

        # Emotional segment should rank first despite lower similarity
        assert results[0].segment.segment_id == "seg_emotional"
        assert results[0].rank == 0
        assert results[0].retrieval_reason == "emotional_context"
        assert "original_similarity" in results[0].metadata

        # Technical segment ranked second
        assert results[1].segment.segment_id == "seg_technical"
        assert results[1].rank == 1

    def test_search_emotional_context_contract_filtering_logic(self) -> None:
        """Contract: emotional search must apply appropriate filters."""
        query_vector = [0.4] * 384

        # Track the filter used
        used_filter = None

        def capture_search(**kwargs: Any) -> list[Any]:
            nonlocal used_filter
            used_filter = kwargs.get("query_filter")
            return []

        self.mock_client.search.side_effect = capture_search

        # Perform emotional context search
        self.store.search_emotional_context(
            query_vector=query_vector,
            min_emotional_score=0.4,
            max_technical_score=0.6,
        )

        # Verify filter construction
        assert used_filter is not None
        assert isinstance(used_filter, Filter)
        assert len(used_filter.should) == 2

        # Check emotional score filter
        emotional_filter = used_filter.should[0]
        assert emotional_filter.key == "emotional_score"
        assert emotional_filter.range.gte == 0.4

        # Check technical score filter
        technical_filter = used_filter.should[1]
        assert technical_filter.key == "technical_score"
        assert technical_filter.range.lt == 0.6

    def test_get_conversation_segments_contract_complete_retrieval(self) -> None:
        """Contract: must retrieve all segments for a conversation."""
        conversation_id = "conv_test"

        # Configure mock to simulate pagination
        first_batch = [
            Mock(
                id=f"seg_{i}",
                payload={
                    "content": f"Content {i}",
                    "speaker": "User" if i % 2 else "Assistant",
                    "emotional_score": 0.5,
                    "emotional_labels": [],
                    "technical_score": 0.3,
                    "importance_weight": 0.4,
                    "segment_id": f"seg_{i}",
                    "conversation_id": conversation_id,
                    "metadata": {},
                },
            )
            for i in range(100)
        ]

        second_batch = [
            Mock(
                id=f"seg_{i}",
                payload={
                    "content": f"Content {i}",
                    "speaker": "AI",
                    "emotional_score": 0.6,
                    "emotional_labels": ["joy"],
                    "technical_score": 0.2,
                    "importance_weight": 0.5,
                    "segment_id": f"seg_{i}",
                    "conversation_id": conversation_id,
                    "metadata": {},
                },
            )
            for i in range(100, 120)
        ]

        # Simulate pagination: first call returns 100 items with offset, second returns 20 with None
        self.mock_client.scroll.side_effect = [
            (first_batch, "offset_token"),
            (second_batch, None),
        ]

        # Retrieve all segments
        segments = self.store.get_conversation_segments(conversation_id)

        # Contract verification: all segments retrieved
        assert len(segments) == 120
        assert all(isinstance(seg, ConversationSegment) for seg in segments)

        # Verify pagination was used correctly
        assert self.mock_client.scroll.call_count == 2

        # First call should have no offset
        first_call = self.mock_client.scroll.call_args_list[0]
        assert first_call.kwargs["offset"] is None
        assert first_call.kwargs["limit"] == 100

        # Second call should use the offset
        second_call = self.mock_client.scroll.call_args_list[1]
        assert second_call.kwargs["offset"] == "offset_token"

    def test_search_similar_contract_error_resilience(self) -> None:
        """Contract: search errors must be handled gracefully."""
        query_vector = [0.1] * 384

        # Configure mock to fail
        self.mock_client.search.side_effect = Exception("Search index corrupted")

        # Verify error handling
        with pytest.raises(RuntimeError, match="Could not perform similarity search"):
            self.store.search_similar(query_vector)


class TestVectorStoreCollectionManagementContracts:
    """Test collection management contracts for vector store initialization and maintenance."""

    def setup_method(self) -> None:
        """Set up test environment with collection management focus."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(
            collection_name="test_emotional_context",
            embedding_dimension=384,
            distance_metric=Distance.COSINE,
            enable_verification=False,  # Disable for unit tests with behavioral mocks
        )

    def test_client_initialization_contract_lazy_connection(self) -> None:
        """Contract: client connection must be lazy and cached."""
        # Initial state: no client
        assert self.store._client is None

        # Mock QdrantClient creation
        with patch("emotional_processor.storage.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = self.mock_client

            # First access creates client
            client1 = self.store.client
            assert client1 == self.mock_client
            mock_client_class.assert_called_once_with(host="localhost", port=6333)

            # Second access returns cached client
            client2 = self.store.client
            assert client2 == client1
            assert mock_client_class.call_count == 1  # Not called again

    def test_client_initialization_contract_connection_error_handling(self) -> None:
        """Contract: connection errors must be wrapped with meaningful messages."""
        with patch("emotional_processor.storage.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection refused")

            with pytest.raises(RuntimeError, match="Could not connect to Qdrant server"):
                _ = self.store.client

    def test_ensure_collection_exists_contract_creation(self) -> None:
        """Contract: collection must be created if it doesn't exist."""
        # Inject mock client
        self.store._client = self.mock_client

        # Configure mock: collection doesn't exist
        mock_collections = Mock()
        mock_collections.collections = []
        self.mock_client.get_collections.return_value = mock_collections

        # Ensure collection exists
        self.store._ensure_collection_exists()

        # Verify collection was created
        self.mock_client.create_collection.assert_called_once()
        call_args = self.mock_client.create_collection.call_args

        assert call_args.kwargs["collection_name"] == "test_emotional_context"

        # Verify vector configuration
        vectors_config = call_args.kwargs["vectors_config"]
        assert isinstance(vectors_config, models.VectorParams)
        assert vectors_config.size == 384
        assert vectors_config.distance == Distance.COSINE

        # Verify initialization flag set
        assert self.store._collection_initialized is True

    def test_ensure_collection_exists_contract_already_exists(self) -> None:
        """Contract: existing collections must not be recreated."""
        # Inject mock client
        self.store._client = self.mock_client

        # Configure mock: collection exists
        existing_collection = Mock()
        existing_collection.name = "test_emotional_context"

        mock_collections = Mock()
        mock_collections.collections = [existing_collection]
        self.mock_client.get_collections.return_value = mock_collections

        # Ensure collection exists
        self.store._ensure_collection_exists()

        # Verify collection was NOT created
        self.mock_client.create_collection.assert_not_called()

        # Verify initialization flag still set
        assert self.store._collection_initialized is True

    def test_ensure_collection_exists_contract_idempotency(self) -> None:
        """Contract: ensure_collection_exists must be idempotent."""
        # Inject mock client
        self.store._client = self.mock_client

        # First call
        self.store._collection_initialized = True
        self.store._ensure_collection_exists()

        # Should not check collections when already initialized
        self.mock_client.get_collections.assert_not_called()

    def test_delete_segment_contract_point_removal(self) -> None:
        """Contract: delete_segment must remove the specified point."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        segment_id = "seg_to_delete"

        # Delete segment
        success = self.store.delete_segment(segment_id)

        # Verify deletion
        assert success is True
        self.mock_client.delete.assert_called_once()

        call_args = self.mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "test_emotional_context"

        # Verify correct point selector
        points_selector = call_args.kwargs["points_selector"]
        assert isinstance(points_selector, models.PointIdsList)
        assert points_selector.points == [segment_id]

    def test_delete_segment_contract_error_handling(self) -> None:
        """Contract: deletion errors must be logged and return False."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Configure mock to fail
        self.mock_client.delete.side_effect = Exception("Point not found")

        # Delete should not raise but return False
        success = self.store.delete_segment("seg_missing")
        assert success is False

    def test_clear_collection_contract_complete_removal(self) -> None:
        """Contract: clear_collection must remove all points."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Clear collection
        success = self.store.clear_collection()

        # Verify all points deleted
        assert success is True
        self.mock_client.delete.assert_called_once()

        call_args = self.mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "test_emotional_context"

        # Verify filter selector targets all points
        points_selector = call_args.kwargs["points_selector"]
        assert isinstance(points_selector, models.FilterSelector)
        # Empty filter matches all points
        assert isinstance(points_selector.filter, Filter)

    def test_get_collection_info_contract_statistics(self) -> None:
        """Contract: collection info must provide comprehensive statistics."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Configure mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.segments_count = 4
        mock_info.disk_data_size = 10485760  # 10MB
        mock_info.ram_data_size = 5242880  # 5MB

        # Single vector configuration
        mock_vectors_config = Mock()
        mock_vectors_config.size = 384
        mock_vectors_config.distance = Distance.COSINE

        mock_info.config.params.vectors = mock_vectors_config

        self.mock_client.get_collection.return_value = mock_info

        # Get collection info
        info = self.store.get_collection_info()

        # Verify comprehensive statistics
        assert info["name"] == "test_emotional_context"
        assert info["vectors_count"] == 1000
        assert info["points_count"] == 1000
        assert info["segments_count"] == 4
        assert info["disk_data_size"] == 10485760
        assert info["ram_data_size"] == 5242880
        assert info["config"]["distance"] == "Cosine"  # Distance enum string representation
        assert info["config"]["vector_size"] == 384

    def test_get_collection_info_contract_error_resilience(self) -> None:
        """Contract: collection info errors must return empty dict."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Configure mock to fail
        self.mock_client.get_collection.side_effect = Exception("Collection not found")

        # Should not raise but return empty dict
        info = self.store.get_collection_info()
        assert info == {}


class TestVectorStoreIntegrationBehavior:
    """Test behavioral contracts that span multiple methods."""

    def setup_method(self) -> None:
        """Set up test environment with behavioral tracking."""
        self.mock_client = MagicMock()
        self.store = ConversationVectorStore(enable_verification=False)  # Disable for unit tests

        # Track stored segments for behavioral verification
        self.stored_segments: dict[str, Any] = {}

        def track_upsert(collection_name: str, points: list[Any]) -> None:
            for point in points:
                self.stored_segments[point.id] = {
                    "vector": point.vector,
                    "payload": point.payload,
                }

        self.mock_client.upsert.side_effect = track_upsert

    def test_store_then_search_behavioral_contract(self) -> None:
        """Contract: stored segments must be searchable with preserved metadata."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        # Store emotional segment
        segment = ConversationSegment(
            content="This project means everything to me",
            speaker=SpeakerType.USER,
            emotional_score=0.9,
            emotional_labels=["passion", "dedication"],
            technical_score=0.1,
            importance_weight=0.85,
            segment_id="emotional_seg",
            conversation_id="conv_123",
        )

        embedding = [0.8] * 384
        self.store.store_segment(segment, embedding)

        # Configure search to return what was stored
        def search_behavior(**kwargs: Any) -> list[Any]:
            # Return stored segment as search result
            stored = self.stored_segments.get("emotional_seg")
            if stored:
                return [
                    Mock(
                        id="emotional_seg",
                        score=0.95,
                        payload=stored["payload"],
                    )
                ]
            return []

        self.mock_client.search.side_effect = search_behavior

        # Search for similar content
        results = self.store.search_similar([0.75] * 384)

        # Verify behavioral contract: stored data is searchable
        assert len(results) == 1
        retrieved = results[0].segment

        assert retrieved.content == segment.content
        assert retrieved.emotional_score == segment.emotional_score
        assert retrieved.emotional_labels == segment.emotional_labels
        assert retrieved.importance_weight == segment.importance_weight

    def test_conversation_lifecycle_behavioral_contract(self) -> None:
        """Contract: support full conversation lifecycle (store, retrieve, delete)."""
        # Inject mock client
        self.store._client = self.mock_client
        self.store._collection_initialized = True

        conversation_id = "conv_lifecycle"

        # Store multiple segments for a conversation
        segments = [
            ConversationSegment(
                content=f"Message {i}",
                speaker=SpeakerType.USER if i % 2 else SpeakerType.ASSISTANT,
                emotional_score=0.5 + i * 0.1,
                segment_id=f"seg_{i}",
                conversation_id=conversation_id,
            )
            for i in range(3)
        ]

        embeddings = [[i * 0.1] * 384 for i in range(3)]
        self.store.store_batch_segments(segments, embeddings)

        # Configure scroll to return stored segments
        def scroll_behavior(**kwargs: Any) -> tuple[list[Any], Any]:
            filter_cond = kwargs.get("scroll_filter")
            if filter_cond and filter_cond.must:
                # Return segments matching conversation_id
                conv_segments = []
                for seg_id, data in self.stored_segments.items():
                    if data["payload"].get("conversation_id") == conversation_id:
                        conv_segments.append(Mock(id=seg_id, payload=data["payload"]))
                return (conv_segments, None)
            return ([], None)

        self.mock_client.scroll.side_effect = scroll_behavior

        # Retrieve conversation segments
        retrieved = self.store.get_conversation_segments(conversation_id)

        # Verify all segments retrieved
        assert len(retrieved) == 3
        assert all(seg.conversation_id == conversation_id for seg in retrieved)

        # Delete one segment
        self.mock_client.delete.return_value = None
        success = self.store.delete_segment("seg_1")
        assert success is True

        # Simulate deletion in our tracking
        if "seg_1" in self.stored_segments:
            del self.stored_segments["seg_1"]

        # Retrieve again - should have one less
        retrieved_after_delete = self.store.get_conversation_segments(conversation_id)
        assert len(retrieved_after_delete) == 2
        assert not any(seg.segment_id == "seg_1" for seg in retrieved_after_delete)
