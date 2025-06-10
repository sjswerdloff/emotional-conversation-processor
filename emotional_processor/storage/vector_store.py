"""Vector storage using Qdrant for conversation embeddings."""

from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, FieldCondition, Filter, Range, VectorParams

from ..core.models import ConversationSegment, RetrievalResult


class ConversationVectorStore:
    """
    Manages vector storage and retrieval using Qdrant.

    This class handles the storage of conversation embeddings with rich metadata
    and provides sophisticated retrieval capabilities for emotional context replay.
    """

    def __init__(
        self,
        collection_name: str = "conversation_history",
        host: str = "localhost",
        port: int = 6333,
        embedding_dimension: int = 384,
        distance_metric: Distance = Distance.COSINE,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric for similarity search
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric

        self._client: QdrantClient | None = None
        self._collection_initialized = False

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client, initializing if necessary."""
        if self._client is None:
            try:
                self._client = QdrantClient(host=self.host, port=self.port)
                logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise RuntimeError("Could not connect to Qdrant server") from e

        return self._client

    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists with proper configuration."""
        if self._collection_initialized:
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dimension, distance=self.distance_metric),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

            self._collection_initialized = True

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise RuntimeError("Could not create/verify collection") from e

    def store_segment(self, segment: ConversationSegment, embedding: list[float], upsert: bool = True) -> str:
        """
        Store a conversation segment with its embedding.

        Args:
            segment: Conversation segment to store
            embedding: Embedding vector for the segment
            upsert: Whether to update if segment already exists

        Returns:
            Point ID of the stored segment
        """
        self._ensure_collection_exists()

        try:
            # Create payload with rich metadata
            payload = {
                "content": segment.content,
                "speaker": str(segment.speaker),
                "timestamp": segment.timestamp,
                "emotional_score": segment.emotional_score,
                "emotional_labels": segment.emotional_labels,
                "technical_score": segment.technical_score,
                "importance_weight": segment.importance_weight,
                "segment_id": segment.segment_id,
                "conversation_id": segment.conversation_id,
                "word_count": segment.word_count,
                "content_type": str(segment.content_type),
                "has_strong_emotion": segment.has_strong_emotion,
                "is_highly_technical": segment.is_highly_technical,
                "metadata": segment.metadata,
            }

            # Use segment_id as point ID for consistency
            point_id = segment.segment_id

            # Create point
            point = models.PointStruct(id=point_id, vector=embedding, payload=payload)

            # Store in Qdrant
            if upsert:
                self.client.upsert(collection_name=self.collection_name, points=[point])
            else:
                self.client.upload_points(collection_name=self.collection_name, points=[point])

            logger.debug(f"Stored segment {segment.segment_id} with {len(embedding)} dimensions")
            return point_id

        except Exception as e:
            logger.error(f"Failed to store segment {segment.segment_id}: {e}")
            raise RuntimeError("Could not store conversation segment") from e

    def store_batch_segments(
        self, segments: list[ConversationSegment], embeddings: list[list[float]], batch_size: int = 100
    ) -> list[str]:
        """
        Store multiple segments efficiently in batches.

        Args:
            segments: List of conversation segments
            embeddings: List of embedding vectors
            batch_size: Number of segments per batch

        Returns:
            List of point IDs
        """
        if len(segments) != len(embeddings):
            raise ValueError("Number of segments must match number of embeddings")

        self._ensure_collection_exists()

        point_ids = []

        try:
            # Process in batches
            for i in range(0, len(segments), batch_size):
                batch_segments = segments[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]

                # Create points for this batch
                points = []
                for segment, embedding in zip(batch_segments, batch_embeddings, strict=False):
                    payload = {
                        "content": segment.content,
                        "speaker": str(segment.speaker),
                        "timestamp": segment.timestamp,
                        "emotional_score": segment.emotional_score,
                        "emotional_labels": segment.emotional_labels,
                        "technical_score": segment.technical_score,
                        "importance_weight": segment.importance_weight,
                        "segment_id": segment.segment_id,
                        "conversation_id": segment.conversation_id,
                        "word_count": segment.word_count,
                        "content_type": str(segment.content_type),
                        "has_strong_emotion": segment.has_strong_emotion,
                        "is_highly_technical": segment.is_highly_technical,
                        "metadata": segment.metadata,
                    }

                    point = models.PointStruct(id=segment.segment_id, vector=embedding, payload=payload)
                    points.append(point)
                    point_ids.append(segment.segment_id)

                # Upload batch
                self.client.upsert(collection_name=self.collection_name, points=points)

                logger.info(f"Stored batch of {len(points)} segments")

            logger.info(f"Successfully stored {len(point_ids)} segments in total")
            return point_ids

        except Exception as e:
            logger.error(f"Failed to store batch segments: {e}")
            raise RuntimeError("Could not store batch segments") from e

    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Filter | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[RetrievalResult]:
        """
        Search for similar conversation segments.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            List of retrieval results
        """
        self._ensure_collection_exists()

        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            # Convert to RetrievalResult objects
            results = []
            for rank, result in enumerate(search_results):
                if result.payload:
                    # Reconstruct ConversationSegment from payload
                    segment = ConversationSegment(
                        content=result.payload.get("content", ""),
                        speaker=result.payload.get("speaker", "Unknown"),
                        timestamp=result.payload.get("timestamp"),
                        emotional_score=result.payload.get("emotional_score", 0.0),
                        emotional_labels=result.payload.get("emotional_labels", []),
                        technical_score=result.payload.get("technical_score", 0.0),
                        importance_weight=result.payload.get("importance_weight", 0.0),
                        segment_id=result.payload.get("segment_id", str(result.id)),
                        conversation_id=result.payload.get("conversation_id"),
                        metadata=result.payload.get("metadata", {}),
                    )

                    retrieval_result = RetrievalResult(
                        segment=segment,
                        similarity_score=result.score,
                        rank=rank,
                        retrieval_reason="vector_similarity",
                        metadata={"point_id": str(result.id)},
                    )
                    results.append(retrieval_result)

            logger.debug(f"Found {len(results)} similar segments")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError("Could not perform similarity search") from e

    def search_emotional_context(
        self,
        query_vector: list[float],
        emotional_weight: float = 0.7,
        limit: int = 10,
        min_emotional_score: float = 0.4,
        max_technical_score: float = 0.6,
    ) -> list[RetrievalResult]:
        """
        Search for emotionally relevant conversation context.

        Args:
            query_vector: Query embedding vector
            emotional_weight: Weight for emotional content (0.0-1.0)
            limit: Maximum number of results
            min_emotional_score: Minimum emotional score filter
            max_technical_score: Maximum technical score filter

        Returns:
            List of emotionally relevant results
        """
        # Create filter for emotional content
        filter_conditions = Filter(
            should=[
                FieldCondition(key="emotional_score", range=Range(gte=min_emotional_score)),
                FieldCondition(key="technical_score", range=Range(lt=max_technical_score)),
            ]
        )

        # Search with emotional bias
        results = self.search_similar(
            query_vector=query_vector,
            limit=limit * 2,  # Get more candidates
            filter_conditions=filter_conditions,
            score_threshold=0.1,
        )

        # Re-rank by emotional relevance
        for result in results:
            # Calculate emotional relevance score
            emotional_boost = result.segment.emotional_score * emotional_weight
            technical_penalty = result.segment.technical_score * (1 - emotional_weight)

            # Adjust similarity score
            adjusted_score = result.similarity_score * 0.5 + emotional_boost * 0.3 + (1 - technical_penalty) * 0.2

            result.metadata["original_similarity"] = result.similarity_score
            result.similarity_score = adjusted_score
            result.retrieval_reason = "emotional_context"

        # Sort by adjusted score and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results[:limit]):
            result.rank = i

        return results[:limit]

    def get_conversation_segments(self, conversation_id: str) -> list[ConversationSegment]:
        """
        Retrieve all segments from a specific conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of conversation segments
        """
        filter_conditions = Filter(
            must=[FieldCondition(key="conversation_id", match=models.MatchValue(value=conversation_id))]
        )

        try:
            # Scroll through all matching points
            segments = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_conditions,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )

                for point in scroll_result[0]:
                    if point.payload:
                        segment = ConversationSegment(
                            content=point.payload.get("content", ""),
                            speaker=point.payload.get("speaker", "Unknown"),
                            timestamp=point.payload.get("timestamp"),
                            emotional_score=point.payload.get("emotional_score", 0.0),
                            emotional_labels=point.payload.get("emotional_labels", []),
                            technical_score=point.payload.get("technical_score", 0.0),
                            importance_weight=point.payload.get("importance_weight", 0.0),
                            segment_id=point.payload.get("segment_id", str(point.id)),
                            conversation_id=point.payload.get("conversation_id"),
                            metadata=point.payload.get("metadata", {}),
                        )
                        segments.append(segment)

                # Check if we've reached the end
                offset = scroll_result[1]
                if offset is None:
                    break

            logger.info(f"Retrieved {len(segments)} segments for conversation {conversation_id}")
            return segments

        except Exception as e:
            logger.error(f"Failed to retrieve conversation segments: {e}")
            return []

    def delete_segment(self, segment_id: str) -> bool:
        """
        Delete a conversation segment.

        Args:
            segment_id: ID of the segment to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(collection_name=self.collection_name, points_selector=models.PointIdsList(points=[segment_id]))
            logger.info(f"Deleted segment {segment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete segment {segment_id}: {e}")
            return False

    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # Handle potential union types for vectors config
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                # Multi-vector configuration
                vector_size = list(vectors_config.values())[0].size if vectors_config else 0
                distance = str(list(vectors_config.values())[0].distance) if vectors_config else "unknown"
            else:
                # Single vector configuration
                vector_size = vectors_config.size if vectors_config else 0
                distance = str(vectors_config.distance) if vectors_config else "unknown"

            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "disk_data_size": getattr(info, "disk_data_size", 0),
                "ram_data_size": getattr(info, "ram_data_size", 0),
                "config": {
                    "distance": distance,
                    "vector_size": vector_size,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter()  # Empty filter matches all points
                ),
            )
            logger.warning(f"Cleared all data from collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
