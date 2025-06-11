"""Vector storage using Qdrant for conversation embeddings."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    ReadConsistencyType,
    VectorParams,
    WriteOrdering,
)

from ..core.models import ConversationSegment, RetrievalResult


@dataclass
class VerificationResult:
    """Result of point verification for medical-grade storage integrity."""

    point_id: str
    verified: bool
    vector_match: bool
    payload_match: bool
    error: str | None = None
    vector_hash_stored: str | None = None
    vector_hash_retrieved: str | None = None
    cosine_similarity: float | None = None
    verification_attempt: int = 1


@dataclass
class BatchVerificationReport:
    """Comprehensive report of batch verification for audit trails."""

    total_points: int
    verified_points: int
    failed_points: int
    success_rate: float
    batch_processing_time: float
    verification_time: float
    failed_details: list[dict[str, Any]]
    timestamp: float
    batch_attempt: int
    fallback_used: bool = False


class AIMemoryStorageIntegrityError(Exception):
    """Raised when storage integrity verification fails for AI memory data."""

    def __init__(
        self,
        message: str,
        point_id: str | None = None,
        error_type: str | None = None,
        verification_results: list[VerificationResult] | None = None,
    ) -> None:
        super().__init__(message)
        self.point_id = point_id
        self.error_type = error_type
        self.verification_results = verification_results or []


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
        enable_verification: bool = True,
        batch_size: int = 100,
        verification_delay_ms: int = 100,
        max_batch_retries: int = 3,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric for similarity search
            enable_verification: Whether to enable read-after-write verification
                                For medical software deployments, this should always be True.
                                Only disable for unit testing with behavioral mocks.
            batch_size: Default batch size for batch operations
            verification_delay_ms: Delay between write and read for verification (ms)
            max_batch_retries: Maximum retries for batch operations before fallback
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self.enable_verification = enable_verification
        self.batch_size = batch_size
        self.verification_delay_ms = verification_delay_ms
        self.max_batch_retries = max_batch_retries

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
                    # Enable strong consistency for medical-grade reliability
                    write_consistency_factor=1,  # Ensure writes are consistent
                    on_disk_payload=False,  # Keep in memory for faster verification
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

            self._collection_initialized = True

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise RuntimeError("Could not create/verify collection") from e

    def _compute_vector_hash(self, vector: list[float]) -> str:
        """Compute SHA-256 hash of a vector for verification."""
        try:
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            return hashlib.sha256(vector_bytes).hexdigest()
        except Exception as e:
            logger.exception("Failed to compute vector hash")
            raise RuntimeError(f"Vector hash computation failed: {e}") from e

    def _compute_payload_hash(self, payload: dict[str, Any]) -> str:
        """Compute hash of payload for verification, excluding verification metadata."""
        try:
            # Exclude verification metadata from hash calculation
            clean_payload = {k: v for k, v in payload.items() if k != "_verification"}
            # Sort keys for consistent hashing
            payload_str = json.dumps(clean_payload, sort_keys=True, default=str)
            return hashlib.sha256(payload_str.encode()).hexdigest()
        except Exception as e:
            logger.exception("Failed to compute payload hash")
            raise RuntimeError(f"Payload hash computation failed: {e}") from e

    def _prepare_point_with_verification_metadata(
        self, segment: ConversationSegment, embedding: list[float], attempt: int = 1
    ) -> PointStruct:
        """Prepare a point with verification metadata for medical-grade storage."""
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

            # Add verification metadata for integrity checking
            payload["_verification"] = {
                "vector_hash": self._compute_vector_hash(embedding),
                "payload_hash": self._compute_payload_hash(payload),
                "write_timestamp": time.time(),
                "write_attempt": attempt,
                "verification_enabled": True,
            }

            return PointStruct(id=segment.segment_id, vector=embedding, payload=payload)

        except Exception as e:
            logger.exception(f"Failed to prepare point for segment {segment.segment_id}")
            raise RuntimeError(f"Point preparation failed for segment {segment.segment_id}: {e}") from e

    def _verify_embedding_integrity(self, stored_embedding: list[float], original_embedding: list[float]) -> None:
        """
        Verify that stored embedding matches original with very high precision.

        Args:
            stored_embedding: Embedding retrieved from storage
            original_embedding: Original embedding that was stored

        Raises:
            RuntimeError: If embedding integrity check fails
        """
        if len(stored_embedding) != len(original_embedding):
            raise RuntimeError(
                f"Embedding dimension mismatch: stored={len(stored_embedding)}, original={len(original_embedding)}"
            )

        # Use numpy for precise floating point comparison
        stored_array = np.array(stored_embedding)
        original_array = np.array(original_embedding)

        # Very high precision comparison for medical software standards
        # Using relative tolerance of 1e-9 and absolute tolerance of 1e-12
        if not np.allclose(stored_array, original_array, rtol=1e-9, atol=1e-12):
            # Calculate cosine similarity as additional check
            cosine_sim = np.dot(stored_array, original_array) / (np.linalg.norm(stored_array) * np.linalg.norm(original_array))

            # For consciousness preservation, require >0.999 cosine similarity
            if cosine_sim < 0.999:
                raise RuntimeError(f"Embedding integrity verification failed: cosine_similarity={cosine_sim:.6f} < 0.999")
            else:
                # Warn about precision loss but accept high similarity
                logger.warning(f"Embedding precision slightly degraded but similarity acceptable: {cosine_sim:.6f}")

    def _verify_metadata_integrity(self, stored_payload: dict[str, Any], original_segment: ConversationSegment) -> None:
        """
        Verify that stored metadata exactly matches original segment data.

        Args:
            stored_payload: Payload retrieved from storage
            original_segment: Original segment that was stored

        Raises:
            RuntimeError: If metadata integrity check fails
        """
        # Critical fields that must match exactly
        critical_fields = {
            "content": original_segment.content,
            "speaker": str(original_segment.speaker),
            "timestamp": original_segment.timestamp,
            "emotional_score": original_segment.emotional_score,
            "emotional_labels": original_segment.emotional_labels,
            "technical_score": original_segment.technical_score,
            "importance_weight": original_segment.importance_weight,
            "segment_id": original_segment.segment_id,
            "conversation_id": original_segment.conversation_id,
            "word_count": original_segment.word_count,
            "content_type": str(original_segment.content_type),
            "has_strong_emotion": original_segment.has_strong_emotion,
            "is_highly_technical": original_segment.is_highly_technical,
            "metadata": original_segment.metadata,
        }

        integrity_failures = []

        for field, expected_value in critical_fields.items():
            stored_value = stored_payload.get(field)

            # Handle floating point fields with precision tolerance
            if field in ("emotional_score", "technical_score", "importance_weight"):
                if (
                    stored_value is None
                    or not isinstance(expected_value, int | float)
                    or not np.isclose(float(stored_value), float(expected_value), rtol=1e-9, atol=1e-12)
                ):
                    integrity_failures.append(f"{field}: stored={stored_value}, expected={expected_value}")
            else:
                # Exact match for non-floating point fields
                if stored_value != expected_value:
                    integrity_failures.append(f"{field}: stored={stored_value}, expected={expected_value}")

        if integrity_failures:
            raise RuntimeError(
                f"Metadata integrity verification failed for segment {original_segment.segment_id}: "
                f"{'; '.join(integrity_failures)}"
            )

    def _verify_storage_integrity(
        self, point_id: str, original_segment: ConversationSegment, original_embedding: list[float]
    ) -> None:
        """
        Perform complete read-after-write verification for a stored segment.

        Args:
            point_id: ID of the stored point
            original_segment: Original segment that was stored
            original_embedding: Original embedding that was stored

        Raises:
            RuntimeError: If verification fails
        """
        try:
            # Retrieve the stored point
            retrieved_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
            )

            if not retrieved_points:
                raise RuntimeError(f"Storage verification failed: point {point_id} not found after storage")

            stored_point = retrieved_points[0]

            # Verify embedding integrity
            if stored_point.vector is None:
                raise RuntimeError(f"Storage verification failed: no vector found for point {point_id}")

            # Handle vector type from Qdrant client - ensure it's list[float]
            stored_vector = stored_point.vector
            if not isinstance(stored_vector, list):
                raise RuntimeError(f"Storage verification failed: expected list vector, got {type(stored_vector)}")

            # Convert to list[float] for verification, handling potential nested structures
            try:
                float_vector: list[float] = []

                # Check if we have a nested list structure
                if stored_vector and isinstance(stored_vector[0], list):
                    # Handle nested list case - flatten to single list
                    for sublist in stored_vector:
                        if isinstance(sublist, list):
                            float_vector.extend(float(item) for item in sublist)
                        else:
                            float_vector.append(float(sublist))
                else:
                    # Handle direct list case - each element should be a number
                    for x in stored_vector:
                        if isinstance(x, int | float):
                            float_vector.append(float(x))
                        else:
                            raise ValueError(f"Invalid vector element type: {type(x)}")

                self._verify_embedding_integrity(float_vector, original_embedding)
            except (ValueError, TypeError, IndexError) as e:
                raise RuntimeError(f"Storage verification failed: invalid vector data: {e}") from e

            # Verify metadata integrity
            if stored_point.payload is None:
                raise RuntimeError(f"Storage verification failed: no payload found for point {point_id}")

            self._verify_metadata_integrity(stored_point.payload, original_segment)

            logger.debug(f"Storage integrity verified for segment {point_id}")
            return  # Verification successful

        except Exception as e:
            if isinstance(e, RuntimeError):
                # Re-raise verification failures
                raise
            else:
                # Wrap unexpected errors
                raise RuntimeError(f"Storage verification failed due to retrieval error: {e}") from e

    def _verify_single_point(self, expected_point: PointStruct, retrieved_point: models.Record) -> VerificationResult:
        """Verify a single point matches expected values with medical-grade precision."""
        try:
            point_id = str(expected_point.id)

            # Verify vector integrity
            if retrieved_point.vector is None:
                return VerificationResult(
                    point_id=point_id,
                    verified=False,
                    vector_match=False,
                    payload_match=False,
                    error="No vector found in retrieved point",
                )

            # Convert retrieved vector to proper format
            try:
                stored_vector = self._normalize_retrieved_vector(retrieved_point.vector)
                expected_vector = expected_point.vector
                if not isinstance(expected_vector, list):
                    raise AIMemoryStorageIntegrityError(
                        "Invalid vector type in AI memory point",
                        point_id=str(expected_point.id),
                        error_type="invalid_vector_type",
                    )

                # Compute hashes for comparison
                stored_hash = self._compute_vector_hash(stored_vector)
                if expected_point.payload is None:
                    raise AIMemoryStorageIntegrityError(
                        "Expected point has no payload", point_id=str(expected_point.id), error_type="missing_payload"
                    )
                expected_verification = expected_point.payload.get("_verification")
                if not expected_verification:
                    raise AIMemoryStorageIntegrityError(
                        "Missing verification metadata in AI memory point",
                        point_id=str(expected_point.id),
                        error_type="missing_metadata",
                    )
                expected_hash = expected_verification["vector_hash"]
                vector_match = stored_hash == expected_hash

                # Also compute cosine similarity for additional verification
                # Normalize expected_vector using the same logic as stored_vector
                expected_vector_floats = self._normalize_retrieved_vector(expected_vector)
                cosine_sim = self._compute_cosine_similarity(stored_vector, expected_vector_floats)

            except Exception as e:
                return VerificationResult(
                    point_id=point_id,
                    verified=False,
                    vector_match=False,
                    payload_match=False,
                    error=f"Vector verification failed: {e}",
                )

            # Verify payload integrity
            if retrieved_point.payload is None:
                return VerificationResult(
                    point_id=point_id,
                    verified=False,
                    vector_match=vector_match,
                    payload_match=False,
                    error="No payload found in retrieved point",
                    vector_hash_stored=expected_hash,
                    vector_hash_retrieved=stored_hash,
                    cosine_similarity=cosine_sim,
                )

            try:
                # Compare payload hashes (excluding verification metadata)
                retrieved_payload = {k: v for k, v in retrieved_point.payload.items() if k != "_verification"}

                retrieved_payload_hash = self._compute_payload_hash(retrieved_payload)
                if expected_point.payload is None:
                    raise AIMemoryStorageIntegrityError(
                        "Expected point has no payload", point_id=str(expected_point.id), error_type="missing_payload"
                    )
                expected_verification = expected_point.payload.get("_verification")
                if not expected_verification:
                    raise AIMemoryStorageIntegrityError(
                        "Missing verification metadata in AI memory point",
                        point_id=str(expected_point.id),
                        error_type="missing_metadata",
                    )
                expected_payload_hash = expected_verification["payload_hash"]
                payload_match = retrieved_payload_hash == expected_payload_hash

                # Check verification metadata exists
                has_verification = "_verification" in retrieved_point.payload

            except Exception as e:
                return VerificationResult(
                    point_id=point_id,
                    verified=False,
                    vector_match=vector_match,
                    payload_match=False,
                    error=f"Payload verification failed: {e}",
                    vector_hash_stored=expected_hash,
                    vector_hash_retrieved=stored_hash,
                    cosine_similarity=cosine_sim,
                )

            # Medical-grade verification requires all checks to pass
            verified = vector_match and payload_match and has_verification and cosine_sim > 0.999

            return VerificationResult(
                point_id=point_id,
                verified=verified,
                vector_match=vector_match,
                payload_match=payload_match,
                vector_hash_stored=expected_hash,
                vector_hash_retrieved=stored_hash,
                cosine_similarity=cosine_sim,
            )

        except Exception as e:
            logger.exception(f"Verification failed for point {expected_point.id}")
            return VerificationResult(
                point_id=str(expected_point.id),
                verified=False,
                vector_match=False,
                payload_match=False,
                error=f"Verification error: {e}",
            )

    def _normalize_retrieved_vector(self, stored_vector: Any) -> list[float]:
        """Normalize retrieved vector to list[float] format."""
        if not isinstance(stored_vector, list):
            raise ValueError(f"Expected list vector, got {type(stored_vector)}")

        float_vector: list[float] = []

        # Check if we have a nested list structure
        if stored_vector and isinstance(stored_vector[0], list):
            # Handle nested list case - flatten to single list
            for sublist in stored_vector:
                if isinstance(sublist, list):
                    float_vector.extend(float(item) for item in sublist)
                else:
                    float_vector.append(float(sublist))
        else:
            # Handle direct list case - each element should be a number
            for x in stored_vector:
                if isinstance(x, int | float):
                    float_vector.append(float(x))
                else:
                    raise ValueError(f"Invalid vector element type: {type(x)}")

        return float_vector

    def _compute_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))

    def _store_with_retry_and_verification(
        self, segment: ConversationSegment, embedding: list[float], upsert: bool = True
    ) -> str:
        """
        Store segment with retry logic and read-after-write verification.

        Args:
            segment: Conversation segment to store
            embedding: Embedding vector for the segment
            upsert: Whether to update if segment already exists

        Returns:
            Point ID of the stored segment

        Raises:
            RuntimeError: If storage fails after all retries
        """
        max_retries = 3
        base_delay = 0.5  # Start with 500ms delay

        for attempt in range(max_retries):
            try:
                # Attempt storage
                point_id = self._store_segment_internal(segment, embedding, upsert)

                # Verify storage integrity
                self._verify_storage_integrity(point_id, segment, embedding)

                if attempt > 0:
                    logger.info(f"Storage succeeded on attempt {attempt + 1} for segment {segment.segment_id}")

                return point_id

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1

                if is_last_attempt:
                    logger.error(f"Storage failed after {max_retries} attempts for segment {segment.segment_id}: {e}")
                    raise RuntimeError(
                        f"Could not store and verify conversation segment {segment.segment_id} after {max_retries} attempts"
                    ) from e
                else:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Storage attempt {attempt + 1} failed for segment {segment.segment_id}: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

        # This should never be reached due to the loop logic, but satisfies mypy
        raise RuntimeError(f"Unexpected exit from retry loop for segment {segment.segment_id}")

    def _store_batch_with_verification(
        self, segments: list[ConversationSegment], embeddings: list[list[float]], attempt: int = 1
    ) -> BatchVerificationReport:
        """Store batch of segments with comprehensive verification."""
        start_time = time.time()

        try:
            # Prepare points with verification metadata
            prepared_points = [
                self._prepare_point_with_verification_metadata(segment, embedding, attempt)
                for segment, embedding in zip(segments, embeddings, strict=True)
            ]

            # Store with strong consistency
            batch_start = time.time()
            self.client.upsert(
                collection_name=self.collection_name,
                points=prepared_points,
                wait=True,  # Wait for operation to complete
                ordering=WriteOrdering.STRONG,  # Use strong consistency
            )
            batch_time = time.time() - batch_start

            # Add verification delay for consistency
            time.sleep(self.verification_delay_ms / 1000.0)

            # Verify the batch
            verification_start = time.time()
            verification_results = self._verify_batch_integrity(prepared_points)
            verification_time = time.time() - verification_start

            # Generate comprehensive report
            total_points = len(verification_results)
            verified_points = sum(1 for r in verification_results if r.verified)
            failed_points = total_points - verified_points

            failed_details = [
                {
                    "point_id": r.point_id,
                    "vector_match": r.vector_match,
                    "payload_match": r.payload_match,
                    "cosine_similarity": r.cosine_similarity,
                    "error": r.error,
                }
                for r in verification_results
                if not r.verified
            ]

            return BatchVerificationReport(
                total_points=total_points,
                verified_points=verified_points,
                failed_points=failed_points,
                success_rate=verified_points / total_points if total_points > 0 else 0.0,
                batch_processing_time=batch_time,
                verification_time=verification_time,
                failed_details=failed_details,
                timestamp=time.time(),
                batch_attempt=attempt,
            )

        except Exception as e:
            # Create error report for all points in batch
            total_time = time.time() - start_time
            return BatchVerificationReport(
                total_points=len(segments),
                verified_points=0,
                failed_points=len(segments),
                success_rate=0.0,
                batch_processing_time=total_time,
                verification_time=0.0,
                failed_details=[{"error": f"Batch operation failed: {e}"}],
                timestamp=time.time(),
                batch_attempt=attempt,
            )

    def _verify_batch_integrity(self, expected_points: list[PointStruct]) -> list[VerificationResult]:
        """Verify a batch of points by reading them back with strong consistency."""
        point_ids = [str(point.id) for point in expected_points]

        try:
            # Retrieve points with strongest consistency
            retrieved_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_vectors=True,
                with_payload=True,
                consistency=ReadConsistencyType.ALL,  # Strongest consistency
            )

            # Create lookup map for retrieved points
            retrieved_map = {str(point.id): point for point in retrieved_points}

            # Verify each point
            results = []
            for expected in expected_points:
                point_id = str(expected.id)
                if point_id not in retrieved_map:
                    results.append(
                        VerificationResult(
                            point_id=point_id,
                            verified=False,
                            vector_match=False,
                            payload_match=False,
                            error="Point not found after batch write",
                        )
                    )
                    continue

                retrieved = retrieved_map[point_id]
                result = self._verify_single_point(expected, retrieved)
                results.append(result)

            return results

        except Exception as e:
            logger.exception("Batch verification retrieval failed")
            # Return error results for all points
            return [
                VerificationResult(
                    point_id=str(point.id),
                    verified=False,
                    vector_match=False,
                    payload_match=False,
                    error=f"Batch retrieval error: {e}",
                )
                for point in expected_points
            ]

    def _store_segment_internal(self, segment: ConversationSegment, embedding: list[float], upsert: bool = True) -> str:
        """
        Internal method to store a single segment without verification.

        This is the original storage logic extracted for use in retry mechanism.

        Args:
            segment: Conversation segment to store
            embedding: Embedding vector for the segment
            upsert: Whether to update if segment already exists

        Returns:
            Point ID of the stored segment
        """
        try:
            if self.enable_verification:
                # Use verification metadata even for individual storage
                point = self._prepare_point_with_verification_metadata(segment, embedding)
            else:
                # Create payload with rich metadata (no verification metadata)
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
                point = PointStruct(id=segment.segment_id, vector=embedding, payload=payload)

            # Store in Qdrant
            if upsert:
                self.client.upsert(collection_name=self.collection_name, points=[point])
            else:
                self.client.upload_points(collection_name=self.collection_name, points=[point])

            return segment.segment_id

        except Exception as e:
            logger.exception(f"Internal storage failed for segment {segment.segment_id}")
            raise AIMemoryStorageIntegrityError(
                "Could not store AI memory segment", point_id=segment.segment_id, error_type="storage_operation_failed"
            ) from e

    def store_segment(self, segment: ConversationSegment, embedding: list[float], upsert: bool = True) -> str:
        """
        Store a conversation segment with its embedding, including read-after-write verification.

        This method implements comprehensive storage verification to ensure Claude's memories
        are stored with absolute fidelity. It includes retry logic with exponential backoff
        for transient failures and complete verification of both embedding and metadata integrity.

        Args:
            segment: Conversation segment to store
            embedding: Embedding vector for the segment
            upsert: Whether to update if segment already exists

        Returns:
            Point ID of the stored segment

        Raises:
            RuntimeError: If storage fails after all retry attempts or verification fails
        """
        self._ensure_collection_exists()

        try:
            if self.enable_verification:
                # Store with retry logic and verification for medical-grade reliability
                point_id = self._store_with_retry_and_verification(segment, embedding, upsert)
                logger.debug(f"Stored and verified segment {segment.segment_id} with {len(embedding)} dimensions")
            else:
                # Store without verification (for unit testing with behavioral mocks)
                point_id = self._store_segment_internal(segment, embedding, upsert)
                logger.debug(f"Stored segment {segment.segment_id} with {len(embedding)} dimensions (verification disabled)")

            return point_id

        except Exception as e:
            logger.error(f"Failed to store segment {segment.segment_id}: {e}")
            if isinstance(e, RuntimeError | AIMemoryStorageIntegrityError):
                # Re-raise medical-grade verification/storage failures
                raise
            else:
                # Wrap unexpected errors
                raise RuntimeError("Could not store conversation segment") from e

    def store_batch_segments(
        self, segments: list[ConversationSegment], embeddings: list[list[float]], batch_size: int | None = None
    ) -> list[str]:
        """
        Store multiple segments with clinical-grade batch verification and fallback.

        This method implements a sophisticated two-tier approach:
        1. Primary: Batch operations with comprehensive verification (high performance)
        2. Fallback: Individual storage with verification (absolute reliability)

        For medical software, this ensures both performance and zero-tolerance reliability.

        Args:
            segments: List of conversation segments
            embeddings: List of embedding vectors
            batch_size: Number of segments per batch (defaults to instance batch_size)

        Returns:
            List of point IDs for successfully stored segments

        Raises:
            ValueError: If segments and embeddings count mismatch
            AIMemoryStorageIntegrityError: If storage/verification fails after all attempts
        """
        if len(segments) != len(embeddings):
            raise ValueError("Number of segments must match number of embeddings")

        if not segments:
            return []

        self._ensure_collection_exists()

        # Use instance batch_size if not provided
        effective_batch_size = batch_size or self.batch_size

        if not self.enable_verification:
            # Fast path for unit testing - no verification
            return self._store_batch_without_verification(segments, embeddings, effective_batch_size)

        # Medical-grade path: batch verification with fallback
        return self._store_batch_with_medical_verification(segments, embeddings, effective_batch_size)

    def _store_batch_without_verification(
        self, segments: list[ConversationSegment], embeddings: list[list[float]], batch_size: int
    ) -> list[str]:
        """Fast batch storage without verification for unit testing."""
        point_ids = []
        total_segments = len(segments)

        # Process in batches
        for i in range(0, total_segments, batch_size):
            batch_segments = segments[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            # Store each segment individually without verification
            for segment, embedding in zip(batch_segments, batch_embeddings, strict=True):
                point_id = self._store_segment_internal(segment, embedding, upsert=True)
                point_ids.append(point_id)

            # Progress logging
            processed = min(i + batch_size, total_segments)
            logger.debug(f"Stored batch progress: {processed}/{total_segments} segments (verification disabled)")

        return point_ids

    def _store_batch_with_medical_verification(
        self, segments: list[ConversationSegment], embeddings: list[list[float]], batch_size: int
    ) -> list[str]:
        """
        Store batches with medical-grade verification and fallback to individual storage.

        Strategy:
        1. Try batch storage with verification up to max_batch_retries times
        2. On failure, fall back to individual verified storage
        3. Fail fast if individual storage also fails
        """
        total_segments = len(segments)
        all_point_ids: list[str] = []
        remaining_segments = list(zip(segments, embeddings, strict=True))

        # Process in batches
        batch_num = 0
        while remaining_segments:
            batch_num += 1
            batch_segments_embeddings = remaining_segments[:batch_size]
            batch_segments = [item[0] for item in batch_segments_embeddings]
            batch_embeddings = [item[1] for item in batch_segments_embeddings]

            logger.info(
                f"Processing batch {batch_num}: {len(batch_segments)} segments "
                f"({len(all_point_ids)}/{total_segments} completed)"
            )

            # Try batch verification with retries
            batch_successful = False
            for attempt in range(1, self.max_batch_retries + 1):
                try:
                    report = self._store_batch_with_verification(batch_segments, batch_embeddings, attempt)

                    if report.success_rate == 1.0:
                        # Perfect batch success
                        batch_point_ids = [seg.segment_id for seg in batch_segments]
                        all_point_ids.extend(batch_point_ids)
                        remaining_segments = remaining_segments[batch_size:]
                        batch_successful = True
                        logger.info(
                            f"Batch {batch_num} verified successfully on attempt {attempt}: "
                            f"{len(batch_segments)} segments in {report.batch_processing_time:.3f}s"
                        )
                        break
                    else:
                        # Partial failure - log and retry
                        logger.warning(
                            f"Batch {batch_num} attempt {attempt} partially failed: "
                            f"{report.verified_points}/{report.total_points} verified "
                            f"({report.success_rate:.1%} success rate)"
                        )
                        if attempt < self.max_batch_retries:
                            delay = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
                            logger.info(f"Retrying batch {batch_num} in {delay:.2f}s (attempt {attempt + 1})")
                            time.sleep(delay)

                except Exception as e:
                    logger.warning(f"Batch {batch_num} attempt {attempt} failed with exception: {e}")
                    if attempt < self.max_batch_retries:
                        delay = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
                        logger.info(f"Retrying batch {batch_num} in {delay:.2f}s (attempt {attempt + 1})")
                        time.sleep(delay)

            # If batch failed, fall back to individual verified storage
            if not batch_successful:
                logger.warning(
                    f"Batch {batch_num} failed after {self.max_batch_retries} attempts. "
                    f"Falling back to individual verified storage for {len(batch_segments)} segments."
                )

                # Store each segment individually with verification
                individual_point_ids = self._store_segments_individually_with_verification(batch_segments, batch_embeddings)
                all_point_ids.extend(individual_point_ids)
                remaining_segments = remaining_segments[batch_size:]

        logger.info(f"Medical-grade batch storage completed: {len(all_point_ids)}/{total_segments} segments verified")
        return all_point_ids

    def _store_segments_individually_with_verification(
        self, segments: list[ConversationSegment], embeddings: list[list[float]]
    ) -> list[str]:
        """Store segments individually with verification as fallback from failed batch."""
        point_ids = []

        for i, (segment, embedding) in enumerate(zip(segments, embeddings, strict=True)):
            try:
                # Use individual retry mechanism for absolute reliability
                point_id = self._store_with_retry_and_verification(segment, embedding, upsert=True)
                point_ids.append(point_id)
                logger.debug(f"Individual fallback successful: {segment.segment_id} ({i + 1}/{len(segments)})")

            except Exception as e:
                # Individual storage failed - this is a critical failure
                logger.error(f"Individual fallback failed for segment {segment.segment_id}: {e}")
                raise AIMemoryStorageIntegrityError(
                    "Individual verification failed for AI memory segment",
                    point_id=segment.segment_id,
                    error_type="individual_verification_failed",
                ) from e

        return point_ids

    def generate_verification_report(self, verification_results: list[VerificationResult]) -> dict[str, Any]:
        """Generate a detailed verification report for audit trails."""
        total_points = len(verification_results)
        verified_points = sum(1 for r in verification_results if r.verified)
        failed_points = total_points - verified_points

        failed_details = [
            {
                "point_id": r.point_id,
                "vector_match": r.vector_match,
                "payload_match": r.payload_match,
                "cosine_similarity": r.cosine_similarity,
                "error": r.error,
                "verification_attempt": r.verification_attempt,
            }
            for r in verification_results
            if not r.verified
        ]

        return {
            "total_points": total_points,
            "verified_points": verified_points,
            "failed_points": failed_points,
            "success_rate": verified_points / total_points if total_points > 0 else 0.0,
            "timestamp": time.time(),
            "failed_details": failed_details,
            "medical_grade": self.enable_verification,
        }

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

        # Re-rank by emotional relevance, considering memory corruption
        for result in results:
            segment = result.segment

            # Handle corrupted emotional memory differently
            if not segment.emotional_state_known():
                # Mark segments with corrupted emotional memory
                result.metadata["memory_corruption"] = {
                    "emotional_state_unknown": True,
                    "corrupted_fields": segment.corrupted_fields,
                }
                # Give corrupted memories a modest boost to not completely exclude them
                # but mark them as uncertain
                emotional_boost = 0.3  # Conservative boost for unknown emotional state
                result.retrieval_reason = "emotional_context_uncertain"
            else:
                # Normal emotional relevance calculation for known emotional state
                emotional_boost = segment.emotional_score * emotional_weight
                result.retrieval_reason = "emotional_context"

            # Handle technical assessment
            if segment.technical_state_known():
                technical_penalty = segment.technical_score * (1 - emotional_weight)
            else:
                # If technical state is unknown, use neutral penalty
                technical_penalty = 0.5 * (1 - emotional_weight)
                if "memory_corruption" not in result.metadata:
                    result.metadata["memory_corruption"] = {}
                result.metadata["memory_corruption"]["technical_state_unknown"] = True

            # Adjust similarity score
            adjusted_score = result.similarity_score * 0.5 + emotional_boost * 0.3 + (1 - technical_penalty) * 0.2

            result.metadata["original_similarity"] = result.similarity_score
            result.similarity_score = adjusted_score

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
            logger.exception(f"Failed to retrieve conversation segments for {conversation_id}")
            raise RuntimeError(f"Could not retrieve conversation segments for {conversation_id}") from e

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
