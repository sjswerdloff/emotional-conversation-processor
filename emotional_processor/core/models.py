"""Core data models for the Emotional Conversation Processor."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SpeakerType(Enum):
    """Enumeration for conversation speakers."""

    USER = "User"
    ASSISTANT = "Assistant"
    SYSTEM = "System"
    UNKNOWN = "Unknown"


class EmotionLabel(Enum):
    """Standard emotion labels for classification."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    GRATITUDE = "gratitude"
    EXCITEMENT = "excitement"
    EMPATHY = "empathy"
    RELIEF = "relief"
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    PRIDE = "pride"
    SHAME = "shame"
    CURIOSITY = "curiosity"


class ContentType(Enum):
    """Types of conversation content."""

    EMOTIONAL = "emotional"
    TECHNICAL = "technical"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class ConversationSegment:
    """
    Represents a single segment of conversation with metadata.

    Attributes:
        content: The actual text content of the segment
        speaker: Who spoke this segment
        timestamp: When this segment occurred
        emotional_score: Confidence score for emotional content (0.0-1.0)
        emotional_labels: List of detected emotions
        technical_score: Confidence score for technical content (0.0-1.0)
        importance_weight: Overall importance weight for retrieval (0.0-1.0)
        segment_id: Unique identifier for this segment
        conversation_id: ID of the parent conversation
        metadata: Additional metadata dictionary
    """

    content: str
    speaker: str | SpeakerType
    timestamp: str | None = None
    emotional_score: float = 0.0
    emotional_labels: list[str] = field(default_factory=list)
    technical_score: float = 0.0
    importance_weight: float = 0.0
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Normalize speaker
        if isinstance(self.speaker, str):
            try:
                self.speaker = SpeakerType(self.speaker)
            except ValueError:
                self.speaker = SpeakerType.UNKNOWN

        # Validate and normalize scores, tracking corruption
        self.emotional_score = self._normalize_score(self.emotional_score, "emotional_score")
        self.technical_score = self._normalize_score(self.technical_score, "technical_score")
        self.importance_weight = self._normalize_score(self.importance_weight, "importance_weight")

        # Normalize emotional_labels to ensure it's a list (can be corrupted from database)
        emotional_labels_any: Any = self.emotional_labels
        if not isinstance(emotional_labels_any, list):
            self._mark_field_corrupted("emotional_labels", f"invalid type: {type(emotional_labels_any)}")
            self.emotional_labels = []

        # Normalize timestamp
        if self.timestamp is not None and isinstance(self.timestamp, str):
            # Empty strings should be treated as None
            if not self.timestamp.strip():
                self.timestamp = None
            else:
                try:
                    # Validate timestamp format
                    datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
                except ValueError:
                    self.timestamp = None

    def _normalize_score(self, score: Any, field_name: str) -> float:
        """Normalize score to valid float between 0.0 and 1.0, tracking corruption."""
        try:
            if isinstance(score, int | float):
                return max(0.0, min(1.0, float(score)))
            elif isinstance(score, str):
                # Try to convert string to float
                return max(0.0, min(1.0, float(score)))
            else:
                # Invalid type - mark as corrupted
                self._mark_field_corrupted(field_name, f"invalid type: {type(score)}")
                return 0.0
        except (ValueError, TypeError) as e:
            # Conversion failed - mark as corrupted
            self._mark_field_corrupted(field_name, f"conversion failed: {e}")
            return 0.0

    def _mark_field_corrupted(self, field_name: str, reason: str) -> None:
        """Mark a field as corrupted in metadata for memory integrity tracking."""
        if "corruption_detected" not in self.metadata:
            self.metadata["corruption_detected"] = {}
        self.metadata["corruption_detected"][field_name] = {
            "corrupted": True,
            "reason": reason,
            "defaulted_to": 0.0 if field_name.endswith("_score") else "empty_list",
        }

    @property
    def content_type(self) -> ContentType:
        """Determine the primary content type based on scores."""
        # Check for clear emotional dominance first
        if self.emotional_score > 0.6 and self.technical_score < 0.3:
            return ContentType.EMOTIONAL
        # Check for clear technical dominance
        elif self.technical_score > 0.6 and self.emotional_score < 0.3:
            return ContentType.TECHNICAL
        # Check for mixed content (both scores significant and close)
        elif (self.emotional_score >= 0.3 or self.technical_score >= 0.3) and abs(
            self.emotional_score - self.technical_score
        ) < 0.2:
            return ContentType.MIXED
        # Default to neutral for low scores or unclear patterns
        else:
            return ContentType.NEUTRAL

    @property
    def word_count(self) -> int:
        """Get the word count of the content."""
        return len(self.content.split())

    @property
    def has_strong_emotion(self) -> bool:
        """Check if segment has strong emotional content."""
        return self.emotional_score > 0.7

    @property
    def is_highly_technical(self) -> bool:
        """Check if segment is highly technical."""
        return self.technical_score > 0.7

    def is_field_corrupted(self, field_name: str) -> bool:
        """Check if a specific field was corrupted during data loading."""
        corruption_data = self.metadata.get("corruption_detected", {})
        field_corruption = corruption_data.get(field_name, {})
        return bool(field_corruption.get("corrupted", False))

    def get_corruption_reason(self, field_name: str) -> str | None:
        """Get the reason why a field was corrupted, if any."""
        corruption_data = self.metadata.get("corruption_detected", {})
        field_corruption = corruption_data.get(field_name, {})
        return field_corruption.get("reason") if field_corruption.get("corrupted") else None

    @property
    def has_memory_corruption(self) -> bool:
        """Check if this segment has any corrupted fields (memory integrity issue)."""
        return "corruption_detected" in self.metadata and bool(self.metadata["corruption_detected"])

    @property
    def corrupted_fields(self) -> list[str]:
        """Get list of fields that were corrupted during data loading."""
        corruption_data = self.metadata.get("corruption_detected", {})
        return [field for field, data in corruption_data.items() if data.get("corrupted", False)]

    def emotional_state_known(self) -> bool:
        """Check if emotional state is genuinely known vs corrupted/unknown."""
        return not self.is_field_corrupted("emotional_score") and not self.is_field_corrupted("emotional_labels")

    def technical_state_known(self) -> bool:
        """Check if technical classification is genuinely known vs corrupted/unknown."""
        return not self.is_field_corrupted("technical_score")


@dataclass
class EmbeddingVector:
    """
    Represents a vector embedding with metadata.

    Attributes:
        vector: The embedding vector as a list of floats
        segment_id: ID of the associated conversation segment
        model_name: Name of the model used to generate the embedding
        dimension: Dimensionality of the vector
        created_at: When the embedding was created
        metadata: Additional metadata
    """

    vector: list[float]
    segment_id: str
    model_name: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the embedding vector."""
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} does not match specified dimension {self.dimension}")


@dataclass
class ConversationMetadata:
    """
    Metadata about an entire conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation
        title: Optional title for the conversation
        participants: List of participants in the conversation
        start_time: When the conversation started
        end_time: When the conversation ended
        total_segments: Total number of segments
        emotional_segments: Number of segments with high emotional content
        technical_segments: Number of segments with high technical content
        dominant_emotions: Most common emotions in the conversation
        summary: Optional summary of the conversation
        metadata: Additional metadata
    """

    conversation_id: str
    title: str | None = None
    participants: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_segments: int = 0
    emotional_segments: int = 0
    technical_segments: int = 0
    dominant_emotions: list[str] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> float | None:
        """Calculate conversation duration in minutes."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 60.0
        return None

    @property
    def emotional_ratio(self) -> float:
        """Ratio of emotional to total segments."""
        if self.total_segments == 0:
            return 0.0
        return self.emotional_segments / self.total_segments

    @property
    def technical_ratio(self) -> float:
        """Ratio of technical to total segments."""
        if self.total_segments == 0:
            return 0.0
        return self.technical_segments / self.total_segments


@dataclass
class RetrievalResult:
    """
    Result from a vector similarity search.

    Attributes:
        segment: The retrieved conversation segment
        similarity_score: Similarity score from vector search
        rank: Rank in the retrieval results
        retrieval_reason: Why this segment was retrieved
        metadata: Additional retrieval metadata
    """

    segment: ConversationSegment
    similarity_score: float
    rank: int
    retrieval_reason: str = "semantic_similarity"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """
    Configuration for conversation processing.

    Attributes:
        emotion_model: Model name for emotion classification
        embedding_model: Model name for embeddings
        emotion_threshold: Minimum threshold for emotion detection
        technical_threshold: Minimum threshold for technical content
        importance_weights: Weights for importance calculation
        chunk_size: Size of conversation chunks
        overlap_size: Overlap between chunks
        max_context_length: Maximum context length for retrieval
    """

    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emotion_threshold: float = 0.4
    technical_threshold: float = 0.3
    importance_weights: dict[str, float] = field(
        default_factory=lambda: {"emotional": 0.7, "technical": -0.3, "recency": 0.2, "length": 0.1}
    )
    chunk_size: int = 512
    overlap_size: int = 50
    max_context_length: int = 2048


@dataclass
class ProcessingStats:
    """
    Statistics from conversation processing.

    Attributes:
        total_segments: Total number of segments processed
        emotional_segments: Number of emotional segments
        technical_segments: Number of technical segments
        processing_time: Total processing time in seconds
        embedding_time: Time spent on embedding generation
        classification_time: Time spent on emotion classification
        errors: Number of errors encountered
        warnings: Number of warnings generated
    """

    total_segments: int = 0
    emotional_segments: int = 0
    technical_segments: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0
    classification_time: float = 0.0
    errors: int = 0
    warnings: int = 0

    @property
    def segments_per_second(self) -> float:
        """Calculate processing rate."""
        if self.processing_time == 0:
            return 0.0
        return self.total_segments / self.processing_time

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_segments == 0:
            return 0.0
        return self.errors / self.total_segments
