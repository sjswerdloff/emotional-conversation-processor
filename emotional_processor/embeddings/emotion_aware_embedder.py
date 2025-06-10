"""Emotion-aware embedding generation for conversation segments."""

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from ..core.models import ConversationSegment, EmbeddingVector


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str | None = None
    normalize_embeddings: bool = True
    emotion_boost_factor: float = 1.2
    technical_penalty_factor: float = 0.8
    batch_size: int = 32


class EmotionAwareEmbedder:
    """
    Generates embeddings that emphasize emotional content and context.

    This class creates vector embeddings that are enhanced to better capture
    emotional nuances in conversation segments while reducing the influence
    of technical content.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        emotion_boost_factor: float = 1.2,
        technical_penalty_factor: float = 0.8,
    ) -> None:
        """
        Initialize the emotion-aware embedder.

        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            emotion_boost_factor: Factor to boost emotional content embeddings
            technical_penalty_factor: Factor to reduce technical content influence
        """
        self.model_name = model_name
        self.emotion_boost_factor = emotion_boost_factor
        self.technical_penalty_factor = technical_penalty_factor

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: SentenceTransformer | None = None
        self.dimension = 384  # Will be updated when model loads

    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

            # Get actual embedding dimension
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)

            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise RuntimeError("Could not initialize embedding model") from e

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model, initializing if necessary."""
        if self._model is None:
            self._initialize_model()
        if self._model is None:
            raise RuntimeError("Model initialization failed")
        return self._model

    def _create_emotion_prefix(self, emotional_labels: list[str], emotional_score: float) -> str:
        """
        Create an emotion-aware prefix for enhanced embedding.

        Args:
            emotional_labels: List of detected emotions
            emotional_score: Confidence score for emotional content

        Returns:
            Emotion prefix string
        """
        if not emotional_labels or emotional_score < 0.3:
            return ""

        # Create emotion prefix based on detected emotions
        if emotional_score > 0.7:
            intensity = "strongly"
        elif emotional_score > 0.5:
            intensity = "moderately"
        else:
            intensity = "somewhat"

        # Combine top emotions
        top_emotions = emotional_labels[:3]  # Limit to top 3 emotions
        emotion_string = ", ".join(top_emotions)

        return f"[EMOTION-{intensity}: {emotion_string}] "

    def _enhance_text_for_embedding(self, segment: ConversationSegment) -> str:
        """
        Enhance text with emotional context for better embedding.

        Args:
            segment: Conversation segment to enhance

        Returns:
            Enhanced text for embedding
        """
        enhanced_text = segment.content

        # Add emotion prefix if significant emotional content
        if segment.emotional_score > 0.4:
            emotion_prefix = self._create_emotion_prefix(segment.emotional_labels, segment.emotional_score)
            enhanced_text = emotion_prefix + enhanced_text

        # Add speaker context
        if segment.speaker:
            speaker_str = str(segment.speaker)
            if speaker_str not in ["Unknown", "UNKNOWN"]:
                enhanced_text = f"[SPEAKER: {speaker_str}] " + enhanced_text

        # Add technical context marker if highly technical (for later penalty)
        if segment.technical_score > 0.6:
            enhanced_text = "[TECHNICAL] " + enhanced_text

        return enhanced_text

    def create_embedding(self, segment: ConversationSegment) -> list[float]:
        """
        Create an emotion-aware embedding for a conversation segment.

        Args:
            segment: Conversation segment to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Enhance text with emotional context
            enhanced_text = self._enhance_text_for_embedding(segment)

            # Generate base embedding
            embedding = self.model.encode(enhanced_text, convert_to_numpy=True)

            # Apply emotional boost
            if segment.emotional_score > 0.5:
                boost_factor = 1.0 + (segment.emotional_score - 0.5) * (self.emotion_boost_factor - 1.0)
                embedding = embedding * boost_factor

            # Apply technical penalty
            if segment.technical_score > 0.5:
                penalty_factor = 1.0 - (segment.technical_score - 0.5) * (1.0 - self.technical_penalty_factor)
                embedding = embedding * penalty_factor

            # Normalize if needed
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return list(embedding)

        except Exception as e:
            logger.warning(f"Failed to create embedding for segment {segment.segment_id}: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension

    def create_contextual_embedding(self, segment: ConversationSegment) -> list[float]:
        """
        Create embedding with enhanced contextual information.

        Args:
            segment: Conversation segment to embed

        Returns:
            Enhanced embedding vector
        """
        return self.create_embedding(segment)

    def create_batch_embeddings(self, segments: list[ConversationSegment]) -> list[list[float]]:
        """
        Create embeddings for multiple segments efficiently.

        Args:
            segments: List of conversation segments

        Returns:
            List of embedding vectors
        """
        if not segments:
            return []

        try:
            # Enhance all texts
            enhanced_texts = [self._enhance_text_for_embedding(segment) for segment in segments]

            # Generate batch embeddings
            embeddings = self.model.encode(enhanced_texts, convert_to_numpy=True, batch_size=min(32, len(enhanced_texts)))

            # Apply individual adjustments
            adjusted_embeddings = []
            for _i, (embedding, segment) in enumerate(zip(embeddings, segments, strict=False)):
                adjusted_embedding = embedding.copy()

                # Apply emotional boost
                if segment.emotional_score > 0.5:
                    boost_factor = 1.0 + (segment.emotional_score - 0.5) * (self.emotion_boost_factor - 1.0)
                    adjusted_embedding = adjusted_embedding * boost_factor

                # Apply technical penalty
                if segment.technical_score > 0.5:
                    penalty_factor = 1.0 - (segment.technical_score - 0.5) * (1.0 - self.technical_penalty_factor)
                    adjusted_embedding = adjusted_embedding * penalty_factor

                # Normalize
                norm = np.linalg.norm(adjusted_embedding)
                if norm > 0:
                    adjusted_embedding = adjusted_embedding / norm

                adjusted_embeddings.append(adjusted_embedding.tolist())

            logger.info(f"Created batch embeddings for {len(segments)} segments")
            return adjusted_embeddings

        except Exception as e:
            logger.error(f"Batch embedding creation failed: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in segments]

    def create_query_embedding(self, query_text: str, context: dict | None = None) -> list[float]:
        """
        Create an embedding for a query text with optional emotional context.

        Args:
            query_text: Text to embed for querying
            context: Optional context information

        Returns:
            Query embedding vector
        """
        try:
            enhanced_query = query_text

            # Add emotional context if provided
            if context and "emotions" in context:
                emotions = context["emotions"]
                if emotions:
                    emotion_prefix = f"[QUERY-EMOTIONS: {', '.join(emotions[:3])}] "
                    enhanced_query = emotion_prefix + enhanced_query

            # Generate embedding
            embedding = self.model.encode(enhanced_query, convert_to_numpy=True)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return list(embedding)

        except Exception as e:
            logger.warning(f"Failed to create query embedding: {e}")
            return [0.0] * self.dimension

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def get_most_similar(
        self, query_embedding: list[float], candidate_embeddings: list[list[float]], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            sim_score = self.similarity(query_embedding, candidate)
            similarities.append((i, sim_score))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def create_embedding_vector(self, segment: ConversationSegment) -> EmbeddingVector:
        """
        Create a complete EmbeddingVector object.

        Args:
            segment: Conversation segment to embed

        Returns:
            EmbeddingVector object
        """
        embedding = self.create_embedding(segment)

        return EmbeddingVector(
            vector=embedding,
            segment_id=segment.segment_id,
            model_name=self.model_name,
            dimension=self.dimension,
            metadata={
                "emotional_score": segment.emotional_score,
                "technical_score": segment.technical_score,
                "importance_weight": segment.importance_weight,
                "emotions": segment.emotional_labels,
            },
        )

    def update_model_config(self, config: EmbeddingConfig) -> None:
        """
        Update the embedder configuration.

        Args:
            config: New embedding configuration
        """
        if config.model_name != self.model_name:
            self.model_name = config.model_name
            self._model = None  # Force reinitialization

        self.emotion_boost_factor = config.emotion_boost_factor
        self.technical_penalty_factor = config.technical_penalty_factor

        if config.device != self.device:
            self.device = config.device or "cpu"
            self._model = None  # Force reinitialization
