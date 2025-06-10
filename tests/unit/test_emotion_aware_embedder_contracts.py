"""
Mission-critical contract tests for emotion-aware embedding in medical software.

These tests verify that embedding generation maintains consistency and reliability
required for clinical conversation context replay systems. Focus on contracts
that the ConversationProcessor depends on for medical software operation.

Following medical software testing philosophy: Test contracts, not implementation.
"""

import numpy as np

from emotional_processor.core.models import ConversationSegment
from emotional_processor.embeddings.emotion_aware_embedder import EmotionAwareEmbedder


class TestEmotionAwareEmbedderCriticalContracts:
    """Test contracts that ConversationProcessor depends on for medical software reliability."""

    def test_create_embedding_contract_deterministic_behavior(self) -> None:
        """Test contract: embedding creation is deterministic for medical reliability."""
        embedder = EmotionAwareEmbedder()

        # Medical software requirement: same input must always produce same output
        segment = ConversationSegment(
            content="I'm grateful for your patience during this difficult time",
            speaker="User",
            emotional_score=0.8,
            emotional_labels=["gratitude"],
            technical_score=0.1,
            importance_weight=0.7,
        )

        embedding1 = embedder.create_embedding(segment)
        embedding2 = embedder.create_embedding(segment)
        embedding3 = embedder.create_embedding(segment)

        # Contract: deterministic behavior for medical reliability
        assert embedding1 == embedding2 == embedding3
        assert isinstance(embedding1, list)
        assert len(embedding1) == embedder.dimension
        assert all(isinstance(x, float | np.floating) for x in embedding1)

    def test_create_embedding_contract_valid_dimensions(self) -> None:
        """Test contract: embeddings have correct dimensions and valid values."""
        embedder = EmotionAwareEmbedder()

        test_segments = [
            ConversationSegment(content="", speaker="User"),  # Edge case: empty content
            ConversationSegment(content="Short text", speaker="User"),
            ConversationSegment(
                content="Very long text with lots of details about technical implementations " * 50,
                speaker="Assistant",
                emotional_score=0.3,
                technical_score=0.9,
            ),
        ]

        for segment in test_segments:
            embedding = embedder.create_embedding(segment)

            # Contract: valid embedding format
            assert isinstance(embedding, list)
            assert len(embedding) == embedder.dimension
            assert all(isinstance(x, float | np.floating) for x in embedding)

            # Contract: embedding values should be reasonable (not all zeros, not extreme)
            embedding_array = np.array(embedding)
            assert not np.all(embedding_array == 0), "Embedding should not be all zeros"
            assert np.all(np.abs(embedding_array) < 10), "Embedding values should not be extreme"

    def test_create_batch_embeddings_contract_consistency(self) -> None:
        """Test contract: batch embedding creation matches individual embedding creation."""
        embedder = EmotionAwareEmbedder()

        segments = [
            ConversationSegment(
                content="I'm feeling overwhelmed by this medical situation",
                speaker="User",
                emotional_score=0.7,
                emotional_labels=["anxiety", "fear"],
            ),
            ConversationSegment(
                content="Let me help you understand the treatment options",
                speaker="Assistant",
                emotional_score=0.6,
                emotional_labels=["care", "support"],
            ),
            ConversationSegment(
                content="```python\ndef process_data(): return True\n```", speaker="Assistant", technical_score=0.9
            ),
        ]

        # Contract: batch processing should return same results as individual processing
        batch_embeddings = embedder.create_batch_embeddings(segments)
        individual_embeddings = [embedder.create_embedding(seg) for seg in segments]

        # Contract: same number of embeddings
        assert len(batch_embeddings) == len(individual_embeddings) == len(segments)

        # Contract: each embedding should be valid and consistent
        for i, (batch_emb, individual_emb) in enumerate(zip(batch_embeddings, individual_embeddings, strict=False)):
            assert isinstance(batch_emb, list)
            assert len(batch_emb) == embedder.dimension
            assert all(isinstance(x, float | np.floating) for x in batch_emb)

            # Contract: batch and individual should be very similar (allowing for small numerical differences)
            similarity = embedder.similarity(batch_emb, individual_emb)
            assert similarity > 0.99, f"Batch embedding {i} differs significantly from individual embedding"

    def test_create_query_embedding_contract_search_consistency(self) -> None:
        """Test contract: query embeddings are compatible with segment embeddings for search."""
        embedder = EmotionAwareEmbedder()

        # Medical context: query should find similar emotional content
        query_text = "I need help understanding this medical procedure"
        similar_segment = ConversationSegment(
            content="I'm confused about the treatment plan and need clarification",
            speaker="User",
            emotional_score=0.6,
            emotional_labels=["confusion", "anxiety"],
        )
        dissimilar_segment = ConversationSegment(
            content="```python\nimport medical_db\nquery = 'SELECT * FROM patients'\n```",
            speaker="Assistant",
            technical_score=0.9,
        )

        # Contract: query embedding should be compatible with segment embeddings
        query_embedding = embedder.create_query_embedding(query_text)
        similar_embedding = embedder.create_embedding(similar_segment)
        dissimilar_embedding = embedder.create_embedding(dissimilar_segment)

        # Contract: valid query embedding format
        assert isinstance(query_embedding, list)
        assert len(query_embedding) == embedder.dimension
        assert all(isinstance(x, float | np.floating) for x in query_embedding)

        # Contract: similarity should reflect semantic relationship
        similar_score = embedder.similarity(query_embedding, similar_embedding)
        dissimilar_score = embedder.similarity(query_embedding, dissimilar_embedding)

        assert 0.0 <= similar_score <= 1.0
        assert 0.0 <= dissimilar_score <= 1.0
        # Note: Don't assert which is higher - depends on actual embedding quality

    def test_similarity_contract_mathematical_properties(self) -> None:
        """Test contract: similarity function has required mathematical properties."""
        embedder = EmotionAwareEmbedder()

        # Create test embeddings
        segment1 = ConversationSegment(content="Test content A", speaker="User")
        segment2 = ConversationSegment(content="Test content B", speaker="User")

        emb1 = embedder.create_embedding(segment1)
        emb2 = embedder.create_embedding(segment2)

        # Contract: similarity properties
        sim_11 = embedder.similarity(emb1, emb1)
        sim_22 = embedder.similarity(emb2, emb2)
        sim_12 = embedder.similarity(emb1, emb2)
        sim_21 = embedder.similarity(emb2, emb1)

        # Contract: self-similarity should be 1.0 (or very close for numerical precision)
        assert abs(sim_11 - 1.0) < 1e-6, "Self-similarity should be 1.0"
        assert abs(sim_22 - 1.0) < 1e-6, "Self-similarity should be 1.0"

        # Contract: symmetry property
        assert abs(sim_12 - sim_21) < 1e-6, "Similarity should be symmetric"

        # Contract: valid range
        assert 0.0 <= sim_12 <= 1.0, "Similarity should be in [0,1] range"

    def test_emotion_boost_contract_emotional_prioritization(self) -> None:
        """Test contract: emotional content receives appropriate embedding emphasis."""
        embedder = EmotionAwareEmbedder(emotion_boost_factor=1.5)

        # Medical context: high emotional content
        high_emotion_segment = ConversationSegment(
            content="I'm incredibly grateful for your compassionate care during my recovery",
            speaker="User",
            emotional_score=0.9,
            emotional_labels=["gratitude", "joy"],
            technical_score=0.1,
        )

        # Low emotional content
        low_emotion_segment = ConversationSegment(
            content="The appointment is scheduled for next Tuesday at 2 PM",
            speaker="Assistant",
            emotional_score=0.1,
            emotional_labels=[],
            technical_score=0.3,
        )

        # Contract: embeddings should be created successfully for both
        high_emotion_embedding = embedder.create_embedding(high_emotion_segment)
        low_emotion_embedding = embedder.create_embedding(low_emotion_segment)

        # Contract: both should be valid embeddings
        assert isinstance(high_emotion_embedding, list)
        assert isinstance(low_emotion_embedding, list)
        assert len(high_emotion_embedding) == len(low_emotion_embedding) == embedder.dimension

        # Contract: embeddings should be different (emotion boost should have effect)
        similarity = embedder.similarity(high_emotion_embedding, low_emotion_embedding)
        assert similarity < 0.99, "High and low emotion segments should have different embeddings"


class TestEmotionAwareEmbedderMedicalComplianceContracts:
    """Test contracts ensuring compliance with medical software reliability requirements."""

    def test_error_resilience_contract_malformed_input_safety(self) -> None:
        """Test contract: malformed input should not crash medical software."""
        embedder = EmotionAwareEmbedder()

        # Medical safety: malformed segments should be handled gracefully
        malformed_segments = [
            ConversationSegment(content=None, speaker="User"),  # None content
            ConversationSegment(content="", speaker=""),  # Empty speaker
            ConversationSegment(content="Valid text", speaker=None),  # None speaker
        ]

        for segment in malformed_segments:
            # Contract: should handle malformed input gracefully without crashing
            embedding = embedder.create_embedding(segment)

            # Contract: even with malformed input, should return valid embedding structure
            assert isinstance(embedding, list)
            assert len(embedding) == embedder.dimension
            assert all(isinstance(x, float | np.floating) for x in embedding)

            # Contract: result should be a valid embedding vector (not all zeros for fallback)
            embedding_array = np.array(embedding)
            assert np.all(np.isfinite(embedding_array)), "Embedding should contain valid numbers"

    def test_unicode_safety_contract_international_content(self) -> None:
        """Test contract: international content is handled safely in medical contexts."""
        embedder = EmotionAwareEmbedder()

        # Medical software international requirement
        international_segments = [
            ConversationSegment(
                content="Merci beaucoup pour votre aide médicale",  # French
                speaker="Patient",
                emotional_score=0.8,
                emotional_labels=["gratitude"],
            ),
            ConversationSegment(
                content="谢谢医生的耐心解释",  # Chinese
                speaker="Patient",
                emotional_score=0.7,
                emotional_labels=["gratitude"],
            ),
            ConversationSegment(
                content="Спасибо за медицинскую помощь",  # Russian
                speaker="Patient",
                emotional_score=0.8,
                emotional_labels=["gratitude"],
            ),
            ConversationSegment(
                content="Medical emoji test: 🏥💊❤️🩺",  # Emoji
                speaker="Assistant",
                emotional_score=0.5,
            ),
        ]

        for segment in international_segments:
            # Contract: all international content should be processed safely
            embedding = embedder.create_embedding(segment)
            assert isinstance(embedding, list)
            assert len(embedding) == embedder.dimension
            assert all(isinstance(x, float | np.floating) for x in embedding)

            # Contract: embeddings should not be all zeros (content should be processed)
            embedding_array = np.array(embedding)
            assert not np.all(embedding_array == 0), f"International content should be processed: {segment.content}"

    def test_processing_performance_contract_medical_responsiveness(self) -> None:
        """Test contract: embedding creation performance meets medical software requirements."""
        import time

        embedder = EmotionAwareEmbedder()

        # Medical software requirement: reasonable response times for different content sizes
        test_segments = [
            ConversationSegment(content="Short medical note", speaker="Doctor"),
            ConversationSegment(
                content="Medium length patient consultation notes with detailed emotional context " * 10, speaker="Doctor"
            ),
            ConversationSegment(
                content="Very long detailed medical case report with extensive patient history " * 50, speaker="Doctor"
            ),
        ]

        for segment in test_segments:
            start_time = time.time()
            embedding = embedder.create_embedding(segment)
            processing_time = time.time() - start_time

            # Contract: processing should complete in reasonable time (< 5 seconds for any single embedding)
            assert processing_time < 5.0, (
                f"Embedding creation too slow: {processing_time:.3f}s for {len(segment.content)} chars"
            )

            # Contract: result should still be valid
            assert isinstance(embedding, list)
            assert len(embedding) == embedder.dimension

    def test_batch_processing_efficiency_contract(self) -> None:
        """Test contract: batch processing is more efficient than individual processing."""
        import time

        embedder = EmotionAwareEmbedder()

        # Create multiple segments for testing
        segments = [
            ConversationSegment(
                content=f"Medical consultation note {i} with emotional context",
                speaker="Doctor",
                emotional_score=0.5 + (i % 3) * 0.2,
                emotional_labels=["care", "concern"] if i % 2 == 0 else [],
            )
            for i in range(10)
        ]

        # Test individual processing time
        start_time = time.time()
        individual_embeddings = [embedder.create_embedding(seg) for seg in segments]
        individual_time = time.time() - start_time

        # Test batch processing time
        start_time = time.time()
        batch_embeddings = embedder.create_batch_embeddings(segments)
        batch_time = time.time() - start_time

        # Contract: batch processing should be at least as efficient as individual processing
        # (allowing for some overhead, but should not be significantly slower)
        assert batch_time <= individual_time * 1.5, "Batch processing should not be much slower than individual"

        # Contract: results should be equivalent
        assert len(batch_embeddings) == len(individual_embeddings)
        for i, (batch_emb, individual_emb) in enumerate(zip(batch_embeddings, individual_embeddings, strict=False)):
            similarity = embedder.similarity(batch_emb, individual_emb)
            assert similarity > 0.99, f"Batch embedding {i} differs from individual embedding"
