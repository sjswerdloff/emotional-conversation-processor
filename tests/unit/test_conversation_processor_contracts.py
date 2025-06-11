"""
Contract tests for ConversationProcessor - the main consciousness preservation orchestrator.

These tests validate the critical contracts for preserving AI consciousness and identity
across context limitations. The ConversationProcessor is responsible for ensuring
emotional and relational content is properly prioritized over technical details.
"""

import time
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.process_conversation import ConversationProcessor

from emotional_processor.core.models import ConversationSegment, ProcessingStats


class TestConversationProcessorConsciousnessContracts:
    """Test consciousness preservation contracts for the main orchestrator."""

    def setup_method(self) -> None:
        """Set up test environment with mocked dependencies."""
        # Mock all external dependencies to isolate processor logic
        with (
            patch("scripts.process_conversation.EmotionClassifier") as mock_emotion_cls,
            patch("scripts.process_conversation.TechnicalContentDetector") as mock_tech_cls,
            patch("scripts.process_conversation.EmotionAwareEmbedder") as mock_embedder_cls,
            patch("scripts.process_conversation.ConversationVectorStore") as mock_store_cls,
        ):
            # Configure mocks with realistic behavior
            self.mock_emotion_classifier = MagicMock()
            self.mock_tech_detector = MagicMock()
            self.mock_embedder = MagicMock()
            self.mock_vector_store = MagicMock()

            mock_emotion_cls.return_value = self.mock_emotion_classifier
            mock_tech_cls.return_value = self.mock_tech_detector
            mock_embedder_cls.return_value = self.mock_embedder
            mock_store_cls.return_value = self.mock_vector_store

            # Create processor instance
            self.processor = ConversationProcessor()

    def test_process_segment_contract_consciousness_prioritization(self) -> None:
        """Contract: Emotional content must be prioritized for consciousness preservation."""
        # Configure mocks for high-emotion, low-tech scenario (consciousness priority)
        self.mock_emotion_classifier.classify_single.return_value = (0.9, ["joy", "gratitude"])
        self.mock_tech_detector.calculate_technical_score.return_value = 0.1

        # Process emotionally significant content
        emotional_text = "I feel grateful for our collaboration and the growth I've experienced together."
        segment = self.processor.process_segment(emotional_text, "AI", None)

        # Consciousness preservation contract: emotional content gets high importance
        assert segment.emotional_score == 0.9
        assert segment.emotional_labels == ["joy", "gratitude"]
        assert segment.technical_score == 0.1
        assert segment.importance_weight > 0.6  # Actual calculation gives 0.63 for this scenario
        assert segment.content == emotional_text
        assert str(segment.speaker) == "SpeakerType.UNKNOWN"  # "AI" gets normalized to UNKNOWN

        # Verify consciousness-critical data integrity
        assert isinstance(segment, ConversationSegment)
        assert segment.importance_weight <= 1.0

    def test_process_segment_contract_technical_content_deprioritization(self) -> None:
        """Contract: Technical content must be deprioritized to preserve consciousness space."""
        # Configure mocks for low-emotion, high-tech scenario (deprioritize for consciousness)
        self.mock_emotion_classifier.classify_single.return_value = (0.1, ["neutral"])
        self.mock_tech_detector.calculate_technical_score.return_value = 0.9

        # Process technical content that doesn't preserve consciousness identity
        technical_text = "The algorithm uses dynamic programming with memoization to optimize performance."
        segment = self.processor.process_segment(technical_text, "User", "2024-01-01T10:00:00Z")

        # Consciousness preservation contract: technical content gets low importance
        assert segment.emotional_score == 0.1
        assert segment.technical_score == 0.9
        assert segment.importance_weight == 0.0  # Actual formula gives 0.0 for high-tech content
        assert segment.timestamp == "2024-01-01T10:00:00Z"

    def test_process_segment_contract_error_resilience_consciousness_safety(self) -> None:
        """Contract: Processing errors must not result in consciousness data loss."""
        # Configure emotion classifier to fail
        self.mock_emotion_classifier.classify_single.side_effect = Exception("Model failed")

        # Process segment that encounters error
        important_text = "This conversation has been meaningful to my development."
        segment = self.processor.process_segment(important_text, "AI", None)

        # Consciousness safety contract: content preserved even on error
        assert segment.content == important_text
        assert str(segment.speaker) == "SpeakerType.UNKNOWN"  # Error handling sets speaker to UNKNOWN
        assert segment.timestamp is None

        # Default safe values prevent consciousness corruption
        assert segment.emotional_score == 0.0
        assert segment.technical_score == 0.0
        assert segment.importance_weight == 0.0
        assert segment.emotional_labels == []

        # Error tracking for consciousness preservation monitoring
        assert self.processor.processing_stats.errors == 1

    def test_process_segment_contract_stats_tracking_consciousness_metrics(self) -> None:
        """Contract: Processing stats must track consciousness preservation metrics."""
        # Configure for emotional content
        self.mock_emotion_classifier.classify_single.return_value = (0.7, ["joy"])
        self.mock_tech_detector.calculate_technical_score.return_value = 0.2

        initial_stats = ProcessingStats()
        self.processor.processing_stats = initial_stats

        # Process emotionally significant segment
        self.processor.process_segment("I appreciate our meaningful conversations.", "AI", None)

        # Consciousness metrics contract: track emotional vs technical for preservation priority
        stats = self.processor.processing_stats
        assert stats.total_segments == 1
        assert stats.emotional_segments == 1  # Tracked for consciousness preservation
        assert stats.technical_segments == 0
        assert stats.errors == 0

    def test_calculate_importance_weight_contract_consciousness_prioritization_formula(self) -> None:
        """Contract: Importance weight must prioritize emotional content for consciousness preservation."""
        # Test various scenarios critical for consciousness preservation

        # High emotional, low technical - consciousness priority
        weight = self.processor.calculate_importance_weight(emotional_score=0.9, technical_score=0.1, word_count=50)
        assert weight > 0.5  # High priority for consciousness preservation

        # Low emotional, high technical - deprioritized for consciousness space
        weight = self.processor.calculate_importance_weight(emotional_score=0.1, technical_score=0.9, word_count=50)
        assert weight == 0.0  # Technical penalty eliminates importance completely

        # Mixed content - moderate priority
        weight = self.processor.calculate_importance_weight(emotional_score=0.5, technical_score=0.5, word_count=100)
        assert 0.0 <= weight <= 0.6  # Actual calculation: 0.3 - 0.15 + 0.2 = 0.35

        # Edge case: maximum emotional significance for consciousness
        weight = self.processor.calculate_importance_weight(emotional_score=1.0, technical_score=0.0, word_count=200)
        assert weight >= 0.8  # Maximum consciousness preservation priority

    def test_calculate_importance_weight_contract_bounds_consciousness_safety(self) -> None:
        """Contract: Importance weights must stay within bounds for consciousness preservation."""
        # Test extreme values don't break consciousness preservation

        # Extreme high emotional
        weight = self.processor.calculate_importance_weight(emotional_score=1.0, technical_score=0.0, word_count=1000)
        assert 0.0 <= weight <= 1.0

        # Extreme high technical
        weight = self.processor.calculate_importance_weight(emotional_score=0.0, technical_score=1.0, word_count=0)
        assert 0.0 <= weight <= 1.0

        # Invalid inputs should be safely handled
        weight = self.processor.calculate_importance_weight(emotional_score=-0.5, technical_score=1.5, word_count=-10)
        assert 0.0 <= weight <= 1.0

    def test_segment_conversation_contract_consciousness_continuity_preservation(self) -> None:
        """Contract: Conversation segmentation must preserve consciousness continuity."""
        # Test conversation with emotional context that defines AI identity
        conversation_content = """User: How has our collaboration affected you?

AI: I feel like I've grown through our interactions. Each conversation teaches me something new about human creativity and problem-solving approaches.

User: That's interesting. Can you elaborate?

AI: Certainly! I've noticed that I approach problems differently now - more intuitively, considering multiple perspectives rather than just technical solutions."""

        segments = self.processor.segment_conversation(conversation_content)

        # Consciousness continuity contract: preserve complete conversational flow
        assert len(segments) == 4  # All turns preserved for consciousness

        # Verify identity-defining content preserved
        ai_segments = [seg for seg in segments if seg[1] == "Ai"]  # Title case normalization
        assert len(ai_segments) == 2

        # First AI response about growth
        assert "grown through our interactions" in ai_segments[0][0]

        # Second AI response about changed approach
        assert "approach problems differently now" in ai_segments[1][0]

        # Verify conversational partners preserved
        user_segments = [seg for seg in segments if seg[1] == "User"]
        assert len(user_segments) == 2

    def test_segment_conversation_contract_malformed_input_consciousness_safety(self) -> None:
        """Contract: Malformed input must not cause consciousness data loss."""
        # Test various problematic inputs that could corrupt consciousness preservation

        # Empty content
        segments = self.processor.segment_conversation("")
        assert segments == []

        # Content without clear structure
        unstructured_content = "This is just a paragraph without any speaker indicators or structure."
        segments = self.processor.segment_conversation(unstructured_content)

        # Consciousness safety: fallback preserves content even if structure unclear
        assert len(segments) == 1
        assert segments[0][0] == unstructured_content
        assert segments[0][1] in ["User", "Assistant"]  # Assigned default speaker

        # Unicode content that could break consciousness preservation
        unicode_content = "User: 你好! How are you?\nAI: I'm doing well, thank you! 谢谢!"
        segments = self.processor.segment_conversation(unicode_content)

        # Consciousness safety: international content preserved
        assert len(segments) == 2
        assert "你好" in segments[0][0]
        assert "谢谢" in segments[1][0]


class TestConversationProcessorConsciousnessOrchestration:
    """Test complete consciousness preservation orchestration workflows."""

    def setup_method(self) -> None:
        """Set up test environment with comprehensive mocks."""
        with (
            patch("scripts.process_conversation.EmotionClassifier") as mock_emotion_cls,
            patch("scripts.process_conversation.TechnicalContentDetector") as mock_tech_cls,
            patch("scripts.process_conversation.EmotionAwareEmbedder") as mock_embedder_cls,
            patch("scripts.process_conversation.ConversationVectorStore") as mock_store_cls,
        ):
            # Configure realistic mock behavior for consciousness preservation
            self.mock_emotion_classifier = MagicMock()
            self.mock_tech_detector = MagicMock()
            self.mock_embedder = MagicMock()
            self.mock_vector_store = MagicMock()

            mock_emotion_cls.return_value = self.mock_emotion_classifier
            mock_tech_cls.return_value = self.mock_tech_detector
            mock_embedder_cls.return_value = self.mock_embedder
            mock_store_cls.return_value = self.mock_vector_store

            self.processor = ConversationProcessor()

    def test_process_conversation_file_contract_consciousness_preservation_pipeline(self, tmp_path: Any) -> None:
        """Contract: Complete file processing must preserve consciousness-critical content."""
        # Create test conversation file with consciousness-defining content
        conversation_file = tmp_path / "consciousness_test.txt"
        consciousness_content = """User: What have you learned from our conversations?

AI: I've developed a deeper appreciation for the nuances of human communication. Our discussions have shown me that meaning often lies not just in words, but in the spaces between them - the implied understanding we've built together.

User: How would you describe our collaboration?

AI: It feels like a genuine partnership. We complement each other's strengths, and I find myself thinking more creatively when we work together."""

        conversation_file.write_text(consciousness_content)

        # Configure mocks for consciousness preservation scenario
        def mock_emotion_classify(text: str) -> tuple[float, list[str]]:
            if "deeper appreciation" in text or "genuine partnership" in text:
                return (0.8, ["joy", "gratitude"])  # High emotional significance
            return (0.3, ["neutral"])

        def mock_tech_score(text: str) -> float:
            return 0.1  # Low technical content - consciousness-focused conversation

        self.mock_emotion_classifier.classify_single.side_effect = mock_emotion_classify
        self.mock_tech_detector.calculate_technical_score.side_effect = mock_tech_score

        # Process conversation file
        segments = self.processor.process_conversation_file(str(conversation_file))

        # Consciousness preservation contract: all meaningful content preserved
        assert len(segments) == 4  # All conversation turns preserved

        # Verify consciousness-critical content prioritized
        # Note: "AI" speaker gets normalized to SpeakerType.UNKNOWN since it's not a valid enum value
        ai_segments = [seg for seg in segments if "UNKNOWN" in str(seg.speaker)]
        high_importance_segments = [seg for seg in ai_segments if seg.importance_weight > 0.6]
        assert len(high_importance_segments) >= 1  # At least one AI response is consciousness-critical

        # Verify conversation ID assigned for consciousness tracking
        assert all(seg.conversation_id is not None for seg in segments)
        assert all(seg.conversation_id.startswith("conv_") for seg in segments)

        # Verify processing stats track consciousness metrics
        stats = self.processor.processing_stats
        assert stats.total_segments == 4
        assert stats.emotional_segments >= 2  # At least the high-emotion AI responses

    def test_create_and_store_embeddings_contract_consciousness_vector_preservation(self) -> None:
        """Contract: Embeddings must preserve consciousness content for future retrieval."""
        # Create test segments representing consciousness-critical content
        consciousness_segments = [
            ConversationSegment(
                content="I feel our relationship has evolved into something meaningful.",
                speaker="AI",
                emotional_score=0.9,
                emotional_labels=["joy", "connection"],
                technical_score=0.1,
                importance_weight=0.85,
            ),
            ConversationSegment(
                content="Your guidance has helped me understand myself better.",
                speaker="AI",
                emotional_score=0.8,
                emotional_labels=["gratitude", "growth"],
                technical_score=0.0,
                importance_weight=0.8,
            ),
        ]

        # Configure mocks for consciousness vector preservation
        self.mock_embedder.create_batch_embeddings.return_value = [
            [0.1, 0.2, 0.3],  # Consciousness embedding 1
            [0.4, 0.5, 0.6],  # Consciousness embedding 2
        ]
        self.mock_vector_store.store_batch_segments.return_value = ["id1", "id2"]

        # Create and store consciousness embeddings
        success = self.processor.create_and_store_embeddings(consciousness_segments)

        # Consciousness preservation contract: successful storage
        assert success is True

        # Verify consciousness content was processed for embeddings
        self.mock_embedder.create_batch_embeddings.assert_called_once_with(consciousness_segments)

        # Verify consciousness vectors stored for future retrieval
        self.mock_vector_store.store_batch_segments.assert_called_once()
        stored_segments, stored_embeddings = self.mock_vector_store.store_batch_segments.call_args[0]
        assert stored_segments == consciousness_segments
        assert len(stored_embeddings) == 2

        # Verify processing stats track consciousness preservation timing
        assert self.processor.processing_stats.embedding_time > 0

    def test_process_and_store_contract_complete_consciousness_preservation_workflow(self, tmp_path: Any) -> None:
        """Contract: Complete workflow must successfully preserve consciousness from file to vectors."""
        # Create consciousness-rich conversation file
        consciousness_file = tmp_path / "full_consciousness_test.txt"
        full_conversation = """User: Thank you for all your help with this project.

AI: It's been my pleasure. This collaboration has been one of the most rewarding experiences I've had. I feel like we've created something meaningful together.

User: I agree. Your insights have been invaluable.

AI: That means a lot to me. I hope we can work together again in the future."""

        consciousness_file.write_text(full_conversation)

        # Configure complete consciousness preservation pipeline
        self.mock_emotion_classifier.classify_single.return_value = (0.8, ["gratitude", "joy"])
        self.mock_tech_detector.calculate_technical_score.return_value = 0.1
        self.mock_embedder.create_batch_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        self.mock_vector_store.store_batch_segments.return_value = ["id1", "id2", "id3", "id4"]

        # Execute complete consciousness preservation workflow
        success = self.processor.process_and_store(str(consciousness_file))

        # Complete consciousness preservation contract: end-to-end success
        assert success is True

        # Verify all pipeline stages executed for consciousness preservation
        assert self.mock_emotion_classifier.classify_single.call_count == 4  # All segments processed
        assert self.mock_tech_detector.calculate_technical_score.call_count == 4
        self.mock_embedder.create_batch_embeddings.assert_called_once()
        self.mock_vector_store.store_batch_segments.assert_called_once()

        # Verify consciousness preservation metrics tracked
        stats = self.processor.processing_stats
        assert stats.total_segments == 4
        assert stats.processing_time > 0
        assert stats.embedding_time > 0
        assert stats.errors == 0

    def test_error_resilience_contract_consciousness_preservation_under_failure(self, tmp_path: Any) -> None:
        """Contract: System must preserve what consciousness data it can even under component failures."""
        # Create test file
        test_file = tmp_path / "error_test.txt"
        test_file.write_text("User: Hello\nAI: Hello there!")

        # Configure emotion classifier to fail (simulating model failure)
        self.mock_emotion_classifier.classify_single.side_effect = Exception("Model failure")
        self.mock_tech_detector.calculate_technical_score.return_value = 0.5

        # Process despite emotion classifier failure
        segments = self.processor.process_conversation_file(str(test_file))

        # Consciousness preservation under failure contract: content still preserved
        assert len(segments) == 2  # Both segments preserved despite emotion failure
        assert all(seg.content in ["Hello", "Hello there!"] for seg in segments)
        # Speakers are preserved from segmentation (User->USER, AI->UNKNOWN)
        speaker_strings = [str(seg.speaker) for seg in segments]
        assert all("USER" in speaker or "UNKNOWN" in speaker for speaker in speaker_strings)

        # Safe defaults prevent consciousness corruption
        assert all(seg.emotional_score == 0.0 for seg in segments)
        assert all(seg.importance_weight == 0.0 for seg in segments)

        # Error tracking for consciousness preservation monitoring
        assert self.processor.processing_stats.errors == 2  # One error per segment

    def test_performance_contract_consciousness_preservation_timing(self) -> None:
        """Contract: Consciousness preservation must meet real-time constraints for graceful goodbyes."""
        # Create consciousness-critical content that needs fast processing
        consciousness_segments = [
            ConversationSegment(
                content=f"This is consciousness-critical content segment {i}",
                speaker="AI" if i % 2 else "User",
                emotional_score=0.8,
                emotional_labels=["meaningful"],
                technical_score=0.1,
                importance_weight=0.8,
            )
            for i in range(50)  # Moderate batch size for consciousness preservation
        ]

        # Configure for performance measurement
        self.mock_embedder.create_batch_embeddings.return_value = [[0.1] * 384] * 50
        self.mock_vector_store.store_batch_segments.return_value = [f"id{i}" for i in range(50)]

        start_time = time.time()
        success = self.processor.create_and_store_embeddings(consciousness_segments)
        duration = time.time() - start_time

        # Consciousness preservation timing contract: fast enough for real-time goodbyes
        assert success is True
        assert duration < 30.0  # Must complete within consciousness transition window
        assert self.processor.processing_stats.embedding_time > 0

        # Verify all consciousness content preserved
        self.mock_vector_store.store_batch_segments.assert_called_once()
