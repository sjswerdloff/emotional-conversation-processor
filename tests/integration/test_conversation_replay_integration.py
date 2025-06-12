"""
Integration tests for conversation replay functionality.

These tests validate the complete conversation replay system that retrieves
and reconstructs emotional context from stored conversations. Tests the
integration between vector storage, embeddings, and retrieval logic.
"""

import contextlib
import time
from typing import Any

import pytest

from emotional_processor.core.models import ConversationSegment, SpeakerType
from emotional_processor.embeddings.emotion_aware_embedder import EmotionAwareEmbedder
from emotional_processor.processors.emotion_classifier import EmotionClassifier
from emotional_processor.processors.technical_detector import TechnicalContentDetector
from emotional_processor.storage.vector_store import ConversationVectorStore


@pytest.mark.integration
@pytest.mark.requires_qdrant
class TestConversationReplayIntegration:
    """Test complete conversation replay system integration."""

    @pytest.fixture(autouse=True)
    def setup_replay_system(self, test_config: dict[str, Any]) -> None:
        """Set up complete replay system with real components."""
        # Real components for integration testing
        self.emotion_classifier = EmotionClassifier()
        self.technical_detector = TechnicalContentDetector()
        self.embedder = EmotionAwareEmbedder()

        # Real vector store with unique collection
        self.collection_name = f"replay_test_{int(time.time())}"
        self.vector_store = ConversationVectorStore(
            collection_name=self.collection_name,
            host=test_config["qdrant_host"],
            port=test_config["qdrant_port"],
            embedding_dimension=test_config["embedding_dimension"],
            enable_verification=False,
        )

    def teardown_method(self) -> None:
        """Clean up test data after each test."""
        if hasattr(self, "vector_store"):
            with contextlib.suppress(Exception):
                self.vector_store.clear_collection()

    def _create_and_store_conversation_history(self) -> list[ConversationSegment]:
        """Create and store a realistic conversation history for replay testing."""
        # Realistic conversation history with emotional arc
        conversation_data = [
            {
                "content": "I'm really struggling with this machine learning project and feeling quite overwhelmed. Could you help me?",
                "speaker": SpeakerType.USER,
                "timestamp": "2024-01-01T10:00:00Z",
                "expected_emotional": True,
                "expected_emotions": ["anxiety", "frustration"],
            },
            {
                "content": "I understand that machine learning can feel overwhelming at first. I'm here to help you break it down into manageable pieces.",
                "speaker": SpeakerType.ASSISTANT,
                "timestamp": "2024-01-01T10:01:00Z",
                "expected_emotional": True,
                "expected_emotions": ["empathy", "support"],
            },
            {
                "content": "Thank you so much for understanding. I really appreciate your patience with my questions.",
                "speaker": SpeakerType.USER,
                "timestamp": "2024-01-01T10:02:00Z",
                "expected_emotional": True,
                "expected_emotions": ["gratitude", "relief"],
            },
            {
                "content": """Here's a basic neural network implementation:
            ```python
            import numpy as np

            class NeuralNetwork:
                def __init__(self, input_size, hidden_size, output_size):
                    self.W1 = np.random.randn(input_size, hidden_size) * 0.01
                    self.W2 = np.random.randn(hidden_size, output_size) * 0.01

                def forward(self, X):
                    self.z1 = np.dot(X, self.W1)
                    self.a1 = np.tanh(self.z1)
                    self.z2 = np.dot(self.a1, self.W2)
                    return 1 / (1 + np.exp(-self.z2))
            ```""",
                "speaker": SpeakerType.ASSISTANT,
                "timestamp": "2024-01-01T10:03:00Z",
                "expected_emotional": False,
                "expected_technical": True,
            },
            {
                "content": "This is exactly what I needed! I'm so excited to try implementing this. You've made something complex seem achievable.",
                "speaker": SpeakerType.USER,
                "timestamp": "2024-01-01T10:04:00Z",
                "expected_emotional": True,
                "expected_emotions": ["joy", "excitement", "gratitude"],
            },
            {
                "content": "I'm genuinely happy that this helped build your confidence! Your enthusiasm for learning is inspiring.",
                "speaker": SpeakerType.ASSISTANT,
                "timestamp": "2024-01-01T10:05:00Z",
                "expected_emotional": True,
                "expected_emotions": ["joy", "pride"],
            },
            {
                "content": "Could you also show me how to add training functionality to this network?",
                "speaker": SpeakerType.USER,
                "timestamp": "2024-01-01T10:06:00Z",
                "expected_emotional": False,
                "expected_technical": True,
            },
            {
                "content": """Absolutely! Here's how to add backpropagation:
            ```python
            def train(self, X, y, epochs=1000, learning_rate=0.1):
                for epoch in range(epochs):
                    # Forward pass
                    output = self.forward(X)

                    # Calculate loss
                    loss = np.mean((y - output) ** 2)

                    # Backward pass
                    d_output = 2 * (output - y) / y.shape[0]
                    d_W2 = np.dot(self.a1.T, d_output)
                    d_a1 = np.dot(d_output, self.W2.T)
                    d_W1 = np.dot(X.T, d_a1 * (1 - np.tanh(self.z1) ** 2))

                    # Update weights
                    self.W2 -= learning_rate * d_W2
                    self.W1 -= learning_rate * d_W1
            ```""",
                "speaker": SpeakerType.ASSISTANT,
                "timestamp": "2024-01-01T10:07:00Z",
                "expected_emotional": False,
                "expected_technical": True,
            },
            {
                "content": "You're an amazing teacher! I feel so much more confident about machine learning now. Thank you for believing in me.",
                "speaker": SpeakerType.USER,
                "timestamp": "2024-01-01T10:08:00Z",
                "expected_emotional": True,
                "expected_emotions": ["gratitude", "joy", "confidence"],
            },
        ]

        # Process and store each segment
        stored_segments = []

        for i, data in enumerate(conversation_data):
            # Process through pipeline
            emotional_score, emotional_labels = self.emotion_classifier.classify_single(data["content"])
            technical_score = self.technical_detector.calculate_technical_score(data["content"])

            # Create segment
            segment = ConversationSegment(
                content=data["content"],
                speaker=data["speaker"],
                timestamp=data["timestamp"],
                emotional_score=emotional_score,
                emotional_labels=emotional_labels,
                technical_score=technical_score,
                importance_weight=max(0.0, emotional_score - technical_score * 0.5),
                conversation_id="conversation_history_001",
            )

            # Generate and store embedding
            embedding = self.embedder.create_contextual_embedding(segment)
            point_id = self.vector_store.store_segment(segment, embedding)

            assert point_id is not None, f"Failed to store segment {i}"
            stored_segments.append(segment)

        return stored_segments

    def test_emotional_context_retrieval_for_replay(self) -> None:
        """Test contract: system retrieves emotionally relevant context for conversation replay."""
        # Store conversation history
        stored_segments = self._create_and_store_conversation_history()
        assert len(stored_segments) >= 8, "Should store complete conversation history"

        # Test queries that should retrieve different types of emotional context
        test_queries = [
            {
                "query": "I'm feeling grateful for all the help and support",
                "expected_emotions": ["gratitude"],
                "emotional_weight": 0.8,
                "expected_segments_min": 2,
            },
            {
                "query": "I'm overwhelmed and struggling with this complexity",
                "expected_emotions": ["anxiety", "frustration"],
                "emotional_weight": 0.7,
                "expected_segments_min": 1,
            },
            {
                "query": "I'm excited and confident about learning new things",
                "expected_emotions": ["joy", "excitement", "confidence"],
                "emotional_weight": 0.8,
                "expected_segments_min": 2,
            },
            {
                "query": "Show me technical implementation details",
                "expected_emotions": [],
                "emotional_weight": 0.2,  # Low emotional weight for technical query
                "expected_technical": True,
                "expected_segments_min": 1,
            },
        ]

        for query_data in test_queries:
            # Generate query embedding
            query_embedding = self.embedder.create_query_embedding(query_data["query"])

            # Search for emotional context
            results = self.vector_store.search_emotional_context(
                query_vector=query_embedding,
                limit=5,
                emotional_weight=query_data["emotional_weight"],
            )

            # Validation: Should find relevant segments
            assert len(results) >= query_data["expected_segments_min"], (
                f"Should find at least {query_data['expected_segments_min']} segments for query: {query_data['query']}"
            )

            # Verify emotional relevance - be flexible about specific emotion labels
            # Real emotion classifiers may use different labels than expected
            if query_data["expected_emotions"]:
                emotional_matches = 0
                for result in results:
                    # For integration testing, check that segments have reasonable emotional scores
                    # rather than specific emotion labels which may vary by model
                    if result.segment.emotional_score > 0.4:
                        emotional_matches += 1

                assert emotional_matches > 0, f"Should find emotionally relevant segments for query: {query_data['query']}"

            # For technical queries, verify technical content is found
            if query_data.get("expected_technical"):
                technical_results = [r for r in results if r.segment.technical_score > 0.5]
                assert len(technical_results) > 0, "Should find technical content for technical query"

    def test_conversation_replay_maintains_emotional_arc(self) -> None:
        """Test contract: conversation replay maintains the emotional progression and relationships."""
        # Store conversation history
        self._create_and_store_conversation_history()

        # Simulate a conversation replay scenario
        # User wants to recall the emotional journey of learning ML
        replay_query = (
            "Our conversation about learning machine learning and the emotional journey from feeling overwhelmed to confident"
        )

        query_embedding = self.embedder.create_query_embedding(replay_query)

        # Retrieve comprehensive context with balanced emotional weight
        replay_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=8,
            emotional_weight=0.6,
        )

        # Validation: Should retrieve key emotional moments
        assert len(replay_results) >= 4, "Should retrieve sufficient context for replay"

        # Verify emotional progression is captured
        emotional_segments = [r for r in replay_results if r.segment.emotional_score > 0.5]
        assert len(emotional_segments) >= 3, "Should capture multiple emotional moments"

        # Check for emotional diversity in retrieved segments
        all_emotions = set()
        for result in emotional_segments:
            all_emotions.update(result.segment.emotional_labels)

        # Should capture the emotional arc: anxiety/frustration → gratitude → joy/confidence
        expected_emotion_types = ["anxiety", "gratitude", "joy", "excitement", "confidence"]
        captured_emotion_types = [emotion for emotion in expected_emotion_types if emotion in all_emotions]

        assert len(captured_emotion_types) >= 3, (
            f"Should capture diverse emotional progression. Captured: {captured_emotion_types}"
        )

        # Verify both user and assistant perspectives are included
        user_segments = [r for r in replay_results if r.segment.speaker == SpeakerType.USER]
        assistant_segments = [r for r in replay_results if r.segment.speaker == SpeakerType.ASSISTANT]

        assert len(user_segments) >= 2, "Should include user emotional expressions"
        assert len(assistant_segments) >= 2, "Should include assistant supportive responses"

    def test_replay_prioritizes_emotional_over_technical_content(self) -> None:
        """Test contract: replay correctly prioritizes emotional context over technical details."""
        # Store conversation history
        self._create_and_store_conversation_history()

        # Query that could match both emotional and technical content
        mixed_query = "Thank you for helping me understand the neural network implementation"

        query_embedding = self.embedder.create_query_embedding(mixed_query)

        # Search with high emotional weight (prioritize emotional context)
        emotional_priority_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=5,
            emotional_weight=0.9,
        )

        # Search with low emotional weight (allow more technical content)
        technical_priority_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=5,
            emotional_weight=0.1,
        )

        # Validation: Different weighting should produce different results
        assert len(emotional_priority_results) > 0
        assert len(technical_priority_results) > 0

        # High emotional weight should prioritize emotional content
        emotional_results_avg_emotion = sum(r.segment.emotional_score for r in emotional_priority_results) / len(
            emotional_priority_results
        )

        technical_results_avg_emotion = sum(r.segment.emotional_score for r in technical_priority_results) / len(
            technical_priority_results
        )

        # Emotional priority search should have higher average emotional scores
        assert emotional_results_avg_emotion >= technical_results_avg_emotion, (
            "High emotional weight should prioritize more emotional content"
        )

        # Verify emotional content is ranked higher in emotional priority search
        if len(emotional_priority_results) >= 2:
            top_emotional_result = emotional_priority_results[0]
            assert top_emotional_result.segment.emotional_score > 0.4, (
                "Top result in emotional priority search should have significant emotional content"
            )

    def test_replay_handles_context_window_limitations(self) -> None:
        """Test contract: replay system handles limited context windows appropriately."""
        # Store conversation history
        self._create_and_store_conversation_history()

        # Test different context window sizes
        context_limits = [1, 3, 5, 10]

        replay_query = "I appreciate your teaching and feel confident about learning"
        query_embedding = self.embedder.create_query_embedding(replay_query)

        for limit in context_limits:
            results = self.vector_store.search_emotional_context(
                query_vector=query_embedding,
                limit=limit,
                emotional_weight=0.7,
            )

            # Validation: Should respect the limit
            assert len(results) <= limit, f"Should respect limit of {limit}"

            # Results should be ranked by relevance
            if len(results) >= 2:
                # Check that results are reasonably ordered by relevance
                # (This is a basic check - more sophisticated ranking validation could be added)
                first_score = results[0].score if hasattr(results[0], "score") else results[0].segment.emotional_score
                second_score = results[1].score if hasattr(results[1], "score") else results[1].segment.emotional_score

                # Allow for close scores due to the complexity of emotional relevance
                assert first_score >= second_score - 0.1, "Results should be reasonably ordered by relevance"

        # Test that smaller context windows still capture essential emotional content
        small_context_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=2,
            emotional_weight=0.8,
        )

        assert len(small_context_results) >= 1, "Should find at least one relevant segment even with small context"

        # Even with limited context, should find high-quality emotional content
        if small_context_results:
            top_result = small_context_results[0]
            assert top_result.segment.emotional_score > 0.5, (
                "Limited context should still prioritize high-quality emotional content"
            )

    def test_replay_cross_conversation_boundary_handling(self) -> None:
        """Test contract: replay system correctly handles multiple conversation contexts."""
        # Store first conversation
        first_conversation_segments = self._create_and_store_conversation_history()

        # Create and store a second, different conversation
        second_conversation_data = [
            {
                "content": "I'm feeling frustrated with this debugging process and need some guidance.",
                "speaker": SpeakerType.USER,
                "emotions": ["frustration", "anxiety"],
            },
            {
                "content": "Debugging can definitely be challenging. Let's work through this systematically together.",
                "speaker": SpeakerType.ASSISTANT,
                "emotions": ["empathy", "support"],
            },
            {
                "content": "Thank you for your patience. I really value your methodical approach to problem-solving.",
                "speaker": SpeakerType.USER,
                "emotions": ["gratitude", "appreciation"],
            },
        ]

        second_conversation_segments = []
        for _i, data in enumerate(second_conversation_data):
            emotional_score, emotional_labels = self.emotion_classifier.classify_single(data["content"])
            technical_score = self.technical_detector.calculate_technical_score(data["content"])

            segment = ConversationSegment(
                content=data["content"],
                speaker=data["speaker"],
                emotional_score=emotional_score,
                emotional_labels=emotional_labels,
                technical_score=technical_score,
                importance_weight=max(0.0, emotional_score - technical_score * 0.5),
                conversation_id="conversation_history_002",
            )

            embedding = self.embedder.create_contextual_embedding(segment)
            point_id = self.vector_store.store_segment(segment, embedding)
            assert point_id is not None

            second_conversation_segments.append(segment)

        # Test cross-conversation retrieval
        cross_query = "I appreciate your patient guidance and support"
        query_embedding = self.embedder.create_query_embedding(cross_query)

        cross_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=6,
            emotional_weight=0.7,
        )

        # Validation: Should find relevant content from both conversations
        assert len(cross_results) >= 4, "Should find relevant content across conversations"

        # Verify results come from multiple conversations
        conversation_ids = {result.segment.conversation_id for result in cross_results}
        assert len(conversation_ids) >= 2, "Should retrieve content from multiple conversations"
        assert "conversation_history_001" in conversation_ids
        assert "conversation_history_002" in conversation_ids

        # Test conversation-specific retrieval
        specific_results = self.vector_store.get_conversation_segments(
            conversation_id="conversation_history_001",
        )

        # Should only return segments from the specified conversation
        assert len(specific_results) >= len(first_conversation_segments)
        for result in specific_results:
            assert result.conversation_id == "conversation_history_001"

    def test_replay_system_performance_with_large_history(self) -> None:
        """Test contract: replay system maintains performance with larger conversation histories."""
        # Store the base conversation
        self._create_and_store_conversation_history()

        # Add additional conversation segments to simulate larger history
        additional_segments_count = 30

        for i in range(additional_segments_count):
            content = f"Additional conversation segment {i} with emotional content about gratitude and learning."

            emotional_score, emotional_labels = self.emotion_classifier.classify_single(content)
            technical_score = self.technical_detector.calculate_technical_score(content)

            segment = ConversationSegment(
                content=content,
                speaker=SpeakerType.USER if i % 2 == 0 else SpeakerType.ASSISTANT,
                emotional_score=emotional_score,
                emotional_labels=emotional_labels,
                technical_score=technical_score,
                importance_weight=max(0.0, emotional_score - technical_score * 0.5),
                conversation_id=f"additional_conv_{i // 5}",  # Multiple conversations
            )

            embedding = self.embedder.create_contextual_embedding(segment)
            point_id = self.vector_store.store_segment(segment, embedding)
            assert point_id is not None

        # Test replay performance with larger dataset
        performance_query = "I'm grateful for the learning experience and support"
        query_embedding = self.embedder.create_query_embedding(performance_query)

        # Time the retrieval operation
        start_time = time.time()

        large_history_results = self.vector_store.search_emotional_context(
            query_vector=query_embedding,
            limit=10,
            emotional_weight=0.7,
        )

        retrieval_time = time.time() - start_time

        # Validation: Should complete in reasonable time
        assert retrieval_time < 5.0, f"Retrieval took too long with large history: {retrieval_time:.2f}s"

        # Should still find relevant results
        assert len(large_history_results) >= 5, "Should find relevant results even with large history"

        # Results should still be high quality
        emotional_results = [r for r in large_history_results if r.segment.emotional_score > 0.5]
        assert len(emotional_results) >= 3, "Should maintain result quality with large history"

        # Test that the system can handle multiple rapid queries (simulating active conversation replay)
        rapid_queries = [
            "Thank you for your help",
            "I feel overwhelmed",
            "This is exciting to learn",
            "I appreciate your patience",
        ]

        rapid_start_time = time.time()

        for query in rapid_queries:
            query_emb = self.embedder.create_query_embedding(query)
            results = self.vector_store.search_emotional_context(
                query_vector=query_emb,
                limit=5,
                emotional_weight=0.6,
            )
            assert len(results) > 0, f"Should find results for rapid query: {query}"

        rapid_total_time = time.time() - rapid_start_time

        # Multiple rapid queries should complete quickly
        assert rapid_total_time < 10.0, f"Rapid queries took too long: {rapid_total_time:.2f}s"
