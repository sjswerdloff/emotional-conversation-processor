"""
Mission-critical contract tests for conversation segmentation in medical software.

These tests verify that conversation parsing maintains reliability and accuracy
required for clinical conversation context replay systems. Focus on contracts
that the ConversationProcessor depends on for medical software operation.

Following medical software testing philosophy: Test contracts, not implementation.
"""

from emotional_processor.processors.conversation_segmenter import ConversationSegmenter, ConversationTurn


class TestConversationSegmenterCriticalContracts:
    """Test contracts that ConversationProcessor depends on for medical software reliability."""

    def test_segmenter_initialization_contract(self) -> None:
        """Test contract: segmenter initializes and is ready for segmentation."""
        segmenter = ConversationSegmenter()

        # Contract verification: segmenter is ready to process conversations
        assert hasattr(segmenter, "claude_desktop_patterns")
        assert hasattr(segmenter, "standard_patterns")
        assert isinstance(segmenter.claude_desktop_patterns, list)
        assert isinstance(segmenter.standard_patterns, list)
        assert len(segmenter.claude_desktop_patterns) > 0
        assert len(segmenter.standard_patterns) > 0

    def test_simple_conversation_segmentation_contract(self) -> None:
        """Test contract: simple conversation format produces correct turns."""
        segmenter = ConversationSegmenter()

        conversation = """User: Hello, how are you today?"""
        turns = segmenter.segment_conversation(conversation)

        # Contract verification: correct number of turns
        assert len(turns) == 1
        assert turns[0].speaker == "User"
        assert turns[0].content == "Hello, how are you today?"

    def test_multi_turn_conversation_contract_deterministic_parsing(self) -> None:
        """Test contract: multi-turn conversations are parsed deterministically."""
        segmenter = ConversationSegmenter()

        # Medical context: patient-provider conversation
        conversation = """User: I'm experiencing chest pain and feeling anxious about it.
Assistant: I understand your concern. Can you describe the type of pain?
User: It's a sharp pain that comes and goes. I'm worried it might be serious.
Assistant: Let's discuss some immediate steps and when to seek emergency care."""

        turns = segmenter.segment_conversation(conversation)

        # Contract: deterministic parsing for medical reliability
        assert len(turns) == 4
        assert turns[0].speaker == "User"
        assert turns[1].speaker == "Assistant"
        assert turns[2].speaker == "User"
        assert turns[3].speaker == "Assistant"

        # Contract: content preservation for medical documentation
        assert "chest pain" in turns[0].content
        assert "describe the type of pain" in turns[1].content
        assert "sharp pain" in turns[2].content
        assert "emergency care" in turns[3].content

    def test_timestamp_preservation_contract_medical_documentation(self) -> None:
        """Test contract: timestamps are preserved for medical audit trails."""
        segmenter = ConversationSegmenter()

        # Medical context: timestamped patient consultation
        conversation = """User [2024-01-15T14:30:00Z]: I need to discuss my medication side effects.
Assistant [2024-01-15T14:30:15Z]: Of course. Which medication are you concerned about?
User [2024-01-15T14:31:00Z]: The blood pressure medication is causing dizziness."""

        turns = segmenter.segment_conversation(conversation)

        # Contract: timestamp preservation for medical audit trails
        assert len(turns) == 3
        assert turns[0].timestamp == "2024-01-15T14:30:00Z"
        assert turns[1].timestamp == "2024-01-15T14:30:15Z"
        assert turns[2].timestamp == "2024-01-15T14:31:00Z"

        # Contract: medical content preservation
        assert "medication side effects" in turns[0].content
        assert "blood pressure medication" in turns[2].content

    def test_mixed_content_parsing_contract_resilience(self) -> None:
        """Test contract: mixed emotional and technical content is parsed without errors."""
        segmenter = ConversationSegmenter()

        # Mixed emotional and technical content (realistic medical software context)
        conversation = """User: I'm struggling with implementing the patient data validation.
Assistant: I understand your frustration. Let me help with that code.
User: Thank you! This helps reduce my anxiety about the project."""

        turns = segmenter.segment_conversation(conversation)

        # Contract: conversation is parsed without errors
        assert len(turns) >= 2

        # Contract: speakers correctly identified
        speakers = [turn.speaker for turn in turns]
        assert "User" in speakers
        assert "Assistant" in speakers

        # Contract: emotional and technical content preserved
        contents = [turn.content for turn in turns]
        assert any("struggling" in content for content in contents)
        assert any("anxiety" in content for content in contents)

    def test_malformed_input_resilience_contract_medical_safety(self) -> None:
        """Test contract: malformed input should not crash medical software."""
        segmenter = ConversationSegmenter()

        # Medical safety: malformed conversations should be handled gracefully
        malformed_inputs = [
            "",  # Empty string
            "   \n\t   ",  # Whitespace only
            "Random text without speaker labels",  # No clear structure
            "User: \nAssistant:",  # Empty messages
            "Patient[broken timestamp: I need help",  # Malformed timestamp
        ]

        for malformed_input in malformed_inputs:
            # Contract: should not crash, should return some valid structure
            turns = segmenter.segment_conversation(malformed_input)
            assert isinstance(turns, list)

            # Contract: all returned turns should be valid ConversationTurn objects
            for turn in turns:
                assert isinstance(turn, ConversationTurn)
                assert hasattr(turn, "content")
                assert hasattr(turn, "speaker")
                assert hasattr(turn, "timestamp")

    def test_unicode_safety_contract_international_medical_content(self) -> None:
        """Test contract: international content is handled safely for global medical systems."""
        segmenter = ConversationSegmenter()

        # Medical software international requirement
        international_conversation = """User: Merci docteur pour votre aide médicale.
Assistant: De rien! Comment vous sentez-vous aujourd'hui?
User: 谢谢医生的耐心解释治疗方案。
Assistant: 很高兴能帮助您。有什么其他问题吗？
User: Спасибо за медицинскую консультацию."""

        turns = segmenter.segment_conversation(international_conversation)

        # Contract: international content processed safely
        assert len(turns) == 5

        # Contract: all turns should be valid
        for turn in turns:
            assert isinstance(turn, ConversationTurn)
            assert turn.content.strip()  # Should have content
            assert turn.speaker in ["User", "Assistant"]

        # Contract: specific international content preserved
        contents = [turn.content for turn in turns]
        assert any("Merci docteur" in content for content in contents)
        assert any("谢谢医生" in content for content in contents)
        assert any("Спасибо" in content for content in contents)

    def test_conversation_stats_contract_medical_documentation(self) -> None:
        """Test contract: conversation statistics provide medical documentation requirements."""
        segmenter = ConversationSegmenter()

        # Medical consultation conversation
        conversation = """User: I'm experiencing severe headaches for the past week.
Assistant: I'm sorry to hear about your headaches. Can you describe the pain?
User: It's throbbing, mainly on the left side. Gets worse with light.
Assistant: Thank you for those details. Let's discuss potential triggers."""

        turns = segmenter.segment_conversation(conversation)
        stats = segmenter.get_conversation_stats(turns)

        # Contract: required medical documentation statistics
        assert isinstance(stats, dict)
        assert "total_turns" in stats
        assert "speakers" in stats
        assert "total_words" in stats

        # Contract: accurate statistics for medical records
        assert stats["total_turns"] == len(turns)
        assert "User" in stats["speakers"]
        assert "Assistant" in stats["speakers"]
        assert stats["total_words"] > 0

        # Contract: medical keywords tracking (if implemented)
        if "keywords" in stats:
            assert isinstance(stats["keywords"], list)


class TestConversationSegmenterMedicalComplianceContracts:
    """Test contracts ensuring compliance with medical software reliability requirements."""

    def test_content_preservation_contract_data_integrity(self) -> None:
        """Test contract: original content is preserved for medical audit trails."""
        segmenter = ConversationSegmenter()

        # Medical content with critical information
        critical_content = "Patient reports severe chest pain radiating to left arm, onset 30 minutes ago"
        conversation = f"User: {critical_content}"

        turns = segmenter.segment_conversation(conversation)

        # Contract: critical medical content must be preserved exactly
        assert len(turns) == 1
        assert critical_content in turns[0].content
        assert turns[0].content.strip() == critical_content

    def test_speaker_normalization_contract_consistency(self) -> None:
        """Test contract: speaker names are normalized consistently for medical records."""
        segmenter = ConversationSegmenter()

        # Various speaker name formats that should be normalized
        conversation = """human: I have a medical question.
ASSISTANT: I'm here to help.
user: What about this symptom?
ai: Let me provide information."""

        turns = segmenter.segment_conversation(conversation)

        # Contract: speaker names should be normalized consistently
        speakers = [turn.speaker for turn in turns]

        # Contract: normalized speakers should be consistent
        for speaker in speakers:
            assert speaker in ["Human", "Assistant", "User", "AI", "System"]
            # Should be properly capitalized
            assert speaker[0].isupper()

    def test_deterministic_behavior_contract_medical_reproducibility(self) -> None:
        """Test contract: segmentation is deterministic for medical reproducibility."""
        segmenter = ConversationSegmenter()

        # Medical consultation content
        conversation = """User: I'm worried about these test results.
Assistant: Let's review them together. What specific concerns do you have?
User: The blood work shows elevated levels. Should I be concerned?"""

        # Contract: multiple segmentations should produce identical results
        turns1 = segmenter.segment_conversation(conversation)
        turns2 = segmenter.segment_conversation(conversation)
        turns3 = segmenter.segment_conversation(conversation)

        # Contract: deterministic behavior for medical reliability
        assert len(turns1) == len(turns2) == len(turns3)

        for i in range(len(turns1)):
            assert turns1[i].speaker == turns2[i].speaker == turns3[i].speaker
            assert turns1[i].content == turns2[i].content == turns3[i].content
            assert turns1[i].timestamp == turns2[i].timestamp == turns3[i].timestamp

    def test_performance_contract_medical_responsiveness(self) -> None:
        """Test contract: segmentation performance meets medical software requirements."""
        import time

        segmenter = ConversationSegmenter()

        # Medical software requirement: process conversations of various sizes efficiently
        conversations = [
            "User: Short question.\nAssistant: Short answer.",
            "User: Medium question about symptoms.\nAssistant: " + "Detailed response. " * 50,
            "User: Long consultation notes.\nAssistant: " + "Very detailed medical information. " * 200,
        ]

        for conversation in conversations:
            start_time = time.time()
            turns = segmenter.segment_conversation(conversation)
            processing_time = time.time() - start_time

            # Contract: processing should complete quickly (< 1 second for typical conversations)
            assert processing_time < 1.0, f"Segmentation too slow: {processing_time:.3f}s for {len(conversation)} chars"

            # Contract: result should still be valid
            assert isinstance(turns, list)
            assert len(turns) > 0
            assert all(isinstance(turn, ConversationTurn) for turn in turns)
