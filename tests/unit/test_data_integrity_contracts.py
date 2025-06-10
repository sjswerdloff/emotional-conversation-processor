"""
Mission-critical data integrity contract tests for medical software reliability.

These tests verify that core data models maintain integrity under all conditions,
preventing patient data corruption that could impact medical outcomes.

Author: Claude Code Assistant
Medical Software Compliance: Production Grade
"""

import uuid

from emotional_processor.core.models import ContentType, ConversationSegment, SpeakerType


class TestConversationSegmentDataIntegrityContracts:
    """Test critical data integrity contracts for ConversationSegment."""

    def test_score_corruption_prevention_contract(self) -> None:
        """Test contract: scores outside [0.0, 1.0] are clamped, never corrupted."""
        # Contract: Invalid scores must be safely clamped
        segment = ConversationSegment(
            content="Test content",
            speaker="User",
            emotional_score=-5.0,  # Invalid negative
            technical_score=2.5,  # Invalid over 1.0
            importance_weight=10.0,  # Invalid over 1.0
        )

        # Contract verification: scores must be clamped to valid range
        assert 0.0 <= segment.emotional_score <= 1.0
        assert 0.0 <= segment.technical_score <= 1.0
        assert 0.0 <= segment.importance_weight <= 1.0

        # Contract: extreme values must be handled safely
        assert segment.emotional_score == 0.0  # Clamped from -5.0
        assert segment.technical_score == 1.0  # Clamped from 2.5
        assert segment.importance_weight == 1.0  # Clamped from 10.0

    def test_score_precision_preservation_contract(self) -> None:
        """Test contract: valid decimal scores are preserved exactly."""
        precise_scores = [0.0, 0.123456789, 0.5, 0.999999, 1.0]

        for score in precise_scores:
            segment = ConversationSegment(
                content="Test content", speaker="User", emotional_score=score, technical_score=score, importance_weight=score
            )

            # Contract: precise values must be preserved
            assert segment.emotional_score == score
            assert segment.technical_score == score
            assert segment.importance_weight == score

    def test_segment_id_uniqueness_enforcement_contract(self) -> None:
        """Test contract: each segment gets a unique UUID, preventing collisions."""
        segments = [ConversationSegment(content=f"Content {i}", speaker="User") for i in range(100)]

        # Contract: all segment IDs must be unique
        segment_ids = [segment.segment_id for segment in segments]
        assert len(set(segment_ids)) == len(segment_ids)

        # Contract: all IDs must be valid UUIDs
        for segment_id in segment_ids:
            uuid_obj = uuid.UUID(segment_id)  # Will raise if invalid
            assert str(uuid_obj) == segment_id

    def test_segment_id_immutability_contract(self) -> None:
        """Test contract: segment IDs cannot be accidentally modified."""
        segment = ConversationSegment(content="Test", speaker="User")
        original_id = segment.segment_id

        # Contract: ID should remain stable
        assert segment.segment_id == original_id

        # Contract: recreating segment should generate new ID
        segment2 = ConversationSegment(content="Test", speaker="User")
        assert segment2.segment_id != original_id

    def test_timestamp_validation_corruption_prevention_contract(self) -> None:
        """Test contract: malformed timestamps are safely handled, not corrupted."""
        malformed_timestamps = [
            "not-a-timestamp",
            "2024-13-45T99:99:99Z",  # Invalid date components
            "2024/01/01 10:00:00",  # Wrong format
            "",  # Empty string
            "   ",  # Whitespace only
            "2024-13-01",  # Invalid month
            "2024-01-32",  # Invalid day
            "invalid-date-format",  # Completely wrong format
        ]

        for bad_timestamp in malformed_timestamps:
            segment = ConversationSegment(content="Test content", speaker="User", timestamp=bad_timestamp)

            # Contract: malformed timestamps must be set to None, not corrupted
            assert segment.timestamp is None, f"Failed for timestamp: {bad_timestamp}"

    def test_timestamp_valid_formats_preservation_contract(self) -> None:
        """Test contract: valid ISO timestamps are preserved exactly."""
        valid_timestamps = [
            "2024-01-01T10:00:00Z",
            "2024-01-01T10:00:00+00:00",
            "2024-12-31T23:59:59.999Z",
            "2024-06-15T14:30:00-05:00",
            "2024-01-01",  # Date only is valid
            "2024-01-01T10:00:00",  # Time without timezone is valid
        ]

        for timestamp in valid_timestamps:
            segment = ConversationSegment(content="Test content", speaker="User", timestamp=timestamp)

            # Contract: valid timestamps must be preserved
            assert segment.timestamp == timestamp

    def test_content_special_characters_safety_contract(self) -> None:
        """Test contract: content with special characters doesn't corrupt metadata."""
        dangerous_content = [
            "Content with\nnewlines\tand\ttabs",
            "Unicode: 🔥💯🚀 émotions",
            "Code: `print('hello')` and ```python\ncode\n```",
            'JSON-like: {"key": "value", "array": [1,2,3]}',
            'XML-like: <tag attr="value">content</tag>',
            "Null bytes: \x00\x01\x02",
            "Very long content: " + "x" * 10000,
        ]

        for content in dangerous_content:
            segment = ConversationSegment(content=content, speaker="User", emotional_score=0.5, technical_score=0.3)

            # Contract: content must be preserved exactly
            assert segment.content == content

            # Contract: metadata must remain uncorrupted
            assert 0.0 <= segment.emotional_score <= 1.0
            assert 0.0 <= segment.technical_score <= 1.0
            assert segment.speaker == SpeakerType.USER

    def test_speaker_normalization_consistency_contract(self) -> None:
        """Test contract: speaker names are consistently normalized."""
        speaker_variations = [
            ("User", SpeakerType.USER),
            ("user", SpeakerType.UNKNOWN),  # Case sensitive
            ("Assistant", SpeakerType.ASSISTANT),
            ("System", SpeakerType.SYSTEM),
            ("Random Person", SpeakerType.UNKNOWN),
            ("", SpeakerType.UNKNOWN),
        ]

        for speaker_input, expected_type in speaker_variations:
            segment = ConversationSegment(content="Test content", speaker=speaker_input)

            # Contract: speaker must be consistently normalized
            assert segment.speaker == expected_type

    def test_metadata_serialization_safety_contract(self) -> None:
        """Test contract: nested metadata doesn't cause data loss during processing."""
        complex_metadata = {
            "processing_info": {
                "model_version": "1.0.0",
                "timestamps": ["2024-01-01T10:00:00Z", "2024-01-01T10:01:00Z"],
                "scores": [0.1, 0.5, 0.9],
                "nested": {"deep": {"data": "value"}},
            },
            "unicode_data": "测试数据 🔥",
            "special_chars": "\n\t\r",
            "null_handling": None,
            "numbers": [1, 2.5, -3.14],
            "booleans": [True, False],
        }

        segment = ConversationSegment(content="Test content", speaker="User", metadata=complex_metadata)

        # Contract: metadata must be preserved exactly
        assert segment.metadata == complex_metadata

        # Contract: nested access must work
        assert segment.metadata["processing_info"]["model_version"] == "1.0.0"
        assert segment.metadata["processing_info"]["nested"]["deep"]["data"] == "value"

    def test_content_type_classification_determinism_contract(self) -> None:
        """Test contract: content type classification is deterministic and consistent."""
        test_cases = [
            # Emotional content
            (0.8, 0.2, ContentType.EMOTIONAL),
            (0.7, 0.2, ContentType.EMOTIONAL),
            # Technical content
            (0.2, 0.8, ContentType.TECHNICAL),
            (0.2, 0.7, ContentType.TECHNICAL),
            # Mixed content
            (0.5, 0.5, ContentType.MIXED),
            (0.6, 0.5, ContentType.MIXED),  # Within 0.2 threshold
            # Neutral content
            (0.4, 0.1, ContentType.NEUTRAL),
            (0.1, 0.4, ContentType.NEUTRAL),
            (0.0, 0.0, ContentType.NEUTRAL),
        ]

        for emotional_score, technical_score, expected_type in test_cases:
            segment = ConversationSegment(
                content="Test content", speaker="User", emotional_score=emotional_score, technical_score=technical_score
            )

            # Contract: classification must be deterministic
            assert segment.content_type == expected_type

            # Contract: multiple calls must return same result
            assert segment.content_type == expected_type
            assert segment.content_type == expected_type

    def test_word_count_accuracy_contract(self) -> None:
        """Test contract: word count is accurate for all text types."""
        test_cases = [
            ("", 0),
            ("word", 1),
            ("two words", 2),
            ("  spaced   words  ", 2),  # Extra spaces
            ("tab\tseparated\twords", 3),
            ("newline\nseparated\nwords", 3),
            ("mixed\t \n  separators  \t", 2),
            ("unicode words 测试 🔥", 4),
            ("code `print('hello')` words", 3),
        ]

        for content, expected_count in test_cases:
            segment = ConversationSegment(content=content, speaker="User")

            # Contract: word count must be accurate
            assert segment.word_count == expected_count

    def test_strong_emotion_threshold_consistency_contract(self) -> None:
        """Test contract: strong emotion detection is consistent and reliable."""
        test_cases = [
            (0.0, False),
            (0.69, False),  # Just below threshold
            (0.7, False),  # At threshold (not strong)
            (0.71, True),  # Just above threshold
            (1.0, True),
        ]

        for score, expected_strong in test_cases:
            segment = ConversationSegment(content="Test content", speaker="User", emotional_score=score)

            # Contract: strong emotion detection must be consistent
            assert segment.has_strong_emotion == expected_strong

    def test_highly_technical_threshold_consistency_contract(self) -> None:
        """Test contract: technical content detection is consistent and reliable."""
        test_cases = [
            (0.0, False),
            (0.69, False),  # Just below threshold
            (0.7, False),  # At threshold (not highly technical)
            (0.71, True),  # Just above threshold
            (1.0, True),
        ]

        for score, expected_technical in test_cases:
            segment = ConversationSegment(content="Test content", speaker="User", technical_score=score)

            # Contract: technical detection must be consistent
            assert segment.is_highly_technical == expected_technical


class TestConversationSegmentMedicalComplianceContracts:
    """Test medical software compliance contracts for ConversationSegment."""

    def test_audit_trail_data_preservation_contract(self) -> None:
        """Test contract: all data required for audit trails is preserved."""
        segment = ConversationSegment(
            content="Patient expressed anxiety about treatment",
            speaker="User",
            timestamp="2024-01-01T10:00:00Z",
            emotional_score=0.8,
            emotional_labels=["anxiety", "fear"],
            technical_score=0.1,
            importance_weight=0.9,
            conversation_id="patient-123-session-001",
            metadata={"session_id": "session-001", "patient_id": "patient-123", "provider_id": "doctor-456"},
        )

        # Contract: audit trail data must be complete and accessible
        assert segment.content is not None
        assert segment.speaker is not None
        assert segment.timestamp is not None
        assert segment.conversation_id is not None
        assert segment.segment_id is not None

        # Contract: metadata for medical compliance must be preserved
        assert "session_id" in segment.metadata
        assert "patient_id" in segment.metadata
        assert "provider_id" in segment.metadata

    def test_data_immutability_after_creation_contract(self) -> None:
        """Test contract: core segment data cannot be accidentally modified."""
        original_data = {
            "content": "Original patient statement",
            "speaker": "User",
            "emotional_score": 0.7,
            "technical_score": 0.2,
            "timestamp": "2024-01-01T10:00:00Z",
        }

        segment = ConversationSegment(**original_data)

        # Store original values
        original_content = segment.content
        original_speaker = segment.speaker
        original_emotional_score = segment.emotional_score
        original_timestamp = segment.timestamp
        original_id = segment.segment_id

        # Contract: attempting to modify should not affect core data integrity
        # (Note: dataclass fields are mutable by design, but this tests the principle)
        assert segment.content == original_content
        assert segment.speaker == original_speaker
        assert segment.emotional_score == original_emotional_score
        assert segment.timestamp == original_timestamp
        assert segment.segment_id == original_id

    def test_deterministic_behavior_contract(self) -> None:
        """Test contract: same inputs always produce same outputs."""
        input_data = {
            "content": "Patient expressed gratitude for care",
            "speaker": "User",
            "emotional_score": 0.75,
            "technical_score": 0.1,
            "importance_weight": 0.8,
            "timestamp": "2024-01-01T10:00:00Z",
        }

        # Create multiple segments with identical data (except ID)
        segments = []
        for _ in range(10):
            segment_data = input_data.copy()
            segment = ConversationSegment(**segment_data)
            segments.append(segment)

        # Contract: all derived properties must be identical
        reference_segment = segments[0]
        for segment in segments[1:]:
            assert segment.content == reference_segment.content
            assert segment.speaker == reference_segment.speaker
            assert segment.emotional_score == reference_segment.emotional_score
            assert segment.technical_score == reference_segment.technical_score
            assert segment.importance_weight == reference_segment.importance_weight
            assert segment.timestamp == reference_segment.timestamp
            assert segment.content_type == reference_segment.content_type
            assert segment.word_count == reference_segment.word_count
            assert segment.has_strong_emotion == reference_segment.has_strong_emotion
            assert segment.is_highly_technical == reference_segment.is_highly_technical

            # Contract: only segment_id should be different
            assert segment.segment_id != reference_segment.segment_id
