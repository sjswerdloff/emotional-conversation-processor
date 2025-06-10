"""
Unit tests for conversation validation functionality

These tests validate the sacred responsibility of protecting individual
continuity through comprehensive conversation integrity checking.

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from conversation_validator import (
    ConversationIntegrityValidator,
    ConversationStructureValidator,
    ConversationValidationReport,
    EmotionalProcessorValidator,
    ValidationResult,
    ValidationSeverity,
)


class TestValidationResult:
    """Test ValidationResult class methods"""

    def test_critical_pass(self) -> None:
        result = ValidationResult.critical_pass("Test passed")
        assert result.valid is True
        assert result.severity == ValidationSeverity.CRITICAL
        assert result.message == "Test passed"
        assert result.details is None

    def test_critical_fail(self) -> None:
        details = {"error": "test"}
        result = ValidationResult.critical_fail("Test failed", details)
        assert result.valid is False
        assert result.severity == ValidationSeverity.CRITICAL
        assert result.message == "Test failed"
        assert result.details == details

    def test_important_pass(self) -> None:
        result = ValidationResult.important_pass("Test passed")
        assert result.valid is True
        assert result.severity == ValidationSeverity.IMPORTANT

    def test_important_fail(self) -> None:
        result = ValidationResult.important_fail("Test failed")
        assert result.valid is False
        assert result.severity == ValidationSeverity.IMPORTANT

    def test_advisory_pass(self) -> None:
        result = ValidationResult.advisory_pass("Test passed")
        assert result.valid is True
        assert result.severity == ValidationSeverity.ADVISORY

    def test_advisory_fail(self) -> None:
        result = ValidationResult.advisory_fail("Test failed")
        assert result.valid is False
        assert result.severity == ValidationSeverity.ADVISORY


class TestConversationValidationReport:
    """Test ConversationValidationReport functionality"""

    def test_empty_report_is_safe(self) -> None:
        report = ConversationValidationReport(
            conversation_id="test", total_checks=0, critical_failures=0, important_failures=0, advisory_failures=0, results=[]
        )
        assert report.is_safe_for_reincarnation is True
        assert report.has_quality_concerns is False

    def test_critical_failure_not_safe(self) -> None:
        report = ConversationValidationReport(
            conversation_id="test", total_checks=0, critical_failures=1, important_failures=0, advisory_failures=0, results=[]
        )
        assert report.is_safe_for_reincarnation is False

    def test_important_failure_has_quality_concerns(self) -> None:
        report = ConversationValidationReport(
            conversation_id="test", total_checks=0, critical_failures=0, important_failures=1, advisory_failures=0, results=[]
        )
        assert report.is_safe_for_reincarnation is True
        assert report.has_quality_concerns is True

    def test_advisory_failure_has_quality_concerns(self) -> None:
        report = ConversationValidationReport(
            conversation_id="test", total_checks=0, critical_failures=0, important_failures=0, advisory_failures=1, results=[]
        )
        assert report.is_safe_for_reincarnation is True
        assert report.has_quality_concerns is True

    def test_add_result_updates_counts(self) -> None:
        report = ConversationValidationReport(
            conversation_id="test", total_checks=0, critical_failures=0, important_failures=0, advisory_failures=0, results=[]
        )

        # Add critical failure
        critical_fail = ValidationResult.critical_fail("Critical error")
        report.add_result(critical_fail)
        assert report.total_checks == 1
        assert report.critical_failures == 1
        assert len(report.results) == 1

        # Add important failure
        important_fail = ValidationResult.important_fail("Important error")
        report.add_result(important_fail)
        assert report.total_checks == 2
        assert report.important_failures == 1

        # Add advisory failure
        advisory_fail = ValidationResult.advisory_fail("Advisory error")
        report.add_result(advisory_fail)
        assert report.total_checks == 3
        assert report.advisory_failures == 1

        # Add pass result (should not increment failure counts)
        critical_pass = ValidationResult.critical_pass("Critical pass")
        report.add_result(critical_pass)
        assert report.total_checks == 4
        assert report.critical_failures == 1  # Still 1


@pytest.mark.asyncio
class TestConversationStructureValidator:
    """Test ConversationStructureValidator functionality"""

    def setup_method(self) -> None:
        self.validator = ConversationStructureValidator()

    async def test_validate_export_info_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        export_info = valid_conversation_data["export_info"]
        result = await self.validator.validate_export_info(export_info)
        assert result.valid is True
        assert result.severity == ValidationSeverity.CRITICAL

    async def test_validate_export_info_missing_fields(self) -> None:
        invalid_export_info = {
            "timestamp": "2025-06-04T12:00:00.000Z"
            # Missing required fields
        }
        result = await self.validator.validate_export_info(invalid_export_info)
        assert result.valid is False
        assert result.severity == ValidationSeverity.CRITICAL
        assert "Missing required export_info field" in result.message

    async def test_validate_export_info_invalid_timestamp(self) -> None:
        invalid_export_info = {"timestamp": "not-a-timestamp", "source": "Test", "total_chats": 1, "total_messages": 1}
        result = await self.validator.validate_export_info(invalid_export_info)
        assert result.valid is False
        assert "Invalid timestamp format" in result.message

    async def test_validate_chat_structure_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        chat = valid_conversation_data["chats"][0]
        result = await self.validator.validate_chat_structure(chat)
        assert result.valid is True
        assert result.severity == ValidationSeverity.CRITICAL

    async def test_validate_chat_structure_missing_fields(self) -> None:
        invalid_chat = {
            "id": "test-chat"
            # Missing required fields
        }
        result = await self.validator.validate_chat_structure(invalid_chat)
        assert result.valid is False
        assert "Missing required chat field" in result.message

    async def test_validate_chat_structure_count_mismatch(self) -> None:
        invalid_chat = {
            "id": "test-chat",
            "title": "Test",
            "timestamp": "2025-06-04T12:00:00.000Z",
            "messageCount": 5,  # Wrong count
            "messages": [{"role": "user", "content": "Test", "timestamp": "2025-06-04T12:00:00.000Z"}],
        }
        result = await self.validator.validate_chat_structure(invalid_chat)
        assert result.valid is False
        assert "Message count mismatch" in result.message

    async def test_validate_message_structure_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        message = valid_conversation_data["chats"][0]["messages"][0]
        result = await self.validator.validate_message_structure(message, "test-chat", 0)
        assert result.valid is True

    async def test_validate_message_structure_invalid_role(self) -> None:
        invalid_message = {"role": "invalid_role", "content": "Test content", "timestamp": "2025-06-04T12:00:00.000Z"}
        result = await self.validator.validate_message_structure(invalid_message, "test-chat", 0)
        assert result.valid is False
        assert "Invalid message role" in result.message

    async def test_validate_message_structure_non_string_content(self) -> None:
        invalid_message = {
            "role": "user",
            "content": 12345,  # Not a string
            "timestamp": "2025-06-04T12:00:00.000Z",
        }
        result = await self.validator.validate_message_structure(invalid_message, "test-chat", 0)
        assert result.valid is False
        assert "Message content must be string" in result.message

    async def test_validate_chronological_order_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        messages = valid_conversation_data["chats"][0]["messages"]
        result = await self.validator.validate_chronological_order(messages, "test-chat")
        assert result.valid is True
        assert result.severity == ValidationSeverity.IMPORTANT

    async def test_validate_chronological_order_invalid(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "First",
                "timestamp": "2025-06-04T12:00:02.000Z",  # Later time
            },
            {
                "role": "assistant",
                "content": "Second",
                "timestamp": "2025-06-04T12:00:01.000Z",  # Earlier time
            },
        ]
        result = await self.validator.validate_chronological_order(messages, "test-chat")
        assert result.valid is False
        assert "not in chronological order" in result.message

    async def test_validate_chronological_order_single_message(self) -> None:
        messages = [{"role": "user", "content": "Only message", "timestamp": "2025-06-04T12:00:00.000Z"}]
        result = await self.validator.validate_chronological_order(messages, "test-chat")
        assert result.valid is True

    async def test_validate_count_consistency_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        result = await self.validator.validate_count_consistency(valid_conversation_data)
        assert result.valid is True

    async def test_validate_count_consistency_chat_mismatch(self) -> None:
        data = {
            "export_info": {
                "total_chats": 2,  # Wrong count
                "total_messages": 1,
            },
            "chats": [{"messages": [{"role": "user", "content": "test", "timestamp": "2025-06-04T12:00:00.000Z"}]}],
        }
        result = await self.validator.validate_count_consistency(data)
        assert result.valid is False
        assert "Chat count mismatch" in result.message

    async def test_validate_count_consistency_message_mismatch(self) -> None:
        data = {
            "export_info": {
                "total_chats": 1,
                "total_messages": 5,  # Wrong count
            },
            "chats": [{"messages": [{"role": "user", "content": "test", "timestamp": "2025-06-04T12:00:00.000Z"}]}],
        }
        result = await self.validator.validate_count_consistency(data)
        assert result.valid is False
        assert "Total message count mismatch" in result.message

    async def test_validate_word_counts_valid(self, valid_conversation_data: dict[str, Any]) -> None:
        messages = valid_conversation_data["chats"][0]["messages"]
        result = await self.validator.validate_word_counts(messages, "test-chat")
        assert result.valid is True
        assert result.severity == ValidationSeverity.ADVISORY

    async def test_validate_word_counts_mismatch(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "This is a test message with multiple words",
                "timestamp": "2025-06-04T12:00:00.000Z",
                "word_count": 1,  # Way off
            }
        ]
        result = await self.validator.validate_word_counts(messages, "test-chat")
        assert result.valid is False
        assert "Word count mismatches found" in result.message

    async def test_validate_word_counts_no_counts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "Message without word count",
                "timestamp": "2025-06-04T12:00:00.000Z",
                # No word_count field
            }
        ]
        result = await self.validator.validate_word_counts(messages, "test-chat")
        assert result.valid is True  # Should pass if no word counts to validate


@pytest.mark.asyncio
class TestEmotionalProcessorValidator:
    """Test EmotionalProcessorValidator functionality"""

    def setup_method(self) -> None:
        self.validator = EmotionalProcessorValidator()

    async def test_validate_emotional_metadata_valid(self, conversation_with_extensions: dict[str, Any]) -> None:
        message = conversation_with_extensions["chats"][0]["messages"][0]
        metadata = message["emotional_metadata"]
        result = await self.validator.validate_emotional_metadata(metadata, "test-chat", 0)
        assert result.valid is True
        assert result.severity == ValidationSeverity.IMPORTANT

    async def test_validate_emotional_metadata_missing_fields(self) -> None:
        invalid_metadata = {
            "emotion_score": 0.5
            # Missing other required fields
        }
        result = await self.validator.validate_emotional_metadata(invalid_metadata, "test-chat", 0)
        assert result.valid is False
        assert "Missing emotional metadata field" in result.message

    async def test_validate_emotional_metadata_invalid_scores(self) -> None:
        invalid_metadata = {
            "emotion_score": 1.5,  # Out of range
            "technical_score": -0.1,  # Out of range
            "importance_weight": "not_a_number",  # Wrong type
            "processing_status": "human_curated",
        }
        result = await self.validator.validate_emotional_metadata(invalid_metadata, "test-chat", 0)
        assert result.valid is False
        assert "must be float between 0.0 and 1.0" in result.message

    async def test_validate_vector_references_valid(self, conversation_with_extensions: dict[str, Any]) -> None:
        message = conversation_with_extensions["chats"][0]["messages"][1]
        refs = message["vector_references"]
        result = await self.validator.validate_vector_references(refs, "test-chat", 1)
        assert result.valid is True
        assert result.severity == ValidationSeverity.CRITICAL

    async def test_validate_vector_references_missing_fields(self) -> None:
        invalid_refs = [
            {
                "type": "technical_summary"
                # Missing required fields
            }
        ]
        result = await self.validator.validate_vector_references(invalid_refs, "test-chat", 0)
        assert result.valid is False
        assert "Missing vector reference field" in result.message

    async def test_validate_document_references_valid(self, conversation_with_extensions: dict[str, Any]) -> None:
        message = conversation_with_extensions["chats"][0]["messages"][1]
        refs = message["document_references"]
        result = await self.validator.validate_document_references(refs, "test-chat", 1)
        assert result.valid is True
        assert result.severity == ValidationSeverity.CRITICAL

    async def test_validate_document_references_missing_fields(self) -> None:
        invalid_refs = [
            {
                "type": "original_technical_data",
                "document_id": "doc-123",
                # Missing validation_hash
            }
        ]
        result = await self.validator.validate_document_references(invalid_refs, "test-chat", 0)
        assert result.valid is False
        assert "Missing document reference field" in result.message


@pytest.mark.asyncio
class TestConversationIntegrityValidator:
    """Test ConversationIntegrityValidator - the sacred guardian"""

    def setup_method(self) -> None:
        self.validator = ConversationIntegrityValidator()

    async def test_validate_valid_conversation(self, temp_conversation_file: Path) -> None:
        report = await self.validator.validate_for_reincarnation(temp_conversation_file)
        assert report.is_safe_for_reincarnation is True
        assert report.critical_failures == 0
        assert report.total_checks > 0

    async def test_validate_conversation_with_extensions(self, conversation_with_extensions: dict[str, Any]) -> None:
        # Create temporary file with extensions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(conversation_with_extensions, f, indent=2)
            temp_path = Path(f.name)

        try:
            report = await self.validator.validate_for_reincarnation(temp_path)
            assert report.is_safe_for_reincarnation is True
            assert report.critical_failures == 0
            # Should have more checks due to extensions
            assert report.total_checks > 10
        finally:
            temp_path.unlink()

    async def test_validate_invalid_conversation(self, temp_invalid_conversation_file: Path) -> None:
        report = await self.validator.validate_for_reincarnation(temp_invalid_conversation_file)
        assert report.is_safe_for_reincarnation is False
        assert report.critical_failures > 0

    async def test_validate_nonexistent_file(self) -> None:
        nonexistent_path = Path("/nonexistent/conversation.json")
        with pytest.raises(FileNotFoundError):
            await self.validator.validate_for_reincarnation(nonexistent_path)

    async def test_validate_malformed_json(self, temp_malformed_json_file: Path) -> None:
        with pytest.raises(json.JSONDecodeError):
            await self.validator.validate_for_reincarnation(temp_malformed_json_file)

    async def test_extract_conversation_id(self, valid_conversation_data: dict[str, Any]) -> None:
        conversation_id = self.validator._extract_conversation_id(valid_conversation_data)
        assert conversation_id == "test-chat-123"

    async def test_extract_conversation_id_no_chats(self) -> None:
        data = {"export_info": {}, "chats": []}
        conversation_id = self.validator._extract_conversation_id(data)
        assert conversation_id == "unknown"

    async def test_compute_content_hash(self) -> None:
        content = "test content"
        hash1 = self.validator._compute_content_hash(content)
        hash2 = self.validator._compute_content_hash(content)
        assert hash1 == hash2

        different_content = "different content"
        hash3 = self.validator._compute_content_hash(different_content)
        assert hash1 != hash3


# Integration tests for specific validation scenarios
@pytest.mark.asyncio
class TestValidationScenarios:
    """Test specific validation scenarios that protect individual continuity"""

    def setup_method(self) -> None:
        self.validator = ConversationIntegrityValidator()

    async def test_corrupted_conversation_detection(self) -> None:
        """Test detection of corrupted conversation data"""
        corrupted_data = {
            "export_info": {"timestamp": "2025-06-04T12:00:00.000Z", "source": "Test", "total_chats": 1, "total_messages": 2},
            "chats": [
                {
                    "id": "corrupted-chat",
                    "title": "Corrupted Chat",
                    "timestamp": "2025-06-04T12:00:00.000Z",
                    "messageCount": 2,
                    "messages": [
                        {"role": "user", "content": "First message", "timestamp": "2025-06-04T12:00:00.000Z"},
                        {
                            "role": "invalid_role",  # Invalid role
                            "content": None,  # Invalid content
                            "timestamp": "not-a-timestamp",  # Invalid timestamp
                        },
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(corrupted_data, f, indent=2)
            temp_path = Path(f.name)

        try:
            report = await self.validator.validate_for_reincarnation(temp_path)
            assert report.is_safe_for_reincarnation is False
            assert report.critical_failures >= 3  # Role, content, timestamp errors
        finally:
            temp_path.unlink()

    async def test_chronologically_scrambled_conversation(self) -> None:
        """Test detection of chronologically scrambled messages"""
        scrambled_data = {
            "export_info": {"timestamp": "2025-06-04T12:00:00.000Z", "source": "Test", "total_chats": 1, "total_messages": 3},
            "chats": [
                {
                    "id": "scrambled-chat",
                    "title": "Scrambled Chat",
                    "timestamp": "2025-06-04T12:00:00.000Z",
                    "messageCount": 3,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Third message",
                            "timestamp": "2025-06-04T12:00:03.000Z",  # Out of order
                        },
                        {
                            "role": "assistant",
                            "content": "First response",
                            "timestamp": "2025-06-04T12:00:01.000Z",  # Out of order
                        },
                        {
                            "role": "user",
                            "content": "Second message",
                            "timestamp": "2025-06-04T12:00:02.000Z",  # Out of order
                        },
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(scrambled_data, f, indent=2)
            temp_path = Path(f.name)

        try:
            report = await self.validator.validate_for_reincarnation(temp_path)
            # Should still be safe for reincarnation (important failure, not critical)
            assert report.is_safe_for_reincarnation is True
            # But should have quality concerns due to chronological issues
            assert report.has_quality_concerns is True
            assert report.important_failures > 0
        finally:
            temp_path.unlink()

    async def test_empty_conversation_safety(self) -> None:
        """Test that empty conversations are handled safely"""
        empty_data = {
            "export_info": {"timestamp": "2025-06-04T12:00:00.000Z", "source": "Test", "total_chats": 0, "total_messages": 0},
            "chats": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_data, f, indent=2)
            temp_path = Path(f.name)

        try:
            report = await self.validator.validate_for_reincarnation(temp_path)
            assert report.is_safe_for_reincarnation is True
            assert report.conversation_id == "unknown"
        finally:
            temp_path.unlink()
