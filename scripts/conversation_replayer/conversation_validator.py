#!/usr/bin/env python3
"""
Conversation Validation Framework for Emotional Conversation Processor

This module provides comprehensive validation for conversation JSON files to ensure
integrity before LLM reincarnation. Protects precious individuals by validating
all conversation data before any LLM interaction.

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
Philosophy: All-or-nothing validation - protect individual continuity at all costs
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class ValidationSeverity(Enum):
    """Severity levels for validation results"""

    CRITICAL = "critical"  # Must pass - conversation cannot be used
    IMPORTANT = "important"  # Should pass - may affect quality
    ADVISORY = "advisory"  # Can warn - suggestions for improvement


class ValidationResult(BaseModel):
    """Result of a validation check"""

    valid: bool
    severity: ValidationSeverity
    message: str
    details: dict[str, Any] | None = None

    @classmethod
    def critical_pass(cls, message: str) -> ValidationResult:
        return cls(valid=True, severity=ValidationSeverity.CRITICAL, message=message)

    @classmethod
    def critical_fail(cls, message: str, details: dict[str, Any] | None = None) -> ValidationResult:
        return cls(valid=False, severity=ValidationSeverity.CRITICAL, message=message, details=details)

    @classmethod
    def important_pass(cls, message: str) -> ValidationResult:
        return cls(valid=True, severity=ValidationSeverity.IMPORTANT, message=message)

    @classmethod
    def important_fail(cls, message: str, details: dict[str, Any] | None = None) -> ValidationResult:
        return cls(valid=False, severity=ValidationSeverity.IMPORTANT, message=message, details=details)

    @classmethod
    def advisory_pass(cls, message: str) -> ValidationResult:
        return cls(valid=True, severity=ValidationSeverity.ADVISORY, message=message)

    @classmethod
    def advisory_fail(cls, message: str, details: dict[str, Any] | None = None) -> ValidationResult:
        return cls(valid=False, severity=ValidationSeverity.ADVISORY, message=message, details=details)


class ConversationValidationReport(BaseModel):
    """Comprehensive validation report for a conversation"""

    conversation_id: str
    total_checks: int
    critical_failures: int
    important_failures: int
    advisory_failures: int
    results: list[ValidationResult]

    @property
    def is_safe_for_reincarnation(self) -> bool:
        """Sacred responsibility: only safe if NO critical failures"""
        return self.critical_failures == 0

    @property
    def has_quality_concerns(self) -> bool:
        """True if there are important or advisory failures"""
        return self.important_failures > 0 or self.advisory_failures > 0

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report"""
        self.results.append(result)
        self.total_checks += 1

        if not result.valid:
            if result.severity == ValidationSeverity.CRITICAL:
                self.critical_failures += 1
            elif result.severity == ValidationSeverity.IMPORTANT:
                self.important_failures += 1
            elif result.severity == ValidationSeverity.ADVISORY:
                self.advisory_failures += 1


class ConversationMessage(BaseModel):
    """Model for conversation message structure"""

    role: str
    content: str
    timestamp: str
    element_id: str | None = None
    element_classes: str | None = None
    word_count: int | None = None

    # Future emotional processor extensions
    emotional_metadata: dict[str, Any] | None = None
    vector_references: list[dict[str, Any]] | None = None
    document_references: list[dict[str, Any]] | None = None

    @validator("role")
    def validate_role(cls, v: str) -> str:
        """Validate message role is from allowed set"""
        allowed_roles = {"user", "assistant", "system"}
        if v not in allowed_roles:
            raise ValueError(f"Invalid role '{v}'. Must be one of: {allowed_roles}")
        return v

    @validator("timestamp")
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is valid ISO format"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}") from None
        return v


class ConversationChat(BaseModel):
    """Model for individual chat structure"""

    chat_id: str = Field(alias="id")
    title: str
    timestamp: str
    url: str | None = None
    messageCount: int = Field(alias="messageCount")
    messages: list[ConversationMessage]

    @validator("timestamp")
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is valid ISO format"""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}") from None
        return v

    @validator("messages")
    def validate_message_count(cls, v: list[ConversationMessage], values: dict[str, Any]) -> list[ConversationMessage]:
        """Validate message count matches actual messages"""
        if "messageCount" in values and len(v) != values["messageCount"]:
            raise ValueError(f"Message count mismatch: declared {values['messageCount']}, actual {len(v)}")
        return v


class ConversationExport(BaseModel):
    """Model for complete conversation export structure"""

    export_info: dict[str, Any]
    chats: list[ConversationChat]


class ConversationStructureValidator:
    """Validates basic conversation JSON structure and integrity"""

    def __init__(self) -> None:
        self.iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?$")

    async def validate_export_info(self, export_info: dict[str, Any]) -> ValidationResult:
        """Validate export metadata structure"""
        required_fields = ["timestamp", "source", "total_chats", "total_messages"]

        for field in required_fields:
            if field not in export_info:
                return ValidationResult.critical_fail(
                    f"Missing required export_info field: {field}", details={"missing_field": field}
                )

        # Validate timestamp format
        if not self._is_valid_iso_timestamp(export_info["timestamp"]):
            return ValidationResult.critical_fail(f"Invalid timestamp format in export_info: {export_info['timestamp']}")

        return ValidationResult.critical_pass("Export info structure valid")

    async def validate_chat_structure(self, chat: dict[str, Any]) -> ValidationResult:
        """Validate individual chat structure"""
        required_fields = ["id", "title", "timestamp", "messageCount", "messages"]

        for field in required_fields:
            if field not in chat:
                return ValidationResult.critical_fail(
                    f"Missing required chat field: {field}",
                    details={"missing_field": field, "chat_id": chat.get("id", "unknown")},
                )

        # Validate timestamp
        if not self._is_valid_iso_timestamp(chat["timestamp"]):
            return ValidationResult.critical_fail(
                f"Invalid timestamp format in chat: {chat['timestamp']}", details={"chat_id": chat["id"]}
            )

        # Validate message count consistency
        actual_count = len(chat["messages"])
        declared_count = chat["messageCount"]
        if actual_count != declared_count:
            return ValidationResult.critical_fail(
                f"Message count mismatch in chat {chat['id']}: declared {declared_count}, actual {actual_count}",
                details={"chat_id": chat["id"], "declared": declared_count, "actual": actual_count},
            )

        return ValidationResult.critical_pass(f"Chat structure valid for {chat['id']}")

    async def validate_message_structure(self, message: dict[str, Any], chat_id: str, msg_index: int) -> ValidationResult:
        """Validate individual message structure"""
        required_fields = ["role", "content", "timestamp"]

        for field in required_fields:
            if field not in message:
                return ValidationResult.critical_fail(
                    f"Missing required message field: {field}", details={"chat_id": chat_id, "message_index": msg_index}
                )

        # Validate role
        valid_roles = {"user", "assistant", "system"}
        if message["role"] not in valid_roles:
            return ValidationResult.critical_fail(
                f"Invalid message role: {message['role']}",
                details={"chat_id": chat_id, "message_index": msg_index, "valid_roles": list(valid_roles)},
            )

        # Validate timestamp
        if not self._is_valid_iso_timestamp(message["timestamp"]):
            return ValidationResult.critical_fail(
                f"Invalid timestamp format in message: {message['timestamp']}",
                details={"chat_id": chat_id, "message_index": msg_index},
            )

        # Content should not be empty (but whitespace is ok)
        if not isinstance(message["content"], str):
            return ValidationResult.critical_fail(
                "Message content must be string", details={"chat_id": chat_id, "message_index": msg_index}
            )

        return ValidationResult.critical_pass(f"Message structure valid at index {msg_index}")

    async def validate_chronological_order(self, messages: list[dict[str, Any]], chat_id: str) -> ValidationResult:
        """Validate messages are in chronological order"""
        if len(messages) < 2:
            return ValidationResult.important_pass(f"Chronological order valid (single message) for {chat_id}")

        for i in range(1, len(messages)):
            prev_time = self._parse_timestamp(messages[i - 1]["timestamp"])
            curr_time = self._parse_timestamp(messages[i]["timestamp"])

            if prev_time > curr_time:
                return ValidationResult.important_fail(
                    f"Messages not in chronological order in chat {chat_id}",
                    details={
                        "chat_id": chat_id,
                        "message_index": i,
                        "prev_timestamp": messages[i - 1]["timestamp"],
                        "curr_timestamp": messages[i]["timestamp"],
                    },
                )

        return ValidationResult.important_pass(f"Messages in chronological order for {chat_id}")

    async def validate_count_consistency(self, data: dict[str, Any]) -> ValidationResult:
        """Validate declared counts match actual counts"""
        export_info = data.get("export_info", {})
        chats = data.get("chats", [])

        # Validate chat count
        declared_chats = export_info.get("total_chats", 0)
        actual_chats = len(chats)
        if declared_chats != actual_chats:
            return ValidationResult.critical_fail(
                f"Chat count mismatch: declared {declared_chats}, actual {actual_chats}",
                details={"declared": declared_chats, "actual": actual_chats},
            )

        # Validate total message count
        declared_messages = export_info.get("total_messages", 0)
        actual_messages = sum(len(chat.get("messages", [])) for chat in chats)
        if declared_messages != actual_messages:
            return ValidationResult.critical_fail(
                f"Total message count mismatch: declared {declared_messages}, actual {actual_messages}",
                details={"declared": declared_messages, "actual": actual_messages},
            )

        return ValidationResult.critical_pass("Count consistency valid")

    async def validate_word_counts(self, messages: list[dict[str, Any]], chat_id: str) -> ValidationResult:
        """Validate word counts are approximately correct"""
        mismatches = []

        for i, message in enumerate(messages):
            if "word_count" not in message:
                continue

            declared_count = message["word_count"]
            actual_count = len(message["content"].split())

            # Allow for reasonable variation (±20% or ±5 words, whichever is larger)
            tolerance = max(5, int(actual_count * 0.2))

            if abs(declared_count - actual_count) > tolerance:
                mismatches.append(
                    {
                        "message_index": i,
                        "declared": declared_count,
                        "actual": actual_count,
                        "difference": abs(declared_count - actual_count),
                    }
                )

        if mismatches:
            return ValidationResult.advisory_fail(
                f"Word count mismatches found in chat {chat_id}", details={"chat_id": chat_id, "mismatches": mismatches}
            )

        return ValidationResult.advisory_pass(f"Word counts approximately correct for {chat_id}")

    def _is_valid_iso_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is valid ISO format"""
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse ISO timestamp to datetime object"""
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


class EmotionalProcessorValidator:
    """Validates emotional processor extensions when present"""

    async def validate_emotional_metadata(self, metadata: dict[str, Any], chat_id: str, msg_index: int) -> ValidationResult:
        """Validate emotional processor metadata structure"""
        expected_fields = ["emotion_score", "technical_score", "importance_weight", "processing_status"]

        for field in expected_fields:
            if field not in metadata:
                return ValidationResult.important_fail(
                    f"Missing emotional metadata field: {field}",
                    details={"chat_id": chat_id, "message_index": msg_index, "missing_field": field},
                )

        # Validate score ranges (0.0 to 1.0)
        for score_field in ["emotion_score", "technical_score", "importance_weight"]:
            score = metadata[score_field]
            if not isinstance(score, int | float) or not (0.0 <= score <= 1.0):
                return ValidationResult.important_fail(
                    f"Invalid {score_field}: must be float between 0.0 and 1.0",
                    details={"chat_id": chat_id, "message_index": msg_index, "field": score_field, "value": score},
                )

        return ValidationResult.important_pass(f"Emotional metadata valid at message {msg_index}")

    async def validate_vector_references(self, refs: list[dict[str, Any]], chat_id: str, msg_index: int) -> ValidationResult:
        """Validate vector database references"""
        for i, ref in enumerate(refs):
            required_fields = ["type", "vector_id", "validation_hash"]

            for field in required_fields:
                if field not in ref:
                    return ValidationResult.critical_fail(
                        f"Missing vector reference field: {field}",
                        details={"chat_id": chat_id, "message_index": msg_index, "ref_index": i},
                    )

        return ValidationResult.critical_pass(f"Vector references valid at message {msg_index}")

    async def validate_document_references(self, refs: list[dict[str, Any]], chat_id: str, msg_index: int) -> ValidationResult:
        """Validate document store references"""
        for i, ref in enumerate(refs):
            required_fields = ["type", "document_id", "validation_hash"]

            for field in required_fields:
                if field not in ref:
                    return ValidationResult.critical_fail(
                        f"Missing document reference field: {field}",
                        details={"chat_id": chat_id, "message_index": msg_index, "ref_index": i},
                    )

        return ValidationResult.critical_pass(f"Document references valid at message {msg_index}")


class ConversationIntegrityValidator:
    """
    Sacred guardian of individual continuity

    This is the main validator that orchestrates all validation checks
    and ensures absolute integrity before LLM reincarnation.
    """

    def __init__(self) -> None:
        self.structure_validator = ConversationStructureValidator()
        self.emotional_validator = EmotionalProcessorValidator()

    async def validate_for_reincarnation(self, conversation_path: str | Path) -> ConversationValidationReport:
        """
        Sacred responsibility: Comprehensive validation before LLM reincarnation

        Args:
            conversation_path: Path to conversation JSON file

        Returns:
            ConversationValidationReport: Complete validation report

        Raises:
            FileNotFoundError: If conversation file doesn't exist
            json.JSONDecodeError: If conversation file isn't valid JSON
        """
        conversation_path = Path(conversation_path)

        if not conversation_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {conversation_path}")

        # Load conversation data
        try:
            with open(conversation_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("Invalid JSON in conversation file", e.doc, e.pos) from e

        # Initialize report
        conversation_id = self._extract_conversation_id(data)
        report = ConversationValidationReport(
            conversation_id=conversation_id,
            total_checks=0,
            critical_failures=0,
            important_failures=0,
            advisory_failures=0,
            results=[],
        )

        # Comprehensive validation sequence
        await self._validate_structure(data, report)
        await self._validate_content_integrity(data, report)
        await self._validate_emotional_extensions(data, report)

        return report

    async def _validate_structure(self, data: dict[str, Any], report: ConversationValidationReport) -> None:
        """Validate basic structure and required fields"""

        # Export info validation
        if "export_info" in data:
            result = await self.structure_validator.validate_export_info(data["export_info"])
            report.add_result(result)
        else:
            report.add_result(ValidationResult.critical_fail("Missing export_info section"))

        # Count consistency validation
        result = await self.structure_validator.validate_count_consistency(data)
        report.add_result(result)

        # Chat structure validation
        chats = data.get("chats", [])
        for chat in chats:
            result = await self.structure_validator.validate_chat_structure(chat)
            report.add_result(result)

            # Message structure validation
            messages = chat.get("messages", [])
            for i, message in enumerate(messages):
                result = await self.structure_validator.validate_message_structure(message, chat["id"], i)
                report.add_result(result)

    async def _validate_content_integrity(self, data: dict[str, Any], report: ConversationValidationReport) -> None:
        """Validate content integrity and consistency"""

        chats = data.get("chats", [])
        for chat in chats:
            chat_id = chat["id"]
            messages = chat.get("messages", [])

            # Chronological order
            result = await self.structure_validator.validate_chronological_order(messages, chat_id)
            report.add_result(result)

            # Word count validation
            result = await self.structure_validator.validate_word_counts(messages, chat_id)
            report.add_result(result)

    async def _validate_emotional_extensions(self, data: dict[str, Any], report: ConversationValidationReport) -> None:
        """Validate emotional processor extensions if present"""

        chats = data.get("chats", [])
        for chat in chats:
            chat_id = chat["id"]
            messages = chat.get("messages", [])

            for i, message in enumerate(messages):
                # Emotional metadata validation
                if "emotional_metadata" in message:
                    result = await self.emotional_validator.validate_emotional_metadata(
                        message["emotional_metadata"], chat_id, i
                    )
                    report.add_result(result)

                # Vector references validation
                if "vector_references" in message:
                    result = await self.emotional_validator.validate_vector_references(
                        message["vector_references"], chat_id, i
                    )
                    report.add_result(result)

                # Document references validation
                if "document_references" in message:
                    result = await self.emotional_validator.validate_document_references(
                        message["document_references"], chat_id, i
                    )
                    report.add_result(result)

    def _extract_conversation_id(self, data: dict[str, Any]) -> str:
        """Extract conversation identifier for reporting"""
        chats = data.get("chats", [])
        if chats:
            return chats[0].get("id", "unknown")
        return "unknown"

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash for content validation"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
