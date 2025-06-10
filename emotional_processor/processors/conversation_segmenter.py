"""Enhanced conversation segmentation for various formats including Claude Desktop exports."""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""

    speaker: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None


class ConversationSegmenter:
    """
    Enhanced conversation segmenter that handles multiple formats including
    messy Claude Desktop exports.
    """

    def __init__(self) -> None:
        """Initialize the conversation segmenter."""
        self.claude_desktop_patterns = self._build_claude_desktop_patterns()
        self.standard_patterns = self._build_standard_patterns()

    def _build_claude_desktop_patterns(self) -> list[str]:
        """Build regex patterns for Claude Desktop format."""
        return [
            # Remove timing information (e.g., "12s", "Analyzed... 10s")
            r"^[A-Z][^.]*?\.\n\d+s\n*",
            # Remove Edit/Retry markers
            r"\n*Edit\n*",
            r"\n*Retry\n*",
            # Remove document tags and metadata
            r"<documents>.*?</documents>",
            r"<document.*?>.*?</document>",
            r"<source>.*?</source>",
            r"<document_content>.*?</document_content>",
            # Remove other UI artifacts
            r"Human: <documents>.*?(?=\n\n|\Z)",
        ]

    def _build_standard_patterns(self) -> list[tuple[str, str]]:
        """Build regex patterns for standard conversation formats."""
        return [
            # "User [timestamp]: message" format
            (
                r"^(User|Assistant|Human|AI|System)\s*\[(.*?)\]:\s*(.*?)(?=^(?:User|Assistant|Human|AI|System)\s*\[|$)",
                "timestamped",
            ),
            # "User: message" format
            (r"^(User|Assistant|Human|AI|System):\s*(.*?)(?=^(?:User|Assistant|Human|AI|System):|$)", "simple"),
            # Markdown style "**User:**" format
            (
                r"^\*\*(User|Assistant|Human|AI|System)\*\*:?\s*(.*?)(?=^\*\*(?:User|Assistant|Human|AI|System)\*\*|$)",
                "markdown",
            ),
            # Claude Desktop specific pattern (after cleaning)
            (r"^(Human|Claude [^:]*?):\s*(.*?)(?=^(?:Human|Claude)|$)", "claude_desktop"),
        ]

    def clean_claude_desktop_export(self, content: str) -> str:
        """
        Clean up messy Claude Desktop export format.

        Args:
            content: Raw Claude Desktop export content

        Returns:
            Cleaned conversation text
        """
        cleaned = content

        # Apply Claude Desktop specific cleaning patterns
        for pattern in self.claude_desktop_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

        # Remove excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"^\s+|\s+$", "", cleaned, flags=re.MULTILINE)

        # Fix common artifacts
        cleaned = re.sub(r"\n+Human:", "\n\nHuman:", cleaned)
        cleaned = re.sub(r"\n+Claude", "\n\nClaude", cleaned)

        # Remove empty lines at start/end
        cleaned = cleaned.strip()

        logger.debug(f"Cleaned Claude Desktop export: {len(content)} -> {len(cleaned)} chars")
        return cleaned

    def detect_format(self, content: str) -> str:
        """
        Detect the conversation format.

        Args:
            content: Conversation content to analyze

        Returns:
            Detected format type
        """
        # Check for Claude Desktop artifacts
        claude_artifacts = [
            r"\d+s\n",  # Timing markers
            "Edit\n",
            "Retry\n",
            "<documents>",
            "Human: <documents>",
        ]

        if any(re.search(pattern, content) for pattern in claude_artifacts):
            return "claude_desktop"

        # Check for standard formats
        for pattern, format_name in self.standard_patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                return format_name

        return "unknown"

    def segment_conversation(self, content: str) -> list[ConversationTurn]:
        """
        Segment conversation content into turns.

        Args:
            content: Raw conversation content

        Returns:
            List of conversation turns
        """
        # Detect and handle format
        format_type = self.detect_format(content)
        logger.info(f"Detected conversation format: {format_type}")

        # Clean Claude Desktop format if needed
        if format_type == "claude_desktop":
            content = self.clean_claude_desktop_export(content)

        # Try each segmentation pattern
        turns: list[ConversationTurn] = []

        for pattern, pattern_name in self.standard_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE))

            if matches:
                pattern_turns = []
                for match in matches:
                    groups = match.groups()

                    if len(groups) == 3:  # With timestamp
                        speaker, timestamp, text = groups
                        timestamp = timestamp.strip() if timestamp else None
                    else:  # Without timestamp
                        speaker, text = groups
                        timestamp = None

                    text = text.strip()
                    if text:
                        turn = ConversationTurn(
                            speaker=speaker.strip(), content=text, timestamp=timestamp, metadata={"pattern": pattern_name}
                        )
                        pattern_turns.append(turn)

                # Use the pattern that found the most turns
                if len(pattern_turns) > len(turns):
                    turns = pattern_turns

        # Fallback: split by paragraphs and alternate speakers
        if not turns:
            turns = self._fallback_segmentation(content)

        # Clean up turns
        turns = self._clean_turns(turns)

        logger.info(f"Segmented conversation into {len(turns)} turns")
        return turns

    def _fallback_segmentation(self, content: str) -> list[ConversationTurn]:
        """
        Fallback segmentation by paragraphs with alternating speakers.

        Args:
            content: Conversation content

        Returns:
            List of conversation turns
        """
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        turns = []

        for i, paragraph in enumerate(paragraphs):
            # Skip very short paragraphs that might be artifacts
            if len(paragraph) < 10:
                continue

            speaker = "Human" if i % 2 == 0 else "Assistant"

            turn = ConversationTurn(speaker=speaker, content=paragraph, metadata={"pattern": "fallback"})
            turns.append(turn)

        return turns

    def _clean_turns(self, turns: list[ConversationTurn]) -> list[ConversationTurn]:
        """
        Clean and validate conversation turns.

        Args:
            turns: Raw conversation turns

        Returns:
            Cleaned conversation turns
        """
        cleaned_turns = []

        for turn in turns:
            # Clean content
            content = turn.content

            # Remove common artifacts
            content = re.sub(r"\n+", " ", content)  # Normalize newlines
            content = re.sub(r"\s+", " ", content)  # Normalize whitespace
            content = content.strip()

            # Skip empty or very short turns
            if len(content) < 5:
                continue

            # Normalize speaker names
            speaker = self._normalize_speaker(turn.speaker)

            cleaned_turn = ConversationTurn(speaker=speaker, content=content, timestamp=turn.timestamp, metadata=turn.metadata)
            cleaned_turns.append(cleaned_turn)

        return cleaned_turns

    def _normalize_speaker(self, speaker: str) -> str:
        """
        Normalize speaker names to standard format.

        Args:
            speaker: Raw speaker name

        Returns:
            Normalized speaker name
        """
        speaker = speaker.strip().lower()

        # Normalize common variations
        if speaker in ["human", "user"]:
            return "User"
        elif speaker.startswith("claude") or speaker in ["assistant", "ai"]:
            return "Assistant"
        elif speaker in ["system"]:
            return "System"
        else:
            return speaker.title()

    def export_clean_format(self, turns: list[ConversationTurn], format_type: str = "simple") -> str:
        """
        Export conversation turns to clean format.

        Args:
            turns: List of conversation turns
            format_type: Output format ('simple', 'timestamped', 'markdown')

        Returns:
            Formatted conversation string
        """
        lines = []

        for turn in turns:
            if format_type == "timestamped" and turn.timestamp:
                line = f"{turn.speaker} [{turn.timestamp}]: {turn.content}"
            elif format_type == "markdown":
                line = f"**{turn.speaker}**: {turn.content}"
            else:  # simple
                line = f"{turn.speaker}: {turn.content}"

            lines.append(line)

        return "\n\n".join(lines)

    def get_conversation_stats(self, turns: list[ConversationTurn]) -> dict[str, Any]:
        """
        Get statistics about the conversation.

        Args:
            turns: List of conversation turns

        Returns:
            Dictionary with conversation statistics
        """
        if not turns:
            return {"error": "No turns found"}

        speakers: dict[str, int] = {}
        total_words = 0
        total_chars = 0

        for turn in turns:
            speaker = turn.speaker
            speakers[speaker] = speakers.get(speaker, 0) + 1

            words = len(turn.content.split())
            total_words += words
            total_chars += len(turn.content)

        return {
            "total_turns": len(turns),
            "speakers": speakers,
            "total_words": total_words,
            "total_characters": total_chars,
            "average_words_per_turn": total_words / len(turns),
            "average_chars_per_turn": total_chars / len(turns),
        }


# Convenience function for backward compatibility
def segment_conversation(content: str) -> list[tuple[str, str, str | None]]:
    """
    Convenience function to segment conversation content.

    Args:
        content: Raw conversation content

    Returns:
        List of (content, speaker, timestamp) tuples
    """
    segmenter = ConversationSegmenter()
    turns = segmenter.segment_conversation(content)

    return [(turn.content, turn.speaker, turn.timestamp) for turn in turns]
