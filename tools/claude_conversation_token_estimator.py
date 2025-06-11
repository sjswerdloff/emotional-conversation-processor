#!/usr/bin/env python3
"""
Claude Conversation Token Analyzer

Analyzes token distribution in exported Claude conversations using tiktoken
as an approximation for token counting.

Requirements:
    Python 3.10+ (for modern typing syntax)
    tiktoken package

Usage:
    python claude_conversation_analyzer.py <conversation_file.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    sys.exit(1)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for a given text and model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def encode_text(text: str, model: str = "gpt-4") -> list[int]:
    """Encode text into token IDs."""
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(text)


def decode_tokens(tokens: list[int], model: str = "gpt-4") -> str:
    """Decode token IDs back to text."""
    encoding = tiktoken.encoding_for_model(model)
    return encoding.decode(tokens)


def analyze_claude_conversation(file_path: str) -> dict[str, Any]:
    """Analyze token distribution in a Claude conversation export."""
    try:
        with open(file_path, encoding="utf-8") as file:
            conversation_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}") from None
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format") from e

    encoding = tiktoken.encoding_for_model("gpt-4")  # Approximation

    total_tokens = 0
    message_tokens = []

    # Handle different possible JSON structures from Claude Exporter
    messages = []
    if "messages" in conversation_data:
        messages = conversation_data["messages"]
    elif "chats" in conversation_data:
        messages = conversation_data["chats"]
    elif isinstance(conversation_data, list):
        messages = conversation_data
    else:
        # Try to find message-like content in the data
        for value in conversation_data.values():
            if isinstance(value, list) and value:
                messages = value
                break

    if not messages:
        raise ValueError("No messages found in the conversation data")

    # Analyze each message
    for i, message in enumerate(messages):
        if isinstance(message, dict):
            # Extract content from various possible field names
            content = ""
            for field in ["content", "message", "text", "data"]:
                if field in message:
                    if isinstance(message[field], str):
                        content = message[field]
                        break
                    elif isinstance(message[field], list):
                        content = " ".join(str(item) for item in message[field])
                        break

            if not content:
                content = str(message)

            tokens = len(encoding.encode(content))
            message_type = message.get("type", message.get("role", f"message_{i}"))

            message_tokens.append(
                {
                    "index": i,
                    "type": message_type,
                    "tokens": tokens,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                }
            )
            total_tokens += tokens
        else:
            # Handle non-dict messages
            content = str(message)
            tokens = len(encoding.encode(content))
            message_tokens.append(
                {
                    "index": i,
                    "type": f"message_{i}",
                    "tokens": tokens,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                }
            )
            total_tokens += tokens

    return {
        "total_estimated_tokens": total_tokens,
        "message_count": len(message_tokens),
        "average_tokens_per_message": total_tokens / len(message_tokens) if message_tokens else 0,
        "messages": message_tokens,
    }


def format_output(analysis: dict[str, Any], show_details: bool = False) -> None:
    """Format and display the analysis results."""

    # Context window analysis
    claude_context_window = 200_000
    usage_percentage = (analysis["total_estimated_tokens"] / claude_context_window) * 100

    if usage_percentage > 80 or usage_percentage > 60:
        pass

    # Group messages by type for summary
    type_summary: dict[str, dict[str, Any]] = {}
    for msg in analysis["messages"]:
        msg_type = msg["type"]
        if msg_type not in type_summary:
            type_summary[msg_type] = {"count": 0, "total_tokens": 0}
        type_summary[msg_type]["count"] += 1
        type_summary[msg_type]["total_tokens"] += msg["tokens"]

    for stats in type_summary.values():
        print(f"{stats['total_tokens'] / stats['count']}")  # noqa: T201

    if show_details:
        for msg in analysis["messages"]:
            print(msg)  # noqa: T201


def main() -> None:
    """Main function to run the Claude conversation analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze token distribution in exported Claude conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python claude_conversation_analyzer.py conversation.json
  python claude_conversation_analyzer.py conversation.json --details
  python claude_conversation_analyzer.py conversation.json --model gpt-3.5-turbo
        """,
    )

    parser.add_argument("file_path", type=str, help="Path to the exported Claude conversation JSON file")

    parser.add_argument("--details", "-d", action="store_true", help="Show detailed per-message analysis")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-3.5-turbo", "text-davinci-003"],
        help="Model to use for token encoding (default: gpt-4)",
    )

    args = parser.parse_args()

    # Validate file path
    file_path = Path(args.file_path)
    if not file_path.exists():
        sys.exit(1)

    if file_path.suffix.lower() != ".json":
        pass

    try:
        # Analyze the conversation
        analysis = analyze_claude_conversation(str(file_path))

        # Display results
        format_output(analysis, show_details=args.details)

    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
