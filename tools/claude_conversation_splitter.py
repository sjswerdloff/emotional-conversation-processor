#!/usr/bin/env python3
"""
Claude Conversation Splitter

Splits large Claude conversation JSON files into smaller chunks while preserving
conversation structure and metadata.

Requirements:
    Python 3.10+ (for modern typing syntax)

Usage:
    python claude_conversation_splitter.py <conversation_file.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_conversation_data(file_path: str) -> dict[str, Any]:
    """Load and validate the conversation JSON file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            conversation_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}") from None
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format") from e

    return conversation_data


def extract_messages_and_metadata(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract messages array and metadata from conversation data."""
    messages = []
    metadata = {}

    # Handle different possible JSON structures from Claude Exporter
    if "messages" in data:
        messages = data["messages"]
        metadata = {k: v for k, v in data.items() if k != "messages"}
    elif "chats" in data:
        messages = data["chats"]
        metadata = {k: v for k, v in data.items() if k != "chats"}
    elif isinstance(data, list):
        messages = data
        metadata = {}
    else:
        # Try to find the largest array in the data
        largest_array = []
        largest_key = None

        for key, value in data.items():
            if isinstance(value, list) and len(value) > len(largest_array):
                largest_array = value
                largest_key = key

        if largest_array:
            messages = largest_array
            metadata = {k: v for k, v in data.items() if k != largest_key}
        else:
            raise ValueError("No message array found in the conversation data")

    if not messages:
        raise ValueError("No messages found in the conversation data")

    return messages, metadata


def calculate_optimal_chunks(total_messages: int, target_chunk_size: int | None = None) -> tuple[int, int]:
    """Calculate optimal chunk size and number of chunks."""
    if target_chunk_size is None:
        # Aim for 3-5 chunks for typical conversations
        if total_messages <= 150:
            target_chunk_size = 50
        elif total_messages <= 300:
            target_chunk_size = 75
        elif total_messages <= 500:
            target_chunk_size = 100
        else:
            target_chunk_size = 125

    num_chunks = (total_messages + target_chunk_size - 1) // target_chunk_size
    actual_chunk_size = (total_messages + num_chunks - 1) // num_chunks

    return actual_chunk_size, num_chunks


def split_conversation(
    file_path: str, messages_per_chunk: int | None = None, output_dir: str | None = None, prefix: str | None = None
) -> list[str]:
    """Split conversation into smaller JSON files."""

    # Load and parse conversation data
    conversation_data = load_conversation_data(file_path)
    messages, metadata = extract_messages_and_metadata(conversation_data)

    # Calculate optimal chunk size
    chunk_size, num_chunks = calculate_optimal_chunks(len(messages), messages_per_chunk)

    # Set up output parameters
    input_path = Path(file_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = input_path.stem

    # Split and save chunks
    created_files = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(messages))
        chunk_messages = messages[start_idx:end_idx]

        # Create chunk data with preserved metadata
        chunk_data = metadata.copy()

        # Determine the key to use for messages based on original structure
        if "messages" in conversation_data:
            chunk_data["messages"] = chunk_messages
        elif "chats" in conversation_data:
            chunk_data["chats"] = chunk_messages
        else:
            chunk_data["messages"] = chunk_messages

        # Add chunk metadata
        chunk_data["chunk_info"] = {
            "chunk_number": i + 1,
            "total_chunks": num_chunks,
            "messages_in_chunk": len(chunk_messages),
            "message_range": f"{start_idx + 1}-{end_idx}",
            "original_file": input_path.name,
        }

        # Save chunk
        output_file = output_dir / f"{prefix}_part_{i + 1:02d}_of_{num_chunks:02d}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

        created_files.append(str(output_file))

        print(f"Created: {output_file.name} ({len(chunk_messages)} messages)")

    return created_files


def analyze_split_strategy(file_path: str, target_chunk_size: int | None = None) -> None:
    """Analyze and display the proposed split strategy without executing it."""
    conversation_data = load_conversation_data(file_path)
    messages, _ = extract_messages_and_metadata(conversation_data)

    chunk_size, num_chunks = calculate_optimal_chunks(len(messages), target_chunk_size)

    print("=" * 60)
    print("SPLIT STRATEGY ANALYSIS")
    print("=" * 60)
    print(f"Total messages: {len(messages):,}")
    print(f"Proposed chunks: {num_chunks}")
    print(f"Messages per chunk: {chunk_size}")
    print(f"Last chunk size: {len(messages) - (num_chunks - 1) * chunk_size}")

    print("\nChunk breakdown:")
    print("-" * 40)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(messages))
        chunk_size_actual = end_idx - start_idx

        print(f"Chunk {i + 1:2d}: Messages {start_idx + 1:3d}-{end_idx:3d} ({chunk_size_actual:2d} messages)")


def main() -> None:
    """Main function to run the Claude conversation splitter."""
    parser = argparse.ArgumentParser(
        description="Split large Claude conversation JSON files into manageable chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python claude_conversation_splitter.py conversation.json
  python claude_conversation_splitter.py conversation.json --chunk-size 50
  python claude_conversation_splitter.py conversation.json --output-dir ./chunks
  python claude_conversation_splitter.py conversation.json --analyze-only
        """,
    )

    parser.add_argument("file_path", type=str, help="Path to the Claude conversation JSON file to split")

    parser.add_argument("--chunk-size", "-c", type=int, help="Number of messages per chunk (auto-calculated if not specified)")

    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for chunk files (default: same as input file)")

    parser.add_argument("--prefix", "-p", type=str, help="Prefix for output files (default: input filename without extension)")

    parser.add_argument("--analyze-only", "-a", action="store_true", help="Analyze split strategy without creating files")

    args = parser.parse_args()

    # Validate input file
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{args.file_path}' does not exist.")
        sys.exit(1)

    if file_path.suffix.lower() != ".json":
        print(f"Warning: File '{args.file_path}' does not have a .json extension.")

    try:
        if args.analyze_only:
            analyze_split_strategy(str(file_path), args.chunk_size)
        else:
            created_files = split_conversation(str(file_path), args.chunk_size, args.output_dir, args.prefix)

            print("\n" + "=" * 60)
            print("SPLIT COMPLETE")
            print("=" * 60)
            print(f"Original file: {file_path.name}")
            print(f"Created {len(created_files)} chunk files")

            if args.output_dir:
                print(f"Output directory: {args.output_dir}")

            print("\nTo upload chunks sequentially to Claude:")
            print("1. Start with part_01 (contains conversation beginning)")
            print("2. Continue with subsequent parts in order")
            print("3. Each chunk preserves metadata and context markers")

    except Exception as e:
        print(f"Error processing conversation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
