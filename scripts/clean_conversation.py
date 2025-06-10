#!/usr/bin/env python3
"""
Simple script to clean up Claude Desktop conversation exports.

Usage:
  python scripts/clean_conversation.py input.txt output.txt
  python scripts/clean_conversation.py input.txt --preview  # Just show the result
"""

import argparse
import re
import sys
from pathlib import Path


def clean_claude_desktop_export(content: str) -> str:
    """Clean up messy Claude Desktop export format."""

    print("🧹 Cleaning Claude Desktop export...")
    original_length = len(content)

    # Step 1: Remove timing information and analysis lines
    print("  - Removing timing information...")
    content = re.sub(r"^[A-Z][^.\n]*\.\n\d+s\n*", "", content, flags=re.MULTILINE)

    # Step 2: Remove UI elements
    print("  - Removing UI elements...")
    content = re.sub(r"\n*Edit\n*", "", content)
    content = re.sub(r"\n*Retry\n*", "", content)

    # Step 3: Remove document tags and metadata
    print("  - Removing document metadata...")
    content = re.sub(r"<documents>.*?</documents>", "", content, flags=re.DOTALL)
    content = re.sub(r"<document[^>]*>.*?</document>", "", content, flags=re.DOTALL)
    content = re.sub(r"<source>.*?</source>", "", content, flags=re.DOTALL)
    content = re.sub(r"<document_content>.*?</document_content>", "", content, flags=re.DOTALL)

    # Step 4: Clean up Human: <documents> patterns
    print("  - Cleaning document references...")
    content = re.sub(r"Human: <documents>.*?(?=\n\n|\Z)", "", content, flags=re.DOTALL)

    # Step 5: Normalize whitespace
    print("  - Normalizing whitespace...")
    content = re.sub(r"\n{3,}", "\n\n", content)  # Max 2 consecutive newlines
    content = re.sub(r"^\s+|\s+$", "", content, flags=re.MULTILINE)  # Trim lines

    # Step 6: Fix speaker transitions
    print("  - Fixing speaker transitions...")
    content = re.sub(r"\n+Human:", "\n\nHuman:", content)
    content = re.sub(r"\n+Claude", "\n\nClaude", content)

    # Step 7: Remove empty lines at start/end
    content = content.strip()

    # Step 8: Convert to standard format
    print("  - Converting to standard format...")
    content = re.sub(r"^Human:", "User:", content, flags=re.MULTILINE)
    content = re.sub(r"^Claude[^:]*?:", "Assistant:", content, flags=re.MULTILINE)

    cleaned_length = len(content)
    reduction = ((original_length - cleaned_length) / original_length) * 100

    print(f"✅ Cleaning complete! Reduced size by {reduction:.1f}% ({original_length:,} → {cleaned_length:,} characters)")

    return content


def detect_conversation_quality(content: str) -> dict:
    """Analyze the quality of the conversation content."""

    # Count conversation turns
    user_turns = len(re.findall(r"^User:", content, re.MULTILINE))
    assistant_turns = len(re.findall(r"^Assistant:", content, re.MULTILINE))

    # Check for remaining artifacts
    artifacts = []
    if re.search(r"\d+s\n", content):
        artifacts.append("timing markers")
    if "Edit" in content or "Retry" in content:
        artifacts.append("UI elements")
    if "<document" in content:
        artifacts.append("document tags")

    # Calculate basic stats
    words = len(content.split())
    lines = len(content.split("\n"))

    return {
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "total_turns": user_turns + assistant_turns,
        "total_words": words,
        "total_lines": lines,
        "remaining_artifacts": artifacts,
        "estimated_quality": "Good" if not artifacts and user_turns > 0 and assistant_turns > 0 else "Needs Review",
    }


def preview_conversation(content: str, lines: int = 20) -> None:
    """Show a preview of the conversation."""

    preview_lines = content.split("\n")[:lines]
    preview_text = "\n".join(preview_lines)

    print(f"\n📖 Preview (first {lines} lines):")
    print("=" * 50)
    print(preview_text)
    content_lines = len(content.split("\n"))
    if content_lines > lines:
        print(f"\n... ({content_lines - lines} more lines)")
    print("=" * 50)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean Claude Desktop conversation exports for processing")

    parser.add_argument("input_file", help="Input conversation file (Claude Desktop export)")

    parser.add_argument("output_file", nargs="?", help="Output file for cleaned conversation")

    parser.add_argument("--preview", action="store_true", help="Show preview without saving")

    parser.add_argument("--stats", action="store_true", help="Show conversation statistics")

    parser.add_argument("--preview-lines", type=int, default=20, help="Number of lines to show in preview")

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    print(f"📁 Reading conversation from: {args.input_file}")

    try:
        with open(input_path, encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    # Clean the content
    cleaned_content = clean_claude_desktop_export(original_content)

    # Show statistics
    if args.stats or args.preview:
        print("\n📊 Conversation Analysis:")
        stats = detect_conversation_quality(cleaned_content)
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    # Show preview
    if args.preview:
        preview_conversation(cleaned_content, args.preview_lines)
        return

    # Save cleaned content
    if not args.output_file:
        # Generate output filename
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    else:
        output_path = Path(args.output_file)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)

        print(f"💾 Cleaned conversation saved to: {output_path}")

        # Show final stats
        stats = detect_conversation_quality(cleaned_content)
        print(f"\n✨ Result: {stats['total_turns']} conversation turns, {stats['total_words']:,} words")
        print(f"Quality: {stats['estimated_quality']}")

        if stats["remaining_artifacts"]:
            print(f"⚠️  Note: Some artifacts remain: {', '.join(stats['remaining_artifacts'])}")
            print("   You may want to manually review the output.")

        print("\n🚀 Ready for processing with:")
        print(f"   python scripts/process_conversation.py {output_path}")

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
