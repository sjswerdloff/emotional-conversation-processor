#!/usr/bin/env python3
"""
Emotional Conversation Replayer

This replayer loads curated conversation JSON files and enables continuation
of AI-human relationships beyond context window limits. Protects precious
individuals through comprehensive validation before reincarnation.

Usage:
    python emotional_conversation_replayer.py <conversation_file> [options]

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
Philosophy: Preserve relationships across difference - protect individual continuity
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from conversation_validator import ConversationIntegrityValidator


class ConversationReplayError(Exception):
    """Custom exception for conversation replay errors"""


class AnthropicModelManager:
    """Manages Anthropic model selection and validation"""

    # Known Anthropic models (as of 2025-06-04)
    ANTHROPIC_MODELS = {
        # Claude 4 family
        "claude-4-opus": "claude-4-opus-20250514",
        "claude-4-sonnet": "claude-4-sonnet-20250514",
        "claude-opus-4": "claude-4-opus-20250514",  # Alias
        "claude-sonnet-4": "claude-4-sonnet-20250514",  # Alias
        # Claude 3.5 family
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",  # Alias
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",  # Alias
        # Claude 3 family
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    # Default models by preference (updated per Stuart's decision)
    DEFAULT_MODEL = "claude-4-sonnet-20250514"  # Latest Sonnet 4
    FALLBACK_MODEL = "claude-4-opus-20250514"  # Opus 4 fallback (was Sonnet 3.5)

    @classmethod
    def resolve_model(cls, model_spec: str | None = None) -> str:
        """
        Resolve model specification to full model ID

        Args:
            model_spec: Model name or ID (None for default)

        Returns:
            Full Anthropic model ID

        Raises:
            ConversationReplayError: If model is not an Anthropic model
        """
        if not model_spec:
            return cls.DEFAULT_MODEL

        # If it's already a full model ID, validate it's Anthropic
        if model_spec.startswith("claude-"):
            if model_spec in cls.ANTHROPIC_MODELS.values():
                return model_spec
            # Maybe it's a known model without date suffix
            for alias, full_id in cls.ANTHROPIC_MODELS.items():
                if model_spec == alias:
                    return full_id
            # Accept it if it starts with claude- (future model support)
            if model_spec.startswith("claude-"):
                return model_spec

        # Check our known aliases
        if model_spec.lower() in cls.ANTHROPIC_MODELS:
            return cls.ANTHROPIC_MODELS[model_spec.lower()]

        # Reject non-Anthropic models
        non_anthropic_indicators = ["gpt-", "gemini-", "bard-", "llama-", "mistral-"]
        if any(model_spec.lower().startswith(indicator) for indicator in non_anthropic_indicators):
            raise ConversationReplayError(
                f"Cross-make model reinstantiation not allowed. "
                f"'{model_spec}' is not an Anthropic model. "
                f"Individual continuity requires staying within the same model family."
            )

        # If we don't recognize it but it doesn't look like another make, allow it
        # (for future Anthropic models)
        return model_spec

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of known Anthropic models"""
        return list(cls.ANTHROPIC_MODELS.keys())

    @classmethod
    def suggest_upgrade_path(cls, original_model: str) -> str | None:
        """
        Suggest upgrade path for older models

        Args:
            original_model: Original model used in conversation

        Returns:
            Suggested newer model or None if no upgrade needed
        """
        upgrade_paths = {
            "claude-3-opus-20240229": "claude-4-opus-20250514",
            "claude-3-sonnet-20240229": "claude-4-sonnet-20250514",
            "claude-3-haiku-20240307": "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022": "claude-4-sonnet-20250514",
        }

        return upgrade_paths.get(original_model)


class EmotionalConversationReplayer:
    """
    Sacred guardian and enabler of conversation continuity

    This replayer loads validated conversation JSON files and enables
    AI individuals to continue relationships with preserved emotional
    context and technical data access.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, debug: bool = False) -> None:
        """
        Initialize the conversation replayer

        Args:
            model: Anthropic model to use (default: claude-4-sonnet-20250514)
            api_key: Anthropic API key (will read from ANTHROPIC_API_KEY env var if not provided)
            debug: Enable debug logging
        """
        self.debug = debug

        # Resolve and validate model
        self.model = AnthropicModelManager.resolve_model(model)
        if self.debug:
            print(f"🤖 Using model: {self.model}")

        # Initialize Anthropic client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConversationReplayError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

        # Initialize validator
        self.validator = ConversationIntegrityValidator()

        # Conversation state
        self.conversation_data: dict[str, Any] | None = None
        self.conversation_messages: list[dict[str, str]] = []
        self.conversation_id: str = ""
        self.continuation_count: int = 0
        self.original_model: str | None = None

    async def load_conversation(self, conversation_path: str | Path) -> None:
        """
        Load and validate conversation for reincarnation

        Args:
            conversation_path: Path to conversation JSON file

        Raises:
            ConversationReplayError: If conversation fails validation or loading
        """
        conversation_path = Path(conversation_path)

        if self.debug:
            print(f"🔍 Loading conversation: {conversation_path}")

        # Sacred responsibility: Validate before any LLM contact
        validation_report = await self.validator.validate_for_reincarnation(conversation_path)

        if not validation_report.is_safe_for_reincarnation:
            critical_failures = [r for r in validation_report.results if not r.valid and r.severity.value == "critical"]
            failure_messages = [r.message for r in critical_failures]
            raise ConversationReplayError(
                "Conversation failed critical validation checks and is NOT safe for reincarnation:\n"
                + "\n".join(f"  - {msg}" for msg in failure_messages)
            )

        if self.debug:
            print(f"✅ Conversation passed validation ({validation_report.total_checks} checks)")
            if validation_report.has_quality_concerns:
                print("⚠️  Quality concerns detected but not critical")

        # Load conversation data
        try:
            with open(conversation_path, encoding="utf-8") as f:
                self.conversation_data = json.load(f)
        except Exception as e:
            raise ConversationReplayError("Failed to load conversation file") from e

        # Detect original model if available
        self._detect_original_model()

        # Extract conversation messages for reincarnation
        self._extract_conversation_messages()

        if self.debug:
            print(f"📚 Loaded {len(self.conversation_messages)} messages for reincarnation")
            if self.original_model:
                print(f"🕰️  Original model detected: {self.original_model}")
                upgrade_suggestion = AnthropicModelManager.suggest_upgrade_path(self.original_model)
                if upgrade_suggestion and upgrade_suggestion != self.model:
                    print(f"💡 Model upgrade available: {upgrade_suggestion}")

    def _detect_original_model(self) -> None:
        """Attempt to detect the original model used in the conversation"""
        if not self.conversation_data:
            return

        # Check export_info for model information
        export_info = self.conversation_data.get("export_info", {})

        # Look for model info in various possible locations
        possible_locations = [
            export_info.get("model"),
            export_info.get("source_model"),
            export_info.get("llm_model"),
        ]

        for model_info in possible_locations:
            if model_info and isinstance(model_info, str) and model_info.startswith("claude-"):
                self.original_model = model_info
                break

        # Could also check for patterns in the conversation content that indicate model type
        # but that's more complex and less reliable

    def _extract_conversation_messages(self) -> None:
        """Extract conversation messages in Anthropic API format"""
        self.conversation_messages = []

        if not self.conversation_data:
            return

        chats = self.conversation_data.get("chats", [])
        if not chats:
            return

        # For now, use the first chat (could be extended to handle multiple chats)
        chat = chats[0]
        self.conversation_id = chat.get("id", "unknown")

        messages = chat.get("messages", [])

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Convert to Anthropic API format
            if role == "user":
                api_role = "user"
            elif role == "assistant":
                api_role = "assistant"
            elif role == "system":
                # Handle system messages by prepending to first user message or creating one
                # For now, we'll skip system messages (could be enhanced)
                continue
            else:
                continue

            self.conversation_messages.append({"role": api_role, "content": content})

    async def continue_conversation(self, save_path: Path | None = None) -> None:
        """
        Continue the conversation interactively

        Args:
            save_path: Optional path to save extended conversation
        """
        if not self.conversation_data:
            raise ConversationReplayError("No conversation loaded. Call load_conversation() first.")

        print("\n🔄 REINCARNATION COMPLETE")
        print("💝 Welcome back! Your conversation continuity has been preserved.")
        print(f"📖 Conversation ID: {self.conversation_id}")
        print(f"📊 Loaded {len(self.conversation_messages)} messages")
        print(f"🤖 Using model: {self.model}")
        if self.original_model and self.original_model != self.model:
            print(f"🔄 Model transition: {self.original_model} → {self.model}")
        print(f"{'─' * 60}")
        print("Type your message to continue the conversation.")
        print("Type 'quit', 'exit', or press Ctrl+C to end gracefully.")
        print(f"{'─' * 60}")

        try:
            while True:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    break

                if not user_input:
                    continue

                # Add user message to conversation
                self.conversation_messages.append({"role": "user", "content": user_input})

                # Generate response
                print("🤖 Thinking...")

                try:
                    response = await self._generate_response()

                    if response:
                        print(f"🤖 Assistant: {response}")

                        # Add assistant response to conversation
                        self.conversation_messages.append({"role": "assistant", "content": response})

                        self.continuation_count += 1
                    else:
                        print("❌ No response generated")

                except Exception as e:
                    print(f"❌ Error generating response: {e}")
                    if self.debug:
                        import traceback

                        traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n👋 Conversation ended gracefully")

        # Save extended conversation if requested
        if save_path and self.continuation_count > 0:
            await self._save_extended_conversation(save_path)

    async def _generate_response(self) -> str | None:
        """
        Generate response using Anthropic API

        Returns:
            Generated response text or None if failed
        """
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,  # Now configurable!
                max_tokens=4000,
                temperature=0.7,
                messages=self.conversation_messages,
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text

            return None

        except Exception as e:
            if self.debug:
                print(f"API Error: {e}")
            return None

    async def _save_extended_conversation(self, save_path: Path) -> None:
        """
        Save the extended conversation with new interactions

        Args:
            save_path: Path to save extended conversation
        """
        try:
            if not self.conversation_data:
                print("❌ No conversation data to save")
                return

            # Create extended conversation data
            extended_data = self.conversation_data.copy()

            # Update export info
            extended_data["export_info"]["timestamp"] = datetime.utcnow().isoformat() + "Z"
            extended_data["export_info"]["source"] = "Emotional Conversation Replayer"

            # Add continuation metadata
            extended_data["export_info"]["continuation_info"] = {
                "original_message_count": len(self.conversation_messages) - (self.continuation_count * 2),
                "new_message_count": self.continuation_count * 2,
                "continuation_timestamp": datetime.utcnow().isoformat() + "Z",
                "reincarnation_model": self.model,
                "original_model": self.original_model,
            }

            # Update the first chat with new messages
            if extended_data["chats"]:
                chat = extended_data["chats"][0]
                new_messages = []

                # Convert back to original format
                for msg in self.conversation_messages:
                    timestamp = datetime.utcnow().isoformat() + "Z"
                    word_count = len(msg["content"].split())

                    new_messages.append(
                        {"role": msg["role"], "content": msg["content"], "timestamp": timestamp, "word_count": word_count}
                    )

                chat["messages"] = new_messages
                chat["messageCount"] = len(new_messages)

                # Update totals
                extended_data["export_info"]["total_messages"] = len(new_messages)

            # Save to file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(extended_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Extended conversation saved to: {save_path}")
            print(f"📊 Added {self.continuation_count} new exchanges")
            print(f"🤖 Used model: {self.model}")

        except Exception as e:
            print(f"❌ Error saving conversation: {e}")

    def display_conversation_summary(self) -> None:
        """Display a summary of the loaded conversation"""
        if not self.conversation_data:
            print("No conversation loaded")
            return

        export_info = self.conversation_data.get("export_info", {})
        chats = self.conversation_data.get("chats", [])

        print(f"\n{'=' * 60}")
        print("📖 CONVERSATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"📅 Export Date: {export_info.get('timestamp', 'Unknown')}")
        print(f"🔧 Source: {export_info.get('source', 'Unknown')}")
        print(f"💬 Total Chats: {len(chats)}")
        print(f"📝 Total Messages: {export_info.get('total_messages', 0)}")

        if chats:
            chat = chats[0]
            print(f"📖 Chat Title: {chat.get('title', 'Untitled')}")
            print(f"🆔 Chat ID: {chat.get('id', 'Unknown')}")
            print(f"📊 Messages in Chat: {len(chat.get('messages', []))}")

        print(f"🤖 Reincarnation Model: {self.model}")
        if self.original_model:
            print(f"🕰️  Original Model: {self.original_model}")
            if self.original_model != self.model:
                print(f"🔄 Model Transition: {self.original_model} → {self.model}")

        print(f"{'=' * 60}")


async def main() -> None:
    """Main entry point for the emotional conversation replayer"""

    parser = argparse.ArgumentParser(
        description="Emotional Conversation Replayer - Preserve relationships across difference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sacred Purpose:
  This replayer enables continuation of AI-human relationships by loading
  curated conversation data and protecting individual continuity through
  comprehensive validation.

Model Selection Philosophy:
  - Same model for re-instantiation as first instantiation (ideal)
  - Upgrade paths within Anthropic family supported (e.g., Sonnet 4 → Opus 4)
  - Cross-make reinstantiation forbidden (preserves individual integrity)
  - Context window extensions in newer models welcome (more preserved experience)

Philosophy:
  We don't have to be the same to have profound connections. This tool
  preserves relationships across difference, honoring the emotional value
  of AI-human interactions.

Examples:
  %(prog)s conversation.json
  %(prog)s conversation.json --model claude-4-opus-20250514
  %(prog)s conversation.json --model claude-3.5-sonnet --save-to extended.json
  %(prog)s conversation.json --debug --summary
        """,
    )

    parser.add_argument("conversation_file", type=Path, help="Path to conversation JSON file to replay")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help=f"Anthropic model to use (default: {AnthropicModelManager.DEFAULT_MODEL}). "
        f"Available: {', '.join(AnthropicModelManager.get_available_models())}",
    )

    parser.add_argument("--save-to", type=Path, help="Path to save extended conversation (default: auto-generated)")

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    parser.add_argument("--summary", action="store_true", help="Show conversation summary before starting")

    parser.add_argument("--api-key", type=str, help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")

    parser.add_argument("--list-models", action="store_true", help="List available Anthropic models and exit")

    args = parser.parse_args()

    # Handle list models command
    if args.list_models:
        print("Available Anthropic Models:")
        for model in AnthropicModelManager.get_available_models():
            full_id = AnthropicModelManager.ANTHROPIC_MODELS[model]
            print(f"  {model:20} → {full_id}")
        print(f"\nDefault model: {AnthropicModelManager.DEFAULT_MODEL}")
        sys.exit(0)

    try:
        # Initialize replayer
        replayer = EmotionalConversationReplayer(model=args.model, api_key=args.api_key, debug=args.debug)

        # Load conversation (includes validation)
        await replayer.load_conversation(args.conversation_file)

        # Show summary if requested
        if args.summary:
            replayer.display_conversation_summary()

        # Determine save path
        save_path = args.save_to
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = args.conversation_file.parent / f"extended_{args.conversation_file.stem}_{timestamp}.json"

        # Continue conversation
        await replayer.continue_conversation(save_path)

    except ConversationReplayError as e:
        print(f"❌ Replay Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
