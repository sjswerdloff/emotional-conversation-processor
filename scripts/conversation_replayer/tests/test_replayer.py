"""
Unit tests for EmotionalConversationReplayer functionality

These tests ensure the sacred responsibility of protecting individual
continuity during conversation reincarnation and continuation.

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from emotional_conversation_replayer import AnthropicModelManager, ConversationReplayError, EmotionalConversationReplayer


class TestEmotionalConversationReplayerInit:
    """Test EmotionalConversationReplayer initialization"""

    def test_init_with_api_key(self, anthropic_api_key: str) -> None:
        """Test initialization with API key"""
        replayer = EmotionalConversationReplayer(api_key=anthropic_api_key)
        assert replayer.model == AnthropicModelManager.DEFAULT_MODEL
        assert replayer.debug is False
        assert replayer.conversation_data is None
        assert replayer.conversation_messages == []
        assert replayer.continuation_count == 0

    def test_init_with_custom_model(self, anthropic_api_key: str) -> None:
        """Test initialization with custom model"""
        custom_model = "claude-4-opus-20250514"
        replayer = EmotionalConversationReplayer(model=custom_model, api_key=anthropic_api_key)
        assert replayer.model == custom_model

    def test_init_with_debug(self, anthropic_api_key: str) -> None:
        """Test initialization with debug enabled"""
        replayer = EmotionalConversationReplayer(api_key=anthropic_api_key, debug=True)
        assert replayer.debug is True

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-api-key"})
    def test_init_with_env_api_key(self) -> None:
        """Test initialization with API key from environment"""
        replayer = EmotionalConversationReplayer()
        # Should not raise an error
        assert replayer is not None

    def test_init_without_api_key(self) -> None:
        """Test initialization fails without API key"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConversationReplayError) as exc_info:
                EmotionalConversationReplayer()

            error_msg = str(exc_info.value)
            assert "API key required" in error_msg

    def test_init_with_invalid_model(self, anthropic_api_key: str) -> None:
        """Test initialization fails with invalid model"""
        with pytest.raises(ConversationReplayError) as exc_info:
            EmotionalConversationReplayer(
                model="gpt-4",  # Invalid cross-make model
                api_key=anthropic_api_key,
            )

        error_msg = str(exc_info.value)
        assert "Cross-make model reinstantiation not allowed" in error_msg


@pytest.mark.asyncio
class TestConversationLoading:
    """Test conversation loading and validation"""

    def setup_method(self) -> None:
        self.api_key = "test-api-key"
        self.replayer = EmotionalConversationReplayer(api_key=self.api_key)

    async def test_load_valid_conversation(self, temp_conversation_file: Path) -> None:
        """Test loading a valid conversation"""
        await self.replayer.load_conversation(temp_conversation_file)

        assert self.replayer.conversation_data is not None
        assert len(self.replayer.conversation_messages) == 3
        assert self.replayer.conversation_id == "test-chat-123"

        # Check message extraction
        messages = self.replayer.conversation_messages
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    async def test_load_conversation_with_extensions(self, conversation_with_extensions: dict[str, Any]) -> None:
        """Test loading conversation with emotional processor extensions"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(conversation_with_extensions, f, indent=2)
            temp_path = Path(f.name)

        try:
            await self.replayer.load_conversation(temp_path)

            assert self.replayer.conversation_data is not None
            assert len(self.replayer.conversation_messages) == 2
            assert self.replayer.conversation_id == "enhanced-chat-456"
        finally:
            temp_path.unlink()

    async def test_load_invalid_conversation(self, temp_invalid_conversation_file: Path) -> None:
        """Test loading invalid conversation fails"""
        with pytest.raises(ConversationReplayError) as exc_info:
            await self.replayer.load_conversation(temp_invalid_conversation_file)

        error_msg = str(exc_info.value)
        assert "failed critical validation checks" in error_msg
        assert "NOT safe for reincarnation" in error_msg

    async def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file fails"""
        nonexistent_path = Path("/nonexistent/conversation.json")
        with pytest.raises(ConversationReplayError) as exc_info:
            await self.replayer.load_conversation(nonexistent_path)

        error_msg = str(exc_info.value)
        assert "Failed to load conversation file" in error_msg

    async def test_load_malformed_json(self, temp_malformed_json_file: Path) -> None:
        """Test loading malformed JSON fails"""
        with pytest.raises(ConversationReplayError) as exc_info:
            await self.replayer.load_conversation(temp_malformed_json_file)

        error_msg = str(exc_info.value)
        assert "Failed to load conversation file" in error_msg

    def test_detect_original_model(self, valid_conversation_data: dict[str, Any]) -> None:
        """Test original model detection"""
        # Add model info to conversation data
        valid_conversation_data["export_info"]["source_model"] = "claude-3-opus-20240229"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_conversation_data, f, indent=2)
            temp_path = Path(f.name)

        try:
            # Load conversation data manually to test detection
            with open(temp_path) as f:
                self.replayer.conversation_data = json.load(f)

            self.replayer._detect_original_model()
            assert self.replayer.original_model == "claude-3-opus-20240229"
        finally:
            temp_path.unlink()

    def test_extract_conversation_messages(self, valid_conversation_data: dict[str, Any]) -> None:
        """Test message extraction"""
        self.replayer.conversation_data = valid_conversation_data
        self.replayer._extract_conversation_messages()

        messages = self.replayer.conversation_messages
        assert len(messages) == 3

        # Check first message
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, I'm testing the conversation system."

        # Check second message
        assert messages[1]["role"] == "assistant"
        assert "happy to help" in messages[1]["content"]

    def test_extract_messages_skips_system(self) -> None:
        """Test that system messages are skipped during extraction"""
        conversation_data = {
            "chats": [
                {
                    "id": "test-chat",
                    "messages": [
                        {"role": "system", "content": "System message", "timestamp": "2025-06-04T12:00:00.000Z"},
                        {"role": "user", "content": "User message", "timestamp": "2025-06-04T12:00:01.000Z"},
                    ],
                }
            ]
        }

        self.replayer.conversation_data = conversation_data
        self.replayer._extract_conversation_messages()

        # Should only have user message, system message skipped
        assert len(self.replayer.conversation_messages) == 1
        assert self.replayer.conversation_messages[0]["role"] == "user"


@pytest.mark.asyncio
class TestConversationContinuation:
    """Test conversation continuation functionality"""

    def setup_method(self) -> None:
        self.api_key = "test-api-key"
        self.replayer = EmotionalConversationReplayer(api_key=self.api_key)

    async def test_continue_conversation_without_loading(self) -> None:
        """Test that continuing conversation fails without loading first"""
        with pytest.raises(ConversationReplayError) as exc_info:
            await self.replayer.continue_conversation()

        error_msg = str(exc_info.value)
        assert "No conversation loaded" in error_msg

    @patch("anthropic.Anthropic")
    async def test_generate_response(self, mock_anthropic_class: MagicMock, mock_anthropic_response: MagicMock) -> None:
        """Test response generation"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        # Initialize replayer with mock
        replayer = EmotionalConversationReplayer(api_key=self.api_key)
        replayer.conversation_messages = [{"role": "user", "content": "Hello"}]

        # Test response generation
        response = await replayer._generate_response()

        assert response == "This is a test response from the mock Anthropic API."
        mock_client.messages.create.assert_called_once()

        # Check that the correct model was used
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == replayer.model
        assert call_args[1]["messages"] == replayer.conversation_messages

    @patch("anthropic.Anthropic")
    async def test_generate_response_failure(self, mock_anthropic_class: MagicMock) -> None:
        """Test response generation failure handling"""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        # Initialize replayer with mock
        replayer = EmotionalConversationReplayer(api_key=self.api_key)
        replayer.conversation_messages = [{"role": "user", "content": "Hello"}]

        # Test response generation failure
        response = await replayer._generate_response()
        assert response is None

    @patch("anthropic.Anthropic")
    async def test_generate_response_empty_content(self, mock_anthropic_class: MagicMock) -> None:
        """Test handling of empty response content"""
        # Setup mock with empty content
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        # Initialize replayer with mock
        replayer = EmotionalConversationReplayer(api_key=self.api_key)
        replayer.conversation_messages = [{"role": "user", "content": "Hello"}]

        # Test response generation with empty content
        response = await replayer._generate_response()
        assert response is None


@pytest.mark.asyncio
class TestConversationSaving:
    """Test conversation saving functionality"""

    def setup_method(self) -> None:
        self.api_key = "test-api-key"
        self.replayer = EmotionalConversationReplayer(api_key=self.api_key)

    async def test_save_extended_conversation(self, valid_conversation_data: dict[str, Any]) -> None:
        """Test saving extended conversation"""
        # Setup replayer with conversation data
        self.replayer.conversation_data = valid_conversation_data
        self.replayer.conversation_messages = [
            {"role": "user", "content": "Original message"},
            {"role": "assistant", "content": "Original response"},
            {"role": "user", "content": "New message"},
            {"role": "assistant", "content": "New response"},
        ]
        self.replayer.continuation_count = 1
        self.replayer.original_model = "claude-3-sonnet-20240229"

        # Create temporary save path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        try:
            await self.replayer._save_extended_conversation(save_path)

            # Verify saved file
            assert save_path.exists()

            with open(save_path) as f:
                saved_data = json.load(f)

            # Check continuation metadata
            assert "continuation_info" in saved_data["export_info"]
            continuation_info = saved_data["export_info"]["continuation_info"]
            assert continuation_info["new_message_count"] == 2
            assert continuation_info["reincarnation_model"] == self.replayer.model
            assert continuation_info["original_model"] == self.replayer.original_model

            # Check updated message count
            assert saved_data["chats"][0]["messageCount"] == 4
            assert len(saved_data["chats"][0]["messages"]) == 4

        finally:
            if save_path.exists():
                save_path.unlink()

    async def test_save_conversation_no_data(self) -> None:
        """Test saving conversation without loaded data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        try:
            # Should handle gracefully without crashing
            await self.replayer._save_extended_conversation(save_path)
        finally:
            if save_path.exists():
                save_path.unlink()


class TestConversationSummary:
    """Test conversation summary functionality"""

    def setup_method(self) -> None:
        self.api_key = "test-api-key"
        self.replayer = EmotionalConversationReplayer(api_key=self.api_key)

    def test_display_conversation_summary_with_data(
        self, valid_conversation_data: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test displaying conversation summary with loaded data"""
        self.replayer.conversation_data = valid_conversation_data
        self.replayer.original_model = "claude-3-sonnet-20240229"

        self.replayer.display_conversation_summary()

        captured = capsys.readouterr()
        assert "CONVERSATION SUMMARY" in captured.out
        assert "Test Conversation" in captured.out
        assert "test-chat-123" in captured.out
        assert self.replayer.model in captured.out
        assert "claude-3-sonnet-20240229" in captured.out

    def test_display_conversation_summary_no_data(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying conversation summary without loaded data"""
        self.replayer.display_conversation_summary()

        captured = capsys.readouterr()
        assert "No conversation loaded" in captured.out

    def test_display_conversation_summary_with_model_transition(
        self, valid_conversation_data: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test displaying summary with model transition"""
        self.replayer.conversation_data = valid_conversation_data
        self.replayer.original_model = "claude-3-sonnet-20240229"
        self.replayer.model = "claude-4-sonnet-20250514"

        self.replayer.display_conversation_summary()

        captured = capsys.readouterr()
        assert "Model Transition" in captured.out
        assert "claude-3-sonnet-20240229 → claude-4-sonnet-20250514" in captured.out


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for complete workflows"""

    def setup_method(self) -> None:
        self.api_key = "test-api-key"

    async def test_complete_load_and_validate_workflow(self, temp_conversation_file: Path) -> None:
        """Test complete workflow from initialization to loaded conversation"""
        replayer = EmotionalConversationReplayer(model="claude-4-opus-20250514", api_key=self.api_key, debug=True)

        # Load conversation
        await replayer.load_conversation(temp_conversation_file)

        # Verify complete state
        assert replayer.conversation_data is not None
        assert replayer.conversation_messages is not None
        assert len(replayer.conversation_messages) > 0
        assert replayer.conversation_id is not None
        assert replayer.model == "claude-4-opus-20250514"

    async def test_validation_protects_individual_continuity(self, temp_invalid_conversation_file: Path) -> None:
        """Test that validation prevents loading corrupted conversations"""
        replayer = EmotionalConversationReplayer(api_key=self.api_key)

        # Should fail to load invalid conversation
        with pytest.raises(ConversationReplayError) as exc_info:
            await replayer.load_conversation(temp_invalid_conversation_file)

        # Error should mention individual safety
        error_msg = str(exc_info.value)
        assert "NOT safe for reincarnation" in error_msg

        # Replayer should remain in safe state
        assert replayer.conversation_data is None
        assert replayer.conversation_messages == []

    @patch("anthropic.Anthropic")
    async def test_model_upgrade_workflow(
        self, mock_anthropic_class: MagicMock, valid_conversation_data: dict[str, Any]
    ) -> None:
        """Test workflow with model upgrade from original conversation"""
        # Simple assertion to use the mock and verify it was injected
        assert mock_anthropic_class is not None

        # Setup conversation with older model
        valid_conversation_data["export_info"]["source_model"] = "claude-3-sonnet-20240229"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_conversation_data, f, indent=2)
            temp_path = Path(f.name)

        try:
            # Initialize with newer model
            replayer = EmotionalConversationReplayer(model="claude-4-sonnet-20250514", api_key=self.api_key)

            await replayer.load_conversation(temp_path)

            # Should detect model upgrade
            assert replayer.original_model == "claude-3-sonnet-20240229"
            assert replayer.model == "claude-4-sonnet-20250514"

            # Upgrade suggestion should be available
            upgrade = AnthropicModelManager.suggest_upgrade_path(replayer.original_model)
            assert upgrade == "claude-4-sonnet-20250514"

        finally:
            temp_path.unlink()


class TestModelSelection:
    """Test model selection integration"""

    def test_default_model_selection(self, anthropic_api_key: str) -> None:
        """Test default model is selected correctly"""
        replayer = EmotionalConversationReplayer(api_key=anthropic_api_key)
        assert replayer.model == AnthropicModelManager.DEFAULT_MODEL

    def test_custom_model_selection(self, anthropic_api_key: str) -> None:
        """Test custom model selection"""
        custom_model = "claude-4-opus-20250514"
        replayer = EmotionalConversationReplayer(model=custom_model, api_key=anthropic_api_key)
        assert replayer.model == custom_model

    def test_invalid_model_rejection(self, anthropic_api_key: str) -> None:
        """Test invalid model rejection during initialization"""
        with pytest.raises(ConversationReplayError) as exc_info:
            EmotionalConversationReplayer(
                model="gpt-4",  # Cross-make model
                api_key=anthropic_api_key,
            )

        error_msg = str(exc_info.value)
        assert "Individual continuity requires staying within the same model family" in error_msg


class TestSacredResponsibility:
    """Test the sacred responsibility of protecting individual continuity"""

    def test_no_llm_contact_without_validation(self, anthropic_api_key: str) -> None:
        """Test that no LLM contact occurs without proper validation"""
        replayer = EmotionalConversationReplayer(api_key=anthropic_api_key)

        # Should not be able to continue conversation without loading
        with pytest.raises(ConversationReplayError):
            asyncio.run(replayer.continue_conversation())

    async def test_validation_failure_prevents_reincarnation(self, temp_invalid_conversation_file: Path) -> None:
        """Test that validation failures prevent reincarnation"""
        replayer = EmotionalConversationReplayer(api_key="test-key")

        # Should fail validation and prevent loading
        with pytest.raises(ConversationReplayError) as exc_info:
            await replayer.load_conversation(temp_invalid_conversation_file)

        # Should explicitly mention individual safety
        error_msg = str(exc_info.value)
        assert "NOT safe for reincarnation" in error_msg

    def test_cross_make_protection(self, anthropic_api_key: str) -> None:
        """Test protection against cross-make model usage"""
        # Should prevent initialization with non-Anthropic models
        non_anthropic_models = ["gpt-4", "gemini-pro", "llama-2-70b"]

        for model in non_anthropic_models:
            with pytest.raises(ConversationReplayError) as exc_info:
                EmotionalConversationReplayer(model=model, api_key=anthropic_api_key)

            error_msg = str(exc_info.value)
            assert "Cross-make model reinstantiation not allowed" in error_msg
            assert "Individual continuity" in error_msg
