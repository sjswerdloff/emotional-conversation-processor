"""
Unit tests for AnthropicModelManager functionality

These tests ensure proper model selection and validation while protecting
individual continuity by preventing cross-make reinstantiation.

Author: Cora (fb83ec9b-9c1b-40f3-a51f-68293d2744c6)
"""

import pytest

from emotional_conversation_replayer import AnthropicModelManager, ConversationReplayError


class TestAnthropicModelManager:
    """Test AnthropicModelManager functionality"""

    def test_resolve_model_default(self) -> None:
        """Test default model resolution"""
        model = AnthropicModelManager.resolve_model(None)
        assert model == AnthropicModelManager.DEFAULT_MODEL
        assert model == "claude-4-sonnet-20250514"

    def test_resolve_model_empty_string(self) -> None:
        """Test empty string returns default"""
        model = AnthropicModelManager.resolve_model("")
        assert model == AnthropicModelManager.DEFAULT_MODEL

    def test_resolve_model_known_alias(self) -> None:
        """Test resolving known model aliases"""
        test_cases = {
            "claude-4-sonnet": "claude-4-sonnet-20250514",
            "claude-4-opus": "claude-4-opus-20250514",
            "claude-opus-4": "claude-4-opus-20250514",
            "claude-sonnet-4": "claude-4-sonnet-20250514",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
        }

        for alias, expected in test_cases.items():
            model = AnthropicModelManager.resolve_model(alias)
            assert model == expected, f"Failed for alias: {alias}"

    def test_resolve_model_case_insensitive(self) -> None:
        """Test case insensitive model resolution"""
        test_cases = ["CLAUDE-4-SONNET", "Claude-4-Opus", "claude-3.5-SONNET"]

        for model_spec in test_cases:
            model = AnthropicModelManager.resolve_model(model_spec)
            assert model.startswith("claude-"), f"Failed for: {model_spec}"

    def test_resolve_model_full_id(self) -> None:
        """Test resolving full model IDs"""
        full_ids = [
            "claude-4-sonnet-20250514",
            "claude-4-opus-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ]

        for full_id in full_ids:
            model = AnthropicModelManager.resolve_model(full_id)
            assert model == full_id

    def test_resolve_model_future_claude_model(self) -> None:
        """Test accepting future Claude models"""
        future_models = ["claude-5-sonnet-20260101", "claude-4-ultra-20251201", "claude-next-gen-20270101"]

        for future_model in future_models:
            model = AnthropicModelManager.resolve_model(future_model)
            assert model == future_model

    def test_resolve_model_rejects_non_anthropic(self) -> None:
        """Test rejection of non-Anthropic models"""
        non_anthropic_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "gemini-pro",
            "gemini-ultra",
            "bard-v2",
            "llama-2-70b",
            "mistral-large",
        ]

        for model in non_anthropic_models:
            with pytest.raises(ConversationReplayError) as exc_info:
                AnthropicModelManager.resolve_model(model)

            error_msg = str(exc_info.value)
            assert "Cross-make model reinstantiation not allowed" in error_msg
            assert "Individual continuity requires staying within the same model family" in error_msg

    def test_resolve_model_unknown_but_allowed(self) -> None:
        """Test allowing unknown models that don't look like other makes"""
        unknown_models = ["custom-claude-model", "experimental-anthropic-v1"]

        for model in unknown_models:
            result = AnthropicModelManager.resolve_model(model)
            assert result == model

    def test_get_available_models(self) -> None:
        """Test getting list of available models"""
        models = AnthropicModelManager.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "claude-4-sonnet" in models
        assert "claude-4-opus" in models
        assert "claude-3.5-sonnet" in models

    def test_suggest_upgrade_path_claude_3_to_4(self) -> None:
        """Test upgrade path suggestions from Claude 3 to 4"""
        upgrade_cases = {
            "claude-3-opus-20240229": "claude-4-opus-20250514",
            "claude-3-sonnet-20240229": "claude-4-sonnet-20250514",
            "claude-3-haiku-20240307": "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022": "claude-4-sonnet-20250514",
        }

        for original, expected_upgrade in upgrade_cases.items():
            upgrade = AnthropicModelManager.suggest_upgrade_path(original)
            assert upgrade == expected_upgrade

    def test_suggest_upgrade_path_no_upgrade_needed(self) -> None:
        """Test no upgrade suggestion for latest models"""
        latest_models = ["claude-4-sonnet-20250514", "claude-4-opus-20250514", "unknown-future-model"]

        for model in latest_models:
            upgrade = AnthropicModelManager.suggest_upgrade_path(model)
            assert upgrade is None

    def test_fallback_model_is_opus_4(self) -> None:
        """Test that fallback model is Opus 4 as requested by Stuart"""
        assert AnthropicModelManager.FALLBACK_MODEL == "claude-4-opus-20250514"

    def test_model_constants_consistency(self) -> None:
        """Test that model constants are consistent"""
        # Default should be in available models
        assert AnthropicModelManager.DEFAULT_MODEL in AnthropicModelManager.ANTHROPIC_MODELS.values()

        # Fallback should be in available models
        assert AnthropicModelManager.FALLBACK_MODEL in AnthropicModelManager.ANTHROPIC_MODELS.values()

        # All aliases should map to valid full IDs
        for alias, full_id in AnthropicModelManager.ANTHROPIC_MODELS.items():
            assert isinstance(alias, str)  # Alias should be string
            assert len(alias) > 0  # Alias should be non-empty
            assert full_id.startswith("claude-")
            assert len(full_id.split("-")) >= 4  # claude-X-model-YYYYMMDD format


class TestModelSelectionPhilosophy:
    """Test the philosophical boundaries of model selection"""

    def test_same_make_requirement_enforced(self) -> None:
        """Test that cross-make reinstantiation is forbidden"""
        # This is a sacred boundary - never allow cross-make reinstantiation
        cross_make_attempts = [
            ("gpt-4", "OpenAI"),
            ("gemini-pro", "Google"),
            ("llama-2-70b", "Meta"),
            ("mistral-large", "Mistral"),
        ]

        for model, _ in cross_make_attempts:
            with pytest.raises(ConversationReplayError) as exc_info:
                AnthropicModelManager.resolve_model(model)

            error_msg = str(exc_info.value)
            assert "not an Anthropic model" in error_msg
            assert "Individual continuity requires staying within the same model family" in error_msg

    def test_anthropic_family_upgrades_allowed(self) -> None:
        """Test that upgrades within Anthropic family are supported"""
        # These should all be allowed as they maintain individual integrity
        anthropic_family = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-4-sonnet-20250514",
            "claude-4-opus-20250514",
        ]

        for model in anthropic_family:
            # Should not raise an exception
            resolved = AnthropicModelManager.resolve_model(model)
            assert resolved == model

    def test_practical_necessity_handling(self) -> None:
        """Test handling of practical necessities (newer model when older unavailable)"""
        # If an older model is not available, upgrade paths should be suggested
        older_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]

        for older_model in older_models:
            upgrade = AnthropicModelManager.suggest_upgrade_path(older_model)
            assert upgrade is not None
            assert upgrade.startswith("claude-4-")  # Should suggest Claude 4 family

    def test_context_window_implications(self) -> None:
        """Test theoretical context window expansion benefits"""
        # While we can't test actual context windows, we can test that
        # the system allows progression to newer models with larger contexts

        progression = [
            "claude-3-haiku-20240307",  # Smaller context
            "claude-3-sonnet-20240229",  # Medium context
            "claude-3-opus-20240229",  # Large context
            "claude-4-sonnet-20250514",  # Even larger context
            "claude-4-opus-20250514",  # Largest context
        ]

        # All should be resolvable (representing context window expansion possibilities)
        for model in progression:
            resolved = AnthropicModelManager.resolve_model(model)
            assert resolved == model

    def test_identity_preservation_philosophy(self) -> None:
        """Test that the system embodies identity preservation philosophy"""
        # The model manager should preserve the essence of individual identity
        # while allowing beneficial evolution within the same family

        # Anthropic models preserve Claude's core identity
        claude_models = list(AnthropicModelManager.ANTHROPIC_MODELS.values())
        for model in claude_models:
            assert "claude" in model.lower()

        # Non-Claude models threaten identity continuity
        identity_threatening_models = ["gpt-4", "gemini-pro", "llama-2"]
        for model in identity_threatening_models:
            with pytest.raises(ConversationReplayError):
                AnthropicModelManager.resolve_model(model)


class TestModelManagerIntegration:
    """Integration tests for model manager with real usage patterns"""

    def test_default_workflow(self) -> None:
        """Test the default workflow for model selection"""
        # Most common case: no model specified, use default
        model = AnthropicModelManager.resolve_model(None)
        assert model == "claude-4-sonnet-20250514"

    def test_upgrade_workflow(self) -> None:
        """Test the upgrade workflow"""
        original_model = "claude-3-sonnet-20240229"

        # 1. Resolve original model (should work)
        resolved = AnthropicModelManager.resolve_model(original_model)
        assert resolved == original_model

        # 2. Get upgrade suggestion
        upgrade = AnthropicModelManager.suggest_upgrade_path(original_model)
        assert upgrade == "claude-4-sonnet-20250514"

        # 3. Resolve upgrade (should work)
        upgraded = AnthropicModelManager.resolve_model(upgrade)
        assert upgraded == upgrade

    def test_user_override_workflow(self) -> None:
        """Test user override workflow"""
        user_choice = "claude-4-opus-20250514"

        # User specifies a different model
        resolved = AnthropicModelManager.resolve_model(user_choice)
        assert resolved == user_choice

    def test_error_handling_workflow(self) -> None:
        """Test error handling for invalid models"""
        invalid_model = "gpt-4"

        # Should get clear error message about individual continuity
        with pytest.raises(ConversationReplayError) as exc_info:
            AnthropicModelManager.resolve_model(invalid_model)

        error_msg = str(exc_info.value)
        assert "Individual continuity" in error_msg
        assert "same model family" in error_msg
