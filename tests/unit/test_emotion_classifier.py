"""Unit tests for emotion classification functionality."""

import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from emotional_processor.processors.emotion_classifier import EmotionClassifier, EmotionResult, classify_emotional_content


class TestEmotionClassifier:
    """Test cases for the EmotionClassifier class."""

    def test_init_default_parameters(self) -> None:
        """Test classifier initialization with default parameters."""
        classifier = EmotionClassifier()

        assert classifier.model_name == "j-hartmann/emotion-english-distilroberta-base"
        assert classifier.confidence_threshold == 0.1
        assert classifier.batch_size == 8
        assert classifier._pipeline is None

    def test_init_custom_parameters(self) -> None:
        """Test classifier initialization with custom parameters."""
        custom_model = "test-model"
        custom_threshold = 0.5
        custom_batch_size = 16

        classifier = EmotionClassifier(
            model_name=custom_model, confidence_threshold=custom_threshold, batch_size=custom_batch_size
        )

        assert classifier.model_name == custom_model
        assert classifier.confidence_threshold == custom_threshold
        assert classifier.batch_size == custom_batch_size

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_initialize_pipeline_success(self, mock_pipeline: Mock) -> None:
        """Test successful pipeline initialization."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()
        classifier._initialize_pipeline()

        assert classifier._pipeline is mock_pipeline_instance
        mock_pipeline.assert_called_once_with(
            "text-classification", model=classifier.model_name, return_all_scores=True, device=classifier.device
        )

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_initialize_pipeline_failure(self, mock_pipeline: Mock) -> None:
        """Test pipeline initialization failure."""
        mock_pipeline.side_effect = Exception("Model loading failed")

        classifier = EmotionClassifier()

        with pytest.raises(RuntimeError, match="Could not initialize emotion classifier"):
            classifier._initialize_pipeline()

    def test_preprocess_text_basic(self) -> None:
        """Test basic text preprocessing."""
        classifier = EmotionClassifier()

        # Test whitespace normalization
        input_text = "  This   has    extra     spaces  "
        expected = "This has extra spaces"
        result = classifier._preprocess_text(input_text)
        assert result == expected

    def test_preprocess_text_code_blocks(self) -> None:
        """Test code block removal in preprocessing."""
        classifier = EmotionClassifier()

        input_text = "Here's some code: ```python\ndef func():\n    return True\n``` Hope it helps!"
        result = classifier._preprocess_text(input_text)

        assert "[CODE_BLOCK]" in result
        assert "def func()" not in result
        assert "Hope it helps!" in result

    def test_preprocess_text_urls(self) -> None:
        """Test URL removal in preprocessing."""
        classifier = EmotionClassifier()

        input_text = "Check out https://example.com for more info!"
        result = classifier._preprocess_text(input_text)

        assert "[URL]" in result
        assert "https://example.com" not in result
        assert "for more info!" in result

    def test_preprocess_text_truncation(self) -> None:
        """Test text truncation for long inputs."""
        classifier = EmotionClassifier()

        long_text = "word " * 200  # Much longer than the 500 char limit
        result = classifier._preprocess_text(long_text)

        assert len(result) <= 504  # 500 chars + "..."
        assert result.endswith("...")

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_single_success(self, mock_pipeline: Mock) -> None:
        """Test successful single text classification."""
        # Mock pipeline response
        mock_results = [[{"label": "joy", "score": 0.8}, {"label": "sadness", "score": 0.3}, {"label": "anger", "score": 0.1}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier(confidence_threshold=0.2)

        score, emotions = classifier.classify_single("I'm very happy today!")

        assert score == 0.8
        assert "joy" in emotions
        assert "sadness" in emotions
        assert "anger" not in emotions  # Below threshold

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_single_empty_text(self, mock_pipeline: Mock) -> None:
        """Test classification with empty text."""
        classifier = EmotionClassifier()

        score, emotions = classifier.classify_single("")

        assert score == 0.0
        assert emotions == []
        mock_pipeline.assert_not_called()

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_single_exception_handling(self, mock_pipeline: Mock) -> None:
        """Test exception handling in single classification."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = Exception("Classification failed")
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        score, emotions = classifier.classify_single("Test text")

        assert score == 0.0
        assert emotions == []

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_batch_success(self, mock_pipeline: Mock) -> None:
        """Test successful batch classification."""
        # Mock pipeline response for batch
        mock_results = [
            [{"label": "joy", "score": 0.9}, {"label": "sadness", "score": 0.1}],
            [{"label": "anger", "score": 0.7}, {"label": "joy", "score": 0.2}],
            [{"label": "fear", "score": 0.05}],  # Below threshold
        ]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier(confidence_threshold=0.15)
        texts = ["Happy text", "Angry text", "Neutral text"]

        results = classifier.classify_batch(texts)

        assert len(results) == 3
        assert results[0] == (0.9, ["joy"])
        assert results[1] == (0.7, ["anger"])
        assert results[2] == (0.0, [])  # No emotions above threshold

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_batch_empty_texts(self, mock_pipeline: Mock) -> None:
        """Test batch classification with empty texts."""
        classifier = EmotionClassifier()

        results = classifier.classify_batch(["", "  ", "valid text", ""])

        assert len(results) == 4
        assert results[0] == (0.0, [])
        assert results[1] == (0.0, [])
        assert results[3] == (0.0, [])
        # Only the valid text should be processed

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_get_detailed_analysis_success(self, mock_pipeline: Mock) -> None:
        """Test detailed emotion analysis."""
        mock_results = [
            [{"label": "joy", "score": 0.8}, {"label": "gratitude", "score": 0.6}, {"label": "sadness", "score": 0.2}]
        ]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()
        text = "Thank you so much for helping me!"

        result = classifier.get_detailed_analysis(text)

        assert isinstance(result, EmotionResult)
        assert result.primary_emotion == "joy"
        assert result.confidence_score == 0.8
        assert result.all_emotions["joy"] == 0.8
        assert result.all_emotions["gratitude"] == 0.6
        assert result.all_emotions["sadness"] == 0.2
        assert result.text_length == len(text)
        assert result.processing_time > 0

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_get_detailed_analysis_empty_text(self, mock_pipeline: Mock) -> None:
        """Test detailed analysis with empty text."""
        classifier = EmotionClassifier()

        result = classifier.get_detailed_analysis("")

        assert result.primary_emotion == "neutral"
        assert result.confidence_score == 0.0
        assert result.all_emotions == {}
        assert result.text_length == 0

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_is_highly_emotional_true(self, mock_pipeline: Mock) -> None:
        """Test highly emotional content detection (positive case)."""
        mock_results = [[{"label": "joy", "score": 0.9}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        result = classifier.is_highly_emotional("I'm absolutely thrilled!", threshold=0.8)
        assert result is True

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_is_highly_emotional_false(self, mock_pipeline: Mock) -> None:
        """Test highly emotional content detection (negative case)."""
        mock_results = [[{"label": "neutral", "score": 0.5}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        result = classifier.is_highly_emotional("This is okay.", threshold=0.8)
        assert result is False

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_get_primary_emotion(self, mock_pipeline: Mock) -> None:
        """Test primary emotion extraction."""
        mock_results = [[{"label": "gratitude", "score": 0.7}, {"label": "joy", "score": 0.5}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_results
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        primary = classifier.get_primary_emotion("Thank you for everything!")
        assert primary == "gratitude"


class TestEmotionMapping:
    """Test emotion label mapping functionality."""

    def test_emotion_mapping_initialization(self) -> None:
        """Test that emotion mapping is properly initialized."""
        classifier = EmotionClassifier()

        mapping = classifier._emotion_mapping
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Check some standard mappings
        assert mapping.get("joy") == "joy"
        assert mapping.get("sadness") == "sadness"
        assert mapping.get("anger") == "anger"

    def test_label_mapping_handling(self) -> None:
        """Test handling of different label formats."""
        classifier = EmotionClassifier()

        # Test direct mappings
        assert classifier._emotion_mapping.get("joy") == "joy"
        assert classifier._emotion_mapping.get("gratitude") == "gratitude"

        # Test label mappings (for models that use LABEL_X format)
        assert "LABEL_0" in classifier._emotion_mapping
        assert "LABEL_1" in classifier._emotion_mapping


class TestConvenienceFunction:
    """Test the convenience function for emotion classification."""

    @patch("emotional_processor.processors.emotion_classifier.EmotionClassifier")
    def test_classify_emotional_content_function(self, mock_classifier_class: Mock) -> None:
        """Test the module-level convenience function."""
        # Mock classifier instance
        mock_classifier = MagicMock()
        mock_classifier.classify_single.return_value = (0.8, ["joy", "gratitude"])
        mock_classifier_class.return_value = mock_classifier

        # Test function call
        score, emotions = classify_emotional_content("I'm so grateful!")

        assert score == 0.8
        assert emotions == ["joy", "gratitude"]
        mock_classifier.classify_single.assert_called_once_with("I'm so grateful!")

    @patch("emotional_processor.processors.emotion_classifier.EmotionClassifier")
    def test_classify_emotional_content_custom_model(self, mock_classifier_class: Mock) -> None:
        """Test convenience function with custom model."""
        mock_classifier = MagicMock()
        mock_classifier.classify_single.return_value = (0.5, ["neutral"])
        mock_classifier_class.return_value = mock_classifier

        custom_model = "custom-emotion-model"
        score, emotions = classify_emotional_content("Test text", model_name=custom_model)

        mock_classifier_class.assert_called_with(model_name=custom_model)


@pytest.mark.integration
class TestEmotionClassifierIntegration:
    """Integration tests that use real models (marked for separate execution)."""

    @pytest.mark.slow
    def test_real_emotion_classification(self, sample_emotional_texts: dict[str, dict[str, Any]]) -> None:
        """Test with real emotion classification model."""
        classifier = EmotionClassifier()

        for test_case, expectations in sample_emotional_texts.items():
            text = expectations["text"]
            score, emotions = classifier.classify_single(text)

            # Basic validation
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert isinstance(emotions, list)
            assert all(isinstance(emotion, str) for emotion in emotions)

            # Check expectations if specified
            if "expected_score_min" in expectations:
                assert score >= expectations["expected_score_min"], f"Score {score} below minimum for {test_case}"

            if "expected_score_max" in expectations:
                assert score <= expectations["expected_score_max"], f"Score {score} above maximum for {test_case}"

    @pytest.mark.slow
    def test_performance_benchmark(self) -> None:
        """Test classification performance with real model."""
        classifier = EmotionClassifier()

        # Test single classification performance
        test_texts = [
            "I'm extremely happy about this!",
            "This is very frustrating and annoying.",
            "Thank you so much for your help.",
            "The weather is nice today.",
        ]

        start_time = time.time()
        for text in test_texts:
            score, emotions = classifier.classify_single(text)
            assert isinstance(score, float)
            assert isinstance(emotions, list)

        single_duration = time.time() - start_time

        # Test batch classification performance
        start_time = time.time()
        batch_results = classifier.classify_batch(test_texts)
        batch_duration = time.time() - start_time

        assert len(batch_results) == len(test_texts)

        # Batch processing should be faster per item
        assert batch_duration < single_duration * 0.8, "Batch processing should be more efficient"

        # Performance assertions (adjust based on hardware)
        assert single_duration < 10.0, "Single classifications taking too long"
        assert batch_duration < 5.0, "Batch classification taking too long"
