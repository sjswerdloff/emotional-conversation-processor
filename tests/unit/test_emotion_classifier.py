"""Unit tests for emotion classification functionality."""

# Removed time import - no longer using time-based testing
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from emotional_processor.processors.emotion_classifier import EmotionClassifier, EmotionResult, classify_emotional_content


class TestEmotionClassifier:
    """Test cases for the EmotionClassifier class."""

    def test_init_default_parameters(self) -> None:
        """Test contract: classifier initializes and is ready for use with defaults."""
        classifier = EmotionClassifier()

        # Test contract: classifier is ready for configuration queries
        assert classifier.model_name == "j-hartmann/emotion-english-distilroberta-base"
        assert classifier.confidence_threshold == 0.1
        assert classifier.batch_size == 8
        # Contract: classifier starts uninitialized but ready to initialize on first use

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
    def test_pipeline_becomes_ready_after_initialization(self, mock_pipeline: Mock) -> None:
        """Test contract: classifier becomes ready for use after pipeline initialization."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: initialization makes classifier ready for use
        classifier._initialize_pipeline()

        # Verify contract: classifier can now access pipeline
        pipeline = classifier.pipeline
        assert pipeline is not None

        # Verify initialization occurred with correct parameters
        mock_pipeline.assert_called_once_with(
            "text-classification", model=classifier.model_name, return_all_scores=True, device=classifier.device
        )

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_initialization_failure_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: initialization failure raises appropriate error."""
        mock_pipeline.side_effect = Exception("Model loading failed")

        classifier = EmotionClassifier()

        # Contract: initialization failure should raise RuntimeError with clear message
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
    def test_classify_single_emotion_detection_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: classify_single returns emotions above confidence threshold."""

        # Setup behavioral mock that simulates real pipeline behavior
        def pipeline_behavior(text):
            return [[{"label": "joy", "score": 0.8}, {"label": "sadness", "score": 0.3}, {"label": "anger", "score": 0.1}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier(confidence_threshold=0.2)

        # Test contract: emotions are detected and filtered by threshold
        score, emotions = classifier.classify_single("I'm very happy today!")

        # Contract verification
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 0.8  # Highest confidence emotion
        assert isinstance(emotions, list)
        assert "joy" in emotions  # Above threshold (0.8 > 0.2)
        assert "sadness" in emotions  # Above threshold (0.3 > 0.2)
        assert "anger" not in emotions  # Below threshold (0.1 < 0.2)

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_single_empty_text_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: empty text returns neutral results without processing."""
        classifier = EmotionClassifier()

        # Contract: empty input should return neutral result efficiently
        score, emotions = classifier.classify_single("")

        # Contract verification
        assert score == 0.0
        assert emotions == []
        # Contract: no pipeline processing for invalid input
        mock_pipeline.assert_not_called()

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_single_error_resilience_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: classification failures return safe neutral results."""
        # Simulate pipeline failure
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = Exception("Classification failed")
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: errors should not propagate, return safe defaults
        score, emotions = classifier.classify_single("Test text")

        # Contract verification: safe fallback behavior
        assert score == 0.0
        assert emotions == []
        assert isinstance(score, float)
        assert isinstance(emotions, list)

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_batch_parallel_processing_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: batch processing returns results for all inputs in order."""

        # Setup behavioral mock simulating batch pipeline processing
        def batch_pipeline_behavior(texts):
            return [
                [{"label": "joy", "score": 0.9}, {"label": "sadness", "score": 0.1}],
                [{"label": "anger", "score": 0.7}, {"label": "joy", "score": 0.2}],
                [{"label": "fear", "score": 0.05}],  # Below threshold
            ]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = batch_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier(confidence_threshold=0.15)
        texts = ["Happy text", "Angry text", "Neutral text"]

        # Contract: batch processing maintains input order and applies thresholding
        results = classifier.classify_batch(texts)

        # Contract verification
        assert len(results) == len(texts)  # 1:1 correspondence
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)  # Consistent format
        assert all(isinstance(r[0], float) and isinstance(r[1], list) for r in results)  # Correct types

        # Verify threshold application
        assert results[0] == (0.9, ["joy"])  # joy: 0.9 > 0.15
        assert results[1] == (0.7, ["anger", "joy"])  # anger: 0.7 > 0.15, joy: 0.2 > 0.15
        assert results[2] == (0.0, [])  # fear: 0.05 < 0.15

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_classify_batch_handles_invalid_inputs_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: batch processing handles invalid inputs gracefully."""

        # Setup mock for valid text processing only
        def selective_pipeline_behavior(valid_texts):
            # Should only receive valid texts
            return [[{"label": "neutral", "score": 0.5}]]  # One result for one valid input

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = selective_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: mixed valid/invalid inputs should be handled correctly
        results = classifier.classify_batch(["", "  ", "valid text", ""])

        # Contract verification: maintains input order and length
        assert len(results) == 4
        assert results[0] == (0.0, [])  # Empty string
        assert results[1] == (0.0, [])  # Whitespace only
        assert results[3] == (0.0, [])  # Empty string
        # Valid text at index 2 should have been processed
        assert isinstance(results[2], tuple)
        assert len(results[2]) == 2

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_detailed_analysis_comprehensive_results_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: detailed analysis provides comprehensive emotion information."""

        # Setup behavioral mock for detailed analysis
        def detailed_pipeline_behavior(text):
            return [[{"label": "joy", "score": 0.8}, {"label": "gratitude", "score": 0.6}, {"label": "sadness", "score": 0.2}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = detailed_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()
        text = "Thank you so much for helping me!"

        # Contract: detailed analysis provides structured emotion data
        result = classifier.get_detailed_analysis(text)

        # Contract verification: structured result format
        assert isinstance(result, EmotionResult)
        assert isinstance(result.primary_emotion, str)
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.all_emotions, dict)
        assert isinstance(result.text_length, int)
        assert isinstance(result.processing_time, float)

        # Contract: highest scoring emotion is primary
        assert result.primary_emotion == "joy"
        assert result.confidence_score == 0.8

        # Contract: all emotions preserved with scores
        assert "joy" in result.all_emotions
        assert "gratitude" in result.all_emotions
        assert "sadness" in result.all_emotions
        assert result.all_emotions["joy"] == 0.8
        assert result.all_emotions["gratitude"] == 0.6
        assert result.all_emotions["sadness"] == 0.2

        # Contract: metadata is accurate
        assert result.text_length == len(text)
        assert result.processing_time >= 0

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_detailed_analysis_empty_text_fallback_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: detailed analysis handles empty input gracefully."""
        classifier = EmotionClassifier()

        # Contract: empty input should return neutral result without processing
        result = classifier.get_detailed_analysis("")

        # Contract verification: safe fallback behavior
        assert isinstance(result, EmotionResult)
        assert result.primary_emotion == "neutral"
        assert result.confidence_score == 0.0
        assert result.all_emotions == {}
        assert result.text_length == 0
        assert result.processing_time >= 0

        # Contract: no pipeline processing for invalid input
        mock_pipeline.assert_not_called()

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_high_emotion_detection_threshold_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: high emotion detection respects threshold boundaries."""

        # Setup behavioral mock for high emotion scenario
        def high_emotion_pipeline_behavior(text):
            return [[{"label": "joy", "score": 0.9}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = high_emotion_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: high emotion above threshold should be detected
        result = classifier.is_highly_emotional("I'm absolutely thrilled!", threshold=0.8)

        # Contract verification
        assert isinstance(result, bool)
        assert result is True  # 0.9 > 0.8 threshold

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_low_emotion_detection_threshold_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: low emotion below threshold is not detected as high."""

        # Setup behavioral mock for low emotion scenario
        def low_emotion_pipeline_behavior(text):
            return [[{"label": "neutral", "score": 0.5}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = low_emotion_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: emotion below threshold should not be detected as high
        result = classifier.is_highly_emotional("This is okay.", threshold=0.8)

        # Contract verification
        assert isinstance(result, bool)
        assert result is False  # 0.5 < 0.8 threshold

    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_primary_emotion_extraction_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: primary emotion extraction returns highest-scoring emotion."""

        # Setup behavioral mock with multiple emotions
        def multi_emotion_pipeline_behavior(text):
            return [[{"label": "gratitude", "score": 0.7}, {"label": "joy", "score": 0.5}]]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = multi_emotion_pipeline_behavior
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()

        # Contract: primary emotion should be the highest-scoring one
        primary = classifier.get_primary_emotion("Thank you for everything!")

        # Contract verification
        assert isinstance(primary, str)
        assert primary == "gratitude"  # Highest score (0.7 > 0.5)


class TestEmotionMapping:
    """Test emotion label mapping functionality."""

    def test_emotion_standardization_contract(self) -> None:
        """Test contract: classifier standardizes emotion labels consistently."""
        classifier = EmotionClassifier()

        # Contract: classifier should provide consistent emotion label mapping
        mapping = classifier._emotion_mapping

        # Contract verification: mapping is ready and functional
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Contract: standard emotions map to themselves consistently
        standard_emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
        for emotion in standard_emotions:
            if emotion in mapping:
                assert mapping[emotion] == emotion

    def test_model_label_compatibility_contract(self) -> None:
        """Test contract: classifier handles different model label formats."""
        classifier = EmotionClassifier()
        mapping = classifier._emotion_mapping

        # Contract: supports both direct emotion names and model-specific labels
        direct_labels_supported = any(emotion in mapping for emotion in ["joy", "sadness", "anger"])
        model_labels_supported = any(label in mapping for label in ["LABEL_0", "LABEL_1", "LABEL_2"])

        # Contract verification: at least one format is supported
        assert direct_labels_supported or model_labels_supported


class TestConvenienceFunction:
    """Test the convenience function for emotion classification."""

    @patch("emotional_processor.processors.emotion_classifier.EmotionClassifier")
    def test_convenience_function_delegation_contract(self, mock_classifier_class: Mock) -> None:
        """Test contract: convenience function provides same interface as classifier."""
        # Setup behavioral mock with state tracking
        mock_classifier = MagicMock()
        mock_classifier.classify_single.return_value = (0.8, ["joy", "gratitude"])
        mock_classifier_class.return_value = mock_classifier

        # Contract: convenience function should delegate to classifier instance
        score, emotions = classify_emotional_content("I'm so grateful!")

        # Contract verification: same return format as classifier
        assert isinstance(score, float)
        assert isinstance(emotions, list)
        assert score == 0.8
        assert emotions == ["joy", "gratitude"]

        # Verify delegation occurred
        mock_classifier.classify_single.assert_called_once_with("I'm so grateful!")

    def test_convenience_function_model_configuration_contract(self) -> None:
        """Test contract: convenience function supports custom model configuration."""
        # Test by calling with custom model and verifying it creates new classifier
        # Since the function uses a global classifier, we test the behavior indirectly

        # Clear any existing global classifier
        import emotional_processor.processors.emotion_classifier as ec_module

        ec_module._global_classifier = None

        with patch("emotional_processor.processors.emotion_classifier.EmotionClassifier") as mock_classifier_class:
            mock_classifier = MagicMock()
            mock_classifier.classify_single.return_value = (0.5, ["neutral"])
            mock_classifier_class.return_value = mock_classifier

            custom_model = "custom-emotion-model"

            # Contract: custom model should be passed through to classifier
            score, emotions = classify_emotional_content("Test text", model_name=custom_model)

            # Contract verification: classifier created with custom configuration
            mock_classifier_class.assert_called_with(model_name=custom_model)

            # Contract: function still returns expected format
            assert isinstance(score, float)
            assert isinstance(emotions, list)


@pytest.mark.integration
class TestEmotionClassifierIntegration:
    """Integration tests that verify real model behavior contracts."""

    @pytest.mark.slow
    def test_real_model_contract_compliance(self, sample_emotional_texts: dict[str, dict[str, Any]]) -> None:
        """Test contract: real model produces expected emotional response patterns."""
        classifier = EmotionClassifier()

        for test_case, expectations in sample_emotional_texts.items():
            text = expectations["text"]

            # Contract: classifier should handle real text and return valid results
            score, emotions = classifier.classify_single(text)

            # Contract verification: valid output format
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert isinstance(emotions, list)
            assert all(isinstance(emotion, str) for emotion in emotions)

            # Contract: results should meet domain expectations if specified
            # Note: Real model performance may vary, so we test within reasonable ranges
            if "expected_score_min" in expectations:
                # Allow 10% tolerance for real model variability
                min_threshold = max(0.0, expectations["expected_score_min"] - 0.1)
                assert score >= min_threshold, f"Score {score} significantly below expected minimum for {test_case}"

            if "expected_score_max" in expectations:
                # Allow 10% tolerance for real model variability
                max_threshold = min(1.0, expectations["expected_score_max"] + 0.1)
                assert score <= max_threshold, f"Score {score} significantly above expected maximum for {test_case}"

    @pytest.mark.slow
    @patch("emotional_processor.processors.emotion_classifier.pipeline")
    def test_batch_efficiency_contract(self, mock_pipeline: Mock) -> None:
        """Test contract: batch processing is more efficient than individual calls."""
        # Setup mock to track call patterns
        call_count = {"single": 0, "batch": 0}

        def track_single_calls(text):
            call_count["single"] += 1
            return [[{"label": "neutral", "score": 0.5}]]

        def track_batch_calls(texts):
            call_count["batch"] += 1
            return [[{"label": "neutral", "score": 0.5}] for _ in texts]

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        classifier = EmotionClassifier()
        test_texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

        # Test individual processing
        mock_pipeline_instance.side_effect = track_single_calls
        for text in test_texts:
            classifier.classify_single(text)

        single_calls = call_count["single"]

        # Reset and test batch processing
        call_count = {"single": 0, "batch": 0}
        mock_pipeline_instance.side_effect = track_batch_calls
        classifier.classify_batch(test_texts)

        batch_calls = call_count["batch"]

        # Contract: batch should make fewer pipeline calls than individual
        assert batch_calls < single_calls, "Batch processing should be more efficient"
        assert batch_calls == 1, "Batch should make single pipeline call"
        assert single_calls == len(test_texts), "Individual calls should equal text count"
