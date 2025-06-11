"""Emotion classification module for conversation analysis."""

import re
from dataclasses import dataclass

import torch
from loguru import logger
from transformers import pipeline  # type: ignore[attr-defined]
from transformers.pipelines import Pipeline  # type: ignore[attr-defined]

# Global instance for efficiency
_global_classifier: "EmotionClassifier | None" = None


@dataclass
class EmotionResult:
    """Result of emotion classification."""

    primary_emotion: str
    confidence_score: float
    all_emotions: dict[str, float]
    text_length: int
    processing_time: float


class EmotionClassifier:
    """
    Handles emotion classification using transformer models.

    This class provides robust emotion detection with support for
    multiple models and fallback mechanisms.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        confidence_threshold: float = 0.1,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        """
        Initialize the emotion classifier.

        Args:
            model_name: HuggingFace model name for emotion classification
            confidence_threshold: Minimum confidence for emotion detection
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for processing multiple texts
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        # Determine device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1

        self._pipeline: Pipeline | None = None
        self._emotion_mapping = self._create_emotion_mapping()

    def _create_emotion_mapping(self) -> dict[str, str]:
        """Create mapping from model outputs to standardized emotion labels."""
        return {
            # Common mappings for different models
            "LABEL_0": "sadness",
            "LABEL_1": "joy",
            "LABEL_2": "love",
            "LABEL_3": "anger",
            "LABEL_4": "fear",
            "LABEL_5": "surprise",
            # Direct mappings
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "love": "love",
            "surprise": "surprise",
            "disgust": "disgust",
            "gratitude": "gratitude",
            "excitement": "excitement",
            "empathy": "empathy",
            "relief": "relief",
            "frustration": "frustration",
            "anxiety": "anxiety",
            "pride": "pride",
            "shame": "shame",
            "curiosity": "curiosity",
        }

    def _initialize_pipeline(self) -> None:
        """Initialize the emotion classification pipeline."""
        try:
            logger.info(f"Loading emotion classification model: {self.model_name}")
            self._pipeline = pipeline("text-classification", model=self.model_name, return_all_scores=True, device=self.device)
            logger.info("Emotion classification model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion model {self.model_name}: {e}")
            raise RuntimeError("Could not initialize emotion classifier") from e

    @property
    def pipeline(self) -> Pipeline:
        """Get the classification pipeline, initializing if necessary."""
        if self._pipeline is None:
            self._initialize_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Pipeline initialization failed")
        return self._pipeline

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for emotion classification.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove code blocks (they rarely contain emotional content)
        text = re.sub(r"```[\s\S]*?```", "[CODE_BLOCK]", text)

        # Remove URLs
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "[URL]", text)

        # Truncate if too long (most models have token limits)
        max_chars = 500  # Approximate limit for most emotion models
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    def classify_single(self, text: str) -> tuple[float, list[str]]:
        """
        Classify emotions in a single text.

        Args:
            text: Text to classify

        Returns:
            Tuple of (max_confidence_score, list_of_detected_emotions)
        """
        import time

        start_time = time.time()

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            if not processed_text or len(processed_text.strip()) < 3:
                return 0.0, []

            # Get emotion predictions
            results = self.pipeline(processed_text)

            # Extract emotions above threshold
            detected_emotions = []
            max_score = 0.0

            for result in results[0]:  # results is a list with one element for single input
                emotion_label = result["label"].lower()
                confidence = result["score"]

                # Map to standardized emotion if needed
                standardized_emotion = self._emotion_mapping.get(emotion_label, emotion_label)

                if confidence >= self.confidence_threshold:
                    detected_emotions.append(standardized_emotion)
                    max_score = max(max_score, confidence)

            processing_time = time.time() - start_time
            logger.debug(f"Classified emotions in {processing_time:.3f}s: {detected_emotions}")

            return max_score, detected_emotions

        except Exception as e:
            logger.exception("Emotion classification failed")
            raise RuntimeError("Could not classify emotions") from e

    def classify_batch(self, texts: list[str]) -> list[tuple[float, list[str]]]:
        """
        Classify emotions in multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of (max_confidence_score, list_of_detected_emotions) tuples
        """
        import time

        start_time = time.time()

        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Filter out empty texts but keep track of original indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(processed_texts):
                if text and len(text.strip()) >= 3:
                    valid_texts.append(text)
                    valid_indices.append(i)

            if not valid_texts:
                return [(0.0, []) for _ in texts]

            # Process in batches
            results = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i : i + self.batch_size]
                batch_results = self.pipeline(batch)
                results.extend(batch_results)

            # Process results and map back to original indices
            final_results: list[tuple[float, list[str]]] = [(0.0, []) for _ in texts]

            for result_idx, original_idx in enumerate(valid_indices):
                detected_emotions = []
                max_score = 0.0

                for emotion_result in results[result_idx]:
                    emotion_label = emotion_result["label"].lower()
                    confidence = emotion_result["score"]

                    # Map to standardized emotion if needed
                    standardized_emotion = self._emotion_mapping.get(emotion_label, emotion_label)

                    if confidence >= self.confidence_threshold:
                        detected_emotions.append(standardized_emotion)
                        max_score = max(max_score, confidence)

                final_results[original_idx] = (max_score, detected_emotions)

            processing_time = time.time() - start_time
            logger.info(f"Batch classified {len(texts)} texts in {processing_time:.3f}s")

            return final_results

        except Exception as e:
            logger.exception("Batch emotion classification failed")
            raise RuntimeError("Could not classify batch emotions") from e

    def get_detailed_analysis(self, text: str) -> EmotionResult:
        """
        Get detailed emotion analysis for a single text.

        Args:
            text: Text to analyze

        Returns:
            Detailed emotion analysis result
        """
        import time

        start_time = time.time()

        try:
            processed_text = self._preprocess_text(text)

            if not processed_text or len(processed_text.strip()) < 3:
                return EmotionResult(
                    primary_emotion="neutral",
                    confidence_score=0.0,
                    all_emotions={},
                    text_length=len(text),
                    processing_time=time.time() - start_time,
                )

            # Get all emotion scores
            results = self.pipeline(processed_text)

            # Process all emotions
            all_emotions = {}
            max_score = 0.0
            primary_emotion = "neutral"

            for result in results[0]:
                emotion_label = result["label"].lower()
                confidence = result["score"]

                # Map to standardized emotion
                standardized_emotion = self._emotion_mapping.get(emotion_label, emotion_label)
                all_emotions[standardized_emotion] = confidence

                if confidence > max_score:
                    max_score = confidence
                    primary_emotion = standardized_emotion

            processing_time = time.time() - start_time

            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence_score=max_score,
                all_emotions=all_emotions,
                text_length=len(text),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.warning(f"Detailed emotion analysis failed: {e}")
            return EmotionResult(
                primary_emotion="neutral",
                confidence_score=0.0,
                all_emotions={},
                text_length=len(text),
                processing_time=time.time() - start_time,
            )

    def is_highly_emotional(self, text: str, threshold: float = 0.7) -> bool:
        """
        Check if text contains highly emotional content.

        Args:
            text: Text to check
            threshold: Confidence threshold for high emotion

        Returns:
            True if text is highly emotional
        """
        max_score, emotions = self.classify_single(text)
        return max_score >= threshold and len(emotions) > 0

    def get_primary_emotion(self, text: str) -> str:
        """
        Get the primary emotion from text.

        Args:
            text: Text to analyze

        Returns:
            Primary emotion label or "neutral" if none detected
        """
        analysis = self.get_detailed_analysis(text)
        return analysis.primary_emotion


def classify_emotional_content(text: str, model_name: str | None = None) -> tuple[float, list[str]]:
    """
    Convenience function for emotion classification.

    Args:
        text: Text to classify
        model_name: Optional model name to use

    Returns:
        Tuple of (max_confidence_score, list_of_detected_emotions)
    """
    # Use a module-level classifier instance for efficiency
    global _global_classifier

    if _global_classifier is None:
        _global_classifier = EmotionClassifier(model_name=model_name or "j-hartmann/emotion-english-distilroberta-base")

    return _global_classifier.classify_single(text)
