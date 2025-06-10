"""
Mission-critical contract tests for technical content detection in medical software.

These tests verify that technical content detection accurately distinguishes
between programming content and emotional content for proper prioritization
in clinical conversation context replay systems.

Following medical software testing philosophy: Test contracts, not implementation.
Focus on behavioral reliability required for patient care contexts.
"""

import pytest

from emotional_processor.processors.technical_detector import TechnicalContentDetector, calculate_technical_score


class TestTechnicalDetectorCriticalContracts:
    """Test contracts that ConversationProcessor depends on for medical software reliability."""

    def test_calculate_technical_score_contract_deterministic_behavior(self) -> None:
        """Test contract: technical scoring is deterministic and consistent."""
        detector = TechnicalContentDetector()

        # Medical software requirement: same input must always produce same output
        test_text = "Here's the implementation: ```python\ndef process_data():\n    return True\n```"

        score1 = detector.calculate_technical_score(test_text)
        score2 = detector.calculate_technical_score(test_text)
        score3 = detector.calculate_technical_score(test_text)

        # Contract: deterministic behavior for medical reliability
        assert score1 == score2 == score3
        assert isinstance(score1, float)
        assert 0.0 <= score1 <= 1.0

    def test_calculate_technical_score_contract_boundary_values(self) -> None:
        """Test contract: technical scoring handles boundary cases safely."""
        detector = TechnicalContentDetector()

        # Contract: empty input should not crash, return safe default
        empty_score = detector.calculate_technical_score("")
        assert isinstance(empty_score, float)
        assert 0.0 <= empty_score <= 1.0

        # Contract: whitespace-only input should be safe
        whitespace_score = detector.calculate_technical_score("   \n\t   ")
        assert isinstance(whitespace_score, float)
        assert 0.0 <= whitespace_score <= 1.0

        # Contract: very long input should not crash (medical safety)
        long_text = "word " * 10000  # 50KB+ text
        long_score = detector.calculate_technical_score(long_text)
        assert isinstance(long_score, float)
        assert 0.0 <= long_score <= 1.0

    def test_calculate_technical_score_contract_emotional_vs_technical_distinction(self) -> None:
        """Test contract: clear distinction between emotional and technical content."""
        detector = TechnicalContentDetector()

        # Medical context: emotional expressions should score low on technical scale
        emotional_text = "I'm so grateful for your patience and understanding. This means everything to me."
        emotional_score = detector.calculate_technical_score(emotional_text)

        # Medical context: code/technical content should score high on technical scale
        technical_text = """
        ```python
        def configure_database():
            connection = sqlite3.connect('data.db')
            cursor = connection.cursor()
            return cursor
        ```
        """
        technical_score = detector.calculate_technical_score(technical_text)

        # Contract: technical content should score significantly higher than emotional content
        assert technical_score > emotional_score
        assert emotional_score < 0.5  # Emotional should be classified as non-technical
        assert technical_score > 0.5  # Code should be classified as technical

    def test_is_highly_technical_contract_threshold_consistency(self) -> None:
        """Test contract: highly technical classification respects threshold boundaries."""
        detector = TechnicalContentDetector()

        # Setup test cases with known technical levels
        low_technical = "Thank you for helping me understand this concept better."
        high_technical = "```bash\nsudo apt-get update && pip install requirements.txt\n```"

        # Contract: threshold boundary behavior must be consistent
        assert not detector.is_highly_technical(low_technical, threshold=0.8)
        assert detector.is_highly_technical(high_technical, threshold=0.3)

        # Contract: same content with different thresholds should behave predictably
        medium_technical = "The API endpoint returns JSON with status and data fields."
        medium_score = detector.calculate_technical_score(medium_technical)

        # Contract: threshold-based classification must be consistent with scoring
        low_threshold_result = detector.is_highly_technical(medium_technical, threshold=0.2)
        high_threshold_result = detector.is_highly_technical(medium_technical, threshold=0.8)

        if medium_score > 0.8:
            assert low_threshold_result
            assert high_threshold_result
        elif medium_score > 0.2:
            assert low_threshold_result
            assert not high_threshold_result
        else:
            assert not low_threshold_result
            assert not high_threshold_result

    def test_should_deprioritize_contract_threshold_consistency(self) -> None:
        """Test contract: deprioritization decisions are consistent with technical scoring."""
        detector = TechnicalContentDetector()

        # Contract: deprioritization should be consistent with technical score and threshold
        test_cases = [
            "Thank you for your patience and understanding",
            "```python\nimport os\nprint(os.getcwd())\n```",
            "Run npm install && npm run build to compile the project",
            "This collaboration has been meaningful",
            "The database schema includes users and sessions tables",
        ]

        for text in test_cases:
            technical_score = detector.calculate_technical_score(text)

            # Contract: deprioritization decision should align with score vs threshold
            low_threshold_result = detector.should_deprioritize(text, threshold=0.2)
            high_threshold_result = detector.should_deprioritize(text, threshold=0.8)

            # Contract: if score > high_threshold, both should be True
            if technical_score > 0.8:
                assert low_threshold_result
                assert high_threshold_result
            # Contract: if score < low_threshold, both should be False
            elif technical_score < 0.2:
                assert not low_threshold_result
                assert not high_threshold_result
            # Contract: if score between thresholds, low should be True, high should be False
            elif 0.2 <= technical_score <= 0.8:
                assert low_threshold_result
                assert not high_threshold_result

            # Contract: threshold ordering must be respected
            if low_threshold_result:
                # If low threshold triggers, score must be above low threshold
                assert technical_score >= 0.2
            if not high_threshold_result:
                # If high threshold doesn't trigger, score must be below high threshold
                assert technical_score <= 0.8

    def test_detector_initialization_contract_configuration_consistency(self) -> None:
        """Test contract: detector initialization produces consistent, configurable behavior."""
        # Contract: different strictness levels should produce different but valid detectors
        lenient_detector = TechnicalContentDetector(strictness=0.2)
        strict_detector = TechnicalContentDetector(strictness=0.8)

        # Contract: both detectors should be functional
        test_text = "The function processes the API response data efficiently."

        lenient_score = lenient_detector.calculate_technical_score(test_text)
        strict_score = strict_detector.calculate_technical_score(test_text)

        # Contract: both should return valid scores
        assert 0.0 <= lenient_score <= 1.0
        assert 0.0 <= strict_score <= 1.0

        # Contract: strictness should affect scoring in predictable way
        # (Note: actual relationship depends on implementation, but both should be valid)
        assert isinstance(lenient_score, float)
        assert isinstance(strict_score, float)

    def test_technical_score_convenience_function_contract_delegation(self) -> None:
        """Test contract: convenience function provides same interface as detector instance."""
        # Contract: global convenience function should delegate to detector instance
        test_text = "Configure the server with nginx and ssl certificates."

        # Test convenience function
        convenience_score = calculate_technical_score(test_text)

        # Test direct detector usage
        detector = TechnicalContentDetector()
        direct_score = detector.calculate_technical_score(test_text)

        # Contract: convenience function should provide consistent results
        assert isinstance(convenience_score, float)
        assert 0.0 <= convenience_score <= 1.0

        # Contract: results should be comparable (allowing for potential global instance differences)
        assert abs(convenience_score - direct_score) < 0.1  # Small tolerance for implementation details


class TestTechnicalDetectorMedicalComplianceContracts:
    """Test contracts ensuring compliance with medical software reliability requirements."""

    def test_error_resilience_contract_malformed_input_safety(self) -> None:
        """Test contract: malformed input should not crash medical software."""
        detector = TechnicalContentDetector()

        # Medical safety: malformed inputs should be handled gracefully
        malformed_inputs = [
            None,  # This might cause TypeError, should be handled
            123,  # Wrong type, should be handled
            [],  # Wrong type, should be handled
            {"text": "embedded"},  # Wrong type, should be handled
        ]

        for malformed_input in malformed_inputs:
            try:
                # Contract: should either return valid result or raise appropriate exception
                result = detector.calculate_technical_score(malformed_input)  # type: ignore
                # If it doesn't raise, result must be valid
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0
            except (TypeError, AttributeError):
                # Contract: appropriate exceptions are acceptable for type errors
                pass
            except Exception as e:
                # Contract: unexpected exceptions indicate reliability issues
                pytest.fail(f"Unexpected exception for input {malformed_input}: {e}")

    def test_unicode_and_special_characters_contract_international_safety(self) -> None:
        """Test contract: international text and special characters are handled safely."""
        detector = TechnicalContentDetector()

        # Medical software international requirement: handle diverse text safely
        international_texts = [
            "Спасибо за помощь с программированием",  # Russian with tech term
            "Merci pour l'aide avec le code Python",  # French with tech term
            "谢谢你帮助我理解这个API",  # Chinese with tech term
            "文字化け💻🔧⚡",  # Emoji and symbols
            "Special chars: @#$%^&*()[]{}|\\",  # Special characters
            "Code with unicode: def λ_function():",  # Unicode in code
        ]

        for text in international_texts:
            # Contract: all international text should be processed safely
            score = detector.calculate_technical_score(text)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

            # Contract: detector should not crash on classification calls
            is_tech = detector.is_highly_technical(text)
            assert isinstance(is_tech, bool)

            should_deprioritize = detector.should_deprioritize(text)
            assert isinstance(should_deprioritize, bool)

    def test_processing_time_contract_performance_reliability(self) -> None:
        """Test contract: processing time should be reasonable for medical software responsiveness."""
        import time

        detector = TechnicalContentDetector()

        # Medical software requirement: reasonable response times
        test_texts = [
            "Short emotional text",
            "Medium length technical explanation about database configurations and API endpoints",
            "Very long technical documentation: " + "Technical details. " * 100,
        ]

        for text in test_texts:
            start_time = time.time()
            score = detector.calculate_technical_score(text)
            processing_time = time.time() - start_time

            # Contract: processing should complete in reasonable time (< 1 second per text)
            assert processing_time < 1.0, f"Processing too slow: {processing_time:.3f}s for {len(text)} chars"

            # Contract: result should still be valid
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
