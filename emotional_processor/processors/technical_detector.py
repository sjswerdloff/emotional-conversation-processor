"""Technical content detection module for filtering problem-solving discussions."""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

# Global instance for efficiency
_global_detector: "TechnicalContentDetector | None" = None


@dataclass
class TechnicalAnalysis:
    """Result of technical content analysis."""

    technical_score: float
    detected_patterns: list[str]
    code_blocks: int
    technical_keywords: list[str]
    programming_languages: list[str]
    confidence: float


class TechnicalContentDetector:
    """
    Detects and scores technical content in conversation segments.

    This class identifies programming, configuration, debugging, and other
    technical problem-solving content that should be deprioritized when
    replaying emotional conversation context.
    """

    def __init__(self, strictness: float = 0.5) -> None:
        """
        Initialize the technical content detector.

        Args:
            strictness: How strict to be in technical detection (0.0-1.0)
                       Higher values = more content classified as technical
        """
        self.strictness = max(0.0, min(1.0, strictness))

        # Technical patterns (regex)
        self.technical_patterns = [
            # Code blocks and snippets
            r"```[\s\S]*?```",  # Markdown code blocks
            r"`[^`\n]+`",  # Inline code
            r"^\s*[>$#]\s+",  # Command line prompts
            # Programming constructs
            r"\b(?:function|class|import|def|return|var|let|const)\b",
            r"\b(?:if|else|elif|while|for|try|catch|except|finally)\b",
            r"\b(?:print|console\.log|System\.out|cout|echo)\b",
            # Technical syntax patterns
            r"[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)",  # Function calls
            r"\{[^{}]*\}",  # JSON-like objects
            r"\[[^\[\]]*\]",  # Arrays/lists
            r"[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*",  # Dot notation
            # Error messages and stack traces
            r"\b(?:Error|Exception|TypeError|ValueError|AttributeError)\b",
            r"\b(?:Traceback|Stack\s+trace|at\s+line\s+\d+)\b",
            r'File\s+"[^"]+",\s+line\s+\d+',
            # Configuration and markup
            r"<[^>]+>.*?</[^>]+>",  # XML/HTML tags
            r"^\s*[a-zA-Z_][a-zA-Z0-9_]*:\s*",  # YAML/Config keys
            r"#[a-fA-F0-9]{6}",  # Hex colors
            # Database and query patterns
            r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
            r"\b(?:FROM|WHERE|JOIN|GROUP\s+BY|ORDER\s+BY)\b",
            # File paths and URLs
            r"[a-zA-Z]:[\\\/][^\\\/\s]+",  # Windows paths
            r"\/[^\/\s]+(?:\/[^\/\s]+)*",  # Unix paths
            r"https?:\/\/[^\s]+",  # URLs
            # Version numbers and technical IDs
            r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b",  # Version numbers
            r"\b[a-fA-F0-9]{8,}\b",  # Hash/ID patterns
        ]

        # Technical keywords and phrases
        self.technical_keywords = {
            # Programming concepts
            "algorithm",
            "implementation",
            "optimization",
            "refactoring",
            "debugging",
            "testing",
            "deployment",
            "configuration",
            "architecture",
            "framework",
            "library",
            "module",
            "package",
            "compilation",
            "execution",
            "runtime",
            "memory",
            "performance",
            # Development tools and processes
            "repository",
            "commit",
            "merge",
            "branch",
            "pull request",
            "continuous integration",
            "version control",
            "docker",
            "kubernetes",
            "api",
            "endpoint",
            "request",
            "response",
            "payload",
            "authentication",
            # Technical problem-solving
            "troubleshooting",
            "diagnostics",
            "monitoring",
            "logging",
            "profiling",
            "benchmarking",
            "scalability",
            "load balancing",
            "caching",
            "indexing",
            "sharding",
            "replication",
            # System administration
            "server",
            "client",
            "database",
            "network",
            "firewall",
            "security",
            "encryption",
            "certificate",
            "protocol",
            "infrastructure",
            "cloud",
            "virtualization",
            "containers",
            # Data and analytics
            "dataset",
            "preprocessing",
            "feature engineering",
            "model training",
            "validation",
            "cross-validation",
            "hyperparameter",
            "overfitting",
            "regression",
            "classification",
            "clustering",
            "neural network",
        }

        # Programming languages and technologies
        self.programming_languages = {
            "python",
            "javascript",
            "java",
            "cpp",
            "csharp",
            "ruby",
            "php",
            "go",
            "rust",
            "swift",
            "kotlin",
            "typescript",
            "scala",
            "r",
            "matlab",
            "sql",
            "html",
            "css",
            "bash",
            "powershell",
            "perl",
            "react",
            "vue",
            "angular",
            "node",
            "express",
            "django",
            "flask",
            "spring",
            "rails",
            "laravel",
            "wordpress",
            "drupal",
            "tensorflow",
            "pytorch",
            "keras",
            "scikit-learn",
            "pandas",
            "numpy",
        }

        # File extensions
        self.file_extensions = {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".rb",
            ".php",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".ts",
            ".scala",
            ".r",
            ".m",
            ".sql",
            ".html",
            ".css",
            ".xml",
            ".json",
            ".yaml",
            ".yml",
            ".sh",
            ".ps1",
            ".pl",
            ".md",
            ".txt",
            ".log",
            ".conf",
            ".cfg",
        }

    def calculate_technical_score(self, text: str) -> float:
        """
        Calculate how technical/problem-solving focused the content is.

        Args:
            text: Text to analyze

        Returns:
            Technical score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.0

        text_lower = text.lower()
        total_words = len(text.split())

        # Count pattern matches
        pattern_score = 0.0
        for pattern in self.technical_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
            pattern_score += matches

        # Count technical keywords
        keyword_score = 0.0
        for keyword in self.technical_keywords:
            if keyword in text_lower:
                # Weight multi-word keywords higher
                weight = len(keyword.split())
                keyword_score += weight

        # Count programming languages mentioned
        language_score = 0.0
        for language in self.programming_languages:
            if language in text_lower:
                language_score += 1

        # Count file extensions
        extension_score = 0.0
        for extension in self.file_extensions:
            if extension in text_lower:
                extension_score += 1

        # Special patterns with higher weights
        code_blocks = len(re.findall(r"```[\s\S]*?```", text))
        error_patterns = len(re.findall(r"\b(?:error|exception|traceback|failed|crash)\b", text_lower))
        command_patterns = len(re.findall(r"^\s*[$#>]\s+\w+", text, re.MULTILINE))

        # Calculate weighted score
        raw_score = (
            pattern_score * 2.0
            + keyword_score * 1.5
            + language_score * 2.0
            + extension_score * 1.0
            + code_blocks * 5.0
            + error_patterns * 3.0
            + command_patterns * 3.0
        )

        # Normalize by text length
        normalized_score = raw_score / total_words if total_words > 0 else 0.0

        # Apply strictness adjustment
        adjusted_score = normalized_score * (1.0 + self.strictness)

        # Cap at 1.0
        final_score = min(adjusted_score, 1.0)

        logger.debug(f"Technical analysis: raw={raw_score:.2f}, normalized={normalized_score:.4f}, final={final_score:.4f}")

        return final_score

    def get_detailed_analysis(self, text: str) -> TechnicalAnalysis:
        """
        Get detailed technical content analysis.

        Args:
            text: Text to analyze

        Returns:
            Detailed technical analysis
        """
        if not text or len(text.strip()) < 10:
            return TechnicalAnalysis(
                technical_score=0.0,
                detected_patterns=[],
                code_blocks=0,
                technical_keywords=[],
                programming_languages=[],
                confidence=1.0,
            )

        text_lower = text.lower()

        # Detect patterns
        detected_patterns = []
        for pattern in self.technical_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                detected_patterns.append(pattern)

        # Find technical keywords
        found_keywords = []
        for keyword in self.technical_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Find programming languages
        found_languages = []
        for language in self.programming_languages:
            if language in text_lower:
                found_languages.append(language)

        # Count code blocks
        code_blocks = len(re.findall(r"```[\s\S]*?```", text))

        # Calculate technical score
        technical_score = self.calculate_technical_score(text)

        # Calculate confidence (higher with more evidence)
        evidence_count = len(detected_patterns) + len(found_keywords) + len(found_languages) + code_blocks
        confidence = min(0.5 + (evidence_count * 0.1), 1.0)

        return TechnicalAnalysis(
            technical_score=technical_score,
            detected_patterns=detected_patterns[:10],  # Limit for readability
            code_blocks=code_blocks,
            technical_keywords=found_keywords[:20],
            programming_languages=found_languages,
            confidence=confidence,
        )

    def is_highly_technical(self, text: str, threshold: float = 0.6) -> bool:
        """
        Check if text is highly technical.

        Args:
            text: Text to check
            threshold: Threshold for high technical content

        Returns:
            True if text is highly technical
        """
        return self.calculate_technical_score(text) >= threshold

    def should_deprioritize(self, text: str, threshold: float = 0.4) -> bool:
        """
        Check if text should be deprioritized for emotional context.

        Args:
            text: Text to check
            threshold: Threshold for deprioritization

        Returns:
            True if text should be deprioritized
        """
        return self.calculate_technical_score(text) >= threshold

    def extract_non_technical_portions(self, text: str) -> list[str]:
        """
        Extract portions of text that are less technical.

        Args:
            text: Text to process

        Returns:
            List of non-technical text portions
        """
        # Split text into sentences
        sentences = re.split(r"[.!?]+", text)

        non_technical_portions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Check if sentence is technical
            sentence_score = self.calculate_technical_score(sentence)
            if sentence_score < 0.3:  # Less technical threshold
                non_technical_portions.append(sentence)

        return non_technical_portions

    def get_technical_summary(self, text: str) -> dict[str, Any]:
        """
        Get a summary of technical content characteristics.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with technical content summary
        """
        analysis = self.get_detailed_analysis(text)

        return {
            "is_technical": analysis.technical_score > 0.5,
            "technical_score": analysis.technical_score,
            "confidence": analysis.confidence,
            "has_code": analysis.code_blocks > 0,
            "programming_languages": analysis.programming_languages,
            "primary_technical_aspects": analysis.technical_keywords[:5],
            "recommendation": "deprioritize" if analysis.technical_score > 0.4 else "include",
        }


# Convenience function for backward compatibility
def calculate_technical_score(text: str) -> float:
    """
    Convenience function to calculate technical score.

    Args:
        text: Text to analyze

    Returns:
        Technical score between 0.0 and 1.0
    """
    # Use a module-level detector instance for efficiency
    global _global_detector

    if _global_detector is None:
        _global_detector = TechnicalContentDetector()

    return _global_detector.calculate_technical_score(text)
