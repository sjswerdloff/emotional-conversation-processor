#!/usr/bin/env python3
"""
Main script for processing conversation files and building emotional context vectors.

This script handles the complete pipeline from raw conversation files to
searchable vector embeddings with emotional context preservation.
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotional_processor.core.models import ConversationSegment, ProcessingStats  # noqa: E402
from emotional_processor.embeddings.emotion_aware_embedder import EmotionAwareEmbedder  # noqa: E402
from emotional_processor.processors.emotion_classifier import EmotionClassifier  # noqa: E402
from emotional_processor.processors.technical_detector import TechnicalContentDetector  # noqa: E402
from emotional_processor.storage.vector_store import ConversationVectorStore  # noqa: E402


class ConversationProcessor:
    """
    Main processor for converting conversations to emotional context vectors.
    """

    def __init__(
        self,
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ) -> None:
        """
        Initialize the conversation processor.

        Args:
            emotion_model: Model for emotion classification
            embedding_model: Model for embeddings
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.emotion_classifier = EmotionClassifier(model_name=emotion_model)
        self.technical_detector = TechnicalContentDetector()
        self.embedder = EmotionAwareEmbedder(model_name=embedding_model)
        self.vector_store = ConversationVectorStore(host=qdrant_host, port=qdrant_port)

        self.processing_stats = ProcessingStats()

    def segment_conversation(self, content: str) -> list[tuple[str, str, str | None]]:
        """
        Segment conversation content into individual turns.

        Args:
            content: Raw conversation content

        Returns:
            List of (text, speaker, timestamp) tuples
        """
        segments = []

        # Pattern for common conversation formats
        patterns = [
            # "User [timestamp]: message" or "Assistant [timestamp]: message"
            r"^(User|Assistant|Human|AI|System)\s*\[(.*?)\]:\s*(.*?)(?=^(?:User|Assistant|Human|AI|System)\s*\[|$)",
            # "User: message" or "Assistant: message"
            r"^(User|Assistant|Human|AI|System):\s*(.*?)(?=^(?:User|Assistant|Human|AI|System):|$)",
            # Markdown style "**User:**" or "**Assistant:**"
            r"^\*\*(User|Assistant|Human|AI|System)\*\*:?\s*(.*?)(?=^\*\*(?:User|Assistant|Human|AI|System)\*\*|$)",
            # Simple alternating format (assume User starts)
        ]

        # Try each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            pattern_segments = []

            for match in matches:
                if len(match.groups()) == 3:
                    speaker, timestamp, text = match.groups()
                    text = text.strip()
                    timestamp = timestamp.strip() if timestamp else None
                else:
                    speaker, text = match.groups()
                    text = text.strip()
                    timestamp = None

                if text:
                    pattern_segments.append((text, speaker.title(), timestamp))

            # Use the pattern that found the most segments
            if len(pattern_segments) > len(segments):
                segments = pattern_segments

        # Fallback: split by paragraphs and alternate speakers
        if not segments:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                speaker = "User" if i % 2 == 0 else "Assistant"
                segments.append((paragraph, speaker, None))

        logger.info(f"Segmented conversation into {len(segments)} turns")
        return segments

    def calculate_importance_weight(self, emotional_score: float, technical_score: float, word_count: int = 0) -> float:
        """
        Calculate importance weight for a segment.

        Args:
            emotional_score: Emotional content score
            technical_score: Technical content score
            word_count: Number of words in segment

        Returns:
            Importance weight (0.0-1.0)
        """
        # Base importance from emotional content
        emotional_weight = emotional_score * 0.6

        # Penalty for technical content
        technical_penalty = technical_score * 0.3

        # Small bonus for reasonable length
        length_bonus = min(word_count / 100, 0.2) if word_count > 0 else 0

        # Calculate final weight
        importance = emotional_weight - technical_penalty + length_bonus

        # Ensure it's between 0 and 1
        return max(0.0, min(1.0, importance))

    def process_segment(self, text: str, speaker: str, timestamp: str | None) -> ConversationSegment:
        """
        Process a single conversation segment.

        Args:
            text: Segment text
            speaker: Speaker name
            timestamp: Optional timestamp

        Returns:
            Processed conversation segment
        """
        try:
            # Classify emotions
            emotional_score, emotional_labels = self.emotion_classifier.classify_single(text)

            # Detect technical content
            technical_score = self.technical_detector.calculate_technical_score(text)

            # Calculate importance weight
            word_count = len(text.split())
            importance_weight = self.calculate_importance_weight(emotional_score, technical_score, word_count)

            # Create segment
            segment = ConversationSegment(
                content=text,
                speaker=speaker,
                timestamp=timestamp,
                emotional_score=emotional_score,
                emotional_labels=emotional_labels,
                technical_score=technical_score,
                importance_weight=importance_weight,
            )

            # Update stats
            self.processing_stats.total_segments += 1
            if emotional_score > 0.6:
                self.processing_stats.emotional_segments += 1
            if technical_score > 0.6:
                self.processing_stats.technical_segments += 1

            return segment

        except Exception as e:
            logger.error(f"Failed to process segment: {e}")
            self.processing_stats.errors += 1

            # Return minimal segment
            return ConversationSegment(content=text, speaker=speaker, timestamp=timestamp)

    def process_conversation_file(
        self,
        file_path: str,
        conversation_id: str | None = None,
        batch_size: int = 50,  # noqa: ARG002
    ) -> list[ConversationSegment]:
        """
        Process a complete conversation file.

        Args:
            file_path: Path to conversation file
            conversation_id: Optional conversation ID
            batch_size: Batch size for processing

        Returns:
            List of processed conversation segments
        """
        start_time = time.time()

        try:
            # Read file
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            logger.info(f"Processing conversation file: {file_path}")

            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = f"conv_{int(time.time())}_{Path(file_path).stem}"

            # Segment conversation
            raw_segments = self.segment_conversation(content)

            # Process segments
            processed_segments = []
            for i, (text, speaker, timestamp) in enumerate(raw_segments):
                logger.debug(f"Processing segment {i + 1}/{len(raw_segments)}")

                segment = self.process_segment(text, speaker, timestamp)
                segment.conversation_id = conversation_id
                processed_segments.append(segment)

                # Log progress for large conversations
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(raw_segments)} segments")

            # Update processing time
            self.processing_stats.processing_time = time.time() - start_time

            logger.info(f"Processed {len(processed_segments)} segments in {self.processing_stats.processing_time:.2f} seconds")

            return processed_segments

        except Exception as e:
            logger.error(f"Failed to process conversation file {file_path}: {e}")
            self.processing_stats.errors += 1
            return []

    def create_and_store_embeddings(self, segments: list[ConversationSegment], batch_size: int = 32) -> bool:
        """
        Create embeddings and store in vector database.

        Args:
            segments: List of conversation segments
            batch_size: Batch size for embedding creation

        Returns:
            True if successful
        """
        if not segments:
            return False

        try:
            embedding_start = time.time()

            # Create embeddings in batches
            all_embeddings = []
            for i in range(0, len(segments), batch_size):
                batch = segments[i : i + batch_size]
                logger.info(f"Creating embeddings for batch {i // batch_size + 1}")

                batch_embeddings = self.embedder.create_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)

            self.processing_stats.embedding_time = time.time() - embedding_start

            # Store in vector database
            time.time()
            point_ids = self.vector_store.store_batch_segments(segments, all_embeddings)

            logger.info(f"Stored {len(point_ids)} segments in vector database")

            return True

        except Exception as e:
            logger.error(f"Failed to create and store embeddings: {e}")
            self.processing_stats.errors += 1
            return False

    def process_and_store(self, file_path: str, conversation_id: str | None = None) -> bool:
        """
        Complete processing pipeline: file -> segments -> embeddings -> storage.

        Args:
            file_path: Path to conversation file
            conversation_id: Optional conversation ID

        Returns:
            True if successful
        """
        try:
            # Process conversation file
            segments = self.process_conversation_file(file_path, conversation_id)

            if not segments:
                logger.error("No segments produced from conversation file")
                return False

            # Create and store embeddings
            success = self.create_and_store_embeddings(segments)

            if success:
                logger.info(f"Successfully processed and stored conversation: {file_path}")
                self._log_processing_stats()
                return True
            else:
                logger.error("Failed to create and store embeddings")
                return False

        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            return False

    def _log_processing_stats(self) -> None:
        """Log processing statistics."""
        stats = self.processing_stats

        logger.info("=== Processing Statistics ===")
        logger.info(f"Total segments: {stats.total_segments}")
        logger.info(f"Emotional segments: {stats.emotional_segments}")
        logger.info(f"Technical segments: {stats.technical_segments}")
        logger.info(f"Processing time: {stats.processing_time:.2f}s")
        logger.info(f"Embedding time: {stats.embedding_time:.2f}s")
        logger.info(f"Processing rate: {stats.segments_per_second:.1f} segments/sec")
        logger.info(f"Errors: {stats.errors}")

        if stats.total_segments > 0:
            emotional_pct = (stats.emotional_segments / stats.total_segments) * 100
            technical_pct = (stats.technical_segments / stats.total_segments) * 100
            logger.info(f"Emotional content: {emotional_pct:.1f}%")
            logger.info(f"Technical content: {technical_pct:.1f}%")


def main() -> None:
    """Main entry point for the conversation processor."""
    parser = argparse.ArgumentParser(description="Process conversation files for emotional context vector storage")

    parser.add_argument("input", help="Input conversation file or directory")

    parser.add_argument("--conversation-id", help="Conversation ID (auto-generated if not provided)")

    parser.add_argument(
        "--emotion-model", default="j-hartmann/emotion-english-distilroberta-base", help="Emotion classification model"
    )

    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")

    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")

    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument("--output-stats", help="Output file for processing statistics (JSON)")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Initialize processor
    try:
        processor = ConversationProcessor(
            emotion_model=args.emotion_model,
            embedding_model=args.embedding_model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
        )
        logger.info("Initialized conversation processor")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        success = processor.process_and_store(str(input_path), args.conversation_id)

        if not success:
            logger.error("Processing failed")
            sys.exit(1)

    elif input_path.is_dir():
        # Process directory
        conversation_files = []
        for ext in ["*.txt", "*.md", "*.json"]:
            conversation_files.extend(input_path.glob(ext))

        if not conversation_files:
            logger.error(f"No conversation files found in {input_path}")
            sys.exit(1)

        logger.info(f"Found {len(conversation_files)} conversation files")

        success_count = 0
        for file_path in conversation_files:
            logger.info(f"Processing: {file_path}")

            # Generate conversation ID from filename
            conv_id = f"conv_{file_path.stem}_{int(time.time())}"

            if processor.process_and_store(str(file_path), conv_id):
                success_count += 1
            else:
                logger.warning(f"Failed to process: {file_path}")

        logger.info(f"Successfully processed {success_count}/{len(conversation_files)} files")

    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Output statistics if requested
    if args.output_stats:
        import json

        stats_dict = {
            "total_segments": processor.processing_stats.total_segments,
            "emotional_segments": processor.processing_stats.emotional_segments,
            "technical_segments": processor.processing_stats.technical_segments,
            "processing_time": processor.processing_stats.processing_time,
            "embedding_time": processor.processing_stats.embedding_time,
            "segments_per_second": processor.processing_stats.segments_per_second,
            "errors": processor.processing_stats.errors,
            "timestamp": datetime.now().isoformat(),
        }

        with open(args.output_stats, "w") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Statistics saved to: {args.output_stats}")

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
