#!/usr/bin/env python3
"""
Health check script for Emotional Conversation Processor.

This script performs comprehensive health checks on all system components
including the vector database, ML models, and processing pipeline.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotional_processor.core.models import ConversationSegment  # noqa: E402
from emotional_processor.embeddings.emotion_aware_embedder import EmotionAwareEmbedder  # noqa: E402
from emotional_processor.processors.emotion_classifier import EmotionClassifier  # noqa: E402
from emotional_processor.processors.technical_detector import TechnicalContentDetector  # noqa: E402
from emotional_processor.storage.vector_store import ConversationVectorStore  # noqa: E402


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    component: str
    status: str  # "healthy", "warning", "error"
    message: str
    details: dict[str, Any]
    duration_ms: float


class SystemHealthChecker:
    """Comprehensive system health checker."""

    def __init__(
        self, qdrant_host: str = "localhost", qdrant_port: int = 6333, collection_name: str = "conversation_history"
    ) -> None:
        """
        Initialize health checker.

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Collection name to check
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name

        self.results: list[HealthCheckResult] = []

    def _time_operation(self, operation_name: str, func: Any, *args: Any, **kwargs: Any) -> HealthCheckResult:
        """
        Time an operation and return a health check result.

        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component=operation_name,
                status="healthy",
                message="Operation completed successfully",
                details=result if isinstance(result, dict) else {"result": result},
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component=operation_name,
                status="error",
                message=str(e),
                details={"error_type": type(e).__name__},
                duration_ms=duration_ms,
            )

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            # Get system info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Check thresholds
            status = "healthy"
            warnings = []

            if cpu_percent > 80:
                status = "warning"
                warnings.append("High CPU usage")

            if memory.percent > 85:
                status = "warning"
                warnings.append("High memory usage")

            if disk.percent > 90:
                status = "warning"
                warnings.append("Low disk space")

            return HealthCheckResult(
                component="system_resources",
                status=status,
                message="System resources checked" + (f" - {', '.join(warnings)}" if warnings else ""),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "warnings": warnings,
                },
                duration_ms=1000,  # Approximate duration
            )

        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status="error",
                message=f"Failed to check system resources: {e}",
                details={"error_type": type(e).__name__},
                duration_ms=0,
            )

    def check_qdrant_connection(self) -> dict[str, Any]:
        """Check Qdrant database connection."""
        vector_store = ConversationVectorStore(
            collection_name=self.collection_name, host=self.qdrant_host, port=self.qdrant_port
        )

        # Test connection
        client = vector_store.client
        collections = client.get_collections()

        # Get cluster info
        cluster_info = client.get_cluster_info()

        return {
            "connected": True,
            "collections_count": len(collections.collections),
            "target_collection_exists": any(col.name == self.collection_name for col in collections.collections),
            "peer_id": cluster_info.peer_id,
            "raft_leader": cluster_info.raft_info.leader,
            "raft_role": cluster_info.raft_info.role,
        }

    def check_collection_health(self) -> dict[str, Any]:
        """Check collection-specific health."""
        vector_store = ConversationVectorStore(
            collection_name=self.collection_name, host=self.qdrant_host, port=self.qdrant_port
        )

        # Get collection info
        info = vector_store.get_collection_info()

        # Performance test: simple search
        test_vector = [0.1] * 384  # Simple test vector
        start_time = time.time()
        try:
            search_results = vector_store.search_similar(query_vector=test_vector, limit=5, score_threshold=0.0)
            search_duration_ms = (time.time() - start_time) * 1000
            search_success = True
        except Exception as e:
            search_duration_ms = (time.time() - start_time) * 1000
            search_success = False
            search_error = str(e)

        result = {
            "collection_info": info,
            "search_test": {
                "success": search_success,
                "duration_ms": search_duration_ms,
                "results_count": len(search_results) if search_success else 0,
            },
        }

        if not search_success:
            result["search_test"]["error"] = search_error

        return result

    def check_emotion_classifier(self) -> dict[str, Any]:
        """Check emotion classification model."""
        classifier = EmotionClassifier()

        # Test with sample texts
        test_texts = [
            "I'm really excited about this project!",
            "This is frustrating and difficult.",
            "Thank you so much for your help.",
            "The function returns a dictionary.",
        ]

        results = []
        total_start = time.time()

        for text in test_texts:
            start_time = time.time()
            score, emotions = classifier.classify_single(text)
            duration_ms = (time.time() - start_time) * 1000

            results.append({"text": text, "emotional_score": score, "emotions": emotions, "duration_ms": duration_ms})

        total_duration_ms = (time.time() - total_start) * 1000

        return {
            "model_name": classifier.model_name,
            "test_results": results,
            "total_duration_ms": total_duration_ms,
            "average_duration_ms": total_duration_ms / len(test_texts),
        }

    def check_technical_detector(self) -> dict[str, Any]:
        """Check technical content detector."""
        detector = TechnicalContentDetector()

        # Test with sample texts
        test_texts = [
            "I'm excited about our collaboration!",
            "Here's the Python code: def func(): return True",
            "The API endpoint returns JSON data with status codes.",
            "Thank you for your patience and understanding.",
        ]

        results = []
        total_start = time.time()

        for text in test_texts:
            start_time = time.time()
            score = detector.calculate_technical_score(text)
            duration_ms = (time.time() - start_time) * 1000

            results.append({"text": text, "technical_score": score, "duration_ms": duration_ms})

        total_duration_ms = (time.time() - total_start) * 1000

        return {
            "test_results": results,
            "total_duration_ms": total_duration_ms,
            "average_duration_ms": total_duration_ms / len(test_texts),
        }

    def check_embedder(self) -> dict[str, Any]:
        """Check embedding generation."""
        embedder = EmotionAwareEmbedder()

        # Test with sample segment
        test_segment = ConversationSegment(
            content="I'm grateful for your help with this project!",
            speaker="User",
            emotional_score=0.8,
            emotional_labels=["gratitude", "joy"],
            technical_score=0.1,
            importance_weight=0.7,
        )

        start_time = time.time()
        embedding = embedder.create_embedding(test_segment)
        duration_ms = (time.time() - start_time) * 1000

        # Validate embedding
        embedding_valid = (
            isinstance(embedding, list)
            and len(embedding) == embedder.dimension
            and all(isinstance(x, float) for x in embedding)
        )

        # Test similarity calculation
        embedding2 = embedder.create_embedding(test_segment)  # Same segment
        similarity = embedder.similarity(embedding, embedding2)

        return {
            "model_name": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "test_embedding": {
                "valid": embedding_valid,
                "dimension": len(embedding),
                "duration_ms": duration_ms,
                "sample_values": embedding[:5],  # First 5 values
            },
            "similarity_test": {"self_similarity": similarity, "expected_high": similarity > 0.95},
        }

    def check_end_to_end_pipeline(self) -> dict[str, Any]:
        """Check complete processing pipeline."""
        # Create test conversation segment
        test_content = "I'm really excited about this collaboration! It means so much to me."

        start_time = time.time()

        # Step 1: Emotion classification
        classifier = EmotionClassifier()
        emotional_score, emotional_labels = classifier.classify_single(test_content)

        # Step 2: Technical detection
        detector = TechnicalContentDetector()
        technical_score = detector.calculate_technical_score(test_content)

        # Step 3: Create segment
        segment = ConversationSegment(
            content=test_content,
            speaker="User",
            emotional_score=emotional_score,
            emotional_labels=emotional_labels,
            technical_score=technical_score,
            importance_weight=max(0.0, emotional_score - technical_score * 0.5),
        )

        # Step 4: Create embedding
        embedder = EmotionAwareEmbedder()
        embedding = embedder.create_embedding(segment)

        # Step 5: Test vector storage (if available)
        try:
            vector_store = ConversationVectorStore(
                collection_name=self.collection_name, host=self.qdrant_host, port=self.qdrant_port
            )

            # Store and retrieve test
            point_id = vector_store.store_segment(segment, embedding)

            # Search test
            vector_store.search_similar(query_vector=embedding, limit=1)

            # Cleanup
            vector_store.delete_segment(point_id)

            storage_success = True
            storage_error = None

        except Exception as e:
            storage_success = False
            storage_error = str(e)

        total_duration_ms = (time.time() - start_time) * 1000

        return {
            "test_content": test_content,
            "pipeline_steps": {
                "emotion_classification": {"score": emotional_score, "labels": emotional_labels},
                "technical_detection": {"score": technical_score},
                "embedding_generation": {"dimension": len(embedding), "valid": len(embedding) == 384},
                "vector_storage": {"success": storage_success, "error": storage_error},
            },
            "total_duration_ms": total_duration_ms,
        }

    def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        logger.info("Starting comprehensive health checks...")

        # System resources
        self.results.append(self.check_system_resources())

        # Qdrant connection
        self.results.append(self._time_operation("qdrant_connection", self.check_qdrant_connection))

        # Collection health
        self.results.append(self._time_operation("collection_health", self.check_collection_health))

        # Emotion classifier
        self.results.append(self._time_operation("emotion_classifier", self.check_emotion_classifier))

        # Technical detector
        self.results.append(self._time_operation("technical_detector", self.check_technical_detector))

        # Embedder
        self.results.append(self._time_operation("embedder", self.check_embedder))

        # End-to-end pipeline
        self.results.append(self._time_operation("e2e_pipeline", self.check_end_to_end_pipeline))

        return self.results

    def get_overall_status(self) -> tuple[str, dict[str, Any]]:
        """
        Get overall system status.

        Returns:
            Tuple of (status, summary)
        """
        if not self.results:
            return "unknown", {"message": "No health checks performed"}

        error_count = sum(1 for r in self.results if r.status == "error")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        healthy_count = sum(1 for r in self.results if r.status == "healthy")

        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        summary = {
            "overall_status": overall_status,
            "total_checks": len(self.results),
            "healthy": healthy_count,
            "warnings": warning_count,
            "errors": error_count,
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "checks": [
                {"component": r.component, "status": r.status, "message": r.message, "duration_ms": r.duration_ms}
                for r in self.results
            ],
        }

        return overall_status, summary


def main() -> None:
    """Main entry point for health checker."""
    parser = argparse.ArgumentParser(description="Perform health checks on Emotional Conversation Processor")

    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")

    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")

    parser.add_argument("--collection-name", default="conversation_history", help="Collection name to check")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    parser.add_argument(
        "--component",
        choices=["system", "qdrant", "collection", "emotion", "technical", "embedder", "e2e"],
        help="Check only specific component",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Initialize health checker
    checker = SystemHealthChecker(
        qdrant_host=args.qdrant_host, qdrant_port=args.qdrant_port, collection_name=args.collection_name
    )

    # Run checks
    if args.component:
        # Run specific component check
        component_map = {
            "system": checker.check_system_resources,
            "qdrant": lambda: checker._time_operation("qdrant_connection", checker.check_qdrant_connection),
            "collection": lambda: checker._time_operation("collection_health", checker.check_collection_health),
            "emotion": lambda: checker._time_operation("emotion_classifier", checker.check_emotion_classifier),
            "technical": lambda: checker._time_operation("technical_detector", checker.check_technical_detector),
            "embedder": lambda: checker._time_operation("embedder", checker.check_embedder),
            "e2e": lambda: checker._time_operation("e2e_pipeline", checker.check_end_to_end_pipeline),
        }
        result = component_map[args.component]()
        checker.results = [result]
    else:
        # Run all checks
        checker.run_all_checks()

    # Get overall status
    overall_status, summary = checker.get_overall_status()

    # Output results
    if args.json:
        import json

        print(json.dumps(summary, indent=2))
    else:
        # Human-readable output
        logger.info("=== Health Check Results ===")
        logger.info(f"Overall Status: {overall_status.upper()}")
        logger.info(f"Total Checks: {summary['total_checks']}")
        logger.info(f"Healthy: {summary['healthy']}, Warnings: {summary['warnings']}, Errors: {summary['errors']}")
        logger.info(f"Total Duration: {summary['total_duration_ms']:.1f}ms")

        for check in summary["checks"]:
            status_symbol = {"healthy": "✅", "warning": "⚠️", "error": "❌"}[check["status"]]
            logger.info(f"{status_symbol} {check['component']}: {check['message']} ({check['duration_ms']:.1f}ms)")

            if args.verbose and check["component"] in [r.component for r in checker.results]:
                # Show details for verbose mode
                result_details = next(r.details for r in checker.results if r.component == check["component"])
                for key, value in result_details.items():
                    if isinstance(value, dict):
                        logger.debug(f"  {key}: {value}")
                    else:
                        logger.debug(f"  {key}: {value}")

    # Exit with appropriate code
    if overall_status == "error":
        sys.exit(1)
    elif overall_status == "warning":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
