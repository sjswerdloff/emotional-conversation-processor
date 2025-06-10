#!/usr/bin/env python3
"""
Database setup script for Emotional Conversation Processor.

This script handles the initialization and configuration of the Qdrant
vector database for storing conversation embeddings.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402
from qdrant_client.http.models import Distance, VectorParams  # noqa: E402


class DatabaseSetup:
    """Handles database initialization and configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "conversation_history",
        embedding_dimension: int = 384,
    ) -> None:
        """
        Initialize database setup.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to create
            embedding_dimension: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.client: QdrantClient | None = None

    def connect(self) -> bool:
        """
        Connect to Qdrant server.

        Returns:
            True if connection successful
        """
        try:
            self.client = QdrantClient(host=self.host, port=self.port)

            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Server has {len(collections.collections)} existing collections")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False

    def check_server_health(self) -> dict[str, Any]:
        """
        Check Qdrant server health and status.

        Returns:
            Dictionary with health information
        """
        if not self.client:
            return {"status": "disconnected", "error": "No connection to server"}

        try:
            # Get cluster info
            cluster_info = self.client.get_cluster_info()

            # Get collections info
            collections = self.client.get_collections()

            health_info = {
                "status": "healthy",
                "peer_id": cluster_info.peer_id,
                "raft_info": {
                    "term": cluster_info.raft_info.term,
                    "commit": cluster_info.raft_info.commit,
                    "pending_operations": cluster_info.raft_info.pending_operations,
                    "leader": cluster_info.raft_info.leader,
                    "role": cluster_info.raft_info.role,
                },
                "collections_count": len(collections.collections),
                "collections": [col.name for col in collections.collections],
            }

            return health_info

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def create_collection(self, distance_metric: Distance = Distance.COSINE, overwrite: bool = False) -> bool:
        """
        Create the conversation collection.

        Args:
            distance_metric: Distance metric for similarity search
            overwrite: Whether to overwrite existing collection

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("Not connected to Qdrant server")
            return False

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)

            if collection_exists:
                if overwrite:
                    logger.warning(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True

            # Create new collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dimension, distance=distance_metric),
            )

            logger.info(f"Successfully created collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def configure_collection(self) -> bool:
        """
        Configure collection with optimal settings.

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("Not connected to Qdrant server")
            return False

        try:
            # Update collection configuration for better performance
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    # Optimize for search performance
                    indexing_threshold=10000,
                    # Reduce memory usage
                    memmap_threshold=50000,
                ),
                hnsw_config=models.HnswConfigDiff(
                    # Optimize HNSW parameters for emotional context search
                    m=16,  # Number of bi-directional links for each node
                    ef_construct=200,  # Size of dynamic candidate list
                    full_scan_threshold=10000,  # Use exact search for small collections
                ),
            )

            logger.info(f"Configured collection {self.collection_name} for optimal performance")
            return True

        except Exception as e:
            logger.warning(f"Failed to configure collection (this is usually okay): {e}")
            return True  # Not critical for basic functionality

    def create_indexes(self) -> bool:
        """
        Create payload indexes for efficient filtering.

        Returns:
            True if successful
        """
        if not self.client:
            logger.error("Not connected to Qdrant server")
            return False

        try:
            # Create indexes for commonly filtered fields
            indexes_to_create = [
                ("emotional_score", models.PayloadSchemaType.FLOAT),
                ("technical_score", models.PayloadSchemaType.FLOAT),
                ("importance_weight", models.PayloadSchemaType.FLOAT),
                ("speaker", models.PayloadSchemaType.KEYWORD),
                ("conversation_id", models.PayloadSchemaType.KEYWORD),
                ("content_type", models.PayloadSchemaType.KEYWORD),
                ("has_strong_emotion", models.PayloadSchemaType.BOOL),
                ("is_highly_technical", models.PayloadSchemaType.BOOL),
            ]

            for field_name, field_type in indexes_to_create:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name, field_name=field_name, field_schema=field_type
                    )
                    logger.debug(f"Created index for field: {field_name}")
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index for {field_name} might already exist: {e}")

            logger.info("Created payload indexes for efficient filtering")
            return True

        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
            return True  # Not critical for basic functionality

    def verify_setup(self) -> bool:
        """
        Verify that the database setup is correct.

        Returns:
            True if setup is valid
        """
        if not self.client:
            logger.error("Not connected to Qdrant server")
            return False

        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)

            # Verify configuration
            expected_dim = self.embedding_dimension
            actual_dim = collection_info.config.params.vectors.size

            if actual_dim != expected_dim:
                logger.error(f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}")
                return False

            logger.info("Database setup verification passed")
            logger.info(f"Collection: {self.collection_name}")
            logger.info(f"Vector dimension: {actual_dim}")
            logger.info(f"Distance metric: {collection_info.config.params.vectors.distance}")
            logger.info(f"Points count: {collection_info.points_count}")
            logger.info(f"Vectors count: {collection_info.vectors_count}")

            return True

        except Exception as e:
            logger.error(f"Setup verification failed: {e}")
            return False

    def setup_complete_database(self, distance_metric: Distance = Distance.COSINE, overwrite: bool = False) -> bool:
        """
        Perform complete database setup.

        Args:
            distance_metric: Distance metric to use
            overwrite: Whether to overwrite existing collection

        Returns:
            True if successful
        """
        logger.info("Starting complete database setup...")

        # Connect to server
        if not self.connect():
            return False

        # Check server health
        health = self.check_server_health()
        if health["status"] != "healthy":
            logger.error(f"Server is not healthy: {health}")
            return False

        # Create collection
        if not self.create_collection(distance_metric, overwrite):
            return False

        # Configure collection
        self.configure_collection()  # Non-critical

        # Create indexes
        self.create_indexes()  # Non-critical

        # Verify setup
        if not self.verify_setup():
            return False

        logger.info("Database setup completed successfully!")
        return True

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.client:
            return {"error": "Not connected to server"}

        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                },
            }

        except Exception as e:
            return {"error": str(e)}


def main() -> None:
    """Main entry point for database setup."""
    parser = argparse.ArgumentParser(description="Set up Qdrant database for Emotional Conversation Processor")

    parser.add_argument("--host", default="localhost", help="Qdrant server host")

    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port")

    parser.add_argument("--collection-name", default="conversation_history", help="Name of the collection to create")

    parser.add_argument("--embedding-dimension", type=int, default=384, help="Dimension of embedding vectors")

    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean", "dot"],
        default="cosine",
        help="Distance metric for similarity search",
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing collection")

    parser.add_argument("--check-health", action="store_true", help="Only check server health")

    parser.add_argument("--stats", action="store_true", help="Show collection statistics")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Map distance metric
    distance_map = {"cosine": Distance.COSINE, "euclidean": Distance.EUCLID, "dot": Distance.DOT}
    distance_metric = distance_map[args.distance_metric]

    # Initialize setup
    setup = DatabaseSetup(
        host=args.host, port=args.port, collection_name=args.collection_name, embedding_dimension=args.embedding_dimension
    )

    # Handle different operations
    if args.check_health:
        if setup.connect():
            health = setup.check_server_health()
            logger.info(f"Server health: {health}")

            if health["status"] == "healthy":
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(1)

    elif args.stats:
        if setup.connect():
            stats = setup.get_collection_stats()
            logger.info(f"Collection statistics: {stats}")
        else:
            sys.exit(1)

    else:
        # Full setup
        success = setup.setup_complete_database(distance_metric, args.overwrite)

        if success:
            logger.info("Database setup completed successfully")

            # Show final stats
            stats = setup.get_collection_stats()
            if "error" not in stats:
                logger.info("Final collection statistics:")
                for key, value in stats.items():
                    if key != "config":
                        logger.info(f"  {key}: {value}")
        else:
            logger.error("Database setup failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
