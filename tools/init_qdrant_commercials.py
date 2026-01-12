"""
================================================================================
INIT QDRANT COMMERCIALS - Initialize Commercial Styles Collection
================================================================================
Creates and configures the Qdrant collection for storing commercial examples.

Usage:
    python tools/init_qdrant_commercials.py

Author: Barrios A2I | 2026-01-12
================================================================================
"""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "commercial_styles"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small


# =============================================================================
# MAIN
# =============================================================================

def init_qdrant_collection(force_recreate: bool = False) -> QdrantClient:
    """
    Initialize Qdrant collection for commercial examples.

    Args:
        force_recreate: If True, delete and recreate existing collection

    Returns:
        QdrantClient instance
    """
    print("=" * 60)
    print("QDRANT COMMERCIAL STYLES COLLECTION INIT")
    print("=" * 60)
    print(f"URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print()

    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
    )

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if exists:
        if force_recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
            exists = False
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists")
            info = client.get_collection(COLLECTION_NAME)
            print(f"Points: {info.points_count}")
            print(f"Vectors: {info.config.params.vectors}")
            return client

    # Create collection with dual vectors
    print(f"Creating collection '{COLLECTION_NAME}'...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "visual": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
            "text": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
    )

    print("Created collection with dual vectors (visual, text)")

    # Create payload indexes for efficient filtering
    print("Creating payload indexes...")

    indexes = [
        ("industry", PayloadSchemaType.KEYWORD),
        ("visual_style", PayloadSchemaType.KEYWORD),
        ("brand", PayloadSchemaType.KEYWORD),
        ("pacing", PayloadSchemaType.KEYWORD),
        ("hook_type", PayloadSchemaType.KEYWORD),
        ("duration_seconds", PayloadSchemaType.INTEGER),
        ("quality_score", PayloadSchemaType.FLOAT),
    ]

    for field_name, field_schema in indexes:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"  - Created index: {field_name} ({field_schema})")

    print()
    print("=" * 60)
    print("COLLECTION INITIALIZED SUCCESSFULLY")
    print("=" * 60)

    return client


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Qdrant commercial styles collection")
    parser.add_argument("--force", action="store_true", help="Force recreate collection")
    args = parser.parse_args()

    init_qdrant_collection(force_recreate=args.force)
