"""
================================================================================
SEED QDRANT - Load Commercial Training Data into Qdrant
================================================================================
Initializes the commercial_styles collection and populates it with seed data.

Usage:
    python tools/seed_qdrant.py

Author: Barrios A2I | 2026-01-12
================================================================================
"""
import os
import sys
import json
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.init_qdrant_commercials import init_qdrant_collection
from tools.ingest_commercials import CommercialIngester


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED_FILE = Path(__file__).parent.parent / "data" / "seed_commercials.json"


# =============================================================================
# MAIN
# =============================================================================

async def seed_database(force_recreate: bool = False) -> int:
    """
    Load seed commercials into Qdrant.

    Args:
        force_recreate: If True, delete and recreate collection first

    Returns:
        Number of commercials ingested
    """
    print("=" * 60)
    print("BARRIOS A2I - COMMERCIAL TRAINING DATA SEEDER")
    print("=" * 60)
    print()

    # Step 1: Initialize collection
    print("[1/3] Initializing Qdrant collection...")
    client = init_qdrant_collection(force_recreate=force_recreate)
    print()

    # Step 2: Load seed data
    print("[2/3] Loading seed data...")

    if not SEED_FILE.exists():
        print(f"ERROR: Seed file not found: {SEED_FILE}")
        return 0

    with open(SEED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    commercials = data.get("commercials", [])
    print(f"Found {len(commercials)} commercial examples")
    print()

    # Step 3: Ingest commercials
    print("[3/3] Ingesting commercials into Qdrant...")
    print()

    ingester = CommercialIngester()
    success_count = 0
    fail_count = 0

    for commercial in commercials:
        try:
            point_id = await ingester.ingest_commercial(**commercial)
            success_count += 1
        except Exception as e:
            print(f"  ERROR: {commercial.get('brand', 'Unknown')} - {e}")
            fail_count += 1

    print()

    # Verify
    info = client.get_collection("commercial_styles")
    print("=" * 60)
    print("SEEDING COMPLETE")
    print("=" * 60)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total points in collection: {info.points_count}")
    print("=" * 60)

    return success_count


async def verify_data() -> None:
    """Verify the seeded data by running a test query."""
    print()
    print("=" * 60)
    print("VERIFICATION - Testing RAG Retrieval")
    print("=" * 60)

    ingester = CommercialIngester()

    # Test query
    results = ingester.query_similar(
        query="cinematic technology commercial with dramatic lighting for AI software",
        industry="technology",
        min_quality=8.0,
        top_k=3,
    )

    print(f"Query: 'cinematic technology commercial with dramatic lighting'")
    print(f"Results: {len(results)}")
    print()

    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title', 'Unknown')} ({r.get('brand', '')})")
        print(f"   Score: {r.get('score', 0):.3f}")
        print(f"   Style: {r.get('visual_style', '')} | Industry: {r.get('industry', '')}")
        print(f"   Learnings: {r.get('key_learnings', [])[:2]}")
        print()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed Qdrant with commercial training data")
    parser.add_argument("--force", action="store_true", help="Force recreate collection")
    parser.add_argument("--verify", action="store_true", help="Run verification query after seeding")
    args = parser.parse_args()

    async def main():
        count = await seed_database(force_recreate=args.force)

        if count > 0 and args.verify:
            await verify_data()

    asyncio.run(main())
