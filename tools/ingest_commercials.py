"""
================================================================================
INGEST COMMERCIALS - Commercial Data Ingestion Pipeline
================================================================================
Ingests commercial examples into Qdrant with embeddings for RAG retrieval.

Usage:
    from tools.ingest_commercials import CommercialIngester

    ingester = CommercialIngester()
    await ingester.ingest_commercial(
        brand="Nike",
        industry="sports_apparel",
        description="High energy athletic B-roll commercial"
    )

Author: Barrios A2I | 2026-01-12
================================================================================
"""
import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Range
from openai import OpenAI
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "commercial_styles"


# =============================================================================
# INGESTER CLASS
# =============================================================================

class CommercialIngester:
    """Ingest commercial examples into Qdrant vector database."""

    def __init__(self):
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        )
        self._openai = None
        self._claude = None
        self._openai_available = None

    @property
    def openai(self):
        """Lazy init OpenAI client."""
        if self._openai is None:
            try:
                self._openai = OpenAI()
                # Test if API key is valid
                self._openai.models.list()
                self._openai_available = True
            except Exception as e:
                print(f"OpenAI not available: {e}")
                self._openai_available = False
        return self._openai if self._openai_available else None

    @property
    def claude(self):
        """Lazy init Claude client."""
        if self._claude is None:
            self._claude = anthropic.Anthropic()
        return self._claude

    def _generate_id(self, data: Dict[str, Any]) -> str:
        """Generate deterministic ID from content."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_deterministic_embedding(self, text: str, seed: int = 42) -> List[float]:
        """
        Generate a deterministic pseudo-embedding from text.

        Used as fallback when OpenAI is unavailable. Creates consistent
        embeddings based on text content hash, suitable for basic similarity
        matching in seed data scenarios.
        """
        import random

        # Create seed from text content
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = random.Random(text_hash + seed)

        # Generate 1536-dim embedding (same as OpenAI text-embedding-3-small)
        embedding = [rng.gauss(0, 0.1) for _ in range(1536)]

        # Normalize to unit length
        magnitude = sum(x**2 for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using OpenAI if available or deterministic fallback."""
        if self.openai:
            try:
                response = self.openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=text[:8000],  # Truncate to avoid token limits
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"OpenAI embedding failed: {e}, using deterministic fallback")

        # Fallback to deterministic embedding
        return self._get_deterministic_embedding(text)

    async def analyze_commercial(
        self,
        video_url: Optional[str] = None,
        description: str = "",
        brand: str = "",
        industry: str = "",
    ) -> Dict[str, Any]:
        """
        Use Claude to analyze a commercial and extract structured data.

        Args:
            video_url: Optional URL to video
            description: Text description of the commercial
            brand: Brand name
            industry: Industry category

        Returns:
            Structured commercial data
        """
        analysis_prompt = f"""Analyze this commercial and extract structured data:

Brand: {brand}
Industry: {industry}
Description/Notes: {description}
Video URL: {video_url or 'Not provided'}

Extract the following in JSON format:
{{
    "title": "Brief title for this commercial",
    "visual_style": "cinematic|documentary|testimonial|product_demo|lifestyle|animated|b_roll",
    "color_palette": ["#hex1", "#hex2", "#hex3"],
    "camera_movements": ["dolly", "pan", "crane", "static", "handheld", "tracking", "drone"],
    "lighting_style": "moody|bright|natural|dramatic|mixed",
    "pacing": "fast|medium|slow|dynamic",
    "estimated_duration_seconds": 30,
    "estimated_scene_count": 5,
    "scenes": [
        {{
            "scene_number": 1,
            "duration_seconds": 5,
            "visual_description": "Detailed B-roll description for video generation",
            "camera": "camera movement type",
            "mood": "emotional tone"
        }}
    ],
    "hook_type": "question|statement|visual_shock|emotional",
    "cta_type": "visit_website|call|download|buy_now|none",
    "emotional_tone": "inspirational|urgent|trustworthy|fun|serious",
    "voiceover_style": "professional|conversational|urgent|none",
    "music_genre": "electronic|orchestral|acoustic|pop|none",
    "quality_score": 8.5,
    "key_learnings": ["What makes this commercial effective", "..."],
    "scene_descriptions": ["Scene 1 B-roll description", "Scene 2...", ...]
}}

Focus on extracting VISUAL STYLE information that can help generate similar B-roll footage.
The scene_descriptions should be detailed prompts suitable for AI video generation.
Return ONLY valid JSON, no markdown."""

        response = self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": analysis_prompt}],
        )

        content = response.content[0].text.strip()

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)

    async def ingest_commercial(
        self,
        brand: str,
        industry: str,
        description: str = "",
        video_url: Optional[str] = None,
        manual_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest a single commercial into Qdrant.

        Args:
            brand: Brand name
            industry: Industry category
            description: Text description
            video_url: Optional video URL
            manual_data: Pre-structured data (skips Claude analysis)

        Returns:
            Point ID of ingested commercial
        """
        # Get analysis from Claude or use manual data
        if manual_data:
            data = manual_data.copy()
        else:
            data = await self.analyze_commercial(
                video_url=video_url,
                description=description,
                brand=brand,
                industry=industry,
            )

        # Add metadata
        data["brand"] = brand
        data["industry"] = industry
        data["source_url"] = video_url
        data["ingested_at"] = datetime.utcnow().isoformat()

        # Generate embeddings
        # Visual embedding: based on visual descriptions
        visual_text = " ".join([
            data.get("visual_style", ""),
            data.get("lighting_style", ""),
            " ".join(data.get("camera_movements", [])),
            " ".join(data.get("scene_descriptions", [])[:5]),  # First 5 scenes
        ])
        visual_embedding = self._get_embedding(visual_text)

        # Text embedding: based on brand/industry/learnings
        text_content = " ".join([
            data.get("title", ""),
            brand,
            industry,
            data.get("emotional_tone", ""),
            " ".join(data.get("key_learnings", [])),
        ])
        text_embedding = self._get_embedding(text_content)

        # Generate ID
        point_id = self._generate_id({"brand": brand, "title": data.get("title", "")})

        # Create point
        point = PointStruct(
            id=point_id,
            vector={
                "visual": visual_embedding,
                "text": text_embedding,
            },
            payload=data,
        )

        # Upsert to Qdrant
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
        )

        print(f"  Ingested: {data.get('title', 'Unknown')} ({brand})")
        return point_id

    async def ingest_batch(
        self,
        commercials: List[Dict[str, Any]],
    ) -> List[str]:
        """Ingest multiple commercials."""
        ids = []
        for c in commercials:
            try:
                point_id = await self.ingest_commercial(**c)
                ids.append(point_id)
            except Exception as e:
                print(f"  ERROR ingesting {c.get('brand', 'Unknown')}: {e}")
        return ids

    def query_similar(
        self,
        query: str,
        industry: Optional[str] = None,
        visual_style: Optional[str] = None,
        min_quality: float = 7.0,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar commercials.

        Args:
            query: Search query text
            industry: Filter by industry
            visual_style: Filter by visual style
            min_quality: Minimum quality score
            top_k: Number of results

        Returns:
            List of matching commercial data
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter conditions
        must_conditions = []
        should_conditions = []

        if industry:
            should_conditions.append(
                FieldCondition(key="industry", match=MatchValue(value=industry))
            )
        if visual_style:
            should_conditions.append(
                FieldCondition(key="visual_style", match=MatchValue(value=visual_style))
            )
        if min_quality > 0:
            must_conditions.append(
                FieldCondition(key="quality_score", range=Range(gte=min_quality))
            )

        query_filter = None
        if must_conditions or should_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None,
            )

        # Search
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=("visual", query_embedding),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                **r.payload,
            }
            for r in results
        ]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Quick test
    async def test():
        ingester = CommercialIngester()
        results = ingester.query_similar(
            query="cinematic technology commercial with dramatic lighting",
            industry="technology",
            top_k=3,
        )
        print(f"Found {len(results)} results")
        for r in results:
            print(f"  - {r.get('title', 'Unknown')} (score: {r.get('score', 0):.3f})")

    asyncio.run(test())
