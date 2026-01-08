"""
GENESIS Music Selection Agent (Agent 6)
===============================================================================
Standalone music selector for video production pipeline.

Adapted from RAGNAROK v3.1.0 MusicSelectionAgent - made standalone without
RagnarokCore dependency.

Key Features:
- Local library of 14 royalty-free tracks ($0 per use)
- Mood-based track matching
- FFmpeg sidechain ducking configuration
- Industry-aware music selection

Author: Barrios A2I
Version: 1.0.0 (GENESIS Standalone)
===============================================================================
"""

import asyncio
import hashlib
import logging
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger("genesis.music_selector")


# =============================================================================
# DATA MODELS
# =============================================================================

class MusicProvider(str, Enum):
    LOCAL = "local"           # Free, pre-licensed tracks
    EPIDEMIC = "epidemic"     # Epidemic Sound API (future)
    ARTLIST = "artlist"       # Artlist API (future)
    SUNO = "suno"            # Suno AI generation (future)


class MusicTrack(BaseModel):
    """Selected music track."""
    id: str
    title: str
    artist: str = "Stock Library"
    url: str                  # Local path or CDN URL
    duration: float           # Seconds
    bpm: int = 120
    key: str = "C"
    mood: List[str] = []
    genre: List[str] = []
    provider: MusicProvider = MusicProvider.LOCAL
    cost: float = 0.0         # Cost to use this track
    license_type: str = "royalty_free"


class DuckingConfig(BaseModel):
    """FFmpeg sidechain compression config for audio ducking."""
    type: str = "sidechain"
    threshold: float = 0.1    # Compressor threshold
    ratio: int = 4            # Compression ratio
    attack: int = 200         # Attack time (ms)
    release: int = 500        # Release time (ms)
    music_base_volume: float = 0.4  # Base music volume


class MusicRequest(BaseModel):
    """Input request for music selection."""
    industry: str
    mood: str                 # Primary mood
    duration: float           # Required duration in seconds
    emotional_arc: List[str] = []
    bpm_preference: Optional[int] = None
    genre_preference: Optional[str] = None
    exclude_tracks: List[str] = []


class MusicResponse(BaseModel):
    """Output from music selection."""
    primary_track: MusicTrack
    backup_tracks: List[MusicTrack] = []
    ducking_config: DuckingConfig
    confidence: float
    source: str               # "cache", "local", "api"
    cost: float = 0.0
    processing_time_ms: float = 0.0


# =============================================================================
# LOCAL MUSIC LIBRARY (14 Royalty-Free Tracks)
# =============================================================================

LOCAL_LIBRARY: List[MusicTrack] = [
    # Corporate / Professional
    MusicTrack(
        id="local_corp_01", title="Corporate Uplift", artist="Stock Library",
        url="assets/music/corporate_uplift.mp3", duration=180, bpm=120,
        mood=["professional", "confident", "optimistic"],
        genre=["corporate", "ambient"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_corp_02", title="Business Forward", artist="Stock Library",
        url="assets/music/business_forward.mp3", duration=200, bpm=110,
        mood=["professional", "determined", "modern"],
        genre=["corporate", "electronic"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_corp_03", title="Clean Success", artist="Stock Library",
        url="assets/music/clean_success.mp3", duration=240, bpm=100,
        mood=["professional", "trustworthy", "calm"],
        genre=["corporate", "piano"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Warm / Inviting
    MusicTrack(
        id="local_warm_01", title="Warm Welcome", artist="Stock Library",
        url="assets/music/warm_welcome.mp3", duration=180, bpm=90,
        mood=["warm", "inviting", "friendly"],
        genre=["acoustic", "folk"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_warm_02", title="Cozy Afternoon", artist="Stock Library",
        url="assets/music/cozy_afternoon.mp3", duration=200, bpm=85,
        mood=["warm", "relaxed", "comfortable"],
        genre=["jazz", "lounge"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Energetic / Upbeat
    MusicTrack(
        id="local_energy_01", title="Power Drive", artist="Stock Library",
        url="assets/music/power_drive.mp3", duration=180, bpm=140,
        mood=["energetic", "powerful", "motivational"],
        genre=["electronic", "edm"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_energy_02", title="Rise Up", artist="Stock Library",
        url="assets/music/rise_up.mp3", duration=200, bpm=130,
        mood=["energetic", "inspiring", "uplifting"],
        genre=["pop", "electronic"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Cinematic / Emotional
    MusicTrack(
        id="local_cine_01", title="Cinematic Dreams", artist="Stock Library",
        url="assets/music/cinematic_dreams.mp3", duration=240, bpm=80,
        mood=["cinematic", "emotional", "dramatic"],
        genre=["orchestral", "cinematic"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_cine_02", title="Epic Journey", artist="Stock Library",
        url="assets/music/epic_journey.mp3", duration=220, bpm=95,
        mood=["cinematic", "inspiring", "epic"],
        genre=["orchestral", "trailer"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Calm / Trustworthy
    MusicTrack(
        id="local_calm_01", title="Gentle Care", artist="Stock Library",
        url="assets/music/gentle_care.mp3", duration=200, bpm=70,
        mood=["calm", "trustworthy", "reassuring"],
        genre=["ambient", "piano"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_calm_02", title="Safe Hands", artist="Stock Library",
        url="assets/music/safe_hands.mp3", duration=180, bpm=75,
        mood=["calm", "professional", "gentle"],
        genre=["ambient", "soft"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Modern / Tech
    MusicTrack(
        id="local_tech_01", title="Digital Pulse", artist="Stock Library",
        url="assets/music/digital_pulse.mp3", duration=180, bpm=128,
        mood=["modern", "innovative", "tech"],
        genre=["electronic", "synthwave"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
    MusicTrack(
        id="local_tech_02", title="Future Forward", artist="Stock Library",
        url="assets/music/future_forward.mp3", duration=200, bpm=125,
        mood=["modern", "futuristic", "dynamic"],
        genre=["electronic", "tech"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),

    # Happy / Cheerful
    MusicTrack(
        id="local_happy_01", title="Sunshine Day", artist="Stock Library",
        url="assets/music/sunshine_day.mp3", duration=180, bpm=115,
        mood=["happy", "cheerful", "bright"],
        genre=["pop", "upbeat"],
        provider=MusicProvider.LOCAL, cost=0.0
    ),
]


# =============================================================================
# INDUSTRY MOOD MAPPINGS
# =============================================================================

INDUSTRY_PROFILES = {
    "technology": {
        "mood": "modern",
        "music_genre": "electronic",
        "ducking": {"threshold": 0.12, "ratio": 3, "music_volume": 0.45}
    },
    "healthcare": {
        "mood": "calm",
        "music_genre": "ambient",
        "ducking": {"threshold": 0.08, "ratio": 5, "music_volume": 0.35}
    },
    "finance": {
        "mood": "professional",
        "music_genre": "corporate",
        "ducking": {"threshold": 0.1, "ratio": 4, "music_volume": 0.4}
    },
    "retail": {
        "mood": "energetic",
        "music_genre": "pop",
        "ducking": {"threshold": 0.12, "ratio": 4, "music_volume": 0.45}
    },
    "real_estate": {
        "mood": "warm",
        "music_genre": "acoustic",
        "ducking": {"threshold": 0.08, "ratio": 5, "music_volume": 0.35}
    },
    "fitness": {
        "mood": "energetic",
        "music_genre": "electronic",
        "ducking": {"threshold": 0.15, "ratio": 3, "music_volume": 0.5}
    },
    "food": {
        "mood": "warm",
        "music_genre": "acoustic",
        "ducking": {"threshold": 0.1, "ratio": 4, "music_volume": 0.4}
    },
    "automotive": {
        "mood": "cinematic",
        "music_genre": "orchestral",
        "ducking": {"threshold": 0.12, "ratio": 4, "music_volume": 0.45}
    },
    "travel": {
        "mood": "inspiring",
        "music_genre": "cinematic",
        "ducking": {"threshold": 0.1, "ratio": 4, "music_volume": 0.4}
    },
    "education": {
        "mood": "calm",
        "music_genre": "ambient",
        "ducking": {"threshold": 0.08, "ratio": 5, "music_volume": 0.35}
    },
    "flooring": {
        "mood": "professional",
        "music_genre": "corporate",
        "ducking": {"threshold": 0.1, "ratio": 4, "music_volume": 0.4}
    },
}


# =============================================================================
# SIMPLE CACHE
# =============================================================================

@dataclass
class SimpleCache:
    """Simple in-memory cache for music selections."""
    _cache: Dict[str, Dict] = field(default_factory=dict)
    max_size: int = 100

    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict):
        # Simple LRU eviction
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear(self):
        self._cache.clear()


# =============================================================================
# MUSIC SELECTION AGENT
# =============================================================================

class MusicSelectionAgent:
    """
    Standalone music selection agent for GENESIS pipeline.

    Selection Strategy (in order):
    1. Check cache (previous successful selections)
    2. Search local library (free, instant)
    3. Return best-effort fallback
    """

    def __init__(self, local_library: Optional[List[MusicTrack]] = None):
        self.local_library = local_library or LOCAL_LIBRARY
        self.industry_profiles = INDUSTRY_PROFILES
        self.cache = SimpleCache()

        # Stats tracking
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "local_hits": 0,
            "fallbacks": 0,
            "total_cost": 0.0
        }

        logger.info(f"[MusicSelector] Initialized with {len(self.local_library)} local tracks")

    async def select_music(self, request: MusicRequest) -> MusicResponse:
        """
        Select music for video production.

        Args:
            request: Music selection request with industry, mood, duration

        Returns:
            MusicResponse with selected track and ducking config
        """
        start_time = time.time()
        self.stats["requests"] += 1

        logger.info(f"[MusicSelector] Selecting: industry={request.industry}, mood={request.mood}, duration={request.duration}s")

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            elapsed = (time.time() - start_time) * 1000

            return MusicResponse(
                primary_track=MusicTrack(**cached["primary_track"]),
                backup_tracks=[MusicTrack(**t) for t in cached.get("backup_tracks", [])],
                ducking_config=DuckingConfig(**cached.get("ducking_config", {})),
                confidence=cached.get("confidence", 0.9),
                source="cache",
                cost=0.0,
                processing_time_ms=elapsed
            )

        # Get industry profile
        industry_profile = self.industry_profiles.get(
            request.industry.lower(),
            {"mood": request.mood, "music_genre": None, "ducking": {}}
        )

        target_mood = industry_profile.get("mood", request.mood)
        target_genre = industry_profile.get("music_genre", request.genre_preference)

        # Search local library
        local_matches = self._search_local_library(
            mood=target_mood,
            genre=target_genre,
            min_duration=request.duration,
            exclude=request.exclude_tracks
        )

        if local_matches:
            self.stats["local_hits"] += 1
            primary = local_matches[0]
            backups = local_matches[1:3]

            ducking_config = self._generate_ducking_config(request, industry_profile)
            confidence = self._calculate_confidence(primary, request)

            response = MusicResponse(
                primary_track=primary,
                backup_tracks=backups,
                ducking_config=ducking_config,
                confidence=confidence,
                source="local",
                cost=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

            # Cache if high confidence
            if confidence >= 0.8:
                self.cache.set(cache_key, response.dict())

            logger.info(f"[MusicSelector] Selected: {primary.title} (confidence: {confidence:.2f})")
            return response

        # Fallback to best available
        self.stats["fallbacks"] += 1
        fallback = self._get_fallback_track(request)
        ducking_config = self._generate_ducking_config(request, industry_profile)

        logger.warning(f"[MusicSelector] Using fallback: {fallback.title}")

        return MusicResponse(
            primary_track=fallback,
            backup_tracks=[],
            ducking_config=ducking_config,
            confidence=0.6,
            source="fallback",
            cost=0.0,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    def _generate_cache_key(self, request: MusicRequest) -> str:
        """Generate cache key from request parameters."""
        key_parts = [
            request.industry.lower(),
            request.mood.lower(),
            str(int(request.duration / 30) * 30)  # Round to 30s buckets
        ]
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _search_local_library(
        self,
        mood: str,
        genre: Optional[str],
        min_duration: float,
        exclude: List[str]
    ) -> List[MusicTrack]:
        """Search local library with mood/genre matching."""
        matches = []

        for track in self.local_library:
            # Skip excluded tracks
            if track.id in exclude:
                continue

            # Duration must be sufficient
            if track.duration < min_duration:
                continue

            # Calculate match score
            score = 0.0

            # Mood matching (fuzzy)
            mood_lower = mood.lower()
            for track_mood in track.mood:
                if mood_lower in track_mood.lower() or track_mood.lower() in mood_lower:
                    score += 0.4
                    break

            # Genre matching (fuzzy)
            if genre:
                genre_lower = genre.lower()
                for track_genre in track.genre:
                    if genre_lower in track_genre.lower() or track_genre.lower() in genre_lower:
                        score += 0.3
                        break

            # Duration bonus (closer to target = better)
            duration_diff = abs(track.duration - min_duration)
            if duration_diff < 30:
                score += 0.2
            elif duration_diff < 60:
                score += 0.1

            if score > 0.2:  # Minimum threshold
                matches.append((score, track))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [track for _, track in matches]

    def _get_fallback_track(self, request: MusicRequest) -> MusicTrack:
        """Get best-effort fallback track from local library."""
        # Find tracks that meet duration
        valid = [t for t in self.local_library if t.duration >= request.duration]
        if valid:
            return valid[0]

        # If nothing long enough, return longest available
        return max(self.local_library, key=lambda t: t.duration)

    def _generate_ducking_config(
        self,
        request: MusicRequest,
        industry_profile: Dict[str, Any]
    ) -> DuckingConfig:
        """Generate FFmpeg sidechain compression config."""
        ducking_prefs = industry_profile.get("ducking", {})

        return DuckingConfig(
            threshold=ducking_prefs.get("threshold", 0.1),
            ratio=ducking_prefs.get("ratio", 4),
            attack=ducking_prefs.get("attack", 200),
            release=ducking_prefs.get("release", 500),
            music_base_volume=ducking_prefs.get("music_volume", 0.4)
        )

    def _calculate_confidence(self, track: MusicTrack, request: MusicRequest) -> float:
        """Calculate confidence score for track selection."""
        score = 0.7  # Base score

        # Duration match
        if track.duration >= request.duration:
            score += 0.15

        # Mood match
        request_mood = request.mood.lower()
        if any(request_mood in m.lower() for m in track.mood):
            score += 0.1

        # BPM preference match
        if request.bpm_preference:
            bpm_diff = abs(track.bpm - request.bpm_preference)
            if bpm_diff < 10:
                score += 0.05

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "local_library_size": len(self.local_library),
            "stats": self.stats,
            "cache_size": len(self.cache._cache)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_music_selector() -> MusicSelectionAgent:
    """Create MusicSelectionAgent instance."""
    return MusicSelectionAgent()


# =============================================================================
# TESTING
# =============================================================================

async def test_music_selector():
    """Test the music selection agent."""
    agent = create_music_selector()

    # Test various industries
    test_cases = [
        {"industry": "technology", "mood": "modern", "duration": 30},
        {"industry": "healthcare", "mood": "calm", "duration": 45},
        {"industry": "fitness", "mood": "energetic", "duration": 60},
        {"industry": "finance", "mood": "professional", "duration": 30},
    ]

    print("\n[MusicSelector] Testing music selection")
    print("=" * 60)

    for case in test_cases:
        request = MusicRequest(**case)
        response = await agent.select_music(request)

        print(f"\nIndustry: {case['industry']}")
        print(f"  Track: {response.primary_track.title}")
        print(f"  Source: {response.source}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Ducking: threshold={response.ducking_config.threshold}, vol={response.ducking_config.music_base_volume}")

    print(f"\nStats: {agent.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_music_selector())
