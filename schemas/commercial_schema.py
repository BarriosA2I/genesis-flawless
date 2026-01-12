"""
================================================================================
COMMERCIAL SCHEMA - Data Models for Commercial Training Data
================================================================================
Defines the schema for commercial examples stored in Qdrant vector database.

Author: Barrios A2I | 2026-01-12
================================================================================
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class AdPlatform(Enum):
    """Source platform for commercial"""
    META = "meta"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    TV = "tv"
    CUSTOM = "custom"


class VisualStyle(Enum):
    """Visual style classification"""
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    TESTIMONIAL = "testimonial"
    PRODUCT_DEMO = "product_demo"
    LIFESTYLE = "lifestyle"
    ANIMATED = "animated"
    B_ROLL = "b_roll"


class CommercialPacing(Enum):
    """Editing pace classification"""
    FAST = "fast"        # Quick cuts, high energy (5-15 cuts/min)
    MEDIUM = "medium"    # Balanced pacing (3-8 cuts/min)
    SLOW = "slow"        # Long takes, contemplative (1-4 cuts/min)
    DYNAMIC = "dynamic"  # Varies throughout


@dataclass
class SceneDescription:
    """Individual scene within a commercial"""
    scene_number: int
    duration_seconds: int
    visual_description: str  # Detailed B-roll prompt for video generation
    camera: str  # Camera movement type
    mood: str  # Emotional tone


@dataclass
class CommercialExample:
    """
    Schema for commercial training examples in Qdrant.

    This data model captures all the visual, audio, and structural
    patterns from successful commercials to inform AI video generation.
    """

    # Identification
    id: str = ""
    title: str = ""
    brand: str = ""
    industry: str = ""

    # Source
    platform: str = "custom"  # AdPlatform value
    source_url: Optional[str] = None
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Duration & Structure
    duration_seconds: int = 30
    scene_count: int = 4
    scenes: List[Dict[str, Any]] = field(default_factory=list)

    # Visual Analysis
    visual_style: str = "cinematic"  # VisualStyle value
    color_palette: List[str] = field(default_factory=list)  # Hex colors
    camera_movements: List[str] = field(default_factory=list)  # dolly, pan, crane, etc.
    lighting_style: str = "dramatic"  # moody, bright, natural, dramatic

    # Pacing & Rhythm
    pacing: str = "medium"  # CommercialPacing value
    cuts_per_minute: float = 6.0
    music_tempo_bpm: Optional[int] = None

    # Audio
    has_voiceover: bool = True
    voiceover_style: Optional[str] = "professional"  # professional, conversational, urgent
    has_music: bool = True
    music_genre: Optional[str] = "electronic"

    # Content Analysis
    hook_type: str = "visual_shock"  # question, statement, visual_shock, emotional
    cta_type: str = "visit_website"  # visit_website, call, download, buy_now
    emotional_tone: str = "inspirational"  # inspirational, urgent, trustworthy, fun

    # Performance (if available)
    engagement_score: Optional[float] = None  # 0-100
    view_count: Optional[int] = None

    # Raw data for learning
    script_text: Optional[str] = None
    scene_descriptions: List[str] = field(default_factory=list)  # One per scene

    # Key learnings for prompts
    key_learnings: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    quality_score: float = 8.0  # Manual rating 1-10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant payload"""
        return {
            "id": self.id,
            "title": self.title,
            "brand": self.brand,
            "industry": self.industry,
            "platform": self.platform,
            "source_url": self.source_url,
            "ingested_at": self.ingested_at,
            "duration_seconds": self.duration_seconds,
            "scene_count": self.scene_count,
            "scenes": self.scenes,
            "visual_style": self.visual_style,
            "color_palette": self.color_palette,
            "camera_movements": self.camera_movements,
            "lighting_style": self.lighting_style,
            "pacing": self.pacing,
            "cuts_per_minute": self.cuts_per_minute,
            "music_tempo_bpm": self.music_tempo_bpm,
            "has_voiceover": self.has_voiceover,
            "voiceover_style": self.voiceover_style,
            "has_music": self.has_music,
            "music_genre": self.music_genre,
            "hook_type": self.hook_type,
            "cta_type": self.cta_type,
            "emotional_tone": self.emotional_tone,
            "engagement_score": self.engagement_score,
            "view_count": self.view_count,
            "script_text": self.script_text,
            "scene_descriptions": self.scene_descriptions,
            "key_learnings": self.key_learnings,
            "tags": self.tags,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommercialExample":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
