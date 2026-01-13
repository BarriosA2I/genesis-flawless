"""
================================================================================
VORTEX v2.1 LEGENDARY - STATE MACHINE
================================================================================
Immutable state models with dual-process routing for AI video assembly.

Processing Modes:
- SYSTEM1_FAST: <200ms rule-based transitions
- SYSTEM1_HYBRID: Fast path with optional AI enhancement
- SYSTEM2_DEEP: Full AI analysis for complex briefs

Author: Barrios A2I | VORTEX v2.1
================================================================================
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class PipelinePhase(str, Enum):
    """Pipeline execution phases"""
    INIT = "init"
    ROUTING = "routing"
    ASSET_DOWNLOAD = "asset_download"
    SCENE_ANALYSIS = "scene_analysis"
    TRANSITION_SELECTION = "transition_selection"
    CLIP_ASSEMBLY = "clip_assembly"
    AUDIO_SYNC = "audio_sync"
    FORMAT_RENDER = "format_render"
    QUALITY_CHECK = "quality_check"
    UPLOAD = "upload"  # Upload to catbox.moe
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(str, Enum):
    """Dual-process routing modes"""
    SYSTEM1_FAST = "system1_fast"         # Rule-based, <200ms
    SYSTEM1_HYBRID = "system1_hybrid"     # Fast + optional enhancement
    SYSTEM2_DEEP = "system2_deep"         # Full AI analysis


class TransitionType(str, Enum):
    """Supported video transitions"""
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"


# =============================================================================
# STATE MODELS
# =============================================================================

class ErrorRecord(BaseModel):
    """Error tracking for pipeline failures"""
    phase: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recoverable: bool = True
    details: Optional[Dict[str, Any]] = None


class SceneAnalysis(BaseModel):
    """AI analysis of a video scene"""
    scene_index: int
    duration_ms: int
    dominant_colors: List[str] = []
    motion_intensity: float = 0.5  # 0-1
    brightness: float = 0.5  # 0-1
    key_objects: List[str] = []
    mood: str = "neutral"
    suggested_transition: TransitionType = TransitionType.CUT


class TransitionDecision(BaseModel):
    """Transition between scenes"""
    from_scene: int
    to_scene: int
    type: TransitionType
    duration_ms: int = 500
    confidence: float = 0.8
    reasoning: Optional[str] = None


class Checkpoint(BaseModel):
    """State checkpoint for recovery"""
    phase: PipelinePhase
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state_hash: str
    version: int


class GlobalState(BaseModel):
    """
    Immutable global state for VORTEX pipeline.
    Uses Pydantic for validation and serialization.
    """
    # Identity
    job_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

    # Pipeline status
    phase: PipelinePhase = PipelinePhase.INIT
    processing_mode: ProcessingMode = ProcessingMode.SYSTEM1_FAST
    complexity_score: float = 0.0  # 0-1
    confidence: float = 1.0

    # Input data
    video_urls: List[str] = []
    voiceover_url: Optional[str] = None
    music_url: Optional[str] = None
    brief_metadata: Dict[str, Any] = {}
    output_formats: List[str] = ["youtube_1080p"]

    # Processing data
    local_clip_paths: List[str] = []
    local_voiceover_path: Optional[str] = None
    local_music_path: Optional[str] = None
    scene_analyses: List[SceneAnalysis] = []
    transitions: List[TransitionDecision] = []

    # Output paths
    assembled_video_path: Optional[str] = None
    video_with_audio_path: Optional[str] = None
    final_output_paths: Dict[str, str] = {}

    # Tracking
    phase_history: List[Dict[str, Any]] = []
    errors: List[ErrorRecord] = []
    warnings: List[str] = []
    checkpoints: List[Checkpoint] = []

    # Metrics
    cost_usd: float = 0.0
    processing_time_ms: float = 0.0

    class Config:
        use_enum_values = True

    def transition_to(self, new_phase: PipelinePhase, **kwargs) -> "GlobalState":
        """
        Create a new state with updated phase.
        Immutable - returns new instance.
        """
        history_entry = {
            "from": self.phase if isinstance(self.phase, str) else self.phase.value,
            "to": new_phase.value,
            "timestamp": datetime.utcnow().isoformat(),
            "version": self.version
        }

        new_history = self.phase_history + [history_entry]

        return self.copy(update={
            "phase": new_phase,
            "phase_history": new_history,
            "updated_at": datetime.utcnow(),
            "version": self.version + 1,
            **kwargs
        })

    def add_error(self, phase: str, message: str, recoverable: bool = True) -> "GlobalState":
        """Add an error and return new state"""
        error = ErrorRecord(
            phase=phase,
            message=message,
            recoverable=recoverable
        )
        return self.copy(update={
            "errors": self.errors + [error],
            "updated_at": datetime.utcnow()
        })

    def create_checkpoint(self) -> "GlobalState":
        """Create a checkpoint of current state"""
        checkpoint = Checkpoint(
            phase=PipelinePhase(self.phase) if isinstance(self.phase, str) else self.phase,
            state_hash=str(hash(self.json())),
            version=self.version
        )
        return self.copy(update={
            "checkpoints": self.checkpoints + [checkpoint]
        })


# =============================================================================
# COMPLEXITY ANALYZER
# =============================================================================

class ComplexityAnalyzer:
    """
    Analyze brief complexity to determine processing mode.
    Implements dual-process routing decision.
    """

    SIMPLE_THRESHOLD = 0.33
    COMPLEX_THRESHOLD = 0.66

    @classmethod
    def analyze(cls, state: GlobalState) -> tuple[ProcessingMode, float]:
        """
        Analyze complexity and return (mode, score).

        Scoring factors:
        - Number of clips (more = complex)
        - Metadata richness (more = complex)
        - Duration variance (high = complex)
        - Custom transitions requested (complex)
        """
        score = 0.0
        weights = {
            "clip_count": 0.25,
            "metadata": 0.25,
            "custom_transitions": 0.30,
            "output_formats": 0.20
        }

        # Clip count factor
        clip_count = len(state.video_urls)
        if clip_count <= 4:
            score += 0.1 * weights["clip_count"]
        elif clip_count <= 8:
            score += 0.5 * weights["clip_count"]
        else:
            score += 1.0 * weights["clip_count"]

        # Metadata richness
        meta = state.brief_metadata
        meta_score = min(len(meta) / 10, 1.0)
        score += meta_score * weights["metadata"]

        # Custom transitions
        if meta.get("custom_transitions"):
            score += 1.0 * weights["custom_transitions"]
        elif meta.get("transition_style"):
            score += 0.5 * weights["custom_transitions"]

        # Output formats
        format_count = len(state.output_formats)
        score += min(format_count / 4, 1.0) * weights["output_formats"]

        # Determine mode
        if score < cls.SIMPLE_THRESHOLD:
            mode = ProcessingMode.SYSTEM1_FAST
        elif score < cls.COMPLEX_THRESHOLD:
            mode = ProcessingMode.SYSTEM1_HYBRID
        else:
            mode = ProcessingMode.SYSTEM2_DEEP

        logger.info(f"Complexity analysis: score={score:.3f}, mode={mode.value}")

        return mode, score


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def produce(state: GlobalState, updater) -> GlobalState:
    """
    Immer-style immutable update helper.
    Takes a state and an updater function, returns new state.
    """
    # Create a copy to work with
    data = state.dict()

    # Apply updates via the updater
    if callable(updater):
        # Create a simple namespace for mutation
        class MutableProxy:
            def __init__(self, d):
                self.__dict__.update(d)

        proxy = MutableProxy(data)
        updater(proxy)

        # Extract changes
        for key in data:
            if hasattr(proxy, key):
                data[key] = getattr(proxy, key)

    data["version"] = state.version + 1
    data["updated_at"] = datetime.utcnow()

    return GlobalState(**data)


# =============================================================================
# TRANSITION MAPPINGS
# =============================================================================

TRANSITION_XFADE_MAP = {
    TransitionType.CUT: "fade",  # Actually instant
    TransitionType.FADE: "fade",
    TransitionType.DISSOLVE: "dissolve",
    TransitionType.WIPE_LEFT: "wipeleft",
    TransitionType.WIPE_RIGHT: "wiperight",
    TransitionType.ZOOM_IN: "smoothup",
    TransitionType.ZOOM_OUT: "smoothdown",
    TransitionType.SLIDE_UP: "slideup",
    TransitionType.SLIDE_DOWN: "slidedown",
}

FORMAT_SPECS = {
    "youtube_1080p": {"width": 1920, "height": 1080, "crf": 18, "fps": 30},
    "youtube_4k": {"width": 3840, "height": 2160, "crf": 15, "fps": 30},
    "tiktok_9x16": {"width": 1080, "height": 1920, "crf": 20, "fps": 30},
    "instagram_1x1": {"width": 1080, "height": 1080, "crf": 20, "fps": 30},
    "instagram_4x5": {"width": 1080, "height": 1350, "crf": 20, "fps": 30},
    "linkedin_16x9": {"width": 1920, "height": 1080, "crf": 20, "fps": 30},
}
