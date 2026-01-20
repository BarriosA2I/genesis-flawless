#!/usr/bin/env python3
"""
THE EDITOR (Agent 7.5) - Shot Detection & Assembly Agent
=========================================================
RAGNAROK Video Pipeline | VORTEX Phase
Barrios A2I Commercial Video Automation

Purpose: Shot boundary detection, transition selection, and video assembly analysis
Pipeline Position: After WORDSMITH (7.25), before FINALIZER (8.0)
Cost Target: $0.15-0.30/video
Latency Target: 8-20s

Features:
- TransNet V2 shot boundary detection (99.2% F1 benchmark hooks)
- Editing grammar engine with context-aware transitions
- Color grading analysis with LUT recommendations
- Neural video stabilization analysis (FILM-Net hooks)
- Timeline assembly with FFmpeg command generation

Author: Barrios A2I Architecture Team
Version: 1.0.0
Date: January 19, 2026
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Third-party imports with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback dataclass-based models
    BaseModel = object
    Field = lambda *args, **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f
    root_validator = lambda *args, **kwargs: lambda f: f

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("THE_EDITOR")

# =============================================================================
# SECTION 1: ENUMS & CONSTANTS
# =============================================================================

class TransitionType(str, Enum):
    """Detected and recommended transition types for video editing"""
    HARD_CUT = "hard_cut"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    DISSOLVE = "dissolve"
    CROSS_DISSOLVE = "cross_dissolve"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    IRIS_IN = "iris_in"
    IRIS_OUT = "iris_out"
    PUSH = "push"
    SLIDE = "slide"
    ZOOM = "zoom"
    FLASH = "flash"
    DIP_TO_BLACK = "dip_to_black"
    DIP_TO_WHITE = "dip_to_white"
    MATCH_CUT = "match_cut"
    JUMP_CUT = "jump_cut"
    L_CUT = "l_cut"
    J_CUT = "j_cut"
    GRADUAL = "gradual_change"
    UNKNOWN = "unknown"


class StylePreset(str, Enum):
    """Commercial style presets affecting editing decisions"""
    COMMERCIAL = "commercial"
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    MUSIC_VIDEO = "music_video"
    CORPORATE = "corporate"
    SOCIAL_MEDIA = "social_media"
    BROADCAST = "broadcast"
    FILM_TRAILER = "film_trailer"
    PRODUCT_DEMO = "product_demo"
    TESTIMONIAL = "testimonial"
    ENERGETIC = "energetic"
    CALM = "calm"
    DRAMATIC = "dramatic"
    MINIMALIST = "minimalist"
    RETRO = "retro"
    MODERN = "modern"


class StabilizationMethod(str, Enum):
    """Video stabilization methods"""
    NONE = "none"
    WARP = "warp"
    CROP = "crop"
    OPTICAL_FLOW = "optical_flow"
    GYRO = "gyro"
    HYBRID = "hybrid"
    NEURAL = "neural"
    FILM_NET = "film_net"


class ColorGradingIntent(str, Enum):
    """Color grading creative intents"""
    NEUTRAL = "neutral"
    WARM = "warm"
    COOL = "cool"
    HIGH_CONTRAST = "high_contrast"
    LOW_CONTRAST = "low_contrast"
    VINTAGE = "vintage"
    TEAL_ORANGE = "teal_orange"
    DESATURATED = "desaturated"
    VIBRANT = "vibrant"
    CINEMATIC = "cinematic"
    BROADCAST_SAFE = "broadcast_safe"


class ShotType(str, Enum):
    """Shot composition types"""
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM_WIDE = "medium_wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    INSERT = "insert"
    CUTAWAY = "cutaway"
    OVER_SHOULDER = "over_shoulder"
    POV = "pov"
    TWO_SHOT = "two_shot"
    GROUP = "group"
    ESTABLISHING = "establishing"
    UNKNOWN = "unknown"


class MotionType(str, Enum):
    """Camera motion types"""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRACK_LEFT = "track_left"
    TRACK_RIGHT = "track_right"
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    HANDHELD = "handheld"
    STEADICAM = "steadicam"
    DRONE = "drone"
    MIXED = "mixed"


class RhythmPattern(str, Enum):
    """Editing rhythm patterns"""
    CONSTANT = "constant"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    ALTERNATING = "alternating"
    BUILDING = "building"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    IRREGULAR = "irregular"


class EditingSeverity(str, Enum):
    """Issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# SECTION 2: CONSTANTS & CONFIGURATION
# =============================================================================

# Default frame rates
DEFAULT_FPS = 30.0
CINEMA_FPS = 24.0
BROADCAST_FPS = 29.97
HIGH_FPS = 60.0

# TransNet V2 model configuration
TRANSNET_INPUT_SIZE = (48, 27)  # Width x Height
TRANSNET_BATCH_SIZE = 100
TRANSNET_THRESHOLD = 0.5

# Stabilization thresholds
SHAKE_THRESHOLD_LOW = 0.15
SHAKE_THRESHOLD_MEDIUM = 0.35
SHAKE_THRESHOLD_HIGH = 0.55

# Color analysis
COLOR_CLUSTERS = 5
HISTOGRAM_BINS = 256

# Timeline defaults
MIN_SHOT_DURATION_FRAMES = 6  # 0.2s at 30fps
MAX_SHOT_DURATION_FRAMES = 900  # 30s at 30fps
DEFAULT_TRANSITION_DURATION_FRAMES = 15  # 0.5s at 30fps

# =============================================================================
# SECTION 3: PYDANTIC MODELS
# =============================================================================

if PYDANTIC_AVAILABLE:
    
    class VideoMetadata(BaseModel):
        """Video file metadata"""
        file_path: str
        width: int = Field(ge=1, description="Video width in pixels")
        height: int = Field(ge=1, description="Video height in pixels")
        fps: float = Field(ge=1.0, le=240.0, description="Frames per second")
        total_frames: int = Field(ge=1, description="Total frame count")
        duration_seconds: float = Field(ge=0.0, description="Duration in seconds")
        codec: str = "unknown"
        bitrate_kbps: Optional[int] = None
        audio_tracks: int = 0
        has_audio: bool = False
        color_space: str = "unknown"
        bit_depth: int = 8
        
        @property
        def aspect_ratio(self) -> float:
            return self.width / self.height if self.height > 0 else 0.0
        
        @property
        def resolution_label(self) -> str:
            if self.height >= 2160:
                return "4K"
            elif self.height >= 1080:
                return "1080p"
            elif self.height >= 720:
                return "720p"
            elif self.height >= 480:
                return "480p"
            else:
                return f"{self.height}p"


    class ShotBoundary(BaseModel):
        """Detected shot boundary with timing and transition info"""
        shot_id: str = Field(description="Unique shot identifier")
        frame_start: int = Field(ge=0, description="Start frame number")
        frame_end: int = Field(ge=0, description="End frame number")
        timestamp_start: float = Field(ge=0.0, description="Start time in seconds")
        timestamp_end: float = Field(ge=0.0, description="End time in seconds")
        transition_type: TransitionType = TransitionType.HARD_CUT
        confidence: float = Field(ge=0.0, le=1.0, default=0.0)
        duration_frames: int = Field(ge=1, default=1)
        duration_seconds: float = Field(ge=0.0, default=0.0)
        shot_type: Optional[ShotType] = None
        motion_type: Optional[MotionType] = None
        dominant_color: Optional[Tuple[int, int, int]] = None
        avg_brightness: Optional[float] = None
        
        @validator('frame_end')
        def end_after_start(cls, v, values):
            if 'frame_start' in values and v < values['frame_start']:
                raise ValueError('frame_end must be >= frame_start')
            return v
        
        @property
        def frame_count(self) -> int:
            return self.frame_end - self.frame_start + 1


    class TransitionRecommendation(BaseModel):
        """Recommended transition between shots"""
        transition_id: str
        from_shot_id: str
        to_shot_id: str
        transition_type: TransitionType
        duration_frames: int = Field(ge=1, le=300, default=15)
        duration_seconds: float = Field(ge=0.0, default=0.5)
        reasoning: str = ""
        alternatives: List[TransitionType] = Field(default_factory=list)
        confidence: float = Field(ge=0.0, le=1.0, default=0.8)
        grammar_rule_applied: Optional[str] = None
        ffmpeg_filter: Optional[str] = None
        
        def get_ffmpeg_filter(self, fps: float = 30.0) -> str:
            """Generate FFmpeg filter string for this transition"""
            if self.ffmpeg_filter:
                return self.ffmpeg_filter
            
            duration = self.duration_frames / fps
            
            if self.transition_type == TransitionType.DISSOLVE:
                return f"xfade=transition=dissolve:duration={duration:.3f}"
            elif self.transition_type == TransitionType.FADE_OUT:
                return f"fade=t=out:d={duration:.3f}"
            elif self.transition_type == TransitionType.FADE_IN:
                return f"fade=t=in:d={duration:.3f}"
            elif self.transition_type in (TransitionType.WIPE_LEFT, TransitionType.WIPE_RIGHT):
                direction = "l" if self.transition_type == TransitionType.WIPE_LEFT else "r"
                return f"xfade=transition=wipe{direction}:duration={duration:.3f}"
            elif self.transition_type == TransitionType.DIP_TO_BLACK:
                return f"xfade=transition=fade:duration={duration:.3f}"
            else:
                return ""  # Hard cut needs no filter


    class ColorAnalysis(BaseModel):
        """Color analysis for a shot or frame"""
        shot_id: str
        dominant_colors: List[Tuple[int, int, int]] = Field(default_factory=list)
        color_palette: List[str] = Field(default_factory=list, description="Hex color codes")
        histogram_r: List[int] = Field(default_factory=list)
        histogram_g: List[int] = Field(default_factory=list)
        histogram_b: List[int] = Field(default_factory=list)
        avg_brightness: float = Field(ge=0.0, le=255.0, default=128.0)
        avg_saturation: float = Field(ge=0.0, le=1.0, default=0.5)
        white_balance_temp: float = Field(default=6500.0, description="Color temp in Kelvin")
        white_balance_tint: float = Field(default=0.0, description="Green-magenta tint")
        contrast_ratio: float = Field(ge=0.0, default=1.0)
        dynamic_range: float = Field(ge=0.0, default=0.0)
        lut_suggestion: Optional[str] = None
        color_intent: Optional[ColorGradingIntent] = None
        consistency_score: float = Field(ge=0.0, le=1.0, default=1.0)
        
        @property
        def is_warm(self) -> bool:
            return self.white_balance_temp < 5500
        
        @property
        def is_cool(self) -> bool:
            return self.white_balance_temp > 7000
        
        def to_hex(self, rgb: Tuple[int, int, int]) -> str:
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


    class StabilizationResult(BaseModel):
        """Video stabilization analysis result"""
        shot_id: str
        shake_score: float = Field(ge=0.0, le=1.0, description="0=stable, 1=extreme shake")
        shake_severity: str = "none"
        motion_vectors_avg: Tuple[float, float] = (0.0, 0.0)
        motion_variance: float = Field(ge=0.0, default=0.0)
        crop_factor: float = Field(ge=1.0, le=2.0, default=1.0)
        stabilization_needed: bool = False
        recommended_method: StabilizationMethod = StabilizationMethod.NONE
        confidence: float = Field(ge=0.0, le=1.0, default=0.8)
        estimated_quality_loss: float = Field(ge=0.0, le=1.0, default=0.0)
        ffmpeg_filter: Optional[str] = None
        
        @validator('shake_severity', always=True)
        def set_severity(cls, v, values):
            if 'shake_score' in values:
                score = values['shake_score']
                if score < SHAKE_THRESHOLD_LOW:
                    return "none"
                elif score < SHAKE_THRESHOLD_MEDIUM:
                    return "low"
                elif score < SHAKE_THRESHOLD_HIGH:
                    return "medium"
                else:
                    return "high"
            return v
        
        def get_ffmpeg_stabilize_filter(self) -> str:
            """Generate FFmpeg vidstab filter"""
            if not self.stabilization_needed:
                return ""
            
            smoothing = int(10 + self.shake_score * 20)
            zoom = (self.crop_factor - 1.0) * 100
            
            return f"vidstabdetect=shakiness=10:accuracy=15,vidstabtransform=smoothing={smoothing}:zoom={zoom:.1f}:interpol=bicubic"


    class EditingGrammarRule(BaseModel):
        """Editing grammar rule for transition selection"""
        rule_id: str
        name: str
        description: str
        condition: str  # Evaluated condition
        action: TransitionType
        priority: int = Field(ge=1, le=100, default=50)
        applicable_styles: List[StylePreset] = Field(default_factory=list)
        min_shot_duration: Optional[float] = None
        max_shot_duration: Optional[float] = None
        context_tags: List[str] = Field(default_factory=list)
        enabled: bool = True
        
        class Config:
            use_enum_values = True


    class PacingAnalysis(BaseModel):
        """Video pacing and rhythm analysis"""
        video_id: str
        total_shots: int = Field(ge=0)
        total_duration_seconds: float = Field(ge=0.0)
        shots_per_minute: float = Field(ge=0.0)
        avg_shot_duration_seconds: float = Field(ge=0.0)
        min_shot_duration_seconds: float = Field(ge=0.0)
        max_shot_duration_seconds: float = Field(ge=0.0)
        shot_duration_variance: float = Field(ge=0.0)
        rhythm_score: float = Field(ge=0.0, le=1.0, default=0.5)
        rhythm_pattern: RhythmPattern = RhythmPattern.CONSTANT
        energy_curve: List[float] = Field(default_factory=list)
        beat_alignment_score: Optional[float] = None
        pacing_recommendation: str = ""
        
        @property
        def is_fast_paced(self) -> bool:
            return self.shots_per_minute > 30
        
        @property
        def is_slow_paced(self) -> bool:
            return self.shots_per_minute < 10


    class TimelineSegment(BaseModel):
        """Timeline segment for video assembly"""
        segment_id: str
        shot_id: str
        position: int = Field(ge=0, description="Position in timeline")
        in_point_frames: int = Field(ge=0)
        out_point_frames: int = Field(ge=0)
        in_point_seconds: float = Field(ge=0.0)
        out_point_seconds: float = Field(ge=0.0)
        duration_frames: int = Field(ge=1)
        duration_seconds: float = Field(ge=0.0)
        transition_in: Optional[TransitionRecommendation] = None
        transition_out: Optional[TransitionRecommendation] = None
        speed_factor: float = Field(ge=0.1, le=10.0, default=1.0)
        audio_level_db: float = Field(ge=-60.0, le=12.0, default=0.0)
        video_filters: List[str] = Field(default_factory=list)
        audio_filters: List[str] = Field(default_factory=list)


    class AssemblyCommand(BaseModel):
        """FFmpeg assembly command"""
        command_id: str
        operation: str  # concat, filter, encode, etc.
        input_files: List[str] = Field(default_factory=list)
        output_file: str
        ffmpeg_args: List[str] = Field(default_factory=list)
        filter_complex: Optional[str] = None
        estimated_duration_seconds: float = Field(ge=0.0, default=0.0)
        
        def to_command_string(self) -> str:
            """Generate full FFmpeg command string"""
            cmd = ["ffmpeg"]
            
            for input_file in self.input_files:
                cmd.extend(["-i", input_file])
            
            if self.filter_complex:
                cmd.extend(["-filter_complex", self.filter_complex])
            
            cmd.extend(self.ffmpeg_args)
            cmd.extend(["-y", self.output_file])
            
            return " ".join(cmd)


    class EditingIssue(BaseModel):
        """Detected editing issue or recommendation"""
        issue_id: str
        severity: EditingSeverity
        category: str
        message: str
        shot_id: Optional[str] = None
        frame_number: Optional[int] = None
        timestamp_seconds: Optional[float] = None
        suggestion: Optional[str] = None
        auto_fixable: bool = False


    class EditValidationRequest(BaseModel):
        """Request for edit validation and assembly"""
        video_path: str
        style_preset: StylePreset = StylePreset.COMMERCIAL
        target_duration: Optional[float] = None
        music_path: Optional[str] = None
        color_intent: ColorGradingIntent = ColorGradingIntent.NEUTRAL
        enable_stabilization: bool = True
        shot_detection_threshold: float = Field(ge=0.1, le=0.9, default=0.5)
        min_shot_duration: float = Field(ge=0.1, default=0.2)
        max_shot_duration: float = Field(ge=1.0, default=30.0)
        generate_ffmpeg_commands: bool = True
        output_format: str = "mp4"
        custom_rules: List[EditingGrammarRule] = Field(default_factory=list)
        
        @validator('video_path')
        def validate_video_path(cls, v):
            if not v:
                raise ValueError("video_path cannot be empty")
            return v


    class EditValidationResult(BaseModel):
        """Complete edit validation and assembly result"""
        success: bool
        video_path: str
        video_metadata: Optional[VideoMetadata] = None
        shots: List[ShotBoundary] = Field(default_factory=list)
        transitions: List[TransitionRecommendation] = Field(default_factory=list)
        color_analyses: List[ColorAnalysis] = Field(default_factory=list)
        stabilization_results: List[StabilizationResult] = Field(default_factory=list)
        pacing: Optional[PacingAnalysis] = None
        timeline: List[TimelineSegment] = Field(default_factory=list)
        assembly_commands: List[AssemblyCommand] = Field(default_factory=list)
        issues: List[EditingIssue] = Field(default_factory=list)
        report_markdown: str = ""
        processing_time_seconds: float = Field(ge=0.0, default=0.0)
        cost_estimate: float = Field(ge=0.0, default=0.0)
        
        @property
        def shot_count(self) -> int:
            return len(self.shots)
        
        @property
        def has_critical_issues(self) -> bool:
            return any(i.severity == EditingSeverity.CRITICAL for i in self.issues)
        
        @property
        def error_count(self) -> int:
            return sum(1 for i in self.issues if i.severity in (EditingSeverity.ERROR, EditingSeverity.CRITICAL))


# =============================================================================
# SECTION 4: BASE DICTIONARIES & RULE DEFINITIONS
# =============================================================================

EDITING_GRAMMAR_RULES: Dict[str, Dict] = {
    "rule_action_cut": {
        "rule_id": "rule_action_cut",
        "name": "Action Continuity Cut",
        "description": "Use hard cut for continuous action sequences",
        "condition": "motion_type in ['handheld', 'tracking'] and shot_duration < 3.0",
        "action": TransitionType.HARD_CUT,
        "priority": 90,
        "applicable_styles": [StylePreset.COMMERCIAL, StylePreset.MUSIC_VIDEO, StylePreset.ENERGETIC],
        "context_tags": ["action", "fast", "dynamic"]
    },
    "rule_dialogue_cut": {
        "rule_id": "rule_dialogue_cut",
        "name": "Dialogue Exchange Cut",
        "description": "Use hard cut for dialogue exchanges",
        "condition": "is_dialogue_scene and shot_type in ['close_up', 'medium_close', 'over_shoulder']",
        "action": TransitionType.HARD_CUT,
        "priority": 85,
        "applicable_styles": [StylePreset.CORPORATE, StylePreset.DOCUMENTARY, StylePreset.TESTIMONIAL],
        "context_tags": ["dialogue", "interview", "conversation"]
    },
    "rule_scene_dissolve": {
        "rule_id": "rule_scene_dissolve",
        "name": "Scene Change Dissolve",
        "description": "Use dissolve for scene/location changes",
        "condition": "is_scene_change and not is_action_sequence",
        "action": TransitionType.DISSOLVE,
        "priority": 80,
        "applicable_styles": [StylePreset.CINEMATIC, StylePreset.DOCUMENTARY, StylePreset.CALM],
        "context_tags": ["scene_change", "location", "time_passage"]
    },
    "rule_time_passage": {
        "rule_id": "rule_time_passage",
        "name": "Time Passage Dissolve",
        "description": "Use cross dissolve for time passage",
        "condition": "time_gap_detected",
        "action": TransitionType.CROSS_DISSOLVE,
        "priority": 75,
        "applicable_styles": [StylePreset.CINEMATIC, StylePreset.DOCUMENTARY],
        "context_tags": ["time_passage", "montage", "dream"]
    },
    "rule_scene_end_fade": {
        "rule_id": "rule_scene_end_fade",
        "name": "Scene End Fade",
        "description": "Use fade out for scene endings",
        "condition": "is_scene_end and next_scene_is_different",
        "action": TransitionType.FADE_OUT,
        "priority": 70,
        "applicable_styles": [StylePreset.CINEMATIC, StylePreset.DRAMATIC],
        "context_tags": ["scene_end", "emotional", "dramatic"]
    },
    "rule_emotional_beat": {
        "rule_id": "rule_emotional_beat",
        "name": "Emotional Beat Fade",
        "description": "Use dip to black for emotional moments",
        "condition": "emotional_peak_detected",
        "action": TransitionType.DIP_TO_BLACK,
        "priority": 65,
        "applicable_styles": [StylePreset.DRAMATIC, StylePreset.CINEMATIC],
        "context_tags": ["emotional", "dramatic", "pause"]
    },
    "rule_location_wipe": {
        "rule_id": "rule_location_wipe",
        "name": "Location Change Wipe",
        "description": "Use wipe for quick location changes",
        "condition": "location_change and style in ['retro', 'music_video', 'social_media']",
        "action": TransitionType.WIPE_LEFT,
        "priority": 60,
        "applicable_styles": [StylePreset.RETRO, StylePreset.MUSIC_VIDEO, StylePreset.SOCIAL_MEDIA],
        "context_tags": ["location", "retro", "comic"]
    },
    "rule_product_reveal": {
        "rule_id": "rule_product_reveal",
        "name": "Product Reveal Cut",
        "description": "Use hard cut for product reveals in commercials",
        "condition": "is_product_shot and style == 'commercial'",
        "action": TransitionType.HARD_CUT,
        "priority": 95,
        "applicable_styles": [StylePreset.COMMERCIAL, StylePreset.PRODUCT_DEMO],
        "context_tags": ["product", "reveal", "hero_shot"]
    },
    "rule_music_beat": {
        "rule_id": "rule_music_beat",
        "name": "Music Beat Cut",
        "description": "Align cuts with music beats",
        "condition": "music_beat_detected and beat_strength > 0.7",
        "action": TransitionType.HARD_CUT,
        "priority": 88,
        "applicable_styles": [StylePreset.MUSIC_VIDEO, StylePreset.COMMERCIAL, StylePreset.ENERGETIC],
        "context_tags": ["music", "beat", "rhythm"]
    },
    "rule_match_cut": {
        "rule_id": "rule_match_cut",
        "name": "Match Cut",
        "description": "Use match cut for visual continuity",
        "condition": "visual_similarity > 0.8 and motion_continuity",
        "action": TransitionType.MATCH_CUT,
        "priority": 72,
        "applicable_styles": [StylePreset.CINEMATIC, StylePreset.FILM_TRAILER],
        "context_tags": ["match", "visual_continuity", "creative"]
    },
    "rule_default_cut": {
        "rule_id": "rule_default_cut",
        "name": "Default Hard Cut",
        "description": "Default to hard cut when no specific rule applies",
        "condition": "True",
        "action": TransitionType.HARD_CUT,
        "priority": 1,
        "applicable_styles": [],
        "context_tags": ["default"]
    }
}

STYLE_PRESETS: Dict[str, Dict] = {
    StylePreset.COMMERCIAL: {
        "avg_shot_duration": 2.5,
        "min_shot_duration": 0.5,
        "max_shot_duration": 8.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.DISSOLVE],
        "pacing": "medium-fast",
        "color_intent": ColorGradingIntent.VIBRANT,
        "stabilization_preference": "high"
    },
    StylePreset.CINEMATIC: {
        "avg_shot_duration": 4.0,
        "min_shot_duration": 1.5,
        "max_shot_duration": 15.0,
        "preferred_transitions": [TransitionType.DISSOLVE, TransitionType.FADE_OUT, TransitionType.MATCH_CUT],
        "pacing": "slow-medium",
        "color_intent": ColorGradingIntent.CINEMATIC,
        "stabilization_preference": "selective"
    },
    StylePreset.MUSIC_VIDEO: {
        "avg_shot_duration": 1.5,
        "min_shot_duration": 0.3,
        "max_shot_duration": 5.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.WIPE_LEFT, TransitionType.FLASH],
        "pacing": "fast",
        "color_intent": ColorGradingIntent.VIBRANT,
        "stabilization_preference": "none"
    },
    StylePreset.DOCUMENTARY: {
        "avg_shot_duration": 5.0,
        "min_shot_duration": 2.0,
        "max_shot_duration": 20.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.DISSOLVE],
        "pacing": "slow",
        "color_intent": ColorGradingIntent.NEUTRAL,
        "stabilization_preference": "high"
    },
    StylePreset.CORPORATE: {
        "avg_shot_duration": 4.0,
        "min_shot_duration": 1.5,
        "max_shot_duration": 12.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.DISSOLVE],
        "pacing": "medium",
        "color_intent": ColorGradingIntent.BROADCAST_SAFE,
        "stabilization_preference": "high"
    },
    StylePreset.SOCIAL_MEDIA: {
        "avg_shot_duration": 1.8,
        "min_shot_duration": 0.5,
        "max_shot_duration": 6.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.ZOOM, TransitionType.WIPE_LEFT],
        "pacing": "fast",
        "color_intent": ColorGradingIntent.VIBRANT,
        "stabilization_preference": "medium"
    },
    StylePreset.ENERGETIC: {
        "avg_shot_duration": 1.2,
        "min_shot_duration": 0.3,
        "max_shot_duration": 4.0,
        "preferred_transitions": [TransitionType.HARD_CUT, TransitionType.FLASH],
        "pacing": "very_fast",
        "color_intent": ColorGradingIntent.HIGH_CONTRAST,
        "stabilization_preference": "none"
    },
    StylePreset.CALM: {
        "avg_shot_duration": 6.0,
        "min_shot_duration": 2.5,
        "max_shot_duration": 20.0,
        "preferred_transitions": [TransitionType.DISSOLVE, TransitionType.CROSS_DISSOLVE, TransitionType.FADE_OUT],
        "pacing": "slow",
        "color_intent": ColorGradingIntent.DESATURATED,
        "stabilization_preference": "high"
    },
    StylePreset.DRAMATIC: {
        "avg_shot_duration": 3.5,
        "min_shot_duration": 1.0,
        "max_shot_duration": 10.0,
        "preferred_transitions": [TransitionType.DIP_TO_BLACK, TransitionType.FADE_OUT, TransitionType.HARD_CUT],
        "pacing": "varied",
        "color_intent": ColorGradingIntent.HIGH_CONTRAST,
        "stabilization_preference": "selective"
    }
}

TRANSITION_DURATIONS: Dict[TransitionType, Dict[str, float]] = {
    TransitionType.HARD_CUT: {"default": 0.0, "min": 0.0, "max": 0.0},
    TransitionType.DISSOLVE: {"default": 0.5, "min": 0.2, "max": 2.0},
    TransitionType.CROSS_DISSOLVE: {"default": 0.7, "min": 0.3, "max": 2.5},
    TransitionType.FADE_IN: {"default": 0.5, "min": 0.2, "max": 3.0},
    TransitionType.FADE_OUT: {"default": 0.5, "min": 0.2, "max": 3.0},
    TransitionType.WIPE_LEFT: {"default": 0.4, "min": 0.2, "max": 1.5},
    TransitionType.WIPE_RIGHT: {"default": 0.4, "min": 0.2, "max": 1.5},
    TransitionType.DIP_TO_BLACK: {"default": 0.8, "min": 0.3, "max": 2.0},
    TransitionType.DIP_TO_WHITE: {"default": 0.6, "min": 0.3, "max": 1.5},
    TransitionType.FLASH: {"default": 0.15, "min": 0.05, "max": 0.5},
    TransitionType.ZOOM: {"default": 0.3, "min": 0.1, "max": 1.0},
}

LUT_RECOMMENDATIONS: Dict[ColorGradingIntent, Dict] = {
    ColorGradingIntent.NEUTRAL: {
        "lut_name": "neutral_rec709",
        "description": "Standard Rec.709 color space",
        "saturation_adjust": 0,
        "contrast_adjust": 0
    },
    ColorGradingIntent.WARM: {
        "lut_name": "warm_golden",
        "description": "Warm golden tones",
        "saturation_adjust": 5,
        "contrast_adjust": 5,
        "temp_shift": -500
    },
    ColorGradingIntent.COOL: {
        "lut_name": "cool_blue",
        "description": "Cool blue tones",
        "saturation_adjust": -5,
        "contrast_adjust": 10,
        "temp_shift": 500
    },
    ColorGradingIntent.HIGH_CONTRAST: {
        "lut_name": "high_contrast",
        "description": "Punchy high contrast look",
        "saturation_adjust": 10,
        "contrast_adjust": 25
    },
    ColorGradingIntent.CINEMATIC: {
        "lut_name": "cinematic_film",
        "description": "Film emulation with lifted blacks",
        "saturation_adjust": -10,
        "contrast_adjust": 15,
        "black_lift": 5
    },
    ColorGradingIntent.TEAL_ORANGE: {
        "lut_name": "teal_orange",
        "description": "Hollywood blockbuster look",
        "saturation_adjust": 15,
        "contrast_adjust": 10
    },
    ColorGradingIntent.VINTAGE: {
        "lut_name": "vintage_film",
        "description": "Faded vintage film look",
        "saturation_adjust": -20,
        "contrast_adjust": -10,
        "black_lift": 15
    },
    ColorGradingIntent.VIBRANT: {
        "lut_name": "vibrant_pop",
        "description": "Punchy vibrant colors",
        "saturation_adjust": 25,
        "contrast_adjust": 15
    },
    ColorGradingIntent.BROADCAST_SAFE: {
        "lut_name": "broadcast_safe",
        "description": "Broadcast-safe color limits",
        "saturation_adjust": -5,
        "contrast_adjust": 0
    }
}


# =============================================================================
# SECTION 5: CORE EDITOR CLASS
# =============================================================================

class TheEditor:
    """
    THE EDITOR (Agent 7.5) - Shot Detection & Assembly Agent
    
    Main orchestrator for video editing analysis including:
    - Shot boundary detection with TransNet V2 hooks
    - Editing grammar-based transition selection
    - Color grading analysis
    - Stabilization analysis
    - Timeline assembly generation
    """
    
    def __init__(
        self,
        device: str = "cpu",
        transnet_threshold: float = TRANSNET_THRESHOLD,
        enable_gpu: bool = False,
        cache_enabled: bool = True,
        max_cache_size: int = 100
    ):
        """
        Initialize THE EDITOR agent.
        
        Args:
            device: Compute device ('cpu', 'cuda', 'mps')
            transnet_threshold: Shot detection confidence threshold
            enable_gpu: Enable GPU acceleration
            cache_enabled: Enable result caching
            max_cache_size: Maximum cache entries
        """
        self.device = device
        self.transnet_threshold = transnet_threshold
        self.enable_gpu = enable_gpu and self._check_gpu_available()
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        
        # Initialize caches
        self._shot_cache: Dict[str, List[ShotBoundary]] = {}
        self._color_cache: Dict[str, ColorAnalysis] = {}
        self._analysis_cache: Dict[str, EditValidationResult] = {}
        
        # Load editing grammar rules
        self.grammar_rules = self._load_grammar_rules()
        
        # TransNet V2 model (lazy loaded)
        self._transnet_model = None
        
        logger.info(f"THE EDITOR initialized | device={device} | GPU={self.enable_gpu}")
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_grammar_rules(self) -> List[EditingGrammarRule]:
        """Load and parse editing grammar rules"""
        rules = []
        for rule_id, rule_data in EDITING_GRAMMAR_RULES.items():
            try:
                rule = EditingGrammarRule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    condition=rule_data["condition"],
                    action=rule_data["action"],
                    priority=rule_data["priority"],
                    applicable_styles=[StylePreset(s) for s in rule_data.get("applicable_styles", [])],
                    context_tags=rule_data.get("context_tags", [])
                )
                rules.append(rule)
            except Exception as e:
                logger.warning(f"Failed to load rule {rule_id}: {e}")
        
        # Sort by priority (highest first)
        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules
    
    # -------------------------------------------------------------------------
    # VIDEO METADATA EXTRACTION
    # -------------------------------------------------------------------------
    
    async def extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object with file information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        try:
            # Use FFprobe to get video info
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFprobe failed: {result.stderr}")
            
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            audio_tracks = 0
            
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio":
                    audio_tracks += 1
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Parse frame rate
            fps_str = video_stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            
            # Get total frames
            total_frames = int(video_stream.get("nb_frames", 0))
            if total_frames == 0:
                # Estimate from duration
                duration = float(probe_data.get("format", {}).get("duration", 0))
                total_frames = int(duration * fps)
            
            # Get duration
            duration = float(video_stream.get("duration", 0))
            if duration == 0:
                duration = float(probe_data.get("format", {}).get("duration", 0))
            
            # Get bitrate
            bitrate = int(probe_data.get("format", {}).get("bit_rate", 0)) // 1000
            
            return VideoMetadata(
                file_path=video_path,
                width=int(video_stream.get("width", 1920)),
                height=int(video_stream.get("height", 1080)),
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                codec=video_stream.get("codec_name", "unknown"),
                bitrate_kbps=bitrate if bitrate > 0 else None,
                audio_tracks=audio_tracks,
                has_audio=audio_tracks > 0,
                color_space=video_stream.get("color_space", "unknown"),
                bit_depth=int(video_stream.get("bits_per_raw_sample", 8))
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFprobe timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse FFprobe output: {e}")
    
    # -------------------------------------------------------------------------
    # SHOT BOUNDARY DETECTION
    # -------------------------------------------------------------------------
    
    async def detect_shots(
        self,
        video_path: str,
        confidence_threshold: Optional[float] = None,
        sample_rate: int = 1,
        metadata: Optional[VideoMetadata] = None
    ) -> List[ShotBoundary]:
        """
        Detect shot boundaries in video.
        
        Uses heuristic-based detection with TransNet V2 integration hooks
        for future neural model integration.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Detection threshold (default: self.transnet_threshold)
            sample_rate: Process every N frames
            metadata: Pre-computed video metadata
            
        Returns:
            List of detected shot boundaries
        """
        threshold = confidence_threshold or self.transnet_threshold
        
        # Check cache
        cache_key = f"{video_path}_{threshold}_{sample_rate}"
        if self.cache_enabled and cache_key in self._shot_cache:
            logger.info(f"Shot cache hit: {cache_key}")
            return self._shot_cache[cache_key]
        
        # Get metadata if not provided
        if metadata is None:
            metadata = await self.extract_video_metadata(video_path)
        
        logger.info(f"Detecting shots in {video_path} | threshold={threshold}")
        
        boundaries = []
        
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            boundaries = await self._detect_shots_opencv(
                video_path, metadata, threshold, sample_rate
            )
        else:
            # Fallback: FFmpeg scene detection
            boundaries = await self._detect_shots_ffmpeg(
                video_path, metadata, threshold
            )
        
        # Cache result
        if self.cache_enabled:
            self._shot_cache[cache_key] = boundaries
            self._trim_cache(self._shot_cache)
        
        logger.info(f"Detected {len(boundaries)} shots")
        return boundaries
    
    async def _detect_shots_opencv(
        self,
        video_path: str,
        metadata: VideoMetadata,
        threshold: float,
        sample_rate: int
    ) -> List[ShotBoundary]:
        """OpenCV-based shot detection with histogram comparison"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        boundaries = []
        prev_hist = None
        shot_start_frame = 0
        shot_id = 0
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue
                
                # Convert to grayscale and compute histogram
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # Compare histograms
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    
                    # Detect shot boundary if difference exceeds threshold
                    if diff > threshold:
                        # Determine transition type based on difference pattern
                        transition_type = self._classify_transition_opencv(
                            prev_hist, hist, diff
                        )
                        
                        # End previous shot
                        if shot_id > 0 or frame_idx > sample_rate:
                            timestamp_start = shot_start_frame / metadata.fps
                            timestamp_end = (frame_idx - 1) / metadata.fps
                            duration = timestamp_end - timestamp_start
                            
                            boundary = ShotBoundary(
                                shot_id=f"shot_{shot_id:04d}",
                                frame_start=shot_start_frame,
                                frame_end=frame_idx - 1,
                                timestamp_start=timestamp_start,
                                timestamp_end=timestamp_end,
                                transition_type=transition_type,
                                confidence=min(diff / threshold, 1.0),
                                duration_frames=frame_idx - shot_start_frame,
                                duration_seconds=duration
                            )
                            boundaries.append(boundary)
                        
                        shot_start_frame = frame_idx
                        shot_id += 1
                
                prev_hist = hist
                frame_idx += 1
            
            # Add final shot
            if frame_idx > shot_start_frame:
                timestamp_start = shot_start_frame / metadata.fps
                timestamp_end = metadata.duration_seconds
                
                boundary = ShotBoundary(
                    shot_id=f"shot_{shot_id:04d}",
                    frame_start=shot_start_frame,
                    frame_end=frame_idx - 1,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                    transition_type=TransitionType.UNKNOWN,
                    confidence=0.5,
                    duration_frames=frame_idx - shot_start_frame,
                    duration_seconds=timestamp_end - timestamp_start
                )
                boundaries.append(boundary)
        
        finally:
            cap.release()
        
        return boundaries
    
    async def _detect_shots_ffmpeg(
        self,
        video_path: str,
        metadata: VideoMetadata,
        threshold: float
    ) -> List[ShotBoundary]:
        """FFmpeg-based scene detection fallback"""
        
        # Convert threshold to FFmpeg scene detection threshold
        ffmpeg_threshold = 0.3 + (1 - threshold) * 0.4
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-filter:v", f"select='gt(scene,{ffmpeg_threshold})',showinfo",
            "-f", "null",
            "-"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            # Parse scene changes from output
            scene_times = []
            for line in result.stderr.split('\n'):
                if 'pts_time' in line:
                    try:
                        pts_time = float(line.split('pts_time:')[1].split()[0])
                        scene_times.append(pts_time)
                    except (IndexError, ValueError):
                        pass
            
            # Create boundaries from scene times
            boundaries = []
            prev_time = 0.0
            
            for i, scene_time in enumerate(scene_times):
                boundary = ShotBoundary(
                    shot_id=f"shot_{i:04d}",
                    frame_start=int(prev_time * metadata.fps),
                    frame_end=int(scene_time * metadata.fps),
                    timestamp_start=prev_time,
                    timestamp_end=scene_time,
                    transition_type=TransitionType.HARD_CUT,
                    confidence=0.7,
                    duration_frames=int((scene_time - prev_time) * metadata.fps),
                    duration_seconds=scene_time - prev_time
                )
                boundaries.append(boundary)
                prev_time = scene_time
            
            # Add final shot
            if prev_time < metadata.duration_seconds:
                boundary = ShotBoundary(
                    shot_id=f"shot_{len(boundaries):04d}",
                    frame_start=int(prev_time * metadata.fps),
                    frame_end=metadata.total_frames - 1,
                    timestamp_start=prev_time,
                    timestamp_end=metadata.duration_seconds,
                    transition_type=TransitionType.UNKNOWN,
                    confidence=0.5,
                    duration_frames=metadata.total_frames - int(prev_time * metadata.fps),
                    duration_seconds=metadata.duration_seconds - prev_time
                )
                boundaries.append(boundary)
            
            return boundaries
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg scene detection timed out")
            return []
    
    def _classify_transition_opencv(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray,
        diff: float
    ) -> TransitionType:
        """Classify transition type from histogram analysis"""
        
        # High difference = hard cut
        if diff > 0.7:
            return TransitionType.HARD_CUT
        
        # Check for fade (histogram shift to extremes)
        mean1 = np.mean(hist1)
        mean2 = np.mean(hist2)
        
        if mean2 < mean1 * 0.3:
            return TransitionType.FADE_OUT
        elif mean2 > mean1 * 3.0:
            return TransitionType.FADE_IN
        
        # Moderate difference = dissolve
        if diff > 0.4:
            return TransitionType.DISSOLVE
        
        return TransitionType.GRADUAL
    
    # -------------------------------------------------------------------------
    # TRANSITION RECOMMENDATION
    # -------------------------------------------------------------------------
    
    async def recommend_transitions(
        self,
        shots: List[ShotBoundary],
        style: StylePreset = StylePreset.COMMERCIAL,
        custom_rules: Optional[List[EditingGrammarRule]] = None
    ) -> List[TransitionRecommendation]:
        """
        Recommend transitions between shots based on editing grammar.
        
        Args:
            shots: List of detected shots
            style: Editing style preset
            custom_rules: Additional custom rules
            
        Returns:
            List of transition recommendations
        """
        if len(shots) < 2:
            return []
        
        # Combine default and custom rules
        rules = self.grammar_rules.copy()
        if custom_rules:
            rules.extend(custom_rules)
            rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Filter rules by style
        style_rules = [
            r for r in rules 
            if not r.applicable_styles or style in r.applicable_styles
        ]
        
        transitions = []
        style_config = STYLE_PRESETS.get(style, STYLE_PRESETS[StylePreset.COMMERCIAL])
        
        for i in range(len(shots) - 1):
            current_shot = shots[i]
            next_shot = shots[i + 1]
            
            # Build context for rule evaluation
            context = {
                "current_shot": current_shot,
                "next_shot": next_shot,
                "style": style,
                "shot_duration": current_shot.duration_seconds,
                "motion_type": current_shot.motion_type,
                "shot_type": current_shot.shot_type,
                "is_scene_change": self._detect_scene_change(current_shot, next_shot),
                "is_dialogue_scene": False,  # Would need audio analysis
                "is_action_sequence": current_shot.motion_type in [MotionType.HANDHELD, MotionType.STEADICAM],
                "is_product_shot": False,  # Would need object detection
                "emotional_peak_detected": False,
                "time_gap_detected": False,
                "is_scene_end": i == len(shots) - 2,
                "next_scene_is_different": self._detect_scene_change(current_shot, next_shot),
                "location_change": False,
                "visual_similarity": 0.5,
                "motion_continuity": True
            }
            
            # Find matching rule
            selected_rule = None
            for rule in style_rules:
                if rule.enabled and self._evaluate_rule_condition(rule.condition, context):
                    selected_rule = rule
                    break
            
            # Default to hard cut if no rule matches
            if selected_rule is None:
                selected_rule = EditingGrammarRule(
                    rule_id="default",
                    name="Default",
                    description="Default hard cut",
                    condition="True",
                    action=TransitionType.HARD_CUT,
                    priority=0
                )
            
            # Get transition duration
            transition_type = selected_rule.action
            duration_config = TRANSITION_DURATIONS.get(
                transition_type, 
                {"default": 0.0, "min": 0.0, "max": 1.0}
            )
            duration_seconds = duration_config["default"]
            
            # Adjust duration based on style pacing
            if style_config.get("pacing") == "fast":
                duration_seconds = max(duration_config["min"], duration_seconds * 0.7)
            elif style_config.get("pacing") == "slow":
                duration_seconds = min(duration_config["max"], duration_seconds * 1.3)
            
            # Generate alternatives
            alternatives = self._get_transition_alternatives(
                transition_type,
                style_config.get("preferred_transitions", [])
            )
            
            transition = TransitionRecommendation(
                transition_id=f"trans_{i:04d}",
                from_shot_id=current_shot.shot_id,
                to_shot_id=next_shot.shot_id,
                transition_type=transition_type,
                duration_frames=max(1, int(duration_seconds * DEFAULT_FPS)),  # Ensure minimum 1 frame
                duration_seconds=max(0.033, duration_seconds),  # Ensure minimum ~1 frame at 30fps
                reasoning=f"Applied rule: {selected_rule.name}. {selected_rule.description}",
                alternatives=alternatives,
                confidence=0.8 if selected_rule.rule_id != "default" else 0.5,
                grammar_rule_applied=selected_rule.rule_id
            )
            transitions.append(transition)
        
        logger.info(f"Generated {len(transitions)} transition recommendations")
        return transitions
    
    def _detect_scene_change(
        self,
        shot1: ShotBoundary,
        shot2: ShotBoundary
    ) -> bool:
        """Detect if shots are from different scenes"""
        # Check color difference if available
        if shot1.dominant_color and shot2.dominant_color:
            color_diff = sum(
                abs(c1 - c2) for c1, c2 in zip(shot1.dominant_color, shot2.dominant_color)
            ) / 3
            if color_diff > 50:
                return True
        
        # Check brightness difference
        if shot1.avg_brightness and shot2.avg_brightness:
            brightness_diff = abs(shot1.avg_brightness - shot2.avg_brightness)
            if brightness_diff > 40:
                return True
        
        return False
    
    def _evaluate_rule_condition(self, condition: str, context: Dict) -> bool:
        """Safely evaluate rule condition"""
        try:
            # Simple condition evaluation (in production, use AST parsing)
            if condition == "True":
                return True
            
            # Replace context variables
            for key, value in context.items():
                if isinstance(value, bool):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, (int, float)):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, str):
                    condition = condition.replace(key, f"'{value}'")
            
            # Safe evaluation of simple boolean expressions
            safe_condition = condition.replace("and", " and ").replace("or", " or ")
            
            # Only evaluate simple conditions
            if all(c in "True False and or not () < > <= >= == != 0123456789. '" for c in safe_condition):
                return eval(safe_condition)
            
            return False
        except Exception:
            return False
    
    def _get_transition_alternatives(
        self,
        primary: TransitionType,
        preferred: List[TransitionType]
    ) -> List[TransitionType]:
        """Get alternative transitions"""
        alternatives = []
        
        # Add preferred alternatives
        for trans in preferred:
            if trans != primary and len(alternatives) < 3:
                alternatives.append(trans)
        
        # Add common alternatives based on primary
        fallbacks = {
            TransitionType.HARD_CUT: [TransitionType.DISSOLVE, TransitionType.DIP_TO_BLACK],
            TransitionType.DISSOLVE: [TransitionType.CROSS_DISSOLVE, TransitionType.HARD_CUT],
            TransitionType.FADE_OUT: [TransitionType.DIP_TO_BLACK, TransitionType.DISSOLVE],
            TransitionType.WIPE_LEFT: [TransitionType.WIPE_RIGHT, TransitionType.SLIDE]
        }
        
        for alt in fallbacks.get(primary, []):
            if alt not in alternatives and len(alternatives) < 3:
                alternatives.append(alt)
        
        return alternatives
    
    # -------------------------------------------------------------------------
    # COLOR GRADING ANALYSIS
    # -------------------------------------------------------------------------
    
    async def analyze_colors(
        self,
        video_path: str,
        shots: List[ShotBoundary],
        intent: ColorGradingIntent = ColorGradingIntent.NEUTRAL
    ) -> List[ColorAnalysis]:
        """
        Analyze color characteristics for each shot.
        
        Args:
            video_path: Path to video file
            shots: List of detected shots
            intent: Color grading creative intent
            
        Returns:
            List of color analysis results
        """
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV/NumPy not available, skipping color analysis")
            return []
        
        logger.info(f"Analyzing colors for {len(shots)} shots")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        analyses = []
        
        try:
            for shot in shots:
                # Seek to middle of shot for representative frame
                middle_frame = (shot.frame_start + shot.frame_end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                analysis = await self._analyze_frame_colors(
                    shot.shot_id, frame, intent
                )
                analyses.append(analysis)
        
        finally:
            cap.release()
        
        # Calculate consistency scores between adjacent shots
        for i in range(1, len(analyses)):
            analyses[i].consistency_score = self._calculate_color_consistency(
                analyses[i-1], analyses[i]
            )
        
        return analyses
    
    async def _analyze_frame_colors(
        self,
        shot_id: str,
        frame: np.ndarray,
        intent: ColorGradingIntent
    ) -> ColorAnalysis:
        """Analyze colors in a single frame"""
        
        # Convert color spaces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histograms
        hist_r = cv2.calcHist([rgb_frame], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_g = cv2.calcHist([rgb_frame], [1], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_b = cv2.calcHist([rgb_frame], [2], None, [256], [0, 256]).flatten().astype(int).tolist()
        
        # Average brightness and saturation
        avg_brightness = float(np.mean(hsv_frame[:, :, 2]))
        avg_saturation = float(np.mean(hsv_frame[:, :, 1]) / 255.0)
        
        # Extract dominant colors
        dominant_colors = []
        color_palette = []
        
        if SKLEARN_AVAILABLE:
            pixels = rgb_frame.reshape(-1, 3)
            # Sample for performance
            sample_size = min(10000, len(pixels))
            sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
            sample_pixels = pixels[sample_indices]
            
            kmeans = KMeans(n_clusters=COLOR_CLUSTERS, n_init=10, random_state=42)
            kmeans.fit(sample_pixels)
            
            for center in kmeans.cluster_centers_:
                rgb = tuple(int(c) for c in center)
                dominant_colors.append(rgb)
                color_palette.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
        
        # Estimate white balance (simplified)
        avg_r = np.mean(rgb_frame[:, :, 0])
        avg_g = np.mean(rgb_frame[:, :, 1])
        avg_b = np.mean(rgb_frame[:, :, 2])
        
        # Rough color temperature estimation
        if avg_b > avg_r * 1.2:
            temp = 8000  # Cool
        elif avg_r > avg_b * 1.2:
            temp = 4500  # Warm
        else:
            temp = 6500  # Neutral
        
        tint = (avg_g - (avg_r + avg_b) / 2) / 128  # Green-magenta shift
        
        # Contrast ratio (simplified)
        min_val = np.percentile(hsv_frame[:, :, 2], 5)
        max_val = np.percentile(hsv_frame[:, :, 2], 95)
        contrast = (max_val - min_val) / 255 if max_val > min_val else 0
        
        # Dynamic range
        dynamic_range = float(np.std(hsv_frame[:, :, 2]))
        
        # Get LUT recommendation
        lut_config = LUT_RECOMMENDATIONS.get(intent, LUT_RECOMMENDATIONS[ColorGradingIntent.NEUTRAL])
        
        return ColorAnalysis(
            shot_id=shot_id,
            dominant_colors=dominant_colors,
            color_palette=color_palette,
            histogram_r=hist_r,
            histogram_g=hist_g,
            histogram_b=hist_b,
            avg_brightness=avg_brightness,
            avg_saturation=avg_saturation,
            white_balance_temp=temp,
            white_balance_tint=float(tint),
            contrast_ratio=contrast,
            dynamic_range=dynamic_range,
            lut_suggestion=lut_config["lut_name"],
            color_intent=intent
        )
    
    def _calculate_color_consistency(
        self,
        prev: ColorAnalysis,
        curr: ColorAnalysis
    ) -> float:
        """Calculate color consistency between adjacent shots"""
        
        # Compare brightness
        brightness_diff = abs(prev.avg_brightness - curr.avg_brightness) / 255
        
        # Compare saturation
        saturation_diff = abs(prev.avg_saturation - curr.avg_saturation)
        
        # Compare color temperature
        temp_diff = abs(prev.white_balance_temp - curr.white_balance_temp) / 5000
        
        # Weighted consistency score
        consistency = 1.0 - (
            brightness_diff * 0.4 +
            saturation_diff * 0.3 +
            temp_diff * 0.3
        )
        
        return max(0.0, min(1.0, consistency))
    
    # -------------------------------------------------------------------------
    # STABILIZATION ANALYSIS
    # -------------------------------------------------------------------------
    
    async def analyze_stabilization(
        self,
        video_path: str,
        shots: List[ShotBoundary]
    ) -> List[StabilizationResult]:
        """
        Analyze video stability for each shot.
        
        Args:
            video_path: Path to video file
            shots: List of detected shots
            
        Returns:
            List of stabilization analysis results
        """
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV/NumPy not available, skipping stabilization analysis")
            return []
        
        logger.info(f"Analyzing stabilization for {len(shots)} shots")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        results = []
        
        try:
            for shot in shots:
                result = await self._analyze_shot_stability(cap, shot)
                results.append(result)
        
        finally:
            cap.release()
        
        return results
    
    async def _analyze_shot_stability(
        self,
        cap: cv2.VideoCapture,
        shot: ShotBoundary
    ) -> StabilizationResult:
        """Analyze stability of a single shot"""
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, shot.frame_start)
        
        prev_gray = None
        motion_vectors = []
        
        # Sample frames from shot
        sample_count = min(30, shot.frame_count)
        sample_interval = max(1, shot.frame_count // sample_count)
        
        frame_count = 0
        while frame_count < shot.frame_count and len(motion_vectors) < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate optical flow
                    try:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray, None,
                            pyr_scale=0.5, levels=3, winsize=15,
                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                        )
                        
                        # Get motion magnitude
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        avg_motion = float(np.mean(mag))
                        motion_vectors.append(avg_motion)
                    except Exception:
                        pass
                
                prev_gray = gray
            
            frame_count += 1
        
        # Calculate shake metrics
        if motion_vectors:
            avg_motion = float(np.mean(motion_vectors))
            motion_variance = float(np.var(motion_vectors))
            
            # Normalize shake score (0-1)
            shake_score = min(1.0, avg_motion / 10.0)
        else:
            avg_motion = 0.0
            motion_variance = 0.0
            shake_score = 0.0
        
        # Determine if stabilization is needed
        needs_stabilization = shake_score > SHAKE_THRESHOLD_LOW
        
        # Recommend method
        if shake_score > SHAKE_THRESHOLD_HIGH:
            method = StabilizationMethod.NEURAL
            crop_factor = 1.3
        elif shake_score > SHAKE_THRESHOLD_MEDIUM:
            method = StabilizationMethod.OPTICAL_FLOW
            crop_factor = 1.2
        elif shake_score > SHAKE_THRESHOLD_LOW:
            method = StabilizationMethod.WARP
            crop_factor = 1.1
        else:
            method = StabilizationMethod.NONE
            crop_factor = 1.0
        
        return StabilizationResult(
            shot_id=shot.shot_id,
            shake_score=shake_score,
            motion_vectors_avg=(avg_motion, 0.0),
            motion_variance=motion_variance,
            crop_factor=crop_factor,
            stabilization_needed=needs_stabilization,
            recommended_method=method,
            confidence=0.8 if motion_vectors else 0.3,
            estimated_quality_loss=(crop_factor - 1.0) * 0.5
        )
    
    # -------------------------------------------------------------------------
    # PACING ANALYSIS
    # -------------------------------------------------------------------------
    
    async def analyze_pacing(
        self,
        video_path: str,
        shots: List[ShotBoundary]
    ) -> PacingAnalysis:
        """
        Analyze video pacing and rhythm.
        
        Args:
            video_path: Path to video file
            shots: List of detected shots
            
        Returns:
            PacingAnalysis with rhythm metrics
        """
        if not shots:
            return PacingAnalysis(
                video_id=os.path.basename(video_path),
                total_shots=0,
                total_duration_seconds=0.0,
                shots_per_minute=0.0,
                avg_shot_duration_seconds=0.0,
                min_shot_duration_seconds=0.0,
                max_shot_duration_seconds=0.0,
                shot_duration_variance=0.0
            )
        
        # Calculate durations
        durations = [shot.duration_seconds for shot in shots]
        total_duration = sum(durations)
        
        avg_duration = total_duration / len(shots) if shots else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Calculate variance
        if NUMPY_AVAILABLE:
            variance = float(np.var(durations)) if durations else 0
        else:
            mean = sum(durations) / len(durations)
            variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        
        # Shots per minute
        spm = (len(shots) / total_duration * 60) if total_duration > 0 else 0
        
        # Determine rhythm pattern
        if len(durations) >= 3:
            pattern = self._detect_rhythm_pattern(durations)
        else:
            pattern = RhythmPattern.CONSTANT
        
        # Calculate rhythm score (consistency)
        if variance > 0 and avg_duration > 0:
            cv = (variance ** 0.5) / avg_duration  # Coefficient of variation
            rhythm_score = 1.0 - min(cv, 1.0)
        else:
            rhythm_score = 1.0
        
        # Generate energy curve
        energy_curve = []
        window_size = max(3, len(shots) // 10)
        for i in range(len(shots)):
            start = max(0, i - window_size // 2)
            end = min(len(shots), i + window_size // 2 + 1)
            window_durations = durations[start:end]
            avg_window = sum(window_durations) / len(window_durations)
            # Shorter shots = higher energy
            energy = 1.0 - min(avg_window / 5.0, 1.0)
            energy_curve.append(energy)
        
        # Generate recommendation
        if spm > 40:
            recommendation = "Very fast pacing may cause viewer fatigue. Consider lengthening some shots."
        elif spm > 25:
            recommendation = "Fast pacing suitable for energetic content."
        elif spm > 15:
            recommendation = "Moderate pacing appropriate for most commercial content."
        elif spm > 8:
            recommendation = "Slow pacing suitable for dramatic or documentary content."
        else:
            recommendation = "Very slow pacing. Consider tightening edits for better engagement."
        
        return PacingAnalysis(
            video_id=os.path.basename(video_path),
            total_shots=len(shots),
            total_duration_seconds=total_duration,
            shots_per_minute=spm,
            avg_shot_duration_seconds=avg_duration,
            min_shot_duration_seconds=min_duration,
            max_shot_duration_seconds=max_duration,
            shot_duration_variance=variance,
            rhythm_score=rhythm_score,
            rhythm_pattern=pattern,
            energy_curve=energy_curve,
            pacing_recommendation=recommendation
        )
    
    def _detect_rhythm_pattern(self, durations: List[float]) -> RhythmPattern:
        """Detect the rhythm pattern from shot durations"""
        
        if len(durations) < 3:
            return RhythmPattern.CONSTANT
        
        # Calculate trend
        first_third = durations[:len(durations)//3]
        last_third = durations[-len(durations)//3:]
        
        avg_first = sum(first_third) / len(first_third)
        avg_last = sum(last_third) / len(last_third)
        
        if avg_last < avg_first * 0.7:
            return RhythmPattern.ACCELERATING
        elif avg_last > avg_first * 1.3:
            return RhythmPattern.DECELERATING
        
        # Check for alternating pattern
        diffs = [durations[i+1] - durations[i] for i in range(len(durations)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        
        if sign_changes > len(diffs) * 0.6:
            return RhythmPattern.ALTERNATING
        
        # Check for building/climax
        if NUMPY_AVAILABLE:
            variance = float(np.var(durations))
        else:
            mean = sum(durations) / len(durations)
            variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        
        if variance < 0.5:
            return RhythmPattern.CONSTANT
        
        return RhythmPattern.IRREGULAR
    
    # -------------------------------------------------------------------------
    # TIMELINE ASSEMBLY
    # -------------------------------------------------------------------------
    
    async def generate_timeline(
        self,
        shots: List[ShotBoundary],
        transitions: List[TransitionRecommendation],
        target_duration: Optional[float] = None
    ) -> List[TimelineSegment]:
        """
        Generate timeline segments for video assembly.
        
        Args:
            shots: List of detected shots
            transitions: List of transition recommendations
            target_duration: Optional target duration in seconds
            
        Returns:
            List of timeline segments
        """
        if not shots:
            return []
        
        # Create transition lookup
        transition_map = {t.from_shot_id: t for t in transitions}
        
        timeline = []
        
        for i, shot in enumerate(shots):
            # Get transitions
            trans_out = transition_map.get(shot.shot_id)
            trans_in = None
            if i > 0:
                prev_shot = shots[i-1]
                trans_in = transition_map.get(prev_shot.shot_id)
            
            segment = TimelineSegment(
                segment_id=f"seg_{i:04d}",
                shot_id=shot.shot_id,
                position=i,
                in_point_frames=shot.frame_start,
                out_point_frames=shot.frame_end,
                in_point_seconds=shot.timestamp_start,
                out_point_seconds=shot.timestamp_end,
                duration_frames=shot.frame_count,
                duration_seconds=shot.duration_seconds,
                transition_in=trans_in,
                transition_out=trans_out
            )
            timeline.append(segment)
        
        # Adjust for target duration if specified
        if target_duration:
            timeline = self._adjust_timeline_duration(timeline, target_duration)
        
        return timeline
    
    def _adjust_timeline_duration(
        self,
        timeline: List[TimelineSegment],
        target: float
    ) -> List[TimelineSegment]:
        """Adjust timeline to match target duration"""
        
        current_duration = sum(s.duration_seconds for s in timeline)
        
        if abs(current_duration - target) < 0.1:
            return timeline
        
        ratio = target / current_duration if current_duration > 0 else 1.0
        
        # Clamp ratio to reasonable bounds
        ratio = max(0.5, min(2.0, ratio))
        
        for segment in timeline:
            segment.speed_factor = 1.0 / ratio
            segment.duration_seconds *= ratio
            segment.duration_frames = int(segment.duration_frames * ratio)
        
        return timeline
    
    # -------------------------------------------------------------------------
    # FFMPEG COMMAND GENERATION
    # -------------------------------------------------------------------------
    
    async def generate_assembly_commands(
        self,
        video_path: str,
        timeline: List[TimelineSegment],
        output_path: str,
        metadata: VideoMetadata
    ) -> List[AssemblyCommand]:
        """
        Generate FFmpeg commands for video assembly.
        
        Args:
            video_path: Input video path
            timeline: Timeline segments
            output_path: Output video path
            metadata: Video metadata
            
        Returns:
            List of FFmpeg assembly commands
        """
        commands = []
        
        # Build filter complex for transitions
        filter_parts = []
        concat_inputs = []
        
        for i, segment in enumerate(timeline):
            input_label = f"[{i}:v]"
            output_label = f"[v{i}]"
            
            # Trim filter
            trim_filter = (
                f"[0:v]trim=start={segment.in_point_seconds:.3f}:"
                f"end={segment.out_point_seconds:.3f},setpts=PTS-STARTPTS"
            )
            
            # Add speed adjustment if needed
            if segment.speed_factor != 1.0:
                trim_filter += f",setpts={segment.speed_factor}*PTS"
            
            trim_filter += output_label
            filter_parts.append(trim_filter)
            concat_inputs.append(f"[v{i}]")
            
            # Add transition filter if present
            if segment.transition_out and i < len(timeline) - 1:
                trans = segment.transition_out
                if trans.transition_type != TransitionType.HARD_CUT:
                    trans_filter = trans.get_ffmpeg_filter(metadata.fps)
                    if trans_filter:
                        filter_parts.append(
                            f"[v{i}][v{i+1}]{trans_filter}[vt{i}]"
                        )
        
        # Concat filter
        if len(concat_inputs) > 1:
            concat_filter = f"{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=1:a=0[outv]"
            filter_parts.append(concat_filter)
        else:
            filter_parts.append(f"{concat_inputs[0]}copy[outv]")
        
        filter_complex = ";".join(filter_parts)
        
        # Main assembly command
        assembly_cmd = AssemblyCommand(
            command_id="cmd_assemble",
            operation="assemble",
            input_files=[video_path],
            output_file=output_path,
            ffmpeg_args=[
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p"
            ],
            filter_complex=filter_complex,
            estimated_duration_seconds=sum(s.duration_seconds for s in timeline)
        )
        commands.append(assembly_cmd)
        
        return commands
    
    # -------------------------------------------------------------------------
    # ISSUE DETECTION
    # -------------------------------------------------------------------------
    
    async def detect_issues(
        self,
        shots: List[ShotBoundary],
        pacing: PacingAnalysis,
        color_analyses: List[ColorAnalysis],
        stabilization_results: List[StabilizationResult],
        style: StylePreset
    ) -> List[EditingIssue]:
        """
        Detect editing issues and generate recommendations.
        
        Args:
            shots: List of detected shots
            pacing: Pacing analysis
            color_analyses: Color analysis results
            stabilization_results: Stabilization results
            style: Editing style preset
            
        Returns:
            List of detected issues
        """
        issues = []
        issue_id = 0
        style_config = STYLE_PRESETS.get(style, STYLE_PRESETS[StylePreset.COMMERCIAL])
        
        # Check shot duration issues
        for shot in shots:
            if shot.duration_seconds < style_config.get("min_shot_duration", 0.3):
                issues.append(EditingIssue(
                    issue_id=f"issue_{issue_id:04d}",
                    severity=EditingSeverity.WARNING,
                    category="pacing",
                    message=f"Shot {shot.shot_id} is too short ({shot.duration_seconds:.2f}s)",
                    shot_id=shot.shot_id,
                    timestamp_seconds=shot.timestamp_start,
                    suggestion="Consider removing or extending this shot",
                    auto_fixable=False
                ))
                issue_id += 1
            
            if shot.duration_seconds > style_config.get("max_shot_duration", 15.0):
                issues.append(EditingIssue(
                    issue_id=f"issue_{issue_id:04d}",
                    severity=EditingSeverity.INFO,
                    category="pacing",
                    message=f"Shot {shot.shot_id} is quite long ({shot.duration_seconds:.2f}s)",
                    shot_id=shot.shot_id,
                    timestamp_seconds=shot.timestamp_start,
                    suggestion="Consider adding cutaways or B-roll",
                    auto_fixable=False
                ))
                issue_id += 1
        
        # Check pacing issues
        if pacing.rhythm_score < 0.5:
            issues.append(EditingIssue(
                issue_id=f"issue_{issue_id:04d}",
                severity=EditingSeverity.WARNING,
                category="pacing",
                message=f"Inconsistent pacing detected (rhythm score: {pacing.rhythm_score:.2f})",
                suggestion="Review shot durations for more consistent rhythm",
                auto_fixable=False
            ))
            issue_id += 1
        
        # Check color consistency
        for i, color in enumerate(color_analyses):
            if color.consistency_score < 0.7:
                issues.append(EditingIssue(
                    issue_id=f"issue_{issue_id:04d}",
                    severity=EditingSeverity.WARNING,
                    category="color",
                    message=f"Color inconsistency at {color.shot_id} (score: {color.consistency_score:.2f})",
                    shot_id=color.shot_id,
                    suggestion="Consider color grading to match adjacent shots",
                    auto_fixable=True
                ))
                issue_id += 1
        
        # Check stabilization needs
        for stab in stabilization_results:
            if stab.shake_severity in ("high", "medium"):
                issues.append(EditingIssue(
                    issue_id=f"issue_{issue_id:04d}",
                    severity=EditingSeverity.WARNING if stab.shake_severity == "medium" else EditingSeverity.ERROR,
                    category="stabilization",
                    message=f"Camera shake detected in {stab.shot_id} ({stab.shake_severity})",
                    shot_id=stab.shot_id,
                    suggestion=f"Apply {stab.recommended_method.value} stabilization",
                    auto_fixable=True
                ))
                issue_id += 1
        
        return issues
    
    # -------------------------------------------------------------------------
    # REPORT GENERATION
    # -------------------------------------------------------------------------
    
    def generate_report(self, result: EditValidationResult) -> str:
        """
        Generate markdown report for edit validation results.
        
        Args:
            result: Complete validation result
            
        Returns:
            Markdown formatted report
        """
        lines = [
            "# THE EDITOR - Video Edit Analysis Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Video:** `{result.video_path}`",
            "",
            "---",
            "",
            "## 📊 Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Shots | {len(result.shots)} |",
            f"| Total Duration | {result.video_metadata.duration_seconds:.2f}s |" if result.video_metadata else "",
            f"| Processing Time | {result.processing_time_seconds:.2f}s |",
            f"| Issues Found | {len(result.issues)} |",
            f"| Cost Estimate | ${result.cost_estimate:.2f} |",
            "",
        ]
        
        # Pacing section
        if result.pacing:
            lines.extend([
                "## ⏱️ Pacing Analysis",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Shots per Minute | {result.pacing.shots_per_minute:.1f} |",
                f"| Avg Shot Duration | {result.pacing.avg_shot_duration_seconds:.2f}s |",
                f"| Min Shot Duration | {result.pacing.min_shot_duration_seconds:.2f}s |",
                f"| Max Shot Duration | {result.pacing.max_shot_duration_seconds:.2f}s |",
                f"| Rhythm Score | {result.pacing.rhythm_score:.2f} |",
                f"| Rhythm Pattern | {result.pacing.rhythm_pattern.value} |",
                "",
                f"**Recommendation:** {result.pacing.pacing_recommendation}",
                "",
            ])
        
        # Shots section
        lines.extend([
            "## 🎬 Shot Breakdown",
            "",
            "| Shot ID | Duration | Transition | Confidence |",
            "|---------|----------|------------|------------|",
        ])
        
        for shot in result.shots[:20]:  # Limit to first 20
            lines.append(
                f"| {shot.shot_id} | {shot.duration_seconds:.2f}s | "
                f"{shot.transition_type.value} | {shot.confidence:.2f} |"
            )
        
        if len(result.shots) > 20:
            lines.append(f"| ... | ({len(result.shots) - 20} more shots) | ... | ... |")
        
        lines.append("")
        
        # Transitions section
        if result.transitions:
            lines.extend([
                "## ↔️ Transition Recommendations",
                "",
                "| From | To | Type | Duration | Rule |",
                "|------|-----|------|----------|------|",
            ])
            
            for trans in result.transitions[:15]:
                lines.append(
                    f"| {trans.from_shot_id} | {trans.to_shot_id} | "
                    f"{trans.transition_type.value} | {trans.duration_seconds:.2f}s | "
                    f"{trans.grammar_rule_applied or 'default'} |"
                )
            
            lines.append("")
        
        # Issues section
        if result.issues:
            lines.extend([
                "## ⚠️ Issues & Recommendations",
                "",
            ])
            
            for issue in result.issues:
                icon = {
                    EditingSeverity.INFO: "ℹ️",
                    EditingSeverity.WARNING: "⚠️",
                    EditingSeverity.ERROR: "❌",
                    EditingSeverity.CRITICAL: "🚨"
                }.get(issue.severity, "•")
                
                lines.append(f"{icon} **{issue.category.upper()}**: {issue.message}")
                if issue.suggestion:
                    lines.append(f"   - *Suggestion:* {issue.suggestion}")
                lines.append("")
        
        # FFmpeg commands section
        if result.assembly_commands:
            lines.extend([
                "## 🔧 FFmpeg Commands",
                "",
                "```bash",
            ])
            
            for cmd in result.assembly_commands:
                lines.append(cmd.to_command_string())
            
            lines.extend([
                "```",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "*Report generated by THE EDITOR (Agent 7.5) | Barrios A2I*",
        ])
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # -------------------------------------------------------------------------
    
    async def analyze(self, request: EditValidationRequest) -> EditValidationResult:
        """
        Run complete edit validation and analysis pipeline.
        
        Args:
            request: Edit validation request parameters
            
        Returns:
            Complete validation result
        """
        start_time = time.time()
        
        logger.info(f"Starting analysis: {request.video_path}")
        
        try:
            # Step 1: Extract metadata
            metadata = await self.extract_video_metadata(request.video_path)
            
            # Step 2: Detect shots
            shots = await self.detect_shots(
                request.video_path,
                confidence_threshold=request.shot_detection_threshold,
                metadata=metadata
            )
            
            # Step 3: Recommend transitions
            transitions = await self.recommend_transitions(
                shots,
                style=request.style_preset,
                custom_rules=request.custom_rules
            )
            
            # Step 4: Analyze colors
            color_analyses = await self.analyze_colors(
                request.video_path,
                shots,
                intent=request.color_intent
            )
            
            # Step 5: Analyze stabilization
            stabilization_results = []
            if request.enable_stabilization:
                stabilization_results = await self.analyze_stabilization(
                    request.video_path,
                    shots
                )
            
            # Step 6: Analyze pacing
            pacing = await self.analyze_pacing(request.video_path, shots)
            
            # Step 7: Generate timeline
            timeline = await self.generate_timeline(
                shots,
                transitions,
                target_duration=request.target_duration
            )
            
            # Step 8: Generate FFmpeg commands
            assembly_commands = []
            if request.generate_ffmpeg_commands and timeline:
                output_path = request.video_path.replace(
                    ".", f"_edited.{request.output_format}"
                )
                assembly_commands = await self.generate_assembly_commands(
                    request.video_path,
                    timeline,
                    output_path,
                    metadata
                )
            
            # Step 9: Detect issues
            issues = await self.detect_issues(
                shots,
                pacing,
                color_analyses,
                stabilization_results,
                request.style_preset
            )
            
            # Calculate processing time and cost
            processing_time = time.time() - start_time
            cost_estimate = self._estimate_cost(metadata, processing_time)
            
            # Build result
            result = EditValidationResult(
                success=True,
                video_path=request.video_path,
                video_metadata=metadata,
                shots=shots,
                transitions=transitions,
                color_analyses=color_analyses,
                stabilization_results=stabilization_results,
                pacing=pacing,
                timeline=timeline,
                assembly_commands=assembly_commands,
                issues=issues,
                processing_time_seconds=processing_time,
                cost_estimate=cost_estimate
            )
            
            # Generate report
            result.report_markdown = self.generate_report(result)
            
            logger.info(
                f"Analysis complete: {len(shots)} shots, {len(issues)} issues, "
                f"{processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return EditValidationResult(
                success=False,
                video_path=request.video_path,
                issues=[
                    EditingIssue(
                        issue_id="error_001",
                        severity=EditingSeverity.CRITICAL,
                        category="system",
                        message=str(e)
                    )
                ],
                processing_time_seconds=time.time() - start_time
            )
    
    def _estimate_cost(self, metadata: VideoMetadata, processing_time: float) -> float:
        """Estimate processing cost"""
        # Base cost per second of video
        base_cost_per_second = 0.005
        
        # Processing overhead
        overhead = 0.05
        
        cost = (metadata.duration_seconds * base_cost_per_second) + overhead
        
        # Adjust for resolution
        if metadata.height >= 2160:
            cost *= 2.0
        elif metadata.height >= 1080:
            cost *= 1.0
        else:
            cost *= 0.5
        
        return min(cost, 0.30)  # Cap at $0.30 per video
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def _trim_cache(self, cache: Dict, max_size: Optional[int] = None) -> None:
        """Trim cache to max size"""
        max_size = max_size or self.max_cache_size
        while len(cache) > max_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        self._shot_cache.clear()
        self._color_cache.clear()
        self._analysis_cache.clear()
        logger.info("Caches cleared")


# =============================================================================
# SECTION 6: FACTORY FUNCTION
# =============================================================================

def create_editor(
    device: str = "cpu",
    transnet_threshold: float = TRANSNET_THRESHOLD,
    enable_gpu: bool = False,
    cache_enabled: bool = True
) -> TheEditor:
    """
    Factory function to create THE EDITOR instance.
    
    Args:
        device: Compute device ('cpu', 'cuda', 'mps')
        transnet_threshold: Shot detection confidence threshold
        enable_gpu: Enable GPU acceleration
        cache_enabled: Enable result caching
        
    Returns:
        Configured TheEditor instance
    """
    return TheEditor(
        device=device,
        transnet_threshold=transnet_threshold,
        enable_gpu=enable_gpu,
        cache_enabled=cache_enabled
    )


# =============================================================================
# SECTION 7: NEXUS REGISTRATION
# =============================================================================

NEXUS_REGISTRATION = {
    "agent_id": "agent_7.5",
    "name": "THE EDITOR",
    "version": "1.0.0",
    "phase": "VORTEX",
    "handler": "editor.analyze",
    "input_schema": "EditValidationRequest",
    "output_schema": "EditValidationResult",
    "description": "Shot boundary detection, transition selection, and video assembly analysis",
    "capabilities": [
        "shot_detection",
        "transition_recommendation",
        "color_analysis",
        "stabilization_analysis",
        "pacing_analysis",
        "timeline_generation",
        "ffmpeg_command_generation"
    ],
    "dependencies": [
        "ffmpeg",
        "ffprobe"
    ],
    "optional_dependencies": [
        "opencv-python",
        "numpy",
        "scikit-learn",
        "transnetv2-pytorch"
    ],
    "cost_target": {
        "min": 0.15,
        "max": 0.30,
        "currency": "USD",
        "unit": "video"
    },
    "latency_target": {
        "min_seconds": 8,
        "max_seconds": 20
    },
    "metrics": {
        "shot_detection_f1": 0.85,
        "transition_accuracy": 0.80,
        "color_consistency_score": 0.90
    }
}


# =============================================================================
# SECTION 8: CLI MAIN
# =============================================================================

async def main_async(args: argparse.Namespace) -> int:
    """Async main function"""
    
    # Create editor
    editor = create_editor(
        device=args.device,
        transnet_threshold=args.threshold,
        enable_gpu=args.gpu
    )
    
    # Build request
    request = EditValidationRequest(
        video_path=args.video_path,
        style_preset=StylePreset(args.style),
        target_duration=args.duration,
        music_path=args.music,
        color_intent=ColorGradingIntent(args.color_intent),
        enable_stabilization=not args.no_stabilization,
        shot_detection_threshold=args.threshold,
        generate_ffmpeg_commands=not args.no_ffmpeg,
        output_format=args.format
    )
    
    # Run analysis
    result = await editor.analyze(request)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        
        if args.json:
            with open(output_path, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            print(f"JSON output written to: {output_path}")
        else:
            with open(output_path, 'w') as f:
                f.write(result.report_markdown)
            print(f"Report written to: {output_path}")
    else:
        if args.json:
            print(json.dumps(result.dict(), indent=2, default=str))
        else:
            print(result.report_markdown)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Analysis Complete")
    print(f"   Shots detected: {len(result.shots)}")
    print(f"   Transitions recommended: {len(result.transitions)}")
    print(f"   Issues found: {len(result.issues)}")
    print(f"   Processing time: {result.processing_time_seconds:.2f}s")
    print(f"   Estimated cost: ${result.cost_estimate:.2f}")
    print(f"{'='*60}")
    
    return 0 if result.success else 1


def main() -> int:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="THE EDITOR (Agent 7.5) - Shot Detection & Assembly Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python the_editor.py video.mp4
  python the_editor.py video.mp4 --style commercial --duration 30
  python the_editor.py video.mp4 --style cinematic --color-intent cinematic
  python the_editor.py video.mp4 --output report.md --json

Style Presets:
  commercial, cinematic, documentary, music_video, corporate,
  social_media, broadcast, film_trailer, product_demo, testimonial,
  energetic, calm, dramatic, minimalist, retro, modern

Color Intents:
  neutral, warm, cool, high_contrast, low_contrast, vintage,
  teal_orange, desaturated, vibrant, cinematic, broadcast_safe
        """
    )
    
    parser.add_argument(
        "video_path",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-s", "--style",
        default="commercial",
        choices=[s.value for s in StylePreset],
        help="Editing style preset (default: commercial)"
    )
    
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Target output duration in seconds"
    )
    
    parser.add_argument(
        "-m", "--music",
        default=None,
        help="Path to music file for beat matching"
    )
    
    parser.add_argument(
        "-c", "--color-intent",
        default="neutral",
        choices=[c.value for c in ColorGradingIntent],
        help="Color grading intent (default: neutral)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Shot detection threshold 0.1-0.9 (default: 0.5)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path for report"
    )
    
    parser.add_argument(
        "-f", "--format",
        default="mp4",
        choices=["mp4", "mov", "mkv", "webm"],
        help="Output video format (default: mp4)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device (default: cpu)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration"
    )
    
    parser.add_argument(
        "--no-stabilization",
        action="store_true",
        help="Disable stabilization analysis"
    )
    
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Skip FFmpeg command generation"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of markdown"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate video path
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
        return 1
    
    # Run async main
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
