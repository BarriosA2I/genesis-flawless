#!/usr/bin/env python3
"""
THE WORDSMITH (Agent 7.25) - Text Detection & QA Agent
=======================================================
RAGNAROK Video Pipeline | VORTEX Phase
Barrios A2I Cognitive Systems Division

Purpose: Detect, validate, and correct all text appearing in commercial videos
Pipeline Position: VORTEX Phase (after SOUNDSCAPER, before EDITOR)
Cost Target: $0.10-0.20/video
Latency Target: 5-12s

Features:
- Text Detection & OCR with confidence scoring
- Spelling & Grammar Validation via LanguageTool hooks
- Brand Compliance Checking (forbidden words, legal disclaimers)
- WCAG 2.2 Accessibility Validation (contrast, font sizes)
- Auto-correction Pipeline with priority ranking

Architecture:
- Keyframe extraction from video
- Multi-engine OCR (Tesseract + PaddleOCR hooks)
- NLP validation pipeline
- Brand rule engine
- Accessibility analyzer
- Correction suggester with confidence scoring
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
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
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, field_validator, computed_field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("THE_WORDSMITH")


# =============================================================================
# SECTION 1: ENUMS & CONSTANTS
# =============================================================================

class OCREngine(str, Enum):
    """Supported OCR engines"""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    EASYOCR = "easyocr"
    HYBRID = "hybrid"  # Tesseract + fallback to PaddleOCR


class ValidationSeverity(str, Enum):
    """Severity levels for text validation errors"""
    CRITICAL = "critical"  # Must fix before publishing (legal, safety)
    MAJOR = "major"        # Should fix (brand compliance, accessibility fail)
    MINOR = "minor"        # Nice to fix (style, minor typos)
    INFO = "info"          # Informational only


class ErrorCategory(str, Enum):
    """Categories of text validation errors"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    BRAND_VIOLATION = "brand_violation"
    LEGAL_MISSING = "legal_missing"
    ACCESSIBILITY = "accessibility"
    CONTRAST = "contrast"
    FONT_SIZE = "font_size"
    TRADEMARK = "trademark"
    COMPETITOR = "competitor"
    FORBIDDEN_WORD = "forbidden_word"
    READABILITY = "readability"
    TIMING = "timing"
    CONSISTENCY = "consistency"


class WCAGLevel(str, Enum):
    """WCAG accessibility levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"
    FAIL = "FAIL"


class ColorBlindnessType(str, Enum):
    """Types of color blindness for simulation"""
    PROTANOPIA = "protanopia"      # Red-blind
    DEUTERANOPIA = "deuteranopia"  # Green-blind
    TRITANOPIA = "tritanopia"      # Blue-blind
    ACHROMATOPSIA = "achromatopsia"  # Complete color blindness


class TextPosition(str, Enum):
    """Text position in frame"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    MIDDLE_CENTER = "middle_center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    LOWER_THIRD = "lower_third"


class FontCategory(str, Enum):
    """Estimated font categories"""
    SERIF = "serif"
    SANS_SERIF = "sans_serif"
    SCRIPT = "script"
    DISPLAY = "display"
    MONOSPACE = "monospace"
    HANDWRITTEN = "handwritten"
    UNKNOWN = "unknown"


class CheckFlag(str, Enum):
    """Flags for what validations to perform"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    BRAND = "brand"
    ACCESSIBILITY = "accessibility"
    LEGAL = "legal"
    CONSISTENCY = "consistency"
    ALL = "all"


# =============================================================================
# SECTION 2: CONSTANTS & CONFIGURATION
# =============================================================================

class WordsmithConfig:
    """Configuration constants for THE WORDSMITH"""
    
    # OCR Settings
    DEFAULT_OCR_ENGINE: OCREngine = OCREngine.HYBRID
    OCR_CONFIDENCE_THRESHOLD: float = 0.65
    MIN_TEXT_LENGTH: int = 1
    MAX_TEXT_LENGTH: int = 500
    
    # Keyframe Extraction
    DEFAULT_KEYFRAME_INTERVAL: float = 1.0  # seconds
    MAX_KEYFRAMES: int = 60
    MIN_KEYFRAMES: int = 3
    
    # WCAG Thresholds
    WCAG_AA_CONTRAST_NORMAL: float = 4.5
    WCAG_AA_CONTRAST_LARGE: float = 3.0
    WCAG_AAA_CONTRAST_NORMAL: float = 7.0
    WCAG_AAA_CONTRAST_LARGE: float = 4.5
    LARGE_TEXT_THRESHOLD_PX: int = 24  # 18pt bold or 24pt regular
    MIN_FONT_SIZE_PX: int = 12
    
    # Reading Time (words per minute)
    AVERAGE_WPM: int = 250
    MIN_DISPLAY_TIME_SEC: float = 1.5
    MAX_CHARS_PER_LINE: int = 42
    
    # Validation
    MAX_TYPOS_BEFORE_FLAG: int = 2
    MAX_GRAMMAR_ERRORS: int = 3
    
    # Cost Targets
    COST_PER_FRAME_OCR: float = 0.001
    COST_PER_VALIDATION_CALL: float = 0.005
    TARGET_COST_MIN: float = 0.10
    TARGET_COST_MAX: float = 0.20
    
    # Latency
    TARGET_LATENCY_MIN_SEC: float = 5.0
    TARGET_LATENCY_MAX_SEC: float = 12.0


# =============================================================================
# SECTION 3: BASE DICTIONARIES & COMMON DATA
# =============================================================================

# Common typos and their corrections
COMMON_TYPOS: Dict[str, str] = {
    "teh": "the",
    "thier": "their",
    "recieve": "receive",
    "occured": "occurred",
    "seperate": "separate",
    "definately": "definitely",
    "accomodate": "accommodate",
    "occassion": "occasion",
    "untill": "until",
    "beleive": "believe",
    "existance": "existence",
    "occurence": "occurrence",
    "persue": "pursue",
    "wierd": "weird",
    "concensus": "consensus",
    "entreprenuer": "entrepreneur",
    "liason": "liaison",
    "millenium": "millennium",
    "priviledge": "privilege",
    "truely": "truly",
    "goverment": "government",
    "enviroment": "environment",
    "calender": "calendar",
    "begining": "beginning",
    "comming": "coming",
    "diferent": "different",
    "excercise": "exercise",
    "finaly": "finally",
    "grammer": "grammar",
    "intresting": "interesting",
    "knowlege": "knowledge",
    "libary": "library",
    "mispell": "misspell",
    "neccessary": "necessary",
    "peice": "piece",
    "probaly": "probably",
    "realy": "really",
    "refering": "referring",
    "similiar": "similar",
    "tommorow": "tomorrow",
    "writting": "writing",
}

# Industry-specific terms that should NOT be flagged as misspellings
INDUSTRY_TERMS: Set[str] = {
    # Tech
    "ai", "ml", "api", "sdk", "saas", "paas", "iaas", "gpu", "cpu",
    "blockchain", "cryptocurrency", "bitcoin", "ethereum", "nft",
    "oauth", "jwt", "ssl", "https", "url", "uri", "json", "xml",
    "frontend", "backend", "fullstack", "devops", "cicd", "kubernetes",
    
    # Marketing
    "cta", "cpc", "cpm", "roi", "kpi", "seo", "sem", "ppc", "cro",
    "b2b", "b2c", "d2c", "omnichannel", "multichannel", "retargeting",
    
    # Video/Media
    "4k", "8k", "hdr", "uhd", "fps", "bitrate", "codec", "hevc",
    "prores", "dnxhd", "mp4", "mov", "webm", "mkv",
    
    # Legal/Business
    "llc", "inc", "corp", "ltd", "plc", "gmbh", "pty",
    "hipaa", "gdpr", "ccpa", "sox", "pci", "dss",
    
    # Common abbreviations
    "etc", "ie", "eg", "vs", "approx", "misc", "qty", "amt",
}

# Common forbidden words for brand safety
DEFAULT_FORBIDDEN_WORDS: Set[str] = {
    # Profanity (mild set - expand as needed)
    "damn", "hell", "crap",
    
    # Potentially problematic claims
    "guaranteed", "100%", "miracle", "cure", "instant",
    "risk-free", "no-risk", "foolproof",
    
    # Legal red flags
    "patent pending",  # Unless verified
    "fda approved",    # Unless verified
    "clinically proven",  # Unless verified
}

# Required legal disclaimers by industry
LEGAL_DISCLAIMERS: Dict[str, List[str]] = {
    "finance": [
        "past performance",
        "not guaranteed",
        "investment risk",
        "consult advisor",
    ],
    "pharma": [
        "side effects",
        "consult doctor",
        "not for everyone",
        "ask your doctor",
    ],
    "alcohol": [
        "drink responsibly",
        "21+",
        "legal drinking age",
    ],
    "gambling": [
        "gamble responsibly",
        "18+",
        "odds may vary",
    ],
    "auto": [
        "professional driver",
        "closed course",
        "do not attempt",
    ],
}

# Common competitor names (generic - should be customized per client)
DEFAULT_COMPETITOR_TERMS: Set[str] = {
    "competitor", "other brands", "leading brand", "brand x",
}

# Grammar rules (simplified - LanguageTool does heavy lifting)
GRAMMAR_PATTERNS: List[Tuple[str, str, str]] = [
    # (pattern, description, severity)
    (r"\byour\s+welcome\b", "Should be 'you're welcome'", "major"),
    (r"\bits\s+been\b", "Check 'it's been' vs 'its been'", "minor"),
    (r"\bcould\s+of\b", "Should be 'could have'", "major"),
    (r"\bshould\s+of\b", "Should be 'should have'", "major"),
    (r"\bwould\s+of\b", "Should be 'would have'", "major"),
    (r"\balot\b", "Should be 'a lot'", "minor"),
    (r"\beveryday\b(?!\s+low)", "Check 'everyday' vs 'every day'", "info"),
    (r"\beffect\b.*\bchange\b", "Check 'effect' vs 'affect'", "info"),
    (r"[.!?]\s*[a-z]", "Capitalize after sentence end", "minor"),
    (r"\s{2,}", "Multiple spaces", "minor"),
]


# =============================================================================
# SECTION 4: PYDANTIC MODELS
# =============================================================================

class BoundingBox(BaseModel):
    """Bounding box for text region"""
    x1: int = Field(ge=0, description="Left edge pixel")
    y1: int = Field(ge=0, description="Top edge pixel")
    x2: int = Field(ge=0, description="Right edge pixel")
    y2: int = Field(ge=0, description="Bottom edge pixel")
    
    @computed_field
    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)
    
    @computed_field
    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)
    
    @computed_field
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @computed_field
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box"""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        
        return inter_area / max(union_area, 1)


class ColorRGB(BaseModel):
    """RGB color representation"""
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)
    
    @computed_field
    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    @computed_field
    @property
    def luminance(self) -> float:
        """Calculate relative luminance per WCAG"""
        def linearize(c: int) -> float:
            c_srgb = c / 255.0
            if c_srgb <= 0.03928:
                return c_srgb / 12.92
            return ((c_srgb + 0.055) / 1.055) ** 2.4
        
        return 0.2126 * linearize(self.r) + 0.7152 * linearize(self.g) + 0.0722 * linearize(self.b)
    
    def contrast_ratio(self, other: "ColorRGB") -> float:
        """Calculate contrast ratio with another color per WCAG"""
        l1 = max(self.luminance, other.luminance)
        l2 = min(self.luminance, other.luminance)
        return (l1 + 0.05) / (l2 + 0.05)


class FontEstimate(BaseModel):
    """Estimated font properties from detected text"""
    category: FontCategory = FontCategory.UNKNOWN
    estimated_size_px: int = Field(ge=1, default=16)
    is_bold: bool = False
    is_italic: bool = False
    is_uppercase: bool = False
    estimated_weight: int = Field(ge=100, le=900, default=400)


class TextDetection(BaseModel):
    """Single text detection from a frame"""
    frame_id: int = Field(ge=0, description="Frame number in video")
    timestamp_sec: float = Field(ge=0.0, description="Timestamp in seconds")
    bbox: BoundingBox
    text: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_engine: OCREngine = OCREngine.TESSERACT
    font_estimate: FontEstimate = Field(default_factory=FontEstimate)
    
    # Color analysis
    text_color: Optional[ColorRGB] = None
    background_color: Optional[ColorRGB] = None
    contrast_ratio: Optional[float] = None
    
    # Position classification
    position: TextPosition = TextPosition.MIDDLE_CENTER
    
    # Motion/blur indicators
    motion_blur_score: float = Field(ge=0.0, le=1.0, default=0.0)
    is_static: bool = True  # Same text in multiple frames
    
    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)


class ValidationError(BaseModel):
    """A single validation error found in text"""
    error_id: str = Field(description="Unique error identifier")
    category: ErrorCategory
    severity: ValidationSeverity
    
    # Location
    frame_id: Optional[int] = None
    timestamp_sec: Optional[float] = None
    text_span: Optional[Tuple[int, int]] = None  # Character indices
    
    # Error details
    original_text: str
    problematic_segment: str
    message: str
    
    # Correction
    suggestion: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    auto_fixable: bool = False
    
    # Context
    rule_id: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class CorrectionSuggestion(BaseModel):
    """Suggested correction for a validation error"""
    error_id: str
    original: str
    corrected: str
    confidence: float = Field(ge=0.0, le=1.0)
    auto_apply: bool = False
    requires_approval: bool = True
    reasoning: str = ""


class BrandGuideline(BaseModel):
    """Brand compliance guidelines"""
    company_name: str
    
    # Word lists
    forbidden_words: Set[str] = Field(default_factory=set)
    required_phrases: Set[str] = Field(default_factory=set)
    preferred_terms: Dict[str, str] = Field(default_factory=dict)  # wrong -> right
    
    # Competitors
    competitor_names: Set[str] = Field(default_factory=set)
    competitor_check_enabled: bool = True
    
    # Legal
    industry: Optional[str] = None
    required_disclaimers: List[str] = Field(default_factory=list)
    
    # Logo/trademark
    trademark_terms: Set[str] = Field(default_factory=set)
    trademark_symbols_required: bool = False  # ® or ™
    
    # Style
    tone_keywords: Set[str] = Field(default_factory=set)
    avoid_keywords: Set[str] = Field(default_factory=set)
    
    # Custom rules
    custom_rules: Dict[str, Any] = Field(default_factory=dict)


class AccessibilityResult(BaseModel):
    """WCAG accessibility validation result"""
    # Overall
    wcag_level: WCAGLevel = WCAGLevel.FAIL
    passes_aa: bool = False
    passes_aaa: bool = False
    
    # Contrast
    contrast_ratio: float = Field(ge=1.0, le=21.0, default=1.0)
    text_color: Optional[ColorRGB] = None
    background_color: Optional[ColorRGB] = None
    
    # Font
    font_size_px: int = Field(ge=1, default=16)
    is_large_text: bool = False
    meets_min_size: bool = True
    
    # Readability
    chars_per_line: int = Field(ge=0, default=0)
    display_duration_sec: float = Field(ge=0.0, default=0.0)
    reading_time_adequate: bool = True
    
    # Color blindness
    color_blind_safe: Dict[ColorBlindnessType, bool] = Field(default_factory=dict)
    
    # Issues
    issues: List[str] = Field(default_factory=list)


class FrameAnalysis(BaseModel):
    """Analysis results for a single frame"""
    frame_id: int
    timestamp_sec: float
    frame_path: Optional[str] = None
    
    # Detections
    text_detections: List[TextDetection] = Field(default_factory=list)
    total_text_regions: int = 0
    
    # Validation
    validation_errors: List[ValidationError] = Field(default_factory=list)
    accessibility_results: List[AccessibilityResult] = Field(default_factory=list)
    
    # Metrics
    ocr_confidence_avg: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time_ms: float = Field(ge=0.0, default=0.0)


class TextValidationRequest(BaseModel):
    """Request for text validation"""
    # Input
    video_path: str
    
    # Brand/client
    brand_guidelines: Optional[BrandGuideline] = None
    
    # What to check
    check_flags: List[CheckFlag] = Field(default_factory=lambda: [CheckFlag.ALL])
    
    # OCR settings
    ocr_engine: OCREngine = OCREngine.HYBRID
    keyframe_interval_sec: float = Field(ge=0.1, le=10.0, default=1.0)
    ocr_confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.65)
    
    # Accessibility
    target_wcag_level: WCAGLevel = WCAGLevel.AA
    
    # Options
    include_suggestions: bool = True
    auto_fix_minor: bool = False
    generate_report: bool = True
    
    # Custom dictionaries
    custom_allowed_words: Set[str] = Field(default_factory=set)
    custom_forbidden_words: Set[str] = Field(default_factory=set)
    
    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v:
            raise ValueError("video_path cannot be empty")
        return v


class ValidationSummary(BaseModel):
    """Summary of validation results"""
    total_frames_analyzed: int = 0
    total_text_regions: int = 0
    total_errors: int = 0
    
    # By severity
    critical_errors: int = 0
    major_errors: int = 0
    minor_errors: int = 0
    info_items: int = 0
    
    # By category
    errors_by_category: Dict[str, int] = Field(default_factory=dict)
    
    # Accessibility
    accessibility_pass_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_contrast_ratio: float = Field(ge=1.0, default=1.0)
    
    # Corrections
    auto_fixable_count: int = 0
    suggestions_count: int = 0
    
    # Quality score
    quality_score: float = Field(ge=0.0, le=100.0, default=0.0)


class TextValidationResult(BaseModel):
    """Complete validation result"""
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Request info
    video_path: str
    video_duration_sec: float = 0.0
    video_resolution: Tuple[int, int] = (1920, 1080)
    
    # Results
    frame_analyses: List[FrameAnalysis] = Field(default_factory=list)
    all_detections: List[TextDetection] = Field(default_factory=list)
    all_errors: List[ValidationError] = Field(default_factory=list)
    corrections: List[CorrectionSuggestion] = Field(default_factory=list)
    
    # Summary
    summary: ValidationSummary = Field(default_factory=ValidationSummary)
    
    # Accessibility
    overall_wcag_level: WCAGLevel = WCAGLevel.FAIL
    accessibility_details: List[AccessibilityResult] = Field(default_factory=list)
    
    # Report
    report_path: Optional[str] = None
    report_markdown: Optional[str] = None
    
    # Metadata
    processing_time_sec: float = 0.0
    estimated_cost: float = 0.0
    ocr_engine_used: OCREngine = OCREngine.TESSERACT
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# SECTION 5: HELPER FUNCTIONS
# =============================================================================

def generate_error_id() -> str:
    """Generate unique error ID"""
    return hashlib.md5(f"{time.time()}{os.urandom(8).hex()}".encode()).hexdigest()[:12]


def classify_position(bbox: BoundingBox, frame_width: int, frame_height: int) -> TextPosition:
    """Classify text position within frame"""
    center_x, center_y = bbox.center
    
    # Normalize to 0-1
    norm_x = center_x / max(frame_width, 1)
    norm_y = center_y / max(frame_height, 1)
    
    # Lower third detection
    if norm_y > 0.75:
        if 0.2 < norm_x < 0.8:
            return TextPosition.LOWER_THIRD
    
    # 3x3 grid classification
    if norm_y < 0.33:
        row = "top"
    elif norm_y < 0.66:
        row = "middle"
    else:
        row = "bottom"
    
    if norm_x < 0.33:
        col = "left"
    elif norm_x < 0.66:
        col = "center"
    else:
        col = "right"
    
    position_map = {
        ("top", "left"): TextPosition.TOP_LEFT,
        ("top", "center"): TextPosition.TOP_CENTER,
        ("top", "right"): TextPosition.TOP_RIGHT,
        ("middle", "left"): TextPosition.MIDDLE_LEFT,
        ("middle", "center"): TextPosition.MIDDLE_CENTER,
        ("middle", "right"): TextPosition.MIDDLE_RIGHT,
        ("bottom", "left"): TextPosition.BOTTOM_LEFT,
        ("bottom", "center"): TextPosition.BOTTOM_CENTER,
        ("bottom", "right"): TextPosition.BOTTOM_RIGHT,
    }
    
    return position_map.get((row, col), TextPosition.MIDDLE_CENTER)


def estimate_font_properties(
    text: str,
    bbox: BoundingBox,
    frame_height: int
) -> FontEstimate:
    """Estimate font properties from text and bounding box"""
    # Estimate font size from bbox height
    char_height = bbox.height
    estimated_size = max(8, int(char_height * 0.75))  # Rough heuristic
    
    # Check if uppercase
    is_uppercase = text.isupper() and len(text) > 2
    
    # Check for bold indicators (heuristic: wider characters)
    avg_char_width = bbox.width / max(len(text), 1)
    char_width_ratio = avg_char_width / max(char_height, 1)
    is_bold = char_width_ratio > 0.7
    
    # Font category estimation (very rough)
    # This would need actual font detection for accuracy
    category = FontCategory.SANS_SERIF  # Default assumption for commercials
    
    return FontEstimate(
        category=category,
        estimated_size_px=estimated_size,
        is_bold=is_bold,
        is_italic=False,
        is_uppercase=is_uppercase,
        estimated_weight=700 if is_bold else 400,
    )


def check_spelling(word: str, custom_allowed: Set[str] = None) -> Optional[str]:
    """
    Check if word is misspelled. Returns correction if found.
    
    This is a simplified checker. In production, integrate with:
    - PySpellChecker
    - LanguageTool
    - symspellpy for fast fuzzy matching
    """
    word_lower = word.lower().strip()
    custom_allowed = custom_allowed or set()
    
    # Skip checks for certain patterns
    if len(word_lower) < 2:
        return None
    if word_lower.isdigit():
        return None
    if word_lower in INDUSTRY_TERMS:
        return None
    if word_lower in custom_allowed:
        return None
    
    # Check common typos
    if word_lower in COMMON_TYPOS:
        correction = COMMON_TYPOS[word_lower]
        # Preserve original case
        if word.isupper():
            return correction.upper()
        elif word[0].isupper():
            return correction.capitalize()
        return correction
    
    return None


def check_grammar_patterns(text: str) -> List[Tuple[str, str, str]]:
    """
    Check text against grammar patterns.
    Returns list of (matched_text, description, severity)
    """
    issues = []
    
    for pattern, description, severity in GRAMMAR_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            issues.append((match.group(), description, severity))
    
    return issues


def calculate_reading_time(text: str, wpm: int = WordsmithConfig.AVERAGE_WPM) -> float:
    """Calculate estimated reading time in seconds"""
    word_count = len(text.split())
    return (word_count / wpm) * 60


def check_wcag_contrast(
    text_color: ColorRGB,
    bg_color: ColorRGB,
    font_size_px: int,
    is_bold: bool = False
) -> Tuple[WCAGLevel, float]:
    """
    Check WCAG contrast compliance.
    Returns (level achieved, contrast ratio)
    """
    ratio = text_color.contrast_ratio(bg_color)
    
    # Large text: 18pt+ or 14pt+ bold (roughly 24px+ or 19px+ bold)
    is_large = font_size_px >= 24 or (is_bold and font_size_px >= 19)
    
    if is_large:
        if ratio >= WordsmithConfig.WCAG_AAA_CONTRAST_LARGE:
            return WCAGLevel.AAA, ratio
        elif ratio >= WordsmithConfig.WCAG_AA_CONTRAST_LARGE:
            return WCAGLevel.AA, ratio
        elif ratio >= 2.0:
            return WCAGLevel.A, ratio
    else:
        if ratio >= WordsmithConfig.WCAG_AAA_CONTRAST_NORMAL:
            return WCAGLevel.AAA, ratio
        elif ratio >= WordsmithConfig.WCAG_AA_CONTRAST_NORMAL:
            return WCAGLevel.AA, ratio
        elif ratio >= 3.0:
            return WCAGLevel.A, ratio
    
    return WCAGLevel.FAIL, ratio


def simulate_color_blindness(color: ColorRGB, cb_type: ColorBlindnessType) -> ColorRGB:
    """
    Simulate how a color appears to someone with color blindness.
    Uses simplified transformation matrices.
    """
    r, g, b = color.r / 255.0, color.g / 255.0, color.b / 255.0
    
    # Transformation matrices (simplified)
    matrices = {
        ColorBlindnessType.PROTANOPIA: [
            [0.567, 0.433, 0.000],
            [0.558, 0.442, 0.000],
            [0.000, 0.242, 0.758],
        ],
        ColorBlindnessType.DEUTERANOPIA: [
            [0.625, 0.375, 0.000],
            [0.700, 0.300, 0.000],
            [0.000, 0.300, 0.700],
        ],
        ColorBlindnessType.TRITANOPIA: [
            [0.950, 0.050, 0.000],
            [0.000, 0.433, 0.567],
            [0.000, 0.475, 0.525],
        ],
        ColorBlindnessType.ACHROMATOPSIA: [
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
        ],
    }
    
    matrix = matrices.get(cb_type, matrices[ColorBlindnessType.DEUTERANOPIA])
    
    new_r = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b
    new_g = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b
    new_b = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b
    
    return ColorRGB(
        r=min(255, max(0, int(new_r * 255))),
        g=min(255, max(0, int(new_g * 255))),
        b=min(255, max(0, int(new_b * 255))),
    )


# =============================================================================
# SECTION 6: CORE WORDSMITH CLASS
# =============================================================================

class TheWordsmith:
    """
    THE WORDSMITH - Text Detection & QA Agent
    
    Detects, validates, and corrects all text appearing in commercial videos.
    Implements OCR, spelling/grammar checking, brand compliance, and WCAG accessibility.
    """
    
    def __init__(
        self,
        ocr_engine: OCREngine = OCREngine.HYBRID,
        brand_guidelines: Optional[BrandGuideline] = None,
        custom_allowed_words: Optional[Set[str]] = None,
        enable_languagetool: bool = False,
        languagetool_url: str = "http://localhost:8081/v2",
    ):
        self.ocr_engine = ocr_engine
        self.brand_guidelines = brand_guidelines
        self.custom_allowed_words = custom_allowed_words or set()
        self.enable_languagetool = enable_languagetool
        self.languagetool_url = languagetool_url
        
        # Metrics
        self.total_frames_processed = 0
        self.total_detections = 0
        self.total_errors_found = 0
        self.processing_times: List[float] = []
        
        logger.info(f"THE WORDSMITH initialized | OCR: {ocr_engine.value}")
    
    # -------------------------------------------------------------------------
    # VIDEO PROCESSING
    # -------------------------------------------------------------------------
    
    async def extract_keyframes(
        self,
        video_path: str,
        interval_sec: float = 1.0,
        output_dir: Optional[str] = None,
    ) -> List[Tuple[int, float, str]]:
        """
        Extract keyframes from video at specified interval.
        Returns list of (frame_number, timestamp, frame_path)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="wordsmith_frames_")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video info
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            raise ValueError(f"Cannot probe video: {video_path}")
        
        # Get duration and fps
        duration = float(video_info["format"]["duration"])
        fps = 30.0  # default
        for stream in video_info.get("streams", []):
            if stream["codec_type"] == "video":
                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                break
        
        # Extract frames
        frame_pattern = os.path.join(output_dir, "frame_%06d.png")
        extract_cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval_sec}",
            "-frame_pts", "1",
            frame_pattern,
            "-y", "-hide_banner", "-loglevel", "error"
        ]
        
        try:
            subprocess.run(extract_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg frame extraction failed: {e}")
            raise
        
        # Collect extracted frames
        frames = []
        for i, frame_file in enumerate(sorted(os.listdir(output_dir))):
            if frame_file.startswith("frame_") and frame_file.endswith(".png"):
                frame_path = os.path.join(output_dir, frame_file)
                frame_number = int(i * interval_sec * fps)
                timestamp = i * interval_sec
                frames.append((frame_number, timestamp, frame_path))
        
        logger.info(f"Extracted {len(frames)} keyframes from {video_path}")
        return frames
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata"""
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            duration = float(info["format"].get("duration", 0))
            width, height = 1920, 1080
            
            for stream in info.get("streams", []):
                if stream["codec_type"] == "video":
                    width = stream.get("width", 1920)
                    height = stream.get("height", 1080)
                    break
            
            return {
                "duration": duration,
                "width": width,
                "height": height,
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {"duration": 0, "width": 1920, "height": 1080}
    
    # -------------------------------------------------------------------------
    # OCR PROCESSING
    # -------------------------------------------------------------------------
    
    async def ocr_frame_tesseract(
        self,
        frame_path: str,
        frame_id: int,
        timestamp: float,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> List[TextDetection]:
        """
        Run Tesseract OCR on a single frame.
        Returns list of text detections.
        """
        detections = []
        
        try:
            # Use tesseract CLI with TSV output for bounding boxes
            cmd = [
                "tesseract", frame_path, "stdout",
                "--psm", "11",  # Sparse text
                "-c", "tessedit_pageseg_mode=11",
                "tsv"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"Tesseract returned non-zero: {result.stderr}")
                return detections
            
            # Parse TSV output
            lines = result.stdout.strip().split("\n")
            if len(lines) <= 1:
                return detections
            
            # Skip header
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 12:
                    continue
                
                try:
                    level = int(parts[0])
                    conf = float(parts[10])
                    text = parts[11].strip()
                    
                    # Skip low confidence or empty
                    if conf < 0 or not text or len(text) < WordsmithConfig.MIN_TEXT_LENGTH:
                        continue
                    
                    left = int(parts[6])
                    top = int(parts[7])
                    width = int(parts[8])
                    height = int(parts[9])
                    
                    bbox = BoundingBox(
                        x1=left,
                        y1=top,
                        x2=left + width,
                        y2=top + height
                    )
                    
                    position = classify_position(bbox, frame_width, frame_height)
                    font_est = estimate_font_properties(text, bbox, frame_height)
                    
                    detection = TextDetection(
                        frame_id=frame_id,
                        timestamp_sec=timestamp,
                        bbox=bbox,
                        text=text,
                        confidence=conf / 100.0,
                        ocr_engine=OCREngine.TESSERACT,
                        font_estimate=font_est,
                        position=position,
                    )
                    
                    detections.append(detection)
                    
                except (ValueError, IndexError) as e:
                    continue
            
        except subprocess.TimeoutExpired:
            logger.error(f"Tesseract timeout on {frame_path}")
        except FileNotFoundError:
            logger.error("Tesseract not found. Install with: apt install tesseract-ocr")
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
        
        return detections
    
    async def ocr_frame_paddleocr(
        self,
        frame_path: str,
        frame_id: int,
        timestamp: float,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> List[TextDetection]:
        """
        PaddleOCR wrapper (requires paddleocr package).
        Falls back to tesseract if not available.
        """
        try:
            # Try importing PaddleOCR
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            result = ocr.ocr(frame_path, cls=True)
            
            detections = []
            
            if not result or not result[0]:
                return detections
            
            for line in result[0]:
                points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # Convert polygon to bbox
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                bbox = BoundingBox(
                    x1=int(min(x_coords)),
                    y1=int(min(y_coords)),
                    x2=int(max(x_coords)),
                    y2=int(max(y_coords)),
                )
                
                position = classify_position(bbox, frame_width, frame_height)
                font_est = estimate_font_properties(text, bbox, frame_height)
                
                detection = TextDetection(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    ocr_engine=OCREngine.PADDLEOCR,
                    font_estimate=font_est,
                    position=position,
                )
                
                detections.append(detection)
            
            return detections
            
        except ImportError:
            logger.warning("PaddleOCR not installed. Falling back to Tesseract.")
            return await self.ocr_frame_tesseract(
                frame_path, frame_id, timestamp, frame_width, frame_height
            )
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return await self.ocr_frame_tesseract(
                frame_path, frame_id, timestamp, frame_width, frame_height
            )
    
    async def ocr_frame(
        self,
        frame_path: str,
        frame_id: int,
        timestamp: float,
        frame_width: int = 1920,
        frame_height: int = 1080,
        confidence_threshold: float = 0.65,
    ) -> List[TextDetection]:
        """
        Run OCR on a frame using configured engine.
        """
        if self.ocr_engine == OCREngine.PADDLEOCR:
            detections = await self.ocr_frame_paddleocr(
                frame_path, frame_id, timestamp, frame_width, frame_height
            )
        elif self.ocr_engine == OCREngine.HYBRID:
            # Try Tesseract first
            detections = await self.ocr_frame_tesseract(
                frame_path, frame_id, timestamp, frame_width, frame_height
            )
            
            # If low confidence, try PaddleOCR
            avg_conf = sum(d.confidence for d in detections) / max(len(detections), 1)
            if avg_conf < confidence_threshold and detections:
                paddle_detections = await self.ocr_frame_paddleocr(
                    frame_path, frame_id, timestamp, frame_width, frame_height
                )
                paddle_conf = sum(d.confidence for d in paddle_detections) / max(len(paddle_detections), 1)
                
                if paddle_conf > avg_conf:
                    detections = paddle_detections
        else:
            detections = await self.ocr_frame_tesseract(
                frame_path, frame_id, timestamp, frame_width, frame_height
            )
        
        # Filter by confidence
        detections = [d for d in detections if d.confidence >= confidence_threshold]
        
        return detections
    
    # -------------------------------------------------------------------------
    # SPELLING & GRAMMAR VALIDATION
    # -------------------------------------------------------------------------
    
    async def validate_spelling(
        self,
        text: str,
        frame_id: int,
        timestamp: float,
    ) -> List[ValidationError]:
        """Check spelling in text"""
        errors = []
        words = re.findall(r'\b[A-Za-z]+\b', text)
        
        for word in words:
            correction = check_spelling(word, self.custom_allowed_words)
            if correction:
                error = ValidationError(
                    error_id=generate_error_id(),
                    category=ErrorCategory.SPELLING,
                    severity=ValidationSeverity.MINOR,
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    original_text=text,
                    problematic_segment=word,
                    message=f"Possible misspelling: '{word}'",
                    suggestion=correction,
                    confidence=0.85,
                    auto_fixable=True,
                    rule_id="SPELLING_TYPO",
                )
                errors.append(error)
        
        return errors
    
    async def validate_grammar(
        self,
        text: str,
        frame_id: int,
        timestamp: float,
    ) -> List[ValidationError]:
        """Check grammar patterns in text"""
        errors = []
        
        # Pattern-based checks
        issues = check_grammar_patterns(text)
        
        for matched_text, description, severity in issues:
            sev_map = {
                "critical": ValidationSeverity.CRITICAL,
                "major": ValidationSeverity.MAJOR,
                "minor": ValidationSeverity.MINOR,
                "info": ValidationSeverity.INFO,
            }
            
            error = ValidationError(
                error_id=generate_error_id(),
                category=ErrorCategory.GRAMMAR,
                severity=sev_map.get(severity, ValidationSeverity.MINOR),
                frame_id=frame_id,
                timestamp_sec=timestamp,
                original_text=text,
                problematic_segment=matched_text,
                message=description,
                confidence=0.75,
                auto_fixable=False,
                rule_id="GRAMMAR_PATTERN",
            )
            errors.append(error)
        
        # LanguageTool integration (if enabled)
        if self.enable_languagetool:
            lt_errors = await self._check_languagetool(text, frame_id, timestamp)
            errors.extend(lt_errors)
        
        return errors
    
    async def _check_languagetool(
        self,
        text: str,
        frame_id: int,
        timestamp: float,
    ) -> List[ValidationError]:
        """Check text with LanguageTool API"""
        errors = []
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": text,
                    "language": "en-US",
                }
                
                async with session.post(
                    f"{self.languagetool_url}/check",
                    data=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for match in data.get("matches", []):
                            sev = ValidationSeverity.MINOR
                            if match.get("rule", {}).get("issueType") == "grammar":
                                sev = ValidationSeverity.MAJOR
                            
                            suggestion = None
                            if match.get("replacements"):
                                suggestion = match["replacements"][0].get("value")
                            
                            error = ValidationError(
                                error_id=generate_error_id(),
                                category=ErrorCategory.GRAMMAR,
                                severity=sev,
                                frame_id=frame_id,
                                timestamp_sec=timestamp,
                                original_text=text,
                                problematic_segment=text[match["offset"]:match["offset"] + match["length"]],
                                message=match.get("message", "Grammar issue"),
                                suggestion=suggestion,
                                confidence=0.8,
                                auto_fixable=suggestion is not None,
                                rule_id=match.get("rule", {}).get("id", "LT_RULE"),
                            )
                            errors.append(error)
        
        except Exception as e:
            logger.debug(f"LanguageTool check failed: {e}")
        
        return errors
    
    # -------------------------------------------------------------------------
    # BRAND COMPLIANCE VALIDATION
    # -------------------------------------------------------------------------
    
    async def validate_brand_compliance(
        self,
        text: str,
        frame_id: int,
        timestamp: float,
    ) -> List[ValidationError]:
        """Check brand compliance rules"""
        errors = []
        text_lower = text.lower()
        
        if not self.brand_guidelines:
            return errors
        
        bg = self.brand_guidelines
        
        # Check forbidden words
        all_forbidden = bg.forbidden_words | DEFAULT_FORBIDDEN_WORDS
        for word in all_forbidden:
            if word.lower() in text_lower:
                error = ValidationError(
                    error_id=generate_error_id(),
                    category=ErrorCategory.FORBIDDEN_WORD,
                    severity=ValidationSeverity.CRITICAL,
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    original_text=text,
                    problematic_segment=word,
                    message=f"Forbidden word/phrase detected: '{word}'",
                    confidence=0.95,
                    auto_fixable=False,
                    rule_id="BRAND_FORBIDDEN",
                )
                errors.append(error)
        
        # Check competitor names
        if bg.competitor_check_enabled:
            all_competitors = bg.competitor_names | DEFAULT_COMPETITOR_TERMS
            for comp in all_competitors:
                if comp.lower() in text_lower:
                    error = ValidationError(
                        error_id=generate_error_id(),
                        category=ErrorCategory.COMPETITOR,
                        severity=ValidationSeverity.CRITICAL,
                        frame_id=frame_id,
                        timestamp_sec=timestamp,
                        original_text=text,
                        problematic_segment=comp,
                        message=f"Competitor mention detected: '{comp}'",
                        confidence=0.9,
                        auto_fixable=False,
                        rule_id="BRAND_COMPETITOR",
                    )
                    errors.append(error)
        
        # Check preferred terms
        for wrong, right in bg.preferred_terms.items():
            if wrong.lower() in text_lower:
                error = ValidationError(
                    error_id=generate_error_id(),
                    category=ErrorCategory.BRAND_VIOLATION,
                    severity=ValidationSeverity.MAJOR,
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    original_text=text,
                    problematic_segment=wrong,
                    message=f"Use '{right}' instead of '{wrong}'",
                    suggestion=text.replace(wrong, right),
                    confidence=0.85,
                    auto_fixable=True,
                    rule_id="BRAND_PREFERRED_TERM",
                )
                errors.append(error)
        
        # Check trademark symbols
        if bg.trademark_symbols_required:
            for tm in bg.trademark_terms:
                if tm.lower() in text_lower:
                    # Check if ® or ™ follows
                    pattern = rf'\b{re.escape(tm)}\b(?!\s*[®™])'
                    if re.search(pattern, text, re.IGNORECASE):
                        error = ValidationError(
                            error_id=generate_error_id(),
                            category=ErrorCategory.TRADEMARK,
                            severity=ValidationSeverity.MAJOR,
                            frame_id=frame_id,
                            timestamp_sec=timestamp,
                            original_text=text,
                            problematic_segment=tm,
                            message=f"Trademark '{tm}' requires ® or ™ symbol",
                            suggestion=f"{tm}®",
                            confidence=0.9,
                            auto_fixable=True,
                            rule_id="BRAND_TRADEMARK_SYMBOL",
                        )
                        errors.append(error)
        
        return errors
    
    async def validate_legal_disclaimers(
        self,
        all_text: str,
    ) -> List[ValidationError]:
        """Check if required legal disclaimers are present"""
        errors = []
        
        if not self.brand_guidelines or not self.brand_guidelines.industry:
            return errors
        
        industry = self.brand_guidelines.industry.lower()
        required = LEGAL_DISCLAIMERS.get(industry, [])
        required.extend(self.brand_guidelines.required_disclaimers)
        
        text_lower = all_text.lower()
        
        for disclaimer in required:
            if disclaimer.lower() not in text_lower:
                error = ValidationError(
                    error_id=generate_error_id(),
                    category=ErrorCategory.LEGAL_MISSING,
                    severity=ValidationSeverity.CRITICAL,
                    original_text="[Video text]",
                    problematic_segment="",
                    message=f"Required disclaimer missing: '{disclaimer}'",
                    suggestion=f"Add text containing: '{disclaimer}'",
                    confidence=0.95,
                    auto_fixable=False,
                    rule_id="LEGAL_DISCLAIMER_MISSING",
                )
                errors.append(error)
        
        return errors
    
    # -------------------------------------------------------------------------
    # ACCESSIBILITY VALIDATION
    # -------------------------------------------------------------------------
    
    async def validate_accessibility(
        self,
        detection: TextDetection,
        target_level: WCAGLevel = WCAGLevel.AA,
    ) -> AccessibilityResult:
        """Validate WCAG accessibility for a text detection"""
        issues = []
        
        # Default colors if not detected
        text_color = detection.text_color or ColorRGB(r=255, g=255, b=255)
        bg_color = detection.background_color or ColorRGB(r=0, g=0, b=0)
        
        # Contrast check
        font_size = detection.font_estimate.estimated_size_px
        is_bold = detection.font_estimate.is_bold
        wcag_level, contrast_ratio = check_wcag_contrast(
            text_color, bg_color, font_size, is_bold
        )
        
        # Large text determination
        is_large = font_size >= 24 or (is_bold and font_size >= 19)
        
        # Font size check
        meets_min_size = font_size >= WordsmithConfig.MIN_FONT_SIZE_PX
        if not meets_min_size:
            issues.append(f"Font size {font_size}px below minimum {WordsmithConfig.MIN_FONT_SIZE_PX}px")
        
        # Characters per line
        chars_per_line = len(detection.text)
        if chars_per_line > WordsmithConfig.MAX_CHARS_PER_LINE:
            issues.append(f"Line length {chars_per_line} chars exceeds {WordsmithConfig.MAX_CHARS_PER_LINE}")
        
        # Reading time (estimate display duration as 2 seconds)
        display_duration = 2.0  # Would need actual frame timing
        min_time = max(
            WordsmithConfig.MIN_DISPLAY_TIME_SEC,
            calculate_reading_time(detection.text)
        )
        reading_time_adequate = display_duration >= min_time
        if not reading_time_adequate:
            issues.append(f"Display time may be insufficient for reading")
        
        # Color blindness simulation
        color_blind_safe = {}
        for cb_type in ColorBlindnessType:
            sim_text = simulate_color_blindness(text_color, cb_type)
            sim_bg = simulate_color_blindness(bg_color, cb_type)
            cb_level, _ = check_wcag_contrast(sim_text, sim_bg, font_size, is_bold)
            color_blind_safe[cb_type] = cb_level != WCAGLevel.FAIL
            if not color_blind_safe[cb_type]:
                issues.append(f"Low contrast for {cb_type.value} color blindness")
        
        # Determine pass levels
        passes_aa = wcag_level in [WCAGLevel.AA, WCAGLevel.AAA]
        passes_aaa = wcag_level == WCAGLevel.AAA
        
        if wcag_level == WCAGLevel.FAIL:
            issues.append(f"Contrast ratio {contrast_ratio:.2f}:1 fails WCAG requirements")
        
        return AccessibilityResult(
            wcag_level=wcag_level,
            passes_aa=passes_aa,
            passes_aaa=passes_aaa,
            contrast_ratio=contrast_ratio,
            text_color=text_color,
            background_color=bg_color,
            font_size_px=font_size,
            is_large_text=is_large,
            meets_min_size=meets_min_size,
            chars_per_line=chars_per_line,
            display_duration_sec=display_duration,
            reading_time_adequate=reading_time_adequate,
            color_blind_safe=color_blind_safe,
            issues=issues,
        )
    
    def accessibility_to_errors(
        self,
        result: AccessibilityResult,
        frame_id: int,
        timestamp: float,
        original_text: str,
        target_level: WCAGLevel = WCAGLevel.AA,
    ) -> List[ValidationError]:
        """Convert accessibility result to validation errors"""
        errors = []
        
        # Check target level not met
        target_met = (
            (target_level == WCAGLevel.A) or
            (target_level == WCAGLevel.AA and result.passes_aa) or
            (target_level == WCAGLevel.AAA and result.passes_aaa)
        )
        
        if not target_met:
            sev = ValidationSeverity.MAJOR if target_level == WCAGLevel.AA else ValidationSeverity.MINOR
            error = ValidationError(
                error_id=generate_error_id(),
                category=ErrorCategory.ACCESSIBILITY,
                severity=sev,
                frame_id=frame_id,
                timestamp_sec=timestamp,
                original_text=original_text,
                problematic_segment="",
                message=f"Does not meet WCAG {target_level.value}: contrast {result.contrast_ratio:.2f}:1",
                confidence=0.95,
                auto_fixable=False,
                rule_id=f"WCAG_{target_level.value}_CONTRAST",
            )
            errors.append(error)
        
        if not result.meets_min_size:
            error = ValidationError(
                error_id=generate_error_id(),
                category=ErrorCategory.FONT_SIZE,
                severity=ValidationSeverity.MAJOR,
                frame_id=frame_id,
                timestamp_sec=timestamp,
                original_text=original_text,
                problematic_segment="",
                message=f"Font size {result.font_size_px}px below minimum {WordsmithConfig.MIN_FONT_SIZE_PX}px",
                confidence=0.9,
                auto_fixable=False,
                rule_id="WCAG_FONT_SIZE",
            )
            errors.append(error)
        
        if result.chars_per_line > WordsmithConfig.MAX_CHARS_PER_LINE:
            error = ValidationError(
                error_id=generate_error_id(),
                category=ErrorCategory.READABILITY,
                severity=ValidationSeverity.MINOR,
                frame_id=frame_id,
                timestamp_sec=timestamp,
                original_text=original_text,
                problematic_segment="",
                message=f"Line too long: {result.chars_per_line} chars (max {WordsmithConfig.MAX_CHARS_PER_LINE})",
                confidence=0.85,
                auto_fixable=False,
                rule_id="WCAG_LINE_LENGTH",
            )
            errors.append(error)
        
        return errors
    
    # -------------------------------------------------------------------------
    # CORRECTION PIPELINE
    # -------------------------------------------------------------------------
    
    def generate_corrections(
        self,
        errors: List[ValidationError],
    ) -> List[CorrectionSuggestion]:
        """Generate correction suggestions from validation errors"""
        corrections = []
        
        for error in errors:
            if error.suggestion:
                # Determine if auto-apply is safe
                auto_apply = (
                    error.auto_fixable and
                    error.severity == ValidationSeverity.MINOR and
                    error.confidence >= 0.85
                )
                
                correction = CorrectionSuggestion(
                    error_id=error.error_id,
                    original=error.problematic_segment,
                    corrected=error.suggestion,
                    confidence=error.confidence,
                    auto_apply=auto_apply,
                    requires_approval=not auto_apply,
                    reasoning=error.message,
                )
                corrections.append(correction)
        
        # Sort by confidence (highest first)
        corrections.sort(key=lambda x: x.confidence, reverse=True)
        
        return corrections
    
    # -------------------------------------------------------------------------
    # MAIN VALIDATION PIPELINE
    # -------------------------------------------------------------------------
    
    async def validate(
        self,
        request: TextValidationRequest,
    ) -> TextValidationResult:
        """
        Main validation pipeline.
        
        Process:
        1. Extract keyframes from video
        2. Run OCR on each frame
        3. Validate spelling/grammar
        4. Check brand compliance
        5. Validate accessibility
        6. Generate corrections
        7. Compile report
        """
        start_time = time.time()
        
        logger.info(f"Starting validation: {request.video_path}")
        
        # Initialize result
        result = TextValidationResult(
            video_path=request.video_path,
            ocr_engine_used=request.ocr_engine,
        )
        
        try:
            # Get video info
            video_info = await self.get_video_info(request.video_path)
            result.video_duration_sec = video_info["duration"]
            result.video_resolution = (video_info["width"], video_info["height"])
            
            # Determine check flags
            check_all = CheckFlag.ALL in request.check_flags
            check_spelling = check_all or CheckFlag.SPELLING in request.check_flags
            check_grammar = check_all or CheckFlag.GRAMMAR in request.check_flags
            check_brand = check_all or CheckFlag.BRAND in request.check_flags
            check_accessibility = check_all or CheckFlag.ACCESSIBILITY in request.check_flags
            check_legal = check_all or CheckFlag.LEGAL in request.check_flags
            
            # Update brand guidelines if provided
            if request.brand_guidelines:
                self.brand_guidelines = request.brand_guidelines
            
            # Update custom allowed words
            if request.custom_allowed_words:
                self.custom_allowed_words.update(request.custom_allowed_words)
            if request.custom_forbidden_words and self.brand_guidelines:
                self.brand_guidelines.forbidden_words.update(request.custom_forbidden_words)
            
            # Extract keyframes
            frames = await self.extract_keyframes(
                request.video_path,
                interval_sec=request.keyframe_interval_sec,
            )
            
            all_detections: List[TextDetection] = []
            all_errors: List[ValidationError] = []
            all_accessibility: List[AccessibilityResult] = []
            frame_analyses: List[FrameAnalysis] = []
            all_text_combined = ""
            
            # Process each frame
            for frame_id, timestamp, frame_path in frames:
                frame_start = time.time()
                
                # OCR
                detections = await self.ocr_frame(
                    frame_path=frame_path,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_width=video_info["width"],
                    frame_height=video_info["height"],
                    confidence_threshold=request.ocr_confidence_threshold,
                )
                
                frame_errors: List[ValidationError] = []
                frame_accessibility: List[AccessibilityResult] = []
                
                for detection in detections:
                    all_text_combined += " " + detection.text
                    
                    # Spelling
                    if check_spelling:
                        spell_errors = await self.validate_spelling(
                            detection.text, frame_id, timestamp
                        )
                        frame_errors.extend(spell_errors)
                    
                    # Grammar
                    if check_grammar:
                        grammar_errors = await self.validate_grammar(
                            detection.text, frame_id, timestamp
                        )
                        frame_errors.extend(grammar_errors)
                    
                    # Brand compliance
                    if check_brand:
                        brand_errors = await self.validate_brand_compliance(
                            detection.text, frame_id, timestamp
                        )
                        frame_errors.extend(brand_errors)
                    
                    # Accessibility
                    if check_accessibility:
                        access_result = await self.validate_accessibility(
                            detection, request.target_wcag_level
                        )
                        frame_accessibility.append(access_result)
                        
                        access_errors = self.accessibility_to_errors(
                            access_result, frame_id, timestamp,
                            detection.text, request.target_wcag_level
                        )
                        frame_errors.extend(access_errors)
                
                # Frame analysis
                processing_time_ms = (time.time() - frame_start) * 1000
                avg_conf = sum(d.confidence for d in detections) / max(len(detections), 1)
                
                frame_analysis = FrameAnalysis(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    frame_path=frame_path,
                    text_detections=detections,
                    total_text_regions=len(detections),
                    validation_errors=frame_errors,
                    accessibility_results=frame_accessibility,
                    ocr_confidence_avg=avg_conf,
                    processing_time_ms=processing_time_ms,
                )
                
                frame_analyses.append(frame_analysis)
                all_detections.extend(detections)
                all_errors.extend(frame_errors)
                all_accessibility.extend(frame_accessibility)
                
                # Clean up frame file
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            # Legal disclaimer check (needs all text)
            if check_legal:
                legal_errors = await self.validate_legal_disclaimers(all_text_combined)
                all_errors.extend(legal_errors)
            
            # Generate corrections
            corrections = []
            if request.include_suggestions:
                corrections = self.generate_corrections(all_errors)
            
            # Apply auto-fixes if enabled
            if request.auto_fix_minor:
                for corr in corrections:
                    if corr.auto_apply:
                        corr.requires_approval = False
            
            # Calculate summary
            summary = self._compute_summary(all_detections, all_errors, all_accessibility)
            
            # Determine overall WCAG level
            if all_accessibility:
                pass_aa = all(a.passes_aa for a in all_accessibility)
                pass_aaa = all(a.passes_aaa for a in all_accessibility)
                if pass_aaa:
                    overall_wcag = WCAGLevel.AAA
                elif pass_aa:
                    overall_wcag = WCAGLevel.AA
                elif any(a.wcag_level != WCAGLevel.FAIL for a in all_accessibility):
                    overall_wcag = WCAGLevel.A
                else:
                    overall_wcag = WCAGLevel.FAIL
            else:
                overall_wcag = WCAGLevel.AA  # No text = passes
            
            # Populate result
            result.frame_analyses = frame_analyses
            result.all_detections = all_detections
            result.all_errors = all_errors
            result.corrections = corrections
            result.summary = summary
            result.overall_wcag_level = overall_wcag
            result.accessibility_details = all_accessibility
            
            # Generate report
            if request.generate_report:
                result.report_markdown = self._generate_report(result)
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result.success = False
            result.error_message = str(e)
        
        # Finalize metrics
        result.processing_time_sec = time.time() - start_time
        result.estimated_cost = self._estimate_cost(result)
        
        logger.info(
            f"Validation complete: {len(result.all_detections)} detections, "
            f"{len(result.all_errors)} errors, {result.processing_time_sec:.2f}s"
        )
        
        return result
    
    def _compute_summary(
        self,
        detections: List[TextDetection],
        errors: List[ValidationError],
        accessibility: List[AccessibilityResult],
    ) -> ValidationSummary:
        """Compute validation summary statistics"""
        # Count by severity
        critical = sum(1 for e in errors if e.severity == ValidationSeverity.CRITICAL)
        major = sum(1 for e in errors if e.severity == ValidationSeverity.MAJOR)
        minor = sum(1 for e in errors if e.severity == ValidationSeverity.MINOR)
        info = sum(1 for e in errors if e.severity == ValidationSeverity.INFO)
        
        # Count by category
        by_category: Dict[str, int] = {}
        for e in errors:
            cat = e.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # Accessibility stats
        if accessibility:
            pass_rate = sum(1 for a in accessibility if a.passes_aa) / len(accessibility)
            avg_contrast = sum(a.contrast_ratio for a in accessibility) / len(accessibility)
        else:
            pass_rate = 1.0
            avg_contrast = 21.0
        
        # Auto-fixable count
        auto_fixable = sum(1 for e in errors if e.auto_fixable)
        suggestions = sum(1 for e in errors if e.suggestion)
        
        # Quality score (100 - penalties)
        score = 100.0
        score -= critical * 20  # -20 per critical
        score -= major * 10    # -10 per major
        score -= minor * 2     # -2 per minor
        score -= info * 0.5    # -0.5 per info
        score -= (1 - pass_rate) * 20  # -20 for no accessibility
        score = max(0, min(100, score))
        
        return ValidationSummary(
            total_frames_analyzed=len(set(d.frame_id for d in detections)) if detections else 0,
            total_text_regions=len(detections),
            total_errors=len(errors),
            critical_errors=critical,
            major_errors=major,
            minor_errors=minor,
            info_items=info,
            errors_by_category=by_category,
            accessibility_pass_rate=pass_rate,
            avg_contrast_ratio=avg_contrast,
            auto_fixable_count=auto_fixable,
            suggestions_count=suggestions,
            quality_score=score,
        )
    
    def _estimate_cost(self, result: TextValidationResult) -> float:
        """Estimate processing cost"""
        frame_cost = len(result.frame_analyses) * WordsmithConfig.COST_PER_FRAME_OCR
        validation_cost = len(result.all_errors) * WordsmithConfig.COST_PER_VALIDATION_CALL
        return frame_cost + validation_cost
    
    def _generate_report(self, result: TextValidationResult) -> str:
        """Generate markdown report"""
        lines = [
            "# THE WORDSMITH - Text Validation Report",
            "",
            f"**Video:** `{result.video_path}`",
            f"**Duration:** {result.video_duration_sec:.2f}s",
            f"**Resolution:** {result.video_resolution[0]}x{result.video_resolution[1]}",
            f"**Processed:** {result.timestamp}",
            f"**Processing Time:** {result.processing_time_sec:.2f}s",
            f"**Estimated Cost:** ${result.estimated_cost:.4f}",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Frames Analyzed | {result.summary.total_frames_analyzed} |",
            f"| Text Regions | {result.summary.total_text_regions} |",
            f"| Total Errors | {result.summary.total_errors} |",
            f"| Quality Score | {result.summary.quality_score:.1f}/100 |",
            f"| WCAG Level | {result.overall_wcag_level.value} |",
            f"| Accessibility Pass Rate | {result.summary.accessibility_pass_rate*100:.1f}% |",
            "",
            "### Error Breakdown",
            "",
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| 🔴 Critical | {result.summary.critical_errors} |",
            f"| 🟠 Major | {result.summary.major_errors} |",
            f"| 🟡 Minor | {result.summary.minor_errors} |",
            f"| ℹ️ Info | {result.summary.info_items} |",
            "",
        ]
        
        # Errors by category
        if result.summary.errors_by_category:
            lines.extend([
                "### Errors by Category",
                "",
                "| Category | Count |",
                "|----------|-------|",
            ])
            for cat, count in sorted(result.summary.errors_by_category.items()):
                lines.append(f"| {cat} | {count} |")
            lines.append("")
        
        # Critical errors detail
        critical_errors = [e for e in result.all_errors if e.severity == ValidationSeverity.CRITICAL]
        if critical_errors:
            lines.extend([
                "---",
                "",
                "## 🔴 Critical Issues",
                "",
            ])
            for i, error in enumerate(critical_errors, 1):
                lines.extend([
                    f"### {i}. {error.category.value.upper()}",
                    f"- **Frame:** {error.frame_id} ({error.timestamp_sec:.2f}s)",
                    f"- **Issue:** {error.message}",
                    f"- **Text:** `{error.problematic_segment}`",
                ])
                if error.suggestion:
                    lines.append(f"- **Suggestion:** `{error.suggestion}`")
                lines.append("")
        
        # Corrections
        if result.corrections:
            lines.extend([
                "---",
                "",
                "## Suggested Corrections",
                "",
                "| Original | Corrected | Confidence | Auto-Apply |",
                "|----------|-----------|------------|------------|",
            ])
            for corr in result.corrections[:20]:
                auto = "✅" if corr.auto_apply else "❌"
                lines.append(
                    f"| `{corr.original[:30]}` | `{corr.corrected[:30]}` | "
                    f"{corr.confidence*100:.0f}% | {auto} |"
                )
            if len(result.corrections) > 20:
                lines.append(f"| ... | +{len(result.corrections)-20} more | | |")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "*Generated by THE WORDSMITH (Agent 7.25) | Barrios A2I*",
        ])
        
        return "\n".join(lines)


# =============================================================================
# SECTION 7: FACTORY FUNCTION
# =============================================================================

def create_wordsmith(
    ocr_engine: OCREngine = OCREngine.HYBRID,
    brand_guidelines: Optional[BrandGuideline] = None,
    custom_allowed_words: Optional[Set[str]] = None,
    enable_languagetool: bool = False,
    languagetool_url: str = "http://localhost:8081/v2",
) -> TheWordsmith:
    """
    Factory function to create a configured Wordsmith instance.
    
    Args:
        ocr_engine: OCR engine to use (TESSERACT, PADDLEOCR, or HYBRID)
        brand_guidelines: Brand compliance rules
        custom_allowed_words: Additional words to allow in spell check
        enable_languagetool: Enable LanguageTool API for grammar
        languagetool_url: LanguageTool API endpoint
    
    Returns:
        Configured TheWordsmith instance
    """
    return TheWordsmith(
        ocr_engine=ocr_engine,
        brand_guidelines=brand_guidelines,
        custom_allowed_words=custom_allowed_words,
        enable_languagetool=enable_languagetool,
        languagetool_url=languagetool_url,
    )


# =============================================================================
# SECTION 8: NEXUS REGISTRATION
# =============================================================================

NEXUS_REGISTRATION = {
    "agent_id": "agent_7.25",
    "name": "THE WORDSMITH",
    "version": "1.0.0",
    "phase": "VORTEX",
    "handler": "wordsmith.validate",
    "input_schema": "TextValidationRequest",
    "output_schema": "TextValidationResult",
    "description": "Text Detection & QA Agent - OCR, spelling/grammar, brand compliance, WCAG accessibility",
    "cost_target": {"min": 0.10, "max": 0.20, "unit": "USD/video"},
    "latency_target": {"min": 5.0, "max": 12.0, "unit": "seconds"},
    "capabilities": [
        "text_detection",
        "ocr_multi_engine",
        "spelling_check",
        "grammar_check",
        "brand_compliance",
        "wcag_accessibility",
        "auto_correction",
    ],
    "dependencies": {
        "required": ["ffmpeg", "tesseract"],
        "optional": ["paddleocr", "languagetool"],
    },
    "author": "Barrios A2I",
    "created": "2026-01-19",
}


# =============================================================================
# SECTION 9: CLI INTERFACE
# =============================================================================

async def main():
    """CLI entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="THE WORDSMITH - Text Detection & QA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python the_wordsmith.py video.mp4
  python the_wordsmith.py video.mp4 --brand "Acme Corp"
  python the_wordsmith.py video.mp4 --wcag AAA --interval 0.5
  python the_wordsmith.py video.mp4 --forbidden "competitor,banned"
        """,
    )
    
    parser.add_argument(
        "video_path",
        help="Path to video file"
    )
    parser.add_argument(
        "--brand",
        help="Company/brand name for compliance checking"
    )
    parser.add_argument(
        "--industry",
        choices=["finance", "pharma", "alcohol", "gambling", "auto"],
        help="Industry for legal disclaimer checking"
    )
    parser.add_argument(
        "--forbidden",
        help="Comma-separated list of forbidden words"
    )
    parser.add_argument(
        "--competitors",
        help="Comma-separated list of competitor names"
    )
    parser.add_argument(
        "--wcag",
        choices=["A", "AA", "AAA"],
        default="AA",
        help="Target WCAG compliance level (default: AA)"
    )
    parser.add_argument(
        "--ocr",
        choices=["tesseract", "paddleocr", "hybrid"],
        default="hybrid",
        help="OCR engine to use (default: hybrid)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Keyframe extraction interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--languagetool",
        action="store_true",
        help="Enable LanguageTool grammar checking"
    )
    parser.add_argument(
        "--output",
        help="Output path for report markdown"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Build brand guidelines if provided
    brand_guidelines = None
    if args.brand:
        forbidden = set()
        if args.forbidden:
            forbidden = set(w.strip() for w in args.forbidden.split(","))
        
        competitors = set()
        if args.competitors:
            competitors = set(w.strip() for w in args.competitors.split(","))
        
        brand_guidelines = BrandGuideline(
            company_name=args.brand,
            forbidden_words=forbidden,
            competitor_names=competitors,
            industry=args.industry,
        )
    
    # Map OCR engine
    ocr_map = {
        "tesseract": OCREngine.TESSERACT,
        "paddleocr": OCREngine.PADDLEOCR,
        "hybrid": OCREngine.HYBRID,
    }
    
    # Map WCAG level
    wcag_map = {
        "A": WCAGLevel.A,
        "AA": WCAGLevel.AA,
        "AAA": WCAGLevel.AAA,
    }
    
    # Create wordsmith
    wordsmith = create_wordsmith(
        ocr_engine=ocr_map[args.ocr],
        brand_guidelines=brand_guidelines,
        enable_languagetool=args.languagetool,
    )
    
    # Build request
    request = TextValidationRequest(
        video_path=args.video_path,
        brand_guidelines=brand_guidelines,
        ocr_engine=ocr_map[args.ocr],
        keyframe_interval_sec=args.interval,
        target_wcag_level=wcag_map[args.wcag],
        generate_report=True,
    )
    
    # Run validation
    print(f"\n🔍 THE WORDSMITH - Text Detection & QA Agent")
    print(f"{'='*50}")
    print(f"Video: {args.video_path}")
    print(f"OCR Engine: {args.ocr}")
    print(f"WCAG Target: {args.wcag}")
    if args.brand:
        print(f"Brand: {args.brand}")
    print(f"{'='*50}\n")
    
    result = await wordsmith.validate(request)
    
    # Output
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        # Summary output
        print(f"\n✅ Validation Complete!")
        print(f"{'='*50}")
        print(f"Processing Time: {result.processing_time_sec:.2f}s")
        print(f"Estimated Cost: ${result.estimated_cost:.4f}")
        print(f"Text Regions Found: {result.summary.total_text_regions}")
        print(f"Total Errors: {result.summary.total_errors}")
        print(f"  🔴 Critical: {result.summary.critical_errors}")
        print(f"  🟠 Major: {result.summary.major_errors}")
        print(f"  🟡 Minor: {result.summary.minor_errors}")
        print(f"  ℹ️ Info: {result.summary.info_items}")
        print(f"Quality Score: {result.summary.quality_score:.1f}/100")
        print(f"WCAG Level: {result.overall_wcag_level.value}")
        print(f"Accessibility Pass Rate: {result.summary.accessibility_pass_rate*100:.1f}%")
        
        if result.corrections:
            print(f"\n📝 Top Corrections ({len(result.corrections)} total):")
            for corr in result.corrections[:5]:
                auto = "🤖" if corr.auto_apply else "👤"
                print(f"  {auto} '{corr.original}' → '{corr.corrected}' ({corr.confidence*100:.0f}%)")
        
        if result.report_markdown:
            if args.output:
                with open(args.output, "w") as f:
                    f.write(result.report_markdown)
                print(f"\n📄 Report saved to: {args.output}")
            else:
                print(f"\n📄 Use --output to save the full report")
    
    # Exit code based on critical errors
    if result.summary.critical_errors > 0:
        sys.exit(2)
    elif result.summary.major_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
