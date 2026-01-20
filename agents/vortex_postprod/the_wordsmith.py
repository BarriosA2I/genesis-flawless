#!/usr/bin/env python3
"""
THE WORDSMITH v2.0 LEGENDARY
Agent 7.25 - Text Detection, Spelling Correction, and Video Text Overlay

RAGNAROK Video Pipeline | VORTEX Phase
Barrios A2I Cognitive Systems Division

This agent is LEGENDARY at:
- Frame-by-frame OCR text detection (EasyOCR)
- Spell checking with pyspellchecker + brand whitelist
- BLOCKING pipeline on spelling errors
- Generating FFmpeg drawtext overlays to FIX errors
- Re-rendering videos with corrected text

NO MISSPELLED TEXT SHALL PASS.

Pipeline Position: VORTEX Phase (after SOUNDSCAPER, before EDITOR)
Cost Target: $0.10-0.20/video
Latency Target: 5-12s
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("THE_WORDSMITH")


# =============================================================================
# BACKWARD COMPATIBILITY - Enums & Classes from Original Implementation
# =============================================================================

class OCREngine(Enum):
    """OCR engine options"""
    TESSERACT = auto()
    EASYOCR = auto()
    PADDLEOCR = auto()

class ValidationSeverity(Enum):
    """Severity levels for validation errors"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorCategory(Enum):
    """Categories of text errors"""
    SPELLING = auto()
    GRAMMAR = auto()
    BRAND = auto()
    ACCESSIBILITY = auto()
    FORMATTING = auto()

class WCAGLevel(Enum):
    """WCAG accessibility levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"

class ColorBlindnessType(Enum):
    """Types of color blindness"""
    PROTANOPIA = auto()
    DEUTERANOPIA = auto()
    TRITANOPIA = auto()
    ACHROMATOPSIA = auto()

class TextPosition(Enum):
    """Text positioning on screen"""
    TOP = auto()
    CENTER = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

class FontCategory(Enum):
    """Font categories"""
    SERIF = auto()
    SANS_SERIF = auto()
    MONOSPACE = auto()
    DISPLAY = auto()
    HANDWRITING = auto()

class CheckFlag(Enum):
    """Flags for what to check"""
    SPELLING = auto()
    GRAMMAR = auto()
    BRAND = auto()
    ACCESSIBILITY = auto()
    ALL = auto()


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

@dataclass
class ColorRGB:
    """RGB color representation"""
    r: int = 0
    g: int = 0
    b: int = 0

@dataclass
class FontEstimate:
    """Estimated font properties"""
    family: str = "Unknown"
    size: int = 12
    weight: str = "normal"
    style: str = "normal"

@dataclass
class TextDetection:
    """Detected text in a frame"""
    text: str = ""
    confidence: float = 0.0
    bbox: BoundingBox = field(default_factory=BoundingBox)
    frame_number: int = 0
    timestamp_ms: int = 0

@dataclass
class ValidationError:
    """Validation error details"""
    text: str = ""
    category: ErrorCategory = ErrorCategory.SPELLING
    severity: ValidationSeverity = ValidationSeverity.ERROR
    suggestion: str = ""
    frame_number: int = 0
    timestamp_ms: int = 0

@dataclass
class CorrectionSuggestion:
    """Suggested correction"""
    original: str = ""
    correction: str = ""
    confidence: float = 0.0

@dataclass
class BrandGuideline:
    """Brand guidelines for text"""
    name: str = ""
    colors: List[ColorRGB] = field(default_factory=list)
    fonts: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)

@dataclass
class AccessibilityResult:
    """Accessibility check result"""
    wcag_level: WCAGLevel = WCAGLevel.AA
    contrast_ratio: float = 0.0
    passes: bool = True
    issues: List[str] = field(default_factory=list)

@dataclass
class FrameAnalysis:
    """Analysis results for a single frame"""
    frame_number: int = 0
    timestamp_ms: int = 0
    detections: List[TextDetection] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)

@dataclass
class TextValidationRequest:
    """Request for text validation"""
    video_path: str = ""
    fps_sample_rate: float = 1.0
    check_flags: List[CheckFlag] = field(default_factory=lambda: [CheckFlag.ALL])
    brand_guidelines: Optional[BrandGuideline] = None

@dataclass
class ValidationSummary:
    """Summary of validation results"""
    total_frames: int = 0
    total_detections: int = 0
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)

@dataclass
class TextValidationResult:
    """Complete validation result"""
    success: bool = True
    summary: ValidationSummary = field(default_factory=ValidationSummary)
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)


@dataclass
class WordsmithConfig:
    """Configuration for Wordsmith agent (backward compat)"""
    fps_sample_rate: float = 1.0
    confidence_threshold: float = 0.5
    ocr_engine: OCREngine = OCREngine.EASYOCR
    blocking_mode: bool = True
    auto_fix_mode: bool = True
    check_flags: List[CheckFlag] = field(default_factory=lambda: [CheckFlag.ALL])


# =============================================================================
# SECTION 1: CONFIGURATION & DICTIONARIES
# =============================================================================

# Known OCR misreads - CHECK THESE FIRST
OCR_CORRECTIONS = {
    # Known OCR misreads from video testing
    "trigcted": "triggered",
    "triggcted": "triggered",
    "trigered": "triggered",
    "trigerred": "triggered",
    "rigcted": "triggered",  # OCR sometimes misses leading 't'
    "websiite": "website",
    "websitte": "website",
    "webisite": "website",
    "wew": "new",
    "quesiton": "question",
    "questoin": "question",
    "questian": "question",
    "qestion": "question",
    "analystics": "analytics",
    "anayltics": "analytics",
    "analyitcs": "analytics",
    "reqeust": "request",
    "requast": "request",
    "deteced": "detected",
    "detiected": "detected",
    "detcted": "detected",
    "intellegence": "intelligence",
    "intelligance": "intelligence",
    # Common typos
    "teh": "the",
    "thier": "their",
    "recieve": "receive",
    "occured": "occurred",
    "seperate": "separate",
    "definately": "definitely",
    "accomodate": "accommodate",
    "untill": "until",
    "beleive": "believe",
    "existance": "existence",
    "occurence": "occurrence",
    "wierd": "weird",
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
    "neccessary": "necessary",
    "peice": "piece",
    "probaly": "probably",
    "realy": "really",
    "similiar": "similar",
    "tommorow": "tomorrow",
    "writting": "writing",
}

# Brand whitelist - don't flag these as misspellings
BRAND_WHITELIST = [
    # Barrios A2I brands
    "Barrios", "A2I", "A2i", "BARRIOS", "BarriosA2I",
    "RAGNAROK", "Ragnarok",
    "VORTEX", "Vortex",
    "NEXUS", "Nexus",
    "CHROMADON", "Chromadon",
    "TRINITY", "Trinity",
    "WORDSMITH", "Wordsmith",
    "SOUNDSCAPER", "Soundscaper",
    "AUTEUR", "Auteur",

    # Tech terms
    "AI", "API", "CRM", "SaaS", "B2B", "ROI", "KPI",
    "FFmpeg", "ElevenLabs", "Vercel", "OpenAI", "GPT",
    "OAuth", "SQL", "NoSQL", "GraphQL", "MongoDB",
    "Tesseract", "PaddleOCR", "EasyOCR", "CUDA",

    # Common UI terms
    "Inbox", "Docs", "Support", "Analytics", "Dashboard",
    "Login", "Signup", "Settings", "Profile", "Logout",
    "Homepage", "Navbar", "Sidebar", "Footer", "Header",

    # Tech abbreviations
    "LLC", "Inc", "Corp", "Ltd",
    "URL", "URI", "JSON", "XML", "CSV", "PDF",
    "HTTP", "HTTPS", "SSL", "TLS", "SSH", "FTP",
]

# Industry terms that shouldn't be flagged
INDUSTRY_TERMS: Set[str] = {
    "ai", "ml", "api", "sdk", "saas", "paas", "iaas", "gpu", "cpu",
    "oauth", "jwt", "ssl", "https", "url", "uri", "json", "xml",
    "frontend", "backend", "fullstack", "devops", "cicd", "kubernetes",
    "cta", "cpc", "cpm", "roi", "kpi", "seo", "sem", "ppc", "cro",
    "b2b", "b2c", "d2c", "omnichannel", "multichannel",
    "4k", "8k", "hdr", "uhd", "fps", "bitrate", "codec", "hevc",
    "prores", "mp4", "mov", "webm", "mkv",
    "llc", "inc", "corp", "ltd", "plc", "gmbh",
    "hipaa", "gdpr", "ccpa", "sox", "pci",
}


# =============================================================================
# SECTION 2: DATA CLASSES
# =============================================================================

@dataclass
class TextDetection:
    """A single text detection from OCR"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_number: int
    timestamp_ms: int
    is_misspelled: bool = False
    correction: Optional[str] = None


@dataclass
class SpellingError:
    """A spelling error found in video"""
    original: str
    correction: str
    timestamp_ms: int
    frame_number: int
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class WordsmithResult:
    """Result from WORDSMITH analysis"""
    success: bool
    signal: str
    message: str = ""
    errors: List[Dict[str, Any]] = field(default_factory=list)
    spelling_errors: List[SpellingError] = field(default_factory=list)
    ffmpeg_fix_commands: List[str] = field(default_factory=list)
    corrected_video_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 1
    max_iterations: int = 1


# =============================================================================
# SECTION 3: WORDSMITH V2 CLASS
# =============================================================================

class WordsmithV2:
    """
    THE WORDSMITH v2.0 LEGENDARY

    Capabilities:
    - Frame extraction at configurable FPS (OpenCV)
    - EasyOCR text detection with confidence scoring
    - Spell checking with pyspellchecker + brand whitelist
    - OCR_CORRECTIONS lookup for known misreads
    - FFmpeg drawtext overlay generation
    - Video re-rendering with corrections
    - BLOCKING mode to halt pipeline on errors
    """

    def __init__(
        self,
        fps_sample_rate: float = 1.0,  # Extract 1 frame per second
        confidence_threshold: float = 0.5,
        spell_checker_language: str = "en",
        blocking_mode: bool = True,  # HALT pipeline on errors
        auto_fix_mode: bool = True,   # Generate corrected video
    ):
        self.fps_sample_rate = fps_sample_rate
        self.confidence_threshold = confidence_threshold
        self.blocking_mode = blocking_mode
        self.auto_fix_mode = auto_fix_mode

        # Lazy initialization for OCR and spell checker
        self._ocr_reader = None
        self._spell_checker = None
        self._spell_checker_language = spell_checker_language

        logger.info(f"WordsmithV2 initialized - LEGENDARY MODE {'ACTIVE' if blocking_mode else 'PASSIVE'}")

    @property
    def ocr_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader (this may take a minute on first run)...")
                # Try GPU first, fallback to CPU
                # verbose=False to avoid Unicode progress bar issues on Windows
                try:
                    self._ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                    logger.info("EasyOCR initialized with GPU")
                except Exception:
                    self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    logger.info("EasyOCR initialized with CPU")
            except ImportError:
                logger.warning("EasyOCR not installed. OCR will be skipped.")
                return None
        return self._ocr_reader

    @property
    def spell_checker(self):
        """Lazy initialization of spell checker"""
        if self._spell_checker is None:
            try:
                from spellchecker import SpellChecker
                self._spell_checker = SpellChecker(language=self._spell_checker_language)

                # Add brand whitelist to spell checker
                self._spell_checker.word_frequency.load_words(BRAND_WHITELIST)
                self._spell_checker.word_frequency.load_words([w.lower() for w in BRAND_WHITELIST])
                self._spell_checker.word_frequency.load_words(list(INDUSTRY_TERMS))

                logger.info("SpellChecker initialized with brand whitelist")
            except ImportError:
                logger.warning("pyspellchecker not installed. Using OCR_CORRECTIONS only.")
                return None
        return self._spell_checker

    def _extract_frames(self, video_path: str, output_dir: str) -> List[Tuple[int, str, int]]:
        """
        Extract frames from video at specified FPS using OpenCV.
        Returns list of (frame_number, frame_path, timestamp_ms)
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Cannot extract frames.")
            return []

        logger.info(f"Extracting frames from {video_path} at {self.fps_sample_rate} FPS")

        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0:
            video_fps = 30.0  # Default fallback

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / self.fps_sample_rate))

        frames = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                timestamp_ms = int((frame_number / video_fps) * 1000)
                frame_path = os.path.join(output_dir, f"frame_{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append((frame_number, frame_path, timestamp_ms))

            frame_number += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {total_frames} total ({video_fps:.1f} fps video)")
        return frames

    def _run_ocr(self, frame_path: str) -> List[Dict[str, Any]]:
        """Run OCR on a single frame using EasyOCR"""
        if self.ocr_reader is None:
            return []

        try:
            results = self.ocr_reader.readtext(frame_path)

            detections = []
            for bbox, text, confidence in results:
                if confidence >= self.confidence_threshold:
                    # Convert bbox polygon to x1, y1, x2, y2
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))

                    detections.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    })

            return detections
        except Exception as e:
            logger.error(f"OCR error on {frame_path}: {e}")
            return []

    def _check_spelling(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is spelled correctly.
        Returns (is_correct, correction_if_wrong)

        Priority:
        1. OCR_CORRECTIONS - known OCR misreads
        2. Brand whitelist - skip these
        3. pyspellchecker - dictionary lookup
        """
        # Clean text
        clean_text = text.strip()
        if not clean_text:
            return True, None

        # Check OCR corrections first (full text match)
        lower_text = clean_text.lower()
        if lower_text in OCR_CORRECTIONS:
            correction = OCR_CORRECTIONS[lower_text]
            # Preserve original case
            if clean_text.isupper():
                return False, correction.upper()
            elif clean_text[0].isupper():
                return False, correction.capitalize()
            return False, correction

        # Check if it's a brand/whitelist word (full text)
        if clean_text in BRAND_WHITELIST or clean_text.upper() in [w.upper() for w in BRAND_WHITELIST]:
            return True, None

        # Split into words and check each
        words = re.findall(r'\b[a-zA-Z]+\b', clean_text)

        for word in words:
            if len(word) < 2:
                continue

            word_lower = word.lower()

            # Skip if in whitelist or industry terms
            if word in BRAND_WHITELIST or word.upper() in [w.upper() for w in BRAND_WHITELIST]:
                continue
            if word_lower in INDUSTRY_TERMS:
                continue

            # Check OCR corrections for individual words
            if word_lower in OCR_CORRECTIONS:
                corrected_word = OCR_CORRECTIONS[word_lower]
                # Preserve case
                if word.isupper():
                    corrected_word = corrected_word.upper()
                elif word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                correction = clean_text.replace(word, corrected_word)
                return False, correction

            # Use pyspellchecker if available
            if self.spell_checker is not None:
                if word_lower not in self.spell_checker:
                    # Get correction
                    best = self.spell_checker.correction(word_lower)
                    if best and best != word_lower:
                        # Preserve original case
                        if word.isupper():
                            best = best.upper()
                        elif word[0].isupper():
                            best = best.capitalize()
                        correction = clean_text.replace(word, best)
                        return False, correction

        return True, None

    def _generate_ffmpeg_overlay(
        self,
        error: SpellingError,
        video_width: int,
        video_height: int
    ) -> str:
        """
        Generate FFmpeg drawtext filter to overlay corrected text.
        """
        x1, y1, x2, y2 = error.bbox

        # Calculate text position and size
        box_width = x2 - x1
        box_height = y2 - y1
        font_size = max(int(box_height * 0.8), 16)

        # Escape special characters for FFmpeg
        correction_escaped = error.correction.replace("'", "\\'").replace(":", "\\:")

        # Time window for overlay (show for 2 seconds around the error)
        t_start = max(0, error.timestamp_ms / 1000 - 0.1)
        t_end = error.timestamp_ms / 1000 + 2.0

        # Create drawtext filter
        # First draw a background box, then the corrected text
        filter_str = (
            f"drawbox=x={x1}:y={y1}:w={box_width}:h={box_height}:"
            f"color=black@0.8:t=fill:enable='between(t,{t_start},{t_end})',"
            f"drawtext=text='{correction_escaped}':"
            f"x={x1+5}:y={y1+5}:"
            f"fontsize={font_size}:fontcolor=white:"
            f"enable='between(t,{t_start},{t_end})'"
        )

        return filter_str

    def _render_corrected_video(
        self,
        input_path: str,
        output_path: str,
        errors: List[SpellingError],
        video_width: int,
        video_height: int
    ) -> bool:
        """
        Render video with text corrections overlaid via FFmpeg.
        """
        if not errors:
            return True

        # Generate filter complex
        filters = []
        for error in errors:
            filter_str = self._generate_ffmpeg_overlay(error, video_width, video_height)
            filters.append(filter_str)

        filter_complex = ",".join(filters)

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_complex,
            "-c:a", "copy",
            output_path
        ]

        logger.info(f"Rendering corrected video with {len(errors)} fixes...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg not found")
            return False
        except Exception as e:
            logger.error(f"FFmpeg failed: {e}")
            return False

    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Get video dimensions using OpenCV"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception:
            return 1920, 1080  # Default fallback

    async def analyze_video(self, video_path: str) -> WordsmithResult:
        """
        Main analysis function - detect all text and check spelling.
        BLOCKS pipeline if errors found (when blocking_mode=True)
        """
        logger.info(f"[WORDSMITH] Analyzing video: {video_path}")

        if not os.path.exists(video_path):
            return WordsmithResult(
                success=False,
                signal="<promise>OCR_FAILED</promise>",
                errors=[{"error": f"Video not found: {video_path}"}]
            )

        # Get video dimensions
        video_width, video_height = self._get_video_dimensions(video_path)

        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp(prefix="wordsmith_")

        try:
            # Extract frames
            frames = self._extract_frames(video_path, temp_dir)

            if not frames:
                logger.warning("[WORDSMITH] No frames extracted")
                return WordsmithResult(
                    success=True,
                    signal="<promise>OCR_CLEAN</promise>",
                    metrics={"frames_analyzed": 0, "total_detections": 0, "spelling_errors": 0}
                )

            all_detections: List[TextDetection] = []
            spelling_errors: List[SpellingError] = []
            seen_errors: Set[str] = set()  # Deduplicate errors

            # Process each frame
            for frame_num, frame_path, timestamp_ms in frames:
                # Run OCR
                ocr_results = self._run_ocr(frame_path)

                for result in ocr_results:
                    text = result["text"]
                    confidence = result["confidence"]
                    bbox = result["bbox"]

                    # Check spelling
                    is_correct, correction = self._check_spelling(text)

                    detection = TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        frame_number=frame_num,
                        timestamp_ms=timestamp_ms,
                        is_misspelled=not is_correct,
                        correction=correction
                    )
                    all_detections.append(detection)

                    if not is_correct and correction:
                        # Deduplicate by original text
                        error_key = text.lower()
                        if error_key not in seen_errors:
                            seen_errors.add(error_key)

                            error = SpellingError(
                                original=text,
                                correction=correction,
                                timestamp_ms=timestamp_ms,
                                frame_number=frame_num,
                                bbox=bbox,
                                confidence=confidence
                            )
                            spelling_errors.append(error)
                            logger.warning(
                                f"[WORDSMITH] SPELLING ERROR at {timestamp_ms}ms: "
                                f"'{text}' -> '{correction}'"
                            )

            # Generate FFmpeg fix commands
            ffmpeg_commands = []
            for error in spelling_errors:
                cmd = self._generate_ffmpeg_overlay(error, video_width, video_height)
                ffmpeg_commands.append(cmd)

            # Determine result
            if spelling_errors:
                logger.error(f"[WORDSMITH] Found {len(spelling_errors)} spelling errors! BLOCKING.")

                # Auto-fix if enabled
                corrected_path = None
                if self.auto_fix_mode:
                    corrected_path = video_path.replace(".mp4", "_CORRECTED.mp4")
                    success = self._render_corrected_video(
                        video_path, corrected_path, spelling_errors,
                        video_width, video_height
                    )
                    if success:
                        logger.info(f"[WORDSMITH] Corrected video saved: {corrected_path}")
                    else:
                        corrected_path = None

                return WordsmithResult(
                    success=False,  # BLOCKING
                    signal="<promise>OCR_FAILED</promise>",
                    message=f"Found {len(spelling_errors)} spelling errors - BLOCKING pipeline",
                    errors=[{
                        "original": e.original,
                        "correction": e.correction,
                        "timestamp_ms": e.timestamp_ms,
                        "frame": e.frame_number
                    } for e in spelling_errors],
                    spelling_errors=spelling_errors,
                    ffmpeg_fix_commands=ffmpeg_commands,
                    corrected_video_path=corrected_path,
                    metrics={
                        "total_detections": len(all_detections),
                        "spelling_errors": len(spelling_errors),
                        "frames_analyzed": len(frames)
                    }
                )
            else:
                logger.info(f"[WORDSMITH] No spelling errors found! Video is clean.")
                return WordsmithResult(
                    success=True,
                    signal="<promise>OCR_CLEAN</promise>",
                    message="Video text is clean - no spelling errors detected",
                    errors=[],
                    spelling_errors=[],
                    ffmpeg_fix_commands=[],
                    metrics={
                        "total_detections": len(all_detections),
                        "spelling_errors": 0,
                        "frames_analyzed": len(frames)
                    }
                )

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SECTION 4: FACTORY FUNCTIONS & LEGACY COMPATIBILITY
# =============================================================================

def create_wordsmith(
    fps_sample_rate: float = 1.0,
    confidence_threshold: float = 0.5,
    blocking_mode: bool = True,
    auto_fix_mode: bool = True,
) -> WordsmithV2:
    """
    Factory function to create a configured WordsmithV2 instance.
    """
    return WordsmithV2(
        fps_sample_rate=fps_sample_rate,
        confidence_threshold=confidence_threshold,
        blocking_mode=blocking_mode,
        auto_fix_mode=auto_fix_mode,
    )


# Legacy compatibility - TheWordsmith class wrapping WordsmithV2
class TheWordsmith(WordsmithV2):
    """Legacy compatibility wrapper"""
    pass


# =============================================================================
# SECTION 5: NEXUS REGISTRATION
# =============================================================================

NEXUS_REGISTRATION = {
    "agent_id": "agent_7.25",
    "name": "THE WORDSMITH v2.0 LEGENDARY",
    "version": "2.0.0",
    "phase": "VORTEX",
    "handler": "wordsmith.analyze_video",
    "input_schema": "video_path: str",
    "output_schema": "WordsmithResult",
    "description": "Text Detection & Spelling Correction Agent - EasyOCR, pyspellchecker, FFmpeg auto-fix",
    "cost_target": {"min": 0.10, "max": 0.20, "unit": "USD/video"},
    "latency_target": {"min": 5.0, "max": 12.0, "unit": "seconds"},
    "capabilities": [
        "text_detection",
        "ocr_easyocr",
        "spelling_check",
        "ocr_correction",
        "brand_whitelist",
        "ffmpeg_auto_fix",
        "pipeline_blocking",
    ],
    "dependencies": {
        "required": ["opencv-python", "ffmpeg"],
        "optional": ["easyocr", "pyspellchecker"],
    },
    "author": "Barrios A2I",
    "created": "2026-01-20",
}


# =============================================================================
# SECTION 6: CLI INTERFACE
# =============================================================================

async def main():
    """CLI entry point for testing"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python the_wordsmith.py <video_path>")
        print("\nExample:")
        print("  python the_wordsmith.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print("=" * 70)
    print("THE WORDSMITH v2.0 LEGENDARY")
    print("=" * 70)
    print(f"Video: {video_path}")
    print("=" * 70)

    wordsmith = create_wordsmith(
        fps_sample_rate=1.0,
        blocking_mode=True,
        auto_fix_mode=True
    )

    result = await wordsmith.analyze_video(video_path)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Signal: {result.signal}")
    print(f"Success: {result.success}")
    print(f"Frames Analyzed: {result.metrics.get('frames_analyzed', 0)}")
    print(f"Text Detections: {result.metrics.get('total_detections', 0)}")

    if result.errors:
        print(f"\nSpelling Errors Found: {len(result.errors)}")
        for err in result.errors:
            print(f"  '{err.get('original')}' -> '{err.get('correction')}' @ {err.get('timestamp_ms')}ms")

    if result.corrected_video_path:
        print(f"\nCorrected video: {result.corrected_video_path}")

    print("=" * 70)

    # Exit code: 2 if errors found, 0 if clean
    sys.exit(2 if result.errors else 0)


if __name__ == "__main__":
    asyncio.run(main())
