"""
WORDSMITH v6.0 TYPESETTER - Text Overlay Engine

FFmpeg-based text overlay system for adding clean, professional text to videos.
Replaces the failed inpainting approach with a "prevention over cure" strategy.

Author: Barrios A2I / NEXUS Brain
Version: 6.0.0
Date: January 2026
"""

import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class TextPosition(Enum):
    """Preset text positions"""
    CENTER = "center"
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    LOWER_THIRD = "lower_third"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class TextAnimation(Enum):
    """Text animation effects"""
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADE_IN_OUT = "fade_in_out"


@dataclass
class TextOverlay:
    """
    Specification for a single text overlay.

    Attributes:
        text: The text content to display
        position: Either a TextPosition preset or custom (x, y) FFmpeg expressions
        start_time: When the text appears (seconds)
        end_time: When the text disappears (seconds)
        fontsize: Font size in pixels
        fontcolor: FFmpeg color (e.g., "white", "#00CED1", "cyan")
        fontfile: Path to TTF/OTF font file (optional, uses default if None)
        animation: Animation effect for the text
        fade_duration: Duration of fade in/out in seconds
        box: Whether to show a background box
        boxcolor: Background box color with alpha (e.g., "black@0.5")
        boxborderw: Padding around text in the box
        shadowcolor: Shadow color (None for no shadow)
        shadowx: Shadow X offset
        shadowy: Shadow Y offset
    """
    text: str
    position: Union[TextPosition, Tuple[str, str]] = TextPosition.CENTER
    start_time: float = 0.0
    end_time: float = 5.0
    fontsize: int = 48
    fontcolor: str = "white"
    fontfile: Optional[str] = None
    animation: TextAnimation = TextAnimation.FADE_IN_OUT
    fade_duration: float = 0.5
    box: bool = False
    boxcolor: str = "black@0.5"
    boxborderw: int = 10
    shadowcolor: Optional[str] = "black@0.5"
    shadowx: int = 2
    shadowy: int = 2


@dataclass
class VideoTextSpec:
    """
    Complete text specification for a video.

    Attributes:
        overlays: List of TextOverlay objects to apply
        title: Optional main title (convenience shorthand)
        subtitle: Optional subtitle
        end_card_text: Optional end card text
    """
    overlays: List[TextOverlay] = field(default_factory=list)
    title: Optional[str] = None
    title_duration: Tuple[float, float] = (0.5, 4.0)
    subtitle: Optional[str] = None
    subtitle_duration: Tuple[float, float] = (1.0, 4.0)
    end_card_text: Optional[str] = None
    end_card_duration: float = 3.0


@dataclass
class OverlayResult:
    """Result of text overlay operation"""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    overlays_applied: int = 0
    processing_time: float = 0.0


class TextOverlayEngine:
    """
    FFmpeg-based text overlay engine.

    Adds clean, professional text overlays to videos using FFmpeg's drawtext filter.
    No GPU required, fast processing, perfect text quality.
    """

    # Barrios A2I brand colors
    BRAND_CYAN = "#00CED1"
    BRAND_GOLD = "#D4AF37"
    BRAND_WHITE = "white"

    def __init__(
        self,
        fonts_dir: Optional[str] = None,
        default_font: Optional[str] = None,
        ffmpeg_path: str = "ffmpeg"
    ):
        """
        Initialize the text overlay engine.

        Args:
            fonts_dir: Directory containing brand fonts
            default_font: Default font file path
            ffmpeg_path: Path to FFmpeg executable
        """
        self.fonts_dir = Path(fonts_dir) if fonts_dir else None
        self.default_font = default_font
        self.ffmpeg_path = ffmpeg_path

        # Verify FFmpeg is available
        self._verify_ffmpeg()

        logger.info("[TYPESETTER] Text Overlay Engine initialized")

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found")
            logger.debug("[TYPESETTER] FFmpeg verified")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and ensure it's in PATH."
            )

    def _get_position_expression(
        self,
        position: Union[TextPosition, Tuple[str, str]],
        margin: int = 50
    ) -> Tuple[str, str]:
        """
        Convert TextPosition enum to FFmpeg x,y expressions.

        Args:
            position: TextPosition preset or (x, y) tuple of FFmpeg expressions
            margin: Margin from edges in pixels

        Returns:
            Tuple of (x_expr, y_expr) FFmpeg expressions
        """
        if isinstance(position, tuple):
            return position

        positions = {
            TextPosition.CENTER: ("(w-text_w)/2", "(h-text_h)/2"),
            TextPosition.TOP_CENTER: ("(w-text_w)/2", f"{margin}"),
            TextPosition.BOTTOM_CENTER: ("(w-text_w)/2", f"h-text_h-{margin}"),
            TextPosition.LOWER_THIRD: ("(w-text_w)/2", "h-h/4"),
            TextPosition.TOP_LEFT: (f"{margin}", f"{margin}"),
            TextPosition.TOP_RIGHT: (f"w-text_w-{margin}", f"{margin}"),
            TextPosition.BOTTOM_LEFT: (f"{margin}", f"h-text_h-{margin}"),
            TextPosition.BOTTOM_RIGHT: (f"w-text_w-{margin}", f"h-text_h-{margin}"),
        }
        return positions.get(position, positions[TextPosition.CENTER])

    def _build_drawtext_filter(self, overlay: TextOverlay) -> str:
        """
        Build FFmpeg drawtext filter string for a single overlay.

        Args:
            overlay: TextOverlay specification

        Returns:
            FFmpeg drawtext filter string
        """
        x_expr, y_expr = self._get_position_expression(overlay.position)

        # Escape special characters in text
        escaped_text = (
            overlay.text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("%", "\\%")
        )

        # Base filter components
        parts = [
            f"text='{escaped_text}'",
            f"fontsize={overlay.fontsize}",
            f"x={x_expr}",
            f"y={y_expr}",
        ]

        # Font file
        if overlay.fontfile:
            font_path = overlay.fontfile.replace("\\", "/").replace(":", "\\:")
            parts.append(f"fontfile='{font_path}'")
        elif self.default_font:
            font_path = self.default_font.replace("\\", "/").replace(":", "\\:")
            parts.append(f"fontfile='{font_path}'")

        # Handle animation/alpha
        if overlay.animation == TextAnimation.NONE:
            parts.append(f"fontcolor={overlay.fontcolor}")
        else:
            # Build alpha expression for fade effects
            start = overlay.start_time
            end = overlay.end_time
            fade = overlay.fade_duration

            if overlay.animation == TextAnimation.FADE_IN:
                alpha_expr = f"if(lt(t-{start},{fade}),(t-{start})/{fade},1)"
            elif overlay.animation == TextAnimation.FADE_OUT:
                alpha_expr = f"if(gt(t,{end-fade}),({end}-t)/{fade},1)"
            elif overlay.animation == TextAnimation.FADE_IN_OUT:
                alpha_expr = (
                    f"if(lt(t-{start},{fade}),(t-{start})/{fade},"
                    f"if(gt(t,{end-fade}),({end}-t)/{fade},1))"
                )
            else:
                alpha_expr = "1"

            # Use fontcolor_expr for animated alpha
            color = overlay.fontcolor.lstrip("#")
            if len(color) == 6:
                parts.append(f"fontcolor_expr={color}%{{eif\\:255*{alpha_expr}\\:x\\:2}}")
            else:
                parts.append(f"fontcolor={overlay.fontcolor}@{alpha_expr}")

        # Background box
        if overlay.box:
            parts.append("box=1")
            parts.append(f"boxcolor={overlay.boxcolor}")
            parts.append(f"boxborderw={overlay.boxborderw}")

        # Shadow
        if overlay.shadowcolor:
            parts.append(f"shadowcolor={overlay.shadowcolor}")
            parts.append(f"shadowx={overlay.shadowx}")
            parts.append(f"shadowy={overlay.shadowy}")

        # Timing enable expression
        parts.append(f"enable='between(t,{overlay.start_time},{overlay.end_time})'")

        return "drawtext=" + ":".join(parts)

    def add_overlays(
        self,
        video_path: str,
        overlays: List[TextOverlay],
        output_path: Optional[str] = None
    ) -> OverlayResult:
        """
        Add multiple text overlays to a video.

        Args:
            video_path: Path to input video
            overlays: List of TextOverlay specifications
            output_path: Output video path (auto-generated if None)

        Returns:
            OverlayResult with success status and output path
        """
        import time
        start_time = time.time()

        if not overlays:
            return OverlayResult(
                success=False,
                error_message="No overlays specified"
            )

        video_path = Path(video_path)
        if not video_path.exists():
            return OverlayResult(
                success=False,
                error_message=f"Video not found: {video_path}"
            )

        # Generate output path
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_TEXT{video_path.suffix}"
        output_path = Path(output_path)

        # Build filter chain
        filters = [self._build_drawtext_filter(o) for o in overlays]
        filter_complex = ",".join(filters)

        logger.info(f"[TYPESETTER] Adding {len(overlays)} text overlay(s) to {video_path.name}")
        logger.debug(f"[TYPESETTER] Filter: {filter_complex[:200]}...")

        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",  # Copy audio without re-encoding
            "-y",  # Overwrite output
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"[TYPESETTER] FFmpeg error: {result.stderr}")
                return OverlayResult(
                    success=False,
                    error_message=f"FFmpeg error: {result.stderr[-500:]}"
                )

            processing_time = time.time() - start_time
            logger.info(
                f"[TYPESETTER] Success! Added {len(overlays)} overlays "
                f"in {processing_time:.1f}s -> {output_path.name}"
            )

            return OverlayResult(
                success=True,
                output_path=str(output_path),
                overlays_applied=len(overlays),
                processing_time=processing_time
            )

        except subprocess.TimeoutExpired:
            return OverlayResult(
                success=False,
                error_message="FFmpeg timed out after 10 minutes"
            )
        except Exception as e:
            logger.exception("[TYPESETTER] Unexpected error")
            return OverlayResult(
                success=False,
                error_message=str(e)
            )

    def add_text_spec(
        self,
        video_path: str,
        spec: VideoTextSpec,
        output_path: Optional[str] = None
    ) -> OverlayResult:
        """
        Add text overlays from a VideoTextSpec.

        Convenience method that handles title/subtitle/end_card shortcuts.

        Args:
            video_path: Path to input video
            spec: VideoTextSpec with overlay definitions
            output_path: Output video path (auto-generated if None)

        Returns:
            OverlayResult with success status and output path
        """
        # Get video duration for end card positioning
        duration = self._get_video_duration(video_path)

        overlays = list(spec.overlays)  # Copy existing overlays

        # Add title if specified
        if spec.title:
            overlays.append(TextOverlay(
                text=spec.title,
                position=TextPosition.CENTER,
                start_time=spec.title_duration[0],
                end_time=spec.title_duration[1],
                fontsize=72,
                fontcolor=self.BRAND_CYAN,
                animation=TextAnimation.FADE_IN_OUT,
                shadowcolor="black@0.7",
                shadowx=3,
                shadowy=3
            ))

        # Add subtitle if specified
        if spec.subtitle:
            overlays.append(TextOverlay(
                text=spec.subtitle,
                position=TextPosition.LOWER_THIRD,
                start_time=spec.subtitle_duration[0],
                end_time=spec.subtitle_duration[1],
                fontsize=36,
                fontcolor=self.BRAND_WHITE,
                animation=TextAnimation.FADE_IN_OUT,
                box=True,
                boxcolor="black@0.6"
            ))

        # Add end card if specified
        if spec.end_card_text and duration:
            overlays.append(TextOverlay(
                text=spec.end_card_text,
                position=TextPosition.CENTER,
                start_time=duration - spec.end_card_duration,
                end_time=duration,
                fontsize=48,
                fontcolor=self.BRAND_GOLD,
                animation=TextAnimation.FADE_IN_OUT
            ))

        return self.add_overlays(video_path, overlays, output_path)

    def add_lower_third(
        self,
        video_path: str,
        title: str,
        subtitle: Optional[str] = None,
        start_time: float = 0.0,
        duration: float = 5.0,
        output_path: Optional[str] = None
    ) -> OverlayResult:
        """
        Add a professional lower-third title.

        Args:
            video_path: Path to input video
            title: Main title text
            subtitle: Optional subtitle text
            start_time: When to show the lower third
            duration: How long to show it
            output_path: Output video path

        Returns:
            OverlayResult
        """
        overlays = [
            TextOverlay(
                text=title,
                position=(f"50", f"h-120"),
                start_time=start_time,
                end_time=start_time + duration,
                fontsize=42,
                fontcolor=self.BRAND_WHITE,
                animation=TextAnimation.FADE_IN_OUT,
                box=True,
                boxcolor="black@0.7",
                boxborderw=15
            )
        ]

        if subtitle:
            overlays.append(TextOverlay(
                text=subtitle,
                position=(f"50", f"h-70"),
                start_time=start_time + 0.3,
                end_time=start_time + duration,
                fontsize=28,
                fontcolor=self.BRAND_CYAN,
                animation=TextAnimation.FADE_IN_OUT
            ))

        return self.add_overlays(video_path, overlays, output_path)

    def add_centered_title(
        self,
        video_path: str,
        text: str,
        start_time: float = 0.0,
        duration: float = 3.0,
        fontsize: int = 72,
        output_path: Optional[str] = None
    ) -> OverlayResult:
        """
        Add a centered title card.

        Args:
            video_path: Path to input video
            text: Title text
            start_time: When to show the title
            duration: How long to show it
            fontsize: Font size
            output_path: Output video path

        Returns:
            OverlayResult
        """
        overlay = TextOverlay(
            text=text,
            position=TextPosition.CENTER,
            start_time=start_time,
            end_time=start_time + duration,
            fontsize=fontsize,
            fontcolor=self.BRAND_WHITE,
            animation=TextAnimation.FADE_IN_OUT,
            shadowcolor="black@0.8",
            shadowx=4,
            shadowy=4
        )

        return self.add_overlays(video_path, [overlay], output_path)

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="WORDSMITH v6.0 TYPESETTER - Text Overlay Engine"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--text", required=True, help="Text to overlay")
    parser.add_argument("--position", default="center",
                       choices=["center", "top", "bottom", "lower_third"],
                       help="Text position")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=5.0, help="End time (seconds)")
    parser.add_argument("--fontsize", type=int, default=48, help="Font size")
    parser.add_argument("--output", help="Output video path")

    args = parser.parse_args()

    # Map position string to enum
    position_map = {
        "center": TextPosition.CENTER,
        "top": TextPosition.TOP_CENTER,
        "bottom": TextPosition.BOTTOM_CENTER,
        "lower_third": TextPosition.LOWER_THIRD
    }

    engine = TextOverlayEngine()
    overlay = TextOverlay(
        text=args.text,
        position=position_map.get(args.position, TextPosition.CENTER),
        start_time=args.start,
        end_time=args.end,
        fontsize=args.fontsize
    )

    result = engine.add_overlays(args.video, [overlay], args.output)

    if result.success:
        print(f"Success! Output: {result.output_path}")
        print(f"Processing time: {result.processing_time:.1f}s")
    else:
        print(f"Failed: {result.error_message}")
        exit(1)
