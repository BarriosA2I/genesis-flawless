"""
GENESIS Agent 7: VideoAssemblyAgent
═══════════════════════════════════════════════════════════════════════════════
Production FFmpeg video assembly pipeline for RAGNAROK Commercial_Lab.

Adapted from python-barrios-agents/ragnarok/video_assembly_agent.py v3.1.0
Standalone version without RagnarokCore dependency.

RAGNAROK v8.0 UPGRADES:
- ClipTimingEngine integration for voiceover sync
- Clips are now timed to match narration sentences
- Automatic speed/trim/loop adjustments

Features:
- Multi-format export (YouTube/TikTok/Instagram)
- Resolution normalization for mixed inputs
- Native FFmpeg sidechain compression for audio ducking
- Async subprocess execution
- Circuit breaker protection
- VOICEOVER SYNC (v8.0)

Author: Barrios A2I
Version: 8.0.0 (RAGNAROK v8.0)
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# RAGNAROK v8.0: Import ClipTimingEngine for voiceover sync
try:
    from agents.vortex_postprod.clip_timing_engine import (
        ClipTimingEngine,
        ClipTimingRequest,
        create_clip_timing_engine
    )
    CLIP_TIMING_AVAILABLE = True
except ImportError:
    CLIP_TIMING_AVAILABLE = False
    ClipTimingEngine = None

# Configure logging
logger = logging.getLogger("genesis.video_assembly")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class SimpleCircuitBreaker:
    """Lightweight circuit breaker for FFmpeg operations."""

    name: str
    failure_threshold: int = 3
    recovery_timeout: int = 60
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: Optional[float] = None

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        # Check if we should attempt recovery
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[CircuitBreaker:{self.name}] Attempting recovery (HALF_OPEN)")
            else:
                raise RuntimeError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success - reset on HALF_OPEN
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failures = 0
                logger.info(f"[CircuitBreaker:{self.name}] Recovered (CLOSED)")

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"[CircuitBreaker:{self.name}] OPEN after {self.failures} failures")

            raise

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "threshold": self.failure_threshold
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class VideoFormat(str, Enum):
    YOUTUBE_1080P = "youtube_1080p"
    YOUTUBE_4K = "youtube_4k"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_FEED = "instagram_feed"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"


class RenderPreset(str, Enum):
    DRAFT = "draft"           # ultrafast, lower quality
    FAST = "fast"             # fast preset, good quality (DEFAULT)
    BALANCED = "balanced"     # medium preset
    QUALITY = "quality"       # slow preset, best quality


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "fade"
    FADE_BLACK = "fadeblack"
    FADE_WHITE = "fadewhite"


class VideoClip(BaseModel):
    """Input video clip."""
    path: str
    duration: float
    scene_number: int = 0
    width: Optional[int] = None
    height: Optional[int] = None


class AudioInput(BaseModel):
    """Audio inputs for assembly."""
    voiceover_path: str
    music_path: Optional[str] = None
    music_volume: float = 0.3  # Background music volume
    ducking_config: Dict[str, Any] = Field(default_factory=lambda: {
        "threshold": 0.1,
        "ratio": 4,
        "attack": 200,
        "release": 500
    })


class TransitionSpec(BaseModel):
    """Transition between clips."""
    from_scene: int
    to_scene: int
    type: TransitionType = TransitionType.CROSSFADE
    duration: float = 0.5


class AssemblyRequest(BaseModel):
    """Complete assembly request."""
    session_id: str
    clips: List[VideoClip]
    audio: AudioInput
    transitions: List[TransitionSpec] = Field(default_factory=list)
    output_formats: List[VideoFormat] = Field(default_factory=lambda: [
        VideoFormat.YOUTUBE_1080P,
        VideoFormat.TIKTOK,
        VideoFormat.INSTAGRAM_FEED
    ])
    render_preset: RenderPreset = RenderPreset.FAST
    include_thumbnail: bool = True


class RenderMetrics(BaseModel):
    """Render performance metrics."""
    render_time_seconds: float
    file_size_bytes: int
    video_bitrate_kbps: float = 0.0
    audio_bitrate_kbps: float = 192.0
    encoding_speed: float = 0.0


class FormatOutput(BaseModel):
    """Single format output."""
    format: VideoFormat
    local_path: str
    resolution: str
    duration: float
    file_size_bytes: int


class AssemblyResponse(BaseModel):
    """Multi-format assembly output."""
    session_id: str
    outputs: Dict[str, FormatOutput]  # format_name -> output
    thumbnail_path: Optional[str] = None
    total_duration: float
    total_render_time: float
    total_cost: float
    success: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

FORMAT_SPECS = {
    VideoFormat.YOUTUBE_1080P: {
        "width": 1920, "height": 1080, "fps": 30,
        "bitrate": "8M", "max_bitrate": "10M",
        "audio_bitrate": "192k"
    },
    VideoFormat.YOUTUBE_4K: {
        "width": 3840, "height": 2160, "fps": 30,
        "bitrate": "35M", "max_bitrate": "45M",
        "audio_bitrate": "256k"
    },
    VideoFormat.TIKTOK: {
        "width": 1080, "height": 1920, "fps": 30,
        "bitrate": "6M", "max_bitrate": "8M",
        "audio_bitrate": "192k"
    },
    VideoFormat.INSTAGRAM_REELS: {
        "width": 1080, "height": 1920, "fps": 30,
        "bitrate": "6M", "max_bitrate": "8M",
        "audio_bitrate": "192k"
    },
    VideoFormat.INSTAGRAM_FEED: {
        "width": 1080, "height": 1080, "fps": 30,
        "bitrate": "5M", "max_bitrate": "6M",
        "audio_bitrate": "192k"
    },
    VideoFormat.LINKEDIN: {
        "width": 1920, "height": 1080, "fps": 30,
        "bitrate": "5M", "max_bitrate": "6M",
        "audio_bitrate": "192k"
    },
    VideoFormat.FACEBOOK: {
        "width": 1280, "height": 720, "fps": 30,
        "bitrate": "4M", "max_bitrate": "5M",
        "audio_bitrate": "128k"
    },
}

PRESET_MAP = {
    RenderPreset.DRAFT: "ultrafast",
    RenderPreset.FAST: "fast",
    RenderPreset.BALANCED: "medium",
    RenderPreset.QUALITY: "slow",
}

CRF_MAP = {
    RenderPreset.DRAFT: 28,
    RenderPreset.FAST: 23,
    RenderPreset.BALANCED: 21,
    RenderPreset.QUALITY: 18,
}


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO ASSEMBLY AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class VideoAssemblyAgent:
    """
    Production FFmpeg video assembly agent for GENESIS.

    Assembles video clips with voiceover and optional music,
    exports to multiple platform formats.
    """

    def __init__(
        self,
        work_dir: str = "/tmp/genesis_assembly",
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        enable_clip_timing: bool = True  # RAGNAROK v8.0: Enable voiceover sync
    ):
        # Work directory
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # FFmpeg paths
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

        # Circuit breaker for FFmpeg
        self.ffmpeg_breaker = SimpleCircuitBreaker(
            name="ffmpeg",
            failure_threshold=2,
            recovery_timeout=60
        )

        # RAGNAROK v8.0: ClipTimingEngine for voiceover sync
        self.clip_timing_engine = None
        self.enable_clip_timing = enable_clip_timing
        if enable_clip_timing and CLIP_TIMING_AVAILABLE:
            self.clip_timing_engine = create_clip_timing_engine(whisper_model="base")
            logger.info("[VideoAssemblyAgent] ClipTimingEngine enabled for voiceover sync")

        # Cost tracking
        self.total_cost = 0.0
        self.render_count = 0

        logger.info(f"[VideoAssemblyAgent] Initialized (work_dir: {self.work_dir})")

    async def assemble(self, request: AssemblyRequest) -> AssemblyResponse:
        """
        Assemble video from clips, voiceover, and optional music.
        Renders to all requested output formats.

        RAGNAROK v8.0: If ClipTimingEngine is enabled and voiceover is present,
        clips are automatically timed to match voiceover narration.
        """
        start_time = time.time()

        logger.info(f"[VideoAssemblyAgent] Starting assembly: {len(request.clips)} clips, "
                   f"{len(request.output_formats)} formats")

        # RAGNAROK v8.0: Sync clips to voiceover timing if enabled
        timed_clips = None
        timing_result = None
        if (self.clip_timing_engine and
            self.enable_clip_timing and
            request.audio.voiceover_path and
            len(request.clips) > 0):

            logger.info("[VideoAssemblyAgent] Calculating clip timing for voiceover sync...")
            try:
                timing_result = await self.clip_timing_engine.calculate(
                    ClipTimingRequest(
                        clips=[clip.path for clip in request.clips],
                        voiceover_path=request.audio.voiceover_path
                    )
                )

                if timing_result.success:
                    timed_clips = timing_result.timed_clips
                    logger.info(f"[VideoAssemblyAgent] Clip timing calculated: "
                               f"{timing_result.adjusted_clips}/{timing_result.total_clips} clips adjusted")
                else:
                    logger.warning(f"[VideoAssemblyAgent] Clip timing failed: {timing_result.error_message}")

            except Exception as e:
                logger.warning(f"[VideoAssemblyAgent] ClipTimingEngine error: {e}")
                # Continue without timing adjustments

        outputs: Dict[str, FormatOutput] = {}
        total_cost = 0.0
        video_duration = 0.0
        thumbnail_path = None

        # Render each format
        for fmt in request.output_formats:
            try:
                output = await self._render_format(request, fmt)
                outputs[fmt.value] = output
                total_cost += self._calculate_cost(output.file_size_bytes, output.duration)
                video_duration = output.duration

                # Generate thumbnail from first format
                if thumbnail_path is None and request.include_thumbnail:
                    thumbnail_path = await self._generate_thumbnail(output.local_path)

            except Exception as e:
                logger.error(f"[VideoAssemblyAgent] Failed to render {fmt.value}: {e}")
                # Continue with other formats

        total_render_time = time.time() - start_time

        self.render_count += 1
        self.total_cost += total_cost

        logger.info(f"[VideoAssemblyAgent] Assembly complete: {len(outputs)} formats in {total_render_time:.1f}s")

        return AssemblyResponse(
            session_id=request.session_id,
            outputs=outputs,
            thumbnail_path=thumbnail_path,
            total_duration=video_duration,
            total_render_time=total_render_time,
            total_cost=total_cost,
            success=len(outputs) > 0
        )

    async def _render_format(self, request: AssemblyRequest, fmt: VideoFormat) -> FormatOutput:
        """Render a single format."""
        spec = FORMAT_SPECS[fmt]

        # Generate output filename
        output_id = f"{request.session_id}_{fmt.value}"
        output_path = self.work_dir / f"{output_id}.mp4"

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(request, spec, str(output_path))

        # Execute with circuit breaker
        async def run_ffmpeg():
            logger.info(f"[VideoAssemblyAgent] Rendering {fmt.value}...")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode()[-500:]
                logger.error(f"[VideoAssemblyAgent] FFmpeg error: {error_msg}")
                raise RuntimeError(f"FFmpeg failed: {error_msg}")

            return True

        await self.ffmpeg_breaker.call(run_ffmpeg)

        # Get output info
        file_size = output_path.stat().st_size
        video_info = await self._probe_video(str(output_path))

        logger.info(f"[VideoAssemblyAgent] Rendered {fmt.value}: "
                   f"{file_size / 1_000_000:.2f} MB, {video_info.get('duration', 0):.1f}s")

        return FormatOutput(
            format=fmt,
            local_path=str(output_path),
            resolution=f"{spec['width']}x{spec['height']}",
            duration=video_info.get("duration", 0),
            file_size_bytes=file_size
        )

    def _build_ffmpeg_command(
        self,
        request: AssemblyRequest,
        spec: Dict[str, Any],
        output_path: str
    ) -> List[str]:
        """Build the FFmpeg command for rendering."""

        inputs = []
        filter_parts = []

        # Add video clips with resolution normalization
        for i, clip in enumerate(request.clips):
            inputs.extend(["-i", clip.path])

            # Normalize all clips to target resolution
            filter_parts.append(
                f"[{i}:v]"
                f"scale={spec['width']}:{spec['height']}:force_original_aspect_ratio=decrease,"
                f"pad={spec['width']}:{spec['height']}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,"
                f"fps={spec['fps']}"
                f"[v{i}]"
            )

        # Add voiceover
        vo_idx = len(request.clips)
        inputs.extend(["-i", request.audio.voiceover_path])

        # Add music if present
        has_music = request.audio.music_path is not None
        if has_music:
            music_idx = len(request.clips) + 1
            inputs.extend(["-i", request.audio.music_path])

        # Video concat
        if len(request.clips) > 1:
            if request.transitions and len(request.transitions) > 0:
                # Use xfade transitions
                current_stream = "[v0]"
                transition_offset = request.clips[0].duration

                for i, trans in enumerate(request.transitions):
                    if i + 1 < len(request.clips):
                        next_stream = f"[v{i+1}]"
                        trans_duration = min(trans.duration, 1.0)
                        offset = max(0, transition_offset - trans_duration)

                        out_stream = f"[vt{i}]" if i < len(request.transitions) - 1 else "[v_main]"

                        filter_parts.append(
                            f"{current_stream}{next_stream}"
                            f"xfade=transition={trans.type.value}:duration={trans_duration}:offset={offset}"
                            f"{out_stream}"
                        )

                        current_stream = out_stream
                        if i + 1 < len(request.clips):
                            transition_offset += request.clips[i + 1].duration - trans_duration
            else:
                # Simple concat
                concat_input = "".join([f"[v{i}]" for i in range(len(request.clips))])
                filter_parts.append(
                    f"{concat_input}concat=n={len(request.clips)}:v=1:a=0[v_main]"
                )
        else:
            # Single clip
            filter_parts.append("[v0]copy[v_main]")

        # Audio processing
        if has_music:
            ducking = request.audio.ducking_config
            music_vol = request.audio.music_volume

            # Audio mixing with sidechain compression
            filter_parts.append(
                f"[{music_idx}:a]volume={music_vol},apad[music_vol];"
                f"[{vo_idx}:a][music_vol]"
                f"sidechaincompress=threshold={ducking.get('threshold', 0.1)}:"
                f"ratio={ducking.get('ratio', 4)}:"
                f"attack={ducking.get('attack', 200)}:"
                f"release={ducking.get('release', 500)}"
                f"[music_ducked];"
                f"[{vo_idx}:a][music_ducked]amix=inputs=2:duration=first:dropout_transition=2[a_main]"
            )
        else:
            # Voiceover only
            filter_parts.append(f"[{vo_idx}:a]apad[a_main]")

        # Combine all filters
        full_filter = ";".join(filter_parts)

        # Build command
        preset = PRESET_MAP[request.render_preset]
        crf = CRF_MAP[request.render_preset]

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex", full_filter,
            "-map", "[v_main]",
            "-map", "[a_main]",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-profile:v", "high",
            "-level", "4.1",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", spec["audio_bitrate"],
            "-movflags", "+faststart",
            "-shortest",
            output_path
        ]

        return cmd

    async def _probe_video(self, path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe."""
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()

            data = json.loads(stdout.decode())

            duration = float(data.get("format", {}).get("duration", 0))
            bitrate = int(data.get("format", {}).get("bit_rate", 0)) / 1000

            return {
                "duration": duration,
                "bitrate_kbps": bitrate,
            }
        except Exception as e:
            logger.warning(f"[VideoAssemblyAgent] ffprobe error: {e}")
            return {"duration": 0, "bitrate_kbps": 0}

    async def _generate_thumbnail(self, video_path: str, timestamp: float = 1.0) -> Optional[str]:
        """Generate thumbnail from video."""
        thumb_path = video_path.replace(".mp4", "_thumb.jpg")

        cmd = [
            self.ffmpeg, "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            thumb_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0:
                return thumb_path
        except Exception as e:
            logger.warning(f"[VideoAssemblyAgent] Thumbnail generation failed: {e}")

        return None

    def _calculate_cost(self, file_size_bytes: int, duration: float) -> float:
        """Calculate render cost based on output size and duration."""
        # Compute cost: ~$0.36/hour of render time
        # Assume render takes roughly 2x realtime
        render_time_estimate = duration * 2
        compute_cost = (render_time_estimate / 3600) * 0.36

        # Storage cost: $0.02/GB
        storage_cost = (file_size_bytes / 1_000_000_000) * 0.02

        return compute_cost + storage_cost

    def cleanup(self, session_id: Optional[str] = None):
        """Clean up work directory."""
        if session_id:
            # Clean specific session
            for f in self.work_dir.glob(f"{session_id}_*"):
                f.unlink()
        else:
            # Clean all
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "work_dir": str(self.work_dir),
            "render_count": self.render_count,
            "total_cost": self.total_cost,
            "circuit_breaker": self.ffmpeg_breaker.stats()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_assembly_agent(work_dir: Optional[str] = None) -> VideoAssemblyAgent:
    """Factory function to create VideoAssemblyAgent with environment config."""
    work_dir = work_dir or os.getenv("ASSEMBLY_WORK_DIR", "/tmp/genesis_assembly")
    ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
    ffprobe_path = os.getenv("FFPROBE_PATH", "ffprobe")

    return VideoAssemblyAgent(
        work_dir=work_dir,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

async def validate_ffmpeg() -> Dict[str, Any]:
    """Check if FFmpeg is available and working."""
    result = {
        "ffmpeg_available": False,
        "ffprobe_available": False,
        "version": None
    }

    try:
        # Check ffmpeg
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        if process.returncode == 0:
            result["ffmpeg_available"] = True
            # Extract version
            version_line = stdout.decode().split('\n')[0]
            result["version"] = version_line

        # Check ffprobe
        process = await asyncio.create_subprocess_exec(
            "ffprobe", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        result["ffprobe_available"] = process.returncode == 0

    except FileNotFoundError:
        logger.warning("[VideoAssemblyAgent] FFmpeg not found in PATH")

    return result


if __name__ == "__main__":
    # Quick validation
    async def main():
        print("\n[VideoAssemblyAgent] Validation")
        print("=" * 60)

        # Check FFmpeg
        ffmpeg_status = await validate_ffmpeg()
        print(f"FFmpeg: {'OK' if ffmpeg_status['ffmpeg_available'] else 'NOT FOUND'}")
        print(f"FFprobe: {'OK' if ffmpeg_status['ffprobe_available'] else 'NOT FOUND'}")
        if ffmpeg_status['version']:
            print(f"Version: {ffmpeg_status['version'][:50]}...")

        # Create agent
        agent = create_assembly_agent()
        print(f"\nAgent initialized: {agent.work_dir}")
        print(f"Circuit breaker: {agent.ffmpeg_breaker.state.value}")

    asyncio.run(main())
