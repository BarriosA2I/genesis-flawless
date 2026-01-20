"""
ClipTimingEngine - Syncs Video Clips to Voiceover Timing
═══════════════════════════════════════════════════════════════════════════════
RAGNAROK v8.0 | VORTEX Post-Production Pipeline

Uses Whisper for voiceover transcription with word-level timestamps.
Maps video clips to voiceover sentences for perfect sync.

Features:
- Whisper-based voiceover transcription with timestamps
- Intelligent clip duration adjustment (trim, loop, speed)
- FFmpeg filter generation for timed assembly
- Export timing JSON for video assembly
- Speed adjustment within 0.8x-1.2x range

Pipeline Position: Before video assembly (after clips generated, after voiceover)
Cost Target: $0.00 (local processing)
Latency Target: 2-5s per video

Author: Barrios A2I
Version: 8.0.0
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("ragnarok.clip_timing")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VoiceoverSegment:
    """A segment of the voiceover with timestamps."""
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    word_count: int = 0
    segment_index: int = 0

    def __post_init__(self):
        self.duration_sec = self.end_sec - self.start_sec
        self.word_count = len(self.text.split())


@dataclass
class TimedClip:
    """A video clip with timing adjustment information."""
    clip_path: str
    clip_index: int
    target_start: float
    target_end: float
    target_duration: float
    actual_duration: float
    adjustment: str  # "none", "trim", "loop", "speed", "transition"
    speed_factor: float  # 0.8-1.2 range
    trim_start: float
    trim_end: float
    loop_count: int = 1
    mapped_segment_indices: List[int] = field(default_factory=list)

    @property
    def needs_adjustment(self) -> bool:
        return self.adjustment != "none"


class ClipTimingRequest(BaseModel):
    """Request for clip timing calculation."""
    clips: List[str] = Field(..., description="Paths to video clips")
    voiceover_path: str = Field(..., description="Path to voiceover audio")
    total_duration: Optional[float] = Field(None, description="Target total duration")
    whisper_model: str = Field("base", description="Whisper model size")


class ClipTimingResult(BaseModel):
    """Result of clip timing calculation."""
    success: bool = True
    error_message: Optional[str] = None
    timed_clips: List[Dict[str, Any]] = Field(default_factory=list)
    voiceover_segments: List[Dict[str, Any]] = Field(default_factory=list)
    total_duration: float = 0.0
    total_clips: int = 0
    adjusted_clips: int = 0
    timing_json_path: Optional[str] = None
    ffmpeg_filter: Optional[str] = None
    processing_time_ms: float = 0.0


# =============================================================================
# CLIP TIMING ENGINE
# =============================================================================

class ClipTimingEngine:
    """
    Synchronize video clips to voiceover timing.

    Uses Whisper for transcription and intelligent clip adjustment
    to ensure video matches narration perfectly.
    """

    # Acceptable speed adjustment range
    SPEED_RANGE = (0.8, 1.2)

    # Minimum clip duration (avoid too-short clips)
    MIN_CLIP_DURATION = 1.0

    def __init__(
        self,
        whisper_model: str = "base",
        ffprobe_path: str = "ffprobe"
    ):
        """
        Initialize ClipTimingEngine.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            ffprobe_path: Path to ffprobe executable
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None  # Lazy load
        self.ffprobe = ffprobe_path

        logger.info(f"[ClipTimingEngine] Initialized with Whisper model: {whisper_model}")

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self.whisper_model is None:
            try:
                import whisper
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                logger.info(f"[ClipTimingEngine] Whisper model '{self.whisper_model_name}' loaded")
            except ImportError:
                logger.warning("[ClipTimingEngine] Whisper not installed, using fallback timing")
                self.whisper_model = None
            except Exception as e:
                logger.warning(f"[ClipTimingEngine] Failed to load Whisper: {e}")
                self.whisper_model = None

    async def parse_voiceover(self, audio_path: str) -> List[VoiceoverSegment]:
        """
        Transcribe voiceover and extract sentence timestamps.

        Args:
            audio_path: Path to voiceover audio file

        Returns:
            List of VoiceoverSegment with timestamps
        """
        self._load_whisper()

        if self.whisper_model is None:
            # Fallback: estimate based on duration
            return await self._fallback_parse_voiceover(audio_path)

        try:
            # Run Whisper transcription with word timestamps
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en",
                verbose=False
            )

            segments = []
            for i, seg in enumerate(result.get("segments", [])):
                segment = VoiceoverSegment(
                    text=seg.get("text", "").strip(),
                    start_sec=seg.get("start", 0.0),
                    end_sec=seg.get("end", 0.0),
                    duration_sec=seg.get("end", 0.0) - seg.get("start", 0.0),
                    segment_index=i
                )
                segments.append(segment)

            logger.info(f"[ClipTimingEngine] Parsed {len(segments)} voiceover segments")
            return segments

        except Exception as e:
            logger.warning(f"[ClipTimingEngine] Whisper transcription failed: {e}")
            return await self._fallback_parse_voiceover(audio_path)

    async def _fallback_parse_voiceover(self, audio_path: str) -> List[VoiceoverSegment]:
        """Fallback timing estimation when Whisper unavailable."""
        duration = self.get_media_duration_sync(audio_path)

        # Estimate ~3 seconds per segment (average sentence)
        num_segments = max(1, int(duration / 3.0))
        segment_duration = duration / num_segments

        segments = []
        for i in range(num_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            segments.append(VoiceoverSegment(
                text=f"[Segment {i+1}]",
                start_sec=start,
                end_sec=end,
                duration_sec=segment_duration,
                segment_index=i
            ))

        logger.info(f"[ClipTimingEngine] Fallback: estimated {len(segments)} segments")
        return segments

    def get_media_duration_sync(self, file_path: str) -> float:
        """Get media duration using ffprobe (synchronous)."""
        try:
            cmd = [
                self.ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception as e:
            logger.warning(f"[ClipTimingEngine] ffprobe error for {file_path}: {e}")

        return 0.0

    async def get_media_duration(self, file_path: str) -> float:
        """Get media duration using ffprobe (async wrapper)."""
        return self.get_media_duration_sync(file_path)

    def calculate_timing(
        self,
        clips: List[str],
        segments: List[VoiceoverSegment]
    ) -> List[TimedClip]:
        """
        Map clips to voiceover segments and calculate timing adjustments.

        Args:
            clips: List of video clip paths
            segments: Parsed voiceover segments

        Returns:
            List of TimedClip with timing information
        """
        if not clips or not segments:
            return []

        timed_clips = []
        total_vo_duration = segments[-1].end_sec if segments else 0

        # Calculate how many clips per segment
        clips_per_segment = len(clips) / max(len(segments), 1)

        # Calculate cumulative segment durations for clip mapping
        segment_cumulative = [0.0]
        for seg in segments:
            segment_cumulative.append(segment_cumulative[-1] + seg.duration_sec)

        for i, clip_path in enumerate(clips):
            # Determine which segment(s) this clip maps to
            segment_idx = min(int(i / clips_per_segment), len(segments) - 1)
            segment = segments[segment_idx]

            # Get actual clip duration (use sync version to avoid event loop issues)
            actual_dur = self.get_media_duration_sync(clip_path)
            if actual_dur == 0:
                actual_dur = 5.0  # Default assumption

            # Calculate target duration for this clip
            # Each clip gets an equal portion of the segment it belongs to
            clips_in_this_segment = sum(
                1 for j in range(len(clips))
                if int(j / clips_per_segment) == segment_idx
            )
            target_dur = segment.duration_sec / max(clips_in_this_segment, 1)
            target_dur = max(target_dur, self.MIN_CLIP_DURATION)

            # Calculate target start/end times
            clip_position_in_segment = i - int(segment_idx * clips_per_segment)
            target_start = segment.start_sec + (clip_position_in_segment * target_dur)
            target_end = target_start + target_dur

            # Determine adjustment strategy
            ratio = actual_dur / target_dur if target_dur > 0 else 1.0

            if 0.95 <= ratio <= 1.05:
                # Close enough - minor trim if needed
                adjustment = "trim" if ratio != 1.0 else "none"
                speed = 1.0
                trim_end = min(actual_dur, target_dur)
                loop_count = 1
            elif self.SPEED_RANGE[0] <= ratio <= self.SPEED_RANGE[1]:
                # Adjust speed (within acceptable range)
                adjustment = "speed"
                speed = ratio  # Speed up or slow down
                trim_end = actual_dur
                loop_count = 1
            elif actual_dur > target_dur:
                # Too long - trim to target
                adjustment = "trim"
                speed = 1.0
                trim_end = target_dur
                loop_count = 1
            else:
                # Too short - need to loop or add transition
                if actual_dur < target_dur * 0.5:
                    # Very short - loop
                    adjustment = "loop"
                    speed = 1.0
                    trim_end = actual_dur
                    loop_count = int(target_dur / actual_dur) + 1
                else:
                    # Slightly short - add transition
                    adjustment = "transition"
                    speed = 1.0
                    trim_end = actual_dur
                    loop_count = 1

            timed_clip = TimedClip(
                clip_path=clip_path,
                clip_index=i,
                target_start=target_start,
                target_end=target_end,
                target_duration=target_dur,
                actual_duration=actual_dur,
                adjustment=adjustment,
                speed_factor=speed,
                trim_start=0,
                trim_end=trim_end,
                loop_count=loop_count,
                mapped_segment_indices=[segment_idx]
            )

            timed_clips.append(timed_clip)

        logger.info(f"[ClipTimingEngine] Calculated timing for {len(timed_clips)} clips")
        return timed_clips

    def generate_ffmpeg_filter(self, timed_clips: List[TimedClip]) -> str:
        """
        Generate FFmpeg filter_complex for timed assembly.

        Args:
            timed_clips: List of TimedClip with timing info

        Returns:
            FFmpeg filter_complex string
        """
        filters = []
        output_streams = []

        for i, clip in enumerate(timed_clips):
            input_ref = f"[{i}:v]"

            if clip.adjustment == "speed":
                # Adjust speed using setpts
                pts_factor = 1.0 / clip.speed_factor
                filters.append(f"{input_ref}setpts={pts_factor}*PTS[v{i}]")
                output_streams.append(f"[v{i}]")

            elif clip.adjustment == "trim":
                # Trim to target duration
                filters.append(
                    f"{input_ref}trim=0:{clip.trim_end},setpts=PTS-STARTPTS[v{i}]"
                )
                output_streams.append(f"[v{i}]")

            elif clip.adjustment == "loop":
                # Loop short clips
                loop_frames = int(clip.actual_duration * 30)  # Assume 30fps
                filters.append(
                    f"{input_ref}loop={clip.loop_count}:size={loop_frames},"
                    f"trim=0:{clip.target_duration},setpts=PTS-STARTPTS[v{i}]"
                )
                output_streams.append(f"[v{i}]")

            elif clip.adjustment == "transition":
                # Add fade transition
                filters.append(
                    f"{input_ref}fade=t=out:st={clip.actual_duration-0.5}:d=0.5[v{i}]"
                )
                output_streams.append(f"[v{i}]")

            else:
                # No adjustment needed - just copy
                filters.append(f"{input_ref}copy[v{i}]")
                output_streams.append(f"[v{i}]")

        # Concatenate all streams
        if output_streams:
            concat_input = "".join(output_streams)
            filters.append(
                f"{concat_input}concat=n={len(output_streams)}:v=1:a=0[vout]"
            )

        return ";".join(filters)

    async def export_timing_json(
        self,
        timed_clips: List[TimedClip],
        segments: List[VoiceoverSegment],
        output_path: str
    ) -> str:
        """
        Export timed_clips.json for video assembly.

        Args:
            timed_clips: List of TimedClip
            segments: Voiceover segments
            output_path: Path to save JSON

        Returns:
            Path to exported JSON file
        """
        data = {
            "version": "8.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_duration": sum(c.target_duration for c in timed_clips),
            "total_clips": len(timed_clips),
            "clips": [
                {
                    "index": c.clip_index,
                    "path": c.clip_path,
                    "start": c.target_start,
                    "end": c.target_end,
                    "duration": c.target_duration,
                    "actual_duration": c.actual_duration,
                    "adjustment": c.adjustment,
                    "speed": c.speed_factor,
                    "trim_end": c.trim_end,
                    "loop_count": c.loop_count,
                    "segment_indices": c.mapped_segment_indices
                }
                for c in timed_clips
            ],
            "voiceover_segments": [
                {
                    "index": s.segment_index,
                    "text": s.text,
                    "start": s.start_sec,
                    "end": s.end_sec,
                    "duration": s.duration_sec
                }
                for s in segments
            ]
        }

        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.info(f"[ClipTimingEngine] Exported timing JSON to {output_path}")
        return output_path

    async def calculate(self, request: ClipTimingRequest) -> ClipTimingResult:
        """
        Main entry point - calculate clip timing for voiceover sync.

        Args:
            request: ClipTimingRequest with clips and voiceover path

        Returns:
            ClipTimingResult with timed clips and FFmpeg filter
        """
        import time
        start_time = time.time()

        try:
            # Parse voiceover
            segments = await self.parse_voiceover(request.voiceover_path)

            if not segments:
                return ClipTimingResult(
                    success=False,
                    error_message="Failed to parse voiceover - no segments found"
                )

            # Calculate timing
            timed_clips = self.calculate_timing(request.clips, segments)

            if not timed_clips:
                return ClipTimingResult(
                    success=False,
                    error_message="No clips to process"
                )

            # Generate FFmpeg filter
            ffmpeg_filter = self.generate_ffmpeg_filter(timed_clips)

            # Export timing JSON
            timing_json_path = f"/tmp/timed_clips_{int(time.time())}.json"
            await self.export_timing_json(timed_clips, segments, timing_json_path)

            # Count adjusted clips
            adjusted_count = sum(1 for c in timed_clips if c.needs_adjustment)

            processing_time = (time.time() - start_time) * 1000

            return ClipTimingResult(
                success=True,
                timed_clips=[
                    {
                        "path": c.clip_path,
                        "start": c.target_start,
                        "end": c.target_end,
                        "duration": c.target_duration,
                        "adjustment": c.adjustment,
                        "speed": c.speed_factor
                    }
                    for c in timed_clips
                ],
                voiceover_segments=[
                    {"text": s.text, "start": s.start_sec, "end": s.end_sec}
                    for s in segments
                ],
                total_duration=sum(c.target_duration for c in timed_clips),
                total_clips=len(timed_clips),
                adjusted_clips=adjusted_count,
                timing_json_path=timing_json_path,
                ffmpeg_filter=ffmpeg_filter,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"[ClipTimingEngine] Error: {e}")
            return ClipTimingResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_clip_timing_engine(whisper_model: str = "base") -> ClipTimingEngine:
    """Create ClipTimingEngine instance."""
    return ClipTimingEngine(whisper_model=whisper_model)


# =============================================================================
# CLI TEST
# =============================================================================

async def test_clip_timing():
    """Test the clip timing engine."""
    print("\n[ClipTimingEngine] Testing...")
    print("=" * 60)

    engine = create_clip_timing_engine(whisper_model="base")

    # Test with sample files (if available)
    test_clips = [
        "/tmp/clip_01.mp4",
        "/tmp/clip_02.mp4",
        "/tmp/clip_03.mp4",
    ]
    test_voiceover = "/tmp/voiceover.mp3"

    # Check if test files exist
    from pathlib import Path
    if not Path(test_voiceover).exists():
        print("Test files not found - skipping actual test")
        print("Engine initialized successfully")
        return

    result = await engine.calculate(ClipTimingRequest(
        clips=test_clips,
        voiceover_path=test_voiceover
    ))

    if result.success:
        print(f"Success! Processed {result.total_clips} clips")
        print(f"Total duration: {result.total_duration:.2f}s")
        print(f"Adjusted clips: {result.adjusted_clips}")
        print(f"Timing JSON: {result.timing_json_path}")
    else:
        print(f"Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(test_clip_timing())
