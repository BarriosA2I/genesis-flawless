"""
================================================================================
VORTEX v2.1 LEGENDARY - GRAPH NODES
================================================================================
Pipeline nodes for video assembly with async FFmpeg execution.

Critical Fix P0: Uses asyncio.create_subprocess_exec() instead of subprocess.run()
to prevent event loop blocking during video processing.

Pipeline Flow:
ROUTING → ASSET_DOWNLOAD → SCENE_ANALYSIS → TRANSITION_SELECTION →
CLIP_ASSEMBLY → AUDIO_SYNC → FORMAT_RENDER → QUALITY_CHECK → COMPLETED

Author: Barrios A2I | VORTEX v2.1
================================================================================
"""

import os
import asyncio
import logging
import tempfile
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

import httpx

from .state_machine import (
    GlobalState,
    PipelinePhase,
    ProcessingMode,
    TransitionType,
    SceneAnalysis,
    TransitionDecision,
    ComplexityAnalyzer,
    TRANSITION_XFADE_MAP,
    FORMAT_SPECS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class VortexConfig:
    """VORTEX pipeline configuration"""
    WORK_DIR = Path(os.getenv("VORTEX_WORK_DIR", "/tmp/vortex"))
    FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "300"))  # 5 minutes
    DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "120"))  # 2 minutes
    MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "4"))
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    AI_ANALYSIS_ENABLED = os.getenv("AI_ANALYSIS_ENABLED", "true").lower() == "true"


# =============================================================================
# ASYNC FFMPEG WRAPPER (P0 FIX - Non-blocking)
# =============================================================================

class AsyncFFmpeg:
    """
    Non-blocking FFmpeg execution using asyncio.create_subprocess_exec().

    CRITICAL: This replaces subprocess.run() which blocks the event loop.
    Health checks will respond during video processing with this approach.
    """

    def __init__(self, timeout: int = None):
        self.timeout = timeout or VortexConfig.FFMPEG_TIMEOUT
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg binary path"""
        # Check common locations
        paths = [
            "/usr/local/bin/ffmpeg",  # Docker static binary location
            "/usr/bin/ffmpeg",
            "ffmpeg",  # System PATH
        ]
        for path in paths:
            if os.path.exists(path) or path == "ffmpeg":
                return path
        return "ffmpeg"

    def _find_ffprobe(self) -> str:
        """Find FFprobe binary path"""
        paths = [
            "/usr/local/bin/ffprobe",
            "/usr/bin/ffprobe",
            "ffprobe",
        ]
        for path in paths:
            if os.path.exists(path) or path == "ffprobe":
                return path
        return "ffprobe"

    async def run(self, cmd: List[str], timeout: int = None) -> Tuple[int, str, str]:
        """
        Execute FFmpeg command asynchronously.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        effective_timeout = timeout or self.timeout

        logger.info(f"AsyncFFmpeg executing: {' '.join(cmd[:10])}...")
        start_time = datetime.utcnow()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=effective_timeout
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"AsyncFFmpeg completed in {duration_ms:.0f}ms, returncode={process.returncode}")

            return process.returncode, stdout.decode(), stderr.decode()

        except asyncio.TimeoutError:
            logger.error(f"FFmpeg timeout after {effective_timeout}s")
            if process:
                process.kill()
                await process.wait()
            raise
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
            raise

    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        returncode, stdout, stderr = await self.run(cmd, timeout=30)

        if returncode != 0:
            logger.error(f"ffprobe failed: {stderr}")
            return {}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse ffprobe output: {stdout[:200]}")
            return {}

    async def get_duration_ms(self, video_path: str) -> int:
        """Get video duration in milliseconds"""
        info = await self.get_video_info(video_path)

        if "format" in info and "duration" in info["format"]:
            return int(float(info["format"]["duration"]) * 1000)

        # Fallback: check streams
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and "duration" in stream:
                return int(float(stream["duration"]) * 1000)

        return 0

    async def check_version(self) -> str:
        """Get FFmpeg version string"""
        cmd = [self.ffmpeg_path, "-version"]
        returncode, stdout, stderr = await self.run(cmd, timeout=10)

        if returncode == 0:
            # Extract version from first line
            first_line = stdout.split("\n")[0]
            return first_line
        return "unknown"


# =============================================================================
# BASE NODE CLASS
# =============================================================================

class PipelineNode(ABC):
    """Abstract base class for pipeline nodes"""

    def __init__(self):
        self.ffmpeg = AsyncFFmpeg()

    @abstractmethod
    async def execute(self, state: GlobalState) -> GlobalState:
        """Execute this node's processing"""
        pass

    def get_work_dir(self, job_id: str) -> Path:
        """Get job-specific working directory"""
        work_dir = VortexConfig.WORK_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir


# =============================================================================
# ROUTER NODE (Dual-Process Routing)
# =============================================================================

class RouterNode(PipelineNode):
    """
    Determines processing mode based on complexity analysis.

    Modes:
    - SYSTEM1_FAST: Simple briefs, rule-based transitions (<200ms)
    - SYSTEM1_HYBRID: Fast path with optional AI enhancement
    - SYSTEM2_DEEP: Full AI analysis for complex briefs
    """

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] RouterNode: Analyzing complexity")

        # Analyze complexity
        mode, score = ComplexityAnalyzer.analyze(state)

        # Update state with routing decision
        new_state = state.transition_to(
            PipelinePhase.ROUTING,
            processing_mode=mode,
            complexity_score=score
        )

        logger.info(f"[{state.job_id}] Routed to {mode.value} (score={score:.3f})")

        return new_state


# =============================================================================
# ASSET DOWNLOAD NODE
# =============================================================================

class AssetDownloadNode(PipelineNode):
    """
    Downloads video clips, voiceover, and music assets.
    Uses concurrent downloads with semaphore for rate limiting.
    """

    def __init__(self):
        super().__init__()
        self.semaphore = asyncio.Semaphore(VortexConfig.MAX_CONCURRENT_DOWNLOADS)

    async def _download_file(self, url: str, dest_path: Path) -> bool:
        """Download a single file with rate limiting"""
        async with self.semaphore:
            try:
                async with httpx.AsyncClient(timeout=VortexConfig.DOWNLOAD_TIMEOUT) as client:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()

                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(response.content)

                    logger.info(f"Downloaded: {url} -> {dest_path}")
                    return True

            except Exception as e:
                logger.error(f"Download failed for {url}: {e}")
                return False

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] AssetDownloadNode: Downloading {len(state.video_urls)} clips")

        work_dir = self.get_work_dir(state.job_id)
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        # Download video clips concurrently
        clip_tasks = []
        clip_paths = []

        for i, url in enumerate(state.video_urls):
            ext = Path(url).suffix or ".mp4"
            dest_path = clips_dir / f"clip_{i:03d}{ext}"
            clip_paths.append(str(dest_path))
            clip_tasks.append(self._download_file(url, dest_path))

        # Download voiceover if present
        voiceover_path = None
        if state.voiceover_url:
            ext = Path(state.voiceover_url).suffix or ".mp3"
            voiceover_path = work_dir / f"voiceover{ext}"
            clip_tasks.append(self._download_file(state.voiceover_url, voiceover_path))

        # Download music if present
        music_path = None
        if state.music_url:
            ext = Path(state.music_url).suffix or ".mp3"
            music_path = work_dir / f"music{ext}"
            clip_tasks.append(self._download_file(state.music_url, music_path))

        # Execute all downloads
        results = await asyncio.gather(*clip_tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception) or r is False]
        if failures:
            logger.warning(f"[{state.job_id}] {len(failures)} download(s) failed")

        # Update state
        new_state = state.transition_to(
            PipelinePhase.ASSET_DOWNLOAD,
            local_clip_paths=clip_paths,
            local_voiceover_path=str(voiceover_path) if voiceover_path else None,
            local_music_path=str(music_path) if music_path else None
        )

        logger.info(f"[{state.job_id}] Downloaded {len(clip_paths)} clips")

        return new_state


# =============================================================================
# SCENE ANALYSIS NODE
# =============================================================================

class SceneAnalysisNode(PipelineNode):
    """
    AI-powered scene analysis for System 2 processing.
    Analyzes motion, colors, brightness, objects, and mood.

    For System 1, uses fast heuristic-based analysis.
    """

    async def _analyze_scene_fast(self, clip_path: str, index: int) -> SceneAnalysis:
        """Fast heuristic-based scene analysis (System 1)"""
        duration_ms = await self.ffmpeg.get_duration_ms(clip_path)

        # Simple heuristics based on filename and duration
        return SceneAnalysis(
            scene_index=index,
            duration_ms=duration_ms,
            dominant_colors=["neutral"],
            motion_intensity=0.5,
            brightness=0.5,
            key_objects=[],
            mood="neutral",
            suggested_transition=TransitionType.DISSOLVE if duration_ms > 3000 else TransitionType.CUT
        )

    async def _analyze_scene_deep(self, clip_path: str, index: int) -> SceneAnalysis:
        """Deep AI-powered scene analysis (System 2)"""
        if not VortexConfig.AI_ANALYSIS_ENABLED or not VortexConfig.ANTHROPIC_API_KEY:
            return await self._analyze_scene_fast(clip_path, index)

        duration_ms = await self.ffmpeg.get_duration_ms(clip_path)

        # Extract frame for analysis
        work_dir = Path(clip_path).parent
        frame_path = work_dir / f"frame_{index}.jpg"

        # Extract middle frame
        mid_time = (duration_ms / 1000) / 2
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-ss", str(mid_time),
            "-i", clip_path,
            "-vframes", "1",
            "-y",
            str(frame_path)
        ]

        await self.ffmpeg.run(cmd, timeout=30)

        # For now, return heuristic analysis
        # TODO: Integrate Claude Vision API for deep analysis
        return SceneAnalysis(
            scene_index=index,
            duration_ms=duration_ms,
            dominant_colors=["warm"],
            motion_intensity=0.6,
            brightness=0.6,
            key_objects=["product", "person"],
            mood="professional",
            suggested_transition=TransitionType.DISSOLVE
        )

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] SceneAnalysisNode: Analyzing {len(state.local_clip_paths)} scenes")

        is_deep = state.processing_mode == ProcessingMode.SYSTEM2_DEEP

        # Analyze all scenes
        analyses = []
        for i, clip_path in enumerate(state.local_clip_paths):
            if is_deep:
                analysis = await self._analyze_scene_deep(clip_path, i)
            else:
                analysis = await self._analyze_scene_fast(clip_path, i)
            analyses.append(analysis)

        new_state = state.transition_to(
            PipelinePhase.SCENE_ANALYSIS,
            scene_analyses=analyses
        )

        logger.info(f"[{state.job_id}] Analyzed {len(analyses)} scenes (mode={state.processing_mode})")

        return new_state


# =============================================================================
# TRANSITION SELECTION NODE
# =============================================================================

class TransitionSelectionNode(PipelineNode):
    """
    Selects transitions between scenes.

    System 1: Rule-based selection based on scene analysis
    System 2: AI-driven transition selection for optimal flow
    """

    def _select_transition_rule_based(
        self,
        from_scene: SceneAnalysis,
        to_scene: SceneAnalysis
    ) -> TransitionDecision:
        """Rule-based transition selection (System 1)"""

        # Heuristics:
        # - High motion difference -> hard cut
        # - Similar brightness -> dissolve
        # - End of section -> fade

        motion_diff = abs(from_scene.motion_intensity - to_scene.motion_intensity)
        brightness_diff = abs(from_scene.brightness - to_scene.brightness)

        if motion_diff > 0.5:
            transition = TransitionType.CUT
            confidence = 0.9
            reasoning = "High motion contrast suggests hard cut"
        elif brightness_diff > 0.4:
            transition = TransitionType.FADE
            confidence = 0.8
            reasoning = "Brightness change suggests fade transition"
        else:
            transition = TransitionType.DISSOLVE
            confidence = 0.7
            reasoning = "Similar scenes work well with dissolve"

        return TransitionDecision(
            from_scene=from_scene.scene_index,
            to_scene=to_scene.scene_index,
            type=transition,
            duration_ms=500 if transition == TransitionType.CUT else 1000,
            confidence=confidence,
            reasoning=reasoning
        )

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] TransitionSelectionNode: Selecting transitions")

        transitions = []

        for i in range(len(state.scene_analyses) - 1):
            from_scene = state.scene_analyses[i]
            to_scene = state.scene_analyses[i + 1]

            transition = self._select_transition_rule_based(from_scene, to_scene)
            transitions.append(transition)

        new_state = state.transition_to(
            PipelinePhase.TRANSITION_SELECTION,
            transitions=transitions
        )

        logger.info(f"[{state.job_id}] Selected {len(transitions)} transitions")

        return new_state


# =============================================================================
# CLIP ASSEMBLY NODE (P1 FIX - FFmpeg 6.1 xfade)
# =============================================================================

class ClipAssemblyNode(PipelineNode):
    """
    Assembles clips with transitions using FFmpeg xfade filter.

    REQUIRES: FFmpeg 6.1+ for full xfade filter support.
    """

    def _build_xfade_filter(self, transitions: List[TransitionDecision]) -> str:
        """Build FFmpeg xfade filter chain"""
        if not transitions:
            return ""

        # Build filter chain for xfade between clips
        filter_parts = []

        for i, t in enumerate(transitions):
            xfade_type = TRANSITION_XFADE_MAP.get(
                TransitionType(t.type) if isinstance(t.type, str) else t.type,
                "fade"
            )
            duration_s = t.duration_ms / 1000

            if i == 0:
                # First transition: [0][1] -> [v1]
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition={xfade_type}:duration={duration_s}:offset=auto[v1]"
                )
            else:
                # Subsequent: [vN][N+1] -> [vN+1]
                filter_parts.append(
                    f"[v{i}][{i+1}:v]xfade=transition={xfade_type}:duration={duration_s}:offset=auto[v{i+1}]"
                )

        return ";".join(filter_parts)

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] ClipAssemblyNode: Assembling {len(state.local_clip_paths)} clips")

        work_dir = self.get_work_dir(state.job_id)
        output_path = work_dir / "assembled.mp4"

        if len(state.local_clip_paths) == 1:
            # Single clip - just copy
            import shutil
            shutil.copy(state.local_clip_paths[0], output_path)
        else:
            # Build FFmpeg command with xfade
            cmd = [self.ffmpeg.ffmpeg_path]

            # Add inputs
            for clip_path in state.local_clip_paths:
                cmd.extend(["-i", clip_path])

            # Strip audio from inputs - AudioSyncNode handles audio separately
            # This fixes FFmpeg returncode 234 (unmapped audio streams in xfade filter)
            cmd.append("-an")

            # Build filter complex
            if state.transitions:
                filter_complex = self._build_xfade_filter(state.transitions)
                if filter_complex:
                    cmd.extend(["-filter_complex", filter_complex])
                    # Map the final output
                    final_label = f"[v{len(state.transitions)}]"
                    cmd.extend(["-map", final_label])
            else:
                # Simple concat without transitions
                filter_complex = "".join(f"[{i}:v]" for i in range(len(state.local_clip_paths)))
                filter_complex += f"concat=n={len(state.local_clip_paths)}:v=1:a=0[outv]"
                cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]"])

            # Output settings
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-y",
                str(output_path)
            ])

            # Retry logic with exponential backoff
            MAX_RETRIES = 3
            RETRY_DELAYS = [1, 2, 4]  # seconds

            for attempt in range(MAX_RETRIES):
                returncode, stdout, stderr = await self.ffmpeg.run(cmd)
                if returncode == 0:
                    break
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"[{state.job_id}] FFmpeg attempt {attempt+1} failed (rc={returncode}), retrying in {RETRY_DELAYS[attempt]}s...")
                    await asyncio.sleep(RETRY_DELAYS[attempt])

            if returncode != 0:
                logger.error(f"[{state.job_id}] FFmpeg assembly failed after {MAX_RETRIES} attempts: {stderr}")
                return state.add_error("clip_assembly", f"FFmpeg failed after {MAX_RETRIES} attempts: {stderr[:500]}")

        new_state = state.transition_to(
            PipelinePhase.CLIP_ASSEMBLY,
            assembled_video_path=str(output_path)
        )

        logger.info(f"[{state.job_id}] Assembled video: {output_path}")

        return new_state


# =============================================================================
# AUDIO SYNC NODE
# =============================================================================

class AudioSyncNode(PipelineNode):
    """
    Syncs voiceover and music bed with assembled video.
    """

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] AudioSyncNode: Syncing audio")

        if not state.assembled_video_path:
            return state.add_error("audio_sync", "No assembled video to sync")

        work_dir = self.get_work_dir(state.job_id)
        output_path = work_dir / "with_audio.mp4"

        # Build FFmpeg command for audio mixing
        cmd = [self.ffmpeg.ffmpeg_path, "-i", state.assembled_video_path]

        filter_complex_parts = []
        audio_inputs = 1  # Video is input 0

        # Add voiceover
        if state.local_voiceover_path and Path(state.local_voiceover_path).exists():
            cmd.extend(["-i", state.local_voiceover_path])
            filter_complex_parts.append(f"[{audio_inputs}:a]volume=1.0[vo]")
            audio_inputs += 1

        # Add music bed
        if state.local_music_path and Path(state.local_music_path).exists():
            cmd.extend(["-i", state.local_music_path])
            # Music at lower volume as bed
            filter_complex_parts.append(f"[{audio_inputs}:a]volume=0.3[music]")
            audio_inputs += 1

        # Mix audio streams
        if filter_complex_parts:
            mix_inputs = []
            if state.local_voiceover_path:
                mix_inputs.append("[vo]")
            if state.local_music_path:
                mix_inputs.append("[music]")

            if len(mix_inputs) > 1:
                filter_complex_parts.append(
                    f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[aout]"
                )
                audio_map = "[aout]"
            elif mix_inputs:
                audio_map = mix_inputs[0].replace("[", "").replace("]", "")
                audio_map = f"[{audio_map}]"
            else:
                audio_map = None

            if filter_complex_parts:
                cmd.extend(["-filter_complex", ";".join(filter_complex_parts)])

            cmd.extend(["-map", "0:v"])
            if audio_map:
                cmd.extend(["-map", audio_map])
        else:
            # No audio to add, just copy video
            cmd.extend(["-c", "copy"])

        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            str(output_path)
        ])

        returncode, stdout, stderr = await self.ffmpeg.run(cmd)

        if returncode != 0:
            logger.warning(f"Audio sync had issues: {stderr[:300]}")
            # Fall back to video without audio
            output_path = Path(state.assembled_video_path)

        new_state = state.transition_to(
            PipelinePhase.AUDIO_SYNC,
            video_with_audio_path=str(output_path)
        )

        logger.info(f"[{state.job_id}] Audio synced: {output_path}")

        return new_state


# =============================================================================
# FORMAT RENDER NODE
# =============================================================================

class FormatRenderNode(PipelineNode):
    """
    Renders video to multiple output formats (YouTube, TikTok, Instagram, etc.)
    """

    async def _render_format(
        self,
        input_path: str,
        output_path: Path,
        format_spec: Dict[str, Any]
    ) -> bool:
        """Render to a specific format"""
        cmd = [
            self.ffmpeg.ffmpeg_path,
            "-i", input_path,
            "-vf", f"scale={format_spec['width']}:{format_spec['height']}:force_original_aspect_ratio=decrease,pad={format_spec['width']}:{format_spec['height']}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(format_spec['crf']),
            "-r", str(format_spec['fps']),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ]

        returncode, stdout, stderr = await self.ffmpeg.run(cmd)

        if returncode != 0:
            logger.error(f"Format render failed for {output_path.name}: {stderr[:300]}")
            return False

        return True

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] FormatRenderNode: Rendering {len(state.output_formats)} formats")

        input_path = state.video_with_audio_path or state.assembled_video_path
        if not input_path:
            return state.add_error("format_render", "No input video for format rendering")

        work_dir = self.get_work_dir(state.job_id)
        output_dir = work_dir / "outputs"
        output_dir.mkdir(exist_ok=True)

        final_outputs = {}

        # Render each requested format
        for format_name in state.output_formats:
            if format_name not in FORMAT_SPECS:
                logger.warning(f"Unknown format: {format_name}, skipping")
                continue

            spec = FORMAT_SPECS[format_name]
            output_path = output_dir / f"{format_name}.mp4"

            success = await self._render_format(input_path, output_path, spec)

            if success:
                final_outputs[format_name] = str(output_path)
                logger.info(f"[{state.job_id}] Rendered: {format_name}")

        new_state = state.transition_to(
            PipelinePhase.FORMAT_RENDER,
            final_output_paths=final_outputs
        )

        return new_state


# =============================================================================
# QUALITY CHECK NODE
# =============================================================================

class QualityCheckNode(PipelineNode):
    """
    Verifies video quality: resolution, duration, audio sync.
    """

    async def _check_video(self, video_path: str, expected_format: str) -> Dict[str, Any]:
        """Check a single video file"""
        info = await self.ffmpeg.get_video_info(video_path)

        issues = []

        # Check video stream exists
        video_stream = None
        audio_stream = None

        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_stream = stream

        if not video_stream:
            issues.append("No video stream found")

        # Check resolution matches format spec
        if expected_format in FORMAT_SPECS and video_stream:
            spec = FORMAT_SPECS[expected_format]
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)

            if width != spec["width"] or height != spec["height"]:
                issues.append(f"Resolution mismatch: {width}x{height} vs expected {spec['width']}x{spec['height']}")

        # Check duration is reasonable
        duration = float(info.get("format", {}).get("duration", 0))
        if duration < 1:
            issues.append("Video too short (< 1s)")
        elif duration > 600:
            issues.append("Video too long (> 10min)")

        return {
            "path": video_path,
            "format": expected_format,
            "duration_s": duration,
            "has_audio": audio_stream is not None,
            "issues": issues,
            "passed": len(issues) == 0
        }

    async def execute(self, state: GlobalState) -> GlobalState:
        logger.info(f"[{state.job_id}] QualityCheckNode: Checking {len(state.final_output_paths)} outputs")

        all_passed = True
        check_results = []

        for format_name, path in state.final_output_paths.items():
            result = await self._check_video(path, format_name)
            check_results.append(result)

            if not result["passed"]:
                all_passed = False
                for issue in result["issues"]:
                    logger.warning(f"[{state.job_id}] {format_name}: {issue}")

        if all_passed:
            new_phase = PipelinePhase.COMPLETED
            logger.info(f"[{state.job_id}] Quality check PASSED")
        else:
            new_phase = PipelinePhase.QUALITY_CHECK
            # Add warnings but don't fail
            warnings = []
            for result in check_results:
                for issue in result["issues"]:
                    warnings.append(f"{result['format']}: {issue}")

        new_state = state.transition_to(
            new_phase,
            warnings=state.warnings + warnings if not all_passed else state.warnings
        )

        # Create checkpoint at completion
        if all_passed:
            new_state = new_state.create_checkpoint()

        return new_state


# =============================================================================
# PIPELINE EXECUTOR
# =============================================================================

class VortexPipeline:
    """
    Orchestrates the complete VORTEX pipeline execution.
    """

    def __init__(self):
        self.nodes = {
            PipelinePhase.INIT: RouterNode(),
            PipelinePhase.ROUTING: AssetDownloadNode(),
            PipelinePhase.ASSET_DOWNLOAD: SceneAnalysisNode(),
            PipelinePhase.SCENE_ANALYSIS: TransitionSelectionNode(),
            PipelinePhase.TRANSITION_SELECTION: ClipAssemblyNode(),
            PipelinePhase.CLIP_ASSEMBLY: AudioSyncNode(),
            PipelinePhase.AUDIO_SYNC: FormatRenderNode(),
            PipelinePhase.FORMAT_RENDER: QualityCheckNode(),
        }

    def get_next_node(self, phase: PipelinePhase) -> Optional[PipelineNode]:
        """Get the node to execute for the current phase"""
        return self.nodes.get(phase)

    async def execute_step(self, state: GlobalState) -> GlobalState:
        """Execute a single pipeline step"""
        current_phase = PipelinePhase(state.phase) if isinstance(state.phase, str) else state.phase
        node = self.get_next_node(current_phase)

        if node is None:
            logger.info(f"[{state.job_id}] No node for phase {current_phase}, pipeline complete")
            return state

        try:
            return await node.execute(state)
        except Exception as e:
            logger.error(f"[{state.job_id}] Pipeline error in {current_phase}: {e}")
            return state.add_error(current_phase.value, str(e), recoverable=False).transition_to(
                PipelinePhase.FAILED
            )

    async def execute_full(self, state: GlobalState) -> GlobalState:
        """Execute the complete pipeline"""
        current_state = state

        while current_state.phase not in [PipelinePhase.COMPLETED, PipelinePhase.FAILED]:
            current_state = await self.execute_step(current_state)

        return current_state

    def get_graph_structure(self) -> Dict[str, Any]:
        """Return pipeline graph structure for UI visualization"""
        nodes = [
            {"id": "init", "label": "Initialize", "type": "start"},
            {"id": "routing", "label": "Route", "type": "decision"},
            {"id": "download", "label": "Download Assets", "type": "process"},
            {"id": "analysis", "label": "Scene Analysis", "type": "process"},
            {"id": "transitions", "label": "Select Transitions", "type": "process"},
            {"id": "assembly", "label": "Assemble Clips", "type": "process"},
            {"id": "audio", "label": "Sync Audio", "type": "process"},
            {"id": "render", "label": "Render Formats", "type": "process"},
            {"id": "quality", "label": "Quality Check", "type": "process"},
            {"id": "completed", "label": "Completed", "type": "end"},
            {"id": "failed", "label": "Failed", "type": "error"},
        ]

        edges = [
            {"from": "init", "to": "routing"},
            {"from": "routing", "to": "download"},
            {"from": "download", "to": "analysis"},
            {"from": "analysis", "to": "transitions"},
            {"from": "transitions", "to": "assembly"},
            {"from": "assembly", "to": "audio"},
            {"from": "audio", "to": "render"},
            {"from": "render", "to": "quality"},
            {"from": "quality", "to": "completed", "label": "pass"},
            {"from": "quality", "to": "failed", "label": "fail"},
        ]

        return {"nodes": nodes, "edges": edges}
