"""
================================================================================
RAGNAROK POST-PRODUCTION PIPELINE
================================================================================
Full pipeline for existing video scenes:
- Agent 5: ElevenLabs Voiceover Generation
- Agent 6: Music Selection
- Agent 7: VORTEX Video Assembly

Takes scene batches and runs them through voiceover + music + assembly.

Usage:
    python ragnarok_postprod.py --source "path/to/scenes" --output "path/to/output"

Author: Barrios A2I | 2026-01-11
================================================================================
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
import base64
import tempfile
import re
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import anthropic

# Load environment variables from .env files
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".env")
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the RAGNAROK post-production pipeline"""
    source_dir: Path
    output_dir: Path
    batch_threshold_seconds: int = 120
    elevenlabs_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    voice_style: str = "professional"  # professional, friendly, energetic
    music_style: str = "corporate"     # corporate, upbeat, emotional
    verbose: bool = True


@dataclass
class SceneFile:
    """Represents a single scene video file"""
    path: Path
    created_at: datetime
    size_bytes: int
    duration_ms: int = 0
    frame_path: Optional[Path] = None
    description: Optional[str] = None

    @classmethod
    def from_path(cls, file_path: Path) -> 'SceneFile':
        """Create SceneFile from filesystem path"""
        stat = file_path.stat()

        # Try to extract Unix timestamp from filename
        timestamp_match = re.search(r'_(\d{10})(?:\s*\(\d+\))?\.\w+$', file_path.name)

        if timestamp_match:
            unix_ts = int(timestamp_match.group(1))
            created_at = datetime.fromtimestamp(unix_ts)
        else:
            created_at = datetime.fromtimestamp(stat.st_ctime)

        return cls(
            path=file_path,
            created_at=created_at,
            size_bytes=stat.st_size
        )


@dataclass
class ProductionBatch:
    """A group of scene files that belong to the same commercial"""
    batch_id: str
    scenes: List[SceneFile] = field(default_factory=list)
    created_at: datetime = None
    voiceover_script: Optional[str] = None
    voiceover_path: Optional[Path] = None
    music_path: Optional[Path] = None
    final_output: Optional[Path] = None

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def add_scene(self, scene: SceneFile):
        self.scenes.append(scene)
        self.scenes.sort(key=lambda s: s.created_at)
        if not self.created_at or scene.created_at < self.created_at:
            self.created_at = scene.created_at


# =============================================================================
# VOICE SETTINGS (Agent 5)
# =============================================================================

VOICE_MAPPINGS = {
    "professional": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Rachel - Professional
        "name": "Rachel",
        "description": "Professional female voice"
    },
    "friendly": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Bella - Warm
        "name": "Bella",
        "description": "Warm friendly female voice"
    },
    "energetic": {
        "voice_id": "ErXwobaYiN019PkySvjV",  # Antoni - Energetic
        "name": "Antoni",
        "description": "Energetic male voice"
    },
    "authoritative": {
        "voice_id": "VR6AewLTigWG4xSOukaG",  # Arnold - Deep
        "name": "Arnold",
        "description": "Deep authoritative male voice"
    }
}


# =============================================================================
# SCENE ANALYZER (Claude Vision)
# =============================================================================

class SceneAnalyzer:
    """Analyze video scenes using Claude Vision"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_frame(self, video_path: Path, output_path: Path, time_offset: float = 0.5) -> bool:
        """Extract a frame from video at the middle point"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(time_offset),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0

    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except:
            pass
        return 0

    def analyze_frame(self, frame_path: Path, scene_index: int, scene_type: str) -> str:
        """Analyze a frame using Claude Vision"""

        with open(frame_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt = f"""Analyze this frame from scene {scene_index + 1} (type: {scene_type}) of a commercial video.

Describe in 1-2 sentences:
1. What is shown in the scene (visuals, colors, objects, people)
2. What emotion or message this scene conveys
3. What type of business/product this might be advertising

Be concise and specific."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        return response.content[0].text


# =============================================================================
# SCRIPT GENERATOR (Claude)
# =============================================================================

class ScriptGenerator:
    """Generate voiceover scripts based on scene analysis"""

    SCENE_STRUCTURE = [
        ("HOOK", "Attention-grabbing opening that stops the scroll"),
        ("PROBLEM", "Identify the pain point the viewer can relate to"),
        ("SOLUTION", "Present the product/service as THE answer"),
        ("CTA", "Clear call to action with urgency")
    ]

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_script(self, scene_descriptions: List[str], tone: str = "professional") -> str:
        """Generate a voiceover script based on scene descriptions"""

        # Build scene context
        scene_context = "\n".join([
            f"Scene {i+1} ({self.SCENE_STRUCTURE[i][0] if i < 4 else 'EXTRA'}): {desc}"
            for i, desc in enumerate(scene_descriptions)
        ])

        prompt = f"""Based on these commercial video scenes, write a voiceover script.

SCENE DESCRIPTIONS:
{scene_context}

REQUIREMENTS:
- Total duration: ~30-45 seconds of narration
- Tone: {tone}
- Structure: Follow the HOOK → PROBLEM → SOLUTION → CTA formula
- Each scene gets 2-3 sentences max
- Natural conversational flow
- End with a strong call-to-action

Write ONLY the voiceover script text, nothing else. No scene labels, just the continuous narration."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()


# =============================================================================
# VOICEOVER GENERATOR (ElevenLabs - Agent 5)
# =============================================================================

class VoiceoverAgent:
    """Agent 5: ElevenLabs Voiceover Generation"""

    BASE_URL = "https://api.elevenlabs.io/v1"
    MODEL_ID = "eleven_turbo_v2_5"

    def __init__(self, api_key: str):
        self.api_key = api_key
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY required")

    async def generate(
        self,
        script: str,
        voice_style: str = "professional",
        output_path: Path = None,
        max_duration_seconds: float = 64.0
    ) -> Path:
        """Generate voiceover audio from script

        Args:
            script: The voiceover script text
            voice_style: Voice style from VOICE_MAPPINGS
            output_path: Optional output file path
            max_duration_seconds: Maximum duration for the voiceover (default 64s)
        """
        # Validate and truncate script if too long for target duration
        # At 2.5 words/second, 64 seconds = 160 words max
        words_per_second = 2.5
        max_words = int(max_duration_seconds * words_per_second)

        words = script.split()
        original_word_count = len(words)

        if original_word_count > max_words:
            print(f"    ⚠️  Script too long: {original_word_count} words > {max_words} max for {max_duration_seconds}s")
            print(f"    ⚠️  Truncating script to {max_words} words...")

            # Truncate to max words
            script = " ".join(words[:max_words])

            # Try to end on a complete sentence
            if not script.endswith((".", "!", "?")):
                script = script.rstrip(",;:") + "..."

            print(f"    ✅ Script truncated: {original_word_count} → {max_words} words")
        else:
            print(f"    ✅ Script word count OK: {original_word_count}/{max_words} words")

        voice_info = VOICE_MAPPINGS.get(voice_style, VOICE_MAPPINGS["professional"])

        url = f"{self.BASE_URL}/text-to-speech/{voice_info['voice_id']}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": script,
            "model_id": self.MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "output_format": "mp3_44100_192"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ElevenLabs API error: {response.status} - {error}")

                audio_bytes = await response.read()

                if not output_path:
                    output_path = Path(tempfile.mktemp(suffix=".mp3"))

                output_path.write_bytes(audio_bytes)

                print(f"    Voiceover generated: {output_path.name} ({len(audio_bytes) / 1024:.1f} KB)")
                return output_path


# =============================================================================
# MUSIC SELECTOR (Agent 6 - Simplified)
# =============================================================================

class MusicSelector:
    """Agent 6: Music Selection (uses royalty-free library)"""

    # Placeholder URLs for royalty-free music tracks
    # In production, these would come from a music library API
    MUSIC_LIBRARY = {
        "corporate": [
            "https://cdn.pixabay.com/download/audio/2022/03/15/audio_8cb749e6a9.mp3",  # Corporate
        ],
        "upbeat": [
            "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3",  # Upbeat
        ],
        "emotional": [
            "https://cdn.pixabay.com/download/audio/2021/11/25/audio_91b32e02f9.mp3",  # Emotional
        ],
        "energetic": [
            "https://cdn.pixabay.com/download/audio/2022/10/25/audio_3c18d78d8e.mp3",  # Energetic
        ]
    }

    async def select_and_download(self, style: str, output_path: Path) -> Optional[Path]:
        """Select and download a music track"""

        tracks = self.MUSIC_LIBRARY.get(style, self.MUSIC_LIBRARY["corporate"])
        if not tracks:
            return None

        music_url = tracks[0]  # Take first track for now

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(music_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        audio_bytes = await response.read()
                        output_path.write_bytes(audio_bytes)
                        print(f"    Music downloaded: {output_path.name}")
                        return output_path
        except Exception as e:
            print(f"    Music download failed: {e}")

        return None


# =============================================================================
# VIDEO ASSEMBLER (Agent 7 - VORTEX)
# =============================================================================

class VideoAssembler:
    """Agent 7: VORTEX Video Assembly with FFmpeg"""

    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        self.ffprobe_path = "ffprobe"

    def get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        cmd = [
            self.ffprobe_path, "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except:
            pass
        return 8.0  # Default 8 seconds per scene

    def assemble(
        self,
        scenes: List[SceneFile],
        voiceover_path: Optional[Path],
        music_path: Optional[Path],
        output_path: Path,
        music_volume: float = 0.2
    ) -> bool:
        """Assemble scenes with voiceover and music"""

        # Step 1: Concatenate video scenes with fade transitions
        work_dir = output_path.parent
        video_only = work_dir / f"{output_path.stem}_video.mp4"

        if len(scenes) == 1:
            # Single scene - just copy
            subprocess.run([
                self.ffmpeg_path, "-y",
                "-i", str(scenes[0].path),
                "-c", "copy",
                str(video_only)
            ], capture_output=True, timeout=120)
        else:
            # Multiple scenes - concatenate with transitions
            success = self._concat_with_transitions(scenes, video_only)
            if not success:
                # Fallback to simple concat
                success = self._simple_concat(scenes, video_only)
                if not success:
                    return False

        # Step 2: Mix in voiceover and music
        if voiceover_path or music_path:
            success = self._add_audio_tracks(
                video_only, voiceover_path, music_path,
                output_path, music_volume
            )
            # Cleanup intermediate file
            if video_only.exists() and output_path.exists():
                video_only.unlink()
            return success
        else:
            # No audio to add, just rename
            video_only.rename(output_path)
            return True

    def _concat_with_transitions(self, scenes: List[SceneFile], output_path: Path) -> bool:
        """Concatenate scenes with xfade transitions"""

        cmd = [self.ffmpeg_path]

        # Add inputs
        for scene in scenes:
            cmd.extend(["-i", str(scene.path)])

        # Build xfade filter
        transition_duration = 0.5
        filter_parts = []

        for i in range(len(scenes) - 1):
            duration = self.get_duration(scenes[i].path)
            offset = duration - transition_duration

            if i == 0:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[v1]"
                )
            else:
                filter_parts.append(
                    f"[v{i}][{i+1}:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[v{i+1}]"
                )

        filter_complex = ";".join(filter_parts)
        final_label = f"[v{len(scenes)-1}]"

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", final_label,
            "-an",  # Strip audio for now
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-y",
            str(output_path)
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0

    def _simple_concat(self, scenes: List[SceneFile], output_path: Path) -> bool:
        """Simple concatenation without transitions"""

        concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
        with open(concat_file, "w") as f:
            for scene in scenes:
                escaped = str(scene.path).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-an",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        try:
            concat_file.unlink()
        except:
            pass

        return result.returncode == 0

    def _add_audio_tracks(
        self,
        video_path: Path,
        voiceover_path: Optional[Path],
        music_path: Optional[Path],
        output_path: Path,
        music_volume: float
    ) -> bool:
        """Add voiceover and music to video"""

        cmd = [self.ffmpeg_path, "-y", "-i", str(video_path)]

        filter_parts = []
        audio_inputs = 1

        # Add voiceover
        if voiceover_path and voiceover_path.exists():
            cmd.extend(["-i", str(voiceover_path)])
            filter_parts.append(f"[{audio_inputs}:a]volume=1.0[vo]")
            audio_inputs += 1

        # Add music
        if music_path and music_path.exists():
            cmd.extend(["-i", str(music_path)])
            filter_parts.append(f"[{audio_inputs}:a]volume={music_volume}[music]")
            audio_inputs += 1

        # Mix audio
        if filter_parts:
            mix_inputs = []
            if voiceover_path and voiceover_path.exists():
                mix_inputs.append("[vo]")
            if music_path and music_path.exists():
                mix_inputs.append("[music]")

            if len(mix_inputs) > 1:
                filter_parts.append(
                    f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2[aout]"
                )
                audio_map = "[aout]"
            elif mix_inputs:
                audio_map = mix_inputs[0]
            else:
                audio_map = None

            cmd.extend(["-filter_complex", ";".join(filter_parts)])
            cmd.extend(["-map", "0:v"])
            if audio_map:
                cmd.extend(["-map", audio_map])

        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path)
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def scan_and_group(source_dir: Path, threshold_seconds: int = 120) -> List[ProductionBatch]:
    """Scan source directory and group into production batches"""

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    scene_files = []

    for file_path in source_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            scene = SceneFile.from_path(file_path)
            scene_files.append(scene)

    scene_files.sort(key=lambda s: s.created_at)

    if not scene_files:
        return []

    batches = []
    current_batch = ProductionBatch(batch_id="batch_001")
    current_batch.add_scene(scene_files[0])

    for scene in scene_files[1:]:
        last_scene_time = current_batch.scenes[-1].created_at
        time_diff = (scene.created_at - last_scene_time).total_seconds()

        if time_diff <= threshold_seconds:
            current_batch.add_scene(scene)
        else:
            batches.append(current_batch)
            batch_num = len(batches) + 1
            current_batch = ProductionBatch(batch_id=f"batch_{batch_num:03d}")
            current_batch.add_scene(scene)

    batches.append(current_batch)
    return batches


async def process_batch(
    batch: ProductionBatch,
    config: PipelineConfig,
    scene_analyzer: SceneAnalyzer,
    script_generator: ScriptGenerator,
    voiceover_agent: VoiceoverAgent,
    music_selector: MusicSelector,
    video_assembler: VideoAssembler
) -> bool:
    """Process a single batch through the full RAGNAROK pipeline"""

    work_dir = config.output_dir / batch.batch_id
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PROCESSING {batch.batch_id} ({batch.scene_count} scenes)")
    print(f"{'='*60}")

    # Step 1: Analyze scenes with Claude Vision
    print("\n[1/4] Analyzing scenes with Claude Vision...")
    scene_descriptions = []
    scene_types = ["HOOK", "PROBLEM", "SOLUTION", "CTA", "EXTRA", "EXTRA", "EXTRA"]

    for i, scene in enumerate(batch.scenes):
        frame_path = work_dir / f"frame_{i:02d}.jpg"

        # Get duration
        duration = video_assembler.get_duration(scene.path)
        scene.duration_ms = int(duration * 1000)

        # Extract frame from middle of video
        mid_time = duration / 2
        if scene_analyzer.extract_frame(scene.path, frame_path, mid_time):
            scene.frame_path = frame_path

            # Analyze with Claude Vision
            try:
                description = scene_analyzer.analyze_frame(
                    frame_path, i, scene_types[i] if i < len(scene_types) else "EXTRA"
                )
                scene.description = description
                scene_descriptions.append(description)
                print(f"    Scene {i+1}: {description[:80]}...")
            except Exception as e:
                print(f"    Scene {i+1}: Analysis failed - {e}")
                scene_descriptions.append(f"Scene {i+1} visual content")
        else:
            print(f"    Scene {i+1}: Frame extraction failed")
            scene_descriptions.append(f"Scene {i+1} visual content")

    # Step 2: Generate voiceover script
    print("\n[2/4] Generating voiceover script...")
    try:
        script = script_generator.generate_script(
            scene_descriptions,
            tone=config.voice_style
        )
        batch.voiceover_script = script
        print(f"    Script: {script[:100]}...")

        # Save script to file
        script_path = work_dir / "script.txt"
        script_path.write_text(script)
    except Exception as e:
        print(f"    Script generation failed: {e}")
        return False

    # Step 3: Generate voiceover with ElevenLabs
    print("\n[3/4] Generating voiceover with ElevenLabs...")
    voiceover_path = work_dir / "voiceover.mp3"
    try:
        await voiceover_agent.generate(
            script=batch.voiceover_script,
            voice_style=config.voice_style,
            output_path=voiceover_path
        )
        batch.voiceover_path = voiceover_path
    except Exception as e:
        print(f"    Voiceover generation failed: {e}")
        batch.voiceover_path = None

    # Step 4: Select and download music
    print("\n[4/4] Selecting background music...")
    music_path = work_dir / "music.mp3"
    try:
        result = await music_selector.select_and_download(
            style=config.music_style,
            output_path=music_path
        )
        if result:
            batch.music_path = music_path
    except Exception as e:
        print(f"    Music selection failed: {e}")
        batch.music_path = None

    # Step 5: Assemble final video with VORTEX
    print("\n[5/5] Assembling final video with VORTEX...")
    timestamp = batch.created_at.strftime("%Y%m%d_%H%M%S")
    output_path = config.output_dir / f"{batch.batch_id}_{timestamp}_final.mp4"

    success = video_assembler.assemble(
        scenes=batch.scenes,
        voiceover_path=batch.voiceover_path,
        music_path=batch.music_path,
        output_path=output_path,
        music_volume=0.2
    )

    if success and output_path.exists():
        batch.final_output = output_path
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n    SUCCESS: {output_path.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"\n    FAILED: Assembly failed")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="RAGNAROK Post-Production Pipeline - Full voiceover + music + assembly"
    )

    parser.add_argument("--source", required=True, help="Source directory with scene files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch", help="Process only specific batch ID")
    parser.add_argument("--voice-style", default="professional",
                       choices=["professional", "friendly", "energetic", "authoritative"])
    parser.add_argument("--music-style", default="corporate",
                       choices=["corporate", "upbeat", "emotional", "energetic"])
    parser.add_argument("--dry-run", action="store_true", help="Show batches without processing")
    parser.add_argument("--skip-voiceover", action="store_true", help="Skip voiceover generation")
    parser.add_argument("--skip-music", action="store_true", help="Skip music selection")
    parser.add_argument("--anthropic-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--elevenlabs-key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")

    args = parser.parse_args()

    # Validate paths
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API keys (command line takes precedence over environment)
    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    elevenlabs_key = args.elevenlabs_key or os.getenv("ELEVENLABS_API_KEY")

    # Create config
    config = PipelineConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        anthropic_api_key=anthropic_key,
        elevenlabs_api_key=elevenlabs_key,
        voice_style=args.voice_style,
        music_style=args.music_style
    )

    print("\n" + "=" * 70)
    print("RAGNAROK POST-PRODUCTION PIPELINE")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Voice Style: {args.voice_style}")
    print(f"Music Style: {args.music_style}")

    # Scan and group batches
    print("\nScanning source directory...")
    batches = scan_and_group(source_dir)

    print(f"\nFound {len(batches)} production batches:")
    for batch in batches:
        batch_type = "4-SCENE COMMERCIAL" if batch.scene_count == 4 else f"{batch.scene_count}-SCENE"
        print(f"  {batch.batch_id}: {batch_type} ({batch.created_at.strftime('%Y-%m-%d %H:%M')})")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above batches")
        sys.exit(0)

    # Validate API keys for actual processing
    if not anthropic_key:
        print("\nError: Anthropic API key required. Set ANTHROPIC_API_KEY or use --anthropic-key")
        sys.exit(1)

    if not elevenlabs_key and not args.skip_voiceover:
        print("\nWarning: ElevenLabs API key not set - voiceover will be skipped")
        args.skip_voiceover = True

    # Filter to specific batch if requested
    if args.batch:
        batches = [b for b in batches if b.batch_id == args.batch]
        if not batches:
            print(f"Error: Batch '{args.batch}' not found")
            sys.exit(1)

    # Initialize agents
    scene_analyzer = SceneAnalyzer(anthropic_key)
    script_generator = ScriptGenerator(anthropic_key)

    voiceover_agent = None
    if elevenlabs_key and not args.skip_voiceover:
        voiceover_agent = VoiceoverAgent(elevenlabs_key)

    music_selector = MusicSelector() if not args.skip_music else None
    video_assembler = VideoAssembler()

    # Process each batch
    success_count = 0
    fail_count = 0

    for batch in batches:
        try:
            # Create a modified voiceover agent that returns None if skipped
            vo_agent = voiceover_agent if voiceover_agent else type('DummyAgent', (), {
                'generate': lambda *a, **k: asyncio.coroutine(lambda: None)()
            })()

            music_sel = music_selector if music_selector else type('DummySelector', (), {
                'select_and_download': lambda *a, **k: asyncio.coroutine(lambda: None)()
            })()

            success = await process_batch(
                batch, config,
                scene_analyzer, script_generator,
                voiceover_agent or vo_agent, music_sel,
                video_assembler
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"\nError processing {batch.batch_id}: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
