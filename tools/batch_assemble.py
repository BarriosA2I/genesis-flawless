"""
================================================================================
BATCH VIDEO ASSEMBLY TOOL
================================================================================
Assembles scene batches from the BARRIOS A2I LAUNCH folder into finished
commercials using FFmpeg with professional transitions.

Usage:
    python batch_assemble.py --source "path/to/scenes" --output "path/to/output"

Features:
- Automatically groups scenes by creation timestamp (within 120s = same batch)
- Applies dissolve/fade transitions between scenes
- Supports optional voiceover and music track overlay
- Outputs 1080p YouTube-ready MP4s

Author: Barrios A2I | 2026-01-11
================================================================================
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AssemblyConfig:
    """Configuration for batch assembly"""
    source_dir: Path
    output_dir: Path
    batch_threshold_seconds: int = 120  # Files within 120s are same batch
    transition_duration_ms: int = 500   # 0.5s dissolve between scenes
    transition_type: str = "fade"        # fade, dissolve, wipe
    output_format: str = "youtube_1080p"
    voiceover_path: Optional[Path] = None
    music_path: Optional[Path] = None
    music_volume: float = 0.3           # Background music at 30%
    verbose: bool = True


@dataclass
class SceneFile:
    """Represents a single scene video file"""
    path: Path
    created_at: datetime
    size_bytes: int
    duration_ms: int = 0

    @classmethod
    def from_path(cls, file_path: Path) -> 'SceneFile':
        """Create SceneFile from filesystem path"""
        import re

        stat = file_path.stat()

        # Try to extract Unix timestamp from filename (e.g., "hash_1767936258.mp4")
        timestamp_match = re.search(r'_(\d{10})(?:\s*\(\d+\))?\.\w+$', file_path.name)

        if timestamp_match:
            # Use timestamp from filename
            unix_ts = int(timestamp_match.group(1))
            created_at = datetime.fromtimestamp(unix_ts)
        else:
            # Fall back to filesystem creation time
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

    @property
    def total_size_mb(self) -> float:
        return sum(s.size_bytes for s in self.scenes) / (1024 * 1024)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def add_scene(self, scene: SceneFile):
        self.scenes.append(scene)
        self.scenes.sort(key=lambda s: s.created_at)
        if not self.created_at or scene.created_at < self.created_at:
            self.created_at = scene.created_at


# =============================================================================
# FFMPEG UTILITIES
# =============================================================================

class FFmpegAssembler:
    """FFmpeg-based video assembly with transitions"""

    def __init__(self, config: AssemblyConfig):
        self.config = config
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg binary"""
        for path in ["/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "ffmpeg", "C:\\ffmpeg\\bin\\ffmpeg.exe"]:
            if os.path.exists(path) or path in ["ffmpeg", "ffmpeg.exe"]:
                return path
        return "ffmpeg"

    def _find_ffprobe(self) -> str:
        """Find FFprobe binary"""
        for path in ["/usr/local/bin/ffprobe", "/usr/bin/ffprobe", "ffprobe", "C:\\ffmpeg\\bin\\ffprobe.exe"]:
            if os.path.exists(path) or path in ["ffprobe", "ffprobe.exe"]:
                return path
        return "ffprobe"

    def get_video_duration_ms(self, video_path: Path) -> int:
        """Get video duration in milliseconds using ffprobe"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data.get("format", {}).get("duration", 0))
                return int(duration * 1000)
        except Exception as e:
            print(f"  Warning: Could not get duration for {video_path.name}: {e}")

        return 0

    def assemble_batch(self, batch: ProductionBatch, output_path: Path) -> bool:
        """Assemble a batch of scenes into a single video with transitions"""

        if len(batch.scenes) == 0:
            print(f"  Error: Batch {batch.batch_id} has no scenes")
            return False

        print(f"\n  Assembling {batch.scene_count} scenes...")

        # Single scene - just copy
        if len(batch.scenes) == 1:
            return self._copy_single(batch.scenes[0].path, output_path)

        # Multiple scenes - concatenate with transitions
        return self._assemble_with_transitions(batch.scenes, output_path)

    def _copy_single(self, source: Path, dest: Path) -> bool:
        """Copy a single video file"""
        cmd = [
            self.ffmpeg_path,
            "-i", str(source),
            "-c", "copy",
            "-y",
            str(dest)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0

    def _assemble_with_transitions(self, scenes: List[SceneFile], output_path: Path) -> bool:
        """Assemble multiple scenes with xfade transitions"""

        # Get durations for calculating offsets
        durations = []
        for scene in scenes:
            duration_ms = self.get_video_duration_ms(scene.path)
            durations.append(duration_ms)
            if self.config.verbose:
                print(f"    Scene {scene.path.name}: {duration_ms}ms")

        # Build FFmpeg command
        cmd = [self.ffmpeg_path]

        # Add all inputs
        for scene in scenes:
            cmd.extend(["-i", str(scene.path)])

        # Strip audio from video streams (handle separately)
        # Build filter complex for xfade transitions
        transition_duration = self.config.transition_duration_ms / 1000
        filter_parts = []

        # Calculate offsets for each transition
        # offset = sum of previous clip durations - sum of previous transitions
        cumulative_offset = 0

        for i in range(len(scenes) - 1):
            clip_duration = durations[i] / 1000  # Convert to seconds

            if i == 0:
                # First transition: [0][1] -> [v1]
                offset = clip_duration - transition_duration
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition={self.config.transition_type}:duration={transition_duration}:offset={offset}[v1]"
                )
                cumulative_offset = offset + (durations[1] / 1000)
            else:
                # Subsequent transitions: [vN][N+1] -> [vN+1]
                offset = cumulative_offset - transition_duration
                filter_parts.append(
                    f"[v{i}][{i+1}:v]xfade=transition={self.config.transition_type}:duration={transition_duration}:offset={offset}[v{i+1}]"
                )
                if i + 1 < len(durations):
                    cumulative_offset = offset + (durations[i+1] / 1000)

        filter_complex = ";".join(filter_parts)
        final_video_label = f"[v{len(scenes)-1}]"

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", final_video_label])

        # Try to include audio from first video that has it
        # Use audio from first input as base
        cmd.extend(["-map", "0:a?"])

        # Output settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ])

        if self.config.verbose:
            print(f"    Running FFmpeg assembly...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"    Error: FFmpeg failed with code {result.returncode}")
            print(f"    stderr: {result.stderr[:500]}")

            # Fallback: simple concat without transitions
            print(f"    Retrying with simple concat...")
            return self._simple_concat(scenes, output_path)

        return True

    def _simple_concat(self, scenes: List[SceneFile], output_path: Path) -> bool:
        """Fallback: Simple concatenation without transitions"""

        # Create concat file
        concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
        with open(concat_file, "w") as f:
            for scene in scenes:
                # Escape Windows paths
                escaped_path = str(scene.path).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Cleanup concat file
        try:
            concat_file.unlink()
        except:
            pass

        return result.returncode == 0

    def add_audio_tracks(self, video_path: Path, voiceover: Optional[Path], music: Optional[Path], output_path: Path) -> bool:
        """Add voiceover and/or music to assembled video"""

        if not voiceover and not music:
            return True  # Nothing to add

        cmd = [self.ffmpeg_path, "-i", str(video_path)]

        filter_parts = []
        audio_inputs = 1

        if voiceover and voiceover.exists():
            cmd.extend(["-i", str(voiceover)])
            filter_parts.append(f"[{audio_inputs}:a]volume=1.0[vo]")
            audio_inputs += 1

        if music and music.exists():
            cmd.extend(["-i", str(music)])
            filter_parts.append(f"[{audio_inputs}:a]volume={self.config.music_volume}[music]")
            audio_inputs += 1

        if not filter_parts:
            return True

        # Mix audio
        mix_inputs = []
        if voiceover and voiceover.exists():
            mix_inputs.append("[vo]")
        if music and music.exists():
            mix_inputs.append("[music]")

        if len(mix_inputs) > 1:
            filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[aout]")
            audio_map = "[aout]"
        else:
            audio_map = mix_inputs[0] if mix_inputs else None

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", "0:v"])
        if audio_map:
            cmd.extend(["-map", audio_map])

        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            str(output_path)
        ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0


# =============================================================================
# BATCH GROUPING
# =============================================================================

def scan_source_directory(source_dir: Path) -> List[SceneFile]:
    """Scan source directory for video files"""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    scene_files = []
    for file_path in source_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            scene = SceneFile.from_path(file_path)
            scene_files.append(scene)

    # Sort by creation time
    scene_files.sort(key=lambda s: s.created_at)
    return scene_files


def group_into_batches(scene_files: List[SceneFile], threshold_seconds: int = 120) -> List[ProductionBatch]:
    """Group scene files into production batches based on creation time proximity"""

    if not scene_files:
        return []

    batches = []
    current_batch = ProductionBatch(batch_id=f"batch_001")
    current_batch.add_scene(scene_files[0])

    for scene in scene_files[1:]:
        # Check time difference from last scene in batch
        last_scene_time = current_batch.scenes[-1].created_at
        time_diff = (scene.created_at - last_scene_time).total_seconds()

        if time_diff <= threshold_seconds:
            # Same batch
            current_batch.add_scene(scene)
        else:
            # New batch
            batches.append(current_batch)
            batch_num = len(batches) + 1
            current_batch = ProductionBatch(batch_id=f"batch_{batch_num:03d}")
            current_batch.add_scene(scene)

    # Add final batch
    batches.append(current_batch)

    return batches


def print_batch_summary(batches: List[ProductionBatch]):
    """Print summary of detected batches"""
    print("\n" + "=" * 70)
    print("PRODUCTION BATCHES DETECTED")
    print("=" * 70)

    total_scenes = 0
    total_size_mb = 0

    for batch in batches:
        total_scenes += batch.scene_count
        total_size_mb += batch.total_size_mb

        batch_type = "4-SCENE COMMERCIAL" if batch.scene_count == 4 else f"{batch.scene_count}-SCENE VARIANT"

        print(f"\n{batch.batch_id}: {batch_type}")
        print(f"  Created: {batch.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Size: {batch.total_size_mb:.1f} MB")
        print(f"  Scenes:")
        for scene in batch.scenes:
            print(f"    - {scene.path.name}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {len(batches)} batches, {total_scenes} scenes, {total_size_mb:.1f} MB")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch assemble scene files into finished commercials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic assembly
  python batch_assemble.py --source "./scenes" --output "./assembled"

  # With music track
  python batch_assemble.py --source "./scenes" --output "./assembled" --music "./music/background.mp3"

  # Assemble only specific batch
  python batch_assemble.py --source "./scenes" --output "./assembled" --batch batch_003
        """
    )

    parser.add_argument("--source", required=True, help="Source directory with scene files")
    parser.add_argument("--output", required=True, help="Output directory for assembled videos")
    parser.add_argument("--voiceover", help="Optional voiceover audio file")
    parser.add_argument("--music", help="Optional background music file")
    parser.add_argument("--music-volume", type=float, default=0.3, help="Music volume (0.0-1.0, default: 0.3)")
    parser.add_argument("--batch", help="Assemble only specific batch ID (e.g., batch_003)")
    parser.add_argument("--transition", choices=["fade", "dissolve", "wipe"], default="fade", help="Transition type")
    parser.add_argument("--transition-duration", type=int, default=500, help="Transition duration in ms")
    parser.add_argument("--threshold", type=int, default=120, help="Time threshold for batch grouping in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be assembled without doing it")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate paths
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = AssemblyConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        batch_threshold_seconds=args.threshold,
        transition_duration_ms=args.transition_duration,
        transition_type=args.transition,
        voiceover_path=Path(args.voiceover) if args.voiceover else None,
        music_path=Path(args.music) if args.music else None,
        music_volume=args.music_volume,
        verbose=not args.quiet
    )

    print("\n" + "=" * 70)
    print("BARRIOS A2I BATCH ASSEMBLY TOOL")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    # Scan and group
    print("\nScanning source directory...")
    scene_files = scan_source_directory(source_dir)
    print(f"Found {len(scene_files)} video files")

    if not scene_files:
        print("Error: No video files found in source directory")
        sys.exit(1)

    # Group into batches
    batches = group_into_batches(scene_files, config.batch_threshold_seconds)
    print_batch_summary(batches)

    if args.dry_run:
        print("\n[DRY RUN] Would assemble the above batches. Use --no-dry-run to execute.")
        sys.exit(0)

    # Filter to specific batch if requested
    if args.batch:
        batches = [b for b in batches if b.batch_id == args.batch]
        if not batches:
            print(f"Error: Batch '{args.batch}' not found")
            sys.exit(1)

    # Assemble each batch
    assembler = FFmpegAssembler(config)

    success_count = 0
    fail_count = 0

    print("\n" + "=" * 70)
    print("STARTING ASSEMBLY")
    print("=" * 70)

    for batch in batches:
        print(f"\n>>> Processing {batch.batch_id} ({batch.scene_count} scenes)...")

        # Generate output filename with timestamp
        timestamp = batch.created_at.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{batch.batch_id}_{timestamp}_assembled.mp4"
        output_path = output_dir / output_filename

        success = assembler.assemble_batch(batch, output_path)

        if success:
            # Add audio if provided
            if config.voiceover_path or config.music_path:
                audio_output = output_dir / f"{batch.batch_id}_{timestamp}_final.mp4"
                success = assembler.add_audio_tracks(
                    output_path,
                    config.voiceover_path,
                    config.music_path,
                    audio_output
                )
                if success:
                    # Remove intermediate file
                    output_path.unlink()
                    output_path = audio_output

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"  SUCCESS: {output_path.name} ({file_size:.1f} MB)")
            success_count += 1
        else:
            print(f"  FAILED: Could not assemble {batch.batch_id}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("ASSEMBLY COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
