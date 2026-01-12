"""
================================================================================
REASSEMBLE 64s - Video Batch Assembly Tool
================================================================================
Reassembles 8-second clips into 64-second commercials.
Groups every 8 consecutive clips (sorted by timestamp) regardless of time gaps.

Usage:
    python reassemble_64s.py

Output:
    - batch_001_YYYYMMDD_HHMMSS_64s.mp4 (8 clips = 64 seconds)
    - batch_002_YYYYMMDD_HHMMSS_64s.mp4 (8 clips = 64 seconds)
    - ...
    - batch_006_YYYYMMDD_HHMMSS_64s.mp4 (7 clips = 56 seconds, partial)

Author: Barrios A2I | 2026-01-12
================================================================================
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_DIR = Path(r"C:\Users\gary\Desktop\BARRIOS A2I LAUNCH")
OUTPUT_DIR = Path(r"C:\Users\gary\Desktop\RAGNAROK_OUTPUT")
CLIPS_PER_BATCH = 8
START_BATCH_NUM = 1


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_sorted_clips() -> List[Tuple[Path, int]]:
    """Get all video clips sorted by timestamp extracted from filename.

    Returns:
        List of (path, timestamp) tuples sorted by timestamp
    """
    clips = []

    for f in SOURCE_DIR.glob("*.mp4"):
        # Extract Unix timestamp from filename (e.g., "hash_1767936258.mp4")
        match = re.search(r'_(\d{10})', f.name)
        if match:
            ts = int(match.group(1))
            clips.append((f, ts))
        else:
            # Fallback to file modification time
            ts = int(f.stat().st_mtime)
            clips.append((f, ts))

    # Sort by timestamp
    return sorted(clips, key=lambda x: x[1])


def get_video_duration(path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def create_concat_file(clips: List[Tuple[Path, int]], output_path: Path) -> Path:
    """Create FFmpeg concat demuxer file.

    Args:
        clips: List of (path, timestamp) tuples
        output_path: Path for the output video (used to name concat file)

    Returns:
        Path to the concat file
    """
    concat_file = output_path.with_suffix('.txt')

    with open(concat_file, 'w', encoding='utf-8') as f:
        for clip_path, _ in clips:
            # Use forward slashes and escape single quotes
            escaped = str(clip_path).replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    return concat_file


def assemble_batch(clips: List[Tuple[Path, int]], batch_num: int) -> Path:
    """Assemble clips into a single video using FFmpeg concat demuxer.

    Args:
        clips: List of (path, timestamp) tuples to assemble
        batch_num: Batch number for output filename

    Returns:
        Path to the assembled video
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"batch_{batch_num:03d}_{timestamp}_64s.mp4"
    output_path = OUTPUT_DIR / output_name

    # Create concat file
    concat_file = create_concat_file(clips, output_path)

    # FFmpeg command for concatenation
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-y',
        str(output_path)
    ]

    print(f"    Running FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Cleanup concat file
    try:
        concat_file.unlink()
    except:
        pass

    if result.returncode != 0:
        print(f"    ERROR: FFmpeg failed")
        print(f"    stderr: {result.stderr[:500]}")
        raise RuntimeError(f"FFmpeg failed for batch {batch_num}")

    return output_path


def main():
    print("=" * 70)
    print("BARRIOS A2I - 64-SECOND BATCH REASSEMBLY")
    print("=" * 70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Clips per batch: {CLIPS_PER_BATCH}")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get sorted clips
    clips = get_sorted_clips()
    print(f"Found {len(clips)} video clips")

    if not clips:
        print("ERROR: No video clips found in source directory")
        sys.exit(1)

    # Show clip info
    print("\nClip durations:")
    total_duration = 0
    for clip_path, ts in clips[:5]:
        dur = get_video_duration(clip_path)
        total_duration += dur
        dt = datetime.fromtimestamp(ts)
        print(f"  {clip_path.name[:40]}... {dur:.1f}s @ {dt}")
    if len(clips) > 5:
        print(f"  ... and {len(clips) - 5} more clips")

    # Calculate expected batches
    num_batches = (len(clips) + CLIPS_PER_BATCH - 1) // CLIPS_PER_BATCH
    print(f"\nWill create {num_batches} batches")

    # Assemble batches
    print("\n" + "=" * 70)
    print("STARTING ASSEMBLY")
    print("=" * 70)

    batch_num = START_BATCH_NUM
    success_count = 0
    fail_count = 0

    for i in range(0, len(clips), CLIPS_PER_BATCH):
        batch_clips = clips[i:i + CLIPS_PER_BATCH]
        expected_duration = len(batch_clips) * 8  # 8 seconds per clip

        print(f"\n>>> Batch {batch_num:03d}: {len(batch_clips)} clips (~{expected_duration}s)")
        for clip_path, _ in batch_clips:
            print(f"    - {clip_path.name[:50]}")

        try:
            output_path = assemble_batch(batch_clips, batch_num)

            # Verify output
            actual_duration = get_video_duration(output_path)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            print(f"    SUCCESS: {output_path.name}")
            print(f"    Duration: {actual_duration:.1f}s | Size: {file_size_mb:.1f} MB")

            success_count += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            fail_count += 1

        batch_num += 1

    # Summary
    print("\n" + "=" * 70)
    print("ASSEMBLY COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # List output files
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob("batch_*_64s.mp4")):
        dur = get_video_duration(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {dur:.1f}s, {size_mb:.1f} MB")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
