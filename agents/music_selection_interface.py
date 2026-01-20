"""
RAGNAROK v8.0 - Music Selection Interface
═══════════════════════════════════════════════════════════════════════════════
Interactive music selection - user MUST choose before video production.

This interface scans available music directories, displays track information,
and BLOCKS the pipeline until the user confirms their music selection.

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

logger = logging.getLogger("ragnarok.music_selection")


# =============================================================================
# DATA MODELS
# =============================================================================

class MusicTrack(BaseModel):
    """Represents a music track available for selection."""
    id: str
    title: str
    artist: str = "Unknown Artist"
    path: str
    duration_sec: float
    duration_formatted: str = ""
    bpm: Optional[int] = None
    genre: List[str] = Field(default_factory=list)
    mood: List[str] = Field(default_factory=list)
    file_size_mb: float = 0.0
    sample_rate: Optional[int] = None
    bitrate_kbps: Optional[int] = None
    source_dir: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-format duration
        if self.duration_sec and not self.duration_formatted:
            mins = int(self.duration_sec // 60)
            secs = int(self.duration_sec % 60)
            self.duration_formatted = f"{mins}:{secs:02d}"


class DuckingConfig(BaseModel):
    """FFmpeg sidechain compression config for audio ducking."""
    type: str = "sidechain"
    threshold: float = 0.1
    ratio: int = 4
    attack: int = 200
    release: int = 500
    music_base_volume: float = 0.20  # 15-25% range for background


class MusicSelectionResult(BaseModel):
    """Result of music selection."""
    selected_track: MusicTrack
    ducking_config: DuckingConfig
    user_confirmed: bool = True
    selection_method: str = "user_interactive"  # or "default", "api"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# MUSIC SELECTION INTERFACE
# =============================================================================

class MusicSelectionInterface:
    """
    Interactive music selection interface.

    BLOCKS pipeline until user confirms their music choice.
    Scans directories, displays options, and requires explicit selection.
    """

    MUSIC_DIRS = [
        Path(r"C:\Users\gary\Downloads\_BARROSA2I\MEDIA\A2I_ASSETS"),
        Path(r"C:\Users\gary\Downloads\_BARROSA2I\MEDIA\MUSIC"),
        Path(r"C:\Users\gary\Downloads\_BARROSA2I\MEDIA\AUDIO"),
    ]

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

    # Default ducking config (15-25% volume range)
    DEFAULT_DUCKING = DuckingConfig(
        threshold=0.1,
        ratio=4,
        attack=200,
        release=500,
        music_base_volume=0.20
    )

    def __init__(
        self,
        additional_dirs: Optional[List[Path]] = None,
        ffprobe_path: str = "ffprobe"
    ):
        self.music_dirs = list(self.MUSIC_DIRS)
        if additional_dirs:
            self.music_dirs.extend(additional_dirs)

        self.ffprobe = ffprobe_path
        self._tracks_cache: List[MusicTrack] = []
        self._last_scan_time: Optional[datetime] = None

        logger.info(f"[MusicSelectionInterface] Initialized with {len(self.music_dirs)} search directories")

    async def scan_available_tracks(self, force_rescan: bool = False) -> List[MusicTrack]:
        """
        Scan all music directories for available tracks.

        Returns:
            List of MusicTrack objects with metadata
        """
        # Use cache if available and recent (within 5 minutes)
        if (not force_rescan and
            self._tracks_cache and
            self._last_scan_time and
            (datetime.now() - self._last_scan_time).seconds < 300):
            return self._tracks_cache

        tracks: List[MusicTrack] = []
        track_id = 0

        for music_dir in self.music_dirs:
            if not music_dir.exists():
                logger.debug(f"[MusicSelectionInterface] Directory not found: {music_dir}")
                continue

            logger.info(f"[MusicSelectionInterface] Scanning: {music_dir}")

            for file_path in music_dir.rglob("*"):
                if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                    continue

                if not file_path.is_file():
                    continue

                try:
                    # Get duration and metadata using ffprobe
                    metadata = await self._get_track_metadata(str(file_path))

                    # Extract title from filename (remove extension)
                    title = file_path.stem
                    # Try to extract artist if filename contains " - "
                    artist = "Stock Library"
                    if " - " in title:
                        parts = title.split(" - ", 1)
                        artist = parts[0].strip()
                        title = parts[1].strip()

                    track = MusicTrack(
                        id=f"track_{track_id:04d}",
                        title=title,
                        artist=artist,
                        path=str(file_path),
                        duration_sec=metadata.get("duration", 0.0),
                        file_size_mb=file_path.stat().st_size / (1024 * 1024),
                        sample_rate=metadata.get("sample_rate"),
                        bitrate_kbps=metadata.get("bitrate_kbps"),
                        source_dir=str(music_dir),
                    )

                    tracks.append(track)
                    track_id += 1

                except Exception as e:
                    logger.warning(f"[MusicSelectionInterface] Failed to process {file_path}: {e}")

        # Sort by title
        tracks.sort(key=lambda t: t.title.lower())

        # Update cache
        self._tracks_cache = tracks
        self._last_scan_time = datetime.now()

        logger.info(f"[MusicSelectionInterface] Found {len(tracks)} music tracks")
        return tracks

    async def _get_track_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get track metadata using ffprobe."""
        try:
            cmd = [
                self.ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await process.communicate()

            if process.returncode != 0:
                return {}

            data = json.loads(stdout.decode())
            format_info = data.get("format", {})

            duration = float(format_info.get("duration", 0))
            bitrate = int(format_info.get("bit_rate", 0)) // 1000 if format_info.get("bit_rate") else None

            # Get sample rate from audio stream
            sample_rate = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sample_rate = int(stream.get("sample_rate", 0)) if stream.get("sample_rate") else None
                    break

            return {
                "duration": duration,
                "bitrate_kbps": bitrate,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            logger.debug(f"[MusicSelectionInterface] ffprobe error for {file_path}: {e}")
            return {}

    def display_track_list(self, tracks: List[MusicTrack]) -> str:
        """
        Format track list for display.

        Returns:
            Formatted string showing all available tracks
        """
        lines = [
            "",
            "=" * 80,
            "                    RAGNAROK v8.0 - MUSIC SELECTION",
            "=" * 80,
            "",
            "Available Music Tracks:",
            "-" * 80,
            f"{'#':<4} {'Title':<35} {'Artist':<20} {'Duration':<10} {'Size':<8}",
            "-" * 80,
        ]

        for i, track in enumerate(tracks, 1):
            duration = track.duration_formatted or f"{int(track.duration_sec)}s"
            size = f"{track.file_size_mb:.1f}MB"
            title = track.title[:33] + ".." if len(track.title) > 35 else track.title
            artist = track.artist[:18] + ".." if len(track.artist) > 20 else track.artist

            lines.append(f"{i:<4} {title:<35} {artist:<20} {duration:<10} {size:<8}")

        lines.extend([
            "-" * 80,
            "",
            "Options:",
            "  [number]     - Select track by number",
            "  [path]       - Provide custom file path or URL",
            "  [0]          - Cancel / No music",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    async def prompt_user_selection(
        self,
        tracks: List[MusicTrack],
        timeout_sec: float = 300.0
    ) -> Optional[MusicTrack]:
        """
        Prompt user to select a music track.

        BLOCKING: Pipeline halts until user makes selection.

        Args:
            tracks: List of available tracks
            timeout_sec: Maximum wait time (default 5 minutes)

        Returns:
            Selected MusicTrack or None if cancelled/timeout
        """
        # Display available tracks
        display = self.display_track_list(tracks)
        print(display)

        try:
            # Wait for user input
            print("\nEnter your selection: ", end="", flush=True)

            # Use asyncio for input with timeout
            loop = asyncio.get_event_loop()

            async def get_input():
                return await loop.run_in_executor(None, input)

            try:
                user_input = await asyncio.wait_for(get_input(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                print("\n[TIMEOUT] No selection made. Using no music.")
                return None

            user_input = user_input.strip()

            # Handle cancel
            if user_input == "0" or user_input.lower() in ("cancel", "none", "skip"):
                print("\n[INFO] Music selection cancelled.")
                return None

            # Handle number selection
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(tracks):
                    selected = tracks[idx]
                    print(f"\n[SELECTED] {selected.title} by {selected.artist} ({selected.duration_formatted})")
                    return selected
                else:
                    print(f"\n[ERROR] Invalid selection. Please enter 1-{len(tracks)}")
                    return await self.prompt_user_selection(tracks, timeout_sec)

            # Handle custom path
            custom_path = Path(user_input)
            if custom_path.exists() and custom_path.suffix.lower() in self.SUPPORTED_FORMATS:
                metadata = await self._get_track_metadata(str(custom_path))
                custom_track = MusicTrack(
                    id="custom_track",
                    title=custom_path.stem,
                    artist="Custom",
                    path=str(custom_path),
                    duration_sec=metadata.get("duration", 0.0),
                    file_size_mb=custom_path.stat().st_size / (1024 * 1024),
                    source_dir="custom"
                )
                print(f"\n[SELECTED] Custom track: {custom_track.title}")
                return custom_track

            # Invalid input
            print(f"\n[ERROR] Invalid input: '{user_input}'. Please try again.")
            return await self.prompt_user_selection(tracks, timeout_sec)

        except KeyboardInterrupt:
            print("\n[CANCELLED] Music selection interrupted.")
            return None
        except Exception as e:
            logger.error(f"[MusicSelectionInterface] Selection error: {e}")
            return None

    async def select_music(
        self,
        allow_default: bool = False,
        default_track_index: int = 0,
        video_duration: Optional[float] = None
    ) -> Optional[MusicSelectionResult]:
        """
        Main entry point for music selection.

        BLOCKS pipeline until user confirms selection.

        Args:
            allow_default: If True, auto-select first track without prompting
            default_track_index: Index of default track if allow_default=True
            video_duration: Target video duration to filter tracks

        Returns:
            MusicSelectionResult with selected track and ducking config
        """
        # Scan for available tracks
        tracks = await self.scan_available_tracks()

        if not tracks:
            logger.warning("[MusicSelectionInterface] No music tracks found in configured directories")
            print("\n[WARNING] No music tracks found. Proceeding without music.")
            return None

        # Filter by duration if specified
        if video_duration:
            suitable_tracks = [t for t in tracks if t.duration_sec >= video_duration * 0.8]
            if suitable_tracks:
                tracks = suitable_tracks
            else:
                logger.warning(f"[MusicSelectionInterface] No tracks long enough for {video_duration}s video")

        # Auto-select if allowed (for testing/API use)
        if allow_default:
            idx = min(default_track_index, len(tracks) - 1)
            selected = tracks[idx]
            logger.info(f"[MusicSelectionInterface] Auto-selected: {selected.title}")
            return MusicSelectionResult(
                selected_track=selected,
                ducking_config=self.DEFAULT_DUCKING,
                user_confirmed=False,
                selection_method="default"
            )

        # Interactive selection (BLOCKING)
        selected = await self.prompt_user_selection(tracks)

        if not selected:
            return None

        return MusicSelectionResult(
            selected_track=selected,
            ducking_config=self.DEFAULT_DUCKING,
            user_confirmed=True,
            selection_method="user_interactive"
        )

    async def select_music_api(
        self,
        track_path: str,
        music_volume: float = 0.20
    ) -> MusicSelectionResult:
        """
        API-based music selection (non-interactive).

        Used when music path is provided via API call.

        Args:
            track_path: Path to music file
            music_volume: Volume level (0.0-1.0, default 0.20 = 20%)

        Returns:
            MusicSelectionResult with specified track
        """
        path = Path(track_path)

        if not path.exists():
            raise FileNotFoundError(f"Music file not found: {track_path}")

        metadata = await self._get_track_metadata(track_path)

        track = MusicTrack(
            id="api_track",
            title=path.stem,
            artist="Provided via API",
            path=track_path,
            duration_sec=metadata.get("duration", 0.0),
            file_size_mb=path.stat().st_size / (1024 * 1024),
            source_dir="api"
        )

        ducking = DuckingConfig(
            threshold=0.1,
            ratio=4,
            attack=200,
            release=500,
            music_base_volume=max(0.15, min(0.25, music_volume))  # Clamp to 15-25%
        )

        return MusicSelectionResult(
            selected_track=track,
            ducking_config=ducking,
            user_confirmed=True,
            selection_method="api"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_music_selector(
    additional_dirs: Optional[List[str]] = None
) -> MusicSelectionInterface:
    """Create MusicSelectionInterface with optional additional directories."""
    dirs = [Path(d) for d in additional_dirs] if additional_dirs else None
    return MusicSelectionInterface(additional_dirs=dirs)


# =============================================================================
# CLI TEST
# =============================================================================

async def test_music_selection():
    """Test the music selection interface."""
    print("\n[MusicSelectionInterface] Testing...")
    print("=" * 60)

    selector = create_music_selector()

    # Scan tracks
    tracks = await selector.scan_available_tracks()
    print(f"\nFound {len(tracks)} tracks")

    if tracks:
        for i, track in enumerate(tracks[:5], 1):
            print(f"  {i}. {track.title} ({track.duration_formatted}) - {track.file_size_mb:.1f}MB")

        if len(tracks) > 5:
            print(f"  ... and {len(tracks) - 5} more")

    # Test interactive selection (comment out for automated testing)
    # result = await selector.select_music()
    # if result:
    #     print(f"\nSelected: {result.selected_track.title}")
    #     print(f"Ducking volume: {result.ducking_config.music_base_volume}")


if __name__ == "__main__":
    asyncio.run(test_music_selection())
