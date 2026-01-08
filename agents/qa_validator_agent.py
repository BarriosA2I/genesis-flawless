"""
GENESIS QA Validator Agent (Agent 8)
===============================================================================
Standalone video quality assurance validator using FFprobe.

Performs basic validation checks before video delivery:
- Duration matches expected length
- Resolution matches format specs
- Audio tracks present
- File size reasonable
- Codec compatibility

Author: Barrios A2I
Version: 1.0.0 (GENESIS Standalone)
===============================================================================
"""

import asyncio
import json
import logging
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger("genesis.qa_validator")


# =============================================================================
# ENUMS & DATA MODELS
# =============================================================================

class QAStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class IssueSeverity(str, Enum):
    CRITICAL = "critical"      # Blocks delivery
    WARNING = "warning"        # Allow delivery but flag
    INFO = "info"             # Informational only


class QAIssue(BaseModel):
    """Quality assurance issue."""
    check: str
    severity: IssueSeverity
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class QARequest(BaseModel):
    """QA validation request."""
    video_path: str                      # Local path or URL
    format_name: str                     # e.g., "youtube_1080p"
    expected_duration: float             # Expected duration in seconds
    tolerance_percent: float = 10.0      # Duration tolerance


class QAResponse(BaseModel):
    """QA validation response."""
    status: QAStatus
    passed: bool
    score: float                         # 0-100
    checks_passed: int
    total_checks: int
    issues: List[QAIssue] = []
    metadata: Dict[str, Any] = {}
    processing_time_ms: float = 0.0


# =============================================================================
# FORMAT SPECIFICATIONS
# =============================================================================

FORMAT_SPECS = {
    "youtube_1080p": {
        "width": 1920,
        "height": 1080,
        "aspect_ratio": "16:9",
        "fps_min": 24,
        "fps_max": 60,
        "video_codecs": ["h264", "hevc", "vp9"],
        "audio_codecs": ["aac", "mp3", "opus"],
        "max_file_size_mb": 5000,
        "min_bitrate_kbps": 2000,
    },
    "youtube_4k": {
        "width": 3840,
        "height": 2160,
        "aspect_ratio": "16:9",
        "fps_min": 24,
        "fps_max": 60,
        "video_codecs": ["h264", "hevc", "vp9", "av1"],
        "audio_codecs": ["aac", "mp3", "opus"],
        "max_file_size_mb": 20000,
        "min_bitrate_kbps": 10000,
    },
    "tiktok": {
        "width": 1080,
        "height": 1920,
        "aspect_ratio": "9:16",
        "fps_min": 24,
        "fps_max": 60,
        "video_codecs": ["h264"],
        "audio_codecs": ["aac"],
        "max_file_size_mb": 500,
        "min_bitrate_kbps": 1500,
        "max_duration": 600,
    },
    "instagram_feed": {
        "width": 1080,
        "height": 1080,
        "aspect_ratio": "1:1",
        "fps_min": 24,
        "fps_max": 30,
        "video_codecs": ["h264"],
        "audio_codecs": ["aac"],
        "max_file_size_mb": 250,
        "min_bitrate_kbps": 1500,
        "max_duration": 60,
    },
    "instagram_reels": {
        "width": 1080,
        "height": 1920,
        "aspect_ratio": "9:16",
        "fps_min": 24,
        "fps_max": 30,
        "video_codecs": ["h264"],
        "audio_codecs": ["aac"],
        "max_file_size_mb": 250,
        "min_bitrate_kbps": 1500,
        "max_duration": 90,
    },
    "linkedin": {
        "width": 1920,
        "height": 1080,
        "aspect_ratio": "16:9",
        "fps_min": 24,
        "fps_max": 30,
        "video_codecs": ["h264"],
        "audio_codecs": ["aac"],
        "max_file_size_mb": 5000,
        "min_bitrate_kbps": 2000,
        "max_duration": 600,
    },
}


# =============================================================================
# FFPROBE WRAPPER
# =============================================================================

async def run_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Run FFprobe to extract video metadata.

    Args:
        video_path: Path to video file

    Returns:
        FFprobe JSON output or None on error
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    try:
        # Run async
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if process.returncode == 0:
            return json.loads(stdout.decode())
        else:
            logger.error(f"FFprobe failed: {stderr.decode()[:200]}")
            return None

    except asyncio.TimeoutError:
        logger.error("FFprobe timed out")
        return None
    except FileNotFoundError:
        logger.error("FFprobe not found in PATH")
        return None
    except Exception as e:
        logger.error(f"FFprobe error: {e}")
        return None


def extract_video_info(ffprobe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant video info from FFprobe output."""
    info = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "video_codec": None,
        "audio_codec": None,
        "audio_channels": 0,
        "bitrate_kbps": 0,
        "file_size_mb": 0.0,
        "has_video": False,
        "has_audio": False,
    }

    # Get format info
    format_info = ffprobe_data.get("format", {})
    info["duration"] = float(format_info.get("duration", 0))
    info["bitrate_kbps"] = int(format_info.get("bit_rate", 0)) // 1000
    info["file_size_mb"] = int(format_info.get("size", 0)) / (1024 * 1024)

    # Get stream info
    for stream in ffprobe_data.get("streams", []):
        codec_type = stream.get("codec_type")

        if codec_type == "video" and not info["has_video"]:
            info["has_video"] = True
            info["width"] = stream.get("width", 0)
            info["height"] = stream.get("height", 0)
            info["video_codec"] = stream.get("codec_name", "").lower()

            # Calculate FPS from frame rate string (e.g., "30/1" or "30000/1001")
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = float(num) / float(den) if float(den) > 0 else 0
            except:
                info["fps"] = 0

        elif codec_type == "audio" and not info["has_audio"]:
            info["has_audio"] = True
            info["audio_codec"] = stream.get("codec_name", "").lower()
            info["audio_channels"] = stream.get("channels", 0)

    return info


# =============================================================================
# QA VALIDATOR AGENT
# =============================================================================

class QAValidatorAgent:
    """
    Standalone QA validator using FFprobe.

    Performs validation checks:
    1. File exists and is readable
    2. Duration within tolerance
    3. Resolution matches format spec
    4. Video codec supported
    5. Audio track present
    6. Audio codec supported
    7. File size reasonable
    8. Bitrate adequate
    """

    def __init__(self, format_specs: Optional[Dict] = None):
        self.format_specs = format_specs or FORMAT_SPECS
        self.stats = {
            "validations": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
        }
        logger.info("[QAValidator] Initialized")

    async def validate(self, request: QARequest) -> QAResponse:
        """
        Validate video quality.

        Args:
            request: QA validation request

        Returns:
            QAResponse with status, score, and issues
        """
        start_time = time.time()
        self.stats["validations"] += 1

        issues: List[QAIssue] = []
        checks_passed = 0
        total_checks = 0

        video_path = Path(request.video_path)
        format_spec = self.format_specs.get(request.format_name, {})

        logger.info(f"[QAValidator] Validating: {request.format_name} - {video_path}")

        # =========================================
        # Check 1: File exists
        # =========================================
        total_checks += 1
        if not video_path.exists():
            issues.append(QAIssue(
                check="file_exists",
                severity=IssueSeverity.CRITICAL,
                message=f"Video file not found: {video_path}"
            ))
            # Can't continue without file
            return self._build_response(
                issues, checks_passed, total_checks, {},
                time.time() - start_time
            )
        checks_passed += 1

        # =========================================
        # Check 2: Run FFprobe
        # =========================================
        total_checks += 1
        ffprobe_data = await run_ffprobe(str(video_path))
        if not ffprobe_data:
            issues.append(QAIssue(
                check="ffprobe_readable",
                severity=IssueSeverity.CRITICAL,
                message="FFprobe failed to read video file"
            ))
            return self._build_response(
                issues, checks_passed, total_checks, {},
                time.time() - start_time
            )
        checks_passed += 1

        # Extract video info
        info = extract_video_info(ffprobe_data)

        # =========================================
        # Check 3: Has video stream
        # =========================================
        total_checks += 1
        if not info["has_video"]:
            issues.append(QAIssue(
                check="has_video_stream",
                severity=IssueSeverity.CRITICAL,
                message="No video stream found"
            ))
        else:
            checks_passed += 1

        # =========================================
        # Check 4: Duration within tolerance
        # =========================================
        total_checks += 1
        expected = request.expected_duration
        actual = info["duration"]
        tolerance = expected * (request.tolerance_percent / 100)

        if abs(actual - expected) > tolerance:
            severity = IssueSeverity.WARNING if abs(actual - expected) < tolerance * 2 else IssueSeverity.CRITICAL
            issues.append(QAIssue(
                check="duration",
                severity=severity,
                message=f"Duration mismatch: expected {expected:.1f}s, got {actual:.1f}s",
                expected=f"{expected:.1f}s",
                actual=f"{actual:.1f}s"
            ))
            if severity == IssueSeverity.WARNING:
                checks_passed += 1
        else:
            checks_passed += 1

        # =========================================
        # Check 5: Resolution matches spec
        # =========================================
        if format_spec:
            total_checks += 1
            expected_width = format_spec.get("width", 0)
            expected_height = format_spec.get("height", 0)

            if info["width"] != expected_width or info["height"] != expected_height:
                # Allow close matches (within 10%)
                width_ok = abs(info["width"] - expected_width) <= expected_width * 0.1
                height_ok = abs(info["height"] - expected_height) <= expected_height * 0.1

                if width_ok and height_ok:
                    issues.append(QAIssue(
                        check="resolution",
                        severity=IssueSeverity.WARNING,
                        message=f"Resolution slightly off: {info['width']}x{info['height']}",
                        expected=f"{expected_width}x{expected_height}",
                        actual=f"{info['width']}x{info['height']}"
                    ))
                    checks_passed += 1
                else:
                    issues.append(QAIssue(
                        check="resolution",
                        severity=IssueSeverity.CRITICAL,
                        message=f"Resolution mismatch: {info['width']}x{info['height']}",
                        expected=f"{expected_width}x{expected_height}",
                        actual=f"{info['width']}x{info['height']}"
                    ))
            else:
                checks_passed += 1

        # =========================================
        # Check 6: Video codec supported
        # =========================================
        if format_spec and "video_codecs" in format_spec:
            total_checks += 1
            supported_codecs = format_spec["video_codecs"]
            if info["video_codec"] not in supported_codecs:
                issues.append(QAIssue(
                    check="video_codec",
                    severity=IssueSeverity.WARNING,
                    message=f"Video codec '{info['video_codec']}' not in recommended list",
                    expected=str(supported_codecs),
                    actual=info["video_codec"]
                ))
            else:
                checks_passed += 1

        # =========================================
        # Check 7: Has audio stream
        # =========================================
        total_checks += 1
        if not info["has_audio"]:
            issues.append(QAIssue(
                check="has_audio_stream",
                severity=IssueSeverity.WARNING,
                message="No audio stream found (video may be silent)"
            ))
        else:
            checks_passed += 1

        # =========================================
        # Check 8: Audio codec supported
        # =========================================
        if format_spec and "audio_codecs" in format_spec and info["has_audio"]:
            total_checks += 1
            supported_codecs = format_spec["audio_codecs"]
            if info["audio_codec"] not in supported_codecs:
                issues.append(QAIssue(
                    check="audio_codec",
                    severity=IssueSeverity.INFO,
                    message=f"Audio codec '{info['audio_codec']}' not in recommended list",
                    expected=str(supported_codecs),
                    actual=info["audio_codec"]
                ))
            else:
                checks_passed += 1

        # =========================================
        # Check 9: File size reasonable
        # =========================================
        if format_spec and "max_file_size_mb" in format_spec:
            total_checks += 1
            max_size = format_spec["max_file_size_mb"]
            if info["file_size_mb"] > max_size:
                issues.append(QAIssue(
                    check="file_size",
                    severity=IssueSeverity.WARNING,
                    message=f"File size ({info['file_size_mb']:.1f}MB) exceeds max ({max_size}MB)",
                    expected=f"{max_size}MB",
                    actual=f"{info['file_size_mb']:.1f}MB"
                ))
            else:
                checks_passed += 1

        # =========================================
        # Check 10: Bitrate adequate
        # =========================================
        if format_spec and "min_bitrate_kbps" in format_spec:
            total_checks += 1
            min_bitrate = format_spec["min_bitrate_kbps"]
            if info["bitrate_kbps"] < min_bitrate:
                issues.append(QAIssue(
                    check="bitrate",
                    severity=IssueSeverity.WARNING,
                    message=f"Bitrate ({info['bitrate_kbps']}kbps) below recommended ({min_bitrate}kbps)",
                    expected=f"{min_bitrate}kbps",
                    actual=f"{info['bitrate_kbps']}kbps"
                ))
            else:
                checks_passed += 1

        # Build response
        processing_time = (time.time() - start_time) * 1000
        return self._build_response(issues, checks_passed, total_checks, info, processing_time)

    def _build_response(
        self,
        issues: List[QAIssue],
        checks_passed: int,
        total_checks: int,
        metadata: Dict[str, Any],
        processing_time_ms: float
    ) -> QAResponse:
        """Build QA response from validation results."""
        # Count issues by severity
        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

        # Determine status
        if critical_count > 0:
            status = QAStatus.FAILED
            passed = False
            self.stats["failed"] += 1
        elif warning_count > 0:
            status = QAStatus.PASSED_WITH_WARNINGS
            passed = True
            self.stats["warnings"] += 1
        else:
            status = QAStatus.PASSED
            passed = True
            self.stats["passed"] += 1

        # Calculate score (0-100)
        if total_checks > 0:
            base_score = (checks_passed / total_checks) * 100
            # Penalize for critical issues
            score = max(0, base_score - (critical_count * 20) - (warning_count * 5))
        else:
            score = 0

        return QAResponse(
            status=status,
            passed=passed,
            score=round(score, 1),
            checks_passed=checks_passed,
            total_checks=total_checks,
            issues=issues,
            metadata=metadata,
            processing_time_ms=round(processing_time_ms, 2)
        )

    async def validate_multiple(
        self,
        video_paths: Dict[str, str],
        expected_duration: float
    ) -> Dict[str, QAResponse]:
        """
        Validate multiple video formats.

        Args:
            video_paths: Dict of format_name -> video_path
            expected_duration: Expected duration for all videos

        Returns:
            Dict of format_name -> QAResponse
        """
        results = {}

        for format_name, video_path in video_paths.items():
            request = QARequest(
                video_path=video_path,
                format_name=format_name,
                expected_duration=expected_duration
            )
            results[format_name] = await self.validate(request)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "stats": self.stats,
            "supported_formats": list(self.format_specs.keys())
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_qa_validator() -> QAValidatorAgent:
    """Create QAValidatorAgent instance."""
    return QAValidatorAgent()


# =============================================================================
# TESTING
# =============================================================================

async def test_qa_validator():
    """Test the QA validator agent."""
    agent = create_qa_validator()

    # Mock test (would need actual video file)
    print("\n[QAValidator] Test Mode")
    print("=" * 60)
    print(f"Supported formats: {list(agent.format_specs.keys())}")
    print(f"Stats: {agent.get_stats()}")

    # Test with a non-existent file to see error handling
    request = QARequest(
        video_path="/nonexistent/video.mp4",
        format_name="youtube_1080p",
        expected_duration=30.0
    )

    response = await agent.validate(request)
    print(f"\nTest validation result:")
    print(f"  Status: {response.status}")
    print(f"  Passed: {response.passed}")
    print(f"  Score: {response.score}")
    print(f"  Checks: {response.checks_passed}/{response.total_checks}")
    print(f"  Issues: {len(response.issues)}")


if __name__ == "__main__":
    asyncio.run(test_qa_validator())
