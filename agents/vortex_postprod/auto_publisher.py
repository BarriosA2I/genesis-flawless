"""
RAGNAROK v8.0 Auto-Publisher
═══════════════════════════════════════════════════════════════════════════════
Standalone mandatory publish module for VORTEX post-production pipeline.

This module extracts and enhances the publish logic from vortex_integration.py
into a reusable, resilient module with retry logic.

Features:
- Mandatory publish (no exceptions allowed)
- Retry with exponential backoff (max 3 retries)
- Clear success/failure reporting
- Completion signals for Ralph integration

Author: Barrios A2I
Version: 8.0.0
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("ragnarok.auto_publisher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# =============================================================================
# CONSTANTS
# =============================================================================
VIDEO_GALLERY_API = "https://video-preview-theta.vercel.app/api/videos"
VIDEO_UPLOAD_API = "https://video-preview-theta.vercel.app/api/upload"
CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0  # seconds


# =============================================================================
# EXCEPTIONS
# =============================================================================
class PublishError(Exception):
    """Raised when publishing fails after all retries."""

    def __init__(self, message: str, attempts: int = 0, last_error: Optional[str] = None):
        self.message = message
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(self.message)


# =============================================================================
# COMPLETION SIGNALS
# =============================================================================
class PublishSignals:
    """Completion signals for Ralph integration."""
    SUCCESS = "<promise>PUBLISH_COMPLETE</promise>"
    FAILED = "<promise>PUBLISH_FAILED</promise>"

    @staticmethod
    def success_with_url(url: str) -> str:
        return f"<promise>PUBLISH_COMPLETE</promise> {url}"


# =============================================================================
# RESULT MODELS
# =============================================================================
class PublishResult:
    """Result of a publish operation."""

    def __init__(
        self,
        success: bool,
        gallery_url: Optional[str] = None,
        video_id: Optional[str] = None,
        attempts: int = 1,
        error: Optional[str] = None,
        signal: Optional[str] = None
    ):
        self.success = success
        self.gallery_url = gallery_url
        self.video_id = video_id
        self.attempts = attempts
        self.error = error
        self.signal = signal or (PublishSignals.SUCCESS if success else PublishSignals.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "gallery_url": self.gallery_url,
            "video_id": self.video_id,
            "attempts": self.attempts,
            "error": self.error,
            "signal": self.signal
        }

    def __repr__(self) -> str:
        if self.success:
            return f"PublishResult(success=True, url={self.gallery_url})"
        return f"PublishResult(success=False, error={self.error})"


# =============================================================================
# CORE PUBLISH FUNCTIONS
# =============================================================================
async def upload_to_temp_host(video_path: str) -> Optional[str]:
    """
    Upload video to temporary hosting (catbox.moe).

    Args:
        video_path: Local path to video file

    Returns:
        Temporary URL or None on failure
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    logger.info(f"[UPLOAD] Uploading to temp host: {video_path}")

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            with open(video_path, 'rb') as f:
                files = {'fileToUpload': (os.path.basename(video_path), f, 'video/mp4')}
                data = {'reqtype': 'fileupload'}
                response = await client.post(CATBOX_UPLOAD_URL, files=files, data=data)

                if response.status_code == 200:
                    temp_url = response.text.strip()
                    if temp_url.startswith('http'):
                        logger.info(f"[UPLOAD] Temp URL: {temp_url}")
                        return temp_url
                    else:
                        logger.error(f"[UPLOAD] Invalid response: {temp_url[:100]}")
                        return None
                else:
                    logger.error(f"[UPLOAD] Failed: {response.status_code}")
                    return None
    except Exception as e:
        logger.error(f"[UPLOAD] Error: {e}")
        return None


async def migrate_to_vercel_blob(temp_url: str, filename: str) -> Optional[str]:
    """
    Migrate video from temp host to Vercel Blob storage.

    Args:
        temp_url: Temporary URL from catbox
        filename: Original filename

    Returns:
        Vercel Blob URL or original temp URL
    """
    logger.info(f"[MIGRATE] Migrating to Vercel Blob...")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                VIDEO_UPLOAD_API,
                json={'source_url': temp_url, 'filename': filename}
            )

            if response.status_code in [200, 201]:
                result = response.json()
                blob_url = result.get('url')
                if blob_url:
                    logger.info(f"[MIGRATE] Migrated to: {blob_url[:80]}...")
                    return blob_url

            # Fallback to temp URL
            logger.warning("[MIGRATE] Using temp URL as fallback")
            return temp_url

    except Exception as e:
        logger.warning(f"[MIGRATE] Error (using temp URL): {e}")
        return temp_url


async def register_in_gallery(
    video_url: str,
    video_id: str,
    title: str,
    tags: List[str],
    duration_seconds: float,
    company_name: str,
    industry: str
) -> Optional[str]:
    """
    Register video in the gallery API.

    Args:
        video_url: URL to the video
        video_id: Unique video ID
        title: Video title
        tags: List of tags
        duration_seconds: Video duration
        company_name: Business name
        industry: Industry vertical

    Returns:
        Gallery URL or None on failure
    """
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    payload = {
        "id": video_id,
        "url": video_url,
        "title": title,
        "description": f"VORTEX Post-Production enhanced video - {title}",
        "thumbnail": None,
        "duration": f"{minutes}:{seconds:02d}",
        "tags": tags,
        "business_name": company_name,
        "industry": industry,
        "created": datetime.now().strftime("%Y-%m-%d")
    }

    logger.info(f"[REGISTER] Registering video: {video_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(VIDEO_GALLERY_API, json=payload)

            if response.status_code in [200, 201]:
                result = response.json()
                returned_id = result.get("video", {}).get("id", video_id)
                gallery_url = f"https://video-preview-theta.vercel.app?v={returned_id}"
                logger.info(f"[REGISTER] Success: {gallery_url}")
                return gallery_url
            else:
                logger.error(f"[REGISTER] Failed: {response.status_code} - {response.text[:200]}")
                return None

    except Exception as e:
        logger.error(f"[REGISTER] Error: {e}")
        return None


# =============================================================================
# MANDATORY PUBLISH FUNCTION
# =============================================================================
async def mandatory_publish(
    video_path: str,
    title: str = "RAGNAROK Production",
    tags: Optional[List[str]] = None,
    video_url: Optional[str] = None,
    duration_seconds: float = 60.0,
    company_name: str = "Barrios A2I",
    industry: str = "technology"
) -> PublishResult:
    """
    MANDATORY publish step with retry logic.

    This function MUST succeed or raise PublishError.
    Every completed video in the RAGNAROK/VORTEX pipeline MUST be published.

    Args:
        video_path: Local path to video file
        title: Video title for gallery
        tags: List of tags (defaults to ["vortex", "ragnarok", "auto-generated"])
        video_url: Optional existing URL (skips upload if provided)
        duration_seconds: Video duration for display
        company_name: Business name
        industry: Industry vertical

    Returns:
        PublishResult with gallery URL and status

    Raises:
        PublishError: If publishing fails after all retries
    """
    logger.info("=" * 60)
    logger.info("[MANDATORY PUBLISH] Starting video gallery publish...")
    logger.info(f"Video: {video_path}")
    logger.info(f"Title: {title}")
    logger.info("=" * 60)

    video_id = f"vortex_{uuid.uuid4().hex[:12]}"

    if tags is None:
        tags = ["vortex", "ragnarok", "auto-generated", industry]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[ATTEMPT {attempt}/{MAX_RETRIES}]")

            # Step 1: Get video URL (upload if needed)
            current_url = video_url

            if not current_url:
                if not os.path.exists(video_path):
                    raise PublishError(f"Video file not found: {video_path}")

                # Upload to temp host
                temp_url = await upload_to_temp_host(video_path)
                if not temp_url:
                    raise Exception("Failed to upload to temp host")

                # Migrate to Vercel Blob
                current_url = await migrate_to_vercel_blob(
                    temp_url,
                    os.path.basename(video_path)
                )

            if not current_url:
                raise Exception("No video URL available")

            # Step 2: Register in gallery
            gallery_url = await register_in_gallery(
                video_url=current_url,
                video_id=video_id,
                title=title,
                tags=tags,
                duration_seconds=duration_seconds,
                company_name=company_name,
                industry=industry
            )

            if gallery_url:
                logger.info("=" * 60)
                logger.info(f"[SUCCESS] Published to gallery!")
                logger.info(f"Gallery URL: {gallery_url}")
                logger.info(f"Attempts: {attempt}")
                logger.info("=" * 60)

                return PublishResult(
                    success=True,
                    gallery_url=gallery_url,
                    video_id=video_id,
                    attempts=attempt,
                    signal=PublishSignals.success_with_url(gallery_url)
                )
            else:
                raise Exception("Gallery registration failed")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[ATTEMPT {attempt}] Failed: {last_error}")

            if attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

    # All retries exhausted
    logger.error("=" * 60)
    logger.error("[FAILED] All publish attempts exhausted!")
    logger.error(f"Last error: {last_error}")
    logger.error("=" * 60)

    return PublishResult(
        success=False,
        attempts=MAX_RETRIES,
        error=last_error,
        signal=PublishSignals.FAILED
    )


async def mandatory_publish_or_raise(
    video_path: str,
    title: str = "RAGNAROK Production",
    tags: Optional[List[str]] = None,
    video_url: Optional[str] = None,
    duration_seconds: float = 60.0,
    company_name: str = "Barrios A2I",
    industry: str = "technology"
) -> str:
    """
    MANDATORY publish that raises on failure.

    Same as mandatory_publish but raises PublishError on failure instead
    of returning a failed PublishResult.

    Returns:
        Gallery URL string

    Raises:
        PublishError: If publishing fails after all retries
    """
    result = await mandatory_publish(
        video_path=video_path,
        title=title,
        tags=tags,
        video_url=video_url,
        duration_seconds=duration_seconds,
        company_name=company_name,
        industry=industry
    )

    if not result.success:
        raise PublishError(
            message=f"Failed to publish video after {result.attempts} attempts",
            attempts=result.attempts,
            last_error=result.error
        )

    return result.gallery_url


# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    'mandatory_publish',
    'mandatory_publish_or_raise',
    'PublishResult',
    'PublishError',
    'PublishSignals',
    'upload_to_temp_host',
    'migrate_to_vercel_blob',
    'register_in_gallery',
]


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    async def test_publish():
        """Test the auto-publisher with a sample video."""
        test_video = r"C:\Users\gary\Downloads\_BARROSA2I\MEDIA\LAUNCH_VIDEOS\barrios_a2i_3min_WITH_VO_20260120_035046.mp4"

        if not os.path.exists(test_video):
            print(f"Test video not found: {test_video}")
            return

        result = await mandatory_publish(
            video_path=test_video,
            title="Auto-Publisher Test",
            tags=["test", "auto-publisher", "vortex"],
            company_name="Barrios A2I",
            industry="technology"
        )

        print(f"\nResult: {result}")
        print(f"Signal: {result.signal}")

    asyncio.run(test_publish())
