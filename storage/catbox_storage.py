"""
GENESIS Catbox.moe Storage
===========================
Upload videos to catbox.moe for public hosting.
Catbox provides free, reliable video hosting with direct links.

Author: Barrios A2I
Version: 1.0.0
"""

import asyncio
import aiohttp
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("genesis.catbox_storage")

CATBOX_API_URL = "https://catbox.moe/user/api.php"
LITTERBOX_API_URL = "https://litterbox.catbox.moe/resources/internals/api.php"


class CatboxStorage:
    """
    Catbox.moe video storage.

    Uploads files to catbox.moe for permanent public hosting.
    Uses litterbox.catbox.moe for temporary files (1h, 12h, 24h, 72h).
    """

    def __init__(self, userhash: Optional[str] = None):
        """
        Initialize Catbox storage.

        Args:
            userhash: Optional catbox user hash for account uploads
        """
        self.userhash = userhash or os.getenv("CATBOX_USERHASH")
        logger.info(f"[CatboxStorage] Initialized (userhash: {'set' if self.userhash else 'anonymous'})")

    @property
    def is_configured(self) -> bool:
        """Always returns True - catbox works without auth."""
        return True

    async def upload_video(
        self,
        local_path: str,
        session_id: str = "",
        format_name: str = "",
        content_type: str = "video/mp4"
    ) -> str:
        """
        Upload video file to catbox.moe.

        Args:
            local_path: Path to local video file
            session_id: Production session ID (for logging)
            format_name: Format name (for logging)
            content_type: MIME type (unused, catbox auto-detects)

        Returns:
            Public URL for the uploaded file (https://files.catbox.moe/xxxxx.mp4)
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"[CatboxStorage] Uploading {path.name} ({file_size_mb:.1f}MB)")

        try:
            async with aiohttp.ClientSession() as session:
                # Prepare multipart form data
                data = aiohttp.FormData()
                data.add_field('reqtype', 'fileupload')

                # Add userhash if available (for account uploads)
                if self.userhash:
                    data.add_field('userhash', self.userhash)

                # Add the file
                data.add_field(
                    'fileToUpload',
                    open(local_path, 'rb'),
                    filename=path.name,
                    content_type=content_type
                )

                # Upload with timeout (5 minutes for large files)
                timeout = aiohttp.ClientTimeout(total=300)
                async with session.post(CATBOX_API_URL, data=data, timeout=timeout) as response:
                    result = await response.text()
                    result = result.strip()

                    if result.startswith("https://files.catbox.moe/"):
                        logger.info(f"[CatboxStorage] Upload success: {result}")
                        return result
                    else:
                        logger.error(f"[CatboxStorage] Upload failed: {result}")
                        raise RuntimeError(f"Catbox upload failed: {result}")

        except asyncio.TimeoutError:
            logger.error(f"[CatboxStorage] Upload timeout for {path.name}")
            raise RuntimeError("Catbox upload timed out")
        except Exception as e:
            logger.error(f"[CatboxStorage] Upload error: {e}")
            raise

    async def upload_to_litterbox(
        self,
        local_path: str,
        expiry: str = "72h"
    ) -> str:
        """
        Upload to litterbox (temporary hosting).

        Args:
            local_path: Path to local file
            expiry: Expiration time (1h, 12h, 24h, 72h)

        Returns:
            Temporary URL (https://litter.catbox.moe/xxxxx.mp4)
        """
        valid_expiry = ["1h", "12h", "24h", "72h"]
        if expiry not in valid_expiry:
            expiry = "72h"

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        logger.info(f"[CatboxStorage] Uploading to litterbox ({expiry}): {path.name}")

        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('reqtype', 'fileupload')
                data.add_field('time', expiry)
                data.add_field(
                    'fileToUpload',
                    open(local_path, 'rb'),
                    filename=path.name
                )

                timeout = aiohttp.ClientTimeout(total=300)
                async with session.post(LITTERBOX_API_URL, data=data, timeout=timeout) as response:
                    result = await response.text()
                    result = result.strip()

                    if result.startswith("https://litter.catbox.moe/"):
                        logger.info(f"[CatboxStorage] Litterbox upload success: {result}")
                        return result
                    else:
                        raise RuntimeError(f"Litterbox upload failed: {result}")

        except Exception as e:
            logger.error(f"[CatboxStorage] Litterbox error: {e}")
            raise

    async def upload_thumbnail(
        self,
        local_path: str,
        session_id: str = ""
    ) -> str:
        """Upload thumbnail image to catbox."""
        return await self.upload_video(local_path, session_id, "thumbnail", "image/jpeg")

    async def check_connection(self) -> Dict[str, Any]:
        """Test catbox connectivity."""
        try:
            async with aiohttp.ClientSession() as session:
                # Just check if the API endpoint is reachable
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get("https://catbox.moe/", timeout=timeout) as response:
                    return {
                        "configured": True,
                        "connected": response.status == 200,
                        "error": None
                    }
        except Exception as e:
            return {
                "configured": True,
                "connected": False,
                "error": str(e)
            }


def get_catbox_storage() -> CatboxStorage:
    """Get CatboxStorage instance."""
    return CatboxStorage()


if __name__ == "__main__":
    async def main():
        print("\n[CatboxStorage] Connection Test")
        print("=" * 60)

        storage = get_catbox_storage()
        status = await storage.check_connection()

        print(f"Configured: {status['configured']}")
        print(f"Connected: {status['connected']}")
        if status['error']:
            print(f"Error: {status['error']}")

    asyncio.run(main())
