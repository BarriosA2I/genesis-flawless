"""
GENESIS R2 Video Storage
═══════════════════════════════════════════════════════════════════════════════
Cloudflare R2 storage integration for video assets.

R2 is S3-compatible, so we use boto3 with custom endpoint.

Environment Variables Required:
- R2_ACCOUNT_ID: Cloudflare account ID
- R2_ACCESS_KEY_ID: R2 API access key
- R2_SECRET_ACCESS_KEY: R2 API secret key
- R2_BUCKET_NAME: Bucket name (e.g., "barrios-videos")
- R2_PUBLIC_URL: Public CDN URL (e.g., "https://pub-xxx.r2.dev")

Author: Barrios A2I
Version: 1.0.0
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger("genesis.r2_storage")

# Thread pool for async boto3 operations
_executor = ThreadPoolExecutor(max_workers=4)


class R2VideoStorage:
    """
    Cloudflare R2 storage for video assets.

    Handles upload, download, and URL generation for rendered videos.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        public_url: Optional[str] = None
    ):
        # Load from environment if not provided
        self.account_id = account_id or os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "barrios-videos")
        self.public_url = public_url or os.getenv("R2_PUBLIC_URL", "https://pub-7cc63ed6b93a4f75933fa8ac7b8a358f.r2.dev")

        # Validate required config
        self._validate_config()

        # R2 endpoint URL
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        # Create boto3 client
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            )
        )

        logger.info(f"[R2Storage] Initialized (bucket: {self.bucket_name})")

    def _validate_config(self):
        """Validate required configuration."""
        missing = []
        if not self.account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")

        if missing:
            logger.warning(f"[R2Storage] Missing config: {missing}. Storage operations will fail.")

    @property
    def is_configured(self) -> bool:
        """Check if R2 is fully configured."""
        return all([
            self.account_id,
            self.access_key_id,
            self.secret_access_key,
            self.bucket_name
        ])

    async def upload_video(
        self,
        local_path: str,
        session_id: str,
        format_name: str,
        content_type: str = "video/mp4"
    ) -> str:
        """
        Upload video file to R2.

        Args:
            local_path: Path to local video file
            session_id: Production session ID
            format_name: Format name (e.g., "youtube_1080p")
            content_type: MIME type

        Returns:
            Public CDN URL for the uploaded file
        """
        if not self.is_configured:
            logger.warning("[R2Storage] Not configured, returning mock URL")
            return f"{self.public_url}/productions/{session_id}/{format_name}.mp4"

        # Generate R2 key
        key = f"productions/{session_id}/{format_name}.mp4"

        logger.info(f"[R2Storage] Uploading {local_path} -> {key}")

        # Upload in thread pool (boto3 is sync)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            self._sync_upload,
            local_path,
            key,
            content_type
        )

        # Return public URL
        public_url = f"{self.public_url}/{key}"
        logger.info(f"[R2Storage] Uploaded: {public_url}")

        return public_url

    def _sync_upload(self, local_path: str, key: str, content_type: str):
        """Synchronous upload for thread pool."""
        file_size = Path(local_path).stat().st_size

        # Use multipart for large files (>100MB)
        if file_size > 100_000_000:
            self._multipart_upload(local_path, key, content_type)
        else:
            self.client.upload_file(
                local_path,
                self.bucket_name,
                key,
                ExtraArgs={
                    "ContentType": content_type,
                    "CacheControl": "public, max-age=31536000"  # 1 year cache
                }
            )

    def _multipart_upload(self, local_path: str, key: str, content_type: str):
        """Multipart upload for large files."""
        # Initiate multipart upload
        response = self.client.create_multipart_upload(
            Bucket=self.bucket_name,
            Key=key,
            ContentType=content_type,
            CacheControl="public, max-age=31536000"
        )
        upload_id = response["UploadId"]

        parts = []
        part_size = 50 * 1024 * 1024  # 50MB parts

        try:
            with open(local_path, "rb") as f:
                part_number = 1
                while True:
                    data = f.read(part_size)
                    if not data:
                        break

                    response = self.client.upload_part(
                        Bucket=self.bucket_name,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=data
                    )

                    parts.append({
                        "PartNumber": part_number,
                        "ETag": response["ETag"]
                    })
                    part_number += 1

            # Complete multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )

        except Exception as e:
            # Abort on error
            self.client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id
            )
            raise

    async def upload_thumbnail(
        self,
        local_path: str,
        session_id: str
    ) -> str:
        """Upload thumbnail image to R2."""
        if not self.is_configured:
            return f"{self.public_url}/productions/{session_id}/thumbnail.jpg"

        key = f"productions/{session_id}/thumbnail.jpg"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            self._sync_upload,
            local_path,
            key,
            "image/jpeg"
        )

        return f"{self.public_url}/{key}"

    async def get_signed_url(
        self,
        key: str,
        expires_in: int = 3600
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            key: Object key in R2
            expires_in: URL expiration in seconds

        Returns:
            Presigned URL
        """
        if not self.is_configured:
            return f"{self.public_url}/{key}"

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            _executor,
            lambda: self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expires_in
            )
        )

        return url

    async def delete_production(self, session_id: str) -> int:
        """
        Delete all files for a production session.

        Returns:
            Number of files deleted
        """
        if not self.is_configured:
            return 0

        prefix = f"productions/{session_id}/"

        loop = asyncio.get_event_loop()
        deleted_count = await loop.run_in_executor(
            _executor,
            self._sync_delete_prefix,
            prefix
        )

        logger.info(f"[R2Storage] Deleted {deleted_count} files for session {session_id}")
        return deleted_count

    def _sync_delete_prefix(self, prefix: str) -> int:
        """Delete all objects with prefix."""
        deleted = 0

        # List objects
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

        if "Contents" not in response:
            return 0

        # Delete objects
        objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
        if objects:
            self.client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": objects}
            )
            deleted = len(objects)

        return deleted

    async def list_production_files(self, session_id: str) -> List[Dict[str, Any]]:
        """List all files for a production session."""
        if not self.is_configured:
            return []

        prefix = f"productions/{session_id}/"

        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(
            _executor,
            self._sync_list_prefix,
            prefix
        )

        return files

    def _sync_list_prefix(self, prefix: str) -> List[Dict[str, Any]]:
        """List objects with prefix."""
        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

        if "Contents" not in response:
            return []

        return [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
                "url": f"{self.public_url}/{obj['Key']}"
            }
            for obj in response["Contents"]
        ]

    async def check_connection(self) -> Dict[str, Any]:
        """Test R2 connection and return status."""
        result = {
            "configured": self.is_configured,
            "connected": False,
            "bucket_exists": False,
            "error": None
        }

        if not self.is_configured:
            result["error"] = "Missing R2 configuration"
            return result

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _executor,
                lambda: self.client.head_bucket(Bucket=self.bucket_name)
            )
            result["connected"] = True
            result["bucket_exists"] = True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            result["error"] = f"R2 error: {error_code}"
            if error_code == "404":
                result["connected"] = True  # Connected but bucket not found

        except Exception as e:
            result["error"] = str(e)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_r2_storage() -> R2VideoStorage:
    """Create R2VideoStorage instance from environment."""
    return R2VideoStorage()


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK STORAGE (for testing without R2)
# ═══════════════════════════════════════════════════════════════════════════════

class MockVideoStorage:
    """Mock storage for testing without R2 credentials."""

    def __init__(self, public_url: str = "https://pub-7cc63ed6b93a4f75933fa8ac7b8a358f.r2.dev"):
        self.public_url = public_url
        self.uploaded_files: Dict[str, str] = {}
        logger.info("[MockVideoStorage] Initialized (no actual uploads)")

    @property
    def is_configured(self) -> bool:
        return True

    async def upload_video(
        self,
        local_path: str,
        session_id: str,
        format_name: str,
        content_type: str = "video/mp4"
    ) -> str:
        key = f"productions/{session_id}/{format_name}.mp4"
        url = f"{self.public_url}/{key}"
        self.uploaded_files[key] = local_path
        logger.info(f"[MockVideoStorage] Mock upload: {local_path} -> {url}")
        return url

    async def upload_thumbnail(self, local_path: str, session_id: str) -> str:
        return f"{self.public_url}/productions/{session_id}/thumbnail.jpg"

    async def get_signed_url(self, key: str, expires_in: int = 3600) -> str:
        return f"{self.public_url}/{key}"

    async def delete_production(self, session_id: str) -> int:
        return 0

    async def list_production_files(self, session_id: str) -> List[Dict[str, Any]]:
        return []

    async def check_connection(self) -> Dict[str, Any]:
        return {"configured": True, "connected": True, "bucket_exists": True, "error": None}


def get_video_storage() -> R2VideoStorage:
    """
    Get video storage instance.

    Returns R2VideoStorage if configured, otherwise MockVideoStorage.
    """
    storage = R2VideoStorage()
    if storage.is_configured:
        return storage

    logger.warning("[Storage] R2 not configured, using MockVideoStorage")
    return MockVideoStorage()


if __name__ == "__main__":
    # Quick test
    async def main():
        print("\n[R2Storage] Connection Test")
        print("=" * 60)

        storage = get_video_storage()
        status = await storage.check_connection()

        print(f"Configured: {status['configured']}")
        print(f"Connected: {status['connected']}")
        print(f"Bucket exists: {status['bucket_exists']}")
        if status['error']:
            print(f"Error: {status['error']}")

    asyncio.run(main())
