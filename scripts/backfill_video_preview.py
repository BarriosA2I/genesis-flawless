"""
GENESIS Video Preview Backfill Script
=============================================================================
Migrates all historical videos from R2 storage to video-preview gallery.

Usage:
    python scripts/backfill_video_preview.py              # List videos (dry run)
    python scripts/backfill_video_preview.py --execute    # Actually upload

Environment Variables Required:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

Author: Barrios A2I | January 2026
=============================================================================
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

VIDEO_PREVIEW_API = "https://video-preview-theta.vercel.app/api/videos"
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "https://pub-7cc63ed6b93a4f75933fa8ac7b8a358f.r2.dev")


def get_r2_client():
    """Create boto3 client for R2."""
    import boto3
    from botocore.config import Config

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key_id, secret_access_key]):
        print("ERROR: Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        return None

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version="s3v4")
    )

    return client


def list_all_videos(client) -> List[Dict[str, Any]]:
    """List all videos from R2 bucket."""
    bucket_name = os.getenv("R2_BUCKET_NAME", "barrios-videos")

    videos = []
    sessions = {}  # Group by session_id

    paginator = client.get_paginator("list_objects_v2")

    print(f"\n[R2] Listing objects in bucket: {bucket_name}")
    print(f"[R2] Prefix: productions/")

    for page in paginator.paginate(Bucket=bucket_name, Prefix="productions/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Key format: productions/{session_id}/{format}.mp4
            parts = key.split("/")
            if len(parts) >= 3:
                session_id = parts[1]
                filename = parts[2]

                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "videos": [],
                        "thumbnail": None,
                        "created_at": obj["LastModified"].isoformat()
                    }

                if filename.endswith(".mp4"):
                    sessions[session_id]["videos"].append({
                        "key": key,
                        "format": filename.replace(".mp4", ""),
                        "size": obj["Size"],
                        "url": f"{R2_PUBLIC_URL}/{key}"
                    })
                elif filename.endswith(".jpg") or filename.endswith(".png"):
                    sessions[session_id]["thumbnail"] = f"{R2_PUBLIC_URL}/{key}"

    # Convert to list of videos (one per session)
    for session_id, data in sessions.items():
        if data["videos"]:
            # Prefer youtube_1080p, then any 1080p, then first available
            video = None
            for v in data["videos"]:
                if "youtube_1080p" in v["format"]:
                    video = v
                    break
                elif "1080p" in v["format"]:
                    video = v
            if not video:
                video = data["videos"][0]

            videos.append({
                "session_id": session_id,
                "video_url": video["url"],
                "format": video["format"],
                "size_mb": round(video["size"] / (1024 * 1024), 2),
                "thumbnail": data["thumbnail"],
                "created_at": data["created_at"],
                "all_formats": [v["format"] for v in data["videos"]]
            })

    # Sort by created_at (newest first)
    videos.sort(key=lambda x: x["created_at"], reverse=True)

    return videos


async def send_to_preview(video: Dict, index: int, total: int) -> Dict:
    """Send a single video to video-preview gallery."""
    video_id = f"genesis_{video['session_id']}"

    # Try to extract business name from session_id
    # Session IDs often contain timestamp or random string
    business_name = "AI Generated Commercial"

    payload = {
        "id": video_id,
        "url": video["video_url"],
        "title": f"{business_name} - {video['format']}",
        "description": f"64-second AI commercial by Barrios A2I. Format: {video['format']}. Size: {video['size_mb']}MB",
        "thumbnail": video.get("thumbnail"),
        "duration": "1:04",
        "tags": ["commercial", "ai-generated", "barrios-a2i", "64-seconds", "backfill"],
        "business_name": business_name,
        "industry": "",
        "created": video["created_at"].split("T")[0] if video.get("created_at") else datetime.now().strftime("%Y-%m-%d")
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(VIDEO_PREVIEW_API, json=payload)

            if response.status_code in [200, 201]:
                result = response.json()
                action = result.get("action", "created")
                preview_id = result.get("video", {}).get("id", video_id)
                preview_url = f"https://video-preview-theta.vercel.app?v={preview_id}"
                print(f"  [{index+1}/{total}] {action.upper()}: {preview_url}")
                return {"success": True, "action": action, "preview_url": preview_url}
            else:
                print(f"  [{index+1}/{total}] FAILED: HTTP {response.status_code}")
                return {"success": False, "error": response.text}

    except Exception as e:
        print(f"  [{index+1}/{total}] ERROR: {e}")
        return {"success": False, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Backfill videos to video-preview gallery")
    parser.add_argument("--execute", action="store_true", help="Actually upload videos (default is dry run)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to process")
    args = parser.parse_args()

    print("=" * 70)
    print("GENESIS VIDEO PREVIEW BACKFILL")
    print("=" * 70)
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN (use --execute to upload)'}")
    print(f"Target: {VIDEO_PREVIEW_API}")

    # Get R2 client
    client = get_r2_client()
    if not client:
        print("\nAborting: R2 credentials not configured")
        return

    # List all videos
    print("\n[1/3] Scanning R2 bucket for videos...")
    videos = list_all_videos(client)

    if not videos:
        print("\nNo videos found in R2 bucket.")
        return

    print(f"\nFound {len(videos)} production sessions with videos:")
    print("-" * 70)

    for i, v in enumerate(videos[:20]):
        formats = ", ".join(v["all_formats"][:3])
        if len(v["all_formats"]) > 3:
            formats += f" (+{len(v['all_formats']) - 3} more)"
        print(f"  {i+1}. {v['session_id'][:30]}...")
        print(f"     URL: {v['video_url'][:60]}...")
        print(f"     Formats: {formats}")
        print(f"     Size: {v['size_mb']}MB | Created: {v['created_at'][:10]}")
        print()

    if len(videos) > 20:
        print(f"  ... and {len(videos) - 20} more\n")

    # Apply limit if specified
    if args.limit > 0:
        videos = videos[:args.limit]
        print(f"\n[Limited to {args.limit} videos]")

    if not args.execute:
        print("\n" + "=" * 70)
        print("DRY RUN COMPLETE")
        print("=" * 70)
        print(f"\nTo actually upload these {len(videos)} videos, run:")
        print(f"  python scripts/backfill_video_preview.py --execute")
        print(f"\nOr with limit:")
        print(f"  python scripts/backfill_video_preview.py --execute --limit 10")
        return

    # Execute upload
    print("\n[2/3] Uploading videos to gallery...")
    print("-" * 70)

    results = {"success": 0, "failed": 0, "updated": 0}

    for i, video in enumerate(videos):
        result = await send_to_preview(video, i, len(videos))
        if result["success"]:
            if result.get("action") == "updated":
                results["updated"] += 1
            else:
                results["success"] += 1
        else:
            results["failed"] += 1

        # Rate limit: 1 per second
        await asyncio.sleep(1)

    print("\n" + "=" * 70)
    print("[3/3] BACKFILL COMPLETE")
    print("=" * 70)
    print(f"\n  Created: {results['success']}")
    print(f"  Updated: {results['updated']}")
    print(f"  Failed:  {results['failed']}")
    print(f"\nView gallery: https://video-preview-theta.vercel.app")


if __name__ == "__main__":
    asyncio.run(main())
