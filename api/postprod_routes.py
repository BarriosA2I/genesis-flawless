"""
================================================================================
RAGNAROK POST-PRODUCTION API
================================================================================
Process existing video scenes through the full RAGNAROK pipeline:
- Scene analysis with Claude Vision
- Script generation
- ElevenLabs voiceover
- Music selection
- VORTEX video assembly

Author: Barrios A2I | 2026-01-11
================================================================================
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import anthropic
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/postprod", tags=["Post-Production"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PostProdRequest(BaseModel):
    """Request to process existing scenes through RAGNAROK pipeline"""
    video_urls: List[str] = Field(..., min_items=1, description="URLs of video scenes to process")
    voice_style: str = Field("professional", description="Voice style: professional, friendly, energetic, authoritative")
    music_style: str = Field("corporate", description="Music style: corporate, upbeat, emotional, energetic")
    brand_name: Optional[str] = Field(None, description="Brand name for script personalization")
    industry: Optional[str] = Field(None, description="Industry for context")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class PostProdResponse(BaseModel):
    """Post-production job response"""
    job_id: str
    status: str
    message: str
    stream_url: str


class PostProdStatusResponse(BaseModel):
    """Post-production status response"""
    job_id: str
    status: str
    current_step: str
    progress: int
    script: Optional[str] = None
    voiceover_url: Optional[str] = None
    music_url: Optional[str] = None
    final_video_url: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# VOICE SETTINGS
# =============================================================================

VOICE_MAPPINGS = {
    "professional": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Rachel"
    },
    "friendly": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Bella"
    },
    "energetic": {
        "voice_id": "ErXwobaYiN019PkySvjV",
        "name": "Antoni"
    },
    "authoritative": {
        "voice_id": "VR6AewLTigWG4xSOukaG",
        "name": "Arnold"
    }
}

SCENE_TYPES = ["HOOK", "PROBLEM", "SOLUTION", "CTA", "EXTRA", "EXTRA", "EXTRA"]


# =============================================================================
# JOB STORAGE
# =============================================================================

# In-memory job storage (use Redis in production)
_jobs: Dict[str, Dict[str, Any]] = {}


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()


# =============================================================================
# PIPELINE STEPS
# =============================================================================

async def download_video(url: str, dest_path: Path) -> bool:
    """Download video from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    content = await response.read()
                    dest_path.write_bytes(content)
                    return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
    return False


async def extract_frame(video_path: Path, output_path: Path, time_offset: float = 0.5) -> bool:
    """Extract a frame from video using FFmpeg"""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_offset),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path)
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    return proc.returncode == 0


async def analyze_scene(client: anthropic.Anthropic, frame_path: Path, scene_index: int, scene_type: str) -> str:
    """Analyze a scene frame using Claude Vision"""

    with open(frame_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
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
                {
                    "type": "text",
                    "text": f"""Analyze this frame from scene {scene_index + 1} (type: {scene_type}) of a commercial video.
Describe in 1-2 sentences what is shown and what message it conveys."""
                }
            ]
        }]
    )

    return response.content[0].text


async def generate_script(client: anthropic.Anthropic, scene_descriptions: List[str], tone: str, brand_name: str = None) -> str:
    """Generate voiceover script from scene descriptions"""

    scene_context = "\n".join([
        f"Scene {i+1} ({SCENE_TYPES[i] if i < len(SCENE_TYPES) else 'EXTRA'}): {desc}"
        for i, desc in enumerate(scene_descriptions)
    ])

    brand_context = f" for {brand_name}" if brand_name else ""

    prompt = f"""Based on these commercial video scenes, write a voiceover script{brand_context}.

SCENE DESCRIPTIONS:
{scene_context}

REQUIREMENTS:
- Total duration: ~30-45 seconds of narration
- Tone: {tone}
- Structure: HOOK → PROBLEM → SOLUTION → CTA
- Each scene gets 2-3 sentences max
- Natural conversational flow
- End with a strong call-to-action

Write ONLY the voiceover script text, nothing else."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


async def generate_voiceover(script: str, voice_style: str, output_path: Path) -> bool:
    """Generate voiceover using ElevenLabs"""

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.error("ELEVENLABS_API_KEY not set")
        return False

    voice_info = VOICE_MAPPINGS.get(voice_style, VOICE_MAPPINGS["professional"])

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_info['voice_id']}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": script,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        },
        "output_format": "mp3_44100_192"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    output_path.write_bytes(audio_bytes)
                    logger.info(f"Voiceover generated: {len(audio_bytes)} bytes")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"ElevenLabs error: {response.status} - {error}")
    except Exception as e:
        logger.error(f"Voiceover generation failed: {e}")

    return False


async def select_music(style: str, output_path: Path) -> bool:
    """Download royalty-free music based on style"""

    MUSIC_URLS = {
        "corporate": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_8cb749e6a9.mp3",
        "upbeat": "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3",
        "emotional": "https://cdn.pixabay.com/download/audio/2021/11/25/audio_91b32e02f9.mp3",
        "energetic": "https://cdn.pixabay.com/download/audio/2022/10/25/audio_3c18d78d8e.mp3"
    }

    url = MUSIC_URLS.get(style, MUSIC_URLS["corporate"])

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    output_path.write_bytes(audio_bytes)
                    return True
    except Exception as e:
        logger.error(f"Music download failed: {e}")

    return False


async def assemble_video(
    video_paths: List[Path],
    voiceover_path: Optional[Path],
    music_path: Optional[Path],
    output_path: Path,
    music_volume: float = 0.2
) -> bool:
    """Assemble final video with FFmpeg"""
    import subprocess

    work_dir = output_path.parent

    # Step 1: Concatenate videos with transitions
    if len(video_paths) == 1:
        video_only = video_paths[0]
    else:
        video_only = work_dir / "video_only.mp4"

        # Create concat file
        concat_file = work_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for vp in video_paths:
                f.write(f"file '{str(vp).replace(chr(92), '/')}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-an",
            str(video_only)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        if proc.returncode != 0:
            return False

    # Step 2: Add audio tracks
    if voiceover_path or music_path:
        cmd = ["ffmpeg", "-y", "-i", str(video_only)]

        filter_parts = []
        audio_inputs = 1

        if voiceover_path and voiceover_path.exists():
            cmd.extend(["-i", str(voiceover_path)])
            filter_parts.append(f"[{audio_inputs}:a]volume=1.0[vo]")
            audio_inputs += 1

        if music_path and music_path.exists():
            cmd.extend(["-i", str(music_path)])
            filter_parts.append(f"[{audio_inputs}:a]volume={music_volume}[music]")
            audio_inputs += 1

        mix_inputs = []
        if voiceover_path and voiceover_path.exists():
            mix_inputs.append("[vo]")
        if music_path and music_path.exists():
            mix_inputs.append("[music]")

        if len(mix_inputs) > 1:
            filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first[aout]")
            audio_map = "[aout]"
        elif mix_inputs:
            audio_map = mix_inputs[0]
        else:
            audio_map = None

        if filter_parts:
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

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        return proc.returncode == 0
    else:
        # Just copy video
        import shutil
        shutil.copy(video_only, output_path)
        return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_postprod_pipeline(job_id: str, request: PostProdRequest):
    """Run the full post-production pipeline"""

    try:
        work_dir = Path(tempfile.mkdtemp(prefix=f"postprod_{job_id}_"))

        # Initialize Anthropic client
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            update_job(job_id, status="failed", error="ANTHROPIC_API_KEY not configured")
            return

        client = anthropic.Anthropic(api_key=anthropic_key)

        # Step 1: Download videos
        update_job(job_id, current_step="downloading", progress=5)

        video_paths = []
        for i, url in enumerate(request.video_urls):
            video_path = work_dir / f"scene_{i:02d}.mp4"
            if await download_video(url, video_path):
                video_paths.append(video_path)
            else:
                logger.warning(f"Failed to download: {url}")

        if not video_paths:
            update_job(job_id, status="failed", error="Failed to download any videos")
            return

        # Step 2: Analyze scenes with Claude Vision
        update_job(job_id, current_step="analyzing", progress=20)

        scene_descriptions = []
        for i, video_path in enumerate(video_paths):
            frame_path = work_dir / f"frame_{i:02d}.jpg"

            if await extract_frame(video_path, frame_path, 4.0):  # Mid-point of 8s scene
                try:
                    desc = await analyze_scene(
                        client, frame_path, i,
                        SCENE_TYPES[i] if i < len(SCENE_TYPES) else "EXTRA"
                    )
                    scene_descriptions.append(desc)
                    logger.info(f"Scene {i+1}: {desc[:80]}...")
                except Exception as e:
                    logger.error(f"Scene analysis failed: {e}")
                    scene_descriptions.append(f"Scene {i+1} content")
            else:
                scene_descriptions.append(f"Scene {i+1} content")

        # Step 3: Generate script
        update_job(job_id, current_step="scripting", progress=40)

        script = await generate_script(
            client, scene_descriptions,
            request.voice_style,
            request.brand_name
        )
        update_job(job_id, script=script)
        logger.info(f"Script generated: {len(script)} chars")

        # Step 4: Generate voiceover
        update_job(job_id, current_step="voiceover", progress=55)

        voiceover_path = work_dir / "voiceover.mp3"
        voiceover_success = await generate_voiceover(script, request.voice_style, voiceover_path)

        if voiceover_success:
            # TODO: Upload to R2 and get URL
            update_job(job_id, voiceover_url=str(voiceover_path))
        else:
            voiceover_path = None
            logger.warning("Voiceover generation failed, continuing without")

        # Step 5: Select music
        update_job(job_id, current_step="music", progress=70)

        music_path = work_dir / "music.mp3"
        music_success = await select_music(request.music_style, music_path)

        if music_success:
            update_job(job_id, music_url=str(music_path))
        else:
            music_path = None

        # Step 6: Assemble final video
        update_job(job_id, current_step="assembling", progress=85)

        output_path = work_dir / f"{job_id}_final.mp4"
        assembly_success = await assemble_video(
            video_paths,
            voiceover_path,
            music_path,
            output_path
        )

        if assembly_success and output_path.exists():
            # TODO: Upload to R2 and get public URL
            update_job(
                job_id,
                status="completed",
                current_step="complete",
                progress=100,
                final_video_url=str(output_path)
            )
            logger.info(f"Pipeline complete: {output_path}")
        else:
            update_job(job_id, status="failed", error="Video assembly failed")

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        update_job(job_id, status="failed", error=str(e))


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/process", response_model=PostProdResponse)
async def start_postprod(request: PostProdRequest, background_tasks: BackgroundTasks):
    """
    Start post-production processing for existing video scenes.

    Takes a list of video URLs and runs them through:
    1. Scene analysis (Claude Vision)
    2. Script generation (Claude)
    3. Voiceover generation (ElevenLabs)
    4. Music selection
    5. Video assembly (VORTEX/FFmpeg)

    Returns a job_id to track progress.
    """

    job_id = request.session_id or str(uuid.uuid4())

    # Create job
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "current_step": "queued",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }

    # Start pipeline in background
    background_tasks.add_task(run_postprod_pipeline, job_id, request)

    return PostProdResponse(
        job_id=job_id,
        status="processing",
        message=f"Processing {len(request.video_urls)} scenes through RAGNAROK pipeline",
        stream_url=f"/api/postprod/status/{job_id}"
    )


@router.get("/status/{job_id}", response_model=PostProdStatusResponse)
async def get_postprod_status(job_id: str):
    """Get current status of a post-production job"""

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return PostProdStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        current_step=job.get("current_step", "unknown"),
        progress=job.get("progress", 0),
        script=job.get("script"),
        voiceover_url=job.get("voiceover_url"),
        music_url=job.get("music_url"),
        final_video_url=job.get("final_video_url"),
        error=job.get("error")
    )


@router.get("/jobs")
async def list_postprod_jobs():
    """List all post-production jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.get("status"),
                "progress": job.get("progress", 0),
                "created_at": job.get("created_at")
            }
            for job_id, job in _jobs.items()
        ],
        "total": len(_jobs)
    }
