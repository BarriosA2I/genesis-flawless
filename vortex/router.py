"""
================================================================================
VORTEX v2.1 LEGENDARY - FastAPI Router
================================================================================
API routes for video assembly integrated into FLAWLESS GENESIS.

Endpoints:
- GET  /api/vortex/health         - Health check with FFmpeg version
- POST /api/vortex/assemble       - Start video assembly job
- GET  /api/vortex/job/{job_id}   - Get job status
- GET  /api/vortex/jobs           - List active jobs
- GET  /api/vortex/stream/{job_id} - SSE progress streaming
- GET  /api/vortex/graph-structure - Pipeline graph for UI

Author: Barrios A2I | VORTEX v2.1
================================================================================
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .state_machine import GlobalState, PipelinePhase
from .graph_nodes import VortexPipeline, AsyncFFmpeg, VortexConfig

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AssembleRequest(BaseModel):
    """Request to start video assembly"""
    video_urls: List[str] = Field(..., min_items=1, description="URLs of video clips")
    voiceover_url: Optional[str] = Field(None, description="URL of voiceover audio")
    music_url: Optional[str] = Field(None, description="URL of background music")
    output_formats: List[str] = Field(default=["youtube_1080p"])
    metadata: Optional[Dict[str, Any]] = Field(default={})
    session_id: Optional[str] = Field(None)


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    phase: str
    phase_label: str
    progress_pct: int
    processing_mode: str
    created_at: str
    updated_at: str
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []
    final_outputs: Dict[str, str] = {}


class VortexHealthResponse(BaseModel):
    """VORTEX health check response"""
    status: str
    version: str
    ffmpeg_version: str
    active_jobs: int


# =============================================================================
# STATE MANAGER (In-Process)
# =============================================================================

class VortexStateManager:
    """In-process state manager for VORTEX jobs within GENESIS"""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._local_cache: Dict[str, GlobalState] = {}
        self.JOB_PREFIX = "vortex:job:"
        self.JOB_TTL = 86400  # 24 hours

    async def save_job_state(self, job_id: str, state: GlobalState) -> bool:
        """Save job state"""
        self._local_cache[job_id] = state

        if self.redis:
            try:
                key = f"{self.JOB_PREFIX}{job_id}"
                await self.redis.set(key, state.json(), ex=self.JOB_TTL)
                return True
            except Exception as e:
                logger.error(f"Redis save failed for {job_id}: {e}")

        return True  # Local cache always works

    async def get_job_state(self, job_id: str) -> Optional[GlobalState]:
        """Get job state"""
        if job_id in self._local_cache:
            return self._local_cache[job_id]

        if self.redis:
            try:
                key = f"{self.JOB_PREFIX}{job_id}"
                data = await self.redis.get(key)
                if data:
                    state = GlobalState.parse_raw(data)
                    self._local_cache[job_id] = state
                    return state
            except Exception as e:
                logger.error(f"Redis get failed for {job_id}: {e}")

        return None

    async def list_active_jobs(self) -> List[str]:
        """List all active job IDs"""
        return list(self._local_cache.keys())


# =============================================================================
# PHASE HELPERS
# =============================================================================

PHASE_LABELS = {
    "init": "Initializing",
    "routing": "Analyzing complexity",
    "asset_download": "Downloading assets",
    "scene_analysis": "Analyzing scenes",
    "transition_selection": "Selecting transitions",
    "clip_assembly": "Assembling clips",
    "audio_sync": "Syncing audio",
    "format_render": "Rendering formats",
    "quality_check": "Quality check",
    "completed": "Completed",
    "failed": "Failed",
}

PHASE_PROGRESS = {
    "init": 0,
    "routing": 5,
    "asset_download": 15,
    "scene_analysis": 30,
    "transition_selection": 40,
    "clip_assembly": 60,
    "audio_sync": 75,
    "format_render": 90,
    "quality_check": 95,
    "completed": 100,
    "failed": 100,
}


def get_phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase)


def get_progress_pct(phase: str) -> int:
    return PHASE_PROGRESS.get(phase, 0)


# =============================================================================
# ROUTER INSTANCE
# =============================================================================

router = APIRouter(prefix="/api/vortex", tags=["VORTEX Video Assembly"])

# Global state (initialized when router is included)
state_manager: Optional[VortexStateManager] = None
pipeline: Optional[VortexPipeline] = None
ffmpeg: Optional[AsyncFFmpeg] = None
active_job_tasks: Dict[str, asyncio.Task] = {}


def initialize_vortex(redis_client=None):
    """Initialize VORTEX components (called from GENESIS lifespan)"""
    global state_manager, pipeline, ffmpeg

    state_manager = VortexStateManager(redis_client)
    pipeline = VortexPipeline()
    ffmpeg = AsyncFFmpeg()

    logger.info("VORTEX v2.1 initialized within GENESIS")


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

async def execute_pipeline_job(job_id: str):
    """Execute pipeline in background"""
    logger.info(f"[VORTEX:{job_id}] Starting pipeline execution")

    state = await state_manager.get_job_state(job_id)
    if not state:
        logger.error(f"[VORTEX:{job_id}] Job not found")
        return

    try:
        while state.phase not in [PipelinePhase.COMPLETED.value, PipelinePhase.FAILED.value, "completed", "failed"]:
            state = await pipeline.execute_step(state)
            await state_manager.save_job_state(job_id, state)
            await asyncio.sleep(0.1)

        logger.info(f"[VORTEX:{job_id}] Pipeline completed: {state.phase}")

    except Exception as e:
        logger.error(f"[VORTEX:{job_id}] Pipeline error: {e}")
        state = state.add_error("execution", str(e), recoverable=False)
        state = state.transition_to(PipelinePhase.FAILED)
        await state_manager.save_job_state(job_id, state)

    finally:
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/health", response_model=VortexHealthResponse)
async def vortex_health():
    """VORTEX health check"""
    try:
        version = await asyncio.wait_for(ffmpeg.check_version(), timeout=5.0)
    except:
        version = "unavailable"

    return VortexHealthResponse(
        status="healthy" if ffmpeg else "initializing",
        version="2.1.0",
        ffmpeg_version=version[:60] if version else "unknown",
        active_jobs=len(active_job_tasks)
    )


@router.post("/assemble")
async def start_assembly(request: AssembleRequest, background_tasks: BackgroundTasks):
    """Start a new video assembly job"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    # Check job limit
    if len(active_job_tasks) >= 10:
        raise HTTPException(status_code=429, detail="Too many active jobs")

    job_id = str(uuid.uuid4())

    initial_state = GlobalState(
        job_id=job_id,
        video_urls=request.video_urls,
        voiceover_url=request.voiceover_url,
        music_url=request.music_url,
        output_formats=request.output_formats,
        brief_metadata=request.metadata or {},
    )

    await state_manager.save_job_state(job_id, initial_state)

    task = asyncio.create_task(execute_pipeline_job(job_id))
    active_job_tasks[job_id] = task

    logger.info(f"[VORTEX] Started job {job_id} with {len(request.video_urls)} clips")

    return {
        "job_id": job_id,
        "status": "started",
        "stream_url": f"/api/vortex/stream/{job_id}"
    }


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get current job status"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    state = await state_manager.get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    phase = state.phase if isinstance(state.phase, str) else state.phase.value

    return JobStatusResponse(
        job_id=state.job_id,
        phase=phase,
        phase_label=get_phase_label(phase),
        progress_pct=get_progress_pct(phase),
        processing_mode=state.processing_mode if isinstance(state.processing_mode, str) else state.processing_mode.value,
        created_at=state.created_at.isoformat(),
        updated_at=state.updated_at.isoformat(),
        errors=[e.dict() for e in state.errors],
        warnings=state.warnings,
        final_outputs=state.final_output_paths
    )


@router.get("/jobs")
async def list_jobs():
    """List all active jobs"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    job_ids = await state_manager.list_active_jobs()

    jobs = []
    for jid in job_ids:
        state = await state_manager.get_job_state(jid)
        if state:
            phase = state.phase if isinstance(state.phase, str) else state.phase.value
            jobs.append({
                "job_id": jid,
                "phase": phase,
                "progress_pct": get_progress_pct(phase),
                "created_at": state.created_at.isoformat()
            })

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/stream/{job_id}")
async def stream_job_progress(job_id: str, request: Request):
    """SSE endpoint for real-time job progress"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    state = await state_manager.get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[dict, None]:
        last_phase = None
        last_version = 0
        heartbeat_counter = 0

        while True:
            if await request.is_disconnected():
                break

            current_state = await state_manager.get_job_state(job_id)

            if current_state:
                phase = current_state.phase if isinstance(current_state.phase, str) else current_state.phase.value

                if phase != last_phase or current_state.version != last_version:
                    last_phase = phase
                    last_version = current_state.version

                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "job_id": job_id,
                            "phase": phase,
                            "phase_label": get_phase_label(phase),
                            "progress_pct": get_progress_pct(phase),
                            "version": current_state.version,
                            "has_errors": len(current_state.errors) > 0,
                            "final_outputs": current_state.final_output_paths if phase in ["completed", "COMPLETED"] else {}
                        })
                    }

                if phase in ["completed", "failed", "COMPLETED", "FAILED"]:
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "job_id": job_id,
                            "phase": phase,
                            "final_outputs": current_state.final_output_paths,
                            "errors": [e.dict() for e in current_state.errors]
                        })
                    }
                    break

            heartbeat_counter += 1
            if heartbeat_counter >= 5:
                heartbeat_counter = 0
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                }

            await asyncio.sleep(1.0)

    return EventSourceResponse(event_generator())


@router.get("/graph-structure")
async def get_graph_structure():
    """Pipeline graph structure for UI visualization"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    return pipeline.get_graph_structure()


@router.get("/job/{job_id}/trace")
async def get_job_trace(job_id: str):
    """Get execution trace for a job"""
    if not state_manager:
        raise HTTPException(status_code=503, detail="VORTEX not initialized")

    state = await state_manager.get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "phase_history": state.phase_history,
        "checkpoints": [c.dict() for c in state.checkpoints],
        "errors": [e.dict() for e in state.errors],
        "processing_mode": state.processing_mode if isinstance(state.processing_mode, str) else state.processing_mode.value,
        "complexity_score": state.complexity_score
    }


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id in active_job_tasks:
        active_job_tasks[job_id].cancel()
        del active_job_tasks[job_id]

    state = await state_manager.get_job_state(job_id)
    if state:
        state = state.transition_to(PipelinePhase.FAILED)
        state = state.add_error("cancel", "Job cancelled by user")
        await state_manager.save_job_state(job_id, state)

    return {"status": "cancelled", "job_id": job_id}


# =============================================================================
# IN-PROCESS EXECUTION (for GENESIS pipeline integration)
# =============================================================================

async def assemble_video_inprocess(
    video_urls: List[str],
    voiceover_url: Optional[str] = None,
    music_url: Optional[str] = None,
    output_formats: List[str] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Execute VORTEX pipeline in-process (not via HTTP).

    Called directly from GENESIS FlawlessOrchestrator for Agent 6.
    Returns dict of format -> output_path.
    """
    if not pipeline or not state_manager:
        raise RuntimeError("VORTEX not initialized - call initialize_vortex() first")

    job_id = f"genesis-{uuid.uuid4().hex[:12]}"

    initial_state = GlobalState(
        job_id=job_id,
        video_urls=video_urls,
        voiceover_url=voiceover_url,
        music_url=music_url,
        output_formats=output_formats or ["youtube_1080p"],
        brief_metadata=metadata or {},
    )

    await state_manager.save_job_state(job_id, initial_state)

    # Execute pipeline synchronously (blocking but async-safe)
    state = initial_state
    while state.phase not in [PipelinePhase.COMPLETED.value, PipelinePhase.FAILED.value, "completed", "failed"]:
        state = await pipeline.execute_step(state)
        await state_manager.save_job_state(job_id, state)

    if state.phase in ["failed", "FAILED", PipelinePhase.FAILED.value]:
        errors = [e.message for e in state.errors]
        raise RuntimeError(f"Video assembly failed: {errors}")

    return state.final_output_paths
