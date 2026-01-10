"""
================================================================================
GENESIS PRODUCTION ROUTES
================================================================================
Server-Sent Events (SSE) endpoint for real-time video production status updates.
Supports Phase 2 of Creative Director AI.

Author: Barrios A2I | Created: 2026-01-10
================================================================================
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import production status tracking from intake module
try:
    from intake.video_brief_intake import (
        get_production_status,
        create_production_status,
        update_production_step,
        ProductionStep,
        ProductionStatus,
        PRODUCTION_STATUSES,
    )
    PRODUCTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Production module not available: {e}")
    PRODUCTION_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/production", tags=["Production"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ProductionStatusResponse(BaseModel):
    """Production status response model."""
    session_id: str
    current_step: str
    step_progress: int
    overall_progress: int
    started_at: Optional[float] = None
    elapsed_seconds: Optional[int] = None
    estimated_remaining: Optional[int] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class WebhookPayload(BaseModel):
    """RAGNAROK webhook payload model."""
    session_id: str
    step: str
    progress: int = 0
    video_url: Optional[str] = None
    error: Optional[str] = None


class SimulateRequest(BaseModel):
    """Simulation request model."""
    speed: float = 1.0  # Multiplier for simulation speed (1.0 = normal, 2.0 = 2x faster)


# =============================================================================
# PRODUCTION STATUS ENDPOINTS
# =============================================================================

@router.get("/status/{session_id}", response_model=ProductionStatusResponse)
async def get_status(session_id: str):
    """
    Get current production status for a session.

    Returns the current step, progress percentage, elapsed time, and video URL if complete.
    """
    if not PRODUCTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Production tracking not available")

    status = get_production_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="No production found for this session")

    elapsed = None
    if status.started_at:
        elapsed = int(time.time() - status.started_at)

    return ProductionStatusResponse(
        session_id=session_id,
        current_step=status.current_step.value,
        step_progress=status.step_progress,
        overall_progress=status.calculate_overall_progress(),
        started_at=status.started_at,
        elapsed_seconds=elapsed,
        estimated_remaining=status.estimate_remaining_time(),
        video_url=status.video_url,
        error_message=status.error_message,
        retry_count=status.retry_count,
    )


@router.get("/status/{session_id}/stream")
async def stream_status(session_id: str, request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time production status updates.

    Connect to this endpoint to receive live updates as video production progresses.
    Events are sent every 2-3 seconds with current status.

    Event format:
    ```
    event: status
    data: {"step": "visuals", "progress": 45, "elapsed": 120, ...}

    event: complete
    data: {"video_url": "https://...", "total_time": 243}

    event: error
    data: {"message": "Production failed", "retry_count": 1}
    ```
    """
    if not PRODUCTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Production tracking not available")

    status = get_production_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="No production found for this session")

    async def event_generator():
        """Generate SSE events for production status."""
        last_step = None
        last_progress = -1

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"SSE client disconnected: {session_id}")
                break

            # Get current status
            current_status = get_production_status(session_id)
            if not current_status:
                yield f"event: error\ndata: {{\"message\": \"Production not found\"}}\n\n"
                break

            # Calculate elapsed time
            elapsed = 0
            if current_status.started_at:
                elapsed = int(time.time() - current_status.started_at)

            overall_progress = current_status.calculate_overall_progress()

            # Send update if step or progress changed
            step_changed = current_status.current_step != last_step
            progress_changed = overall_progress != last_progress

            if step_changed or progress_changed:
                last_step = current_status.current_step
                last_progress = overall_progress

                # Check for completion
                if current_status.current_step == ProductionStep.COMPLETE:
                    total_time = 0
                    if current_status.started_at and current_status.completed_at:
                        total_time = int(current_status.completed_at - current_status.started_at)

                    yield f"event: complete\ndata: {{\"video_url\": \"{current_status.video_url or ''}\", \"total_time\": {total_time}}}\n\n"
                    break

                # Check for error
                elif current_status.current_step == ProductionStep.ERROR:
                    yield f"event: error\ndata: {{\"message\": \"{current_status.error_message or 'Unknown error'}\", \"retry_count\": {current_status.retry_count}}}\n\n"
                    break

                # Send status update
                else:
                    data = {
                        "step": current_status.current_step.value,
                        "step_progress": current_status.step_progress,
                        "overall_progress": overall_progress,
                        "elapsed": elapsed,
                        "estimated_remaining": current_status.estimate_remaining_time(),
                    }
                    yield f"event: status\ndata: {data}\n\n"

            # Wait before next update
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/webhook/ragnarok")
async def ragnarok_webhook(payload: WebhookPayload):
    """
    Webhook endpoint for RAGNAROK pipeline to report progress.

    Called by each stage of the RAGNAROK v7.0 APEX pipeline:
    - script: Script generation complete
    - visuals: Visual asset selection complete
    - voice: Voiceover generation complete
    - edit: Video editing complete
    - upload: Upload to CDN complete
    - complete: Full pipeline complete
    - error: Pipeline failed
    """
    if not PRODUCTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Production tracking not available")

    logger.info(f"RAGNAROK webhook: session={payload.session_id}, step={payload.step}, progress={payload.progress}")

    try:
        # Map step string to enum
        step_map = {
            "queued": ProductionStep.QUEUED,
            "script": ProductionStep.SCRIPT_GENERATION,
            "visuals": ProductionStep.VISUAL_SELECTION,
            "voice": ProductionStep.VOICEOVER_GENERATION,
            "edit": ProductionStep.VIDEO_EDITING,
            "upload": ProductionStep.UPLOAD,
            "complete": ProductionStep.COMPLETE,
            "error": ProductionStep.ERROR,
        }

        step = step_map.get(payload.step.lower())
        if not step:
            raise HTTPException(status_code=400, detail=f"Unknown step: {payload.step}")

        # Update production status
        await update_production_step(
            session_id=payload.session_id,
            step=step,
            progress=payload.progress,
            video_url=payload.video_url,
            error=payload.error,
        )

        return {"status": "ok", "step": step.value}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/{session_id}")
async def simulate_production(session_id: str, request: SimulateRequest):
    """
    DEV ONLY: Simulate production pipeline for testing.

    Advances through all production steps with configurable speed.
    Useful for testing the SSE endpoint and frontend UI without actual video generation.
    """
    if not PRODUCTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Production tracking not available")

    status = get_production_status(session_id)
    if not status:
        # Create a mock production status for testing
        status = create_production_status(session_id, {
            "business_name": "Test Business",
            "product": "Test Product",
            "audience": "Test Audience",
            "cta": "Learn More",
        })

    async def run_simulation():
        """Run production simulation in background."""
        steps = [
            (ProductionStep.SCRIPT_GENERATION, 15),
            (ProductionStep.VISUAL_SELECTION, 45),
            (ProductionStep.VOICEOVER_GENERATION, 30),
            (ProductionStep.VIDEO_EDITING, 60),
            (ProductionStep.UPLOAD, 20),
        ]

        speed = max(0.1, min(10.0, request.speed))  # Clamp speed

        for step, duration in steps:
            # Simulate step progress
            for progress in range(0, 101, 10):
                await update_production_step(session_id, step, progress)
                await asyncio.sleep((duration / 10) / speed)

        # Mark as complete
        await update_production_step(
            session_id,
            ProductionStep.COMPLETE,
            100,
            video_url=f"https://cdn.barriosa2i.com/videos/{session_id}/final.mp4"
        )

    # Start simulation in background
    asyncio.create_task(run_simulation())

    return {
        "status": "simulation_started",
        "session_id": session_id,
        "speed": request.speed,
        "message": "Connect to /api/production/status/{session_id}/stream to watch progress"
    }
