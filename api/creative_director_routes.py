"""
============================================================================
CREATIVE DIRECTOR V3.1 - FASTAPI ROUTES
============================================================================

File: api/creative_director_routes.py

Endpoints:
- POST /api/creative-director/start         - Start new session
- POST /api/creative-director/chat          - Send message (triggers RAGNAROK)
- POST /api/creative-director/chat/stream   - Send message (SSE streaming)
- POST /api/creative-director/upload-logo   - Upload logo file
- GET  /api/creative-director/session/{id}  - Get session state
- GET  /api/creative-director/health        - Health check

V3.1 Changes:
- Added SSE streaming endpoint for real-time UI updates
- Wired production trigger to RAGNAROK pipeline
- When user confirms brief, RAGNAROK video generation starts automatically

Author: Barrios A2I | Version: 3.1.0 | January 2026
============================================================================
"""

import os
import json
import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from intake.creative_director_fsm import get_fsm

logger = logging.getLogger("creative_director_routes")

router = APIRouter(prefix="/api/creative-director", tags=["Creative Director V3"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StartRequest(BaseModel):
    """Request to start a new Creative Director session."""
    session_id: Optional[str] = Field(None, description="Optional session ID (auto-generated if not provided)")


class ChatRequest(BaseModel):
    """Chat message request."""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    """Chat response with session state."""
    session_id: str
    response: str
    state: str
    progress: int
    production_triggered: bool = False
    brief: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Session state response."""
    session_id: str
    state: str
    brief: Dict[str, Any]
    progress: int
    production_triggered: bool
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    api_key_configured: bool


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/start", response_model=ChatResponse)
async def start_session(request: StartRequest):
    """
    Start a new Creative Director session.

    Returns welcome message and initial state.
    """
    session_id = request.session_id or str(uuid.uuid4())
    fsm = get_fsm()

    try:
        result = fsm.start_session(session_id)
        logger.info(f"[V3] Session started: {session_id}")

        return ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            state=result["state"],
            progress=result["progress"],
            production_triggered=result["production_triggered"]
        )
    except Exception as e:
        logger.error(f"[V3] Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the Creative Director.

    The FSM will:
    1. Extract relevant slot values from the message
    2. Transition to the next state
    3. Generate a conversational response
    4. Trigger RAGNAROK production if confirmed
    """
    fsm = get_fsm()

    try:
        result = await fsm.process_message(request.session_id, request.message)
        logger.info(f"[V3] Message processed for {request.session_id}, state={result['state']}, progress={result['progress']}")

        if result.get("production_triggered"):
            prod_result = result.get("production_result", {})
            if prod_result.get("success"):
                logger.info(f"[V3] Production triggered successfully for {request.session_id}")
            else:
                logger.warning(f"[V3] Production trigger failed: {prod_result.get('error')}")

        return ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            state=result["state"],
            progress=result["progress"],
            production_triggered=result["production_triggered"],
            brief=result.get("brief")
        )
    except Exception as e:
        logger.error(f"[V3] Chat error for {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming version of chat.

    Sends real-time events for UI updates:
    - typing: Agent is thinking
    - state: State transition occurred
    - response: Agent response text
    - production: Production was triggered
    - brief: Brief data (when complete)
    - done: Stream complete
    - error: Error occurred
    """
    fsm = get_fsm()

    async def event_generator():
        try:
            # Send typing indicator
            yield f"data: {json.dumps({'type': 'typing', 'message': 'Thinking...'})}\n\n"

            # Process message
            result = await fsm.process_message(request.session_id, request.message)

            # Send state update
            yield f"data: {json.dumps({'type': 'state', 'state': result['state'], 'progress': result['progress']})}\n\n"

            # Send response
            yield f"data: {json.dumps({'type': 'response', 'content': result['response']})}\n\n"

            # Send brief if available
            if result.get('brief'):
                yield f"data: {json.dumps({'type': 'brief', 'brief': result['brief']})}\n\n"

            # Send production trigger if applicable
            if result.get('production_triggered'):
                prod_result = result.get('production_result', {})
                yield f"data: {json.dumps({'type': 'production', 'triggered': True, 'success': prod_result.get('success', False)})}\n\n"

            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'session_id': result['session_id']})}\n\n"

        except Exception as e:
            logger.error(f"[V3-Stream] Error for {request.session_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/upload-logo")
async def upload_logo(
    session_id: str = Form(..., description="Session ID"),
    file: UploadFile = File(..., description="Logo image file"),
    instructions: Optional[str] = Form(None, description="Logo placement instructions")
):
    """
    Upload a logo for a session.

    The logo will be associated with the session's brief.
    In production, this would upload to cloud storage (S3, GCS, etc.)
    """
    fsm = get_fsm()

    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )

    # In production, upload to cloud storage and get URL
    # For now, use a placeholder URL
    logo_url = f"/uploads/{session_id}/{file.filename}"

    # Set logo in session
    success = fsm.set_logo(session_id, logo_url, instructions)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"[V3] Logo uploaded for {session_id}: {file.filename}")

    return {
        "status": "uploaded",
        "session_id": session_id,
        "logo_url": logo_url,
        "filename": file.filename,
        "instructions": instructions
    }


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get the current state of a session.

    Returns the full session state including brief data and progress.
    """
    fsm = get_fsm()

    session = fsm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(**session)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session.

    This removes the session from memory.
    """
    fsm = get_fsm()

    if session_id in fsm.sessions:
        del fsm.sessions[session_id]
        logger.info(f"[V3] Session deleted: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns service status and configuration.
    """
    return HealthResponse(
        status="healthy",
        version="v3-fsm",
        api_key_configured=bool(os.getenv("ANTHROPIC_API_KEY"))
    )


@router.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Creative Director V3",
        "version": "3.1.0",
        "architecture": "State Machine + LLM Hybrid + RAGNAROK Integration",
        "endpoints": {
            "start": "POST /api/creative-director/start",
            "chat": "POST /api/creative-director/chat",
            "chat_stream": "POST /api/creative-director/chat/stream (SSE)",
            "upload_logo": "POST /api/creative-director/upload-logo",
            "session": "GET /api/creative-director/session/{session_id}",
            "health": "GET /api/creative-director/health"
        },
        "state_flow": "START -> BUSINESS_NAME -> PRODUCT -> AUDIENCE -> CTA -> TONE -> LOGO -> CONFIRM -> PRODUCTION -> COMPLETE",
        "production": "RAGNAROK video generation triggered on CONFIRM"
    }
