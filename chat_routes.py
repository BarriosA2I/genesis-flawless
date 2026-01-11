"""
================================================================================
GENESIS CHAT ROUTES
================================================================================
Migrated from creative-director-api and integrated with GENESIS orchestrator.
This provides the /api/chat endpoint for the Neural Interface.

Author: Barrios A2I | Merged: 2026-01-07
================================================================================
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# Import migrated intake system
try:
    from intake.video_brief_intake import CreativeDirectorOrchestrator, VideoBriefState, BriefPhase
    INTAKE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Intake module not available: {e}")
    INTAKE_AVAILABLE = False
    CreativeDirectorOrchestrator = None

# Import session storage
try:
    from storage.redis_session_store import RedisSessionStore, SessionData
    SESSION_STORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Redis session store not available: {e}")
    SESSION_STORE_AVAILABLE = False
    RedisSessionStore = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])

# =============================================================================
# GLOBAL INSTANCES (initialized on startup)
# =============================================================================

_orchestrator: Optional[CreativeDirectorOrchestrator] = None
_session_store: Optional[RedisSessionStore] = None


def get_orchestrator() -> CreativeDirectorOrchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        if not INTAKE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Intake module not available")

        # Initialize with Anthropic client
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except Exception as e:
            logger.warning(f"Anthropic client init failed: {e}, orchestrator will try fallback")
            client = None

        _orchestrator = CreativeDirectorOrchestrator(
            anthropic_client=client,
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        )
        logger.info("CreativeDirectorOrchestrator initialized")

    return _orchestrator


def get_session_store() -> Optional[RedisSessionStore]:
    """Get or create the session store instance."""
    global _session_store
    if _session_store is None and SESSION_STORE_AVAILABLE:
        _session_store = RedisSessionStore()
        logger.info("RedisSessionStore initialized")
    return _session_store


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response model with Phase 1 enhancements."""
    session_id: str
    response: str
    phase: str
    progress: float
    is_complete: bool
    progress_percentage: int = 0
    missing_fields: List[str] = []
    trigger_production: bool = False
    ragnarok_ready: bool = False
    mode: str = "intake"
    metadata: Optional[Dict[str, Any]] = None


class SessionRequest(BaseModel):
    """Session creation request."""
    client_name: Optional[str] = Field(None, description="Client name")
    project_type: Optional[str] = Field("video_commercial", description="Project type")
    context: Optional[Dict[str, Any]] = Field(None, description="Initial context")


class SessionResponse(BaseModel):
    """Session response model."""
    session_id: str
    status: str
    phase: str
    created_at: str


class BriefResponse(BaseModel):
    """RAGNAROK-ready brief response."""
    session_id: str
    is_complete: bool
    brief: Optional[Dict[str, Any]] = None
    missing_fields: List[str] = []
    completion_percentage: float


# =============================================================================
# CHAT ENDPOINTS
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the Neural Interface.

    This is the 23-agent RAGNAROK system's conversational interface.
    Handles intake, qualification, and prepares briefs for the full pipeline.

    The conversation collects:
    - Business identity (name, industry, description)
    - Target audience and pain points
    - Video goals and creative direction
    - Call-to-action and special requests
    """
    try:
        start_time = time.time()

        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Get orchestrator
        orchestrator = get_orchestrator()

        # Process message through the intake orchestrator
        result = await orchestrator.process_message(
            session_id=session_id,
            user_message=request.message
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Chat processed: session={session_id}, latency={latency_ms:.0f}ms")

        # Extract state info
        state = result.get("state", {})

        return ChatResponse(
            session_id=session_id,
            response=result.get("response", ""),
            phase=state.get("phase", "greeting"),
            progress=result.get("completion_percentage", 0.0),
            is_complete=result.get("is_complete", False),
            progress_percentage=result.get("progress_percentage", 0),
            missing_fields=result.get("missing_fields", []),
            trigger_production=result.get("trigger_production", False),
            ragnarok_ready=result.get("ragnarok_ready", False),
            mode=result.get("mode", "intake"),
            metadata={
                "latency_ms": latency_ms,
                "turns_count": state.get("turns_count", 0),
                "extracted_fields": list(state.get("brief_data", {}).keys()) if state.get("brief_data") else [],
                "extracted_data": state.get("brief_data", {}),  # Full brief data for voice selector
                "sentiment": result.get("sentiment", {}),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """
    Create a new chat session.

    Sessions track conversation state and collected brief data.
    Sessions expire after 24 hours of inactivity.
    """
    try:
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Store in Redis if available
        store = get_session_store()
        if store and store._available:
            store.save(session_id, {
                "session_id": session_id,
                "client_name": request.client_name or "",
                "project_type": request.project_type or "video_commercial",
                "phase": "greeting",
                "created_at": created_at
            })

        logger.info(f"Session created: {session_id}")

        return SessionResponse(
            session_id=session_id,
            status="created",
            phase="greeting",
            created_at=created_at
        )

    except Exception as e:
        logger.error(f"Session creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session status and current phase.
    """
    try:
        orchestrator = get_orchestrator()

        # Check if session exists in orchestrator
        if session_id in orchestrator.sessions:
            state = orchestrator.sessions[session_id]
            return SessionResponse(
                session_id=session_id,
                status="active",
                phase=state.phase.value if hasattr(state.phase, 'value') else str(state.phase),
                created_at=state.created_at if hasattr(state, 'created_at') else datetime.utcnow().isoformat()
            )

        # Check Redis store
        store = get_session_store()
        if store and store._available:
            session_data = store.load(session_id)
            if session_data:
                return SessionResponse(
                    session_id=session_id,
                    status="stored",
                    phase=session_data.get("phase", "unknown"),
                    created_at=session_data.get("created_at", datetime.utcnow().isoformat())
                )

        raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session lookup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its associated data.
    """
    try:
        orchestrator = get_orchestrator()

        # Remove from orchestrator
        if session_id in orchestrator.sessions:
            del orchestrator.sessions[session_id]

        # Remove from Redis
        store = get_session_store()
        if store and store._available:
            store.delete(session_id)

        logger.info(f"Session deleted: {session_id}")
        return {"status": "deleted", "session_id": session_id}

    except Exception as e:
        logger.error(f"Session deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current session state for frontend progress bar.
    Returns Phase 1 tracking data.
    """
    try:
        # Import BriefSessionState functions
        from intake.video_brief_intake import get_or_create_brief_session

        session = get_or_create_brief_session(session_id)

        return {
            "session_id": session_id,
            "progress_percentage": session.completion_percentage(),
            "fields_gathered": list(session.get_filled_fields().keys()),
            "missing_fields": session.get_missing_fields(),
            "awaiting_confirmation": session.awaiting_confirmation,
            "conversation_count": session.conversation_count,
            "is_complete": session.is_complete(),
            "pricing_ask_count": session.pricing_ask_count,
        }

    except Exception as e:
        logger.error(f"Session status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ragnarok/brief/{session_id}", response_model=BriefResponse)
async def get_ragnarok_brief(session_id: str):
    """
    Get the RAGNAROK-ready brief for a session.

    This endpoint returns the structured brief data that can be
    passed to the RAGNAROK v7.0 APEX pipeline for video generation.

    The brief is ready when:
    - Phase is 'complete' or 'confirm'
    - All required fields are captured
    - Completion percentage >= 80%
    """
    try:
        orchestrator = get_orchestrator()

        if session_id not in orchestrator.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        state = orchestrator.sessions[session_id]

        # Check completion
        is_complete = state.phase in [BriefPhase.COMPLETE, BriefPhase.CONFIRM] if INTAKE_AVAILABLE else False
        completion_pct = state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0.0

        # Get missing fields
        missing = []
        if hasattr(state, 'brief_data'):
            required = ['business_name', 'industry', 'target_audience', 'video_goal', 'cta']
            for field in required:
                if not state.brief_data.get(field):
                    missing.append(field)

        # Build RAGNAROK-format brief
        brief = None
        if is_complete or completion_pct >= 80:
            brief = state.to_ragnarok_input() if hasattr(state, 'to_ragnarok_input') else state.to_dict()

        return BriefResponse(
            session_id=session_id,
            is_complete=is_complete,
            brief=brief,
            missing_fields=missing,
            completion_percentage=completion_pct
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brief retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GENESIS CHAT ALIASES (for backwards compatibility)
# =============================================================================

@router.post("/genesis/chat", response_model=ChatResponse)
async def genesis_chat(request: ChatRequest):
    """Alias for /api/chat - for GENESIS routing compatibility."""
    return await chat(request)


@router.post("/genesis/session", response_model=SessionResponse)
async def genesis_session(request: SessionRequest):
    """Alias for /api/session - for GENESIS routing compatibility."""
    return await create_session(request)


@router.get("/genesis/session/{session_id}", response_model=SessionResponse)
async def genesis_get_session(session_id: str):
    """Alias for /api/session/{id} - for GENESIS routing compatibility."""
    return await get_session(session_id)
