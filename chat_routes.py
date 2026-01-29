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

# Import V2 LangGraph Creative Director
try:
    from intake.creative_director_v2 import creative_director_v2
    V2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative Director V2 not available: {e}")
    V2_AVAILABLE = False
    creative_director_v2 = None

# Import V3 Natural Conversation Creative Director
try:
    from intake.creative_director_v3 import CreativeDirectorV3
    creative_director_v3 = CreativeDirectorV3()
    V3_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative Director V3 not available: {e}")
    V3_AVAILABLE = False
    creative_director_v3 = None

# Import NEXUS Unified Agent (V4 - token-gated with knowledge)
try:
    from intake.nexus_unified_agent import NexusUnifiedAgent
    nexus_unified_agent = NexusUnifiedAgent()
    UNIFIED_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NEXUS Unified Agent not available: {e}")
    UNIFIED_AVAILABLE = False
    nexus_unified_agent = None

# Import session storage
try:
    from storage.redis_session_store import RedisSessionStore, SessionData
    SESSION_STORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Redis session store not available: {e}")
    SESSION_STORE_AVAILABLE = False
    RedisSessionStore = None

# Import NEXUS Q&A handler
try:
    from nexus_qa import nexus_qa_handler
    QA_HANDLER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NEXUS Q&A handler not available: {e}")
    QA_HANDLER_AVAILABLE = False
    nexus_qa_handler = None

# Import Integration Hub client for lead capture
try:
    from nexus_integration_client import (
        get_integration_client,
        detect_trigger_keywords,
        enrich_response_with_hooks,
        Platform,
        LeadSource,
        ConversionStatus,
    )
    INTEGRATION_CLIENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Integration client not available: {e}")
    INTEGRATION_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])

# =============================================================================
# INTENT DETECTION - Routes customer Q&A vs video intake
# =============================================================================

# Keywords that indicate customer Q&A (not video production)
QA_INTENT_KEYWORDS = [
    "pricing", "price", "cost", "how much", "tier", "plan", "subscription",
    "what is", "what are", "what do", "explain", "tell me about", "describe",
    "services", "features", "capabilities", "roi", "results", "benefits",
    "consultation", "meeting", "schedule", "book", "call", "demo",
    "who is", "who are", "company", "barrios", "about you",
    "how does", "how do", "why should", "can you", "do you",
    "help me understand", "i have a question", "question about",
    "contact", "support", "talk to", "speak with",
]

# Keywords that indicate lead/buying intent (capture to Integration Hub)
LEAD_INTENT_KEYWORDS = [
    "pricing", "cost", "quote", "hire", "work with",
    "schedule", "consultation", "demo", "interested",
    "how much", "get started", "contact", "proposal",
    "a2i", "commercial",
]

# Keywords that indicate explicit video production intent
VIDEO_INTENT_KEYWORDS = [
    "video", "commercial", "ad ", "advertisement", "create a", "make a",
    "produce", "film", "content for", "marketing video", "promo",
    "i want to create", "i need a video", "let's make", "start production",
    "generate a", "build a commercial",
]


def detect_intent(message: str) -> str:
    """
    Classify message as 'qa' (customer questions) or 'video_intake' (production).

    Intent Priority:
    1. Explicit video keywords → video_intake
    2. Q&A keywords → qa
    3. Questions (ends with ?) → qa
    4. Default → video_intake (for statements that may be brief data)
    """
    message_lower = message.lower().strip()

    # First check for explicit video intent (highest priority)
    for keyword in VIDEO_INTENT_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"[INTENT] Detected VIDEO intent: '{keyword}' in '{message[:50]}...'")
            return "video_intake"

    # Then check for Q&A intent
    for keyword in QA_INTENT_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"[INTENT] Detected QA intent: '{keyword}' in '{message[:50]}...'")
            return "qa"

    # Questions default to Q&A
    if message_lower.endswith("?"):
        logger.info(f"[INTENT] Detected QA intent: message is a question")
        return "qa"

    # Default to video intake for statements (likely brief data)
    logger.info(f"[INTENT] Defaulting to video_intake for: '{message[:50]}...'")
    return "video_intake"

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

    ROUTING LOGIC:
    1. Detect intent (Q&A vs video production)
    2. Q&A questions → NEXUS Q&A handler (pricing, services, general questions)
    3. Video intent → Orchestrator for brief collection

    The video intake conversation collects:
    - Business identity (name, industry, description)
    - Target audience and pain points
    - Video goals and creative direction
    - Call-to-action and special requests
    """
    try:
        start_time = time.time()

        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())

        # =====================================================================
        # INTENT ROUTING - Route Q&A separately from video intake
        # =====================================================================
        intent = detect_intent(request.message)
        logger.info(f"[CHAT] Intent detected: {intent} for message: '{request.message[:50]}...'")

        # =====================================================================
        # LEAD CAPTURE - Track buying intent via Integration Hub
        # =====================================================================
        lead_captured = False
        lead_id = None
        suggested_hook = ""  # SCRIPTWRITER-X personalized hook
        if INTEGRATION_CLIENT_AVAILABLE:
            message_lower = request.message.lower()
            if any(keyword in message_lower for keyword in LEAD_INTENT_KEYWORDS):
                try:
                    client = get_integration_client()
                    engagement_result = await client.handle_social_engagement(
                        platform=Platform.WEBSITE,
                        contact_handle=session_id,
                        message=request.message,
                        engagement_type="website_chat"
                    )
                    # Check for errors in the result (client returns {"error": ...} on failure)
                    if "error" in engagement_result:
                        logger.warning(f"[CHAT] Lead capture hub error: {engagement_result.get('error')}")
                    else:
                        lead_captured = True
                        lead_id = engagement_result.get("lead_id")
                        suggested_hook = engagement_result.get("suggested_response", "")
                        logger.info(f"[CHAT] Lead captured: {lead_id} | Hook: {suggested_hook[:50] if suggested_hook else 'None'}...")
                except Exception as e:
                    logger.warning(f"[CHAT] Lead capture failed: {e}")

        # Route to Q&A handler for customer questions
        if intent == "qa" and QA_HANDLER_AVAILABLE and nexus_qa_handler:
            logger.info(f"[CHAT] Routing to NEXUS Q&A handler")
            qa_result = await nexus_qa_handler(request.message, session_id)

            # Get base response
            response_text = qa_result.get("response", "")

            # Enrich response with SCRIPTWRITER-X hooks for lead conversion
            if INTEGRATION_CLIENT_AVAILABLE and lead_captured and suggested_hook:
                # Prepend the personalized hook to the response
                response_text = f"{suggested_hook}\n\n{response_text}"
                logger.info(f"[CHAT] Response enriched with SCRIPTWRITER-X hook")
            elif INTEGRATION_CLIENT_AVAILABLE and lead_captured:
                # Try async enrichment if no suggested_hook was returned
                try:
                    enriched = await enrich_response_with_hooks(
                        base_response=response_text,
                        context=request.message,
                        visitor_profile={"session_id": session_id}
                    )
                    response_text = enriched
                    logger.info(f"[CHAT] Response enriched via enrich_response_with_hooks")
                except Exception as e:
                    logger.warning(f"[CHAT] Response enrichment failed: {e}")

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"[CHAT] Q&A processed: session={session_id}, latency={latency_ms:.0f}ms")

            # Return Q&A response in ChatResponse format
            return ChatResponse(
                session_id=session_id,
                response=response_text,
                phase=qa_result.get("phase", "qa"),
                progress=qa_result.get("progress", 0.0),
                is_complete=qa_result.get("is_complete", False),
                progress_percentage=qa_result.get("progress_percentage", 0),
                missing_fields=qa_result.get("missing_fields", []),
                trigger_production=False,  # Never trigger production from Q&A
                ragnarok_ready=False,
                mode="qa",
                metadata={
                    "latency_ms": latency_ms,
                    "intent": intent,
                    "handler": "nexus_qa",
                    "lead_captured": lead_captured,
                    "lead_id": lead_id,
                    "suggested_hook": suggested_hook if suggested_hook else None,
                    **qa_result.get("metadata", {})
                }
            )

        # =====================================================================
        # VIDEO INTAKE - Route to orchestrator for brief collection
        # =====================================================================
        logger.info(f"[CHAT] Routing to video intake orchestrator")

        # Get orchestrator
        orchestrator = get_orchestrator()

        # Process message through the intake orchestrator
        result = await orchestrator.process_message(
            session_id=session_id,
            user_message=request.message
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"[CHAT] Intake processed: session={session_id}, latency={latency_ms:.0f}ms")

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
                "intent": intent,
                "handler": "orchestrator",
                "turns_count": state.get("turns_count", 0),
                "extracted_fields": list(state.get("brief_data", {}).keys()) if state.get("brief_data") else [],
                "extracted_data": state.get("brief_data", {}),
                "sentiment": result.get("sentiment", {}),
                "lead_captured": lead_captured,
                "lead_id": lead_id,
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


# =============================================================================
# V2 ENDPOINTS - LangGraph Creative Director (Beta)
# =============================================================================

@router.post("/chat/v2")
async def chat_v2(request: ChatRequest):
    """
    LangGraph-powered Creative Director (Beta)

    This endpoint runs the new state machine architecture in parallel
    with the existing /api/chat endpoint.

    Key differences from v1:
    - Schema-first extraction (no regex)
    - Deterministic routing (code decides, not AI)
    - Proper session state management
    - Normalized field names

    Returns same format as v1 for frontend compatibility.
    """
    if not V2_AVAILABLE or not creative_director_v2:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V2 not available. Check langgraph dependency."
        )

    try:
        result = await creative_director_v2.process_message(
            session_id=request.session_id,
            message=request.message
        )
        return result

    except Exception as e:
        logger.error(f"V2 endpoint error: {e}", exc_info=True)
        return {
            "response": "I encountered an issue. Please try again.",
            "progress_percentage": 0,
            "missing_fields": [],
            "error": str(e),
            "version": "v2-langgraph"
        }


@router.get("/chat/v2/session/{session_id}")
async def get_session_v2(session_id: str):
    """
    Debug endpoint: Get current V2 session state.
    Useful for troubleshooting extraction issues.
    """
    if not V2_AVAILABLE or not creative_director_v2:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V2 not available"
        )

    state = creative_director_v2.sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@router.delete("/chat/v2/session/{session_id}")
async def reset_session_v2(session_id: str):
    """
    Reset a V2 session to start fresh.
    """
    if not V2_AVAILABLE or not creative_director_v2:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V2 not available"
        )

    if session_id in creative_director_v2.sessions:
        del creative_director_v2.sessions[session_id]
        return {"status": "reset", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# =============================================================================
# V3 ENDPOINTS - Natural Conversation Creative Director
# =============================================================================

@router.post("/chat/v3")
async def chat_v3(request: ChatRequest):
    """
    V3 Creative Director - Natural conversation that HELPS users.

    Key fix from V2: When user says "I'm not sure", V3 provides helpful
    suggestions instead of just repeating the question.

    Acts like a creative partner, not a robotic form.
    """
    if not V3_AVAILABLE or not creative_director_v3:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V3 not available"
        )

    try:
        result = await creative_director_v3.process_message(
            session_id=request.session_id,
            user_message=request.message
        )
        return result

    except Exception as e:
        logger.error(f"V3 endpoint error: {e}", exc_info=True)
        return {
            "response": "I encountered an issue. Please try again.",
            "brief": {},
            "phase": "intake",
            "error": str(e),
            "version": "v3-natural"
        }


@router.get("/chat/v3/greeting")
async def get_v3_greeting():
    """Get V3 initial greeting."""
    if not V3_AVAILABLE or not creative_director_v3:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V3 not available"
        )
    return {"greeting": creative_director_v3.get_greeting()}


@router.get("/chat/v3/session/{session_id}")
async def get_session_v3(session_id: str):
    """Get V3 session state."""
    if not V3_AVAILABLE or not creative_director_v3:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V3 not available"
        )

    state = creative_director_v3.sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "brief": state.brief.to_dict(),
        "phase": state.phase,
        "message_count": len(state.messages)
    }


@router.delete("/chat/v3/session/{session_id}")
async def reset_session_v3(session_id: str):
    """Reset a V3 session to start fresh."""
    if not V3_AVAILABLE or not creative_director_v3:
        raise HTTPException(
            status_code=503,
            detail="Creative Director V3 not available"
        )

    if session_id in creative_director_v3.sessions:
        del creative_director_v3.sessions[session_id]
        return {"status": "reset", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# =============================================================================
# UNIFIED ENDPOINTS - Token-Gated AI with Knowledge Injection
# =============================================================================

class UnifiedChatRequest(BaseModel):
    """Request model for unified chat endpoint."""
    session_id: str
    message: str
    user_id: Optional[str] = None


@router.post("/chat/unified")
async def chat_unified(request: UnifiedChatRequest):
    """
    NEXUS Unified Agent - Token-gated AI assistant with knowledge injection.

    Features:
    - Answers pricing/service questions from NEXUS Brain knowledge
    - Checks user token balance before generation
    - Offers purchase links if tokens = 0
    - Deducts tokens on generation confirmation
    - Preserves V3 behaviors: short responses, one question, texting vibe
    """
    if not UNIFIED_AVAILABLE or not nexus_unified_agent:
        raise HTTPException(
            status_code=503,
            detail="NEXUS Unified Agent not available"
        )

    try:
        result = await nexus_unified_agent.process_message(
            session_id=request.session_id,
            user_message=request.message,
            user_id=request.user_id
        )
        return result

    except Exception as e:
        logger.error(f"Unified endpoint error: {e}", exc_info=True)
        return {
            "response": "Hit a snag. Try that again?",
            "brief": {},
            "phase": "intake",
            "session_id": request.session_id,
            "tokens": 0,
            "intent": "error",
            "error": str(e)
        }


@router.get("/chat/unified/greeting")
async def get_unified_greeting():
    """Get unified agent initial greeting."""
    if not UNIFIED_AVAILABLE or not nexus_unified_agent:
        raise HTTPException(
            status_code=503,
            detail="NEXUS Unified Agent not available"
        )
    return {"greeting": nexus_unified_agent.get_greeting()}


# =============================================================================
# NEXUS INTEGRATION HUB ENDPOINTS - Phase 5
# =============================================================================

@router.get("/nexus/integration/health")
async def integration_health():
    """
    Check Integration Hub connectivity.
    Returns hub status, circuit breaker state, and lead stats.
    """
    if not INTEGRATION_CLIENT_AVAILABLE:
        return {
            "status": "unavailable",
            "hub_connected": False,
            "reason": "Integration client not imported",
        }

    try:
        client = get_integration_client()
        hub_health = await client.health_check()
        return {
            "status": "healthy",
            "hub_connected": True,
            "hub_status": hub_health.get("status"),
            "hub_services": hub_health.get("services", {}),
            "hub_stats": hub_health.get("stats", {}),
        }
    except Exception as e:
        logger.error(f"[INTEGRATION] Health check failed: {e}")
        return {
            "status": "degraded",
            "hub_connected": False,
            "error": str(e),
        }


class LeadStatusUpdate(BaseModel):
    """Request model for lead status updates."""
    status: str
    deal_value: Optional[float] = None
    notes: Optional[str] = None


@router.patch("/leads/{lead_id}/status")
async def update_lead_status_endpoint(lead_id: str, update: LeadStatusUpdate):
    """
    Update lead status - triggers feedback loop on conversion.

    Status values: new, engaged, qualified, proposal_sent, converted, lost, ghosted

    When status is 'converted', provide deal_value to trigger the SCRIPTWRITER-X
    feedback loop, marking the source content as "legendary".
    """
    if not INTEGRATION_CLIENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Integration client not available"
        )

    try:
        client = get_integration_client()
        result = await client.update_lead_status(
            lead_id=lead_id,
            status=ConversionStatus(update.status),
            deal_value=update.deal_value,
        )

        logger.info(f"[INTEGRATION] Lead {lead_id} status updated to {update.status}")
        if update.status == "converted" and update.deal_value:
            logger.info(f"[INTEGRATION] Conversion feedback triggered: ${update.deal_value:,.2f}")

        return {
            "success": True,
            "lead_id": lead_id,
            "new_status": update.status,
            "feedback_triggered": update.status == "converted",
            "result": result,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status value: {update.status}. Valid: new, engaged, qualified, proposal_sent, converted, lost, ghosted"
        )
    except Exception as e:
        logger.error(f"[INTEGRATION] Lead status update failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update lead: {str(e)}"
        )
