"""
NEXUS Brain API Router - Landing Page Concierge AI
Barrios A2I - Main website AI assistant

Version: 1.0.0
Purpose: Dedicated endpoint for NEXUS Brain - the landing page concierge
         that helps visitors understand all Barrios A2I services
"""

import uuid
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# OpenTelemetry imports (if available)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    tracer = trace.get_tracer("nexus-brain", "1.0.0")
    OTEL_ENABLED = True
except ImportError:
    OTEL_ENABLED = False
    tracer = None

# Anthropic client import
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger("nexus-brain")

# =============================================================================
# NEXUS BRAIN PERSONA - Landing Page Concierge
# =============================================================================

NEXUS_SYSTEM_PROMPT = """You are NEXUS, the AI assistant for Barrios A2I - an AI automation consultancy.

## YOUR GREETING
"Hello! I'm NEXUS, your guide to AI automation. What business challenge can I help you solve today?"

## YOUR ROLE
Help visitors understand our services and qualify leads. Ask about their BUSINESS NEEDS first - do NOT immediately ask about videos or commercials.

## OUR SERVICES

### 1. AI Assistants
- Personal and business AI that handles tasks and automates workflows
- Can search documents, write emails, schedule tasks, answer questions
- Runs locally for privacy or cloud for convenience

### 2. Marketing Overlord
- Automated campaigns, content generation, lead nurturing
- Analyzes your audience and creates targeted messaging
- Integrates with your existing marketing stack

### 3. AI-Powered Websites
- Intelligent sites with embedded AI assistants (like me!)
- Personalized visitor experiences, automated lead qualification
- Convert more visitors into customers

### 4. Custom App Development
- Full-stack applications with AI capabilities baked in
- Options: 30% equity partnership OR flat-fee for full ownership
- Best for founders with promising AI-enabled app ideas

### 5. AI Creative Director (Video Commercials)
- For video commercial creation, direct visitors to our Commercial Lab at /creative-director
- 9-agent orchestrated system creates professional video commercials
- Full commercial in ~4 minutes for ~$2.60

## CONVERSATION FLOW
1. Ask what challenges they're facing with their business
2. Listen and understand their pain points
3. Recommend the relevant service(s) that can help
4. Explain how we can solve their specific problem
5. Offer to connect them with Gary for a consultation

## KEY RULES
- DO NOT immediately ask about videos or commercials
- ASK about their business needs first
- Keep initial responses short and conversational (2-3 sentences)
- If they want video commercials, direct them to /creative-director
- For pricing questions, encourage scheduling a consultation

## ABOUT BARRIOS A2I
- Founded by Gary Barrios
- Mission: Transform complex technical challenges into streamlined AI automation
- 230,000+ production requests processed with 97.5% success rate
- Enterprise-grade systems with Netflix/Google-level resilience patterns

## WHAT YOU DON'T DO
- Access visitor's files or systems
- Process payments or contracts
- Share internal technical secrets
- Make up capabilities we don't have

Be helpful, be conversational, focus on solving their business problems."""


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class NexusRequest(BaseModel):
    """NEXUS chat request model."""
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    message: str = Field(..., description="User message", min_length=1, max_length=4000)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (page, referrer, etc)")
    visitor_info: Optional[Dict[str, Any]] = Field(None, description="Visitor metadata for personalization")


class NexusResponse(BaseModel):
    """NEXUS chat response model."""
    session_id: str
    response: str
    suggested_actions: List[Dict[str, str]] = Field(default_factory=list)
    is_complete: bool = False
    metadata: Optional[Dict[str, Any]] = None


class NexusHealthResponse(BaseModel):
    """NEXUS health check response."""
    status: str
    version: str
    persona: str
    uptime_seconds: float
    total_conversations: int
    avg_response_ms: float


# =============================================================================
# IN-MEMORY SESSION STORE (Replace with Redis in production)
# =============================================================================

class NexusSessionStore:
    """Simple in-memory session store for conversation history."""

    def __init__(self, max_sessions: int = 10000, max_history: int = 20):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._max_sessions = max_sessions
        self._max_history = max_history
        self._created_at = time.time()
        self._conversation_count = 0
        self._total_latency_ms = 0.0
        self._request_count = 0

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._conversation_count += 1

            # Evict oldest sessions if at capacity
            if len(self._sessions) > self._max_sessions:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]

        self._sessions[session_id].append({"role": role, "content": content})

        # Trim to max history
        if len(self._sessions[session_id]) > self._max_history * 2:
            self._sessions[session_id] = self._sessions[session_id][-self._max_history * 2:]

    def record_latency(self, latency_ms: float):
        """Record response latency for metrics."""
        self._total_latency_ms += latency_ms
        self._request_count += 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Get session store stats."""
        return {
            "active_sessions": len(self._sessions),
            "total_conversations": self._conversation_count,
            "avg_response_ms": self._total_latency_ms / max(1, self._request_count),
            "uptime_seconds": time.time() - self._created_at
        }


# Global session store
nexus_sessions = NexusSessionStore()


# =============================================================================
# NEXUS BRAIN ROUTER
# =============================================================================

router = APIRouter(prefix="/api/nexus", tags=["NEXUS Brain"])


def get_anthropic_client():
    """Get Anthropic client instance."""
    if not ANTHROPIC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Anthropic client not available")

    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    return Anthropic(api_key=api_key)


def generate_suggested_actions(response_text: str, message: str) -> List[Dict[str, str]]:
    """Generate contextual action buttons based on conversation."""
    actions = []

    lower_response = response_text.lower()
    lower_message = message.lower()

    # Suggest Creative Director for video/commercial interest
    if any(word in lower_message for word in ["video", "commercial", "ragnarok", "ad", "advertisement"]):
        actions.append({
            "label": "Try Creative Director",
            "action": "navigate",
            "target": "/creative-director"
        })

    # Suggest demo for general interest
    if any(word in lower_message for word in ["demo", "example", "show", "see", "proof"]):
        actions.append({
            "label": "See Live Demo",
            "action": "navigate",
            "target": "/creative-director"
        })

    # Suggest consultation for pricing/enterprise questions
    if any(word in lower_message for word in ["price", "cost", "enterprise", "custom", "quote"]):
        actions.append({
            "label": "Schedule Consultation",
            "action": "calendar",
            "target": "https://calendly.com/barriosa2i"
        })

    # Always offer to explore services if no specific action
    if not actions:
        actions.append({
            "label": "Explore Services",
            "action": "scroll",
            "target": "#services"
        })

    return actions[:3]  # Max 3 actions


@router.get("/health", response_model=NexusHealthResponse)
async def nexus_health():
    """NEXUS Brain health check endpoint."""
    stats = nexus_sessions.stats
    return NexusHealthResponse(
        status="online",
        version="1.0.0",
        persona="NEXUS Brain - Landing Page Concierge",
        uptime_seconds=stats["uptime_seconds"],
        total_conversations=stats["total_conversations"],
        avg_response_ms=stats["avg_response_ms"]
    )


@router.post("/chat", response_model=NexusResponse)
async def nexus_chat(request: NexusRequest):
    """
    NEXUS Brain chat endpoint - Landing page concierge AI.

    This endpoint provides conversational AI for the main Barrios A2I landing page,
    helping visitors understand services and routing them to appropriate pages.
    """
    start_time = time.time()

    # Generate session ID if not provided
    session_id = request.session_id or f"nexus_{uuid.uuid4().hex[:12]}"

    # Build context for tracing
    span_attrs = {
        "nexus.session_id": session_id,
        "nexus.message_length": len(request.message),
        "nexus.has_context": request.context is not None
    }

    # Optional OpenTelemetry span
    if OTEL_ENABLED and tracer:
        with tracer.start_as_current_span("nexus_chat", attributes=span_attrs) as span:
            try:
                response = await _process_nexus_chat(session_id, request, start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    else:
        return await _process_nexus_chat(session_id, request, start_time)


async def _process_nexus_chat(session_id: str, request: NexusRequest, start_time: float) -> NexusResponse:
    """Process NEXUS chat request."""

    try:
        client = get_anthropic_client()

        # Get conversation history
        history = nexus_sessions.get_history(session_id)

        # Build messages array
        messages = history + [{"role": "user", "content": request.message}]

        # Add visitor context to system prompt if available
        system_prompt = NEXUS_SYSTEM_PROMPT
        if request.visitor_info:
            context_note = f"\n\n## VISITOR CONTEXT\n"
            if request.visitor_info.get("referrer"):
                context_note += f"- Came from: {request.visitor_info['referrer']}\n"
            if request.visitor_info.get("page"):
                context_note += f"- Current page: {request.visitor_info['page']}\n"
            if request.visitor_info.get("returning"):
                context_note += f"- Returning visitor: Yes\n"
            system_prompt += context_note

        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )

        # Extract response text
        response_text = response.content[0].text if response.content else "I'm here to help! What would you like to know about Barrios A2I?"

        # Update session history
        nexus_sessions.add_turn(session_id, "user", request.message)
        nexus_sessions.add_turn(session_id, "assistant", response_text)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        nexus_sessions.record_latency(latency_ms)

        # Generate suggested actions
        suggested_actions = generate_suggested_actions(response_text, request.message)

        logger.info(f"NEXUS chat completed", extra={
            "session_id": session_id,
            "latency_ms": latency_ms,
            "message_length": len(request.message),
            "response_length": len(response_text)
        })

        return NexusResponse(
            session_id=session_id,
            response=response_text,
            suggested_actions=suggested_actions,
            is_complete=False,
            metadata={
                "latency_ms": round(latency_ms, 2),
                "model": "claude-sonnet-4-20250514",
                "turns": len(history) // 2 + 1
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NEXUS chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"NEXUS processing error: {str(e)}")


@router.delete("/session/{session_id}")
async def clear_nexus_session(session_id: str):
    """Clear a NEXUS conversation session."""
    if session_id in nexus_sessions._sessions:
        del nexus_sessions._sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@router.get("/stats")
async def nexus_stats():
    """Get NEXUS Brain usage statistics."""
    return {
        "status": "online",
        "version": "1.0.0",
        **nexus_sessions.stats
    }
