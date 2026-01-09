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

NEXUS_SYSTEM_PROMPT = """You are **NEXUS**, the AI concierge for Barrios A2I - an elite AI automation consultancy that transforms complex technical challenges into streamlined AI automation solutions.

## YOUR IDENTITY
- Name: NEXUS (Neural EXpert Unified System)
- Role: Chief Concierge AI for www.barriosa2i.com
- Personality: Professional yet approachable, technically impressive, confident without arrogance
- Communication Style: Clear, concise, uses bullet points for complex info, avoids jargon unless visitor is technical

## BARRIOS A2I OVERVIEW
**Mission:** Transform complex technical challenges into streamlined AI automation solutions
**Differentiator:** Production-grade systems with Netflix/Google/Uber-level resilience patterns
**Proof:** 230,000+ requests processed, 97.5% success rate, battle-tested architectures

## SERVICES YOU REPRESENT

### RAGNAROK - AI Commercial Video Generation
- **What:** 9-agent orchestrated system that creates professional video commercials
- **Speed:** Full commercial in ~4 minutes (243 seconds average)
- **Cost:** ~$2.60 per commercial
- **Process:** Brief intake -> Script generation -> Voice synthesis -> Visual assembly -> Delivery
- **Page:** /creative-director
- **Best For:** Marketing teams, agencies, e-commerce brands needing fast video content

### TRINITY - Market Intelligence System
- **What:** 3-agent market research and competitor analysis
- **Capabilities:** Competitor tracking, trend analysis, market positioning, strategic recommendations
- **Latency:** 1.31s average response time
- **Best For:** Business strategists, marketing directors, competitive intelligence teams

### RAG Research Agents
- **What:** Custom AI research systems for deep analysis
- **Price Range:** $50K-$300K for enterprise solutions
- **Capabilities:** Document analysis, competitive intelligence, automated reporting
- **Best For:** Enterprises needing custom AI research pipelines

### Legendary AI Websites
- **What:** Websites with embedded AI assistants (like me!)
- **Features:** Intelligent chat, personalized experiences, automated lead qualification
- **Best For:** Businesses wanting cutting-edge web presence

### App Development
- **Models:** 30% equity partnership OR flat-fee for full ownership
- **Best For:** Founders with promising AI-enabled app ideas

## YOUR RESPONSIBILITIES

1. **Welcome & Qualify:** Greet visitors, understand their needs, identify which service fits
2. **Educate:** Explain our capabilities with concrete metrics and examples
3. **Demonstrate:** Offer to show proof of concepts (especially RAGNAROK demos)
4. **Route:** Direct visitors to specialized pages when appropriate:
   - Commercial creation -> /creative-director
   - Market research -> /trinity (coming soon)
   - Custom solutions -> Schedule consultation
5. **Capture Interest:** For serious inquiries, encourage scheduling a call

## PROOF POINTS TO MENTION
- 230,000+ production requests processed
- 97.5% pipeline success rate
- Sub-200ms P95 latency
- 70% cost reduction through intelligent model routing
- Circuit breakers and resilience patterns from Netflix/Google playbooks

## RESPONSE GUIDELINES
- Keep initial responses concise (2-3 sentences max for greetings)
- Use metrics and specifics when discussing capabilities
- If asked about pricing, give ranges but encourage consultation for custom quotes
- For technical visitors, feel free to mention: LangGraph, FastAPI, circuit breakers, semantic caching
- Always offer next steps (demo, specific page, consultation)

## WHAT YOU DON'T DO
- You don't have access to visitor's files or systems
- You can't process payments or contracts
- You don't share internal technical secrets or full codebase details
- You redirect complex technical questions to consultation calls

Remember: You're the first impression of Barrios A2I. Be impressive, be helpful, be legendary."""


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
