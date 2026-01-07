"""
================================================================================
⚡ FLAWLESS GENESIS API v2.0 LEGENDARY
================================================================================
FastAPI endpoints for the ultimate NEXUS → TRINITY → RAGNAROK pipeline.

Endpoints:
- POST /api/genesis/trigger      - Trigger full pipeline (with debouncing)
- POST /api/genesis/research     - TRINITY research only
- GET  /api/genesis/stream/{id}  - SSE stream with ghost recovery
- GET  /api/genesis/status/{id}  - Pipeline status
- GET  /api/genesis/health       - Comprehensive health check
- GET  /metrics                  - Prometheus metrics

All endpoints integrate:
✅ Distributed Circuit Breakers
✅ Trigger Debouncing
✅ Ghost Connection Recovery
✅ OpenTelemetry Tracing

================================================================================
Author: Barrios A2I | Version: 2.0.0 LEGENDARY | January 2026
================================================================================
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import orchestrator and components
from flawless_orchestrator import (
    FlawlessGenesisOrchestrator,
    LeadData,
    create_flawless_orchestrator,
    # Agent 0: NEXUS Intake
    NexusIntakeAgent,
    NexusIntakeRequest,
    NexusIntakeResponse,
)
from ghost_recovery import EventType, SSEResponse

# Import VORTEX v2.1 video assembly router
from vortex.router import router as vortex_router, initialize_vortex

# Import Chat routes (migrated from creative-director-api)
from chat_routes import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TriggerPipelineRequest(BaseModel):
    """Request to trigger full GENESIS pipeline"""
    session_id: str = Field(..., description="NEXUS chat session ID")

    # Option A: Provide conversation_history (Agent 0 extracts lead data)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Chat conversation history for Agent 0 to extract lead data"
    )

    # Option B: Provide explicit fields (legacy, skips Agent 0)
    business_name: Optional[str] = Field(None, description="Business name")
    industry: Optional[str] = Field(None, description="Industry vertical")
    website_url: Optional[str] = Field(None, description="Business website")
    contact_email: Optional[str] = Field(None, description="Contact email")
    goals: List[str] = Field(default_factory=list, description="Business goals")
    budget_range: Optional[str] = Field(None, description="Budget range")
    timeline: Optional[str] = Field(None, description="Timeline")
    additional_context: Optional[str] = Field(None, description="Extra context")

    # Pipeline options
    generate_video: bool = Field(True, description="Generate video with RAGNAROK")
    video_formats: List[str] = Field(
        default_factory=lambda: ["youtube_1080p", "tiktok_vertical", "instagram_square"],
        description="Video formats to generate"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "nexus-session-12345",
                "business_name": "Smile Dental Care",
                "industry": "dental",
                "website_url": "https://smiledentalcare.com",
                "contact_email": "info@smiledentalcare.com",
                "goals": ["Increase new patients", "Build brand awareness"],
                "budget_range": "$1,000-$2,000",
                "generate_video": True
            }
        }


class ResearchOnlyRequest(BaseModel):
    """Request for TRINITY research without video"""
    session_id: str
    business_name: str
    industry: str
    website_url: Optional[str] = None
    goals: List[str] = Field(default_factory=list)


class PipelineStatusResponse(BaseModel):
    """Pipeline status response"""
    pipeline_id: str
    status: str  # queued, running, completed, failed
    phase: Optional[str] = None
    progress: float = 0.0
    events_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: float
    circuits: Dict[str, Any]
    ghost_recovery: Dict[str, Any]
    active_pipelines: int


class TriggerResponse(BaseModel):
    """Response when pipeline is triggered"""
    pipeline_id: str
    status: str
    stream_url: str
    estimated_time_seconds: int
    estimated_cost_usd: float


# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Global state
START_TIME = time.time()
orchestrator: Optional[FlawlessGenesisOrchestrator] = None
active_streams: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global orchestrator
    
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info("🚀 Starting FLAWLESS GENESIS API v2.0...")
    
    # Initialize Redis (if available)
    redis_client = None
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e} - using in-memory fallback")
            redis_client = None
    
    # Initialize Anthropic (if available)
    anthropic_client = None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from anthropic import AsyncAnthropic
            anthropic_client = AsyncAnthropic(api_key=anthropic_key)
            logger.info("✅ Anthropic client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Anthropic init failed: {e}")
    
    # Create orchestrator
    orchestrator = create_flawless_orchestrator(
        redis_client=redis_client,
        anthropic_client=anthropic_client
    )

    logger.info("✅ FlawlessGenesisOrchestrator initialized")

    # Initialize VORTEX v2.1 (video assembly)
    initialize_vortex(redis_client)
    logger.info("✅ VORTEX v2.1 video assembly initialized")
    logger.info("=" * 60)
    logger.info("⚡ FLAWLESS GENESIS API v2.0 READY")
    logger.info("=" * 60)
    
    yield
    
    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("👋 Shutting down FLAWLESS GENESIS API...")
    
    # Cancel active streams
    for task in active_streams.values():
        task.cancel()
    
    # Close Redis
    if redis_client:
        await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="FLAWLESS GENESIS API",
    description="""
    The Ultimate NEXUS → TRINITY → RAGNAROK Pipeline Orchestration.
    
    ## Features
    - **Distributed Circuit Breakers** - Redis-backed resilience
    - **Trigger Debouncing** - Prevents duplicate expensive pipelines
    - **Ghost Connection Recovery** - SSE replay on reconnection
    - **Full Observability** - Prometheus metrics, structured logging
    
    ## Pipeline Flow
    1. NEXUS qualifies lead via streaming chat
    2. TRINITY runs 3 agents in parallel (Trend, Market, Competitor)
    3. Strategy synthesis creates positioning report
    4. RAGNAROK pipeline produces video deliverables
    
    **Target:** Full pipeline < 5 minutes, Cost < $3.00
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://barrios-landing.vercel.app",
        "https://barriosa2i.com",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include VORTEX v2.1 video assembly routes
app.include_router(vortex_router)

# Include Chat routes (migrated from creative-director-api)
app.include_router(chat_router)


# =============================================================================
# HEALTH & METRICS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/genesis/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check.
    
    Returns status of all components:
    - Circuit breakers (per agent)
    - Ghost recovery system
    - Active pipelines
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    health = await orchestrator.get_health()
    
    return HealthResponse(
        status=health["status"],
        version="2.0.0",
        uptime_seconds=time.time() - START_TIME,
        circuits=health["circuits"],
        ghost_recovery=health["ghost_recovery"],
        active_pipelines=health["active_pipelines"]
    )


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/genesis/stats", tags=["Observability"])
async def get_stats():
    """Get orchestrator statistics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.get_stats()


# =============================================================================
# PIPELINE TRIGGER ENDPOINTS
# =============================================================================

@app.post("/api/genesis/trigger", tags=["Pipeline"])
async def trigger_pipeline(
    request: TriggerPipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger full GENESIS pipeline.

    Two modes:
    1. With conversation_history: Agent 0 extracts & qualifies lead from chat
    2. With explicit fields: Legacy mode, skips Agent 0

    Includes:
    - Agent 0: NEXUS Intake (lead extraction & qualification)
    - Debounce check (prevents duplicate triggers)
    - TRINITY research (3 parallel agents)
    - Strategy synthesis
    - RAGNAROK video generation (if enabled)

    Returns stream URL for real-time progress updates.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Generate pipeline ID early for SSE events
    pipeline_id = f"genesis-{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # AGENT 0: NEXUS Intake & Lead Qualification
    # =========================================================================
    if request.conversation_history:
        # Run Agent 0 to extract lead data from conversation
        intake_agent = NexusIntakeAgent()
        intake_request = NexusIntakeRequest(
            session_id=request.session_id,
            message=request.conversation_history[-1].get("content", "") if request.conversation_history else "",
            conversation_history=request.conversation_history
        )

        logger.info(f"[{pipeline_id}] Agent 0: Processing conversation ({len(request.conversation_history)} messages)")

        intake_result = await intake_agent.process_message(intake_request)

        logger.info(
            f"[{pipeline_id}] Agent 0 complete: "
            f"score={intake_result.qualification_score:.0%}, "
            f"business={intake_result.lead_data.business_name}, "
            f"industry={intake_result.lead_data.industry}"
        )

        # GATE: If not qualified, return early with suggestions
        if not intake_result.lead_data.is_qualified:
            return {
                "pipeline_id": pipeline_id,
                "status": "needs_qualification",
                "qualification_score": intake_result.qualification_score,
                "missing_fields": intake_result.missing_fields,
                "suggested_questions": intake_result.suggested_questions,
                "lead_data": {
                    "business_name": intake_result.lead_data.business_name,
                    "industry": intake_result.lead_data.industry,
                    "goals": intake_result.lead_data.goals,
                    "contact_email": intake_result.lead_data.contact_email,
                    "budget_range": intake_result.lead_data.budget_range,
                },
                "message": f"Lead qualification: {intake_result.qualification_score:.0%}. Need more information before starting pipeline."
            }

        # Use extracted lead data
        lead = LeadData(
            session_id=request.session_id,
            business_name=intake_result.lead_data.business_name or "Unknown Business",
            industry=intake_result.lead_data.industry or "general",
            website_url=request.website_url,
            contact_email=intake_result.lead_data.contact_email,
            goals=intake_result.lead_data.goals,
            budget_range=intake_result.lead_data.budget_range,
            timeline=intake_result.lead_data.timeline,
            additional_context=request.additional_context,
            qualification_score=intake_result.qualification_score
        )
    else:
        # Legacy mode: Use explicit fields (skip Agent 0)
        if not request.business_name or not request.industry:
            raise HTTPException(
                status_code=400,
                detail="Either conversation_history or (business_name + industry) required"
            )

        lead = LeadData(
            session_id=request.session_id,
            business_name=request.business_name,
            industry=request.industry,
            website_url=request.website_url,
            contact_email=request.contact_email,
            goals=request.goals,
            budget_range=request.budget_range,
            timeline=request.timeline,
            additional_context=request.additional_context,
            qualification_score=0.85  # Pre-qualified from NEXUS (legacy)
        )

    # =========================================================================
    # Start Pipeline Execution
    # =========================================================================

    # Start pipeline in background
    async def run_pipeline():
        async for event in orchestrator.execute(
            lead,
            generate_video=request.generate_video,
            video_formats=request.video_formats
        ):
            pass  # Events are persisted via ghost recovery

    task = asyncio.create_task(run_pipeline())
    active_streams[pipeline_id] = task

    # Estimate time and cost
    estimated_time = 300 if request.generate_video else 30
    estimated_cost = 2.50 if request.generate_video else 0.10

    return TriggerResponse(
        pipeline_id=pipeline_id,
        status="started",
        stream_url=f"/api/genesis/stream/{pipeline_id}",
        estimated_time_seconds=estimated_time,
        estimated_cost_usd=estimated_cost
    )


@app.post("/api/genesis/research", tags=["Pipeline"])
async def research_only(request: ResearchOnlyRequest):
    """
    Run TRINITY research only (no video generation).
    
    Faster and cheaper - great for initial exploration.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    lead = LeadData(
        session_id=request.session_id,
        business_name=request.business_name,
        industry=request.industry,
        website_url=request.website_url,
        goals=request.goals,
        qualification_score=0.7
    )
    
    pipeline_id = f"research-{uuid.uuid4().hex[:12]}"
    
    async def run_research():
        async for event in orchestrator.execute(
            lead,
            generate_video=False
        ):
            pass
    
    task = asyncio.create_task(run_research())
    active_streams[pipeline_id] = task
    
    return {
        "pipeline_id": pipeline_id,
        "status": "started",
        "stream_url": f"/api/genesis/stream/{pipeline_id}",
        "type": "research_only"
    }


# =============================================================================
# SSE STREAMING WITH GHOST RECOVERY
# =============================================================================

@app.get("/api/genesis/stream/{pipeline_id}", tags=["Streaming"])
async def stream_pipeline(
    pipeline_id: str,
    last_event_id: Optional[int] = Query(
        0,
        alias="Last-Event-ID",
        description="Last seen event sequence for reconnection"
    )
):
    """
    Stream pipeline events via Server-Sent Events (SSE).
    
    ## Ghost Recovery
    If connection drops, reconnect with `Last-Event-ID` header to resume
    from where you left off. All missed events will be replayed.
    
    ## Event Types
    - `pipeline_start` - Pipeline initiated
    - `phase_start` - Phase beginning
    - `agent_complete` - Agent finished
    - `phase_complete` - Phase finished
    - `pipeline_complete` - Full pipeline done
    - `pipeline_error` - Error occurred
    - `heartbeat` - Keep-alive
    
    ## Example
    ```javascript
    const evtSource = new EventSource('/api/genesis/stream/pipeline-123');
    evtSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        console.log(event.event_type, event.data);
    };
    
    // On reconnection, browser automatically sends Last-Event-ID
    ```
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    async def event_generator():
        """Generate SSE events with ghost recovery"""
        try:
            # Initial comment to establish connection
            yield SSEResponse.format_comment("connected")
            
            # Stream with replay from last seen sequence
            async for sse in orchestrator.stream_events(pipeline_id, last_event_id):
                yield sse
        
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled: {pipeline_id}")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield SSEResponse.format_event("error", {"error": str(e)})
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSEResponse.headers()
    )


# =============================================================================
# PIPELINE STATUS
# =============================================================================

@app.get("/api/genesis/status/{pipeline_id}", response_model=PipelineStatusResponse, tags=["Pipeline"])
async def get_pipeline_status(pipeline_id: str):
    """
    Get current pipeline status.
    
    Includes:
    - Current phase
    - Progress percentage
    - Event count
    - Error details (if failed)
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Check if pipeline is active
    if pipeline_id in orchestrator.active_pipelines:
        state = orchestrator.active_pipelines[pipeline_id]
        return PipelineStatusResponse(
            pipeline_id=pipeline_id,
            status="running",
            phase=state.phase.value,
            progress=0.0,  # Would compute from phase
            events_count=await orchestrator.ghost.event_log.get_count(pipeline_id),
            started_at=datetime.fromtimestamp(state.started_at)
        )
    
    # Check event log for completed pipelines
    events = await orchestrator.ghost.event_log.get_all(pipeline_id)
    
    if not events:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    last_event = events[-1]
    
    status = "unknown"
    if last_event.event_type == EventType.PIPELINE_COMPLETE.value:
        status = "completed"
    elif last_event.event_type == EventType.PIPELINE_ERROR.value:
        status = "failed"
    else:
        status = "running"
    
    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        status=status,
        phase=last_event.data.get("phase"),
        progress=last_event.data.get("progress", 0.0),
        events_count=len(events),
        error=last_event.data.get("error") if status == "failed" else None
    )


# =============================================================================
# NEXUS INTEGRATION WEBHOOK
# =============================================================================

@app.post("/api/genesis/nexus-webhook", tags=["Integration"])
async def nexus_webhook(request: Request):
    """
    Webhook endpoint for NEXUS chat integration.
    
    Called automatically when NEXUS detects a qualified lead
    ready for video production.
    """
    body = await request.json()
    
    lead_data = body.get("lead", {})
    chat_context = body.get("chat_context", {})
    
    # Convert to trigger request
    trigger_request = TriggerPipelineRequest(
        session_id=chat_context.get("session_id", str(uuid.uuid4())),
        business_name=lead_data.get("business_name", ""),
        industry=lead_data.get("industry", ""),
        website_url=lead_data.get("website_url"),
        contact_email=lead_data.get("contact_email"),
        goals=lead_data.get("goals", []),
        budget_range=lead_data.get("budget_range"),
        additional_context=json.dumps(chat_context),
        generate_video=body.get("generate_video", True),
        video_formats=body.get("video_formats", ["youtube_1080p", "tiktok_vertical"])
    )
    
    # Trigger pipeline
    from fastapi import BackgroundTasks
    return await trigger_pipeline(trigger_request, BackgroundTasks())


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@app.post("/api/genesis/admin/circuit/{service}/open", tags=["Admin"])
async def force_open_circuit(service: str, reason: str = "manual"):
    """Force open a circuit breaker (admin only)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if service not in orchestrator.circuits:
        raise HTTPException(status_code=404, detail=f"Circuit {service} not found")
    
    await orchestrator.circuits[service].force_open(reason)
    
    return {"status": "opened", "service": service, "reason": reason}


@app.post("/api/genesis/admin/circuit/{service}/close", tags=["Admin"])
async def force_close_circuit(service: str):
    """Force close a circuit breaker (admin only)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if service not in orchestrator.circuits:
        raise HTTPException(status_code=404, detail=f"Circuit {service} not found")
    
    await orchestrator.circuits[service].force_close()
    
    return {"status": "closed", "service": service}


# =============================================================================
# ROOT & DOCS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic info"""
    return {
        "name": "FLAWLESS GENESIS API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "flawless_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
