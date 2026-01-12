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

# Import Legendary Agents (7.5-15) - THE APEX TIER
from legendary_agents import (
    create_legendary_coordinator,
    LegendaryCoordinator,
    TheAuteur,
    TheGeneticist,
    TheOracle,
    TheChameleon,
    TheMemory,
    TheHunter,
    TheAccountant,
)

# Import THE CURATOR (Agent 16) - Autonomous Commercial Intelligence
from commercial_curator import (
    TheCurator, CuratorConfig, Platform, PatternType,
    create_curator, ExtractedPattern, TrendSignal
)

# Import NEXUS BRIDGE (Commercial_Lab Production Pipeline)
from nexus_bridge import (
    NexusBridge, ProductionPhase, ProductionStatus, ProductionState,
    create_nexus_bridge
)

# Import NEXUS Brain concierge router (Landing Page AI)
from nexus_router import router as nexus_brain_router

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
legendary_coordinator: Optional[LegendaryCoordinator] = None
curator: Optional[TheCurator] = None
nexus_bridge: Optional[NexusBridge] = None
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

    # Check FFmpeg availability for Agent 7 (Video Assembly)
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            ffmpeg_version = result.stdout.decode().split('\n')[0]
            logger.info(f"✅ FFmpeg available: {ffmpeg_version[:50]}")
        else:
            logger.warning("⚠️ FFmpeg not available - video assembly will use mock output")
    except FileNotFoundError:
        logger.warning("⚠️ FFmpeg not found in PATH - video assembly will use mock output")
    except Exception as e:
        logger.warning(f"⚠️ FFmpeg check failed: {e}")

    # Initialize Legendary Coordinator (Agents 7.5-15)
    global legendary_coordinator
    try:
        # Initialize Qdrant client for client DNA storage
        qdrant_client = None
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY")
            )
            logger.info("✅ Qdrant client initialized for Legendary Agents")

        legendary_coordinator = create_legendary_coordinator(
            anthropic_client=anthropic_client,
            qdrant_client=qdrant_client,
            trend_api_key=os.getenv("TREND_API_KEY")
        )
        logger.info("✅ Legendary Coordinator initialized (Agents 7.5-15)")
    except Exception as e:
        logger.warning(f"⚠️ Legendary Coordinator init failed: {e} - using mock fallback")
        legendary_coordinator = create_legendary_coordinator()

    # Initialize THE CURATOR (Agent 16) - Autonomous Commercial Intelligence
    global curator
    try:
        curator_config = CuratorConfig(
            discovery_interval_hours=6,
            consolidation_hour=2,
            cleanup_day=0,
            max_ads_per_cycle=100,
            industries=[
                "technology", "healthcare", "finance", "retail",
                "automotive", "travel", "food", "entertainment",
                "real_estate", "education", "fitness", "beauty"
            ]
        )

        curator = create_curator(
            llm_client=None,  # Uses mock for now
            qdrant_client=qdrant_client,
            meta_token=os.getenv("META_AD_LIBRARY_TOKEN"),
            tiktok_token=os.getenv("TIKTOK_CREATIVE_TOKEN"),
            config=curator_config
        )
        logger.info("✅ THE CURATOR initialized (Agent 16)")
    except Exception as e:
        logger.warning(f"⚠️ Curator init failed: {e} - using mock")
        curator = create_curator()

    # Initialize NEXUS BRIDGE (Commercial_Lab Production Pipeline)
    global nexus_bridge
    try:
        nexus_bridge = create_nexus_bridge(
            redis_client=redis_client,
            curator=curator
        )
        logger.info("✅ NEXUS BRIDGE initialized (Commercial_Lab Pipeline)")
    except Exception as e:
        logger.warning(f"⚠️ Nexus Bridge init failed: {e} - using default")
        nexus_bridge = create_nexus_bridge()

    logger.info("=" * 60)
    logger.info("⚡ FLAWLESS GENESIS API v2.0 READY")
    logger.info("⚡ 24 AGENTS + COMMERCIAL_LAB PIPELINE ACTIVATED")
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
        "https://www.barriosa2i.com",
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

# Include Commercial Review routes (publish workflow)
from commercial_review_routes import router as review_router
app.include_router(review_router)

# Include NEXUS Brain concierge routes (Landing Page AI)
app.include_router(nexus_brain_router)

# Include Production Status routes (Phase 2 - SSE streaming)
from production_routes import router as production_router
app.include_router(production_router)

# Include Post-Production routes (existing scenes through RAGNAROK pipeline)
try:
    from api.postprod_routes import router as postprod_router
    app.include_router(postprod_router)
    logger.info("Post-production routes loaded")
except ImportError as e:
    logger.warning(f"Post-production routes not available: {e}")


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


@app.get("/api/debug/agents", tags=["Debug"])
async def debug_agents():
    """Debug endpoint to check agent configuration status."""
    import os

    kie_key = os.getenv("KIE_API_KEY")

    try:
        # nexus_bridge is the production pipeline (used by /api/production/start)
        video_available = bool(nexus_bridge and nexus_bridge.video_agent)
        video_configured = False
        video_api_key_set = False
        if video_available:
            video_configured = getattr(nexus_bridge.video_agent, 'is_configured', False)
            video_api_key_set = bool(getattr(nexus_bridge.video_agent, 'api_key', None))

        return {
            "video_agent": {
                "available": video_available,
                "configured": video_configured,
                "api_key_loaded": video_api_key_set,
                "env_kie_api_key_present": bool(kie_key),
                "env_kie_api_key_prefix": (kie_key[:10] + "...") if kie_key and len(kie_key) >= 10 else kie_key
            },
            "auteur": {
                "available": bool(nexus_bridge and nexus_bridge.auteur)
            },
            "intake": {
                "available": bool(nexus_bridge and nexus_bridge.intake_agent)
            },
            "nexus_bridge_initialized": bool(nexus_bridge)
        }
    except Exception as e:
        return {"error": str(e)}


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


@app.post("/api/admin/refresh-website-knowledge", tags=["Admin"])
async def refresh_website_knowledge():
    """
    Re-index barriosa2i.com website content into RAG system.

    This crawls all website pages, chunks content, embeds with OpenAI,
    and stores in Qdrant for real-time knowledge retrieval.

    Use this after website content updates to keep AI knowledge current.
    """
    try:
        from knowledge.website_rag import get_website_rag

        rag = get_website_rag()
        result = await rag.index_website()

        logger.info(f"Website knowledge re-indexed: {result}")

        return {
            "status": "success",
            "result": result,
            "message": "Website knowledge updated successfully"
        }
    except Exception as e:
        logger.error(f"Website knowledge refresh failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh website knowledge: {str(e)}"
        )


@app.get("/api/admin/website-knowledge-stats", tags=["Admin"])
async def get_website_knowledge_stats():
    """Get statistics about the website knowledge RAG system."""
    try:
        from knowledge.website_rag import get_website_rag

        rag = get_website_rag()
        stats = rag.get_stats()

        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get website knowledge stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/api/admin/backfill-videos", tags=["Admin"])
async def backfill_videos_to_preview(
    execute: bool = Query(False, description="Set to true to actually upload"),
    limit: int = Query(0, description="Limit number of videos (0 = all)")
):
    """
    Backfill historical videos from R2 to video-preview gallery.

    This scans the R2 bucket for all production videos and sends them
    to the video-preview gallery API.

    Use execute=false (default) for dry run.
    """
    try:
        from storage.r2_storage import get_video_storage
        import httpx

        VIDEO_PREVIEW_API = "https://video-preview-theta.vercel.app/api/videos"

        storage = get_video_storage()
        if not storage.is_configured:
            return {
                "status": "error",
                "error": "R2 storage not configured",
                "message": "Missing R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, or R2_SECRET_ACCESS_KEY"
            }

        # List all objects in R2 bucket
        import boto3
        from botocore.config import Config

        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("R2_BUCKET_NAME", "barrios-videos")
        public_url = os.getenv("R2_PUBLIC_URL", "https://videos.barriosa2i.com")

        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4")
        )

        # Scan for videos
        sessions = {}
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket_name, Prefix="productions/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                if len(parts) >= 3:
                    session_id = parts[1]
                    filename = parts[2]

                    if session_id not in sessions:
                        sessions[session_id] = {
                            "videos": [],
                            "thumbnail": None,
                            "created_at": obj["LastModified"].isoformat()
                        }

                    if filename.endswith(".mp4"):
                        sessions[session_id]["videos"].append({
                            "format": filename.replace(".mp4", ""),
                            "url": f"{public_url}/{key}",
                            "size": obj["Size"]
                        })
                    elif filename.endswith((".jpg", ".png")):
                        sessions[session_id]["thumbnail"] = f"{public_url}/{key}"

        # Build video list (prefer youtube_1080p or 1080p formats)
        videos = []
        for session_id, data in sessions.items():
            if data["videos"]:
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
                    "formats_available": [v["format"] for v in data["videos"]]
                })

        # Sort by created_at (newest first)
        videos.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply limit
        if limit > 0:
            videos = videos[:limit]

        if not execute:
            return {
                "status": "dry_run",
                "total_videos": len(videos),
                "videos": videos[:50],  # Show first 50 in response
                "message": f"Found {len(videos)} videos. Use execute=true to upload.",
                "execute_url": f"/api/admin/backfill-videos?execute=true&limit={limit if limit else ''}"
            }

        # Execute upload
        results = {"created": 0, "updated": 0, "failed": 0, "details": []}

        async with httpx.AsyncClient(timeout=30) as http_client:
            for video in videos:
                video_id = f"genesis_{video['session_id']}"
                payload = {
                    "id": video_id,
                    "url": video["video_url"],
                    "title": f"AI Commercial - {video['format']}",
                    "description": f"64-second AI commercial. Format: {video['format']}. Size: {video['size_mb']}MB",
                    "thumbnail": video.get("thumbnail"),
                    "duration": "1:04",
                    "tags": ["commercial", "ai-generated", "barrios-a2i", "backfill"],
                    "created": video["created_at"].split("T")[0]
                }

                try:
                    response = await http_client.post(VIDEO_PREVIEW_API, json=payload)
                    if response.status_code in [200, 201]:
                        result = response.json()
                        action = result.get("action", "created")
                        if action == "updated":
                            results["updated"] += 1
                        else:
                            results["created"] += 1
                        results["details"].append({
                            "session_id": video["session_id"],
                            "action": action,
                            "preview_url": f"https://video-preview-theta.vercel.app?v={video_id}"
                        })
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "session_id": video["session_id"],
                            "error": f"HTTP {response.status_code}"
                        })
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append({
                        "session_id": video["session_id"],
                        "error": str(e)
                    })

                await asyncio.sleep(0.5)  # Rate limit

        return {
            "status": "complete",
            "created": results["created"],
            "updated": results["updated"],
            "failed": results["failed"],
            "total_processed": len(videos),
            "details": results["details"][:50],  # Limit details in response
            "gallery_url": "https://video-preview-theta.vercel.app"
        }

    except Exception as e:
        logger.error(f"Video backfill failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to backfill videos: {str(e)}"
        )


@app.get("/api/admin/r2-contents", tags=["Admin"])
async def list_r2_contents(
    prefix: str = Query("", description="Prefix/folder to list"),
    limit: int = Query(100, description="Max objects to return")
):
    """
    List contents of R2 bucket for debugging.

    Use this to inspect what's actually stored in R2.
    """
    try:
        import boto3
        from botocore.config import Config

        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("R2_BUCKET_NAME", "barrios-videos")
        public_url = os.getenv("R2_PUBLIC_URL", "https://videos.barriosa2i.com")

        if not all([account_id, access_key_id, secret_access_key]):
            return {
                "error": "R2 not configured",
                "configured": False,
                "missing": [
                    k for k, v in {
                        "R2_ACCOUNT_ID": account_id,
                        "R2_ACCESS_KEY_ID": access_key_id,
                        "R2_SECRET_ACCESS_KEY": secret_access_key
                    }.items() if not v
                ]
            }

        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4")
        )

        # List prefixes (folders) at this level
        prefixes_response = client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )

        # List objects
        objects_response = client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=limit
        )

        prefixes = [p['Prefix'] for p in prefixes_response.get('CommonPrefixes', [])]
        objects = [
            {
                "key": obj['Key'],
                "size_mb": round(obj['Size'] / (1024 * 1024), 2),
                "size_bytes": obj['Size'],
                "modified": obj['LastModified'].isoformat(),
                "url": f"{public_url}/{obj['Key']}"
            }
            for obj in objects_response.get('Contents', [])
        ]

        return {
            "configured": True,
            "bucket": bucket_name,
            "prefix": prefix,
            "prefixes": prefixes,
            "objects": objects,
            "total_objects": len(objects),
            "public_url_base": public_url
        }

    except Exception as e:
        logger.error(f"R2 listing failed: {e}")
        return {"error": str(e), "configured": True}


@app.get("/api/admin/redis-sessions", tags=["Admin"])
async def list_redis_sessions(
    pattern: str = Query("*", description="Redis key pattern to search"),
    limit: int = Query(50, description="Max keys to return")
):
    """
    List Redis keys for debugging production/session history.

    Patterns:
    - commercial:review:* - Commercial review records
    - production:* - Production status records
    - session:* - Chat sessions
    """
    try:
        import redis.asyncio as redis_async

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return {"error": "REDIS_URL not configured", "configured": False}

        client = redis_async.from_url(redis_url, decode_responses=True)

        # Scan for keys matching pattern
        keys = []
        async for key in client.scan_iter(match=pattern, count=100):
            if len(keys) >= limit:
                break
            keys.append(key)

        # Get data for each key
        results = []
        for key in keys:
            try:
                key_type = await client.type(key)
                ttl = await client.ttl(key)

                item = {
                    "key": key,
                    "type": key_type,
                    "ttl_seconds": ttl if ttl > 0 else "no expiry"
                }

                # Get value preview for string keys
                if key_type == "string":
                    value = await client.get(key)
                    if value:
                        try:
                            data = json.loads(value)
                            item["preview"] = {
                                "session_id": data.get("session_id"),
                                "video_url": data.get("video_url"),
                                "status": data.get("status"),
                                "created_at": data.get("created_at"),
                                "business_name": data.get("business_name")
                            }
                        except:
                            item["preview"] = value[:200] if len(value) > 200 else value

                results.append(item)
            except Exception as e:
                results.append({"key": key, "error": str(e)})

        await client.close()

        return {
            "pattern": pattern,
            "total_found": len(results),
            "keys": results
        }

    except Exception as e:
        logger.error(f"Redis listing failed: {e}")
        return {"error": str(e)}


@app.post("/api/admin/seed-commercials", tags=["Admin"])
async def seed_commercial_training_data(
    force_recreate: bool = Query(False, description="Force recreate collection"),
    verify: bool = Query(True, description="Run verification query after seeding")
):
    """
    Initialize and seed the commercial_styles Qdrant collection.

    This populates the commercial training data used by RAGNAROK to generate
    cinematic B-roll video prompts instead of generic talking-head videos.

    Seed data includes 5 high-quality commercial examples:
    - Apple (technology) - Product hero, dramatic lighting
    - Nike (sports) - Fast cuts, slow-mo impact
    - Tesla (automotive) - Environment + vehicle beauty
    - Airbnb (travel) - Spaces not faces, warm color grading
    - Barrios A2I (ai_automation) - Abstract visualizations
    """
    import sys
    from pathlib import Path

    # Add tools to path
    sys.path.insert(0, str(Path(__file__).parent / "tools"))

    try:
        from tools.init_qdrant_commercials import init_qdrant_collection
        from tools.ingest_commercials import CommercialIngester
        import json

        # Step 1: Initialize collection
        logger.info("Initializing Qdrant commercial_styles collection...")
        client = init_qdrant_collection(force_recreate=force_recreate)

        # Step 2: Load seed data
        seed_file = Path(__file__).parent / "data" / "seed_commercials.json"
        if not seed_file.exists():
            return {"error": f"Seed file not found: {seed_file}", "success": False}

        with open(seed_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        commercials = data.get("commercials", [])
        logger.info(f"Found {len(commercials)} commercial examples to seed")

        # Step 3: Ingest commercials
        ingester = CommercialIngester()
        results = []
        success_count = 0

        for commercial in commercials:
            try:
                point_id = await ingester.ingest_commercial(**commercial)
                results.append({
                    "brand": commercial.get("brand"),
                    "industry": commercial.get("manual_data", {}).get("title", "Unknown"),
                    "status": "success",
                    "point_id": point_id
                })
                success_count += 1
            except Exception as e:
                results.append({
                    "brand": commercial.get("brand"),
                    "status": "error",
                    "error": str(e)
                })

        # Step 4: Verify
        collection_info = client.get_collection("commercial_styles")
        verification = {
            "collection": "commercial_styles",
            "points_count": collection_info.points_count,
            "vectors_config": str(collection_info.config.params.vectors)
        }

        # Optional: Run test query
        if verify and success_count > 0:
            try:
                test_results = ingester.query_similar(
                    query="cinematic technology commercial with dramatic lighting",
                    min_quality=8.0,
                    top_k=3
                )
                verification["test_query_results"] = len(test_results)
                verification["test_query_brands"] = [r.get("brand") for r in test_results]
            except Exception as e:
                verification["test_query_error"] = str(e)

        logger.info(f"Commercial seeding complete: {success_count}/{len(commercials)} succeeded")

        return {
            "success": True,
            "seeded": success_count,
            "failed": len(commercials) - success_count,
            "results": results,
            "verification": verification
        }

    except Exception as e:
        logger.error(f"Commercial seeding failed: {e}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# =============================================================================
# LEGENDARY AGENTS ENDPOINTS (7.5-15) - THE APEX TIER
# =============================================================================

class LegendaryEnhanceRequest(BaseModel):
    """Request for full pre/post production enhancement"""
    session_id: str
    brief: Dict[str, Any] = Field(..., description="Creative brief data")
    client_id: Optional[str] = Field(None, description="Client ID for DNA profiling")
    video_frames: Optional[List[str]] = Field(None, description="Base64 video frame samples for post-production")
    target_platforms: List[str] = Field(default_factory=lambda: ["youtube", "tiktok", "instagram"])


class ViralPredictionRequest(BaseModel):
    """Request for viral prediction analysis"""
    hook_text: str = Field(..., description="Opening hook text")
    visual_style: str = Field(..., description="Visual style description")
    target_audience: Dict[str, Any] = Field(..., description="Target audience profile")
    platform: str = Field("youtube", description="Target platform")
    industry: str = Field(..., description="Industry vertical")


class PlatformAdaptRequest(BaseModel):
    """Request for platform adaptation"""
    content: Dict[str, Any] = Field(..., description="Original content")
    source_platform: str = Field("youtube", description="Source platform")
    target_platforms: List[str] = Field(..., description="Target platforms to adapt for")


class TrendHuntRequest(BaseModel):
    """Request for trend hunting"""
    industry: str = Field(..., description="Industry to scout")
    keywords: List[str] = Field(default_factory=lambda: ["visual", "audio", "narrative"], description="Keywords/categories to search for")
    lookback_days: int = Field(7, description="Days to look back for trends")


class BudgetOptimizeRequest(BaseModel):
    """Request for budget optimization"""
    total_budget: float = Field(..., description="Total campaign budget in USD")
    platforms: List[str] = Field(..., description="Platforms to distribute across")
    campaign_goal: str = Field("conversions", description="Primary goal: awareness, engagement, conversions")
    duration_days: int = Field(30, description="Campaign duration")
    industry: str = Field(..., description="Industry vertical")


# =============================================================================
# THE CURATOR REQUEST MODELS (Agent 16)
# =============================================================================

class CuratorEnhanceBriefRequest(BaseModel):
    """Request to enhance a creative brief with curator intelligence"""
    brief: Dict[str, Any] = Field(..., description="Creative brief to enhance")
    industry: str = Field(..., description="Industry vertical")


class CuratorTrendsRequest(BaseModel):
    """Request for trending patterns"""
    industry: Optional[str] = Field(None, description="Filter by industry")
    lookback_days: int = Field(14, description="Days to look back")


# =============================================================================
# COMMERCIAL_LAB PRODUCTION REQUEST MODELS
# =============================================================================

class ProductionStartRequest(BaseModel):
    """Request to start Commercial_Lab video production"""
    brief: Dict[str, Any] = Field(..., description="Approved creative brief")
    industry: str = Field(..., description="Industry vertical")
    business_name: str = Field(..., description="Business name")
    style: str = Field("modern", description="Visual style preference")
    goals: List[str] = Field(default_factory=list, description="Business goals")
    target_platforms: List[str] = Field(
        default_factory=lambda: ["youtube", "tiktok", "instagram"],
        description="Target platforms for video delivery"
    )


@app.post("/api/legendary/enhance", tags=["Legendary"])
async def legendary_enhance(request: LegendaryEnhanceRequest):
    """
    Full Legendary Enhancement Pipeline.

    Runs pre-production enhancement (trend scouting, client DNA, budget optimization)
    and optionally post-production enhancement (vision QA, platform adaptation, viral prediction).

    This is the 23-agent APEX tier in action.
    """
    if not legendary_coordinator:
        raise HTTPException(status_code=503, detail="Legendary Coordinator not initialized")

    try:
        # Run pre-production enhancement
        pre_results = await legendary_coordinator.pre_production_enhancement(
            brief=request.brief,
            client_id=request.client_id
        )

        # If video frames provided, run post-production enhancement
        post_results = {}
        if request.video_frames:
            post_results = await legendary_coordinator.post_production_enhancement(
                video_frames=request.video_frames,
                brief=request.brief,
                target_platforms=request.target_platforms
            )

        return {
            "status": "success",
            "session_id": request.session_id,
            "pre_production": pre_results,
            "post_production": post_results if post_results else None,
            "agents_executed": ["7.5", "8.5", "11", "12", "13", "14", "15"]
        }

    except Exception as e:
        logger.error(f"Legendary enhance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/legendary/predict-viral", tags=["Legendary"])
async def predict_viral(request: ViralPredictionRequest):
    """
    THE ORACLE (Agent 11) - Viral Prediction Analysis.

    Predicts viral potential using multi-factor analysis:
    - Emotional resonance scoring
    - Platform-specific virality factors
    - Trend alignment
    - Hook effectiveness

    Returns confidence score (0-100) and detailed breakdown.
    """
    if not legendary_coordinator:
        raise HTTPException(status_code=503, detail="Legendary Coordinator not initialized")

    try:
        # Build content dict from request fields
        content = {
            "hook_text": request.hook_text,
            "visual_style": request.visual_style,
            "industry": request.industry,
        }
        target_platforms = [request.platform] if request.platform else ["tiktok"]

        prediction = await legendary_coordinator.oracle.predict_virality(
            content=content,
            target_platforms=target_platforms,
            target_audience=request.target_audience
        )

        return {
            "status": "success",
            "agent": "THE ORACLE (11)",
            "prediction": {
                "viral_score": prediction.viral_score,
                "confidence": prediction.confidence,
                "momentum": prediction.momentum,
                "peak_timing": prediction.peak_timing,
                "risk_factors": prediction.risk_factors,
                "amplification_tips": prediction.amplification_tips
            }
        }

    except Exception as e:
        logger.error(f"Viral prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/legendary/adapt-platform", tags=["Legendary"])
async def adapt_platform(request: PlatformAdaptRequest):
    """
    THE CHAMELEON (Agent 12) - Platform Adaptation.

    Automatically adapts content for multiple platforms:
    - Aspect ratio adjustments
    - Caption/text overlay optimization
    - Hashtag strategy per platform
    - Posting time recommendations
    - Platform-specific hooks
    """
    if not legendary_coordinator:
        raise HTTPException(status_code=503, detail="Legendary Coordinator not initialized")

    try:
        # adapt_for_platform takes a single platform, so loop through targets
        adaptations = {}
        for platform in request.target_platforms:
            adapt = await legendary_coordinator.chameleon.adapt_for_platform(
                content=request.content,
                target_platform=platform
            )
            adaptations[platform] = {
                "platform": adapt.platform,
                "adapted_content": adapt.adapted_content,
                "format_specs": adapt.format_specs,
                "hashtags": adapt.hashtags,
                "optimal_posting_time": adapt.optimal_posting_time,
                "engagement_prediction": adapt.engagement_prediction
            }

        return {
            "status": "success",
            "agent": "THE CHAMELEON (12)",
            "source_platform": request.source_platform,
            "adaptations": adaptations
        }

    except Exception as e:
        logger.error(f"Platform adaptation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/legendary/hunt-trends", tags=["Legendary"])
async def hunt_trends(request: TrendHuntRequest):
    """
    THE HUNTER (Agent 14) - Trend Scouting.

    Scans multiple data sources for emerging trends:
    - Social media trending topics
    - Industry-specific patterns
    - Visual style trends
    - Audio/music trends
    - Narrative format trends

    Returns actionable trend insights for creative direction.
    """
    if not legendary_coordinator:
        raise HTTPException(status_code=503, detail="Legendary Coordinator not initialized")

    try:
        trend_report = await legendary_coordinator.hunter.hunt_trends(
            industry=request.industry,
            keywords=request.keywords,
            lookback_days=request.lookback_days
        )

        return {
            "status": "success",
            "agent": "THE HUNTER (14)",
            "industry": request.industry,
            "report": {
                "trends": trend_report.trends,
                "emerging": trend_report.emerging,
                "declining": trend_report.declining,
                "opportunities": trend_report.opportunities,
                "risks": trend_report.risks,
                "recommended_topics": trend_report.recommended_topics
            }
        }

    except Exception as e:
        logger.error(f"Trend hunting error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/legendary/optimize-budget", tags=["Legendary"])
async def optimize_budget(request: BudgetOptimizeRequest):
    """
    THE ACCOUNTANT (Agent 15) - Budget Optimization.

    Optimizes campaign budget allocation using:
    - Historical performance data
    - Platform CPM/CPC analysis
    - Audience reach modeling
    - ROI prediction

    Returns optimal budget distribution and expected metrics.
    """
    if not legendary_coordinator:
        raise HTTPException(status_code=503, detail="Legendary Coordinator not initialized")

    try:
        # optimize_budget expects project dict, constraints dict, and goals list
        project = {
            "platforms": request.platforms,
            "duration_days": request.duration_days,
            "industry": request.industry
        }
        constraints = {
            "total_budget": request.total_budget,
            "max_daily": request.total_budget / max(request.duration_days, 1)
        }
        goals = [request.campaign_goal] if isinstance(request.campaign_goal, str) else request.campaign_goal

        budget_plan = await legendary_coordinator.accountant.optimize_budget(
            project=project,
            constraints=constraints,
            goals=goals
        )

        return {
            "status": "success",
            "agent": "THE ACCOUNTANT (15)",
            "total_budget": request.total_budget,
            "optimization": {
                "allocation": budget_plan.allocation,
                "expected_roi": budget_plan.expected_roi,
                "cost_per_result": budget_plan.cost_per_result,
                "savings_opportunities": budget_plan.savings_opportunities,
                "risk_assessment": budget_plan.risk_assessment
            }
        }

    except Exception as e:
        logger.error(f"Budget optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/legendary/status", tags=["Legendary"])
async def legendary_status():
    """Get status of all Legendary Agents (7.5-16)."""
    if not legendary_coordinator:
        return {
            "status": "offline",
            "message": "Legendary Coordinator not initialized"
        }

    return {
        "status": "online",
        "agents": {
            "7.5": {"name": "THE AUTEUR", "role": "Vision-Language QA", "status": "active"},
            "8.5": {"name": "THE GENETICIST", "role": "DSPy Prompt Evolution", "status": "active"},
            "11": {"name": "THE ORACLE", "role": "Viral Prediction", "status": "active"},
            "12": {"name": "THE CHAMELEON", "role": "Platform Adapter", "status": "active"},
            "13": {"name": "THE MEMORY", "role": "Client DNA Profiling", "status": "active"},
            "14": {"name": "THE HUNTER", "role": "Trend Scouting", "status": "active"},
            "15": {"name": "THE ACCOUNTANT", "role": "Budget Optimization", "status": "active"},
            "16": {"name": "THE CURATOR", "role": "Commercial Intelligence", "status": "active" if curator else "initializing"}
        },
        "total_agents": 24,
        "apex_tier": "RAGNAROK v3.0 APEX"
    }


# =============================================================================
# THE CURATOR ENDPOINTS (Agent 16) - Autonomous Commercial Intelligence
# =============================================================================

@app.post("/api/curator/enhance-brief", tags=["Curator"])
async def curator_enhance_brief(request: CuratorEnhanceBriefRequest):
    """
    THE CURATOR (Agent 16) - Enhance Creative Brief.

    Enriches a creative brief with competitive intelligence:
    - Top-performing hooks from similar ads
    - Proven CTA patterns for the industry
    - Trending visual styles
    - Platform-specific optimizations
    """
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        enhanced = await curator.enhance_brief(
            brief=request.brief,
            industry=request.industry
        )

        return {
            "status": "success",
            "agent": "THE CURATOR (16)",
            "enhanced_brief": enhanced
        }
    except Exception as e:
        logger.error(f"Brief enhancement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curator/hooks/{industry}", tags=["Curator"])
async def curator_get_hooks(
    industry: str,
    limit: int = Query(10, ge=1, le=50, description="Number of hooks to return")
):
    """
    THE CURATOR (Agent 16) - Get Top Hooks.

    Returns the highest-performing hook patterns for an industry.
    Hooks are ranked by engagement score and recency.
    """
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        hooks = await curator.pattern_indexer.get_top_patterns(
            pattern_type=PatternType.HOOK,
            industry=industry,
            limit=limit
        )

        return {
            "status": "success",
            "agent": "THE CURATOR (16)",
            "industry": industry,
            "hooks": [
                {
                    "text": h.text,
                    "effectiveness_score": h.effectiveness_score,
                    "platform": h.platform.value if h.platform else "multi",
                    "industry": h.industry
                }
                for h in hooks
            ]
        }
    except Exception as e:
        logger.error(f"Hook retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curator/ctas/{industry}", tags=["Curator"])
async def curator_get_ctas(
    industry: str,
    limit: int = Query(10, ge=1, le=50, description="Number of CTAs to return")
):
    """
    THE CURATOR (Agent 16) - Get Top CTAs.

    Returns proven call-to-action patterns for an industry.
    CTAs are ranked by conversion effectiveness.
    """
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        ctas = await curator.pattern_indexer.get_top_patterns(
            pattern_type=PatternType.CTA,
            industry=industry,
            limit=limit
        )

        return {
            "status": "success",
            "agent": "THE CURATOR (16)",
            "industry": industry,
            "ctas": [
                {
                    "text": c.text,
                    "effectiveness_score": c.effectiveness_score,
                    "platform": c.platform.value if c.platform else "multi",
                    "industry": c.industry
                }
                for c in ctas
            ]
        }
    except Exception as e:
        logger.error(f"CTA retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curator/styles/{industry}", tags=["Curator"])
async def curator_get_styles(
    industry: str,
    limit: int = Query(10, ge=1, le=50, description="Number of styles to return")
):
    """
    THE CURATOR (Agent 16) - Get Trending Visual Styles.

    Returns trending visual styles for an industry.
    Styles include color palettes, typography, and composition patterns.
    """
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        styles = await curator.pattern_indexer.get_top_patterns(
            pattern_type=PatternType.VISUAL_STYLE,
            industry=industry,
            limit=limit
        )

        return {
            "status": "success",
            "agent": "THE CURATOR (16)",
            "industry": industry,
            "styles": [
                {
                    "text": s.text,
                    "effectiveness_score": s.effectiveness_score,
                    "platform": s.platform.value if s.platform else "multi",
                    "industry": s.industry
                }
                for s in styles
            ]
        }
    except Exception as e:
        logger.error(f"Style retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/curator/trends", tags=["Curator"])
async def curator_get_trends(request: CuratorTrendsRequest):
    """
    THE CURATOR (Agent 16) - Get Trending Patterns.

    Detects emerging trends across all platforms:
    - Rising hooks and CTAs
    - Viral visual styles
    - Platform-specific trends
    - Industry momentum signals
    """
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        trends = await curator.trend_detector.detect_trends(
            industry=request.industry,
            lookback_days=request.lookback_days
        )

        return {
            "status": "success",
            "agent": "THE CURATOR (16)",
            "industry": request.industry or "all",
            "lookback_days": request.lookback_days,
            "trends": [
                {
                    "pattern_type": t.pattern_type.value,
                    "momentum_score": t.momentum_score,
                    "growth_rate": t.growth_rate,
                    "platform": t.platform.value if t.platform else "multi",
                    "representative_examples": t.representative_examples[:3]
                }
                for t in trends
            ]
        }
    except Exception as e:
        logger.error(f"Trend detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/curator/status", tags=["Curator"])
async def curator_status():
    """Get THE CURATOR (Agent 16) status and statistics."""
    if not curator:
        return {
            "status": "offline",
            "message": "Curator not initialized"
        }

    return {
        "status": "online",
        "agent": "THE CURATOR (16)",
        "role": "Autonomous Commercial Intelligence",
        "capabilities": [
            "Ad discovery (Meta, TikTok, YouTube)",
            "Pattern extraction (hooks, CTAs, styles)",
            "Trend detection with momentum scoring",
            "Brief enhancement with competitive intel"
        ],
        "stats": {
            "running": curator.is_running,
            "industries_tracked": len(curator.config.industries),
            "discovery_interval_hours": curator.config.discovery_interval_hours
        }
    }


@app.post("/api/curator/start", tags=["Curator"])
async def curator_start():
    """Start THE CURATOR's autonomous discovery cycle."""
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        await curator.start()
        return {
            "status": "success",
            "message": "Curator discovery cycle started",
            "is_running": curator.is_running
        }
    except Exception as e:
        logger.error(f"Curator start error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/curator/stop", tags=["Curator"])
async def curator_stop():
    """Stop THE CURATOR's autonomous discovery cycle."""
    if not curator:
        raise HTTPException(status_code=503, detail="Curator not initialized")

    try:
        await curator.stop()
        return {
            "status": "success",
            "message": "Curator discovery cycle stopped",
            "is_running": curator.is_running
        }
    except Exception as e:
        logger.error(f"Curator stop error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# COMMERCIAL_LAB PRODUCTION ENDPOINTS
# =============================================================================

@app.post("/api/production/start/{session_id}", tags=["Commercial_Lab"])
async def start_production(session_id: str, request: ProductionStartRequest):
    """
    Start Commercial_Lab video production.

    Triggers the full RAGNAROK 8-agent pipeline:
    1. Intelligence (TRINITY + Curator)
    2. Story Creation
    3. Prompt Engineering
    4. Video Generation
    5. Voiceover
    6. Assembly
    7. QA

    Returns SSE stream of status updates for real-time UI feedback.
    """
    if not nexus_bridge:
        raise HTTPException(status_code=503, detail="Production pipeline not initialized")

    # Build full brief
    approved_brief = {
        **request.brief,
        "business_name": request.business_name,
        "industry": request.industry,
        "style": request.style,
        "goals": request.goals,
        "target_platforms": request.target_platforms
    }

    async def generate():
        async for state in nexus_bridge.start_production(session_id, approved_brief):
            yield f"data: {json.dumps(state.to_dict())}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/production/status/{session_id}", tags=["Commercial_Lab"])
async def get_production_status(session_id: str):
    """Get current production status for a session."""
    if not nexus_bridge:
        raise HTTPException(status_code=503, detail="Production pipeline not initialized")

    state = nexus_bridge.get_production_status(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Production not found")

    return state.to_dict()


@app.get("/api/production/phases", tags=["Commercial_Lab"])
async def get_production_phases():
    """Get all production phase definitions."""
    return {
        "phases": [
            {"id": p.value, "name": p.name, "order": i}
            for i, p in enumerate(ProductionPhase)
        ],
        "total_phases": len(ProductionPhase),
        "pipeline": "RAGNAROK v3.0 APEX"
    }


# =============================================================================
# VOICE PREVIEW (ElevenLabs)
# =============================================================================

class VoicePreviewRequest(BaseModel):
    """Request model for voice preview"""
    voice_id: str = Field(..., description="ElevenLabs voice ID")
    text: str = Field(default="Hello, I'll be the voice of your commercial.", description="Text to speak")

@app.post("/api/voice/preview", tags=["Voice"])
async def voice_preview(request: VoicePreviewRequest):
    """
    Generate ElevenLabs voice preview.
    Returns audio/mpeg stream.
    """
    import httpx

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}",
                headers={
                    "xi-api-key": elevenlabs_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
            )

            if response.status_code != 200:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="ElevenLabs API error")

            # Return audio stream
            return Response(
                content=response.content,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=preview_{request.voice_id}.mp3"
                }
            )

    except httpx.TimeoutException:
        logger.error("ElevenLabs API timeout")
        raise HTTPException(status_code=504, detail="Voice generation timeout")
    except Exception as e:
        logger.error(f"Voice preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
@app.get("/api/voices", tags=["Voice"])
async def list_voices():
    """
    List available ElevenLabs voices for the voice selector UI.

    Returns a curated list of professional voices suitable for commercials.
    Voices are categorized by gender and style for easy selection.
    """
    import httpx

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    # Default curated voices (fallback if API fails)
    default_voices = [
        {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "female", "description": "Warm, professional American female", "preview_url": None},
        {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "category": "female", "description": "Strong, confident female", "preview_url": None},
        {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "category": "female", "description": "Soft, gentle female", "preview_url": None},
        {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "category": "male", "description": "Well-rounded, warm male", "preview_url": None},
        {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "category": "male", "description": "Crisp, authoritative male", "preview_url": None},
        {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "category": "male", "description": "Deep, narrative male", "preview_url": None},
        {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam", "category": "male", "description": "Raspy, dynamic male", "preview_url": None},
        {"voice_id": "jBpfuIE2acCO8z3wKNLl", "name": "Gigi", "category": "female", "description": "Childlike, animated female", "preview_url": None},
    ]

    if not elevenlabs_api_key:
        logger.warning("ElevenLabs API key not configured, returning default voices")
        return {"voices": default_voices, "source": "default"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": elevenlabs_api_key}
            )

            if response.status_code == 200:
                data = response.json()
                voices = []
                for voice in data.get("voices", []):
                    voices.append({
                        "voice_id": voice.get("voice_id"),
                        "name": voice.get("name"),
                        "category": voice.get("labels", {}).get("gender", "unknown"),
                        "description": voice.get("labels", {}).get("description", voice.get("description", "")),
                        "preview_url": voice.get("preview_url"),
                        "labels": voice.get("labels", {})
                    })
                return {"voices": voices, "source": "elevenlabs_api", "count": len(voices)}
            else:
                logger.warning(f"ElevenLabs voices API returned {response.status_code}, using defaults")
                return {"voices": default_voices, "source": "default"}

    except httpx.TimeoutException:
        logger.warning("ElevenLabs voices API timeout, using defaults")
        return {"voices": default_voices, "source": "default"}
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return {"voices": default_voices, "source": "default", "error": str(e)}

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
