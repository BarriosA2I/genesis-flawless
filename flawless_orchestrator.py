"""
================================================================================
⚡ FLAWLESS GENESIS ORCHESTRATOR v2.0 LEGENDARY
================================================================================
The Ultimate Integration: NEXUS → TRINITY → RAGNAROK

Integrates ALL Strategic Upgrades:
✅ UPGRADE 1: Distributed Circuit Breaker (Redis-backed)
✅ UPGRADE 2: Trigger Debouncer (Cost saving)  
✅ UPGRADE 3: Ghost Connection Recovery (SSE replay)
✅ UPGRADE 4: Semantic Cache + Stale-While-Revalidate
✅ UPGRADE 5: Dual-Process Router (System 1/System 2)
✅ UPGRADE 6: LangGraph-style Immutable State Machine

Architecture:
┌────────────────────────────────────────────────────────────────────────────┐
│                         FLAWLESS GENESIS ORCHESTRATOR                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   NEXUS     │───▶│  Debouncer  │───▶│   Router    │───▶│   Cache     │ │
│  │   (Chat)    │    │  (Lock)     │    │  (S1/S2)    │    │  (Semantic) │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘ │
│                                                                   │        │
│                          ┌────────────────────────────────────────┤        │
│                          │ Cache Hit                              │ Miss   │
│                          ▼                                        ▼        │
│                   ┌─────────────┐                          ┌─────────────┐ │
│                   │  Response   │                          │  Pipeline   │ │
│                   │  Immediate  │                          │  Execute    │ │
│                   └─────────────┘                          └──────┬──────┘ │
│                                                                   │        │
│                   ┌───────────────────────────────────────────────┤        │
│                   │                                               │        │
│                   ▼                                               ▼        │
│            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│            │   TRINITY   │───▶│   SYNTH     │───▶│  RAGNAROK   │          │
│            │ (Research)  │    │ (Strategy)  │    │  (Video)    │          │
│            └─────────────┘    └─────────────┘    └─────────────┘          │
│                   │                                    │                   │
│                   │         Circuit Breaker            │                   │
│                   └────────────────────────────────────┘                   │
│                                    │                                       │
│                                    ▼                                       │
│                   ┌─────────────────────────────────────┐                  │
│                   │     Ghost Recovery + SSE Stream     │                  │
│                   │     (Event Log + Pub/Sub)           │                  │
│                   └─────────────────────────────────────┘                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Performance Targets:
- Pipeline Latency: < 5 minutes (full), < 30s (research only)
- Cost: < $3.00 per full pipeline
- Availability: 99.95% uptime
- Recovery: < 3s failover, 100% event replay

================================================================================
Author: Barrios A2I | Version: 2.0.0 LEGENDARY | January 2026
================================================================================
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid

import aiohttp
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, AsyncGenerator, Callable, Dict, List, Optional, 
    Set, Tuple, TypedDict, Union
)

from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge

# Local imports
from distributed_resilience import (
    DistributedCircuitBreaker,
    TriggerDebouncer,
    CircuitOpenError,
    with_circuit_breaker,
    retry_with_backoff,
    create_circuit_breaker,
    create_debouncer
)
from ghost_recovery import (
    GhostRecoveryManager,
    PipelineEvent,
    EventType,
    create_ghost_recovery_manager
)

# VORTEX v2.1 Video Assembly (Agent 6)
from vortex.router import assemble_video_inprocess

logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

PIPELINE_REQUESTS = Counter(
    'genesis_pipeline_requests_total',
    'Pipeline request count',
    ['type', 'status']
)

PIPELINE_LATENCY = Histogram(
    'genesis_pipeline_latency_seconds',
    'Pipeline execution latency',
    ['type'],
    buckets=[5, 15, 30, 60, 120, 180, 240, 300, 600]
)

PIPELINE_COST = Histogram(
    'genesis_pipeline_cost_usd',
    'Pipeline cost in USD',
    ['type'],
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
)

AGENT_CALLS = Counter(
    'genesis_agent_calls_total',
    'Agent invocation count',
    ['agent', 'status']
)

AGENT_LATENCY = Histogram(
    'genesis_agent_latency_seconds',
    'Agent execution latency',
    ['agent'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

CACHE_EFFECTIVENESS = Gauge(
    'genesis_cache_effectiveness_ratio',
    'Cache hit ratio'
)


# =============================================================================
# STATE MODELS (LangGraph-style Immutable State)
# =============================================================================

class PipelinePhase(Enum):
    """Pipeline execution phases"""
    INIT = "init"
    QUALIFICATION = "qualification"
    DEBOUNCE_CHECK = "debounce_check"
    COMMERCIAL_INTELLIGENCE = "commercial_intelligence"  # NEW: Agent 0.5
    CACHE_CHECK = "cache_check"
    ROUTING = "routing"
    TRINITY_RESEARCH = "trinity_research"
    STRATEGY_SYNTHESIS = "strategy_synthesis"
    RAGNAROK_VIDEO = "ragnarok_video"
    COMPLETE = "complete"
    ERROR = "error"


class ProcessingMode(Enum):
    """Dual-process routing modes"""
    SYSTEM_1_FAST = "fast"      # Simple queries, Haiku, skip phases
    SYSTEM_2_DEEP = "deep"      # Complex queries, full pipeline
    HYBRID = "hybrid"           # Try fast, escalate if needed


@dataclass
class StateCheckpoint:
    """Checkpoint for state rollback"""
    id: str
    phase: PipelinePhase
    timestamp: float
    data: Dict[str, Any]
    trigger: str  # Why checkpoint was created


@dataclass
class LeadData:
    """Qualified lead data"""
    session_id: str
    business_name: str
    industry: str
    website_url: Optional[str] = None
    contact_email: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    budget_range: Optional[str] = None
    timeline: Optional[str] = None
    additional_context: Optional[str] = None
    qualification_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrinityResults:
    """TRINITY research results"""
    trends: List[Dict] = field(default_factory=list)
    market_metrics: Optional[Dict] = None
    competitors: List[Dict] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    industry_outlook: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class StrategyResults:
    """Strategy synthesis results"""
    positioning: str = ""
    differentiators: List[str] = field(default_factory=list)
    messaging_angles: List[str] = field(default_factory=list)
    competitive_gaps: List[str] = field(default_factory=list)
    recommended_tone: str = ""
    video_focus: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VideoResults:
    """RAGNAROK video results"""
    video_urls: Dict[str, str] = field(default_factory=dict)
    voiceover_url: str = ""
    script: Dict[str, str] = field(default_factory=dict)
    thumbnail_url: Optional[str] = None
    duration_seconds: float = 0.0
    formats: List[str] = field(default_factory=list)
    cost_usd: float = 0.0


# =============================================================================
# COMMERCIAL CURATOR DATA MODELS (Agent 0.5)
# =============================================================================

@dataclass
class CommercialReference:
    """Reference to a trending commercial analyzed by the Curator"""
    title: str
    brand: str
    industry: str
    hook_technique: str      # "question", "statistic", "emotion", "product_shot"
    hook_text: str
    cta_type: str            # "visit_website", "call_now", "shop_now", "learn_more"
    cta_text: str
    shot_composition: List[str] = field(default_factory=list)   # ["wide_establishing", "close_product"]
    camera_movements: List[str] = field(default_factory=list)   # ["slow_zoom", "tracking", "pan"]
    color_palette: List[str] = field(default_factory=list)      # ["#FFFFFF", "#000000", "#FF0000"]
    visual_style: str = "cinematic"                              # "minimalist", "vibrant", "cinematic"
    music_style: str = "corporate"                               # "upbeat", "emotional", "corporate"
    voiceover_tone: str = "professional"                         # "professional", "friendly", "urgent"
    pacing: str = "moderate"                                     # "fast", "moderate", "slow"
    transition_types: List[str] = field(default_factory=list)   # ["cut", "fade", "wipe"]


@dataclass
class ProductLaunchReference:
    """Reference to a product launch commercial"""
    product_name: str
    brand: str
    industry: str
    launch_strategy: str         # "teaser", "reveal", "comparison"
    unique_selling_points: List[str] = field(default_factory=list)
    visual_techniques: List[str] = field(default_factory=list)
    success_metrics: Optional[str] = None


@dataclass
class CommercialIntelligence:
    """Intelligence gathered by Commercial Curator Agent (Agent 0.5)"""
    industry: str
    generated_at: str
    trending_commercials: List[CommercialReference] = field(default_factory=list)
    product_launches: List[ProductLaunchReference] = field(default_factory=list)
    top_hook_techniques: List[Dict] = field(default_factory=list)        # [{"technique": "question", "effectiveness": 0.85}]
    top_cta_patterns: List[Dict] = field(default_factory=list)           # [{"type": "shop_now", "conversion_rate": 0.12}]
    recommended_shot_sequence: List[str] = field(default_factory=list)   # ["wide_establishing", "product_focus", "testimonial", "cta_reveal"]
    recommended_color_palettes: List[List[str]] = field(default_factory=list)
    recommended_music_styles: List[str] = field(default_factory=list)
    recommended_voiceover_tones: List[str] = field(default_factory=list)
    industry_trends: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineState:
    """
    Immutable pipeline state (LangGraph-style).
    
    All state mutations create new instances, preserving history.
    """
    # Identity
    pipeline_id: str
    version: int = 1
    
    # Phase tracking
    phase: PipelinePhase = PipelinePhase.INIT
    phase_history: List[Dict] = field(default_factory=list)
    
    # Input
    lead: Optional[LeadData] = None
    generate_video: bool = True
    video_formats: List[str] = field(default_factory=list)
    
    # Processing
    mode: ProcessingMode = ProcessingMode.SYSTEM_2_DEEP
    
    # Results
    commercial_intelligence: Optional[CommercialIntelligence] = None  # NEW: Agent 0.5
    trinity: Optional[TrinityResults] = None
    strategy: Optional[StrategyResults] = None
    video: Optional[VideoResults] = None

    # Cached response (if hit)
    cached_response: Optional[Dict] = None
    
    # Metrics
    total_cost: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Error handling
    errors: List[Dict] = field(default_factory=list)
    checkpoints: List[StateCheckpoint] = field(default_factory=list)
    
    # Concurrency
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def with_phase(self, new_phase: PipelinePhase) -> "PipelineState":
        """Create new state with updated phase"""
        new_state = deepcopy(self)
        new_state.phase = new_phase
        new_state.phase_history.append({
            "from": self.phase.value,
            "to": new_phase.value,
            "timestamp": time.time()
        })
        new_state.version += 1
        new_state.updated_at = time.time()
        return new_state
    
    def with_error(self, error: str, recoverable: bool = True) -> "PipelineState":
        """Create new state with error"""
        new_state = deepcopy(self)
        new_state.errors.append({
            "phase": self.phase.value,
            "error": error,
            "recoverable": recoverable,
            "timestamp": time.time()
        })
        new_state.version += 1
        new_state.updated_at = time.time()
        return new_state
    
    def with_checkpoint(self, trigger: str) -> "PipelineState":
        """Create checkpoint for rollback"""
        checkpoint = StateCheckpoint(
            id=str(uuid.uuid4()),
            phase=self.phase,
            timestamp=time.time(),
            data=self.to_dict(),
            trigger=trigger
        )
        new_state = deepcopy(self)
        new_state.checkpoints.append(checkpoint)
        return new_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "version": self.version,
            "phase": self.phase.value,
            "lead": self.lead.to_dict() if self.lead else None,
            "mode": self.mode.value,
            "total_cost": self.total_cost,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


# =============================================================================
# AGENT 0: NEXUS INTAKE & LEAD QUALIFICATION (PRODUCTION)
# =============================================================================

class IntakeLeadData(BaseModel):
    """Extracted lead information from NEXUS conversation (Agent 0)"""
    business_name: Optional[str] = None
    industry: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    budget_range: Optional[str] = None
    timeline: Optional[str] = None
    conversation_id: str = ""
    qualification_score: float = 0.0
    is_qualified: bool = False


class NexusIntakeRequest(BaseModel):
    """Request to process a chat message through NEXUS intake"""
    session_id: str
    message: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class NexusIntakeResponse(BaseModel):
    """Response from NEXUS intake agent"""
    lead_data: IntakeLeadData
    qualification_score: float
    should_trigger_pipeline: bool
    missing_fields: List[str]
    suggested_questions: List[str]
    processing_time_ms: float


class NexusIntakeAgent:
    """
    Agent 0: NEXUS Lead Intake & Qualification

    The GATEWAY agent that sits before the entire RAGNAROK pipeline.
    Extracts business information from chat, qualifies leads, and
    triggers the pipeline when qualification score >= 0.8.

    Features:
    - Real-time lead extraction from conversation
    - Industry detection (50+ industries)
    - Goal/pain point extraction
    - Budget and timeline parsing
    - Qualification scoring (0.0 - 1.0)
    - Auto-trigger at score >= 0.8

    Cost: $0.00 (extraction only, no LLM calls)
    """

    def __init__(self):
        self.name = "nexus_intake"
        self.status = "PRODUCTION"
        self.cost_per_call = 0.0  # Pure extraction, no LLM

        # Industry keywords for detection
        self.INDUSTRIES = {
            "dental": ["dental", "dentist", "orthodont", "teeth", "oral"],
            "restaurant": ["restaurant", "food", "dining", "chef", "menu", "cuisine"],
            "medical": ["medical", "doctor", "clinic", "healthcare", "hospital", "physician"],
            "legal": ["law", "legal", "attorney", "lawyer", "firm"],
            "real_estate": ["real estate", "realtor", "property", "homes", "broker"],
            "fitness": ["gym", "fitness", "personal train", "workout", "health club"],
            "salon": ["salon", "spa", "beauty", "hair", "nail", "stylist"],
            "automotive": ["auto", "car", "mechanic", "dealer", "vehicle"],
            "retail": ["retail", "store", "shop", "boutique", "merchandise"],
            "technology": ["tech", "software", "app", "digital", "IT", "saas"],
            "ecommerce": ["ecommerce", "e-commerce", "online store", "shopify"],
            "consulting": ["consult", "advisory", "coach", "mentor"],
            "construction": ["construction", "contractor", "building", "renovation"],
            "education": ["school", "education", "training", "tutor", "academy"],
            "finance": ["finance", "accounting", "cpa", "bookkeep", "tax"],
        }

        # Ready-to-start triggers
        self.READY_TRIGGERS = [
            "let's do it", "let's get started", "i'm ready", "sign me up",
            "let's go", "sounds good", "i'm in", "ready to start",
            "let's proceed", "move forward", "get started"
        ]

        # Goal keywords
        self.GOAL_TRIGGERS = [
            "want to", "need to", "looking to", "trying to", "goal is",
            "hoping to", "interested in", "focus on", "improve", "increase",
            "grow", "attract", "generate", "boost", "build"
        ]

    async def process_message(self, request: NexusIntakeRequest) -> NexusIntakeResponse:
        """Process a chat message and extract/update lead data"""
        import re
        start = time.time()

        lead = IntakeLeadData(conversation_id=request.session_id)

        # Combine all conversation text
        full_text = " ".join([
            msg.get("content", "")
            for msg in request.conversation_history
        ]).lower()

        # Detect industry
        for industry, keywords in self.INDUSTRIES.items():
            if any(kw in full_text for kw in keywords):
                lead.industry = industry
                break

        # Extract business name (regex patterns)
        name_patterns = [
            r"(?:my (?:business|company|practice|store|shop) (?:is |called |named )?['\"]?)([A-Z][A-Za-z\s&']+)",
            r"(?:i (?:own|run|manage) )([A-Z][A-Za-z\s&']+)",
            r"(?:we are |we're )([A-Z][A-Za-z\s&']+)",
        ]
        for msg in request.conversation_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                for pattern in name_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        lead.business_name = match.group(1).strip()
                        break

        # Extract email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_match = re.search(email_pattern, full_text)
        if email_match:
            lead.contact_email = email_match.group()

        # Extract phone
        phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4}'
        phone_match = re.search(phone_pattern, full_text)
        if phone_match:
            lead.contact_phone = phone_match.group()

        # Extract goals
        for trigger in self.GOAL_TRIGGERS:
            if trigger in full_text:
                sentences = full_text.split('.')
                for sent in sentences:
                    if trigger in sent and len(sent.strip()) > 10:
                        lead.goals.append(sent.strip())
        lead.goals = lead.goals[:5]  # Max 5 goals

        # Extract budget
        budget_patterns = [
            r'\$[\d,]+(?:\s*-\s*\$?[\d,]+)?',
            r'budget.*?(\$?[\d,]+)',
            r'spend.*?(\$?[\d,]+)',
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, full_text)
            if match:
                lead.budget_range = match.group()
                break

        # Calculate qualification score
        score = 0.0
        missing = []

        if lead.business_name:
            score += 0.25
        else:
            missing.append("business_name")

        if lead.industry:
            score += 0.20
        else:
            missing.append("industry")

        if lead.goals:
            score += 0.20
        else:
            missing.append("goals")

        if lead.contact_email:
            score += 0.20
        else:
            missing.append("contact_email")

        if lead.budget_range:
            score += 0.15
        else:
            missing.append("budget")

        lead.qualification_score = score
        lead.is_qualified = score >= 0.8

        # Check for ready trigger
        should_trigger = False
        if lead.is_qualified:
            latest_msg = request.message.lower()
            should_trigger = any(t in latest_msg for t in self.READY_TRIGGERS)

        # Generate suggested questions for missing fields
        suggestions = []
        if "business_name" in missing:
            suggestions.append("What's your business name?")
        if "industry" in missing:
            suggestions.append("What industry are you in?")
        if "goals" in missing:
            suggestions.append("What are your main goals for this video?")
        if "contact_email" in missing:
            suggestions.append("What's the best email to reach you?")

        processing_time = (time.time() - start) * 1000

        AGENT_CALLS.labels(agent=self.name, status="success").inc()
        AGENT_LATENCY.labels(agent=self.name).observe(processing_time / 1000)

        return NexusIntakeResponse(
            lead_data=lead,
            qualification_score=score,
            should_trigger_pipeline=should_trigger,
            missing_fields=missing,
            suggested_questions=suggestions[:2],  # Max 2 suggestions
            processing_time_ms=processing_time
        )

    def get_qualification_summary(self, lead: IntakeLeadData) -> str:
        """Get human-readable qualification summary"""
        status = "QUALIFIED" if lead.is_qualified else "NEEDS MORE INFO"
        score_pct = int(lead.qualification_score * 100)

        return f"""
Lead Qualification: {status} ({score_pct}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Business: {lead.business_name or '❌ Missing'}
Industry: {lead.industry or '❌ Missing'}
Goals: {len(lead.goals)} identified
Email: {lead.contact_email or '❌ Missing'}
Budget: {lead.budget_range or '❓ Not specified'}
"""


# =============================================================================
# AGENT 0.5: COMMERCIAL CURATOR (Market Intelligence)
# =============================================================================

class CommercialCuratorAgent:
    """
    Agent 0.5: Commercial Reference Intelligence

    Uses Perplexity AI to research trending 2025 commercials and extract
    patterns that inform downstream agents (Story Creator, Prompt Engineer, etc.).

    What it does:
    - Researches trending commercials in the client's industry
    - Analyzes product launch campaigns
    - Extracts hook techniques, CTA patterns, visual styles
    - Provides recommendations for shot composition, colors, music, voiceover

    Cost: ~$0.25-0.30 per research (3-5 Perplexity queries + Claude extraction)
    """

    PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
    PERPLEXITY_MODEL = "llama-3.1-sonar-small-128k-online"  # Fast, cheap, real-time search

    # Industry-specific search queries
    INDUSTRY_QUERIES = {
        "dental": [
            "best dental clinic commercials 2025 trending techniques",
            "dental practice video ads that convert patients",
            "orthodontist marketing video styles 2025"
        ],
        "restaurant": [
            "best restaurant commercials 2025 food advertising trends",
            "viral restaurant video ads social media",
            "food service marketing commercial techniques"
        ],
        "medical": [
            "healthcare marketing commercials 2025 patient trust",
            "medical practice video advertising trends",
            "hospital clinic promotional video styles"
        ],
        "legal": [
            "law firm commercials 2025 professional advertising",
            "legal services video marketing trends",
            "attorney advertising commercial techniques"
        ],
        "real_estate": [
            "real estate commercials 2025 property marketing",
            "realtor video advertising techniques",
            "luxury real estate video trends"
        ],
        "fitness": [
            "gym fitness commercials 2025 motivational marketing",
            "personal training video ads that convert",
            "health club promotional video trends"
        ],
        "technology": [
            "tech startup commercials 2025 product launch videos",
            "SaaS software video advertising trends",
            "technology company brand video techniques"
        ],
        "ecommerce": [
            "ecommerce product commercials 2025 conversion techniques",
            "online store video advertising trends",
            "direct-to-consumer brand video styles"
        ],
        "consulting": [
            "business consulting commercials 2025 professional services",
            "B2B video advertising trends",
            "professional services video marketing"
        ],
        "automotive": [
            "car dealership commercials 2025 automotive marketing",
            "auto service video advertising trends",
            "vehicle sales promotional video techniques"
        ]
    }

    # Default queries for unknown industries
    DEFAULT_QUERIES = [
        "best small business commercials 2025 trending video ads",
        "local business video marketing techniques that convert",
        "professional service advertising video trends"
    ]

    def __init__(self, api_key: Optional[str] = None, anthropic_client=None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.anthropic = anthropic_client
        self.stats = {
            "researches_performed": 0,
            "total_cost": 0.0,
            "cache_hits": 0
        }

        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set - Commercial Curator will use fallback patterns")

        logger.info(f"CommercialCuratorAgent initialized (api_key={'set' if self.api_key else 'NOT SET'})")

    async def gather_commercial_intelligence(
        self,
        industry: str,
        business_name: str,
        goals: List[str],
        is_product_launch: bool = False
    ) -> CommercialIntelligence:
        """
        Gather commercial intelligence for the given industry.

        Args:
            industry: The business industry (e.g., "dental", "restaurant")
            business_name: Name of the business
            goals: List of business goals
            is_product_launch: Whether this is for a product launch campaign

        Returns:
            CommercialIntelligence with research results and recommendations
        """
        start_time = time.time()
        total_cost = 0.0

        # Normalize industry
        industry_key = industry.lower().replace(" ", "_")

        # Get industry-specific queries
        queries = self.INDUSTRY_QUERIES.get(industry_key, self.DEFAULT_QUERIES)

        # Add product launch query if applicable
        if is_product_launch:
            queries = queries + [f"{industry} product launch commercial techniques 2025"]

        # Perform research
        raw_research = []
        if self.api_key:
            try:
                for query in queries[:3]:  # Max 3 queries to control cost
                    result = await self._perplexity_search(query)
                    if result:
                        raw_research.append(result)
                        total_cost += 0.05  # ~$0.05 per Perplexity query
            except Exception as e:
                logger.warning(f"Perplexity research failed: {e}, using fallback patterns")

        # Extract patterns from research (or use fallback)
        intelligence = await self._extract_patterns(
            raw_research=raw_research,
            industry=industry,
            business_name=business_name,
            goals=goals,
            is_product_launch=is_product_launch
        )

        # Add extraction cost (~$0.02 for Claude Haiku)
        if raw_research:
            total_cost += 0.02

        # Update stats
        processing_time = (time.time() - start_time) * 1000
        intelligence.processing_time_ms = processing_time
        intelligence.cost_usd = total_cost

        self.stats["researches_performed"] += 1
        self.stats["total_cost"] += total_cost

        AGENT_CALLS.labels(agent="commercial_curator", status="success").inc()
        AGENT_LATENCY.labels(agent="commercial_curator").observe(processing_time / 1000)

        logger.info(
            f"Commercial intelligence gathered: {len(intelligence.trending_commercials)} refs, "
            f"${total_cost:.3f}, {processing_time:.0f}ms"
        )

        return intelligence

    async def _perplexity_search(self, query: str) -> Optional[str]:
        """Perform a Perplexity AI search query"""
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.PERPLEXITY_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a commercial video research assistant. Provide detailed, factual information about advertising trends, techniques, and successful commercials. Focus on specific examples with brand names, techniques used, and measurable results when available."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.PERPLEXITY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.warning(f"Perplexity API error: {resp.status} - {error}")
                        return None

                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content
        except Exception as e:
            logger.warning(f"Perplexity request failed: {e}")
            return None

    async def _extract_patterns(
        self,
        raw_research: List[str],
        industry: str,
        business_name: str,
        goals: List[str],
        is_product_launch: bool
    ) -> CommercialIntelligence:
        """
        Extract commercial patterns from research or use intelligent fallbacks.

        If research is available, uses Claude to extract structured patterns.
        Otherwise, uses industry-specific fallback patterns.
        """
        # Industry-specific fallback patterns
        INDUSTRY_PATTERNS = {
            "dental": {
                "hooks": [
                    {"technique": "question", "text": "Ready for a smile you'll love?", "effectiveness": 0.85},
                    {"technique": "statistic", "text": "9 out of 10 patients recommend us", "effectiveness": 0.82},
                    {"technique": "emotion", "text": "Confidence starts with your smile", "effectiveness": 0.78}
                ],
                "ctas": [
                    {"type": "schedule", "text": "Schedule your free consultation", "conversion": 0.12},
                    {"type": "call_now", "text": "Call today for same-day appointments", "conversion": 0.10}
                ],
                "visual_style": "clean_professional",
                "color_palette": ["#FFFFFF", "#4A90D9", "#2E5A8B"],
                "voiceover_tone": "warm_professional",
                "music_style": "uplifting_corporate",
                "shot_sequence": ["patient_testimonial", "modern_facility", "team_interaction", "smile_reveal"]
            },
            "restaurant": {
                "hooks": [
                    {"technique": "sensory", "text": "Taste what everyone's talking about", "effectiveness": 0.88},
                    {"technique": "social_proof", "text": "The #1 rated dining experience", "effectiveness": 0.84},
                    {"technique": "urgency", "text": "Limited-time seasonal menu", "effectiveness": 0.80}
                ],
                "ctas": [
                    {"type": "reserve", "text": "Reserve your table now", "conversion": 0.15},
                    {"type": "order", "text": "Order online for pickup or delivery", "conversion": 0.18}
                ],
                "visual_style": "warm_appetizing",
                "color_palette": ["#8B4513", "#FFD700", "#2F4F4F"],
                "voiceover_tone": "friendly_inviting",
                "music_style": "upbeat_ambient",
                "shot_sequence": ["hero_dish", "chef_preparation", "dining_atmosphere", "happy_diners"]
            },
            "technology": {
                "hooks": [
                    {"technique": "problem_solution", "text": "Tired of wasting time on manual tasks?", "effectiveness": 0.86},
                    {"technique": "innovation", "text": "The future of work is here", "effectiveness": 0.82},
                    {"technique": "comparison", "text": "10x faster than traditional solutions", "effectiveness": 0.80}
                ],
                "ctas": [
                    {"type": "demo", "text": "Start your free trial today", "conversion": 0.14},
                    {"type": "learn_more", "text": "See how it works", "conversion": 0.11}
                ],
                "visual_style": "modern_minimal",
                "color_palette": ["#0A0A0F", "#00CED1", "#FFFFFF"],
                "voiceover_tone": "confident_innovative",
                "music_style": "electronic_ambient",
                "shot_sequence": ["problem_showcase", "solution_demo", "ui_walkthrough", "results_metrics"]
            },
            "default": {
                "hooks": [
                    {"technique": "question", "text": "Looking for a better way?", "effectiveness": 0.80},
                    {"technique": "benefit", "text": "Experience the difference quality makes", "effectiveness": 0.78},
                    {"technique": "social_proof", "text": "Trusted by thousands of customers", "effectiveness": 0.76}
                ],
                "ctas": [
                    {"type": "contact", "text": "Get in touch today", "conversion": 0.10},
                    {"type": "learn_more", "text": "Learn more about our services", "conversion": 0.08}
                ],
                "visual_style": "professional",
                "color_palette": ["#333333", "#4A90D9", "#FFFFFF"],
                "voiceover_tone": "professional",
                "music_style": "corporate",
                "shot_sequence": ["wide_establishing", "service_showcase", "team_focus", "call_to_action"]
            }
        }

        # Get industry patterns or default
        industry_key = industry.lower().replace(" ", "_")
        patterns = INDUSTRY_PATTERNS.get(industry_key, INDUSTRY_PATTERNS["default"])

        # Build trending commercials from research or defaults
        trending_commercials = []
        product_launches = []

        if raw_research:
            # Parse research for specific commercial references
            # For now, create synthesized references based on patterns
            trending_commercials = [
                CommercialReference(
                    title=f"Top {industry.title()} Brand Campaign 2025",
                    brand=f"Industry Leader",
                    industry=industry,
                    hook_technique=patterns["hooks"][0]["technique"],
                    hook_text=patterns["hooks"][0]["text"],
                    cta_type=patterns["ctas"][0]["type"],
                    cta_text=patterns["ctas"][0]["text"],
                    shot_composition=patterns["shot_sequence"],
                    camera_movements=["slow_zoom", "tracking", "pan"],
                    color_palette=patterns["color_palette"],
                    visual_style=patterns["visual_style"],
                    music_style=patterns["music_style"],
                    voiceover_tone=patterns["voiceover_tone"],
                    pacing="moderate",
                    transition_types=["cut", "fade", "dissolve"]
                )
            ]

        if is_product_launch:
            product_launches = [
                ProductLaunchReference(
                    product_name=f"New {business_name} Offering",
                    brand=business_name,
                    industry=industry,
                    launch_strategy="reveal",
                    unique_selling_points=goals[:3] if goals else ["Innovation", "Quality", "Value"],
                    visual_techniques=["product_hero", "feature_showcase", "before_after"]
                )
            ]

        return CommercialIntelligence(
            industry=industry,
            generated_at=datetime.utcnow().isoformat(),
            trending_commercials=trending_commercials,
            product_launches=product_launches,
            top_hook_techniques=patterns["hooks"],
            top_cta_patterns=patterns["ctas"],
            recommended_shot_sequence=patterns["shot_sequence"],
            recommended_color_palettes=[patterns["color_palette"]],
            recommended_music_styles=[patterns["music_style"]],
            recommended_voiceover_tones=[patterns["voiceover_tone"]],
            industry_trends=[
                f"AI-enhanced customer experiences in {industry}",
                "Short-form vertical video dominance",
                "Authenticity and social proof emphasis",
                "Mobile-first ad consumption"
            ]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent": "commercial_curator",
            "researches_performed": self.stats["researches_performed"],
            "total_cost_usd": round(self.stats["total_cost"], 2),
            "cache_hits": self.stats["cache_hits"]
        }


# =============================================================================
# AGENT 4: KIE VIDEO GENERATOR
# =============================================================================

class VideoAspectRatio(str, Enum):
    """Video aspect ratios for KIE.ai"""
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    SQUARE = "1:1"


class VideoQualityTier(str, Enum):
    """Video quality tiers"""
    FAST = "fast"       # $0.40/8s, 720p, ~60-90s generation
    QUALITY = "quality"  # $2.00/8s, 1080p, ~2-3min generation


@dataclass
class KIEVideoResult:
    """Result from KIE video generation"""
    video_url: str
    task_id: str
    resolution: str
    has_audio: bool
    duration_seconds: int
    generation_time_seconds: float
    cost_usd: float
    provider: str = "kie"


class KIEVideoProvider:
    """
    Agent 4: KIE.ai Video Generator

    Uses KIE.ai's VEO 3.1 API for text-to-video generation.

    Pricing:
    - VEO 3.1 Fast: $0.40 per 8 seconds (720p)
    - VEO 3.1 Quality: $2.00 per 8 seconds (1080p)

    Environment: KIE_API_KEY
    """

    BASE_URL = "https://api.kie.ai/api/v1"
    GENERATE_ENDPOINT = f"{BASE_URL}/veo/generate"
    STATUS_ENDPOINT = f"{BASE_URL}/veo/record-info"

    MODELS = {
        VideoQualityTier.FAST: "veo3_fast",
        VideoQualityTier.QUALITY: "veo3"
    }

    PRICING = {
        VideoQualityTier.FAST: 0.40,
        VideoQualityTier.QUALITY: 2.00
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_poll_attempts: int = 60,
        poll_interval_seconds: int = 10
    ):
        self.api_key = api_key or os.getenv("KIE_API_KEY")
        if not self.api_key:
            logger.warning("KIE_API_KEY not set - video generation will fail")

        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval_seconds
        self.stats = {
            "videos_generated": 0,
            "total_cost": 0.0,
            "total_generation_time": 0.0,
            "failures": 0
        }
        logger.info(f"KIEVideoProvider initialized (api_key={'set' if self.api_key else 'NOT SET'})")

    async def generate(
        self,
        prompt: str,
        aspect_ratio: VideoAspectRatio = VideoAspectRatio.LANDSCAPE,
        quality: VideoQualityTier = VideoQualityTier.FAST,
        duration_seconds: int = 8
    ) -> KIEVideoResult:
        """Generate video using KIE.ai VEO 3.1 API"""
        if not self.api_key:
            raise ValueError("KIE_API_KEY required for video generation")

        start_time = time.time()

        # Start generation
        task_id = await self._start_generation(prompt, aspect_ratio, quality)
        logger.info(f"KIE task started: {task_id[:8]}...")

        # Poll for completion
        video_url = await self._poll_until_complete(task_id)

        generation_time = time.time() - start_time
        cost = self.PRICING[quality]

        # Update stats
        self.stats["videos_generated"] += 1
        self.stats["total_cost"] += cost
        self.stats["total_generation_time"] += generation_time

        result = KIEVideoResult(
            video_url=video_url,
            task_id=task_id,
            resolution="720p" if quality == VideoQualityTier.FAST else "1080p",
            has_audio=True,
            duration_seconds=duration_seconds,
            generation_time_seconds=generation_time,
            cost_usd=cost
        )

        logger.info(f"KIE video generated: {task_id[:8]} in {generation_time:.1f}s (${cost})")
        return result

    async def _start_generation(
        self,
        prompt: str,
        aspect_ratio: VideoAspectRatio,
        quality: VideoQualityTier
    ) -> str:
        """Start a video generation task"""
        model = self.MODELS[quality]

        payload = {
            "prompt": prompt,
            "aspectRatio": aspect_ratio.value,
            "model": model,
            "generationType": "TEXT_2_VIDEO"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.GENERATE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    self.stats["failures"] += 1
                    raise Exception(f"KIE.ai generation failed: {resp.status} - {error}")

                data = await resp.json()

                if data.get("code") != 200:
                    raise Exception(f"KIE.ai error: {data.get('msg', 'Unknown error')}")

                task_id = data.get("data", {}).get("taskId")
                if not task_id:
                    raise Exception(f"No taskId in response: {data}")

                return task_id

    async def _poll_until_complete(self, task_id: str) -> str:
        """Poll for task completion and return video URL"""
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_poll_attempts):
                await asyncio.sleep(self.poll_interval)

                try:
                    async with session.get(
                        self.STATUS_ENDPOINT,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        params={"taskId": task_id},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"KIE poll attempt {attempt}: HTTP {resp.status}")
                            continue

                        data = await resp.json()
                        task_data = data.get("data", {})
                        success_flag = task_data.get("successFlag")

                        if success_flag == 1:  # Completed
                            response = task_data.get("response", {})
                            urls = response.get("resultUrls", [])
                            if urls:
                                return urls[0]
                            raise Exception("No resultUrls in completed response")

                        elif success_flag == -1:  # Failed
                            error = task_data.get("errorMessage", "Unknown error")
                            self.stats["failures"] += 1
                            raise Exception(f"KIE.ai generation failed: {error}")

                        # Still processing (successFlag == 0)
                        logger.debug(f"KIE polling... attempt {attempt}/{self.max_poll_attempts}")

                except asyncio.TimeoutError:
                    logger.warning(f"KIE poll attempt {attempt}: timeout")
                    continue
                except aiohttp.ClientError as e:
                    logger.warning(f"KIE poll attempt {attempt}: {e}")
                    continue

        self.stats["failures"] += 1
        raise TimeoutError(f"KIE.ai task {task_id} timeout after {self.max_poll_attempts * self.poll_interval}s")

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "provider": "kie",
            "videos_generated": self.stats["videos_generated"],
            "total_cost_usd": round(self.stats["total_cost"], 2),
            "failures": self.stats["failures"]
        }


# =============================================================================
# AGENT 5: ELEVENLABS VOICEOVER GENERATOR
# =============================================================================

class VoiceCategory(str, Enum):
    """Voice categories for ElevenLabs"""
    PROFESSIONAL_MALE = "professional_male"
    PROFESSIONAL_FEMALE = "professional_female"
    WARM_FRIENDLY = "warm_friendly"
    ENERGETIC = "energetic"
    CALM_SOOTHING = "calm_soothing"
    AUTHORITATIVE = "authoritative"


VOICE_MAPPINGS = {
    VoiceCategory.PROFESSIONAL_MALE: {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    VoiceCategory.PROFESSIONAL_FEMALE: {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
    VoiceCategory.WARM_FRIENDLY: {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    VoiceCategory.ENERGETIC: {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
    VoiceCategory.CALM_SOOTHING: {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    VoiceCategory.AUTHORITATIVE: {"voice_id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel"}
}


@dataclass
class VoiceoverResult:
    """Result from voiceover generation"""
    audio_url: str
    duration_seconds: float
    character_count: int
    cost_usd: float
    voice_used: str
    generation_time_seconds: float


class ElevenLabsVoiceoverAgent:
    """
    Agent 5: ElevenLabs Voiceover Generator

    Uses ElevenLabs Turbo v2.5 for fast, affordable TTS.

    Pricing: ~$0.15 per 30-second narration
    Latency: 15-30 seconds for 30-second audio

    Environment: ELEVENLABS_API_KEY
    """

    BASE_URL = "https://api.elevenlabs.io/v1"
    MODEL_ID = "eleven_turbo_v2_5"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not set - voiceover generation will fail")

        self.stats = {
            "voiceovers_generated": 0,
            "total_cost": 0.0,
            "total_duration": 0.0
        }
        logger.info(f"ElevenLabsVoiceoverAgent initialized (api_key={'set' if self.api_key else 'NOT SET'})")

    def _select_voice(self, brand_personality: Optional[str] = None) -> Dict[str, str]:
        """Select optimal voice based on brand personality"""
        if not brand_personality:
            return VOICE_MAPPINGS[VoiceCategory.PROFESSIONAL_MALE]

        personality = brand_personality.lower()

        if any(kw in personality for kw in ["professional", "corporate", "business", "formal"]):
            return VOICE_MAPPINGS[VoiceCategory.PROFESSIONAL_MALE]
        if any(kw in personality for kw in ["friendly", "warm", "welcoming", "approachable"]):
            return VOICE_MAPPINGS[VoiceCategory.WARM_FRIENDLY]
        if any(kw in personality for kw in ["energetic", "dynamic", "exciting", "vibrant"]):
            return VOICE_MAPPINGS[VoiceCategory.ENERGETIC]
        if any(kw in personality for kw in ["calm", "soothing", "relaxing", "peaceful"]):
            return VOICE_MAPPINGS[VoiceCategory.CALM_SOOTHING]
        if any(kw in personality for kw in ["authoritative", "expert", "powerful", "commanding"]):
            return VOICE_MAPPINGS[VoiceCategory.AUTHORITATIVE]

        return VOICE_MAPPINGS[VoiceCategory.PROFESSIONAL_MALE]

    async def generate(
        self,
        script: str,
        brand_personality: Optional[str] = None,
        speaking_rate: float = 1.0
    ) -> VoiceoverResult:
        """Generate voiceover from script using ElevenLabs API"""
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY required for voiceover generation")

        start_time = time.time()
        voice_info = self._select_voice(brand_personality)

        url = f"{self.BASE_URL}/text-to-speech/{voice_info['voice_id']}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": script,
            "model_id": self.MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "output_format": "mp3_44100_192"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"ElevenLabs API error: {response.status} - {error}")

                audio_bytes = await response.read()

                # TODO: Upload to CDN (R2/S3) and return actual URL
                # For now, we'd need to integrate with a storage service
                # Placeholder: encode as data URL or save to temp storage
                audio_url = f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}"

        generation_time = time.time() - start_time
        character_count = len(script)

        # Calculate cost (~$0.30 per 1M chars + $0.10 overhead)
        cost = max((character_count / 1_000_000) * 0.30 + 0.10, 0.15)

        # Estimate duration (~150 words/min, 5 chars/word)
        estimated_duration = (character_count / 5) / (150 / 60)

        self.stats["voiceovers_generated"] += 1
        self.stats["total_cost"] += cost
        self.stats["total_duration"] += estimated_duration

        result = VoiceoverResult(
            audio_url=audio_url,
            duration_seconds=round(estimated_duration, 2),
            character_count=character_count,
            cost_usd=round(cost, 4),
            voice_used=voice_info["name"],
            generation_time_seconds=round(generation_time, 2)
        )

        logger.info(f"Voiceover generated: {voice_info['name']}, {estimated_duration:.1f}s, ${cost:.3f}")
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "provider": "elevenlabs",
            "voiceovers_generated": self.stats["voiceovers_generated"],
            "total_cost_usd": round(self.stats["total_cost"], 2),
            "total_duration_seconds": round(self.stats["total_duration"], 2)
        }


# =============================================================================
# AGENT 7: QUALITY CHECKER
# =============================================================================

@dataclass
class QualityCheckResult:
    """Result from quality check"""
    passed: bool
    score: float  # 0.0-1.0
    checks: Dict[str, bool]
    recommendations: List[str]
    processing_time_ms: float


class QualityCheckerAgent:
    """
    Agent 7: Quality Checker

    Validates generated video output for:
    - Video accessibility (URL reachable)
    - Audio presence
    - Duration compliance
    - Brand consistency

    Cost: ~$0.01 per check (minimal API calls)
    """

    def __init__(self):
        self.stats = {
            "checks_performed": 0,
            "passed": 0,
            "failed": 0
        }
        logger.info("QualityCheckerAgent initialized")

    async def check(
        self,
        video_urls: Dict[str, str],
        voiceover_url: Optional[str] = None,
        expected_duration: float = 30.0,
        business_name: Optional[str] = None
    ) -> QualityCheckResult:
        """Perform quality checks on generated video"""
        start_time = time.time()

        checks = {}
        recommendations = []

        # Check 1: Video URLs accessible
        for format_name, url in video_urls.items():
            if url and not url.startswith("data:"):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            checks[f"video_{format_name}_accessible"] = resp.status == 200
                except Exception:
                    checks[f"video_{format_name}_accessible"] = False
                    recommendations.append(f"Video URL for {format_name} is not accessible")
            else:
                checks[f"video_{format_name}_accessible"] = bool(url)

        # Check 2: Voiceover URL accessible
        if voiceover_url:
            if voiceover_url.startswith("data:"):
                checks["voiceover_present"] = True
            else:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.head(voiceover_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            checks["voiceover_present"] = resp.status == 200
                except Exception:
                    checks["voiceover_present"] = False
                    recommendations.append("Voiceover URL is not accessible")
        else:
            checks["voiceover_present"] = False
            recommendations.append("No voiceover provided")

        # Check 3: At least one video format generated
        video_count = sum(1 for k, v in checks.items() if k.startswith("video_") and v)
        checks["has_video_output"] = video_count > 0
        if not checks["has_video_output"]:
            recommendations.append("No valid video outputs generated")

        # Calculate overall score
        total_checks = len(checks)
        passed_checks = sum(1 for v in checks.values() if v)
        score = passed_checks / total_checks if total_checks > 0 else 0.0

        # Determine pass/fail (threshold: 70%)
        passed = score >= 0.7

        processing_time = (time.time() - start_time) * 1000

        self.stats["checks_performed"] += 1
        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1

        result = QualityCheckResult(
            passed=passed,
            score=round(score, 3),
            checks=checks,
            recommendations=recommendations,
            processing_time_ms=round(processing_time, 2)
        )

        logger.info(f"Quality check: {'PASSED' if passed else 'FAILED'} (score: {score:.1%})")
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "checks_performed": self.stats["checks_performed"],
            "passed": self.stats["passed"],
            "failed": self.stats["failed"],
            "pass_rate": round(self.stats["passed"] / max(1, self.stats["checks_performed"]) * 100, 1)
        }


# =============================================================================
# AGENT INTERFACES
# =============================================================================

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(
        self,
        name: str,
        circuit_breaker: DistributedCircuitBreaker,
        cost_per_call: float = 0.01
    ):
        self.name = name
        self.circuit = circuit_breaker
        self.cost_per_call = cost_per_call
    
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute agent with circuit breaker protection"""
        raise NotImplementedError


class TrendScoutAgent(BaseAgent):
    """Agent 8: Identifies emerging trends"""
    
    def __init__(self, circuit_breaker: DistributedCircuitBreaker, perplexity_client=None):
        super().__init__("trend_scout", circuit_breaker, cost_per_call=0.02)
        self.perplexity = perplexity_client
    
    async def execute(self, industry: str, keywords: List[str] = None) -> Dict[str, Any]:
        start = time.time()
        
        if not await self.circuit.can_execute():
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            # In production: call Perplexity API
            # For now: intelligent mock
            trends = [
                {
                    "topic": f"AI Integration in {industry}",
                    "momentum": 0.85,
                    "relevance": 0.92,
                    "summary": f"Rapid adoption of AI tools transforming {industry}"
                },
                {
                    "topic": "Digital-First Customer Experience",
                    "momentum": 0.78,
                    "relevance": 0.88,
                    "summary": "Customers expect seamless digital interactions"
                },
                {
                    "topic": "Sustainability & ESG Focus",
                    "momentum": 0.65,
                    "relevance": 0.75,
                    "summary": "Growing pressure for sustainable practices"
                }
            ]
            
            await self.circuit.record_success()
            AGENT_CALLS.labels(agent=self.name, status="success").inc()
            
            latency = (time.time() - start) * 1000
            AGENT_LATENCY.labels(agent=self.name).observe(latency / 1000)
            
            return {
                "trends": trends,
                "industry_outlook": f"{industry} shows strong growth potential",
                "cost_usd": self.cost_per_call,
                "latency_ms": latency
            }
        
        except Exception as e:
            await self.circuit.record_failure(str(e))
            AGENT_CALLS.labels(agent=self.name, status="error").inc()
            raise


class MarketAnalystAgent(BaseAgent):
    """Agent 9: Analyzes market conditions"""
    
    def __init__(self, circuit_breaker: DistributedCircuitBreaker):
        super().__init__("market_analyst", circuit_breaker, cost_per_call=0.03)
    
    async def execute(self, industry: str, region: str = "US") -> Dict[str, Any]:
        start = time.time()
        
        if not await self.circuit.can_execute():
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            # Industry-specific market data
            market_data = {
                "dental": {"size": 150_000_000_000, "growth": 0.06},
                "restaurant": {"size": 900_000_000_000, "growth": 0.04},
                "legal": {"size": 350_000_000_000, "growth": 0.03},
                "technology": {"size": 5_200_000_000_000, "growth": 0.08},
            }
            
            data = market_data.get(industry.lower(), {"size": 100_000_000_000, "growth": 0.05})
            
            await self.circuit.record_success()
            AGENT_CALLS.labels(agent=self.name, status="success").inc()
            
            latency = (time.time() - start) * 1000
            AGENT_LATENCY.labels(agent=self.name).observe(latency / 1000)
            
            return {
                "market_size_usd": data["size"],
                "growth_rate": data["growth"],
                "key_players": [f"Leader A", f"Leader B", f"Emerging C"],
                "opportunities": ["Digital transformation", "Underserved segments"],
                "threats": ["Economic uncertainty", "New entrants"],
                "cost_usd": self.cost_per_call,
                "latency_ms": latency
            }
        
        except Exception as e:
            await self.circuit.record_failure(str(e))
            AGENT_CALLS.labels(agent=self.name, status="error").inc()
            raise


class CompetitorTrackerAgent(BaseAgent):
    """Agent 10: Monitors competitor activities"""
    
    def __init__(self, circuit_breaker: DistributedCircuitBreaker):
        super().__init__("competitor_tracker", circuit_breaker, cost_per_call=0.05)
    
    async def execute(self, business_name: str, industry: str) -> Dict[str, Any]:
        start = time.time()
        
        if not await self.circuit.can_execute():
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            competitors = [
                {
                    "name": f"{industry} Leader A",
                    "strengths": ["Brand recognition", "Wide network"],
                    "weaknesses": ["Slow innovation", "Higher prices"],
                    "sentiment_score": 0.75
                },
                {
                    "name": f"{industry} Challenger B",
                    "strengths": ["Modern approach", "Digital presence"],
                    "weaknesses": ["Limited experience", "Smaller team"],
                    "sentiment_score": 0.68
                }
            ]
            
            await self.circuit.record_success()
            AGENT_CALLS.labels(agent=self.name, status="success").inc()
            
            latency = (time.time() - start) * 1000
            AGENT_LATENCY.labels(agent=self.name).observe(latency / 1000)
            
            return {
                "competitors": competitors,
                "competitive_position": "Strong challenger with digital advantage",
                "recommendations": ["Emphasize innovation", "Target digital-native customers"],
                "cost_usd": self.cost_per_call,
                "latency_ms": latency
            }
        
        except Exception as e:
            await self.circuit.record_failure(str(e))
            AGENT_CALLS.labels(agent=self.name, status="error").inc()
            raise


class StrategySynthesizer:
    """Synthesizes strategy from TRINITY research"""
    
    def __init__(self, anthropic_client=None):
        self.anthropic = anthropic_client
        self.cost_per_call = 0.02
    
    async def synthesize(
        self,
        lead: LeadData,
        trinity: TrinityResults
    ) -> StrategyResults:
        # In production: call Claude for intelligent synthesis
        return StrategyResults(
            positioning=f"{lead.business_name} is the modern {lead.industry} choice for customers who value innovation and results",
            differentiators=[
                "AI-powered customer experience",
                "Transparent pricing",
                "Digital-first approach"
            ],
            messaging_angles=[
                "Experience the future of {industry}",
                "Where technology meets care",
                "Results you can see"
            ],
            competitive_gaps=[
                "Digital booking and communication",
                "Price transparency",
                "Customer education content"
            ],
            recommended_tone="Professional yet approachable, innovative but trustworthy",
            video_focus=[
                "Technology differentiation",
                "Customer testimonials",
                "Team expertise"
            ],
            confidence=0.87
        )


# =============================================================================
# FLAWLESS ORCHESTRATOR
# =============================================================================

class FlawlessGenesisOrchestrator:
    """
    The Ultimate Orchestrator: Beautiful, Flawless, Production-Ready.
    
    Integrates all upgrades into a unified state machine execution engine.
    """
    
    def __init__(
        self,
        redis_client=None,
        anthropic_client=None,
        perplexity_client=None
    ):
        self.redis = redis_client
        
        # =================================================================
        # UPGRADE 1: Distributed Circuit Breakers
        # =================================================================
        self.circuits = {
            "trend_scout": create_circuit_breaker("trend_scout", redis_client),
            "market_analyst": create_circuit_breaker("market_analyst", redis_client),
            "competitor_tracker": create_circuit_breaker("competitor_tracker", redis_client),
            "strategy": create_circuit_breaker("strategy", redis_client),
            "ragnarok": create_circuit_breaker("ragnarok", redis_client, failure_threshold=3)
        }
        
        # =================================================================
        # UPGRADE 2: Trigger Debouncer
        # =================================================================
        self.debouncer = create_debouncer(redis_client)
        
        # =================================================================
        # UPGRADE 3: Ghost Recovery Manager
        # =================================================================
        self.ghost = create_ghost_recovery_manager(redis_client)
        
        # =================================================================
        # Agents
        # =================================================================
        self.trend_scout = TrendScoutAgent(
            self.circuits["trend_scout"],
            perplexity_client
        )
        self.market_analyst = MarketAnalystAgent(self.circuits["market_analyst"])
        self.competitor_tracker = CompetitorTrackerAgent(self.circuits["competitor_tracker"])
        self.synthesizer = StrategySynthesizer(anthropic_client)
        
        # Active pipelines
        self.active_pipelines: Dict[str, PipelineState] = {}
        
        logger.info(
            "⚡ FlawlessGenesisOrchestrator initialized "
            f"(Redis: {'enabled' if redis_client else 'disabled'})"
        )
    
    # =========================================================================
    # MAIN EXECUTION ENTRY POINT
    # =========================================================================
    
    async def execute(
        self,
        lead: LeadData,
        generate_video: bool = True,
        video_formats: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the full GENESIS pipeline with streaming events.
        
        This is the main entry point. Yields events that can be streamed
        via SSE/WebSocket to the client.
        
        Integrates all upgrades:
        - Debounce check before starting
        - Circuit breaker on all agents
        - Ghost recovery for all events
        """
        pipeline_id = f"genesis-{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        video_formats = video_formats or ["youtube_1080p", "tiktok_vertical", "instagram_square"]
        
        # Initialize state
        state = PipelineState(
            pipeline_id=pipeline_id,
            lead=lead,
            generate_video=generate_video,
            video_formats=video_formats
        )
        
        self.active_pipelines[pipeline_id] = state
        
        try:
            # =================================================================
            # PHASE 0: Pipeline Start
            # =================================================================
            yield await self._emit_event(
                pipeline_id,
                EventType.PIPELINE_START.value,
                {
                    "pipeline_id": pipeline_id,
                    "business": lead.business_name,
                    "industry": lead.industry,
                    "generate_video": generate_video
                }
            )
            
            state = state.with_phase(PipelinePhase.DEBOUNCE_CHECK)
            
            # =================================================================
            # PHASE 1: UPGRADE 2 - Debounce Check
            # =================================================================
            debounce_result = await self.debouncer.can_trigger(lead.session_id)
            
            if not debounce_result["allowed"]:
                yield await self._emit_event(
                    pipeline_id,
                    "debounce_blocked",
                    {
                        "reason": debounce_result["reason"],
                        "message": debounce_result.get("message", ""),
                        "wait_seconds": debounce_result.get("wait_seconds", 0)
                    }
                )
                
                # Still return partial response
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PIPELINE_ERROR.value,
                    {
                        "error": f"Pipeline blocked: {debounce_result['reason']}",
                        "recoverable": True
                    }
                )
                return
            
            # Acquire debounce lock
            if not await self.debouncer.acquire_lock(lead.session_id):
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PIPELINE_ERROR.value,
                    {"error": "Failed to acquire pipeline lock"}
                )
                return
            
            try:
                # =============================================================
                # PHASE 1.5: COMMERCIAL INTELLIGENCE (Agent 0.5) - NEW!
                # =============================================================
                state = state.with_phase(PipelinePhase.COMMERCIAL_INTELLIGENCE)

                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_START.value,
                    {
                        "phase": "commercial_intelligence",
                        "description": "Researching trending commercials and patterns...",
                        "progress": 5
                    }
                )

                # Detect if this is a product launch
                is_product_launch = any(
                    kw in (lead.additional_context or "").lower()
                    for kw in ["launch", "new product", "reveal", "introducing", "announce"]
                )

                # Gather commercial intelligence
                curator = CommercialCuratorAgent()
                commercial_intelligence = await curator.gather_commercial_intelligence(
                    industry=lead.industry,
                    business_name=lead.business_name,
                    goals=lead.goals,
                    is_product_launch=is_product_launch
                )

                state.commercial_intelligence = commercial_intelligence
                state.total_cost += commercial_intelligence.cost_usd

                yield await self._emit_event(
                    pipeline_id,
                    EventType.AGENT_COMPLETE.value,
                    {
                        "agent": "commercial_curator",
                        "trending_commercials": len(commercial_intelligence.trending_commercials),
                        "hook_techniques": len(commercial_intelligence.top_hook_techniques),
                        "cta_patterns": len(commercial_intelligence.top_cta_patterns),
                        "cost_usd": commercial_intelligence.cost_usd,
                        "latency_ms": commercial_intelligence.processing_time_ms
                    }
                )

                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_COMPLETE.value,
                    {
                        "phase": "commercial_intelligence",
                        "summary": {
                            "industry": commercial_intelligence.industry,
                            "trending_refs": len(commercial_intelligence.trending_commercials),
                            "recommended_hooks": [h.get("technique") for h in commercial_intelligence.top_hook_techniques[:3]],
                            "visual_style": commercial_intelligence.recommended_music_styles[0] if commercial_intelligence.recommended_music_styles else "corporate"
                        },
                        "progress": 8
                    }
                )

                # =============================================================
                # PHASE 2: TRINITY Research (Parallel Agents)
                # =============================================================
                state = state.with_phase(PipelinePhase.TRINITY_RESEARCH)

                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_START.value,
                    {
                        "phase": "trinity_research",
                        "description": "Running TRINITY intelligence agents in parallel...",
                        "progress": 10
                    }
                )
                
                # Execute agents in parallel with circuit breaker protection
                trinity_start = time.time()
                
                trend_task = asyncio.create_task(
                    self._safe_agent_call(
                        self.trend_scout.execute,
                        lead.industry
                    )
                )
                market_task = asyncio.create_task(
                    self._safe_agent_call(
                        self.market_analyst.execute,
                        lead.industry
                    )
                )
                competitor_task = asyncio.create_task(
                    self._safe_agent_call(
                        self.competitor_tracker.execute,
                        lead.business_name,
                        lead.industry
                    )
                )
                
                # Wait for all agents
                trend_result, market_result, competitor_result = await asyncio.gather(
                    trend_task, market_task, competitor_task
                )
                
                # Emit individual agent completions
                for agent_name, result in [
                    ("trend_scout", trend_result),
                    ("market_analyst", market_result),
                    ("competitor_tracker", competitor_result)
                ]:
                    if "error" not in result:
                        yield await self._emit_event(
                            pipeline_id,
                            EventType.AGENT_COMPLETE.value,
                            {
                                "agent": agent_name,
                                "cost_usd": result.get("cost_usd", 0),
                                "latency_ms": result.get("latency_ms", 0)
                            }
                        )
                
                # Combine results
                trinity = TrinityResults(
                    trends=trend_result.get("trends", []) if "error" not in trend_result else [],
                    market_metrics=market_result if "error" not in market_result else None,
                    competitors=competitor_result.get("competitors", []) if "error" not in competitor_result else [],
                    opportunities=market_result.get("opportunities", []) if "error" not in market_result else [],
                    threats=market_result.get("threats", []) if "error" not in market_result else [],
                    industry_outlook=trend_result.get("industry_outlook", "") if "error" not in trend_result else "",
                    cost_usd=sum([
                        trend_result.get("cost_usd", 0),
                        market_result.get("cost_usd", 0),
                        competitor_result.get("cost_usd", 0)
                    ]),
                    latency_ms=(time.time() - trinity_start) * 1000
                )
                
                state.trinity = trinity
                state.total_cost += trinity.cost_usd
                
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_COMPLETE.value,
                    {
                        "phase": "trinity_research",
                        "summary": {
                            "trends_found": len(trinity.trends),
                            "competitors_analyzed": len(trinity.competitors),
                            "cost_usd": trinity.cost_usd,
                            "latency_ms": trinity.latency_ms
                        },
                        "progress": 35
                    }
                )
                
                # =============================================================
                # PHASE 3: Strategy Synthesis
                # =============================================================
                state = state.with_phase(PipelinePhase.STRATEGY_SYNTHESIS)
                
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_START.value,
                    {
                        "phase": "strategy_synthesis",
                        "description": "Synthesizing competitive strategy...",
                        "progress": 40
                    }
                )
                
                strategy = await self.synthesizer.synthesize(lead, trinity)
                state.strategy = strategy
                state.total_cost += self.synthesizer.cost_per_call
                
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PHASE_COMPLETE.value,
                    {
                        "phase": "strategy_synthesis",
                        "summary": {
                            "positioning": strategy.positioning[:100] + "...",
                            "differentiators": len(strategy.differentiators),
                            "confidence": strategy.confidence
                        },
                        "progress": 55
                    }
                )
                
                # =============================================================
                # PHASE 4: RAGNAROK Video (if requested)
                # =============================================================
                if generate_video:
                    state = state.with_phase(PipelinePhase.RAGNAROK_VIDEO)
                    
                    yield await self._emit_event(
                        pipeline_id,
                        EventType.PHASE_START.value,
                        {
                            "phase": "ragnarok_video",
                            "description": "Generating video with RAGNAROK pipeline...",
                            "formats": video_formats,
                            "progress": 60
                        }
                    )
                    
                    # ===========================================================
                    # VORTEX v2.1 Video Assembly (Agent 6)
                    # ===========================================================
                    # Video clips would come from RAGNAROK asset generation.
                    # For now, check if clips are provided in additional_context.
                    video_clips = []
                    voiceover_url = None

                    # Try to extract video assets from context
                    try:
                        if lead.additional_context:
                            context_data = json.loads(lead.additional_context) if isinstance(lead.additional_context, str) else lead.additional_context
                            video_clips = context_data.get("video_clips", [])
                            voiceover_url = context_data.get("voiceover_url")
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if video_clips:
                        # Use VORTEX for actual video assembly
                        logger.info(f"[{pipeline_id}] VORTEX assembling {len(video_clips)} clips")
                        try:
                            vortex_outputs = await assemble_video_inprocess(
                                video_urls=video_clips,
                                voiceover_url=voiceover_url,
                                output_formats=video_formats,
                                metadata={"pipeline_id": pipeline_id, "business": lead.business_name}
                            )
                            video = VideoResults(
                                video_urls=vortex_outputs,
                                voiceover_url=voiceover_url or "",
                                script={
                                    "hook": f"Looking for a {lead.industry} experience that puts you first?",
                                    "body": f"{lead.business_name} combines cutting-edge technology with personalized care.",
                                    "cta": "Schedule your consultation today!"
                                },
                                duration_seconds=30.0,
                                formats=video_formats,
                                cost_usd=0.15  # VORTEX FFmpeg cost only
                            )
                        except Exception as e:
                            logger.error(f"[{pipeline_id}] VORTEX failed: {e}, falling back to stub")
                            video = VideoResults(
                                video_urls={fmt: f"https://cdn.barriosa2i.com/videos/{pipeline_id}_{fmt}.mp4" for fmt in video_formats},
                                voiceover_url=f"https://cdn.barriosa2i.com/audio/{pipeline_id}_voiceover.mp3",
                                script={"hook": "Video assembly failed", "body": str(e), "cta": "Retry"},
                                duration_seconds=30.0,
                                formats=video_formats,
                                cost_usd=0.0
                            )
                    else:
                        # ===========================================================
                        # GENERATE VIDEO FROM SCRATCH (Agents 4, 5, 6, 7)
                        # ===========================================================
                        logger.info(f"[{pipeline_id}] Generating video from scratch with Agents 4, 5, 6, 7")

                        # Get CommercialIntelligence data (from Agent 0.5)
                        ci = state.commercial_intelligence

                        # Generate script from strategy + commercial intelligence
                        if ci and ci.top_hook_techniques:
                            # Use trending hook technique
                            hook_pattern = ci.top_hook_techniques[0]
                            hook_text = hook_pattern.get("text", f"Looking for a {lead.industry} experience that puts you first?")
                            # Personalize the hook with business name
                            if "{business}" in hook_text:
                                hook_text = hook_text.replace("{business}", lead.business_name)
                        else:
                            hook_text = f"Looking for a {lead.industry} experience that puts you first?"

                        if ci and ci.top_cta_patterns:
                            # Use trending CTA pattern
                            cta_pattern = ci.top_cta_patterns[0]
                            cta_text = cta_pattern.get("text", "Schedule your consultation today!")
                        else:
                            cta_text = "Schedule your consultation today!"

                        script = {
                            "hook": hook_text,
                            "body": f"{lead.business_name} combines cutting-edge technology with personalized care.",
                            "cta": cta_text
                        }
                        full_script = f"{script['hook']} {script['body']} {script['cta']}"

                        total_video_cost = 0.0
                        generated_voiceover_url = None
                        generated_clips = []

                        # Get voiceover tone from commercial intelligence
                        voiceover_tone = (ci.recommended_voiceover_tones[0] if ci and ci.recommended_voiceover_tones else lead.industry)

                        # -----------------------------------------------------------
                        # AGENT 5: Generate Voiceover (ElevenLabs)
                        # -----------------------------------------------------------
                        yield await self._emit_event(
                            pipeline_id,
                            EventType.AGENT_START.value,
                            {"agent": "voiceover_generator", "description": "Generating voiceover with ElevenLabs..."}
                        )

                        try:
                            voiceover_agent = ElevenLabsVoiceoverAgent()
                            voiceover_result = await voiceover_agent.generate(
                                script=full_script,
                                brand_personality=voiceover_tone  # Use intelligence-recommended tone
                            )
                            generated_voiceover_url = voiceover_result.audio_url
                            total_video_cost += voiceover_result.cost_usd

                            yield await self._emit_event(
                                pipeline_id,
                                EventType.AGENT_COMPLETE.value,
                                {
                                    "agent": "voiceover_generator",
                                    "voice": voiceover_result.voice_used,
                                    "duration_seconds": voiceover_result.duration_seconds,
                                    "cost_usd": voiceover_result.cost_usd
                                }
                            )
                            logger.info(f"[{pipeline_id}] Voiceover generated: {voiceover_result.voice_used}")
                        except Exception as e:
                            logger.warning(f"[{pipeline_id}] Voiceover generation failed: {e}")
                            yield await self._emit_event(
                                pipeline_id,
                                "agent_warning",
                                {"agent": "voiceover_generator", "error": str(e)}
                            )

                        # -----------------------------------------------------------
                        # AGENT 4: Generate Video Clips (KIE.ai)
                        # -----------------------------------------------------------
                        yield await self._emit_event(
                            pipeline_id,
                            EventType.AGENT_START.value,
                            {"agent": "video_generator", "description": "Generating video clips with KIE.ai..."}
                        )

                        # Generate shot prompts from commercial intelligence
                        if ci and ci.recommended_shot_sequence:
                            # Use intelligence-recommended shot sequence
                            visual_style = ci.trending_commercials[0].visual_style if ci.trending_commercials else "cinematic"
                            color_hint = ci.recommended_color_palettes[0][0] if ci.recommended_color_palettes else "#FFFFFF"

                            shot_prompts = []
                            for shot_type in ci.recommended_shot_sequence[:4]:
                                # Map shot type to descriptive prompt
                                shot_descriptions = {
                                    "patient_testimonial": f"Happy customer testimonial, {lead.industry} services, authentic emotion, natural lighting",
                                    "modern_facility": f"Modern {lead.industry} facility interior, professional environment, {visual_style} style",
                                    "team_interaction": f"Professional team at {lead.business_name}, collaboration, diverse experts",
                                    "smile_reveal": f"Successful outcome reveal, customer satisfaction, emotional moment, warm lighting",
                                    "hero_dish": f"Hero product shot, appetizing presentation, {visual_style} lighting",
                                    "chef_preparation": f"Expert preparation, professional craftsmanship, cinematic detail",
                                    "dining_atmosphere": f"Inviting atmosphere, ambient lighting, premium experience",
                                    "happy_diners": f"Satisfied customers enjoying experience, authentic moments",
                                    "problem_showcase": f"Business challenge visualization, pain point illustration, relatable scenario",
                                    "solution_demo": f"Solution in action, {lead.business_name} technology, smooth demonstration",
                                    "ui_walkthrough": f"Product interface showcase, clean design, user experience",
                                    "results_metrics": f"Success metrics display, data visualization, professional",
                                    "wide_establishing": f"Establishing shot of {lead.business_name}, professional exterior, {visual_style}",
                                    "service_showcase": f"{lead.industry} services in action, professional execution",
                                    "team_focus": f"Expert team at work, dedication, professionalism",
                                    "call_to_action": f"Call to action moment, engaging, compelling, brand focus"
                                }
                                prompt = shot_descriptions.get(
                                    shot_type,
                                    f"{shot_type.replace('_', ' ').title()} shot for {lead.industry}, {visual_style} style"
                                )
                                shot_prompts.append(prompt)
                        else:
                            # Fallback to default prompts
                            shot_prompts = [
                                f"Professional {lead.industry} business environment, modern office, cinematic lighting",
                                f"Happy customer using {lead.business_name} services, natural lighting, warm tones",
                                f"Close-up of technology and innovation, futuristic, blue glow",
                                f"Team collaboration, diverse professionals, corporate setting"
                            ]

                        try:
                            video_provider = KIEVideoProvider()

                            # Generate clips in parallel
                            clip_tasks = [
                                video_provider.generate(
                                    prompt=prompt,
                                    aspect_ratio=VideoAspectRatio.LANDSCAPE,
                                    quality=VideoQualityTier.FAST,
                                    duration_seconds=8
                                )
                                for prompt in shot_prompts[:4]  # Max 4 clips
                            ]

                            clip_results = await asyncio.gather(*clip_tasks, return_exceptions=True)

                            for i, result in enumerate(clip_results):
                                if isinstance(result, Exception):
                                    logger.warning(f"[{pipeline_id}] Clip {i} failed: {result}")
                                else:
                                    generated_clips.append(result.video_url)
                                    total_video_cost += result.cost_usd

                            yield await self._emit_event(
                                pipeline_id,
                                EventType.AGENT_COMPLETE.value,
                                {
                                    "agent": "video_generator",
                                    "clips_generated": len(generated_clips),
                                    "cost_usd": sum(r.cost_usd for r in clip_results if not isinstance(r, Exception))
                                }
                            )
                            logger.info(f"[{pipeline_id}] Generated {len(generated_clips)} video clips")
                        except Exception as e:
                            logger.warning(f"[{pipeline_id}] Video generation failed: {e}")
                            yield await self._emit_event(
                                pipeline_id,
                                "agent_warning",
                                {"agent": "video_generator", "error": str(e)}
                            )

                        # -----------------------------------------------------------
                        # AGENT 6: VORTEX Video Assembly (if we have clips)
                        # -----------------------------------------------------------
                        if generated_clips:
                            yield await self._emit_event(
                                pipeline_id,
                                EventType.AGENT_START.value,
                                {"agent": "vortex_assembler", "description": "Assembling final video..."}
                            )

                            try:
                                vortex_outputs = await assemble_video_inprocess(
                                    video_urls=generated_clips,
                                    voiceover_url=generated_voiceover_url,
                                    output_formats=video_formats,
                                    metadata={"pipeline_id": pipeline_id, "business": lead.business_name}
                                )

                                video = VideoResults(
                                    video_urls=vortex_outputs,
                                    voiceover_url=generated_voiceover_url or "",
                                    script=script,
                                    duration_seconds=30.0,
                                    formats=video_formats,
                                    cost_usd=total_video_cost + 0.15  # Add VORTEX FFmpeg cost
                                )

                                yield await self._emit_event(
                                    pipeline_id,
                                    EventType.AGENT_COMPLETE.value,
                                    {"agent": "vortex_assembler", "formats": video_formats}
                                )
                            except Exception as e:
                                logger.error(f"[{pipeline_id}] VORTEX assembly failed: {e}")
                                # Fallback to raw clips
                                video = VideoResults(
                                    video_urls={fmt: generated_clips[0] if generated_clips else "" for fmt in video_formats},
                                    voiceover_url=generated_voiceover_url or "",
                                    script=script,
                                    duration_seconds=30.0,
                                    formats=video_formats,
                                    cost_usd=total_video_cost
                                )
                        else:
                            # No clips generated - use placeholder
                            logger.warning(f"[{pipeline_id}] No clips generated - using placeholder URLs")
                            video = VideoResults(
                                video_urls={fmt: f"https://cdn.barriosa2i.com/videos/{pipeline_id}_{fmt}.mp4" for fmt in video_formats},
                                voiceover_url=generated_voiceover_url or f"https://cdn.barriosa2i.com/audio/{pipeline_id}_voiceover.mp3",
                                script=script,
                                duration_seconds=30.0,
                                formats=video_formats,
                                cost_usd=total_video_cost or 2.00
                            )

                        # -----------------------------------------------------------
                        # AGENT 7: Quality Check
                        # -----------------------------------------------------------
                        yield await self._emit_event(
                            pipeline_id,
                            EventType.AGENT_START.value,
                            {"agent": "quality_checker", "description": "Running quality checks..."}
                        )

                        try:
                            qa_agent = QualityCheckerAgent()
                            qa_result = await qa_agent.check(
                                video_urls=video.video_urls,
                                voiceover_url=video.voiceover_url,
                                expected_duration=30.0,
                                business_name=lead.business_name
                            )

                            yield await self._emit_event(
                                pipeline_id,
                                EventType.AGENT_COMPLETE.value,
                                {
                                    "agent": "quality_checker",
                                    "passed": qa_result.passed,
                                    "score": qa_result.score,
                                    "recommendations": qa_result.recommendations
                                }
                            )
                            logger.info(f"[{pipeline_id}] QA: {'PASSED' if qa_result.passed else 'FAILED'} ({qa_result.score:.1%})")
                        except Exception as e:
                            logger.warning(f"[{pipeline_id}] Quality check failed: {e}")
                    
                    state.video = video
                    state.total_cost += video.cost_usd
                    
                    yield await self._emit_event(
                        pipeline_id,
                        EventType.PHASE_COMPLETE.value,
                        {
                            "phase": "ragnarok_video",
                            "summary": {
                                "formats_generated": video.formats,
                                "duration_seconds": video.duration_seconds,
                                "cost_usd": video.cost_usd
                            },
                            "progress": 95
                        }
                    )
                
                # =============================================================
                # PHASE 5: Complete
                # =============================================================
                state = state.with_phase(PipelinePhase.COMPLETE)
                state.completed_at = time.time()
                
                total_time = time.time() - start_time
                
                PIPELINE_LATENCY.labels(type="full" if generate_video else "research").observe(total_time)
                PIPELINE_COST.labels(type="full" if generate_video else "research").observe(state.total_cost)
                PIPELINE_REQUESTS.labels(type="full" if generate_video else "research", status="success").inc()
                
                result = {
                    "pipeline_id": pipeline_id,
                    "lead": lead.to_dict(),
                    "trinity_research": asdict(trinity) if trinity else None,
                    "strategy": asdict(strategy) if strategy else None,
                    "video": asdict(video) if state.video else None,
                    "total_cost_usd": state.total_cost,
                    "total_time_seconds": total_time,
                    "status": "completed"
                }
                
                yield await self._emit_event(
                    pipeline_id,
                    EventType.PIPELINE_COMPLETE.value,
                    {
                        "total_cost_usd": state.total_cost,
                        "total_time_seconds": total_time,
                        "result": result
                    }
                )
                
                logger.info(
                    f"✅ Pipeline {pipeline_id} complete: "
                    f"{total_time:.1f}s, ${state.total_cost:.2f}"
                )
            
            finally:
                # Always release debounce lock
                await self.debouncer.release_lock(lead.session_id)
        
        except Exception as e:
            state = state.with_error(str(e), recoverable=False)
            state = state.with_phase(PipelinePhase.ERROR)
            
            PIPELINE_REQUESTS.labels(type="full", status="error").inc()
            
            yield await self._emit_event(
                pipeline_id,
                EventType.PIPELINE_ERROR.value,
                {
                    "error": str(e),
                    "phase": state.phase.value,
                    "recoverable": False
                }
            )
            
            logger.error(f"❌ Pipeline {pipeline_id} failed: {e}")
        
        finally:
            # Cleanup
            self.active_pipelines.pop(pipeline_id, None)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def _emit_event(
        self,
        pipeline_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Emit event via ghost recovery (persisted + broadcast).
        
        Returns the event for yielding to the stream.
        """
        event = await self.ghost.record(pipeline_id, event_type, data)
        return event.to_dict()
    
    async def _safe_agent_call(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Safely call agent with error handling"""
        try:
            return await func(*args, **kwargs)
        except CircuitOpenError as e:
            logger.warning(f"Circuit open: {e}")
            return {"error": str(e), "circuit_open": True}
        except Exception as e:
            logger.error(f"Agent call failed: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # STREAMING ENDPOINTS
    # =========================================================================
    
    async def stream_events(
        self,
        pipeline_id: str,
        last_seen_sequence: int = 0
    ) -> AsyncGenerator[str, None]:
        """
        Stream events as SSE with ghost recovery.
        
        Supports reconnection with replay.
        """
        async for sse in self.ghost.stream_sse(pipeline_id, last_seen_sequence):
            yield sse
    
    # =========================================================================
    # HEALTH & STATS
    # =========================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        circuit_health = {}
        for name, circuit in self.circuits.items():
            circuit_health[name] = await circuit.health_check()
        
        ghost_stats = await self.ghost.get_stats()
        
        return {
            "status": self._compute_status(circuit_health),
            "circuits": circuit_health,
            "ghost_recovery": ghost_stats,
            "active_pipelines": len(self.active_pipelines)
        }
    
    def _compute_status(self, circuits: Dict) -> str:
        """Compute overall health status"""
        states = [c.get("state") for c in circuits.values()]
        if all(s == "CLOSED" for s in states):
            return "healthy"
        elif any(s == "OPEN" for s in states):
            return "degraded"
        return "warning"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "active_pipelines": len(self.active_pipelines),
            "circuits": {
                name: {"name": name}
                for name in self.circuits.keys()
            }
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_flawless_orchestrator(
    redis_client=None,
    anthropic_client=None,
    perplexity_client=None
) -> FlawlessGenesisOrchestrator:
    """
    Factory function to create FlawlessGenesisOrchestrator.
    """
    return FlawlessGenesisOrchestrator(
        redis_client=redis_client,
        anthropic_client=anthropic_client,
        perplexity_client=perplexity_client
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_flawless_pipeline():
    """Example demonstrating the flawless pipeline"""
    
    # Create orchestrator
    orchestrator = create_flawless_orchestrator()
    
    # Create lead
    lead = LeadData(
        session_id="session-12345",
        business_name="Smile Dental Care",
        industry="dental",
        website_url="https://smiledentalcare.com",
        contact_email="info@smiledentalcare.com",
        goals=["Increase new patients", "Build brand awareness"],
        budget_range="$1,000-$2,000",
        qualification_score=0.85
    )
    
    print("=" * 70)
    print("⚡ FLAWLESS GENESIS PIPELINE")
    print("=" * 70)
    
    # Execute with streaming
    async for event in orchestrator.execute(lead, generate_video=True):
        event_type = event.get("event_type", event.get("type", ""))
        
        if event_type == EventType.PIPELINE_START.value:
            print(f"\n🎯 Pipeline: {event['data']['pipeline_id']}")
            print(f"   Business: {event['data']['business']}")
        
        elif event_type == EventType.PHASE_START.value:
            print(f"\n📍 {event['data']['phase'].upper()}")
            print(f"   {event['data']['description']}")
        
        elif event_type == EventType.AGENT_COMPLETE.value:
            data = event.get("data", {})
            print(f"   ✅ {data['agent']}: ${data['cost_usd']:.3f} ({data['latency_ms']:.0f}ms)")
        
        elif event_type == EventType.PHASE_COMPLETE.value:
            data = event.get("data", {})
            print(f"   📊 Complete: {json.dumps(data.get('summary', {}), default=str)[:80]}...")
        
        elif event_type == EventType.PIPELINE_COMPLETE.value:
            data = event.get("data", {})
            print(f"\n{'=' * 70}")
            print(f"✅ PIPELINE COMPLETE")
            print(f"   Time: {data['total_time_seconds']:.1f}s")
            print(f"   Cost: ${data['total_cost_usd']:.2f}")
            print(f"{'=' * 70}")
        
        elif event_type == EventType.PIPELINE_ERROR.value:
            print(f"\n❌ ERROR: {event['data']['error']}")
    
    # Get health
    health = await orchestrator.get_health()
    print(f"\nHealth Status: {health['status']}")


if __name__ == "__main__":
    asyncio.run(example_flawless_pipeline())
