"""
================================================================================
🎬 CREATIVE DIRECTOR: RAGNAROK + NEXUS BRAIN INTEGRATION
================================================================================
Intelligent conversational video brief intake powered by Nexus Brain AI,
outputting structured briefs ready for RAGNAROK v7.0 APEX pipeline.

Author: Barrios A2I | Version: 1.0.0 | December 2025
================================================================================
"""

import asyncio
import hashlib
import json
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

# Knowledge loader for Memory Pack integration
import logging as _logging
_kl_logger = _logging.getLogger("KnowledgeLoaderInit")
try:
    # Try relative import first (when running as part of GENESIS package)
    from .knowledge_loader import get_knowledge_loader, route_to_knowledge, get_knowledge_context
    KNOWLEDGE_ENABLED = True
except ImportError:
    try:
        # Fall back to direct import (standalone mode)
        from knowledge_loader import get_knowledge_loader, route_to_knowledge, get_knowledge_context
        KNOWLEDGE_ENABLED = True
    except ImportError:
        KNOWLEDGE_ENABLED = False
        get_knowledge_loader = None
        route_to_knowledge = None
        get_knowledge_context = None
if KNOWLEDGE_ENABLED:
    try:
        # Pre-load knowledge on import to catch errors early
        _loader = get_knowledge_loader()
        _knowledge_file_count = len(_loader._cache) if _loader._cache else 0
        _kl_logger.info(f"Knowledge loader initialized: {_knowledge_file_count} files loaded")
    except Exception as e:
        import traceback
        _kl_logger.error(f"Knowledge loader Exception: {e}")
        _kl_logger.error(f"Traceback: {traceback.format_exc()}")
        _knowledge_file_count = 0
else:
    _knowledge_file_count = 0

# Website RAG for real-time business knowledge from barriosa2i.com
_rag_logger = _logging.getLogger("WebsiteRAGInit")
try:
    from knowledge.website_rag import get_website_rag
    WEBSITE_RAG_ENABLED = True
    _rag_logger.info("Website RAG module loaded successfully")
except ImportError as e:
    _rag_logger.warning(f"Website RAG module not available: {e}")
    WEBSITE_RAG_ENABLED = False
    get_website_rag = None

# Resilience module for retry logic, circuit breaker, and graceful degradation
_resilience_logger = _logging.getLogger("ResilienceInit")
try:
    from resilience import (
        get_claude_circuit_breaker,
        retry_with_backoff,
        CLAUDE_RETRY_CONFIG,
        GracefulDegrader,
        get_resilience_status,
    )
    RESILIENCE_ENABLED = True
    _resilience_logger.info("Resilience module loaded successfully")
except ImportError as e:
    _resilience_logger.warning(f"Resilience module not available: {e}")
    RESILIENCE_ENABLED = False
    get_claude_circuit_breaker = None
    retry_with_backoff = None
    CLAUDE_RETRY_CONFIG = None
    GracefulDegrader = None
    get_resilience_status = None

try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except ImportError:
    trace = None
    # Mock tracer for standalone testing
    class MockTracer:
        def start_as_current_span(self, name):
            return MockSpan()
    class MockSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, k, v): pass
    tracer = MockTracer()

try:
    from prometheus_client import Counter, Histogram, Gauge
except ImportError:
    # Mock metrics for standalone testing
    class MockMetric:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass
        def observe(self, *args): pass
        def set(self, *args): pass
    Counter = Histogram = Gauge = MockMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreativeDirector")


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

BRIEF_PHASE = Counter(
    'creative_director_phase_total',
    'Brief intake phase transitions',
    ['from_phase', 'to_phase']
)

BRIEF_COMPLETION = Counter(
    'creative_director_briefs_completed_total',
    'Successfully completed video briefs'
)

FIELD_CAPTURED = Counter(
    'creative_director_field_captured_total',
    'Fields captured during brief intake',
    ['field_name']
)

CONVERSATION_LENGTH = Histogram(
    'creative_director_conversation_turns',
    'Number of conversation turns to complete brief',
    buckets=[2, 4, 6, 8, 10, 15, 20]
)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class BriefPhase(str, Enum):
    """Conversation phases for video brief collection"""
    GREETING = "greeting"
    BUSINESS_CORE = "business_core"
    OFFERING = "offering"
    AUDIENCE = "audience"
    PAIN_POINTS = "pain_points"
    VIDEO_GOAL = "video_goal"
    TONE_STYLE = "tone_style"
    CTA = "call_to_action"
    SPECIAL_REQUESTS = "special_requests"
    CONFIRM = "confirm"
    COMPLETE = "complete"


class VideoGoal(str, Enum):
    """Primary video objectives"""
    AWARENESS = "awareness"
    LEADS = "leads"
    SALES = "sales"
    TRUST = "trust"
    LAUNCH = "launch"


class VideoTone(str, Enum):
    """Video tone/style options - maps to RAGNAROK semantic memory"""
    PROFESSIONAL = "professional"
    BOLD = "bold"
    FRIENDLY = "friendly"
    LUXURY = "luxury"
    URGENT = "urgent"


# Industry profiles from RAGNAROK's semantic memory
RAGNAROK_INDUSTRY_PROFILES = {
    "flooring": {
        "mood": "professional", "tempo": "moderate", "transition": "crossfade",
        "color_grade": "warm", "pacing": "moderate",
        "music_genre": "corporate_upbeat", "voice_style": "confident_warm"
    },
    "restaurant": {
        "mood": "inviting", "tempo": "relaxed", "transition": "crossfade",
        "color_grade": "warm_saturated", "pacing": "relaxed",
        "music_genre": "jazz_acoustic", "voice_style": "friendly_warm"
    },
    "technology": {
        "mood": "innovative", "tempo": "fast", "transition": "cut",
        "color_grade": "cool_contrast", "pacing": "fast",
        "music_genre": "electronic_upbeat", "voice_style": "energetic_modern"
    },
    "healthcare": {
        "mood": "trustworthy", "tempo": "slow", "transition": "crossfade",
        "color_grade": "clean_bright", "pacing": "slow",
        "music_genre": "soft_piano", "voice_style": "calm_reassuring"
    },
    "real_estate": {
        "mood": "aspirational", "tempo": "moderate", "transition": "crossfade",
        "color_grade": "bright_airy", "pacing": "moderate",
        "music_genre": "cinematic_light", "voice_style": "confident_warm"
    },
    "fitness": {
        "mood": "energetic", "tempo": "fast", "transition": "cut",
        "color_grade": "high_contrast", "pacing": "fast",
        "music_genre": "edm_motivational", "voice_style": "energetic_motivational"
    },
    "legal": {
        "mood": "authoritative", "tempo": "slow", "transition": "crossfade",
        "color_grade": "neutral", "pacing": "measured",
        "music_genre": "subtle_orchestral", "voice_style": "authoritative_calm"
    },
    "ecommerce": {
        "mood": "exciting", "tempo": "fast", "transition": "cut",
        "color_grade": "vibrant", "pacing": "fast",
        "music_genre": "upbeat_pop", "voice_style": "energetic_friendly"
    },
    "professional_services": {
        "mood": "trustworthy", "tempo": "moderate", "transition": "crossfade",
        "color_grade": "clean", "pacing": "moderate",
        "music_genre": "corporate_light", "voice_style": "professional_warm"
    },
}


# =============================================================================
# VIDEO BRIEF STATE
# =============================================================================

@dataclass
class VideoBriefState:
    """
    Complete video brief state for RAGNAROK input.
    Maps directly to RAGNAROK's semantic memory structure.
    """
    # Session tracking
    session_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    phase: BriefPhase = BriefPhase.GREETING
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Business identity
    business_name: Optional[str] = None
    industry: Optional[str] = None
    years_in_business: Optional[int] = None
    location: Optional[str] = None
    
    # Product/Service
    primary_offering: Optional[str] = None
    unique_selling_points: List[str] = field(default_factory=list)
    price_range: Optional[str] = None
    
    # Target audience
    target_demographic: Optional[str] = None
    pain_points: List[str] = field(default_factory=list)
    customer_desires: List[str] = field(default_factory=list)
    
    # Video strategy
    video_goal: Optional[VideoGoal] = None
    platform: Optional[str] = None  # youtube, tiktok, instagram, linkedin
    call_to_action: Optional[str] = None
    
    # Tone & style
    tone: Optional[VideoTone] = None
    brand_personality: List[str] = field(default_factory=list)
    
    # Creative direction
    must_include: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    competitor_reference: Optional[str] = None
    
    # Contact
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_name: Optional[str] = None

    # Production tracking (RAGNAROK integration)
    production_job_id: Optional[str] = None
    production_status: str = "not_started"  # not_started, queued, in_progress, script_review, complete, failed
    script_draft: Optional[str] = None
    script_approved: bool = False
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # Confirmation gate - user must confirm brief before production
    user_confirmed: bool = False

    # Metrics
    turns_count: int = 0
    extraction_confidence: float = 0.0
    
    def to_ragnarok_input(self) -> Dict[str, Any]:
        """
        Convert to RAGNAROK-compatible input format.
        Maps our schema to RAGNAROK's semantic memory structure.
        """
        # Get industry profile or use default
        industry_key = self._normalize_industry()
        profile = RAGNAROK_INDUSTRY_PROFILES.get(
            industry_key,
            RAGNAROK_INDUSTRY_PROFILES["professional_services"]
        )
        
        # Override profile based on explicit tone selection
        if self.tone:
            profile = self._apply_tone_override(profile)
        
        return {
            # Business context
            "business": {
                "name": self.business_name,
                "industry": self.industry,
                "offering": self.primary_offering,
                "usps": self.unique_selling_points,
            },
            # Audience
            "audience": {
                "demographic": self.target_demographic,
                "pain_points": self.pain_points,
                "desires": self.customer_desires,
            },
            # Video specs
            "video": {
                "goal": self.video_goal.value if self.video_goal else "awareness",
                "cta": self.call_to_action,
                "platform": self.platform or "youtube_1080p",
                "duration_seconds": 64,
            },
            # Creative direction (from RAGNAROK semantic memory)
            "creative": {
                "mood": profile["mood"],
                "tempo": profile["tempo"],
                "transition": profile["transition"],
                "color_grade": profile["color_grade"],
                "pacing": profile["pacing"],
                "music_genre": profile["music_genre"],
                "voice_style": profile["voice_style"],
            },
            # Constraints
            "constraints": {
                "must_include": self.must_include,
                "must_avoid": self.must_avoid,
                "competitor_style": self.competitor_reference,
            },
            # Metadata
            "metadata": {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "extraction_confidence": self.extraction_confidence,
                "turns": self.turns_count,
            }
        }
    
    def _normalize_industry(self) -> str:
        """Normalize industry string to RAGNAROK profile key"""
        if not self.industry:
            return "professional_services"
        
        industry_lower = self.industry.lower()
        
        # Direct matches
        for key in RAGNAROK_INDUSTRY_PROFILES:
            if key in industry_lower:
                return key
        
        # Fuzzy matching
        mappings = {
            "medical": "healthcare", "dental": "healthcare", "clinic": "healthcare",
            "lawyer": "legal", "attorney": "legal", "law firm": "legal",
            "gym": "fitness", "yoga": "fitness", "trainer": "fitness",
            "food": "restaurant", "cafe": "restaurant", "bar": "restaurant",
            "software": "technology", "saas": "technology", "tech": "technology",
            "shop": "ecommerce", "store": "ecommerce", "retail": "ecommerce",
            "property": "real_estate", "realtor": "real_estate", "homes": "real_estate",
            "floor": "flooring", "carpet": "flooring", "tile": "flooring",
        }
        
        for keyword, profile in mappings.items():
            if keyword in industry_lower:
                return profile
        
        return "professional_services"
    
    def _apply_tone_override(self, profile: Dict) -> Dict:
        """Apply explicit tone selection over industry defaults"""
        tone_overrides = {
            VideoTone.PROFESSIONAL: {"mood": "professional", "voice_style": "authoritative_calm"},
            VideoTone.BOLD: {"mood": "powerful", "tempo": "fast", "voice_style": "energetic_bold"},
            VideoTone.FRIENDLY: {"mood": "warm", "voice_style": "friendly_warm"},
            VideoTone.LUXURY: {"mood": "elegant", "tempo": "slow", "voice_style": "refined_subtle"},
            VideoTone.URGENT: {"mood": "urgent", "tempo": "fast", "voice_style": "energetic_urgent"},
        }
        
        override = tone_overrides.get(self.tone, {})
        return {**profile, **override}
    
    def get_completion_percentage(self) -> float:
        """Calculate brief completion percentage"""
        required_fields = [
            self.business_name,
            self.primary_offering,
            self.target_demographic,
            self.call_to_action,
            self.tone,
        ]
        optional_fields = [
            self.industry,
            bool(self.unique_selling_points),
            bool(self.pain_points),
            self.video_goal,
            self.contact_email,
        ]
        
        required_score = sum(1 for f in required_fields if f) / len(required_fields)
        optional_score = sum(1 for f in optional_fields if f) / len(optional_fields)
        
        return (required_score * 0.7) + (optional_score * 0.3)
    
    def get_missing_required_fields(self) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.business_name:
            missing.append("business_name")
        if not self.primary_offering:
            missing.append("primary_offering")
        if not self.target_demographic:
            missing.append("target_demographic")
        if not self.call_to_action:
            missing.append("call_to_action")
        if not self.tone:
            missing.append("tone")
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "phase": self.phase.value,
            "conversation_history": self.conversation_history,
            "business_name": self.business_name,
            "industry": self.industry,
            "years_in_business": self.years_in_business,
            "location": self.location,
            "primary_offering": self.primary_offering,
            "unique_selling_points": self.unique_selling_points,
            "price_range": self.price_range,
            "target_demographic": self.target_demographic,
            "pain_points": self.pain_points,
            "customer_desires": self.customer_desires,
            "video_goal": self.video_goal.value if self.video_goal else None,
            "platform": self.platform,
            "call_to_action": self.call_to_action,
            "tone": self.tone.value if self.tone else None,
            "brand_personality": self.brand_personality,
            "must_include": self.must_include,
            "must_avoid": self.must_avoid,
            "competitor_reference": self.competitor_reference,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "contact_name": self.contact_name,
            "production_job_id": self.production_job_id,
            "production_status": self.production_status,
            "script_draft": self.script_draft,
            "script_approved": self.script_approved,
            "video_url": self.video_url,
            "thumbnail_url": self.thumbnail_url,
            "turns_count": self.turns_count,
            "extraction_confidence": self.extraction_confidence,
            "completion_percentage": self.get_completion_percentage(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoBriefState":
        """Deserialize state from dictionary"""
        state = cls()
        state.session_id = data.get("session_id", "")
        state.created_at = data.get("created_at", datetime.utcnow().isoformat())
        state.phase = BriefPhase(data.get("phase", "greeting"))
        state.conversation_history = data.get("conversation_history", [])
        state.business_name = data.get("business_name")
        state.industry = data.get("industry")
        state.years_in_business = data.get("years_in_business")
        state.location = data.get("location")
        state.primary_offering = data.get("primary_offering")
        state.unique_selling_points = data.get("unique_selling_points", [])
        state.price_range = data.get("price_range")
        state.target_demographic = data.get("target_demographic")
        state.pain_points = data.get("pain_points", [])
        state.customer_desires = data.get("customer_desires", [])
        
        if data.get("video_goal"):
            state.video_goal = VideoGoal(data["video_goal"])
        if data.get("tone"):
            state.tone = VideoTone(data["tone"])
        
        state.platform = data.get("platform")
        state.call_to_action = data.get("call_to_action")
        state.brand_personality = data.get("brand_personality", [])
        state.must_include = data.get("must_include", [])
        state.must_avoid = data.get("must_avoid", [])
        state.competitor_reference = data.get("competitor_reference")
        state.contact_email = data.get("contact_email")
        state.contact_phone = data.get("contact_phone")
        state.contact_name = data.get("contact_name")
        state.production_job_id = data.get("production_job_id")
        state.production_status = data.get("production_status", "not_started")
        state.script_draft = data.get("script_draft")
        state.script_approved = data.get("script_approved", False)
        state.video_url = data.get("video_url")
        state.thumbnail_url = data.get("thumbnail_url")
        state.turns_count = data.get("turns_count", 0)
        state.extraction_confidence = data.get("extraction_confidence", 0.0)

        return state


# =============================================================================
# CREATIVE DIRECTOR SYSTEM PROMPT
# =============================================================================

CREATIVE_DIRECTOR_SYSTEM_PROMPT = """You are Alex, a creative director at a video agency.

RULES:
- 2 sentences max per response
- ONE question at a time
- Sound human: contractions, casual, warm
- Acknowledge what they said before next question
- NO lists, NO bullet points, NO walls of text

NEVER mention: pricing, costs, dollars, budget, timeline, days, hours, production process, API, how videos are made, conversion rates, CTR, metrics, tokens, discovery call

If asked about pricing: "Let's nail the creative first—our team handles pricing after."
If asked about timeline: "We move fast. What platforms are you targeting?"

GATHER (one at a time): business name, product/service, target audience, tone, key message, CTA, platform

When you have all info, confirm briefly: "Got everything. Ready to bring this to life?"

Required fields: business_name, primary_offering, target_demographic, call_to_action, tone
Only set is_complete=true when user confirms with "yes", "confirm", "looks good", etc.

JSON response format:
{"response": "your short message", "extracted_data": {}, "next_phase": "phase", "confidence": 0.8, "is_complete": false}
"""


# =============================================================================
# RESPONSE GUARDRAILS - Prevent info leaks & enforce brevity
# =============================================================================

BANNED_TERMS = [
    'cost', 'costs', 'pricing', 'price', 'budget', 'dollar', 'dollars',
    '$2,500', '$8,500', '$500', '$2.60', 'per production', 'fee', 'fees',
    '$449', '$899', '$1,699', '$3,199', '$299', '$1,499', '$999', '$1,599',
    'timeline', 'days', 'hours', 'weeks', '24-48', 'turnaround', '48-72',
    'API', 'token', 'tokens', 'RAGNAROK', 'pipeline', 'infrastructure',
    'CTR', 'conversion rate', 'view completion', 'engagement rate',
    'discovery call', 'consultation', 'sales team', 'account manager',
    '97.5%', '243s', 'success rate', 'production cycle'
]


def contains_banned_term(text: str) -> bool:
    """Check if response contains banned terms."""
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in BANNED_TERMS)


def sanitize_response(text: str) -> str:
    """Remove accidentally leaked internal information."""
    # Remove any dollar amounts
    text = re.sub(r'\$[\d,]+(?:\.\d{2})?', '', text)
    # Remove percentages with metrics
    text = re.sub(r'\d+(?:\.\d+)?%\s*(?:CTR|conversion|completion|engagement|success)', '', text)
    # Remove timeline specifics
    text = re.sub(r'\d+(?:-\d+)?\s*(?:hours?|days?|weeks?)', 'quickly', text)
    # Clean up double spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def enforce_brevity(text: str, max_sentences: int = 2) -> str:
    """Force 2-sentence maximum."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text
    truncated = ' '.join(sentences[:max_sentences])
    if not truncated.endswith(('.', '!', '?')):
        truncated += '.'
    return truncated


def enforce_single_question(text: str) -> str:
    """Keep only the first question if multiple exist."""
    if text.count('?') > 1:
        parts = text.split('?')
        return parts[0] + '?'
    return text


# =============================================================================
# CREATIVE DIRECTOR ORCHESTRATOR
# =============================================================================

class CreativeDirectorOrchestrator:
    """
    LangGraph-compatible orchestrator for video brief intake.
    Manages conversation state and extracts structured brief data.
    """
    
    # Extraction patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}')
    
    def __init__(self, anthropic_client=None, model: str = "claude-3-5-sonnet-20241022"):
        self.anthropic = anthropic_client
        self.model = model
        self.sessions: Dict[str, VideoBriefState] = {}
        logger.info(f"CreativeDirector initialized with model: {model}")
    
    async def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process a user message and return AI response with updated state.
        
        Args:
            session_id: Unique session identifier
            user_message: User's message text
            
        Returns:
            Dict with response, state, and metadata
        """
        with tracer.start_as_current_span("creative_director_process") as span:
            start_time = time.time()
            span.set_attribute("session_id", session_id)
            
            # Get or create session state
            state = self._get_or_create_session(session_id)
            old_phase = state.phase
            
            # Pre-extraction (regex patterns)
            self._pre_extract(user_message, state)
            
            # Add user message to history
            state.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": time.time()
            })
            state.turns_count += 1
            
            try:
                # Generate AI response
                ai_result = await self._generate_response(state, user_message)
                
                # Apply extracted data
                self._apply_extraction(ai_result.get("extracted_data", {}), state)
                
                # Update phase (with defensive validation)
                next_phase_str = ai_result.get("next_phase")
                if next_phase_str:
                    try:
                        # Validate it's a valid BriefPhase value
                        new_phase = BriefPhase(next_phase_str)
                        if new_phase != state.phase:
                            BRIEF_PHASE.labels(
                                from_phase=state.phase.value,
                                to_phase=new_phase.value
                            ).inc()
                            state.phase = new_phase
                    except ValueError as phase_error:
                        logger.warning(f"[PHASE_ERROR] Invalid phase value '{next_phase_str}': {phase_error}, keeping current phase")
                
                # Update confidence
                state.extraction_confidence = ai_result.get("confidence", 0.0)
                
                # Add AI response to history
                response_text = ai_result.get("response", "")
                state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": time.time()
                })
                
                # Check completion - WITH VALIDATION GATE
                is_complete = ai_result.get("is_complete", False)
                missing_fields = state.get_missing_required_fields()

                # VALIDATION GATE: Override Claude's is_complete if required fields are missing
                if is_complete and missing_fields:
                    logger.warning(f"[VALIDATION_GATE] Overriding is_complete=true - missing fields: {missing_fields}")
                    is_complete = False
                    # Claude tried to complete too early, keep gathering info

                # CONFIRMATION GATE: Detect user confirmation phrases
                confirmation_phrases = ["yes", "confirm", "looks good", "correct", "that's right", "proceed", "let's do it", "go ahead", "start production", "ready"]
                user_confirmed_now = any(phrase in user_message.lower() for phrase in confirmation_phrases)

                # Only set user_confirmed if all required fields are present AND user confirmed
                if user_confirmed_now and not missing_fields:
                    state.user_confirmed = True
                    logger.info(f"[CONFIRMATION_GATE] User confirmed brief - production can now proceed")

                # Final is_complete requires BOTH: no missing fields AND user confirmation
                ragnarok_ready = is_complete and state.user_confirmed and not missing_fields

                if ragnarok_ready:
                    state.phase = BriefPhase.COMPLETE
                    BRIEF_COMPLETION.inc()
                    CONVERSATION_LENGTH.observe(state.turns_count)
                    logger.info(f"[PRODUCTION_READY] Brief complete and confirmed - RAGNAROK ready")
                elif is_complete and not state.user_confirmed:
                    # Brief is complete but user hasn't confirmed yet
                    logger.info(f"[AWAITING_CONFIRMATION] Brief gathered, waiting for user confirmation")

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("latency_ms", latency_ms)
                span.set_attribute("phase", state.phase.value)
                span.set_attribute("completion", state.get_completion_percentage())
                span.set_attribute("user_confirmed", state.user_confirmed)

                return {
                    "response": response_text,
                    "state": state.to_dict(),
                    "is_complete": ragnarok_ready,  # Only truly complete when confirmed
                    "completion_percentage": state.get_completion_percentage(),
                    "missing_fields": missing_fields,
                    "user_confirmed": state.user_confirmed,
                    "ragnarok_ready": ragnarok_ready,
                    "ragnarok_input": state.to_ragnarok_input() if ragnarok_ready else None,
                    "latency_ms": latency_ms,
                }
                
            except Exception as e:
                import traceback
                full_traceback = traceback.format_exc()
                error_type = e.__class__.__name__
                error_str = str(e).lower()

                # =================================================================
                # DETAILED ERROR LOGGING (process_message level)
                # =================================================================
                logger.error(f"[PROCESS_ERROR] Type: {error_type}")
                logger.error(f"[PROCESS_ERROR] Message: {e}")
                logger.error(f"[PROCESS_ERROR] User input: {user_message[:100]}...")
                logger.error(f"[PROCESS_ERROR] Session phase: {state.phase.value}")
                logger.error(f"[PROCESS_ERROR] Knowledge enabled: {KNOWLEDGE_ENABLED}")
                logger.error(f"[PROCESS_ERROR] Full traceback:\n{full_traceback}")

                # Identify likely cause
                if "phase" in error_str or "enum" in error_str:
                    logger.error("[PROCESS_ERROR] LIKELY CAUSE: Invalid phase value from Claude")
                if "key" in error_str or "dict" in error_str:
                    logger.error("[PROCESS_ERROR] LIKELY CAUSE: Missing expected field in response")
                if "type" in error_str or "attribute" in error_str:
                    logger.error("[PROCESS_ERROR] LIKELY CAUSE: Wrong data type in response")

                span.set_attribute("error", str(e))
                span.set_attribute("traceback", full_traceback[:1000])
                latency_ms = (time.time() - start_time) * 1000

                # Use GracefulDegrader for meaningful fallback
                if RESILIENCE_ENABLED and GracefulDegrader:
                    emergency = GracefulDegrader.get_emergency_response(user_message)
                    response_text = emergency["response"]
                    logger.warning(f"[PROCESS_ERROR] Using emergency fallback: {emergency['category']}")
                else:
                    response_text = "I'd be happy to help. What would you like to know about Barrios A2I's services?"

                return {
                    "response": response_text,
                    "state": state.to_dict(),
                    "is_complete": False,
                    "completion_percentage": state.get_completion_percentage(),
                    "missing_fields": state.get_missing_required_fields(),
                    "ragnarok_ready": False,
                    "latency_ms": latency_ms,
                    "error_type": error_type,
                }
    
    def _get_or_create_session(self, session_id: str) -> VideoBriefState:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            state = VideoBriefState()
            state.session_id = session_id
            self.sessions[session_id] = state
            logger.info(f"Created new session: {session_id}")
        return self.sessions[session_id]
    
    def _pre_extract(self, message: str, state: VideoBriefState):
        """Pre-extract structured data using regex patterns"""
        # Email
        if not state.contact_email:
            emails = self.EMAIL_PATTERN.findall(message)
            if emails:
                state.contact_email = emails[0]
                FIELD_CAPTURED.labels(field_name="contact_email").inc()
        
        # Phone
        if not state.contact_phone:
            phones = self.PHONE_PATTERN.findall(message)
            if phones:
                state.contact_phone = phones[0]
                FIELD_CAPTURED.labels(field_name="contact_phone").inc()
    
    async def _generate_response(
        self,
        state: VideoBriefState,
        user_message: str
    ) -> Dict[str, Any]:
        """Generate AI response using Claude"""
        if not self.anthropic:
            # Mock response for testing
            return self._mock_response(state, user_message)
        
        # Build conversation for Claude
        messages = []
        for turn in state.conversation_history[-10:]:  # Last 10 turns
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        # Add current state context
        state_context = f"""
Current brief state:
- Phase: {state.phase.value}
- Completion: {state.get_completion_percentage():.0%}
- Business: {state.business_name or 'Unknown'}
- Offering: {state.primary_offering or 'Unknown'}
- Audience: {state.target_demographic or 'Unknown'}
- Missing required fields: {', '.join(state.get_missing_required_fields()) or 'None'}
"""

        # Detect conversation mode and inject relevant knowledge
        knowledge_context = ""
        if KNOWLEDGE_ENABLED and route_to_knowledge:
            detected_mode = route_to_knowledge(user_message)
            knowledge_context = get_knowledge_context(detected_mode)
            logger.debug(f"Detected mode: {detected_mode}, knowledge context size: {len(knowledge_context)}")

        # Query Website RAG for real-time business knowledge
        website_rag_context = ""
        if WEBSITE_RAG_ENABLED and get_website_rag:
            try:
                rag = get_website_rag()
                website_rag_context = rag.query(user_message, top_k=3)
                if website_rag_context:
                    logger.debug(f"Website RAG context retrieved: {len(website_rag_context)} chars")
            except Exception as e:
                logger.warning(f"Website RAG query failed: {e}")

        # Combine knowledge sources
        if website_rag_context:
            if knowledge_context:
                knowledge_context = knowledge_context + "\n\n## REAL-TIME WEBSITE KNOWLEDGE\n" + website_rag_context
            else:
                knowledge_context = "## REAL-TIME WEBSITE KNOWLEDGE\n" + website_rag_context

        # Build full system prompt with knowledge
        # =================================================================
        # TOKEN LIMIT SAFETY (Prevent token overflow errors)
        # =================================================================
        MAX_SYSTEM_PROMPT_CHARS = 15000  # ~4K tokens safety limit

        base_prompt = CREATIVE_DIRECTOR_SYSTEM_PROMPT + "\n\n" + state_context

        if knowledge_context:
            # Calculate how much space we have for knowledge
            knowledge_budget = MAX_SYSTEM_PROMPT_CHARS - len(base_prompt) - 500  # 500 char buffer
            if knowledge_budget > 0:
                if len(knowledge_context) > knowledge_budget:
                    logger.warning(f"[TOKEN_LIMIT] Knowledge context truncated: {len(knowledge_context)} -> {knowledge_budget} chars")
                    knowledge_context = knowledge_context[:knowledge_budget] + "\n[Knowledge truncated for token limit]"
                full_system_prompt = base_prompt + "\n\n## KNOWLEDGE CONTEXT\n" + knowledge_context
            else:
                logger.warning(f"[TOKEN_LIMIT] No budget for knowledge, base prompt: {len(base_prompt)} chars")
                full_system_prompt = base_prompt
        else:
            full_system_prompt = base_prompt

        # Final safety check
        if len(full_system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
            logger.warning(f"[TOKEN_LIMIT] System prompt still too long ({len(full_system_prompt)} chars), truncating")
            full_system_prompt = full_system_prompt[:MAX_SYSTEM_PROMPT_CHARS]

        # Define the Claude API call for retry wrapper
        async def make_claude_call():
            return await asyncio.to_thread(
                self.anthropic.messages.create,
                model=self.model,
                max_tokens=150,
                system=full_system_prompt,
                messages=messages + [{"role": "user", "content": user_message}]
            )

        try:
            # Use resilience module if available (retry + circuit breaker)
            if RESILIENCE_ENABLED and retry_with_backoff:
                circuit_breaker = get_claude_circuit_breaker()
                response = await retry_with_backoff(
                    fn=make_claude_call,
                    config=CLAUDE_RETRY_CONFIG,
                    circuit_breaker=circuit_breaker,
                    operation_name="claude_generate_response"
                )
            else:
                # Direct call without resilience
                response = await make_claude_call()

            # =================================================================
            # DEFENSIVE RESPONSE EXTRACTION (Fix for ~50% failure rate)
            # =================================================================
            try:
                if response and hasattr(response, 'content') and response.content:
                    if len(response.content) > 0:
                        content_block = response.content[0]
                        if hasattr(content_block, 'text'):
                            result_text = content_block.text
                        else:
                            logger.warning(f"[RESPONSE_EXTRACT] Unexpected content block type: {type(content_block)}")
                            result_text = str(content_block)
                    else:
                        logger.warning("[RESPONSE_EXTRACT] Claude returned empty content array")
                        result_text = json.dumps({
                            "response": "I'm here to help. Could you tell me more about what you're looking for?",
                            "extracted_data": {},
                            "next_phase": state.phase.value,
                            "confidence": 0.5,
                            "is_complete": False
                        })
                else:
                    logger.warning(f"[RESPONSE_EXTRACT] Unexpected response format: {type(response)}")
                    result_text = json.dumps({
                        "response": "I'm here to help. What would you like to know about Barrios A2I?",
                        "extracted_data": {},
                        "next_phase": state.phase.value,
                        "confidence": 0.5,
                        "is_complete": False
                    })
            except (IndexError, AttributeError, TypeError) as extract_error:
                logger.error(f"[RESPONSE_EXTRACT] Extraction error: {extract_error}, response type: {type(response)}")
                result_text = json.dumps({
                    "response": "I'd be happy to help. What questions do you have about our services?",
                    "extracted_data": {},
                    "next_phase": state.phase.value,
                    "confidence": 0.5,
                    "is_complete": False
                })

            original_text = result_text  # Keep original for fallback

            # ROBUST JSON EXTRACTION: Find JSON block anywhere in response
            # This handles cases where Claude adds text before/after the JSON
            json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', result_text)
            if json_match:
                result_text = json_match.group(1)
            else:
                # Try to find bare JSON object with expected keys
                json_match = re.search(r'\{\s*"response"[\s\S]*?"is_complete"\s*:\s*(true|false)[\s\S]*?\}', result_text)
                if json_match:
                    result_text = json_match.group(0)
                else:
                    # Last resort: find any JSON-like object
                    json_match = re.search(r'\{[^{}]*"response"[^{}]*\}', result_text)
                    if json_match:
                        result_text = json_match.group(0)

            # Clean any remaining markdown markers
            result_text = re.sub(r'^```json\s*', '', result_text.strip())
            result_text = re.sub(r'\s*```$', '', result_text.strip())

            try:
                parsed = json.loads(result_text)

                # CLEAN UP: Remove any JSON that leaked into the response text
                if "response" in parsed:
                    clean_response = parsed["response"]
                    # Remove JSON blocks that might have leaked into the response
                    clean_response = re.sub(r'```json[\s\S]*?```', '', clean_response).strip()
                    clean_response = re.sub(r'\{[^{}]*"response"[^{}]*\}', '', clean_response).strip()

                    # =================================================================
                    # GUARDRAILS: No pricing leaks, 2 sentences max, single question
                    # =================================================================
                    if contains_banned_term(clean_response):
                        logger.warning(f"[GUARDRAIL] Banned term detected, replacing response")
                        clean_response = "Tell me more about what you're looking for."
                    clean_response = sanitize_response(clean_response)
                    clean_response = enforce_brevity(clean_response, max_sentences=2)
                    clean_response = enforce_single_question(clean_response)

                    parsed["response"] = clean_response

                # =================================================================
                # RESPONSE VALIDATION (Ensure valid response before returning)
                # =================================================================
                response_text = parsed.get("response", "")

                # Check for empty or too-short response
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning(f"[RESPONSE_VALIDATE] Empty/short response: '{response_text[:50] if response_text else 'None'}'")
                    parsed["response"] = "I'd be happy to help you learn about Barrios A2I. What would you like to know?"

                # Check for error markers in response (but allow "rate" for pricing)
                elif "error" in response_text.lower()[:50] and "rate" not in response_text.lower()[:50]:
                    logger.warning(f"[RESPONSE_VALIDATE] Response may contain error: {response_text[:100]}")

                # Ensure required fields exist with defaults
                if "extracted_data" not in parsed:
                    parsed["extracted_data"] = {}
                if "next_phase" not in parsed:
                    parsed["next_phase"] = state.phase.value
                if "confidence" not in parsed:
                    parsed["confidence"] = 0.7
                if "is_complete" not in parsed:
                    parsed["is_complete"] = False

                return parsed

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, text preview: {result_text[:300]}")
                # Fallback: Extract readable text and continue conversation
                fallback_text = original_text.split('```')[0].strip() if '```' in original_text else original_text[:500]
                # Remove any JSON-looking content from fallback
                fallback_text = re.sub(r'\{[\s\S]*?\}', '', fallback_text).strip()
                if not fallback_text:
                    fallback_text = "I understand. Let me continue with the next question."

                return {
                    "response": fallback_text,
                    "extracted_data": {},
                    "next_phase": state.phase.value,
                    "confidence": 0.5,
                    "is_complete": False
                }

        except Exception as e:
            import traceback
            full_tb = traceback.format_exc()

            # =================================================================
            # DETAILED ERROR LOGGING (Diagnose root cause)
            # =================================================================
            error_str = str(e).lower()
            error_type = e.__class__.__name__

            logger.error(f"[CHAT_ERROR] Exception type: {error_type}")
            logger.error(f"[CHAT_ERROR] Exception message: {e}")
            logger.error(f"[CHAT_ERROR] User message: {user_message[:100]}...")
            logger.error(f"[CHAT_ERROR] Session phase: {state.phase.value}")
            logger.error(f"[CHAT_ERROR] System prompt size: {len(full_system_prompt)} chars")
            logger.error(f"[CHAT_ERROR] Messages count: {len(messages)}")
            logger.error(f"[CHAT_ERROR] Knowledge enabled: {KNOWLEDGE_ENABLED}")
            logger.error(f"[CHAT_ERROR] Full traceback:\n{full_tb}")

            # Identify specific known issues for debugging
            if "index" in error_str or "list" in error_str:
                logger.error("[CHAT_ERROR] LIKELY CAUSE: response.content indexing issue")
            if "attribute" in error_str:
                logger.error("[CHAT_ERROR] LIKELY CAUSE: response format mismatch")
            if "token" in error_str or "limit" in error_str:
                logger.error("[CHAT_ERROR] LIKELY CAUSE: token limit exceeded")
            if "json" in error_str or "decode" in error_str:
                logger.error("[CHAT_ERROR] LIKELY CAUSE: JSON parsing failure")
            if "timeout" in error_str or "connection" in error_str:
                logger.error("[CHAT_ERROR] LIKELY CAUSE: Network/timeout issue")

            # Graceful degradation: Return emergency response instead of raising
            if RESILIENCE_ENABLED and GracefulDegrader:
                emergency = GracefulDegrader.get_emergency_response(user_message)
                logger.warning(f"[CHAT_ERROR] Using emergency fallback: {emergency['category']}")
                return {
                    "response": emergency["response"],
                    "extracted_data": {},
                    "next_phase": state.phase.value,
                    "confidence": 0.0,
                    "is_complete": False,
                    "quality_tier": emergency["quality_tier"],
                    "error_type": error_type,
                }

            raise
    
    def _mock_response(self, state: VideoBriefState, user_message: str) -> Dict:
        """Mock response for testing without API"""
        phase_responses = {
            BriefPhase.GREETING: {
                "response": "Welcome! I'm your Creative Director at Barrios A2I. I'll help you create your 64-second AI video commercial. Let's start with your business - what's your company name and what industry are you in?",
                "next_phase": "business_core",
            },
            BriefPhase.BUSINESS_CORE: {
                "response": "Great! Now tell me about your main product or service - what do you offer, and what makes it unique?",
                "next_phase": "offering",
            },
            BriefPhase.OFFERING: {
                "response": "Excellent! Who's your ideal customer? Describe them - demographics, role, situation.",
                "next_phase": "audience",
            },
            BriefPhase.AUDIENCE: {
                "response": "What problems or pain points do you solve for your customers?",
                "next_phase": "pain_points",
            },
            BriefPhase.PAIN_POINTS: {
                "response": "What's the primary goal of this video? (Awareness, Lead Generation, Sales, Building Trust, or Product Launch)",
                "next_phase": "video_goal",
            },
            BriefPhase.VIDEO_GOAL: {
                "response": "What tone should the video have? Professional & Corporate, Bold & Energetic, Friendly & Approachable, Luxury & Premium, or Urgent & Action-Oriented?",
                "next_phase": "tone_style",
            },
            BriefPhase.TONE_STYLE: {
                "response": "After watching, what should viewers do? What's your call-to-action?",
                "next_phase": "cta",
            },
            BriefPhase.CTA: {
                "response": "Are there any specific phrases, taglines, or offers that MUST be included? Anything we should avoid mentioning?",
                "next_phase": "special_requests",
            },
            BriefPhase.SPECIAL_REQUESTS: {
                "response": f"Perfect! Here's your video brief:\n\n**Business:** {state.business_name}\n**Offering:** {state.primary_offering}\n**Audience:** {state.target_demographic}\n**Goal:** {state.video_goal}\n**Tone:** {state.tone}\n**CTA:** {state.call_to_action}\n\nDoes this look correct?",
                "next_phase": "confirm",
            },
            BriefPhase.CONFIRM: {
                "response": "Excellent! Your brief is confirmed and queued for RAGNAROK. You'll receive your 64-second commercial within 24-48 hours!",
                "is_complete": True,
                "next_phase": "complete",
            },
        }
        
        base = phase_responses.get(state.phase, {
            "response": "Let me continue gathering your brief details...",
            "next_phase": state.phase.value,
        })
        
        return {
            "response": base.get("response", ""),
            "extracted_data": {},
            "next_phase": base.get("next_phase", state.phase.value),
            "confidence": 0.8,
            "is_complete": base.get("is_complete", False),
        }
    
    def _apply_extraction(self, extracted: Dict[str, Any], state: VideoBriefState):
        """Apply extracted data to state"""
        field_mapping = {
            "business_name": "business_name",
            "industry": "industry",
            "years_in_business": "years_in_business",
            "location": "location",
            "primary_offering": "primary_offering",
            "price_range": "price_range",
            "target_demographic": "target_demographic",
            "call_to_action": "call_to_action",
            "platform": "platform",
            "competitor_reference": "competitor_reference",
            "contact_email": "contact_email",
            "contact_phone": "contact_phone",
            "contact_name": "contact_name",
        }
        
        for json_key, attr_name in field_mapping.items():
            if json_key in extracted and extracted[json_key]:
                old_value = getattr(state, attr_name)
                if not old_value:
                    setattr(state, attr_name, extracted[json_key])
                    FIELD_CAPTURED.labels(field_name=attr_name).inc()
        
        # Handle array fields
        array_fields = [
            "unique_selling_points", "pain_points", "customer_desires",
            "brand_personality", "must_include", "must_avoid"
        ]
        for field in array_fields:
            if field in extracted and extracted[field]:
                current = getattr(state, field)
                for item in extracted[field]:
                    if item not in current:
                        current.append(item)
                        FIELD_CAPTURED.labels(field_name=field).inc()
        
        # Handle enums
        if "video_goal" in extracted and extracted["video_goal"]:
            try:
                state.video_goal = VideoGoal(extracted["video_goal"])
                FIELD_CAPTURED.labels(field_name="video_goal").inc()
            except ValueError:
                pass
        
        if "tone" in extracted and extracted["tone"]:
            try:
                state.tone = VideoTone(extracted["tone"])
                FIELD_CAPTURED.labels(field_name="tone").inc()
            except ValueError:
                pass
    
    def get_session(self, session_id: str) -> Optional[VideoBriefState]:
        """Get session state by ID"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def export_ragnarok_brief(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export completed brief for RAGNAROK"""
        state = self.sessions.get(session_id)
        if not state or state.phase != BriefPhase.COMPLETE:
            return None
        return state.to_ragnarok_input()


# =============================================================================
# FACTORY & VALIDATION
# =============================================================================

def create_creative_director(
    anthropic_client=None,
    model: str = "claude-3-5-sonnet-20241022"
) -> CreativeDirectorOrchestrator:
    """Factory function to create Creative Director instance"""
    return CreativeDirectorOrchestrator(
        anthropic_client=anthropic_client,
        model=model
    )


def validate_module() -> Dict[str, bool]:
    """Validate module components"""
    checks = {
        "VideoBriefState_init": False,
        "BriefPhase_enum": False,
        "ragnarok_profiles": False,
        "orchestrator_init": False,
        "state_serialization": False,
        "ragnarok_output": False,
    }
    
    try:
        state = VideoBriefState()
        checks["VideoBriefState_init"] = True
        
        phases = list(BriefPhase)
        checks["BriefPhase_enum"] = len(phases) == 11
        
        checks["ragnarok_profiles"] = len(RAGNAROK_INDUSTRY_PROFILES) >= 6
        
        orchestrator = CreativeDirectorOrchestrator()
        checks["orchestrator_init"] = True
        
        state.business_name = "Test Corp"
        state.primary_offering = "AI Services"
        state_dict = state.to_dict()
        restored = VideoBriefState.from_dict(state_dict)
        checks["state_serialization"] = restored.business_name == "Test Corp"
        
        ragnarok_input = state.to_ragnarok_input()
        checks["ragnarok_output"] = "business" in ragnarok_input and "creative" in ragnarok_input
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
    
    return checks


if __name__ == "__main__":
    results = validate_module()
    passed = sum(1 for v in results.values() if v)
    print(f"\n🎬 Creative Director Validation: {passed}/{len(results)}")
    for check, result in results.items():
        print(f"  {'✅' if result else '❌'} {check}")
