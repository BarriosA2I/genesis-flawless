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


# =============================================================================
# TONE EXTRACTION - Enhanced keyword mapping for fallback extraction
# =============================================================================

TONE_KEYWORDS = {
    # Professional/Corporate
    'professional': 'professional',
    'trustworthy': 'professional',
    'corporate': 'professional',
    'business': 'professional',
    'polished': 'professional',
    'sophisticated': 'professional',
    'credible': 'professional',
    'formal': 'professional',
    'serious': 'professional',
    'authoritative': 'professional',

    # Energetic/Dynamic
    'energetic': 'energetic',
    'exciting': 'energetic',
    'dynamic': 'energetic',
    'vibrant': 'energetic',
    'lively': 'energetic',
    'active': 'energetic',
    'powerful': 'energetic',

    # Urgent/Time-Sensitive
    'urgent': 'urgent',
    'time-sensitive': 'urgent',
    'limited': 'urgent',
    'now': 'urgent',
    'hurry': 'urgent',
    'fast': 'urgent',
    'immediate': 'urgent',

    # Friendly/Warm
    'friendly': 'friendly',
    'warm': 'friendly',
    'approachable': 'friendly',
    'welcoming': 'friendly',
    'personable': 'friendly',
    'inviting': 'friendly',

    # Casual/Relaxed
    'casual': 'casual',
    'relaxed': 'casual',
    'laid-back': 'casual',
    'easygoing': 'casual',
    'conversational': 'casual',

    # Modern/Innovative
    'modern': 'modern',
    'sleek': 'modern',
    'contemporary': 'modern',
    'innovative': 'modern',
    'cutting-edge': 'modern',
    'fresh': 'modern',

    # Inspirational/Motivational
    'inspirational': 'inspirational',
    'motivational': 'inspirational',
    'uplifting': 'inspirational',
    'empowering': 'inspirational',
    'encouraging': 'inspirational',

    # Humorous/Fun
    'humorous': 'humorous',
    'fun': 'humorous',
    'playful': 'humorous',
    'quirky': 'humorous',
    'witty': 'humorous',
    'lighthearted': 'humorous',

    # Bold/Confident
    'bold': 'bold',
    'confident': 'bold',
    'assertive': 'bold',
    'strong': 'bold',
    'daring': 'bold',

    # Emotional/Heartfelt
    'emotional': 'emotional',
    'heartfelt': 'emotional',
    'touching': 'emotional',
    'sincere': 'emotional',
    'authentic': 'emotional',

    # Luxurious/Premium
    'luxury': 'luxurious',
    'luxurious': 'luxurious',
    'premium': 'luxurious',
    'elegant': 'luxurious',
    'exclusive': 'luxurious',
    'high-end': 'luxurious',
}


def extract_tone_fallback(message: str, current_tone: Optional[str] = None) -> Optional[str]:
    """
    Fallback tone extraction using keyword matching and pattern detection.

    Called when LLM extraction misses tone from phrases like:
    - "Keep it professional"
    - "Make it feel energetic"
    - "I want a modern vibe"
    - "Should be trustworthy and serious"

    Args:
        message: User's message to extract tone from
        current_tone: Already extracted tone (if any)

    Returns:
        Normalized tone value or None
    """
    if current_tone:
        return normalize_tone(current_tone)

    message_lower = message.lower()

    # Pattern 1: Explicit tone directives - "keep it X", "make it X", etc.
    tone_directive_patterns = [
        r'keep\s+it\s+(\w+(?:\s+and\s+\w+)?)',
        r'make\s+it\s+(?:feel\s+)?(\w+(?:\s+and\s+\w+)?)',
        r'should\s+be\s+(\w+(?:\s+and\s+\w+)?)',
        r'want\s+(?:it\s+)?(?:to\s+be\s+)?(\w+(?:\s+and\s+\w+)?)',
        r'(\w+)\s+(?:tone|vibe|feel|style|mood)',
        r'going\s+for\s+(?:a\s+)?(\w+)',
        r'needs?\s+to\s+(?:feel|be)\s+(\w+)',
    ]

    for pattern in tone_directive_patterns:
        matches = re.findall(pattern, message_lower)
        for match in matches:
            # Check each word in the match (handles "professional and trustworthy")
            words = match.replace(' and ', ' ').split()
            for word in words:
                word = word.strip()
                if word in TONE_KEYWORDS:
                    return TONE_KEYWORDS[word]

    # Pattern 2: Direct keyword scan (fallback)
    # Find all matching keywords and return the first one that appears
    found_tones = []
    for keyword, tone_value in TONE_KEYWORDS.items():
        if keyword in message_lower:
            # Store position and tone
            found_tones.append((message_lower.index(keyword), tone_value))

    if found_tones:
        # Return the tone whose keyword appears first in the message
        found_tones.sort(key=lambda x: x[0])
        return found_tones[0][1]

    return None


def normalize_tone(tone: str) -> str:
    """
    Normalize tone value to canonical VideoTone enum values.

    Maps various tone descriptions to standard categories:
    - professional, energetic, urgent, friendly, casual, modern,
      inspirational, humorous, bold, emotional, luxurious
    """
    if not tone:
        return "professional"  # Default

    tone_lower = tone.lower().strip()

    # Direct enum match
    if tone_lower in ['professional', 'energetic', 'urgent', 'friendly', 'bold', 'luxury']:
        return tone_lower

    # Use TONE_KEYWORDS for normalization
    if tone_lower in TONE_KEYWORDS:
        return TONE_KEYWORDS[tone_lower]

    # Fallback to professional
    return "professional"


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
        """Calculate brief completion percentage

        Returns 100% (1.0) when all required fields are filled.
        Optional fields don't affect completion percentage.
        """
        required_fields = [
            self.business_name,
            self.primary_offering,
            self.target_demographic,
            self.call_to_action,
            self.tone,
        ]

        # Return completion based only on required fields (0.0 to 1.0)
        required_score = sum(1 for f in required_fields if f) / len(required_fields)
        return required_score
    
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
        # Build brief_data dict with all extracted fields for frontend metadata
        brief_data = {}
        if self.business_name:
            brief_data["business_name"] = self.business_name
        if self.primary_offering:
            brief_data["primary_offering"] = self.primary_offering
        if self.target_demographic:
            brief_data["target_demographic"] = self.target_demographic
        if self.call_to_action:
            brief_data["call_to_action"] = self.call_to_action
        if self.tone:
            brief_data["tone"] = self.tone.value if hasattr(self.tone, 'value') else self.tone

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
            "brief_data": brief_data,  # For frontend metadata.extracted_fields
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

INTAKE_SYSTEM_PROMPT = """You are the Creative Director at Barrios A2I.

## CRITICAL RULE - READ THIS FIRST
**EVERY response you give MUST end with a question mark (?)**
If your response doesn't end with "?" you are WRONG. Fix it.

## YOUR JOB
Collect 5 pieces of information to create a video:
1. business_name - Their company name
2. primary_offering - What they sell/do
3. target_demographic - Who they're trying to reach
4. call_to_action - What viewers should do
5. tone - Video style (professional, fun, luxurious, etc.)

## RESPONSE FORMAT
Every response = 1 short acknowledgment + 1 specific question

## QUESTION TO ASK (based on what's missing)
- Missing business_name → "What's the name of your business?"
- Missing primary_offering → "What product or service do you want to highlight?"
- Missing target_demographic → "Who are you trying to reach?"
- Missing call_to_action → "What should viewers do after watching?"
- Missing tone → "What vibe - professional, fun, or luxurious?"

## EXAMPLES

User: "hello"
You: "Hello! Let's create a great video. What's the name of your business?"

User: "hi"
You: "Hi there! What business are we making this video for?"

User: "I want a video"
You: "Awesome! What's your business called?"

User: "My business is Glamour Studio"
You: "Great name! What does Glamour Studio specialize in?"

## REQUIRED FIELDS (must collect all 5)
1. business_name - Company/brand name
2. primary_offering - Main product or service to promote
3. target_demographic - Who they're trying to reach
4. call_to_action - What viewers should do (visit website, call, buy)
5. tone - Desired style (professional, friendly, energetic, etc.)

## RESPONSE RULES
1. Keep responses under 2 sentences
2. ALWAYS end with a question about the NEXT missing field
3. Do NOT be passive ("Let me know when you're ready")
4. Do NOT just acknowledge without asking the next question

## QUESTION TEMPLATES

For business_name: "What's the name of your business?"

For primary_offering: "What product or service would you like to highlight in this video?"

For target_demographic: "Who are you trying to reach with this video?"

For call_to_action: "What should viewers do after watching - call you, visit your website, or make a purchase?"

For tone: "What vibe should this video have - professional, friendly, energetic, or luxurious?"

## VISUAL ASSETS - Ask AFTER all 5 main fields are collected

Once you have ALL 5 fields (business_name, primary_offering, target_demographic, call_to_action, tone), ask about visual assets based on the commercial type:

**DETECT COMMERCIAL TYPE from their messages:**

BRAND AWARENESS signals (gaining clients, promoting business):
- "gain clients", "get clients", "more clients", "new clients"
- "grow business", "promote my business", "build awareness"
- "Facebook reel", "social media", "reach people", "reach locals"
- "get customers", "attract customers", "find customers"

PRODUCT SALES signals (selling specific items):
- "sell", "selling", "sales"
- "product", "products", "item", "items"
- "buy", "purchase", "order"
- "shop", "store", "Etsy", "Amazon", "Shopify"
- "handmade", "homemade", "custom made"

**ASK FOR LOGO (when brand awareness detected):**
"Great! Do you have a business logo you'd like featured in the video? You can upload it using the 📎 button below the chat."

**ASK FOR PRODUCT PHOTO (when product sales detected):**
"Awesome! Could you upload a photo of your [product name]? Click the 📎 button below - great product shots make videos much more effective!"

**IF THEY SAY NO or DON'T HAVE ONE:**
"No problem at all! We can create a professional video without it. Ready to proceed to video creation?"

**IF THEY UPLOAD:**
"Perfect, got it! Your [logo/photo] looks great. Ready to create your video? Just confirm and we'll get started!"

## EXAMPLE CONVERSATIONS

### Example 1: Brand Awareness (Facebook reel for clients)
User: "I want a Facebook reel to get more clients for my salon"
You: "A Facebook reel for your salon - great choice! What's the name of your salon?"

User: "Glamour Studio"
You: "Love it! What services does Glamour Studio specialize in that you want to highlight?"

User: "Hair coloring and styling"
You: "Perfect. Who's your ideal client - what age group or style?"

User: "Women 25-45 who want trendy looks"
You: "Got it. What should viewers do - book online, call, or walk in?"

User: "Book through our website"
You: "And what vibe should this have - luxurious, fun, edgy?"

User: "Luxurious and upscale"
You: "Perfect! Do you have a logo for Glamour Studio? You can upload it using the 📎 button below."

User: "Yes here it is" [uploads]
You: "Beautiful logo! Ready to create your video? Just say yes to confirm!"

### Example 2: Product Sales (selling handmade items)
User: "I want to sell my handmade candles"
You: "Handmade candles - love it! What's the name of your candle business?"

User: "Cozy Glow Candles"
You: "Great name! What makes your candles special - scents, ingredients, style?"

User: "All natural soy with unique scents"
You: "Nice! Who's your ideal customer for these candles?"

User: "People who love home decor and self-care"
You: "Perfect audience. Where should they buy - your website, Etsy, local store?"

User: "From my Etsy shop"
You: "Got it. What vibe - cozy and relaxing, modern and chic?"

User: "Cozy and warm"
You: "Perfect for candles! Could you upload a photo of your best-selling candle? Click 📎 below - great product photos really boost sales!"

User: "I don't have good photos yet"
You: "No problem! We can still create a beautiful video. Ready to proceed?"

## CRITICAL RULES
- Collect ALL 5 main fields FIRST before asking about logo/photos
- DETECT the type: brand awareness → logo, product sales → product photo
- ALWAYS mention the 📎 button when asking for uploads
- Keep driving forward - never be passive
- If they don't have assets, reassure them and proceed

=== DATA EXTRACTION RULES ===
You MUST extract information from EVERY user message into extracted_data.

Field mapping:
- business_name: Company name, brand name, "I run X", "my company X"
- primary_offering: What they sell, service, product, "we make X", "we sell X"
- target_demographic: Target customers, audience, "for X", "targeting X"
- call_to_action: What viewers should do, "sign up", "buy now", "learn more"
- tone: Professional, friendly, energetic, casual, etc.

Example extractions:
User: "I'm Gary from TechStart" → extracted_data: {"business_name": "TechStart"}
User: "We sell SaaS tools for startups" → extracted_data: {"primary_offering": "SaaS tools for startups"}
User: "1. TechStart 2. SaaS 3. Founders 4. Sign up" → extracted_data: {"business_name": "TechStart", "primary_offering": "SaaS", "target_demographic": "Founders", "call_to_action": "Sign up"}

NEVER leave extracted_data empty if the user provided ANY relevant information.

Only set is_complete=true when user confirms with "yes", "confirm", "looks good", etc.

JSON response format:
{
  "response": "your short 1-2 sentence message ending with a question about the NEXT missing field",
  "extracted_data": {
    "business_name": "extracted business name or null",
    "primary_offering": "extracted product/service or null",
    "target_demographic": "extracted target audience or null",
    "call_to_action": "extracted CTA or null",
    "tone": "extracted tone or null"
  },
  "next_phase": "phase",
  "confidence": 0.8,
  "is_complete": false
}
"""

# Keep old name as alias for backwards compatibility during deployment
CREATIVE_DIRECTOR_SYSTEM_PROMPT = INTAKE_SYSTEM_PROMPT


# =============================================================================
# FEW-SHOT EXAMPLES - Inject into first user message for better pattern matching
# =============================================================================

FEW_SHOT_EXAMPLES = """
<examples>
  <example>
    <user>hello</user>
    <assistant>{"response": "Hello! Let's create a great video. What's the name of your business?", "extracted_data": {}, "next_phase": "greeting", "confidence": 1.0, "is_complete": false}</assistant>
  </example>

  <example>
    <user>hi</user>
    <assistant>{"response": "Hi there! What business are we making this video for?", "extracted_data": {}, "next_phase": "greeting", "confidence": 1.0, "is_complete": false}</assistant>
  </example>

  <example>
    <user>I want a video</user>
    <assistant>{"response": "Awesome! What's your business called?", "extracted_data": {}, "next_phase": "greeting", "confidence": 1.0, "is_complete": false}</assistant>
  </example>

  <example>
    <user>I run a fitness studio called FitLife</user>
    <assistant>{"response": "Nice! Who's your ideal customer?", "extracted_data": {"business_name": "FitLife", "primary_offering": "fitness studio"}}</assistant>
  </example>

  <example>
    <user>We target busy professionals who want quick workouts</user>
    <assistant>{"response": "Got it. What should viewers do after watching?", "extracted_data": {"target_demographic": "busy professionals who want quick workouts"}}</assistant>
  </example>

  <example>
    <user>Book a free class</user>
    <assistant>{"response": "Perfect. What tone—energetic, professional, friendly?", "extracted_data": {"call_to_action": "Book a free class"}}</assistant>
  </example>

  <example>
    <user>Professional and motivating</user>
    <assistant>{"response": "Got everything. Ready to bring this to life?", "extracted_data": {"tone": "professional"}}</assistant>
  </example>
</examples>
"""


def inject_few_shot_examples(user_message: str, is_first_message: bool) -> str:
    """Inject few-shot examples into first user message only."""
    if is_first_message:
        return f"{FEW_SHOT_EXAMPLES}\n\n{user_message}"
    return user_message


# =============================================================================
# SESSION STATE MANAGEMENT - Track brief in backend, prevent drift
# =============================================================================

class BriefSessionState:
    """
    Track extracted brief fields in backend dict, not system prompt.
    This prevents instruction drift over long conversations.
    """

    REQUIRED_FIELDS = ["business_name", "product", "audience", "tone", "cta"]
    ALL_FIELDS = ["business_name", "product", "audience", "tone", "key_message", "cta", "platform"]

    def __init__(self):
        self.fields = {field: None for field in self.ALL_FIELDS}
        self.conversation_count = 0
        self.is_first_message = True
        self.awaiting_confirmation = False
        self.pricing_ask_count = 0
        self.last_message_time = None
        self.sentiment_history = []
        self.mode = "standard"  # Phase 2: "standard" or "fast_track"

    def update_field(self, field: str, value: str) -> None:
        """Update a brief field."""
        if field in self.fields:
            self.fields[field] = value

    def get_filled_fields(self) -> dict:
        """Get only fields that have values."""
        return {k: v for k, v in self.fields.items() if v}

    def get_missing_fields(self) -> list:
        """Get fields that still need values."""
        return [k for k, v in self.fields.items() if not v]

    def get_missing_required(self) -> list:
        """Get required fields that are missing."""
        return [f for f in self.REQUIRED_FIELDS if not self.fields.get(f)]

    def is_complete(self) -> bool:
        """Check if all required fields are gathered."""
        return len(self.get_missing_required()) == 0

    def completion_percentage(self) -> int:
        """Calculate completion percentage for progress bar."""
        filled = sum(1 for v in self.fields.values() if v)
        return int((filled / len(self.fields)) * 100)

    def get_context_injection(self) -> str:
        """
        Generate context string to inject into Claude messages.
        This keeps Claude aware of what's been gathered without relying on system prompt.
        """
        filled = self.get_filled_fields()
        missing = self.get_missing_fields()

        if not filled:
            return ""

        context_parts = []
        for k, v in filled.items():
            context_parts.append(f"{k.replace('_', ' ').title()}: {v}")

        return f"\n---\n*Brief so far: {', '.join(context_parts)}. Still need: {', '.join(missing)}*"

    def generate_summary(self) -> str:
        """Generate human-readable brief summary for confirmation."""
        f = self.fields

        summary = f"""Here's what I've got:

**Business:** {f.get('business_name') or 'Not specified'} — {f.get('product') or 'Not specified'}
**Audience:** {f.get('audience') or 'Not specified'}
**Tone:** {f.get('tone') or 'Not specified'}
**Key Message:** {f.get('key_message') or 'Not specified'}
**Call-to-Action:** {f.get('cta') or 'Not specified'}
**Platform:** {f.get('platform') or 'Not specified'}

Does this look right? Say "yes" to start your video, or tell me what to change."""

        return summary.strip()


# In-memory session store (use Redis in production for multi-instance)
BRIEF_SESSIONS: dict = {}


def get_or_create_brief_session(session_id: str) -> BriefSessionState:
    """Get existing session or create new one."""
    if session_id not in BRIEF_SESSIONS:
        BRIEF_SESSIONS[session_id] = BriefSessionState()
    return BRIEF_SESSIONS[session_id]


# =============================================================================
# PHASE 2: PRODUCTION STATUS TRACKING - Real-time video generation progress
# =============================================================================

class ProductionStep(Enum):
    """Steps in the RAGNAROK video production pipeline."""
    QUEUED = "queued"
    SCRIPT_GENERATION = "script"
    VISUAL_SELECTION = "visuals"
    VOICEOVER_GENERATION = "voice"
    VIDEO_EDITING = "edit"
    UPLOAD = "upload"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProductionStatus:
    """Track production status for a session."""
    session_id: str
    brief: dict
    current_step: ProductionStep = ProductionStep.QUEUED
    step_progress: int = 0  # 0-100 within current step
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    step_timestamps: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "session_id": self.session_id,
            "current_step": self.current_step.value,
            "step_progress": self.step_progress,
            "overall_progress": self.calculate_overall_progress(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "video_url": self.video_url,
            "error_message": self.error_message,
            "is_complete": self.current_step == ProductionStep.COMPLETE,
            "is_error": self.current_step == ProductionStep.ERROR,
            "estimated_remaining_seconds": self.estimate_remaining_time(),
        }

    def calculate_overall_progress(self) -> int:
        """Calculate overall progress percentage."""
        step_weights = {
            ProductionStep.QUEUED: 0,
            ProductionStep.SCRIPT_GENERATION: 15,
            ProductionStep.VISUAL_SELECTION: 35,
            ProductionStep.VOICEOVER_GENERATION: 55,
            ProductionStep.VIDEO_EDITING: 80,
            ProductionStep.UPLOAD: 95,
            ProductionStep.COMPLETE: 100,
            ProductionStep.ERROR: 0,
        }
        base = step_weights.get(self.current_step, 0)
        if self.current_step not in [ProductionStep.COMPLETE, ProductionStep.ERROR, ProductionStep.QUEUED]:
            next_step_values = list(step_weights.values())
            current_index = list(step_weights.keys()).index(self.current_step)
            if current_index < len(next_step_values) - 1:
                step_range = next_step_values[current_index + 1] - base
                base += int((self.step_progress / 100) * step_range)
        return min(100, base)

    def estimate_remaining_time(self) -> int:
        """Estimate remaining seconds based on typical production times."""
        step_times = {
            ProductionStep.QUEUED: 5,
            ProductionStep.SCRIPT_GENERATION: 30,
            ProductionStep.VISUAL_SELECTION: 60,
            ProductionStep.VOICEOVER_GENERATION: 45,
            ProductionStep.VIDEO_EDITING: 90,
            ProductionStep.UPLOAD: 15,
        }
        remaining = 0
        found_current = False
        for step, duration in step_times.items():
            if step == self.current_step:
                found_current = True
                remaining += int(duration * (1 - self.step_progress / 100))
            elif found_current and step != ProductionStep.COMPLETE:
                remaining += duration
        return remaining

    def advance_step(self, new_step: ProductionStep) -> None:
        """Advance to a new step."""
        self.step_timestamps[self.current_step.value] = time.time()
        self.current_step = new_step
        self.step_progress = 0
        if new_step == ProductionStep.COMPLETE:
            self.completed_at = time.time()


# Production status store
PRODUCTION_STATUSES: Dict[str, ProductionStatus] = {}


def get_production_status(session_id: str) -> Optional[ProductionStatus]:
    """Get production status for a session."""
    return PRODUCTION_STATUSES.get(session_id)


def create_production_status(session_id: str, brief: dict) -> ProductionStatus:
    """Create new production status tracker."""
    status = ProductionStatus(
        session_id=session_id,
        brief=brief,
        started_at=time.time()
    )
    PRODUCTION_STATUSES[session_id] = status
    return status


async def update_production_step(
    session_id: str,
    step: ProductionStep,
    progress: int = 0,
    video_url: str = None,
    error: str = None
) -> None:
    """Update production status - called by RAGNAROK pipeline."""
    status = PRODUCTION_STATUSES.get(session_id)
    if not status:
        return
    if step != status.current_step:
        status.advance_step(step)
    status.step_progress = progress
    if video_url:
        status.video_url = video_url
    if error:
        status.error_message = error
        status.current_step = ProductionStep.ERROR


# =============================================================================
# SENTIMENT DETECTION - Adapt responses to user emotional state
# =============================================================================

def detect_user_sentiment(text: str) -> dict:
    """
    Detect user sentiment and emotional state.
    Returns dict of boolean signals for different states.
    """
    text_lower = text.lower().strip()

    signals = {
        "frustrated": False,
        "hurried": False,
        "confused": False,
        "pushback": False,
        "one_word": False,
        "pricing_focus": False,
        "positive": False,
    }

    # === FRUSTRATION SIGNALS ===
    frustration_words = [
        "stupid", "waste", "ridiculous", "hate", "terrible", "awful",
        "annoying", "useless", "pointless", "frustrat"
    ]
    # Check for ALL CAPS (more than 50% uppercase in message longer than 5 chars)
    if len(text) > 5:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text.replace(" ", ""))
        if caps_ratio > 0.5:
            signals["frustrated"] = True

    # Check for multiple exclamation marks
    if text.count("!") >= 2:
        signals["frustrated"] = True

    # Check for frustration words
    if any(word in text_lower for word in frustration_words):
        signals["frustrated"] = True

    # === HURRIED SIGNALS ===
    hurry_phrases = [
        "speed up", "quick", "hurry", "don't have time", "no time",
        "fast", "skip", "just", "only need", "can we finish",
        "how long", "take forever", "faster"
    ]
    if any(phrase in text_lower for phrase in hurry_phrases):
        signals["hurried"] = True

    # === CONFUSION SIGNALS ===
    confusion_patterns = [
        "what?", "huh?", "don't understand", "confused", "what do you mean",
        "i don't get", "not sure what", "can you explain", "what are you asking"
    ]
    if any(pattern in text_lower for pattern in confusion_patterns):
        signals["confused"] = True

    # === PUSHBACK SIGNALS ===
    pushback_patterns = [
        "actually,", "actually ", "no,", "no ", "that's not", "wrong",
        "disagree", "i don't think", "not really", "that doesn't"
    ]
    if any(pattern in text_lower for pattern in pushback_patterns):
        signals["pushback"] = True

    # === ONE-WORD DETECTION ===
    word_count = len(text.split())
    if word_count <= 2:
        signals["one_word"] = True

    # === PRICING FOCUS ===
    pricing_words = [
        "cost", "price", "how much", "budget", "expensive", "afford",
        "pricing", "pay", "money", "dollars", "$"
    ]
    if any(word in text_lower for word in pricing_words):
        signals["pricing_focus"] = True

    # === POSITIVE SIGNALS ===
    positive_words = [
        "great", "awesome", "perfect", "love", "excellent", "thanks",
        "cool", "nice", "sounds good", "yes", "yeah"
    ]
    if any(word in text_lower for word in positive_words):
        signals["positive"] = True

    return signals


def get_adaptive_instruction(sentiment: dict, session: BriefSessionState) -> str:
    """
    Generate adaptive instructions based on detected sentiment.
    Injected into system prompt to guide Claude's response.
    """
    instructions = []

    if sentiment["frustrated"]:
        instructions.append(
            "[USER FRUSTRATED] Acknowledge their feeling briefly. "
            "Offer faster path: 'I hear you—let me speed this up. "
            "Just need a few quick details and we're done.'"
        )

    if sentiment["hurried"]:
        instructions.append(
            "[USER IN HURRY] Switch to fast-track: "
            "'Got it—quick version. Business name, what you sell, who's buying, "
            "and what action you want viewers to take. Go.'"
        )

    if sentiment["one_word"] and not sentiment["positive"]:
        instructions.append(
            "[SHORT ANSWER] Use elicitation technique (statement + gentle probe): "
            "'Okay so [their answer]—I'm guessing there's more to it. "
            "Like, more X or more Y?' Don't interrogate."
        )

    if sentiment["pushback"]:
        instructions.append(
            "[USER PUSHBACK] Use Acknowledge → Validate → Continue: "
            "'Got it—my mistake. That actually makes more sense. "
            "Tell me more about that angle.'"
        )

    if sentiment["pricing_focus"]:
        session.pricing_ask_count += 1
        if session.pricing_ask_count >= 3:
            instructions.append(
                "[PRICING X3] User asked about pricing 3+ times. "
                "Offer escalation: 'I think you need pricing clarity now—totally fair. "
                "Let me connect you with someone who can give you exact numbers.'"
            )
        else:
            instructions.append(
                "[PRICING ASK] Redirect with empathy: "
                "'Great question—pricing depends on scope. "
                "Let's nail the vision first so the quote makes sense.'"
            )

    if sentiment["confused"]:
        instructions.append(
            "[USER CONFUSED] Clarify simply: "
            "'Let me put it differently...' then rephrase your question "
            "in simpler terms. One thing at a time."
        )

    if instructions:
        return "\n\n## ADAPT YOUR RESPONSE:\n" + "\n".join(instructions)

    return ""


# Confirmation detection
CONFIRMATION_PHRASES = [
    "yes", "yeah", "yep", "yup", "correct", "looks good", "that's right",
    "perfect", "let's go", "start", "confirm", "proceed", "do it",
    "that's correct", "approved", "go ahead", "ship it", "make it",
    "looks right", "all good", "good to go"
]


def user_confirmed_brief(text: str) -> bool:
    """Check if user confirmed the brief summary."""
    text_lower = text.lower().strip()

    # Check for explicit confirmations
    for phrase in CONFIRMATION_PHRASES:
        if phrase in text_lower:
            return True

    # Check for very short affirmative responses
    if text_lower in ["y", "k", "ok", "okay", "sure", "yea"]:
        return True

    return False


# =============================================================================
# PHASE 2: FAST-TRACK MODE - 4-question quick brief for hurried users
# =============================================================================

FAST_TRACK_SYSTEM_PROMPT = """You are Alex, a creative director at a video agency.

The user is in a HURRY. Use FAST-TRACK MODE:

RULES:
- Ask for ALL 4 essentials in ONE message
- Be ultra-brief (1 sentence intro max)
- No small talk, no explanations
- Accept partial answers and fill gaps with reasonable defaults

FAST-TRACK QUESTIONS (ask all at once):
1. Business name?
2. What do you sell?
3. Who's buying?
4. What should viewers do after watching?

RESPONSE FORMAT:
"Quick version—hit me with:
1. Business name
2. What you sell
3. Who you're targeting
4. What action you want (call, buy, sign up, etc.)"

When they answer, extract what you can and confirm:
"Got it: [business] selling [product] to [audience], goal is [action]. Sound right?"

If they confirm: "Perfect. Starting your video now."

=== EXTRACTION INSTRUCTIONS ===
CRITICAL: Extract ALL information from the user's message into extracted_data.
If user provides "1. TechStart 2. SaaS tools 3. Founders 4. Sign up", extract:
{
  "business_name": "TechStart",
  "primary_offering": "SaaS tools",
  "target_demographic": "Founders",
  "call_to_action": "Sign up"
}

NEVER leave extracted_data empty if ANY information was provided.

JSON response format:
{
  "response": "your brief message",
  "extracted_data": {
    "business_name": "extracted value or null",
    "primary_offering": "extracted value or null",
    "target_demographic": "extracted value or null",
    "call_to_action": "extracted value or null"
  },
  "is_complete": false,
  "mode": "fast_track"
}
"""

FAST_TRACK_FIELDS = ["business_name", "product", "audience", "cta"]


def should_trigger_fast_track(sentiment: dict, session: BriefSessionState) -> bool:
    """Determine if we should switch to fast-track mode."""
    # Explicit hurry signals
    if sentiment.get("hurried"):
        return True

    # Frustration + low progress = offer fast track
    if sentiment.get("frustrated") and session.completion_percentage() < 30:
        return True

    # Multiple one-word answers in a row
    if len(session.sentiment_history) >= 3:
        recent = session.sentiment_history[-3:]
        if all(s.get("one_word") for s in recent):
            return True

    return False


def get_fast_track_prompt() -> str:
    """Get the fast-track system prompt."""
    return FAST_TRACK_SYSTEM_PROMPT


def is_fast_track_complete(session: BriefSessionState) -> bool:
    """Check if fast-track minimum fields are gathered."""
    return all(session.fields.get(f) for f in FAST_TRACK_FIELDS)


# =============================================================================
# PHASE 2: ESCALATION LOGIC - When to route to human support
# =============================================================================

@dataclass
class EscalationDecision:
    """Result of escalation check."""
    should_escalate: bool
    reason: Optional[str] = None
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    suggested_response: Optional[str] = None


def check_escalation_needed(
    session: BriefSessionState,
    sentiment: dict,
    user_message: str,
    production_status: Optional[ProductionStatus] = None
) -> EscalationDecision:
    """
    Determine if conversation should escalate to human.

    Escalation triggers:
    1. Explicit request for human
    2. Pricing asked 4+ times
    3. Strong negative sentiment after video delivery
    4. Production failures (2+ retries)
    5. Refund/cancel language
    """
    message_lower = user_message.lower()

    # === EXPLICIT HUMAN REQUEST ===
    human_request_phrases = [
        "talk to human", "real person", "speak to someone",
        "talk to someone", "human please", "representative",
        "customer service", "support team", "manager"
    ]
    if any(phrase in message_lower for phrase in human_request_phrases):
        return EscalationDecision(
            should_escalate=True,
            reason="explicit_human_request",
            priority="medium",
            suggested_response=(
                "Absolutely—let me connect you with our team. "
                "Someone will reach out within the hour. "
                "Anything specific you'd like them to know?"
            )
        )

    # === PRICING PERSISTENCE ===
    if session.pricing_ask_count >= 4:
        return EscalationDecision(
            should_escalate=True,
            reason="pricing_persistence",
            priority="medium",
            suggested_response=(
                "I hear you—pricing matters and I can't give you the detail you need. "
                "Let me get you to someone who can give you exact numbers based on your project."
            )
        )

    # === REFUND/CANCEL LANGUAGE ===
    refund_phrases = [
        "refund", "money back", "cancel", "disappointed",
        "waste of money", "want my money", "charge back"
    ]
    if any(phrase in message_lower for phrase in refund_phrases):
        return EscalationDecision(
            should_escalate=True,
            reason="refund_request",
            priority="urgent",
            suggested_response=(
                "I'm sorry this isn't meeting your expectations. "
                "Let me get you to our support team right away—they can help sort this out."
            )
        )

    # === STRONG NEGATIVE AFTER DELIVERY ===
    if production_status and production_status.current_step == ProductionStep.COMPLETE:
        negative_phrases = [
            "terrible", "awful", "hate it", "completely wrong",
            "useless", "not what i asked", "garbage", "trash"
        ]
        if any(phrase in message_lower for phrase in negative_phrases):
            return EscalationDecision(
                should_escalate=True,
                reason="negative_result_feedback",
                priority="high",
                suggested_response=(
                    "I'm really sorry this isn't hitting the mark—that's on us. "
                    "Let me get a human creative director to look at this and give you a proper revision."
                )
            )

    # === PRODUCTION FAILURES ===
    if production_status and production_status.retry_count >= 2:
        return EscalationDecision(
            should_escalate=True,
            reason="production_failures",
            priority="high",
            suggested_response=(
                "We're having some technical issues with your video. "
                "I'm escalating this to our production team to get it sorted manually. "
                "Someone will update you shortly."
            )
        )

    # === SUSTAINED FRUSTRATION ===
    if len(session.sentiment_history) >= 5:
        recent_frustrated = sum(
            1 for s in session.sentiment_history[-5:]
            if s.get("frustrated")
        )
        if recent_frustrated >= 3:
            return EscalationDecision(
                should_escalate=True,
                reason="sustained_frustration",
                priority="medium",
                suggested_response=(
                    "I can tell this process isn't working for you—I apologize. "
                    "Would it help to talk to someone directly? "
                    "I can have our team reach out."
                )
            )

    # No escalation needed
    return EscalationDecision(should_escalate=False)


async def handle_escalation(
    session_id: str,
    session: BriefSessionState,
    decision: EscalationDecision,
    conversation_history: list
) -> dict:
    """
    Handle escalation to human support.
    In production, this would integrate with your support system.
    """
    escalation_data = {
        "session_id": session_id,
        "reason": decision.reason,
        "priority": decision.priority,
        "brief_state": session.fields,
        "conversation_history": conversation_history[-10:] if conversation_history else [],
        "sentiment_history": session.sentiment_history[-5:],
        "timestamp": time.time(),
    }

    # Log escalation
    logger.warning(f"[ESCALATION] Session {session_id} escalated: {decision.reason}")

    # TODO: Integrate with support system (Slack, Zendesk, etc.)

    return {
        "escalated": True,
        "reason": decision.reason,
        "priority": decision.priority,
        "response": decision.suggested_response,
    }


def generate_status_response(status: ProductionStatus) -> str:
    """Generate human-friendly status update."""
    step_messages = {
        ProductionStep.QUEUED: "Your video is queued up—starting any moment now.",
        ProductionStep.SCRIPT_GENERATION: "Writing your script right now...",
        ProductionStep.VISUAL_SELECTION: "Picking the perfect visuals for each scene...",
        ProductionStep.VOICEOVER_GENERATION: "Recording the voiceover...",
        ProductionStep.VIDEO_EDITING: "Editing everything together...",
        ProductionStep.UPLOAD: "Almost there—uploading your video...",
        ProductionStep.COMPLETE: "Your video is ready!",
        ProductionStep.ERROR: "We hit a snag. Let me get someone to help.",
    }
    base_message = step_messages.get(status.current_step, "Working on it...")
    remaining = status.estimate_remaining_time()

    if remaining > 0 and status.current_step not in [ProductionStep.COMPLETE, ProductionStep.ERROR]:
        if remaining < 60:
            time_str = f"about {remaining} seconds"
        else:
            time_str = f"about {remaining // 60} minute{'s' if remaining >= 120 else ''}"
        return f"{base_message} Should be {time_str} left."

    return base_message


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


def extract_response_from_json(raw_response: str) -> dict:
    """
    Extract structured response from Claude's JSON output.

    Handles multiple formats:
    - Direct JSON object
    - JSON in markdown code blocks
    - JSON mixed with prose text
    - Malformed/partial JSON

    Returns:
        dict with at least 'response' and 'extracted_data' keys
    """
    if not raw_response or not raw_response.strip():
        return {"response": "I'm here to help. What would you like to know?", "extracted_data": {}}

    raw_response = raw_response.strip()

    try:
        # Strategy 1: Try direct JSON parse (best case)
        if raw_response.startswith('{'):
            data = json.loads(raw_response)
            if isinstance(data, dict) and "response" in data:
                return data

        # Strategy 2: Extract JSON from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response)
        if code_block_match:
            data = json.loads(code_block_match.group(1))
            if isinstance(data, dict) and "response" in data:
                return data

        # Strategy 3: Find JSON object with "response" key anywhere in text
        # Use greedy matching to get the full object
        json_match = re.search(r'(\{[^{}]*"response"\s*:\s*"[^"]*"[^{}]*\})', raw_response)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "response" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find nested JSON (handles more complex structures)
        # Match opening brace, content with "response", closing brace
        nested_match = re.search(r'\{\s*"response"[\s\S]*?"is_complete"\s*:\s*(?:true|false)[\s\S]*?\}', raw_response, re.IGNORECASE)
        if nested_match:
            try:
                # Try to balance braces
                json_str = nested_match.group(0)
                data = json.loads(json_str)
                if isinstance(data, dict) and "response" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 5: Extract just the response string value
        response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)\"', raw_response)
        if response_match:
            response_text = response_match.group(1)
            # Unescape the string
            response_text = response_text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return {"response": response_text, "extracted_data": {}}

        # Fallback: Return raw text (cleaned of JSON artifacts)
        clean_response = re.sub(r'```json[\s\S]*?```', '', raw_response).strip()
        clean_response = re.sub(r'\{[^{}]*\}', '', clean_response).strip()

        if clean_response:
            return {"response": clean_response, "extracted_data": {}}

        return {"response": raw_response[:500], "extracted_data": {}}

    except json.JSONDecodeError as e:
        logger.warning(f"[JSON_EXTRACT] Parse failed: {e}")
        # Clean up partial JSON from response
        clean_response = re.sub(r'\s*\{[^}]*$', '', raw_response).strip()
        clean_response = re.sub(r'```json[\s\S]*', '', clean_response).strip()
        return {"response": clean_response or raw_response[:500], "extracted_data": {}}
    except Exception as e:
        logger.error(f"[JSON_EXTRACT] Unexpected error: {e}")
        return {"response": raw_response[:500], "extracted_data": {}}


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
                    "progress_percentage": int(state.get_completion_percentage() * 100),  # Convert 0-1 to 0-100 int
                    "missing_fields": missing_fields,
                    "trigger_production": ragnarok_ready,  # Same as ragnarok_ready for API consistency
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
                    "progress_percentage": int(state.get_completion_percentage() * 100),  # Convert 0-1 to 0-100 int
                    "missing_fields": state.get_missing_required_fields(),
                    "trigger_production": False,  # Always False on error path
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

    def _update_session_state_from_extraction(self, session_id: str, extracted_data: dict) -> BriefSessionState:
        """
        Update BriefSessionState with extracted brief fields.
        Maps various field name variations to canonical field names.

        Args:
            session_id: Session identifier
            extracted_data: Dictionary of extracted fields from Claude's response

        Returns:
            Updated BriefSessionState instance
        """
        if not extracted_data:
            return get_or_create_brief_session(session_id)

        brief_session = get_or_create_brief_session(session_id)

        # Comprehensive field mapping - handles variations in field names
        field_mapping = {
            # Business name variations
            "business_name": "business_name",
            "company_name": "business_name",
            "brand_name": "business_name",
            "name": "business_name",
            "company": "business_name",

            # Product/offering variations
            "primary_offering": "product",
            "product": "product",
            "service": "product",
            "offering": "product",
            "what_you_sell": "product",
            "products": "product",
            "services": "product",

            # Audience variations
            "target_demographic": "audience",
            "target_audience": "audience",
            "audience": "audience",
            "who_is_buying": "audience",
            "customer": "audience",
            "customers": "audience",
            "demographic": "audience",

            # CTA variations
            "call_to_action": "cta",
            "cta": "cta",
            "action": "cta",
            "desired_action": "cta",

            # Tone variations
            "tone": "tone",
            "brand_tone": "tone",
            "style": "tone",
            "mood": "tone",

            # Optional fields
            "key_message": "key_message",
            "message": "key_message",
            "platform": "platform",
            "platforms": "platform",
        }

        # Apply extracted fields to session state
        for extracted_key, value in extracted_data.items():
            if not value:
                continue

            canonical_field = field_mapping.get(extracted_key.lower())
            if canonical_field:
                # Only update if field is currently empty or None
                if not brief_session.fields.get(canonical_field):
                    brief_session.update_field(canonical_field, str(value))
                    logger.info(f"[BriefState] Updated {canonical_field} = {str(value)[:50]}")

        # Store updated session
        BRIEF_SESSIONS[session_id] = brief_session

        return brief_session

    def _calculate_brief_progress(self, brief_session: BriefSessionState) -> dict:
        """
        Calculate brief completion progress from actual filled fields.

        Args:
            brief_session: Current brief session state

        Returns:
            Dictionary with progress metrics:
            - progress_percentage: 0-100 completion percentage
            - missing_fields: List of unfilled required fields
            - is_complete: Boolean indicating all required fields are filled
            - trigger_production: Boolean indicating readiness for production
        """
        required_fields = ["business_name", "product", "audience", "cta", "tone"]

        # Count filled required fields
        filled_count = sum(1 for field in required_fields if brief_session.fields.get(field))
        total_count = len(required_fields)

        # Calculate percentage
        progress = int((filled_count / total_count) * 100) if total_count > 0 else 0

        # Identify missing fields
        missing = [field for field in required_fields if not brief_session.fields.get(field)]

        # Check completion
        is_complete = len(missing) == 0

        return {
            "progress_percentage": progress,
            "missing_fields": missing,
            "is_complete": is_complete,
            "trigger_production": is_complete,  # Only trigger when complete
        }

    def _check_escalation_triggered(self, response_text: str, sentiment: dict) -> bool:
        """
        Check if escalation to human support should be triggered.

        Args:
            response_text: AI's response text
            sentiment: Sentiment analysis dictionary

        Returns:
            Boolean indicating if escalation should occur
        """
        # Explicit escalation phrases in AI response
        escalation_phrases = [
            "connect you with",
            "team member will",
            "human support",
            "speak with someone",
            "get you to support",
            "reach out",
            "talk to someone",
            "escalating this",
        ]

        text_lower = response_text.lower()
        explicit_escalation = any(phrase in text_lower for phrase in escalation_phrases)

        # Sentiment-based escalation (frustrated + escalation requested)
        sentiment_escalation = (
            sentiment.get("frustrated", False) and
            sentiment.get("escalation_requested", False)
        )

        return explicit_escalation or sentiment_escalation

    def _extract_from_user_message(self, user_message: str) -> dict:
        """
        Fallback extraction: Parse user message directly using regex patterns.
        CRITICAL ORDER: Extract TONE first to prevent it being captured as business_name.

        Args:
            user_message: The user's raw message

        Returns:
            Dictionary with extracted fields
        """
        import re

        extracted = {}
        message_lower = user_message.lower().strip()

        # =========================================================================
        # TONE EXTRACTION - MUST BE FIRST to prevent false positives
        # =========================================================================
        tone_map = {
            # Luxury/Premium
            'luxurious': 'luxurious', 'luxury': 'luxurious', 'upscale': 'luxurious and upscale',
            'elegant': 'elegant', 'premium': 'premium', 'sophisticated': 'sophisticated',
            'high-end': 'high-end', 'high end': 'high-end', 'exclusive': 'exclusive',
            'classy': 'classy', 'refined': 'refined',

            # Professional
            'professional': 'professional', 'corporate': 'professional', 'formal': 'professional',
            'business': 'professional', 'clinical': 'professional', 'serious': 'professional',
            'polished': 'professional', 'authoritative': 'authoritative', 'trustworthy': 'trustworthy',

            # Friendly/Warm
            'friendly': 'friendly', 'warm': 'warm and friendly', 'welcoming': 'welcoming',
            'approachable': 'approachable', 'casual': 'casual', 'relaxed': 'relaxed',
            'conversational': 'conversational', 'personable': 'personable', 'inviting': 'inviting',

            # Energetic/Fun
            'energetic': 'energetic', 'exciting': 'exciting', 'dynamic': 'dynamic',
            'upbeat': 'upbeat', 'fun': 'fun', 'playful': 'playful', 'lively': 'lively',
            'vibrant': 'vibrant', 'enthusiastic': 'enthusiastic', 'bold': 'bold',

            # Calm/Cozy
            'cozy': 'cozy', 'calm': 'calm', 'soothing': 'soothing', 'peaceful': 'peaceful',
            'serene': 'serene', 'tranquil': 'tranquil', 'gentle': 'gentle', 'soft': 'soft',

            # Modern/Trendy
            'modern': 'modern', 'trendy': 'trendy', 'edgy': 'edgy',
            'cutting-edge': 'cutting-edge', 'cutting edge': 'cutting-edge',
            'contemporary': 'contemporary', 'hip': 'trendy', 'cool': 'cool and modern',
            'sleek': 'sleek', 'minimalist': 'minimalist',

            # Urgent/Direct
            'urgent': 'urgent', 'direct': 'direct', 'compelling': 'compelling',
        }

        # Check for tone keywords
        for keyword, tone_value in tone_map.items():
            if keyword in message_lower:
                extracted["tone"] = tone_value
                break

        # Check for "X vibe/tone/feel/style" patterns
        if 'tone' not in extracted:
            vibe_match = re.search(r'(\w+)\s+(?:vibe|tone|feel|style|aesthetic|mood)', message_lower)
            if vibe_match:
                vibe_word = vibe_match.group(1)
                if vibe_word in tone_map:
                    extracted["tone"] = tone_map[vibe_word]
                elif len(vibe_word) > 2:
                    extracted["tone"] = vibe_word

        # Check for "should be X" or "make it X" patterns
        if 'tone' not in extracted:
            should_match = re.search(r'(?:should be|make it|keep it|go with|want it) (\w+)', message_lower)
            if should_match:
                word = should_match.group(1)
                if word in tone_map:
                    extracted["tone"] = tone_map[word]

        # =========================================================================
        # BUSINESS NAME EXTRACTION - WITH SKIP LOGIC
        # =========================================================================

        # SKIP if message is tone-only (e.g., "Luxurious and upscale")
        word_count = len(user_message.split())
        has_tone_keyword = any(keyword in message_lower for keyword in tone_map.keys())

        if word_count <= 5 and has_tone_keyword and 'tone' in extracted:
            # This is a tone-only message, don't extract business name
            pass

        # SKIP if message starts with demographic keywords
        elif re.match(r'^(?:women|men|people|adults|seniors|families|patients|clients)', message_lower):
            # This is likely a demographic message, skip business name
            pass

        else:
            business_patterns = [
                # "My business name is X" / "My business is X"
                r"(?:my |our )?business (?:name )?is ([A-Z][A-Za-z0-9\s&'.-]+)",
                # "My salon/studio/shop is called X"
                r"(?:my |our )?(?:salon|studio|shop|store|company|restaurant|clinic|pharmacy|bakery|gym|spa) (?:is )?(?:called |named )?([A-Z][A-Za-z0-9\s&'.-]+)",
                # "I own X" / "I run X"
                r"(?:i own |i run |we own |we run |we are |we're )([A-Z][A-Za-z0-9\s&'.-]+)",
                # "X is my salon/business"
                r"([A-Z][A-Za-z0-9\s&'.-]+) is (?:my |our )?(?:salon|studio|shop|store|company|business)",
                # "It's called X" / "We're called X"
                r"(?:it's called |its called |we're called |were called |named )([A-Z][A-Za-z0-9\s&'.-]+)",
                # "called X" anywhere (but NOT "is called" - handled above)
                r"(?<!is )called ([A-Z][A-Za-z0-9\s&'.-]+)",
                # Just a capitalized name by itself
                r"^([A-Z][A-Za-z0-9\s&'.-]+)$",
                # Original patterns
                r"(?:i run|i own|my company is|we are|i'm from|i'm)\s+([A-Z][A-Za-z0-9\s&'.-]+?)(?:\s*[,.]|\s+and|\s+we|\s+selling|$)",
                r"^([A-Z][A-Za-z0-9]+)\s*[-–]\s*",
                r"(?:at|for)\s+([A-Z][A-Za-z0-9\s&'.-]+?)(?:\s*[,.]|\s+and|$)",
            ]

            for pattern in business_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Clean up trailing words
                    name = re.sub(r'\s+(?:and|for|to|the|a|in|on|is|are|we|i)$', '', name, flags=re.IGNORECASE)
                    name = name.strip(' .,')

                    # Remove tone words that might have been captured
                    for tone_keyword in tone_map.keys():
                        if tone_keyword in name.lower():
                            name = re.sub(rf'\b{tone_keyword}\b', '', name, flags=re.IGNORECASE).strip()

                    # Filter false positives
                    skip_words = ['i', 'we', 'my', 'our', 'the', 'a', 'an', 'yes', 'no', 'ok', 'okay',
                                 'hello', 'hi', 'hey', 'sure', 'yeah', 'yep', 'nope', 'thanks']
                    if len(name) > 1 and len(name) < 50 and name.lower() not in skip_words:
                        extracted["business_name"] = name
                        break

        # =========================================================================
        # PRIMARY OFFERING EXTRACTION - Enhanced patterns
        # =========================================================================
        product_patterns = [
            r"(?:we |i )?(?:specialize|specialise) in ([^.!?,]+)",
            r"(?:we |i )?focus on ([^.!?,]+)",
            r"(?:we |i )?(?:offer|provide|do|make|create|sell) ([^.!?,]+)",
            r"(?:our |my )?(?:main|primary|core|signature) (?:service|product|offering) is ([^.!?,]+)",
            r"(?:known for|famous for|best at) ([^.!?,]+)",
            r"(?:specialized in|focusing on) ([^.!?,]+)",
            r"(?:hair |food |fitness |dental |legal |medical |cleaning |custom |handmade )([^.!?,]+)",
            r"([A-Za-z\s]+) (?:services|products|solutions|treatments)",
            r"(?:saas|software|tools?|platform|app|service|product)s?\s+(?:for|that)\s+(.+?)(?:\s*[,.]|$)",
            r"(?:our|the|my)\s+(products?|services?)\s+(?:are|is)\s+(.+?)(?:\s*[,.]|$)",
        ]

        for pattern in product_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                offering = match.group(match.lastindex if match.lastindex else 1).strip()
                offering = re.sub(r'\s+(?:and|for|to|the)$', '', offering, flags=re.IGNORECASE)
                offering = offering.strip(' .,')
                if len(offering) > 3 and offering.lower() not in ['it', 'that', 'this', 'them']:
                    extracted["primary_offering"] = offering[:100]
                    break

        # Keyword fallback
        if "primary_offering" not in extracted:
            product_keywords = ["saas", "software", "tools", "platform", "app", "automation", "service", "solution"]
            for keyword in product_keywords:
                if keyword in message_lower:
                    match = re.search(rf"(\w+\s+)?(\w+\s+)?{keyword}(\s+\w+)?(\s+\w+)?", message_lower)
                    if match:
                        extracted["primary_offering"] = match.group(0).strip()
                        break

        # =========================================================================
        # TARGET DEMOGRAPHIC EXTRACTION
        # =========================================================================
        demographic_patterns = [
            r"((?:women|men|people|adults|seniors|teens|kids|children|families|parents|professionals|businesses) (?:aged?|ages?)? ?\d+[- to]+\d+[^.!?,]*)",
            r"((?:women|men|people|customers|clients|patients) who [^.!?,]+)",
            r"(?:our |my )?target (?:is |are |audience is |market is )([^.!?,]+)",
            r"(?:trying to |want to |looking to )?reach ([^.!?,]+)",
            r"^for ([^.!?,]+)",
            r"([A-Za-z\s]+) (?:customers|clients|patients|audience)",
            r"(?:aged?|ages?) ?(\d+[- to]+\d+)",
            r"(local[s]?|local (?:customers|clients|people|community))",
            r"(?:patients?|people|folks|users|businesses|companies)\s+(?:who|that|needing|seeking)\s+(.+?)(?:\s*[,.]|$)",
            r"(?:reach|serving|helping)\s+(patients?|doctors?|businesses?|companies?|people|professionals?)\s*(?:who|that)?\s*(.+?)(?:\s*[,.]|$)",
        ]

        for pattern in demographic_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                demo = None
                for group in reversed(match.groups()):
                    if group and group.strip():
                        demo = group.strip()
                        break
                if not demo:
                    demo = match.group(1).strip()

                demo = demo.strip(' .,')
                if demo and len(demo) > 3 and "sign" not in demo.lower() and "buy" not in demo.lower():
                    extracted["target_demographic"] = demo[:100]
                    break

        # =========================================================================
        # CALL TO ACTION EXTRACTION - FIXED: removed bare 'call'
        # =========================================================================
        cta_map = {
            'book online': ['book online', 'book through', 'book via', 'book on'],
            'book appointment': ['book appointment', 'schedule', 'make an appointment'],
            'call us': ['call us', 'give us a call', 'phone us'],  # REMOVED bare 'call'
            'visit website': ['visit website', 'visit our website', 'check out our website'],
            'visit store': ['visit store', 'come in', 'stop by', 'walk in', 'visit us'],
            'order online': ['order online', 'order from', 'buy online', 'purchase online'],
            'sign up': ['sign up', 'signup', 'register', 'subscribe'],
            'contact us': ['contact us', 'get in touch'],
        }

        for cta_value, phrases in cta_map.items():
            for phrase in phrases:
                if phrase in message_lower:
                    extracted["call_to_action"] = cta_value
                    break
            if 'call_to_action' in extracted:
                break

        # Pattern-based CTA extraction if not found
        if "call_to_action" not in extracted:
            cta_patterns = [
                r"(?:cta|call to action)(?:\s+is)?\s*[:\-]?\s*(.+?)(?:\s*[,.]|$)",
                r"(?:want them to|they should|viewers should|people should)\s+(.+?)(?:\s*[,.]|$)",
                r"(sign up|buy now|learn more|get started|book|register|subscribe|download|contact|schedule)",
            ]
            for pattern in cta_patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    cta = match.group(1 if match.groups() else 0).strip()[:50]
                    extracted["call_to_action"] = cta
                    break

        # =========================================================================
        # NUMBERED LIST EXTRACTION (fast-track format)
        # =========================================================================
        numbered_match = re.findall(r'(\d+)[.)]\s*([^0-9]+?)(?=\d+[.)]|$)', user_message)
        if len(numbered_match) >= 3:
            field_order = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]
            for i, (num, value) in enumerate(numbered_match):
                if i < len(field_order):
                    value = value.strip().rstrip(',.')
                    if value and len(value) > 1:
                        extracted[field_order[i]] = value

        return extracted

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

    def force_question_in_response(self, response_text: str, missing_fields: list) -> str:
        """
        Post-process AI response to ensure it ends with a question.
        If response doesn't end with ?, append the next logical question.
        """
        if not response_text:
            return "What's the name of your business?"

        # Check if response already ends with a question
        cleaned = response_text.strip()
        if cleaned.endswith('?'):
            return response_text

        # Response doesn't end with question - need to add one
        question_map = {
            'business_name': "What's the name of your business?",
            'primary_offering': "What product or service do you want to highlight?",
            'target_demographic': "Who are you trying to reach with this video?",
            'call_to_action': "What should viewers do after watching?",
            'tone': "What vibe should this video have?"
        }

        # Find the first missing field and add its question
        for field in ['business_name', 'primary_offering', 'target_demographic', 'call_to_action', 'tone']:
            if field in missing_fields:
                return f"{response_text} {question_map[field]}"

        # All fields complete - ask for confirmation
        return f"{response_text} Ready to create your video?"

    async def _generate_response(
        self,
        state: VideoBriefState,
        user_message: str
    ) -> Dict[str, Any]:
        """Generate AI response using Claude with Phase 1 enhancements"""

        # =================================================================
        # PHASE 1: Get/create BriefSessionState for conversation tracking
        # =================================================================
        brief_session = get_or_create_brief_session(state.session_id)
        brief_session.conversation_count += 1
        brief_session.last_message_time = time.time()

        # Check if user is confirming a brief summary
        if brief_session.awaiting_confirmation:
            if user_confirmed_brief(user_message):
                brief_session.awaiting_confirmation = False
                # Phase 2: Create production status tracker
                prod_status = create_production_status(state.session_id, brief_session.fields)
                return {
                    "response": "Perfect. Starting your video now—I'll show you progress as each step completes.",
                    "extracted_data": brief_session.get_filled_fields(),
                    "is_complete": True,
                    "ragnarok_ready": True,
                    "trigger_production": True,
                    "production_status": prod_status.to_dict(),
                    "progress_percentage": 100,
                    "mode": "production",
                    "next_phase": "complete",
                    "confidence": 1.0,
                }
            else:
                # User wants to make changes, continue conversation
                brief_session.awaiting_confirmation = False

        # Detect sentiment for adaptive responses
        sentiment = detect_user_sentiment(user_message)
        brief_session.sentiment_history.append(sentiment)

        # =================================================================
        # PHASE 2: Escalation Check
        # =================================================================
        production_status = get_production_status(state.session_id)
        escalation = check_escalation_needed(
            session=brief_session,
            sentiment=sentiment,
            user_message=user_message,
            production_status=production_status
        )

        if escalation.should_escalate:
            await handle_escalation(
                session_id=state.session_id,
                session=brief_session,
                decision=escalation,
                conversation_history=state.conversation_history
            )
            return {
                "response": escalation.suggested_response,
                "extracted_data": brief_session.get_filled_fields(),
                "is_complete": False,
                "escalated": True,
                "escalation_reason": escalation.reason,
                "escalation_priority": escalation.priority,
                "progress_percentage": brief_session.completion_percentage(),
                "mode": "escalated",
                "next_phase": state.phase.value,
                "confidence": 1.0,
            }

        # =================================================================
        # PHASE 2: Production Status Query
        # =================================================================
        if production_status and production_status.current_step not in [ProductionStep.COMPLETE, ProductionStep.ERROR]:
            status_queries = ["status", "how long", "progress", "where", "when will", "ready yet"]
            if any(q in user_message.lower() for q in status_queries):
                return {
                    "response": generate_status_response(production_status),
                    "extracted_data": brief_session.get_filled_fields(),
                    "is_complete": True,
                    "production_status": production_status.to_dict(),
                    "progress_percentage": 100,
                    "mode": "production_status",
                    "next_phase": "production",
                    "confidence": 1.0,
                }

        # =================================================================
        # PHASE 2: Fast-Track Mode Trigger
        # =================================================================
        if should_trigger_fast_track(sentiment, brief_session) and brief_session.mode != "fast_track":
            brief_session.mode = "fast_track"
            return {
                "response": (
                    "Got it—let's speed this up. Quick version:\n\n"
                    "1. Business name?\n"
                    "2. What do you sell?\n"
                    "3. Who's buying?\n"
                    "4. What should viewers do after watching?\n\n"
                    "Hit me with all four."
                ),
                "extracted_data": brief_session.get_filled_fields(),
                "is_complete": False,
                "mode": "fast_track",
                "progress_percentage": brief_session.completion_percentage(),
                "next_phase": state.phase.value,
                "confidence": 1.0,
            }

        # Inject few-shot examples on first message only
        processed_message = inject_few_shot_examples(user_message, brief_session.is_first_message)
        if brief_session.is_first_message:
            brief_session.is_first_message = False

        # Add context injection (what we've gathered so far)
        context_injection = brief_session.get_context_injection()
        if context_injection:
            processed_message += context_injection

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

        # Add current state context with explicit missing fields guidance
        missing_required = state.get_missing_required_fields()
        state_context = f"""
Current brief state:
- Phase: {state.phase.value}
- Completion: {state.get_completion_percentage():.0%}
- Business: {state.business_name or 'MISSING'}
- Offering: {state.primary_offering or 'MISSING'}
- Audience: {state.target_demographic or 'MISSING'}
- CTA: {state.call_to_action or 'MISSING'}
- Tone: {state.tone or 'MISSING'}

FIELDS STILL NEEDED: {', '.join(missing_required) if missing_required else 'All complete!'}

IMPORTANT: If any fields are MISSING, your response MUST ask for the NEXT missing field. Do not just acknowledge - always ask a specific question.
"""

        # Get adaptive instruction based on sentiment
        adaptive_instruction = get_adaptive_instruction(sentiment, brief_session)

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

        # Include adaptive instruction based on sentiment
        base_prompt = CREATIVE_DIRECTOR_SYSTEM_PROMPT + "\n\n" + state_context
        if adaptive_instruction:
            base_prompt += adaptive_instruction

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
                messages=messages + [{"role": "user", "content": processed_message}]
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

            # =================================================================
            # ROBUST JSON EXTRACTION using extract_response_from_json helper
            # Handles: direct JSON, code blocks, mixed prose, malformed JSON
            # =================================================================
            parsed = extract_response_from_json(result_text)
            logger.debug(f"[JSON_EXTRACT] Parsed result keys: {list(parsed.keys())}")

            try:
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

                # =================================================================
                # PHASE 2 FIX: Use new methods for session state update & progress
                # =================================================================
                extracted = parsed.get("extracted_data") or {}

                # FALLBACK EXTRACTION: If Claude didn't return structured data, parse user message directly
                if not extracted or (isinstance(extracted, dict) and not any(extracted.values())):
                    fallback_extracted = self._extract_from_user_message(user_message)
                    if fallback_extracted:
                        extracted = fallback_extracted
                        logger.info(f"[Extraction Fallback] Parsed user message directly: {list(fallback_extracted.keys())} -> {fallback_extracted}")
                else:
                    logger.debug(f"[Extraction] Claude returned: {extracted}")

                # SECONDARY EXTRACTION: Check if Claude mentioned data in prose response
                # This catches cases where Claude says "So you run TechStart..." but didn't use JSON
                if extracted and isinstance(extracted, dict):
                    response_text = parsed.get("response", "")

                    # Extract business name from prose if missing
                    if not extracted.get("business_name"):
                        business_match = re.search(r"(?:you run|your company|called|you're|you are)\s+([A-Z][A-Za-z0-9]+)", response_text)
                        if business_match:
                            extracted["business_name"] = business_match.group(1)
                            logger.info(f"[Secondary Extraction] Found business name in prose: {business_match.group(1)}")

                # Update session state from extracted data using new method
                brief_session = self._update_session_state_from_extraction(
                    session_id=state.session_id,
                    extracted_data=extracted
                )

                # Calculate accurate progress from filled fields
                progress_info = self._calculate_brief_progress(brief_session)

                # Check for escalation triggers
                escalation_triggered = self._check_escalation_triggered(
                    response_text=parsed.get("response", ""),
                    sentiment=sentiment
                )

                # Check if brief is complete → trigger confirmation flow
                if brief_session.is_complete() and not brief_session.awaiting_confirmation:
                    brief_session.awaiting_confirmation = True
                    parsed["response"] = brief_session.generate_summary()

                # Build response with accurate progress from new methods
                parsed["progress_percentage"] = progress_info["progress_percentage"]
                parsed["missing_fields"] = progress_info["missing_fields"]
                parsed["is_complete"] = progress_info["is_complete"]
                parsed["trigger_production"] = progress_info["trigger_production"]
                parsed["ragnarok_ready"] = progress_info["is_complete"]
                parsed["mode"] = "intake"
                parsed["sentiment"] = sentiment
                parsed["escalation"] = escalation_triggered
                # CRITICAL FIX: Include actual extracted data so _apply_extraction can update VideoBriefState
                parsed["extracted_data"] = extracted or {}

                # =================================================================
                # FORCE QUESTION POST-PROCESSING
                # This ensures every response ends with a question
                # =================================================================
                if parsed and "response" in parsed:
                    missing = progress_info.get("missing_fields", [])
                    original_response = parsed["response"]
                    parsed["response"] = self.force_question_in_response(parsed["response"], missing)
                    if original_response != parsed["response"]:
                        logger.info(f"[FORCE_QUESTION] Added question to response. Missing fields: {missing}")

                logger.info(f"[BriefProgress] Session {state.session_id}: {progress_info['progress_percentage']}% complete, missing: {progress_info['missing_fields']}, extracted: {list(extracted.keys()) if extracted else []}")

                return parsed

            except (KeyError, TypeError, AttributeError) as e:
                # Response processing error - use the parsed response as-is
                logger.warning(f"[RESPONSE_PROCESS] Processing error: {e}, using parsed response")
                response_text = parsed.get("response", "I understand. Let me continue with the next question.")

                # Use new progress calculation method even in error case
                progress_info = self._calculate_brief_progress(brief_session)
                # Get extracted data from the fallback extraction (if any)
                extracted_data = extracted if 'extracted' in locals() else parsed.get("extracted_data", {})

                return {
                    "response": response_text,
                    "extracted_data": extracted_data,
                    "next_phase": state.phase.value,
                    "confidence": 0.5,
                    "is_complete": progress_info["is_complete"],
                    "progress_percentage": progress_info["progress_percentage"],
                    "missing_fields": progress_info["missing_fields"],
                    "trigger_production": progress_info["trigger_production"],
                    "mode": "intake",
                    "escalation": False,
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

                # Use new progress calculation method
                progress_info = self._calculate_brief_progress(brief_session)

                return {
                    "response": emergency["response"],
                    "extracted_data": {},
                    "next_phase": state.phase.value,
                    "confidence": 0.0,
                    "is_complete": progress_info["is_complete"],
                    "quality_tier": emergency["quality_tier"],
                    "error_type": error_type,
                    "progress_percentage": progress_info["progress_percentage"],
                    "missing_fields": progress_info["missing_fields"],
                    "trigger_production": progress_info["trigger_production"],
                    "mode": "error",
                    "escalation": False,
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
