"""
============================================================================
LANGGRAPH CREATIVE DIRECTOR V2
============================================================================

File: intake/creative_director_v2.py

Deploy as: /api/chat/v2
Test alongside: /api/chat (existing v1)
Migrate frontend when validated

This replaces the 2800-line monolith with a proper state machine.
============================================================================
"""

import os
import json
import logging
import time
import traceback
import httpx
from typing import TypedDict, List, Optional, Literal
from datetime import datetime, timezone

# LangGraph
from langgraph.graph import StateGraph, START, END

# Production status tracking (for SSE)
try:
    from intake.video_brief_intake import (
        create_production_status,
        update_production_step,
        ProductionStep,
    )
    PRODUCTION_TRACKING_AVAILABLE = True
except ImportError:
    PRODUCTION_TRACKING_AVAILABLE = False
    create_production_status = None
    update_production_step = None
    ProductionStep = None

# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

# Barrios Product Knowledge
try:
    from knowledge.barrios_product import BARRIOS_PRODUCT_KNOWLEDGE
    PRODUCT_KNOWLEDGE_AVAILABLE = True
except ImportError:
    BARRIOS_PRODUCT_KNOWLEDGE = ""
    PRODUCT_KNOWLEDGE_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creative_director_v2")

# Trinity API Configuration
TRINITY_URL = "https://barrios-genesis-flawless.onrender.com/api/genesis/research"
TRINITY_TIMEOUT = 60.0  # Trinity research can take time

# RAGNAROK Production API Configuration
GENESIS_API_BASE = "https://barrios-genesis-flawless.onrender.com"

# Video Preview Integration - Auto-publish completed commercials
VIDEO_PREVIEW_API = "https://video-preview-theta.vercel.app/api/videos"

# ============================================================================
# BARRIOS A2I COMMERCIAL SPECIFICATIONS
# ============================================================================
# Standard commercial: 64 seconds, 8 scenes (8 seconds each)
COMMERCIAL_CONFIG = {
    "duration_seconds": 64,
    "scene_count": 8,
    "scene_duration_seconds": 8,
    "scenes": [
        {"name": "SYSTEM_ONLINE", "duration": "0:00-0:08", "purpose": "Establish authority and brand presence"},
        {"name": "PIPELINE", "duration": "0:08-0:16", "purpose": "Show the process/workflow"},
        {"name": "AGENTS", "duration": "0:16-0:24", "purpose": "Demonstrate automation/AI capabilities"},
        {"name": "PROOF", "duration": "0:24-0:32", "purpose": "Industry adaptability and versatility"},
        {"name": "OUTPUTS", "duration": "0:32-0:40", "purpose": "Format variety and deliverables"},
        {"name": "HUMAN_REMOVAL", "duration": "0:40-0:48", "purpose": "Speed and efficiency benefits"},
        {"name": "ECOSYSTEM", "duration": "0:48-0:56", "purpose": "Scale potential and integration"},
        {"name": "CTA", "duration": "0:56-1:04", "purpose": "Clear call to action with tagline"}
    ]
}

# Helper function to generate scene structure for prompt
def generate_scene_structure() -> str:
    """Generate the scene structure text for the script prompt based on COMMERCIAL_CONFIG."""
    scenes = COMMERCIAL_CONFIG["scenes"]
    scene_duration = COMMERCIAL_CONFIG["scene_duration_seconds"]

    lines = []
    for i, scene in enumerate(scenes):
        start_sec = i * scene_duration
        end_sec = (i + 1) * scene_duration
        start_time = f"{start_sec // 60}:{start_sec % 60:02d}"
        end_time = f"{end_sec // 60}:{end_sec % 60:02d}"
        lines.append(f"{i + 1}. **{scene['name']}** ({start_time}-{end_time}): {scene['purpose']}")

    return "\n".join(lines)


# ============================================================================
# VIDEO PREVIEW INTEGRATION
# ============================================================================

async def send_to_video_preview(video_url: str, state: dict) -> dict:
    """
    Send completed commercial to video-preview-theta.vercel.app
    Creates a shareable preview link for the client.

    Returns: {"success": bool, "preview_id": str, "preview_url": str}
    """
    try:
        # Generate unique ID for this commercial
        video_id = f"genesis_{state.get('session_id', 'unknown')}_{int(time.time())}"

        payload = {
            "id": video_id,
            "url": video_url,
            "title": f"{state.get('business_name', 'Commercial')} - AI Generated",
            "description": f"64-second commercial for {state.get('business_name', 'client')}. {state.get('primary_offering', '')}",
            "duration": "1:04",
            "tags": ["commercial", "ai-generated", "barrios-a2i", "64-seconds"],
            "business_name": state.get("business_name", ""),
            "industry": _infer_industry(state.get("primary_offering", ""))
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(VIDEO_PREVIEW_API, json=payload)

            if response.status_code in [200, 201]:
                result = response.json()
                preview_id = result.get("video", {}).get("id", video_id)
                preview_url = f"https://video-preview-theta.vercel.app?v={preview_id}"

                logger.info(f"[VideoPreview] ✅ Published: {preview_id}")
                return {
                    "success": True,
                    "preview_id": preview_id,
                    "preview_url": preview_url
                }
            else:
                logger.warning(f"[VideoPreview] API returned {response.status_code}: {response.text}")
                return {"success": False, "preview_id": None, "preview_url": None}

    except Exception as e:
        logger.error(f"[VideoPreview] Failed to send: {e}")
        return {"success": False, "preview_id": None, "preview_url": None}


# Script Writer Prompt (Barrios A2I Standard: configurable scenes)
SCRIPT_WRITER_PROMPT = """You are a world-class commercial scriptwriter for Barrios A2I video advertisements.

## VIDEO BRIEF
- Business: {business_name}
- Product/Service: {primary_offering}
- Target Audience: {target_demographic}
- Call to Action: {call_to_action}
- Desired Tone: {tone}

## MARKET RESEARCH INSIGHTS
{research_summary}

## BARRIOS A2I COMMERCIAL SPECIFICATIONS
Create a **{total_duration}-second** commercial with exactly **{scene_count} scenes** ({scene_duration} seconds each):

{scene_structure}

## CRITICAL TIMING CONSTRAINTS (MUST FOLLOW)
- Speaking pace: 2.5 words per second
- MAXIMUM {words_per_scene} WORDS per scene narration
- MAXIMUM {total_words} WORDS total voiceover script
- If you exceed these limits, the audio will be CUT OFF mid-sentence!

Count your words carefully for each narration field.

## VISUAL STYLE: CINEMATIC B-ROLL ONLY (CRITICAL)
Your visual_description fields must describe HIGH-END COMMERCIAL B-ROLL footage.

**REQUIRED Visual Elements:**
- Product close-ups and macro shots
- Environment/location establishing shots
- Abstract visualizations (data flow, particle effects, light trails)
- Action sequences showing processes or results
- Drone/aerial footage of locations
- Cinematic camera movements (dolly, crane, steadicam, slow pan)

**FORBIDDEN Visual Elements - NEVER describe these:**
- People talking to camera ❌
- Presenters or spokespersons explaining ❌
- Talking heads or avatars ❌
- Direct-to-camera dialogue ❌
- Someone presenting in an office ❌
- Interview-style footage ❌

**GOOD visual_description examples:**
- "Sleek drone shot over a modern data center at dawn, rows of servers glowing blue"
- "Extreme macro of coffee beans being ground in slow motion, aromatic dust particles floating"
- "Split-screen transformation: cluttered desk morphs into organized workspace, time-lapse style"
- "Abstract visualization of AI neural networks processing data, neon pathways pulsing"

**BAD visual_description examples (NEVER use):**
- "CEO explains company benefits to camera" ❌
- "A friendly presenter walks through features" ❌
- "Business owner talks about their experience" ❌

## OUTPUT FORMAT
Return a JSON object with this exact structure (NO comments allowed - pure JSON only):
{{
    "title": "Commercial title",
    "duration_seconds": {total_duration},
    "target_platform": "social_media",
    "scenes": [
        {{
            "scene_number": 1,
            "timestamp": "0:00-0:08",
            "type": "hook",
            "visual_description": "Cinematic B-roll description for scene 1",
            "narration": "Voiceover text (MAX {words_per_scene} words)",
            "text_overlay": "On-screen text or empty string",
            "music_mood": "Music direction"
        }},
        {{
            "scene_number": 2,
            "timestamp": "0:08-0:16",
            "type": "problem",
            "visual_description": "Cinematic B-roll description for scene 2",
            "narration": "Voiceover text (MAX {words_per_scene} words)",
            "text_overlay": "On-screen text or empty string",
            "music_mood": "Music direction"
        }}
    ],
    "voiceover_full_script": "Complete narration (MUST be under {total_words} words total)",
    "visual_style_notes": "Overall visual direction",
    "key_messaging": ["Main message 1", "Main message 2", "Main message 3"],
    "estimated_production_complexity": "low|medium|high"
}}

IMPORTANT: Generate EXACTLY {scene_count} scenes with timestamps at {scene_duration}-second intervals.
The example shows 2 scenes - you must generate all {scene_count} scenes following the same structure.
REMEMBER: {words_per_scene} words per scene MAX. {total_words} words total MAX. Count carefully!
Make it cinematic and compelling for {business_name}.
"""

# Reviewer Node - Approval/Revision detection patterns
APPROVAL_PATTERNS = [
    "approve", "approved", "looks good", "perfect", "love it", "great",
    "yes", "go ahead", "proceed", "let's do it", "ship it", "send it",
    "ready to produce", "ready for production", "make it", "create it",
    "that's perfect", "exactly what i wanted", "nailed it", "awesome",
    "👍", "✅", "🎬", "produce it", "generate", "make the video"
]

REVISION_PATTERNS = [
    "change", "revise", "update", "modify", "tweak", "adjust",
    "make it more", "make it less", "can you", "could you",
    "i'd like", "i want", "different", "instead", "but", "however",
    "not quite", "almost", "close but", "try again", "redo",
    "more", "less", "shorter", "longer", "faster", "slower",
    "upbeat", "serious", "funny", "professional", "casual"
]

REJECTION_PATTERNS = [
    "start over", "cancel", "stop", "forget it", "never mind",
    "don't want", "scrap it", "delete", "trash", "no thanks"
]


# ============================================================================
# 1. SHARED STATE - Single Source of Truth
# ============================================================================

class VideoBriefState(TypedDict):
    """
    Immutable state shared by all agents.
    Field names are NORMALIZED - no PRODUCT vs primary_offering mismatch.
    """
    # Core 5 Fields
    business_name: Optional[str]
    primary_offering: Optional[str]
    target_demographic: Optional[str]
    call_to_action: Optional[str]
    tone: Optional[str]

    # Optional (for Trinity research)
    top_rivals: Optional[List[str]]

    # Conversation
    messages: List[dict]

    # State tracking
    missing_fields: List[str]
    is_complete: bool
    current_phase: str

    # Session metadata
    session_id: str

    # Research findings from Trinity
    research_findings: Optional[dict]

    # Script draft from Script Writer
    script_draft: Optional[dict]

    # Review workflow fields
    script_status: Optional[str]       # "pending_review" | "approved" | "revision_requested" | "rejected" | "parse_error_final" | "retry_needed"
    revision_feedback: Optional[str]   # User's revision request
    revision_count: int                # Track revision iterations (max 3)
    script_parse_attempts: int         # Track JSON parse retry count (max 3)

    # Production tracking (NEW)
    production_id: Optional[str]          # RAGNAROK pipeline ID
    production_status: Optional[str]      # queued, processing, completed, failed
    production_progress: Optional[float]  # 0.0 to 1.0
    production_phase: Optional[str]       # Current production phase
    production_error: Optional[str]       # Error message if failed
    video_urls: Optional[dict]            # Platform → URL mapping {youtube: ..., tiktok: ..., instagram: ...}
    production_cost: Optional[float]      # Estimated cost in USD
    production_started_at: Optional[str]  # ISO timestamp

    # Asset Upload (NEW)
    uploaded_assets: Optional[List[dict]]  # List of {type, url, name}
    assets_reviewed: bool                  # Gate flag to ensure we asked
    _asset_just_received: Optional[str]    # Temp flag for logo upload acknowledgment

    # Video Preview Integration
    preview_url: Optional[str]             # Shareable preview link
    preview_id: Optional[str]              # Video preview gallery ID


# ============================================================================
# 2. EXTRACTION SCHEMA - Replaces Regex
# ============================================================================

class ExtractionSchema(BaseModel):
    """
    Pydantic model for structured LLM extraction.
    Using with_structured_output guarantees this format.
    """
    business_name: Optional[str] = Field(
        None,
        description="The official name of the company, brand, or business"
    )
    primary_offering: Optional[str] = Field(
        None,
        description="The specific product, service, or offering to promote"
    )
    target_demographic: Optional[str] = Field(
        None,
        description="The ideal customer profile - who the video is for"
    )
    call_to_action: Optional[str] = Field(
        None,
        description="What viewers should do after watching (visit, call, buy, book)"
    )
    tone: Optional[str] = Field(
        None,
        description="The style, vibe, or feeling (luxury, energetic, professional)"
    )
    top_rivals: Optional[List[str]] = Field(
        None,
        description="Names of 1-3 direct competitors"
    )


# ============================================================================
# 3. LLM SETUP
# ============================================================================

def get_llm():
    """Create ChatAnthropic client with explicit error logging."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    logger.info(f"[LLM] Creating ChatAnthropic: model={model}, api_key_exists={bool(api_key)}, api_key_prefix={api_key[:10] if api_key else 'NONE'}...")

    if not api_key:
        logger.error("[LLM] ANTHROPIC_API_KEY is not set!")
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    try:
        llm = ChatAnthropic(
            model=model,
            temperature=0.3,
            max_tokens=4096,  # Increased from 1024 - script generation needs ~3000-4000 tokens
            api_key=api_key
        )
        logger.info(f"[LLM] ChatAnthropic created successfully")
        return llm
    except Exception as e:
        logger.error(f"[LLM] Failed to create ChatAnthropic: {type(e).__name__}: {e}")
        raise


# ============================================================================
# 4. INTAKE AGENT NODE
# ============================================================================

INTAKE_PROMPT = """You are a Creative Director gathering information for a video commercial.

## CURRENT STATUS
- Business: {business_name}
- Product/Service: {primary_offering}
- Target Audience: {target_demographic}
- Call to Action: {call_to_action}
- Tone: {tone}
- Uploaded Assets: {uploaded_assets}
- Still need: {missing_fields}

## RULES
1. If user just uploaded a file (check "Uploaded Assets"), acknowledge it warmly FIRST with "Got your logo!" or similar
2. Then ask for the NEXT missing field specifically
3. Keep responses short (2-3 sentences)
4. NEVER say "Ready to create" until ALL 5 fields are filled
5. If user says "ready" but fields missing, explain what you still need

## RESPONSE FORMAT
If asset was just uploaded → "Got your [logo/image]! Now, [ask for next field]..."
Otherwise → Acknowledge their input → Ask for next missing field"""


# ============================================================================
# ASSET DETECTION HELPER
# ============================================================================

def _detect_assets(message: str) -> List[dict]:
    """
    Detect URLs and file references in message.
    Returns list of {type, url, name} dicts.
    """
    import re
    assets = []

    # 1. Regex for URLs (images/docs)
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, message)

    for url in urls:
        # Simple heuristic for file type
        filename = url.split('/')[-1].split('?')[0]
        lower_name = filename.lower()

        if any(ext in lower_name for ext in ['.png', '.jpg', '.jpeg', '.svg', '.webp', '.gif']):
            asset_type = "image"
        elif any(ext in lower_name for ext in ['.pdf', '.doc', '.docx', '.txt']):
            asset_type = "document"
        else:
            asset_type = "link"

        assets.append({"type": asset_type, "url": url, "name": filename})

    # 2. Detect frontend upload notifications
    # Pattern A: [User uploaded logo/image: filename.png]
    upload_pattern = r'\[User uploaded (?:logo/image|document): ([^\]]+)\]'
    upload_match = re.search(upload_pattern, message)

    # Pattern B: 📎 Uploaded: filename.png (emoji format from frontend)
    if not upload_match:
        emoji_upload_pattern = r'📎\s*(?:Uploaded|File attached):\s*([^\s\n]+)'
        upload_match = re.search(emoji_upload_pattern, message)

    if upload_match:
        filename = upload_match.group(1)
        lower_name = filename.lower()

        if any(ext in lower_name for ext in ['.png', '.jpg', '.jpeg', '.svg', '.webp', '.gif']):
            asset_type = "logo"  # Treat uploaded images as logos
        elif any(ext in lower_name for ext in ['.pdf', '.doc', '.docx', '.txt']):
            asset_type = "document"
        else:
            asset_type = "image"

        # Use placeholder URL since file is handled client-side
        assets.append({
            "type": asset_type,
            "url": f"uploaded://{filename}",  # Placeholder - frontend has the actual file
            "name": filename,
            "source": "frontend_upload"
        })
        logger.info(f"[AssetDetection] Detected frontend upload: {filename} (type: {asset_type})")

    # 3. Check for general upload intent keywords if no URL or pattern found
    if not assets and any(w in message.lower() for w in ["uploaded", "attached", "sending file", "here is the logo", "logo attached"]):
        # Generic upload acknowledgment - user mentioned uploading but we can't parse the filename
        logger.info("[AssetDetection] Detected upload intent keywords but no specific file pattern")
        # Still mark as having assets to proceed with the flow
        assets.append({
            "type": "unknown",
            "url": "uploaded://user_asset",
            "name": "user_uploaded_asset",
            "source": "intent_detection"
        })

    return assets


async def intake_node(state: VideoBriefState) -> dict:
    """
    Single LLM call for extraction, inline response generation.
    NO double LLM calls - extraction only, responses are deterministic.
    OPTIMIZATION: Skip LLM call if all 5 fields already filled.
    """
    # Get the last user message
    user_messages = [m for m in state.get("messages", []) if m.get("role") == "user"]
    if not user_messages:
        # First turn - greet
        new_state = dict(state)
        new_state["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": "Welcome! Let's create your commercial. What's the name of your business?"
        }]
        new_state["current_phase"] = "intake"
        new_state["is_complete"] = False
        return new_state

    last_message = user_messages[-1].get("content", "")

    # Check if all fields are already filled BEFORE making LLM call
    required = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]
    already_filled = all(state.get(f) for f in required)

    extracted_dict = {}

    # OPTIMIZATION: Skip LLM extraction if all fields already filled
    # This prevents hangs when processing upload notifications or skip commands
    if already_filled:
        logger.info(f"[IntakeAgent] All 5 fields already filled - skipping LLM extraction")
    else:
        # Build extraction prompt
        extraction_prompt = f"""Extract any of these fields from the user message:
- business_name: The company/brand name
- primary_offering: Product or service being promoted
- target_demographic: Who the ad is targeting
- call_to_action: What viewer should do
- tone: Style/vibe of the video

User said: {last_message}

Current state:
- business_name: {state.get('business_name', 'NOT SET')}
- primary_offering: {state.get('primary_offering', 'NOT SET')}
- target_demographic: {state.get('target_demographic', 'NOT SET')}
- call_to_action: {state.get('call_to_action', 'NOT SET')}
- tone: {state.get('tone', 'NOT SET')}

Extract ONLY what user explicitly stated. Do not guess or infer."""

        # SINGLE LLM call for extraction
        try:
            llm = get_llm()
            extractor = llm.with_structured_output(ExtractionSchema)
            extracted = await extractor.ainvoke(extraction_prompt)
            extracted_dict = extracted.dict(exclude_none=True) if hasattr(extracted, 'dict') else {}
            logger.info(f"[IntakeAgent] Extracted: {extracted_dict}")
        except Exception as e:
            logger.error(f"[IntakeAgent] Extraction failed: {e}")
            extracted_dict = {}

    new_state = dict(state)

    # FILTER: Reject placeholder values
    PLACEHOLDER_VALUES = {"unknown", "<unknown>", "[not provided]", "[not yet provided]",
                          "not provided", "n/a", "na", "none", "null", "undefined",
                          "[unknown]", "tbd", "to be determined", "not set"}

    # Update state with extracted values
    for field, value in extracted_dict.items():
        if value and not state.get(field):
            value_lower = str(value).lower().strip()
            if value_lower in PLACEHOLDER_VALUES:
                logger.warning(f"[IntakeAgent] Skipping placeholder: {field}={value}")
                continue
            new_state[field] = value
            logger.info(f"[IntakeAgent] Set {field} = {value}")

    # Calculate missing fields
    required = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]
    missing = [f for f in required if not new_state.get(f)]
    new_state["missing_fields"] = missing

    logger.info(f"[IntakeAgent] Missing: {missing}, assets_reviewed={new_state.get('assets_reviewed')}")

    # GENERATE RESPONSE INLINE - NO EXTRA LLM CALL
    # Deterministic field questions
    field_questions = {
        "business_name": "What's the name of your business?",
        "primary_offering": "What product or service are you promoting?",
        "target_demographic": "Who's the ideal audience for this commercial?",
        "call_to_action": "What's the one thing you want viewers to do after watching?",
        "tone": "What's the vibe—professional, fun, inspiring, edgy?",
    }

    if missing:
        # CASE A: Still need fields
        next_field = missing[0]

        # Check if asset was just received (flagged by process_message)
        if new_state.get("_asset_just_received"):
            filename = new_state.get("_asset_just_received")
            response_text = f"✅ Got your logo: {filename}! Now, {field_questions[next_field].lower()}"
            del new_state["_asset_just_received"]
        else:
            response_text = field_questions[next_field]

        new_state["is_complete"] = False
        new_state["current_phase"] = "intake"

    elif new_state.get("_asset_just_received"):
        # Asset was just uploaded AND all fields complete
        filename = new_state.get("_asset_just_received")
        response_text = f"✅ Got your logo: {filename}! I have everything I need. Ready to start research?"
        new_state["is_complete"] = True
        new_state["current_phase"] = "research"
        new_state["assets_reviewed"] = True
        del new_state["_asset_just_received"]

    elif not new_state.get("assets_reviewed"):
        # CASE B: All 5 fields done, ask about assets
        response_text = f"Perfect! I have all the details for **{new_state.get('business_name')}**.\n\nDo you have a logo or brand images to include? Upload now, or say 'skip' to proceed."
        new_state["assets_reviewed"] = True
        new_state["is_complete"] = False  # NOT complete until they respond

    else:
        # CASE C: Fields done, assets asked, user responded
        last_lower = last_message.lower()
        skip_words = ["skip", "no", "none", "proceed", "continue", "no logo", "don't have", "nope", "pass"]

        # Check for new assets in this message
        new_assets = _detect_assets(last_message)
        if new_assets:
            current_assets = new_state.get("uploaded_assets") or []
            current_assets.extend(new_assets)
            new_state["uploaded_assets"] = current_assets

        # Check if user is providing additional instructions for previously uploaded assets
        has_existing_assets = bool(new_state.get("uploaded_assets"))
        is_asset_instruction = any(word in last_lower for word in [
            "place", "position", "bottom", "top", "left", "right", "corner",
            "frame", "end", "beginning", "watermark", "overlay", "logo"
        ])

        # Use explicit flag to determine if we should proceed to research
        should_proceed = False

        if new_assets:
            response_text = f"✅ Assets received! Starting research for {new_state.get('business_name')}..."
            should_proceed = True
        elif any(word in last_lower for word in skip_words):
            response_text = f"Got it! Starting research for {new_state.get('business_name')}..."
            should_proceed = True
        elif has_existing_assets and is_asset_instruction:
            # User is giving placement instructions for already-uploaded logo
            response_text = f"✅ Got it - logo placement noted! Starting research for {new_state.get('business_name')}..."
            logger.info(f"[IntakeAgent] Asset instruction detected: {last_message[:50]}")
            should_proceed = True
        elif has_existing_assets:
            # User has assets and is responding with something else - proceed
            response_text = f"Perfect! Starting research for {new_state.get('business_name')}..."
            should_proceed = True
        else:
            # No assets, not skipping - ask again
            response_text = f"Would you like to upload a logo, or say 'skip' to proceed without one?"
            should_proceed = False

        # FIXED: Explicitly set based on flag, not inherited state
        if should_proceed:
            new_state["is_complete"] = True
            new_state["current_phase"] = "research"
        else:
            new_state["is_complete"] = False
            new_state["current_phase"] = "intake"

    new_state["messages"] = state.get("messages", []) + [{
        "role": "assistant",
        "content": response_text
    }]

    return new_state


# ============================================================================
# 5. RESEARCHER NODE (Trinity Integration Placeholder)
# ============================================================================

def _infer_industry(primary_offering: str) -> str:
    """
    Infer industry from the primary offering description.
    Trinity needs an industry for market segmentation.
    """
    offering_lower = primary_offering.lower()

    industry_keywords = {
        "technology": ["software", "app", "saas", "tech", "ai", "cloud", "digital"],
        "healthcare": ["health", "medical", "clinic", "wellness", "therapy", "dental"],
        "beauty": ["salon", "spa", "beauty", "hair", "nail", "skincare", "cosmetic"],
        "fitness": ["gym", "fitness", "workout", "training", "yoga", "pilates"],
        "food": ["restaurant", "food", "catering", "bakery", "cafe", "coffee"],
        "retail": ["shop", "store", "boutique", "retail", "ecommerce", "products"],
        "professional_services": ["consulting", "legal", "accounting", "agency", "marketing"],
        "real_estate": ["real estate", "property", "home", "apartment", "rental"],
        "automotive": ["car", "auto", "vehicle", "mechanic", "dealer"],
        "education": ["school", "training", "course", "tutoring", "education", "learning"],
        "finance": ["finance", "investment", "insurance", "banking", "loan", "mortgage"],
    }

    for industry, keywords in industry_keywords.items():
        if any(kw in offering_lower for kw in keywords):
            return industry

    return "general_business"


def _build_goals(state: VideoBriefState) -> List[str]:
    """
    Build goals list from video brief for Trinity research.
    """
    goals = []

    # Primary goal from CTA
    cta = state.get("call_to_action", "")
    if cta:
        cta_lower = cta.lower()
        if any(w in cta_lower for w in ["book", "schedule", "appointment"]):
            goals.append("drive_appointments")
        elif any(w in cta_lower for w in ["buy", "purchase", "order", "shop"]):
            goals.append("increase_sales")
        elif any(w in cta_lower for w in ["call", "contact", "reach"]):
            goals.append("generate_leads")
        elif any(w in cta_lower for w in ["visit", "website", "learn"]):
            goals.append("drive_traffic")
        elif any(w in cta_lower for w in ["sign up", "subscribe", "join"]):
            goals.append("grow_subscribers")
        else:
            goals.append("increase_awareness")

    # Add tone-based goal
    tone = state.get("tone", "")
    if tone:
        tone_lower = tone.lower()
        if any(w in tone_lower for w in ["luxury", "premium", "exclusive"]):
            goals.append("premium_positioning")
        elif any(w in tone_lower for w in ["fun", "energetic", "exciting"]):
            goals.append("viral_potential")
        elif any(w in tone_lower for w in ["trust", "professional", "reliable"]):
            goals.append("build_trust")

    # Default if no goals inferred
    if not goals:
        goals = ["increase_awareness", "drive_engagement"]

    return goals


def _summarize_insights(research_data: dict) -> str:
    """
    Create a human-readable summary of Trinity research findings.
    """
    summary_parts = []

    # Check for different data structures Trinity might return
    if "market_analysis" in research_data:
        market = research_data["market_analysis"]
        if isinstance(market, dict):
            if "key_insight" in market:
                summary_parts.append(f"**Key Insight:** {market['key_insight']}")
            if "market_size" in market:
                summary_parts.append(f"**Market Size:** {market['market_size']}")

    if "competitors" in research_data:
        competitors = research_data["competitors"]
        if isinstance(competitors, list) and len(competitors) > 0:
            comp_names = [c.get("name", "Unknown") for c in competitors[:3] if isinstance(c, dict)]
            if comp_names:
                summary_parts.append(f"**Top Competitors:** {', '.join(comp_names)}")

    if "audience_insights" in research_data:
        audience = research_data["audience_insights"]
        if isinstance(audience, dict) and "summary" in audience:
            summary_parts.append(f"**Audience:** {audience['summary']}")

    if "positioning" in research_data:
        pos = research_data["positioning"]
        if isinstance(pos, str):
            summary_parts.append(f"**Recommended Position:** {pos}")

    if "trends" in research_data:
        trends = research_data["trends"]
        if isinstance(trends, list) and len(trends) > 0:
            trend_items = trends[:3] if isinstance(trends[0], str) else [t.get("name", "") for t in trends[:3] if isinstance(t, dict)]
            if trend_items:
                summary_parts.append(f"**Trending:** {', '.join(filter(None, trend_items))}")

    # If no structured data, check for raw summary
    if not summary_parts:
        if "summary" in research_data:
            return research_data["summary"]
        elif "message" in research_data:
            return research_data["message"]
        else:
            return "Market intelligence gathered. Ready to create a targeted commercial."

    return "\n".join(summary_parts)


async def researcher_node(state: VideoBriefState) -> dict:
    """
    Trinity Market Intelligence Integration.

    Calls the real Trinity API at /api/genesis/research to get:
    - Competitor analysis
    - Market positioning insights
    - Audience intelligence
    - Trend data

    Maps video brief fields to Trinity's expected schema.
    """
    logger.info(f"[ResearcherAgent] Starting Trinity research for {state.get('business_name', 'unknown')}")

    new_state = dict(state)
    new_state["current_phase"] = "research"

    # Map video brief to Trinity request schema
    trinity_payload = {
        "session_id": state.get("session_id", f"v2-{int(time.time())}"),
        "business_name": state.get("business_name", "Unknown Business"),
        "industry": _infer_industry(state.get("primary_offering", "")),
        "website_url": None,  # Could be added to intake if needed
        "goals": _build_goals(state)
    }

    logger.info(f"[ResearcherAgent] Trinity payload: {trinity_payload}")

    try:
        async with httpx.AsyncClient(timeout=TRINITY_TIMEOUT) as client:
            response = await client.post(
                TRINITY_URL,
                json=trinity_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                research_data = response.json()
                logger.info(f"[ResearcherAgent] Trinity returned: {list(research_data.keys())}")

                # Store research findings
                new_state["research_findings"] = research_data

                # Build success message with key insights
                insights_summary = _summarize_insights(research_data)
                response_text = (
                    f"🔍 **Market Research Complete!**\n\n"
                    f"{insights_summary}\n\n"
                    f"I have deep competitive intelligence on your market. "
                    f"Ready to craft a script that positions {state.get('business_name', 'you')} to win."
                )

            else:
                logger.error(f"[ResearcherAgent] Trinity error: {response.status_code} - {response.text}")
                # Graceful degradation - continue without research
                new_state["research_findings"] = {"status": "unavailable", "reason": f"API returned {response.status_code}"}
                response_text = (
                    f"📊 I've gathered the core information about {state.get('business_name', 'your business')}. "
                    f"Let me craft a compelling script based on what we know."
                )

    except httpx.TimeoutException:
        logger.error("[ResearcherAgent] Trinity timeout")
        new_state["research_findings"] = {"status": "timeout"}
        response_text = (
            f"⚡ Moving forward with script creation for {state.get('business_name', 'your business')}. "
            f"We have everything we need to create something powerful."
        )

    except Exception as e:
        logger.error(f"[ResearcherAgent] Trinity exception: {e}")
        new_state["research_findings"] = {"status": "error", "message": str(e)}
        response_text = (
            f"Let me proceed with creating your commercial script for {state.get('business_name', 'your business')}."
        )

    # Add response to messages
    new_state["messages"] = list(state.get("messages", [])) + [{
        "role": "assistant",
        "content": response_text
    }]

    return new_state


def _build_research_summary(research_findings: dict) -> str:
    """
    Convert Trinity research findings into a summary for the script prompt.
    """
    if not research_findings:
        return "No market research available. Create a compelling general commercial."

    # Handle different Trinity response structures
    summary_parts = []

    # Check for pipeline status (async research)
    if research_findings.get("status") == "started":
        return "Market research is being gathered. Create a compelling commercial based on the brief."

    # Extract key insights
    if "market_analysis" in research_findings:
        market = research_findings["market_analysis"]
        if isinstance(market, dict):
            if market.get("key_insight"):
                summary_parts.append(f"Key Insight: {market['key_insight']}")
            if market.get("opportunity"):
                summary_parts.append(f"Opportunity: {market['opportunity']}")

    if "competitors" in research_findings:
        competitors = research_findings["competitors"]
        if isinstance(competitors, list) and competitors:
            comp_info = []
            for c in competitors[:3]:
                if isinstance(c, dict):
                    name = c.get("name", "Unknown")
                    weakness = c.get("weakness", c.get("gap", ""))
                    if weakness:
                        comp_info.append(f"{name} (gap: {weakness})")
                    else:
                        comp_info.append(name)
            if comp_info:
                summary_parts.append(f"Competitors: {', '.join(comp_info)}")

    if "audience_insights" in research_findings:
        audience = research_findings["audience_insights"]
        if isinstance(audience, dict):
            if audience.get("pain_points"):
                summary_parts.append(f"Audience Pain Points: {audience['pain_points']}")
            if audience.get("desires"):
                summary_parts.append(f"Audience Desires: {audience['desires']}")

    if "positioning" in research_findings:
        summary_parts.append(f"Recommended Positioning: {research_findings['positioning']}")

    if "trends" in research_findings:
        trends = research_findings["trends"]
        if isinstance(trends, list) and trends:
            trend_names = [t if isinstance(t, str) else t.get("name", "") for t in trends[:3]]
            summary_parts.append(f"Current Trends: {', '.join(filter(None, trend_names))}")

    if summary_parts:
        return "\n".join(summary_parts)

    # Fallback for unknown structure
    return f"Research data available: {list(research_findings.keys())}"


def _parse_script_json(response_text: str) -> Optional[dict]:
    """
    Parse JSON from LLM response, handling common formatting issues.
    """
    import re

    def strip_js_comments(text: str) -> str:
        """Strip JavaScript-style // comments that LLMs sometimes add to JSON."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just comments
            if stripped.startswith('//'):
                continue
            # Remove trailing // comments (but not if inside a quoted string)
            if '//' in line and not re.search(r'"[^"]*//[^"]*"', line):
                line = re.sub(r'\s*//.*$', '', line)
            cleaned.append(line)
        return '\n'.join(cleaned)

    # Try direct parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.debug(f"[Parser] Direct parse failed: {e}")

    # Strip JS-style comments and try again
    cleaned_text = strip_js_comments(response_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.debug(f"[Parser] After comment strip failed: {e}")

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        extracted = strip_js_comments(json_match.group(1))
        logger.info(f"[Parser] Markdown extraction found {len(extracted)} chars, first 200: {extracted[:200]}")
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.error(f"[Parser] Markdown JSON parse failed at pos {e.pos}: {e.msg}")
            # Log context around error position
            if e.pos is not None and e.pos < len(extracted):
                start = max(0, e.pos - 50)
                end = min(len(extracted), e.pos + 50)
                logger.error(f"[Parser] Context around error: ...{extracted[start:end]}...")
    else:
        logger.warning(f"[Parser] No markdown code block found in response")

    # Try to find JSON object in response
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        extracted = strip_js_comments(json_match.group(0))
        logger.info(f"[Parser] Brace extraction found {len(extracted)} chars")
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.error(f"[Parser] Brace JSON parse failed at pos {e.pos}: {e.msg}")
            if e.pos is not None and e.pos < len(extracted):
                start = max(0, e.pos - 50)
                end = min(len(extracted), e.pos + 50)
                logger.error(f"[Parser] Context around error: ...{extracted[start:end]}...")

    logger.error(f"[Parser] All parsing methods failed")
    return None


def _format_script_preview(script_data: dict) -> str:
    """
    Format script data into a readable preview for the user.
    """
    preview_parts = []

    scenes = script_data.get("scenes", [])
    for scene in scenes:
        scene_num = scene.get("scene_number", "?")
        timestamp = scene.get("timestamp", "")
        scene_type = scene.get("type", "").upper()
        visual = scene.get("visual_description", "")
        narration = scene.get("narration", "")

        preview_parts.append(
            f"**Scene {scene_num}** ({timestamp}) - {scene_type}\n"
            f"📹 *{visual}*\n"
            f"🎙️ \"{narration}\""
        )

    if preview_parts:
        return "\n\n".join(preview_parts)

    # Fallback to voiceover script
    voiceover = script_data.get("voiceover_full_script", "")
    if voiceover:
        return f"**Voiceover Script:**\n\n\"{voiceover}\""

    return "Script generated - see details below."


def _validate_script_word_count(script_data: dict) -> dict:
    """
    Validate that script narration fits within 64-second duration.

    At 2.5 words/second:
    - 16 seconds per scene = 40 words max
    - 64 seconds total = 160 words max

    Returns validation result with warnings and optionally truncated text.
    """
    MAX_WORDS_PER_SCENE = 40
    MAX_TOTAL_WORDS = 160

    result = {
        "valid": True,
        "warnings": [],
        "scene_word_counts": [],
        "total_words": 0,
        "recommended_cuts": []
    }

    total_words = 0
    scenes = script_data.get("scenes", [])

    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "")
        word_count = len(narration.split())
        result["scene_word_counts"].append({
            "scene": i + 1,
            "type": scene.get("type", "unknown"),
            "word_count": word_count,
            "over_limit": word_count > MAX_WORDS_PER_SCENE
        })

        if word_count > MAX_WORDS_PER_SCENE:
            over_by = word_count - MAX_WORDS_PER_SCENE
            result["warnings"].append(
                f"Scene {i+1} ({scene.get('type', 'unknown')}): {word_count} words "
                f"(OVER by {over_by} words - will be cut off!)"
            )
            result["valid"] = False

        total_words += word_count

    result["total_words"] = total_words

    # Also check voiceover_full_script if present
    voiceover = script_data.get("voiceover_full_script", "")
    if voiceover:
        vo_word_count = len(voiceover.split())
        result["voiceover_word_count"] = vo_word_count
        if vo_word_count > MAX_TOTAL_WORDS:
            over_by = vo_word_count - MAX_TOTAL_WORDS
            result["warnings"].append(
                f"Full voiceover: {vo_word_count} words "
                f"(OVER by {over_by} words - audio will be truncated!)"
            )
            result["valid"] = False

    if total_words > MAX_TOTAL_WORDS:
        over_by = total_words - MAX_TOTAL_WORDS
        result["warnings"].append(
            f"Total narration: {total_words} words "
            f"(OVER by {over_by} words - reduce by ~{over_by} words)"
        )
        result["valid"] = False

    # Log validation result
    if result["warnings"]:
        logger.warning(f"[ScriptValidation] Word count issues: {result['warnings']}")
    else:
        logger.info(f"[ScriptValidation] OK - {total_words} words (max {MAX_TOTAL_WORDS})")

    return result


async def script_writer_node(state: VideoBriefState) -> dict:
    """
    Script Writer Agent - Generates commercial scripts using brief + research.

    Takes:
    - Video brief (5 core fields)
    - Trinity research findings

    Produces:
    - Structured 64-second commercial script (Barrios A2I standard)
    - 4 scenes: HOOK, PROBLEM, SOLUTION, CTA (16 seconds each)
    - Complete voiceover script
    - Visual direction notes
    """
    logger.info(f"[ScriptWriterAgent] Generating script for {state.get('business_name', 'unknown')}")

    new_state = dict(state)
    new_state["current_phase"] = "scripting"

    # Build research summary for the prompt
    research_summary = _build_research_summary(state.get("research_findings", {}))

    # Check if this is a revision
    revision_feedback = state.get("revision_feedback")
    is_revision = revision_feedback is not None and state.get("script_status") == "revision_requested"

    # Build the prompt with configurable scene parameters
    scene_count = COMMERCIAL_CONFIG["scene_count"]
    scene_duration = COMMERCIAL_CONFIG["scene_duration_seconds"]
    total_duration = COMMERCIAL_CONFIG["duration_seconds"]
    words_per_scene = int(scene_duration * 2.5)  # 2.5 words per second
    total_words = scene_count * words_per_scene

    base_prompt = SCRIPT_WRITER_PROMPT.format(
        business_name=state.get("business_name", "the business"),
        primary_offering=state.get("primary_offering", "their product/service"),
        target_demographic=state.get("target_demographic", "their target audience"),
        call_to_action=state.get("call_to_action", "take action"),
        tone=state.get("tone", "professional"),
        research_summary=research_summary,
        scene_count=scene_count,
        scene_duration=scene_duration,
        total_duration=total_duration,
        words_per_scene=words_per_scene,
        total_words=total_words,
        scene_structure=generate_scene_structure()
    )

    # Add revision context if applicable
    if is_revision:
        previous_script = state.get("script_draft", {})
        previous_title = previous_script.get("title", "previous version")

        prompt = (
            f"{base_prompt}\n\n"
            f"## REVISION REQUEST\n"
            f"The previous script titled \"{previous_title}\" needs changes.\n"
            f"User feedback: \"{revision_feedback}\"\n\n"
            f"Please generate a NEW version that addresses this feedback while "
            f"maintaining the core messaging and {scene_count}-scene structure ({total_duration} seconds total)."
        )
        logger.info(f"[ScriptWriterAgent] Revision mode - feedback: {revision_feedback[:100]}")
    else:
        prompt = base_prompt

    try:
        llm = get_llm()

        # Request JSON output
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a commercial scriptwriter. Always respond with valid JSON only, no markdown."},
            {"role": "user", "content": prompt}
        ])

        response_text = response.content

        # Parse JSON from response
        script_data = _parse_script_json(response_text)

        if script_data:
            logger.info(f"[ScriptWriterAgent] Script generated: {script_data.get('title', 'Untitled')}")

            # Validate word count for 64-second video
            word_validation = _validate_script_word_count(script_data)
            new_state["script_word_validation"] = word_validation

            new_state["script_draft"] = script_data
            new_state["script_status"] = "pending_review"

            # Clear revision feedback after applying it
            if is_revision:
                new_state["revision_feedback"] = None

            # Build user-friendly response
            script_preview = _format_script_preview(script_data)

            revision_note = ""
            if is_revision:
                revision_count = state.get("revision_count", 1)
                revision_note = f"**Revision #{revision_count}** - Updated based on your feedback.\n\n"

            # Add word count warning if over limit
            word_count_warning = ""
            if not word_validation["valid"]:
                word_count_warning = (
                    f"\n\n⚠️ **TIMING WARNING:**\n"
                    f"Word count: {word_validation['total_words']}/160 words\n"
                    f"Issues: {'; '.join(word_validation['warnings'][:2])}\n"
                    f"*Consider shortening to fit 64-second duration.*\n"
                )

            response_message = (
                f"✍️ **Script {'Revised' if is_revision else 'Draft Complete'}!**\n\n"
                f"{revision_note}"
                f"**Title:** {script_data.get('title', 'Untitled Commercial')}\n"
                f"**Duration:** {script_data.get('duration_seconds', 30)} seconds\n"
                f"**Scenes:** {len(script_data.get('scenes', []))}\n"
                f"**Word Count:** {word_validation['total_words']}/160 ({'✅' if word_validation['valid'] else '⚠️ OVER'})\n"
                f"{word_count_warning}\n"
                f"---\n\n"
                f"{script_preview}\n\n"
                f"---\n\n"
                f"Does this script capture what you're looking for?\n"
                f"- **Approve** - Say 'looks good' to proceed to production\n"
                f"- **Revise** - Tell me what to change (e.g., 'make it more upbeat')"
            )
        else:
            # Track retry attempts
            attempts = state.get("script_parse_attempts", 0) + 1
            # DEBUG: Log what the LLM actually returned
            logger.error(f"[ScriptWriterAgent] Failed to parse script JSON (attempt {attempts}/3)")
            logger.error(f"[ScriptWriterAgent] Raw response preview: {response_text[:300]}...")
            new_state["script_draft"] = {"status": "parse_error", "raw": response_text[:500]}
            new_state["script_parse_attempts"] = attempts

            if attempts >= 3:
                # Max retries reached - stop retrying, go to reviewer for fallback
                new_state["script_status"] = "parse_error_final"
                response_message = (
                    f"I'm having trouble generating the script in the expected format. "
                    f"Let me show you what I have so far for {state.get('business_name', 'your business')} "
                    f"and we can work together to refine it."
                )
            else:
                # Allow retry
                new_state["script_status"] = "retry_needed"
                response_message = (
                    f"I've drafted a script concept for {state.get('business_name', 'your business')}. "
                    f"Let me refine it and get it ready for your review."
                )

    except Exception as e:
        logger.error(f"[ScriptWriterAgent] Error: {e}")
        new_state["script_draft"] = {"status": "error", "message": str(e)}
        response_message = (
            f"I'm working on your script for {state.get('business_name', 'your business')}. "
            f"Give me a moment to finalize it."
        )

    # Add response to messages
    new_state["messages"] = list(state.get("messages", [])) + [{
        "role": "assistant",
        "content": response_message
    }]

    return new_state


# ============================================================================
# 6. DETERMINISTIC ROUTING
# ============================================================================

def route_after_intake(state: VideoBriefState) -> Literal["intake", "researcher"]:
    """
    Three-gate check for proper state transitions.
    CODE decides routing, NOT the AI.
    """
    required = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]

    # Gate 1: Have all required fields?
    if not all(state.get(f) for f in required):
        logger.info(f"[Router] Gate 1 FAIL: Missing fields → stay in intake")
        return "intake"

    # Gate 2: Have we asked about assets?
    if not state.get("assets_reviewed"):
        logger.info(f"[Router] Gate 2 FAIL: Haven't asked about assets → stay in intake")
        return "intake"

    # Gate 3: Is intake truly complete?
    if state.get("is_complete"):
        logger.info(f"[Router] All gates PASS → researcher")
        return "researcher"

    # Still waiting for user to respond to asset prompt
    logger.info(f"[Router] Gate 3 FAIL: Waiting for asset response → stay in intake")
    return "intake"


def route_after_research(state: VideoBriefState) -> Literal["script_writer", "researcher"]:
    """
    Route after research node.
    If research is complete (or we're continuing without it), go to script writer.
    """
    research = state.get("research_findings")

    # If we have research findings (success or graceful failure), proceed to scripting
    if research is not None:
        logger.info("[Router] Research complete → script_writer")
        return "script_writer"

    # This shouldn't happen, but stay in research if no findings
    logger.info("[Router] No research findings → stay in researcher")
    return "researcher"


def _detect_review_intent(message: str) -> str:
    """
    Detect user's intent from their review message.
    Returns: "approved" | "revision" | "rejected" | "unclear"
    """
    message_lower = message.lower()

    # Check rejection first (highest priority)
    for pattern in REJECTION_PATTERNS:
        if pattern in message_lower:
            return "rejected"

    # Check approval
    approval_score = sum(1 for p in APPROVAL_PATTERNS if p in message_lower)

    # Check revision
    revision_score = sum(1 for p in REVISION_PATTERNS if p in message_lower)

    # If message is long (>50 chars) and contains revision words, likely revision
    if len(message) > 50 and revision_score > 0:
        return "revision"

    # If clear approval signals
    if approval_score >= 1 and revision_score == 0:
        return "approved"

    # If revision signals present
    if revision_score >= 1:
        return "revision"

    # Short positive responses
    if message_lower in ["yes", "ok", "okay", "sure", "yep", "yeah", "y"]:
        return "approved"

    return "unclear"


async def reviewer_node(state: VideoBriefState) -> dict:
    """
    Reviewer Node - Handles user feedback on script draft.

    Detects:
    - APPROVAL: User likes the script → ready for production
    - REVISION: User wants changes → store feedback, regenerate
    - REJECTION: User wants to stop → end workflow

    This creates a human-in-the-loop approval system.
    """
    logger.info(f"[ReviewerAgent] Processing user feedback")

    new_state = dict(state)

    # Get the last user message (their feedback)
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]

    if not user_messages:
        # No user message yet, stay in review
        new_state["script_status"] = "pending_review"
        return new_state

    last_user_message = user_messages[-1].get("content", "").lower().strip()
    logger.info(f"[ReviewerAgent] Analyzing: '{last_user_message[:100]}'")

    # Detect intent
    intent = _detect_review_intent(last_user_message)
    logger.info(f"[ReviewerAgent] Detected intent: {intent}")

    if intent == "approved":
        new_state["script_status"] = "approved"
        new_state["current_phase"] = "approved"
        response_text = (
            "🎬 **Script Approved!**\n\n"
            f"Excellent! Your commercial script for **{state.get('business_name', 'your business')}** "
            f"is locked and ready for production.\n\n"
            f"The next step is video generation with RAGNAROK. "
            f"Would you like me to start producing your commercial?"
        )

    elif intent == "revision":
        revision_count = state.get("revision_count", 0) + 1

        if revision_count > 3:
            # Max revisions reached
            new_state["script_status"] = "max_revisions"
            new_state["current_phase"] = "max_revisions"
            response_text = (
                "📝 We've done several revisions on this script. "
                "To keep things moving, I recommend we either:\n\n"
                "1. **Approve** the current version and refine in production\n"
                "2. **Start fresh** with a new brief\n\n"
                "What would you like to do?"
            )
        else:
            new_state["script_status"] = "revision_requested"
            new_state["revision_feedback"] = last_user_message
            new_state["revision_count"] = revision_count
            new_state["current_phase"] = "revising"
            response_text = (
                f"✏️ **Revision #{revision_count}**\n\n"
                f"Got it! I'll incorporate your feedback:\n"
                f"> *\"{last_user_message[:200]}{'...' if len(last_user_message) > 200 else ''}\"*\n\n"
                f"Generating updated script..."
            )

    elif intent == "rejected":
        new_state["script_status"] = "rejected"
        new_state["current_phase"] = "rejected"
        response_text = (
            "No problem! I've set this script aside.\n\n"
            "Would you like to:\n"
            "1. **Start over** with a new commercial concept?\n"
            "2. **End** this session?\n\n"
            "Just let me know how you'd like to proceed."
        )

    else:
        # Unclear intent - ask for clarification
        new_state["script_status"] = "pending_review"
        response_text = (
            "I want to make sure I understand your feedback.\n\n"
            "Could you clarify:\n"
            "- **Approve** - Say 'looks good' or 'approve' to proceed to production\n"
            "- **Revise** - Tell me what changes you'd like (e.g., 'make it more upbeat')\n"
            "- **Start over** - Say 'start over' to begin fresh\n\n"
            "What would you like to do with this script?"
        )

    # Add response to messages
    new_state["messages"] = list(state.get("messages", [])) + [{
        "role": "assistant",
        "content": response_text
    }]

    return new_state


# ============================================================================
# PRODUCTION HELPER FUNCTIONS
# ============================================================================

async def _call_production_api(
    session_id: str,
    brief: dict,
    script: dict,
    industry: str,
    business_name: str,
    style: str = "modern",
    target_platforms: Optional[list] = None
) -> dict:
    """
    Call the RAGNAROK production API to start video generation.

    Args:
        session_id: Session identifier
        brief: Complete video brief with all 5 fields
        script: Approved script from script_writer_node
        industry: Inferred industry category
        business_name: Business name for tracking
        style: Visual style (modern, premium, dynamic, etc.)
        target_platforms: List of platforms (default: youtube, tiktok, instagram)

    Returns:
        Dict with production_id on success, error message on failure
    """
    if target_platforms is None:
        target_platforms = ["youtube", "tiktok", "instagram"]

    payload = {
        "brief": {
            **brief,
            "script": script,
            "approved": True,
            "approved_at": datetime.now(timezone.utc).isoformat()
        },
        "industry": industry,
        "business_name": business_name,
        "style": style,
        "goals": brief.get("goals", ["brand_awareness"]),
        "target_platforms": target_platforms
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Use /api/production/trigger for fire-and-forget background execution
            # This returns JSON and starts the RAGNAROK pipeline in the background
            response = await client.post(
                f"{GENESIS_API_BASE}/api/production/trigger/{session_id}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"[ProductionAPI] Production triggered for session {session_id}: {result.get('production_id')}")
                return result
            else:
                return {
                    "success": False,
                    "error": f"Production API returned {response.status_code}: {response.text[:200]}"
                }
    except httpx.TimeoutException:
        return {"success": False, "error": "Production API timeout (30s exceeded)"}
    except Exception as e:
        logger.error(f"[ProductionAPI] Error: {e}")
        return {"success": False, "error": f"Production API error: {str(e)}"}


def _infer_industry_for_production(primary_offering: str, research_findings: dict) -> str:
    """
    Infer industry category from product/service description.
    Used by RAGNAROK to select appropriate video templates.

    NOTE: This is the production-specific version that also checks research findings.
    The simpler _infer_industry() at line 449 is used during research phase.
    """
    offering_lower = (primary_offering or "").lower()

    # Keyword matching for industry categories
    industry_keywords = {
        "technology": ["software", "app", "saas", "tech", "digital", "ai", "automation", "platform"],
        "healthcare": ["health", "medical", "clinic", "doctor", "therapy", "wellness", "fitness"],
        "finance": ["finance", "bank", "investment", "insurance", "loan", "mortgage", "crypto"],
        "retail": ["shop", "store", "ecommerce", "product", "clothing", "fashion", "jewelry"],
        "food": ["restaurant", "food", "cafe", "catering", "bakery", "coffee", "meal"],
        "beauty": ["salon", "spa", "beauty", "hair", "skin", "cosmetic", "makeup", "nail"],
        "real_estate": ["real estate", "property", "home", "apartment", "realtor", "housing"],
        "education": ["school", "training", "course", "education", "tutoring", "academy"],
        "automotive": ["car", "auto", "vehicle", "mechanic", "dealership", "automotive"],
        "legal": ["law", "legal", "attorney", "lawyer", "court"],
        "marketing": ["marketing", "advertising", "agency", "creative", "branding"],
    }

    for industry, keywords in industry_keywords.items():
        if any(kw in offering_lower for kw in keywords):
            return industry

    # Check research findings for industry hints
    if research_findings:
        research_industry = research_findings.get("industry", "")
        if research_industry:
            return research_industry

    return "general"  # Default fallback


def _map_tone_to_style(tone: str) -> str:
    """
    Map conversational tone to RAGNAROK visual style engine.

    Tone (from user) → Style (for video generation)
    """
    tone_style_map = {
        "professional": "corporate",
        "friendly": "modern",
        "luxurious": "premium",
        "elegant": "premium",
        "energetic": "dynamic",
        "playful": "fun",
        "serious": "corporate",
        "warm": "lifestyle",
        "casual": "modern",
        "sophisticated": "premium",
        "bold": "dynamic",
        "minimalist": "clean",
    }

    tone_lower = (tone or "").lower()
    for key, style in tone_style_map.items():
        if key in tone_lower:
            return style

    return "modern"  # Default fallback


async def production_node(state: VideoBriefState) -> dict:
    """
    Production Node: Triggers RAGNAROK video generation.

    Entry condition: script_status == "approved"

    Workflow:
    1. Validate script is approved and exists
    2. Build production payload from state
    3. Call RAGNAROK API (/api/production/start)
    4. Store production_id for tracking
    5. Return confirmation or error message

    Exit: Always goes to END (terminal node)
    """
    logger.info(f"[ProductionNode] Starting for session {state.get('session_id', 'unknown')}")

    new_state = dict(state)

    # Validation: Script must be approved
    if state.get("script_status") != "approved":
        logger.warning(f"[ProductionNode] Script not approved: {state.get('script_status')}")
        new_state["production_status"] = "failed"
        new_state["production_error"] = "Script must be approved before production"
        new_state["messages"] = list(state.get("messages", [])) + [{
            "role": "assistant",
            "content": "⚠️ The script needs to be approved before we can start video production. Please review and approve the script first."
        }]
        return new_state

    # Validation: Script draft must exist
    script_draft = state.get("script_draft")
    if not script_draft:
        logger.warning("[ProductionNode] No script draft found")
        new_state["production_status"] = "failed"
        new_state["production_error"] = "No script found"
        new_state["messages"] = list(state.get("messages", [])) + [{
            "role": "assistant",
            "content": "⚠️ No script found. Please generate a script first."
        }]
        return new_state

    # Extract required fields
    business_name = state.get("business_name", "")
    primary_offering = state.get("primary_offering", "")
    target_demographic = state.get("target_demographic", "")
    call_to_action = state.get("call_to_action", "")
    tone = state.get("tone", "professional")
    research_findings = state.get("research_findings", {})

    # Build complete brief
    brief = {
        "business_name": business_name,
        "primary_offering": primary_offering,
        "target_demographic": target_demographic,
        "call_to_action": call_to_action,
        "tone": tone,
        "research_data": research_findings,
        "assets": state.get("uploaded_assets", []),  # Pass assets to RAGNAROK
        "goals": ["brand_awareness", "lead_generation"]
    }

    # Infer industry for template selection
    industry = _infer_industry_for_production(primary_offering, research_findings)
    logger.info(f"[ProductionNode] Inferred industry: {industry}")

    # Call RAGNAROK production API
    result = await _call_production_api(
        session_id=state.get("session_id", ""),
        brief=brief,
        script=script_draft,
        industry=industry,
        business_name=business_name,
        style=_map_tone_to_style(tone),
        target_platforms=["youtube", "tiktok", "instagram"]
    )

    # Handle API response
    if result.get("success"):
        production_id = result.get("production_id", result.get("id", ""))
        session_id = state.get("session_id", "")
        logger.info(f"[ProductionNode] Production started: {production_id}")

        new_state["production_id"] = production_id
        new_state["production_status"] = "queued"
        new_state["production_progress"] = 0.0
        new_state["production_phase"] = "queued"
        new_state["production_started_at"] = datetime.now(timezone.utc).isoformat()
        new_state["current_phase"] = "production"

        # Register with SSE tracking system
        if PRODUCTION_TRACKING_AVAILABLE and create_production_status:
            try:
                create_production_status(session_id, {
                    "business_name": business_name,
                    "primary_offering": primary_offering,
                    "target_demographic": target_demographic,
                    "call_to_action": call_to_action,
                    "tone": tone,
                    "production_id": production_id,
                })
                logger.info(f"[ProductionNode] SSE tracking registered for session: {session_id}")
            except Exception as e:
                logger.warning(f"[ProductionNode] SSE tracking failed: {e}")

        # ========================================
        # AUTO-PUBLISH TO VIDEO PREVIEW
        # ========================================
        # Check if video_url is immediately available (sync production)
        video_url = result.get("video_url") or result.get("url")
        preview_section = ""

        if video_url:
            preview_result = await send_to_video_preview(video_url, new_state)
            if preview_result["success"]:
                new_state["preview_url"] = preview_result["preview_url"]
                new_state["preview_id"] = preview_result["preview_id"]
                logger.info(f"[Production] Commercial published to preview: {preview_result['preview_url']}")
                preview_section = f"""

🔗 **Shareable Preview:** {preview_result['preview_url']}

Share this link with your team or clients to preview the commercial!"""

        response_text = f'''🎬 **Production Started!**

Your commercial for **{business_name}** is now in the RAGNAROK pipeline.

**Production ID:** `{production_id}`

**Pipeline stages:**
1. 📝 Script finalization
2. 🎙️ Voice synthesis (ElevenLabs)
3. 🎥 Video generation
4. 🎵 Music selection
5. ✂️ Final assembly
6. 📦 Multi-platform export

**Estimated time:** 3-5 minutes

You'll receive your commercial in **YouTube**, **TikTok**, and **Instagram** formats.
{preview_section}
I'll update you as each phase completes! 🚀'''

    else:
        error_msg = result.get("error", "Unknown error")
        logger.error(f"[ProductionNode] Production failed: {error_msg}")

        new_state["production_status"] = "failed"
        new_state["production_error"] = error_msg

        response_text = f'''❌ **Production Error**

We encountered an issue starting your video production:

> {error_msg}

**What you can do:**
1. Try again in a few moments
2. Say "retry production" to attempt again
3. Contact support if the issue persists

Your script is still saved and approved, so we won't lose any progress!'''

    new_state["messages"] = list(state.get("messages", [])) + [{
        "role": "assistant",
        "content": response_text
    }]

    return new_state


def route_after_script_writer(state: VideoBriefState) -> Literal["reviewer", "script_writer"]:
    """
    After script writer generates a script, go to reviewer for user feedback.
    Includes retry limit to prevent infinite loops on JSON parse errors.
    """
    script_status = state.get("script_status")

    if script_status == "pending_review":
        logger.info("[Router] Script ready → reviewer (awaiting feedback)")
        return "reviewer"

    if script_status == "parse_error_final":
        logger.info("[Router] Max parse attempts reached → reviewer (for fallback handling)")
        return "reviewer"

    if script_status == "retry_needed":
        attempts = state.get("script_parse_attempts", 0)
        logger.info(f"[Router] Parse error, retrying script_writer ({attempts}/3)")
        return "script_writer"

    # Unknown status - go to reviewer to avoid infinite loop
    logger.warning(f"[Router] Unknown script_status={script_status} → reviewer (safety fallback)")
    return "reviewer"


def route_after_reviewer(state: VideoBriefState) -> Literal["script_writer", "production", "end"]:
    """
    After reviewer processes feedback:
    - approved → production (or end for now)
    - revision_requested → back to script_writer
    - rejected/max_revisions → end
    """
    script_status = state.get("script_status")

    if script_status == "approved":
        logger.info("[Router] Script approved → production")
        return "production"  # Will go to END until production node exists

    if script_status == "revision_requested":
        logger.info("[Router] Revision requested → script_writer")
        return "script_writer"

    if script_status in ["rejected", "max_revisions"]:
        logger.info(f"[Router] Status {script_status} → end")
        return "end"

    # Unclear/pending - stay in reviewer (wait for next input)
    logger.info("[Router] Status unclear → end (await next input)")
    return "end"


# ============================================================================
# 7. BUILD GRAPH
# ============================================================================

def build_graph():
    """Construct the LangGraph state machine with approval loop."""
    workflow = StateGraph(VideoBriefState)

    # Add all nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("script_writer", script_writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("production", production_node)  # PRODUCTION NODE

    # Entry point
    workflow.add_edge(START, "intake")

    # After intake: wait for input OR go to research
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "intake": END,
            "researcher": "researcher"
        }
    )

    # After research: go to script writer
    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "researcher": "researcher",
            "script_writer": "script_writer"
        }
    )

    # After script writer: go to reviewer
    workflow.add_conditional_edges(
        "script_writer",
        route_after_script_writer,
        {
            "reviewer": END,        # END to wait for user input, then reviewer processes
            "script_writer": "script_writer"
        }
    )

    # After reviewer: loop back to script_writer OR go to production
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "script_writer": "script_writer",  # Revision loop
            "production": "production",         # Approved → Production
            "end": END
        }
    )

    # Production always goes to END (terminal node)
    workflow.add_edge("production", END)

    return workflow.compile()


# Create the compiled graph
creative_director_graph = build_graph()


# ============================================================================
# 8. SESSION MANAGER
# ============================================================================

class CreativeDirectorV2:
    """
    Main interface for the LangGraph Creative Director.
    Manages sessions and provides API-compatible responses.
    """

    def __init__(self):
        self.graph = creative_director_graph
        self.sessions: dict[str, VideoBriefState] = {}

    def _create_initial_state(self, session_id: str) -> VideoBriefState:
        return {
            "business_name": None,
            "primary_offering": None,
            "target_demographic": None,
            "call_to_action": None,
            "tone": None,
            "top_rivals": None,
            "research_findings": None,
            "script_draft": None,
            "script_status": None,
            "revision_feedback": None,
            "revision_count": 0,
            # Production tracking (NEW)
            "production_id": None,
            "production_status": None,
            "production_progress": None,
            "production_phase": None,
            "production_error": None,
            "video_urls": None,
            "production_cost": None,
            "production_started_at": None,
            # Asset Upload (NEW)
            "uploaded_assets": [],
            "assets_reviewed": False,
            # Video Preview Integration
            "preview_url": None,
            "preview_id": None,
            "messages": [],
            "missing_fields": [
                "business_name", "primary_offering",
                "target_demographic", "call_to_action", "tone"
            ],
            "is_complete": False,
            "current_phase": "intake",
            "session_id": session_id
        }

    async def process_message(self, session_id: str, message: str) -> dict:
        """Process a user message and return API-compatible response."""

        # Get or create session
        if session_id not in self.sessions:
            state = self._create_initial_state(session_id)
            # Welcome message
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "Welcome to Barrios A2I Creative Director! "
                    "I'll help you create a professional video commercial. "
                    "What's the name of your business?"
                )
            })
            self.sessions[session_id] = state
        else:
            state = self.sessions[session_id]

        # Add user message
        state["messages"].append({"role": "user", "content": message})

        # =========================================================================
        # LOGO UPLOAD DETECTION - Set flag, DO NOT early-return
        # =========================================================================
        import re
        # Pattern 1: Original format [User uploaded logo/image: filename.png]
        upload_pattern = r'\[User uploaded (?:logo/image|document): ([^\]]+)\]'
        upload_match = re.search(upload_pattern, message)

        # Pattern 2: Frontend emoji format 📎 Uploaded: filename.png
        if not upload_match:
            emoji_pattern = r'📎\s*(?:Uploaded|File attached):\s*([^\s\n]+)'
            upload_match = re.search(emoji_pattern, message)

        if upload_match:
            filename = upload_match.group(1)
            logger.info(f"[LOGO] Detected upload: {filename}")

            # Store the logo in state
            current_assets = state.get("uploaded_assets", [])
            current_assets.append({
                "type": "logo",
                "url": f"uploaded://{filename}",
                "name": filename,
                "source": "frontend_upload"
            })
            state["uploaded_assets"] = current_assets

            # FLAG that asset just arrived - intake_node will see this
            state["_asset_just_received"] = filename
            logger.info(f"[LOGO] Asset flagged, continuing to graph invocation")

            # DO NOT RETURN EARLY - let graph invoke normally below

        # Determine which node to invoke based on current state
        current_phase = state.get("current_phase", "intake")
        script_status = state.get("script_status")

        # If script is pending review, route to reviewer
        if script_status == "pending_review" or current_phase == "scripting":
            logger.info(f"[V2] Script pending review - invoking reviewer")
            try:
                state = await reviewer_node(state)

                # If revision requested, immediately generate new script
                if state.get("script_status") == "revision_requested":
                    state = await script_writer_node(state)

                # If approved, immediately trigger production
                if state.get("script_status") == "approved":
                    logger.info("[V2] Script approved - triggering production")
                    state = await production_node(state)

                result = state
                self.sessions[session_id] = result
            except Exception as e:
                logger.error(f"Reviewer error: {e}")
                result = state
                result["messages"].append({
                    "role": "assistant",
                    "content": "I encountered an issue processing your feedback. Could you try again?"
                })

        # If script is already approved and user confirms, trigger production
        elif script_status == "approved" and current_phase == "approved":
            logger.info("[V2] Script already approved - checking for production confirmation")
            user_msg_lower = message.lower().strip()
            if any(p in user_msg_lower for p in ["yes", "start", "produce", "go", "make"]):
                try:
                    state = await production_node(state)
                    result = state
                    self.sessions[session_id] = result
                except Exception as e:
                    logger.error(f"Production error: {e}")
                    result = state
                    result["messages"].append({
                        "role": "assistant",
                        "content": "I encountered an issue starting production. Please try again."
                    })
            else:
                # User doesn't want production, just acknowledge
                result = state
                result["messages"].append({
                    "role": "assistant",
                    "content": "Your script is approved and ready. Say 'start production' when you're ready to generate your video!"
                })
                self.sessions[session_id] = result

        else:
            # Normal flow through graph
            try:
                result = await self.graph.ainvoke(state)
                self.sessions[session_id] = result
            except Exception as e:
                logger.error(f"Graph error: {e}")
                logger.error(f"Graph error traceback: {traceback.format_exc()}")
                result = state
                result["messages"].append({
                    "role": "assistant",
                    "content": "I encountered an issue. Could you repeat that?"
                })
                self.sessions[session_id] = result  # Save session even on error

        # Calculate progress
        required = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]
        filled = sum(1 for f in required if result.get(f))
        progress = int((filled / len(required)) * 100)

        # Map to frontend field names
        field_map = {
            "business_name": "BUSINESS",
            "primary_offering": "PRODUCT",
            "target_demographic": "AUDIENCE",
            "call_to_action": "CTA",
            "tone": "TONE"
        }
        missing_display = [field_map.get(f, f) for f in result.get("missing_fields", [])]

        # Determine if production should be triggered
        # Trigger when script is approved and production was started (has production_id)
        should_trigger = (
            result.get("script_status") == "approved" and
            result.get("production_id") is not None
        )

        return {
            "response": result["messages"][-1]["content"] if result["messages"] else "",
            "progress_percentage": progress,
            "missing_fields": missing_display,
            "current_phase": result.get("current_phase", "intake"),
            "is_complete": result.get("is_complete", False),
            "script_status": result.get("script_status"),
            "revision_count": result.get("revision_count", 0),
            "version": "v2-langgraph",
            # Production trigger flag - frontend uses this to show voice selector
            "trigger_production": should_trigger,
            # Production tracking (for SSE streaming)
            "production_id": result.get("production_id"),
            "production_status": result.get("production_status"),
            "metadata": {
                "session_id": session_id,
                "production_id": result.get("production_id"),
                "preview_url": result.get("preview_url"),
                "preview_id": result.get("preview_id"),
                "extracted_data": {
                    "business_name": result.get("business_name"),
                    "product": result.get("primary_offering"),
                    "audience": result.get("target_demographic"),
                    "cta": result.get("call_to_action"),
                    "tone": result.get("tone")
                }
            }
        }


# ============================================================================
# 9. SINGLETON INSTANCE
# ============================================================================

# Create singleton for import
creative_director_v2 = CreativeDirectorV2()


# ============================================================================
# 10. TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        session = f"test-{datetime.now().strftime('%H%M%S')}"

        messages = [
            "My business is Glamour Studio",
            "We specialize in hair coloring",
            "Women aged 25-45 who want trendy looks",
            "They should book online",
            "Luxurious vibe"
        ]

        for msg in messages:
            print(f"\n>>> User: {msg}")
            result = await creative_director_v2.process_message(session, msg)
            print(f"<<< AI: {result['response']}")
            print(f"    Progress: {result['progress_percentage']}%")
            print(f"    Missing: {result['missing_fields']}")

    asyncio.run(test())
