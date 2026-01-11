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
import httpx
from typing import TypedDict, List, Optional, Literal
from datetime import datetime

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creative_director_v2")

# Trinity API Configuration
TRINITY_URL = "https://barrios-genesis-flawless.onrender.com/api/genesis/research"
TRINITY_TIMEOUT = 60.0  # Trinity research can take time


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
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )


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
- Still need: {missing_fields}

## RULES
1. Acknowledge what the user told you warmly and concisely
2. Ask for the NEXT missing field specifically
3. Keep responses short (2-3 sentences)
4. NEVER say "Ready to create" until ALL 5 fields are filled
5. If user says "ready" but fields missing, explain what you still need

## RESPONSE
Acknowledge their input → Ask for next missing field"""


async def intake_node(state: VideoBriefState) -> dict:
    """
    Schema-First Extraction: Replaces 2800 lines of regex with structured LLM output.
    """
    logger.info(f"[IntakeAgent] Processing for session {state.get('session_id', 'unknown')}")

    llm = get_llm()

    # Get latest user message
    user_messages = [m for m in state["messages"] if m.get("role") == "user"]
    if not user_messages:
        return state
    latest_msg = user_messages[-1]["content"]

    # =========================================================================
    # STEP 1: Structured Extraction
    # =========================================================================
    try:
        extractor = llm.with_structured_output(ExtractionSchema)

        extraction_prompt = f"""Extract video brief information from this message.

User said: "{latest_msg}"

What we already know:
- Business: {state.get('business_name') or 'unknown'}
- Product: {state.get('primary_offering') or 'unknown'}
- Audience: {state.get('target_demographic') or 'unknown'}
- CTA: {state.get('call_to_action') or 'unknown'}
- Tone: {state.get('tone') or 'unknown'}

Extract ONLY what user explicitly stated. Do not guess."""

        extracted = await extractor.ainvoke(extraction_prompt)
        extracted_dict = extracted.dict(exclude_none=True)
        logger.info(f"[IntakeAgent] Extracted: {extracted_dict}")

    except Exception as e:
        logger.error(f"[IntakeAgent] Extraction error: {e}")
        extracted_dict = {}

    # =========================================================================
    # STEP 2: Update State
    # =========================================================================
    new_state = dict(state)

    for field, value in extracted_dict.items():
        if value and not state.get(field):
            new_state[field] = value
            logger.info(f"[IntakeAgent] Set {field} = {value}")

    # =========================================================================
    # STEP 3: Calculate Missing (Deterministic)
    # =========================================================================
    required = ["business_name", "primary_offering", "target_demographic", "call_to_action", "tone"]
    missing = [f for f in required if not new_state.get(f)]

    new_state["missing_fields"] = missing
    new_state["is_complete"] = len(missing) == 0

    logger.info(f"[IntakeAgent] Missing: {missing}, Complete: {new_state['is_complete']}")

    # =========================================================================
    # STEP 4: Generate Response
    # =========================================================================
    if new_state["is_complete"]:
        response_text = (
            f"Excellent! I have everything I need for {new_state['business_name']}. "
            f"Let me connect with our research team to analyze your market."
        )
        new_state["current_phase"] = "research"
    else:
        # Build context-aware prompt
        prompt = INTAKE_PROMPT.format(
            business_name=new_state.get("business_name") or "[not yet provided]",
            primary_offering=new_state.get("primary_offering") or "[not yet provided]",
            target_demographic=new_state.get("target_demographic") or "[not yet provided]",
            call_to_action=new_state.get("call_to_action") or "[not yet provided]",
            tone=new_state.get("tone") or "[not yet provided]",
            missing_fields=", ".join(missing)
        )

        messages = [SystemMessage(content=prompt)]
        for m in state["messages"]:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        response = await llm.ainvoke(messages)
        response_text = response.content

    # Add to history
    new_state["messages"] = list(state["messages"]) + [
        {"role": "assistant", "content": response_text}
    ]

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


# ============================================================================
# 6. DETERMINISTIC ROUTING
# ============================================================================

def route_after_intake(state: VideoBriefState) -> Literal["intake", "researcher"]:
    """
    CODE decides routing, NOT the AI.
    This is the key fix for the 'Ready to create?' bug.
    """
    if state.get("is_complete", False):
        logger.info("[Router] All fields complete → researcher")
        return "researcher"

    logger.info("[Router] Missing fields → stay in intake")
    return "intake"


# ============================================================================
# 7. BUILD GRAPH
# ============================================================================

def build_graph():
    """Construct the LangGraph state machine."""
    workflow = StateGraph(VideoBriefState)

    # Add nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("researcher", researcher_node)

    # Entry point
    workflow.add_edge(START, "intake")

    # Conditional routing
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "intake": END,        # Wait for next user input
            "researcher": "researcher"
        }
    )

    workflow.add_edge("researcher", END)

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

        # Run graph
        try:
            result = await self.graph.ainvoke(state)
            self.sessions[session_id] = result
        except Exception as e:
            logger.error(f"Graph error: {e}")
            result = state
            result["messages"].append({
                "role": "assistant",
                "content": "I encountered an issue. Could you repeat that?"
            })

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

        return {
            "response": result["messages"][-1]["content"] if result["messages"] else "",
            "progress_percentage": progress,
            "missing_fields": missing_display,
            "current_phase": result.get("current_phase", "intake"),
            "is_complete": result.get("is_complete", False),
            "version": "v2-langgraph",
            "metadata": {
                "session_id": session_id,
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
