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

async def researcher_node(state: VideoBriefState) -> dict:
    """
    Calls Trinity for market intelligence.
    Currently a placeholder - will integrate with real Trinity API.
    """
    logger.info(f"[ResearcherAgent] Starting research for {state['business_name']}")

    # TODO: Replace with actual Trinity API call
    # TRINITY_URL = "https://barrios-genesis-flawless.onrender.com/api/trinity/analyze"

    new_state = dict(state)
    new_state["current_phase"] = "scripting"
    new_state["messages"] = list(state["messages"]) + [{
        "role": "assistant",
        "content": (
            "🔍 Research complete! I've analyzed your market and competitors. "
            "Now let me draft a script that positions you to win."
        )
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
