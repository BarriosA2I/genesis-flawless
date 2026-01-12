"""
============================================================================
CREATIVE DIRECTOR V3 - STATE MACHINE IMPLEMENTATION
============================================================================

File: intake/creative_director_fsm.py

Architecture: State Machine + LLM Hybrid
- CODE controls conversation flow (deterministic)
- CLAUDE only generates natural language responses

State Flow:
START -> BUSINESS_NAME -> PRODUCT -> AUDIENCE -> CTA -> TONE -> LOGO -> CONFIRM -> PRODUCTION -> COMPLETE

Author: Barrios A2I | Version: 3.0.0 | January 2026
============================================================================
"""

import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
import anthropic

logger = logging.getLogger("creative_director_v3")

# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class ConvState(Enum):
    """Conversation states for the Creative Director FSM."""
    START = "start"
    BUSINESS_NAME = "business_name"
    PRODUCT = "product"
    AUDIENCE = "audience"
    CTA = "cta"
    TONE = "tone"
    LOGO = "logo"
    CONFIRM = "confirm"
    PRODUCTION = "production"
    COMPLETE = "complete"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Brief:
    """Collected video brief data."""
    business_name: Optional[str] = None
    product: Optional[str] = None
    audience: Optional[str] = None
    cta: Optional[str] = None
    tone: Optional[str] = None
    logo_url: Optional[str] = None
    logo_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert brief to dictionary."""
        return {
            "business_name": self.business_name,
            "product": self.product,
            "audience": self.audience,
            "cta": self.cta,
            "tone": self.tone,
            "logo_url": self.logo_url,
            "logo_instructions": self.logo_instructions
        }

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        return all([
            self.business_name,
            self.product,
            self.audience,
            self.cta,
            self.tone
        ])


@dataclass
class Session:
    """Conversation session state."""
    session_id: str
    state: ConvState = ConvState.START
    brief: Brief = field(default_factory=Brief)
    history: List[Dict[str, str]] = field(default_factory=list)
    error_count: int = 0
    confirmation_attempts: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    production_triggered: bool = False


# ============================================================================
# ROUTER (Pure Python - NO LLM)
# ============================================================================

class Router:
    """Deterministic state transitions. NO LLM calls."""

    STATE_ORDER = [
        ConvState.START,
        ConvState.BUSINESS_NAME,
        ConvState.PRODUCT,
        ConvState.AUDIENCE,
        ConvState.CTA,
        ConvState.TONE,
        ConvState.LOGO,
        ConvState.CONFIRM,
        ConvState.PRODUCTION,
        ConvState.COMPLETE
    ]

    @classmethod
    def get_next_state(cls, current: ConvState, brief: Brief, user_confirmed: bool = False) -> ConvState:
        """
        Pure Python state transition logic.

        This is the CORE of the FSM - deterministic, testable, no LLM.
        """

        if current == ConvState.START:
            return ConvState.BUSINESS_NAME

        if current == ConvState.BUSINESS_NAME and brief.business_name:
            return ConvState.PRODUCT

        if current == ConvState.PRODUCT and brief.product:
            return ConvState.AUDIENCE

        if current == ConvState.AUDIENCE and brief.audience:
            return ConvState.CTA

        if current == ConvState.CTA and brief.cta:
            return ConvState.TONE

        if current == ConvState.TONE and brief.tone:
            return ConvState.LOGO

        if current == ConvState.LOGO:
            # Logo is optional - always proceed to confirm
            return ConvState.CONFIRM

        if current == ConvState.CONFIRM:
            if user_confirmed:
                return ConvState.PRODUCTION
            # Stay in confirm state if not confirmed
            return ConvState.CONFIRM

        if current == ConvState.PRODUCTION:
            return ConvState.COMPLETE

        return current  # Stay in current state if no transition


# ============================================================================
# SLOT EXTRACTOR (Uses LLM)
# ============================================================================

class SlotExtractor:
    """Extract ONE slot value from user message using LLM."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def extract(self, user_message: str, slot_name: str) -> Optional[str]:
        """
        Extract a specific slot value from user message.

        Uses LLM to intelligently extract the value, handling natural language
        variations like "My company is called Acme Corp" -> "Acme Corp"
        """

        prompt = f"""Extract the {slot_name} from this user message.
Return ONLY the extracted value, nothing else.
If the value is not present or unclear, return "NONE".

User message: {user_message}

Extracted {slot_name}:"""

        try:
            response = self.client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            value = response.content[0].text.strip()
            if value.upper() == "NONE" or not value:
                return None
            return value

        except Exception as e:
            logger.error(f"Slot extraction failed: {e}")
            return None


# ============================================================================
# RESPONSE GENERATOR (Uses LLM)
# ============================================================================

class ResponseGenerator:
    """Generate conversational responses using LLM."""

    SYSTEM_PROMPT = """You are a creative director gathering a video brief. Be conversational, not formal.

RULES:
- 2 sentences max per response
- Ask ONE question at a time
- Be enthusiastic but professional
- Never mention you're an AI or state machine
- If user seems confused, gently guide them back"""

    QUESTION_TEMPLATES = {
        ConvState.BUSINESS_NAME: "What's the name of your business?",
        ConvState.PRODUCT: "What product or service does {business_name} offer?",
        ConvState.AUDIENCE: "Who's your target audience for this video?",
        ConvState.CTA: "What action do you want viewers to take? Any specific tagline?",
        ConvState.TONE: "What tone or visual style are you going for?",
        ConvState.LOGO: "Do you have a logo or brand images to include?",
        ConvState.CONFIRM: "Here's your brief:\n\n{brief_summary}\n\nReady to create your commercial?",
    }

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def generate(self, state: ConvState, brief: Brief, history: List[Dict]) -> str:
        """Generate response for current state."""

        # Handle special states with fixed responses
        if state == ConvState.START:
            return "Welcome to the Commercial Lab! I'm your AI Creative Director. Let's create something amazing. What's your business name?"

        if state == ConvState.PRODUCTION:
            return "Excellent! Starting production now. Your commercial is being generated..."

        if state == ConvState.COMPLETE:
            return "Your commercial is ready! Check the gallery to view your video."

        # Get template and fill in values
        template = self.QUESTION_TEMPLATES.get(state, "")

        if state == ConvState.CONFIRM:
            brief_summary = self._format_brief(brief)
            template = template.format(brief_summary=brief_summary)
        elif "{business_name}" in template and brief.business_name:
            template = template.format(business_name=brief.business_name)

        # Generate natural response using LLM
        try:
            prompt = f"""Given this question template: "{template}"

Generate a natural, conversational version (2 sentences max).
Keep the same intent but make it sound human.

Conversational version:"""

            response = self.client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=150,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return template  # Fallback to template

    def _format_brief(self, brief: Brief) -> str:
        """Format brief for confirmation display."""
        lines = []
        if brief.business_name:
            lines.append(f"- Business: {brief.business_name}")
        if brief.product:
            lines.append(f"- Product/Service: {brief.product}")
        if brief.audience:
            lines.append(f"- Target Audience: {brief.audience}")
        if brief.cta:
            lines.append(f"- CTA: {brief.cta}")
        if brief.tone:
            lines.append(f"- Tone/Style: {brief.tone}")
        if brief.logo_url:
            lines.append(f"- Logo: Uploaded")
        return "\n".join(lines)


# ============================================================================
# CREATIVE DIRECTOR FSM (Main Orchestrator)
# ============================================================================

class CreativeDirectorFSM:
    """
    Main FSM orchestrator.

    KEY PRINCIPLE: CODE controls flow, CLAUDE generates text.

    This class:
    - Manages session state
    - Orchestrates state transitions (via Router)
    - Extracts slot values (via SlotExtractor)
    - Generates responses (via ResponseGenerator)
    """

    # Map states to their slot names for extraction
    STATE_TO_SLOT = {
        ConvState.BUSINESS_NAME: "business_name",
        ConvState.PRODUCT: "product",
        ConvState.AUDIENCE: "audience",
        ConvState.CTA: "cta",
        ConvState.TONE: "tone",
    }

    def __init__(self):
        """Initialize the FSM with Anthropic client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set - LLM calls will fail")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.extractor = SlotExtractor(self.client)
        self.generator = ResponseGenerator(self.client)
        self.sessions: Dict[str, Session] = {}

    def start_session(self, session_id: str) -> Dict[str, Any]:
        """
        Start a new session and return welcome message.

        Returns dict with:
        - session_id
        - response (welcome message)
        - state (will be BUSINESS_NAME after welcome)
        - progress (percentage)
        - production_triggered (False)
        """
        session = Session(session_id=session_id)
        self.sessions[session_id] = session

        # Generate welcome and transition to BUSINESS_NAME
        welcome = self.generator.generate(ConvState.START, session.brief, session.history)
        session.state = ConvState.BUSINESS_NAME

        session.history.append({"role": "assistant", "content": welcome})

        logger.info(f"[V3-FSM] Session {session_id} started, state={session.state.value}")

        return {
            "session_id": session_id,
            "response": welcome,
            "state": session.state.value,
            "progress": self._calculate_progress(session),
            "production_triggered": False
        }

    def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process user message and return response.

        This is the main message processing loop:
        1. Get/create session
        2. Add message to history
        3. Extract slot value (if applicable)
        4. Handle special cases (LOGO, CONFIRM)
        5. Transition to next state
        6. Generate response
        """

        session = self.sessions.get(session_id)
        if not session:
            # Auto-create session if not exists
            self.start_session(session_id)
            session = self.sessions[session_id]

        logger.info(f"[V3-FSM] Processing message for {session_id}, current_state={session.state.value}")

        # Add user message to history
        session.history.append({"role": "user", "content": user_message})

        # Extract slot value if applicable
        if session.state in self.STATE_TO_SLOT:
            slot_name = self.STATE_TO_SLOT[session.state]
            extracted_value = self.extractor.extract(user_message, slot_name)

            if extracted_value:
                setattr(session.brief, slot_name, extracted_value)
                logger.info(f"[V3-FSM] Extracted {slot_name}: {extracted_value}")

        # Handle LOGO state specially
        if session.state == ConvState.LOGO:
            # Check if user wants to skip logo
            skip_keywords = ["no", "skip", "none", "don't have", "no logo", "nope"]
            if any(kw in user_message.lower() for kw in skip_keywords):
                logger.info("[V3-FSM] User skipping logo")
            # Logo URL is set via upload endpoint, not message

        # Handle CONFIRM state
        if session.state == ConvState.CONFIRM:
            session.confirmation_attempts += 1
            confirm_keywords = ["yes", "ready", "go", "start", "confirm", "approved", "looks good", "let's do it"]
            user_confirmed = any(kw in user_message.lower() for kw in confirm_keywords)

            if user_confirmed or session.confirmation_attempts >= 2:
                # Transition to production
                logger.info(f"[V3-FSM] User confirmed (attempts={session.confirmation_attempts}), triggering production")
                next_state = Router.get_next_state(session.state, session.brief, user_confirmed=True)
                session.state = next_state
                session.production_triggered = True

                response = self.generator.generate(session.state, session.brief, session.history)
                session.history.append({"role": "assistant", "content": response})

                return {
                    "session_id": session_id,
                    "response": response,
                    "state": session.state.value,
                    "progress": 100,
                    "production_triggered": True,
                    "brief": session.brief.to_dict()
                }

        # Transition to next state
        next_state = Router.get_next_state(session.state, session.brief)
        logger.info(f"[V3-FSM] State transition: {session.state.value} -> {next_state.value}")
        session.state = next_state

        # Generate response for new state
        response = self.generator.generate(session.state, session.brief, session.history)
        session.history.append({"role": "assistant", "content": response})

        return {
            "session_id": session_id,
            "response": response,
            "state": session.state.value,
            "progress": self._calculate_progress(session),
            "production_triggered": session.production_triggered,
            "brief": session.brief.to_dict() if session.brief.is_complete() else None
        }

    def set_logo(self, session_id: str, logo_url: str, instructions: str = None) -> bool:
        """
        Set logo URL for a session.

        Called from the upload-logo endpoint after file is uploaded to storage.
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.brief.logo_url = logo_url
        session.brief.logo_instructions = instructions
        logger.info(f"[V3-FSM] Logo set for {session_id}: {logo_url}")
        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state as dictionary."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "state": session.state.value,
            "brief": session.brief.to_dict(),
            "progress": self._calculate_progress(session),
            "production_triggered": session.production_triggered,
            "created_at": session.created_at.isoformat()
        }

    def _calculate_progress(self, session: Session) -> int:
        """Calculate progress percentage based on current state."""
        state_progress = {
            ConvState.START: 0,
            ConvState.BUSINESS_NAME: 10,
            ConvState.PRODUCT: 25,
            ConvState.AUDIENCE: 40,
            ConvState.CTA: 55,
            ConvState.TONE: 70,
            ConvState.LOGO: 85,
            ConvState.CONFIRM: 95,
            ConvState.PRODUCTION: 100,
            ConvState.COMPLETE: 100
        }
        return state_progress.get(session.state, 0)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_fsm: Optional[CreativeDirectorFSM] = None


def get_fsm() -> CreativeDirectorFSM:
    """Get or create FSM singleton instance."""
    global _fsm
    if _fsm is None:
        _fsm = CreativeDirectorFSM()
        logger.info("[V3-FSM] CreativeDirectorFSM initialized")
    return _fsm
