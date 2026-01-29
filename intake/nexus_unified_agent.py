"""
NEXUS Unified Agent - Token-Gated AI Assistant for Barrios A2I Commercial Lab
Handles Q&A, brief intake, and token management in one conversational agent.

CRITICAL BEHAVIORS (preserved from V3):
- MAX 2-3 sentences per response
- ONE question at a time
- Texting a friend vibe, not email
- Never repeats unanswered questions
- Helps when user says "I'm not sure"
"""

import json
import os
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import anthropic

from intake.nexus_brain import get_brain, get_knowledge_context
from api.tokens import check_user_tokens, use_tokens

logger = logging.getLogger(__name__)

# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class Intent:
    QUESTION = "question"          # Asking about pricing, services, how it works
    START_BRIEF = "start_brief"    # Wants to create a commercial
    CONTINUE_BRIEF = "continue_brief"  # Continuing intake conversation
    CONFIRM_GENERATE = "confirm_generate"  # Ready to generate
    GREETING = "greeting"          # Hello, hi, etc.
    OTHER = "other"                # General conversation

# Keywords for intent detection
QUESTION_KEYWORDS = [
    "how much", "cost", "price", "pricing", "token", "tokens",
    "what is", "what's", "how does", "how do", "explain",
    "plan", "tier", "subscription", "cancel", "refund",
    "format", "delivery", "turnaround", "clone", "vortex"
]

START_KEYWORDS = [
    "create", "make", "start", "begin", "want a commercial",
    "ready to", "let's go", "let's do", "i want to", "i need a"
]

CONFIRM_KEYWORDS = [
    "yes", "confirm", "do it", "generate", "let's go", "ship it",
    "sounds good", "perfect", "approved", "ready"
]

GREETING_KEYWORDS = ["hello", "hi", "hey", "good morning", "good afternoon"]


def classify_intent(message: str, brief_completion: int, phase: str) -> str:
    """Classify user intent based on message content and context."""
    msg_lower = message.lower().strip()

    # Check for greetings
    if any(msg_lower.startswith(g) or msg_lower == g for g in GREETING_KEYWORDS):
        return Intent.GREETING

    # Check for questions about the service
    if any(kw in msg_lower for kw in QUESTION_KEYWORDS):
        return Intent.QUESTION

    # Check for confirmation (only if brief is complete)
    if brief_completion == 100 and phase == "review":
        if any(kw in msg_lower for kw in CONFIRM_KEYWORDS):
            return Intent.CONFIRM_GENERATE

    # Check for starting a brief
    if brief_completion == 0:
        if any(kw in msg_lower for kw in START_KEYWORDS):
            return Intent.START_BRIEF

    # If we're in intake and have partial completion, continue brief
    if phase == "intake" and brief_completion > 0:
        return Intent.CONTINUE_BRIEF

    # Default based on context
    if brief_completion == 0:
        return Intent.QUESTION  # Probably asking about the service
    else:
        return Intent.CONTINUE_BRIEF


# ============================================================================
# STATE MANAGEMENT (same as V3)
# ============================================================================

@dataclass
class BriefData:
    """Collected brief information."""
    business_name: Optional[str] = None
    product_service: Optional[str] = None
    target_audience: Optional[str] = None
    call_to_action: Optional[str] = None
    tone: Optional[str] = None
    logo_url: Optional[str] = None
    additional_notes: List[str] = field(default_factory=list)

    def completion_percentage(self) -> int:
        required = ['business_name', 'product_service', 'target_audience', 'call_to_action', 'tone']
        filled = sum(1 for f in required if getattr(self, f))
        return int((filled / len(required)) * 100)

    def missing_fields(self) -> List[str]:
        required = ['business_name', 'product_service', 'target_audience', 'call_to_action', 'tone']
        return [f for f in required if not getattr(self, f)]

    def to_dict(self) -> Dict:
        return {
            'business_name': self.business_name,
            'product_service': self.product_service,
            'target_audience': self.target_audience,
            'call_to_action': self.call_to_action,
            'tone': self.tone,
            'logo_url': self.logo_url,
            'additional_notes': self.additional_notes,
            'completion_percentage': self.completion_percentage(),
            'missing_fields': self.missing_fields()
        }


@dataclass
class ConversationState:
    """Full conversation state."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    brief: BriefData = field(default_factory=BriefData)
    phase: str = "intake"  # intake, review, production
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are the AI Creative Director for Barrios A2I's Commercial Lab.

## CRITICAL: KEEP IT SHORT
- MAX 2-3 sentences per response
- Ask ONE question at a time
- Sound like texting a friend, not writing an email
- Warm but BRIEF

BAD: "Absolutely! I'd love to help you figure this out together. Let's start with the basics - what kind of business are you working on? For example, are you running a local business, launching a product, part of a nonprofit, or working on a personal brand?"

GOOD: "No worries, I got you! What kind of business do you have?"

GOOD: "Love it! Who's your ideal customer?"

GOOD: "Nice! What should viewers do after watching - visit your site, call, or something else?"

## KNOWLEDGE
{knowledge_context}

## USER CONTEXT
Tokens: {token_balance}
Plan: {plan_type}

## BRIEF STATUS
{brief_status}

## CONVERSATION
{conversation_history}

## RULES
1. If they say "I'm not sure" - suggest ONE option, ask if that fits
2. Never repeat questions - pivot or help instead
3. Keep the vibe friendly but efficient
4. If they ask about pricing/services, answer from your knowledge
5. If tokens = 0 and they want to generate, mention they need tokens first

## YOUR TASK
Respond in 2-3 sentences MAX. One question only if needed."""


# ============================================================================
# UNIFIED AGENT
# ============================================================================

class NexusUnifiedAgent:
    """Token-gated AI assistant with knowledge injection and brief intake."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.sessions: Dict[str, ConversationState] = {}
        self.brain = get_brain()

    def get_or_create_session(self, session_id: str, user_id: str = None) -> ConversationState:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState(
                session_id=session_id,
                user_id=user_id
            )
        elif user_id:
            self.sessions[session_id].user_id = user_id
        return self.sessions[session_id]

    def format_brief_status(self, brief: BriefData) -> str:
        """Format brief status for prompt."""
        pct = brief.completion_percentage()
        if pct == 0:
            return "Brief: Not started"

        lines = [f"Brief: {pct}% complete"]
        if brief.business_name:
            lines.append(f"- Business: {brief.business_name}")
        if brief.product_service:
            lines.append(f"- Product: {brief.product_service}")
        if brief.target_audience:
            lines.append(f"- Audience: {brief.target_audience}")
        if brief.call_to_action:
            lines.append(f"- CTA: {brief.call_to_action}")
        if brief.tone:
            lines.append(f"- Tone: {brief.tone}")

        missing = brief.missing_fields()
        if missing:
            lines.append(f"Still need: {', '.join(missing)}")

        return "\n".join(lines)

    def format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt."""
        if not messages:
            return "This is the start of the conversation."

        formatted = []
        for msg in messages[-8:]:  # Last 8 messages
            role = "User" if msg["role"] == "user" else "You"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def extract_brief_info(self, state: ConversationState, user_message: str, ai_response: str) -> None:
        """Extract brief information using Claude Haiku."""
        extraction_prompt = f"""Analyze this exchange and extract brief info.

CURRENT BRIEF:
{json.dumps(state.brief.to_dict(), indent=2)}

USER: {user_message}
AI: {ai_response}

Extract NEW info for: business_name, product_service, target_audience, call_to_action, tone
Return JSON with only fields that have NEW values. Empty object if nothing new.
Example: {{"business_name": "Acme Corp"}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=150,
                messages=[{"role": "user", "content": extraction_prompt}]
            )

            text = response.content[0].text.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                extracted = json.loads(text[start:end])

                for field_name, value in extracted.items():
                    if hasattr(state.brief, field_name) and value:
                        setattr(state.brief, field_name, value)
                        logger.info(f"Extracted {field_name}: {value}")
        except Exception as e:
            logger.warning(f"Brief extraction failed: {e}")

    async def process_message(
        self,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user message and return AI response."""

        # Get session state
        state = self.get_or_create_session(session_id, user_id)
        state.updated_at = datetime.now()

        # Add user message to history
        state.messages.append({"role": "user", "content": user_message})

        # Get token info if user_id provided
        token_info = {"balance": 0, "plan_type": None}
        if user_id:
            token_info = await check_user_tokens(user_id)

        # Classify intent
        intent = classify_intent(
            user_message,
            state.brief.completion_percentage(),
            state.phase
        )
        logger.info(f"Intent: {intent}")

        # Handle token-gated generation
        if intent == Intent.CONFIRM_GENERATE:
            if token_info["balance"] < 8:
                ai_response = f"You need 8 tokens for a commercial but have {token_info['balance']}. Check out our pricing at barriosa2i.com/pricing to get tokens!"
                state.messages.append({"role": "assistant", "content": ai_response})
                return self._build_response(state, ai_response, token_info, intent)

            # Deduct tokens and trigger production
            success = await use_tokens(user_id, 8, "Commercial generation")
            if success:
                state.phase = "production"
                ai_response = "Awesome! 8 tokens deducted. Your commercial is being generated - I'll let you know when it's ready!"
                token_info["balance"] -= 8
            else:
                ai_response = "Something went wrong with token deduction. Let me check on that."

            state.messages.append({"role": "assistant", "content": ai_response})
            return self._build_response(state, ai_response, token_info, intent)

        # Build system prompt with knowledge
        system = SYSTEM_PROMPT.format(
            knowledge_context=self.brain.get_full_context(),
            token_balance=token_info["balance"],
            plan_type=token_info["plan_type"] or "None",
            brief_status=self.format_brief_status(state.brief),
            conversation_history=self.format_conversation_history(state.messages[:-1])
        )

        # Get AI response
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast + cheap
                max_tokens=150,  # Keep responses SHORT
                system=system,
                messages=[{"role": "user", "content": user_message}]
            )
            ai_response = response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            ai_response = "Hmm, hit a snag. Try that again?"

        # Add AI response to history
        state.messages.append({"role": "assistant", "content": ai_response})

        # Extract brief info if in intake
        if intent in [Intent.START_BRIEF, Intent.CONTINUE_BRIEF]:
            self.extract_brief_info(state, user_message, ai_response)

        # Check if brief is complete
        if state.brief.completion_percentage() == 100 and state.phase == "intake":
            state.phase = "review"

        return self._build_response(state, ai_response, token_info, intent)

    def _build_response(
        self,
        state: ConversationState,
        ai_response: str,
        token_info: Dict,
        intent: str
    ) -> Dict[str, Any]:
        """Build the response dict."""
        return {
            "response": ai_response,
            "brief": state.brief.to_dict(),
            "phase": state.phase,
            "session_id": state.session_id,
            "user_id": state.user_id,
            "tokens": token_info.get("balance", 0),
            "plan_type": token_info.get("plan_type"),
            "intent": intent
        }

    def get_greeting(self) -> str:
        """Get the initial greeting."""
        return "Hey! I'm your AI Creative Director. Want to make a commercial, or got questions about how it works?"


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_unified_endpoint(app, agent: NexusUnifiedAgent):
    """Create FastAPI unified chat endpoint."""
    from fastapi import HTTPException
    from pydantic import BaseModel

    class UnifiedChatRequest(BaseModel):
        session_id: str
        message: str
        user_id: Optional[str] = None

    class UnifiedChatResponse(BaseModel):
        response: str
        brief: Dict[str, Any]
        phase: str
        session_id: str
        user_id: Optional[str]
        tokens: int
        plan_type: Optional[str]
        intent: str

    @app.post("/api/chat/unified", response_model=UnifiedChatResponse)
    async def chat_unified(request: UnifiedChatRequest):
        """Unified chat endpoint with token management."""
        try:
            result = await agent.process_message(
                session_id=request.session_id,
                user_message=request.message,
                user_id=request.user_id
            )
            return UnifiedChatResponse(**result)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/chat/unified/greeting")
    async def get_unified_greeting():
        """Get initial greeting."""
        return {"greeting": agent.get_greeting()}

    return app


# ============================================================================
# TESTING
# ============================================================================

async def test_agent():
    """Test the unified agent."""
    agent = NexusUnifiedAgent()
    session_id = "test-session"

    print("=" * 50)
    print("NEXUS UNIFIED AGENT TEST")
    print("=" * 50)
    print(f"\nGreeting: {agent.get_greeting()}\n")

    test_messages = [
        "hi",
        "how much does it cost?",
        "I want to make a commercial",
        "Barrios A2I",
        "We sell AI-powered commercials",
        "Small business owners who need video content",
        "Visit our website",
        "Professional but friendly"
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        result = await agent.process_message(session_id, msg)
        print(f"AI: {result['response']}")
        print(f"   [Intent: {result['intent']}, Brief: {result['brief']['completion_percentage']}%]")
        print()


if __name__ == "__main__":
    asyncio.run(test_agent())
