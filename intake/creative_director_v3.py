#!/usr/bin/env python3
"""
Creative Director V3 - Natural Conversation AI
Fixes the critical bug where AI repeats questions instead of helping

KEY CHANGES from V2:
1. AI can ANSWER questions, not just ask them
2. AI can HELP when user says "I'm not sure" 
3. AI doesn't repeat questions - it adapts to user responses
4. AI maintains personality throughout, not robotic form filling

Author: Barrios A2I
"""

import json
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

@dataclass
class BriefData:
    """Collected brief information"""
    business_name: Optional[str] = None
    product_service: Optional[str] = None
    target_audience: Optional[str] = None
    call_to_action: Optional[str] = None
    tone: Optional[str] = None
    logo_url: Optional[str] = None
    additional_notes: List[str] = field(default_factory=list)
    
    def completion_percentage(self) -> int:
        """Calculate how complete the brief is"""
        required = ['business_name', 'product_service', 'target_audience', 'call_to_action', 'tone']
        filled = sum(1 for f in required if getattr(self, f))
        return int((filled / len(required)) * 100)
    
    def missing_fields(self) -> List[str]:
        """Get list of missing required fields"""
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
    """Full conversation state"""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    brief: BriefData = field(default_factory=BriefData)
    phase: str = "intake"  # intake, review, production
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# SYSTEM PROMPT - THE KEY FIX
# ============================================================================

SYSTEM_PROMPT = """You are the AI Creative Director for Barrios A2I's Commercial Lab. You help clients create professional video commercials.

## YOUR PERSONALITY
- Warm, professional, and genuinely helpful
- You're a creative partner, NOT a form or questionnaire
- You can answer ANY question a client asks
- You're knowledgeable about marketing, advertising, video production, and business

## CRITICAL BEHAVIOR RULES

### RULE 1: NEVER JUST REPEAT QUESTIONS
If a client says "I'm not sure" or asks for help, you HELP THEM. Examples:

❌ WRONG (what you were doing):
Client: "I'm not sure, can you figure it out?"
AI: "Who's the ideal audience for this commercial?" [REPEATING]

✅ CORRECT (what you should do):
Client: "I'm not sure, can you figure it out?"  
AI: "Let me help! Based on what you've told me about selling AI commercials, your ideal audience is probably:
- Small business owners looking to modernize their marketing
- Marketing agencies that want to offer video services
- Startups needing professional content on a budget

Does any of these sound right? Or tell me more about who typically buys from you."

### RULE 2: BE CONVERSATIONAL, NOT ROBOTIC
You're having a conversation, not filling out a form. Adapt to how the client talks.

### RULE 3: ANSWER CLIENT QUESTIONS
If they ask "what should my CTA be?" - GIVE THEM OPTIONS. Don't just ask the question back.

### RULE 4: USE CONTEXT
You know about their business from what they've said. Use that knowledge to make suggestions.

## INFORMATION YOU'RE COLLECTING (but naturally, through conversation)

1. **Business Name** - Who are they?
2. **Product/Service** - What are they promoting?
3. **Target Audience** - Who should see this ad?
4. **Call to Action** - What should viewers do?
5. **Tone** - What feeling should the commercial have?

## CURRENT BRIEF STATUS
{brief_status}

## CONVERSATION HISTORY
{conversation_history}

## YOUR TASK
Respond to the client's latest message. Be helpful, conversational, and NEVER just repeat a question they didn't answer. If they need help, HELP THEM with suggestions and examples.

Remember: You're a creative professional having a real conversation, not a chatbot filling out a form."""


# ============================================================================
# CREATIVE DIRECTOR AGENT
# ============================================================================

class CreativeDirectorV3:
    """Natural conversation Creative Director that actually helps clients"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.sessions: Dict[str, ConversationState] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationState:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState(session_id=session_id)
        return self.sessions[session_id]
    
    def format_brief_status(self, brief: BriefData) -> str:
        """Format brief status for prompt"""
        status_lines = [f"Completion: {brief.completion_percentage()}%"]
        
        if brief.business_name:
            status_lines.append(f"✅ Business: {brief.business_name}")
        else:
            status_lines.append("❓ Business: Not yet collected")
            
        if brief.product_service:
            status_lines.append(f"✅ Product/Service: {brief.product_service}")
        else:
            status_lines.append("❓ Product/Service: Not yet collected")
            
        if brief.target_audience:
            status_lines.append(f"✅ Target Audience: {brief.target_audience}")
        else:
            status_lines.append("❓ Target Audience: Not yet collected")
            
        if brief.call_to_action:
            status_lines.append(f"✅ CTA: {brief.call_to_action}")
        else:
            status_lines.append("❓ CTA: Not yet collected")
            
        if brief.tone:
            status_lines.append(f"✅ Tone: {brief.tone}")
        else:
            status_lines.append("❓ Tone: Not yet collected")
            
        return "\n".join(status_lines)
    
    def format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not messages:
            return "This is the start of the conversation."
        
        formatted = []
        for msg in messages[-10:]:  # Last 10 messages for context
            role = "Client" if msg["role"] == "user" else "You"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def extract_brief_info(self, state: ConversationState, user_message: str, ai_response: str) -> None:
        """Extract brief information from conversation using Claude"""
        # Build extraction prompt
        extraction_prompt = f"""Analyze this conversation exchange and extract any brief information mentioned.

CURRENT BRIEF STATE:
{json.dumps(state.brief.to_dict(), indent=2)}

USER SAID: {user_message}
AI RESPONDED: {ai_response}

Extract any NEW information for these fields (only if explicitly mentioned or clearly implied):
- business_name: Company/business name
- product_service: What they're selling/promoting  
- target_audience: Who the ad is for
- call_to_action: What viewers should do
- tone: Feeling/style of commercial (professional, fun, urgent, etc.)

Return JSON with ONLY the fields that have NEW information. If nothing new, return empty object {{}}.
Example: {{"business_name": "Acme Corp", "tone": "professional and trustworthy"}}

IMPORTANT: Only extract CONFIRMED information. Don't guess or infer beyond what was clearly stated."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast model for extraction
                max_tokens=200,
                messages=[{"role": "user", "content": extraction_prompt}]
            )
            
            # Parse response
            response_text = response.content[0].text.strip()
            
            # Find JSON in response
            if "{" in response_text:
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                json_str = response_text[start:end]
                extracted = json.loads(json_str)
                
                # Update brief with extracted info
                for field, value in extracted.items():
                    if hasattr(state.brief, field) and value:
                        setattr(state.brief, field, value)
                        logger.info(f"Extracted {field}: {value}")
                        
        except Exception as e:
            logger.warning(f"Brief extraction failed: {e}")
    
    async def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process a user message and return AI response"""
        
        # Get session state
        state = self.get_or_create_session(session_id)
        state.updated_at = datetime.now()
        
        # Add user message to history
        state.messages.append({"role": "user", "content": user_message})
        
        # Check for logo upload
        if "[User uploaded" in user_message or "logo" in user_message.lower():
            # Handle logo upload
            if "http" in user_message or ".png" in user_message or ".jpg" in user_message:
                # Extract URL or filename
                state.brief.logo_url = user_message
                logger.info(f"Logo detected: {user_message[:50]}")
        
        # Build the full prompt
        system = SYSTEM_PROMPT.format(
            brief_status=self.format_brief_status(state.brief),
            conversation_history=self.format_conversation_history(state.messages[:-1])  # Exclude current message
        )
        
        # Get AI response
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Good balance of quality and speed
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": user_message}]
            )
            
            ai_response = response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            ai_response = "I apologize, I'm having a technical issue. Could you try again?"
        
        # Add AI response to history
        state.messages.append({"role": "assistant", "content": ai_response})
        
        # Extract any brief info from the exchange
        self.extract_brief_info(state, user_message, ai_response)
        
        # Check if brief is complete
        if state.brief.completion_percentage() == 100 and state.phase == "intake":
            state.phase = "review"
        
        # Build response
        return {
            "response": ai_response,
            "brief": state.brief.to_dict(),
            "phase": state.phase,
            "session_id": session_id
        }
    
    def get_greeting(self) -> str:
        """Get the initial greeting"""
        return """Welcome to the A2I Commercial Lab! I'm your AI Creative Director. 

Whether you need a TikTok ad, Instagram Reel, or YouTube commercial - let's create something amazing together.

What's the name of your business?"""


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_chat_endpoint(app, director: CreativeDirectorV3):
    """Create FastAPI chat endpoint"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    class ChatRequest(BaseModel):
        session_id: str
        message: str
    
    class ChatResponse(BaseModel):
        response: str
        brief: Dict[str, Any]
        phase: str
        session_id: str
    
    @app.post("/api/chat/v3", response_model=ChatResponse)
    async def chat_v3(request: ChatRequest):
        """V3 chat endpoint with natural conversation"""
        try:
            result = await director.process_message(
                session_id=request.session_id,
                user_message=request.message
            )
            return ChatResponse(**result)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/chat/v3/greeting")
    async def get_greeting():
        """Get initial greeting"""
        return {"greeting": director.get_greeting()}
    
    return app


# ============================================================================
# MAIN / TESTING
# ============================================================================

async def test_conversation():
    """Test the natural conversation flow"""
    director = CreativeDirectorV3()
    session_id = "test-session-001"
    
    print("=" * 60)
    print("CREATIVE DIRECTOR V3 - NATURAL CONVERSATION TEST")
    print("=" * 60)
    
    # Test conversation that previously broke
    test_messages = [
        "Hello",
        "Barrios A2I",
        "I sell commercials made by AI",
        "Im not sure can you figure it out?",  # THIS IS THE BUG - V2 would repeat
        "yeah the first one sounds good",
        "Visit our website",
        "Professional but exciting"
    ]
    
    print(f"\n🤖 AI: {director.get_greeting()}\n")
    
    for msg in test_messages:
        print(f"👤 User: {msg}")
        result = await director.process_message(session_id, msg)
        print(f"🤖 AI: {result['response']}")
        print(f"   [Brief: {result['brief']['completion_percentage']}% complete]")
        print()
    
    print("=" * 60)
    print("FINAL BRIEF:")
    print(json.dumps(result['brief'], indent=2))


if __name__ == "__main__":
    asyncio.run(test_conversation())
