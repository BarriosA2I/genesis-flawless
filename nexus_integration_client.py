#!/usr/bin/env python3
"""
NEXUS BRAIN Integration Client v1.0
Add this to your existing GENESIS backend to integrate with the hub.

Location: C:/Users/gary/python-genesis-flawless/nexus_integration_client.py

This client provides:
1. Connection to the Integration Hub
2. Content requests from SCRIPTWRITER-X
3. Lead capture and tracking
4. Hook Arsenal access for enriched responses
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from functools import lru_cache
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IntegrationConfig:
    """Configuration for NEXUS Integration Hub connection"""
    hub_url: str = "http://localhost:8010"  # Local dev (Integration Hub)
    # Production: "https://barrios-genesis-flawless.onrender.com/integration"
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    cache_ttl_seconds: int = 300  # 5 min cache for hook arsenal

# Singleton config
_config = IntegrationConfig()

def configure_integration(
    hub_url: str = None,
    timeout: float = None
):
    """Configure integration client"""
    global _config
    if hub_url:
        _config.hub_url = hub_url
    if timeout:
        _config.timeout_seconds = timeout

# ============================================================================
# ENUMS (Match the hub)
# ============================================================================

class Platform(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    WEBSITE = "website"
    DIRECT = "direct"

class LeadSource(str, Enum):
    COMMENT = "comment"
    DM = "dm"
    WEBSITE_CHAT = "website_chat"
    FORM_SUBMISSION = "form"
    REFERRAL = "referral"

class ConversionStatus(str, Enum):
    NEW = "new"
    ENGAGED = "engaged"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    CONVERTED = "converted"
    LOST = "lost"
    GHOSTED = "ghosted"

# ============================================================================
# INTEGRATION CLIENT
# ============================================================================

class NexusIntegrationClient:
    """
    Client for NEXUS BRAIN to communicate with the Integration Hub.
    Use this in your chat handlers to:
    - Get personalized content from SCRIPTWRITER-X
    - Capture and track leads
    - Access the Hook Arsenal
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or _config
        self._session: Optional[aiohttp.ClientSession] = None
        self._hook_arsenal_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make request to Integration Hub with retry logic"""
        session = await self._get_session()
        url = f"{self.config.hub_url}{endpoint}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                if method == "GET":
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        raise Exception(f"Hub error: {resp.status}")
                
                elif method == "POST":
                    async with session.post(url, json=data) as resp:
                        if resp.status in (200, 201):
                            return await resp.json()
                        raise Exception(f"Hub error: {resp.status}")
                
                elif method == "PATCH":
                    async with session.patch(url, json=data) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        raise Exception(f"Hub error: {resp.status}")
                        
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    print(f"Integration Hub request failed: {e}")
                    return {"error": str(e)}
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return {"error": "Max retries exceeded"}
    
    # =========================================================================
    # CONTENT GENERATION (SCRIPTWRITER-X Integration)
    # =========================================================================
    
    async def get_personalized_hook(
        self,
        context: str,
        visitor_profile: Dict[str, Any] = None,
        content_type: str = "pitch"
    ) -> Dict[str, Any]:
        """
        Request personalized content from SCRIPTWRITER-X.
        
        Args:
            context: What the visitor asked about
            visitor_profile: Any known info about the visitor
            content_type: "hook", "pitch", or "objection_handler"
        
        Returns:
            {
                "generated_content": {...},
                "arsenal_options": [...],
                "timestamp": "..."
            }
        
        Example:
            result = await client.get_personalized_hook(
                context="How much does a custom AI system cost?",
                content_type="objection_handler"
            )
            response = result["generated_content"]["hook"]
        """
        return await self._request(
            "POST",
            "/api/v1/nexus/content-request",
            data={
                "context": context,
                "visitor_profile": visitor_profile or {},
                "content_type": content_type
            }
        )
    
    async def get_hook_arsenal(
        self,
        category: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Get hooks from SCRIPTWRITER-X arsenal for NEXUS BRAIN knowledge.
        
        Args:
            category: Filter by category (e.g., "pain_point", "curiosity")
            force_refresh: Bypass cache
        
        Returns:
            List of hooks with their text, category, and performance data
        
        Example:
            hooks = await client.get_hook_arsenal(category="value_prop")
            for hook in hooks:
                print(f"Hook: {hook['text']}")
                print(f"Conversion rate: {hook['conversion_rate']}%")
        """
        cache_key = f"arsenal_{category or 'all'}"
        
        # Check cache
        if not force_refresh and self._cache_timestamp:
            elapsed = (datetime.utcnow() - self._cache_timestamp).total_seconds()
            if elapsed < self.config.cache_ttl_seconds:
                if cache_key in self._hook_arsenal_cache:
                    return self._hook_arsenal_cache[cache_key]
        
        # Fetch from hub
        params = {"category": category} if category else {}
        result = await self._request("GET", "/api/v1/nexus/hook-arsenal", params=params)
        
        hooks = result.get("hooks", [])
        
        # Update cache
        self._hook_arsenal_cache[cache_key] = hooks
        self._cache_timestamp = datetime.utcnow()
        
        return hooks
    
    # =========================================================================
    # LEAD CAPTURE & TRACKING
    # =========================================================================
    
    async def handle_social_engagement(
        self,
        platform: Platform,
        contact_handle: str,
        message: str,
        engagement_type: str = "comment"
    ) -> Dict[str, Any]:
        """
        Handle incoming social media engagement.
        Call this when someone comments "A2I" or DMs "COMMERCIAL".
        
        Args:
            platform: Where the engagement came from
            contact_handle: @username or identifier
            message: What they said
            engagement_type: "comment" or "dm"
        
        Returns:
            {
                "lead_id": "...",
                "thread_id": "...",  # For conversation tracking
                "suggested_response": "...",  # AI-generated response
                "attribution": {
                    "source_post_id": "...",  # Which post led to this
                    "trigger_keyword": "..."
                }
            }
        
        Example:
            # In your webhook handler for Twitter DMs:
            result = await client.handle_social_engagement(
                platform=Platform.TWITTER,
                contact_handle="@prospect_user",
                message="I saw your post - COMMERCIAL",
                engagement_type="dm"
            )
            
            # Start conversation
            thread_id = result["thread_id"]
            initial_response = result["suggested_response"]
        """
        return await self._request(
            "POST",
            "/api/v1/nexus/social-engagement",
            data={
                "platform": platform.value,
                "contact_handle": contact_handle,
                "message": message,
                "engagement_type": engagement_type
            }
        )
    
    async def get_lead(self, lead_id: str) -> Dict[str, Any]:
        """Get lead details by ID"""
        return await self._request("GET", f"/api/v1/leads/{lead_id}")
    
    async def update_lead_status(
        self,
        lead_id: str,
        status: ConversionStatus,
        deal_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update lead status - triggers feedback loop on conversion/ghost.
        
        Args:
            lead_id: The lead to update
            status: New status
            deal_value: If converting, the deal value
        
        Example:
            # When lead converts:
            await client.update_lead_status(
                lead_id="abc123",
                status=ConversionStatus.CONVERTED,
                deal_value=75000.00
            )
            # This triggers feedback to SCRIPTWRITER-X marking the source post as "legendary"
        """
        data = {"status": status.value}
        if deal_value is not None:
            data["deal_value"] = deal_value
        
        return await self._request(
            "PATCH",
            f"/api/v1/leads/{lead_id}/status",
            data=data
        )
    
    async def list_leads(
        self,
        status: Optional[ConversionStatus] = None,
        platform: Optional[Platform] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """List leads with optional filters"""
        params = {"limit": limit}
        if status:
            params["status"] = status.value
        if platform:
            params["platform"] = platform.value
        
        return await self._request("GET", "/api/v1/leads", params=params)
    
    # =========================================================================
    # ANALYTICS
    # =========================================================================
    
    async def get_attribution_report(self) -> Dict[str, Any]:
        """Get comprehensive attribution analytics"""
        return await self._request("GET", "/api/v1/analytics/attribution")
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics"""
        return await self._request("GET", "/api/v1/analytics/feedback-loop")
    
    # =========================================================================
    # HEALTH & CLEANUP
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Integration Hub health"""
        return await self._request("GET", "/health")
    
    async def close(self):
        """Close the client session"""
        if self._session:
            await self._session.close()

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_client: Optional[NexusIntegrationClient] = None

def get_integration_client() -> NexusIntegrationClient:
    """Get singleton integration client"""
    global _client
    if _client is None:
        _client = NexusIntegrationClient()
    return _client

async def close_integration_client():
    """Close singleton client (call on shutdown)"""
    global _client
    if _client:
        await _client.close()
        _client = None

# ============================================================================
# HELPER FUNCTIONS FOR NEXUS BRAIN
# ============================================================================

async def enrich_response_with_hooks(
    base_response: str,
    context: str,
    visitor_profile: Dict[str, Any] = None
) -> str:
    """
    Enrich a NEXUS BRAIN response with SCRIPTWRITER-X hooks.
    
    Call this in your chat handler to make responses more compelling.
    
    Example:
        original = "We offer custom AI systems for businesses."
        enriched = await enrich_response_with_hooks(
            base_response=original,
            context="User asked about pricing",
            visitor_profile={"industry": "saas"}
        )
    """
    client = get_integration_client()
    
    # Get personalized hook
    hook_data = await client.get_personalized_hook(
        context=context,
        visitor_profile=visitor_profile,
        content_type="pitch"
    )
    
    generated = hook_data.get("generated_content", {})
    hook_text = generated.get("hook", "")
    
    if hook_text:
        # Prepend hook to response
        return f"{hook_text}\n\n{base_response}"
    
    return base_response

async def detect_trigger_keywords(message: str) -> Optional[str]:
    """
    Detect trigger keywords that indicate lead intent.
    
    Returns the keyword if found, None otherwise.
    
    Example:
        keyword = await detect_trigger_keywords("I want to learn more A2I")
        if keyword:
            # Capture as lead
            pass
    """
    message_lower = message.lower()
    
    trigger_keywords = [
        "a2i",
        "commercial",
        "pricing",
        "cost",
        "demo",
        "interested",
        "consultation",
        "quote",
        "proposal"
    ]
    
    for keyword in trigger_keywords:
        if keyword in message_lower:
            return keyword
    
    return None

# ============================================================================
# EXAMPLE USAGE IN NEXUS BRAIN CHAT HANDLER
# ============================================================================
"""
# In your existing chat endpoint (e.g., /api/chat):

from nexus_integration_client import (
    get_integration_client,
    Platform,
    LeadSource,
    ConversionStatus,
    detect_trigger_keywords,
    enrich_response_with_hooks
)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    client = get_integration_client()
    
    # Check for lead trigger keywords
    trigger = await detect_trigger_keywords(request.message)
    
    if trigger:
        # This is a potential lead - capture it
        engagement_result = await client.handle_social_engagement(
            platform=Platform.WEBSITE,
            contact_handle=request.session_id or request.user_email or "anonymous",
            message=request.message,
            engagement_type="website_chat"
        )
        
        # Use the suggested response as a starting point
        suggested = engagement_result.get("suggested_response", "")
        
        # Generate your normal response
        normal_response = await generate_chat_response(request.message)
        
        # Enrich with SCRIPTWRITER-X hooks
        enriched = await enrich_response_with_hooks(
            base_response=normal_response,
            context=request.message,
            visitor_profile={"session_id": request.session_id}
        )
        
        return {
            "response": enriched,
            "lead_captured": True,
            "lead_id": engagement_result.get("lead_id"),
            "thread_id": engagement_result.get("thread_id")
        }
    
    # Normal chat flow - still use hooks for better responses
    response = await generate_chat_response(request.message)
    enriched = await enrich_response_with_hooks(
        base_response=response,
        context=request.message
    )
    
    return {"response": enriched}


# Webhook for social media engagement (Twitter/LinkedIn):

@app.post("/webhooks/twitter-dm")
async def twitter_dm_webhook(payload: dict):
    client = get_integration_client()
    
    result = await client.handle_social_engagement(
        platform=Platform.TWITTER,
        contact_handle=payload["sender"]["username"],
        message=payload["message"]["text"],
        engagement_type="dm"
    )
    
    # Auto-respond with suggested response
    await send_twitter_dm(
        user_id=payload["sender"]["id"],
        message=result["suggested_response"]
    )
    
    return {"status": "processed", "lead_id": result["lead_id"]}


# Update lead status when conversation progresses:

@app.post("/api/leads/{lead_id}/qualify")
async def qualify_lead(lead_id: str, qualification: LeadQualification):
    client = get_integration_client()
    
    await client.update_lead_status(
        lead_id=lead_id,
        status=ConversionStatus.QUALIFIED
    )
    
    return {"status": "qualified"}


@app.post("/api/leads/{lead_id}/convert")
async def convert_lead(lead_id: str, deal: DealInfo):
    client = get_integration_client()
    
    # This triggers the feedback loop!
    # The source post will be marked as "legendary" in SCRIPTWRITER-X
    await client.update_lead_status(
        lead_id=lead_id,
        status=ConversionStatus.CONVERTED,
        deal_value=deal.value
    )
    
    return {"status": "converted", "feedback_sent": True}
"""
