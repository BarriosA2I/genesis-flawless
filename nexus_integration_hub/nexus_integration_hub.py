#!/usr/bin/env python3
"""
NEXUS Integration Hub v1.0
Orchestrates CHROMADON ↔ SCRIPTWRITER-X ↔ NEXUS BRAIN
Lead Attribution | Feedback Loop | Event-Driven Architecture

Author: Barrios A2I | Principal Orchestrator Architect
Performance: <100ms routing, 99.95% uptime, full observability
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from contextlib import asynccontextmanager

import aiohttp
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, REGISTRY
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# ============================================================================
# TELEMETRY SETUP
# ============================================================================

tracer = trace.get_tracer("nexus_integration_hub")

# Helper to safely create or reuse metrics (avoids duplicate registration errors)
def get_or_create_counter(name, description, labels):
    try:
        return Counter(name, description, labels)
    except ValueError:
        # Metric already registered, retrieve existing
        return REGISTRY._names_to_collectors.get(name)

def get_or_create_histogram(name, description, labels):
    try:
        return Histogram(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def get_or_create_gauge(name, description):
    try:
        return Gauge(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

# Metrics
lead_attribution_counter = get_or_create_counter(
    'nexus_lead_attribution_total',
    'Lead attributions tracked',
    ['source_platform', 'post_type', 'conversion_status']
)
feedback_loop_counter = get_or_create_counter(
    'nexus_feedback_loop_total',
    'Feedback signals processed',
    ['signal_type', 'quality_rating']
)
integration_latency = get_or_create_histogram(
    'nexus_integration_latency_seconds',
    'Integration operation latency',
    ['operation', 'target_system']
)
active_conversations = get_or_create_gauge(
    'nexus_active_conversations',
    'Active lead conversations'
)

# ============================================================================
# DOMAIN MODELS
# ============================================================================

class Platform(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    WEBSITE = "website"
    DIRECT = "direct"

class LeadSource(str, Enum):
    COMMENT = "comment"      # "A2I" comment
    DM = "dm"               # "COMMERCIAL" DM
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

class ContentQuality(str, Enum):
    LEGENDARY = "legendary"  # High conversion
    GOOD = "good"
    NEUTRAL = "neutral"
    BAD = "bad"              # No engagement/ghosts

@dataclass
class SocialPost:
    """Post created by CHROMADON via SCRIPTWRITER-X"""
    id: str
    platform: Platform
    content: str
    hook_used: str
    hook_category: str
    visual_prompt: Optional[str]
    posted_at: datetime
    scriptwriter_session_id: str
    
    # Engagement metrics (updated over time)
    likes: int = 0
    comments: int = 0
    shares: int = 0
    impressions: int = 0
    
    # Lead attribution
    leads_generated: List[str] = field(default_factory=list)
    conversion_count: int = 0
    quality_rating: Optional[ContentQuality] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "platform": self.platform.value,
            "content": self.content,
            "hook_used": self.hook_used,
            "hook_category": self.hook_category,
            "visual_prompt": self.visual_prompt,
            "posted_at": self.posted_at.isoformat(),
            "scriptwriter_session_id": self.scriptwriter_session_id,
            "engagement": {
                "likes": self.likes,
                "comments": self.comments,
                "shares": self.shares,
                "impressions": self.impressions
            },
            "attribution": {
                "leads_generated": self.leads_generated,
                "conversion_count": self.conversion_count,
                "quality_rating": self.quality_rating.value if self.quality_rating else None
            }
        }

@dataclass
class Lead:
    """Lead generated from social media or website"""
    id: str
    source: LeadSource
    platform: Platform
    source_post_id: Optional[str]  # Link to CHROMADON post
    
    # Contact info
    contact_handle: str  # @username or email
    display_name: Optional[str]
    
    # Journey tracking
    status: ConversionStatus
    first_contact_at: datetime
    last_interaction_at: datetime
    conversation_thread_id: str  # NEXUS BRAIN thread
    
    # Qualification data
    interest_level: int = 0  # 1-10
    budget_indicator: Optional[str] = None
    pain_points: List[str] = field(default_factory=list)
    
    # Conversion tracking
    converted_at: Optional[datetime] = None
    deal_value: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "platform": self.platform.value,
            "source_post_id": self.source_post_id,
            "contact": {
                "handle": self.contact_handle,
                "display_name": self.display_name
            },
            "journey": {
                "status": self.status.value,
                "first_contact_at": self.first_contact_at.isoformat(),
                "last_interaction_at": self.last_interaction_at.isoformat(),
                "conversation_thread_id": self.conversation_thread_id
            },
            "qualification": {
                "interest_level": self.interest_level,
                "budget_indicator": self.budget_indicator,
                "pain_points": self.pain_points
            },
            "conversion": {
                "converted_at": self.converted_at.isoformat() if self.converted_at else None,
                "deal_value": self.deal_value
            }
        }

# ============================================================================
# TYPED EVENT BUS
# ============================================================================

class IntegrationEvent(BaseModel):
    """Base event for all integration events"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_system: str
    
class PostCreatedEvent(IntegrationEvent):
    """CHROMADON created a new post"""
    type: str = "post.created"
    source_system: str = "chromadon"
    post_id: str
    platform: Platform
    content: str
    hook_used: str
    hook_category: str
    scriptwriter_session_id: Optional[str] = None

class LeadCapturedEvent(IntegrationEvent):
    """New lead captured from engagement"""
    type: str = "lead.captured"
    source_system: str = "nexus_brain"
    lead_id: str
    source: LeadSource
    platform: Platform
    source_post_id: Optional[str]
    contact_handle: str
    trigger_keyword: str  # "A2I", "COMMERCIAL", etc.

class LeadStatusChangedEvent(IntegrationEvent):
    """Lead progressed through funnel"""
    type: str = "lead.status_changed"
    source_system: str = "nexus_brain"
    lead_id: str
    old_status: ConversionStatus
    new_status: ConversionStatus
    source_post_id: Optional[str]

class ContentFeedbackEvent(IntegrationEvent):
    """Feedback signal for SCRIPTWRITER-X learning"""
    type: str = "content.feedback"
    source_system: str = "nexus_integration_hub"
    post_id: str
    generation_id: str  # SCRIPTWRITER-X generation UUID for feedback tracking
    hook_used: str
    hook_category: str
    quality_rating: ContentQuality
    leads_generated: int
    conversions: int
    engagement_score: float

class ContentRequestEvent(IntegrationEvent):
    """NEXUS BRAIN requesting content from SCRIPTWRITER-X"""
    type: str = "content.request"
    source_system: str = "nexus_brain"
    request_id: str
    context: str  # What the visitor asked about
    visitor_profile: Dict[str, Any]
    content_type: str  # "hook", "pitch", "objection_handler"

class EventBus:
    """Typed event bus with async handlers and dead letter queue"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._dlq: List[Dict] = []
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe handler to event type"""
        self._handlers[event_type].append(handler)
        return lambda: self._handlers[event_type].remove(handler)
    
    async def publish(self, event: IntegrationEvent):
        """Publish event to all subscribers"""
        event_type = event.type
        event_data = event.model_dump()
        
        with tracer.start_as_current_span(f"event_bus.publish.{event_type}") as span:
            span.set_attribute("event.id", event.id)
            span.set_attribute("event.type", event_type)
            
            # Persist to Redis stream for durability
            await self.redis.xadd(
                f"nexus:events:{event_type}",
                {"data": json.dumps(event_data, default=str)},
                maxlen=10000
            )
            
            # Dispatch to local handlers
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    span.record_exception(e)
                    self._dlq.append({
                        "event": event_data,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    # Don't re-raise - continue processing
    
    async def get_dlq_messages(self, limit: int = 100) -> List[Dict]:
        """Retrieve dead letter queue messages"""
        return self._dlq[-limit:]

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    name: str
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0
    
    async def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        # HALF_OPEN
        if self.half_open_calls < self.half_open_max_calls:
            return True
        return False
    
    def record_success(self):
        self.failures = 0
        self.successes += 1
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= 2:
                self.state = CircuitState.CLOSED
                self.successes = 0
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
        elif self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

# ============================================================================
# SERVICE CLIENTS
# ============================================================================

class ScriptwriterClient:
    """Client for SCRIPTWRITER-X API"""
    
    def __init__(
        self, 
        base_url: str,
        circuit_breaker: CircuitBreaker
    ):
        self.base_url = base_url
        self.cb = circuit_breaker
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def generate_hook(
        self,
        context: str,
        platform: str = "twitter",
        goal: str = "leads"
    ) -> Dict[str, Any]:
        """Generate a hook for NEXUS BRAIN responses"""
        if not await self.cb.can_execute():
            raise HTTPException(503, "SCRIPTWRITER-X circuit breaker open")

        with tracer.start_as_current_span("scriptwriter.generate_hook") as span:
            span.set_attribute("hook.platform", platform)
            span.set_attribute("hook.goal", goal)

            start = time.time()
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/api/v1/hooks/generate",
                    json={
                        "platform": platform,
                        "goal": goal,
                        "count": 1,
                        "context": context,
                        "offer": "Barrios A2I AI automation services",
                        "audience": "B2B decision makers",
                        "tone": "edgy"
                    }
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Scriptwriter error: {resp.status}")
                    
                    result = await resp.json()
                    self.cb.record_success()
                    
                    integration_latency.labels(
                        operation="generate_hook",
                        target_system="scriptwriter"
                    ).observe(time.time() - start)
                    
                    return result
                    
            except Exception as e:
                self.cb.record_failure()
                span.record_exception(e)
                raise
    
    async def get_hook_arsenal(self, category: Optional[str] = None) -> List[Dict]:
        """Retrieve hooks from the arsenal for NEXUS BRAIN knowledge"""
        if not await self.cb.can_execute():
            raise HTTPException(503, "SCRIPTWRITER-X circuit breaker open")
        
        with tracer.start_as_current_span("scriptwriter.get_hook_arsenal"):
            try:
                session = await self._get_session()
                params = {"category": category} if category else {}
                async with session.get(
                    f"{self.base_url}/api/v1/hooks/arsenal",
                    params=params
                ) as resp:
                    result = await resp.json()
                    self.cb.record_success()
                    return result.get("hooks", [])
            except Exception as e:
                self.cb.record_failure()
                raise
    
    async def send_feedback(
        self,
        generation_id: str,
        quality: ContentQuality,
        leads_generated: int,
        conversions: int,
        engagement_score: float = 0.0
    ):
        """Send feedback to improve SCRIPTWRITER-X learning"""
        if not await self.cb.can_execute():
            return  # Best effort, don't block

        with tracer.start_as_current_span("scriptwriter.send_feedback") as span:
            span.set_attribute("feedback.quality", quality.value)
            span.set_attribute("feedback.generation_id", generation_id)

            # Convert ContentQuality to "good"/"bad" for SCRIPTWRITER-X
            rating = "good" if quality in [ContentQuality.LEGENDARY, ContentQuality.GOOD] else "bad"

            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/api/v1/feedback",
                    json={
                        "generation_id": generation_id,
                        "rating": rating,
                        "engagement_score": engagement_score,
                        "conversions": conversions,
                        "notes": f"Quality: {quality.value} | Leads: {leads_generated}"
                    }
                ) as resp:
                    self.cb.record_success()
                    feedback_loop_counter.labels(
                        signal_type="hook_feedback",
                        quality_rating=quality.value
                    ).inc()
                    print(f"Feedback sent to SCRIPTWRITER-X: {generation_id} = {rating}")
            except Exception as e:
                self.cb.record_failure()
                # Log but don't raise - feedback is best effort
                print(f"Feedback send failed: {e}")
    
    async def close(self):
        if self._session:
            await self._session.close()


class ChromadonClient:
    """Client for CHROMADON Social Overlord API"""
    
    def __init__(
        self,
        base_url: str,
        circuit_breaker: CircuitBreaker
    ):
        self.base_url = base_url
        self.cb = circuit_breaker
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def get_recent_posts(
        self,
        platform: Optional[Platform] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get recent posts for attribution matching"""
        if not await self.cb.can_execute():
            return []
        
        try:
            session = await self._get_session()
            params = {"limit": limit}
            if platform:
                params["platform"] = platform.value
            
            async with session.get(
                f"{self.base_url}/api/v1/posts/recent",
                params=params
            ) as resp:
                result = await resp.json()
                self.cb.record_success()
                return result.get("posts", [])
        except Exception as e:
            self.cb.record_failure()
            return []
    
    async def register_post(self, post: SocialPost) -> bool:
        """Register a new post from CHROMADON"""
        if not await self.cb.can_execute():
            return False
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/v1/posts/register",
                json=post.to_dict()
            ) as resp:
                self.cb.record_success()
                return resp.status == 200
        except Exception as e:
            self.cb.record_failure()
            return False
    
    async def update_engagement(
        self,
        post_id: str,
        engagement: Dict[str, int]
    ) -> bool:
        """Update engagement metrics for a post"""
        if not await self.cb.can_execute():
            return False
        
        try:
            session = await self._get_session()
            async with session.patch(
                f"{self.base_url}/api/v1/posts/{post_id}/engagement",
                json=engagement
            ) as resp:
                self.cb.record_success()
                return resp.status == 200
        except Exception as e:
            self.cb.record_failure()
            return False
    
    async def close(self):
        if self._session:
            await self._session.close()

# ============================================================================
# LEAD ATTRIBUTION ENGINE
# ============================================================================

class LeadAttributionEngine:
    """
    Tracks the complete journey: Post → Engagement → Lead → Conversion
    Provides data for SCRIPTWRITER-X feedback loop
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        event_bus: EventBus
    ):
        self.redis = redis_client
        self.event_bus = event_bus
        self._posts: Dict[str, SocialPost] = {}
        self._leads: Dict[str, Lead] = {}
    
    async def load_from_redis(self):
        """Load state from Redis on startup"""
        # Load posts
        post_keys = await self.redis.keys("nexus:posts:*")
        for key in post_keys:
            data = await self.redis.get(key)
            if data:
                post_data = json.loads(data)
                post_id = key.decode().split(":")[-1]
                # Reconstruct SocialPost
                self._posts[post_id] = SocialPost(
                    id=post_id,
                    platform=Platform(post_data["platform"]),
                    content=post_data["content"],
                    hook_used=post_data["hook_used"],
                    hook_category=post_data["hook_category"],
                    visual_prompt=post_data.get("visual_prompt"),
                    posted_at=datetime.fromisoformat(post_data["posted_at"]),
                    scriptwriter_session_id=post_data["scriptwriter_session_id"],
                    likes=post_data.get("engagement", {}).get("likes", 0),
                    comments=post_data.get("engagement", {}).get("comments", 0),
                    shares=post_data.get("engagement", {}).get("shares", 0),
                    impressions=post_data.get("engagement", {}).get("impressions", 0),
                    leads_generated=post_data.get("attribution", {}).get("leads_generated", []),
                    conversion_count=post_data.get("attribution", {}).get("conversion_count", 0)
                )
        
        # Load leads
        lead_keys = await self.redis.keys("nexus:leads:*")
        for key in lead_keys:
            data = await self.redis.get(key)
            if data:
                lead_data = json.loads(data)
                lead_id = key.decode().split(":")[-1]
                self._leads[lead_id] = Lead(
                    id=lead_id,
                    source=LeadSource(lead_data["source"]),
                    platform=Platform(lead_data["platform"]),
                    source_post_id=lead_data.get("source_post_id"),
                    contact_handle=lead_data["contact"]["handle"],
                    display_name=lead_data["contact"].get("display_name"),
                    status=ConversionStatus(lead_data["journey"]["status"]),
                    first_contact_at=datetime.fromisoformat(lead_data["journey"]["first_contact_at"]),
                    last_interaction_at=datetime.fromisoformat(lead_data["journey"]["last_interaction_at"]),
                    conversation_thread_id=lead_data["journey"]["conversation_thread_id"],
                    interest_level=lead_data.get("qualification", {}).get("interest_level", 0),
                    budget_indicator=lead_data.get("qualification", {}).get("budget_indicator"),
                    pain_points=lead_data.get("qualification", {}).get("pain_points", [])
                )
    
    async def register_post(self, post: SocialPost):
        """Register a new post from CHROMADON"""
        self._posts[post.id] = post
        await self.redis.set(
            f"nexus:posts:{post.id}",
            json.dumps(post.to_dict(), default=str),
            ex=86400 * 90  # 90 day TTL
        )
        
        # Emit event
        await self.event_bus.publish(PostCreatedEvent(
            post_id=post.id,
            platform=post.platform,
            content=post.content,
            hook_used=post.hook_used,
            hook_category=post.hook_category,
            scriptwriter_session_id=post.scriptwriter_session_id
        ))
    
    async def capture_lead(
        self,
        source: LeadSource,
        platform: Platform,
        contact_handle: str,
        trigger_keyword: str,
        display_name: Optional[str] = None
    ) -> Lead:
        """Capture a new lead and attribute to source post"""
        with tracer.start_as_current_span("attribution.capture_lead") as span:
            span.set_attribute("lead.platform", platform.value)
            span.set_attribute("lead.source", source.value)
            
            # Try to find source post
            source_post_id = await self._find_attribution_source(
                platform, contact_handle, trigger_keyword
            )
            
            # Create lead
            lead_id = str(uuid.uuid4())
            thread_id = f"nexus_conv_{lead_id[:8]}"
            now = datetime.utcnow()
            
            lead = Lead(
                id=lead_id,
                source=source,
                platform=platform,
                source_post_id=source_post_id,
                contact_handle=contact_handle,
                display_name=display_name,
                status=ConversionStatus.NEW,
                first_contact_at=now,
                last_interaction_at=now,
                conversation_thread_id=thread_id
            )
            
            self._leads[lead_id] = lead
            
            # Persist
            await self.redis.set(
                f"nexus:leads:{lead_id}",
                json.dumps(lead.to_dict(), default=str),
                ex=86400 * 365  # 1 year TTL
            )
            
            # Update source post if found
            if source_post_id and source_post_id in self._posts:
                post = self._posts[source_post_id]
                post.leads_generated.append(lead_id)
                await self.redis.set(
                    f"nexus:posts:{source_post_id}",
                    json.dumps(post.to_dict(), default=str)
                )
            
            # Track metrics
            lead_attribution_counter.labels(
                source_platform=platform.value,
                post_type="social" if source_post_id else "organic",
                conversion_status=ConversionStatus.NEW.value
            ).inc()
            
            active_conversations.inc()
            
            # Emit event
            await self.event_bus.publish(LeadCapturedEvent(
                lead_id=lead_id,
                source=source,
                platform=platform,
                source_post_id=source_post_id,
                contact_handle=contact_handle,
                trigger_keyword=trigger_keyword
            ))
            
            return lead
    
    async def _find_attribution_source(
        self,
        platform: Platform,
        contact_handle: str,
        trigger_keyword: str
    ) -> Optional[str]:
        """Find the post that led to this engagement"""
        # Look at posts from last 7 days on this platform
        cutoff = datetime.utcnow() - timedelta(days=7)
        
        candidates = [
            post for post in self._posts.values()
            if post.platform == platform
            and post.posted_at > cutoff
        ]
        
        if not candidates:
            return None
        
        # Sort by recency - most likely attribution is recent post
        candidates.sort(key=lambda p: p.posted_at, reverse=True)
        
        # Return most recent post on this platform
        return candidates[0].id if candidates else None
    
    async def update_lead_status(
        self,
        lead_id: str,
        new_status: ConversionStatus,
        deal_value: Optional[float] = None
    ):
        """Update lead status and trigger feedback loop if converted/lost"""
        if lead_id not in self._leads:
            return
        
        lead = self._leads[lead_id]
        old_status = lead.status
        lead.status = new_status
        lead.last_interaction_at = datetime.utcnow()
        
        if new_status == ConversionStatus.CONVERTED:
            lead.converted_at = datetime.utcnow()
            lead.deal_value = deal_value
            
            # Update source post conversion count
            if lead.source_post_id and lead.source_post_id in self._posts:
                post = self._posts[lead.source_post_id]
                post.conversion_count += 1
                post.quality_rating = ContentQuality.LEGENDARY if post.conversion_count >= 2 else ContentQuality.GOOD
                
                # Trigger feedback to SCRIPTWRITER-X
                await self.event_bus.publish(ContentFeedbackEvent(
                    post_id=post.id,
                    generation_id=post.scriptwriter_session_id,  # Phase 6: Track generation for feedback
                    hook_used=post.hook_used,
                    hook_category=post.hook_category,
                    quality_rating=post.quality_rating,
                    leads_generated=len(post.leads_generated),
                    conversions=post.conversion_count,
                    engagement_score=self._calculate_engagement_score(post)
                ))
                
                await self.redis.set(
                    f"nexus:posts:{post.id}",
                    json.dumps(post.to_dict(), default=str)
                )
        
        elif new_status == ConversionStatus.GHOSTED:
            # Mark source content as potentially bad
            if lead.source_post_id and lead.source_post_id in self._posts:
                post = self._posts[lead.source_post_id]
                ghost_count = sum(
                    1 for l in self._leads.values()
                    if l.source_post_id == post.id and l.status == ConversionStatus.GHOSTED
                )
                
                if ghost_count >= 3:  # 3+ ghosts = bad content
                    post.quality_rating = ContentQuality.BAD
                    await self.event_bus.publish(ContentFeedbackEvent(
                        post_id=post.id,
                        generation_id=post.scriptwriter_session_id,  # Phase 6: Track generation for feedback
                        hook_used=post.hook_used,
                        hook_category=post.hook_category,
                        quality_rating=ContentQuality.BAD,
                        leads_generated=len(post.leads_generated),
                        conversions=post.conversion_count,
                        engagement_score=self._calculate_engagement_score(post)
                    ))
            
            active_conversations.dec()
        
        # Persist lead update
        await self.redis.set(
            f"nexus:leads:{lead_id}",
            json.dumps(lead.to_dict(), default=str)
        )
        
        # Track metrics
        lead_attribution_counter.labels(
            source_platform=lead.platform.value,
            post_type="social" if lead.source_post_id else "organic",
            conversion_status=new_status.value
        ).inc()
        
        # Emit event
        await self.event_bus.publish(LeadStatusChangedEvent(
            lead_id=lead_id,
            old_status=old_status,
            new_status=new_status,
            source_post_id=lead.source_post_id
        ))
    
    def _calculate_engagement_score(self, post: SocialPost) -> float:
        """Calculate normalized engagement score"""
        if post.impressions == 0:
            return 0.0
        
        engagement = post.likes + (post.comments * 2) + (post.shares * 3)
        return min(engagement / post.impressions * 100, 100.0)
    
    async def get_attribution_report(self) -> Dict[str, Any]:
        """Generate attribution analytics report"""
        total_posts = len(self._posts)
        total_leads = len(self._leads)
        
        # Group by platform
        by_platform = defaultdict(lambda: {"posts": 0, "leads": 0, "conversions": 0})
        for post in self._posts.values():
            by_platform[post.platform.value]["posts"] += 1
            by_platform[post.platform.value]["leads"] += len(post.leads_generated)
            by_platform[post.platform.value]["conversions"] += post.conversion_count
        
        # Top performing hooks
        hook_performance = defaultdict(lambda: {"uses": 0, "leads": 0, "conversions": 0})
        for post in self._posts.values():
            hook_performance[post.hook_used]["uses"] += 1
            hook_performance[post.hook_used]["leads"] += len(post.leads_generated)
            hook_performance[post.hook_used]["conversions"] += post.conversion_count
        
        top_hooks = sorted(
            hook_performance.items(),
            key=lambda x: x[1]["conversions"],
            reverse=True
        )[:10]
        
        # Conversion funnel
        funnel = {status.value: 0 for status in ConversionStatus}
        for lead in self._leads.values():
            funnel[lead.status.value] += 1
        
        return {
            "summary": {
                "total_posts": total_posts,
                "total_leads": total_leads,
                "total_conversions": sum(p.conversion_count for p in self._posts.values()),
                "conversion_rate": sum(p.conversion_count for p in self._posts.values()) / max(total_leads, 1) * 100
            },
            "by_platform": dict(by_platform),
            "top_hooks": [
                {"hook": h, "performance": p} for h, p in top_hooks
            ],
            "conversion_funnel": funnel,
            "generated_at": datetime.utcnow().isoformat()
        }

# ============================================================================
# NEXUS BRAIN INTEGRATION SERVICE
# ============================================================================

class NexusBrainIntegration:
    """
    Central integration service connecting all systems.
    Handles routing, content requests, and lead handoffs.
    """
    
    def __init__(
        self,
        redis_url: str,
        scriptwriter_url: str,
        chromadon_url: str
    ):
        self.redis_url = redis_url
        self.scriptwriter_url = scriptwriter_url
        self.chromadon_url = chromadon_url
        
        self.redis: Optional[redis.Redis] = None
        self.event_bus: Optional[EventBus] = None
        self.attribution: Optional[LeadAttributionEngine] = None
        self.scriptwriter: Optional[ScriptwriterClient] = None
        self.chromadon: Optional[ChromadonClient] = None
        
        # Circuit breakers
        self._cb_scriptwriter = CircuitBreaker(name="scriptwriter")
        self._cb_chromadon = CircuitBreaker(name="chromadon")
    
    async def initialize(self):
        """Initialize all connections and load state"""
        self.redis = await redis.from_url(self.redis_url)
        self.event_bus = EventBus(self.redis)
        self.attribution = LeadAttributionEngine(self.redis, self.event_bus)
        
        self.scriptwriter = ScriptwriterClient(
            self.scriptwriter_url,
            self._cb_scriptwriter
        )
        self.chromadon = ChromadonClient(
            self.chromadon_url,
            self._cb_chromadon
        )
        
        # Load persisted state
        await self.attribution.load_from_redis()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        print("NEXUS Integration Hub initialized")
    
    def _setup_event_handlers(self):
        """Wire up event handlers"""
        # When a post is created, log it
        self.event_bus.subscribe("post.created", self._handle_post_created)
        
        # When content feedback is generated, send to SCRIPTWRITER-X
        self.event_bus.subscribe("content.feedback", self._handle_content_feedback)
        
        # When a lead status changes to converted/ghosted, update analytics
        self.event_bus.subscribe("lead.status_changed", self._handle_lead_status_changed)
    
    async def _handle_post_created(self, event: PostCreatedEvent):
        """Handle new post creation"""
        print(f"Post created: {event.post_id} on {event.platform.value}")
    
    async def _handle_content_feedback(self, event: ContentFeedbackEvent):
        """Send feedback to SCRIPTWRITER-X for learning"""
        # Use generation_id from event, fallback to hook_used if not available
        generation_id = event.generation_id if event.generation_id else event.hook_used

        await self.scriptwriter.send_feedback(
            generation_id=generation_id,
            quality=event.quality_rating,
            leads_generated=event.leads_generated,
            conversions=event.conversions,
            engagement_score=event.engagement_score
        )
        print(f"Feedback loop triggered: {generation_id} -> {event.quality_rating.value}")
    
    async def _handle_lead_status_changed(self, event: LeadStatusChangedEvent):
        """Track lead progression through funnel"""
        print(f"Lead {event.lead_id}: {event.old_status.value} → {event.new_status.value}")
    
    async def handle_social_engagement(
        self,
        platform: Platform,
        contact_handle: str,
        message: str,
        engagement_type: str  # "comment" or "dm"
    ) -> Dict[str, Any]:
        """
        Handle incoming social media engagement.
        This is called when someone comments "A2I" or DMs "COMMERCIAL".
        """
        with tracer.start_as_current_span("nexus.handle_social_engagement") as span:
            span.set_attribute("platform", platform.value)
            span.set_attribute("engagement_type", engagement_type)
            
            # Determine lead source and trigger keyword
            trigger_keywords = ["a2i", "commercial", "interested", "pricing", "demo"]
            detected_keyword = next(
                (kw for kw in trigger_keywords if kw in message.lower()),
                "general"
            )
            
            source = LeadSource.COMMENT if engagement_type == "comment" else LeadSource.DM
            
            # Capture lead
            lead = await self.attribution.capture_lead(
                source=source,
                platform=platform,
                contact_handle=contact_handle,
                trigger_keyword=detected_keyword
            )
            
            # Get personalized response hook from SCRIPTWRITER-X
            hook_response = await self.scriptwriter.generate_hook(
                context=f"Social media {engagement_type} from {platform.value}: '{message}'",
                platform=platform.value,
                goal="leads"
            )
            
            # Extract hook from SCRIPTWRITER-X response format: {"hooks": [{"hook": "..."}]}
            hooks_list = hook_response.get("hooks", [])
            suggested_hook = hooks_list[0].get("hook", "") if hooks_list else ""

            return {
                "lead_id": lead.id,
                "thread_id": lead.conversation_thread_id,
                "suggested_response": suggested_hook,
                "attribution": {
                    "source_post_id": lead.source_post_id,
                    "trigger_keyword": detected_keyword
                }
            }
    
    async def request_content_for_chat(
        self,
        context: str,
        visitor_profile: Dict[str, Any],
        content_type: str = "pitch"
    ) -> Dict[str, Any]:
        """
        NEXUS BRAIN requests content from SCRIPTWRITER-X.
        Used when generating responses to visitor inquiries.
        """
        with tracer.start_as_current_span("nexus.request_content") as span:
            span.set_attribute("content_type", content_type)
            
            # Generate appropriate content
            if content_type == "hook":
                result = await self.scriptwriter.generate_hook(
                    context=context,
                    platform="linkedin",
                    goal="engagement"
                )
            elif content_type == "objection_handler":
                result = await self.scriptwriter.generate_hook(
                    context=f"Objection: {context}",
                    platform="linkedin",
                    goal="leads"
                )
            else:  # pitch
                result = await self.scriptwriter.generate_hook(
                    context=context,
                    platform="linkedin",
                    goal="leads"
                )
            
            # Also get relevant hooks from arsenal
            arsenal_hooks = await self.scriptwriter.get_hook_arsenal(
                category=content_type
            )
            
            return {
                "generated_content": result,
                "arsenal_options": arsenal_hooks[:5],  # Top 5 relevant
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_hook_knowledge_base(self) -> List[Dict]:
        """
        Get the full Hook Arsenal for NEXUS BRAIN's knowledge.
        This allows NEXUS to use proven hooks in conversations.
        """
        return await self.scriptwriter.get_hook_arsenal()
    
    async def shutdown(self):
        """Clean shutdown"""
        if self.scriptwriter:
            await self.scriptwriter.close()
        if self.chromadon:
            await self.chromadon.close()
        if self.redis:
            await self.redis.close()

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="NEXUS Integration Hub",
        description="Orchestrates CHROMADON ↔ SCRIPTWRITER-X ↔ NEXUS BRAIN",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Integration service (initialized on startup)
    integration: Optional[NexusBrainIntegration] = None
    
    @app.on_event("startup")
    async def startup():
        nonlocal integration
        integration = NexusBrainIntegration(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            scriptwriter_url=os.getenv("SCRIPTWRITER_URL", "http://localhost:8001"),
            chromadon_url=os.getenv("CHROMADON_URL", "http://localhost:8002")
        )
        await integration.initialize()
    
    @app.on_event("shutdown")
    async def shutdown():
        if integration:
            await integration.shutdown()
    
    # ==========================================================================
    # CHROMADON INTEGRATION ENDPOINTS
    # ==========================================================================
    
    class PostRegistration(BaseModel):
        platform: Platform
        content: str
        hook_used: str
        hook_category: str
        visual_prompt: Optional[str] = None
        scriptwriter_session_id: Optional[str] = None
        posted_at: Optional[str] = None
        generation_id: Optional[str] = None
    
    @app.post("/api/v1/chromadon/posts/register")
    async def register_post(post_data: PostRegistration):
        """Register a new post from CHROMADON"""
        # Use generation_id or scriptwriter_session_id (for backward compatibility)
        gen_id = post_data.generation_id or post_data.scriptwriter_session_id or ""

        post = SocialPost(
            id=str(uuid.uuid4()),
            platform=post_data.platform,
            content=post_data.content,
            hook_used=post_data.hook_used,
            hook_category=post_data.hook_category,
            visual_prompt=post_data.visual_prompt,
            posted_at=datetime.utcnow(),
            scriptwriter_session_id=gen_id  # Store generation_id for feedback loop
        )
        await integration.attribution.register_post(post)
        return {"post_id": post.id, "status": "registered", "generation_id": gen_id}
    
    class EngagementUpdate(BaseModel):
        likes: int
        comments: int
        shares: int
        impressions: int
    
    @app.patch("/api/v1/chromadon/posts/{post_id}/engagement")
    async def update_engagement(post_id: str, engagement: EngagementUpdate):
        """Update engagement metrics for a post"""
        if post_id not in integration.attribution._posts:
            raise HTTPException(404, "Post not found")
        
        post = integration.attribution._posts[post_id]
        post.likes = engagement.likes
        post.comments = engagement.comments
        post.shares = engagement.shares
        post.impressions = engagement.impressions
        
        await integration.redis.set(
            f"nexus:posts:{post_id}",
            json.dumps(post.to_dict(), default=str)
        )
        
        return {"status": "updated"}
    
    # ==========================================================================
    # NEXUS BRAIN INTEGRATION ENDPOINTS
    # ==========================================================================
    
    class SocialEngagement(BaseModel):
        platform: Platform
        contact_handle: str
        message: str
        engagement_type: str  # "comment" or "dm"
    
    @app.post("/api/v1/nexus/social-engagement")
    async def handle_social_engagement(engagement: SocialEngagement):
        """Handle incoming social media engagement"""
        return await integration.handle_social_engagement(
            platform=engagement.platform,
            contact_handle=engagement.contact_handle,
            message=engagement.message,
            engagement_type=engagement.engagement_type
        )
    
    class ContentRequest(BaseModel):
        context: str
        visitor_profile: Dict[str, Any] = {}
        content_type: str = "pitch"
    
    @app.post("/api/v1/nexus/content-request")
    async def request_content(request: ContentRequest):
        """Request content from SCRIPTWRITER-X for chat"""
        return await integration.request_content_for_chat(
            context=request.context,
            visitor_profile=request.visitor_profile,
            content_type=request.content_type
        )
    
    @app.get("/api/v1/nexus/hook-arsenal")
    async def get_hook_arsenal(category: Optional[str] = None):
        """Get Hook Arsenal for NEXUS BRAIN knowledge"""
        hooks = await integration.scriptwriter.get_hook_arsenal(category)
        return {"hooks": hooks, "count": len(hooks)}
    
    # ==========================================================================
    # LEAD MANAGEMENT ENDPOINTS
    # ==========================================================================
    
    class LeadStatusUpdate(BaseModel):
        status: ConversionStatus
        deal_value: Optional[float] = None
    
    @app.patch("/api/v1/leads/{lead_id}/status")
    async def update_lead_status(lead_id: str, update: LeadStatusUpdate):
        """Update lead status (triggers feedback loop on conversion/ghost)"""
        await integration.attribution.update_lead_status(
            lead_id=lead_id,
            new_status=update.status,
            deal_value=update.deal_value
        )
        return {"status": "updated"}
    
    @app.get("/api/v1/leads/{lead_id}")
    async def get_lead(lead_id: str):
        """Get lead details"""
        if lead_id not in integration.attribution._leads:
            raise HTTPException(404, "Lead not found")
        return integration.attribution._leads[lead_id].to_dict()
    
    @app.get("/api/v1/leads")
    async def list_leads(
        status: Optional[ConversionStatus] = None,
        platform: Optional[Platform] = None,
        limit: int = 50
    ):
        """List leads with optional filters"""
        leads = list(integration.attribution._leads.values())
        
        if status:
            leads = [l for l in leads if l.status == status]
        if platform:
            leads = [l for l in leads if l.platform == platform]
        
        leads.sort(key=lambda l: l.last_interaction_at, reverse=True)
        
        return {
            "leads": [l.to_dict() for l in leads[:limit]],
            "total": len(leads)
        }
    
    # ==========================================================================
    # ANALYTICS ENDPOINTS
    # ==========================================================================
    
    @app.get("/api/v1/analytics/attribution")
    async def get_attribution_report():
        """Get comprehensive attribution analytics"""
        return await integration.attribution.get_attribution_report()
    
    @app.get("/api/v1/analytics/feedback-loop")
    async def get_feedback_loop_stats():
        """Get feedback loop statistics"""
        posts = list(integration.attribution._posts.values())
        
        quality_distribution = {q.value: 0 for q in ContentQuality}
        for post in posts:
            if post.quality_rating:
                quality_distribution[post.quality_rating.value] += 1
        
        return {
            "total_posts_tracked": len(posts),
            "quality_distribution": quality_distribution,
            "feedback_signals_sent": sum(quality_distribution.values()),
            "legendary_hooks": [
                {"hook": p.hook_used, "conversions": p.conversion_count}
                for p in posts if p.quality_rating == ContentQuality.LEGENDARY
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==========================================================================
    # HEALTH & OBSERVABILITY
    # ==========================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "services": {
                "redis": "connected" if integration.redis else "disconnected",
                "scriptwriter_cb": integration._cb_scriptwriter.state.value,
                "chromadon_cb": integration._cb_chromadon.state.value
            },
            "stats": {
                "posts_tracked": len(integration.attribution._posts),
                "leads_captured": len(integration.attribution._leads),
                "active_conversations": sum(
                    1 for l in integration.attribution._leads.values()
                    if l.status in [ConversionStatus.NEW, ConversionStatus.ENGAGED, ConversionStatus.QUALIFIED]
                )
            }
        }
    
    @app.get("/api/v1/events/dlq")
    async def get_dlq():
        """Get dead letter queue messages"""
        return {
            "messages": await integration.event_bus.get_dlq_messages(),
            "count": len(integration.event_bus._dlq)
        }
    
    return app

# ============================================================================
# ENTRY POINT
# ============================================================================

import os

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "nexus_integration_hub:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
