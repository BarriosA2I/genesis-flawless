# NEXUS Integration Hub - Architecture & Integration Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            BARRIOS A2I ECOSYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    ┌──────────────────┐         ┌──────────────────┐                           │
│    │  TWITTER/X       │         │  LINKEDIN        │                           │
│    │  ───────────     │         │  ─────────       │                           │
│    │  Posts 3x/day    │         │  Posts 3x/day    │                           │
│    │  DM handling     │         │  DM handling     │                           │
│    └────────┬─────────┘         └────────┬─────────┘                           │
│             │                            │                                      │
│             ▼                            ▼                                      │
│    ┌────────────────────────────────────────────────┐                          │
│    │           CHROMADON SOCIAL OVERLORD            │ ◄──── Browser Automation │
│    │  ─────────────────────────────────────────     │       (MCP Server)       │
│    │  • Multi-agent orchestration                   │                          │
│    │  • Circuit breakers per platform               │                          │
│    │  • MVCC checkpointing                          │                          │
│    │  • Engagement monitoring                       │                          │
│    └────────────────────┬───────────────────────────┘                          │
│                         │                                                       │
│            ┌────────────┴────────────┐                                         │
│            │                         │                                         │
│            ▼                         ▼                                         │
│    ┌─────────────────┐      ┌─────────────────────────────────────┐           │
│    │ SCRIPTWRITER-X  │      │      NEXUS INTEGRATION HUB         │           │
│    │ ───────────────│      │   ────────────────────────────      │           │
│    │                 │      │                                     │           │
│    │ Content Brain:  │◄────►│  Central Orchestrator:              │           │
│    │ • Hook Arsenal  │      │  • Event Bus (Redis Streams)        │           │
│    │ • Multi-Model   │      │  • Lead Attribution Engine          │           │
│    │ • Quality Gate  │      │  • Feedback Loop Manager            │           │
│    │ • Trend Jacker  │      │  • Circuit Breakers                 │           │
│    │ • Feedback Loop │      │  • OpenTelemetry Tracing            │           │
│    │                 │      │                                     │           │
│    └────────┬────────┘      └──────────────┬──────────────────────┘           │
│             │                              │                                   │
│             │         ┌────────────────────┘                                   │
│             │         │                                                        │
│             ▼         ▼                                                        │
│    ┌────────────────────────────────────────────────────────────┐             │
│    │                    NEXUS BRAIN                              │             │
│    │  ──────────────────────────────────────────────────────    │             │
│    │                                                             │             │
│    │  Website Assistant (barriosa2i.com):                        │             │
│    │  • Conversational AI with Generative UI                     │             │
│    │  • Lead capture & qualification                             │             │
│    │  • Service explanations                                     │             │
│    │  • Demo scheduling                                          │             │
│    │                                                             │             │
│    │  Integrations via Client:                                   │             │
│    │  • get_personalized_hook() ─────► SCRIPTWRITER-X            │             │
│    │  • handle_social_engagement() ──► Lead Attribution          │             │
│    │  • update_lead_status() ────────► Feedback Loop             │             │
│    │                                                             │             │
│    └────────────────────────────────────────────────────────────┘             │
│                         │                                                      │
│                         ▼                                                      │
│              ┌─────────────────────┐                                          │
│              │    CONVERSIONS      │                                          │
│              │    ────────────     │                                          │
│              │    $50K-$300K       │                                          │
│              │    Custom AI        │                                          │
│              │    Systems          │                                          │
│              └─────────────────────┘                                          │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

### 1. Content Creation Flow
```
CHROMADON                 SCRIPTWRITER-X              Integration Hub
    │                          │                           │
    ├─── Request Content ─────►│                           │
    │                          │                           │
    │◄── Generated Hook ───────┤                           │
    │    (with category,       │                           │
    │     visual prompt)       │                           │
    │                          │                           │
    ├─── Post to Platform ────►│                           │
    │                          │                           │
    ├─── Register Post ────────┼──────────────────────────►│
    │    (post_id, hook,       │                           │
    │     platform, content)   │                           │
    │                          │                           │
```

### 2. Lead Attribution Flow
```
Social Platform          Integration Hub              NEXUS BRAIN
       │                       │                          │
       ├─── Engagement ───────►│                          │
       │    ("A2I" comment)    │                          │
       │                       │                          │
       │                       ├─── Find Source Post ────►│
       │                       │    (last 7 days match)   │
       │                       │                          │
       │                       ├─── Create Lead ─────────►│
       │                       │    with attribution      │
       │                       │                          │
       │                       │◄── Get Response Hook ────┤
       │                       │                          │
       │◄── Respond ───────────┤                          │
       │    (AI-generated)     │                          │
```

### 3. Feedback Loop Flow
```
NEXUS BRAIN            Integration Hub           SCRIPTWRITER-X
    │                       │                         │
    ├─── Lead Converted ───►│                         │
    │    (deal_value: 75K)  │                         │
    │                       │                         │
    │                       ├─── Find Source Post ───►│
    │                       │                         │
    │                       ├─── Update Quality ─────►│
    │                       │    (LEGENDARY)          │
    │                       │                         │
    │                       ├─── Send Feedback ──────►│
    │                       │    {                    │
    │                       │      hook_id,           │
    │                       │      quality: "good",   │
    │                       │      conversions: 1     │
    │                       │    }                    │
    │                       │                         │
    │                       │                    ┌────┤
    │                       │                    │    │
    │                       │                    │ Learning:
    │                       │                    │ • Increase hook weight
    │                       │                    │ • Update category stats
    │                       │                    │ • Refine prompt templates
    │                       │                    └───►│
```

## 🔌 API Endpoints

### Integration Hub (`/api/v1/`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chromadon/posts/register` | POST | Register new social post |
| `/chromadon/posts/{id}/engagement` | PATCH | Update engagement metrics |
| `/nexus/social-engagement` | POST | Handle social media lead |
| `/nexus/content-request` | POST | Get content from SCRIPTWRITER-X |
| `/nexus/hook-arsenal` | GET | Get Hook Arsenal |
| `/leads/{id}` | GET | Get lead details |
| `/leads/{id}/status` | PATCH | Update lead status |
| `/leads` | GET | List leads (with filters) |
| `/analytics/attribution` | GET | Attribution report |
| `/analytics/feedback-loop` | GET | Feedback statistics |
| `/health` | GET | Health check |

## 📦 Installation

### 1. Add to GENESIS Backend

```bash
# Copy integration client to your backend
cp nexus_integration_client.py C:\Users\gary\python-genesis-flawless\

# Add to requirements.txt
echo "aiohttp>=3.9.0" >> requirements.txt
```

### 2. Initialize in your FastAPI app

```python
# In main.py or app startup
from nexus_integration_client import (
    configure_integration,
    get_integration_client,
    close_integration_client
)

@app.on_event("startup")
async def startup():
    configure_integration(
        hub_url="http://localhost:8000",  # Dev
        # hub_url="https://your-hub.render.com",  # Prod
    )

@app.on_event("shutdown")
async def shutdown():
    await close_integration_client()
```

### 3. Update Chat Handler

```python
from nexus_integration_client import (
    get_integration_client,
    Platform,
    ConversionStatus,
    detect_trigger_keywords,
    enrich_response_with_hooks
)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    client = get_integration_client()
    
    # Check for lead triggers
    trigger = await detect_trigger_keywords(request.message)
    
    if trigger:
        # Capture lead
        result = await client.handle_social_engagement(
            platform=Platform.WEBSITE,
            contact_handle=request.session_id,
            message=request.message,
            engagement_type="website_chat"
        )
        
        # Use SCRIPTWRITER-X enhanced response
        response = result["suggested_response"]
    else:
        # Normal response with hook enrichment
        base_response = await generate_response(request.message)
        response = await enrich_response_with_hooks(
            base_response=base_response,
            context=request.message
        )
    
    return {"response": response}
```

## 🚀 Deployment

### Local Development
```bash
cd nexus_integration_hub
docker-compose up -d
```

### Production (Render)
1. Deploy Integration Hub as new service
2. Set environment variables:
   - `REDIS_URL`: Your Redis connection
   - `SCRIPTWRITER_URL`: SCRIPTWRITER-X endpoint
   - `CHROMADON_URL`: CHROMADON endpoint
   - `DATABASE_URL`: PostgreSQL connection

3. Update GENESIS backend env:
   - `INTEGRATION_HUB_URL`: Hub endpoint

## 📈 Metrics & Monitoring

### Key Metrics
- `nexus_lead_attribution_total` - Leads by source/platform/status
- `nexus_feedback_loop_total` - Feedback signals processed
- `nexus_integration_latency_seconds` - Operation latencies
- `nexus_active_conversations` - Currently active leads

### Grafana Dashboards
- **Attribution Overview**: Posts → Leads → Conversions funnel
- **Platform Performance**: Conversion rates by platform
- **Hook Effectiveness**: Top performing hooks
- **Feedback Loop**: Quality ratings over time

## 🔄 Event Types

| Event | Trigger | Purpose |
|-------|---------|---------|
| `post.created` | CHROMADON posts | Track content for attribution |
| `lead.captured` | New engagement | Start lead tracking |
| `lead.status_changed` | Funnel progression | Track journey, trigger feedback |
| `content.feedback` | Conversion/Ghost | Train SCRIPTWRITER-X |
| `content.request` | NEXUS needs hook | Dynamic content generation |

## 🛡️ Circuit Breakers

Each external service has independent circuit breakers:

| Service | Failure Threshold | Reset Timeout |
|---------|-------------------|---------------|
| SCRIPTWRITER-X | 5 failures | 30 seconds |
| CHROMADON | 5 failures | 30 seconds |

## 📝 Example: Full Lead Journey

```python
# 1. CHROMADON posts content
await chromadon.post_content(
    platform="twitter",
    content="Stop losing 40% of your leads to slow response times...",
    hook_used="pain_point_slow_response"
)
# → Registered in Integration Hub with post_id

# 2. User comments "A2I" on post
# → Integration Hub:
#    - Matches comment to post (attribution)
#    - Creates lead with source_post_id
#    - Returns suggested response

# 3. NEXUS BRAIN handles conversation
response = await client.get_personalized_hook(
    context="User wants faster lead response",
    content_type="pitch"
)
# → SCRIPTWRITER-X generates compelling response

# 4. Lead qualifies and converts
await client.update_lead_status(
    lead_id="lead_123",
    status=ConversionStatus.CONVERTED,
    deal_value=75000.0
)
# → Feedback sent to SCRIPTWRITER-X
# → "pain_point_slow_response" hook marked LEGENDARY
# → Future content prioritizes this hook pattern
```

---

**Built with 💜 by Barrios A2I | Alienation 2 Innovation**
