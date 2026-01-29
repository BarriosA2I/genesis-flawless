# NEXUS Integration Hub - Quick Start

## 🚀 5-Minute Setup

### Step 1: Copy Files to Your Project

```powershell
# From your python-genesis-flawless directory
mkdir nexus_integration

# Copy these files:
# - nexus_integration_hub.py     → Main integration service
# - nexus_integration_client.py  → Client for NEXUS BRAIN
# - docker-compose.yml           → Full stack deployment
# - init_db.sql                  → Database schema
```

### Step 2: Add Client to GENESIS Backend

```python
# In your main.py, add imports:
from nexus_integration_client import (
    get_integration_client,
    close_integration_client,
    configure_integration,
    Platform,
    ConversionStatus,
    detect_trigger_keywords,
    enrich_response_with_hooks
)

# In startup:
@app.on_event("startup")
async def startup():
    configure_integration(hub_url="http://localhost:8000")
    # ... existing startup code

# In shutdown:
@app.on_event("shutdown")
async def shutdown():
    await close_integration_client()
```

### Step 3: Update Chat Handler

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    client = get_integration_client()
    
    # 1. Check for lead triggers ("A2I", "commercial", "pricing", etc.)
    trigger = await detect_trigger_keywords(request.message)
    
    if trigger:
        # 2. Capture as lead
        result = await client.handle_social_engagement(
            platform=Platform.WEBSITE,
            contact_handle=request.session_id or "visitor",
            message=request.message,
            engagement_type="website_chat"
        )
        
        # 3. Use enhanced response from SCRIPTWRITER-X
        return {
            "response": result["suggested_response"],
            "lead_id": result["lead_id"]
        }
    
    # 4. Normal flow - still enrich with hooks
    base_response = await your_existing_response_function(request.message)
    enriched = await enrich_response_with_hooks(
        base_response=base_response,
        context=request.message
    )
    
    return {"response": enriched}
```

### Step 4: Start Services

```bash
# Start the full stack
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Step 5: Test Integration

```python
# Test lead capture
import httpx

async def test():
    async with httpx.AsyncClient() as client:
        # Simulate someone commenting "A2I" on Twitter
        resp = await client.post(
            "http://localhost:8000/api/v1/nexus/social-engagement",
            json={
                "platform": "twitter",
                "contact_handle": "@test_user",
                "message": "This is amazing! A2I",
                "engagement_type": "comment"
            }
        )
        print(resp.json())
        # Returns: {"lead_id": "...", "suggested_response": "...", ...}

import asyncio
asyncio.run(test())
```

## 📊 Verify It's Working

### Check Metrics
```bash
# View active leads
curl http://localhost:8000/api/v1/leads

# View attribution report
curl http://localhost:8000/api/v1/analytics/attribution

# View feedback loop stats
curl http://localhost:8000/api/v1/analytics/feedback-loop
```

### Monitor in Grafana
1. Open http://localhost:3000
2. Login: admin/admin
3. View "NEXUS Attribution" dashboard

## 🔧 Common Integration Points

### 1. CHROMADON → Hub (Post Registration)
```python
# After CHROMADON posts content:
await httpx.post(
    f"{HUB_URL}/api/v1/chromadon/posts/register",
    json={
        "platform": "twitter",
        "content": "Stop losing 40% of leads...",
        "hook_used": "pain_point_slow_response",
        "hook_category": "pain_point",
        "scriptwriter_session_id": session.id
    }
)
```

### 2. NEXUS BRAIN → Hub (Get Content)
```python
# When visitor asks about services:
content = await client.get_personalized_hook(
    context="User asking about AI automation pricing",
    visitor_profile={"industry": "saas"},
    content_type="pitch"
)
response = content["generated_content"]["hook"]
```

### 3. Lead Conversion → Feedback Loop
```python
# When deal closes:
await client.update_lead_status(
    lead_id="lead_abc123",
    status=ConversionStatus.CONVERTED,
    deal_value=75000.0
)
# This automatically:
# 1. Marks source post as "legendary"
# 2. Sends feedback to SCRIPTWRITER-X
# 3. Updates analytics
```

## 🛠️ Troubleshooting

### Hub Not Responding
```bash
# Check logs
docker logs nexus_integration_hub

# Check Redis connection
docker exec -it nexus_redis redis-cli ping
```

### Circuit Breaker Open
```bash
# Check health endpoint
curl http://localhost:8000/health
# Look at "services.scriptwriter_cb" and "services.chromadon_cb"
# Should be "closed" - if "open", service is failing
```

### Leads Not Being Attributed
```bash
# Check if posts are being registered
curl "http://localhost:8000/api/v1/chromadon/posts/recent?limit=5"

# Attribution looks at posts from last 7 days
# Make sure CHROMADON is registering posts
```

## 📈 Key Metrics to Watch

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Integration latency (p99) | <200ms | <500ms | >1000ms |
| Lead capture rate | >5% | 2-5% | <2% |
| Conversion rate | >10% | 5-10% | <5% |
| Circuit breaker state | closed | half_open | open |

---

**Questions?** Check ARCHITECTURE.md for full details.
