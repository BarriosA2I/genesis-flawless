# GENESIS Flawless - Project Instructions

## Overview

GENESIS is the **ONLY backend** for all Barrios A2I services. All other backends have been deprecated and merged into this repository.

## Critical Rules

1. **This IS the backend** - Do not create new backend services
2. **Deploy to Ohio region** on Render (NOT Oregon)
3. **Add new endpoints here** - Do not spin up separate services

## Repository Info

| Property | Value |
|----------|-------|
| Name | GENESIS Flawless |
| URL | https://barrios-genesis-flawless.onrender.com |
| GitHub | BarriosA2I/barrios-genesis-flawless |
| Entry Point | flawless_api.py |

## API Endpoints

```
/api/chat                    → Main chat endpoint (23-agent system)
/api/genesis/trigger         → Trigger full pipeline
/api/genesis/research        → Market research
/api/genesis/stream/{id}     → SSE streaming
/api/trinity/analyze         → TRINITY 3-agent intelligence
/api/ragnarok/generate       → RAGNAROK video generation
/api/vortex/assemble         → Video assembly
/health                      → Health check
/metrics                     → Prometheus metrics
```

## Agent System (23 Agents)

### Core Pipeline (Agents 0-7)
- Agent 0: NEXUS Intake (lead qualification)
- Agent 1: Business Intelligence RAG
- Agent 2: Story Creator
- Agent 3: Video Prompt Engineer
- Agent 4: Video Generator
- Agent 5: Voiceover (ElevenLabs)
- Agent 6: Music Selector
- Agent 7: Video Assembly (VORTEX)

### Enhancement Agents (0.75, 1.5, 3.5, 5.5, 6.5)
- Agent 0.75: Lead Enrichment
- Agent 1.5: Competitor Deep Dive
- Agent 3.5: Scene Optimizer
- Agent 5.5: Voice Emotion Tuner
- Agent 6.5: Audio Mixer

### Strategic Agents (8-10)
- Agent 8: Meta-Learning
- Agent 9: A/B Variant Generator
- Agent 10: Performance Predictor

### Legendary Agents (7.5, 8.5, 11-15)
- Agent 7.5: THE AUTEUR (Vision QA)
- Agent 8.5: THE GENETICIST (DSPy Optimizer)
- Agent 11: THE ORACLE (Viral Predictor)
- Agent 12: THE CHAMELEON (Platform Adapter)
- Agent 13: THE MEMORY (Client DNA)
- Agent 14: THE HUNTER (Trend Scout)
- Agent 15: THE ACCOUNTANT (Budget Optimizer)

### Intelligence (TRINITY)
- TrendScout
- MarketAnalyst
- CompetitorTracker

## Quick Commands

```bash
# Run locally
uvicorn flawless_api:app --reload --port 8000

# Check health
curl https://barrios-genesis-flawless.onrender.com/health

# Deploy (push to trigger auto-deploy on Render)
git add . && git commit -m "feat: Description" && git push
```

## Related Files

- Full backend reference: `C:\Users\gary\.claude\memory\BACKENDS.md`
- Migration report: `C:\Users\gary\BACKEND_MERGE_REPORT.md`
- Frontend: `C:\Users\gary\frontend-barrios-landing`

## Deprecated Backends (Merged Into GENESIS)

| Name | Old URL | Status |
|------|---------|--------|
| Creative Director API | creative-director-api.onrender.com | DEPRECATED |
| NEXUS API | nexus-api.onrender.com | DEPRECATED |
| NEXUS Supergraph | nexus-supergraph.onrender.com | DEPRECATED |

---

*Last Updated: 2026-01-07*
