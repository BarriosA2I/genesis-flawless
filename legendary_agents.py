"""
================================================================================
RAGNAROK LEGENDARY AGENTS (7.5, 8.5, 11-15)
================================================================================
The apex predators of the 23-agent cognitive pipeline.

Agent 7.5:  THE AUTEUR      - Vision-Language Quality Assurance
Agent 8.5:  THE GENETICIST  - DSPy Prompt Evolution
Agent 11:   THE ORACLE      - Viral Prediction Engine
Agent 12:   THE CHAMELEON   - Platform Adaptation
Agent 13:   THE MEMORY      - Client DNA Profiling
Agent 14:   THE HUNTER      - Trend Scouting
Agent 15:   THE ACCOUNTANT  - Budget Optimization

Author: Barrios A2I | Version: 1.0.0 | Neural RAG Brain v3.0
================================================================================
"""

import asyncio
import logging
import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from opentelemetry import trace
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

LEGENDARY_AGENT_CALLS = Counter(
    'ragnarok_legendary_agent_calls_total',
    'Total calls to legendary agents',
    ['agent_name']
)

LEGENDARY_AGENT_LATENCY = Histogram(
    'ragnarok_legendary_agent_latency_ms',
    'Legendary agent processing latency',
    ['agent_name'],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000]
)

VIRAL_SCORE = Gauge(
    'ragnarok_viral_prediction_score',
    'Latest viral prediction score',
    ['content_type']
)


# =============================================================================
# MOCK LLM CLIENT (For standalone operation without full Anthropic setup)
# =============================================================================

class MockLLMClient:
    """Mock LLM client for when Anthropic is unavailable"""

    async def generate(self, prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
        """Generate mock response based on prompt analysis"""
        logger.warning(f"Using MockLLMClient - no real LLM available")

        # Return appropriate mock response based on prompt content
        if "viral" in prompt.lower():
            return json.dumps({
                "viral_score": 0.72,
                "momentum": 0.4,
                "peak_timing": "Tuesday 10am EST",
                "risk_factors": ["Market saturation", "Timing sensitivity"],
                "amplification_tips": ["Use trending hashtags", "Cross-post to LinkedIn"],
                "confidence": 0.75
            })
        elif "platform" in prompt.lower() or "adapt" in prompt.lower():
            return json.dumps({
                "adapted_content": {
                    "title": "AI Automation for Modern Business",
                    "description": "Discover how AI can transform your workflow",
                    "hook": "What if your business could run itself?",
                    "cta": "Book your free consultation"
                },
                "format_specs": {"duration_seconds": 60, "aspect_ratio": "9:16"},
                "hashtags": ["#AIAutomation", "#BusinessGrowth", "#TechInnovation"],
                "optimal_posting_time": "Wednesday 2pm EST",
                "engagement_prediction": 0.68
            })
        elif "trend" in prompt.lower() or "hunt" in prompt.lower():
            return json.dumps({
                "trends": [
                    {"topic": "AI Automation", "momentum": 0.85, "relevance": 0.9},
                    {"topic": "Workflow Optimization", "momentum": 0.7, "relevance": 0.8}
                ],
                "emerging": [{"topic": "AI Agents", "growth_rate": 0.6}],
                "declining": [{"topic": "Manual Processes", "decline_rate": -0.4}],
                "opportunities": ["AI video content", "Automation case studies"],
                "risks": ["AI fatigue", "Over-automation messaging"],
                "recommended_topics": ["AI ROI", "Time savings", "Success stories"]
            })
        elif "budget" in prompt.lower() or "cost" in prompt.lower():
            return json.dumps({
                "total_budget": 1000.00,
                "allocation": {
                    "video_production": 400,
                    "voiceover": 100,
                    "music": 50,
                    "platform_ads": 350,
                    "contingency": 100
                },
                "expected_roi": 3.5,
                "cost_per_result": {
                    "cost_per_view": 0.015,
                    "cost_per_engagement": 0.12,
                    "cost_per_lead": 4.50
                },
                "savings_opportunities": ["Batch video processing", "Reuse audio assets"],
                "risk_assessment": {"budget_risk": "low", "timeline_risk": "medium"}
            })
        elif "quality" in prompt.lower() or "video" in prompt.lower():
            return json.dumps({
                "visual_quality_score": 0.85,
                "brand_consistency": 0.82,
                "message_clarity": 0.88,
                "emotional_impact": 0.75,
                "technical_issues": [],
                "recommendations": ["Increase contrast", "Add captions"],
                "overall_score": 0.83
            })
        elif "evolve" in prompt.lower() or "prompt" in prompt.lower():
            return json.dumps({
                "optimized": True,
                "improvement": 0.15
            })
        else:
            return json.dumps({
                "brand_essence": {"values": ["Innovation", "Excellence"]},
                "preferred_styles": ["Modern", "Professional"],
                "successful_patterns": [{"pattern": "Case study format", "success_rate": 0.85}],
                "avoid_patterns": [{"pattern": "Hard sell", "failure_rate": 0.6}],
                "voice_characteristics": {"tone": "Professional", "formality": "Medium"},
                "historical_performance": {"avg_engagement": 0.12, "avg_conversion": 0.03}
            })

    async def analyze_images(self, prompt: str, images: List[str], model: str) -> str:
        """Mock image analysis"""
        return json.dumps({
            "visual_quality_score": 0.85,
            "brand_consistency": 0.82,
            "message_clarity": 0.88,
            "emotional_impact": 0.75,
            "technical_issues": [],
            "recommendations": ["Consider adding captions for accessibility"],
            "overall_score": 0.83
        })


# =============================================================================
# MOCK VECTOR DB (For standalone operation without Qdrant)
# =============================================================================

class MockVectorDB:
    """Mock vector database for client DNA storage"""

    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}

    async def search(self, collection: str, query: str, filter: Dict = None, top_k: int = 10) -> List[Dict]:
        """Mock search - returns stored data"""
        key = f"{collection}:{filter.get('client_id', 'default')}" if filter else collection
        return self._store.get(key, [])

    async def upsert(self, collection: str, points: List[Dict]) -> None:
        """Mock upsert - stores data in memory"""
        for point in points:
            client_id = point.get('payload', {}).get('client_id', 'default')
            key = f"{collection}:{client_id}"
            if key not in self._store:
                self._store[key] = []
            self._store[key].append(point)


# =============================================================================
# AGENT 7.5: THE AUTEUR - Vision-Language Quality Assurance
# =============================================================================

@dataclass
class AuteurAnalysis:
    """Output from THE AUTEUR vision QA"""
    visual_quality_score: float  # 0-1
    brand_consistency: float  # 0-1
    message_clarity: float  # 0-1
    emotional_impact: float  # 0-1
    technical_issues: List[str]
    recommendations: List[str]
    overall_score: float  # 0-1
    approved: bool


class TheAuteur:
    """
    Agent 7.5: THE AUTEUR

    Vision-Language Quality Assurance using Claude's vision capabilities.
    Analyzes generated video frames for quality, consistency, and impact.
    """

    def __init__(self, llm_client=None, approval_threshold: float = 0.85):
        self.llm = llm_client or MockLLMClient()
        self.threshold = approval_threshold
        self.name = "THE_AUTEUR"

    async def analyze_video(
        self,
        video_frames: List[str],  # Base64 encoded frames
        brief: Dict[str, Any],
        brand_guidelines: Optional[Dict] = None
    ) -> AuteurAnalysis:
        """Analyze video quality against brief and brand guidelines"""

        with tracer.start_as_current_span("auteur_analyze") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            # Sample frames for analysis (every 5th frame, max 10)
            sampled_frames = video_frames[::5][:10] if video_frames else []

            prompt = f"""You are THE AUTEUR, a legendary video quality analyst.

## BRIEF
{json.dumps(brief, indent=2)}

## BRAND GUIDELINES
{json.dumps(brand_guidelines or {}, indent=2)}

## TASK
Analyze these {len(sampled_frames)} video frames and provide quality assessment.

Respond in JSON format with scores 0-1 for:
visual_quality_score, brand_consistency, message_clarity, emotional_impact,
technical_issues (list), recommendations (list), overall_score"""

            if sampled_frames and hasattr(self.llm, 'analyze_images'):
                response = await self.llm.analyze_images(
                    prompt=prompt,
                    images=sampled_frames,
                    model="claude-sonnet-4-20250514"
                )
            else:
                response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")

            result = json.loads(response)

            analysis = AuteurAnalysis(
                visual_quality_score=result.get('visual_quality_score', 0.8),
                brand_consistency=result.get('brand_consistency', 0.8),
                message_clarity=result.get('message_clarity', 0.8),
                emotional_impact=result.get('emotional_impact', 0.7),
                technical_issues=result.get('technical_issues', []),
                recommendations=result.get('recommendations', []),
                overall_score=result.get('overall_score', 0.8),
                approved=result.get('overall_score', 0.8) >= self.threshold
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("overall_score", analysis.overall_score)
            span.set_attribute("approved", analysis.approved)

            logger.info(f"[{self.name}] Analysis complete: score={analysis.overall_score:.2f}, approved={analysis.approved}")

            return analysis


# =============================================================================
# AGENT 8.5: THE GENETICIST - DSPy Prompt Evolution
# =============================================================================

@dataclass
class PromptEvolution:
    """Output from THE GENETICIST"""
    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    generation: int
    mutations_applied: List[str]


class TheGeneticist:
    """
    Agent 8.5: THE GENETICIST

    DSPy-inspired prompt optimization through evolutionary techniques.
    """

    def __init__(self, llm_client=None, generations: int = 3, population_size: int = 5):
        self.llm = llm_client or MockLLMClient()
        self.generations = generations
        self.population_size = population_size
        self.name = "THE_GENETICIST"

    async def evolve_prompt(
        self,
        base_prompt: str,
        task_description: str,
        evaluation_examples: List[Dict] = None
    ) -> PromptEvolution:
        """Evolve a prompt through genetic optimization"""

        with tracer.start_as_current_span("geneticist_evolve") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            # Initialize population with mutations
            population = await self._initialize_population(base_prompt)

            best_prompt = base_prompt
            best_score = 0.0
            all_mutations = []

            for gen in range(self.generations):
                scores = []
                for prompt in population:
                    score = await self._evaluate_fitness(prompt, task_description, evaluation_examples or [])
                    scores.append((prompt, score))

                scores.sort(key=lambda x: x[1], reverse=True)

                if scores[0][1] > best_score:
                    best_score = scores[0][1]
                    best_prompt = scores[0][0]

                survivors = [s[0] for s in scores[:self.population_size // 2]]
                population = await self._breed_population(survivors)
                all_mutations.append(f"Gen{gen+1}: crossover/mutation applied")

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            logger.info(f"[{self.name}] Evolution complete: improvement={best_score - 0.5:.2f}")

            return PromptEvolution(
                original_prompt=base_prompt,
                optimized_prompt=best_prompt,
                improvement_score=best_score - 0.5,
                generation=self.generations,
                mutations_applied=all_mutations
            )

    async def _initialize_population(self, base: str) -> List[str]:
        mutations = [
            f"Be more specific: {base}",
            f"Think step by step: {base}",
            f"Consider edge cases: {base}",
            f"{base}\n\nProvide detailed reasoning.",
            base
        ]
        return mutations[:self.population_size]

    async def _evaluate_fitness(self, prompt: str, task: str, examples: List[Dict]) -> float:
        score = 0.5
        if len(prompt) > 100:
            score += 0.1
        if "step" in prompt.lower():
            score += 0.1
        if any(word in prompt.lower() for word in ["because", "therefore", "thus"]):
            score += 0.1
        return min(score, 1.0)

    async def _breed_population(self, survivors: List[str]) -> List[str]:
        new_pop = survivors.copy()
        while len(new_pop) < self.population_size:
            if len(survivors) >= 2:
                p1, p2 = survivors[0], survivors[1]
                child = p1[:len(p1)//2] + " " + p2[len(p2)//2:]
                new_pop.append(child)
        return new_pop


# =============================================================================
# AGENT 11: THE ORACLE - Viral Prediction Engine
# =============================================================================

@dataclass
class ViralPrediction:
    """Output from THE ORACLE"""
    viral_score: float
    momentum: float
    peak_timing: str
    risk_factors: List[str]
    amplification_tips: List[str]
    confidence: float


class TheOracle:
    """
    Agent 11: THE ORACLE

    Predicts content virality using multi-signal analysis.
    """

    def __init__(self, llm_client=None, trend_api=None):
        self.llm = llm_client or MockLLMClient()
        self.trend_api = trend_api
        self.name = "THE_ORACLE"

    async def predict_virality(
        self,
        content: Dict[str, Any],
        target_platforms: List[str],
        target_audience: Dict[str, Any]
    ) -> ViralPrediction:
        """Predict viral potential of content"""

        with tracer.start_as_current_span("oracle_predict") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            trends = await self._get_trends(target_platforms) if self.trend_api else []

            prompt = f"""You are THE ORACLE, a legendary viral content predictor.

## CONTENT
{json.dumps(content, indent=2)}

## TARGET PLATFORMS
{', '.join(target_platforms)}

## TARGET AUDIENCE
{json.dumps(target_audience, indent=2)}

## CURRENT TRENDS
{json.dumps(trends, indent=2)}

Analyze and predict viral potential. Respond in JSON:
{{
    "viral_score": 0.0-1.0,
    "momentum": -1.0 to +1.0,
    "peak_timing": "e.g., Tuesday 10am EST",
    "risk_factors": ["risk1", "risk2"],
    "amplification_tips": ["tip1", "tip2"],
    "confidence": 0.0-1.0
}}"""

            response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")
            result = json.loads(response)

            prediction = ViralPrediction(
                viral_score=result.get('viral_score', 0.5),
                momentum=result.get('momentum', 0.0),
                peak_timing=result.get('peak_timing', 'Tuesday 10am EST'),
                risk_factors=result.get('risk_factors', []),
                amplification_tips=result.get('amplification_tips', []),
                confidence=result.get('confidence', 0.7)
            )

            VIRAL_SCORE.labels(content_type=content.get('type', 'video')).set(prediction.viral_score)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("viral_score", prediction.viral_score)
            logger.info(f"[{self.name}] Prediction: viral_score={prediction.viral_score:.2f}, confidence={prediction.confidence:.2f}")

            return prediction

    async def _get_trends(self, platforms: List[str]) -> List[Dict]:
        if not self.trend_api:
            return [{"topic": "AI", "momentum": 0.8}]
        return []


# =============================================================================
# AGENT 12: THE CHAMELEON - Platform Adapter
# =============================================================================

@dataclass
class PlatformAdaptation:
    """Output from THE CHAMELEON"""
    platform: str
    adapted_content: Dict[str, Any]
    format_specs: Dict[str, Any]
    hashtags: List[str]
    optimal_posting_time: str
    engagement_prediction: float


class TheChameleon:
    """
    Agent 12: THE CHAMELEON

    Adapts content for optimal performance on each platform.
    """

    PLATFORM_SPECS = {
        "youtube": {"aspect": "16:9", "max_duration": 3600, "min_duration": 15},
        "youtube_shorts": {"aspect": "9:16", "max_duration": 60, "min_duration": 15},
        "linkedin": {"aspect": "16:9", "max_duration": 600, "tone": "professional"},
        "instagram_reels": {"aspect": "9:16", "max_duration": 90, "min_duration": 15},
        "tiktok": {"aspect": "9:16", "max_duration": 180, "min_duration": 15},
        "twitter": {"aspect": "16:9", "max_duration": 140, "min_duration": 10},
        "facebook": {"aspect": "16:9", "max_duration": 240, "tone": "casual"}
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client or MockLLMClient()
        self.name = "THE_CHAMELEON"

    async def adapt_for_platform(
        self,
        content: Dict[str, Any],
        target_platform: str,
        brand_voice: Optional[Dict] = None
    ) -> PlatformAdaptation:
        """Adapt content for specific platform"""

        with tracer.start_as_current_span("chameleon_adapt") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            specs = self.PLATFORM_SPECS.get(target_platform, {})

            prompt = f"""You are THE CHAMELEON, a platform optimization expert.

## ORIGINAL CONTENT
{json.dumps(content, indent=2)}

## TARGET PLATFORM: {target_platform}
Specs: {json.dumps(specs, indent=2)}

## BRAND VOICE
{json.dumps(brand_voice or {}, indent=2)}

Adapt content for {target_platform}. Respond in JSON:
{{
    "adapted_content": {{"title": "...", "description": "...", "hook": "...", "cta": "..."}},
    "format_specs": {{"duration_seconds": 60, "aspect_ratio": "9:16"}},
    "hashtags": ["#tag1", "#tag2"],
    "optimal_posting_time": "Tuesday 2pm EST",
    "engagement_prediction": 0.0-1.0
}}"""

            response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")
            result = json.loads(response)

            adaptation = PlatformAdaptation(
                platform=target_platform,
                adapted_content=result.get('adapted_content', {}),
                format_specs=result.get('format_specs', {}),
                hashtags=result.get('hashtags', []),
                optimal_posting_time=result.get('optimal_posting_time', 'Tuesday 2pm EST'),
                engagement_prediction=result.get('engagement_prediction', 0.6)
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("platform", target_platform)
            logger.info(f"[{self.name}] Adapted for {target_platform}: engagement_pred={adaptation.engagement_prediction:.2f}")

            return adaptation


# =============================================================================
# AGENT 13: THE MEMORY - Client DNA Profiling
# =============================================================================

@dataclass
class ClientDNA:
    """Output from THE MEMORY"""
    client_id: str
    brand_essence: Dict[str, Any]
    preferred_styles: List[str]
    successful_patterns: List[Dict]
    avoid_patterns: List[Dict]
    voice_characteristics: Dict[str, Any]
    historical_performance: Dict[str, float]


class TheMemory:
    """
    Agent 13: THE MEMORY

    Maintains and recalls client preferences, history, and patterns.
    """

    def __init__(self, vector_db=None, llm_client=None):
        self.vector_db = vector_db or MockVectorDB()
        self.llm = llm_client or MockLLMClient()
        self.name = "THE_MEMORY"

    async def get_client_dna(self, client_id: str) -> ClientDNA:
        """Retrieve comprehensive client profile"""

        with tracer.start_as_current_span("memory_retrieve") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            client_data = await self.vector_db.search(
                collection="client_dna",
                query=client_id,
                filter={"client_id": client_id},
                top_k=50
            )

            prompt = f"""Analyze client historical data and create DNA profile:

{json.dumps(client_data, indent=2)}

Respond in JSON with: brand_essence, preferred_styles, successful_patterns,
avoid_patterns, voice_characteristics, historical_performance"""

            response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")
            result = json.loads(response)

            dna = ClientDNA(
                client_id=client_id,
                brand_essence=result.get('brand_essence', {}),
                preferred_styles=result.get('preferred_styles', []),
                successful_patterns=result.get('successful_patterns', []),
                avoid_patterns=result.get('avoid_patterns', []),
                voice_characteristics=result.get('voice_characteristics', {}),
                historical_performance=result.get('historical_performance', {})
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("client_id", client_id)
            logger.info(f"[{self.name}] Retrieved DNA for client: {client_id}")

            return dna

    async def update_client_dna(
        self,
        client_id: str,
        interaction: Dict[str, Any],
        performance: Dict[str, float]
    ) -> None:
        """Update client profile with new interaction data"""

        await self.vector_db.upsert(
            collection="client_dna",
            points=[{
                "id": f"{client_id}_{datetime.now().timestamp()}",
                "payload": {
                    "client_id": client_id,
                    "interaction": interaction,
                    "performance": performance,
                    "timestamp": datetime.now().isoformat()
                }
            }]
        )
        logger.info(f"[{self.name}] Updated DNA for client: {client_id}")


# =============================================================================
# AGENT 14: THE HUNTER - Trend Scouting
# =============================================================================

@dataclass
class TrendReport:
    """Output from THE HUNTER"""
    trends: List[Dict[str, Any]]
    emerging: List[Dict[str, Any]]
    declining: List[Dict[str, Any]]
    opportunities: List[str]
    risks: List[str]
    recommended_topics: List[str]


class TheHunter:
    """
    Agent 14: THE HUNTER

    Scouts and analyzes trends across multiple sources.
    """

    def __init__(self, llm_client=None, trend_apis: Dict[str, Any] = None):
        self.llm = llm_client or MockLLMClient()
        self.trend_apis = trend_apis or {}
        self.name = "THE_HUNTER"

    async def hunt_trends(
        self,
        industry: str,
        keywords: List[str],
        lookback_days: int = 7
    ) -> TrendReport:
        """Hunt for relevant trends"""

        with tracer.start_as_current_span("hunter_scout") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            raw_trends = await self._gather_trends(industry, keywords, lookback_days)

            prompt = f"""You are THE HUNTER, a legendary trend scout.

## INDUSTRY: {industry}
## KEYWORDS: {', '.join(keywords)}
## LOOKBACK: {lookback_days} days

## RAW TREND DATA
{json.dumps(raw_trends, indent=2)}

Analyze trends. Respond in JSON:
{{
    "trends": [{{"topic": "...", "momentum": 0.8, "relevance": 0.9}}],
    "emerging": [{{"topic": "...", "growth_rate": 0.5}}],
    "declining": [{{"topic": "...", "decline_rate": -0.3}}],
    "opportunities": ["opportunity1"],
    "risks": ["risk1"],
    "recommended_topics": ["topic1"]
}}"""

            response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")
            result = json.loads(response)

            report = TrendReport(
                trends=result.get('trends', []),
                emerging=result.get('emerging', []),
                declining=result.get('declining', []),
                opportunities=result.get('opportunities', []),
                risks=result.get('risks', []),
                recommended_topics=result.get('recommended_topics', [])
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("industry", industry)
            logger.info(f"[{self.name}] Hunted {len(report.trends)} trends for {industry}")

            return report

    async def _gather_trends(self, industry: str, keywords: List[str], days: int) -> Dict[str, Any]:
        return {
            "social_trends": [{"topic": f"{industry} AI", "volume": 10000}],
            "search_trends": [{"keyword": f"{industry} automation", "growth": 0.4}],
            "news_mentions": [{"topic": "AI video", "sentiment": 0.7}]
        }


# =============================================================================
# AGENT 15: THE ACCOUNTANT - Budget Optimization
# =============================================================================

@dataclass
class BudgetOptimization:
    """Output from THE ACCOUNTANT"""
    total_budget: float
    allocation: Dict[str, float]
    expected_roi: float
    cost_per_result: Dict[str, float]
    savings_opportunities: List[str]
    risk_assessment: Dict[str, Any]


class TheAccountant:
    """
    Agent 15: THE ACCOUNTANT

    Optimizes budget allocation for maximum ROI.
    """

    # Internal costs - NEVER expose to clients
    COST_RATES = {
        "video_generation": 2.60,
        "voiceover_elevenlabs": 0.30,
        "music_license": 5.00,
        "storage_per_gb": 0.02,
        "api_calls_claude": 0.003,
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client or MockLLMClient()
        self.name = "THE_ACCOUNTANT"

    async def optimize_budget(
        self,
        project: Dict[str, Any],
        constraints: Dict[str, float],
        goals: List[str]
    ) -> BudgetOptimization:
        """Optimize budget allocation for project"""

        with tracer.start_as_current_span("accountant_optimize") as span:
            LEGENDARY_AGENT_CALLS.labels(agent_name=self.name).inc()
            start_time = datetime.now()

            prompt = f"""You are THE ACCOUNTANT, a budget optimization expert.

## PROJECT
{json.dumps(project, indent=2)}

## CONSTRAINTS
{json.dumps(constraints, indent=2)}

## GOALS
{json.dumps(goals, indent=2)}

Optimize budget. Respond in JSON:
{{
    "total_budget": 1000.00,
    "allocation": {{"video_production": 500, "voiceover": 150, "music": 50, "platform_ads": 200, "contingency": 100}},
    "expected_roi": 3.5,
    "cost_per_result": {{"cost_per_view": 0.02, "cost_per_engagement": 0.15, "cost_per_lead": 5.00}},
    "savings_opportunities": ["batch processing"],
    "risk_assessment": {{"budget_risk": "low", "timeline_risk": "medium"}}
}}"""

            response = await self.llm.generate(prompt=prompt, model="claude-sonnet-4-20250514")
            result = json.loads(response)

            optimization = BudgetOptimization(
                total_budget=result.get('total_budget', 1000.0),
                allocation=result.get('allocation', {}),
                expected_roi=result.get('expected_roi', 2.5),
                cost_per_result=result.get('cost_per_result', {}),
                savings_opportunities=result.get('savings_opportunities', []),
                risk_assessment=result.get('risk_assessment', {})
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            LEGENDARY_AGENT_LATENCY.labels(agent_name=self.name).observe(latency_ms)

            span.set_attribute("expected_roi", optimization.expected_roi)
            logger.info(f"[{self.name}] Optimized budget: total=${optimization.total_budget:.2f}, ROI={optimization.expected_roi:.1f}x")

            return optimization


# =============================================================================
# LEGENDARY COORDINATOR
# =============================================================================

class LegendaryCoordinator:
    """
    Coordinates all legendary agents in the RAGNAROK pipeline.
    """

    def __init__(
        self,
        llm_client=None,
        vector_db=None,
        trend_api=None
    ):
        self.auteur = TheAuteur(llm_client)
        self.geneticist = TheGeneticist(llm_client)
        self.oracle = TheOracle(llm_client, trend_api)
        self.chameleon = TheChameleon(llm_client)
        self.memory = TheMemory(vector_db, llm_client)
        self.hunter = TheHunter(llm_client, trend_api)
        self.accountant = TheAccountant(llm_client)

        logger.info("LegendaryCoordinator initialized with all 7 legendary agents")

    async def pre_production_enhancement(
        self,
        brief: Dict[str, Any],
        client_id: str
    ) -> Dict[str, Any]:
        """Run legendary agents before production"""

        logger.info(f"[LegendaryCoordinator] Starting pre-production enhancement for {client_id}")
        results = {}

        # Run agents in parallel where possible
        tasks = [
            self.memory.get_client_dna(client_id),
            self.hunter.hunt_trends(
                industry=brief.get('industry', 'general'),
                keywords=brief.get('keywords', [])
            ),
            self.oracle.predict_virality(
                content=brief,
                target_platforms=brief.get('platforms', ['youtube']),
                target_audience=brief.get('audience', {})
            ),
            self.accountant.optimize_budget(
                project=brief,
                constraints=brief.get('budget_constraints', {}),
                goals=brief.get('goals', [])
            )
        ]

        client_dna, trends, viral_prediction, budget = await asyncio.gather(*tasks)

        results['client_dna'] = {
            'client_id': client_dna.client_id,
            'brand_essence': client_dna.brand_essence,
            'preferred_styles': client_dna.preferred_styles,
            'voice_characteristics': client_dna.voice_characteristics
        }
        results['trends'] = {
            'current': [t for t in trends.trends],
            'emerging': [t for t in trends.emerging],
            'opportunities': trends.opportunities,
            'recommended_topics': trends.recommended_topics
        }
        results['viral_prediction'] = {
            'score': viral_prediction.viral_score,
            'momentum': viral_prediction.momentum,
            'peak_timing': viral_prediction.peak_timing,
            'amplification_tips': viral_prediction.amplification_tips,
            'confidence': viral_prediction.confidence
        }
        results['budget'] = {
            'total': budget.total_budget,
            'allocation': budget.allocation,
            'expected_roi': budget.expected_roi,
            'savings_opportunities': budget.savings_opportunities
        }

        logger.info(f"[LegendaryCoordinator] Pre-production complete: viral_score={viral_prediction.viral_score:.2f}")

        return results

    async def post_production_enhancement(
        self,
        video_frames: List[str],
        brief: Dict[str, Any],
        target_platforms: List[str]
    ) -> Dict[str, Any]:
        """Run legendary agents after production"""

        logger.info("[LegendaryCoordinator] Starting post-production enhancement")
        results = {}

        # Quality assurance
        qa = await self.auteur.analyze_video(
            video_frames=video_frames,
            brief=brief
        )
        results['qa'] = {
            'overall_score': qa.overall_score,
            'approved': qa.approved,
            'visual_quality': qa.visual_quality_score,
            'brand_consistency': qa.brand_consistency,
            'message_clarity': qa.message_clarity,
            'recommendations': qa.recommendations
        }

        # Platform adaptations
        results['adaptations'] = {}
        for platform in target_platforms:
            adaptation = await self.chameleon.adapt_for_platform(
                content=brief,
                target_platform=platform
            )
            results['adaptations'][platform] = {
                'adapted_content': adaptation.adapted_content,
                'format_specs': adaptation.format_specs,
                'hashtags': adaptation.hashtags,
                'optimal_posting_time': adaptation.optimal_posting_time,
                'engagement_prediction': adaptation.engagement_prediction
            }

        logger.info(f"[LegendaryCoordinator] Post-production complete: qa_score={qa.overall_score:.2f}")

        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_legendary_coordinator(
    anthropic_client=None,
    qdrant_client=None,
    trend_api=None
) -> LegendaryCoordinator:
    """Factory function to create LegendaryCoordinator with available clients"""

    return LegendaryCoordinator(
        llm_client=anthropic_client,
        vector_db=qdrant_client,
        trend_api=trend_api
    )


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAGNAROK LEGENDARY AGENTS v1.0")
    print("=" * 60)
    print("\nAgents:")
    print("  7.5  THE AUTEUR      - Vision-Language QA")
    print("  8.5  THE GENETICIST  - DSPy Prompt Evolution")
    print("  11   THE ORACLE      - Viral Prediction")
    print("  12   THE CHAMELEON   - Platform Adapter")
    print("  13   THE MEMORY      - Client DNA Profiling")
    print("  14   THE HUNTER      - Trend Scouting")
    print("  15   THE ACCOUNTANT  - Budget Optimization")
    print("=" * 60)
    print("\nLegendaryCoordinator orchestrates all agents for:")
    print("  - Pre-production enhancement")
    print("  - Post-production QA and optimization")
    print("=" * 60)
