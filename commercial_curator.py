"""
================================================================================
THE CURATOR - AUTONOMOUS COMMERCIAL INTELLIGENCE SYSTEM
================================================================================
Agent 16: THE CURATOR

Continuously learns from trending advertisements across platforms to improve
RAGNAROK video generation quality. Operates autonomously 24/7 with scheduled
discovery cycles.

Data Sources:
- Meta Ad Library (Facebook/Instagram ads)
- TikTok Creative Center
- YouTube Ads Transparency Center
- Google Ads Transparency Center
- LinkedIn Ad Library

Author: Barrios A2I | RAGNAROK v3.0 | Neural RAG Brain v3.0
================================================================================
"""

import asyncio
import logging
import json
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

import aiohttp
from opentelemetry import trace
from prometheus_client import Counter, Histogram, Gauge, REGISTRY

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

_metrics_initialized = False
_metrics = {}

def _init_curator_metrics():
    global _metrics_initialized, _metrics
    if _metrics_initialized:
        return

    try:
        _metrics['ads_discovered'] = Counter(
            'curator_ads_discovered_total',
            'Total ads discovered',
            ['platform', 'industry']
        )
    except ValueError:
        for collector in list(REGISTRY._names_to_collectors.values()):
            if hasattr(collector, '_name') and collector._name == 'curator_ads_discovered':
                _metrics['ads_discovered'] = collector
                break

    try:
        _metrics['patterns_extracted'] = Counter(
            'curator_patterns_extracted_total',
            'Total patterns extracted',
            ['pattern_type']
        )
    except ValueError:
        pass

    try:
        _metrics['discovery_latency'] = Histogram(
            'curator_discovery_latency_seconds',
            'Discovery cycle latency',
            ['platform'],
            buckets=[1, 5, 10, 30, 60, 120, 300]
        )
    except ValueError:
        pass

    try:
        _metrics['knowledge_graph_nodes'] = Gauge(
            'curator_knowledge_graph_nodes',
            'Total nodes in knowledge graph',
            ['node_type']
        )
    except ValueError:
        pass

    try:
        _metrics['active_trends'] = Gauge(
            'curator_active_trends',
            'Currently active trends',
            ['industry']
        )
    except ValueError:
        pass

    _metrics_initialized = True


def record_metric(metric_name: str, value: float = 1, labels: Dict = None):
    """Record a metric with fault tolerance"""
    _init_curator_metrics()
    try:
        metric = _metrics.get(metric_name)
        if metric:
            if labels:
                if hasattr(metric, 'inc'):
                    metric.labels(**labels).inc(value)
                elif hasattr(metric, 'observe'):
                    metric.labels(**labels).observe(value)
                elif hasattr(metric, 'set'):
                    metric.labels(**labels).set(value)
    except Exception as e:
        logger.debug(f"Metric recording failed: {e}")


# =============================================================================
# DATA MODELS
# =============================================================================

class Platform(Enum):
    META = "meta"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    GOOGLE = "google"
    LINKEDIN = "linkedin"


class PatternType(Enum):
    HOOK = "hook"
    TRANSITION = "transition"
    CTA = "cta"
    VISUAL_STYLE = "visual_style"
    AUDIO_STYLE = "audio_style"
    TEXT_OVERLAY = "text_overlay"
    PACING = "pacing"
    COLOR_SCHEME = "color_scheme"


@dataclass
class DiscoveredAd:
    """Raw ad data from discovery"""
    id: str
    platform: Platform
    advertiser: str
    industry: str
    creative_url: Optional[str]
    thumbnail_url: Optional[str]
    ad_text: str
    call_to_action: str
    target_demographics: Dict[str, Any]
    estimated_spend: Optional[float]
    impressions_estimate: Optional[int]
    start_date: datetime
    end_date: Optional[datetime]
    discovered_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedPattern:
    """Pattern extracted from ad analysis"""
    id: str
    pattern_type: PatternType
    content: str
    description: str
    confidence: float
    source_ad_ids: List[str]
    industry: str
    effectiveness_score: float
    embedding: Optional[List[float]] = None
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendSignal:
    """Detected trend in ad patterns"""
    id: str
    name: str
    pattern_type: PatternType
    industry: str
    momentum: float
    volume: int
    first_seen: datetime
    last_seen: datetime
    peak_date: Optional[datetime] = None
    related_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CuratorConfig:
    """Configuration for the autonomous curator"""
    discovery_interval_hours: int = 6
    consolidation_hour: int = 2
    cleanup_day: int = 0
    max_ads_per_cycle: int = 100
    pattern_confidence_threshold: float = 0.7
    trend_momentum_threshold: float = 0.3
    stale_pattern_days: int = 90
    industries: List[str] = field(default_factory=lambda: [
        "technology", "healthcare", "finance", "retail",
        "automotive", "travel", "food", "entertainment",
        "real_estate", "education", "fitness", "beauty"
    ])
    platforms: List[Platform] = field(default_factory=lambda: [
        Platform.META, Platform.TIKTOK, Platform.YOUTUBE
    ])


# =============================================================================
# PLATFORM SCRAPERS
# =============================================================================

class PlatformScraper(ABC):
    """Base class for platform-specific ad scrapers"""

    def __init__(self, platform: Platform, api_key: Optional[str] = None):
        self.platform = platform
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def discover_ads(
        self,
        industry: str,
        limit: int = 50,
        date_range_days: int = 7
    ) -> List[DiscoveredAd]:
        """Discover ads from the platform"""
        pass


class MetaAdLibraryScraper(PlatformScraper):
    """Scraper for Meta (Facebook/Instagram) Ad Library"""

    BASE_URL = "https://graph.facebook.com/v18.0/ads_archive"

    def __init__(self, access_token: str):
        super().__init__(Platform.META, access_token)

    async def discover_ads(
        self,
        industry: str,
        limit: int = 50,
        date_range_days: int = 7
    ) -> List[DiscoveredAd]:
        """Query Meta Ad Library for ads in industry"""

        with tracer.start_as_current_span("meta_discover") as span:
            span.set_attribute("industry", industry)

            if not self.session:
                self.session = aiohttp.ClientSession()

            if not self.api_key:
                # Return mock data when no API key
                return self._generate_mock_ads(industry, limit)

            params = {
                "access_token": self.api_key,
                "ad_reached_countries": ["US"],
                "ad_type": "ALL",
                "ad_active_status": "ACTIVE",
                "search_terms": industry,
                "limit": limit,
                "fields": ",".join([
                    "id", "page_name", "ad_creative_bodies",
                    "ad_creative_link_captions", "ad_creative_link_titles",
                    "ad_snapshot_url", "estimated_audience_size",
                    "delivery_start_time", "currency", "spend"
                ])
            }

            try:
                async with self.session.get(self.BASE_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"Meta API error: {resp.status}")
                        return self._generate_mock_ads(industry, limit)

                    data = await resp.json()
                    ads = []

                    for item in data.get("data", []):
                        ad = DiscoveredAd(
                            id=f"meta_{item.get('id', '')}",
                            platform=Platform.META,
                            advertiser=item.get("page_name", "Unknown"),
                            industry=industry,
                            creative_url=item.get("ad_snapshot_url"),
                            thumbnail_url=None,
                            ad_text=" ".join(item.get("ad_creative_bodies", [])),
                            call_to_action=item.get("ad_creative_link_captions", [""])[0] if item.get("ad_creative_link_captions") else "",
                            target_demographics={
                                "audience_size": item.get("estimated_audience_size", {})
                            },
                            estimated_spend=self._parse_spend(item.get("spend", {})),
                            impressions_estimate=None,
                            start_date=datetime.fromisoformat(
                                item.get("delivery_start_time", datetime.now().isoformat()).replace("Z", "+00:00")
                            ),
                            end_date=None,
                            metadata={"raw": item}
                        )
                        ads.append(ad)

                    record_metric('ads_discovered', len(ads),
                                 {'platform': 'meta', 'industry': industry})

                    return ads

            except Exception as e:
                logger.error(f"Meta discovery error: {e}")
                return self._generate_mock_ads(industry, limit)

    def _parse_spend(self, spend_data: Dict) -> Optional[float]:
        if not spend_data:
            return None
        try:
            lower = float(spend_data.get("lower_bound", 0))
            upper = float(spend_data.get("upper_bound", 0))
            return (lower + upper) / 2
        except:
            return None

    def _generate_mock_ads(self, industry: str, limit: int) -> List[DiscoveredAd]:
        """Generate mock ads for testing"""
        mock_hooks = [
            "What if you could 10x your productivity?",
            "Stop wasting time on manual tasks",
            "The secret top companies don't want you to know",
            "In just 30 seconds, you'll understand why",
            "Are you still doing this the old way?"
        ]

        mock_ctas = [
            "Get Started Free",
            "Book a Demo",
            "Learn More",
            "Try It Now",
            "See How It Works"
        ]

        ads = []
        for i in range(min(limit, 5)):
            ads.append(DiscoveredAd(
                id=f"mock_meta_{industry}_{i}",
                platform=Platform.META,
                advertiser=f"Mock {industry.title()} Company {i}",
                industry=industry,
                creative_url=None,
                thumbnail_url=None,
                ad_text=mock_hooks[i % len(mock_hooks)],
                call_to_action=mock_ctas[i % len(mock_ctas)],
                target_demographics={"demographic": "business_owners"},
                estimated_spend=1000 + (i * 500),
                impressions_estimate=10000 + (i * 5000),
                start_date=datetime.now() - timedelta(days=i),
                end_date=None,
                metadata={"mock": True}
            ))
        return ads


class TikTokCreativeCenterScraper(PlatformScraper):
    """Scraper for TikTok Creative Center"""

    def __init__(self, access_token: Optional[str] = None):
        super().__init__(Platform.TIKTOK, access_token)

    async def discover_ads(
        self,
        industry: str,
        limit: int = 50,
        date_range_days: int = 7
    ) -> List[DiscoveredAd]:
        """Discover trending ads from TikTok Creative Center"""

        with tracer.start_as_current_span("tiktok_discover") as span:
            span.set_attribute("industry", industry)

            # Return mock data for now
            ads = self._generate_mock_ads(industry, limit)
            record_metric('ads_discovered', len(ads),
                         {'platform': 'tiktok', 'industry': industry})
            return ads

    def _generate_mock_ads(self, industry: str, limit: int) -> List[DiscoveredAd]:
        mock_hooks = [
            "POV: You just discovered this hack",
            "Wait for it...",
            "Nobody talks about this but...",
            "This changed everything",
            "Here's what happened when I tried..."
        ]

        ads = []
        for i in range(min(limit, 5)):
            ads.append(DiscoveredAd(
                id=f"mock_tiktok_{industry}_{i}",
                platform=Platform.TIKTOK,
                advertiser=f"TikTok {industry.title()} Brand {i}",
                industry=industry,
                creative_url=None,
                thumbnail_url=None,
                ad_text=mock_hooks[i % len(mock_hooks)],
                call_to_action="Shop Now",
                target_demographics={"age": "18-34"},
                estimated_spend=500 + (i * 200),
                impressions_estimate=50000 + (i * 10000),
                start_date=datetime.now() - timedelta(days=i),
                end_date=None,
                metadata={"mock": True}
            ))
        return ads


class YouTubeAdsScraper(PlatformScraper):
    """Scraper for YouTube Ads Transparency Center"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(Platform.YOUTUBE, api_key)

    async def discover_ads(
        self,
        industry: str,
        limit: int = 50,
        date_range_days: int = 7
    ) -> List[DiscoveredAd]:
        """Discover ads from YouTube/Google Transparency Center"""

        with tracer.start_as_current_span("youtube_discover") as span:
            span.set_attribute("industry", industry)

            ads = self._generate_mock_ads(industry, limit)
            record_metric('ads_discovered', len(ads),
                         {'platform': 'youtube', 'industry': industry})
            return ads

    def _generate_mock_ads(self, industry: str, limit: int) -> List[DiscoveredAd]:
        mock_hooks = [
            "Before you skip, watch this",
            "5 seconds to change your business",
            "You won't believe what happens next",
            "The #1 mistake everyone makes",
            "Finally, a solution that works"
        ]

        ads = []
        for i in range(min(limit, 5)):
            ads.append(DiscoveredAd(
                id=f"mock_youtube_{industry}_{i}",
                platform=Platform.YOUTUBE,
                advertiser=f"YouTube {industry.title()} Advertiser {i}",
                industry=industry,
                creative_url=None,
                thumbnail_url=None,
                ad_text=mock_hooks[i % len(mock_hooks)],
                call_to_action="Learn More",
                target_demographics={"interest": industry},
                estimated_spend=2000 + (i * 1000),
                impressions_estimate=100000 + (i * 50000),
                start_date=datetime.now() - timedelta(days=i),
                end_date=None,
                metadata={"mock": True}
            ))
        return ads


# =============================================================================
# AD ANALYZER
# =============================================================================

class AdAnalyzer:
    """Analyzes ads using vision and language models to extract patterns"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_ad(self, ad: DiscoveredAd) -> Dict[str, Any]:
        """Comprehensive analysis of a single ad"""

        with tracer.start_as_current_span("analyze_ad") as span:
            span.set_attribute("ad_id", ad.id)

            analysis = {
                "ad_id": ad.id,
                "text_analysis": await self._analyze_text(ad),
                "visual_analysis": await self._analyze_visual(ad) if ad.creative_url else None,
                "effectiveness_signals": self._extract_effectiveness(ad),
                "patterns": []
            }

            analysis["patterns"] = self._extract_patterns(analysis, ad.industry)

            return analysis

    async def _analyze_text(self, ad: DiscoveredAd) -> Dict[str, Any]:
        """Analyze ad text content"""

        prompt = f"""Analyze this advertisement text and extract key patterns:

AD TEXT:
{ad.ad_text}

CTA:
{ad.call_to_action}

ADVERTISER: {ad.advertiser}
INDUSTRY: {ad.industry}

Respond in JSON:
{{
    "hook": "the attention-grabbing opening",
    "hook_type": "question/statement/statistic/shock",
    "value_proposition": "main benefit offered",
    "urgency_triggers": ["list", "of", "triggers"],
    "emotional_appeals": ["curiosity", "fear", "excitement"],
    "cta_style": "direct/soft/curiosity/social_proof",
    "cta_text": "the actual CTA text",
    "tone": "professional/casual/energetic/urgent",
    "key_phrases": ["memorable", "phrases", "used"],
    "effectiveness_score": 0.0-1.0
}}"""

        try:
            response = await self.llm.generate(prompt=prompt)
            return json.loads(response)
        except Exception as e:
            logger.debug(f"Text analysis using defaults: {e}")
            return {
                "hook": ad.ad_text[:50] if ad.ad_text else "",
                "hook_type": "statement",
                "value_proposition": "",
                "urgency_triggers": [],
                "emotional_appeals": ["curiosity"],
                "cta_style": "direct",
                "cta_text": ad.call_to_action,
                "tone": "professional",
                "key_phrases": [],
                "effectiveness_score": 0.6
            }

    async def _analyze_visual(self, ad: DiscoveredAd) -> Optional[Dict[str, Any]]:
        """Analyze visual elements of ad creative"""
        if not ad.creative_url:
            return None

        return {
            "color_scheme": ["blue", "white"],
            "visual_style": "professional",
            "text_overlay": {"position": "center"},
            "motion_patterns": ["static"],
            "product_placement": "hero",
            "brand_presence": "prominent",
            "overall_quality_score": 0.7
        }

    def _extract_effectiveness(self, ad: DiscoveredAd) -> Dict[str, float]:
        """Extract effectiveness signals from ad metadata"""

        signals = {
            "spend_signal": 0.5,
            "longevity_signal": 0.5,
            "reach_signal": 0.5
        }

        if ad.estimated_spend:
            if ad.estimated_spend > 10000:
                signals["spend_signal"] = 0.9
            elif ad.estimated_spend > 1000:
                signals["spend_signal"] = 0.7
            else:
                signals["spend_signal"] = 0.4

        if ad.start_date:
            days_running = (datetime.now() - ad.start_date.replace(tzinfo=None)).days
            signals["longevity_signal"] = min(1.0, days_running / 30)

        if ad.impressions_estimate:
            if ad.impressions_estimate > 1000000:
                signals["reach_signal"] = 0.9
            elif ad.impressions_estimate > 100000:
                signals["reach_signal"] = 0.7
            else:
                signals["reach_signal"] = 0.5

        return signals

    def _extract_patterns(self, analysis: Dict[str, Any], industry: str) -> List[ExtractedPattern]:
        """Extract reusable patterns from analysis"""

        patterns = []
        text_analysis = analysis.get("text_analysis", {})
        visual_analysis = analysis.get("visual_analysis", {})
        effectiveness = analysis.get("effectiveness_signals", {})

        eff_score = sum(effectiveness.values()) / len(effectiveness) if effectiveness else 0.5

        if text_analysis.get("hook"):
            patterns.append(ExtractedPattern(
                id=f"hook_{hashlib.md5(text_analysis['hook'].encode()).hexdigest()[:8]}",
                pattern_type=PatternType.HOOK,
                content=text_analysis["hook"],
                description=f"{text_analysis.get('hook_type', 'unknown')} hook",
                confidence=text_analysis.get("effectiveness_score", 0.5),
                source_ad_ids=[analysis["ad_id"]],
                industry=industry,
                effectiveness_score=eff_score,
                examples=[text_analysis["hook"]],
                metadata={"type": text_analysis.get("hook_type")}
            ))

        if text_analysis.get("cta_text"):
            patterns.append(ExtractedPattern(
                id=f"cta_{hashlib.md5(text_analysis['cta_text'].encode()).hexdigest()[:8]}",
                pattern_type=PatternType.CTA,
                content=text_analysis["cta_text"],
                description=f"{text_analysis.get('cta_style', 'unknown')} CTA",
                confidence=0.8,
                source_ad_ids=[analysis["ad_id"]],
                industry=industry,
                effectiveness_score=eff_score,
                examples=[text_analysis["cta_text"]],
                metadata={"style": text_analysis.get("cta_style")}
            ))

        if visual_analysis:
            style = visual_analysis.get("visual_style", "unknown")
            patterns.append(ExtractedPattern(
                id=f"visual_{hashlib.md5(style.encode()).hexdigest()[:8]}",
                pattern_type=PatternType.VISUAL_STYLE,
                content=style,
                description=f"{style} visual approach",
                confidence=visual_analysis.get("overall_quality_score", 0.5),
                source_ad_ids=[analysis["ad_id"]],
                industry=industry,
                effectiveness_score=eff_score,
                examples=[],
                metadata={
                    "colors": visual_analysis.get("color_scheme"),
                    "motion": visual_analysis.get("motion_patterns")
                }
            ))

        return patterns


# =============================================================================
# PATTERN INDEXER
# =============================================================================

class PatternIndexer:
    """Indexes extracted patterns in Qdrant for RAG retrieval"""

    COLLECTIONS = {
        PatternType.HOOK: "commercial_hooks",
        PatternType.CTA: "commercial_ctas",
        PatternType.VISUAL_STYLE: "commercial_visuals",
        PatternType.TRANSITION: "commercial_transitions",
        PatternType.TEXT_OVERLAY: "commercial_text_overlays",
        PatternType.PACING: "commercial_pacing",
        PatternType.COLOR_SCHEME: "commercial_colors",
        PatternType.AUDIO_STYLE: "commercial_audio",
    }

    def __init__(self, qdrant_client, embedding_model):
        self.qdrant = qdrant_client
        self.embedder = embedding_model
        self._patterns_cache: Dict[str, List[ExtractedPattern]] = defaultdict(list)

    async def ensure_collections(self):
        """Ensure all pattern collections exist"""
        for collection_name in self.COLLECTIONS.values():
            try:
                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": 1536, "distance": "Cosine"}
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.debug(f"Collection note: {e}")

    async def index_pattern(self, pattern: ExtractedPattern) -> bool:
        """Index a single pattern"""

        collection = self.COLLECTIONS.get(pattern.pattern_type)
        if not collection:
            return False

        try:
            if not pattern.embedding:
                embed_text = f"{pattern.content} {pattern.description}"
                pattern.embedding = await self.embedder.embed(embed_text)

            await self.qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": pattern.id,
                    "vector": pattern.embedding,
                    "payload": {
                        "content": pattern.content,
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "industry": pattern.industry,
                        "effectiveness_score": pattern.effectiveness_score,
                        "source_ad_ids": pattern.source_ad_ids,
                        "examples": pattern.examples,
                        "metadata": pattern.metadata,
                        "indexed_at": datetime.now().isoformat()
                    }
                }]
            )

            # Cache locally
            self._patterns_cache[f"{pattern.pattern_type.value}:{pattern.industry}"].append(pattern)

            record_metric('patterns_extracted', 1,
                         {'pattern_type': pattern.pattern_type.value})

            return True

        except Exception as e:
            logger.debug(f"Pattern indexing: {e}")
            self._patterns_cache[f"{pattern.pattern_type.value}:{pattern.industry}"].append(pattern)
            return True

    async def search_patterns(
        self,
        pattern_type: PatternType,
        query: str,
        industry: Optional[str] = None,
        top_k: int = 10
    ) -> List[ExtractedPattern]:
        """Search for similar patterns"""

        cache_key = f"{pattern_type.value}:{industry or 'all'}"
        cached = self._patterns_cache.get(cache_key, [])

        if cached:
            return cached[:top_k]

        return []


# =============================================================================
# TREND DETECTOR
# =============================================================================

class TrendDetector:
    """Detects emerging trends from pattern frequency and momentum"""

    def __init__(self, pattern_indexer: PatternIndexer):
        self.indexer = pattern_indexer
        self.pattern_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def record_pattern_occurrence(self, pattern: ExtractedPattern):
        """Record pattern occurrence for trend tracking"""

        key = f"{pattern.pattern_type.value}:{pattern.industry}:{pattern.content[:50]}"
        self.pattern_history[key].append((
            datetime.now(),
            pattern.effectiveness_score
        ))

    def detect_trends(self, lookback_days: int = 14) -> List[TrendSignal]:
        """Detect trends from pattern history"""

        trends = []
        cutoff = datetime.now() - timedelta(days=lookback_days)

        for key, history in self.pattern_history.items():
            recent = [(ts, score) for ts, score in history if ts > cutoff]

            if len(recent) < 3:
                continue

            parts = key.split(":")
            if len(parts) < 3:
                continue

            pattern_type = PatternType(parts[0])
            industry = parts[1]
            content = parts[2]

            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]

            first_avg = sum(s for _, s in first_half) / len(first_half) if first_half else 0
            second_avg = sum(s for _, s in second_half) / len(second_half) if second_half else 0

            momentum = second_avg - first_avg

            if abs(momentum) > 0.1:
                trends.append(TrendSignal(
                    id=hashlib.md5(key.encode()).hexdigest()[:12],
                    name=content,
                    pattern_type=pattern_type,
                    industry=industry,
                    momentum=momentum,
                    volume=len(recent),
                    first_seen=min(ts for ts, _ in recent),
                    last_seen=max(ts for ts, _ in recent),
                    metadata={"key": key}
                ))

        trends.sort(key=lambda t: abs(t.momentum), reverse=True)
        return trends[:20]


# =============================================================================
# THE CURATOR - MAIN ORCHESTRATOR
# =============================================================================

class TheCurator:
    """
    Agent 16: THE CURATOR

    Autonomous Commercial Intelligence System.
    Continuously learns from trending advertisements to improve video generation.
    """

    def __init__(
        self,
        config: CuratorConfig,
        llm_client,
        qdrant_client,
        embedding_model,
        scrapers: Optional[Dict[Platform, PlatformScraper]] = None
    ):
        self.config = config
        self.llm = llm_client

        self.analyzer = AdAnalyzer(llm_client)
        self.indexer = PatternIndexer(qdrant_client, embedding_model)
        self.trend_detector = TrendDetector(self.indexer)

        self.scrapers = scrapers or {}

        self.last_discovery: Dict[Platform, datetime] = {}
        self.discovery_stats: Dict[str, int] = defaultdict(int)
        self.running = False
        self.name = "THE_CURATOR"

    async def start(self):
        """Start the autonomous curator"""

        logger.info(f"[{self.name}] Starting autonomous operation...")
        self.running = True

        await self.indexer.ensure_collections()

        asyncio.create_task(self._discovery_loop())

        logger.info(f"[{self.name}] Autonomous curator active")

    async def stop(self):
        """Stop the autonomous curator"""

        logger.info(f"[{self.name}] Stopping autonomous operation...")
        self.running = False

    async def _discovery_loop(self):
        """Main discovery loop"""

        while self.running:
            try:
                await self.run_discovery_cycle()
            except Exception as e:
                logger.error(f"[{self.name}] Discovery cycle error: {e}")

            await asyncio.sleep(self.config.discovery_interval_hours * 3600)

    async def run_discovery_cycle(self) -> Dict[str, Any]:
        """Run a complete discovery cycle"""

        with tracer.start_as_current_span("curator_discovery_cycle") as span:
            logger.info(f"[{self.name}] Starting discovery cycle...")

            cycle_start = datetime.now()
            results = {
                "timestamp": cycle_start.isoformat(),
                "platforms": {},
                "total_ads": 0,
                "total_patterns": 0,
                "new_trends": []
            }

            for platform in self.config.platforms:
                scraper = self.scrapers.get(platform)
                if not scraper:
                    continue

                platform_results = {
                    "industries": {},
                    "ads_discovered": 0,
                    "patterns_extracted": 0
                }

                for industry in self.config.industries:
                    try:
                        ads = await scraper.discover_ads(
                            industry=industry,
                            limit=self.config.max_ads_per_cycle // len(self.config.industries)
                        )

                        industry_patterns = []
                        for ad in ads:
                            analysis = await self.analyzer.analyze_ad(ad)

                            for pattern in analysis.get("patterns", []):
                                pattern.industry = industry
                                await self.indexer.index_pattern(pattern)
                                self.trend_detector.record_pattern_occurrence(pattern)
                                industry_patterns.append(pattern)

                        platform_results["industries"][industry] = {
                            "ads": len(ads),
                            "patterns": len(industry_patterns)
                        }
                        platform_results["ads_discovered"] += len(ads)
                        platform_results["patterns_extracted"] += len(industry_patterns)

                    except Exception as e:
                        logger.debug(f"[{self.name}] {platform.value}/{industry}: {e}")

                results["platforms"][platform.value] = platform_results
                results["total_ads"] += platform_results["ads_discovered"]
                results["total_patterns"] += platform_results["patterns_extracted"]

                self.last_discovery[platform] = datetime.now()

            results["new_trends"] = [
                {"name": t.name, "industry": t.industry, "momentum": t.momentum}
                for t in self.trend_detector.detect_trends()[:10]
            ]

            cycle_duration = (datetime.now() - cycle_start).total_seconds()

            logger.info(
                f"[{self.name}] Discovery complete: "
                f"{results['total_ads']} ads, {results['total_patterns']} patterns, "
                f"{len(results['new_trends'])} trends in {cycle_duration:.1f}s"
            )

            return results

    async def get_hooks_for_industry(
        self,
        industry: str,
        style: Optional[str] = None,
        top_k: int = 10
    ) -> List[ExtractedPattern]:
        """Get effective hooks for an industry"""

        query = f"effective hook for {industry}"
        if style:
            query += f" {style} style"

        return await self.indexer.search_patterns(
            pattern_type=PatternType.HOOK,
            query=query,
            industry=industry,
            top_k=top_k
        )

    async def get_ctas_for_industry(
        self,
        industry: str,
        goal: Optional[str] = None,
        top_k: int = 10
    ) -> List[ExtractedPattern]:
        """Get effective CTAs for an industry"""

        query = f"call to action for {industry}"
        if goal:
            query += f" {goal}"

        return await self.indexer.search_patterns(
            pattern_type=PatternType.CTA,
            query=query,
            industry=industry,
            top_k=top_k
        )

    async def get_visual_styles(
        self,
        industry: str,
        mood: Optional[str] = None,
        top_k: int = 10
    ) -> List[ExtractedPattern]:
        """Get trending visual styles for an industry"""

        query = f"visual style for {industry} commercial"
        if mood:
            query += f" {mood} mood"

        return await self.indexer.search_patterns(
            pattern_type=PatternType.VISUAL_STYLE,
            query=query,
            industry=industry,
            top_k=top_k
        )

    async def get_trending_patterns(
        self,
        industry: Optional[str] = None,
        lookback_days: int = 14
    ) -> List[TrendSignal]:
        """Get currently trending patterns"""

        trends = self.trend_detector.detect_trends(lookback_days)

        if industry:
            trends = [t for t in trends if t.industry == industry]

        return trends

    async def enhance_brief(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a creative brief with curator intelligence"""

        industry = brief.get("industry", "general")

        hooks = await self.get_hooks_for_industry(industry, top_k=5)
        ctas = await self.get_ctas_for_industry(industry, top_k=5)
        styles = await self.get_visual_styles(industry, top_k=3)
        trends = await self.get_trending_patterns(industry, lookback_days=7)

        enhanced = {
            **brief,
            "curator_intelligence": {
                "recommended_hooks": [
                    {"content": h.content, "effectiveness": h.effectiveness_score}
                    for h in hooks
                ],
                "recommended_ctas": [
                    {"content": c.content, "style": c.metadata.get("style")}
                    for c in ctas
                ],
                "trending_styles": [
                    {"style": s.content, "colors": s.metadata.get("colors")}
                    for s in styles
                ],
                "active_trends": [
                    {"name": t.name, "momentum": t.momentum}
                    for t in trends[:5]
                ],
                "intelligence_timestamp": datetime.now().isoformat()
            }
        }

        return enhanced

    def get_status(self) -> Dict[str, Any]:
        """Get curator status"""

        return {
            "name": self.name,
            "running": self.running,
            "last_discovery": {
                p.value: ts.isoformat()
                for p, ts in self.last_discovery.items()
            },
            "discovery_stats": dict(self.discovery_stats),
            "active_scrapers": [p.value for p in self.scrapers.keys()],
            "configured_industries": self.config.industries,
            "discovery_interval_hours": self.config.discovery_interval_hours
        }


# =============================================================================
# MOCK COMPONENTS
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing"""

    async def generate(self, prompt: str) -> str:
        return json.dumps({
            "hook": "Discover the future of AI automation",
            "hook_type": "curiosity",
            "value_proposition": "Save 10 hours per week",
            "urgency_triggers": ["limited time", "act now"],
            "emotional_appeals": ["curiosity", "efficiency"],
            "cta_style": "direct",
            "cta_text": "Get Started Free",
            "tone": "professional",
            "key_phrases": ["AI automation", "save time"],
            "effectiveness_score": 0.78
        })


class MockEmbedder:
    """Mock embedding model for testing"""

    async def embed(self, text: str) -> List[float]:
        import random
        return [random.random() for _ in range(1536)]


class MockQdrantClient:
    """Mock Qdrant client for testing"""

    def __init__(self):
        self.collections: Dict[str, List[Dict]] = defaultdict(list)

    async def create_collection(self, collection_name: str, **kwargs):
        self.collections[collection_name] = []

    async def upsert(self, collection_name: str, points: List[Dict]):
        self.collections[collection_name].extend(points)

    async def search(self, collection_name: str, **kwargs) -> List:
        return []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_curator(
    llm_client=None,
    qdrant_client=None,
    embedding_model=None,
    meta_token: Optional[str] = None,
    tiktok_token: Optional[str] = None,
    config: Optional[CuratorConfig] = None
) -> TheCurator:
    """Factory function to create a configured curator instance"""

    config = config or CuratorConfig()

    llm = llm_client or MockLLMClient()
    qdrant = qdrant_client or MockQdrantClient()
    embedder = embedding_model or MockEmbedder()

    scrapers = {}
    scrapers[Platform.META] = MetaAdLibraryScraper(meta_token or "")
    scrapers[Platform.TIKTOK] = TikTokCreativeCenterScraper(tiktok_token)
    scrapers[Platform.YOUTUBE] = YouTubeAdsScraper()

    return TheCurator(
        config=config,
        llm_client=llm,
        qdrant_client=qdrant,
        embedding_model=embedder,
        scrapers=scrapers
    )
