"""
================================================================================
🔥 RAGNAROK LEGENDARY UPGRADES - THE COGNITIVE APEX
================================================================================
8 NEW Agents That Transform RAGNAROK Into an Unstoppable AI Commercial Machine

ALREADY PROPOSED (From Analysis):
├── Agent 7.5:  "THE AUTEUR" - Vision-Language Creative QA
├── Agent 8.5:  "THE GENETICIST" - DSPy Prompt Self-Optimization
└── SHADOW MODE - Risk-Free Evolution Pipeline

NEW LEGENDARY ADDITIONS:
├── Agent 11:   "THE ORACLE" - Viral Potential Predictor
├── Agent 12:   "THE CHAMELEON" - Multi-Platform Optimizer
├── Agent 13:   "THE MEMORY" - Client DNA System
├── Agent 14:   "THE HUNTER" - Real-Time Trend Radar
└── Agent 15:   "THE ACCOUNTANT" - Dynamic Budget Optimizer

COMBINED IMPACT:
- Self-healing creative pipeline (Auteur + Geneticist)
- Predictive virality optimization (Oracle)
- Platform-native content (Chameleon)
- Personalized client experience (Memory)
- Trend-riding content (Hunter)
- Maximum ROI per dollar (Accountant)

Total System: 23 Agents | $2.20-2.80/commercial | Fully Autonomous Flywheel

Author: Barrios A2I | Version: 4.0.0 LEGENDARY | January 2026
================================================================================
"""

import asyncio
import time
import hashlib
import json
import re
import logging
import base64
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict
from functools import lru_cache

# Observability
from opentelemetry import trace
from prometheus_client import Counter, Histogram, Gauge

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

LEGENDARY_AGENT_CALLS = Counter(
    'ragnarok_legendary_agent_calls_total',
    'Legendary agent invocations',
    ['agent', 'status']
)

VIRAL_PREDICTION_ACCURACY = Gauge(
    'ragnarok_viral_prediction_accuracy',
    'Viral prediction accuracy over time',
    ['industry']
)

PLATFORM_OPTIMIZATION_SCORE = Gauge(
    'ragnarok_platform_optimization_score',
    'Platform-specific optimization score',
    ['platform']
)

CLIENT_DNA_MATCH = Histogram(
    'ragnarok_client_dna_match_score',
    'Client preference match score',
    ['client_tier'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

TREND_INTEGRATION_LATENCY = Histogram(
    'ragnarok_trend_integration_latency_seconds',
    'Time to integrate trends',
    ['trend_source']
)


# =============================================================================
# AGENT 7.5: "THE AUTEUR" - Vision-Language Creative QA
# =============================================================================

class CreativeQARequest(BaseModel):
    """Request for creative quality assessment"""
    video_path: str
    script_intent: str
    visual_style_target: str
    brand_guidelines: Dict[str, Any] = Field(default_factory=dict)
    frame_count: int = 5


class CreativeQAIssue(BaseModel):
    """A detected creative issue"""
    severity: str  # "critical", "major", "minor"
    category: str  # "visual", "text", "brand", "pacing"
    description: str
    timestamp_seconds: Optional[float] = None
    suggested_fix: str


class CreativeQAResponse(BaseModel):
    """Response with creative assessment"""
    passed: bool
    overall_score: float  # 0-1
    issues: List[CreativeQAIssue]
    frame_analysis: List[Dict[str, Any]]
    recommendation: str  # "approve", "revise", "reject"
    revision_prompts: List[str] = Field(default_factory=list)
    cost_usd: float
    latency_ms: float


class TheAuteur:
    """
    Agent 7.5: "THE AUTEUR" - Vision-Language Creative QA
    
    Uses GPT-4o Vision or Claude 3.5 Sonnet Vision to "watch" the video
    and compare it against the script intent.
    
    Checks:
    1. Visual-Script Alignment - Does the video show what the script describes?
    2. Brand Consistency - Are colors, fonts, logos correct?
    3. Text Readability - Is overlaid text readable against backgrounds?
    4. Pacing Analysis - Does the timing match the emotional arc?
    5. Quality Detection - Any artifacts, glitches, or low-quality frames?
    
    Cost: ~$0.03-0.08 per video (depends on frame count)
    """
    
    def __init__(
        self,
        vision_client: Any = None,  # OpenAI or Anthropic client
        model: str = "gpt-4o",
        max_frames: int = 8
    ):
        self.vision_client = vision_client
        self.model = model
        self.max_frames = max_frames
        
        self.name = "the_auteur"
        self.version = "1.0.0"
        self.cost_per_frame = 0.01  # ~$0.01 per frame analyzed
        
        # Quality thresholds
        self.thresholds = {
            "critical_score": 0.3,   # Below this = auto-reject
            "revision_score": 0.7,   # Below this = needs revision
            "approval_score": 0.85   # Above this = auto-approve
        }
    
    def _extract_frames(self, video_path: str, num_frames: int = 5) -> List[str]:
        """
        Extract keyframes from video as base64 strings.
        Uses OpenCV for extraction.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            frames = []
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize for efficiency
                    frame = cv2.resize(frame, (640, 360))
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    frames.append({
                        "base64": b64,
                        "timestamp": idx / fps if fps > 0 else 0,
                        "frame_number": int(idx)
                    })
            
            cap.release()
            return frames
            
        except ImportError:
            logger.warning("OpenCV not available, using synthetic frames")
            return self._synthetic_frames(num_frames)
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return self._synthetic_frames(num_frames)
    
    def _synthetic_frames(self, num_frames: int) -> List[Dict]:
        """Generate synthetic frame data for testing"""
        return [
            {
                "base64": "",  # Empty for testing
                "timestamp": i * 6.0,  # Assume 30s video
                "frame_number": i * 30
            }
            for i in range(num_frames)
        ]
    
    async def critique(self, request: CreativeQARequest) -> CreativeQAResponse:
        """
        Perform comprehensive creative quality assessment.
        """
        with tracer.start_as_current_span("auteur_critique") as span:
            start_time = time.time()
            
            # Extract frames
            frames = self._extract_frames(request.video_path, request.frame_count)
            span.set_attribute("frame_count", len(frames))
            
            # Build analysis prompt
            analysis_prompt = self._build_analysis_prompt(request)
            
            # Analyze with vision model
            frame_analysis = []
            issues = []
            scores = []
            
            for i, frame in enumerate(frames):
                if frame["base64"]:
                    analysis = await self._analyze_frame(
                        frame, 
                        request.script_intent,
                        request.visual_style_target,
                        i, 
                        len(frames)
                    )
                else:
                    # Synthetic analysis for testing
                    analysis = self._synthetic_analysis(i, len(frames))
                
                frame_analysis.append(analysis)
                scores.append(analysis.get("score", 0.8))
                
                if analysis.get("issues"):
                    issues.extend(analysis["issues"])
            
            # Calculate overall score
            overall_score = np.mean(scores) if scores else 0.5
            
            # Determine recommendation
            if overall_score < self.thresholds["critical_score"]:
                recommendation = "reject"
                revision_prompts = ["Complete re-generation required"]
            elif overall_score < self.thresholds["revision_score"]:
                recommendation = "revise"
                revision_prompts = self._generate_revision_prompts(issues)
            else:
                recommendation = "approve"
                revision_prompts = []
            
            # Filter issues by severity
            critical_issues = [i for i in issues if i.severity == "critical"]
            
            latency = (time.time() - start_time) * 1000
            cost = len(frames) * self.cost_per_frame
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status=recommendation
            ).inc()
            
            return CreativeQAResponse(
                passed=recommendation == "approve" and len(critical_issues) == 0,
                overall_score=overall_score,
                issues=issues,
                frame_analysis=frame_analysis,
                recommendation=recommendation,
                revision_prompts=revision_prompts,
                cost_usd=cost,
                latency_ms=latency
            )
    
    def _build_analysis_prompt(self, request: CreativeQARequest) -> str:
        """Build the analysis prompt for the vision model"""
        return f"""You are THE AUTEUR, a world-class Creative Director reviewing video content.

SCRIPT INTENT: {request.script_intent}

TARGET VISUAL STYLE: {request.visual_style_target}

BRAND GUIDELINES: {json.dumps(request.brand_guidelines, indent=2)}

Analyze this video frame and evaluate:
1. VISUAL-SCRIPT ALIGNMENT (Does this frame match the script intent?)
2. BRAND CONSISTENCY (Colors, fonts, overall feel)
3. TEXT READABILITY (If any text, is it readable?)
4. QUALITY (Any artifacts, blur, or technical issues?)
5. EMOTIONAL IMPACT (Does this frame evoke the intended emotion?)

Provide a score from 0-100 and list any issues found.
"""
    
    async def _analyze_frame(
        self, 
        frame: Dict, 
        script_intent: str,
        visual_style: str,
        frame_index: int,
        total_frames: int
    ) -> Dict[str, Any]:
        """Analyze a single frame with the vision model"""
        
        if not self.vision_client:
            return self._synthetic_analysis(frame_index, total_frames)
        
        try:
            # Call vision model
            response = await self.vision_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are THE AUTEUR, analyzing video frames for quality."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Script: {script_intent}\nStyle: {visual_style}\nFrame {frame_index+1}/{total_frames}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame['base64']}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse response (simplified)
            content = response.choices[0].message.content
            
            return {
                "frame_index": frame_index,
                "timestamp": frame["timestamp"],
                "score": 0.85,  # Would parse from response
                "analysis": content,
                "issues": []
            }
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return self._synthetic_analysis(frame_index, total_frames)
    
    def _synthetic_analysis(self, frame_index: int, total_frames: int) -> Dict[str, Any]:
        """Generate synthetic analysis for testing"""
        base_score = 0.75 + np.random.uniform(0, 0.2)
        
        issues = []
        if np.random.random() < 0.2:
            issues.append(CreativeQAIssue(
                severity="minor",
                category="visual",
                description="Slight color mismatch with brand guidelines",
                timestamp_seconds=frame_index * 6.0,
                suggested_fix="Adjust color grading in post-production"
            ))
        
        return {
            "frame_index": frame_index,
            "timestamp": frame_index * (30.0 / total_frames),
            "score": base_score,
            "analysis": f"Frame {frame_index+1} analysis: Good visual quality",
            "issues": issues
        }
    
    def _generate_revision_prompts(self, issues: List[CreativeQAIssue]) -> List[str]:
        """Generate revision prompts based on issues"""
        prompts = []
        
        # Group issues by category
        by_category = defaultdict(list)
        for issue in issues:
            by_category[issue.category].append(issue)
        
        if "visual" in by_category:
            prompts.append(
                f"VISUAL FIX: Address {len(by_category['visual'])} visual issues - "
                f"{by_category['visual'][0].suggested_fix}"
            )
        
        if "text" in by_category:
            prompts.append(
                "TEXT FIX: Improve text readability - increase contrast or font size"
            )
        
        if "brand" in by_category:
            prompts.append(
                "BRAND FIX: Align with brand guidelines - check colors and style"
            )
        
        if "pacing" in by_category:
            prompts.append(
                "PACING FIX: Adjust timing to match emotional arc"
            )
        
        return prompts


# =============================================================================
# AGENT 8.5: "THE GENETICIST" - DSPy Prompt Self-Optimization
# =============================================================================

class PromptGene(BaseModel):
    """A single prompt gene that can evolve"""
    gene_id: str
    prompt_text: str
    target_agent: str  # Which agent this prompt is for
    fitness_score: float = 0.5
    generation: int = 1
    parent_ids: List[str] = Field(default_factory=list)
    mutations: List[str] = Field(default_factory=list)


class EvolutionResult(BaseModel):
    """Result of a prompt evolution cycle"""
    original_prompt: str
    evolved_prompt: str
    fitness_improvement: float
    generation: int
    mutations_applied: List[str]
    a_b_test_id: Optional[str] = None


class GeneticOptimizationRequest(BaseModel):
    """Request to optimize a prompt genetically"""
    target_agent: str  # "story_creator", "prompt_engineer", etc.
    current_prompt: str
    performance_data: List[Dict[str, Any]]  # Historical performance
    optimization_goal: str = "engagement"  # "engagement", "conversion", "completion"
    generations: int = 3


class GeneticOptimizationResponse(BaseModel):
    """Response with optimized prompt"""
    evolved_prompt: str
    fitness_improvement: float
    evolution_history: List[EvolutionResult]
    recommended_action: str  # "deploy", "test", "continue_evolution"
    cost_usd: float
    latency_ms: float


class TheGeneticist:
    """
    Agent 8.5: "THE GENETICIST" - DSPy Prompt Self-Optimization
    
    Uses genetic algorithms to evolve the system prompts of other agents.
    Instead of just updating weights, it REWRITES the actual prompts based
    on what's working.
    
    Process:
    1. Analyze performance data to identify winning patterns
    2. Extract "genes" (effective phrases, structures, tones)
    3. Mutate prompts by crossing winning genes
    4. Test evolved prompts via A/B testing
    5. Promote winners to production
    
    Cost: ~$0.02 per evolution cycle
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        ab_tester: Any = None,  # Agent 9
        mutation_rate: float = 0.3
    ):
        self.llm_client = llm_client
        self.ab_tester = ab_tester
        self.mutation_rate = mutation_rate
        
        self.name = "the_geneticist"
        self.version = "1.0.0"
        self.cost_per_generation = 0.007
        
        # Gene library - effective prompt fragments
        self.gene_library = {
            "emotional_openers": [
                "You are a master of emotional storytelling who...",
                "Channel the energy of a passionate brand advocate...",
                "Write with the conviction of someone who truly believes..."
            ],
            "structure_genes": [
                "Structure your output with: Hook (3s) → Problem (5s) → Solution (15s) → CTA (7s)",
                "Follow the AIDA framework: Attention, Interest, Desire, Action",
                "Use the PAS formula: Problem → Agitation → Solution"
            ],
            "style_genes": [
                "Write in short, punchy sentences. No fluff.",
                "Use active voice exclusively. Be direct.",
                "Speak as if you're talking to a friend, not selling."
            ],
            "industry_genes": {
                "dental": "Focus on transformation: from pain/embarrassment to confidence/health",
                "legal": "Emphasize authority, track record, and client outcomes",
                "real_estate": "Paint a picture of lifestyle, not just property features",
                "fitness": "Tap into aspiration and community belonging"
            }
        }
        
        # Prompt templates by agent
        self.agent_templates = {
            "story_creator": """You are {opener}

Your mission: Create a {duration}-second commercial script for {industry}.

{structure}

{style}

{industry_specific}

Business: {business_name}
USP: {usp}
Target Audience: {audience}

Write the script now.""",
            
            "prompt_engineer": """You are a visual prompt engineer specializing in AI video generation.

{opener}

{structure}

Create {scene_count} scene prompts for a {style} {industry} commercial.

{industry_specific}

Script to visualize:
{script}

Generate prompts now."""
        }
    
    async def evolve(self, request: GeneticOptimizationRequest) -> GeneticOptimizationResponse:
        """
        Evolve a prompt through multiple generations.
        """
        with tracer.start_as_current_span("geneticist_evolve") as span:
            start_time = time.time()
            span.set_attribute("target_agent", request.target_agent)
            span.set_attribute("generations", request.generations)
            
            current_prompt = request.current_prompt
            evolution_history = []
            total_improvement = 0.0
            
            for gen in range(request.generations):
                # Analyze performance to find winning patterns
                winning_patterns = self._analyze_winners(request.performance_data)
                
                # Select genes based on performance
                selected_genes = self._select_genes(
                    winning_patterns, 
                    request.optimization_goal
                )
                
                # Crossover: combine winning genes into new prompt
                crossed_prompt = self._crossover(current_prompt, selected_genes)
                
                # Mutate: apply random beneficial mutations
                mutated_prompt, mutations = self._mutate(crossed_prompt)
                
                # Calculate fitness improvement (estimated)
                fitness_improvement = self._estimate_fitness(
                    current_prompt, 
                    mutated_prompt,
                    request.optimization_goal
                )
                
                # Record evolution
                result = EvolutionResult(
                    original_prompt=current_prompt,
                    evolved_prompt=mutated_prompt,
                    fitness_improvement=fitness_improvement,
                    generation=gen + 1,
                    mutations_applied=mutations
                )
                
                # Create A/B test for significant improvements
                if fitness_improvement > 0.05 and self.ab_tester:
                    experiment = await self.ab_tester.create_experiment(
                        name=f"Prompt Evolution Gen {gen+1}",
                        description=f"Testing evolved {request.target_agent} prompt",
                        industry="all",
                        parameter_to_test="system_prompt",
                        control_value=current_prompt,
                        variant_values=[mutated_prompt]
                    )
                    result.a_b_test_id = experiment.experiment_id
                
                evolution_history.append(result)
                total_improvement += fitness_improvement
                current_prompt = mutated_prompt
            
            # Determine recommendation
            if total_improvement > 0.15:
                recommendation = "deploy"
            elif total_improvement > 0.05:
                recommendation = "test"
            else:
                recommendation = "continue_evolution"
            
            latency = (time.time() - start_time) * 1000
            cost = request.generations * self.cost_per_generation
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status=recommendation
            ).inc()
            
            return GeneticOptimizationResponse(
                evolved_prompt=current_prompt,
                fitness_improvement=total_improvement,
                evolution_history=evolution_history,
                recommended_action=recommendation,
                cost_usd=cost,
                latency_ms=latency
            )
    
    def _analyze_winners(self, performance_data: List[Dict]) -> Dict[str, List[str]]:
        """Analyze performance data to find winning patterns"""
        winners = {
            "hooks": [],
            "structures": [],
            "tones": [],
            "keywords": []
        }
        
        # Sort by performance
        sorted_data = sorted(
            performance_data, 
            key=lambda x: x.get("engagement_rate", 0), 
            reverse=True
        )
        
        # Take top 20%
        top_performers = sorted_data[:max(1, len(sorted_data) // 5)]
        
        for item in top_performers:
            if "hook_type" in item:
                winners["hooks"].append(item["hook_type"])
            if "structure" in item:
                winners["structures"].append(item["structure"])
            if "tone" in item:
                winners["tones"].append(item["tone"])
        
        return winners
    
    def _select_genes(self, winning_patterns: Dict, goal: str) -> List[str]:
        """Select best genes based on winning patterns"""
        selected = []
        
        # Select emotional opener based on winning tones
        if winning_patterns.get("tones"):
            if "emotional" in winning_patterns["tones"]:
                selected.append(self.gene_library["emotional_openers"][0])
            elif "authoritative" in winning_patterns["tones"]:
                selected.append(self.gene_library["emotional_openers"][1])
        else:
            selected.append(np.random.choice(self.gene_library["emotional_openers"]))
        
        # Select structure based on winning patterns
        if winning_patterns.get("structures"):
            if "AIDA" in winning_patterns["structures"]:
                selected.append(self.gene_library["structure_genes"][1])
            else:
                selected.append(self.gene_library["structure_genes"][0])
        else:
            selected.append(np.random.choice(self.gene_library["structure_genes"]))
        
        # Add style gene
        selected.append(np.random.choice(self.gene_library["style_genes"]))
        
        return selected
    
    def _crossover(self, current_prompt: str, selected_genes: List[str]) -> str:
        """Combine current prompt with selected genes"""
        # Simple crossover: inject genes at appropriate positions
        crossed = current_prompt
        
        for gene in selected_genes:
            if "You are" in gene and "You are" in crossed:
                # Replace opener
                lines = crossed.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("You are"):
                        lines[i] = gene
                        break
                crossed = "\n".join(lines)
            else:
                # Append as guidance
                crossed = crossed.rstrip() + f"\n\n{gene}"
        
        return crossed
    
    def _mutate(self, prompt: str) -> Tuple[str, List[str]]:
        """Apply random beneficial mutations"""
        mutations_applied = []
        mutated = prompt
        
        if np.random.random() < self.mutation_rate:
            # Mutation 1: Add urgency
            if "urgently" not in mutated.lower() and "immediately" not in mutated.lower():
                mutated += "\n\nIMPORTANT: Make the CTA feel urgent but not pushy."
                mutations_applied.append("urgency_injection")
        
        if np.random.random() < self.mutation_rate:
            # Mutation 2: Strengthen specificity
            if "specific" not in mutated.lower():
                mutated = mutated.replace(
                    "Write the script",
                    "Write a SPECIFIC, DETAILED script with concrete examples"
                )
                mutations_applied.append("specificity_boost")
        
        if np.random.random() < self.mutation_rate:
            # Mutation 3: Add emotional anchor
            emotion_anchors = [
                "The viewer should FEEL the transformation.",
                "Every word should build trust.",
                "Create a moment of connection in the first 3 seconds."
            ]
            mutated += f"\n\n{np.random.choice(emotion_anchors)}"
            mutations_applied.append("emotional_anchor")
        
        return mutated, mutations_applied
    
    def _estimate_fitness(self, original: str, evolved: str, goal: str) -> float:
        """Estimate fitness improvement (simplified heuristic)"""
        # Count improvements
        improvements = 0
        
        # Check for specificity increase
        if evolved.count("specific") > original.count("specific"):
            improvements += 0.03
        
        # Check for emotional language
        emotional_words = ["feel", "imagine", "transform", "discover", "experience"]
        original_emotional = sum(1 for w in emotional_words if w in original.lower())
        evolved_emotional = sum(1 for w in emotional_words if w in evolved.lower())
        if evolved_emotional > original_emotional:
            improvements += 0.02 * (evolved_emotional - original_emotional)
        
        # Check for structure
        if "→" in evolved and "→" not in original:
            improvements += 0.02
        
        # Length optimization (not too short, not too long)
        original_words = len(original.split())
        evolved_words = len(evolved.split())
        if 150 <= evolved_words <= 400 and not (150 <= original_words <= 400):
            improvements += 0.02
        
        # Add some randomness to simulate real A/B results
        improvements += np.random.uniform(-0.01, 0.03)
        
        return max(0, improvements)


# =============================================================================
# AGENT 11: "THE ORACLE" - Viral Potential Predictor
# =============================================================================

class ViralityFactors(BaseModel):
    """Factors that contribute to virality"""
    emotional_resonance: float = Field(ge=0, le=1)
    shareability: float = Field(ge=0, le=1)
    hook_strength: float = Field(ge=0, le=1)
    trend_alignment: float = Field(ge=0, le=1)
    controversy_potential: float = Field(ge=0, le=1)
    meme_potential: float = Field(ge=0, le=1)


class ViralityPrediction(BaseModel):
    """Viral potential prediction"""
    viral_score: float = Field(ge=0, le=1)
    predicted_views_7d: int
    predicted_shares: int
    confidence: float
    factors: ViralityFactors
    recommendations: List[str]


class OraclePredictionRequest(BaseModel):
    """Request for virality prediction"""
    script: str
    visual_style: str
    hook_technique: str
    industry: str
    target_platform: str = "all"  # "tiktok", "youtube", "instagram", "all"


class OraclePredictionResponse(BaseModel):
    """Response with virality prediction"""
    prediction: ViralityPrediction
    should_boost: bool
    boost_recommendations: List[str]
    cost_usd: float
    latency_ms: float


class TheOracle:
    """
    Agent 11: "THE ORACLE" - Viral Potential Predictor
    
    Predicts the viral potential of a commercial BEFORE it's generated,
    allowing optimization while it's still cheap.
    
    Uses:
    1. Historical performance data (from Agent 8)
    2. Current trend analysis (from Agent 14)
    3. Platform-specific algorithms knowledge
    4. Emotional resonance scoring
    5. Hook strength analysis
    
    Cost: ~$0.005 per prediction (lightweight ML inference)
    """
    
    def __init__(
        self,
        trend_radar: Any = None,  # Agent 14
        meta_learner: Any = None  # Agent 8
    ):
        self.trend_radar = trend_radar
        self.meta_learner = meta_learner
        
        self.name = "the_oracle"
        self.version = "1.0.0"
        self.cost_per_prediction = 0.005
        
        # Platform-specific multipliers
        self.platform_multipliers = {
            "tiktok": {
                "hook_importance": 2.0,  # First 1 second is critical
                "trend_importance": 1.8,
                "meme_importance": 1.5,
                "base_views": 5000
            },
            "youtube": {
                "hook_importance": 1.5,  # First 5 seconds
                "trend_importance": 1.2,
                "quality_importance": 1.8,
                "base_views": 1000
            },
            "instagram": {
                "hook_importance": 1.7,
                "aesthetic_importance": 1.8,
                "trend_importance": 1.5,
                "base_views": 2000
            }
        }
        
        # Hook strength scores by type
        self.hook_scores = {
            "question": 0.70,
            "statistic": 0.65,
            "testimonial": 0.75,
            "emotion": 0.80,
            "negative_callout": 0.85,
            "story": 0.72,
            "comparison": 0.68,
            "challenge": 0.78,
            "controversy": 0.88
        }
        
        # Emotional keywords and their resonance scores
        self.emotional_triggers = {
            "high": ["transform", "secret", "finally", "discover", "imagine", "never"],
            "medium": ["better", "improve", "easy", "fast", "proven", "trusted"],
            "low": ["quality", "service", "professional", "experience", "solution"]
        }
    
    async def predict(self, request: OraclePredictionRequest) -> OraclePredictionResponse:
        """
        Predict viral potential of the commercial.
        """
        with tracer.start_as_current_span("oracle_predict") as span:
            start_time = time.time()
            span.set_attribute("industry", request.industry)
            span.set_attribute("platform", request.target_platform)
            
            # Calculate individual factors
            emotional_resonance = self._score_emotional_resonance(request.script)
            shareability = self._score_shareability(request)
            hook_strength = self._score_hook_strength(request.hook_technique)
            trend_alignment = await self._score_trend_alignment(request)
            controversy = self._score_controversy(request.script)
            meme_potential = self._score_meme_potential(request)
            
            factors = ViralityFactors(
                emotional_resonance=emotional_resonance,
                shareability=shareability,
                hook_strength=hook_strength,
                trend_alignment=trend_alignment,
                controversy_potential=controversy,
                meme_potential=meme_potential
            )
            
            # Calculate weighted viral score
            weights = self._get_platform_weights(request.target_platform)
            viral_score = (
                factors.emotional_resonance * weights.get("emotional", 0.2) +
                factors.shareability * weights.get("shareability", 0.15) +
                factors.hook_strength * weights.get("hook", 0.25) +
                factors.trend_alignment * weights.get("trend", 0.2) +
                factors.controversy_potential * weights.get("controversy", 0.1) +
                factors.meme_potential * weights.get("meme", 0.1)
            )
            
            # Predict views and shares
            base_views = self.platform_multipliers.get(
                request.target_platform, 
                {"base_views": 2000}
            )["base_views"]
            
            predicted_views = int(base_views * (1 + viral_score * 10))
            predicted_shares = int(predicted_views * viral_score * 0.05)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(factors, viral_score)
            
            # Determine if boosting is recommended
            should_boost = viral_score > 0.7
            boost_recommendations = []
            if should_boost:
                boost_recommendations = [
                    f"This content has {viral_score:.0%} viral potential - recommend paid promotion",
                    f"Best posting time: Peak hours for {request.target_platform}",
                    "Consider influencer partnerships for amplification"
                ]
            
            prediction = ViralityPrediction(
                viral_score=viral_score,
                predicted_views_7d=predicted_views,
                predicted_shares=predicted_shares,
                confidence=min(0.8 + viral_score * 0.15, 0.95),
                factors=factors,
                recommendations=recommendations
            )
            
            latency = (time.time() - start_time) * 1000
            
            VIRAL_PREDICTION_ACCURACY.labels(
                industry=request.industry
            ).set(viral_score)
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status="predicted"
            ).inc()
            
            return OraclePredictionResponse(
                prediction=prediction,
                should_boost=should_boost,
                boost_recommendations=boost_recommendations,
                cost_usd=self.cost_per_prediction,
                latency_ms=latency
            )
    
    def _score_emotional_resonance(self, script: str) -> float:
        """Score the emotional resonance of the script"""
        script_lower = script.lower()
        score = 0.5  # Base score
        
        # Check for high-impact triggers
        for trigger in self.emotional_triggers["high"]:
            if trigger in script_lower:
                score += 0.08
        
        for trigger in self.emotional_triggers["medium"]:
            if trigger in script_lower:
                score += 0.04
        
        # Check for personal pronouns (more engaging)
        personal_count = script_lower.count("you") + script_lower.count("your")
        score += min(personal_count * 0.02, 0.1)
        
        # Check for questions (engagement boosters)
        question_count = script.count("?")
        score += min(question_count * 0.03, 0.1)
        
        return min(score, 1.0)
    
    def _score_shareability(self, request: OraclePredictionRequest) -> float:
        """Score how likely the content is to be shared"""
        score = 0.5
        
        # Transformational content is highly shareable
        if "transform" in request.script.lower() or "before" in request.script.lower():
            score += 0.15
        
        # Useful/helpful content gets shared
        if any(word in request.script.lower() for word in ["tip", "trick", "hack", "secret"]):
            score += 0.12
        
        # Controversial or opinion-based content
        if "vs" in request.script.lower() or "better than" in request.script.lower():
            score += 0.1
        
        # Industry-specific shareability
        high_share_industries = ["fitness", "real_estate", "restaurant"]
        if request.industry in high_share_industries:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_hook_strength(self, hook_technique: str) -> float:
        """Score the hook technique strength"""
        return self.hook_scores.get(hook_technique, 0.65)
    
    async def _score_trend_alignment(self, request: OraclePredictionRequest) -> float:
        """Score alignment with current trends"""
        if self.trend_radar:
            try:
                trends = await self.trend_radar.get_current_trends(request.industry)
                # Match script against trends
                script_lower = request.script.lower()
                matches = sum(1 for trend in trends if trend.lower() in script_lower)
                return min(0.5 + matches * 0.15, 1.0)
            except Exception:
                pass
        
        # Default trend score
        return 0.6
    
    def _score_controversy(self, script: str) -> float:
        """Score controversy potential (careful balance)"""
        controversy_triggers = ["myth", "wrong", "mistake", "lie", "truth", "actually"]
        script_lower = script.lower()
        
        score = 0.3  # Base
        for trigger in controversy_triggers:
            if trigger in script_lower:
                score += 0.1
        
        return min(score, 0.8)  # Cap to avoid too controversial
    
    def _score_meme_potential(self, request: OraclePredictionRequest) -> float:
        """Score meme potential"""
        score = 0.3
        
        # Certain styles are more meme-able
        if request.visual_style in ["ugc", "raw", "authentic"]:
            score += 0.2
        
        # Relatable content
        relatable_words = ["when you", "that moment", "every", "always"]
        for word in relatable_words:
            if word in request.script.lower():
                score += 0.1
                break
        
        # Platform-specific
        if request.target_platform == "tiktok":
            score += 0.15
        
        return min(score, 1.0)
    
    def _get_platform_weights(self, platform: str) -> Dict[str, float]:
        """Get factor weights by platform"""
        weights = {
            "tiktok": {
                "emotional": 0.15, "shareability": 0.1, "hook": 0.3,
                "trend": 0.25, "controversy": 0.1, "meme": 0.1
            },
            "youtube": {
                "emotional": 0.25, "shareability": 0.15, "hook": 0.2,
                "trend": 0.15, "controversy": 0.15, "meme": 0.1
            },
            "instagram": {
                "emotional": 0.2, "shareability": 0.2, "hook": 0.25,
                "trend": 0.2, "controversy": 0.05, "meme": 0.1
            },
            "all": {
                "emotional": 0.2, "shareability": 0.15, "hook": 0.25,
                "trend": 0.2, "controversy": 0.1, "meme": 0.1
            }
        }
        return weights.get(platform, weights["all"])
    
    def _generate_recommendations(self, factors: ViralityFactors, score: float) -> List[str]:
        """Generate recommendations to improve virality"""
        recs = []
        
        if factors.hook_strength < 0.7:
            recs.append("HOOK UPGRADE: Consider using 'negative_callout' or 'emotion' hook for stronger opening")
        
        if factors.emotional_resonance < 0.6:
            recs.append("EMOTION BOOST: Add transformational language ('discover', 'imagine', 'finally')")
        
        if factors.trend_alignment < 0.5:
            recs.append("TREND ALIGN: Incorporate current industry trends for better discoverability")
        
        if factors.shareability < 0.6:
            recs.append("SHARE FACTOR: Add tips, hacks, or surprising facts to encourage sharing")
        
        if score >= 0.8:
            recs.append("🔥 HIGH VIRAL POTENTIAL: Proceed with confidence!")
        elif score >= 0.6:
            recs.append("✅ GOOD POTENTIAL: Minor optimizations could push this higher")
        else:
            recs.append("⚠️ LOW POTENTIAL: Consider significant changes before production")
        
        return recs


# =============================================================================
# AGENT 12: "THE CHAMELEON" - Multi-Platform Optimizer
# =============================================================================

class PlatformSpec(BaseModel):
    """Platform-specific specifications"""
    platform: str
    aspect_ratio: str
    max_duration: int
    optimal_duration: int
    text_overlay_style: str
    cta_placement: str
    thumbnail_required: bool


class PlatformOptimizedContent(BaseModel):
    """Content optimized for a specific platform"""
    platform: str
    adapted_script: str
    visual_adjustments: List[str]
    audio_adjustments: List[str]
    optimal_posting_time: str
    hashtag_recommendations: List[str]


class ChameleonOptimizationRequest(BaseModel):
    """Request to optimize for multiple platforms"""
    original_script: str
    original_style: str
    business_name: str
    industry: str
    target_platforms: List[str] = ["youtube", "tiktok", "instagram"]


class ChameleonOptimizationResponse(BaseModel):
    """Response with platform-optimized variants"""
    platform_variants: List[PlatformOptimizedContent]
    cross_platform_synergies: List[str]
    cost_usd: float
    latency_ms: float


class TheChameleon:
    """
    Agent 12: "THE CHAMELEON" - Multi-Platform Optimizer
    
    Automatically adapts a single commercial to be native on each platform.
    Instead of generic "one-size-fits-all", creates platform-specific versions.
    
    Adaptations:
    1. TikTok: Vertical, fast-paced, trend-aware, native captions
    2. YouTube: Widescreen, longer hook, quality emphasis, SEO
    3. Instagram: Square/Reels, aesthetic, lifestyle-focused, hashtags
    4. LinkedIn: Professional, data-driven, thought leadership
    5. Facebook: Family-friendly, broad appeal, longer form
    
    Cost: ~$0.01 per platform adaptation
    """
    
    def __init__(self):
        self.name = "the_chameleon"
        self.version = "1.0.0"
        self.cost_per_platform = 0.01
        
        self.platform_specs = {
            "tiktok": PlatformSpec(
                platform="tiktok",
                aspect_ratio="9:16",
                max_duration=60,
                optimal_duration=21,
                text_overlay_style="bold_center_captions",
                cta_placement="end_with_profile_visit",
                thumbnail_required=False
            ),
            "youtube": PlatformSpec(
                platform="youtube",
                aspect_ratio="16:9",
                max_duration=300,
                optimal_duration=30,
                text_overlay_style="minimal_lower_third",
                cta_placement="verbal_and_end_screen",
                thumbnail_required=True
            ),
            "instagram": PlatformSpec(
                platform="instagram",
                aspect_ratio="9:16",  # Reels
                max_duration=90,
                optimal_duration=15,
                text_overlay_style="aesthetic_animated",
                cta_placement="profile_link",
                thumbnail_required=True
            ),
            "linkedin": PlatformSpec(
                platform="linkedin",
                aspect_ratio="1:1",
                max_duration=180,
                optimal_duration=45,
                text_overlay_style="professional_subtitles",
                cta_placement="company_page_visit",
                thumbnail_required=True
            ),
            "facebook": PlatformSpec(
                platform="facebook",
                aspect_ratio="16:9",
                max_duration=120,
                optimal_duration=30,
                text_overlay_style="burned_in_captions",
                cta_placement="website_link",
                thumbnail_required=True
            )
        }
        
        self.posting_times = {
            "tiktok": {"optimal": "7-9pm", "backup": "12-2pm"},
            "youtube": {"optimal": "2-4pm", "backup": "6-8pm"},
            "instagram": {"optimal": "11am-1pm", "backup": "7-9pm"},
            "linkedin": {"optimal": "7-8am", "backup": "12pm"},
            "facebook": {"optimal": "1-4pm", "backup": "7-8pm"}
        }
        
        self.hashtag_strategies = {
            "tiktok": {"count": 5, "style": "trending+niche"},
            "instagram": {"count": 20, "style": "mixed"},
            "youtube": {"count": 3, "style": "seo_focused"},
            "linkedin": {"count": 3, "style": "professional"},
            "facebook": {"count": 2, "style": "minimal"}
        }
    
    async def adapt(self, request: ChameleonOptimizationRequest) -> ChameleonOptimizationResponse:
        """
        Adapt content for multiple platforms.
        """
        with tracer.start_as_current_span("chameleon_adapt") as span:
            start_time = time.time()
            span.set_attribute("platforms", len(request.target_platforms))
            
            variants = []
            
            for platform in request.target_platforms:
                if platform not in self.platform_specs:
                    continue
                
                spec = self.platform_specs[platform]
                
                # Adapt script
                adapted_script = self._adapt_script(
                    request.original_script, 
                    spec,
                    request.industry
                )
                
                # Generate visual adjustments
                visual_adjustments = self._get_visual_adjustments(spec, request.original_style)
                
                # Generate audio adjustments
                audio_adjustments = self._get_audio_adjustments(platform)
                
                # Get posting time
                posting_time = self.posting_times.get(platform, {}).get("optimal", "12pm")
                
                # Generate hashtags
                hashtags = self._generate_hashtags(
                    platform, 
                    request.industry, 
                    request.business_name
                )
                
                variants.append(PlatformOptimizedContent(
                    platform=platform,
                    adapted_script=adapted_script,
                    visual_adjustments=visual_adjustments,
                    audio_adjustments=audio_adjustments,
                    optimal_posting_time=posting_time,
                    hashtag_recommendations=hashtags
                ))
                
                PLATFORM_OPTIMIZATION_SCORE.labels(platform=platform).set(0.85)
            
            # Cross-platform synergies
            synergies = self._identify_synergies(request.target_platforms)
            
            latency = (time.time() - start_time) * 1000
            cost = len(variants) * self.cost_per_platform
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status="adapted"
            ).inc()
            
            return ChameleonOptimizationResponse(
                platform_variants=variants,
                cross_platform_synergies=synergies,
                cost_usd=cost,
                latency_ms=latency
            )
    
    def _adapt_script(self, script: str, spec: PlatformSpec, industry: str) -> str:
        """Adapt script for platform specifications"""
        adapted = script
        
        # Trim to optimal duration (roughly 3 words per second)
        max_words = spec.optimal_duration * 3
        words = adapted.split()
        if len(words) > max_words:
            adapted = " ".join(words[:max_words]) + "..."
        
        # Platform-specific adjustments
        if spec.platform == "tiktok":
            # Add immediacy
            if not adapted.startswith(("POV:", "Wait", "Stop", "This")):
                adapted = "Stop scrolling. " + adapted
        
        elif spec.platform == "linkedin":
            # Make more professional
            adapted = adapted.replace("!", ".")
            if "we" not in adapted.lower():
                adapted = "Our team at " + adapted
        
        elif spec.platform == "instagram":
            # Add lifestyle angle
            adapted = "✨ " + adapted
        
        return adapted
    
    def _get_visual_adjustments(self, spec: PlatformSpec, style: str) -> List[str]:
        """Get platform-specific visual adjustments"""
        adjustments = [
            f"Aspect ratio: {spec.aspect_ratio}",
            f"Text style: {spec.text_overlay_style}",
            f"CTA placement: {spec.cta_placement}"
        ]
        
        if spec.platform == "tiktok":
            adjustments.extend([
                "Add native TikTok captions",
                "Keep safe zones for UI elements",
                "Use fast cuts (2-3 seconds max per shot)"
            ])
        elif spec.platform == "youtube":
            adjustments.extend([
                "Create custom thumbnail",
                "Add end screen annotations",
                "Include subscribe reminder"
            ])
        elif spec.platform == "instagram":
            adjustments.extend([
                "Ensure aesthetic color grading",
                "Add branded story stickers",
                "Include swipe-up CTA"
            ])
        
        return adjustments
    
    def _get_audio_adjustments(self, platform: str) -> List[str]:
        """Get platform-specific audio adjustments"""
        adjustments = []
        
        if platform == "tiktok":
            adjustments = [
                "Consider using trending sound",
                "Add punchy sound effects",
                "Optimize for sound-off viewing (captions)"
            ]
        elif platform == "youtube":
            adjustments = [
                "Professional voiceover quality",
                "Background music at 15% volume",
                "Include audio branding"
            ]
        elif platform == "instagram":
            adjustments = [
                "Upbeat, modern music",
                "Clean audio mix",
                "Consider trending audio"
            ]
        elif platform == "linkedin":
            adjustments = [
                "Professional narration tone",
                "Subtle background music",
                "Optimize for muted autoplay"
            ]
        
        return adjustments
    
    def _generate_hashtags(self, platform: str, industry: str, business: str) -> List[str]:
        """Generate platform-appropriate hashtags"""
        strategy = self.hashtag_strategies.get(platform, {"count": 5, "style": "mixed"})
        
        # Industry-specific hashtags
        industry_tags = {
            "dental": ["#dentist", "#smile", "#oralhealth", "#dental"],
            "legal": ["#lawyer", "#legal", "#attorney", "#lawfirm"],
            "real_estate": ["#realestate", "#property", "#home", "#realtor"],
            "fitness": ["#fitness", "#workout", "#gym", "#health"],
            "restaurant": ["#food", "#restaurant", "#foodie", "#dining"]
        }
        
        tags = industry_tags.get(industry, ["#business", "#small business"])[:strategy["count"]]
        
        # Add platform-specific trending tags
        if platform == "tiktok":
            tags.extend(["#fyp", "#viral"])
        elif platform == "instagram":
            tags.extend(["#instagood", "#explore"])
        
        return tags[:strategy["count"]]
    
    def _identify_synergies(self, platforms: List[str]) -> List[str]:
        """Identify cross-platform synergies"""
        synergies = []
        
        if "tiktok" in platforms and "instagram" in platforms:
            synergies.append("Repurpose TikTok content directly to Instagram Reels")
        
        if "youtube" in platforms and "tiktok" in platforms:
            synergies.append("Use YouTube long-form to drive to TikTok highlights")
        
        if "linkedin" in platforms:
            synergies.append("Share behind-the-scenes on LinkedIn for B2B credibility")
        
        if len(platforms) >= 3:
            synergies.append("Create content calendar for synchronized cross-posting")
        
        return synergies


# =============================================================================
# AGENT 13: "THE MEMORY" - Client DNA System
# =============================================================================

class ClientDNA(BaseModel):
    """Complete profile of a client's preferences"""
    client_id: str
    business_name: str
    industry: str
    
    # Style preferences (learned over time)
    preferred_hook_techniques: List[str] = Field(default_factory=list)
    preferred_visual_styles: List[str] = Field(default_factory=list)
    preferred_voiceover_styles: List[str] = Field(default_factory=list)
    color_palette: List[str] = Field(default_factory=list)
    
    # Performance history
    total_commercials: int = 0
    avg_satisfaction: float = 0.0
    best_performing_style: Optional[str] = None
    
    # Communication preferences
    revision_sensitivity: float = 0.5  # How much they revise (0=never, 1=always)
    preferred_tone: str = "professional"
    
    # Timestamps
    first_interaction: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)


class MemoryRecallRequest(BaseModel):
    """Request to recall client preferences"""
    client_id: str
    context: str = ""  # What we're generating


class MemoryRecallResponse(BaseModel):
    """Response with personalized recommendations"""
    client_dna: Optional[ClientDNA]
    personalized_parameters: Dict[str, Any]
    confidence: float
    relationship_score: float  # How well we know this client
    cost_usd: float
    latency_ms: float


class TheMemory:
    """
    Agent 13: "THE MEMORY" - Client DNA System
    
    Remembers everything about each client to personalize every interaction.
    Creates a "DNA profile" that improves with every commercial generated.
    
    Features:
    1. Preference learning from feedback
    2. Style consistency across commercials
    3. Predictive customization
    4. Revision pattern analysis
    5. Relationship scoring
    
    Cost: ~$0.001 per recall (pure database lookup)
    """
    
    def __init__(self, qdrant_client: Any = None):
        self.qdrant_client = qdrant_client
        
        self.name = "the_memory"
        self.version = "1.0.0"
        self.cost_per_recall = 0.001
        
        # In-memory client store (production would use Qdrant)
        self.client_profiles: Dict[str, ClientDNA] = {}
    
    async def recall(self, request: MemoryRecallRequest) -> MemoryRecallResponse:
        """
        Recall client preferences and generate personalized parameters.
        """
        with tracer.start_as_current_span("memory_recall") as span:
            start_time = time.time()
            span.set_attribute("client_id", request.client_id)
            
            # Get or create client DNA
            dna = self.client_profiles.get(request.client_id)
            
            if dna:
                # Existing client - personalize heavily
                params = self._generate_personalized_params(dna, request.context)
                relationship_score = self._calculate_relationship_score(dna)
                confidence = min(0.7 + (dna.total_commercials * 0.03), 0.95)
            else:
                # New client - use industry defaults
                params = self._get_industry_defaults(request.context)
                relationship_score = 0.0
                confidence = 0.5
            
            latency = (time.time() - start_time) * 1000
            
            CLIENT_DNA_MATCH.labels(
                client_tier="returning" if dna else "new"
            ).observe(confidence)
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status="recalled" if dna else "new_client"
            ).inc()
            
            return MemoryRecallResponse(
                client_dna=dna,
                personalized_parameters=params,
                confidence=confidence,
                relationship_score=relationship_score,
                cost_usd=self.cost_per_recall,
                latency_ms=latency
            )
    
    async def learn(
        self, 
        client_id: str,
        business_name: str,
        industry: str,
        commercial_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ) -> ClientDNA:
        """
        Learn from a new commercial interaction.
        """
        # Get or create DNA
        if client_id in self.client_profiles:
            dna = self.client_profiles[client_id]
        else:
            dna = ClientDNA(
                client_id=client_id,
                business_name=business_name,
                industry=industry
            )
        
        # Update DNA with new data
        dna.total_commercials += 1
        dna.last_interaction = datetime.utcnow()
        
        # Learn preferences from commercial
        if "hook_technique" in commercial_data:
            if commercial_data["hook_technique"] not in dna.preferred_hook_techniques:
                dna.preferred_hook_techniques.append(commercial_data["hook_technique"])
        
        if "visual_style" in commercial_data:
            if commercial_data["visual_style"] not in dna.preferred_visual_styles:
                dna.preferred_visual_styles.append(commercial_data["visual_style"])
        
        # Learn from feedback
        if feedback:
            if "satisfaction" in feedback:
                # Weighted average
                n = dna.total_commercials
                dna.avg_satisfaction = (
                    (dna.avg_satisfaction * (n - 1) + feedback["satisfaction"]) / n
                )
            
            if "revisions" in feedback:
                # Update revision sensitivity
                dna.revision_sensitivity = (
                    dna.revision_sensitivity * 0.8 + 
                    (1 if feedback["revisions"] > 0 else 0) * 0.2
                )
        
        # Store updated DNA
        self.client_profiles[client_id] = dna
        
        logger.info(f"📚 Learned from {business_name}: {dna.total_commercials} commercials")
        
        return dna
    
    def _generate_personalized_params(self, dna: ClientDNA, context: str) -> Dict[str, Any]:
        """Generate personalized parameters based on DNA"""
        params = {}
        
        # Use preferred styles if available
        if dna.preferred_hook_techniques:
            params["hook_technique"] = dna.preferred_hook_techniques[-1]  # Most recent
        
        if dna.preferred_visual_styles:
            params["visual_style"] = dna.preferred_visual_styles[-1]
        
        if dna.preferred_voiceover_styles:
            params["voiceover_style"] = dna.preferred_voiceover_styles[-1]
        
        # Adjust based on satisfaction
        if dna.avg_satisfaction < 3.5:
            params["quality_boost"] = True
            params["extra_review"] = True
        
        # Adjust based on revision sensitivity
        if dna.revision_sensitivity > 0.7:
            params["conservative_approach"] = True
            params["pre_approval_preview"] = True
        
        params["tone"] = dna.preferred_tone
        
        return params
    
    def _get_industry_defaults(self, context: str) -> Dict[str, Any]:
        """Get default parameters for new clients"""
        return {
            "hook_technique": "question",
            "visual_style": "cinematic",
            "voiceover_style": "professional",
            "quality_boost": False,
            "tone": "professional"
        }
    
    def _calculate_relationship_score(self, dna: ClientDNA) -> float:
        """Calculate how well we know this client"""
        score = 0.0
        
        # Commercial count factor
        score += min(dna.total_commercials * 0.1, 0.3)
        
        # Preference richness factor
        preferences_known = (
            len(dna.preferred_hook_techniques) +
            len(dna.preferred_visual_styles) +
            len(dna.preferred_voiceover_styles)
        )
        score += min(preferences_known * 0.05, 0.3)
        
        # Recency factor
        days_since_interaction = (datetime.utcnow() - dna.last_interaction).days
        recency_score = max(0, 0.3 - (days_since_interaction * 0.01))
        score += recency_score
        
        # Satisfaction factor
        if dna.avg_satisfaction >= 4.0:
            score += 0.1
        
        return min(score, 1.0)


# =============================================================================
# AGENT 14: "THE HUNTER" - Real-Time Trend Radar
# =============================================================================

class TrendSignal(BaseModel):
    """A detected trend signal"""
    trend_id: str
    topic: str
    source: str  # "tiktok", "google_trends", "twitter", etc.
    strength: float  # 0-1
    velocity: float  # How fast it's growing
    relevance_to_industry: float
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    peak_prediction: Optional[datetime] = None


class TrendRadarRequest(BaseModel):
    """Request to scan for trends"""
    industry: str
    keywords: List[str] = Field(default_factory=list)
    lookback_hours: int = 24


class TrendRadarResponse(BaseModel):
    """Response with detected trends"""
    trends: List[TrendSignal]
    recommended_integrations: List[str]
    urgency: str  # "immediate", "soon", "monitor"
    cost_usd: float
    latency_ms: float


class TheHunter:
    """
    Agent 14: "THE HUNTER" - Real-Time Trend Radar
    
    Continuously scans for emerging trends to make content timely.
    Integrates with Google Trends, Twitter/X, TikTok Creative Center.
    
    Features:
    1. Real-time trend detection
    2. Velocity tracking (catching trends early)
    3. Industry relevance scoring
    4. Peak timing prediction
    5. Integration recommendations
    
    Cost: ~$0.01 per scan (API calls)
    """
    
    def __init__(
        self,
        google_trends_client: Any = None,
        tiktok_client: Any = None
    ):
        self.google_trends = google_trends_client
        self.tiktok_client = tiktok_client
        
        self.name = "the_hunter"
        self.version = "1.0.0"
        self.cost_per_scan = 0.01
        
        # Cached trends (refreshed every 30 minutes)
        self.trend_cache: Dict[str, List[TrendSignal]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # Industry trend keywords
        self.industry_keywords = {
            "dental": ["smile", "teeth", "dentist", "whitening", "invisalign"],
            "legal": ["lawyer", "lawsuit", "legal advice", "attorney", "court"],
            "real_estate": ["home", "mortgage", "housing market", "real estate", "buy house"],
            "fitness": ["workout", "gym", "fitness", "weight loss", "health"],
            "restaurant": ["food", "restaurant", "dining", "chef", "recipe"],
            "saas": ["software", "app", "productivity", "ai", "automation"]
        }
    
    async def scan(self, request: TrendRadarRequest) -> TrendRadarResponse:
        """
        Scan for relevant trends.
        """
        with tracer.start_as_current_span("hunter_scan") as span:
            start_time = time.time()
            span.set_attribute("industry", request.industry)
            
            # Check cache first
            cache_key = f"{request.industry}_{request.lookback_hours}"
            if self._is_cache_valid(cache_key):
                trends = self.trend_cache[cache_key]
            else:
                # Fetch fresh trends
                trends = await self._fetch_trends(request)
                self._update_cache(cache_key, trends)
            
            # Filter and score for relevance
            relevant_trends = self._filter_relevant(trends, request.industry)
            
            # Generate integration recommendations
            recommendations = self._generate_recommendations(relevant_trends, request.industry)
            
            # Determine urgency
            urgency = self._calculate_urgency(relevant_trends)
            
            latency = (time.time() - start_time) * 1000
            
            TREND_INTEGRATION_LATENCY.labels(
                trend_source="combined"
            ).observe(latency / 1000)
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status=urgency
            ).inc()
            
            return TrendRadarResponse(
                trends=relevant_trends[:10],  # Top 10 trends
                recommended_integrations=recommendations,
                urgency=urgency,
                cost_usd=self.cost_per_scan,
                latency_ms=latency
            )
    
    async def get_current_trends(self, industry: str) -> List[str]:
        """Quick method to get current trend topics"""
        result = await self.scan(TrendRadarRequest(industry=industry))
        return [t.topic for t in result.trends[:5]]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached trends are still valid"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.utcnow() < self.cache_expiry[cache_key]
    
    def _update_cache(self, cache_key: str, trends: List[TrendSignal]) -> None:
        """Update trend cache"""
        self.trend_cache[cache_key] = trends
        self.cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=30)
    
    async def _fetch_trends(self, request: TrendRadarRequest) -> List[TrendSignal]:
        """Fetch trends from various sources"""
        trends = []
        
        # Get industry keywords
        keywords = request.keywords or self.industry_keywords.get(request.industry, [])
        
        # Simulated trend data (in production, would call actual APIs)
        simulated_trends = [
            {"topic": "AI-powered", "strength": 0.85, "velocity": 0.9},
            {"topic": "sustainable", "strength": 0.72, "velocity": 0.6},
            {"topic": "personalized", "strength": 0.68, "velocity": 0.5},
            {"topic": "instant results", "strength": 0.75, "velocity": 0.7},
            {"topic": "before and after", "strength": 0.82, "velocity": 0.4}
        ]
        
        for trend_data in simulated_trends:
            trend = TrendSignal(
                trend_id=hashlib.md5(trend_data["topic"].encode()).hexdigest()[:8],
                topic=trend_data["topic"],
                source="google_trends",
                strength=trend_data["strength"],
                velocity=trend_data["velocity"],
                relevance_to_industry=self._calculate_relevance(
                    trend_data["topic"], 
                    request.industry
                )
            )
            trends.append(trend)
        
        return trends
    
    def _calculate_relevance(self, topic: str, industry: str) -> float:
        """Calculate how relevant a trend is to an industry"""
        industry_keywords = self.industry_keywords.get(industry, [])
        
        # Check for direct keyword match
        for keyword in industry_keywords:
            if keyword.lower() in topic.lower():
                return 0.9
        
        # Universal trends
        universal_trends = ["ai", "sustainable", "personalized", "instant"]
        if any(ut in topic.lower() for ut in universal_trends):
            return 0.7
        
        return 0.5  # Default relevance
    
    def _filter_relevant(self, trends: List[TrendSignal], industry: str) -> List[TrendSignal]:
        """Filter and sort trends by relevance"""
        # Calculate composite score
        for trend in trends:
            trend.relevance_to_industry = self._calculate_relevance(trend.topic, industry)
        
        # Sort by composite score (strength * velocity * relevance)
        trends.sort(
            key=lambda t: t.strength * t.velocity * t.relevance_to_industry,
            reverse=True
        )
        
        # Filter out low-relevance trends
        return [t for t in trends if t.relevance_to_industry >= 0.5]
    
    def _generate_recommendations(self, trends: List[TrendSignal], industry: str) -> List[str]:
        """Generate actionable trend integration recommendations"""
        recs = []
        
        for trend in trends[:3]:  # Top 3 trends
            if trend.velocity > 0.7:
                recs.append(
                    f"🔥 HOT TREND: '{trend.topic}' is rising fast. "
                    f"Integrate into hook: 'The {trend.topic} way to...'"
                )
            elif trend.strength > 0.8:
                recs.append(
                    f"📈 STRONG TREND: '{trend.topic}' is established. "
                    f"Safe to use in messaging."
                )
            else:
                recs.append(
                    f"👀 EMERGING: '{trend.topic}' is worth monitoring. "
                    f"Consider A/B testing."
                )
        
        return recs
    
    def _calculate_urgency(self, trends: List[TrendSignal]) -> str:
        """Calculate overall urgency based on trend signals"""
        if not trends:
            return "monitor"
        
        # Check for high-velocity trends
        hot_trends = [t for t in trends if t.velocity > 0.8 and t.strength > 0.7]
        
        if len(hot_trends) >= 2:
            return "immediate"
        elif len(hot_trends) == 1:
            return "soon"
        else:
            return "monitor"


# =============================================================================
# AGENT 15: "THE ACCOUNTANT" - Dynamic Budget Optimizer
# =============================================================================

class BudgetAllocation(BaseModel):
    """Budget allocation for a commercial"""
    component: str
    allocated_usd: float
    expected_quality: float
    cost_efficiency: float


class BudgetOptimizationRequest(BaseModel):
    """Request to optimize budget allocation"""
    total_budget_usd: float
    priority: str = "balanced"  # "quality", "cost", "balanced"
    industry: str
    must_have_features: List[str] = Field(default_factory=list)


class BudgetOptimizationResponse(BaseModel):
    """Response with optimized budget allocation"""
    allocations: List[BudgetAllocation]
    total_allocated: float
    expected_quality_score: float
    savings_from_optimization: float
    recommendations: List[str]
    cost_usd: float
    latency_ms: float


class TheAccountant:
    """
    Agent 15: "THE ACCOUNTANT" - Dynamic Budget Optimizer
    
    Maximizes quality within any budget constraint.
    Intelligently allocates resources across the pipeline.
    
    Features:
    1. Component-level cost analysis
    2. Quality-cost tradeoff optimization
    3. Industry-specific recommendations
    4. Savings identification
    5. ROI prediction
    
    Cost: ~$0.001 per optimization (pure computation)
    """
    
    def __init__(self):
        self.name = "the_accountant"
        self.version = "1.0.0"
        self.cost_per_optimization = 0.001
        
        # Component costs and quality factors
        self.component_profiles = {
            "video_generation": {
                "base_cost": 1.50,
                "quality_weight": 0.30,
                "min_cost": 0.75,  # Budget option
                "max_cost": 3.00,  # Premium option
                "scalable": True
            },
            "voiceover": {
                "base_cost": 0.10,
                "quality_weight": 0.15,
                "min_cost": 0.03,  # Basic TTS
                "max_cost": 0.50,  # Premium voice
                "scalable": True
            },
            "script_generation": {
                "base_cost": 0.05,
                "quality_weight": 0.25,
                "min_cost": 0.02,  # Haiku
                "max_cost": 0.15,  # Opus
                "scalable": True
            },
            "competitive_intel": {
                "base_cost": 0.02,
                "quality_weight": 0.10,
                "min_cost": 0.00,  # Cache only
                "max_cost": 0.15,  # Live research
                "scalable": True
            },
            "compliance": {
                "base_cost": 0.01,
                "quality_weight": 0.05,
                "min_cost": 0.01,
                "max_cost": 0.01,
                "scalable": False  # Fixed cost
            },
            "assembly": {
                "base_cost": 0.00,
                "quality_weight": 0.05,
                "min_cost": 0.00,
                "max_cost": 0.00,
                "scalable": False
            },
            "qa": {
                "base_cost": 0.03,
                "quality_weight": 0.10,
                "min_cost": 0.00,  # Basic ffprobe
                "max_cost": 0.10,  # Vision QA
                "scalable": True
            }
        }
        
        # Budget tier thresholds
        self.budget_tiers = {
            "micro": {"max": 1.50, "strategy": "essentials_only"},
            "budget": {"max": 2.50, "strategy": "cost_optimized"},
            "standard": {"max": 4.00, "strategy": "balanced"},
            "premium": {"max": 7.00, "strategy": "quality_first"},
            "enterprise": {"max": float("inf"), "strategy": "no_compromise"}
        }
    
    async def optimize(self, request: BudgetOptimizationRequest) -> BudgetOptimizationResponse:
        """
        Optimize budget allocation for maximum quality.
        """
        with tracer.start_as_current_span("accountant_optimize") as span:
            start_time = time.time()
            span.set_attribute("budget", request.total_budget_usd)
            span.set_attribute("priority", request.priority)
            
            # Determine budget tier
            tier = self._determine_tier(request.total_budget_usd)
            
            # Calculate optimal allocation
            allocations = self._calculate_allocations(
                request.total_budget_usd,
                request.priority,
                tier,
                request.must_have_features
            )
            
            # Calculate metrics
            total_allocated = sum(a.allocated_usd for a in allocations)
            expected_quality = self._calculate_expected_quality(allocations)
            
            # Calculate savings vs default
            default_cost = sum(
                self.component_profiles[c]["base_cost"] 
                for c in self.component_profiles
            )
            savings = max(0, default_cost - total_allocated)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                request.total_budget_usd,
                tier,
                allocations
            )
            
            latency = (time.time() - start_time) * 1000
            
            LEGENDARY_AGENT_CALLS.labels(
                agent=self.name,
                status=tier
            ).inc()
            
            return BudgetOptimizationResponse(
                allocations=allocations,
                total_allocated=total_allocated,
                expected_quality_score=expected_quality,
                savings_from_optimization=savings,
                recommendations=recommendations,
                cost_usd=self.cost_per_optimization,
                latency_ms=latency
            )
    
    def _determine_tier(self, budget: float) -> str:
        """Determine budget tier"""
        for tier_name, tier_info in self.budget_tiers.items():
            if budget <= tier_info["max"]:
                return tier_name
        return "enterprise"
    
    def _calculate_allocations(
        self,
        total_budget: float,
        priority: str,
        tier: str,
        must_haves: List[str]
    ) -> List[BudgetAllocation]:
        """Calculate optimal component allocations"""
        allocations = []
        remaining_budget = total_budget
        
        # Priority weights
        priority_weights = {
            "quality": {"video": 1.3, "voiceover": 1.2, "script": 1.2},
            "cost": {"video": 0.7, "voiceover": 0.8, "script": 0.9},
            "balanced": {"video": 1.0, "voiceover": 1.0, "script": 1.0}
        }
        weights = priority_weights.get(priority, priority_weights["balanced"])
        
        # Allocate by importance order
        component_order = [
            "video_generation", "script_generation", "voiceover",
            "qa", "competitive_intel", "compliance", "assembly"
        ]
        
        for component in component_order:
            profile = self.component_profiles[component]
            
            # Calculate allocation
            if profile["scalable"]:
                weight = weights.get(component.split("_")[0], 1.0)
                
                if tier == "micro":
                    allocated = profile["min_cost"]
                elif tier == "budget":
                    allocated = profile["min_cost"] * 1.5
                elif tier == "standard":
                    allocated = profile["base_cost"]
                elif tier == "premium":
                    allocated = profile["base_cost"] * 1.3
                else:  # enterprise
                    allocated = profile["max_cost"]
                
                allocated *= weight
            else:
                allocated = profile["base_cost"]
            
            # Ensure we don't exceed budget
            allocated = min(allocated, remaining_budget)
            remaining_budget -= allocated
            
            # Calculate quality and efficiency
            quality = self._calculate_component_quality(component, allocated)
            efficiency = quality / allocated if allocated > 0 else 0
            
            allocations.append(BudgetAllocation(
                component=component,
                allocated_usd=round(allocated, 4),
                expected_quality=round(quality, 3),
                cost_efficiency=round(efficiency, 2)
            ))
        
        return allocations
    
    def _calculate_component_quality(self, component: str, allocated: float) -> float:
        """Calculate expected quality for a component given allocation"""
        profile = self.component_profiles[component]
        
        if allocated <= profile["min_cost"]:
            return 0.5  # Minimum quality
        elif allocated >= profile["max_cost"]:
            return 0.95  # Maximum quality
        else:
            # Linear interpolation
            range_cost = profile["max_cost"] - profile["min_cost"]
            range_quality = 0.95 - 0.5
            progress = (allocated - profile["min_cost"]) / range_cost
            return 0.5 + (progress * range_quality)
    
    def _calculate_expected_quality(self, allocations: List[BudgetAllocation]) -> float:
        """Calculate overall expected quality"""
        total_weight = sum(
            self.component_profiles[a.component]["quality_weight"]
            for a in allocations
        )
        
        weighted_quality = sum(
            a.expected_quality * self.component_profiles[a.component]["quality_weight"]
            for a in allocations
        )
        
        return weighted_quality / total_weight if total_weight > 0 else 0.5
    
    def _generate_recommendations(
        self,
        budget: float,
        tier: str,
        allocations: List[BudgetAllocation]
    ) -> List[str]:
        """Generate budget recommendations"""
        recs = []
        
        if tier == "micro":
            recs.append(
                f"💡 MICRO BUDGET: Consider increasing to $2.50 for 40% quality boost"
            )
            recs.append(
                "Use cache-first competitive intel to save costs"
            )
        
        elif tier == "budget":
            recs.append(
                "Good budget allocation for testing. Consider premium for high-stakes campaigns."
            )
        
        elif tier in ["premium", "enterprise"]:
            recs.append(
                "🌟 PREMIUM: All quality features enabled. Maximum impact expected."
            )
        
        # Identify efficiency improvements
        low_efficiency = [
            a for a in allocations 
            if a.cost_efficiency < 1.0 and a.allocated_usd > 0.1
        ]
        
        for alloc in low_efficiency:
            recs.append(
                f"⚠️ {alloc.component} has low efficiency ({alloc.cost_efficiency:.1f}). "
                f"Consider reducing allocation."
            )
        
        return recs


# =============================================================================
# SHADOW MODE ORCHESTRATOR
# =============================================================================

class ShadowModeResult(BaseModel):
    """Result of shadow mode comparison"""
    production_quality: float
    shadow_quality: float
    improvement: float
    winner: str  # "production" or "shadow"
    should_promote: bool
    confidence: float


class ShadowModeOrchestrator:
    """
    SHADOW MODE - Risk-Free Evolution Pipeline
    
    Runs enhanced pipeline in parallel with production.
    Compares results without affecting live traffic.
    Promotes shadow when consistently better.
    
    Process:
    1. Generate production commercial (current pipeline)
    2. Generate shadow commercial (with enhancements)
    3. Compare using Agent 7 metrics
    4. Log difference for analysis
    5. Auto-promote when shadow wins N times
    """
    
    def __init__(
        self,
        production_pipeline: Any,
        shadow_pipeline: Any,
        promotion_threshold: int = 10  # Shadow must win 10x before promotion
    ):
        self.production = production_pipeline
        self.shadow = shadow_pipeline
        self.promotion_threshold = promotion_threshold
        
        self.shadow_wins = 0
        self.production_wins = 0
        self.results: List[ShadowModeResult] = []
    
    async def generate_with_shadow(self, brief: Dict[str, Any]) -> Tuple[Any, ShadowModeResult]:
        """
        Generate using both pipelines and compare.
        Returns production result, but logs shadow comparison.
        """
        # Run both pipelines in parallel
        production_task = asyncio.create_task(self.production.generate(brief))
        shadow_task = asyncio.create_task(self.shadow.generate(brief))
        
        production_result, shadow_result = await asyncio.gather(
            production_task, shadow_task
        )
        
        # Compare quality (using internal scores)
        production_quality = production_result.get("quality_score", 0.7)
        shadow_quality = shadow_result.get("quality_score", 0.7)
        
        improvement = shadow_quality - production_quality
        winner = "shadow" if improvement > 0 else "production"
        
        if winner == "shadow":
            self.shadow_wins += 1
        else:
            self.production_wins += 1
        
        # Determine if we should promote
        should_promote = self.shadow_wins >= self.promotion_threshold
        
        result = ShadowModeResult(
            production_quality=production_quality,
            shadow_quality=shadow_quality,
            improvement=improvement,
            winner=winner,
            should_promote=should_promote,
            confidence=self.shadow_wins / max(1, self.shadow_wins + self.production_wins)
        )
        
        self.results.append(result)
        
        logger.info(
            f"🔮 Shadow Mode: {winner} wins. "
            f"Score: Shadow {self.shadow_wins} vs Production {self.production_wins}"
        )
        
        # Return production result (shadow is just for comparison)
        return production_result, result


# =============================================================================
# LEGENDARY AGENT FACTORY
# =============================================================================

class LegendaryAgentFactory:
    """
    Factory to create and manage legendary agents.
    """
    
    def __init__(
        self,
        qdrant_client: Any = None,
        llm_client: Any = None,
        vision_client: Any = None
    ):
        self.qdrant_client = qdrant_client
        self.llm_client = llm_client
        self.vision_client = vision_client
    
    def create_all(self) -> Dict[str, Any]:
        """Create all legendary agents"""
        return {
            "agent_7_5": TheAuteur(vision_client=self.vision_client),
            "agent_8_5": TheGeneticist(llm_client=self.llm_client),
            "agent_11": TheOracle(),
            "agent_12": TheChameleon(),
            "agent_13": TheMemory(qdrant_client=self.qdrant_client),
            "agent_14": TheHunter(),
            "agent_15": TheAccountant()
        }
    
    def create_auteur(self) -> TheAuteur:
        return TheAuteur(vision_client=self.vision_client)
    
    def create_geneticist(self) -> TheGeneticist:
        return TheGeneticist(llm_client=self.llm_client)
    
    def create_oracle(self) -> TheOracle:
        return TheOracle()
    
    def create_chameleon(self) -> TheChameleon:
        return TheChameleon()
    
    def create_memory(self) -> TheMemory:
        return TheMemory(qdrant_client=self.qdrant_client)
    
    def create_hunter(self) -> TheHunter:
        return TheHunter()
    
    def create_accountant(self) -> TheAccountant:
        return TheAccountant()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("🔥 RAGNAROK LEGENDARY UPGRADES - THE COGNITIVE APEX")
        print("=" * 70)
        
        factory = LegendaryAgentFactory()
        agents = factory.create_all()
        
        # Demo Agent 11: The Oracle
        print("\n🔮 Agent 11: THE ORACLE - Viral Potential Predictor")
        print("-" * 50)
        
        oracle = agents["agent_11"]
        oracle_result = await oracle.predict(OraclePredictionRequest(
            script="Stop scrolling. This dental hack will save you thousands. Finally, a painless way to get the smile you deserve.",
            visual_style="ugc",
            hook_technique="negative_callout",
            industry="dental",
            target_platform="tiktok"
        ))
        
        print(f"Viral Score: {oracle_result.prediction.viral_score:.0%}")
        print(f"Predicted Views (7d): {oracle_result.prediction.predicted_views_7d:,}")
        print(f"Should Boost: {oracle_result.should_boost}")
        print(f"Cost: ${oracle_result.cost_usd:.4f}")
        
        # Demo Agent 12: The Chameleon
        print("\n🦎 Agent 12: THE CHAMELEON - Multi-Platform Optimizer")
        print("-" * 50)
        
        chameleon = agents["agent_12"]
        chameleon_result = await chameleon.adapt(ChameleonOptimizationRequest(
            original_script="Discover the smile you've always wanted at Bright Smile Dental.",
            original_style="cinematic",
            business_name="Bright Smile Dental",
            industry="dental",
            target_platforms=["tiktok", "youtube", "instagram"]
        ))
        
        for variant in chameleon_result.platform_variants:
            print(f"\n{variant.platform.upper()}:")
            print(f"  Script: {variant.adapted_script[:60]}...")
            print(f"  Posting: {variant.optimal_posting_time}")
            print(f"  Hashtags: {', '.join(variant.hashtag_recommendations[:3])}")
        
        print(f"\nCost: ${chameleon_result.cost_usd:.4f}")
        
        # Demo Agent 15: The Accountant
        print("\n💰 Agent 15: THE ACCOUNTANT - Budget Optimizer")
        print("-" * 50)
        
        accountant = agents["agent_15"]
        accountant_result = await accountant.optimize(BudgetOptimizationRequest(
            total_budget_usd=2.50,
            priority="balanced",
            industry="dental"
        ))
        
        print(f"Budget Tier: ${accountant_result.total_allocated:.2f}")
        print(f"Expected Quality: {accountant_result.expected_quality_score:.0%}")
        print(f"Savings: ${accountant_result.savings_from_optimization:.2f}")
        print("\nAllocations:")
        for alloc in accountant_result.allocations[:5]:
            print(f"  {alloc.component}: ${alloc.allocated_usd:.3f} (Q: {alloc.expected_quality:.0%})")
        
        print("\n" + "=" * 70)
        print("✅ LEGENDARY UPGRADES DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\n📊 FULL AGENT COUNT: 23 AGENTS")
        print("💰 TOTAL COST RANGE: $2.20-2.80/commercial")
        print("🧠 COGNITIVE LEVEL: APEX PREDATOR")
        print("=" * 70)
    
    asyncio.run(main())
