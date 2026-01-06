"""
================================================================================
RAGNAROK v4.0 LEGENDARY - Agent Module
================================================================================
Enhancement, Strategic, and Legendary agents for the commercial video pipeline.

Phase 1 (Enhancement): Agents 0.75, 1.5, 3.5, 5.5, 6.5
Phase 2 (Strategic): Agents 8, 9, 10
Phase 3 (Legendary): Agents 7.5, 8.5, 11-15

Author: Barrios A2I | Version: 4.0.0 | January 2026
================================================================================
"""

# Enhancement Agents (Phase 1)
from .ragnarok_enhancement_agents import (
    # Factory
    EnhancementAgentFactory,
    # Agents
    CompetitiveIntelligenceAgent,
    NarrativeArcValidatorAgent,
    PromptMutationEngine,
    SonicBrandingSynthesizer,
    CulturalComplianceValidator,
    # Request/Response Models
    CompetitiveIntelligenceRequest,
    CompetitiveIntelligenceResponse,
    NarrativeValidationRequest,
    NarrativeValidationResponse,
    PromptMutationRequest,
    PromptMutationResponse,
    SonicBrandingRequest,
    SonicBrandingResponse,
    ComplianceValidationRequest,
    ComplianceValidationResponse,
    # Shared Models
    AgentResult,
    CacheStrategy,
)

# Strategic Agents (Phase 2)
from .ragnarok_strategic_agents import (
    # Factory
    StrategicAgentFactory,
    # Agents
    MetaLearningPerformanceOptimizer,
    ABTestingOrchestrator,
    RealTimeFeedbackIntegrator,
    # Request/Response Models
    MetaLearningRequest,
    MetaLearningResponse,
    ABTestRequest,
    ABTestResponse,
    FeedbackRequest,
    FeedbackResponse,
    ClientFeedback,
    # Integration helper
    integrate_strategic_agents,
)

# Legendary Agents (Phase 3)
from .ragnarok_legendary_upgrades import (
    # Factory
    LegendaryAgentFactory,
    # Agent 7.5: THE AUTEUR - Vision-Language Creative QA
    TheAuteur,
    CreativeQARequest,
    CreativeQAResponse,
    CreativeQAIssue,
    # Agent 8.5: THE GENETICIST - DSPy Prompt Self-Optimization
    TheGeneticist,
    GeneticOptimizationRequest,
    GeneticOptimizationResponse,
    PromptGene,
    EvolutionResult,
    # Agent 11: THE ORACLE - Viral Potential Predictor
    TheOracle,
    OraclePredictionRequest,
    OraclePredictionResponse,
    ViralityFactors,
    ViralityPrediction,
    # Agent 12: THE CHAMELEON - Multi-Platform Optimizer
    TheChameleon,
    ChameleonOptimizationRequest,
    ChameleonOptimizationResponse,
    PlatformSpec,
    PlatformOptimizedContent,
    # Agent 13: THE MEMORY - Client DNA System
    TheMemory,
    MemoryRecallRequest,
    MemoryRecallResponse,
    ClientDNA,
    # Agent 14: THE HUNTER - Real-Time Trend Radar
    TheHunter,
    TrendRadarRequest,
    TrendRadarResponse,
    TrendSignal,
    # Agent 15: THE ACCOUNTANT - Dynamic Budget Optimizer
    TheAccountant,
    BudgetOptimizationRequest,
    BudgetOptimizationResponse,
    BudgetAllocation,
    # Shadow Mode Orchestrator
    ShadowModeOrchestrator,
    ShadowModeResult,
)

__all__ = [
    # Enhancement (Phase 1)
    "EnhancementAgentFactory",
    "CompetitiveIntelligenceAgent",
    "NarrativeArcValidatorAgent",
    "PromptMutationEngine",
    "SonicBrandingSynthesizer",
    "CulturalComplianceValidator",
    "CompetitiveIntelligenceRequest",
    "CompetitiveIntelligenceResponse",
    "NarrativeValidationRequest",
    "NarrativeValidationResponse",
    "PromptMutationRequest",
    "PromptMutationResponse",
    "SonicBrandingRequest",
    "SonicBrandingResponse",
    "ComplianceValidationRequest",
    "ComplianceValidationResponse",
    "AgentResult",
    "CacheStrategy",
    # Strategic (Phase 2)
    "StrategicAgentFactory",
    "MetaLearningPerformanceOptimizer",
    "ABTestingOrchestrator",
    "RealTimeFeedbackIntegrator",
    "MetaLearningRequest",
    "MetaLearningResponse",
    "ABTestRequest",
    "ABTestResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "ClientFeedback",
    "integrate_strategic_agents",
    # Legendary (Phase 3)
    "LegendaryAgentFactory",
    # Agent 7.5: THE AUTEUR
    "TheAuteur",
    "CreativeQARequest",
    "CreativeQAResponse",
    "CreativeQAIssue",
    # Agent 8.5: THE GENETICIST
    "TheGeneticist",
    "GeneticOptimizationRequest",
    "GeneticOptimizationResponse",
    "PromptGene",
    "EvolutionResult",
    # Agent 11: THE ORACLE
    "TheOracle",
    "OraclePredictionRequest",
    "OraclePredictionResponse",
    "ViralityFactors",
    "ViralityPrediction",
    # Agent 12: THE CHAMELEON
    "TheChameleon",
    "ChameleonOptimizationRequest",
    "ChameleonOptimizationResponse",
    "PlatformSpec",
    "PlatformOptimizedContent",
    # Agent 13: THE MEMORY
    "TheMemory",
    "MemoryRecallRequest",
    "MemoryRecallResponse",
    "ClientDNA",
    # Agent 14: THE HUNTER
    "TheHunter",
    "TrendRadarRequest",
    "TrendRadarResponse",
    "TrendSignal",
    # Agent 15: THE ACCOUNTANT
    "TheAccountant",
    "BudgetOptimizationRequest",
    "BudgetOptimizationResponse",
    "BudgetAllocation",
    # Shadow Mode
    "ShadowModeOrchestrator",
    "ShadowModeResult",
]
