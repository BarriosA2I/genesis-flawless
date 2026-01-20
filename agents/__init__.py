"""
================================================================================
RAGNAROK v7.5 LEGENDARY - Agent Module
================================================================================
Enhancement, Strategic, Legendary, and Post-Production agents for the
commercial video pipeline.

Phase 1 (Enhancement): Agents 0.75, 1.5, 3.5, 5.5, 6.5
Phase 2 (Strategic): Agents 8, 9, 10
Phase 3 (Legendary): Agents 7.5, 8.5, 11-15
Phase 4 (Post-Production): Agents 7.75 (EDITOR), 6.5 (SOUNDSCAPER), 7.25 (WORDSMITH)

Author: Barrios A2I | Version: 7.5.0 | January 2026
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

# VORTEX Post-Production Agents (Phase 4)
try:
    from .vortex_postprod import (
        # Orchestrator
        VortexOrchestrator,
        VortexStateMachine,
        ProcessingMode,
        PipelinePhase,
        GlobalState,
        VideoMetadata,
        BriefData,
        create_vortex_orchestrator,
        # Agent 7.75: THE EDITOR
        TheEditor,
        create_editor,
        TransitionType,
        StylePreset,
        ColorGradingIntent,
        ShotType,
        # Agent 6.5: THE SOUNDSCAPER (VORTEX)
        TheSoundscaper,
        create_soundscaper,
        SFXCategory,
        MoodProfile,
        SoundscapeRequest,
        SoundscapeResult,
        # Agent 7.25: THE WORDSMITH
        TheWordsmith,
        create_wordsmith,
        WordsmithConfig,
        TextValidationRequest,
        TextValidationResult,
    )
    VORTEX_POSTPROD_AVAILABLE = True
except ImportError as e:
    VORTEX_POSTPROD_AVAILABLE = False
    VortexOrchestrator = None
    TheEditor = None
    TheSoundscaper = None
    TheWordsmith = None

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
    # VORTEX Post-Production (Phase 4)
    "VORTEX_POSTPROD_AVAILABLE",
    # Orchestrator
    "VortexOrchestrator",
    "VortexStateMachine",
    "ProcessingMode",
    "PipelinePhase",
    "GlobalState",
    "VideoMetadata",
    "BriefData",
    "create_vortex_orchestrator",
    # Agent 7.75: THE EDITOR
    "TheEditor",
    "create_editor",
    "TransitionType",
    "StylePreset",
    "ColorGradingIntent",
    "ShotType",
    # Agent 6.5 (VORTEX): THE SOUNDSCAPER
    "TheSoundscaper",
    "create_soundscaper",
    "SFXCategory",
    "MoodProfile",
    "SoundscapeRequest",
    "SoundscapeResult",
    # Agent 7.25: THE WORDSMITH
    "TheWordsmith",
    "create_wordsmith",
    "WordsmithConfig",
    "TextValidationRequest",
    "TextValidationResult",
]
