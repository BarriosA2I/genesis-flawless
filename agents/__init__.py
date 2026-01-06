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
    LegendaryAgentFactory,
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
]
