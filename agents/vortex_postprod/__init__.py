"""
================================================================================
VORTEX Post-Production Agents - v1.0 LEGENDARY
================================================================================
State machine orchestrator + 3 enhancement agents for post-production pipeline.

Agents:
  - Agent 7.75: THE EDITOR - Shot detection, transitions, color grading
  - Agent 6.5:  THE SOUNDSCAPER - SFX, foley, ambient audio
  - Agent 7.25: THE WORDSMITH - Text detection, spelling/grammar QA

Performance: 15K QPS | Sub-200ms P95 | 99.95% Uptime | $0.30-0.60/video
Author: Barrios A2I | RAGNAROK v7.0 APEX
================================================================================
"""

# =============================================================================
# ORCHESTRATOR EXPORTS
# =============================================================================
from .orchestrator import (
    # Main orchestrator classes
    VortexOrchestrator,
    VortexStateMachine,
    StateMachineGraph,

    # Enums
    PipelinePhase,
    ProcessingMode,
    CircuitState,
    EventType,

    # Data models
    GlobalState,
    VideoMetadata,
    BriefData,
    EditorResult,
    SoundscaperResult,
    WordsmithResult,
    StateCheckpoint,
    ExecutionMetrics,
    PipelineError,

    # Infrastructure
    TypedEventBus,
    PipelineEvent,
    CircuitBreaker,
    CircuitBreakerConfig,
    DualProcessRouter,
    RoutingDecision,
    CheckpointManager,

    # Execution
    ExecutionConfig,
    ExecutionContext,

    # Graph nodes
    GraphNode,
    IntakeNode,
    RoutingNode,
    EditorNode,
    SoundscaperNode,
    WordsmithNode,
    VerificationNode,
    OutputNode,
    ErrorNode,

    # Factory functions
    create_vortex_orchestrator,
    create_initial_state,
)

# =============================================================================
# THE EDITOR EXPORTS (Agent 7.75)
# =============================================================================
from .the_editor import (
    TheEditor,
    create_editor,

    # Enums
    TransitionType,
    StylePreset,
    StabilizationMethod,
    ColorGradingIntent,
    ShotType,
    MotionType,
    RhythmPattern,
    EditingSeverity,
)

# =============================================================================
# THE SOUNDSCAPER EXPORTS (Agent 6.5)
# =============================================================================
from .the_soundscaper import (
    TheSoundscaper,
    create_soundscaper,

    # Enums
    SFXCategory,
    MoodProfile,

    # Models
    SceneAnalysis,
    SFXPlacement,
    AudioLayer,
    SoundscapeRequest,
    SoundscapeResult,

    # Constants
    INDUSTRY_AUDIO_PROFILES,
)

# =============================================================================
# THE WORDSMITH EXPORTS (Agent 7.25)
# =============================================================================
from .the_wordsmith import (
    TheWordsmith,
    create_wordsmith,
    WordsmithConfig,

    # Enums
    OCREngine,
    ValidationSeverity,
    ErrorCategory,
    WCAGLevel,
    ColorBlindnessType,
    TextPosition,
    FontCategory,
    CheckFlag,

    # Models
    BoundingBox,
    ColorRGB,
    FontEstimate,
    TextDetection,
    ValidationError,
    CorrectionSuggestion,
    BrandGuideline,
    AccessibilityResult,
    FrameAnalysis,
    TextValidationRequest,
    ValidationSummary,
    TextValidationResult,
)

# =============================================================================
# MODULE INFO
# =============================================================================
__version__ = "1.0.0"
__author__ = "Barrios A2I"
__status__ = "LEGENDARY"

__all__ = [
    # Orchestrator
    "VortexOrchestrator",
    "VortexStateMachine",
    "StateMachineGraph",
    "PipelinePhase",
    "ProcessingMode",
    "CircuitState",
    "EventType",
    "GlobalState",
    "VideoMetadata",
    "BriefData",
    "EditorResult",
    "SoundscaperResult",
    "WordsmithResult",
    "StateCheckpoint",
    "ExecutionMetrics",
    "PipelineError",
    "TypedEventBus",
    "PipelineEvent",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "DualProcessRouter",
    "RoutingDecision",
    "CheckpointManager",
    "ExecutionConfig",
    "ExecutionContext",
    "GraphNode",
    "IntakeNode",
    "RoutingNode",
    "EditorNode",
    "SoundscaperNode",
    "WordsmithNode",
    "VerificationNode",
    "OutputNode",
    "ErrorNode",
    "create_vortex_orchestrator",
    "create_initial_state",

    # The Editor (Agent 7.75)
    "TheEditor",
    "create_editor",
    "TransitionType",
    "StylePreset",
    "StabilizationMethod",
    "ColorGradingIntent",
    "ShotType",
    "MotionType",
    "RhythmPattern",
    "EditingSeverity",

    # The Soundscaper (Agent 6.5)
    "TheSoundscaper",
    "create_soundscaper",
    "SFXCategory",
    "MoodProfile",
    "SceneAnalysis",
    "SFXPlacement",
    "AudioLayer",
    "SoundscapeRequest",
    "SoundscapeResult",
    "INDUSTRY_AUDIO_PROFILES",

    # The Wordsmith (Agent 7.25)
    "TheWordsmith",
    "create_wordsmith",
    "WordsmithConfig",
    "OCREngine",
    "ValidationSeverity",
    "ErrorCategory",
    "WCAGLevel",
    "ColorBlindnessType",
    "TextPosition",
    "FontCategory",
    "CheckFlag",
    "BoundingBox",
    "ColorRGB",
    "FontEstimate",
    "TextDetection",
    "ValidationError",
    "CorrectionSuggestion",
    "BrandGuideline",
    "AccessibilityResult",
    "FrameAnalysis",
    "TextValidationRequest",
    "ValidationSummary",
    "TextValidationResult",
]
