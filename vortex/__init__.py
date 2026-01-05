"""
================================================================================
VORTEX v2.1 LEGENDARY - Video Assembly Pipeline
================================================================================
Integrated into FLAWLESS GENESIS for in-process video assembly.

RAGNAROK Agent 6 - Assembles video clips with professional transitions
into multiple output formats for social media platforms.
================================================================================
"""

__version__ = "2.1.0"

from .state_machine import (
    GlobalState,
    PipelinePhase,
    ProcessingMode,
    TransitionType,
    SceneAnalysis,
    TransitionDecision,
    ComplexityAnalyzer,
    FORMAT_SPECS,
    TRANSITION_XFADE_MAP,
)

from .graph_nodes import (
    AsyncFFmpeg,
    VortexPipeline,
    VortexConfig,
)

__all__ = [
    "__version__",
    "GlobalState",
    "PipelinePhase",
    "ProcessingMode",
    "TransitionType",
    "SceneAnalysis",
    "TransitionDecision",
    "ComplexityAnalyzer",
    "FORMAT_SPECS",
    "TRANSITION_XFADE_MAP",
    "AsyncFFmpeg",
    "VortexPipeline",
    "VortexConfig",
]
