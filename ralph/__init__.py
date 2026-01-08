"""
Ralph System - Iterative Agent Refinement for RAGNAROK Pipeline

Based on Geoffrey Huntley's Ralph System (June 2025)
Implemented for Barrios A2I RAGNAROK v8.0

Components:
- RalphLoopController: Main iteration orchestrator
- RalphAgentWrapper: Wraps agents in Ralph loops
- QualityGate: Pipeline-level iteration decisions

Key Features:
- Iterative refinement until quality thresholds met
- Self-correction based on previous failures
- Convergence guarantees with max iteration limits
- Filesystem persistence for audit trail and recovery

Usage:
    from ralph import get_ralph_wrapper, quality_gate, GateDecision

    # Wrap an agent
    wrapper = get_ralph_wrapper('story_creator', my_agent.generate)
    result = await wrapper.execute(task, context)

    # Check quality gate after QA
    gate_result = quality_gate.evaluate(
        auteur_score=78,
        technical_qa={'status': 'PASSED'},
        pipeline_iteration=1
    )

    if gate_result.decision == GateDecision.PASS:
        proceed_to_delivery()
    else:
        rerun_from_phase(gate_result.phases_to_rerun[0])
"""

from .loop_controller import (
    RalphConfig,
    RalphLoopController,
    RalphState,
    IterationResult,
    ralph_loop
)

from .agent_wrapper import (
    AgentRalphConfig,
    RalphAgentWrapper,
    get_ralph_wrapper,
    is_ralph_enabled,
    get_agent_config,
    AGENT_RALPH_CONFIGS,
    AGENT_EVALUATORS,
    evaluate_script,
    evaluate_prompts,
    evaluate_video_scene,
    evaluate_research,
    evaluate_auteur_qa
)

from .quality_gate import (
    GateDecision,
    GateResult,
    QualityGate,
    quality_gate
)

__version__ = "1.0.0"
__author__ = "Barrios A2I"

__all__ = [
    # Loop Controller
    'RalphConfig',
    'RalphLoopController',
    'RalphState',
    'IterationResult',
    'ralph_loop',

    # Agent Wrapper
    'AgentRalphConfig',
    'RalphAgentWrapper',
    'get_ralph_wrapper',
    'is_ralph_enabled',
    'get_agent_config',
    'AGENT_RALPH_CONFIGS',
    'AGENT_EVALUATORS',
    'evaluate_script',
    'evaluate_prompts',
    'evaluate_video_scene',
    'evaluate_research',
    'evaluate_auteur_qa',

    # Quality Gate
    'GateDecision',
    'GateResult',
    'QualityGate',
    'quality_gate'
]
