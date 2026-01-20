"""
Ralph Agent Wrapper
Wraps any RAGNAROK agent in a Ralph loop for iterative refinement.

This module provides:
- Per-agent Ralph configurations
- Generic agent wrapper class
- Specialized evaluation functions for each agent type
- Factory function for easy wrapper creation
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import structlog

from .loop_controller import RalphLoopController, RalphConfig, RalphState

logger = structlog.get_logger(__name__)


@dataclass
class AgentRalphConfig:
    """Per-agent Ralph configuration."""
    enabled: bool = True
    max_iterations: int = 5
    completion_threshold: float = 0.85
    timeout_per_iteration: int = 180
    retry_on_failure: bool = True
    min_score_to_continue: float = 0.3  # Stop early if score too low


# Default configurations per agent type
# Based on agent complexity and importance to final quality
AGENT_RALPH_CONFIGS: Dict[str, AgentRalphConfig] = {
    # HIGH PRIORITY - These directly affect creative quality
    'story_creator': AgentRalphConfig(
        enabled=True,
        max_iterations=5,
        completion_threshold=0.90,
        timeout_per_iteration=120
    ),
    'prompt_engineer': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.85,
        timeout_per_iteration=60
    ),
    'video_generator': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.80,
        timeout_per_iteration=300
    ),

    # MEDIUM PRIORITY - Research and analysis
    'trinity_suite': AgentRalphConfig(
        enabled=True,
        max_iterations=5,
        completion_threshold=0.85,
        timeout_per_iteration=180
    ),
    'market_decoder': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.80,
        timeout_per_iteration=120
    ),
    'competitor_intel': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.80,
        timeout_per_iteration=120
    ),
    'viral_pattern': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.80,
        timeout_per_iteration=120
    ),

    # QA AGENTS - Limited iterations
    'the_auteur': AgentRalphConfig(
        enabled=True,
        max_iterations=2,
        completion_threshold=0.85,
        timeout_per_iteration=120
    ),

    # SINGLE-PASS AGENTS - No Ralph loop needed
    'intake_qualifier': AgentRalphConfig(enabled=False),
    'voiceover': AgentRalphConfig(enabled=False),  # External API
    'music_selector': AgentRalphConfig(enabled=False),  # Selection task
    'video_assembly': AgentRalphConfig(enabled=False),  # FFmpeg process
    'qa_validator': AgentRalphConfig(enabled=False),  # Technical checks
    'enhancement_suite': AgentRalphConfig(enabled=False),  # Post-processing

    # ENHANCEMENT AGENTS - Single-pass
    'performance_optimizer': AgentRalphConfig(enabled=False),
    'emotional_resonance': AgentRalphConfig(enabled=False),
    'brand_consistency': AgentRalphConfig(enabled=False),
    'platform_adaptor': AgentRalphConfig(enabled=False),
    'accessibility_expert': AgentRalphConfig(enabled=False),
    'legal_compliance': AgentRalphConfig(enabled=False),
    'cultural_sensitivity': AgentRalphConfig(enabled=False),
    'trend_integrator': AgentRalphConfig(enabled=False),
    'final_polish': AgentRalphConfig(enabled=False),

    # ============================================================
    # VORTEX POST-PRODUCTION AGENTS - Enable Ralph loops for v8.0
    # ============================================================
    'the_wordsmith': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.95,  # High - spelling must be perfect
        timeout_per_iteration=60,
        min_score_to_continue=0.5
    ),
    'the_editor': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.85,
        timeout_per_iteration=180,  # Longer for video processing
        min_score_to_continue=0.4
    ),
    'the_soundscaper': AgentRalphConfig(
        enabled=True,
        max_iterations=3,
        completion_threshold=0.85,
        timeout_per_iteration=120,
        min_score_to_continue=0.4
    ),
    'clip_timing_engine': AgentRalphConfig(
        enabled=True,
        max_iterations=2,  # Limited - sync should converge fast
        completion_threshold=0.90,
        timeout_per_iteration=90,
        min_score_to_continue=0.5
    ),
}


class RalphAgentWrapper:
    """
    Wraps any RAGNAROK agent in a Ralph loop.

    Usage:
        wrapper = RalphAgentWrapper('story_creator', story_agent.generate)
        result = await wrapper.execute(task, context)

        if result['ralph_enabled']:
            print(f"Completed in {result['iterations']} iterations")
            print(f"Final score: {result['final_score']}")
    """

    def __init__(
        self,
        agent_name: str,
        agent_fn: Callable,
        evaluate_fn: Optional[Callable[[Any], float]] = None,
        config: Optional[AgentRalphConfig] = None
    ):
        self.agent_name = agent_name
        self.agent_fn = agent_fn
        self.config = config or AGENT_RALPH_CONFIGS.get(
            agent_name,
            AgentRalphConfig()  # Default enabled
        )
        self.evaluate_fn = evaluate_fn or self._get_default_evaluator()

    def _get_default_evaluator(self) -> Callable[[Any], float]:
        """Get agent-specific evaluator or default."""
        return AGENT_EVALUATORS.get(self.agent_name, self._default_evaluate)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent with Ralph loop if enabled.

        Returns:
            Dict with:
            - 'output': The actual agent output
            - 'ralph_state': Full Ralph state (if looped)
            - 'iterations': Number of iterations taken
            - 'final_score': Final quality score
            - 'total_cost': Total cost across iterations
            - 'ralph_enabled': Whether Ralph was used
            - 'status': 'completed' | 'max_iterations' | 'failed'
        """
        if not self.config.enabled:
            # Single-pass execution (no Ralph loop)
            logger.info("ralph_disabled", agent=self.agent_name)
            try:
                result = await self.agent_fn(task, context, **kwargs)
                return {
                    'output': result,
                    'ralph_state': None,
                    'iterations': 1,
                    'final_score': 1.0,
                    'total_cost': result.get('cost', 0.0) if isinstance(result, dict) else 0.0,
                    'ralph_enabled': False,
                    'status': 'completed'
                }
            except Exception as e:
                logger.error("single_pass_error", agent=self.agent_name, error=str(e))
                return {
                    'output': None,
                    'ralph_state': None,
                    'iterations': 1,
                    'final_score': 0.0,
                    'total_cost': 0.0,
                    'ralph_enabled': False,
                    'status': 'failed',
                    'error': str(e)
                }

        # Ralph loop execution
        logger.info(
            "ralph_agent_starting",
            agent=self.agent_name,
            max_iterations=self.config.max_iterations,
            completion_threshold=self.config.completion_threshold
        )

        ralph_config = RalphConfig(
            max_iterations=self.config.max_iterations,
            completion_threshold=self.config.completion_threshold,
            timeout_per_iteration=self.config.timeout_per_iteration
        )

        controller = RalphLoopController(ralph_config)

        # Wrap agent function to match Ralph interface
        async def wrapped_agent(task: str, ctx: Dict) -> Dict:
            try:
                result = await self.agent_fn(task, ctx, **kwargs)

                # Normalize result format
                if isinstance(result, dict):
                    return {
                        'output': result.get('output', result),
                        'tokens_used': result.get('tokens_used', 0),
                        'cost': result.get('cost', 0.0),
                        'errors': result.get('errors', [])
                    }
                else:
                    return {
                        'output': result,
                        'tokens_used': 0,
                        'cost': 0.0,
                        'errors': []
                    }
            except Exception as e:
                return {
                    'output': None,
                    'tokens_used': 0,
                    'cost': 0.0,
                    'errors': [str(e)]
                }

        ralph_state = await controller.run_loop(
            agent_name=self.agent_name,
            task=task,
            agent_fn=wrapped_agent,
            evaluate_fn=self.evaluate_fn,
            completion_criteria={'threshold': self.config.completion_threshold},
            context=context
        )

        # Extract best result
        best = ralph_state.get('best_result', {})
        return {
            'output': best.get('output'),
            'ralph_state': ralph_state,
            'iterations': ralph_state['iteration'] + 1,
            'final_score': best.get('score', 0),
            'total_cost': ralph_state['total_cost'],
            'total_tokens': ralph_state['total_tokens'],
            'ralph_enabled': True,
            'status': ralph_state['status']
        }

    def _default_evaluate(self, output: Any) -> float:
        """
        Default evaluation function.
        Override with agent-specific evaluation.
        """
        if output is None:
            return 0.0

        if isinstance(output, dict):
            # Check for explicit score
            if 'score' in output:
                return min(output['score'] / 100.0, 1.0)
            if 'confidence' in output:
                return output['confidence']
            if 'quality_score' in output:
                return min(output['quality_score'] / 100.0, 1.0)
            # Check for success indicator
            if output.get('success', False):
                return 0.9
            # Check for errors
            if output.get('errors'):
                return 0.3
            return 0.7  # Default for successful dict output

        if isinstance(output, str):
            # Basic heuristics for string output
            if len(output) > 100:
                return 0.8
            return 0.5

        return 0.6  # Default


# ============================================================
# EVALUATION FUNCTIONS FOR SPECIFIC AGENTS
# ============================================================

def evaluate_script(output: Any) -> float:
    """
    Evaluate story/script quality for commercial.

    Checks:
    - Appropriate length (80-150 words for 30s)
    - Has structure (scenes, visual cues)
    - Strong hook
    - Clear CTA
    - Emotional appeal
    - Brand mention
    """
    if not output:
        return 0.0

    script = output.get('script', '') if isinstance(output, dict) else str(output)
    score = 0.0

    # Length check (30s commercial should be 80-150 words)
    word_count = len(script.split())
    if 80 <= word_count <= 150:
        score += 0.25
    elif 50 <= word_count <= 200:
        score += 0.15
    elif word_count > 0:
        score += 0.05

    # Structure check (has scenes/visual cues)
    structure_keywords = ['SCENE', 'VISUAL', 'CUT TO', 'VOICEOVER', 'VO:', '[', ']', 'SHOT', 'ANGLE']
    structure_count = sum(1 for kw in structure_keywords if kw.upper() in script.upper())
    score += min(structure_count * 0.05, 0.20)

    # Hook check (attention-grabbing opener)
    lines = script.split('\n')
    first_line = lines[0] if lines else ''
    hook_indicators = ['?', '!', '"', 'Imagine', 'What if', 'Stop', 'Wait', 'Did you know']
    if any(ind in first_line for ind in hook_indicators) or len(first_line) < 60:
        score += 0.15

    # CTA check (call to action)
    cta_keywords = ['call', 'visit', 'sign up', 'try', 'get', 'book', 'download', 'start', 'learn more', 'today']
    cta_count = sum(1 for kw in cta_keywords if kw.lower() in script.lower())
    if cta_count > 0:
        score += 0.15

    # Brand mention check
    if isinstance(output, dict):
        business_name = output.get('business_name', '')
        if business_name and business_name.lower() in script.lower():
            score += 0.10

    # Emotional appeal check
    emotion_words = ['imagine', 'transform', 'discover', 'unlock', 'achieve', 'dream',
                     'love', 'amazing', 'incredible', 'powerful', 'easy', 'simple']
    emotion_count = sum(1 for word in emotion_words if word.lower() in script.lower())
    score += min(emotion_count * 0.03, 0.15)

    return min(score, 1.0)


def evaluate_prompts(output: Any) -> float:
    """
    Evaluate video prompts quality.

    Checks:
    - Correct number of prompts (4-6 scenes)
    - Sufficient detail per prompt
    - Visual keywords present
    - Consistency across prompts
    - No duplicates
    """
    if not output:
        return 0.0

    prompts = output.get('prompts', []) if isinstance(output, dict) else []
    if not prompts:
        # Try alternate keys
        prompts = output.get('scene_prompts', []) or output.get('video_prompts', [])
    if not prompts:
        return 0.0

    score = 0.0

    # Count check (should have 4-6 scene prompts for 30s commercial)
    if 4 <= len(prompts) <= 6:
        score += 0.25
    elif 3 <= len(prompts) <= 8:
        score += 0.15
    elif len(prompts) > 0:
        score += 0.05

    # Detail check (each prompt should be descriptive)
    detailed_count = sum(1 for p in prompts if len(str(p)) > 50)
    score += (detailed_count / max(len(prompts), 1)) * 0.25

    # Visual keyword check
    visual_keywords = ['camera', 'shot', 'angle', 'lighting', 'color', 'motion',
                       'close-up', 'wide', 'zoom', 'pan', 'fade', 'transition']
    prompts_str = ' '.join(str(p) for p in prompts).lower()
    visual_count = sum(1 for kw in visual_keywords if kw in prompts_str)
    score += min(visual_count * 0.03, 0.20)

    # Consistency check (similar structure)
    if all(isinstance(p, dict) for p in prompts):
        score += 0.15
    elif all(isinstance(p, str) for p in prompts):
        score += 0.10

    # Uniqueness check (no duplicate prompts)
    unique = len(set(str(p) for p in prompts))
    if unique == len(prompts):
        score += 0.15
    else:
        score += (unique / len(prompts)) * 0.10

    return min(score, 1.0)


def evaluate_video_scene(output: Any) -> float:
    """
    Evaluate generated video scene quality.

    Checks:
    - Video file exists
    - Correct duration (4-8s per scene)
    - Good resolution
    - No errors
    """
    if not output:
        return 0.0

    if isinstance(output, dict):
        # Check explicit quality metrics
        if 'quality_score' in output:
            return min(output['quality_score'] / 100.0, 1.0)

        # Check video file exists
        has_video = output.get('video_path') or output.get('video_url')
        if not has_video:
            return 0.2  # Generated something but no video

        score = 0.5  # Base score for having a video

        # Duration check
        duration = output.get('duration', 0)
        if 4 <= duration <= 8:
            score += 0.20
        elif 2 <= duration <= 12:
            score += 0.10

        # Resolution check
        resolution = output.get('resolution', '')
        if resolution in ['1080p', '4k', '1920x1080', '2160p']:
            score += 0.15
        elif resolution in ['720p', '1280x720']:
            score += 0.10

        # Error-free check
        if not output.get('errors'):
            score += 0.15

        return min(score, 1.0)

    return 0.5  # Default


def evaluate_research(output: Any) -> float:
    """
    Evaluate Trinity research quality.

    Checks:
    - Has key research components
    - Confidence scores
    - Source count
    - Insight quality
    """
    if not output:
        return 0.0

    score = 0.0

    if isinstance(output, dict):
        # Check for key research components
        components = ['market_analysis', 'competitor_intel', 'viral_patterns',
                      'platform_recommendations', 'audience_profile', 'trends',
                      'key_insights', 'recommendations']
        present = sum(1 for c in components if output.get(c))
        score += (present / len(components)) * 0.35

        # Check confidence scores
        confidence = output.get('confidence', 0)
        if confidence > 0.8:
            score += 0.20
        elif confidence > 0.6:
            score += 0.15
        elif confidence > 0.4:
            score += 0.10

        # Check source count
        sources = output.get('sources_count', 0) or len(output.get('sources', []))
        if sources >= 5:
            score += 0.20
        elif sources >= 3:
            score += 0.15
        elif sources >= 1:
            score += 0.10

        # Check insights quality
        insights = output.get('key_insights', []) or output.get('insights', [])
        if len(insights) >= 5:
            score += 0.25
        elif len(insights) >= 3:
            score += 0.20
        elif len(insights) >= 1:
            score += 0.10

    return min(score, 1.0)


def evaluate_auteur_qa(output: Any) -> float:
    """
    Evaluate THE AUTEUR creative QA quality.

    Checks:
    - Overall score
    - Category scores
    - Feedback quality
    """
    if not output:
        return 0.0

    if isinstance(output, dict):
        # Use overall score directly if present
        if 'overall_score' in output:
            return min(output['overall_score'] / 100.0, 1.0)
        if 'score' in output:
            return min(output['score'] / 100.0, 1.0)

        # Calculate from category scores
        categories = ['hook_strength', 'pacing', 'emotional_impact',
                      'brand_alignment', 'cta_effectiveness', 'visual_quality']
        scores = [output.get(cat, 0) for cat in categories]
        if any(scores):
            return sum(scores) / (len(categories) * 100)

    return 0.5


# ============================================================
# VORTEX POST-PRODUCTION EVALUATION FUNCTIONS (v8.0)
# ============================================================

def evaluate_wordsmith(output: Any) -> float:
    """
    Evaluate WORDSMITH text validation quality.

    Checks:
    - Spelling errors (must be 0 for high score)
    - Grammar errors
    - Brand compliance
    - WCAG accessibility
    """
    if not output:
        return 0.0

    score = 0.0

    if isinstance(output, dict):
        # Perfect spelling = high score
        spelling_errors = output.get('spelling_errors', [])
        if len(spelling_errors) == 0:
            score += 0.50
        else:
            score += max(0, 0.30 - len(spelling_errors) * 0.05)

        # Grammar check passed
        grammar_errors = output.get('grammar_errors', [])
        if len(grammar_errors) == 0:
            score += 0.20

        # Brand compliance
        if output.get('brand_compliant', False):
            score += 0.15

        # Accessibility passed
        if output.get('wcag_compliant', False):
            score += 0.15

        # Emit completion signal if no errors
        if len(spelling_errors) == 0 and len(grammar_errors) == 0:
            score = max(score, 0.95)  # Force completion

    return min(score, 1.0)


def evaluate_editor(output: Any) -> float:
    """
    Evaluate THE EDITOR shot detection & assembly quality.

    Checks:
    - Shot detection successful (3+ shots)
    - Transitions applied (2+)
    - Color analysis complete
    - Stabilization analyzed
    - Timeline/FFmpeg commands generated
    """
    if not output:
        return 0.0

    score = 0.0

    if isinstance(output, dict):
        # Shot detection successful
        shots = output.get('shots', [])
        if len(shots) >= 3:
            score += 0.25
        elif len(shots) >= 1:
            score += 0.15

        # Transitions applied
        transitions = output.get('transitions', [])
        if len(transitions) >= 2:
            score += 0.20

        # Color analysis complete
        if output.get('color_analysis'):
            score += 0.15

        # Stabilization analyzed
        if output.get('stabilization_result'):
            score += 0.15

        # Timeline generated
        if output.get('timeline') or output.get('ffmpeg_commands'):
            score += 0.25

    return min(score, 1.0)


def evaluate_soundscaper(output: Any) -> float:
    """
    Evaluate THE SOUNDSCAPER audio mixing quality.

    Checks:
    - Audio layers present (2+)
    - Ducking applied
    - Levels balanced
    - SFX matched
    - No clipping
    """
    if not output:
        return 0.0

    score = 0.0

    if isinstance(output, dict):
        # Audio layers present
        layers = output.get('audio_layers', [])
        if len(layers) >= 2:
            score += 0.25

        # Ducking applied
        if output.get('ducking_applied', False):
            score += 0.25

        # Levels balanced
        if output.get('levels_balanced', False):
            score += 0.20

        # SFX matched
        sfx = output.get('sfx_matches', [])
        if len(sfx) >= 1:
            score += 0.15

        # No clipping
        if not output.get('clipping_detected', True):
            score += 0.15

    return min(score, 1.0)


def evaluate_clip_timing(output: Any) -> float:
    """
    Evaluate ClipTimingEngine voiceover sync quality.

    Checks:
    - Segments parsed (3+)
    - Clips timed (3+)
    - Sync quality > 0.8
    - FFmpeg filter generated
    """
    if not output:
        return 0.0

    score = 0.0

    if isinstance(output, dict):
        # Segments parsed
        segments = output.get('segments', [])
        if len(segments) >= 3:
            score += 0.30

        # Clips timed
        timed_clips = output.get('timed_clips', [])
        if len(timed_clips) >= 3:
            score += 0.30

        # Sync quality (avg offset < 0.5s)
        sync_quality = output.get('sync_quality', 0)
        if sync_quality > 0.8:
            score += 0.25
        elif sync_quality > 0.6:
            score += 0.15

        # FFmpeg filter generated
        if output.get('ffmpeg_filter'):
            score += 0.15

    return min(score, 1.0)


# Registry of evaluation functions
AGENT_EVALUATORS: Dict[str, Callable[[Any], float]] = {
    'story_creator': evaluate_script,
    'prompt_engineer': evaluate_prompts,
    'video_generator': evaluate_video_scene,
    'trinity_suite': evaluate_research,
    'market_decoder': evaluate_research,
    'competitor_intel': evaluate_research,
    'viral_pattern': evaluate_research,
    'the_auteur': evaluate_auteur_qa,
    # VORTEX Post-Production Agents (v8.0)
    'the_wordsmith': evaluate_wordsmith,
    'the_editor': evaluate_editor,
    'the_soundscaper': evaluate_soundscaper,
    'clip_timing_engine': evaluate_clip_timing,
}


def get_ralph_wrapper(
    agent_name: str,
    agent_fn: Callable,
    custom_evaluator: Optional[Callable] = None,
    custom_config: Optional[AgentRalphConfig] = None
) -> RalphAgentWrapper:
    """
    Factory function to create configured Ralph wrapper.

    Usage:
        wrapper = get_ralph_wrapper('story_creator', story_agent.generate)
        result = await wrapper.execute(task, context)
    """
    return RalphAgentWrapper(
        agent_name=agent_name,
        agent_fn=agent_fn,
        evaluate_fn=custom_evaluator or AGENT_EVALUATORS.get(agent_name),
        config=custom_config or AGENT_RALPH_CONFIGS.get(agent_name)
    )


def is_ralph_enabled(agent_name: str) -> bool:
    """Check if Ralph is enabled for an agent."""
    config = AGENT_RALPH_CONFIGS.get(agent_name, AgentRalphConfig())
    return config.enabled


def get_agent_config(agent_name: str) -> AgentRalphConfig:
    """Get Ralph configuration for an agent."""
    return AGENT_RALPH_CONFIGS.get(agent_name, AgentRalphConfig())


# Export for package
__all__ = [
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
    # VORTEX evaluators (v8.0)
    'evaluate_wordsmith',
    'evaluate_editor',
    'evaluate_soundscaper',
    'evaluate_clip_timing'
]
