"""
Ralph Quality Gate
Decides whether to iterate the entire pipeline based on final QA scores.

This module handles pipeline-level iteration decisions after QA phase:
- PASS: Quality threshold met, proceed to delivery
- ITERATE_STORY: Regenerate script/story
- ITERATE_VIDEO: Regenerate video scenes
- ITERATE_FULL: Full pipeline restart
- FAIL: Max iterations reached

Based on AUTEUR scores and Technical QA results.
"""

from typing import Any, Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class GateDecision(Enum):
    """Quality gate decision outcomes."""
    PASS = "pass"
    ITERATE_STORY = "iterate_story"
    ITERATE_VIDEO = "iterate_video"
    ITERATE_PROMPTS = "iterate_prompts"
    ITERATE_FULL = "iterate_full"
    FAIL = "fail"


@dataclass
class GateResult:
    """Result from quality gate evaluation."""
    decision: GateDecision
    reason: str
    phases_to_rerun: List[str]
    feedback: str
    auteur_score: float
    technical_passed: bool
    iteration_number: int

    def should_iterate(self) -> bool:
        """Check if iteration is needed."""
        return self.decision in [
            GateDecision.ITERATE_STORY,
            GateDecision.ITERATE_PROMPTS,
            GateDecision.ITERATE_VIDEO,
            GateDecision.ITERATE_FULL
        ]

    def should_pass(self) -> bool:
        """Check if quality threshold met."""
        return self.decision == GateDecision.PASS

    def should_fail(self) -> bool:
        """Check if max iterations exceeded."""
        return self.decision == GateDecision.FAIL


class QualityGate:
    """
    Evaluates production output and decides next action.

    Decision Matrix:
        - AUTEUR Score >= 85: PASS (proceed to delivery)
        - AUTEUR Score 75-84: ITERATE_STORY (regenerate script)
        - AUTEUR Score 60-74: ITERATE_PROMPTS (regenerate prompts)
        - AUTEUR Score 50-59: ITERATE_VIDEO (regenerate video)
        - AUTEUR Score < 50: ITERATE_FULL (full pipeline retry)
        - Technical QA fails critically: ITERATE_VIDEO
        - Max iterations reached: Use best result or FAIL

    Usage:
        gate = QualityGate()
        result = gate.evaluate(
            auteur_score=78,
            technical_qa={'status': 'PASSED'},
            pipeline_iteration=1
        )

        if result.decision == GateDecision.PASS:
            proceed_to_delivery()
        else:
            rerun_from_phase(result.phases_to_rerun[0])
    """

    def __init__(
        self,
        auteur_pass_threshold: float = 85.0,
        auteur_story_threshold: float = 75.0,
        auteur_prompts_threshold: float = 60.0,
        auteur_video_threshold: float = 50.0,
        technical_qa_threshold: float = 70.0,
        max_pipeline_iterations: int = 3,
        use_best_on_max_iterations: bool = True
    ):
        """
        Initialize Quality Gate with thresholds.

        Args:
            auteur_pass_threshold: Score to PASS (default 85)
            auteur_story_threshold: Below this, iterate story (default 75)
            auteur_prompts_threshold: Below this, iterate prompts (default 60)
            auteur_video_threshold: Below this, iterate video (default 50)
            technical_qa_threshold: Technical QA pass threshold
            max_pipeline_iterations: Max full pipeline retries
            use_best_on_max_iterations: If True, use best result instead of failing
        """
        self.auteur_pass_threshold = auteur_pass_threshold
        self.auteur_story_threshold = auteur_story_threshold
        self.auteur_prompts_threshold = auteur_prompts_threshold
        self.auteur_video_threshold = auteur_video_threshold
        self.technical_qa_threshold = technical_qa_threshold
        self.max_pipeline_iterations = max_pipeline_iterations
        self.use_best_on_max_iterations = use_best_on_max_iterations

    def evaluate(
        self,
        auteur_score: float,
        technical_qa: Dict[str, Any],
        pipeline_iteration: int,
        qa_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> GateResult:
        """
        Evaluate QA results and decide next action.

        Args:
            auteur_score: THE AUTEUR creative QA score (0-100)
            technical_qa: Technical QA results dict
            pipeline_iteration: Current pipeline iteration number
            qa_history: History of previous QA scores
            metadata: Additional context

        Returns:
            GateResult with decision and instructions
        """
        tech_passed = technical_qa.get('status') == 'PASSED'
        tech_score = technical_qa.get('overall_score', 100)

        logger.info(
            "quality_gate_evaluating",
            auteur_score=auteur_score,
            tech_passed=tech_passed,
            tech_score=tech_score,
            pipeline_iteration=pipeline_iteration,
            max_iterations=self.max_pipeline_iterations
        )

        # Check max iterations first
        if pipeline_iteration >= self.max_pipeline_iterations:
            if self.use_best_on_max_iterations and auteur_score >= 70:
                # Use best result even if below threshold
                return GateResult(
                    decision=GateDecision.PASS,
                    reason=f"Max iterations ({self.max_pipeline_iterations}) reached. "
                           f"Using best result with AUTEUR score {auteur_score}/100.",
                    phases_to_rerun=[],
                    feedback="Quality is acceptable after max iterations.",
                    auteur_score=auteur_score,
                    technical_passed=tech_passed,
                    iteration_number=pipeline_iteration
                )
            else:
                return GateResult(
                    decision=GateDecision.FAIL,
                    reason=f"Max pipeline iterations ({self.max_pipeline_iterations}) reached. "
                           f"Best AUTEUR score: {auteur_score}/100",
                    phases_to_rerun=[],
                    feedback="Quality threshold not achievable within iteration limit.",
                    auteur_score=auteur_score,
                    technical_passed=tech_passed,
                    iteration_number=pipeline_iteration
                )

        # Check for critical technical issues
        if not tech_passed:
            critical_issues = [
                issue for issue in technical_qa.get('issues', [])
                if issue.get('severity') == 'CRITICAL'
            ]
            if critical_issues:
                return GateResult(
                    decision=GateDecision.ITERATE_VIDEO,
                    reason=f"Technical QA failed with {len(critical_issues)} critical issues. "
                           f"Issues: {[i.get('type') for i in critical_issues[:3]]}",
                    phases_to_rerun=['video', 'assembly', 'qa'],
                    feedback="Critical technical issues detected. Regenerating video with fixes.",
                    auteur_score=auteur_score,
                    technical_passed=False,
                    iteration_number=pipeline_iteration
                )

        # Check AUTEUR score thresholds
        if auteur_score >= self.auteur_pass_threshold:
            return GateResult(
                decision=GateDecision.PASS,
                reason=f"Quality threshold met! AUTEUR: {auteur_score}/100, "
                       f"Technical QA: {'PASSED' if tech_passed else 'WARNINGS'}",
                phases_to_rerun=[],
                feedback="Excellent quality achieved. Ready for delivery.",
                auteur_score=auteur_score,
                technical_passed=tech_passed,
                iteration_number=pipeline_iteration
            )

        if auteur_score >= self.auteur_story_threshold:
            # Score 75-84: Almost there, iterate on story/script
            return GateResult(
                decision=GateDecision.ITERATE_STORY,
                reason=f"AUTEUR score {auteur_score}/100 just below threshold ({self.auteur_pass_threshold}). "
                       f"Iterating on script/story for better creative quality.",
                phases_to_rerun=['story', 'prompts', 'video', 'voice', 'assembly', 'qa'],
                feedback=self._generate_story_feedback(auteur_score, metadata),
                auteur_score=auteur_score,
                technical_passed=tech_passed,
                iteration_number=pipeline_iteration
            )

        if auteur_score >= self.auteur_prompts_threshold:
            # Score 60-74: Prompts need work
            return GateResult(
                decision=GateDecision.ITERATE_PROMPTS,
                reason=f"AUTEUR score {auteur_score}/100 indicates prompt quality issues. "
                       f"Re-engineering video prompts.",
                phases_to_rerun=['prompts', 'video', 'assembly', 'qa'],
                feedback=self._generate_prompts_feedback(auteur_score, metadata),
                auteur_score=auteur_score,
                technical_passed=tech_passed,
                iteration_number=pipeline_iteration
            )

        if auteur_score >= self.auteur_video_threshold:
            # Score 50-59: Video generation issues
            return GateResult(
                decision=GateDecision.ITERATE_VIDEO,
                reason=f"AUTEUR score {auteur_score}/100 indicates video quality issues. "
                       f"Re-generating video scenes with enhanced settings.",
                phases_to_rerun=['video', 'assembly', 'qa'],
                feedback=self._generate_video_feedback(auteur_score, metadata),
                auteur_score=auteur_score,
                technical_passed=tech_passed,
                iteration_number=pipeline_iteration
            )

        # Score < 50: Full pipeline iteration needed
        return GateResult(
            decision=GateDecision.ITERATE_FULL,
            reason=f"AUTEUR score {auteur_score}/100 too low. "
                   f"Full pipeline iteration required for quality improvement.",
            phases_to_rerun=['story', 'prompts', 'video', 'voice', 'music', 'assembly', 'qa'],
            feedback=self._generate_full_feedback(auteur_score, qa_history, metadata),
            auteur_score=auteur_score,
            technical_passed=tech_passed,
            iteration_number=pipeline_iteration
        )

    def _generate_story_feedback(self, score: float, metadata: Optional[Dict]) -> str:
        """Generate specific feedback for story iteration."""
        feedback = [
            "Story quality is good but needs refinement.",
            "Focus on strengthening the emotional hook.",
            "Ensure the CTA is compelling and clear.",
            "Consider tightening the narrative arc."
        ]
        if metadata and metadata.get('weak_category') == 'hook_strength':
            feedback.append("PRIORITY: Rewrite opening hook for stronger impact.")
        return " ".join(feedback)

    def _generate_prompts_feedback(self, score: float, metadata: Optional[Dict]) -> str:
        """Generate specific feedback for prompts iteration."""
        feedback = [
            "Video prompts need more visual detail.",
            "Add specific camera angles and movements.",
            "Include lighting and color direction.",
            "Ensure scene continuity between prompts."
        ]
        return " ".join(feedback)

    def _generate_video_feedback(self, score: float, metadata: Optional[Dict]) -> str:
        """Generate specific feedback for video iteration."""
        feedback = [
            "Video generation quality below target.",
            "Try higher resolution or quality settings.",
            "Consider alternate visual styles.",
            "Check scene transitions and pacing."
        ]
        return " ".join(feedback)

    def _generate_full_feedback(
        self,
        score: float,
        qa_history: Optional[List[Dict]],
        metadata: Optional[Dict]
    ) -> str:
        """Generate feedback for full pipeline iteration."""
        feedback = [
            "Significant quality improvement needed.",
            "Review brief alignment and target audience.",
            "Consider alternative creative direction.",
            "Focus on coherence and brand consistency."
        ]

        # Learn from history if available
        if qa_history and len(qa_history) >= 2:
            scores = [h.get('auteur_score', 0) for h in qa_history]
            if scores[-1] <= scores[-2]:
                feedback.append("Previous iteration didn't improve. Try a different approach.")

        return " ".join(feedback)

    def get_iteration_config(self, decision: GateDecision) -> Dict[str, Any]:
        """
        Get configuration for the iteration based on decision.

        Returns dict with:
        - phases_to_run: List of phases to execute
        - skip_to_phase: Phase to start from
        - feedback: Improvement guidance
        - priority_focus: What to prioritize
        """
        configs = {
            GateDecision.PASS: {
                'phases_to_run': [],
                'skip_to_phase': 'delivery',
                'feedback': 'Quality threshold met. Proceed to delivery.',
                'priority_focus': 'enhancement'
            },
            GateDecision.ITERATE_STORY: {
                'phases_to_run': ['story', 'prompts', 'video', 'voice', 'assembly', 'qa'],
                'skip_to_phase': 'story',
                'feedback': 'Improve script emotional impact, hook strength, and CTA clarity.',
                'priority_focus': 'narrative'
            },
            GateDecision.ITERATE_PROMPTS: {
                'phases_to_run': ['prompts', 'video', 'assembly', 'qa'],
                'skip_to_phase': 'prompts',
                'feedback': 'Re-engineer prompts with more visual detail and better scene direction.',
                'priority_focus': 'visuals'
            },
            GateDecision.ITERATE_VIDEO: {
                'phases_to_run': ['video', 'assembly', 'qa'],
                'skip_to_phase': 'video',
                'feedback': 'Regenerate video scenes with higher quality settings.',
                'priority_focus': 'generation'
            },
            GateDecision.ITERATE_FULL: {
                'phases_to_run': ['story', 'prompts', 'video', 'voice', 'music', 'assembly', 'qa'],
                'skip_to_phase': 'story',
                'feedback': 'Full regeneration needed. Focus on coherence, brand alignment, and quality.',
                'priority_focus': 'all'
            },
            GateDecision.FAIL: {
                'phases_to_run': [],
                'skip_to_phase': 'error',
                'feedback': 'Quality threshold not achievable within iteration limit.',
                'priority_focus': None
            }
        }
        return configs.get(decision, configs[GateDecision.FAIL])

    def should_continue(self, result: GateResult) -> bool:
        """Check if iteration should continue or stop."""
        return result.decision not in [GateDecision.PASS, GateDecision.FAIL]


# Singleton instance for easy import
quality_gate = QualityGate()


# Export for package
__all__ = [
    'GateDecision',
    'GateResult',
    'QualityGate',
    'quality_gate'
]
