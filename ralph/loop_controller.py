"""
Ralph Loop Controller
Iterative autonomous agent execution with filesystem persistence.

Based on Geoffrey Huntley's Ralph System (June 2025)
Implemented for Barrios A2I RAGNAROK Pipeline

Key Features:
- Iterative refinement until quality thresholds met
- Self-correction based on previous failures
- Convergence guarantees with max iteration limits
- Filesystem persistence for audit trail and recovery
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RalphConfig:
    """Configuration for Ralph loop execution."""
    max_iterations: int = 10
    completion_threshold: float = 0.90
    timeout_per_iteration: int = 300  # 5 minutes
    checkpoint_interval: int = 1  # Checkpoint every N iterations
    enable_git_commits: bool = False  # Disable for Render deployment
    progress_file: str = "progress.txt"
    state_file: str = "state.json"
    work_dir_base: str = ".ralph"


class IterationResult(TypedDict):
    """Result from a single iteration."""
    iteration: int
    output: Any
    score: float
    tokens_used: int
    cost: float
    duration_ms: int
    completion_signal: bool
    errors: List[str]


class RalphState(TypedDict):
    """Persistent state for Ralph loop."""
    loop_id: str
    agent_name: str
    task: str
    iteration: int
    max_iterations: int
    started_at: str
    updated_at: str
    status: str  # 'running' | 'completed' | 'failed' | 'max_iterations'
    history: List[IterationResult]
    best_result: Optional[IterationResult]
    total_cost: float
    total_tokens: int
    completion_criteria: Dict[str, Any]


class RalphLoopController:
    """
    Orchestrates iterative agent execution with:
    - Filesystem persistence (progress.txt, state.json)
    - PostgreSQL checkpointing (optional)
    - Git commits for audit trail (optional)
    - Completion signal detection
    - Quality threshold evaluation

    Usage:
        config = RalphConfig(max_iterations=5, completion_threshold=0.90)
        controller = RalphLoopController(config)

        result = await controller.run_loop(
            agent_name="story_creator",
            task="Create a 30-second commercial script",
            agent_fn=my_agent_function,
            evaluate_fn=my_evaluation_function,
            completion_criteria={'threshold': 0.90}
        )
    """

    def __init__(
        self,
        config: RalphConfig,
        db_session: Optional[Any] = None,
        git_repo_path: Optional[str] = None
    ):
        self.config = config
        self.db_session = db_session
        self.git_repo_path = git_repo_path
        self.state: Optional[RalphState] = None
        self.progress_path: Optional[Path] = None
        self.state_path: Optional[Path] = None

    async def run_loop(
        self,
        agent_name: str,
        task: str,
        agent_fn: Callable[..., Any],
        evaluate_fn: Callable[[Any], float],
        completion_criteria: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        work_dir: Optional[str] = None
    ) -> RalphState:
        """
        Execute Ralph loop until completion or max iterations.

        Args:
            agent_name: Name of the agent being wrapped
            task: Task description
            agent_fn: Async function to invoke agent
            evaluate_fn: Function to score output (returns 0.0-1.0)
            completion_criteria: Dict defining success conditions
            context: Additional context for agent
            work_dir: Working directory for persistence files

        Returns:
            Final RalphState with best result
        """
        # Initialize state
        loop_id = str(uuid.uuid4())[:8]
        work_dir_path = Path(work_dir or f"{self.config.work_dir_base}/{agent_name}_{loop_id}")
        work_dir_path.mkdir(parents=True, exist_ok=True)

        self.progress_path = work_dir_path / self.config.progress_file
        self.state_path = work_dir_path / self.config.state_file

        self.state = RalphState(
            loop_id=loop_id,
            agent_name=agent_name,
            task=task,
            iteration=0,
            max_iterations=self.config.max_iterations,
            started_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status='running',
            history=[],
            best_result=None,
            total_cost=0.0,
            total_tokens=0,
            completion_criteria=completion_criteria
        )

        logger.info(
            "ralph_loop_started",
            loop_id=loop_id,
            agent=agent_name,
            max_iterations=self.config.max_iterations,
            completion_threshold=self.config.completion_threshold
        )

        # Write initial progress
        self._write_progress(f"[{datetime.utcnow().isoformat()}] Ralph loop started for {agent_name}")
        self._write_progress(f"Task: {task[:200]}...")  # Truncate long tasks
        self._write_progress(f"Max iterations: {self.config.max_iterations}")
        self._write_progress(f"Completion threshold: {self.config.completion_threshold}")
        self._write_progress("---")

        # Main loop
        while self.state['iteration'] < self.config.max_iterations:
            iteration_num = self.state['iteration']

            try:
                # Build context with previous attempts
                iteration_context = self._build_iteration_context(context)

                # Execute agent with timeout
                start_time = datetime.utcnow()
                result = await asyncio.wait_for(
                    agent_fn(task, iteration_context),
                    timeout=self.config.timeout_per_iteration
                )
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                # Extract metrics from result
                if isinstance(result, dict):
                    tokens_used = result.get('tokens_used', 0)
                    cost = result.get('cost', 0.0)
                    output = result.get('output', result)
                    errors = result.get('errors', [])
                else:
                    tokens_used = 0
                    cost = 0.0
                    output = result
                    errors = []

                # Evaluate result
                score = evaluate_fn(output)

                # Check for completion signal
                completion_signal = self._detect_completion_signal(output)

                # Record iteration
                iteration_result = IterationResult(
                    iteration=iteration_num,
                    output=output,
                    score=score,
                    tokens_used=tokens_used,
                    cost=cost,
                    duration_ms=duration_ms,
                    completion_signal=completion_signal,
                    errors=errors
                )

                self.state['history'].append(iteration_result)
                self.state['total_cost'] += cost
                self.state['total_tokens'] += tokens_used

                # Track best result
                if self.state['best_result'] is None or score > self.state['best_result']['score']:
                    self.state['best_result'] = iteration_result
                    logger.info(
                        "ralph_new_best_score",
                        loop_id=loop_id,
                        iteration=iteration_num,
                        score=score
                    )

                # Log progress
                self._write_progress(f"\n[Iteration {iteration_num}]")
                self._write_progress(f"Score: {score:.3f} | Tokens: {tokens_used} | Cost: ${cost:.4f}")
                self._write_progress(f"Duration: {duration_ms}ms | Completion signal: {completion_signal}")
                if errors:
                    self._write_progress(f"Errors: {errors}")

                logger.info(
                    "ralph_iteration_complete",
                    loop_id=loop_id,
                    iteration=iteration_num,
                    score=score,
                    cost=cost,
                    duration_ms=duration_ms,
                    completion_signal=completion_signal
                )

                # Check completion conditions
                if completion_signal or score >= self.config.completion_threshold:
                    self.state['status'] = 'completed'
                    self._write_progress(f"\n{'='*50}")
                    self._write_progress(f"COMPLETION ACHIEVED at iteration {iteration_num}!")
                    self._write_progress(f"Final score: {score:.3f}")
                    self._write_progress(f"Total cost: ${self.state['total_cost']:.4f}")
                    self._write_progress(f"Total tokens: {self.state['total_tokens']}")
                    break

                # Checkpoint periodically
                if iteration_num % self.config.checkpoint_interval == 0:
                    await self._checkpoint()

                # Increment iteration counter
                self.state['iteration'] += 1
                self.state['updated_at'] = datetime.utcnow().isoformat()

            except asyncio.TimeoutError:
                self._write_progress(f"\n[WARNING] Iteration {iteration_num} timed out after {self.config.timeout_per_iteration}s")
                logger.warning(
                    "ralph_iteration_timeout",
                    loop_id=loop_id,
                    iteration=iteration_num,
                    timeout=self.config.timeout_per_iteration
                )
                self.state['iteration'] += 1
                continue

            except Exception as e:
                error_msg = str(e)
                self._write_progress(f"\n[ERROR] Iteration {iteration_num} failed: {error_msg}")
                logger.error(
                    "ralph_iteration_error",
                    loop_id=loop_id,
                    iteration=iteration_num,
                    error=error_msg
                )
                self.state['iteration'] += 1
                continue

        # Check if we hit max iterations without completion
        if self.state['status'] == 'running':
            self.state['status'] = 'max_iterations'
            best_score = self.state['best_result']['score'] if self.state['best_result'] else 0
            self._write_progress(f"\n{'='*50}")
            self._write_progress(f"Max iterations ({self.config.max_iterations}) reached")
            self._write_progress(f"Best score achieved: {best_score:.3f}")
            self._write_progress(f"Target was: {self.config.completion_threshold}")
            self._write_progress(f"Total cost: ${self.state['total_cost']:.4f}")

        # Final checkpoint
        await self._checkpoint()

        # Git commit if enabled
        if self.config.enable_git_commits and self.git_repo_path:
            self._git_commit(f"Ralph loop {loop_id} completed: {self.state['status']}")

        logger.info(
            "ralph_loop_completed",
            loop_id=loop_id,
            status=self.state['status'],
            iterations=self.state['iteration'],
            best_score=self.state['best_result']['score'] if self.state['best_result'] else 0,
            total_cost=self.state['total_cost'],
            total_tokens=self.state['total_tokens']
        )

        return self.state

    def _build_iteration_context(self, base_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Build context including previous attempts for self-correction."""
        context = base_context.copy() if base_context else {}

        # Add Ralph iteration info
        context['ralph'] = {
            'iteration': self.state['iteration'],
            'max_iterations': self.config.max_iterations,
            'previous_attempts': len(self.state['history']),
            'best_score_so_far': self.state['best_result']['score'] if self.state['best_result'] else 0,
            'completion_threshold': self.config.completion_threshold
        }

        # Add recent history (last 3 attempts) for self-correction
        if self.state['history']:
            context['ralph']['recent_history'] = [
                {
                    'iteration': h['iteration'],
                    'score': h['score'],
                    'errors': h['errors'],
                    'duration_ms': h['duration_ms']
                }
                for h in self.state['history'][-3:]
            ]

            # Calculate improvement trend
            if len(self.state['history']) >= 2:
                scores = [h['score'] for h in self.state['history'][-3:]]
                context['ralph']['score_trend'] = 'improving' if scores[-1] > scores[0] else 'declining'

        # Add improvement guidance based on last attempt
        if self.state['history']:
            last = self.state['history'][-1]
            if last['score'] < self.config.completion_threshold:
                context['ralph']['improvement_needed'] = True
                context['ralph']['last_score'] = last['score']
                context['ralph']['gap_to_target'] = self.config.completion_threshold - last['score']

                # Specific feedback based on score ranges
                if last['score'] < 0.5:
                    context['ralph']['feedback'] = "Major improvements needed. Review core requirements."
                elif last['score'] < 0.7:
                    context['ralph']['feedback'] = "Good foundation but needs refinement. Focus on quality details."
                elif last['score'] < 0.85:
                    context['ralph']['feedback'] = "Almost there. Fine-tune for excellence."
                else:
                    context['ralph']['feedback'] = "Very close to threshold. Minor polish needed."

        return context

    def _detect_completion_signal(self, output: Any) -> bool:
        """
        Detect completion signal in output.

        Agents can emit these signals to indicate they believe the task is complete,
        regardless of the numerical score.
        """
        signals = [
            '<promise>COMPLETE</promise>',
            '<promise>DONE</promise>',
            '<promise>SUCCESS</promise>',
            '[RALPH_COMPLETE]',
            '{{COMPLETION_SIGNAL}}',
            '[QUALITY_THRESHOLD_MET]',
            '<status>EXCELLENT</status>'
        ]

        output_str = str(output)
        return any(signal in output_str for signal in signals)

    def _write_progress(self, message: str):
        """Append to progress file for audit trail."""
        if self.progress_path:
            try:
                with open(self.progress_path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                logger.warning("progress_write_error", error=str(e))

    async def _checkpoint(self):
        """Save state to file and optionally to database."""
        # Save to JSON file
        if self.state_path:
            try:
                # Serialize state (handle non-serializable outputs)
                serializable_state = self._make_serializable(self.state)
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_state, f, indent=2, default=str)
            except Exception as e:
                logger.warning("checkpoint_file_error", error=str(e))

        # Save to database if available
        if self.db_session:
            try:
                await self.db_session.execute(
                    """
                    INSERT INTO ralph_checkpoints (loop_id, agent_name, state, created_at)
                    VALUES (:loop_id, :agent_name, :state, :created_at)
                    ON CONFLICT (loop_id) DO UPDATE SET
                        state = :state,
                        updated_at = :created_at
                    """,
                    {
                        'loop_id': self.state['loop_id'],
                        'agent_name': self.state['agent_name'],
                        'state': json.dumps(self._make_serializable(self.state), default=str),
                        'created_at': datetime.utcnow()
                    }
                )
                await self.db_session.commit()
            except Exception as e:
                logger.warning("checkpoint_db_error", error=str(e))

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

    def _git_commit(self, message: str):
        """Commit progress to git for audit trail."""
        import subprocess
        try:
            subprocess.run(
                ['git', 'add', '-A'],
                cwd=self.git_repo_path,
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.git_repo_path,
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.warning("git_commit_error", error=str(e))

    @classmethod
    async def resume_loop(
        cls,
        state_path: str,
        agent_fn: Callable[..., Any],
        evaluate_fn: Callable[[Any], float],
        config: Optional[RalphConfig] = None
    ) -> RalphState:
        """
        Resume a Ralph loop from a saved state.

        Args:
            state_path: Path to saved state.json file
            agent_fn: Agent function to use
            evaluate_fn: Evaluation function
            config: Optional config override

        Returns:
            Completed RalphState
        """
        with open(state_path, 'r') as f:
            saved_state = json.load(f)

        config = config or RalphConfig(
            max_iterations=saved_state['max_iterations'],
            completion_threshold=saved_state['completion_criteria'].get('threshold', 0.90)
        )

        controller = cls(config)
        controller.state = saved_state
        controller.state_path = Path(state_path)
        controller.progress_path = Path(state_path).parent / 'progress.txt'

        logger.info(
            "ralph_loop_resumed",
            loop_id=saved_state['loop_id'],
            from_iteration=saved_state['iteration']
        )

        # Continue the loop
        return await controller.run_loop(
            agent_name=saved_state['agent_name'],
            task=saved_state['task'],
            agent_fn=agent_fn,
            evaluate_fn=evaluate_fn,
            completion_criteria=saved_state['completion_criteria']
        )


# Convenience function for simple use cases
async def ralph_loop(
    agent_fn: Callable,
    task: str,
    evaluate_fn: Callable[[Any], float],
    max_iterations: int = 10,
    completion_threshold: float = 0.90,
    agent_name: str = "agent",
    context: Optional[Dict] = None
) -> RalphState:
    """
    Simple Ralph loop wrapper for quick integration.

    Usage:
        result = await ralph_loop(
            agent_fn=my_agent.generate,
            task="Create a commercial script",
            evaluate_fn=evaluate_script,
            max_iterations=5,
            completion_threshold=0.85,
            agent_name="story_creator"
        )

        if result['status'] == 'completed':
            best_output = result['best_result']['output']
    """
    config = RalphConfig(
        max_iterations=max_iterations,
        completion_threshold=completion_threshold
    )
    controller = RalphLoopController(config)
    return await controller.run_loop(
        agent_name=agent_name,
        task=task,
        agent_fn=agent_fn,
        evaluate_fn=evaluate_fn,
        completion_criteria={'threshold': completion_threshold},
        context=context
    )


# Export for package
__all__ = [
    'RalphConfig',
    'RalphLoopController',
    'RalphState',
    'IterationResult',
    'ralph_loop'
]
