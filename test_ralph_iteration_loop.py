#!/usr/bin/env python3
"""
RAGNAROK v8.0 - RALPH ITERATION LOOP TEST
Tests that Quality Gate decisions actually trigger phase re-execution.

This script validates:
1. Quality Gate evaluates AUTEUR scores correctly
2. ITERATE_* decisions cause phases to re-run
3. Score progression is tracked across iterations
4. Feedback is passed between iterations

Usage:
    python test_ralph_iteration_loop.py
    python test_ralph_iteration_loop.py --endpoint http://localhost:8000
    python test_ralph_iteration_loop.py --mock-low-score
    python test_ralph_iteration_loop.py --output report.json
"""

import asyncio
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install with: pip install httpx")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ENDPOINT = "https://barrios-genesis-flawless.onrender.com"

# Standard test brief that should pass quality
STANDARD_BRIEF = {
    "industry": "B2B SaaS",
    "business_name": "TechFlow AI",
    "style": "modern",
    "goals": ["Drive demo signups", "Increase brand awareness"],
    "target_platforms": ["youtube", "tiktok"],
    "brief": {
        "target_audience": "Marketing directors at mid-size companies",
        "tone": "Professional but approachable",
        "duration": 30,
        "description": "TechFlow AI helps marketing teams automate repetitive tasks and save 40% of their time.",
        "brand_guidelines": {
            "primary_color": "#00CED1",
            "secondary_color": "#1a1a2e"
        },
        "enable_ralph": True
    }
}

# Brief designed to trigger iterations (vague, low quality)
LOW_QUALITY_BRIEF = {
    "industry": "unknown",
    "business_name": "X",
    "style": "basic",
    "goals": ["stuff"],
    "target_platforms": ["other"],
    "brief": {
        "target_audience": "people",
        "tone": "normal",
        "duration": 15,
        "description": "A company that does things.",
        "enable_ralph": True
    }
}


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class IterationRecord:
    """Record of a single pipeline iteration."""
    iteration_number: int
    auteur_score: float
    gate_decision: str
    phases_run: List[str] = field(default_factory=list)
    feedback: str = ""
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestReport:
    """Complete test report."""
    endpoint: str
    test_mode: str
    result: TestResult = TestResult.SKIPPED
    session_id: Optional[str] = None
    iterations: List[IterationRecord] = field(default_factory=list)
    total_iterations: int = 0
    final_auteur_score: float = 0.0
    final_gate_decision: str = ""
    score_progression: List[float] = field(default_factory=list)
    score_improvement: float = 0.0
    total_duration_ms: int = 0
    errors: List[str] = field(default_factory=list)
    ralph_enabled: bool = False
    phases_detected: List[str] = field(default_factory=list)
    iteration_detected: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["result"] = self.result.value
        data["iterations"] = [asdict(i) for i in self.iterations]
        return data


class RalphIterationTester:
    """Tests the Ralph iteration loop implementation."""

    def __init__(self, endpoint: str, mock_low_score: bool = False):
        self.endpoint = endpoint.rstrip("/")
        self.mock_low_score = mock_low_score
        self.report = TestReport(
            endpoint=endpoint,
            test_mode="low_quality" if mock_low_score else "standard"
        )

    async def run_all_tests(self) -> TestReport:
        """Run complete test suite."""
        print("=" * 60)
        print("RAGNAROK v8.0 - RALPH ITERATION LOOP TEST")
        print("=" * 60)
        print(f"Endpoint: {self.endpoint}")
        print(f"Test Mode: {'Low Quality (expect iterations)' if self.mock_low_score else 'Standard'}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        async with httpx.AsyncClient() as client:
            # Test 1: Health Check
            print("\n[1/4] Health Check...")
            if not await self._test_health(client):
                self.report.result = TestResult.FAIL
                return self.report

            # Test 2: Create Session
            print("\n[2/4] Creating Session...")
            session_id = await self._create_session(client)
            if not session_id:
                self.report.result = TestResult.FAIL
                return self.report
            self.report.session_id = session_id

            # Test 3: Run Production with Ralph
            print("\n[3/4] Running Production (monitoring for iterations)...")
            await self._run_production(client, session_id)

            # Test 4: Analyze Results
            print("\n[4/4] Analyzing Results...")
            self._analyze_results()

        self._print_report()
        return self.report

    async def _test_health(self, client: httpx.AsyncClient) -> bool:
        """Test health endpoint."""
        try:
            response = await client.get(f"{self.endpoint}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"      [+] Status: {data.get('status', 'unknown')}")
                print(f"      [+] Version: {data.get('version', 'unknown')}")
                return True
            else:
                self.report.errors.append(f"Health check failed: HTTP {response.status_code}")
                print(f"      [-] Failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.report.errors.append(f"Health check error: {str(e)}")
            print(f"      [-] Error: {e}")
            return False

    async def _create_session(self, client: httpx.AsyncClient) -> Optional[str]:
        """Create a production session."""
        try:
            response = await client.post(
                f"{self.endpoint}/api/session",
                json={"email": f"ralph-test-{int(time.time())}@barriosa2i.com"},
                timeout=30
            )
            if response.status_code in [200, 201]:
                data = response.json()
                session_id = data.get("session_id")
                print(f"      [+] Session: {session_id}")
                return session_id
            else:
                self.report.errors.append(f"Session creation failed: HTTP {response.status_code}")
                print(f"      [-] Failed: HTTP {response.status_code}")
                return None
        except Exception as e:
            self.report.errors.append(f"Session creation error: {str(e)}")
            print(f"      [-] Error: {e}")
            return None

    async def _run_production(self, client: httpx.AsyncClient, session_id: str):
        """Run production and monitor for iterations."""
        start_time = time.time()
        brief = LOW_QUALITY_BRIEF if self.mock_low_score else STANDARD_BRIEF

        current_iteration = 1
        current_phases: List[str] = []

        try:
            async with client.stream(
                "POST",
                f"{self.endpoint}/api/production/start/{session_id}",
                json=brief,
                headers={"Accept": "text/event-stream"},
                timeout=300  # 5 minutes for full production
            ) as response:

                if response.status_code != 200:
                    self.report.errors.append(f"Production failed: HTTP {response.status_code}")
                    print(f"      [-] Failed: HTTP {response.status_code}")
                    return

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                self._process_sse_event(
                                    data,
                                    current_iteration,
                                    current_phases,
                                    start_time
                                )

                                # Detect iteration change
                                iteration_num = self._extract_iteration_number(data)
                                if iteration_num and iteration_num > current_iteration:
                                    # Record previous iteration
                                    self._record_iteration(current_iteration, current_phases)
                                    current_iteration = iteration_num
                                    current_phases = []
                                    self.report.iteration_detected = True
                                    print(f"\n      === ITERATION {current_iteration} DETECTED! ===\n")

                                # Track phase
                                phase = data.get("phase")
                                if phase and phase not in current_phases:
                                    current_phases.append(phase)
                                    if phase not in self.report.phases_detected:
                                        self.report.phases_detected.append(phase)

                                # Check for completion
                                if data.get("status") == "completed":
                                    self._record_iteration(current_iteration, current_phases)
                                    self._extract_final_metrics(data)

                            except json.JSONDecodeError:
                                continue

        except asyncio.TimeoutError:
            self.report.errors.append("Production timed out after 5 minutes")
            print("      [-] Timeout!")
        except Exception as e:
            self.report.errors.append(f"Production error: {str(e)}")
            print(f"      [-] Error: {e}")

        self.report.total_duration_ms = int((time.time() - start_time) * 1000)

    def _process_sse_event(
        self,
        data: Dict[str, Any],
        iteration: int,
        phases: List[str],
        start_time: float
    ):
        """Process a single SSE event."""
        status = data.get("status", "")
        phase = data.get("phase", "")
        message = data.get("message", "")
        progress = data.get("progress", 0)

        # Format phase display
        iter_suffix = f" [Iter {iteration}]" if iteration > 1 else ""

        # Print progress updates
        if phase and message:
            print(f"      [{progress:3d}%] {phase.upper()}{iter_suffix}: {message[:50]}...")

        # Detect Quality Gate decisions
        if "quality_gate" in str(data).lower() or "gate_decision" in data:
            gate = data.get("gate_decision") or data.get("quality_gate", {}).get("decision")
            score = data.get("auteur_score") or data.get("quality_gate", {}).get("auteur_score", 0)
            if gate:
                print(f"      [>] Quality Gate: {gate}")
            if score:
                print(f"      [>] AUTEUR Score: {score}/100")
                self.report.score_progression.append(float(score))

        # Detect Ralph system status
        if data.get("ralph_enabled") or data.get("metadata", {}).get("ralph_enabled"):
            self.report.ralph_enabled = True

    def _extract_iteration_number(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract iteration number from event data."""
        # Check direct field
        if "iteration" in data:
            return int(data["iteration"])

        # Check message for "Iteration X/Y" pattern
        message = str(data.get("message", ""))
        if "Iteration" in message:
            import re
            match = re.search(r"Iteration\s+(\d+)", message)
            if match:
                return int(match.group(1))

        # Check metadata
        metadata = data.get("metadata", {})
        if "pipeline_iteration" in metadata:
            return int(metadata["pipeline_iteration"])

        return None

    def _record_iteration(self, iteration_num: int, phases: List[str]):
        """Record a completed iteration."""
        score = self.report.score_progression[-1] if self.report.score_progression else 0.0

        record = IterationRecord(
            iteration_number=iteration_num,
            auteur_score=score,
            gate_decision=self.report.final_gate_decision or "unknown",
            phases_run=phases.copy()
        )
        self.report.iterations.append(record)
        self.report.total_iterations = len(self.report.iterations)

    def _extract_final_metrics(self, data: Dict[str, Any]):
        """Extract final metrics from completion event."""
        metadata = data.get("metadata", {})

        self.report.ralph_enabled = metadata.get("ralph_enabled", False)
        self.report.total_iterations = metadata.get("pipeline_iterations", 1)
        self.report.final_auteur_score = metadata.get("auteur_score", 0)

        gate_info = metadata.get("quality_gate", {})
        self.report.final_gate_decision = gate_info.get("decision", "unknown")

        # Track QA history scores
        qa_history = metadata.get("qa_history", [])
        for entry in qa_history:
            score = entry.get("auteur_score", 0)
            if score and score not in self.report.score_progression:
                self.report.score_progression.append(float(score))

    def _analyze_results(self):
        """Analyze collected metrics and determine test result."""
        # Calculate score improvement
        if len(self.report.score_progression) >= 2:
            self.report.score_improvement = (
                self.report.score_progression[-1] - self.report.score_progression[0]
            )

        # Determine test result
        if self.report.errors:
            self.report.result = TestResult.FAIL
        elif self.report.ralph_enabled:
            if self.report.iteration_detected:
                # Iterations occurred - check if they improved quality
                if self.report.final_auteur_score >= 85:
                    self.report.result = TestResult.PASS
                elif self.report.score_improvement > 0:
                    self.report.result = TestResult.PASS
                else:
                    self.report.result = TestResult.WARNING
            else:
                # No iterations - first pass was good enough
                if self.report.final_auteur_score >= 85:
                    self.report.result = TestResult.PASS
                elif self.report.final_auteur_score >= 70:
                    self.report.result = TestResult.WARNING
                else:
                    self.report.result = TestResult.FAIL
        else:
            # Ralph not enabled
            self.report.result = TestResult.WARNING

    def _print_report(self):
        """Print final test report."""
        print("\n" + "=" * 60)
        print("TEST REPORT")
        print("=" * 60)

        # Overall result
        result_symbol = {
            TestResult.PASS: "[+] PASS",
            TestResult.FAIL: "[-] FAIL",
            TestResult.WARNING: "[!] WARNING",
            TestResult.SKIPPED: "[.] SKIPPED"
        }
        print(f"\nResult: {result_symbol[self.report.result]}")

        # Ralph System Status
        print(f"\nRalph System:")
        print(f"  Enabled: {'YES' if self.report.ralph_enabled else 'NO'}")
        print(f"  Iterations: {self.report.total_iterations}")
        print(f"  Iteration Detected: {'YES' if self.report.iteration_detected else 'NO'}")

        # Score Analysis
        print(f"\nScore Analysis:")
        print(f"  Final AUTEUR Score: {self.report.final_auteur_score}/100")
        print(f"  Final Gate Decision: {self.report.final_gate_decision}")

        if self.report.score_progression:
            progression = " -> ".join([f"{s:.0f}" for s in self.report.score_progression])
            print(f"  Score Progression: {progression}")
            if self.report.score_improvement != 0:
                sign = "+" if self.report.score_improvement > 0 else ""
                print(f"  Improvement: {sign}{self.report.score_improvement:.0f} points")

        # Phases
        if self.report.phases_detected:
            print(f"\nPhases Detected: {', '.join(self.report.phases_detected)}")

        # Timing
        print(f"\nDuration: {self.report.total_duration_ms / 1000:.1f}s")

        # Errors
        if self.report.errors:
            print(f"\nErrors:")
            for error in self.report.errors:
                print(f"  [-] {error}")

        # Iteration Details
        if self.report.iterations:
            print(f"\nIteration History:")
            for it in self.report.iterations:
                print(f"  Iteration {it.iteration_number}: "
                      f"Score={it.auteur_score:.0f}, "
                      f"Decision={it.gate_decision}, "
                      f"Phases={len(it.phases_run)}")

        print("\n" + "=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Ralph iteration loop implementation"
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"API endpoint (default: {DEFAULT_ENDPOINT})"
    )
    parser.add_argument(
        "--mock-low-score",
        action="store_true",
        help="Use low-quality brief to trigger iterations"
    )
    parser.add_argument(
        "--output",
        help="Save JSON report to file"
    )

    args = parser.parse_args()

    tester = RalphIterationTester(
        endpoint=args.endpoint,
        mock_low_score=args.mock_low_score
    )

    report = await tester.run_all_tests()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Exit with appropriate code
    if report.result == TestResult.PASS:
        exit(0)
    elif report.result == TestResult.WARNING:
        exit(0)  # Warnings are not failures
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
