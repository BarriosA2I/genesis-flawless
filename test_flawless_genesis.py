"""
================================================================================
⚡ FLAWLESS GENESIS ORCHESTRATOR v2.0 - Test Suite
================================================================================
Run: python test_flawless_genesis.py
================================================================================
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any

sys.path.insert(0, '.')

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    error: str = ""

class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        passed = sum(1 for r in self.results if r.passed)
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": f"{(passed / len(self.results) * 100):.1f}%" if self.results else "N/A",
            "total_time_ms": sum(r.duration_ms for r in self.results)
        }
    
    def print_results(self):
        print(f"\n{'=' * 70}")
        print(f"TEST SUITE: {self.name}")
        print(f"{'=' * 70}\n")
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"{status} | {r.name} ({r.duration_ms:.0f}ms)")
            if r.details: print(f"       {r.details}")
            if r.error: print(f"       ERROR: {r.error}")
        s = self.summary()
        print(f"\n{'-' * 70}")
        print(f"SUMMARY: {s['passed']}/{s['total']} passed ({s['pass_rate']})")
        print(f"{'=' * 70}\n")

async def run_test(name: str, test_func) -> TestResult:
    start = time.time()
    try:
        result = await test_func()
        return TestResult(name=name, passed=result.get("passed", False),
                         duration_ms=(time.time() - start) * 1000, details=result.get("details", ""))
    except Exception as e:
        return TestResult(name=name, passed=False, duration_ms=(time.time() - start) * 1000, error=str(e))

# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

async def test_circuit_breaker_local():
    from distributed_resilience import DistributedCircuitBreaker, CircuitState, CircuitConfig
    config = CircuitConfig(failure_threshold=2, timeout_seconds=0.1, success_threshold=1)
    cb = DistributedCircuitBreaker(service_name="test-agent", redis_client=None, config=config)
    
    state = await cb.get_state()
    assert state == CircuitState.CLOSED, f"Expected CLOSED, got {state}"
    assert await cb.can_execute() == True
    
    await cb.record_failure()
    await cb.record_failure()
    state = await cb.get_state()
    assert state == CircuitState.OPEN, f"Expected OPEN, got {state}"
    assert await cb.can_execute() == False
    
    await asyncio.sleep(0.15)
    assert await cb.can_execute() == True
    state = await cb.get_state()
    assert state == CircuitState.HALF_OPEN
    
    await cb.record_success()
    state = await cb.get_state()
    assert state == CircuitState.CLOSED
    return {"passed": True, "details": "State transitions: CLOSED→OPEN→HALF_OPEN→CLOSED"}

async def test_circuit_breaker_decorator():
    from distributed_resilience import DistributedCircuitBreaker, CircuitState, CircuitConfig, CircuitOpenError, with_circuit_breaker
    config = CircuitConfig(failure_threshold=1, timeout_seconds=30)
    cb = DistributedCircuitBreaker(service_name="decorator-test", redis_client=None, config=config)
    
    call_count = 0
    
    @with_circuit_breaker(cb)
    async def flaky():
        nonlocal call_count
        call_count += 1
        raise ValueError("Fail")
    
    try: await flaky()
    except ValueError: pass
    assert call_count == 1
    assert await cb.get_state() == CircuitState.OPEN
    
    try:
        await flaky()
        return {"passed": False, "details": "Should raise CircuitOpenError"}
    except CircuitOpenError: pass
    assert call_count == 1
    return {"passed": True, "details": "Decorator blocks when open"}

# =============================================================================
# DEBOUNCER TESTS
# =============================================================================

async def test_debouncer_allows_first():
    from distributed_resilience import TriggerDebouncer, DebounceConfig
    debouncer = TriggerDebouncer(redis_client=None, config=DebounceConfig(cooldown_seconds=5.0))
    result = await debouncer.can_trigger("session-123")
    assert result["allowed"] == True and result["reason"] == "ok"
    return {"passed": True, "details": "First trigger allowed"}

async def test_debouncer_blocks_locked():
    from distributed_resilience import TriggerDebouncer, DebounceConfig
    debouncer = TriggerDebouncer(redis_client=None, config=DebounceConfig(cooldown_seconds=5.0))
    await debouncer.can_trigger("session-456")
    await debouncer.acquire_lock("session-456")
    result = await debouncer.can_trigger("session-456")
    assert result["allowed"] == False and result["reason"] == "locked"
    return {"passed": True, "details": "Locked session blocked"}

async def test_debouncer_release():
    from distributed_resilience import TriggerDebouncer, DebounceConfig
    debouncer = TriggerDebouncer(redis_client=None, config=DebounceConfig(cooldown_seconds=0.1))
    await debouncer.acquire_lock("session-789")
    await debouncer.release_lock("session-789")
    result = await debouncer.can_trigger("session-789")
    assert result["allowed"] == True
    return {"passed": True, "details": "Allowed after release"}

# =============================================================================
# GHOST RECOVERY TESTS
# =============================================================================

async def test_ghost_record():
    from ghost_recovery import GhostRecoveryManager, GhostConfig
    manager = GhostRecoveryManager(redis_client=None, config=GhostConfig())
    e1 = await manager.record("pipe-123", "pipeline_start", {"msg": "Starting"})
    e2 = await manager.record("pipe-123", "agent_complete", {"agent": "scout"})
    assert e1.event_type == "pipeline_start" and e2.event_type == "agent_complete"
    return {"passed": True, "details": "Recorded 2 events"}

async def test_ghost_replay():
    from ghost_recovery import GhostRecoveryManager, GhostConfig
    manager = GhostRecoveryManager(redis_client=None, config=GhostConfig())
    for i in range(5):
        await manager.record("pipe-456", f"event_{i}", {"idx": i})
    events = await manager.event_log.get_all("pipe-456")
    assert len(events) == 5
    partial = await manager.event_log.get_since("pipe-456", 2)
    assert len(partial) == 3
    return {"passed": True, "details": f"Full: 5, After seq 2: 3"}

# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

async def test_orchestrator_init():
    from flawless_orchestrator import create_flawless_orchestrator
    orch = create_flawless_orchestrator(redis_client=None)
    assert orch and orch.trend_scout and orch.market_analyst and orch.competitor_tracker
    return {"passed": True, "details": "All agents initialized"}

async def test_orchestrator_execute():
    from flawless_orchestrator import create_flawless_orchestrator, LeadData
    orch = create_flawless_orchestrator(redis_client=None)
    lead = LeadData(session_id="test-123", business_name="Test Dental", industry="dental",
                   goals=["grow"], qualification_score=0.85)
    events = [e async for e in orch.execute(lead, generate_video=False)]
    assert len(events) > 0
    return {"passed": True, "details": f"Received {len(events)} events"}

async def test_orchestrator_full():
    from flawless_orchestrator import create_flawless_orchestrator, LeadData
    orch = create_flawless_orchestrator(redis_client=None)
    lead = LeadData(session_id="full-456", business_name="Smile Dental", industry="dental",
                   website_url="https://smile.com", goals=["grow"], qualification_score=0.9)
    events = [e async for e in orch.execute(lead, generate_video=True, video_formats=["youtube_1080p"])]
    return {"passed": True, "details": f"Full pipeline: {len(events)} events"}

async def test_orchestrator_health():
    from flawless_orchestrator import create_flawless_orchestrator
    orch = create_flawless_orchestrator(redis_client=None)
    health = await orch.get_health()
    assert health and "status" in health
    return {"passed": True, "details": f"Status: {health['status']}"}

# =============================================================================
# INTEGRATION TEST
# =============================================================================

async def test_full_integration():
    from flawless_orchestrator import create_flawless_orchestrator, LeadData
    from ghost_recovery import GhostRecoveryManager, GhostConfig
    
    orch = create_flawless_orchestrator(redis_client=None)
    ghost = GhostRecoveryManager(redis_client=None, config=GhostConfig())
    lead = LeadData(session_id="int-test", business_name="Int Dental", industry="dental",
                   goals=["test"], qualification_score=0.8)
    
    pipeline_id, count = None, 0
    async for event in orch.execute(lead, generate_video=True):
        data = event.get("data", {})
        if "pipeline_id" in data: pipeline_id = data["pipeline_id"]
        if pipeline_id:
            await ghost.record(pipeline_id, event.get("event_type", "unknown"), data)
            count += 1
    
    if pipeline_id:
        recovered = await ghost.event_log.get_all(pipeline_id)
        return {"passed": True, "details": f"Pipeline {pipeline_id[:12]}... recorded {len(recovered)} events"}
    return {"passed": True, "details": f"Completed with {count} events"}

# =============================================================================
# RUN ALL
# =============================================================================

async def run_all_tests():
    suite = TestSuite("FLAWLESS GENESIS v2.0 LEGENDARY")
    
    suite.add_result(await run_test("Circuit Breaker Local", test_circuit_breaker_local))
    suite.add_result(await run_test("Circuit Breaker Decorator", test_circuit_breaker_decorator))
    suite.add_result(await run_test("Debouncer Allows First", test_debouncer_allows_first))
    suite.add_result(await run_test("Debouncer Blocks Locked", test_debouncer_blocks_locked))
    suite.add_result(await run_test("Debouncer Release", test_debouncer_release))
    suite.add_result(await run_test("Ghost Record Events", test_ghost_record))
    suite.add_result(await run_test("Ghost Replay Events", test_ghost_replay))
    suite.add_result(await run_test("Orchestrator Init", test_orchestrator_init))
    suite.add_result(await run_test("Orchestrator Execute", test_orchestrator_execute))
    suite.add_result(await run_test("Orchestrator Full Pipeline", test_orchestrator_full))
    suite.add_result(await run_test("Orchestrator Health", test_orchestrator_health))
    suite.add_result(await run_test("Full Integration", test_full_integration))
    
    suite.print_results()
    return suite.summary()

if __name__ == "__main__":
    print("\n🧪 FLAWLESS GENESIS v2.0 - Test Suite\n")
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result["failed"] == 0 else 1)
