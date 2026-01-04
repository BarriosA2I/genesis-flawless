"""
================================================================================
⚡ FLAWLESS GENESIS ORCHESTRATOR v2.0 LEGENDARY
================================================================================
Production-grade NEXUS → TRINITY → RAGNAROK pipeline orchestration.

Package Exports:
- FlawlessGenesisOrchestrator: Main orchestrator class
- DistributedCircuitBreaker: Redis-backed circuit breaker
- TriggerDebouncer: Cost-saving duplicate prevention
- GhostRecoveryManager: SSE reconnection recovery
- LeadData: Lead qualification model

================================================================================
Author: Barrios A2I | Version: 2.0.0 LEGENDARY | January 2026
================================================================================
"""

__version__ = "2.0.0"
__author__ = "Barrios A2I"

# Core Orchestrator
from flawless_orchestrator import (
    FlawlessGenesisOrchestrator,
    LeadData,
    EventType,
    create_flawless_orchestrator
)

# Distributed Resilience
from distributed_resilience import (
    DistributedCircuitBreaker,
    CircuitState,
    CircuitConfig,
    TriggerDebouncer,
    DebounceConfig,
    create_circuit_breaker,
    create_debouncer
)

# Ghost Recovery
from ghost_recovery import (
    GhostRecoveryManager,
    GhostConfig,
    PipelineEvent,
    EventLog,
    create_ghost_recovery_manager
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Orchestrator
    "FlawlessGenesisOrchestrator",
    "LeadData",
    "EventType",
    "create_flawless_orchestrator",
    
    # Resilience
    "DistributedCircuitBreaker",
    "CircuitState",
    "CircuitConfig",
    "TriggerDebouncer",
    "DebounceConfig",
    "create_circuit_breaker",
    "create_debouncer",
    
    # Ghost Recovery
    "GhostRecoveryManager",
    "GhostConfig",
    "PipelineEvent",
    "EventLog",
    "create_ghost_recovery_manager"
]


def get_version():
    """Return package version"""
    return __version__


def get_info():
    """Return package information"""
    return {
        "name": "FLAWLESS GENESIS ORCHESTRATOR",
        "version": __version__,
        "author": __author__,
        "components": [
            "FlawlessGenesisOrchestrator",
            "DistributedCircuitBreaker",
            "TriggerDebouncer",
            "GhostRecoveryManager"
        ],
        "capabilities": [
            "NEXUS → TRINITY → RAGNAROK pipeline",
            "Redis-backed distributed state",
            "SSE ghost connection recovery",
            "Prometheus metrics",
            "OpenTelemetry tracing"
        ]
    }
