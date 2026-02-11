"""
Layer 5: Orchestration - Brian-QARI Brain Loop

The orchestrator ties all layers together into a running system:
    Sense -> Think -> Act -> Learn

Usage:
    from brian.orchestrator import BrianOrchestrator, RunMode

    orchestrator = await BrianOrchestrator.create()
    await orchestrator.start()
    orchestrator.set_goal("pick up the cup")
    await orchestrator.stop()
"""

from brian.orchestrator.brian_orchestrator import (
    BrianOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
    RunMode,
)

__all__ = [
    "BrianOrchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
    "RunMode",
]
