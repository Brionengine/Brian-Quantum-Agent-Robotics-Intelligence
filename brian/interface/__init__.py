"""
Layer 7: Human Interaction - Brian-QARI Interface

Provides all human-facing interfaces for Brian:
  - OperatorConsole: Terminal REPL for interactive control
  - BrianAPI: Programmatic interface for dashboards and apps
  - VoiceInterface: Speech input/output
  - CommandParser: Natural language command parsing

Usage:
    from brian.interface import OperatorConsole, BrianAPI

    # Terminal control
    console = OperatorConsole(orchestrator)
    await console.run()

    # Programmatic control
    api = BrianAPI(orchestrator)
    api.send_command("pick up the cup")
"""

from brian.interface.operator_interface import (
    BrianAPI,
    CommandParser,
    CommandType,
    OperatorConsole,
    ParsedCommand,
    VoiceInterface,
)

__all__ = [
    "BrianAPI",
    "CommandParser",
    "CommandType",
    "OperatorConsole",
    "ParsedCommand",
    "VoiceInterface",
]
