"""
Layer 7: Human Interaction - Operator Interface

Provides the human-facing APIs for interacting with Brian:
  - OperatorConsole: Terminal-based interactive control
  - VoiceInterface: Speech input/output for natural interaction
  - BrianAPI: Programmatic REST-like interface for dashboards
  - CommandParser: Natural language goal parsing

This layer sits on top of the BrianOrchestrator and translates
human intent into goals, commands, and queries.

Developed by Brion Quantum - Quantum AI & Intelligence Company
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Command Parser
# ============================================================================

class CommandType(Enum):
    """Types of operator commands."""
    GOAL = auto()           # "pick up the cup"
    MOTION = auto()         # "move left arm to..."
    NAVIGATION = auto()     # "walk to the table"
    SPEECH = auto()         # "say hello"
    GESTURE = auto()        # "wave hello"
    QUERY = auto()          # "what do you see?"
    SYSTEM = auto()         # "pause", "stop", "status"
    EMERGENCY = auto()      # "stop!", "emergency"


@dataclass
class ParsedCommand:
    """Result of parsing a natural language command."""
    command_type: CommandType
    raw_text: str
    intent: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    priority: float = 0.5


class CommandParser:
    """
    Parse natural language commands into structured goals and actions.

    Uses keyword matching for reliability. For production, this would
    integrate with an LLM (Claude) for sophisticated understanding.
    """

    # Intent patterns: (keywords, command_type, intent, priority)
    PATTERNS = [
        # Emergency (highest priority)
        (["stop", "halt", "emergency", "estop", "freeze"],
         CommandType.EMERGENCY, "emergency_stop", 1.0),

        # System commands
        (["status", "report", "how are you"],
         CommandType.SYSTEM, "status_report", 0.5),
        (["pause", "wait", "hold"],
         CommandType.SYSTEM, "pause", 0.5),
        (["resume", "continue", "go"],
         CommandType.SYSTEM, "resume", 0.5),
        (["shutdown", "power off", "sleep"],
         CommandType.SYSTEM, "shutdown", 0.5),
        (["reset", "restart"],
         CommandType.SYSTEM, "reset", 0.5),

        # Manipulation goals
        (["pick up", "grab", "grasp", "take", "lift"],
         CommandType.GOAL, "pick_up", 0.8),
        (["put down", "place", "set down", "drop", "release"],
         CommandType.GOAL, "place", 0.7),
        (["hand", "give", "pass", "deliver"],
         CommandType.GOAL, "handover", 0.8),
        (["pour", "fill"],
         CommandType.GOAL, "pour", 0.7),
        (["push", "slide"],
         CommandType.GOAL, "push", 0.6),
        (["open", "close"],
         CommandType.GOAL, "manipulate", 0.7),

        # Navigation
        (["walk to", "go to", "move to", "navigate", "come here", "approach"],
         CommandType.NAVIGATION, "navigate", 0.7),
        (["turn", "rotate", "face"],
         CommandType.NAVIGATION, "turn", 0.5),
        (["step back", "retreat", "back up"],
         CommandType.NAVIGATION, "retreat", 0.6),

        # Communication
        (["say", "speak", "tell", "announce"],
         CommandType.SPEECH, "speak", 0.5),
        (["wave", "greet", "hello", "hi"],
         CommandType.GESTURE, "wave", 0.5),
        (["nod", "yes"],
         CommandType.GESTURE, "nod", 0.3),
        (["shake head", "no"],
         CommandType.GESTURE, "shake_head", 0.3),
        (["point", "show"],
         CommandType.GESTURE, "point", 0.5),

        # Queries
        (["what do you see", "what's around", "describe", "look around"],
         CommandType.QUERY, "describe_scene", 0.3),
        (["where is", "find", "locate", "search for"],
         CommandType.QUERY, "locate_object", 0.5),
        (["how many", "count"],
         CommandType.QUERY, "count_objects", 0.3),
        (["what are you doing", "current task", "progress"],
         CommandType.QUERY, "current_task", 0.3),
        (["remember", "recall", "what happened"],
         CommandType.QUERY, "recall_memory", 0.3),
    ]

    def parse(self, text: str) -> ParsedCommand:
        """Parse a natural language command."""
        text_lower = text.lower().strip()

        best_match = None
        best_score = 0.0

        for keywords, cmd_type, intent, priority in self.PATTERNS:
            for keyword in keywords:
                if keyword in text_lower:
                    # Score based on keyword length and position
                    score = len(keyword) / max(len(text_lower), 1)
                    if text_lower.startswith(keyword):
                        score *= 1.5

                    if score > best_score:
                        best_score = score
                        best_match = (cmd_type, intent, priority)

        if best_match:
            cmd_type, intent, priority = best_match
            params = self._extract_parameters(text_lower, intent)
            return ParsedCommand(
                command_type=cmd_type,
                raw_text=text,
                intent=intent,
                parameters=params,
                confidence=min(best_score * 2, 1.0),
                priority=priority,
            )

        # Default: treat as a general goal
        return ParsedCommand(
            command_type=CommandType.GOAL,
            raw_text=text,
            intent="general_goal",
            parameters={"description": text},
            confidence=0.3,
            priority=0.5,
        )

    def _extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract relevant parameters from the command text."""
        params: Dict[str, Any] = {"description": text}

        # Extract object references
        objects = ["cup", "bottle", "plate", "bowl", "tool", "box",
                   "ball", "phone", "book", "pen", "glass"]
        for obj in objects:
            if obj in text:
                params["target_object"] = obj
                break

        # Extract color references
        colors = ["red", "blue", "green", "yellow", "white", "black",
                  "orange", "pink", "purple", "brown"]
        for color in colors:
            if color in text:
                params["target_color"] = color
                break

        # Extract location references
        locations = ["table", "desk", "floor", "shelf", "counter",
                     "left", "right", "front", "behind"]
        for loc in locations:
            if loc in text:
                params["location"] = loc
                break

        # Extract person references
        if any(w in text for w in ["person", "human", "people", "him", "her", "them"]):
            params["involves_person"] = True

        # Extract speech content for speak commands
        if intent == "speak":
            for prefix in ["say ", "speak ", "tell "]:
                if prefix in text:
                    params["speech_text"] = text.split(prefix, 1)[1]
                    break

        return params


# ============================================================================
# Voice Interface
# ============================================================================

class VoiceInterface:
    """
    Speech input/output for natural human-robot interaction.

    Input: Microphone -> Speech-to-Text -> CommandParser -> Goals
    Output: Text -> Text-to-Speech -> Speaker

    Designed for integration with:
      - Whisper (local STT)
      - Edge TTS / Coqui (local TTS)
      - Google Cloud STT/TTS (cloud option)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._stt_engine = config.get("stt_engine", "whisper")
        self._tts_engine = config.get("tts_engine", "edge_tts")
        self._language = config.get("language", "en")
        self._voice = config.get("voice", "en-US-GuyNeural")
        self._listening = False
        self._speak_queue: asyncio.Queue = asyncio.Queue()
        self._command_parser = CommandParser()
        self._on_command_callbacks: List[Callable] = []
        logger.info(f"VoiceInterface initialized | stt={self._stt_engine} | tts={self._tts_engine}")

    async def start_listening(self) -> None:
        """Start listening for voice commands."""
        self._listening = True
        logger.info("[Voice] Listening started")

    async def stop_listening(self) -> None:
        """Stop listening."""
        self._listening = False
        logger.info("[Voice] Listening stopped")

    async def speak(self, text: str, priority: float = 0.5) -> None:
        """Queue text for speech synthesis."""
        await self._speak_queue.put((text, priority))
        logger.info(f"[Voice] Speaking: '{text}'")

    def process_text_input(self, text: str) -> ParsedCommand:
        """Process text as if it were spoken (for testing or text chat)."""
        return self._command_parser.parse(text)

    def on_command(self, callback: Callable) -> None:
        """Register callback for recognized voice commands."""
        self._on_command_callbacks.append(callback)

    async def _process_audio_chunk(self, audio_data: np.ndarray,
                                    sample_rate: int = 16000) -> Optional[str]:
        """
        Process audio through STT.
        Returns recognized text or None.
        """
        # Placeholder for STT integration
        # In production: feed to Whisper, Google STT, etc.
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "listening": self._listening,
            "stt_engine": self._stt_engine,
            "tts_engine": self._tts_engine,
            "speak_queue_size": self._speak_queue.qsize(),
        }


# ============================================================================
# Brian API (Programmatic Interface)
# ============================================================================

class BrianAPI:
    """
    Programmatic interface for controlling Brian from external systems.

    Provides a clean API for dashboards, web interfaces, mobile apps,
    and other programs to interact with Brian.

    Designed to be wrapped by a web framework (FastAPI, Flask) for
    HTTP/WebSocket access.
    """

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        self._event_callbacks: Dict[str, List[Callable]] = {
            "goal_completed": [],
            "goal_failed": [],
            "safety_violation": [],
            "emergency_stop": [],
            "state_changed": [],
            "telemetry": [],
        }
        self._command_log: List[Dict[str, Any]] = []
        logger.info("[API] BrianAPI initialized")

    # --- System Control ---

    async def start(self) -> Dict[str, Any]:
        """Start Brian's brain loop."""
        await self._orchestrator.start()
        return {"status": "started", "state": self._orchestrator.state.name}

    async def stop(self) -> Dict[str, Any]:
        """Stop Brian gracefully."""
        await self._orchestrator.stop()
        return {"status": "stopped", "state": self._orchestrator.state.name}

    async def pause(self) -> Dict[str, Any]:
        """Pause brain loop."""
        await self._orchestrator.pause()
        return {"status": "paused"}

    async def resume(self) -> Dict[str, Any]:
        """Resume brain loop."""
        await self._orchestrator.resume()
        return {"status": "resumed"}

    async def emergency_stop(self) -> Dict[str, Any]:
        """Trigger emergency stop."""
        if self._orchestrator.robot:
            await self._orchestrator.robot.emergency_stop()
        return {"status": "emergency_stop_activated"}

    # --- Goal Management ---

    def set_goal(self, description: str, priority: float = 0.5,
                 success_criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Set a new goal."""
        goal = self._orchestrator.set_goal(description, priority, success_criteria)
        self._log_command("set_goal", {"description": description, "priority": priority})
        return {
            "goal_id": goal.goal_id,
            "description": goal.description,
            "priority": goal.priority,
            "status": goal.status,
        }

    def cancel_goal(self, goal_id: str) -> Dict[str, Any]:
        """Cancel a goal."""
        success = self._orchestrator.cancel_goal(goal_id)
        return {"goal_id": goal_id, "cancelled": success}

    def get_goals(self) -> List[Dict[str, Any]]:
        """Get all active goals."""
        return self._orchestrator.get_active_goals()

    # --- Status & Telemetry ---

    def get_status(self) -> Dict[str, Any]:
        """Get full system status."""
        return self._orchestrator.get_status()

    def get_mind_status(self) -> Dict[str, Any]:
        """Get BrianMind status."""
        return self._orchestrator.mind.get_status() if self._orchestrator.mind else {}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return self._orchestrator.memory.get_stats() if self._orchestrator.memory else {}

    def get_safety_report(self) -> Dict[str, Any]:
        """Get safety governor status."""
        return self._orchestrator.safety.get_status_report() if self._orchestrator.safety else {}

    def get_vision_status(self) -> Dict[str, Any]:
        """Get vision pipeline status."""
        return self._orchestrator.vision.get_status() if self._orchestrator.vision else {}

    # --- Natural Language ---

    def send_command(self, text: str) -> Dict[str, Any]:
        """
        Send a natural language command to Brian.

        Examples:
            api.send_command("pick up the red cup")
            api.send_command("walk to the table")
            api.send_command("what do you see?")
            api.send_command("status")
        """
        parser = CommandParser()
        parsed = parser.parse(text)
        self._log_command("nl_command", {"text": text, "parsed": parsed.intent})

        result = {"parsed": {
            "type": parsed.command_type.name,
            "intent": parsed.intent,
            "confidence": parsed.confidence,
            "parameters": parsed.parameters,
        }}

        if parsed.command_type == CommandType.EMERGENCY:
            asyncio.create_task(self.emergency_stop())
            result["action"] = "emergency_stop"

        elif parsed.command_type == CommandType.SYSTEM:
            result["action"] = self._handle_system_command(parsed)

        elif parsed.command_type == CommandType.QUERY:
            result["answer"] = self._handle_query(parsed)

        elif parsed.command_type in (CommandType.GOAL, CommandType.NAVIGATION,
                                      CommandType.SPEECH, CommandType.GESTURE):
            goal = self._orchestrator.set_goal(
                parsed.raw_text, parsed.priority)
            result["goal"] = {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "status": goal.status,
            }

        return result

    def _handle_system_command(self, parsed: ParsedCommand) -> str:
        """Handle system-level commands."""
        if parsed.intent == "status_report":
            return "status_reported"
        elif parsed.intent == "pause":
            asyncio.create_task(self._orchestrator.pause())
            return "paused"
        elif parsed.intent == "resume":
            asyncio.create_task(self._orchestrator.resume())
            return "resumed"
        elif parsed.intent == "shutdown":
            asyncio.create_task(self._orchestrator.stop())
            return "shutting_down"
        return "unknown_system_command"

    def _handle_query(self, parsed: ParsedCommand) -> str:
        """Handle queries about Brian's state and perception."""
        if parsed.intent == "describe_scene":
            if self._orchestrator.vision:
                status = self._orchestrator.vision.get_status()
                return (f"I see the environment with {status.get('tracked_people', 0)} people "
                        f"and {status.get('frame_count', 0)} frames processed. "
                        f"Closest person is {status.get('closest_person', 'inf'):.1f}m away.")
            return "Vision system not active."

        elif parsed.intent == "current_task":
            goals = self._orchestrator.get_active_goals()
            if goals:
                g = goals[0]
                return f"Working on: '{g['description']}' ({g['progress']:.0%} complete)"
            return "No active tasks."

        elif parsed.intent == "recall_memory":
            if self._orchestrator.memory:
                stats = self._orchestrator.memory.get_stats()
                return (f"I have {stats['total_memories']} memories: "
                        f"{stats.get('type_counts', {})}. "
                        f"Consolidated {stats.get('consolidations', 0)} times.")
            return "Memory system not active."

        return "I'm not sure how to answer that."

    # --- Events ---

    def on_event(self, event_type: str, callback: Callable) -> None:
        """Register a callback for events."""
        if event_type in self._event_callbacks:
            self._event_callbacks[event_type].append(callback)

    # --- Logging ---

    def _log_command(self, action: str, details: Dict[str, Any]) -> None:
        self._command_log.append({
            "timestamp": time.time(),
            "action": action,
            "details": details,
        })
        # Keep last 1000 commands
        if len(self._command_log) > 1000:
            self._command_log = self._command_log[-1000:]

    def get_command_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return self._command_log[-n:]


# ============================================================================
# Operator Console (Terminal Interface)
# ============================================================================

class OperatorConsole:
    """
    Interactive terminal interface for controlling Brian.

    Provides a REPL-style interface where operators can type
    natural language commands, query status, and monitor Brian.

    Usage:
        console = OperatorConsole(orchestrator)
        await console.run()
    """

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        self._api = BrianAPI(orchestrator)
        self._parser = CommandParser()
        self._running = False

    async def run(self) -> None:
        """Run the interactive console loop."""
        self._running = True

        print("\n" + "="*60)
        print("  Brian-QARI Operator Console")
        print("  Brion Quantum - Quantum AI & Intelligence")
        print("="*60)
        print("\nCommands:")
        print("  Type natural language goals (e.g., 'pick up the cup')")
        print("  'status'  - Show Brian's status")
        print("  'goals'   - Show active goals")
        print("  'memory'  - Show memory stats")
        print("  'safety'  - Show safety report")
        print("  'pause'   - Pause Brian")
        print("  'resume'  - Resume Brian")
        print("  'stop'    - Emergency stop")
        print("  'quit'    - Shutdown and exit")
        print("-"*60 + "\n")

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Brian> "))

                if not user_input.strip():
                    continue

                await self._process_input(user_input.strip())

            except (EOFError, KeyboardInterrupt):
                print("\nShutting down...")
                self._running = False
                await self._orchestrator.stop()

    async def _process_input(self, text: str) -> None:
        """Process operator input."""
        text_lower = text.lower()

        # System commands (exact match)
        if text_lower == "quit" or text_lower == "exit":
            print("Shutting down Brian...")
            await self._orchestrator.stop()
            self._running = False
            return

        if text_lower == "status":
            self._print_status()
            return

        if text_lower == "goals":
            self._print_goals()
            return

        if text_lower == "memory":
            self._print_memory()
            return

        if text_lower == "safety":
            self._print_safety()
            return

        if text_lower == "help":
            self._print_help()
            return

        # Natural language command
        result = self._api.send_command(text)
        parsed = result.get("parsed", {})

        print(f"  [{parsed.get('type', '?')}] {parsed.get('intent', '?')} "
              f"(confidence: {parsed.get('confidence', 0):.0%})")

        if "goal" in result:
            goal = result["goal"]
            print(f"  Goal set: {goal['goal_id']} - '{goal['description']}'")

        if "answer" in result:
            print(f"  Brian: {result['answer']}")

        if "action" in result:
            print(f"  Action: {result['action']}")

    def _print_status(self) -> None:
        """Print Brian's status."""
        status = self._orchestrator.get_status()
        metrics = status.get("metrics", {})
        mind = status.get("mind", {})
        orch = status.get("orchestrator", {})

        print(f"\n  State: {orch.get('state', '?')}")
        print(f"  Runtime: {orch.get('runtime_s', 0):.1f}s")
        print(f"  Consciousness: {mind.get('consciousness_level', 0):.4f}")
        print(f"  Autonomy: {mind.get('autonomy_level', 0):.4f}")
        print(f"  Cycles: {metrics.get('cycle_count', 0)}")
        print(f"  Perception: {metrics.get('avg_perception_ms', 0):.1f}ms "
              f"({metrics.get('perception_hz_actual', 0):.1f} Hz)")
        print(f"  Decision: {metrics.get('avg_decision_ms', 0):.1f}ms")
        print(f"  Goals completed: {metrics.get('goals_completed', 0)}")
        print(f"  Safety violations: {metrics.get('safety_violations', 0)}")
        print()

    def _print_goals(self) -> None:
        """Print active goals."""
        goals = self._orchestrator.get_active_goals()
        if not goals:
            print("  No active goals.")
            return
        print(f"\n  Active Goals ({len(goals)}):")
        for g in goals:
            print(f"    [{g['goal_id'][:8]}] {g['description']} "
                  f"({g['progress']:.0%}) [{g['status']}]")
        print()

    def _print_memory(self) -> None:
        """Print memory statistics."""
        if not self._orchestrator.memory:
            print("  Memory system not active.")
            return
        stats = self._orchestrator.memory.get_stats()
        print(f"\n  Memory System:")
        print(f"    Total: {stats['total_memories']}")
        print(f"    Types: {stats.get('type_counts', {})}")
        print(f"    Working: {stats['working_memory_items']} items")
        print(f"    Skills: {stats['skill_memories']}")
        print(f"    Avg strength: {stats['avg_strength']:.3f}")
        print(f"    Stored: {stats['total_stored']} | "
              f"Recalled: {stats['total_recalled']} | "
              f"Forgotten: {stats['total_forgotten']}")
        print()

    def _print_safety(self) -> None:
        """Print safety report."""
        if not self._orchestrator.safety:
            print("  Safety system not active.")
            return
        report = self._orchestrator.safety.get_status_report()
        print(f"\n  Safety Governor:")
        print(f"    Level: {report['overall_level']}")
        print(f"    E-stop: {report['emergency_stop']}")
        print(f"    Speed factor: {report['speed_factor']:.2f}")
        print(f"    Zones: {report['zones']}")
        if report['violations']:
            print(f"    Violations ({len(report['violations'])}):")
            for v in report['violations'][-5:]:
                print(f"      Zone {v['zone']} [{v['severity']}]: {v['description']}")
        print()

    def _print_help(self) -> None:
        """Print help text."""
        print("\n  Brian-QARI Operator Console Help")
        print("  --------------------------------")
        print("  Natural language commands:")
        print("    'pick up the red cup'    - Set manipulation goal")
        print("    'walk to the table'      - Set navigation goal")
        print("    'wave hello'             - Gesture command")
        print("    'say good morning'       - Speech command")
        print("    'what do you see?'       - Query perception")
        print()
        print("  System commands:")
        print("    status  - Full system status")
        print("    goals   - Active goals")
        print("    memory  - Memory statistics")
        print("    safety  - Safety report")
        print("    pause   - Pause brain loop")
        print("    resume  - Resume brain loop")
        print("    stop    - Emergency stop")
        print("    quit    - Shutdown and exit")
        print()
