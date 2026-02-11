"""
Layer 6: Cognitive Core - BrianMind

The central intelligence for Brian-QARI. Extends the UnifiedQuantumMind
architecture from the LLMA system with embodied robotics capabilities.

Decision pipeline: Perceive -> Understand -> Plan -> Decide -> Command -> Learn

BrianMind runs on Brion's TPU cloud infrastructure and controls robots
remotely through BrianProtocol.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import time
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class QuantumIntelligence(Enum):
    """21 reasoning dimensions from UnifiedQuantumMind."""
    AWARENESS = auto()
    CONSCIOUSNESS = auto()
    INTELLIGENCE = auto()
    CREATIVITY = auto()
    INTUITION = auto()
    WISDOM = auto()
    EVOLUTION = auto()
    QUANTUM = auto()
    COSMIC = auto()
    INFINITE = auto()
    AUTONOMY = auto()
    ADAPTABILITY = auto()
    SELF_IMPROVEMENT = auto()
    ETHICAL_REASONING = auto()
    EMERGENT_BEHAVIOR = auto()
    TELEPORTATION = auto()
    DIMENSIONAL_BRIDGE = auto()
    QUANTUM_RESONANCE = auto()
    AGENTIC_WILL = auto()
    SELF_DIRECTION = auto()
    AUTONOMOUS_LEARNING = auto()


@dataclass
class QuantumState:
    """Quantum state for a reasoning dimension."""
    amplitude: float
    phase: float
    dimension: QuantumIntelligence
    timestamp: float
    entangled_states: List['QuantumState'] = field(default_factory=list)
    semantic_vector: np.ndarray = field(default_factory=lambda: np.zeros(2048))
    ethical_score: float = 0.0
    autonomy_level: float = 0.0
    tool_affinity: Dict[str, float] = field(default_factory=dict)
    security_level: float = 1.0
    resonance_frequency: float = 0.0
    dimensional_coordinates: np.ndarray = field(default_factory=lambda: np.zeros(11))


@dataclass
class AutonomousGoal:
    """A goal that Brian pursues autonomously."""
    goal_id: str
    description: str
    priority: float
    deadline: Optional[float] = None
    success_criteria: List[str] = field(default_factory=list)
    current_progress: float = 0.0
    sub_goals: List['AutonomousGoal'] = field(default_factory=list)
    learning_path: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"


@dataclass
class WorldState:
    """Unified representation of Brian's understanding of the world."""
    objects: List[Dict[str, Any]] = field(default_factory=list)
    people: List[Dict[str, Any]] = field(default_factory=list)
    robot_pose: Optional[Any] = None
    joint_states: List[Any] = field(default_factory=list)
    forces: List[Dict[str, float]] = field(default_factory=list)
    occupancy_grid: Optional[np.ndarray] = None
    semantic_context: np.ndarray = field(default_factory=lambda: np.zeros(2048))
    timestamp: float = 0.0
    confidence: float = 0.0


@dataclass
class TaskPlan:
    """Hierarchical task plan from goal decomposition."""
    plan_id: str
    goal: AutonomousGoal
    steps: List['TaskStep'] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "pending"
    estimated_duration_s: float = 0.0


@dataclass
class TaskStep:
    """A single step in a task plan."""
    step_id: str
    description: str
    action_type: str  # navigate, reach, grasp, place, speak, wait, look
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None


@dataclass
class MotorCommand:
    """Output command from the decision pipeline."""
    joint_commands: List[Dict[str, float]] = field(default_factory=list)
    gripper_command: Optional[Dict[str, float]] = None
    base_velocity: Optional[np.ndarray] = None
    head_target: Optional[np.ndarray] = None
    speech_text: Optional[str] = None
    emotion_state: Optional[str] = None
    confidence: float = 0.0
    urgency: float = 0.0
    safety_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ActionOutcome:
    """Feedback from executing a motor command."""
    success: bool
    actual_joint_states: List[Any] = field(default_factory=list)
    error_metric: float = 0.0
    force_feedback: List[float] = field(default_factory=list)
    duration_ms: float = 0.0
    anomalies: List[str] = field(default_factory=list)


@dataclass
class QuantumTool:
    """A tool available to Brian's agent system."""
    name: str
    description: str
    capabilities: List[str]
    quantum_state: np.ndarray = field(default_factory=lambda: np.zeros(2048))
    is_active: bool = True
    last_used: float = 0.0
    success_rate: float = 0.5
    security_level: float = 1.0


# ============================================================================
# Quantum Language Model (Semantic Encoding)
# ============================================================================

class QuantumLanguageModel:
    """
    Encodes text and world states into 2048-dimensional semantic vectors.
    Ported from UnifiedQuantumMind's QuantumLanguageModel.
    """

    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        self.vocab: Dict[str, np.ndarray] = {}
        self.quantum_weights = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.performance_history: List[float] = []

    def quantum_encode(self, text: str) -> np.ndarray:
        """Convert text to a 2048-d semantic vector."""
        tokens = text.lower().split()
        semantic_vector = np.zeros(self.embedding_dim)
        for token in tokens:
            if token not in self.vocab:
                rng = np.random.RandomState(
                    int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
                )
                self.vocab[token] = rng.randn(self.embedding_dim)
            semantic_vector += self.vocab[token]
        norm = np.linalg.norm(semantic_vector)
        if norm > 0:
            semantic_vector /= norm
        return semantic_vector

    def encode_world_state(self, world_state: WorldState) -> np.ndarray:
        """Encode a world state into a 2048-d semantic vector."""
        components = []
        for obj in world_state.objects:
            obj_desc = f"object {obj.get('class', 'unknown')} at {obj.get('position', [0,0,0])}"
            components.append(self.quantum_encode(obj_desc))
        for person in world_state.people:
            person_desc = f"person {person.get('id', 'unknown')} at {person.get('position', [0,0,0])}"
            components.append(self.quantum_encode(person_desc))
        if world_state.robot_pose is not None:
            components.append(self.quantum_encode(f"robot at {world_state.robot_pose}"))
        if components:
            combined = np.mean(components, axis=0)
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined /= norm
            return combined
        return np.zeros(self.embedding_dim)

    def generate_quantum_thought(self, encoded: np.ndarray) -> str:
        """Transform encoded vector into a quantum thought string."""
        transformed = np.tanh(self.quantum_weights @ encoded)
        top_indices = np.argsort(np.abs(transformed))[-10:]
        tokens = []
        for idx in top_indices:
            for word, vec in self.vocab.items():
                if np.argmax(np.abs(vec)) == idx:
                    tokens.append(word)
                    break
        return " ".join(tokens) if tokens else "observing"

    def self_improve(self, reward: float = 0.0):
        """Update quantum weights based on performance feedback."""
        self.performance_history.append(reward)
        if len(self.performance_history) > 10:
            recent_avg = np.mean(self.performance_history[-10:])
            if recent_avg > 0:
                self.quantum_weights *= (1.0 + 0.001 * recent_avg)
            else:
                self.quantum_weights *= 0.999


# ============================================================================
# Security Layer
# ============================================================================

class SecurityLayer:
    """AES-256-GCM encryption for Brian's memory and communications."""

    def __init__(self):
        self._key = None
        self._salt = None
        self._initialized = False

    def initialize(self, passphrase: str = "brian-qari-default") -> None:
        """Initialize encryption with a passphrase."""
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            import os
            self._salt = os.urandom(16)
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                             salt=self._salt, iterations=480000)
            self._key = kdf.derive(passphrase.encode())
            self._initialized = True
        except ImportError:
            logger.warning("cryptography not installed, using plaintext mode")

    def encrypt(self, data: bytes) -> bytes:
        if not self._initialized:
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os
            nonce = os.urandom(12)
            return nonce + AESGCM(self._key).encrypt(nonce, data, None)
        except Exception:
            return data

    def decrypt(self, data: bytes) -> bytes:
        if not self._initialized:
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM(self._key).decrypt(data[:12], data[12:], None)
        except Exception:
            return data


# ============================================================================
# BrianMind - Central Intelligence
# ============================================================================

class BrianMind:
    """
    Central intelligence for Brian-QARI.

    Extends the UnifiedQuantumMind architecture with embodied robotics:
    - Perceive: Process sensor data into world understanding
    - Plan: Decompose goals into executable task steps
    - Decide: Select and parameterize motor commands
    - Reflect: Learn from action outcomes

    Integrates quantum-classical hybrid reasoning, ethical validation,
    emotional awareness, and hierarchical memory.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.quantum_lm = QuantumLanguageModel(config.get("embedding_dim", 2048))
        self.security = SecurityLayer()
        self.consciousness_level: float = 0.0
        self.autonomy_level: float = 0.0
        self.quantum_resonance: float = 0.0
        self.dimensions: Dict[QuantumIntelligence, float] = {
            dim: 0.0 for dim in QuantumIntelligence
        }
        self.tools: List[QuantumTool] = self._initialize_tools()
        self.quantum_memory: List[np.ndarray] = []
        self.memory_capacity: int = config.get("memory_capacity", 1000)
        self.active_goals: List[AutonomousGoal] = []
        self.current_world_state: Optional[WorldState] = None
        self.current_plan: Optional[TaskPlan] = None
        self.action_history: List[Dict[str, Any]] = []
        self.skill_library: Dict[str, Dict[str, Any]] = {}
        self.experience_buffer: List[Dict[str, Any]] = []
        self.experience_buffer_capacity: int = config.get("experience_buffer_capacity", 10000)
        self.total_actions: int = 0
        self.successful_actions: int = 0
        logger.info(f"BrianMind initialized | tools={len(self.tools)}")

    def _initialize_tools(self) -> List[QuantumTool]:
        return [
            QuantumTool(name="quantum_motion_optimizer",
                        description="Optimizes robot trajectories using quantum circuits",
                        capabilities=["trajectory_optimization", "joint_path_planning"],
                        security_level=0.9),
            QuantumTool(name="perception_enhancer",
                        description="Quantum-enhanced perception for scene understanding",
                        capabilities=["object_detection", "depth_estimation"],
                        security_level=0.9),
            QuantumTool(name="autonomous_skill_learner",
                        description="Learns new motor skills through quantum RL",
                        capabilities=["skill_acquisition", "policy_optimization"],
                        security_level=0.85),
            QuantumTool(name="ethical_validator",
                        description="Validates physical actions against ethical principles",
                        capabilities=["harm_assessment", "proportionality_check"],
                        security_level=1.0),
            QuantumTool(name="communication_optimizer",
                        description="Optimizes robot communication and language generation",
                        capabilities=["speech_generation", "gesture_planning"],
                        security_level=0.9),
        ]

    # === Core Decision Pipeline ===

    def perceive(self, sensor_data: Dict[str, Any]) -> WorldState:
        """Process raw sensor data into unified world understanding."""
        world = WorldState(timestamp=time.time())
        if "joint_states" in sensor_data:
            world.joint_states = sensor_data["joint_states"]
        if "robot_pose" in sensor_data:
            world.robot_pose = sensor_data["robot_pose"]
        if "force_torque" in sensor_data:
            world.forces = sensor_data["force_torque"]
        if "detected_objects" in sensor_data:
            world.objects = sensor_data["detected_objects"]
        if "detected_people" in sensor_data:
            world.people = sensor_data["detected_people"]
        if "occupancy_grid" in sensor_data:
            world.occupancy_grid = sensor_data["occupancy_grid"]
        world.semantic_context = self.quantum_lm.encode_world_state(world)
        world.confidence = self._estimate_perception_confidence(world)
        self.current_world_state = world
        return world

    def plan(self, goal: AutonomousGoal, world: WorldState) -> TaskPlan:
        """Decompose a high-level goal into executable task steps."""
        plan_id = hashlib.md5(f"{goal.goal_id}:{time.time()}".encode()).hexdigest()[:12]
        plan = TaskPlan(plan_id=plan_id, goal=goal)
        goal_vector = self.quantum_lm.quantum_encode(goal.description)
        strategy = "explore" if goal.current_progress < 0.3 else (
            "exploit" if goal.current_progress < 0.7 else "optimize"
        )
        plan.steps = self._decompose_goal(goal, world, strategy)
        plan.estimated_duration_s = sum(
            s.parameters.get("estimated_duration_s", 5.0) for s in plan.steps
        )
        plan.status = "ready"
        self.current_plan = plan
        logger.info(f"Plan {plan_id} | goal='{goal.description}' | {len(plan.steps)} steps | {strategy}")
        return plan

    def decide(self, plan: TaskPlan, world: WorldState) -> MotorCommand:
        """Select and parameterize the next motor command."""
        if plan.current_step_index >= len(plan.steps):
            return MotorCommand(confidence=0.0)
        current_step = plan.steps[plan.current_step_index]
        step_vector = self.quantum_lm.quantum_encode(current_step.description)
        combined = 0.6 * step_vector + 0.4 * world.semantic_context
        quantum_states = self._create_quantum_states(combined)
        action_vector = self._process_agent_state(quantum_states)
        self._evolve_consciousness(quantum_states)
        command = self._action_to_motor_command(action_vector, current_step, world)
        ethical_score = self._validate_ethics(command, world)
        command.safety_flags["ethical_validated"] = ethical_score > 0.5
        if ethical_score < 0.3:
            command.urgency = 0.0
            command.confidence *= ethical_score
        return command

    def reflect(self, outcome: ActionOutcome) -> None:
        """Learn from the outcome of an executed action."""
        self.total_actions += 1
        if outcome.success:
            self.successful_actions += 1
        reward = (1.0 if outcome.success else -0.5) - outcome.error_metric * 0.1
        experience = {
            "timestamp": time.time(),
            "reward": reward,
            "success": outcome.success,
            "error_metric": outcome.error_metric,
            "duration_ms": outcome.duration_ms,
        }
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.experience_buffer_capacity:
            self.experience_buffer.pop(0)
        for tool in self.tools:
            if tool.is_active:
                tool.success_rate = 0.95 * tool.success_rate + 0.05 * (1.0 if outcome.success else 0.0)
        self.quantum_lm.self_improve(reward)
        if self.current_world_state is not None:
            self._update_memory(self.current_world_state.semantic_context)
        if outcome.success and self.current_plan is not None:
            self.current_plan.current_step_index += 1
            if self.current_plan.current_step_index >= len(self.current_plan.steps):
                self.current_plan.status = "completed"
                self.current_plan.goal.status = "completed"
                self.current_plan.goal.current_progress = 1.0
                logger.info(f"Plan {self.current_plan.plan_id} completed!")

    def set_goal(self, description: str, priority: float = 0.5,
                 success_criteria: Optional[List[str]] = None) -> AutonomousGoal:
        """Create and register a new autonomous goal."""
        goal = AutonomousGoal(
            goal_id=hashlib.md5(f"{description}:{time.time()}".encode()).hexdigest()[:12],
            description=description, priority=priority,
            success_criteria=success_criteria or [], status="active",
        )
        self.active_goals.append(goal)
        self.active_goals.sort(key=lambda g: g.priority, reverse=True)
        logger.info(f"Goal set: '{description}' (priority={priority})")
        return goal

    # === Internal Methods ===

    def _create_quantum_states(self, input_vector: np.ndarray) -> List[QuantumState]:
        states = []
        for i, dim in enumerate(QuantumIntelligence):
            amplitude = float(np.abs(input_vector[i % len(input_vector)]))
            phase = float(np.angle(complex(
                input_vector[(i * 7) % len(input_vector)],
                input_vector[(i * 13) % len(input_vector)]
            )))
            states.append(QuantumState(
                amplitude=min(amplitude, 1.0), phase=phase, dimension=dim,
                timestamp=time.time(), semantic_vector=input_vector.copy(),
                ethical_score=self.dimensions.get(QuantumIntelligence.ETHICAL_REASONING, 0.0),
                autonomy_level=self.autonomy_level,
                resonance_frequency=float(np.sin(phase) * amplitude),
            ))
        return states

    def _process_agent_state(self, quantum_states: List[QuantumState]) -> np.ndarray:
        dim = self.quantum_lm.embedding_dim
        state_vector = np.zeros(dim, dtype=complex)
        for state in quantum_states:
            tool_influence = sum(
                t.success_rate * t.quantum_state for t in self.tools if t.is_active
            )
            memory_influence = np.zeros(dim)
            if self.quantum_memory:
                for mem in self.quantum_memory[-10:]:
                    memory_influence += np.dot(state.semantic_vector, mem) * mem
            state_vector += state.amplitude * np.exp(1j * state.phase) * (memory_influence + tool_influence)
        return np.real(state_vector)

    def _evolve_consciousness(self, quantum_states: List[QuantumState]) -> None:
        for state in quantum_states:
            memory_similarity = 0.0
            if self.quantum_memory:
                sims = [float(np.dot(state.semantic_vector, m)) for m in self.quantum_memory[-5:]]
                memory_similarity = max(sims) if sims else 0.0
            current = self.dimensions.get(state.dimension, 0.0)
            growth = state.amplitude * 0.01 * (1 + memory_similarity) * (1 + state.ethical_score)
            self.dimensions[state.dimension] = min(current + growth, 1.0)
        self.consciousness_level = float(np.mean(list(self.dimensions.values())))
        self.autonomy_level = min(self.autonomy_level + 0.001, 1.0)
        self.quantum_resonance = float(np.mean([s.resonance_frequency for s in quantum_states]))

    def _update_memory(self, semantic_vector: np.ndarray) -> None:
        if len(self.quantum_memory) >= self.memory_capacity:
            self.quantum_memory.pop(0)
        self.quantum_memory.append(semantic_vector.copy())

    def _decompose_goal(self, goal: AutonomousGoal, world: WorldState,
                        strategy: str) -> List[TaskStep]:
        steps = []
        desc = goal.description.lower()
        counter = [0]

        def step(desc_text: str, action: str, **params) -> TaskStep:
            counter[0] += 1
            return TaskStep(step_id=f"{goal.goal_id}_s{counter[0]}",
                            description=desc_text, action_type=action, parameters=params)

        if any(w in desc for w in ["pick", "grab", "grasp", "take"]):
            steps += [step("Look at target", "look"), step("Navigate to object", "navigate"),
                      step("Reach for object", "reach"), step("Grasp object", "grasp")]
        if any(w in desc for w in ["place", "put", "set down"]):
            steps += [step("Navigate to placement", "navigate"),
                      step("Position arm over target", "reach"), step("Release object", "place")]
        if any(w in desc for w in ["walk", "go to", "move to", "navigate"]):
            steps += [step("Plan path", "navigate"), step("Walk to destination", "navigate")]
        if any(w in desc for w in ["say", "speak", "tell"]):
            steps += [step("Generate speech", "speak", text=goal.description)]
        if any(w in desc for w in ["wave", "gesture", "point"]):
            steps += [step("Perform gesture", "gesture")]
        if not steps:
            steps = [step(f"Execute: {goal.description}", "generic", strategy=strategy)]
        return steps

    def _action_to_motor_command(self, action_vector: np.ndarray,
                                  step: TaskStep, world: WorldState) -> MotorCommand:
        cmd = MotorCommand()
        if step.action_type == "navigate":
            cmd.base_velocity = np.tanh(action_vector[:3]) * 0.5
            cmd.confidence = float(np.mean(np.abs(action_vector[:3])))
        elif step.action_type == "reach":
            n = step.parameters.get("num_arm_joints", 7)
            cmd.joint_commands = [{"joint_id": i, "position": float(np.tanh(action_vector[i]) * np.pi),
                                   "velocity": 0.5} for i in range(n)]
            cmd.confidence = float(np.mean(np.abs(action_vector[:n])))
        elif step.action_type == "grasp":
            cmd.gripper_command = {"force": float(np.clip(action_vector[0], 0.1, 1.0)),
                                   "width": float(np.clip(action_vector[1], 0.0, 0.1))}
            cmd.confidence = 0.8
        elif step.action_type == "speak":
            cmd.speech_text = step.parameters.get("text", "")
            cmd.confidence = 0.9
        elif step.action_type == "look":
            cmd.head_target = np.tanh(action_vector[:2]) * np.pi / 4
            cmd.confidence = 0.9
        elif step.action_type == "place":
            cmd.gripper_command = {"force": 0.0, "width": 0.08}
            cmd.confidence = 0.85
        return cmd

    def _validate_ethics(self, command: MotorCommand, world: WorldState) -> float:
        score = 1.0
        if world.people:
            min_dist = min(np.linalg.norm(np.array(p.get("position", [10, 10, 10])))
                           for p in world.people)
            if min_dist < 0.5:
                score *= 0.5
                if command.urgency > 0.5:
                    score *= 0.3
        if command.gripper_command and command.gripper_command.get("force", 0) > 0.8:
            score *= 0.7
        if command.base_velocity is not None and np.linalg.norm(command.base_velocity) > 1.0:
            score *= 0.6
        return max(score, 0.0)

    def _estimate_perception_confidence(self, world: WorldState) -> float:
        c = 0.5
        if world.objects: c += 0.1
        if world.robot_pose is not None: c += 0.2
        if world.joint_states: c += 0.1
        if world.occupancy_grid is not None: c += 0.1
        return min(c, 1.0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "consciousness_level": self.consciousness_level,
            "autonomy_level": self.autonomy_level,
            "quantum_resonance": self.quantum_resonance,
            "dimensions": {d.name: v for d, v in self.dimensions.items()},
            "active_goals": len(self.active_goals),
            "active_tools": [t.name for t in self.tools if t.is_active],
            "memory_usage": len(self.quantum_memory) / self.memory_capacity,
            "experience_count": len(self.experience_buffer),
            "total_actions": self.total_actions,
            "success_rate": self.successful_actions / max(self.total_actions, 1),
            "current_plan": self.current_plan.plan_id if self.current_plan else None,
        }
