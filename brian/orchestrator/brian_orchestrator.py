"""
Layer 5: Orchestration - BrianOrchestrator

The main brain loop that brings Brian to life inside a robot.

Connects all layers into a single running system:
    Sense  -> VisionPipeline + Sensors -> WorldState
    Think  -> BrianMind.perceive -> plan -> decide
    Act    -> SafetyGovernor.validate -> MotionController -> HAL
    Learn  -> BrianMind.reflect -> MemorySystem.consolidate

Control Loop (~30Hz perception, ~1kHz motor):
    1. Read sensors from HAL (via BrianProtocol or direct)
    2. Process vision + depth through VisionPipeline
    3. Build unified WorldState for BrianMind
    4. BrianMind decides next MotorCommand
    5. SafetyGovernor validates the command
    6. MotionController converts to JointCommands
    7. HAL sends to robot hardware
    8. BrianMind reflects on outcome

Lifecycle:
    BrianOrchestrator.create(config) -> orchestrator
    orchestrator.start()  -> initializes all layers, begins loop
    orchestrator.set_goal("pick up the cup")  -> autonomous execution
    orchestrator.stop()   -> graceful shutdown

Developed by Brion Quantum - Quantum AI & Intelligence Company
"""

import asyncio
import json
import signal
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from brian.core.brian_mind import (
    ActionOutcome, AutonomousGoal, BrianMind, MotorCommand, WorldState,
)
from brian.core.memory.memory_system import (
    EpisodicEvent, MemoryImportance, MemorySystem, SkillMemory,
)
from brian.perception.vision.vision_pipeline import CameraIntrinsics, VisionPipeline
from brian.motion.control.motion_controller import MotionController
from brian.safety.safety_governor import SafetyGovernor, SafetyLimits, SafetyLevel
from brian.quantum.qvm_bridge import QVMBridge, QuantumResourceConfig
from brian.communication.brian_protocol import (
    BrianMessageType, BrianProtocolHandler,
)
from brian.hal.robot_interface import (
    ControlMode, JointCommand, JointState, RobotConfig, RobotInterface,
    RobotPose, SensorReading, SensorType,
)
from brian.hal.platforms.simulation_platform import (
    SimulationPlatform, load_simulation_config,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Orchestrator State
# ============================================================================

class OrchestratorState(Enum):
    """Lifecycle states of the orchestrator."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()       # initialized, waiting for goals
    RUNNING = auto()     # active control loop
    PAUSED = auto()      # loop paused but systems alive
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class RunMode(Enum):
    """How Brian is connected to the robot."""
    SIMULATION = auto()   # built-in physics sim (no hardware)
    LOCAL = auto()        # direct HAL connection (on-robot)
    REMOTE = auto()       # via BrianProtocol over network


@dataclass
class OrchestratorConfig:
    """Full configuration for the orchestrator."""
    # Paths
    brian_config_path: str = "config/brian_default.yaml"
    platform_config_path: str = "config/platforms/simulation.yaml"
    safety_config_path: str = "config/safety/safety_limits.yaml"
    memory_storage_path: str = "brian_memory"

    # Run mode
    run_mode: RunMode = RunMode.SIMULATION

    # Control loop timing
    perception_hz: float = 30.0    # vision processing rate
    decision_hz: float = 10.0      # BrianMind decision rate
    motor_hz: float = 100.0        # motor command rate
    telemetry_hz: float = 2.0      # status reporting rate
    memory_consolidation_interval_s: float = 300.0  # 5 min

    # Remote connection (if run_mode == REMOTE)
    robot_host: str = "localhost"
    robot_port: int = 50051
    hmac_key: str = "brian-qari-default-key"

    # Behavior
    auto_home_on_start: bool = True
    load_memory_on_start: bool = True
    save_memory_on_stop: bool = True
    max_idle_time_s: float = 0.0  # 0 = no idle timeout


@dataclass
class LoopMetrics:
    """Performance metrics for the main loop."""
    cycle_count: int = 0
    perception_cycles: int = 0
    decision_cycles: int = 0
    motor_cycles: int = 0
    total_runtime_s: float = 0.0
    avg_perception_ms: float = 0.0
    avg_decision_ms: float = 0.0
    avg_motor_ms: float = 0.0
    goals_completed: int = 0
    goals_failed: int = 0
    safety_violations: int = 0
    emergency_stops: int = 0


# ============================================================================
# BrianOrchestrator
# ============================================================================

class BrianOrchestrator:
    """
    The main brain loop that makes Brian live inside a robot.

    This is where Sense -> Think -> Act -> Learn comes together.
    Every cycle, Brian perceives the world, decides what to do,
    validates it for safety, executes it, and learns from the result.

    Usage:
        orchestrator = await BrianOrchestrator.create("config/brian_default.yaml")
        await orchestrator.start()
        orchestrator.set_goal("pick up the red cup")
        # ... Brian autonomously works toward the goal ...
        await orchestrator.stop()
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state = OrchestratorState.UNINITIALIZED

        # Subsystems (initialized in _initialize)
        self.mind: Optional[BrianMind] = None
        self.memory: Optional[MemorySystem] = None
        self.vision: Optional[VisionPipeline] = None
        self.motion: Optional[MotionController] = None
        self.safety: Optional[SafetyGovernor] = None
        self.quantum: Optional[QVMBridge] = None
        self.protocol: Optional[BrianProtocolHandler] = None
        self.robot: Optional[RobotInterface] = None

        # Runtime state
        self._brian_config: Dict[str, Any] = {}
        self._robot_config: Optional[RobotConfig] = None
        self._current_joint_states: List[JointState] = []
        self._current_world_state: Optional[WorldState] = None
        self._current_command: Optional[MotorCommand] = None
        self._last_perception_time: float = 0.0
        self._last_decision_time: float = 0.0
        self._last_motor_time: float = 0.0
        self._last_telemetry_time: float = 0.0
        self._last_consolidation_time: float = 0.0
        self._start_time: float = 0.0
        self._metrics = LoopMetrics()

        # Async control
        self._main_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Goal queue
        self._goal_queue: List[AutonomousGoal] = []

        # Telemetry callbacks
        self._telemetry_callbacks: List[Any] = []

    # === Factory ===

    @classmethod
    async def create(cls, config_path: str = "config/brian_default.yaml",
                     run_mode: RunMode = RunMode.SIMULATION,
                     platform_config_path: str = "config/platforms/simulation.yaml",
                     ) -> 'BrianOrchestrator':
        """
        Create and initialize a BrianOrchestrator from config files.

        Args:
            config_path: Path to brian_default.yaml.
            run_mode: How to connect to the robot.
            platform_config_path: Path to platform config (for simulation).

        Returns:
            Initialized BrianOrchestrator ready to start().
        """
        orch_config = OrchestratorConfig(
            brian_config_path=config_path,
            platform_config_path=platform_config_path,
            run_mode=run_mode,
        )
        orchestrator = cls(orch_config)
        await orchestrator._initialize()
        return orchestrator

    # === Lifecycle ===

    async def _initialize(self) -> None:
        """Initialize all subsystems from configuration."""
        self.state = OrchestratorState.INITIALIZING
        logger.info("="*60)
        logger.info("  Brian-QARI Orchestrator Initializing")
        logger.info("  Brion Quantum - Quantum AI & Intelligence")
        logger.info("="*60)

        # 1. Load configuration
        self._brian_config = self._load_yaml(self.config.brian_config_path)
        brian_cfg = self._brian_config.get("brian", {})

        # 2. Initialize Quantum VM Bridge
        quantum_cfg = brian_cfg.get("quantum", {})
        self.quantum = QVMBridge(QuantumResourceConfig(
            max_qubits=quantum_cfg.get("max_qubits", 30),
            default_shots=quantum_cfg.get("default_shots", 1000),
            use_gpu=quantum_cfg.get("use_gpu", False),
        ))
        logger.info(f"[QVM] Backends: {self.quantum.get_available_backends()}")

        # 3. Initialize BrianMind (cognitive core)
        core_cfg = brian_cfg.get("core", {})
        self.mind = BrianMind(config=core_cfg)
        logger.info(f"[Mind] Initialized | tools={len(self.mind.tools)}")

        # 4. Initialize Memory System
        self.memory = MemorySystem(config={
            "embedding_dim": core_cfg.get("embedding_dim", 2048),
            "storage_path": self.config.memory_storage_path,
        })
        if self.config.load_memory_on_start:
            loaded = self.memory.load()
            logger.info(f"[Memory] Loaded {loaded} memories from disk")

        # 5. Initialize Safety Governor
        safety_cfg = brian_cfg.get("safety", {})
        safety_limits = SafetyLimits(
            max_cartesian_velocity=safety_cfg.get("max_cartesian_velocity", 1.5),
            max_cartesian_acceleration=safety_cfg.get("max_cartesian_acceleration", 5.0),
            max_contact_force=safety_cfg.get("max_contact_force", 150.0),
            min_human_distance=safety_cfg.get("min_human_distance", 0.5),
            speed_reduction_distance=safety_cfg.get("speed_reduction_distance", 1.5),
            emergency_stop_distance=safety_cfg.get("emergency_stop_distance", 0.2),
            heartbeat_timeout_ms=safety_cfg.get("heartbeat_timeout_ms", 200.0),
            max_command_age_ms=safety_cfg.get("max_command_age_ms", 500.0),
        )
        self.safety = SafetyGovernor(safety_limits)
        self.safety.on_emergency_stop(self._on_emergency_stop)
        self.safety.on_violation(self._on_safety_violation)
        logger.info("[Safety] 5-zone governor active")

        # 6. Initialize Robot HAL
        if self.config.run_mode == RunMode.SIMULATION:
            self._robot_config = load_simulation_config(self.config.platform_config_path)
            self.robot = SimulationPlatform(self._robot_config)
            success = await self.robot.initialize()
            if not success:
                raise RuntimeError("Failed to initialize simulation platform")
            logger.info(f"[HAL] Simulation platform: {self._robot_config.name} "
                        f"({self._robot_config.num_joints} joints)")
        elif self.config.run_mode == RunMode.REMOTE:
            # Initialize protocol for remote connection
            comm_cfg = brian_cfg.get("communication", {})
            self.protocol = BrianProtocolHandler(
                source_id="brian-brain",
                hmac_key=self.config.hmac_key.encode(),
            )
            self.protocol.max_command_age_ms = comm_cfg.get("max_command_age_ms", 500.0)
            logger.info(f"[Protocol] Remote mode -> {self.config.robot_host}:{self.config.robot_port}")

        # 7. Initialize Vision Pipeline
        self.vision = VisionPipeline(config={
            "detection_confidence": 0.5,
            "device": "cpu",
        })
        logger.info("[Vision] Pipeline ready (detector + SLAM + depth + tracking)")

        # 8. Initialize Motion Controller
        if self._robot_config:
            self.motion = MotionController(self._robot_config)
            logger.info(f"[Motion] Controller ready | dt={1000/self._robot_config.control_frequency_hz:.1f}ms")

        # 9. Populate safety limits from joint configs
        if self._robot_config:
            for jc in self._robot_config.joints:
                j = jc if not isinstance(jc, dict) else type('J', (), jc)()
                jid = j.joint_id if hasattr(j, 'joint_id') else j.id
                safety_limits.joint_position_limits[jid] = (j.position_min, j.position_max)
                safety_limits.joint_velocity_limits[jid] = j.velocity_max
                safety_limits.joint_torque_limits[jid] = j.torque_max

        # 10. Home the robot
        if self.config.auto_home_on_start and self.robot:
            logger.info("[HAL] Homing robot...")
            await self.robot.home()

        self.state = OrchestratorState.READY
        logger.info("="*60)
        logger.info("  Brian-QARI Ready")
        logger.info(f"  Mode: {self.config.run_mode.name}")
        logger.info(f"  Perception: {self.config.perception_hz}Hz")
        logger.info(f"  Decision: {self.config.decision_hz}Hz")
        logger.info(f"  Motor: {self.config.motor_hz}Hz")
        logger.info("="*60)

    async def start(self) -> None:
        """Start the main brain loop."""
        if self.state != OrchestratorState.READY:
            raise RuntimeError(f"Cannot start from state {self.state.name}")

        self.state = OrchestratorState.RUNNING
        self._start_time = time.time()
        self._shutdown_event.clear()
        logger.info("Brian is awake. Main loop starting.")

        # Launch the main loop as an async task
        self._main_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Gracefully stop the brain loop and shut down all systems."""
        if self.state not in (OrchestratorState.RUNNING, OrchestratorState.PAUSED):
            return

        self.state = OrchestratorState.STOPPING
        logger.info("Brian is shutting down...")
        self._shutdown_event.set()

        # Wait for main loop to finish
        if self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._main_task.cancel()
                logger.warning("Main loop did not stop gracefully, cancelled")

        # Save memory
        if self.config.save_memory_on_stop and self.memory:
            self.memory.save()
            logger.info(f"[Memory] Saved {self.memory.get_stats()['total_memories']} memories")

        # Shutdown robot
        if self.robot:
            await self.robot.shutdown()
            logger.info("[HAL] Robot shutdown complete")

        self.state = OrchestratorState.STOPPED
        runtime = time.time() - self._start_time
        logger.info(f"Brian has stopped. Runtime: {runtime:.1f}s | "
                    f"Cycles: {self._metrics.cycle_count} | "
                    f"Goals completed: {self._metrics.goals_completed}")

    async def pause(self) -> None:
        """Pause the brain loop (robot holds position)."""
        if self.state == OrchestratorState.RUNNING:
            self.state = OrchestratorState.PAUSED
            logger.info("Brian paused")

    async def resume(self) -> None:
        """Resume the brain loop from pause."""
        if self.state == OrchestratorState.PAUSED:
            self.state = OrchestratorState.RUNNING
            logger.info("Brian resumed")

    # === Goal Management ===

    def set_goal(self, description: str, priority: float = 0.5,
                 success_criteria: Optional[List[str]] = None) -> AutonomousGoal:
        """
        Set a new goal for Brian to pursue autonomously.

        Examples:
            orchestrator.set_goal("pick up the red cup")
            orchestrator.set_goal("walk to the table", priority=0.8)
            orchestrator.set_goal("wave hello to the person", priority=0.3)
        """
        goal = self.mind.set_goal(description, priority, success_criteria)
        self._goal_queue.append(goal)
        logger.info(f"[Goal] New: '{description}' (priority={priority})")
        return goal

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel an active goal."""
        for goal in self.mind.active_goals:
            if goal.goal_id == goal_id:
                goal.status = "cancelled"
                self.mind.active_goals.remove(goal)
                logger.info(f"[Goal] Cancelled: {goal_id}")
                return True
        return False

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get all active goals with status."""
        return [{
            "goal_id": g.goal_id,
            "description": g.description,
            "priority": g.priority,
            "progress": g.current_progress,
            "status": g.status,
        } for g in self.mind.active_goals]

    # === Main Loop ===

    async def _main_loop(self) -> None:
        """
        The core sense-think-act-learn loop.

        Runs at ~100Hz motor rate with perception and decision at lower rates.
        """
        perception_interval = 1.0 / self.config.perception_hz
        decision_interval = 1.0 / self.config.decision_hz
        motor_interval = 1.0 / self.config.motor_hz
        telemetry_interval = 1.0 / self.config.telemetry_hz
        consolidation_interval = self.config.memory_consolidation_interval_s

        self._last_perception_time = 0.0
        self._last_decision_time = 0.0
        self._last_motor_time = 0.0
        self._last_telemetry_time = 0.0
        self._last_consolidation_time = time.time()

        logger.info("[Loop] Entering main brain loop")

        while not self._shutdown_event.is_set():
            try:
                if self.state == OrchestratorState.PAUSED:
                    await asyncio.sleep(0.1)
                    continue

                now = time.time()
                self._metrics.cycle_count += 1

                # === SENSE (perception rate) ===
                if now - self._last_perception_time >= perception_interval:
                    await self._sense()
                    self._last_perception_time = now

                # === THINK (decision rate) ===
                if now - self._last_decision_time >= decision_interval:
                    await self._think()
                    self._last_decision_time = now

                # === ACT (motor rate) ===
                if now - self._last_motor_time >= motor_interval:
                    await self._act()
                    self._last_motor_time = now

                # === TELEMETRY ===
                if now - self._last_telemetry_time >= telemetry_interval:
                    self._emit_telemetry()
                    self._last_telemetry_time = now

                # === MEMORY CONSOLIDATION ===
                if now - self._last_consolidation_time >= consolidation_interval:
                    self._consolidate_memory()
                    self._last_consolidation_time = now

                # Yield to event loop
                await asyncio.sleep(0.0001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Loop] Error in main loop: {e}", exc_info=True)
                self._metrics.safety_violations += 1
                await asyncio.sleep(0.01)

        logger.info("[Loop] Main brain loop exited")

    # === Sense ===

    async def _sense(self) -> None:
        """
        Read sensors and build a world state.
        Runs at perception_hz (~30Hz).
        """
        t_start = time.time()

        if not self.robot:
            return

        # Read joint states
        self._current_joint_states = await self.robot.get_joint_states()

        # Read camera (RGB + depth)
        rgb_reading = await self.robot.get_sensor_reading(SensorType.RGB_CAMERA, "head_cam")
        depth_reading = await self.robot.get_sensor_reading(SensorType.DEPTH_CAMERA, "head_depth")

        rgb_image = rgb_reading.data if rgb_reading else None
        depth_image = depth_reading.data if depth_reading else None

        # Run vision pipeline
        vision_result = {}
        if rgb_image is not None:
            vision_result = self.vision.process(rgb_image, depth_image)

        # Build unified sensor data for BrianMind
        sensor_data = {}

        # Joint states
        if self._current_joint_states:
            sensor_data["joint_states"] = self._current_joint_states

        # Robot pose
        robot_pose = await self.robot.get_robot_pose()
        sensor_data["robot_pose"] = {
            "position": robot_pose.position.tolist(),
            "orientation": robot_pose.orientation.tolist(),
        }

        # Vision-derived data
        if vision_result:
            world_updates = self.vision.get_world_state_updates(vision_result)
            sensor_data.update(world_updates)

        # Force/torque
        ft_reading = await self.robot.get_sensor_reading(SensorType.FORCE_TORQUE, "r_wrist_ft")
        if ft_reading is not None:
            sensor_data["force_torque"] = [{"sensor_id": "r_wrist", "data": ft_reading.data.tolist()}]

        # IMU
        imu_reading = await self.robot.get_sensor_reading(SensorType.IMU, "torso_imu")
        if imu_reading is not None:
            sensor_data["imu"] = imu_reading.data.tolist()

        # Feed into BrianMind's perception
        self._current_world_state = self.mind.perceive(sensor_data)

        # Update working memory with current context
        self.memory.working_memory.push(
            "world_state",
            f"objects={len(self._current_world_state.objects)} "
            f"people={len(self._current_world_state.people)} "
            f"confidence={self._current_world_state.confidence:.2f}",
            priority=0.8,
        )

        dt_ms = (time.time() - t_start) * 1000
        self._metrics.perception_cycles += 1
        self._metrics.avg_perception_ms = (
            0.95 * self._metrics.avg_perception_ms + 0.05 * dt_ms)

    # === Think ===

    async def _think(self) -> None:
        """
        BrianMind decision cycle: plan and decide.
        Runs at decision_hz (~10Hz).
        """
        t_start = time.time()

        if not self.mind or not self._current_world_state:
            return

        # Select highest-priority active goal
        active_goal = None
        for goal in self.mind.active_goals:
            if goal.status == "active":
                active_goal = goal
                break

        if active_goal is None:
            # No active goals - idle behavior (look around slowly)
            self._current_command = MotorCommand(
                head_target=np.array([
                    0.3 * np.sin(time.time() * 0.5),
                    0.1 * np.sin(time.time() * 0.3),
                ]),
                confidence=0.5,
            )
            return

        # Plan if we don't have a current plan for this goal
        if (self.mind.current_plan is None or
                self.mind.current_plan.goal.goal_id != active_goal.goal_id or
                self.mind.current_plan.status in ("completed", "failed")):
            plan = self.mind.plan(active_goal, self._current_world_state)
            logger.info(f"[Think] New plan: {plan.plan_id} | "
                        f"{len(plan.steps)} steps for '{active_goal.description}'")

        # Decide next motor command
        if self.mind.current_plan:
            self._current_command = self.mind.decide(
                self.mind.current_plan, self._current_world_state)

            # Update working memory with current action
            if self.mind.current_plan.current_step_index < len(self.mind.current_plan.steps):
                step = self.mind.current_plan.steps[self.mind.current_plan.current_step_index]
                self.memory.working_memory.push(
                    "current_action",
                    f"{step.action_type}: {step.description}",
                    priority=1.0,
                )

            # Check if plan completed
            if self.mind.current_plan.status == "completed":
                self._metrics.goals_completed += 1
                logger.info(f"[Think] Goal completed: '{active_goal.description}'")

                # Store episodic memory of the completed goal
                self._store_goal_memory(active_goal, "success")

        # Recall relevant memories for context
        if self._current_world_state:
            relevant_memories = self.memory.recall(
                self._current_world_state.semantic_context, k=3)
            for recall in relevant_memories:
                self.memory.working_memory.push(
                    f"memory_{recall.memory.memory_id[:6]}",
                    str(recall.memory.content),
                    priority=recall.relevance,
                )

        dt_ms = (time.time() - t_start) * 1000
        self._metrics.decision_cycles += 1
        self._metrics.avg_decision_ms = (
            0.95 * self._metrics.avg_decision_ms + 0.05 * dt_ms)

    # === Act ===

    async def _act(self) -> None:
        """
        Validate, convert, and send motor commands.
        Runs at motor_hz (~100Hz).
        """
        t_start = time.time()

        if not self._current_command or not self.robot:
            return

        command = self._current_command

        # Build safety context
        safety_context = {
            "joint_states": [
                {"joint_id": js.joint_id, "position": js.position,
                 "velocity": js.velocity, "torque": js.torque}
                for js in self._current_joint_states
            ],
            "detected_people": (
                self._current_world_state.people
                if self._current_world_state else []
            ),
            "robot_position": (
                self._current_world_state.robot_pose.get("position", [0, 0, 0])
                if self._current_world_state and isinstance(
                    getattr(self._current_world_state, 'robot_pose', None), dict)
                else [0, 0, 0]
            ),
            "ethical_score": command.safety_flags.get("ethical_validated", 1.0),
        }

        # Validate through safety governor
        is_safe, validated_command, violations = self.safety.validate_command(
            command, safety_context)

        if not is_safe:
            self._metrics.safety_violations += len(violations)
            for v in violations:
                logger.warning(f"[Safety] {v.severity.name}: {v.description} -> {v.action_taken}")

            if any(v.severity == SafetyLevel.EMERGENCY for v in violations):
                await self.robot.emergency_stop()
                self._metrics.emergency_stops += 1
                return

            # Apply speed reduction
            speed_factor = self.safety.state.speed_reduction_factor
            if command.base_velocity is not None:
                command.base_velocity *= speed_factor
            if command.joint_commands:
                for jc in command.joint_commands:
                    jc["velocity"] = jc.get("velocity", 0) * speed_factor

        # Convert MotorCommand to JointCommands via MotionController
        if self.motion:
            joint_commands = self.motion.execute(command, self._current_joint_states)
        else:
            joint_commands = self._command_to_joints(command)

        # Send to robot
        if joint_commands:
            success = await self.robot.send_joint_commands(joint_commands)

            # Learn from outcome
            outcome = ActionOutcome(
                success=success,
                actual_joint_states=self._current_joint_states,
                error_metric=0.01 if success else 0.5,
                duration_ms=(time.time() - t_start) * 1000,
            )
            self.mind.reflect(outcome)

        dt_ms = (time.time() - t_start) * 1000
        self._metrics.motor_cycles += 1
        self._metrics.avg_motor_ms = (
            0.95 * self._metrics.avg_motor_ms + 0.05 * dt_ms)

    # === Memory ===

    def _store_goal_memory(self, goal: AutonomousGoal, outcome: str) -> None:
        """Store an episodic memory about a completed/failed goal."""
        event = EpisodicEvent(
            event_type=f"goal_{outcome}",
            objects=[],
            outcome=outcome,
            duration_s=0.0,
        )
        importance = (MemoryImportance.HIGH if outcome == "success"
                      else MemoryImportance.MEDIUM)
        vector = self.mind.quantum_lm.quantum_encode(goal.description)
        self.memory.store_episodic(event, vector, importance,
                                   tags=["goal", outcome, goal.description[:30]])

    def _consolidate_memory(self) -> None:
        """Periodically consolidate episodic memories into semantic knowledge."""
        if self.memory and self.mind:
            new_facts = self.memory.consolidate(self.mind.quantum_lm)
            forgotten = self.memory.decay(dt=self.config.memory_consolidation_interval_s)
            if new_facts > 0 or forgotten > 0:
                logger.info(f"[Memory] Consolidated: +{new_facts} facts, -{forgotten} forgotten")

    # === Safety Callbacks ===

    def _on_emergency_stop(self) -> None:
        """Called when safety governor triggers emergency stop."""
        logger.critical("[EMERGENCY] Safety governor triggered E-STOP!")
        self._metrics.emergency_stops += 1
        if self.robot:
            asyncio.create_task(self.robot.emergency_stop())

    def _on_safety_violation(self, violations) -> None:
        """Called when safety violations are detected."""
        for v in violations:
            if v.severity in (SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY):
                logger.warning(f"[Safety] Zone {v.zone} {v.severity.name}: "
                               f"{v.description}")

    # === Telemetry ===

    def _emit_telemetry(self) -> None:
        """Emit status telemetry at regular intervals."""
        self._metrics.total_runtime_s = time.time() - self._start_time

        telemetry = self.get_status()
        for callback in self._telemetry_callbacks:
            try:
                callback(telemetry)
            except Exception as e:
                logger.error(f"Telemetry callback error: {e}")

    def on_telemetry(self, callback) -> None:
        """Register a telemetry callback."""
        self._telemetry_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of Brian and all subsystems."""
        status = {
            "orchestrator": {
                "state": self.state.name,
                "run_mode": self.config.run_mode.name,
                "runtime_s": self._metrics.total_runtime_s,
            },
            "metrics": {
                "cycle_count": self._metrics.cycle_count,
                "perception_hz_actual": (
                    self._metrics.perception_cycles / max(self._metrics.total_runtime_s, 1)),
                "decision_hz_actual": (
                    self._metrics.decision_cycles / max(self._metrics.total_runtime_s, 1)),
                "motor_hz_actual": (
                    self._metrics.motor_cycles / max(self._metrics.total_runtime_s, 1)),
                "avg_perception_ms": self._metrics.avg_perception_ms,
                "avg_decision_ms": self._metrics.avg_decision_ms,
                "avg_motor_ms": self._metrics.avg_motor_ms,
                "goals_completed": self._metrics.goals_completed,
                "goals_failed": self._metrics.goals_failed,
                "safety_violations": self._metrics.safety_violations,
                "emergency_stops": self._metrics.emergency_stops,
            },
        }

        if self.mind:
            status["mind"] = self.mind.get_status()
        if self.memory:
            status["memory"] = self.memory.get_stats()
        if self.vision:
            status["vision"] = self.vision.get_status()
        if self.motion:
            status["motion"] = self.motion.get_status()
        if self.safety:
            status["safety"] = self.safety.get_status_report()
        if self.quantum:
            status["quantum"] = {"backends": self.quantum.get_available_backends()}

        return status

    # === Utilities ===

    def _command_to_joints(self, command: MotorCommand) -> List[JointCommand]:
        """Fallback: convert MotorCommand to JointCommands without MotionController."""
        commands = []
        for jc in command.joint_commands:
            commands.append(JointCommand(
                joint_id=jc.get("joint_id", 0),
                position=jc.get("position", 0.0),
                velocity=jc.get("velocity", 0.0),
                mode=ControlMode.POSITION,
            ))
        return commands

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Config not found: {path}, using defaults")
            return {}
        with open(p, 'r') as f:
            return yaml.safe_load(f) or {}


# ============================================================================
# CLI Entry Point
# ============================================================================

async def main(config_path: str = "config/brian_default.yaml",
               platform_path: str = "config/platforms/simulation.yaml"):
    """
    Launch Brian-QARI from the command line.

    Usage:
        python -m brian.orchestrator.brian_orchestrator
        python -m brian.orchestrator.brian_orchestrator --config config/brian_default.yaml
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting Brian-QARI...")

    orchestrator = await BrianOrchestrator.create(
        config_path=config_path,
        platform_config_path=platform_path,
        run_mode=RunMode.SIMULATION,
    )

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    # Print telemetry to console
    def print_telemetry(status):
        mind = status.get("mind", {})
        metrics = status.get("metrics", {})
        safety = status.get("safety", {})
        print(f"\r[Brian] consciousness={mind.get('consciousness_level', 0):.3f} | "
              f"goals={mind.get('active_goals', 0)} | "
              f"perception={metrics.get('avg_perception_ms', 0):.1f}ms | "
              f"decision={metrics.get('avg_decision_ms', 0):.1f}ms | "
              f"safety={safety.get('overall_level', 'NOMINAL')} | "
              f"violations={metrics.get('safety_violations', 0)}",
              end="", flush=True)

    orchestrator.on_telemetry(print_telemetry)

    # Start the brain
    await orchestrator.start()

    # Set a demo goal
    orchestrator.set_goal("observe the environment and learn", priority=0.3)

    # Run until shutdown signal
    await stop_event.wait()
    await orchestrator.stop()

    # Final status
    status = orchestrator.get_status()
    print(f"\n\nFinal Status:")
    print(f"  Runtime: {status['orchestrator']['runtime_s']:.1f}s")
    print(f"  Cycles: {status['metrics']['cycle_count']}")
    print(f"  Goals Completed: {status['metrics']['goals_completed']}")
    print(f"  Memory: {status.get('memory', {}).get('total_memories', 0)} memories")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brian-QARI Orchestrator")
    parser.add_argument("--config", default="config/brian_default.yaml",
                        help="Path to brian config YAML")
    parser.add_argument("--platform", default="config/platforms/simulation.yaml",
                        help="Path to platform config YAML")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.platform))
