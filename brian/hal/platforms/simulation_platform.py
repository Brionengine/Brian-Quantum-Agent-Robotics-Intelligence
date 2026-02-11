"""
Layer 1: HAL - Simulation Platform Driver

Implements the RobotInterface for a simulated humanoid robot.
This is the primary development and testing platform for Brian-QARI.

Supports three simulation backends:
  - Built-in: Pure Python physics (no dependencies, always available)
  - MuJoCo:   High-fidelity physics via mujoco-py / dm_control
  - Isaac Sim: NVIDIA Isaac for GPU-accelerated simulation

The built-in simulator provides:
  - Joint dynamics with configurable stiffness/damping
  - Basic contact/collision detection
  - Synthetic sensor data generation (RGB, depth, IMU, F/T)
  - Gravity and friction simulation
  - Multi-robot support

This allows full Brian-QARI development without any external simulator,
while supporting drop-in replacement with MuJoCo/Isaac for production testing.
"""

import asyncio
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from brian.hal.robot_interface import (
    ActuatorType, ControlMode, JointCommand, JointConfig, JointState,
    RobotConfig, RobotInterface, RobotPose, RobotStatus,
    SensorReading, SensorType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Physics Simulation
# ============================================================================

@dataclass
class SimJoint:
    """Simulated joint with basic physics."""
    config: JointConfig
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    torque: float = 0.0
    target_position: float = 0.0
    target_velocity: float = 0.0
    target_torque: float = 0.0
    mode: ControlMode = ControlMode.POSITION
    temperature: float = 25.0  # degrees C
    kp: float = 100.0   # position gain
    kd: float = 10.0    # velocity gain
    friction: float = 0.5  # Nm
    inertia: float = 0.1   # kg*m^2


@dataclass
class SimSensor:
    """Simulated sensor with configurable noise."""
    sensor_type: SensorType
    sensor_id: str
    rate_hz: float = 30.0
    noise_stddev: float = 0.01
    last_reading_time: float = 0.0
    resolution: Tuple[int, int] = (640, 480)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicsEngine:
    """
    Lightweight rigid-body physics for humanoid simulation.
    Simulates joint dynamics, gravity, and basic contact.
    """

    def __init__(self, dt: float = 0.001, gravity: float = 9.81):
        self.dt = dt
        self.gravity = gravity
        self.sim_time = 0.0
        self._joints: Dict[int, SimJoint] = {}
        self._base_pose = np.eye(4)
        self._base_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self._com_position = np.zeros(3)
        self._contact_forces: Dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(42)

    def add_joint(self, config: JointConfig, **kwargs) -> None:
        """Add a joint to the simulation."""
        self._joints[config.joint_id] = SimJoint(
            config=config,
            position=config.default_position,
            kp=kwargs.get("kp", config.stiffness_default),
            kd=kwargs.get("kd", config.damping_default),
            inertia=kwargs.get("inertia", self._estimate_inertia(config)),
        )

    def step(self) -> None:
        """Advance simulation by one timestep."""
        for jid, joint in self._joints.items():
            self._step_joint(joint)

        # Update base pose from leg kinematics (simplified)
        self._update_base_pose()
        self._update_contact_forces()
        self.sim_time += self.dt

    def set_command(self, joint_id: int, command: JointCommand) -> None:
        """Set the target command for a joint."""
        if joint_id not in self._joints:
            return
        joint = self._joints[joint_id]
        joint.mode = command.mode
        joint.target_position = command.position
        joint.target_velocity = command.velocity
        joint.target_torque = command.torque
        if command.stiffness > 0:
            joint.kp = command.stiffness
        if command.damping > 0:
            joint.kd = command.damping

    def get_joint_state(self, joint_id: int) -> Optional[JointState]:
        """Get current state of a joint."""
        if joint_id not in self._joints:
            return None
        j = self._joints[joint_id]
        return JointState(
            joint_id=joint_id,
            position=j.position,
            velocity=j.velocity,
            torque=j.torque,
            temperature=j.temperature,
        )

    def get_base_pose(self) -> RobotPose:
        """Get the robot's base pose."""
        pos = self._base_pose[:3, 3]
        R = self._base_pose[:3, :3]
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R).as_quat()  # xyzw
        orientation = np.array([quat[3], quat[0], quat[1], quat[2]])  # wxyz

        return RobotPose(
            position=pos.copy(),
            orientation=orientation,
            linear_velocity=self._base_velocity[:3].copy(),
            angular_velocity=self._base_velocity[3:].copy(),
        )

    def _step_joint(self, joint: SimJoint) -> None:
        """Simulate one joint for one timestep using PD control."""
        cfg = joint.config

        if joint.mode == ControlMode.POSITION:
            # PD controller
            pos_error = joint.target_position - joint.position
            vel_error = joint.target_velocity - joint.velocity
            torque = joint.kp * pos_error + joint.kd * vel_error
        elif joint.mode == ControlMode.VELOCITY:
            vel_error = joint.target_velocity - joint.velocity
            torque = joint.kd * vel_error * 10.0
        elif joint.mode == ControlMode.TORQUE:
            torque = joint.target_torque
        elif joint.mode == ControlMode.IMPEDANCE:
            pos_error = joint.target_position - joint.position
            vel_error = -joint.velocity
            torque = joint.kp * pos_error + joint.kd * vel_error
        else:
            torque = 0.0

        # Friction
        if abs(joint.velocity) > 0.001:
            torque -= joint.friction * np.sign(joint.velocity)

        # Clamp torque
        torque = np.clip(torque, -cfg.torque_max, cfg.torque_max)

        # Integrate dynamics: tau = I * alpha
        alpha = torque / max(joint.inertia, 0.001)
        joint.velocity += alpha * self.dt
        joint.velocity = np.clip(joint.velocity, -cfg.velocity_max, cfg.velocity_max)
        joint.position += joint.velocity * self.dt
        joint.position = np.clip(joint.position, cfg.position_min, cfg.position_max)
        joint.torque = torque
        joint.acceleration = alpha

        # Thermal model (simplified)
        heat_generated = abs(torque * joint.velocity) * 0.1  # W
        heat_dissipated = (joint.temperature - 25.0) * 0.01
        joint.temperature += (heat_generated - heat_dissipated) * self.dt

    def _update_base_pose(self) -> None:
        """Update base pose from leg joint states (simplified standing model)."""
        # COM height from leg extension
        r_knee = self._joints.get(20)
        l_knee = self._joints.get(26)
        if r_knee and l_knee:
            avg_knee = (r_knee.position + l_knee.position) / 2
            leg_length = 0.82  # approximate leg length
            height = leg_length * math.cos(avg_knee / 2)
            self._base_pose[1, 3] = height

        # Forward motion from hip pitch
        r_hip = self._joints.get(17)
        l_hip = self._joints.get(23)
        if r_hip and l_hip:
            avg_hip_pitch = (r_hip.position + l_hip.position) / 2
            self._base_pose[0, 3] += avg_hip_pitch * 0.01 * self.dt

    def _update_contact_forces(self) -> None:
        """Estimate contact forces at feet and wrists."""
        # Right foot contact
        r_ankle = self._joints.get(21)
        if r_ankle:
            ground_contact = max(0, -r_ankle.position) * 500  # spring-like
            self._contact_forces["r_ankle_ft"] = np.array([
                0, 0, ground_contact, r_ankle.torque, 0, 0])

        # Left foot contact
        l_ankle = self._joints.get(27)
        if l_ankle:
            ground_contact = max(0, -l_ankle.position) * 500
            self._contact_forces["l_ankle_ft"] = np.array([
                0, 0, ground_contact, l_ankle.torque, 0, 0])

        # Wrist F/T (from gripper interaction)
        for prefix, gripper_id in [("r_wrist_ft", 29), ("l_wrist_ft", 30)]:
            gripper = self._joints.get(gripper_id)
            if gripper:
                grip_force = abs(gripper.torque)
                self._contact_forces[prefix] = np.array([
                    0, 0, grip_force, 0, 0, 0])

    def _estimate_inertia(self, config: JointConfig) -> float:
        """Estimate joint inertia from name/type."""
        name = config.name.lower()
        if "hip" in name:
            return 2.0
        elif "knee" in name:
            return 1.5
        elif "ankle" in name:
            return 0.5
        elif "shoulder" in name:
            return 1.0
        elif "elbow" in name:
            return 0.5
        elif "wrist" in name:
            return 0.1
        elif "gripper" in name:
            return 0.05
        elif "head" in name:
            return 0.3
        elif "torso" in name:
            return 3.0
        return 0.1


# ============================================================================
# Synthetic Sensor Data Generation
# ============================================================================

class SensorSimulator:
    """
    Generates synthetic sensor data for the simulated robot.
    Produces realistic RGB images, depth maps, IMU readings, and F/T data.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self._rng = rng or np.random.RandomState(42)
        self._imu_bias_gyro = self._rng.randn(3) * 0.001
        self._imu_bias_accel = self._rng.randn(3) * 0.01

    def generate_rgb(self, width: int = 640, height: int = 480,
                     scene_objects: Optional[List[Dict]] = None) -> np.ndarray:
        """Generate a synthetic RGB image with basic scene rendering."""
        image = np.full((height, width, 3), [180, 200, 220], dtype=np.uint8)  # sky

        # Floor gradient
        for y in range(height // 2, height):
            t = (y - height // 2) / (height // 2)
            gray = int(120 + 60 * t)
            image[y, :] = [gray, gray + 10, gray - 10]

        # Render simple objects as colored rectangles
        if scene_objects:
            for obj in scene_objects:
                x, y_pos, w, h = obj.get("bbox", [100, 100, 50, 50])
                color = obj.get("color", [200, 100, 50])
                x1, y1 = max(0, x), max(0, y_pos)
                x2, y2 = min(width, x + w), min(height, y_pos + h)
                image[y1:y2, x1:x2] = color

        # Add noise
        noise = self._rng.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def generate_depth(self, width: int = 640, height: int = 480,
                       scene_objects: Optional[List[Dict]] = None) -> np.ndarray:
        """Generate a synthetic depth image."""
        # Base depth: floor plane receding into distance
        depth = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            if y > height // 2:
                t = (y - height // 2) / (height // 2)
                depth[y, :] = 0.5 + 5.0 * (1.0 - t)  # 0.5m to 5.5m
            else:
                depth[y, :] = 10.0  # far away (sky/ceiling)

        # Add objects at various depths
        if scene_objects:
            for obj in scene_objects:
                x, y_pos, w, h = obj.get("bbox", [100, 100, 50, 50])
                d = obj.get("depth", 2.0)
                x1, y1 = max(0, x), max(0, y_pos)
                x2, y2 = min(width, x + w), min(height, y_pos + h)
                depth[y1:y2, x1:x2] = d

        # Add sensor noise
        noise = self._rng.normal(0, 0.005, depth.shape).astype(np.float32)
        depth += noise
        depth = np.clip(depth, 0.1, 10.0)

        return depth

    def generate_imu(self, base_orientation: np.ndarray,
                     base_angular_velocity: np.ndarray,
                     base_linear_acceleration: np.ndarray) -> np.ndarray:
        """
        Generate synthetic IMU reading.
        Returns: [ax, ay, az, gx, gy, gz] (accel + gyro)
        """
        # Add gravity component
        gravity = np.array([0, 0, -9.81])
        accel = base_linear_acceleration - gravity

        # Add bias and noise
        accel += self._imu_bias_accel + self._rng.randn(3) * 0.02
        gyro = base_angular_velocity + self._imu_bias_gyro + self._rng.randn(3) * 0.005

        return np.concatenate([accel, gyro])

    def generate_force_torque(self, contact_force: np.ndarray) -> np.ndarray:
        """Generate synthetic F/T sensor reading [fx, fy, fz, tx, ty, tz]."""
        noise = self._rng.randn(6) * 0.1
        return contact_force + noise


# ============================================================================
# Simulation Platform Driver
# ============================================================================

class SimulationPlatform(RobotInterface):
    """
    Complete simulation platform implementing the RobotInterface.

    This is the primary development driver for Brian-QARI. It runs
    a built-in physics simulation that requires zero external dependencies
    beyond numpy/scipy, making it available everywhere.

    Usage:
        config = load_config("config/platforms/simulation.yaml")
        robot = SimulationPlatform(config)
        await robot.initialize()
        await robot.send_joint_commands([...])
        states = await robot.get_joint_states()
    """

    def __init__(self, config: RobotConfig,
                 physics_dt: float = 0.001,
                 realtime: bool = False):
        super().__init__(config)
        self.physics_dt = physics_dt
        self.realtime = realtime

        # Physics engine
        self._physics = PhysicsEngine(dt=physics_dt)

        # Sensor simulator
        self._sensor_sim = SensorSimulator()
        self._sensors: Dict[str, SimSensor] = {}

        # Simulation state
        self._scene_objects: List[Dict] = []
        self._step_count = 0
        self._wall_clock_start = 0.0
        self._status = RobotStatus()

        logger.info(f"SimulationPlatform created | robot={config.name} | "
                    f"joints={config.num_joints} | dt={physics_dt*1000:.1f}ms")

    async def initialize(self) -> bool:
        """Initialize the simulated robot."""
        try:
            # Create joints in physics engine
            for joint_data in self.config.joints:
                if isinstance(joint_data, dict):
                    jc = JointConfig(**joint_data)
                else:
                    jc = joint_data
                self._physics.add_joint(jc)

            # Create sensors
            for sensor_data in self.config.sensors:
                s = sensor_data if isinstance(sensor_data, dict) else vars(sensor_data)
                sensor = SimSensor(
                    sensor_type=SensorType[s["type"].upper()] if isinstance(s["type"], str) else s["type"],
                    sensor_id=s["id"],
                    rate_hz=s.get("rate_hz", s.get("fps", 30)),
                    resolution=tuple(s.get("resolution", [640, 480])),
                )
                self._sensors[s["id"]] = sensor

            # Add default scene objects for testing
            self._scene_objects = [
                {"bbox": [200, 200, 60, 80], "color": [220, 50, 50], "depth": 1.5,
                 "class": "cup"},
                {"bbox": [400, 180, 40, 100], "color": [50, 150, 220], "depth": 2.0,
                 "class": "bottle"},
                {"bbox": [100, 300, 200, 180], "color": [139, 90, 43], "depth": 3.0,
                 "class": "table"},
            ]

            self._is_initialized = True
            self._wall_clock_start = time.time()
            self._status.is_connected = True
            self._status.is_powered = True

            logger.info(f"SimulationPlatform initialized | "
                        f"{len(self._physics._joints)} joints | "
                        f"{len(self._sensors)} sensors")
            return True

        except Exception as e:
            logger.error(f"SimulationPlatform initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the simulation."""
        # Move to rest pose
        for jid in self._physics._joints:
            self._physics.set_command(jid, JointCommand(
                joint_id=jid, position=0.0, velocity=0.0,
                mode=ControlMode.POSITION))

        # Run a few steps to settle
        for _ in range(100):
            self._physics.step()

        self._is_initialized = False
        self._status.is_connected = False
        logger.info("SimulationPlatform shutdown complete")

    async def home(self) -> bool:
        """Move all joints to default positions."""
        for jid, joint in self._physics._joints.items():
            self._physics.set_command(jid, JointCommand(
                joint_id=jid, position=joint.config.default_position,
                velocity=0.0, mode=ControlMode.POSITION))

        # Step until converged or timeout
        for _ in range(5000):
            self._physics.step()
            max_error = max(
                abs(j.position - j.config.default_position)
                for j in self._physics._joints.values()
            )
            if max_error < 0.01:
                self._status.is_homed = True
                logger.info("SimulationPlatform homed successfully")
                return True

        logger.warning("SimulationPlatform homing did not fully converge")
        self._status.is_homed = True
        return True

    async def send_joint_commands(self, commands: List[JointCommand]) -> bool:
        """Send joint commands and step the simulation."""
        if not self._is_initialized:
            return False
        if self._emergency_stop_active:
            return False

        # Validate commands
        valid, violations = self.validate_joint_commands(commands)
        if not valid:
            for v in violations:
                logger.warning(f"Joint command violation: {v}")

        # Apply commands to physics
        for cmd in commands:
            self._physics.set_command(cmd.joint_id, cmd)

        # Step physics (multiple sub-steps for stability)
        control_dt = 1.0 / self.config.control_frequency_hz
        n_substeps = max(1, int(control_dt / self.physics_dt))
        for _ in range(n_substeps):
            self._physics.step()
        self._step_count += 1

        # Realtime sync
        if self.realtime:
            elapsed = time.time() - self._wall_clock_start
            sim_time = self._physics.sim_time
            if sim_time > elapsed:
                await asyncio.sleep(sim_time - elapsed)

        return True

    async def get_joint_states(self) -> List[JointState]:
        """Read all joint states."""
        states = []
        for jid in sorted(self._physics._joints.keys()):
            state = self._physics.get_joint_state(jid)
            if state is not None:
                states.append(state)
        return states

    async def get_sensor_reading(self, sensor_type: SensorType,
                                  sensor_id: str = "") -> Optional[SensorReading]:
        """Read from a specific simulated sensor."""
        # Find matching sensor
        sensor = None
        for sid, s in self._sensors.items():
            if s.sensor_type == sensor_type and (not sensor_id or sid == sensor_id):
                sensor = s
                break

        if sensor is None:
            return None

        timestamp = int(self._physics.sim_time * 1e9)

        if sensor_type == SensorType.RGB_CAMERA:
            w, h = sensor.resolution
            data = self._sensor_sim.generate_rgb(w, h, self._scene_objects)
            return SensorReading(
                sensor_type=sensor_type, sensor_id=sensor.sensor_id,
                timestamp_ns=timestamp, data=data)

        elif sensor_type == SensorType.DEPTH_CAMERA:
            w, h = sensor.resolution
            data = self._sensor_sim.generate_depth(w, h, self._scene_objects)
            return SensorReading(
                sensor_type=sensor_type, sensor_id=sensor.sensor_id,
                timestamp_ns=timestamp, data=data)

        elif sensor_type == SensorType.IMU:
            pose = self._physics.get_base_pose()
            data = self._sensor_sim.generate_imu(
                pose.orientation, pose.angular_velocity,
                np.array([0, 0, 0]))  # simplified
            return SensorReading(
                sensor_type=sensor_type, sensor_id=sensor.sensor_id,
                timestamp_ns=timestamp, data=data)

        elif sensor_type == SensorType.FORCE_TORQUE:
            contact = self._physics._contact_forces.get(
                sensor.sensor_id, np.zeros(6))
            data = self._sensor_sim.generate_force_torque(contact)
            return SensorReading(
                sensor_type=sensor_type, sensor_id=sensor.sensor_id,
                timestamp_ns=timestamp, data=data)

        elif sensor_type == SensorType.MICROPHONE:
            # Generate silent audio with background noise
            sample_rate = sensor.metadata.get("sample_rate", 16000)
            duration = 0.1  # 100ms chunk
            n_samples = int(sample_rate * duration)
            data = (self._sensor_sim._rng.randn(n_samples) * 100).astype(np.int16)
            return SensorReading(
                sensor_type=sensor_type, sensor_id=sensor.sensor_id,
                timestamp_ns=timestamp, data=data,
                metadata={"sample_rate": sample_rate})

        return None

    async def get_all_sensor_readings(self) -> List[SensorReading]:
        """Read all sensors."""
        readings = []
        for sensor in self._sensors.values():
            reading = await self.get_sensor_reading(
                sensor.sensor_type, sensor.sensor_id)
            if reading is not None:
                readings.append(reading)
        return readings

    async def get_status(self) -> RobotStatus:
        """Get simulation status."""
        self._status.uptime_seconds = time.time() - self._wall_clock_start
        self._status.battery_percent = 100.0  # infinite battery in sim
        self._status.is_in_error = False

        # Check motor temperatures
        for jid, joint in self._physics._joints.items():
            self._status.motor_temperatures[jid] = joint.temperature
            if joint.temperature > 80.0:
                self._status.is_in_error = True
                self._status.error_messages.append(
                    f"Joint {jid} overheating: {joint.temperature:.1f}C")

        return self._status

    async def emergency_stop(self) -> None:
        """Stop all motion immediately."""
        self._emergency_stop_active = True
        for jid in self._physics._joints:
            joint = self._physics._joints[jid]
            joint.target_velocity = 0.0
            joint.target_torque = 0.0
            joint.velocity = 0.0  # instant stop in sim
        logger.warning("SimulationPlatform: EMERGENCY STOP activated")

    async def clear_emergency_stop(self) -> bool:
        """Clear the emergency stop."""
        self._emergency_stop_active = False
        logger.info("SimulationPlatform: Emergency stop cleared")
        return True

    async def get_robot_pose(self) -> RobotPose:
        """Get the simulated robot's base pose."""
        return self._physics.get_base_pose()

    def get_config(self) -> RobotConfig:
        return self.config

    def get_supported_sensors(self) -> List[SensorType]:
        return list(set(s.sensor_type for s in self._sensors.values()))

    def get_joint_configs(self) -> List[JointConfig]:
        return [j.config for j in self._physics._joints.values()]

    # === Simulation-specific methods ===

    def add_scene_object(self, bbox: List[int], color: List[int],
                         depth: float, class_name: str = "unknown") -> None:
        """Add an object to the simulated scene."""
        self._scene_objects.append({
            "bbox": bbox, "color": color, "depth": depth, "class": class_name})

    def remove_scene_objects(self) -> None:
        """Clear all scene objects."""
        self._scene_objects.clear()

    def get_sim_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._physics.sim_time

    def get_step_count(self) -> int:
        return self._step_count

    def set_gravity(self, g: float) -> None:
        """Change gravity (useful for testing)."""
        self._physics.gravity = g


# ============================================================================
# Config Loader Helper
# ============================================================================

def load_simulation_config(config_path: str) -> RobotConfig:
    """Load a RobotConfig from a YAML platform configuration file."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    robot_data = raw.get("robot", raw)
    joints = []
    for j in robot_data.get("joints", []):
        joints.append(JointConfig(
            joint_id=j["id"],
            name=j["name"],
            joint_type=j["type"],
            position_min=j["pos_min"],
            position_max=j["pos_max"],
            velocity_max=j["vel_max"],
            torque_max=j["torque_max"],
            default_position=j.get("default_position", 0.0),
        ))

    platform = raw.get("platform", {})
    return RobotConfig(
        name=platform.get("name", "Simulation Humanoid"),
        platform_type=platform.get("type", "simulation"),
        num_joints=robot_data.get("num_joints", len(joints)),
        joints=joints,
        sensors=robot_data.get("sensors", []),
        mass_kg=robot_data.get("mass_kg", 75.0),
        height_m=robot_data.get("height_m", 1.75),
        max_payload_kg=robot_data.get("max_payload_kg", 15.0),
        control_frequency_hz=robot_data.get("control_frequency_hz", 1000.0),
        dof_arms=robot_data.get("dof_arms", 7),
        dof_legs=robot_data.get("dof_legs", 6),
        dof_hands=robot_data.get("dof_hands", 5),
        dof_head=robot_data.get("dof_head", 3),
        has_mobile_base=robot_data.get("has_mobile_base", False),
        has_bipedal_locomotion=robot_data.get("has_bipedal_locomotion", True),
    )
