"""
Layer 1: Hardware Abstraction Layer - Abstract Robot Interface

Defines the universal contract that all robot platforms must implement.
This allows BrianMind to control any robot through a unified API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class ControlMode(Enum):
    """Motor control mode."""
    POSITION = auto()
    VELOCITY = auto()
    TORQUE = auto()
    IMPEDANCE = auto()


class SensorType(Enum):
    """Types of sensors available on a robot."""
    RGB_CAMERA = auto()
    DEPTH_CAMERA = auto()
    STEREO_CAMERA = auto()
    LIDAR = auto()
    IMU = auto()
    FORCE_TORQUE = auto()
    JOINT_ENCODER = auto()
    TACTILE = auto()
    MICROPHONE = auto()
    GPS = auto()
    TEMPERATURE = auto()
    BATTERY = auto()


class ActuatorType(Enum):
    """Types of actuators on a robot."""
    SERVO = auto()
    HYDRAULIC = auto()
    PNEUMATIC = auto()
    LINEAR = auto()
    GRIPPER = auto()
    SPEAKER = auto()
    LED_DISPLAY = auto()


@dataclass
class JointConfig:
    """Configuration for a single robot joint."""
    joint_id: int
    name: str
    joint_type: str  # 'revolute', 'prismatic', 'continuous', 'fixed'
    position_min: float  # radians or meters
    position_max: float
    velocity_max: float  # rad/s or m/s
    torque_max: float  # Nm or N
    default_position: float = 0.0
    stiffness_default: float = 100.0
    damping_default: float = 10.0


@dataclass
class JointState:
    """Current state of a single joint."""
    joint_id: int
    position: float
    velocity: float
    torque: float
    temperature: float = 0.0


@dataclass
class JointCommand:
    """Command for a single joint."""
    joint_id: int
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    stiffness: float = 100.0
    damping: float = 10.0
    mode: ControlMode = ControlMode.POSITION


@dataclass
class SensorReading:
    """Generic sensor reading with metadata."""
    sensor_type: SensorType
    sensor_id: str
    timestamp_ns: int
    data: np.ndarray
    metadata: Dict = field(default_factory=dict)


@dataclass
class RobotPose:
    """6-DOF robot base pose in world frame."""
    position: np.ndarray  # [x, y, z] meters
    orientation: np.ndarray  # [qw, qx, qy, qz] quaternion
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class RobotConfig:
    """Full robot configuration."""
    name: str
    platform_type: str
    num_joints: int
    joints: List[JointConfig] = field(default_factory=list)
    sensors: List[Dict] = field(default_factory=list)
    actuators: List[Dict] = field(default_factory=list)
    mass_kg: float = 0.0
    height_m: float = 0.0
    max_payload_kg: float = 0.0
    control_frequency_hz: float = 1000.0
    dof_arms: int = 0
    dof_legs: int = 0
    dof_hands: int = 0
    dof_head: int = 0
    has_mobile_base: bool = False
    has_bipedal_locomotion: bool = False


@dataclass
class RobotStatus:
    """Overall robot health and status."""
    is_connected: bool = False
    is_powered: bool = False
    is_homed: bool = False
    is_in_error: bool = False
    battery_percent: float = 100.0
    uptime_seconds: float = 0.0
    error_codes: List[int] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    cpu_temperature: float = 0.0
    motor_temperatures: Dict[int, float] = field(default_factory=dict)


class RobotInterface(ABC):
    """
    Abstract base class for all robot platform drivers.

    Every robot platform (Optimus, Figure 02, simulation, etc.) must
    implement this interface to be controlled by BrianMind.

    The interface is designed for remote operation: BrianMind runs in the
    cloud and sends commands through BrianProtocol. The HAL driver runs
    on the robot's onboard computer.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self._is_initialized = False
        self._emergency_stop_active = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the robot hardware. Returns True on success."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Safely shutdown the robot, moving to rest pose first."""
        ...

    @abstractmethod
    async def home(self) -> bool:
        """Move robot to home/calibration pose."""
        ...

    @abstractmethod
    async def send_joint_commands(self, commands: List[JointCommand]) -> bool:
        """Send joint commands. Safety limits enforced by HAL before hardware."""
        ...

    @abstractmethod
    async def get_joint_states(self) -> List[JointState]:
        """Read current joint states from all joints."""
        ...

    @abstractmethod
    async def get_sensor_reading(self, sensor_type: SensorType,
                                  sensor_id: str = "") -> Optional[SensorReading]:
        """Read from a specific sensor."""
        ...

    @abstractmethod
    async def get_all_sensor_readings(self) -> List[SensorReading]:
        """Read all available sensors at once."""
        ...

    @abstractmethod
    async def get_status(self) -> RobotStatus:
        """Get overall robot status."""
        ...

    @abstractmethod
    async def emergency_stop(self) -> None:
        """Immediately stop all motion (Zone 1 software E-stop)."""
        ...

    @abstractmethod
    async def clear_emergency_stop(self) -> bool:
        """Clear E-stop state. Requires manual acknowledgment."""
        ...

    @abstractmethod
    async def get_robot_pose(self) -> RobotPose:
        """Get robot base pose in world frame."""
        ...

    @abstractmethod
    def get_config(self) -> RobotConfig:
        """Return the robot's configuration."""
        ...

    @abstractmethod
    def get_supported_sensors(self) -> List[SensorType]:
        """Return available sensor types on this platform."""
        ...

    @abstractmethod
    def get_joint_configs(self) -> List[JointConfig]:
        """Return configuration for all joints."""
        ...

    async def set_led_color(self, r: int, g: int, b: int) -> None:
        """Set status LED color (if available)."""
        pass

    async def play_audio(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """Play audio through robot speaker (if available)."""
        pass

    async def set_display(self, image_data: bytes) -> None:
        """Set display image/expression (if available)."""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def emergency_stop_active(self) -> bool:
        return self._emergency_stop_active

    def validate_joint_commands(self, commands: List[JointCommand]) -> Tuple[bool, List[str]]:
        """Validate joint commands against configured limits."""
        violations = []
        for cmd in commands:
            joint_cfg = next(
                (j for j in self.config.joints if j.joint_id == cmd.joint_id), None
            )
            if joint_cfg is None:
                violations.append(f"Unknown joint_id: {cmd.joint_id}")
                continue

            if cmd.mode == ControlMode.POSITION:
                if not (joint_cfg.position_min <= cmd.position <= joint_cfg.position_max):
                    violations.append(
                        f"Joint {cmd.joint_id} ({joint_cfg.name}): position {cmd.position:.3f} "
                        f"outside [{joint_cfg.position_min:.3f}, {joint_cfg.position_max:.3f}]"
                    )
            if abs(cmd.velocity) > joint_cfg.velocity_max:
                violations.append(
                    f"Joint {cmd.joint_id} ({joint_cfg.name}): velocity {abs(cmd.velocity):.3f} "
                    f"exceeds max {joint_cfg.velocity_max:.3f}"
                )
            if abs(cmd.torque) > joint_cfg.torque_max:
                violations.append(
                    f"Joint {cmd.joint_id} ({joint_cfg.name}): torque {abs(cmd.torque):.3f} "
                    f"exceeds max {joint_cfg.torque_max:.3f}"
                )
        return (len(violations) == 0, violations)
