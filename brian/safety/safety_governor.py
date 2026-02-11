"""
Layer 2: Safety & Security - Safety Governor

Master coordinator for the 5-zone safety architecture.
ISO 10218 / ISO/TS 15066 inspired collaborative robot safety.

Zone 1: Hardware Emergency (< 1ms, on-robot physical)
Zone 2: Real-time Limits (< 5ms, on-robot RTOS)
Zone 3: Behavioral Constraints (< 50ms, on-robot computer)
Zone 4: Ethical Validation (< 200ms, cloud brain)
Zone 5: Remote Oversight (< 1s, cloud + human operator)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    NOMINAL = auto()
    CAUTION = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class ZoneStatus(Enum):
    OK = auto()
    TRIGGERED = auto()
    OVERRIDE = auto()
    DISABLED = auto()
    ERROR = auto()


@dataclass
class SafetyViolation:
    zone: int
    severity: SafetyLevel
    description: str
    timestamp: float
    joint_id: Optional[int] = None
    measured_value: Optional[float] = None
    limit_value: Optional[float] = None
    action_taken: str = ""


@dataclass
class SafetyLimits:
    joint_position_limits: Dict[int, tuple] = field(default_factory=dict)
    joint_velocity_limits: Dict[int, float] = field(default_factory=dict)
    joint_torque_limits: Dict[int, float] = field(default_factory=dict)
    max_cartesian_velocity: float = 1.5
    max_cartesian_acceleration: float = 5.0
    max_contact_force: float = 150.0
    max_contact_pressure: float = 2.0
    min_human_distance: float = 0.5
    speed_reduction_distance: float = 1.5
    emergency_stop_distance: float = 0.2
    heartbeat_timeout_ms: float = 200.0
    max_command_age_ms: float = 500.0
    workspace_bounds_min: List[float] = field(default_factory=lambda: [-5.0, -5.0, 0.0])
    workspace_bounds_max: List[float] = field(default_factory=lambda: [5.0, 5.0, 3.0])
    no_go_zones: List[Dict] = field(default_factory=list)


@dataclass
class SafetyState:
    overall_level: SafetyLevel = SafetyLevel.NOMINAL
    zone_statuses: Dict[int, ZoneStatus] = field(default_factory=lambda: {
        1: ZoneStatus.OK, 2: ZoneStatus.OK, 3: ZoneStatus.OK,
        4: ZoneStatus.OK, 5: ZoneStatus.OK,
    })
    violations: List[SafetyViolation] = field(default_factory=list)
    last_check_timestamp: float = 0.0
    emergency_stop_active: bool = False
    speed_reduction_factor: float = 1.0
    operator_override_active: bool = False


class SafetyZone:
    def __init__(self, zone_number: int, name: str, limits: SafetyLimits):
        self.zone_number = zone_number
        self.name = name
        self.limits = limits
        self.status = ZoneStatus.OK
        self.violations: List[SafetyViolation] = []

    def check(self, context: Dict[str, Any]) -> ZoneStatus:
        return ZoneStatus.OK

    def _add_violation(self, severity: SafetyLevel, description: str,
                       **kwargs) -> SafetyViolation:
        v = SafetyViolation(zone=self.zone_number, severity=severity,
                            description=description, timestamp=time.time(), **kwargs)
        self.violations.append(v)
        return v


class Zone2RealtimeSafety(SafetyZone):
    """Zone 2: Per-joint position, velocity, and torque limits at 1kHz."""

    def __init__(self, limits: SafetyLimits):
        super().__init__(2, "Real-time Limits", limits)

    def check(self, context: Dict[str, Any]) -> ZoneStatus:
        self.violations.clear()
        for js in context.get("joint_states", []):
            jid = js.get("joint_id", -1)
            if jid in self.limits.joint_position_limits:
                lo, hi = self.limits.joint_position_limits[jid]
                pos = js.get("position", 0.0)
                if pos < lo or pos > hi:
                    self._add_violation(SafetyLevel.CRITICAL,
                                        f"Joint {jid} pos {pos:.3f} outside [{lo:.3f},{hi:.3f}]",
                                        joint_id=jid, measured_value=pos,
                                        limit_value=hi if pos > hi else lo,
                                        action_taken="clamp_position")
            if jid in self.limits.joint_velocity_limits:
                vlim = self.limits.joint_velocity_limits[jid]
                vel = abs(js.get("velocity", 0.0))
                if vel > vlim:
                    self._add_violation(SafetyLevel.WARNING,
                                        f"Joint {jid} vel {vel:.3f} > {vlim:.3f}",
                                        joint_id=jid, measured_value=vel,
                                        limit_value=vlim, action_taken="reduce_velocity")
            if jid in self.limits.joint_torque_limits:
                tlim = self.limits.joint_torque_limits[jid]
                torq = abs(js.get("torque", 0.0))
                if torq > tlim:
                    self._add_violation(SafetyLevel.WARNING,
                                        f"Joint {jid} torque {torq:.3f} > {tlim:.3f}",
                                        joint_id=jid, measured_value=torq,
                                        limit_value=tlim, action_taken="limit_torque")
        self.status = ZoneStatus.TRIGGERED if self.violations else ZoneStatus.OK
        return self.status


class Zone3BehavioralSafety(SafetyZone):
    """Zone 3: Workspace bounds, no-go zones, human proximity."""

    def __init__(self, limits: SafetyLimits):
        super().__init__(3, "Behavioral Constraints", limits)

    def check(self, context: Dict[str, Any]) -> ZoneStatus:
        self.violations.clear()
        speed_factor = 1.0
        robot_pos = np.array(context.get("robot_position", [0, 0, 0])[:3])

        for i, (pos, lo, hi) in enumerate(zip(
            robot_pos, self.limits.workspace_bounds_min, self.limits.workspace_bounds_max
        )):
            if pos < lo or pos > hi:
                self._add_violation(SafetyLevel.WARNING,
                                    f"{'XYZ'[i]} pos {pos:.2f} outside [{lo:.2f},{hi:.2f}]",
                                    action_taken="halt_and_retreat")

        for zone in self.limits.no_go_zones:
            center = np.array(zone.get("center", [0, 0, 0]))
            radius = zone.get("radius", 1.0)
            dist = float(np.linalg.norm(robot_pos - center))
            if dist < radius:
                self._add_violation(SafetyLevel.CRITICAL,
                                    f"In no-go zone (dist={dist:.2f}<{radius})",
                                    action_taken="emergency_retreat")

        for person in context.get("detected_people", []):
            person_pos = np.array(person.get("position", [100, 100, 100]))
            dist = float(np.linalg.norm(robot_pos - person_pos))
            if dist < self.limits.emergency_stop_distance:
                self._add_violation(SafetyLevel.EMERGENCY,
                                    f"Human at {dist:.2f}m (E-stop: {self.limits.emergency_stop_distance}m)",
                                    measured_value=dist, action_taken="emergency_stop")
                speed_factor = 0.0
            elif dist < self.limits.min_human_distance:
                self._add_violation(SafetyLevel.CRITICAL,
                                    f"Human at {dist:.2f}m (min: {self.limits.min_human_distance}m)",
                                    measured_value=dist, action_taken="stop_motion")
                speed_factor = 0.0
            elif dist < self.limits.speed_reduction_distance:
                speed_factor = min(speed_factor, dist / self.limits.speed_reduction_distance)

        context["speed_reduction_factor"] = speed_factor
        self.status = ZoneStatus.TRIGGERED if self.violations else ZoneStatus.OK
        return self.status


class Zone4EthicalSafety(SafetyZone):
    """Zone 4: Ethical action validation (cloud-side, < 200ms)."""

    def __init__(self, limits: SafetyLimits):
        super().__init__(4, "Ethical Validation", limits)
        self.harm_threshold = 0.1

    def check(self, context: Dict[str, Any]) -> ZoneStatus:
        self.violations.clear()
        ethical_score = context.get("ethical_score", 1.0)
        if ethical_score < self.harm_threshold:
            self._add_violation(SafetyLevel.CRITICAL,
                                f"Ethical score {ethical_score:.2f} < {self.harm_threshold}",
                                measured_value=ethical_score, action_taken="veto_action")
        self.status = ZoneStatus.TRIGGERED if self.violations else ZoneStatus.OK
        return self.status


class Zone5RemoteOversight(SafetyZone):
    """Zone 5: Operator monitoring and remote kill switch."""

    def __init__(self, limits: SafetyLimits):
        super().__init__(5, "Remote Oversight", limits)
        self.operator_connected = False
        self.last_operator_heartbeat = 0.0
        self.kill_switch_active = False

    def check(self, context: Dict[str, Any]) -> ZoneStatus:
        self.violations.clear()
        if self.kill_switch_active:
            self._add_violation(SafetyLevel.EMERGENCY,
                                "Remote kill switch activated", action_taken="emergency_stop")
        if self.operator_connected and (time.time() - self.last_operator_heartbeat > 30.0):
            self._add_violation(SafetyLevel.CAUTION,
                                "Operator connection lost (>30s)", action_taken="reduce_autonomy")
        self.status = ZoneStatus.TRIGGERED if self.violations else ZoneStatus.OK
        return self.status

    def activate_kill_switch(self):
        self.kill_switch_active = True
        logger.critical("REMOTE KILL SWITCH ACTIVATED")

    def deactivate_kill_switch(self):
        self.kill_switch_active = False

    def operator_heartbeat(self):
        self.operator_connected = True
        self.last_operator_heartbeat = time.time()


class SafetyGovernor:
    """
    Master coordinator for the 5-zone safety architecture.
    Every motor command passes through all zones before reaching hardware.
    """

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.state = SafetyState()
        self.zone2 = Zone2RealtimeSafety(limits)
        self.zone3 = Zone3BehavioralSafety(limits)
        self.zone4 = Zone4EthicalSafety(limits)
        self.zone5 = Zone5RemoteOversight(limits)
        self._violation_callbacks: List[Callable] = []
        self._emergency_callbacks: List[Callable] = []
        logger.info("SafetyGovernor initialized (5-zone architecture)")

    def check_all_zones(self, context: Dict[str, Any]) -> SafetyState:
        self.state.violations.clear()
        self.state.last_check_timestamp = time.time()
        for zone_num, zone in [(2, self.zone2), (3, self.zone3),
                                (4, self.zone4), (5, self.zone5)]:
            status = zone.check(context)
            self.state.zone_statuses[zone_num] = status
            self.state.violations.extend(zone.violations)
        self.state.speed_reduction_factor = context.get("speed_reduction_factor", 1.0)
        self.state.overall_level = self._compute_overall_level()
        if self.state.overall_level == SafetyLevel.EMERGENCY:
            self.state.emergency_stop_active = True
            for cb in self._emergency_callbacks:
                cb()
        if self.state.violations:
            for cb in self._violation_callbacks:
                cb(self.state.violations)
        return self.state

    def validate_command(self, command: Any, context: Dict[str, Any]) -> tuple:
        context["motor_command"] = command
        state = self.check_all_zones(context)
        is_safe = state.overall_level in (SafetyLevel.NOMINAL, SafetyLevel.CAUTION)
        return (is_safe, command, state.violations)

    def on_violation(self, callback: Callable):
        self._violation_callbacks.append(callback)

    def on_emergency_stop(self, callback: Callable):
        self._emergency_callbacks.append(callback)

    def _compute_overall_level(self) -> SafetyLevel:
        order = list(SafetyLevel)
        worst = SafetyLevel.NOMINAL
        for v in self.state.violations:
            if order.index(v.severity) > order.index(worst):
                worst = v.severity
        return worst

    def get_status_report(self) -> Dict[str, Any]:
        return {
            "overall_level": self.state.overall_level.name,
            "emergency_stop": self.state.emergency_stop_active,
            "speed_factor": self.state.speed_reduction_factor,
            "zones": {k: v.name for k, v in self.state.zone_statuses.items()},
            "violations": [{
                "zone": v.zone, "severity": v.severity.name,
                "description": v.description, "action": v.action_taken
            } for v in self.state.violations],
        }
