"""
Layer 4: Sensorimotor - Motion Controller

Implements trajectory planning, inverse kinematics, gait generation,
and real-time motion control for humanoid robots (Optimus, Figure, etc.).

Integrates with BrianMind's MotorCommand output and the HAL's JointCommand
interface. Safety limits are enforced at the HAL layer; this module focuses
on smooth, natural motion generation.

Architecture:
    BrianMind.decide() -> MotorCommand
        -> MotionController.execute(command) -> List[JointCommand]
            -> RobotInterface.send_joint_commands(commands)

Supports:
    - Cartesian end-effector control via analytical/numerical IK
    - Joint-space trajectory interpolation (cubic/quintic spline)
    - Bipedal walking gait generation (ZMP-based)
    - Whole-body motion coordination
    - Impedance control for compliant manipulation
    - Real-time trajectory modification for obstacle avoidance
"""

import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from brian.hal.robot_interface import (
    ControlMode, JointCommand, JointConfig, JointState, RobotConfig, RobotPose,
)
from brian.core.brian_mind import MotorCommand, WorldState

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class MotionPhase(Enum):
    """Phases of a motion execution."""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    CONVERGING = auto()
    COMPLETED = auto()
    ABORTED = auto()


class GaitPhase(Enum):
    """Bipedal walking gait phases."""
    DOUBLE_SUPPORT = auto()
    RIGHT_SWING = auto()
    LEFT_SWING = auto()
    STANDING = auto()


@dataclass
class CartesianTarget:
    """Target pose in Cartesian space for end-effector control."""
    position: np.ndarray          # [x, y, z] in meters
    orientation: np.ndarray       # [qw, qx, qy, qz] quaternion
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_frame: str = "base"  # "base" or "world"
    max_linear_speed: float = 0.5   # m/s
    max_angular_speed: float = 1.0  # rad/s


@dataclass
class TrajectoryPoint:
    """A single point along a joint-space trajectory."""
    time_from_start: float                  # seconds
    positions: np.ndarray                   # joint positions (rad or m)
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None


@dataclass
class Trajectory:
    """A complete joint-space trajectory."""
    joint_ids: List[int]
    points: List[TrajectoryPoint]
    duration: float = 0.0
    phase: MotionPhase = MotionPhase.IDLE
    start_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GaitParameters:
    """Parameters for bipedal walking gait generation."""
    step_length: float = 0.15       # meters per step
    step_height: float = 0.05       # foot lift height (meters)
    step_width: float = 0.20        # lateral distance between feet
    step_duration: float = 0.8      # seconds per step
    double_support_ratio: float = 0.2  # fraction of step in double support
    com_height: float = 0.85        # center of mass height (meters)
    swing_foot_clearance: float = 0.03  # minimum clearance (meters)
    max_step_length: float = 0.30
    max_step_width: float = 0.35
    max_angular_step: float = 0.3   # rad per step (turning)


@dataclass
class FootState:
    """State of a single foot during locomotion."""
    position: np.ndarray       # [x, y, z] in world frame
    orientation: np.ndarray    # [qw, qx, qy, qz]
    is_stance: bool = True
    contact_force: float = 0.0


@dataclass
class ImpedanceParams:
    """Impedance control parameters for compliant behavior."""
    stiffness: np.ndarray      # 6-DOF: [kx, ky, kz, krx, kry, krz]
    damping: np.ndarray        # 6-DOF
    inertia: Optional[np.ndarray] = None  # virtual inertia (optional)
    equilibrium_position: Optional[np.ndarray] = None
    equilibrium_orientation: Optional[np.ndarray] = None
    force_limit: float = 50.0  # N, max interaction force


# ============================================================================
# Forward Kinematics
# ============================================================================

class ForwardKinematics:
    """
    Compute end-effector pose from joint angles using DH parameters.
    Supports arbitrary serial kinematic chains.
    """

    def __init__(self, dh_params: List[Dict[str, float]]):
        """
        Args:
            dh_params: List of DH parameters per joint.
                Each dict: {a, d, alpha, theta_offset}
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def compute(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the 4x4 homogeneous transform from base to end-effector.

        Args:
            joint_angles: Array of joint angles (rad).

        Returns:
            4x4 homogeneous transformation matrix.
        """
        T = np.eye(4)
        for i, (dh, q) in enumerate(zip(self.dh_params, joint_angles)):
            theta = q + dh.get("theta_offset", 0.0)
            d = dh.get("d", 0.0)
            a = dh.get("a", 0.0)
            alpha = dh.get("alpha", 0.0)

            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)

            Ti = np.array([
                [ct,  -st * ca,  st * sa,  a * ct],
                [st,   ct * ca, -ct * sa,  a * st],
                [0.0,  sa,       ca,       d     ],
                [0.0,  0.0,      0.0,      1.0   ],
            ])
            T = T @ Ti
        return T

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the 6xN geometric Jacobian at the given configuration.
        Uses numerical differentiation for robustness across all DH configs.

        Returns:
            6xN Jacobian matrix (linear velocity rows 0-2, angular rows 3-5).
        """
        n = self.num_joints
        J = np.zeros((6, n))
        eps = 1e-6
        T_current = self.compute(joint_angles)
        p_current = T_current[:3, 3]

        for i in range(n):
            q_perturbed = joint_angles.copy()
            q_perturbed[i] += eps
            T_perturbed = self.compute(q_perturbed)
            p_perturbed = T_perturbed[:3, 3]

            # Linear velocity component
            J[:3, i] = (p_perturbed - p_current) / eps

            # Angular velocity component (from rotation difference)
            R_diff = T_perturbed[:3, :3] @ T_current[:3, :3].T
            # Extract angular velocity from rotation matrix difference
            J[3, i] = (R_diff[2, 1] - R_diff[1, 2]) / (2.0 * eps)
            J[4, i] = (R_diff[0, 2] - R_diff[2, 0]) / (2.0 * eps)
            J[5, i] = (R_diff[1, 0] - R_diff[0, 1]) / (2.0 * eps)

        return J


# ============================================================================
# Inverse Kinematics Solver
# ============================================================================

class InverseKinematicsSolver:
    """
    Numerical IK solver using damped least squares (Levenberg-Marquardt).
    Handles joint limits, singularity avoidance, and null-space optimization.
    """

    def __init__(self, fk: ForwardKinematics, joint_configs: List[JointConfig],
                 joint_ids: List[int]):
        self.fk = fk
        self.joint_configs = joint_configs
        self.joint_ids = joint_ids
        self.n_joints = len(joint_ids)

        # Build limit arrays
        self._q_min = np.array([jc.position_min for jc in joint_configs])
        self._q_max = np.array([jc.position_max for jc in joint_configs])
        self._q_center = (self._q_min + self._q_max) / 2.0
        self._q_range = self._q_max - self._q_min

        # Solver parameters
        self.max_iterations = 100
        self.position_tolerance = 1e-4     # meters
        self.orientation_tolerance = 1e-3  # radians
        self.damping_factor = 0.01
        self.step_size = 0.5
        self.null_space_gain = 0.1

    def solve(self, target: CartesianTarget,
              q_initial: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """
        Solve IK for the given Cartesian target.

        Args:
            target: Desired end-effector pose.
            q_initial: Initial joint configuration (uses center if None).

        Returns:
            (success, joint_angles) tuple.
        """
        q = q_initial.copy() if q_initial is not None else self._q_center.copy()
        q = np.clip(q, self._q_min, self._q_max)

        target_pos = target.position
        target_rot = Rotation.from_quat(
            [target.orientation[1], target.orientation[2],
             target.orientation[3], target.orientation[0]]  # xyzw format for scipy
        ).as_matrix()

        for iteration in range(self.max_iterations):
            T_current = self.fk.compute(q)
            current_pos = T_current[:3, 3]
            current_rot = T_current[:3, :3]

            # Position error
            pos_error = target_pos - current_pos

            # Orientation error (axis-angle from rotation difference)
            R_error = target_rot @ current_rot.T
            trace_val = np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
            angle = math.acos(trace_val)
            if abs(angle) < 1e-10:
                ori_error = np.zeros(3)
            else:
                axis = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1],
                ]) / (2.0 * math.sin(angle))
                ori_error = angle * axis

            # Check convergence
            pos_err_norm = np.linalg.norm(pos_error)
            ori_err_norm = np.linalg.norm(ori_error)
            if pos_err_norm < self.position_tolerance and ori_err_norm < self.orientation_tolerance:
                logger.debug(f"IK converged in {iteration + 1} iterations "
                             f"(pos_err={pos_err_norm:.6f}, ori_err={ori_err_norm:.6f})")
                return True, q

            # Stack into 6D error vector
            error = np.concatenate([pos_error, ori_error])

            # Compute Jacobian
            J = self.fk.jacobian(q)

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
            JJT = J @ J.T
            damping = self.damping_factor * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(JJT + damping, error)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.lstsq(JJT + damping, error, rcond=None)[0]

            # Null-space optimization: push toward joint centers
            null_projector = np.eye(self.n_joints) - np.linalg.pinv(J) @ J
            q_null = self.null_space_gain * (self._q_center - q)
            dq += null_projector @ q_null

            # Apply step with limits
            q += self.step_size * dq
            q = np.clip(q, self._q_min, self._q_max)

        logger.warning(f"IK failed to converge after {self.max_iterations} iterations "
                       f"(pos_err={pos_err_norm:.6f}, ori_err={ori_err_norm:.6f})")
        return False, q


# ============================================================================
# Trajectory Planner
# ============================================================================

class TrajectoryPlanner:
    """
    Plans smooth joint-space trajectories between configurations.
    Supports cubic spline, quintic polynomial, and minimum-jerk profiles.
    """

    def __init__(self, joint_configs: List[JointConfig], dt: float = 0.001):
        """
        Args:
            joint_configs: Joint configurations with limits.
            dt: Control loop timestep (seconds).
        """
        self.joint_configs = joint_configs
        self.dt = dt
        self.n_joints = len(joint_configs)
        self._vel_limits = np.array([jc.velocity_max for jc in joint_configs])
        self._acc_limits = self._vel_limits * 5.0  # default accel = 5x velocity

    def plan_point_to_point(self, q_start: np.ndarray, q_goal: np.ndarray,
                            duration: Optional[float] = None,
                            v_start: Optional[np.ndarray] = None,
                            v_goal: Optional[np.ndarray] = None) -> Trajectory:
        """
        Plan a minimum-jerk trajectory from start to goal configuration.

        If duration is None, it's computed from joint velocities to ensure
        all joints stay within limits.
        """
        n = self.n_joints
        v_start = v_start if v_start is not None else np.zeros(n)
        v_goal = v_goal if v_goal is not None else np.zeros(n)

        # Auto-compute duration from velocity limits
        if duration is None:
            dq = np.abs(q_goal - q_start)
            time_per_joint = np.where(
                dq > 1e-6,
                dq / (self._vel_limits * 0.8),  # 80% of max for smoothness
                0.0,
            )
            duration = max(float(np.max(time_per_joint)), 0.5)

        # Generate minimum-jerk profile
        num_points = max(int(duration / self.dt), 2)
        joint_ids = [jc.joint_id for jc in self.joint_configs]
        points = []

        for k in range(num_points + 1):
            t = k * self.dt
            tau = t / duration if duration > 0 else 1.0
            tau = min(tau, 1.0)

            # Minimum jerk polynomial: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
            dds = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (duration**2)

            positions = q_start + s * (q_goal - q_start)
            velocities = ds * (q_goal - q_start) + (1.0 - s) * v_start + s * v_goal
            accelerations = dds * (q_goal - q_start)

            points.append(TrajectoryPoint(
                time_from_start=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
            ))

        traj = Trajectory(
            joint_ids=joint_ids,
            points=points,
            duration=duration,
            metadata={"type": "minimum_jerk", "num_points": len(points)},
        )
        return traj

    def plan_via_waypoints(self, waypoints: List[np.ndarray],
                           durations: Optional[List[float]] = None) -> Trajectory:
        """
        Plan a smooth trajectory through multiple waypoints using cubic splines.

        Args:
            waypoints: List of joint configurations to pass through.
            durations: Time between consecutive waypoints (auto if None).
        """
        n_waypoints = len(waypoints)
        if n_waypoints < 2:
            raise ValueError("Need at least 2 waypoints")

        # Compute cumulative times
        if durations is None:
            durations = []
            for i in range(n_waypoints - 1):
                dq = np.abs(waypoints[i + 1] - waypoints[i])
                dt = max(float(np.max(dq / (self._vel_limits * 0.7))), 0.3)
                durations.append(dt)

        times = [0.0]
        for d in durations:
            times.append(times[-1] + d)
        total_duration = times[-1]

        # Stack waypoints into array: (n_waypoints, n_joints)
        wp_array = np.array(waypoints)

        # Fit cubic spline per joint
        joint_ids = [jc.joint_id for jc in self.joint_configs]
        num_points = max(int(total_duration / self.dt), 2)
        t_interp = np.linspace(0, total_duration, num_points + 1)

        all_positions = np.zeros((num_points + 1, self.n_joints))
        all_velocities = np.zeros((num_points + 1, self.n_joints))
        all_accelerations = np.zeros((num_points + 1, self.n_joints))

        for j in range(self.n_joints):
            cs = CubicSpline(times, wp_array[:, j], bc_type='clamped')
            all_positions[:, j] = cs(t_interp)
            all_velocities[:, j] = cs(t_interp, 1)
            all_accelerations[:, j] = cs(t_interp, 2)

        points = []
        for k in range(num_points + 1):
            points.append(TrajectoryPoint(
                time_from_start=float(t_interp[k]),
                positions=all_positions[k],
                velocities=all_velocities[k],
                accelerations=all_accelerations[k],
            ))

        return Trajectory(
            joint_ids=joint_ids,
            points=points,
            duration=total_duration,
            metadata={"type": "cubic_spline", "n_waypoints": n_waypoints},
        )

    def time_optimal_scaling(self, traj: Trajectory, safety_margin: float = 0.9) -> Trajectory:
        """
        Retime a trajectory to be as fast as possible while respecting
        velocity and acceleration limits.
        """
        if not traj.points:
            return traj

        # Find the maximum velocity and acceleration ratios
        max_vel_ratio = 0.0
        max_acc_ratio = 0.0
        for pt in traj.points:
            if pt.velocities is not None:
                ratios = np.abs(pt.velocities) / (self._vel_limits + 1e-10)
                max_vel_ratio = max(max_vel_ratio, float(np.max(ratios)))
            if pt.accelerations is not None:
                ratios = np.abs(pt.accelerations) / (self._acc_limits + 1e-10)
                max_acc_ratio = max(max_acc_ratio, float(np.max(ratios)))

        # Scale factor: slowest constraint wins
        scale = max(max_vel_ratio, math.sqrt(max_acc_ratio), 1e-6) / safety_margin

        if scale <= 1.0:
            return traj  # Already within limits

        new_duration = traj.duration * scale
        new_points = []
        for pt in traj.points:
            new_points.append(TrajectoryPoint(
                time_from_start=pt.time_from_start * scale,
                positions=pt.positions.copy(),
                velocities=pt.velocities / scale if pt.velocities is not None else None,
                accelerations=pt.accelerations / (scale**2) if pt.accelerations is not None else None,
            ))

        return Trajectory(
            joint_ids=traj.joint_ids,
            points=new_points,
            duration=new_duration,
            metadata={**traj.metadata, "retimed": True, "scale_factor": scale},
        )


# ============================================================================
# Bipedal Gait Generator
# ============================================================================

class GaitGenerator:
    """
    Generates stable bipedal walking gaits using simplified ZMP
    (Zero Moment Point) planning for humanoid robots like Optimus and Figure.

    Produces foot placement trajectories and CoM (Center of Mass) trajectories
    that maintain dynamic balance during locomotion.
    """

    def __init__(self, params: GaitParameters, robot_config: RobotConfig):
        self.params = params
        self.robot_config = robot_config
        self.phase = GaitPhase.STANDING
        self._step_count = 0
        self._phase_timer = 0.0

        # Foot states initialized at standing position
        self.right_foot = FootState(
            position=np.array([0.0, -params.step_width / 2, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            is_stance=True,
        )
        self.left_foot = FootState(
            position=np.array([0.0, params.step_width / 2, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            is_stance=True,
        )
        self.com_position = np.array([0.0, 0.0, params.com_height])
        self.com_velocity = np.zeros(3)

    def plan_footsteps(self, target_velocity: np.ndarray,
                       num_steps: int = 6) -> List[Dict[str, Any]]:
        """
        Plan a sequence of footsteps to achieve the target walking velocity.

        Args:
            target_velocity: [vx, vy, v_theta] desired velocity.
            num_steps: Number of steps to plan.

        Returns:
            List of footstep dicts with position, orientation, and timing.
        """
        vx = np.clip(target_velocity[0], -self.params.max_step_length / self.params.step_duration,
                      self.params.max_step_length / self.params.step_duration)
        vy = np.clip(target_velocity[1], -0.1, 0.1)
        v_theta = np.clip(target_velocity[2] if len(target_velocity) > 2 else 0.0,
                          -self.params.max_angular_step / self.params.step_duration,
                          self.params.max_angular_step / self.params.step_duration)

        step_length = vx * self.params.step_duration
        step_lateral = vy * self.params.step_duration
        step_yaw = v_theta * self.params.step_duration

        footsteps = []
        current_x = 0.0
        current_y = 0.0
        current_yaw = 0.0

        for i in range(num_steps):
            is_right = (i % 2 == 0)
            current_x += step_length
            lateral_offset = -self.params.step_width / 2 if is_right else self.params.step_width / 2
            current_y = lateral_offset + step_lateral * (1 if is_right else -1)
            current_yaw += step_yaw

            footsteps.append({
                "step_index": i,
                "foot": "right" if is_right else "left",
                "position": np.array([current_x, current_y, 0.0]),
                "yaw": current_yaw,
                "time_start": i * self.params.step_duration,
                "time_end": (i + 1) * self.params.step_duration,
                "step_height": self.params.step_height,
            })

        return footsteps

    def generate_com_trajectory(self, footsteps: List[Dict[str, Any]],
                                dt: float = 0.001) -> List[np.ndarray]:
        """
        Generate Center of Mass trajectory using linear inverted pendulum model.

        The CoM moves smoothly between support foot positions to maintain
        the ZMP within the support polygon.
        """
        if not footsteps:
            return [self.com_position.copy()]

        total_time = footsteps[-1]["time_end"]
        num_points = int(total_time / dt)
        com_trajectory = []

        g = 9.81
        zc = self.params.com_height
        omega = math.sqrt(g / zc)  # natural frequency of inverted pendulum

        for k in range(num_points + 1):
            t = k * dt
            # Find current and next support foot
            current_support = self._get_support_foot(footsteps, t)
            next_support = self._get_next_support(footsteps, t)

            # Interpolate CoM between support positions
            if current_support is not None and next_support is not None:
                t_in_step = t - current_support["time_start"]
                step_duration = current_support["time_end"] - current_support["time_start"]
                tau = min(t_in_step / step_duration, 1.0) if step_duration > 0 else 1.0

                # Smooth interpolation using hyperbolic functions (LIPM solution)
                support_pos = current_support["position"][:2]
                next_pos = next_support["position"][:2]

                # Simplified LIPM: x(t) = x_support + A*cosh(omega*t) + B*sinh(omega*t)
                x_com = support_pos + (next_pos - support_pos) * (
                    math.cosh(omega * t_in_step) - 1.0
                ) / (math.cosh(omega * step_duration) - 1.0 + 1e-10)

                com_point = np.array([x_com[0], x_com[1], zc])
            else:
                com_point = np.array([0.0, 0.0, zc])

            com_trajectory.append(com_point)

        return com_trajectory

    def generate_swing_foot_trajectory(self, start_pos: np.ndarray,
                                        end_pos: np.ndarray,
                                        step_height: float,
                                        duration: float,
                                        dt: float = 0.001) -> List[np.ndarray]:
        """
        Generate a smooth swing foot trajectory with clearance.
        Uses a cycloid-inspired profile for natural foot motion.
        """
        num_points = int(duration / dt)
        trajectory = []

        for k in range(num_points + 1):
            t = k * dt
            tau = min(t / duration, 1.0) if duration > 0 else 1.0

            # Horizontal: minimum jerk profile
            s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            xy = start_pos[:2] + s * (end_pos[:2] - start_pos[:2])

            # Vertical: parabolic arc with maximum at midpoint
            z = 4.0 * step_height * tau * (1.0 - tau)
            z += start_pos[2] + s * (end_pos[2] - start_pos[2])

            trajectory.append(np.array([xy[0], xy[1], z]))

        return trajectory

    def _get_support_foot(self, footsteps: List[Dict], t: float) -> Optional[Dict]:
        for step in footsteps:
            if step["time_start"] <= t < step["time_end"]:
                return step
        return footsteps[-1] if footsteps else None

    def _get_next_support(self, footsteps: List[Dict], t: float) -> Optional[Dict]:
        for i, step in enumerate(footsteps):
            if step["time_start"] <= t < step["time_end"] and i + 1 < len(footsteps):
                return footsteps[i + 1]
        return footsteps[-1] if footsteps else None

    def compute_leg_ik(self, foot_target: np.ndarray, hip_position: np.ndarray,
                       leg_lengths: Tuple[float, float] = (0.42, 0.40)) -> Optional[np.ndarray]:
        """
        Analytical IK for a 6-DOF leg (hip pitch/roll/yaw, knee, ankle pitch/roll).
        Uses geometric solution for the hip-knee-ankle plane.

        Args:
            foot_target: [x, y, z] foot position relative to hip.
            hip_position: [x, y, z] hip position in body frame.
            leg_lengths: (upper_leg, lower_leg) in meters.

        Returns:
            6-element array [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
            or None if unreachable.
        """
        L1, L2 = leg_lengths
        target = foot_target - hip_position

        # Distance from hip to foot
        d = np.linalg.norm(target)
        if d > (L1 + L2) * 0.99 or d < abs(L1 - L2) * 1.01:
            return None  # Unreachable

        # Knee angle (law of cosines)
        cos_knee = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_angle = math.pi - math.acos(cos_knee)

        # Hip pitch and roll from target direction
        hip_roll = math.atan2(target[1], -target[2])
        d_sagittal = math.sqrt(target[0]**2 + target[2]**2)

        # Angle from vertical to foot in sagittal plane
        alpha = math.atan2(target[0], -target[2])
        # Angle of upper leg relative to vertical
        cos_beta = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = math.acos(cos_beta)
        hip_pitch = alpha - beta

        # Ankle compensates to keep foot flat
        ankle_pitch = -(hip_pitch + knee_angle)
        ankle_roll = -hip_roll
        hip_yaw = 0.0  # No yaw for straight walking

        return np.array([hip_pitch, hip_roll, hip_yaw, knee_angle, ankle_pitch, ankle_roll])


# ============================================================================
# Impedance Controller
# ============================================================================

class ImpedanceController:
    """
    Cartesian impedance controller for compliant manipulation.
    Makes the robot behave like a mass-spring-damper system in task space,
    allowing safe interaction with objects and humans.
    """

    def __init__(self, fk: ForwardKinematics, joint_configs: List[JointConfig]):
        self.fk = fk
        self.joint_configs = joint_configs
        self.n_joints = len(joint_configs)

    def compute_torques(self, q: np.ndarray, dq: np.ndarray,
                        params: ImpedanceParams) -> np.ndarray:
        """
        Compute joint torques that realize the desired impedance behavior.

        Args:
            q: Current joint positions.
            dq: Current joint velocities.
            params: Impedance parameters (stiffness, damping, equilibrium).

        Returns:
            Joint torques (Nm).
        """
        T_current = self.fk.compute(q)
        current_pos = T_current[:3, 3]
        current_rot = T_current[:3, :3]

        eq_pos = params.equilibrium_position if params.equilibrium_position is not None else current_pos
        eq_rot = Rotation.from_quat(
            [params.equilibrium_orientation[1], params.equilibrium_orientation[2],
             params.equilibrium_orientation[3], params.equilibrium_orientation[0]]
        ).as_matrix() if params.equilibrium_orientation is not None else current_rot

        # Position error
        pos_error = eq_pos - current_pos

        # Orientation error (axis-angle)
        R_error = eq_rot @ current_rot.T
        trace_val = np.clip((np.trace(R_error) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(trace_val)
        if abs(angle) < 1e-10:
            ori_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1],
            ]) / (2.0 * math.sin(angle))
            ori_error = angle * axis

        # 6D error
        x_error = np.concatenate([pos_error, ori_error])

        # Jacobian
        J = self.fk.jacobian(q)

        # Cartesian velocity
        dx = J @ dq

        # Wrench = K * x_error - D * dx
        K = np.diag(params.stiffness)
        D = np.diag(params.damping)
        wrench = K @ x_error - D @ dx

        # Enforce force limit
        force_magnitude = np.linalg.norm(wrench[:3])
        if force_magnitude > params.force_limit:
            wrench[:3] *= params.force_limit / force_magnitude

        # Map to joint torques: tau = J^T * wrench
        torques = J.T @ wrench
        return torques


# ============================================================================
# Main Motion Controller
# ============================================================================

class MotionController:
    """
    Main motion controller that orchestrates all motion subsystems.

    Converts BrianMind's MotorCommands into executable joint trajectories
    using IK, trajectory planning, gait generation, and impedance control.
    """

    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.dt = 1.0 / robot_config.control_frequency_hz

        # Build kinematic chains for each arm
        # Default DH parameters for a 7-DOF arm (generic humanoid)
        self._arm_dh_right = self._build_arm_dh(side="right")
        self._arm_dh_left = self._build_arm_dh(side="left")

        # FK engines
        self.fk_right_arm = ForwardKinematics(self._arm_dh_right)
        self.fk_left_arm = ForwardKinematics(self._arm_dh_left)

        # Get joint configs for each kinematic group
        joint_map = {jc.joint_id: jc for jc in
                     (JointConfig(**j) if isinstance(j, dict) else j
                      for j in robot_config.joints)}

        right_arm_ids = list(range(0, 7))
        left_arm_ids = list(range(7, 14))
        right_leg_ids = list(range(17, 23))
        left_leg_ids = list(range(23, 29))

        self.right_arm_configs = [joint_map[i] for i in right_arm_ids if i in joint_map]
        self.left_arm_configs = [joint_map[i] for i in left_arm_ids if i in joint_map]
        self.right_leg_configs = [joint_map[i] for i in right_leg_ids if i in joint_map]
        self.left_leg_configs = [joint_map[i] for i in left_leg_ids if i in joint_map]

        # IK solvers
        self.ik_right_arm = InverseKinematicsSolver(
            self.fk_right_arm, self.right_arm_configs, right_arm_ids)
        self.ik_left_arm = InverseKinematicsSolver(
            self.fk_left_arm, self.left_arm_configs, left_arm_ids)

        # Trajectory planner (all joints)
        all_configs = [joint_map[i] for i in sorted(joint_map.keys())]
        self.trajectory_planner = TrajectoryPlanner(all_configs, self.dt)

        # Gait generator
        self.gait_generator = GaitGenerator(GaitParameters(), robot_config)

        # Impedance controllers
        self.impedance_right = ImpedanceController(self.fk_right_arm, self.right_arm_configs)
        self.impedance_left = ImpedanceController(self.fk_left_arm, self.left_arm_configs)

        # State tracking
        self._current_trajectory: Optional[Trajectory] = None
        self._trajectory_index: int = 0
        self._last_joint_states: Optional[List[JointState]] = None
        self._motion_phase = MotionPhase.IDLE

        logger.info(f"MotionController initialized | joints={len(joint_map)} | "
                    f"dt={self.dt*1000:.1f}ms | arms=7+7 DOF | legs=6+6 DOF")

    def execute(self, command: MotorCommand,
                current_states: List[JointState]) -> List[JointCommand]:
        """
        Convert a BrianMind MotorCommand into executable JointCommands.

        This is the main entry point called every control cycle.

        Args:
            command: High-level motor command from BrianMind.
            current_states: Current joint states from HAL.

        Returns:
            List of JointCommands to send to the robot.
        """
        self._last_joint_states = current_states
        joint_commands = []

        # Handle base velocity (walking)
        if command.base_velocity is not None and np.linalg.norm(command.base_velocity) > 0.01:
            leg_commands = self._generate_walking_commands(
                command.base_velocity, current_states)
            joint_commands.extend(leg_commands)

        # Handle arm joint commands (direct or from IK)
        if command.joint_commands:
            for jc in command.joint_commands:
                joint_commands.append(JointCommand(
                    joint_id=jc.get("joint_id", 0),
                    position=jc.get("position", 0.0),
                    velocity=jc.get("velocity", 0.0),
                    mode=ControlMode.POSITION,
                ))

        # Handle gripper
        if command.gripper_command is not None:
            force = command.gripper_command.get("force", 0.0)
            width = command.gripper_command.get("width", 0.04)
            # Right gripper (id 29), left gripper (id 30)
            joint_commands.append(JointCommand(
                joint_id=29, position=width,
                torque=force * 40.0,  # scale to gripper torque
                mode=ControlMode.POSITION,
            ))

        # Handle head targeting
        if command.head_target is not None:
            head_cmds = self._generate_head_commands(command.head_target)
            joint_commands.extend(head_cmds)

        return joint_commands

    def execute_cartesian(self, target: CartesianTarget, arm: str = "right",
                          current_states: List[JointState] = None) -> List[JointCommand]:
        """
        Move an arm to a Cartesian target using IK.

        Args:
            target: Desired end-effector pose.
            arm: "right" or "left".
            current_states: Current joint states.

        Returns:
            Joint commands for the specified arm.
        """
        ik_solver = self.ik_right_arm if arm == "right" else self.ik_left_arm
        arm_ids = list(range(0, 7)) if arm == "right" else list(range(7, 14))

        # Get current arm configuration
        q_current = np.zeros(7)
        if current_states:
            state_map = {s.joint_id: s for s in current_states}
            for i, jid in enumerate(arm_ids):
                if jid in state_map:
                    q_current[i] = state_map[jid].position

        success, q_target = ik_solver.solve(target, q_current)
        if not success:
            logger.warning(f"IK failed for {arm} arm target, returning current pose")
            return []

        # Plan smooth trajectory from current to target
        traj = self.trajectory_planner.plan_point_to_point(q_current, q_target)
        traj = self.trajectory_planner.time_optimal_scaling(traj)

        # Return the first trajectory point as immediate command
        if traj.points:
            pt = traj.points[min(1, len(traj.points) - 1)]
            commands = []
            for i, jid in enumerate(arm_ids):
                commands.append(JointCommand(
                    joint_id=jid,
                    position=float(pt.positions[i]),
                    velocity=float(pt.velocities[i]) if pt.velocities is not None else 0.0,
                    mode=ControlMode.POSITION,
                ))
            return commands
        return []

    def execute_impedance(self, params: ImpedanceParams, arm: str = "right",
                          current_states: List[JointState] = None) -> List[JointCommand]:
        """
        Execute impedance control for compliant interaction.

        Args:
            params: Impedance parameters.
            arm: "right" or "left".
            current_states: Current joint states.

        Returns:
            Joint commands with torque mode for compliant behavior.
        """
        controller = self.impedance_right if arm == "right" else self.impedance_left
        arm_ids = list(range(0, 7)) if arm == "right" else list(range(7, 14))

        q = np.zeros(7)
        dq = np.zeros(7)
        if current_states:
            state_map = {s.joint_id: s for s in current_states}
            for i, jid in enumerate(arm_ids):
                if jid in state_map:
                    q[i] = state_map[jid].position
                    dq[i] = state_map[jid].velocity

        torques = controller.compute_torques(q, dq, params)

        commands = []
        for i, jid in enumerate(arm_ids):
            commands.append(JointCommand(
                joint_id=jid,
                torque=float(torques[i]),
                mode=ControlMode.TORQUE,
            ))
        return commands

    def _generate_walking_commands(self, base_velocity: np.ndarray,
                                   current_states: List[JointState]) -> List[JointCommand]:
        """Generate leg joint commands for bipedal walking."""
        # Plan footsteps for the desired velocity
        footsteps = self.gait_generator.plan_footsteps(base_velocity, num_steps=4)
        if not footsteps:
            return []

        # Get first step target
        step = footsteps[0]
        foot_target = step["position"]

        # Determine which leg is swinging
        hip_offset_r = np.array([0.0, -self.gait_generator.params.step_width / 2,
                                 self.gait_generator.params.com_height])
        hip_offset_l = np.array([0.0, self.gait_generator.params.step_width / 2,
                                 self.gait_generator.params.com_height])

        commands = []
        right_leg_ids = list(range(17, 23))
        left_leg_ids = list(range(23, 29))

        if step["foot"] == "right":
            # Right leg swinging, compute IK
            angles = self.gait_generator.compute_leg_ik(foot_target, hip_offset_r)
            if angles is not None:
                for i, jid in enumerate(right_leg_ids):
                    commands.append(JointCommand(
                        joint_id=jid, position=float(angles[i]),
                        velocity=1.0, mode=ControlMode.POSITION))
            # Left leg holds stance
            for jid in left_leg_ids:
                commands.append(JointCommand(
                    joint_id=jid, position=0.0, velocity=0.0, mode=ControlMode.POSITION))
        else:
            # Left leg swinging
            angles = self.gait_generator.compute_leg_ik(foot_target, hip_offset_l)
            if angles is not None:
                for i, jid in enumerate(left_leg_ids):
                    commands.append(JointCommand(
                        joint_id=jid, position=float(angles[i]),
                        velocity=1.0, mode=ControlMode.POSITION))
            for jid in right_leg_ids:
                commands.append(JointCommand(
                    joint_id=jid, position=0.0, velocity=0.0, mode=ControlMode.POSITION))

        return commands

    def _generate_head_commands(self, head_target: np.ndarray) -> List[JointCommand]:
        """Generate head joint commands to look at a target direction."""
        pan = float(np.clip(head_target[0], -2.0, 2.0)) if len(head_target) > 0 else 0.0
        tilt = float(np.clip(head_target[1], -0.7, 0.7)) if len(head_target) > 1 else 0.0
        roll = 0.0  # Keep head level

        return [
            JointCommand(joint_id=14, position=pan, velocity=2.0, mode=ControlMode.POSITION),
            JointCommand(joint_id=15, position=tilt, velocity=2.0, mode=ControlMode.POSITION),
            JointCommand(joint_id=16, position=roll, velocity=1.0, mode=ControlMode.POSITION),
        ]

    def _build_arm_dh(self, side: str = "right") -> List[Dict[str, float]]:
        """
        Build DH parameters for a 7-DOF humanoid arm.
        Generic parameters suitable for Optimus/Figure-class robots.
        """
        sign = 1.0 if side == "right" else -1.0
        return [
            {"a": 0.0,   "d": 0.0,    "alpha": -math.pi / 2, "theta_offset": 0.0},   # shoulder pitch
            {"a": 0.0,   "d": 0.0,    "alpha": math.pi / 2,  "theta_offset": 0.0},    # shoulder roll
            {"a": 0.0,   "d": 0.30,   "alpha": -math.pi / 2, "theta_offset": 0.0},    # shoulder yaw
            {"a": 0.0,   "d": 0.0,    "alpha": math.pi / 2,  "theta_offset": 0.0},    # elbow
            {"a": 0.0,   "d": 0.25,   "alpha": -math.pi / 2, "theta_offset": 0.0},    # wrist roll
            {"a": 0.0,   "d": 0.0,    "alpha": math.pi / 2,  "theta_offset": 0.0},    # wrist pitch
            {"a": 0.0,   "d": 0.08,   "alpha": 0.0,          "theta_offset": 0.0},     # wrist yaw
        ]

    @property
    def motion_phase(self) -> MotionPhase:
        return self._motion_phase

    def get_status(self) -> Dict[str, Any]:
        return {
            "motion_phase": self._motion_phase.name,
            "has_trajectory": self._current_trajectory is not None,
            "trajectory_progress": (
                self._trajectory_index / len(self._current_trajectory.points)
                if self._current_trajectory and self._current_trajectory.points else 0.0
            ),
            "gait_phase": self.gait_generator.phase.name,
            "step_count": self.gait_generator._step_count,
        }
