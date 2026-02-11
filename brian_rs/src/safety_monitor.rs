//! Real-time safety constraint checking at wire speed.
//!
//! Validates joint commands against configured limits before they
//! reach the robot hardware. Runs in the Rust layer for minimal latency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    pub position_min: f64,
    pub position_max: f64,
    pub velocity_max: f64,
    pub torque_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub joint_limits: HashMap<u32, JointLimits>,
    pub max_cartesian_velocity: f64,
    pub max_contact_force: f64,
    pub heartbeat_timeout_ms: f64,
    pub max_command_age_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Caution,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub joint_id: Option<u32>,
    pub severity: ViolationSeverity,
    pub description: String,
    pub measured_value: Option<f64>,
    pub limit_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct JointCommandCheck {
    pub joint_id: u32,
    pub position: f64,
    pub velocity: f64,
    pub torque: f64,
}

pub struct SafetyMonitor {
    config: SafetyConfig,
}

impl SafetyMonitor {
    pub fn new(config: SafetyConfig) -> Self {
        Self { config }
    }

    /// Check a set of joint commands against safety limits.
    /// Returns (is_safe, violations).
    pub fn check_joint_commands(&self, commands: &[JointCommandCheck]) -> (bool, Vec<SafetyViolation>) {
        let mut violations = Vec::new();

        for cmd in commands {
            if let Some(limits) = self.config.joint_limits.get(&cmd.joint_id) {
                // Position check
                if cmd.position < limits.position_min || cmd.position > limits.position_max {
                    violations.push(SafetyViolation {
                        joint_id: Some(cmd.joint_id),
                        severity: ViolationSeverity::Critical,
                        description: format!(
                            "Joint {} position {:.3} outside [{:.3}, {:.3}]",
                            cmd.joint_id, cmd.position,
                            limits.position_min, limits.position_max
                        ),
                        measured_value: Some(cmd.position),
                        limit_value: Some(if cmd.position > limits.position_max {
                            limits.position_max
                        } else {
                            limits.position_min
                        }),
                    });
                }

                // Velocity check
                if cmd.velocity.abs() > limits.velocity_max {
                    violations.push(SafetyViolation {
                        joint_id: Some(cmd.joint_id),
                        severity: ViolationSeverity::Warning,
                        description: format!(
                            "Joint {} velocity {:.3} exceeds max {:.3}",
                            cmd.joint_id, cmd.velocity.abs(), limits.velocity_max
                        ),
                        measured_value: Some(cmd.velocity.abs()),
                        limit_value: Some(limits.velocity_max),
                    });
                }

                // Torque check
                if cmd.torque.abs() > limits.torque_max {
                    violations.push(SafetyViolation {
                        joint_id: Some(cmd.joint_id),
                        severity: ViolationSeverity::Warning,
                        description: format!(
                            "Joint {} torque {:.3} exceeds max {:.3}",
                            cmd.joint_id, cmd.torque.abs(), limits.torque_max
                        ),
                        measured_value: Some(cmd.torque.abs()),
                        limit_value: Some(limits.torque_max),
                    });
                }
            }
        }

        let is_safe = !violations.iter().any(|v| matches!(
            v.severity,
            ViolationSeverity::Critical | ViolationSeverity::Emergency
        ));

        (is_safe, violations)
    }

    /// Clamp joint commands to within safety limits.
    pub fn clamp_commands(&self, commands: &mut [JointCommandCheck]) {
        for cmd in commands.iter_mut() {
            if let Some(limits) = self.config.joint_limits.get(&cmd.joint_id) {
                cmd.position = cmd.position.clamp(limits.position_min, limits.position_max);
                cmd.velocity = cmd.velocity.clamp(-limits.velocity_max, limits.velocity_max);
                cmd.torque = cmd.torque.clamp(-limits.torque_max, limits.torque_max);
            }
        }
    }
}
