//! Brian-QARI Rust Performance Layer
//!
//! High-performance components for real-time robotics:
//! - Task orchestration with priority scheduling
//! - Safety monitoring at wire speed
//! - Protocol message codec
//! - Trajectory smoothing and interpolation
//! - Connection watchdog

pub mod orchestrator;
pub mod task_queue;
pub mod safety_monitor;
pub mod protocol_codec;
pub mod watchdog;
