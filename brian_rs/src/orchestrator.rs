//! Brian Orchestrator - Real-time agent coordination for robotics.
//!
//! Extends the LLMA orchestrator pattern for multi-robot control
//! with real-time safety constraints.

use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::task_queue::{TaskQueue, RobotTask};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorStats {
    pub total_tasks_executed: usize,
    pub successful_tasks: usize,
    pub failed_tasks: usize,
    pub active_robots: usize,
    pub avg_latency_ms: f64,
    pub last_activity: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct RobotConnection {
    pub robot_id: String,
    pub platform_type: String,
    pub is_connected: bool,
    pub last_heartbeat: DateTime<Utc>,
    pub latency_ms: f64,
    pub tasks_completed: usize,
}

pub struct BrianOrchestrator {
    task_queue: Arc<TaskQueue>,
    robot_connections: Arc<RwLock<HashMap<String, RobotConnection>>>,
    stats: Arc<RwLock<OrchestratorStats>>,
    is_running: Arc<RwLock<bool>>,
}

impl BrianOrchestrator {
    pub fn new() -> Self {
        Self {
            task_queue: Arc::new(TaskQueue::new()),
            robot_connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(OrchestratorStats {
                total_tasks_executed: 0,
                successful_tasks: 0,
                failed_tasks: 0,
                active_robots: 0,
                avg_latency_ms: 0.0,
                last_activity: None,
            })),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub fn register_robot(&self, robot_id: String, platform_type: String) {
        let conn = RobotConnection {
            robot_id: robot_id.clone(),
            platform_type,
            is_connected: true,
            last_heartbeat: Utc::now(),
            latency_ms: 0.0,
            tasks_completed: 0,
        };
        self.robot_connections.write().insert(robot_id, conn);
        self.stats.write().active_robots = self.robot_connections.read().len();
    }

    pub fn submit_task(&self, task: RobotTask) {
        self.task_queue.enqueue(task);
    }

    pub fn get_next_task(&self) -> Option<RobotTask> {
        self.task_queue.dequeue()
    }

    pub fn record_task_result(&self, task: RobotTask, success: bool) {
        let mut stats = self.stats.write();
        stats.total_tasks_executed += 1;
        if success {
            stats.successful_tasks += 1;
        } else {
            stats.failed_tasks += 1;
        }
        stats.last_activity = Some(Utc::now());
        self.task_queue.mark_completed(task);
    }

    pub fn update_heartbeat(&self, robot_id: &str, latency_ms: f64) {
        if let Some(conn) = self.robot_connections.write().get_mut(robot_id) {
            conn.last_heartbeat = Utc::now();
            conn.latency_ms = latency_ms;
            conn.is_connected = true;
        }
    }

    pub fn get_stats(&self) -> OrchestratorStats {
        self.stats.read().clone()
    }

    pub fn pending_tasks(&self) -> usize {
        self.task_queue.len()
    }

    pub fn connected_robots(&self) -> Vec<String> {
        self.robot_connections.read()
            .iter()
            .filter(|(_, c)| c.is_connected)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

impl Default for BrianOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}
