//! Priority-based task queue for real-time robot task scheduling.
//!
//! Extends the LLMA system's TaskQueue with deadline support
//! for real-time robotics constraints.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Navigate,
    Reach,
    Grasp,
    Place,
    Speak,
    Gesture,
    Look,
    EmergencyStop,
    SafetyCheck,
    Perception,
    MotorControl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotTask {
    pub id: String,
    pub task_type: TaskType,
    pub priority: u8,  // 0=highest (E-stop), 255=lowest
    pub description: String,
    pub robot_id: Option<String>,
    pub parameters: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct PrioritizedTask {
    task: RobotTask,
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.priority == other.task.priority
    }
}

impl Eq for PrioritizedTask {}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority number = higher priority (reverse order)
        other.task.priority.cmp(&self.task.priority)
            .then_with(|| {
                // Break ties by deadline (earlier deadline first)
                match (&self.task.deadline, &other.task.deadline) {
                    (Some(a), Some(b)) => a.cmp(b),
                    (Some(_), None) => Ordering::Greater,
                    (None, Some(_)) => Ordering::Less,
                    (None, None) => Ordering::Equal,
                }
            })
    }
}

pub struct TaskQueue {
    tasks: Arc<RwLock<BinaryHeap<PrioritizedTask>>>,
    completed_tasks: Arc<RwLock<Vec<RobotTask>>>,
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(BinaryHeap::new())),
            completed_tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn enqueue(&self, task: RobotTask) {
        let prioritized = PrioritizedTask { task };
        self.tasks.write().push(prioritized);
    }

    pub fn dequeue(&self) -> Option<RobotTask> {
        self.tasks.write().pop().map(|p| p.task)
    }

    pub fn peek(&self) -> Option<RobotTask> {
        self.tasks.read().peek().map(|p| p.task.clone())
    }

    pub fn len(&self) -> usize {
        self.tasks.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.read().is_empty()
    }

    pub fn mark_completed(&self, task: RobotTask) {
        self.completed_tasks.write().push(task);
    }

    pub fn completed_count(&self) -> usize {
        self.completed_tasks.read().len()
    }

    /// Remove tasks that have passed their deadline
    pub fn prune_expired(&self) -> usize {
        let now = Utc::now();
        let mut queue = self.tasks.write();
        let before = queue.len();
        let remaining: Vec<_> = queue.drain()
            .filter(|p| {
                p.task.deadline.map_or(true, |d| d > now)
            })
            .collect();
        *queue = remaining.into_iter().collect();
        before - queue.len()
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}
