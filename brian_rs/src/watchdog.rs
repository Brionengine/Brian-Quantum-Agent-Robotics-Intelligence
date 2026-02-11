//! Connection watchdog timer for robot heartbeat monitoring.
//!
//! Triggers fallback behaviors when brain-robot connection drops.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    Connected,
    HighLatency,
    Intermittent,
    Lost,
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct WatchdogEntry {
    pub robot_id: String,
    pub last_heartbeat: Instant,
    pub status: ConnectionStatus,
    pub consecutive_misses: u32,
    pub latency_ms: f64,
}

pub struct Watchdog {
    entries: Arc<RwLock<HashMap<String, WatchdogEntry>>>,
    heartbeat_timeout: Duration,
    high_latency_threshold: Duration,
    intermittent_threshold: u32,  // consecutive misses before intermittent
    lost_threshold: u32,          // consecutive misses before lost
}

impl Watchdog {
    pub fn new(
        heartbeat_timeout_ms: u64,
        high_latency_ms: u64,
        intermittent_misses: u32,
        lost_misses: u32,
    ) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            heartbeat_timeout: Duration::from_millis(heartbeat_timeout_ms),
            high_latency_threshold: Duration::from_millis(high_latency_ms),
            intermittent_threshold: intermittent_misses,
            lost_threshold: lost_misses,
        }
    }

    pub fn register(&self, robot_id: String) {
        self.entries.write().insert(robot_id.clone(), WatchdogEntry {
            robot_id,
            last_heartbeat: Instant::now(),
            status: ConnectionStatus::Connected,
            consecutive_misses: 0,
            latency_ms: 0.0,
        });
    }

    pub fn heartbeat_received(&self, robot_id: &str, latency_ms: f64) {
        if let Some(entry) = self.entries.write().get_mut(robot_id) {
            entry.last_heartbeat = Instant::now();
            entry.consecutive_misses = 0;
            entry.latency_ms = latency_ms;

            if latency_ms > self.high_latency_threshold.as_millis() as f64 {
                entry.status = ConnectionStatus::HighLatency;
            } else {
                entry.status = ConnectionStatus::Connected;
            }
        }
    }

    /// Check all connections and return those that changed status.
    pub fn check(&self) -> Vec<(String, ConnectionStatus, ConnectionStatus)> {
        let now = Instant::now();
        let mut changes = Vec::new();

        for entry in self.entries.write().values_mut() {
            let elapsed = now.duration_since(entry.last_heartbeat);
            let old_status = entry.status;

            if elapsed > self.heartbeat_timeout {
                entry.consecutive_misses += 1;

                if entry.consecutive_misses >= self.lost_threshold {
                    entry.status = ConnectionStatus::Lost;
                } else if entry.consecutive_misses >= self.intermittent_threshold {
                    entry.status = ConnectionStatus::Intermittent;
                }
            }

            if entry.status != old_status {
                changes.push((
                    entry.robot_id.clone(),
                    old_status,
                    entry.status,
                ));
            }
        }

        changes
    }

    pub fn get_status(&self, robot_id: &str) -> Option<ConnectionStatus> {
        self.entries.read().get(robot_id).map(|e| e.status)
    }

    pub fn get_all_statuses(&self) -> HashMap<String, ConnectionStatus> {
        self.entries.read()
            .iter()
            .map(|(id, e)| (id.clone(), e.status))
            .collect()
    }
}

impl Default for Watchdog {
    fn default() -> Self {
        Self::new(200, 100, 3, 10)
    }
}
