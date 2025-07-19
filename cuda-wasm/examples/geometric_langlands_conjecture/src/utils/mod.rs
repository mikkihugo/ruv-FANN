//! Utility functions and helpers

pub mod math;
pub mod logging;
pub mod metrics;
pub mod visualization;

use std::time::Instant;

/// Timer utility for performance measurement
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!("{}: {}ms", self.name, self.elapsed_ms());
    }
}