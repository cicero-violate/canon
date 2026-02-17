use std::time::Duration;

/// Metrics returned by memory pressure handlers after a GC pass.
#[derive(Debug, Clone, Copy)]
pub struct GCMetrics {
    pub reclaimed_pages: usize,
    pub reclaimed_bytes: usize,
    pub duration: Duration,
}

/// Abstraction over memory pressure mitigation strategies.
pub trait MemoryPressureHandler: Send + Sync {
    /// Number of pages the handler attempts to reclaim on each incremental pass.
    fn incremental_batch_pages(&self) -> usize;

    /// Trigger a garbage collection pass with the provided budget.
    fn run_gc(&self, budget_pages: usize) -> Option<GCMetrics>;
}
