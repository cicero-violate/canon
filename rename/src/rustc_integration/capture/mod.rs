//! Capture pipeline infrastructure shared between compiler frontends.

/// Workspace orchestration.
pub mod coordinator;
/// Deduplication helpers.
pub mod dedup;
/// Session tracking utilities.
pub mod session;
/// Validation helpers for captured items.
pub mod validation;

pub use coordinator::CaptureCoordinator;
