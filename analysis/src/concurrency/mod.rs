//! Concurrency analysis: data races and lock discipline.
//!
//! Delegates to algorithms::concurrency::lockset.

pub use algorithms::concurrency::lockset::{LockId, LocksetState, ThreadId, VarId};
