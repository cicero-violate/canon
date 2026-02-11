// mmsb-memory - Domain-separated MMSB architecture
#![allow(dead_code)]
#![allow(hidden_glob_reexports)]

pub mod primitives;

// Memory engine - canonical truth owner
pub mod memory_engine;

// Core data structures
pub mod delta;
pub mod epoch;
pub mod graph_log;
pub mod page;
pub mod tlog;

// Proof artifacts
pub mod proofs;

pub use delta::Delta;
pub use memory_engine::{CanonicalState, StateSlice};
pub use tlog::{TlogEntry, TlogManager};
