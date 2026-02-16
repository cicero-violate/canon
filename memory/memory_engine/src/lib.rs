// mmsb-memory - Domain-separated MMSB architecture
#![allow(dead_code)]
#![allow(hidden_glob_reexports)]

pub mod memory_engine;

// Internal modules — NOT externally accessible
pub(crate) mod primitives;
pub(crate) mod delta;
pub(crate) mod epoch;
pub(crate) mod graph_log;
pub(crate) mod page;
pub(crate) mod tlog;
pub(crate) mod proofs;

// Only expose the authority surface
pub use memory_engine::{
    AdmissionError,
    CommitError,
    MemoryEngine,
    MemoryEngineConfig,
    MemoryEngineError,
    StateSlice,
};
