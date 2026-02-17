// mmsb-memory — Canon-facing API surface

#![allow(hidden_glob_reexports)]

// ===== Internal modules (private) =====

mod canonical_state;
mod hash;
mod memory_engine;
mod engine_commit;
mod page_store;
mod journal;
mod page;
mod tlog;

// ===== Public domain modules =====

pub mod primitives;
pub mod delta;
pub mod epoch;
pub mod graph_log;
pub mod proofs;
pub mod engine;

// ===== Canon-facing API =====

pub use engine::Engine;

pub use memory_engine::{
    MemoryEngine,
    MemoryEngineConfig,
    MemoryEngineError,
    AdmissionError,
    CommitError,
};

// Proof types Canon needs
pub use proofs::{
    JudgmentProof,
    AdmissionProof,
    CommitProof,
    OutcomeProof,
};

// Common types Canon needs
pub use primitives::Hash;
pub use delta::Delta;
pub use graph_log::{GraphDelta, GraphSnapshot};
