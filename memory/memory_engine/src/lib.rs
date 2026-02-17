// mmsb-memory — Canon-facing API surface

#![allow(hidden_glob_reexports)]

// ===== Internal modules (private) =====

pub mod canonical_state;
pub mod hash;
pub mod persistence;
mod memory_engine;
mod engine_commit;
mod page_store;
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

// Re-export CanonicalState for integration tests
pub use canonical_state::CanonicalState;

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
