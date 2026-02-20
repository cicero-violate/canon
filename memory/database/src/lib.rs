// mmsb-memory — Canon-facing API surface

#![allow(hidden_glob_reexports)]

// ===== Internal modules (private) =====

pub mod canonical_state;
mod engine_commit;
pub mod hash;
mod memory_engine;
mod page;
mod page_store;
pub mod persistence;
pub mod transition;

// ===== Public domain modules =====

pub mod delta;
pub mod engine;
pub mod epoch;
pub mod graph_log;
pub mod primitives;
pub mod proofs;

// ===== Canon-facing API =====

pub use engine::Engine;
pub use transition::MemoryTransition;

pub use memory_engine::{AdmissionError, CommitError, MemoryEngine, MemoryEngineConfig, MemoryEngineError};

// Re-export CanonicalState for integration tests
pub use canonical_state::MerkleState;

// Proof types Canon needs
pub use proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof};

// Common types Canon needs
pub use delta::Delta;
pub use graph_log::{GraphDelta, GraphSnapshot};
pub use primitives::Hash;
