#![allow(hidden_glob_reexports)]

// Pure storage + state machine crate

pub mod canonical_state;
pub mod delta;
pub mod epoch;
pub mod graph_log;
pub mod hash;
pub mod page;
pub mod page_store;
pub mod persistence;
pub mod primitives;
pub mod proofs;

// Public surface (state layer only)
pub use canonical_state::MerkleState;
pub use delta::Delta;
pub use graph_log::{GraphDelta, GraphSnapshot};
pub use primitives::StateHash;
pub use proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof};
