// mmsb-memory - Domain-separated MMSB architecture
// #![allow(dead_code)]
#![allow(hidden_glob_reexports)]

mod canonical_state;
mod hash;
mod engine_commit;

pub mod primitives;
pub mod delta;
pub mod epoch;
pub mod graph_log;
pub mod page;
pub mod tlog;
pub mod proofs;
pub mod page_store;
pub mod journal;
pub mod engine;


pub use engine::Engine;
pub use canonical_state::CanonicalState;

pub use memory_engine::{
    AdmissionError,
    CommitError,
    MemoryEngine,
    MemoryEngineConfig,
    MemoryEngineError,
};

// expose proof types needed by canon
pub use proofs::{JudgmentProof, AdmissionProof, CommitProof, OutcomeProof};
