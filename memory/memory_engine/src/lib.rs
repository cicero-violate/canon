// mmsb-memory - Domain-separated MMSB architecture
// #![allow(dead_code)]
#![allow(hidden_glob_reexports)]

mod primitives;
mod delta;
mod epoch;
mod graph_log;
mod page;
mod tlog;
mod proofs;

// Authority surface — ONLY this is public
pub mod memory_engine;

pub use memory_engine::{
    AdmissionError,
    CommitError,
    MemoryEngine,
    MemoryEngineConfig,
    MemoryEngineError,
    StateSlice,
};

// expose proof types needed by canon
// pub use crate::proofs::JudgmentProof;
// pub use primitives::Hash;
pub use proofs::{JudgmentProof, AdmissionProof, CommitProof, OutcomeProof};
