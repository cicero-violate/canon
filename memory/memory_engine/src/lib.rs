// mmsb-memory - Domain-separated MMSB architecture
// #![allow(dead_code)]
#![allow(hidden_glob_reexports)]

pub mod primitives;
pub mod delta;
pub mod epoch;
pub mod graph_log;
pub mod page;
pub mod tlog;
pub mod proofs;

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
pub use proofs::{JudgmentProof, AdmissionProof, CommitProof, OutcomeProof};
