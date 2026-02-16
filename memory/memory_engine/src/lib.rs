// mmsb-memory - Domain-separated MMSB architecture
// #![allow(dead_code)]
#![allow(hidden_glob_reexports)]

// pub mod memory_engine;

// // Internal modules — NOT externally accessible
// pub (crate) mod primitives;
// pub (crate) mod delta;
// pub (crate) mod epoch;
// pub(crate) mod graph_log;
// pub(crate) mod page;
// pub(crate) mod tlog;
// pub(crate) mod proofs;

// // Only expose the authority surface
// pub use memory_engine::{
//     AdmissionError,
//     CommitError,
//     MemoryEngine,
//     MemoryEngineConfig,
//     MemoryEngineError,
//     StateSlice,
// };

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
