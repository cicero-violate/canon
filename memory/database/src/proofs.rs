//! Minimal proof artifacts emitted by the memory engine.
//!
//! These are dumb records that link stages (judgment → admission → commit → outcome).
//! Verification lives outside the memory subsystem.
use crate::primitives::StateHash;
use serde::{Deserialize, Serialize};
/// Judgment proof (Stage C) recorded from the authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentProof {
    pub approved: bool,
    pub timestamp: u64,
    pub hash: StateHash,
}
impl JudgmentProof {
    pub fn hash(&self) -> StateHash {
        self.hash
    }
}
/// Admission proof (Stage D) emitted after replay protection + epoch checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionProof {
    pub judgment_proof_hash: StateHash,
    pub epoch: u64,
    pub nonce: u64,
}
impl AdmissionProof {
    pub fn hash(&self) -> StateHash {
        self.judgment_proof_hash
    }
}
/// Commit proof (Stage E) anchors the applied delta and resulting state hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitProof {
    pub admission_proof_hash: StateHash,
    pub delta_hash: StateHash,
    pub state_hash: StateHash,
}
impl CommitProof {
    pub fn hash(&self) -> StateHash {
        self.state_hash
    }
}
/// Outcome proof (Stage F) summarizes post-commit effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeProof {
    pub commit_proof_hash: StateHash,
    pub success: bool,
}
impl OutcomeProof {
    pub fn hash(&self) -> StateHash {
        self.commit_proof_hash
    }
}
