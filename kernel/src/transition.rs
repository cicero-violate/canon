//! Minimal transition interface for callers that only need `(state, delta) -> (state, proof)`.
//!
//! This trait intentionally hides all snapshot, checkpoint, and WAL plumbing.
use crate::{delta::Delta, memory_engine::KernelError, primitives::StateHash, proofs::CommitProof};
/// Minimal surface that exposes the canonical `(Hash, Delta) -> (Hash, CommitProof)` transition.
pub trait MemoryTransition {
    /// Returns the current canonical root hash.
    fn genesis(&self) -> StateHash;
    /// Applies a delta starting from the provided state hash and emits the resulting hash + proof.
    fn step(&self, state: StateHash, delta: Delta) -> Result<(StateHash, CommitProof), KernelError>;
}
