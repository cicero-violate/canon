use crate::{
    delta::Delta,
    graph_log::{GraphDelta, GraphSnapshot},
    primitives::Hash,
    proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof},
};

/// Public engine API consumed by Canon.
///
/// This trait is the stable boundary between:
/// - Canon (execution layer)
/// - memory_engine (state + persistence layer)
pub trait Engine: Send + Sync {
    type Error;

    // ---------------- Admission ----------------

    fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, Self::Error>;

    // ---------------- Delta registry ----------------

    fn register_delta(&self, delta: Delta) -> Hash;

    fn fetch_delta_by_hash(&self, hash: &Hash) -> Option<Delta>;

    // ---------------- Commit ----------------

    fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta_hash: &Hash,
    ) -> Result<CommitProof, Self::Error>;

    fn commit_batch(
        &self,
        admission: &AdmissionProof,
        delta_hashes: &[Hash],
    ) -> Result<Vec<CommitProof>, Self::Error>;

    // ---------------- Outcome ----------------

    fn record_outcome(&self, commit: &CommitProof) -> OutcomeProof;

    // ---------------- Event ----------------

    fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> Hash;

    // ---------------- Graph ----------------

    fn commit_graph_delta(&self, delta: GraphDelta) -> Result<(), Self::Error>;

    fn materialized_graph(&self) -> Result<GraphSnapshot, Self::Error>;
}
