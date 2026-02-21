use crate::{
    delta::Delta, graph_log::{GraphDelta, GraphSnapshot},
    primitives::StateHash,
    proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof},
};
/// Public engine API consumed by Canon.
///
/// This trait is the stable boundary between:
/// - Canon (execution layer)
/// - database (state + persistence layer)
pub trait DeltaExecutionEngine: Send + Sync {
    type Error;
    fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, Self::Error>;
    fn register_delta(&self, delta: Delta) -> StateHash;
    fn fetch_delta_by_hash(&self, hash: &StateHash) -> Option<Delta>;
    fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta_hash: &StateHash,
    ) -> Result<CommitProof, Self::Error>;
    fn commit_batch(
        &self,
        admission: &AdmissionProof,
        delta_hashes: &[StateHash],
    ) -> Result<Vec<CommitProof>, Self::Error>;
    fn record_outcome(&self, commit: &CommitProof) -> OutcomeProof;
    fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> StateHash;
    fn commit_graph_delta(&self, delta: GraphDelta) -> Result<(), Self::Error>;
    fn materialized_graph(&self) -> Result<GraphSnapshot, Self::Error>;
}
