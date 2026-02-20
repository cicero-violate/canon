use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Proof scopes differentiate structural law, execution law, and meta-law invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProofScope {
    Structure,
    Execution,
    Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProofId(String);

impl ProofId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub id: ProofId,
    pub uri: String,
    pub hash: String,
    pub scope: ProofScope,
}

/// Invariants are fully data-defined and activated only via proof-backed deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub id: String,
    pub proof_id: ProofId,
    pub payload_hash: String,
}

impl Delta {
    pub fn new(id: impl Into<String>, proof: &ProofArtifact, payload_hash: impl Into<String>) -> Self {
        Self { id: id.into(), proof_id: proof.id.clone(), payload_hash: payload_hash.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JudgmentDecision {
    Accept,
    Reject,
}

#[derive(Debug, Clone)]
pub struct Judgment {
    pub id: String,
    pub predicate: JudgmentPredicate,
}

#[derive(Debug, Clone)]
pub enum JudgmentPredicate {
    StateHashEquals(String),
    ProofScopeAllowed(ProofScope),
}

impl Judgment {
    pub fn evaluate(&self, state: &StateLog, invariants: &InvariantRegistry) -> JudgmentDecision {
        match &self.predicate {
            JudgmentPredicate::StateHashEquals(expected) => {
                if state.state_hash() == *expected {
                    JudgmentDecision::Accept
                } else {
                    JudgmentDecision::Reject
                }
            }
            JudgmentPredicate::ProofScopeAllowed(scope) => {
                if invariants.is_scope_allowed(*scope) {
                    JudgmentDecision::Accept
                } else {
                    JudgmentDecision::Reject
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub id: String,
    pub judgment_id: String,
    pub deltas: Vec<Delta>,
}

impl Admission {
    pub fn new(id: impl Into<String>, judgment_id: impl Into<String>, deltas: Vec<Delta>) -> Result<Self, KernelError> {
        if deltas.is_empty() {
            return Err(KernelError::EmptyAdmission);
        }
        Ok(Self { id: id.into(), judgment_id: judgment_id.into(), deltas })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaRecord {
    pub order: u64,
    pub delta: Delta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateLog {
    initial_state: String,
    log: Vec<DeltaRecord>,
}

impl StateLog {
    pub fn new(initial_state: impl Into<String>) -> Self {
        Self { initial_state: initial_state.into(), log: Vec::new() }
    }

    pub fn from_records(initial_state: impl Into<String>, records: Vec<DeltaRecord>) -> Self {
        Self { initial_state: initial_state.into(), log: records }
    }

    pub fn records(&self) -> &[DeltaRecord] {
        &self.log
    }

    pub fn state_hash(&self) -> String {
        let mut hasher = Hasher::new();
        hasher.update(self.initial_state.as_bytes());
        for record in &self.log {
            hasher.update(record.delta.id.as_bytes());
            hasher.update(record.delta.proof_id.as_str().as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    pub fn apply(&self, admission: &Admission) -> Self {
        let mut next = self.clone();
        let mut order = next.log.last().map(|r| r.order + 1).unwrap_or(0);
        for delta in &admission.deltas {
            next.log.push(DeltaRecord { order, delta: delta.clone() });
            order += 1;
        }
        next
    }
}

#[derive(Debug, Default)]
pub struct InvariantRegistry {
    allowed_scopes: BTreeSet<ProofScope>,
}

impl InvariantRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allow_scope(&mut self, scope: ProofScope) {
        self.allowed_scopes.insert(scope);
    }

    pub fn revoke_scope(&mut self, scope: ProofScope) {
        self.allowed_scopes.remove(&scope);
    }

    pub fn is_scope_allowed(&self, scope: ProofScope) -> bool {
        self.allowed_scopes.contains(&scope)
    }
}

#[derive(Debug, Default)]
pub struct ProofRegistry {
    proofs: BTreeMap<ProofId, ProofArtifact>,
}

impl ProofRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, proof: ProofArtifact) {
        self.proofs.insert(proof.id.clone(), proof);
    }

    pub fn get(&self, id: &ProofId) -> Option<&ProofArtifact> {
        self.proofs.get(id)
    }
}

pub fn apply_admission(state: &StateLog, judgment: &Judgment, admission: &Admission, invariants: &InvariantRegistry, proofs: &ProofRegistry) -> Result<StateLog, KernelError> {
    if judgment.id != admission.judgment_id {
        return Err(KernelError::JudgmentMismatch);
    }
    match judgment.evaluate(state, invariants) {
        JudgmentDecision::Accept => {
            for delta in &admission.deltas {
                let proof = proofs.get(&delta.proof_id).ok_or_else(|| KernelError::UnknownProof(delta.proof_id.as_str().to_string()))?;
                if !invariants.is_scope_allowed(proof.scope) {
                    return Err(KernelError::ProofScopeRejected(delta.id.clone()));
                }
            }
            Ok(state.apply(admission))
        }
        JudgmentDecision::Reject => Err(KernelError::JudgmentRejected),
    }
}

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("admission must contain at least one delta")]
    EmptyAdmission,
    #[error("judgment mismatch")]
    JudgmentMismatch,
    #[error("judgment rejected admission")]
    JudgmentRejected,
    #[error("proof `{0}` is unknown")]
    UnknownProof(String),
    #[error("proof scope rejected for delta `{0}`")]
    ProofScopeRejected(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proof(scope: ProofScope) -> ProofArtifact {
        ProofArtifact { id: ProofId::new(format!("proof-{:?}", scope)), uri: "file://proof".into(), hash: "abc123".into(), scope }
    }

    #[test]
    fn delta_requires_proof() {
        let pr = proof(ProofScope::Structure);
        let delta = Delta::new("delta-1", &pr, "hash-hello");
        assert_eq!(delta.proof_id.as_str(), pr.id.as_str());
    }

    #[test]
    fn apply_appends_deterministically() {
        let pr = proof(ProofScope::Execution);
        let admission = Admission::new("admit", "judgment", vec![Delta::new("delta", &pr, "hash-x")]).unwrap();
        let state = StateLog::new("s0");
        let mut invariants = InvariantRegistry::new();
        invariants.allow_scope(ProofScope::Execution);
        let mut proofs = ProofRegistry::new();
        proofs.register(pr.clone());
        let judgment = Judgment { id: "judgment".into(), predicate: JudgmentPredicate::StateHashEquals(state.state_hash()) };
        let next = apply_admission(&state, &judgment, &admission, &invariants, &proofs).unwrap();
        assert_eq!(next.records().len(), 1);
        let again = apply_admission(&state, &judgment, &admission, &invariants, &proofs).unwrap();
        assert_eq!(next.state_hash(), again.state_hash());
    }

    #[test]
    fn replay_is_deterministic() {
        let pr = proof(ProofScope::Execution);
        let admission = Admission::new("admit", "judgment", vec![Delta::new("delta", &pr, "hash-x")]).unwrap();
        let state = StateLog::new("s0");
        let mut invariants = InvariantRegistry::new();
        invariants.allow_scope(ProofScope::Execution);
        let mut proofs = ProofRegistry::new();
        proofs.register(pr.clone());
        let judgment = Judgment { id: "judgment".into(), predicate: JudgmentPredicate::StateHashEquals(state.state_hash()) };
        let next = apply_admission(&state, &judgment, &admission, &invariants, &proofs).unwrap();
        assert_ne!(state.state_hash(), next.state_hash());
        let replay = apply_admission(&state, &judgment, &admission, &invariants, &proofs).unwrap();
        assert_eq!(next.state_hash(), replay.state_hash());
    }

    #[test]
    fn invalid_states_unrepresentable() {
        let pr = proof(ProofScope::Structure);
        let delta = Delta::new("delta", &pr, "hash-hello");
        let admission = Admission::new("admit", "judgment", vec![delta]).unwrap();
        let invariants = InvariantRegistry::new();
        let mut proofs = ProofRegistry::new();
        proofs.register(pr.clone());
        let state = StateLog::new("s0");
        let judgment = Judgment { id: "judgment".into(), predicate: JudgmentPredicate::StateHashEquals("invalid".into()) };
        assert!(apply_admission(&state, &judgment, &admission, &invariants, &proofs).is_err());
    }
}
