use std::collections::HashMap;
use blake3::hash;
use canon_kernel::{
    Admission as KernelAdmission, Delta as KernelDelta, DeltaRecord as KernelDeltaRecord,
    InvariantRegistry as KernelInvariantRegistry, ProofArtifact as KernelProofArtifact,
    ProofId as KernelProofId, ProofRegistry as KernelProofRegistry,
    ProofScope as KernelProofScope, StateLog,
};
use serde_json;
use crate::ir::{SystemState, CanonicalMeta, StateChange, ChangeAdmission, Proof};
use super::EvolutionError;
pub(super) fn build_proof_registry(proofs: &[Proof]) -> KernelProofRegistry {
    let mut registry = KernelProofRegistry::new();
    for proof in proofs {
        let artifact = KernelProofArtifact {
            id: KernelProofId::new(proof.id.clone()),
            uri: proof.evidence.uri.clone(),
            hash: proof.evidence.hash.clone(),
            scope: map_proof_scope(proof.scope),
        };
        registry.register(artifact);
    }
    registry
}
pub(super) fn build_invariant_registry() -> KernelInvariantRegistry {
    let mut registry = KernelInvariantRegistry::new();
    registry.allow_scope(KernelProofScope::Structure);
    registry.allow_scope(KernelProofScope::Execution);
    registry.allow_scope(KernelProofScope::Meta);
    registry
}
pub(super) fn build_state_log(
    ir: &SystemState,
    deltas: &HashMap<&str, &StateChange>,
) -> Result<StateLog, EvolutionError> {
    let records = ir
        .applied_deltas
        .iter()
        .map(|record| {
            let delta = deltas
                .get(record.delta.as_str())
                .ok_or_else(|| EvolutionError::UnknownDelta(record.delta.clone()))?;
            Ok(KernelDeltaRecord {
                order: record.order,
                delta: kernel_delta(delta)?,
            })
        })
        .collect::<Result<Vec<_>, EvolutionError>>()?;
    Ok(StateLog::from_records(initial_state_seed(&ir.meta), records))
}
pub(super) fn build_kernel_admission(
    admission: &ChangeAdmission,
    deltas: &HashMap<&str, &StateChange>,
) -> Result<KernelAdmission, EvolutionError> {
    let kernel_deltas = admission
        .delta_ids
        .iter()
        .map(|delta_id| {
            let delta = deltas
                .get(delta_id.as_str())
                .ok_or_else(|| EvolutionError::UnknownDelta(delta_id.clone()))?;
            kernel_delta(delta)
        })
        .collect::<Result<Vec<_>, EvolutionError>>()?;
    KernelAdmission::new(admission.id.clone(), admission.judgment.clone(), kernel_deltas)
        .map_err(|err| EvolutionError::Kernel(err.to_string()))
}
fn kernel_delta(delta: &StateChange) -> Result<KernelDelta, EvolutionError> {
    Ok(KernelDelta {
        id: delta.id.clone(),
        proof_id: KernelProofId::new(delta.proof.clone()),
        payload_hash: payload_hash(delta)?,
    })
}
fn payload_hash(delta: &StateChange) -> Result<String, EvolutionError> {
    let bytes = serde_json::to_vec(&delta.payload)
        .map_err(|err| EvolutionError::PayloadHash(err.to_string()))?;
    Ok(hash(&bytes).to_hex().to_string())
}
fn initial_state_seed(meta: &CanonicalMeta) -> String {
    format!("{}::{}", meta.law_revision, meta.version)
}
fn map_proof_scope(scope: crate::ir::ProofScope) -> KernelProofScope {
    match scope {
        crate::ir::ProofScope::Structure => KernelProofScope::Structure,
        crate::ir::ProofScope::Execution => KernelProofScope::Execution,
        crate::ir::ProofScope::Law => KernelProofScope::Meta,
    }
}
