pub mod goal_mutation;
mod kernel_bridge;
pub mod lyapunov;
mod structural;
use std::collections::HashMap;
use canon_kernel::{self as kernel, Judgment as KernelJudgment, JudgmentPredicate};
use thiserror::Error;
use crate::ir::{
    AdmissionId, AppliedDeltaRecord, CanonicalIr, Delta, DeltaId, DeltaKind,
    DeltaPayload, JudgmentDecision,
};
use crate::runtime::delta_verifier::{DeltaVerifier, VerificationError};
pub use goal_mutation::{GoalMutationError, mutate_goal};
use kernel_bridge::{
    build_invariant_registry, build_kernel_admission, build_proof_registry,
    build_state_log,
};
pub use lyapunov::{
    DEFAULT_TOPOLOGY_THETA, LyapunovError, TopologyFingerprint, enforce_lyapunov_bound,
};
use structural::apply_structural_delta;
pub fn apply_deltas(
    ir: &CanonicalIr,
    admission_ids: &[AdmissionId],
) -> Result<CanonicalIr, EvolutionError> {
    let snapshot = DeltaVerifier::create_snapshot(ir);
    let judgments: HashMap<_, _> = ir
        .judgments
        .iter()
        .map(|j| (j.id.as_str(), j))
        .collect();
    let deltas: HashMap<_, _> = ir.deltas.iter().map(|d| (d.id.as_str(), d)).collect();
    let admissions: HashMap<_, _> = ir
        .admissions
        .iter()
        .map(|a| (a.id.as_str(), a))
        .collect();
    let mut next = ir.clone();
    let mut order = next.applied_deltas.last().map(|r| r.order + 1).unwrap_or(0);
    let proof_registry = build_proof_registry(&ir.proofs);
    let invariants = build_invariant_registry();
    let mut state = build_state_log(&next, &deltas)?;
    for admission_id in admission_ids {
        let admission = admissions
            .get(admission_id.as_str())
            .ok_or_else(|| EvolutionError::UnknownAdmission(admission_id.clone()))?;
        let judgment = judgments
            .get(admission.judgment.as_str())
            .ok_or_else(|| EvolutionError::UnknownJudgment(admission.judgment.clone()))?;
        if judgment.decision != JudgmentDecision::Accept {
            return Err(EvolutionError::JudgmentNotAccepted(admission.judgment.clone()));
        }
        let kernel_admission = build_kernel_admission(admission, &deltas)?;
        let kernel_judgment = KernelJudgment {
            id: admission.judgment.clone(),
            predicate: JudgmentPredicate::StateHashEquals(state.state_hash()),
        };
        state = kernel::apply_admission(
                &state,
                &kernel_judgment,
                &kernel_admission,
                &invariants,
                &proof_registry,
            )
            .map_err(|err| EvolutionError::Kernel(err.to_string()))?;
        for delta_id in &admission.delta_ids {
            let delta = deltas
                .get(delta_id.as_str())
                .ok_or_else(|| EvolutionError::UnknownDelta(delta_id.clone()))?;
            enforce_delta_application(delta)?;
            if delta.kind == DeltaKind::Structure {
                let proof_ids: Vec<String> = ir
                    .proofs
                    .iter()
                    .map(|p| p.id.clone())
                    .collect();
                let mut candidate = next.clone();
                apply_structural_delta(&mut candidate, delta)?;
                enforce_lyapunov_bound(
                        &next,
                        &candidate,
                        &proof_ids,
                        DEFAULT_TOPOLOGY_THETA,
                    )
                    .map_err(EvolutionError::TopologyDrift)?;
                next = candidate;
            } else {
                apply_structural_delta(&mut next, delta)?;
            }
            next.applied_deltas
                .push(AppliedDeltaRecord {
                    id: format!("{}#{}", admission.id, order),
                    admission: admission.id.clone(),
                    delta: delta_id.clone(),
                    order,
                });
            order += 1;
        }
    }
    match DeltaVerifier::verify_application(ir, &next, admission_ids) {
        Ok(_) => Ok(next),
        Err(verification_err) => {
            eprintln!("Delta verification failed: {}", verification_err);
            eprintln!("Rolling back to snapshot with hash: {}", snapshot.state_hash);
            Err(EvolutionError::VerificationFailed(verification_err))
        }
    }
}
fn enforce_delta_application(delta: &Delta) -> Result<(), EvolutionError> {
    if matches!(delta.payload, Some(DeltaPayload::AddFunction { .. }))
        && delta.related_function.is_none()
    {
        return Err(EvolutionError::MissingContext(delta.id.clone()));
    }
    Ok(())
}
#[derive(Debug, Error)]
pub enum EvolutionError {
    #[error("unknown admission `{0}`")]
    UnknownAdmission(AdmissionId),
    #[error("unknown judgment `{0}`")]
    UnknownJudgment(String),
    #[error("unknown delta `{0}`")]
    UnknownDelta(DeltaId),
    #[error("judgment `{0}` is not accepted")]
    JudgmentNotAccepted(String),
    #[error("delta `{0}` is missing context for application")]
    MissingContext(DeltaId),
    #[error("struct `{0}` does not exist")]
    UnknownStruct(String),
    #[error("field `{field}` does not exist on struct `{struct_id}`")]
    UnknownField { struct_id: String, field: String },
    #[error("module `{0}` does not exist")]
    UnknownModule(String),
    #[error("impl `{0}` does not exist")]
    UnknownImpl(String),
    #[error("trait `{0}` does not exist")]
    UnknownTrait(String),
    #[error("trait function `{0}` does not exist")]
    UnknownTraitFunction(String),
    #[error("execution `{0}` does not exist")]
    UnknownExecution(String),
    #[error("function `{0}` does not exist")]
    UnknownFunction(String),
    #[error("enum `{0}` does not exist")]
    UnknownEnum(String),
    #[error("artifact `{0}` already exists")]
    DuplicateArtifact(String),
    #[error("failed to hash delta payload: {0}")]
    PayloadHash(String),
    #[error("kernel error: {0}")]
    Kernel(String),
    #[error("delta verification failed")]
    VerificationFailed(VerificationError),
    #[error("topology drift rejected: {0}")]
    TopologyDrift(LyapunovError),
}
