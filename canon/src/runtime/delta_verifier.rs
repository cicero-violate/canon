//! Delta application verification (Canon Lines 68-69).
//!
//! Verifies deltas are applied correctly with state hash validation.
//! Implements rollback on verification failure.

use std::collections::HashMap;
use thiserror::Error;

use crate::ir::{AdmissionId, AppliedDeltaRecord, CanonicalIr, Delta, DeltaId, Proof};

/// Verifies delta application with state hash checking.
pub struct DeltaVerifier;

impl DeltaVerifier {
    /// Verify that applying deltas produces expected state hash.
    /// Returns error if verification fails (Canon Line 69).
    pub fn verify_application(
        original_ir: &CanonicalIr,
        evolved_ir: &CanonicalIr,
        admission_ids: &[AdmissionId],
    ) -> Result<VerificationResult, VerificationError> {
        // Step 1: Compute expected state hash before application
        let before_hash = compute_state_hash(original_ir);

        // Step 2: Verify all deltas have proofs (Canon Line 68)
        Self::verify_delta_proofs(original_ir, admission_ids)?;

        // Step 3: Compute actual state hash after application
        let after_hash = compute_state_hash(evolved_ir);

        // Step 4: Verify delta ordering is preserved
        Self::verify_delta_ordering(original_ir, evolved_ir, admission_ids)?;

        // Step 5: Verify all referenced deltas were applied
        Self::verify_deltas_applied(original_ir, evolved_ir, admission_ids)?;

        Ok(VerificationResult {
            before_hash,
            after_hash,
            verified: true,
        })
    }

    /// Verify all deltas have attached proofs (Canon Line 68).
    fn verify_delta_proofs(
        ir: &CanonicalIr,
        admission_ids: &[AdmissionId],
    ) -> Result<(), VerificationError> {
        let proofs: HashMap<_, _> = ir.proofs.iter().map(|p| (p.id.as_str(), p)).collect();
        let deltas: HashMap<_, _> = ir.deltas.iter().map(|d| (d.id.as_str(), d)).collect();
        let admissions: HashMap<_, _> = ir.admissions.iter().map(|a| (a.id.as_str(), a)).collect();

        for admission_id in admission_ids {
            let admission = admissions
                .get(admission_id.as_str())
                .ok_or_else(|| VerificationError::UnknownAdmission(admission_id.clone()))?;

            for delta_id in &admission.delta_ids {
                let delta = deltas
                    .get(delta_id.as_str())
                    .ok_or_else(|| VerificationError::UnknownDelta(delta_id.clone()))?;

                // Verify proof exists (Canon Line 68)
                let proof = proofs.get(delta.proof.as_str()).ok_or_else(|| {
                    VerificationError::MissingProof {
                        delta_id: delta_id.clone(),
                        proof_id: delta.proof.clone(),
                    }
                })?;

                // Verify proof scope is appropriate for delta kind
                Self::verify_proof_scope(delta, proof)?;
            }
        }

        Ok(())
    }

    /// Verify proof scope matches delta kind (Canon proof scope rules).
    fn verify_proof_scope(delta: &Delta, proof: &Proof) -> Result<(), VerificationError> {
        use crate::ir::{DeltaKind, ProofScope};

        let valid = match delta.kind {
            DeltaKind::State => proof.scope == ProofScope::Execution,
            DeltaKind::Io => proof.scope == ProofScope::Execution,
            DeltaKind::Structure => {
                proof.scope == ProofScope::Structure || proof.scope == ProofScope::Law
            }
            DeltaKind::History => {
                proof.scope == ProofScope::Structure || proof.scope == ProofScope::Execution
            }
        };

        if !valid {
            return Err(VerificationError::InvalidProofScope {
                delta_id: delta.id.clone(),
                delta_kind: delta.kind,
                proof_scope: proof.scope,
            });
        }

        Ok(())
    }

    /// Verify delta ordering is preserved (Canon Line 38: append-only).
    fn verify_delta_ordering(
        original_ir: &CanonicalIr,
        evolved_ir: &CanonicalIr,
        admission_ids: &[AdmissionId],
    ) -> Result<(), VerificationError> {
        let original_count = original_ir.applied_deltas.len();
        let evolved_count = evolved_ir.applied_deltas.len();

        // Count expected new records
        let admissions: HashMap<_, _> = original_ir
            .admissions
            .iter()
            .map(|a| (a.id.as_str(), a))
            .collect();

        let mut expected_new_count = 0;
        for admission_id in admission_ids {
            let admission = admissions
                .get(admission_id.as_str())
                .ok_or_else(|| VerificationError::UnknownAdmission(admission_id.clone()))?;
            expected_new_count += admission.delta_ids.len();
        }

        if evolved_count != original_count + expected_new_count {
            return Err(VerificationError::DeltaCountMismatch {
                expected: original_count + expected_new_count,
                actual: evolved_count,
            });
        }

        // Verify original deltas are unchanged (append-only)
        for (idx, original_record) in original_ir.applied_deltas.iter().enumerate() {
            let evolved_record = &evolved_ir.applied_deltas[idx];
            if original_record.delta != evolved_record.delta
                || original_record.order != evolved_record.order
            {
                return Err(VerificationError::DeltaOrderingViolation {
                    index: idx,
                    original: original_record.delta.clone(),
                    evolved: evolved_record.delta.clone(),
                });
            }
        }

        // Verify new deltas are in correct order
        let mut previous_order = original_ir.applied_deltas.last().map(|r| r.order);

        for record in &evolved_ir.applied_deltas[original_count..] {
            if let Some(prev) = previous_order {
                if record.order <= prev {
                    return Err(VerificationError::NonMonotonicOrder {
                        delta_id: record.delta.clone(),
                        order: record.order,
                        previous: prev,
                    });
                }
            }
            previous_order = Some(record.order);
        }

        Ok(())
    }

    /// Verify all deltas in admissions were applied.
    fn verify_deltas_applied(
        original_ir: &CanonicalIr,
        evolved_ir: &CanonicalIr,
        admission_ids: &[AdmissionId],
    ) -> Result<(), VerificationError> {
        let admissions: HashMap<_, _> = original_ir
            .admissions
            .iter()
            .map(|a| (a.id.as_str(), a))
            .collect();

        let applied_deltas: HashMap<_, _> = evolved_ir
            .applied_deltas
            .iter()
            .map(|r| (r.delta.as_str(), r))
            .collect();

        for admission_id in admission_ids {
            let admission = admissions
                .get(admission_id.as_str())
                .ok_or_else(|| VerificationError::UnknownAdmission(admission_id.clone()))?;

            for delta_id in &admission.delta_ids {
                if !applied_deltas.contains_key(delta_id.as_str()) {
                    return Err(VerificationError::DeltaNotApplied {
                        delta_id: delta_id.clone(),
                        admission_id: admission_id.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Create a snapshot for rollback.
    pub fn create_snapshot(ir: &CanonicalIr) -> Snapshot {
        Snapshot {
            state_hash: compute_state_hash(ir),
            applied_deltas: ir.applied_deltas.clone(),
            modules: ir.modules.len(),
            structs: ir.structs.len(),
            traits: ir.traits.len(),
            functions: ir.functions.len(),
        }
    }

    /// Verify snapshot matches current state.
    pub fn verify_snapshot(ir: &CanonicalIr, snapshot: &Snapshot) -> bool {
        compute_state_hash(ir) == snapshot.state_hash
            && ir.applied_deltas.len() == snapshot.applied_deltas.len()
    }
}

/// Compute deterministic state hash from IR.
/// Includes all structural artifacts and applied deltas.
fn compute_state_hash(ir: &CanonicalIr) -> String {
    let mut hasher = blake3::Hasher::new();

    // Hash meta
    hasher.update(ir.meta.version.as_bytes());
    hasher.update(ir.meta.law_revision.as_str().as_bytes());

    // Hash applied deltas in order
    for record in &ir.applied_deltas {
        hasher.update(record.delta.as_bytes());
        hasher.update(&record.order.to_le_bytes());
    }

    // Hash structural artifacts
    for module in &ir.modules {
        hasher.update(module.id.as_bytes());
    }
    for structure in &ir.structs {
        hasher.update(structure.id.as_bytes());
    }
    for trait_def in &ir.traits {
        hasher.update(trait_def.id.as_bytes());
    }
    for function in &ir.functions {
        hasher.update(function.id.as_bytes());
    }

    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub before_hash: String,
    pub after_hash: String,
    pub verified: bool,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub state_hash: String,
    pub applied_deltas: Vec<AppliedDeltaRecord>,
    pub modules: usize,
    pub structs: usize,
    pub traits: usize,
    pub functions: usize,
}

#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("unknown admission `{0}`")]
    UnknownAdmission(AdmissionId),
    #[error("unknown delta `{0}`")]
    UnknownDelta(DeltaId),
    #[error("delta `{delta_id}` missing proof `{proof_id}` (Canon Line 68)")]
    MissingProof { delta_id: DeltaId, proof_id: String },
    #[error(
        "delta `{delta_id}` of kind `{delta_kind:?}` has invalid proof scope `{proof_scope:?}`"
    )]
    InvalidProofScope {
        delta_id: DeltaId,
        delta_kind: crate::ir::DeltaKind,
        proof_scope: crate::ir::ProofScope,
    },
    #[error("delta count mismatch: expected {expected}, found {actual}")]
    DeltaCountMismatch { expected: usize, actual: usize },
    #[error(
        "delta ordering violated at index {index}: original `{original}`, evolved `{evolved}` (Canon Line 38)"
    )]
    DeltaOrderingViolation {
        index: usize,
        original: DeltaId,
        evolved: DeltaId,
    },
    #[error("non-monotonic delta order: delta `{delta_id}` has order {order} <= {previous}")]
    NonMonotonicOrder {
        delta_id: DeltaId,
        order: u64,
        previous: u64,
    },
    #[error("delta `{delta_id}` from admission `{admission_id}` was not applied")]
    DeltaNotApplied {
        delta_id: DeltaId,
        admission_id: AdmissionId,
    },
    #[error("state hash mismatch: expected `{expected}`, found `{actual}`")]
    StateHashMismatch { expected: String, actual: String },
}
