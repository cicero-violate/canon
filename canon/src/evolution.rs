use std::collections::HashMap;

use blake3::hash;
use canon_kernel::{
    self as kernel, Admission as KernelAdmission, Delta as KernelDelta,
    DeltaRecord as KernelDeltaRecord, InvariantRegistry as KernelInvariantRegistry,
    Judgment as KernelJudgment, JudgmentPredicate, ProofArtifact as KernelProofArtifact,
    ProofId as KernelProofId, ProofRegistry as KernelProofRegistry, ProofScope as KernelProofScope,
    StateLog,
};
use serde_json;
use thiserror::Error;

use crate::ir::{
    AdmissionId, AppliedDeltaRecord, CanonicalIr, CanonicalMeta, Delta, DeltaId, DeltaKind,
    DeltaPayload, ImplBlock, JudgmentDecision, Module, ModuleEdge, Proof, Struct, Trait,
    TraitFunction, Visibility,
};

use crate::runtime::delta_verifier::{DeltaVerifier, VerificationError};

pub fn apply_deltas(
    ir: &CanonicalIr,
    admission_ids: &[AdmissionId],
) -> Result<CanonicalIr, EvolutionError> {
    // Create snapshot before application (for rollback)
    let snapshot = DeltaVerifier::create_snapshot(ir);

    let judgments: HashMap<_, _> = ir.judgments.iter().map(|j| (j.id.as_str(), j)).collect();
    let deltas: HashMap<_, _> = ir.deltas.iter().map(|d| (d.id.as_str(), d)).collect();
    let admissions: HashMap<_, _> = ir.admissions.iter().map(|a| (a.id.as_str(), a)).collect();

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
            return Err(EvolutionError::JudgmentNotAccepted(
                admission.judgment.clone(),
            ));
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
            apply_structural_delta(&mut next, delta)?;
            next.applied_deltas.push(AppliedDeltaRecord {
                id: format!("{}#{}", admission.id, order),
                admission: admission.id.clone(),
                delta: delta_id.clone(),
                order,
            });
            order += 1;
        }
    }

    // Verify application was correct (Canon Lines 68-69)
    match DeltaVerifier::verify_application(ir, &next, admission_ids) {
        Ok(_) => Ok(next),
        Err(verification_err) => {
            // Verification failed - rollback to snapshot
            eprintln!("Delta verification failed: {}", verification_err);
            eprintln!(
                "Rolling back to snapshot with hash: {}",
                snapshot.state_hash
            );
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

fn build_proof_registry(proofs: &[Proof]) -> KernelProofRegistry {
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

fn build_invariant_registry() -> KernelInvariantRegistry {
    let mut registry = KernelInvariantRegistry::new();
    registry.allow_scope(KernelProofScope::Structure);
    registry.allow_scope(KernelProofScope::Execution);
    registry.allow_scope(KernelProofScope::Meta);
    registry
}

fn build_state_log(
    ir: &CanonicalIr,
    deltas: &HashMap<&str, &Delta>,
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
    Ok(StateLog::from_records(
        initial_state_seed(&ir.meta),
        records,
    ))
}

fn build_kernel_admission(
    admission: &crate::ir::DeltaAdmission,
    deltas: &HashMap<&str, &Delta>,
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
    KernelAdmission::new(
        admission.id.clone(),
        admission.judgment.clone(),
        kernel_deltas,
    )
    .map_err(|err| EvolutionError::Kernel(err.to_string()))
}

fn kernel_delta(delta: &Delta) -> Result<KernelDelta, EvolutionError> {
    Ok(KernelDelta {
        id: delta.id.clone(),
        proof_id: KernelProofId::new(delta.proof.clone()),
        payload_hash: payload_hash(delta)?,
    })
}

fn payload_hash(delta: &Delta) -> Result<String, EvolutionError> {
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

fn apply_structural_delta(ir: &mut CanonicalIr, delta: &Delta) -> Result<(), EvolutionError> {
    match &delta.payload {
        Some(DeltaPayload::AddModule {
            module_id,
            name,
            visibility,
            description,
        }) => {
            if ir.modules.iter().any(|m| m.id == *module_id) {
                return Err(EvolutionError::DuplicateArtifact(module_id.clone()));
            }
            ir.modules.push(Module {
                id: module_id.clone(),
                name: name.clone(),
                visibility: *visibility,
                description: description.clone(),
                files: Vec::new(),
                file_edges: Vec::new(),
            });
        }
        Some(DeltaPayload::AddStruct {
            module,
            struct_id,
            name,
        }) => {
            if ir.structs.iter().any(|s| s.id == *struct_id) {
                return Err(EvolutionError::DuplicateArtifact(struct_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ir.structs.push(Struct {
                id: struct_id.clone(),
                name: name.clone(),
                module: module.clone(),
                visibility: Visibility::Private,
                fields: vec![],
                history: vec![],
            });
        }
        Some(DeltaPayload::AddField { struct_id, field }) => {
            let structure = ir
                .structs
                .iter_mut()
                .find(|s| s.id == *struct_id)
                .ok_or_else(|| EvolutionError::UnknownStruct(struct_id.clone()))?;
            if structure.fields.iter().any(|f| f.name == field.name) {
                return Err(EvolutionError::DuplicateArtifact(
                    field.name.as_str().to_string(),
                ));
            }
            structure.fields.push(field.clone());
        }
        Some(DeltaPayload::AddTrait {
            module,
            trait_id,
            name,
        }) => {
            if ir.traits.iter().any(|t| t.id == *trait_id) {
                return Err(EvolutionError::DuplicateArtifact(trait_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ir.traits.push(Trait {
                id: trait_id.clone(),
                name: name.clone(),
                module: module.clone(),
                visibility: Visibility::Private,
                functions: vec![],
            });
        }
        Some(DeltaPayload::AddTraitFunction { trait_id, function }) => {
            let target_trait = ir
                .traits
                .iter_mut()
                .find(|t| t.id == *trait_id)
                .ok_or_else(|| EvolutionError::UnknownTrait(trait_id.clone()))?;
            if target_trait.functions.iter().any(|f| f.id == function.id) {
                return Err(EvolutionError::DuplicateArtifact(function.id.clone()));
            }
            target_trait.functions.push(function.clone());
        }
        Some(DeltaPayload::AddImpl {
            module,
            impl_id,
            struct_id,
            trait_id,
        }) => {
            if ir.impl_blocks.iter().any(|i| i.id == *impl_id) {
                return Err(EvolutionError::DuplicateArtifact(impl_id.clone()));
            }
            ensure_module_exists(ir, module)?;
            ensure_struct_exists(ir, struct_id)?;
            ensure_trait_exists(ir, trait_id)?;
            ir.impl_blocks.push(ImplBlock {
                id: impl_id.clone(),
                module: module.clone(),
                struct_id: struct_id.clone(),
                trait_id: trait_id.clone(),
                functions: vec![],
            });
        }
        Some(DeltaPayload::AddFunction {
            function_id,
            impl_id,
            signature,
        }) => {
            if ir.functions.iter().any(|f| f.id == *function_id) {
                return Err(EvolutionError::DuplicateArtifact(function_id.clone()));
            }
            let block_index = ir
                .impl_blocks
                .iter()
                .position(|i| i.id == *impl_id)
                .ok_or_else(|| EvolutionError::UnknownImpl(impl_id.clone()))?;
            let trait_id = ir.impl_blocks[block_index].trait_id.clone();
            let module = ir.impl_blocks[block_index].module.clone();
            ensure_trait_function_exists(ir, &trait_id, &signature.trait_function)?;
            ir.functions.push(crate::ir::Function {
                id: function_id.clone(),
                name: signature.name.clone(),
                module,
                impl_id: impl_id.clone(),
                trait_function: signature.trait_function.clone(),
                visibility: signature.visibility,
                inputs: signature.inputs.clone(),
                outputs: signature.outputs.clone(),
                deltas: vec![],
                contract: crate::ir::FunctionContract {
                    total: true,
                    deterministic: true,
                    explicit_inputs: true,
                    explicit_outputs: true,
                    effects_are_deltas: true,
                },
                metadata: crate::ir::FunctionMetadata::default(),
            });
            ir.impl_blocks[block_index]
                .functions
                .push(crate::ir::ImplFunctionBinding {
                    trait_fn: signature.trait_function.clone(),
                    function: function_id.clone(),
                });
        }
        Some(DeltaPayload::AddModuleEdge {
            from,
            to,
            rationale,
        }) => {
            ensure_module_exists(ir, from)?;
            ensure_module_exists(ir, to)?;
            if ir
                .module_edges
                .iter()
                .any(|edge| &edge.source == from && &edge.target == to)
            {
                return Err(EvolutionError::DuplicateArtifact(format!("{from}->{to}")));
            }
            ir.module_edges.push(ModuleEdge {
                source: from.clone(),
                target: to.clone(),
                rationale: rationale.clone(),
                imported_types: Vec::new(),
            });
        }
        Some(DeltaPayload::AddCallEdge { caller, callee }) => {
            ensure_function_exists(ir, caller)?;
            ensure_function_exists(ir, callee)?;
            if ir
                .call_edges
                .iter()
                .any(|edge| edge.caller == *caller && edge.callee == *callee)
            {
                return Err(EvolutionError::DuplicateArtifact(format!(
                    "{caller}->{callee}"
                )));
            }
            ir.call_edges.push(crate::ir::CallEdge {
                id: format!("call:{}:{}", caller, callee),
                caller: caller.clone(),
                callee: callee.clone(),
                rationale: "delta-applied".to_string(),
            });
        }
        Some(DeltaPayload::AttachExecutionEvent {
            execution_id,
            event,
        }) => {
            let record = ir
                .executions
                .iter_mut()
                .find(|e| e.id == *execution_id)
                .ok_or_else(|| EvolutionError::UnknownExecution(execution_id.clone()))?;
            record.events.push(event.clone());
        }
        _ => {}
        Some(DeltaPayload::UpdateFunctionAst { function_id, ast }) => {
            let function = ir
                .functions
                .iter_mut()
                .find(|f| f.id == *function_id)
                .ok_or_else(|| EvolutionError::UnknownFunction(function_id.clone()))?;
            function.metadata.ast = Some(ast.clone());
        }
    }
    Ok(())
}

fn ensure_module_exists(ir: &CanonicalIr, module: &str) -> Result<(), EvolutionError> {
    if ir.modules.iter().any(|m| m.id == module) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownModule(module.to_string()))
    }
}

fn ensure_struct_exists(ir: &CanonicalIr, struct_id: &str) -> Result<(), EvolutionError> {
    if ir.structs.iter().any(|s| s.id == struct_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownStruct(struct_id.to_string()))
    }
}

fn ensure_trait_exists(ir: &CanonicalIr, trait_id: &str) -> Result<(), EvolutionError> {
    if ir.traits.iter().any(|t| t.id == trait_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownTrait(trait_id.to_string()))
    }
}

fn ensure_trait_function_exists(
    ir: &CanonicalIr,
    trait_id: &str,
    trait_fn: &str,
) -> Result<(), EvolutionError> {
    let tr = ir
        .traits
        .iter()
        .find(|t| t.id == trait_id)
        .ok_or_else(|| EvolutionError::UnknownTrait(trait_id.to_string()))?;
    if tr.functions.iter().any(|f| f.id == trait_fn) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownTraitFunction(trait_fn.to_string()))
    }
}

fn ensure_function_exists(ir: &CanonicalIr, function_id: &str) -> Result<(), EvolutionError> {
    if ir.functions.iter().any(|f| f.id == function_id) {
        Ok(())
    } else {
        Err(EvolutionError::UnknownFunction(function_id.to_string()))
    }
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
    #[error("artifact `{0}` already exists")]
    DuplicateArtifact(String),
    #[error("failed to hash delta payload: {0}")]
    PayloadHash(String),
    #[error("kernel error: {0}")]
    Kernel(String),
    #[error("delta verification failed: {0}")]
    VerificationFailed(#[from] VerificationError),
}
