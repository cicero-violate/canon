use super::error::{Violation, ViolationDetail};
use super::helpers::{pipeline_stage_allows, proof_scope_allows, Indexes};
use super::rules::CanonRule;
use crate::ir::*;
pub fn check_deltas_top<'a>(
    ir: &'a SystemState,
    idx: &Indexes<'a>,
    violations: &mut Vec<Violation>,
) {
    check_deltas(ir, idx, violations);
    check_struct_history(ir, idx, violations);
    check_admissions(ir, idx, violations);
    check_applied_records(ir, idx, violations);
}
fn check_deltas(ir: &SystemState, idx: &Indexes, violations: &mut Vec<Violation>) {
    for delta in &ir.deltas {
        match &delta.payload {
            Some(ChangePayload::UpdateFunctionAst { function_id, .. }) => {
                if idx.functions.get(function_id.as_str()).is_none() {
                    violations
                        .push(
                            Violation::structured(
                                CanonRule::FunctionAst,
                                delta.id.clone(),
                                ViolationDetail::DeltaReferencesUnknownFunction {
                                    delta: delta.id.clone(),
                                    function_id: function_id.clone(),
                                },
                            ),
                        );
                }
            }
            Some(ChangePayload::UpdateFunctionInputs { function_id, .. })
            | Some(ChangePayload::UpdateFunctionOutputs { function_id, .. }) => {
                if idx.functions.get(function_id.as_str()).is_none() {
                    violations
                        .push(
                            Violation::structured(
                                CanonRule::ExplicitArtifacts,
                                delta.id.clone(),
                                ViolationDetail::DeltaReferencesUnknownFunction {
                                    delta: delta.id.clone(),
                                    function_id: function_id.clone(),
                                },
                            ),
                        );
                }
            }
            Some(ChangePayload::UpdateStructVisibility { struct_id, .. }) => {
                if idx.structs.get(struct_id.as_str()).is_none() {
                    violations
                        .push(
                            Violation::structured(
                                CanonRule::ExplicitArtifacts,
                                delta.id.clone(),
                                ViolationDetail::DeltaReferencesUnknownStruct {
                                    delta: delta.id.clone(),
                                    struct_id: struct_id.clone(),
                                },
                            ),
                        );
                }
            }
            Some(ChangePayload::RemoveField { struct_id, field_name }) => {
                match idx.structs.get(struct_id.as_str()) {
                    Some(structure) => {
                        if !structure.fields.iter().any(|f| f.name == *field_name) {
                            violations
                                .push(
                                    Violation::structured(
                                        CanonRule::ExplicitArtifacts,
                                        delta.id.clone(),
                                        ViolationDetail::DeltaMissingField {
                                            delta: delta.id.clone(),
                                            struct_id: struct_id.clone(),
                                            field: field_name.to_string(),
                                        },
                                    ),
                                );
                        }
                    }
                    None => {
                        violations
                            .push(
                                Violation::structured(
                                    CanonRule::ExplicitArtifacts,
                                    delta.id.clone(),
                                    ViolationDetail::DeltaReferencesUnknownStruct {
                                        delta: delta.id.clone(),
                                        struct_id: struct_id.clone(),
                                    },
                                ),
                            )
                    }
                }
            }
            Some(ChangePayload::RenameArtifact { kind, old_id, .. }) => {
                let exists = match kind.as_str() {
                    "module" => idx.modules.get(old_id.as_str()).is_some(),
                    "struct" => idx.structs.get(old_id.as_str()).is_some(),
                    "function" => idx.functions.get(old_id.as_str()).is_some(),
                    _ => false,
                };
                if !exists {
                    violations
                        .push(
                            Violation::structured(
                                CanonRule::ExplicitArtifacts,
                                delta.id.clone(),
                                ViolationDetail::DeltaReferencesUnknownArtifact {
                                    delta: delta.id.clone(),
                                    kind: kind.clone(),
                                    id: old_id.clone(),
                                },
                            ),
                        );
                }
            }
            _ => {}
        }
        if !delta.append_only {
            violations
                .push(
                    Violation::structured(
                        CanonRule::DeltaAppendOnly,
                        delta.id.clone(),
                        ViolationDetail::DeltaAppendOnlyViolation {
                            delta: delta.id.clone(),
                        },
                    ),
                );
        }
        if !pipeline_stage_allows(delta.stage, delta.kind) {
            violations
                .push(
                    Violation::structured(
                        CanonRule::DeltaPipeline,
                        delta.id.clone(),
                        ViolationDetail::DeltaPipelineViolation {
                            delta: delta.id.clone(),
                        },
                    ),
                );
        }
        let Some(proof) = idx.proofs.get(delta.proof.as_str()) else {
            violations
                .push(
                    Violation::structured(
                        CanonRule::DeltaProofs,
                        delta.id.clone(),
                        ViolationDetail::DeltaMissingProof {
                            delta: delta.id.clone(),
                            proof: delta.proof.clone(),
                        },
                    ),
                );
            continue;
        };
        if !proof_scope_allows(delta.kind, proof.scope) {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ProofScope,
                        delta.id.clone(),
                        ViolationDetail::ProofScopeViolation {
                            delta: delta.id.clone(),
                        },
                    ),
                );
        }
        if let Some(fn_id) = &delta.related_function {
            if idx.functions.get(fn_id.as_str()).is_none() {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::EffectsAreDeltas,
                            delta.id.clone(),
                            ViolationDetail::DeltaReferencesUnknownFunction {
                                delta: delta.id.clone(),
                                function_id: fn_id.clone(),
                            },
                        ),
                    );
            }
        }
    }
}
fn check_struct_history(
    ir: &SystemState,
    idx: &Indexes,
    violations: &mut Vec<Violation>,
) {
    for s in &ir.structs {
        for h in &s.history {
            if idx.deltas.get(h.delta.as_str()).is_none() {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::EffectsAreDeltas,
                            s.id.clone(),
                            ViolationDetail::StructHistoryMissingDelta {
                                struct_id: s.id.clone(),
                                delta: h.delta.clone(),
                            },
                        ),
                    );
            }
        }
    }
}
fn check_admissions(ir: &SystemState, idx: &Indexes, violations: &mut Vec<Violation>) {
    for a in &ir.admissions {
        if idx.ticks.get(a.tick.as_str()).is_none() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::AdmissionBridge,
                        a.id.clone(),
                        ViolationDetail::AdmissionMissingTick {
                            admission: a.id.clone(),
                            tick: a.tick.clone(),
                        },
                    ),
                );
        }
        let Some(j) = idx.judgments.get(a.judgment.as_str()) else {
            violations
                .push(
                    Violation::structured(
                        CanonRule::AdmissionBridge,
                        a.id.clone(),
                        ViolationDetail::AdmissionMissingJudgment {
                            admission: a.id.clone(),
                            judgment: a.judgment.clone(),
                        },
                    ),
                );
            continue;
        };
        if j.decision != JudgmentDecision::Accept {
            violations
                .push(
                    Violation::structured(
                        CanonRule::AdmissionBridge,
                        a.id.clone(),
                        ViolationDetail::AdmissionNotAccepted {
                            admission: a.id.clone(),
                        },
                    ),
                );
        }
        for d in &a.delta_ids {
            if idx.deltas.get(d.as_str()).is_none() {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::AdmissionBridge,
                            a.id.clone(),
                            ViolationDetail::AdmissionMissingDelta {
                                admission: a.id.clone(),
                                delta: d.clone(),
                            },
                        ),
                    );
            }
        }
    }
}
fn check_applied_records(
    ir: &SystemState,
    idx: &Indexes,
    violations: &mut Vec<Violation>,
) {
    let mut prev_order = None;
    for r in &ir.applied_deltas {
        if idx.admissions.get(r.admission.as_str()).is_none() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::AdmissionBridge,
                        r.id.clone(),
                        ViolationDetail::AppliedMissingAdmission {
                            applied: r.id.clone(),
                            admission: r.admission.clone(),
                        },
                    ),
                );
        }
        if idx.deltas.get(r.delta.as_str()).is_none() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::AdmissionBridge,
                        r.id.clone(),
                        ViolationDetail::AppliedMissingDelta {
                            applied: r.id.clone(),
                            delta: r.delta.clone(),
                        },
                    ),
                );
        }
        if let Some(prev) = prev_order {
            if r.order < prev {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::AdmissionBridge,
                            r.id.clone(),
                            ViolationDetail::AppliedOrderViolation {
                                applied: r.id.clone(),
                            },
                        ),
                    );
            }
        }
        prev_order = Some(r.order);
    }
}
