use crate::ir::*;
use super::error::Violation;
use super::helpers::{Indexes, pipeline_stage_allows, proof_scope_allows};
use super::rules::CanonRule;

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    check_deltas(ir, idx, violations);
    check_struct_history(ir, idx, violations);
    check_admissions(ir, idx, violations);
    check_applied_records(ir, idx, violations);
}

fn check_deltas(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for delta in &ir.deltas {
        // Gap 2: validate UpdateFunctionAst references an existing function
        if let Some(DeltaPayload::UpdateFunctionAst { function_id, .. }) = &delta.payload {
            if idx.functions.get(function_id.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::FunctionAst,
                    format!("delta `{}` references unknown function `{function_id}` for AST update", delta.id),
                ));
            }
        }
        if !delta.append_only {
            violations.push(Violation::new(CanonRule::DeltaAppendOnly, format!("delta `{}` must be append-only", delta.id)));
        }
        if !pipeline_stage_allows(delta.stage, delta.kind) {
            violations.push(Violation::new(CanonRule::DeltaPipeline, format!("delta `{}` of kind `{:?}` is not legal in stage `{:?}`", delta.id, delta.kind, delta.stage)));
        }
        let Some(proof) = idx.proofs.get(delta.proof.as_str()) else {
            violations.push(Violation::new(CanonRule::DeltaProofs, format!("delta `{}` requires proof `{}` but it was not found", delta.id, delta.proof)));
            continue;
        };
        if !proof_scope_allows(delta.kind, proof.scope) {
            violations.push(Violation::new(CanonRule::ProofScope, format!("delta `{}` of kind `{:?}` cannot carry proof scope `{:?}`", delta.id, delta.kind, proof.scope)));
        }
        if let Some(fn_id) = &delta.related_function {
            if idx.functions.get(fn_id.as_str()).is_none() {
                violations.push(Violation::new(CanonRule::EffectsAreDeltas, format!("delta `{}` references missing function `{fn_id}`", delta.id)));
            }
        }
    }
}

fn check_struct_history(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for s in &ir.structs {
        for h in &s.history {
            if idx.deltas.get(h.delta.as_str()).is_none() {
                violations.push(Violation::new(CanonRule::EffectsAreDeltas, format!("struct `{}` references missing delta `{}` in history", s.name, h.delta)));
            }
        }
    }
}

fn check_admissions(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for a in &ir.admissions {
        if idx.ticks.get(a.tick.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::AdmissionBridge, format!("admission `{}` references missing tick `{}`", a.id, a.tick)));
        }
        let Some(j) = idx.judgments.get(a.judgment.as_str()) else {
            violations.push(Violation::new(CanonRule::AdmissionBridge, format!("admission `{}` references missing judgment `{}`", a.id, a.judgment)));
            continue;
        };
        if j.decision != JudgmentDecision::Accept {
            violations.push(Violation::new(CanonRule::AdmissionBridge, format!("admission `{}` must reference accepted judgment", a.id)));
        }
        for d in &a.delta_ids {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::new(CanonRule::AdmissionBridge, format!("admission `{}` lists unknown delta `{d}`", a.id)));
            }
        }
    }
}

fn check_applied_records(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let mut prev_order = None;
    for r in &ir.applied_deltas {
        if idx.admissions.get(r.admission.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::AdmissionBridge, format!("applied delta `{}` references missing admission `{}`", r.id, r.admission)));
        }
        if idx.deltas.get(r.delta.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::AdmissionBridge, format!("applied delta `{}` references unknown delta `{}`", r.id, r.delta)));
        }
        if let Some(prev) = prev_order {
            if r.order < prev {
                violations.push(Violation::new(CanonRule::AdmissionBridge, format!("applied deltas must be non-decreasing but `{}` < `{prev}`", r.order)));
            }
        }
        prev_order = Some(r.order);
    }
}
