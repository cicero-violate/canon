use crate::ir::{CanonicalIr, JudgmentPredicate, Proof, ProofArtifact, ProofScope, Tick};

use super::{DSL_PREDICATE_ID, DSL_PROOF_ID, DSL_TICK_ID};

pub(super) fn ensure_dsl_proof(ir: &mut CanonicalIr) {
    if ir.proofs.iter().any(|proof| proof.id == DSL_PROOF_ID) {
        return;
    }
    ir.proofs.push(Proof {
        id: DSL_PROOF_ID.to_string(),
        invariant: "DSL submissions are structurally lawful.".to_string(),
        scope: ProofScope::Structure,
        evidence: ProofArtifact {
            uri: "canon://dsl/bootstrap".to_string(),
            hash: "dsl-bootstrap".to_string(),
        },
        proof_object_hash: None,
    });
}

pub(super) fn ensure_dsl_predicate(ir: &mut CanonicalIr) {
    if ir
        .judgment_predicates
        .iter()
        .any(|predicate| predicate.id == DSL_PREDICATE_ID)
    {
        return;
    }
    ir.judgment_predicates.push(JudgmentPredicate {
        id: DSL_PREDICATE_ID.to_string(),
        description: "Auto-accept DSL proposals.".to_string(),
    });
}

pub(super) fn ensure_dsl_tick(ir: &mut CanonicalIr) -> Result<(), ()> {
    if ir.ticks.iter().any(|tick| tick.id == DSL_TICK_ID) {
        return Ok(());
    }
    let graph_id = ir
        .tick_graphs
        .first()
        .map(|graph| graph.id.clone())
        .ok_or(())?;
    ir.ticks.push(Tick {
        id: DSL_TICK_ID.to_string(),
        graph: graph_id,
        input_state: vec![],
        output_deltas: vec![],
    });
    Ok(())
}
