use std::path::PathBuf;

use canon::ir::JudgmentDecision;
use canon::ir::{DeltaKind, ProofScope};
use canon::{validate_ir, CanonRule, CanonicalIr, PipelineStage};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name)
}

fn load_fixture(name: &str) -> CanonicalIr {
    let data = std::fs::read(fixture_path(name)).expect("fixture must exist");
    serde_json::from_slice(&data).expect("fixture must deserialize")
}

#[test]
fn valid_fixture_passes_validation() {
    let ir = load_fixture("valid_ir.json");
    validate_ir(&ir).expect("valid fixture must satisfy Canon");
}

#[test]
fn missing_proof_fixture_fails_validation() {
    let ir = load_fixture("invalid_missing_proof.json");
    let err = validate_ir(&ir).expect_err("fixture lacks proof and must fail");
    assert!(err.violations().iter().any(|v| v.rule() == CanonRule::DeltaProofs), "expected a proof violation, got {:?}", err.violations());
}

#[test]
fn delta_stage_rules_enforced() {
    let mut ir = load_fixture("valid_ir.json");
    ir.deltas[0].stage = PipelineStage::Decide;
    ir.deltas[0].kind = DeltaKind::State;
    let err = validate_ir(&ir).expect_err("decide stage cannot emit state deltas");
    assert!(err.violations().iter().any(|v| v.rule() == CanonRule::DeltaPipeline));
}

#[test]
fn proof_scope_rules_enforced() {
    let mut ir = load_fixture("valid_ir.json");
    ir.deltas[0].kind = DeltaKind::Structure;
    ir.proofs[0].scope = ProofScope::Execution;
    let err = validate_ir(&ir).expect_err("structure delta cannot use execution proof");
    assert!(err.violations().iter().any(|v| v.rule() == CanonRule::ProofScope));
}

#[test]
fn version_contract_requires_law_scoped_proof() {
    let mut ir = load_fixture("valid_ir.json");
    ir.version_contract.migration_proofs = vec!["proof.delta".to_string()];
    let err = validate_ir(&ir).expect_err("version migration must reference law-scoped proofs");
    assert!(err.violations().iter().any(|v| v.rule() == CanonRule::VersionEvolution));
}

#[test]
fn plan_requires_accepted_judgment() {
    let mut ir = load_fixture("valid_ir.json");
    ir.judgments[0].decision = JudgmentDecision::Reject;
    let err = validate_ir(&ir).expect_err("plan must reference accepted judgment");
    assert!(err.violations().iter().any(|v| v.rule() == CanonRule::PlanArtifacts));
}
