use canon::ir::DeltaPayload;
use canon::{CanonicalIr, apply_admitted_deltas, wrap_execution_events_as_deltas};
use std::path::PathBuf;

fn load_fixture() -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("valid_ir.json");
    let data = std::fs::read(path).expect("fixture");
    serde_json::from_slice(&data).expect("canonical ir")
}

#[test]
fn apply_deltas_appends_history() {
    let mut ir = load_fixture();
    ir.applied_deltas.clear();
    let next = apply_admitted_deltas(&ir, &["admission.core".to_string()]).expect("apply");
    assert_eq!(next.applied_deltas.len(), 2);
    assert_eq!(next.applied_deltas[1].delta, "delta.add_field");
}

#[test]
fn execution_events_become_observe_deltas() {
    let ir = load_fixture();
    let exec = ir
        .executions
        .iter()
        .find(|e| e.id == "exec.tick1")
        .expect("execution");
    let deltas = wrap_execution_events_as_deltas(exec, "proof.delta");
    assert!(!deltas.is_empty());
    assert!(deltas[0].proof == "proof.delta");
    assert!(
        deltas[0]
            .payload
            .as_ref()
            .is_some_and(|p| matches!(p, DeltaPayload::AttachExecutionEvent { .. }))
    );
}
