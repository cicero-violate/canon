use canon::{PatchApplier, PatchGate, PatchMetadata, PatchQueue};

fn make_metadata() -> PatchMetadata {
    PatchMetadata {
        agent_id: "agent".into(),
        prompt: "Add feature".into(),
        timestamp: "2025-01-01T00:00:00Z".into(),
    }
}

#[test]
fn rejects_unstructured_patch() {
    let mut queue = PatchQueue::new();
    assert!(
        queue
            .enqueue("invalid patch".into(), make_metadata())
            .is_err()
    );
}

#[test]
fn accepts_structured_patch_and_runs_gate() {
    let mut queue = PatchQueue::new();
    let diff: String = "*** Begin Patch\n*** End Patch".into();
    let proposal = queue
        .enqueue(diff.clone(), make_metadata())
        .unwrap()
        .clone();
    let mut gate = PatchGate::new();
    match gate.evaluate(proposal) {
        canon::PatchDecision::Accepted { verified } => {
            let applier = PatchApplier::new(|_| true);
            applier.apply(&verified).unwrap();
        }
        canon::PatchDecision::Rejected { reason, .. } => panic!("Rejected: {reason}"),
    }
}
