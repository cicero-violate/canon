use canon::proof_object::Expression;
use canon::{ProofObject, evaluate_proof_object};

#[test]
fn proof_object_accepts_true_equal_true() {
    let object = ProofObject {
        id: "proof1".into(),
        expression: Expression::Equals(
            Box::new(Expression::Atom("true".into())),
            Box::new(Expression::Atom("true".into())),
        ),
    };
    let result = evaluate_proof_object(&object);
    assert!(result.valid);
}

#[test]
fn proof_object_rejects_false_equal_true() {
    let object = ProofObject {
        id: "proof2".into(),
        expression: Expression::Equals(
            Box::new(Expression::Atom("false".into())),
            Box::new(Expression::Atom("true".into())),
        ),
    };
    let result = evaluate_proof_object(&object);
    assert!(!result.valid);
}
