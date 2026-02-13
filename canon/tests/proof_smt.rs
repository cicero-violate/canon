use canon::ir::{
    CanonicalIr, Delta, DeltaKind, Function, FunctionContract, FunctionMetadata, PipelineStage,
    Postcondition, TypeKind, TypeRef, ValuePort, Visibility, Word,
};
use canon::proof::smt_bridge::{SmtError, attach_function_proofs, verify_function_postconditions};
use canon::runtime::value::{ScalarValue, Value};
use canon::runtime::{BinOp, Expr, FunctionAst, OutputExpr};

fn load_fixture(name: &str) -> CanonicalIr {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name);
    let data = std::fs::read(path).expect("fixture must exist");
    serde_json::from_slice(&data).expect("valid fixture")
}

fn sample_function(expr: Expr) -> Function {
    let output_name = Word::new("Result").unwrap();
    let ast = FunctionAst {
        outputs: vec![OutputExpr {
            name: output_name.as_str().to_string(),
            expr,
        }],
    };

    Function {
        id: "fn.test".into(),
        name: Word::new("Test").unwrap(),
        module: "module.math".into(),
        impl_id: "impl.math".into(),
        trait_function: "trait_fn.math.test".into(),
        visibility: Visibility::Public,
        doc: None,
        lifetime_params: Vec::new(),
        receiver: canon::ir::Receiver::None,
        is_async: false,
        is_unsafe: false,
        generics: Vec::new(),
        where_clauses: Vec::new(),
        inputs: vec![ValuePort {
            name: Word::new("Input").unwrap(),
            ty: TypeRef {
                name: Word::new("ScalarValue").unwrap(),
                kind: TypeKind::Scalar,
                params: Vec::new(),
                ref_kind: Default::default(),
                lifetime: None,
            },
        }],
        outputs: vec![ValuePort {
            name: output_name.clone(),
            ty: TypeRef {
                name: Word::new("ScalarValue").unwrap(),
                kind: TypeKind::Scalar,
                params: Vec::new(),
                ref_kind: Default::default(),
                lifetime: None,
            },
        }],
        deltas: vec![],
        contract: FunctionContract {
            total: true,
            deterministic: true,
            explicit_inputs: true,
            explicit_outputs: true,
            effects_are_deltas: true,
        },
        metadata: FunctionMetadata {
            bytecode_b64: None,
            ast: Some(serde_json::to_value(&ast).unwrap()),
            postconditions: vec![Postcondition::NonNegative {
                output: output_name,
            }],
        },
    }
}

#[test]
fn smt_proves_square_is_non_negative() {
    let expr = Expr::BinOp {
        left: Box::new(Expr::Input {
            name: "Input".into(),
        }),
        op: BinOp::Mul,
        right: Box::new(Expr::Input {
            name: "Input".into(),
        }),
    };
    let function = sample_function(expr);
    let cert = verify_function_postconditions(&function)
        .expect("verification succeeds")
        .expect("certificate expected");
    assert!(!cert.proof_hash.is_empty());
}

#[test]
fn smt_detects_counterexample_when_condition_fails() {
    let expr = Expr::Input {
        name: "Input".into(),
    };
    let function = sample_function(expr);
    let err = verify_function_postconditions(&function).unwrap_err();
    match err {
        SmtError::Counterexample { .. } => {}
        other => panic!("expected counterexample, got {other:?}"),
    }
}

#[test]
fn attach_function_proofs_sets_delta_hash() {
    let expr = Expr::BinOp {
        left: Box::new(Expr::Input {
            name: "Input".into(),
        }),
        op: BinOp::Mul,
        right: Box::new(Expr::Input {
            name: "Input".into(),
        }),
    };
    let mut ir = load_fixture("valid_ir.json");
    let function = sample_function(expr);
    let function_id = function.id.clone();
    ir.functions.push(function);

    let delta = Delta {
        id: "delta.test.fn".into(),
        kind: DeltaKind::Structure,
        stage: PipelineStage::Decide,
        append_only: true,
        proof: ir.proofs.first().expect("proof").id.clone(),
        description: "test delta".into(),
        related_function: Some(function_id),
        payload: None,
        proof_object_hash: None,
    };
    ir.deltas.push(delta);

    attach_function_proofs(&mut ir).expect("attach proofs succeeds");
    assert!(
        ir.deltas
            .last()
            .unwrap()
            .proof_object_hash
            .as_ref()
            .is_some(),
        "delta must include proof hash"
    );
}
