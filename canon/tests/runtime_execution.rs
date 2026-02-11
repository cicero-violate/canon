//! Integration tests for runtime execution (Phase 1).

use std::collections::BTreeMap;
use std::path::PathBuf;

use canon::CanonicalIr;
use canon::runtime::value::ScalarValue;
use canon::runtime::{
    ExecutionContext, ExecutorError, FunctionExecutor, TickExecutionMode, TickExecutor, Value,
};

#[test]
fn test_execution_context_no_recursion() {
    let inputs = BTreeMap::new();
    let mut ctx = ExecutionContext::new(inputs);

    // Push first call
    assert!(ctx.push_call("fn1".into()).is_ok());

    // Try to call same function again (recursion)
    let result = ctx.push_call("fn1".into());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("recursion"));
}

#[test]
fn test_execution_context_no_mutation() {
    let inputs = BTreeMap::new();
    let mut ctx = ExecutionContext::new(inputs);

    // Bind once
    assert!(ctx.bind("x", Value::Scalar(ScalarValue::I32(42))).is_ok());

    // Try to bind again (mutation not allowed)
    let result = ctx.bind("x", Value::Scalar(ScalarValue::I32(99)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn test_execution_context_delta_append_only() {
    let inputs = BTreeMap::new();
    let mut ctx = ExecutionContext::new(inputs);

    // Emit deltas
    ctx.emit_delta(canon::runtime::value::DeltaValue {
        delta_id: "delta1".into(),
        payload_hash: "hash1".into(),
    });
    ctx.emit_delta(canon::runtime::value::DeltaValue {
        delta_id: "delta2".into(),
        payload_hash: "hash2".into(),
    });

    // Deltas are append-only
    assert_eq!(ctx.deltas().len(), 2);
    assert_eq!(ctx.deltas()[0].delta_id, "delta1");
    assert_eq!(ctx.deltas()[1].delta_id, "delta2");
}

#[test]
fn test_value_type_checking() {
    let scalar = Value::Scalar(ScalarValue::I32(42));
    assert!(scalar.as_scalar().is_ok());
    assert!(scalar.as_struct().is_err());
    assert!(scalar.as_delta().is_err());
}

fn load_fixture(name: &str) -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    let data = std::fs::read(path).expect("fixture must exist");
    serde_json::from_slice(&data).expect("fixture must be valid CanonicalIr")
}

#[test]
fn test_simple_add_fixture_executes() {
    let ir = load_fixture("simple_add.json");
    let executor = FunctionExecutor::new(&ir);
    let mut ctx = ExecutionContext::new(BTreeMap::new());
    let mut inputs = BTreeMap::new();
    inputs.insert("Lhs".into(), Value::Scalar(ScalarValue::I32(2)));
    inputs.insert("Rhs".into(), Value::Scalar(ScalarValue::I32(3)));

    let outputs = executor
        .execute_by_id(&"fn.add".to_string(), inputs, &mut ctx)
        .expect("execution should succeed");
    let sum = outputs.get("Sum").expect("sum output expected");
    assert_eq!(
        sum,
        &Value::Scalar(ScalarValue::I32(5)),
        "bytecode interpreter must perform real addition"
    );
}

#[test]
fn test_recursive_fixture_errors() {
    let ir = load_fixture("recursive_call.json");
    let executor = FunctionExecutor::new(&ir);
    let mut ctx = ExecutionContext::new(BTreeMap::new());
    let mut inputs = BTreeMap::new();
    inputs.insert("Value".into(), Value::Scalar(ScalarValue::I32(1)));

    let result = executor.execute_by_id(&"fn.recurse".to_string(), inputs, &mut ctx);
    assert!(
        matches!(result, Err(ExecutorError::Interpreter(_))),
        "recursion must be detected"
    );
}

#[test]
fn test_delta_emit_fixture_emits_delta() {
    let ir = load_fixture("delta_emit.json");
    let executor = FunctionExecutor::new(&ir);
    let mut ctx = ExecutionContext::new(BTreeMap::new());
    let outputs = executor
        .execute_by_id(&"fn.emit".to_string(), BTreeMap::new(), &mut ctx)
        .expect("delta emission succeeds");
    assert_eq!(
        outputs.get("Ack"),
        Some(&Value::Unit),
        "delta emission returns unit ack"
    );
    assert_eq!(ctx.deltas().len(), 1);
    assert_eq!(ctx.deltas()[0].delta_id, "delta.effect");
}

#[test]
fn test_multi_function_fixture_executes_pipeline() {
    let ir = load_fixture("multi_function.json");
    let executor = FunctionExecutor::new(&ir);
    let mut ctx = ExecutionContext::new(BTreeMap::new());
    let mut inputs = BTreeMap::new();
    inputs.insert("Seed".into(), Value::Scalar(ScalarValue::I32(5)));

    let outputs = executor
        .execute_by_id(&"fn.pipeline".to_string(), inputs, &mut ctx)
        .expect("pipeline executes");
    let total = outputs.get("Total").expect("total output");
    assert_eq!(
        total,
        &Value::Scalar(ScalarValue::I32(20)),
        "pipeline should double twice"
    );
}

#[test]
fn test_tick_executor_parallel_matches() {
    let ir = load_fixture("tick_noop.json");
    let executor = TickExecutor::new(&ir);
    let parallel = executor
        .execute_tick_with_mode("tick.noop", TickExecutionMode::ParallelVerified)
        .expect("parallel execution should succeed");
    assert!(
        parallel.parallel_duration.is_some(),
        "parallel mode must report timing"
    );
    assert!(
        parallel.sequential_duration > std::time::Duration::from_nanos(0),
        "sequential timing captured"
    );
}
