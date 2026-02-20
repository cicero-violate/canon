//! SMT bridge using Z3 to prove simple function postconditions.
//!
//! Functions declare postconditions in metadata, which we discharge by encoding
//! their AST into Z3 constraints and checking for counterexamples.
use std::collections::HashMap;
use blake3::Hasher;
use serde_json::Value as JsonValue;
use thiserror::Error;
use z3::{
    ast::{Bool, Int},
    Config, Context, Model, SatResult, Solver,
};
use crate::ir::{SystemState, Function, FunctionId, Postcondition};
use crate::runtime::ast::{Expr, FunctionAst};
use crate::runtime::value::{ScalarValue, Value};
const INPUT_BOUND: i64 = 1_000;
/// Result of a successful SMT proof.
#[derive(Debug, Clone)]
pub struct SmtCertificate {
    pub function_id: FunctionId,
    pub proof_hash: String,
}
/// SMT verification errors.
#[derive(Debug, Error)]
pub enum SmtError {
    #[error("function `{function}` missing AST metadata for SMT verification")]
    MissingAst { function: FunctionId },
    #[error(
        "function `{function}` undefined output `{output}` referenced in postcondition"
    )]
    UnknownOutput { function: FunctionId, output: String },
    #[error("function `{function}` uses unsupported AST construct: {reason}")]
    UnsupportedAst { function: FunctionId, reason: String },
    #[error(
        "Z3 returned counterexample for `{function}` violating `{condition}`: {model}"
    )]
    Counterexample { function: FunctionId, condition: String, model: String },
    #[error("Z3 returned unknown for `{function}`")]
    Unknown { function: FunctionId },
    #[error("Z3 error for `{function}`: {message}")]
    Solver { function: FunctionId, message: String },
    #[error("AST decode error for `{function}`: {message}")]
    AstDecode { function: FunctionId, message: String },
}
/// Attach SMT proof hashes to deltas referencing verified functions.
pub fn attach_function_proofs(ir: &mut SystemState) -> Result<(), SmtError> {
    let mut certs = HashMap::new();
    for function in &ir.functions {
        if let Some(cert) = verify_function_postconditions(function)? {
            certs.insert(function.id.clone(), cert.proof_hash);
        }
    }
    for delta in &mut ir.deltas {
        if let Some(function_id) = &delta.related_function {
            if let Some(hash) = certs.get(function_id) {
                delta.proof_object_hash = Some(hash.clone());
            }
        }
    }
    Ok(())
}
/// Verify postconditions associated with a function.
pub fn verify_function_postconditions(
    function: &Function,
) -> Result<Option<SmtCertificate>, SmtError> {
    if function.metadata.postconditions.is_empty() {
        return Ok(None);
    }
    let ast_json: &JsonValue = function
        .metadata
        .ast
        .as_ref()
        .ok_or_else(|| SmtError::MissingAst {
            function: function.id.clone(),
        })?;
    let ast: FunctionAst = serde_json::from_value(ast_json.clone())
        .map_err(|err| SmtError::AstDecode {
            function: function.id.clone(),
            message: err.to_string(),
        })?;
    let mut cfg = Config::new();
    cfg.set_proof_generation(true);
    let ctx = Context::new(&cfg);
    let mut solver = Solver::new(&ctx);
    let vars = build_input_vars(&ctx, function);
    apply_input_bounds(&ctx, &mut solver, &vars);
    let outputs = build_output_exprs(&ctx, &vars, &ast, function)?;
    let mut hasher = Hasher::new();
    hasher.update(function.id.as_bytes());
    for condition in &function.metadata.postconditions {
        solver.push();
        let violation = build_violation(condition, &outputs, &ctx, function)?;
        solver.assert(&violation);
        match solver.check() {
            SatResult::Sat => {
                let model = solver
                    .get_model()
                    .map(|m: Model| m.to_string())
                    .unwrap_or_else(|| "<unknown>".into());
                return Err(SmtError::Counterexample {
                    function: function.id.clone(),
                    condition: describe_condition(condition),
                    model,
                });
            }
            SatResult::Unknown => {
                return Err(SmtError::Unknown {
                    function: function.id.clone(),
                });
            }
            SatResult::Unsat => {
                if let Some(proof) = solver.get_proof() {
                    hasher.update(format!("{proof:?}").as_bytes());
                }
            }
        }
        solver.pop(1);
    }
    let proof_hash = hasher.finalize().to_hex().to_string();
    Ok(
        Some(SmtCertificate {
            function_id: function.id.clone(),
            proof_hash,
        }),
    )
}
fn describe_condition(condition: &Postcondition) -> String {
    match condition {
        Postcondition::NonNegative { output } => {
            format!("non_negative({})", output.as_str())
        }
    }
}
fn build_violation<'ctx>(
    condition: &Postcondition,
    outputs: &HashMap<String, Int<'ctx>>,
    ctx: &'ctx Context,
    function: &Function,
) -> Result<Bool<'ctx>, SmtError> {
    match condition {
        Postcondition::NonNegative { output } => {
            let output_expr = outputs
                .get(output.as_str())
                .ok_or_else(|| SmtError::UnknownOutput {
                    function: function.id.clone(),
                    output: output.as_str().to_string(),
                })?;
            Ok(output_expr.lt(&Int::from_i64(ctx, 0)))
        }
    }
}
fn build_input_vars<'ctx>(
    ctx: &'ctx Context,
    function: &Function,
) -> HashMap<String, Int<'ctx>> {
    let mut vars = HashMap::new();
    for input in &function.inputs {
        vars.insert(
            input.name.as_str().to_string(),
            Int::new_const(ctx, input.name.as_str()),
        );
    }
    vars
}
fn apply_input_bounds<'ctx>(
    ctx: &'ctx Context,
    solver: &mut Solver<'ctx>,
    vars: &HashMap<String, Int<'ctx>>,
) {
    for var in vars.values() {
        let lower = Int::from_i64(ctx, -INPUT_BOUND);
        let upper = Int::from_i64(ctx, INPUT_BOUND);
        solver.assert(&var.ge(&lower));
        solver.assert(&var.le(&upper));
    }
}
fn build_output_exprs<'ctx>(
    ctx: &'ctx Context,
    vars: &HashMap<String, Int<'ctx>>,
    ast: &FunctionAst,
    function: &Function,
) -> Result<HashMap<String, Int<'ctx>>, SmtError> {
    let mut outputs = HashMap::new();
    for output in &ast.outputs {
        let expr = encode_expr(&output.expr, ctx, vars, function)?;
        outputs.insert(output.name.clone(), expr);
    }
    Ok(outputs)
}
fn encode_expr<'ctx>(
    expr: &Expr,
    ctx: &'ctx Context,
    vars: &HashMap<String, Int<'ctx>>,
    function: &Function,
) -> Result<Int<'ctx>, SmtError> {
    match expr {
        Expr::Literal { value } => encode_literal(value, ctx, function),
        Expr::Input { name } => {
            vars.get(name)
                .cloned()
                .ok_or_else(|| SmtError::UnsupportedAst {
                    function: function.id.clone(),
                    reason: format!("unknown input `{name}`"),
                })
        }
        Expr::BinOp { left, op, right } => {
            let lhs = encode_expr(left, ctx, vars, function)?;
            let rhs = encode_expr(right, ctx, vars, function)?;
            Ok(
                match op {
                    crate::runtime::ast::BinOp::Add => Int::add(ctx, &[&lhs, &rhs]),
                    crate::runtime::ast::BinOp::Sub => Int::sub(ctx, &[&lhs, &rhs]),
                    crate::runtime::ast::BinOp::Mul => Int::mul(ctx, &[&lhs, &rhs]),
                },
            )
        }
        Expr::FieldAccess { .. } | Expr::Call { .. } | Expr::EmitDelta { .. } => {
            Err(SmtError::UnsupportedAst {
                function: function.id.clone(),
                reason: format!("unsupported expression `{expr:?}`"),
            })
        }
    }
}
fn encode_literal<'ctx>(
    value: &Value,
    ctx: &'ctx Context,
    function: &Function,
) -> Result<Int<'ctx>, SmtError> {
    match value {
        Value::Scalar(ScalarValue::I32(v)) => Ok(Int::from_i64(ctx, *v as i64)),
        _ => {
            Err(SmtError::UnsupportedAst {
                function: function.id.clone(),
                reason: format!("unsupported literal `{value:?}`"),
            })
        }
    }
}
