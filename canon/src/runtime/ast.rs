//! Abstract syntax tree representation for Canon runtime functions.
//!
//! Functions authored in DSLs can be lowered into this AST, which is then
//! compiled into bytecode for execution (Canon Line 27).

use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::FunctionId;
use crate::runtime::bytecode::{FunctionBytecode, Instruction};
use crate::runtime::value::{DeltaValue, Value};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FunctionAst {
    pub outputs: Vec<OutputExpr>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OutputExpr {
    pub name: String,
    pub expr: Expr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Expr {
    Literal { value: Value },
    Input {
        name: String,
    },
    BinOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },
    Call {
        function: FunctionId,
        args: Vec<Expr>,
    },
    EmitDelta {
        delta_id: String,
        payload_hash: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Error)]
pub enum AstError {
    #[error("output `{0}` defined more than once")]
    DuplicateOutput(String),
}

pub fn compile_function_ast(ast: &FunctionAst) -> Result<FunctionBytecode, AstError> {
    let mut instructions = Vec::new();
    let mut outputs = HashSet::new();
    for output in &ast.outputs {
        if !outputs.insert(output.name.clone()) {
            return Err(AstError::DuplicateOutput(output.name.clone()));
        }
        compile_expr(&output.expr, &mut instructions)?;
        instructions.push(Instruction::StoreBinding(output.name.clone()));
    }

    for output in &ast.outputs {
        instructions.push(Instruction::LoadBinding(output.name.clone()));
    }
    instructions.push(Instruction::Return);
    Ok(FunctionBytecode::new(instructions))
}

fn compile_expr(expr: &Expr, instructions: &mut Vec<Instruction>) -> Result<(), AstError> {
    match expr {
        Expr::Literal { value } => instructions.push(Instruction::LoadConst(value.clone())),
        Expr::Input { name } => instructions.push(Instruction::LoadInput(name.clone())),
        Expr::BinOp { left, op, right } => {
            compile_expr(left, instructions)?;
            compile_expr(right, instructions)?;
            match op {
                BinOp::Add => instructions.push(Instruction::Add),
                BinOp::Sub => instructions.push(Instruction::Sub),
                BinOp::Mul => instructions.push(Instruction::Mul),
            }
        }
        Expr::FieldAccess { expr, field } => {
            compile_expr(expr, instructions)?;
            instructions.push(Instruction::FieldAccess(field.clone()));
        }
        Expr::Call { function, args } => {
            for arg in args {
                compile_expr(arg, instructions)?;
            }
            instructions.push(Instruction::Call(function.clone()));
        }
        Expr::EmitDelta {
            delta_id,
            payload_hash,
        } => {
            instructions.push(Instruction::EmitDelta(DeltaValue {
                delta_id: delta_id.clone(),
                payload_hash: payload_hash.clone(),
            }));
            instructions.push(Instruction::LoadConst(Value::Unit));
        }
    }
    Ok(())
}
