//! Bytecode data types shared between ast and bytecode modules.
//! Kept separate to avoid a cyclic dependency between ast ↔ bytecode.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ir::FunctionId;
use crate::runtime::value::{DeltaValue, Value};

/// Canon bytecode instruction set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case", tag = "op", content = "args")]
pub enum Instruction {
    LoadConst(Value),
    LoadInput(String),
    LoadBinding(String),
    StoreBinding(String),
    FieldAccess(String),
    Add,
    Sub,
    Mul,
    Call(FunctionId),
    EmitDelta(DeltaValue),
    Return,
}

/// Serialized bytecode stored alongside Canonical IR.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct FunctionBytecode {
    pub instructions: Vec<Instruction>,
}

impl FunctionBytecode {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self { instructions }
    }
}
