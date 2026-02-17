//! Bytecode program representation for Canon runtime.
//!
//! Bytecode is a portable, serialized form of function execution that can be
//! shipped with Canonical IR metadata. Functions execute by interpreting this
//! bytecode, keeping user-defined source out of the runtime (Canon Line 27).

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bincode;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::{Function, FunctionId};
use crate::runtime::ast::{FunctionAst, compile_function_ast};
pub use crate::runtime::bytecode_types::{FunctionBytecode, Instruction};
use crate::runtime::value::{DeltaValue, Value};

impl FunctionBytecode {
    /// Load bytecode from metadata or compile AST fallback.
    pub fn from_function(function: &Function) -> Result<Self, BytecodeError> {
        if let Some(encoded) = function.metadata.bytecode_b64.as_deref() {
            return Self::decode(function, encoded);
        }

        if let Some(ast_json) = function.metadata.ast.as_ref() {
            let ast: FunctionAst = serde_json::from_value(ast_json.clone()).map_err(|err| {
                BytecodeError::InvalidAst {
                    function: function.id.clone(),
                    message: err.to_string(),
                }
            })?;
            let program = compile_function_ast(&ast).map_err(|err| BytecodeError::InvalidAst {
                function: function.id.clone(),
                message: err.to_string(),
            })?;
            return Ok(program);
        }

        Err(BytecodeError::MissingBytecode {
            function: function.id.clone(),
        })
    }

    pub fn decode(function: &Function, encoded: &str) -> Result<Self, BytecodeError> {
        let bytes = BASE64
            .decode(encoded)
            .map_err(|err| BytecodeError::Decode {
                function: function.id.clone(),
                message: err.to_string(),
            })?;
        bincode::deserialize(&bytes).map_err(|err| BytecodeError::Decode {
            function: function.id.clone(),
            message: err.to_string(),
        })
    }

    pub fn encode(&self, function: &Function) -> Result<String, BytecodeError> {
        let bytes =
            bincode::serialize(&self.instructions).map_err(|err| BytecodeError::Encode {
                function: function.id.clone(),
                message: err.to_string(),
            })?;
        Ok(BASE64.encode(bytes))
    }
}

#[derive(Debug, Error)]
pub enum BytecodeError {
    #[error("function `{function}` missing bytecode metadata")]
    MissingBytecode { function: FunctionId },
    #[error("function `{function}` failed to decode bytecode: {message}")]
    Decode {
        function: FunctionId,
        message: String,
    },
    #[error("function `{function}` failed to encode bytecode: {message}")]
    Encode {
        function: FunctionId,
        message: String,
    },
    #[error("function `{function}` AST invalid: {message}")]
    InvalidAst {
        function: FunctionId,
        message: String,
    },
}
