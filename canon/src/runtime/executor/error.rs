use crate::ir::FunctionId;
use crate::runtime::bytecode::BytecodeError;
use crate::runtime::context::ContextError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("unknown function `{0}`")]
    UnknownFunction(FunctionId),
    #[error("function `{function}` missing required input `{input}`")]
    MissingInput { function: FunctionId, input: String },
    #[error(
        "function `{function}` port `{port}` type mismatch: expected `{expected}`, found `{found}`"
    )]
    TypeMismatch {
        function: FunctionId,
        port: String,
        expected: String,
        found: String,
    },
    #[error("function `{function}` violates contract: {reason}")]
    ContractViolation {
        function: FunctionId,
        reason: String,
    },
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error(transparent)]
    Interpreter(#[from] InterpreterError),
}

#[derive(Debug, Error)]
pub enum InterpreterError {
    #[error("function `{function}` missing required input `{input}`")]
    MissingInput { function: FunctionId, input: String },
    #[error("stack underflow in function `{function}`")]
    StackUnderflow { function: FunctionId },
    #[error("function `{function}` binding `{binding}` not found")]
    BindingNotFound {
        function: FunctionId,
        binding: String,
    },
    #[error("function `{function}` missing output `{output}` from call")]
    MissingOutput {
        function: FunctionId,
        output: String,
    },
    #[error("function `{function}` missing return instruction")]
    MissingReturn { function: FunctionId },
    #[error("function `{function}` type error: {message}")]
    TypeError {
        function: FunctionId,
        message: String,
    },
    #[error("unknown function `{function}` invoked from bytecode")]
    UnknownFunction { function: FunctionId },
    #[error("call to `{callee}` from `{caller}` failed: {source}")]
    CallFailed {
        caller: FunctionId,
        callee: FunctionId,
        #[source]
        source: Box<ExecutorError>,
    },
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error(transparent)]
    Bytecode(#[from] BytecodeError),
}
