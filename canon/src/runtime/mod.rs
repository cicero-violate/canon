//! Runtime execution engine for Canon.
//!
//! Implements function composition and delta emission according to Canon Law.
pub mod ast;
pub mod bytecode;
pub mod bytecode_types;
pub mod context;
pub mod delta_verifier;
pub mod error;
pub mod executor;
pub mod graph;
pub mod parallel;
pub mod policy_updater;
pub mod reward;
pub mod rollout;
pub mod system_interpreter;
pub mod tick_executor;
pub mod value;
pub use ast::{compile_function_ast, BinOp, Expr, FunctionAst, OutputExpr};
pub use bytecode_types::{FunctionBytecode, Instruction};
pub use context::{ExecutionContext, ExecutionState};
pub use delta_verifier::{
    ChangeVerifier, Snapshot, VerificationError, VerificationResult,
};
pub use executor::{Executor, ExecutorError, FunctionExecutor};
pub use policy_updater::{update_policy, PolicyUpdateError, PolicyUpdater};
pub use system_interpreter::{
    DeltaEmission, ProofArtifact, SystemExecutionEvent, SystemExecutionResult,
    SystemInterpreter, SystemInterpreterError,
};
pub use tick_executor::{TickExecutionMode, TickExecutionResult, ExecutionEngine};
pub use value::{Value, ValueKind};
