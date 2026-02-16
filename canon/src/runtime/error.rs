use crate::ir::FunctionId;
use crate::runtime::executor::ExecutorError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("unknown tick `{0}`")]
    UnknownTick(String),
    #[error("unknown graph `{0}`")]
    UnknownGraph(String),
    #[error("cycle detected in tick graph at function `{0}`")]
    CycleDetected(FunctionId),
    #[error("parallel execution mismatch for function `{function}`")]
    ParallelMismatch { function: FunctionId },
    #[error("parallel delta mismatch at index {index}")]
    ParallelDeltaMismatch { index: usize },
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}
