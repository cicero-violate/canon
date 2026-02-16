//! Function executor for Canon runtime.
//!
//! Implements function invocation and composition according to Canon Laws 27-30.

mod arith;
mod error;
mod interp;

use std::collections::BTreeMap;

use crate::ir::{CanonicalIr, Function, FunctionId};
use crate::runtime::context::ExecutionContext;
use crate::runtime::value::Value;

pub use error::{ExecutorError, InterpreterError};

/// Executor trait for functions (Canon Line 27: execution in impl).
pub trait Executor {
    fn execute(
        &self,
        function: &Function,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, ExecutorError>;
}

/// Default function executor.
pub struct FunctionExecutor<'a> {
    ir: &'a CanonicalIr,
}

impl<'a> FunctionExecutor<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self { ir }
    }

    /// Execute a function by ID.
    pub fn execute_by_id(
        &self,
        function_id: &FunctionId,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, ExecutorError> {
        let function = self
            .ir
            .functions
            .iter()
            .find(|f| &f.id == function_id)
            .ok_or_else(|| ExecutorError::UnknownFunction(function_id.clone()))?;
        self.execute(function, inputs, context)
    }

    /// Compose multiple functions in a tick graph.
    /// Data flows from outputs to inputs (Canon Line 30).
    pub fn execute_composition(
        &self,
        functions: &[FunctionId],
        initial_inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<Vec<BTreeMap<String, Value>>, ExecutorError> {
        let mut results = Vec::new();
        let mut current_inputs = initial_inputs;
        for function_id in functions {
            let function = self
                .ir
                .functions
                .iter()
                .find(|f| &f.id == function_id)
                .ok_or_else(|| ExecutorError::UnknownFunction(function_id.clone()))?;
            validate_inputs(function, &current_inputs)?;
            let outputs = self.execute(function, current_inputs.clone(), context)?;
            results.push(outputs.clone());
            current_inputs = outputs;
        }
        Ok(results)
    }
}

impl<'a> Executor for FunctionExecutor<'a> {
    fn execute(
        &self,
        function: &Function,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, ExecutorError> {
        context
            .push_call(function.id.clone())
            .map_err(ExecutorError::Context)?;
        check_contract(function)?;
        validate_inputs(function, &inputs)?;
        let outputs = interp::interpret_bytecode(self, self.ir, function, &inputs, context)?;
        context.pop_call().map_err(ExecutorError::Context)?;
        Ok(outputs)
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

fn check_contract(function: &Function) -> Result<(), ExecutorError> {
    if !function.contract.total {
        return Err(ExecutorError::ContractViolation {
            function: function.id.clone(),
            reason: "function must be total".into(),
        });
    }
    if !function.contract.explicit_inputs || !function.contract.explicit_outputs {
        return Err(ExecutorError::ContractViolation {
            function: function.id.clone(),
            reason: "inputs and outputs must be explicit".into(),
        });
    }
    if !function.contract.effects_are_deltas {
        return Err(ExecutorError::ContractViolation {
            function: function.id.clone(),
            reason: "effects must be deltas".into(),
        });
    }
    Ok(())
}

fn validate_inputs(
    function: &Function,
    inputs: &BTreeMap<String, Value>,
) -> Result<(), ExecutorError> {
    for port in &function.inputs {
        if !inputs.contains_key(port.name.as_str()) {
            return Err(ExecutorError::MissingInput {
                function: function.id.clone(),
                input: port.name.as_str().to_string(),
            });
        }
        let value = &inputs[port.name.as_str()];
        if !value.is_compatible_with(&port.ty) {
            return Err(ExecutorError::TypeMismatch {
                function: function.id.clone(),
                port: port.name.as_str().to_string(),
                expected: format!("{:?}", port.ty),
                found: format!("{:?}", value.kind()),
            });
        }
    }
    Ok(())
}
