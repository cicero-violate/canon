//! Function executor for Canon runtime.
//!
//! Implements function invocation and composition according to Canon Laws 27-30.

use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

use crate::ir::{CanonicalIr, Function, FunctionId};
use crate::runtime::bytecode::{BytecodeError, FunctionBytecode, Instruction};
use crate::runtime::context::{ContextError, ExecutionContext};
use crate::runtime::value::{ScalarValue, Value};

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
/// Currently a stub that returns mock outputs (to be replaced with actual execution).
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

            // Validate inputs match function signature
            self.validate_inputs(function, &current_inputs)?;

            // Execute function
            let outputs = self.execute(function, current_inputs.clone(), context)?;

            // Store results
            results.push(outputs.clone());

            // Outputs become inputs for next function (Canon Line 30: composition)
            current_inputs = outputs;
        }

        Ok(results)
    }

    fn validate_inputs(
        &self,
        function: &Function,
        inputs: &BTreeMap<String, Value>,
    ) -> Result<(), ExecutorError> {
        // Check all required inputs are present (Canon Line 32: explicit inputs)
        for input_port in &function.inputs {
            if !inputs.contains_key(input_port.name.as_str()) {
                return Err(ExecutorError::MissingInput {
                    function: function.id.clone(),
                    input: input_port.name.as_str().to_string(),
                });
            }

            // Type checking
            let value = &inputs[input_port.name.as_str()];
            if !value.is_compatible_with(&input_port.ty) {
                return Err(ExecutorError::TypeMismatch {
                    function: function.id.clone(),
                    port: input_port.name.as_str().to_string(),
                    expected: format!("{:?}", input_port.ty),
                    found: format!("{:?}", value.kind()),
                });
            }
        }

        Ok(())
    }
}

impl<'a> Executor for FunctionExecutor<'a> {
    fn execute(
        &self,
        function: &Function,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, ExecutorError> {
        // Push onto call stack (Canon Line 75: detect recursion)
        context
            .push_call(function.id.clone())
            .map_err(ExecutorError::Context)?;

        // Validate function contract (Canon Lines 31-34)
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

        // Validate inputs
        self.validate_inputs(function, &inputs)?;

        let outputs = self.interpret_bytecode(function, &inputs, context)?;

        // Pop from call stack
        context.pop_call().map_err(ExecutorError::Context)?;

        Ok(outputs)
    }
}

impl<'a> FunctionExecutor<'a> {
    fn interpret_bytecode(
        &self,
        function: &Function,
        inputs: &BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, InterpreterError> {
        let program = FunctionBytecode::from_function(function)?;
        let mut stack: Vec<Value> = Vec::new();
        let mut binding_keys: HashMap<String, String> = HashMap::new();

        for instruction in &program.instructions {
            match instruction {
                Instruction::LoadConst(value) => stack.push(value.clone()),
                Instruction::LoadInput(name) => {
                    let value = inputs.get(name).cloned().ok_or_else(|| {
                        InterpreterError::MissingInput {
                            function: function.id.clone(),
                            input: name.clone(),
                        }
                    })?;
                    stack.push(value);
                }
                Instruction::StoreBinding(name) => {
                    let value =
                        stack
                            .last()
                            .cloned()
                            .ok_or_else(|| InterpreterError::StackUnderflow {
                                function: function.id.clone(),
                            })?;
                    let key = context.bind_scoped(&function.id, name, value)?;
                    binding_keys.insert(name.clone(), key);
                }
                Instruction::LoadBinding(name) => {
                    let key = binding_keys.get(name).ok_or_else(|| {
                        InterpreterError::BindingNotFound {
                            function: function.id.clone(),
                            binding: name.clone(),
                        }
                    })?;
                    let value = context
                        .lookup(key)
                        .map_err(InterpreterError::Context)?
                        .clone();
                    stack.push(value);
                }
                Instruction::FieldAccess(field) => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| InterpreterError::StackUnderflow {
                            function: function.id.clone(),
                        })?;
                    let struct_value =
                        value
                            .as_struct()
                            .map_err(|err| InterpreterError::TypeError {
                                function: function.id.clone(),
                                message: err.to_string(),
                            })?;
                    let field_value = struct_value.get_field(field).map_err(|err| {
                        InterpreterError::TypeError {
                            function: function.id.clone(),
                            message: err.to_string(),
                        }
                    })?;
                    stack.push(field_value.clone());
                }
                Instruction::Add => {
                    let rhs = self.pop_value(function, &mut stack)?;
                    let lhs = self.pop_value(function, &mut stack)?;
                    let result = Self::apply_arithmetic(function, lhs, rhs, ArithmeticOp::Add)?;
                    stack.push(result);
                }
                Instruction::Sub => {
                    let rhs = self.pop_value(function, &mut stack)?;
                    let lhs = self.pop_value(function, &mut stack)?;
                    let result = Self::apply_arithmetic(function, lhs, rhs, ArithmeticOp::Sub)?;
                    stack.push(result);
                }
                Instruction::Mul => {
                    let rhs = self.pop_value(function, &mut stack)?;
                    let lhs = self.pop_value(function, &mut stack)?;
                    let result = Self::apply_arithmetic(function, lhs, rhs, ArithmeticOp::Mul)?;
                    stack.push(result);
                }
                Instruction::Call(target) => {
                    let callee = self
                        .ir
                        .functions
                        .iter()
                        .find(|f| &f.id == target)
                        .ok_or_else(|| InterpreterError::UnknownFunction {
                            function: target.clone(),
                        })?;
                    let mut args = Vec::new();
                    for _ in 0..callee.inputs.len() {
                        let arg = self.pop_value(function, &mut stack)?;
                        args.push(arg);
                    }
                    args.reverse();
                    let mut call_inputs = BTreeMap::new();
                    for (input, value) in callee.inputs.iter().zip(args.into_iter()) {
                        call_inputs.insert(input.name.as_str().to_string(), value);
                    }
                    let outputs = self.execute(callee, call_inputs, context).map_err(|err| {
                        InterpreterError::CallFailed {
                            caller: function.id.clone(),
                            callee: callee.id.clone(),
                            source: Box::new(err),
                        }
                    })?;
                    for output in &callee.outputs {
                        let value = outputs.get(output.name.as_str()).ok_or_else(|| {
                            InterpreterError::MissingOutput {
                                function: callee.id.clone(),
                                output: output.name.as_str().to_string(),
                            }
                        })?;
                        stack.push(value.clone());
                    }
                }
                Instruction::EmitDelta(delta) => context.emit_delta(delta.clone()),
                Instruction::Return => {
                    return self.collect_outputs(function, &mut stack);
                }
            }
        }

        Err(InterpreterError::MissingReturn {
            function: function.id.clone(),
        })
    }

    fn collect_outputs(
        &self,
        function: &Function,
        stack: &mut Vec<Value>,
    ) -> Result<BTreeMap<String, Value>, InterpreterError> {
        let mut outputs = BTreeMap::new();
        for output in function.outputs.iter().rev() {
            let value = stack
                .pop()
                .ok_or_else(|| InterpreterError::StackUnderflow {
                    function: function.id.clone(),
                })?;
            outputs.insert(output.name.as_str().to_string(), value);
        }
        Ok(outputs)
    }

    fn pop_value(
        &self,
        function: &Function,
        stack: &mut Vec<Value>,
    ) -> Result<Value, InterpreterError> {
        stack.pop().ok_or_else(|| InterpreterError::StackUnderflow {
            function: function.id.clone(),
        })
    }

    fn apply_arithmetic(
        function: &Function,
        lhs: Value,
        rhs: Value,
        op: ArithmeticOp,
    ) -> Result<Value, InterpreterError> {
        match (lhs, rhs) {
            (Value::Scalar(ScalarValue::I32(a)), Value::Scalar(ScalarValue::I32(b))) => {
                Ok(Value::Scalar(ScalarValue::I32(match op {
                    ArithmeticOp::Add => a + b,
                    ArithmeticOp::Sub => a - b,
                    ArithmeticOp::Mul => a * b,
                })))
            }
            (Value::Scalar(ScalarValue::F32(a)), Value::Scalar(ScalarValue::F32(b))) => {
                Ok(Value::Scalar(ScalarValue::F32(match op {
                    ArithmeticOp::Add => a + b,
                    ArithmeticOp::Sub => a - b,
                    ArithmeticOp::Mul => a * b,
                })))
            }
            (Value::Scalar(ScalarValue::F64(a)), Value::Scalar(ScalarValue::F64(b))) => {
                Ok(Value::Scalar(ScalarValue::F64(match op {
                    ArithmeticOp::Add => a + b,
                    ArithmeticOp::Sub => a - b,
                    ArithmeticOp::Mul => a * b,
                })))
            }
            (Value::Scalar(ScalarValue::U32(a)), Value::Scalar(ScalarValue::U32(b))) => {
                Ok(Value::Scalar(ScalarValue::U32(match op {
                    ArithmeticOp::Add => a + b,
                    ArithmeticOp::Sub => a - b,
                    ArithmeticOp::Mul => a * b,
                })))
            }
            _ => Err(InterpreterError::TypeError {
                function: function.id.clone(),
                message: "arithmetic requires matching scalar types".into(),
            }),
        }
    }
}

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

enum ArithmeticOp {
    Add,
    Sub,
    Mul,
}
