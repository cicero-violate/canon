use std::collections::{BTreeMap, HashMap};

use crate::ir::Function;
use crate::runtime::bytecode::{FunctionBytecode, Instruction};
use crate::runtime::context::ExecutionContext;
use crate::runtime::value::Value;

use super::Executor;
use super::arith::{ArithmeticOp, apply_arithmetic};
use super::error::{ExecutorError, InterpreterError};

pub fn interpret_bytecode(
    executor: &dyn Executor,
    ir: &crate::ir::CanonicalIr,
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
                let value =
                    inputs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| InterpreterError::MissingInput {
                            function: function.id.clone(),
                            input: name.clone(),
                        })?;
                stack.push(value);
            }
            Instruction::StoreBinding(name) => {
                let value = pop(&function, &mut stack)?;
                let key = context.bind_scoped(&function.id, name, value)?;
                binding_keys.insert(name.clone(), key);
            }
            Instruction::LoadBinding(name) => {
                let key =
                    binding_keys
                        .get(name)
                        .ok_or_else(|| InterpreterError::BindingNotFound {
                            function: function.id.clone(),
                            binding: name.clone(),
                        })?;
                let value = context
                    .lookup(key)
                    .map_err(InterpreterError::Context)?
                    .clone();
                stack.push(value);
            }
            Instruction::FieldAccess(field) => {
                let value = pop(function, &mut stack)?;
                let struct_val = value.as_struct().map_err(|e| InterpreterError::TypeError {
                    function: function.id.clone(),
                    message: e.to_string(),
                })?;
                let field_value =
                    struct_val
                        .get_field(field)
                        .map_err(|e| InterpreterError::TypeError {
                            function: function.id.clone(),
                            message: e.to_string(),
                        })?;
                stack.push(field_value.clone());
            }
            Instruction::Add => {
                let rhs = pop(function, &mut stack)?;
                let lhs = pop(function, &mut stack)?;
                stack.push(apply_arithmetic(function, lhs, rhs, ArithmeticOp::Add)?);
            }
            Instruction::Sub => {
                let rhs = pop(function, &mut stack)?;
                let lhs = pop(function, &mut stack)?;
                stack.push(apply_arithmetic(function, lhs, rhs, ArithmeticOp::Sub)?);
            }
            Instruction::Mul => {
                let rhs = pop(function, &mut stack)?;
                let lhs = pop(function, &mut stack)?;
                stack.push(apply_arithmetic(function, lhs, rhs, ArithmeticOp::Mul)?);
            }
            Instruction::Call(target) => {
                let callee = ir
                    .functions
                    .iter()
                    .find(|f| &f.id == target)
                    .ok_or_else(|| InterpreterError::UnknownFunction {
                        function: target.clone(),
                    })?;
                let mut args = Vec::new();
                for _ in 0..callee.inputs.len() {
                    args.push(pop(function, &mut stack)?);
                }
                args.reverse();
                let mut call_inputs = BTreeMap::new();
                for (port, value) in callee.inputs.iter().zip(args) {
                    call_inputs.insert(port.name.as_str().to_string(), value);
                }
                let mut owned_inputs = call_inputs.clone();
                let outputs = executor
                    .execute(callee, owned_inputs, context)
                    .map_err(|e| InterpreterError::CallFailed {
                        caller: function.id.clone(),
                        callee: callee.id.clone(),
                        source: Box::new(e),
                    })?;
                for port in &callee.outputs {
                    let value = outputs.get(port.name.as_str()).ok_or_else(|| {
                        InterpreterError::MissingOutput {
                            function: callee.id.clone(),
                            output: port.name.as_str().to_string(),
                        }
                    })?;
                    stack.push(value.clone());
                }
            }
            Instruction::EmitDelta(delta) => context.emit_delta(delta.clone()),
            Instruction::Return => return collect_outputs(function, &mut stack),
        }
    }

    Err(InterpreterError::MissingReturn {
        function: function.id.clone(),
    })
}

fn pop(function: &Function, stack: &mut Vec<Value>) -> Result<Value, InterpreterError> {
    stack.pop().ok_or_else(|| InterpreterError::StackUnderflow {
        function: function.id.clone(),
    })
}

fn collect_outputs(
    function: &Function,
    stack: &mut Vec<Value>,
) -> Result<BTreeMap<String, Value>, InterpreterError> {
    let mut outputs = BTreeMap::new();
    for port in function.outputs.iter().rev() {
        let value = pop(function, stack)?;
        outputs.insert(port.name.as_str().to_string(), value);
    }
    Ok(outputs)
}
