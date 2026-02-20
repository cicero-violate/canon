//! Execution context for Canon runtime.
//!
//! Tracks state, deltas, and function call stack during execution.
//! Enforces Canon Laws 29-40 (no hidden mutation, explicit effects, etc.).

use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

use crate::ir::FunctionId;
use crate::runtime::value::{DeltaValue, Value};

/// Execution context maintains state for a single tick execution.
/// All state is explicit and inspectable (Canon Line 35-36).
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Current execution state (immutable during tick)
    state: ExecutionState,
    /// Deltas emitted during this execution (append-only, Canon Line 38)
    emitted_deltas: Vec<DeltaValue>,
    /// Call stack for recursion detection (Canon Line 75)
    call_stack: Vec<FunctionId>,
    /// Max call depth to prevent infinite recursion
    max_call_depth: usize,
    /// Unique binding counter so scoped bindings never collide.
    binding_counter: u64,
}

/// Execution state snapshot.
/// State is immutable during execution (Canon Line 39).
#[derive(Debug, Clone)]
pub struct ExecutionState {
    /// Input values from previous tick
    inputs: BTreeMap<String, Value>,
    /// Intermediate values during composition
    bindings: HashMap<String, Value>,
}

impl ExecutionContext {
    pub fn new(inputs: BTreeMap<String, Value>) -> Self {
        Self {
            state: ExecutionState { inputs, bindings: HashMap::new() },
            emitted_deltas: Vec::new(),
            call_stack: Vec::new(),
            max_call_depth: 100, // Prevent deep recursion (Canon Line 75)
            binding_counter: 0,
        }
    }

    /// Bind a value to a name in the current scope.
    /// No mutation of existing bindings allowed (Canon Line 35).
    pub fn bind(&mut self, name: impl Into<String>, value: Value) -> Result<(), ContextError> {
        let name = name.into();
        if self.state.bindings.contains_key(&name) {
            return Err(ContextError::BindingExists(name));
        }
        self.state.bindings.insert(name, value);
        Ok(())
    }

    /// Lookup a value by name.
    pub fn lookup(&self, name: &str) -> Result<&Value, ContextError> {
        self.state.bindings.get(name).or_else(|| self.state.inputs.get(name)).ok_or_else(|| ContextError::UnboundName(name.to_string()))
    }

    /// Emit a delta (Canon Line 34: effects as deltas).
    /// Deltas are append-only (Canon Line 38).
    pub fn emit_delta(&mut self, delta: DeltaValue) {
        self.emitted_deltas.push(delta);
    }

    /// Get all emitted deltas.
    pub fn deltas(&self) -> &[DeltaValue] {
        &self.emitted_deltas
    }

    /// Push function onto call stack (Canon Line 75: no recursion).
    pub fn push_call(&mut self, function_id: FunctionId) -> Result<(), ContextError> {
        // Check for recursion (Canon Line 75)
        if self.call_stack.contains(&function_id) {
            return Err(ContextError::RecursionDetected(function_id));
        }

        // Check max depth
        if self.call_stack.len() >= self.max_call_depth {
            return Err(ContextError::CallStackOverflow);
        }

        self.call_stack.push(function_id);
        Ok(())
    }

    /// Pop function from call stack.
    pub fn pop_call(&mut self) -> Result<FunctionId, ContextError> {
        self.call_stack.pop().ok_or(ContextError::CallStackUnderflow)
    }

    /// Get current call stack (for debugging).
    pub fn call_stack(&self) -> &[FunctionId] {
        &self.call_stack
    }

    /// Create a child context for nested execution.
    /// Inherits state but maintains separate delta log.
    pub fn create_child(&self) -> Self {
        Self { state: self.state.clone(), emitted_deltas: Vec::new(), call_stack: self.call_stack.clone(), max_call_depth: self.max_call_depth, binding_counter: self.binding_counter }
    }

    /// Merge child context back (Canon Line 30: composition only).
    pub fn merge_child(&mut self, child: ExecutionContext) {
        // Append deltas (Canon Line 38)
        self.emitted_deltas.extend(child.emitted_deltas);
        // Merge bindings
        self.state.bindings.extend(child.state.bindings);
    }

    /// Bind a value with an auto-generated unique key tied to the function ID and slot.
    pub fn bind_scoped(&mut self, function_id: &FunctionId, slot: impl Into<String>, value: Value) -> Result<String, ContextError> {
        let slot = slot.into();
        let key = format!("{}::{}::{}", function_id, slot, self.binding_counter);
        self.binding_counter += 1;
        self.bind(key.clone(), value)?;
        Ok(key)
    }
}

impl ExecutionState {
    pub fn get_input(&self, name: &str) -> Result<&Value, ContextError> {
        self.inputs.get(name).ok_or_else(|| ContextError::UnboundName(name.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("binding `{0}` already exists (no mutation allowed)")]
    BindingExists(String),
    #[error("unbound name `{0}`")]
    UnboundName(String),
    #[error("recursion detected for function `{0}` (Canon Line 75)")]
    RecursionDetected(FunctionId),
    #[error("call stack overflow (max depth exceeded)")]
    CallStackOverflow,
    #[error("call stack underflow (pop on empty stack)")]
    CallStackUnderflow,
}
