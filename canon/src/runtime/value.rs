//! Value representation for runtime execution.
//!
//! Values flow between functions during composition.
//! All values are explicit and serializable (Canon Line 32-33).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::ir::{ScalarType, TypeRef};

/// Runtime value passed between functions.
/// All values are explicit and inspectable (Canon Line 32-33).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub enum Value {
    Scalar(ScalarValue),
    Struct(StructValue),
    Array(Vec<Value>),
    Delta(DeltaValue),
    Unit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    U32(u32),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct StructValue {
    pub type_name: String,
    pub fields: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct DeltaValue {
    pub delta_id: String,
    pub payload_hash: String,
}

impl Value {
    pub fn kind(&self) -> ValueKind {
        match self {
            Value::Scalar(s) => ValueKind::Scalar(s.scalar_type()),
            Value::Struct(s) => ValueKind::Struct(s.type_name.clone()),
            Value::Array(_) => ValueKind::Array,
            Value::Delta(_) => ValueKind::Delta,
            Value::Unit => ValueKind::Unit,
        }
    }

    pub fn is_compatible_with(&self, type_ref: &TypeRef) -> bool {
        match (&self, &type_ref.kind) {
            (Value::Scalar(_scalar), crate::ir::TypeKind::Scalar) => {
                // Type checking would compare s.scalar_type() with type_ref.name
                true
            }
            (Value::Struct(s), crate::ir::TypeKind::Struct) => s.type_name == type_ref.name.as_str(),
            (Value::Delta(_), crate::ir::TypeKind::Delta) => true,
            (Value::Unit, crate::ir::TypeKind::External) => true,
            _ => false,
        }
    }

    pub fn as_scalar(&self) -> Result<&ScalarValue, ValueError> {
        match self {
            Value::Scalar(s) => Ok(s),
            _ => Err(ValueError::TypeMismatch { expected: "Scalar".into(), found: format!("{:?}", self.kind()) }),
        }
    }

    pub fn as_struct(&self) -> Result<&StructValue, ValueError> {
        match self {
            Value::Struct(s) => Ok(s),
            _ => Err(ValueError::TypeMismatch { expected: "Struct".into(), found: format!("{:?}", self.kind()) }),
        }
    }

    pub fn as_delta(&self) -> Result<&DeltaValue, ValueError> {
        match self {
            Value::Delta(d) => Ok(d),
            _ => Err(ValueError::TypeMismatch { expected: "Delta".into(), found: format!("{:?}", self.kind()) }),
        }
    }
}

impl ScalarValue {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ScalarValue::F32(_) => ScalarType::F32,
            ScalarValue::F64(_) => ScalarType::F64,
            ScalarValue::I32(_) => ScalarType::I32,
            ScalarValue::U32(_) => ScalarType::U32,
            ScalarValue::Bool(_) => ScalarType::Bool,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueKind {
    Scalar(ScalarType),
    Struct(String),
    Array,
    Delta,
    Unit,
}

#[derive(Debug, Error)]
pub enum ValueError {
    #[error("type mismatch: expected `{expected}`, found `{found}`")]
    TypeMismatch { expected: String, found: String },
    #[error("field `{0}` not found in struct")]
    FieldNotFound(String),
}

impl StructValue {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self { type_name: type_name.into(), fields: BTreeMap::new() }
    }

    pub fn with_field(mut self, name: impl Into<String>, value: Value) -> Self {
        self.fields.insert(name.into(), value);
        self
    }

    pub fn get_field(&self, name: &str) -> Result<&Value, ValueError> {
        self.fields.get(name).ok_or_else(|| ValueError::FieldNotFound(name.to_string()))
    }
}
