use super::error::InterpreterError;
use crate::ir::Function;
use crate::runtime::value::{ScalarValue, Value};

pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
}

pub fn apply_arithmetic(
    function: &Function,
    lhs: Value,
    rhs: Value,
    op: ArithmeticOp,
) -> Result<Value, InterpreterError> {
    match (lhs, rhs) {
        (Value::Scalar(ScalarValue::I32(a)), Value::Scalar(ScalarValue::I32(b))) => {
            Ok(Value::Scalar(ScalarValue::I32(apply_op(a, b, &op))))
        }
        (Value::Scalar(ScalarValue::U32(a)), Value::Scalar(ScalarValue::U32(b))) => {
            Ok(Value::Scalar(ScalarValue::U32(apply_op(a, b, &op))))
        }
        (Value::Scalar(ScalarValue::F32(a)), Value::Scalar(ScalarValue::F32(b))) => {
            Ok(Value::Scalar(ScalarValue::F32(apply_op_f32(a, b, &op))))
        }
        (Value::Scalar(ScalarValue::F64(a)), Value::Scalar(ScalarValue::F64(b))) => {
            Ok(Value::Scalar(ScalarValue::F64(apply_op_f64(a, b, &op))))
        }
        _ => Err(InterpreterError::TypeError {
            function: function.id.clone(),
            message: "arithmetic requires matching scalar types".into(),
        }),
    }
}

fn apply_op<T>(a: T, b: T, op: &ArithmeticOp) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    match op {
        ArithmeticOp::Add => a + b,
        ArithmeticOp::Sub => a - b,
        ArithmeticOp::Mul => a * b,
    }
}

fn apply_op_f32(a: f32, b: f32, op: &ArithmeticOp) -> f32 {
    match op {
        ArithmeticOp::Add => a + b,
        ArithmeticOp::Sub => a - b,
        ArithmeticOp::Mul => a * b,
    }
}

fn apply_op_f64(a: f64, b: f64, op: &ArithmeticOp) -> f64 {
    match op {
        ArithmeticOp::Add => a + b,
        ArithmeticOp::Sub => a - b,
        ArithmeticOp::Mul => a * b,
    }
}
