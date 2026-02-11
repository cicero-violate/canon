// Derived from Canonical IR. Do not edit.

pub struct State {
    Value: ScalarValue,
}

pub trait Add {
    fn AddTwo(Lhs: ScalarValue, Rhs: ScalarValue) -> ScalarValue;
}
