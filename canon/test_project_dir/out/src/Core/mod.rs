// Derived from Canonical IR. Do not edit.

pub struct State {
    Value: ScalarValue,
}

pub trait Compute {
    fn Process(Input: ScalarValue) -> ScalarValue;
}

impl Compute for State {

    pub fn Process(Input: ScalarValue) -> ScalarValue {
        // Invoke Canon runtime interpreter (generated stub)
        canon_runtime::execute_function("fn.compute");
    }

}
