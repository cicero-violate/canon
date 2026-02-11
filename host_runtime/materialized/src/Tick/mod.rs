// Derived from Canonical IR. Do not edit.

impl Add for State {

    pub fn Add(Lhs: ScalarValue, Rhs: ScalarValue) -> ScalarValue {
        // Invoke Canon runtime interpreter (generated stub)
        canon_runtime::execute_function("fn.add");
    }

}
