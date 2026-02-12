use std::sync::Mutex;

extern crate lazy_static;
extern crate serde;

#[derive(Debug, serde::Serialize)]
pub struct LintSignal {
    pub policy: &'static str,
    pub def_path: String,
    pub kind: &'static str,
    pub module: String,
    pub severity: f32,
}

lazy_static::lazy_static! {
    pub static ref LINT_SIGNALS: Mutex<Vec<LintSignal>> =
        Mutex::new(Vec::new());
}
