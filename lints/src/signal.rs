use std::sync::Mutex;

extern crate lazy_static;
extern crate serde;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LintSignal {
    pub policy: String,
    pub def_path: String,
    pub kind: String,
    pub module: String,
    pub severity: f32,
}

lazy_static::lazy_static! {
    pub static ref LINT_SIGNALS: Mutex<Vec<LintSignal>> =
        Mutex::new(Vec::new());
}
