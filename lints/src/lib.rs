#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_session;
extern crate rustc_span;
extern crate serde;

mod classify;
mod pass;
mod policy;
mod signal;

use rustc_lint::LintStore;

// ---- Public re-exports for runner ----

pub use pass::ApiTraitsOnly;
pub use policy::API_TRAITS_ONLY;
pub use signal::{LINT_SIGNALS, LintSignal};

// ---- Registration entry point ----

pub fn register_lints(store: &mut LintStore) {
    store.register_lints(&[&API_TRAITS_ONLY]);
    store.register_late_pass(|_| Box::new(ApiTraitsOnly));
}
