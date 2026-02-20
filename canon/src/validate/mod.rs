use crate::ir::SystemState;
mod check_artifacts;
mod check_deltas;
mod check_execution;
mod check_graphs;
mod check_project;
mod check_proposals;
pub mod error;
pub mod helpers;
pub mod rules;
pub use error::{ValidationErrors, Violation};
pub use rules::CanonRule;
pub fn validate_ir(ir: &SystemState) -> Result<(), ValidationErrors> {
    let mut violations = Vec::new();
    check_project::check(&ir, &mut violations);
    let indexes = helpers::build_indexes(ir, &mut violations);
    check_artifacts::check_artifacts(ir, &indexes, &mut violations);
    check_deltas::check_deltas_top(ir, &indexes, &mut violations);
    check_graphs::check_graphs(ir, &indexes, &mut violations);
    check_proposals::check(ir, &indexes, &mut violations);
    check_execution::check(ir, &indexes, &mut violations);
    if violations.is_empty() { Ok(()) } else { Err(ValidationErrors::new(violations)) }
}
