mod law_dead_integration;
mod law_file_length_limit;
mod law_reachability;

use rustc_lint::LintStore;

pub use law_dead_integration::{DEAD_INTEGRATION, enforce_dead_integration};
pub use law_file_length_limit::{FILE_TOO_LONG, enforce_file_length, reset_cache};
pub use law_reachability::{best_reconnect_target, collect_dead_items, reset_reachability};

pub fn register_laws(store: &mut LintStore) {
    law_dead_integration::register_law(store);
    law_file_length_limit::register_law(store);
    // Ensure reachability-related signals are consistently driven
    // (even though it does not declare a lint, this keeps behavior explicit)
    // No-op registration placeholder for structural symmetry
}
