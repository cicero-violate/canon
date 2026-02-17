mod law_dead_integration;
mod law_file_length_limit;

use rustc_lint::LintStore;

pub use law_dead_integration::{DEAD_INTEGRATION, enforce_dead_integration};
pub use law_file_length_limit::{FILE_TOO_LONG, enforce_file_length, reset_cache};

pub fn register_laws(store: &mut LintStore) {
    law_dead_integration::register_law(store);
    law_file_length_limit::register_law(store);
}
