mod law_file_length_limit;

use rustc_lint::LintStore;

pub use law_file_length_limit::{enforce_file_length, reset_cache, FILE_TOO_LONG};

pub fn register_laws(store: &mut LintStore) {
    law_file_length_limit::register_law(store);
}
