//! Validation helpers that ensure captured metadata satisfies invariants.

use crate::integration::ExtractionResult;

/// Performs light validation to ensure the extraction result looks sane.
pub fn validate(result: &ExtractionResult) -> bool {
    // TODO: add rich validation rules (missing items, stats consistency, etc.)
    !result.crate_name.is_empty()
}
