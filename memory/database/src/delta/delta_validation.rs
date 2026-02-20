use crate::delta::{Delta, DeltaError};

/// Validate structural consistency of a delta before application.
pub fn validate_delta(delta: &Delta) -> Result<(), DeltaError> {
    if delta.is_sparse {
        let changed = delta.mask.iter().filter(|&&bit| bit).count();
        if changed != delta.payload.len() {
            return Err(DeltaError::SizeMismatch {
                mask_len: changed,
                payload_len: delta.payload.len(),
            });
        }
    } else if delta.mask.len() != delta.payload.len() {
        return Err(DeltaError::SizeMismatch {
            mask_len: delta.mask.len(),
            payload_len: delta.payload.len(),
        });
    }

    Ok(())
}
