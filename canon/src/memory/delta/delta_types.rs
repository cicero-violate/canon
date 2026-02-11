use crate::memory::primitives::PageID;
use thiserror::Error;

use serde::{Deserialize, Serialize}; // Required for bincode in tlog

// #[repr(transparent)]
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
// pub struct DeltaID(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source(pub String);

#[derive(Debug, Error)]
pub enum DeltaError {
    #[error("Mask/payload size mismatch mask={mask_len} payload={payload_len}")]
    SizeMismatch { mask_len: usize, payload_len: usize },

    #[error("PageID mismatch: expected {expected:?}, found {found:?}")]
    PageIDMismatch { expected: PageID, found: PageID },

    #[error("Mask size mismatch: expected {expected}, found {found}")]
    MaskSizeMismatch { expected: usize, found: usize },
}
