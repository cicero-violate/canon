//! MMSB Primitives
//!
//! Shared foundational types (IDs, Epoch, Hash) used across MMSB crates.
//! This crate has ZERO dependencies except serde for serialization.
//! No semantics, no logic — just shapes.

use serde::{Deserialize, Serialize};

/// Canonical hash type (32 bytes)
pub type Hash = [u8; 32];

/// Logical page identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PageID(pub u64);

/// Delta mutation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeltaID(pub u64);

/// Canonical epoch (logical time)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Epoch(pub u32);

/// Timestamp (logical or wall-clock)
pub type Timestamp = u64;

/// Event ID (content-addressed)
pub type EventId = Hash;

impl std::fmt::Display for PageID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PageID({})", self.0)
    }
}
