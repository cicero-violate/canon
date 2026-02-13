use std::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize}; // ← Added for bincode in tlog

/// Page epoch (monotonic counter)
#[repr(transparent)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize, // ← Added derives
)]
pub struct Epoch(pub u32);

impl Epoch {
    pub fn new(value: u32) -> Self {
        Epoch(value)
    }
}

/// Thread-safe epoch cell reused inside the Rust Page structure.
#[derive(Debug)]
pub struct EpochCell {
    inner: AtomicU32,
}

impl EpochCell {
    pub fn new(value: u32) -> Self {
        Self {
            inner: AtomicU32::new(value),
        }
    }

    #[inline]
    pub fn load(&self) -> Epoch {
        Epoch(self.inner.load(Ordering::Acquire))
    }

    #[inline]
    pub fn store(&self, value: Epoch) {
        self.inner.store(value.0, Ordering::Release);
    }

    #[inline]
    pub fn increment(&self) -> Epoch {
        let old = self.inner.fetch_add(1, Ordering::AcqRel);
        println!("EPOCH_INCREMENT: was {} → now {}", old, old + 1);
        Epoch(old)
    }
}
