//! Cross-crate linking helpers used after normalization.

/// Links symbols across crate boundaries.
#[derive(Debug, Default)]
pub struct Linker;

impl Linker {
    /// Creates a new linker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Executes the linking pass.
    pub fn link(&mut self) {
        // TODO: stitch symbols across crate boundaries.
    }
}
