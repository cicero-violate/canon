//! Lightweight capture session bookkeeping.

use crate::rustc_integration::ExtractionResult;

/// Bookkeeping structure tracking capture runs during a workspace crawl.
#[derive(Debug, Default)]
pub struct CaptureSession {
    /// Number of capture runs executed.
    pub runs: usize,
    /// Names of processed crates.
    pub crates: Vec<String>,
}

impl CaptureSession {
    /// Records that a crate was captured.
    pub fn record(&mut self, result: &ExtractionResult) {
        self.runs += 1;
        self.crates.push(result.crate_name.clone());
    }
}
