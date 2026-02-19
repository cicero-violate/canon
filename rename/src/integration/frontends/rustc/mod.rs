//! Rustc integration frontend that captures graph snapshots.

use thiserror::Error;

#[cfg(feature = "rustc_frontend")]
mod collector;
#[cfg(feature = "rustc_frontend")]
mod context;
#[cfg(feature = "rustc_frontend")]
mod crate_meta;
#[cfg(feature = "rustc_frontend")]
mod hir_bodies;
#[cfg(feature = "rustc_frontend")]
mod items;
#[cfg(feature = "rustc_frontend")]
mod metadata;
#[cfg(feature = "rustc_frontend")]
mod mir;
#[cfg(feature = "rustc_frontend")]
mod nodes;
#[cfg(feature = "rustc_frontend")]
mod traits;
#[cfg(feature = "rustc_frontend")]
mod types;

#[cfg(not(feature = "rustc_frontend"))]
mod stub;

/// Errors that may arise when invoking the rustc frontend.
#[derive(Debug, Error)]
pub enum RustcFrontendError {
    /// Feature flag not enabled.
    #[error("rustc frontend feature not enabled")]
    Unavailable,
    /// I/O error while preparing the frontend.
    #[cfg(feature = "rustc_frontend")]
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to determine sysroot for rustc.
    #[cfg(feature = "rustc_frontend")]
    #[error("failed to determine sysroot: {0}")]
    Sysroot(std::io::Error),
    /// Callbacks ran without producing a snapshot.
    #[cfg(feature = "rustc_frontend")]
    #[error("compiler produced no snapshot")]
    MissingSnapshot,
}

#[cfg(feature = "rustc_frontend")]
pub use collector::RustcFrontend;

#[cfg(not(feature = "rustc_frontend"))]
pub use stub::RustcFrontend;
