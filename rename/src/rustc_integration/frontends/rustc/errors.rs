//! Rustc integration frontend that captures graph snapshots.

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
#[derive(Debug)]
pub enum RustcFrontendError {
    /// Feature flag not enabled.
    Unavailable,
    /// I/O error while preparing the frontend.
    #[cfg(feature = "rustc_frontend")]
    Io(std::io::Error),
    /// Failed to determine sysroot for rustc.
    #[cfg(feature = "rustc_frontend")]
    Sysroot(std::io::Error),
    /// Callbacks ran without producing a snapshot.
    #[cfg(feature = "rustc_frontend")]
    MissingSnapshot,
}

impl std::fmt::Display for RustcFrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustcFrontendError::Unavailable => write!(f, "rustc frontend feature not enabled"),
            #[cfg(feature = "rustc_frontend")]
            RustcFrontendError::Io(err) => write!(f, "io error: {err}"),
            #[cfg(feature = "rustc_frontend")]
            RustcFrontendError::Sysroot(err) => write!(f, "failed to determine sysroot: {err}"),
            #[cfg(feature = "rustc_frontend")]
            RustcFrontendError::MissingSnapshot => write!(f, "compiler produced no snapshot"),
        }
    }
}

impl std::error::Error for RustcFrontendError {}

#[cfg(feature = "rustc_frontend")]
impl From<std::io::Error> for RustcFrontendError {
    fn from(err: std::io::Error) -> Self {
        RustcFrontendError::Io(err)
    }
}

#[cfg(feature = "rustc_frontend")]
pub use collector::RustcFrontend;

#[cfg(not(feature = "rustc_frontend"))]
pub use stub::RustcFrontend;
