#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

// #[cfg(feature = "rustc_frontend")]
// extern crate rustc_driver;
// #[cfg(feature = "rustc_frontend")]
// extern crate rustc_hir;
// #[cfg(feature = "rustc_frontend")]
// extern crate rustc_interface;
// #[cfg(feature = "rustc_frontend")]
// extern crate rustc_middle;
// #[cfg(feature = "rustc_frontend")]
// extern crate rustc_span;

pub mod alias;
pub mod api;
pub mod attributes;
// #[path = "../../compiler_capture/src/mod.rs"]
pub mod core;
pub mod fs;
pub mod macros;
pub mod module_path;
pub mod occurrence;
pub mod pattern;
pub mod scope;
pub mod state;
pub mod structured;

// Compatibility shim for external crates expecting `crate::rename::*`
pub mod rename {
    pub use crate::core;
    pub use crate::structured;
    pub use crate::alias;
    pub use crate::scope;
    pub use crate::state;
}

pub use crate::core::{apply_rename, apply_rename_with_map, collect_names, emit_names};
