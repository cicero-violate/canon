#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;
#[cfg(feature = "rustc_frontend")]
extern crate rustc_hir;
#[cfg(feature = "rustc_frontend")]
extern crate rustc_interface;
#[cfg(feature = "rustc_frontend")]
extern crate rustc_middle;
#[cfg(feature = "rustc_frontend")]
extern crate rustc_span;

pub mod fs;
pub mod alias;
pub mod api;
pub mod attributes;
pub mod core;
pub mod macros;
pub mod module_path;
pub mod occurrence;
pub mod pattern;
pub mod scope;
pub mod structured;
pub mod state;
pub mod rename;
#[path = "../../compiler_capture/src/mod.rs"]
pub mod compiler_capture;

pub use rename::core::{apply_rename, apply_rename_with_map, collect_names, emit_names};
