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
pub mod rename;
pub mod state;
pub mod rustc_integration;

pub use rename::core::{apply_rename, apply_rename_with_map, collect_names, emit_names};
