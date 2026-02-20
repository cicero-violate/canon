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


#[path = "mod.rs"]
mod shared;

pub use shared::*;
