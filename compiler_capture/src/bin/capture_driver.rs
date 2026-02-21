#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;

fn main() {
    if let Err(err) = compiler_capture::frontends::rustc::driver_entry::run_capture_driver() {
        eprintln!("{err:?}");
        std::process::exit(1);
    }
}
