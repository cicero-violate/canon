//! Build script for lint_rustc wrapper
//! Usage: rustc build_wrapper.rs && ./build_wrapper

use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Get sysroot
    let sysroot_output = Command::new("rustc")
        .args(&["--print", "sysroot"])
        .output()
        .expect("failed to get rustc sysroot");
    let sysroot = String::from_utf8(sysroot_output.stdout)
        .expect("invalid utf8 from rustc")
        .trim()
        .to_string();

    // Build lints library first
    let status = Command::new("cargo")
        .arg("build")
        .env("LINT_RUSTC_BUILDING", "1")
        .current_dir(".")
        .status()
        .expect("failed to build lints");
    if !status.success() {
        eprintln!("cargo build failed");
        std::process::exit(1);
    }

    // Find libraries
    let target_dir = PathBuf::from("../target/debug");
    let lints_rlib = target_dir.join("liblints.rlib");

    let deps_dir = target_dir.join("deps");
    
    // Find lazy_static from deps (lints was compiled against this version)
    let lazy_static = std::fs::read_dir(&deps_dir)
        .expect("failed to read deps dir")
        .filter_map(Result::ok)
        .find(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("liblazy_static-") && name.ends_with(".rlib")
        })
        .expect("no lazy_static rlib found")
        .path();

    // Get serde_json from sysroot (not deps) to avoid conflicts
    let serde_json_sysroot = std::fs::read_dir(format!("{}/lib/rustlib/x86_64-unknown-linux-gnu/lib", sysroot))
        .expect("failed to read sysroot rustlib")
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("libserde_json-")
                && e.file_name().to_string_lossy().ends_with(".rmeta")
        })
        .expect("no serde_json in sysroot")
        .path();

    let rustc_driver = std::fs::read_dir(format!("{}/lib", sysroot))
        .expect("failed to read sysroot/lib")
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("librustc_driver-")
                && e.file_name().to_string_lossy().ends_with(".so")
        })
        .expect("no rustc_driver.so found")
        .path();

    // Compile lint_rustc
    let status = Command::new("rustc")
        .args(&[
            "-Z", "unstable-options",
            "-C", "prefer-dynamic",
            "-L", target_dir.to_str().unwrap(),
            "-L", deps_dir.to_str().unwrap(),
            "-L", &format!("{}/lib", sysroot),
            "-L", &format!("{}/lib/rustlib/x86_64-unknown-linux-gnu/lib", sysroot),
            "--extern", &format!("lints={}", lints_rlib.display()),
            "--extern", &format!("rustc_driver={}", rustc_driver.display()),
            "--extern", &format!("lazy_static={}", lazy_static.display()),
            "--extern", &format!("serde_json={}", serde_json_sysroot.display()),
            "--edition", "2021",
            "lint_rustc/src/main.rs",
            "-o", target_dir.join("lint_rustc").to_str().unwrap(),
        ])
        .status()
        .expect("failed to compile lint_rustc");

    if status.success() {
        println!("✓ Built lint_rustc at ../target/debug/lint_rustc");
    } else {
        eprintln!("rustc compilation failed");
        std::process::exit(1);
    }
}
